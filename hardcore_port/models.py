"""Sequential actor-critic models for the dedicated hardcore port."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Normal


def _ensure_sequence_batch(state: torch.Tensor) -> torch.Tensor:
    if state.dim() == 2:
        return state.unsqueeze(1)
    return state


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding similar to the reference transformer."""

    def __init__(self, d_model: int, seq_len: int) -> None:
        super().__init__()
        scale = float(d_model) ** 0.5
        positions = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(1000.0) / d_model)
        )
        encoding = torch.zeros(seq_len, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * div_term)
        encoding[:, 1::2] = torch.cos(positions * div_term)
        encoding = torch.flip(encoding.unsqueeze(0), dims=[1]) / scale
        self.register_buffer("encoding", encoding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding[:, -x.size(1) :, :]


class LastStepTransformerEncoder(nn.Module):
    """Pre-LN transformer block that queries only the last sequence element."""

    def __init__(
        self,
        *,
        input_dim: int,
        seq_len: int,
        model_dim: int = 96,
        num_heads: int = 4,
        ff_dim: int = 192,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.zeros_(self.embedding.bias)
        self.position = PositionalEncoding(model_dim, seq_len=seq_len)
        self.norm1 = nn.LayerNorm(model_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(model_dim)
        self.ff1 = nn.Linear(model_dim, ff_dim)
        self.ff2 = nn.Linear(ff_dim, model_dim)
        nn.init.xavier_uniform_(self.ff1.weight)
        nn.init.zeros_(self.ff1.bias)
        nn.init.xavier_uniform_(self.ff2.weight)
        nn.init.zeros_(self.ff2.bias)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        sequence = _ensure_sequence_batch(state)
        encoded = self.embedding(sequence)
        encoded = self.position(encoded)

        normalized = self.norm1(encoded)
        query = normalized[:, -1:, :]
        attn_output, _ = self.attn(query=query, key=normalized, value=normalized, need_weights=False)
        residual = encoded[:, -1:, :] + self.dropout1(attn_output)

        ff_input = self.norm2(residual)
        ff_output = self.ff2(self.dropout2(self.activation(self.ff1(ff_input))))
        output = residual + self.dropout2(ff_output)
        return output[:, -1, :]


class LSTMEncoder(nn.Module):
    """Single-layer LSTM encoder with last-state readout."""

    def __init__(self, *, input_dim: int, hidden_dim: int = 96) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=False,
            num_layers=1,
            dropout=0.0,
        )
        with torch.no_grad():
            self.lstm.bias_hh_l0.fill_(-0.2)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        sequence = _ensure_sequence_batch(state)
        output, _ = self.lstm(sequence)
        return output[:, -1, :]


def build_encoder(
    *,
    backbone: str,
    state_dim: int,
    history_length: int,
) -> nn.Module:
    normalized = backbone.lower()
    if normalized == "lstm":
        return LSTMEncoder(input_dim=state_dim, hidden_dim=96)
    if normalized == "transformer":
        return LastStepTransformerEncoder(
            input_dim=state_dim,
            seq_len=history_length,
            model_dim=96,
            num_heads=4,
            ff_dim=192,
        )
    raise ValueError(f"Unsupported backbone: {backbone}")


class CriticNetwork(nn.Module):
    """Sequential critic used by both SAC and TD3."""

    def __init__(
        self,
        *,
        backbone: str,
        state_dim: int,
        action_dim: int,
        history_length: int,
    ) -> None:
        super().__init__()
        self.encoder = build_encoder(
            backbone=backbone,
            state_dim=state_dim,
            history_length=history_length,
        )
        self.hidden = nn.Linear(96 + action_dim, 192)
        nn.init.xavier_uniform_(self.hidden.weight, gain=nn.init.calculate_gain("tanh"))
        nn.init.zeros_(self.hidden.bias)
        self.out = nn.Linear(192, 1, bias=False)
        nn.init.uniform_(self.out.weight, -0.003, 0.003)
        self.activation = nn.Tanh()

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(state)
        x = torch.cat((encoded, action), dim=1)
        x = self.activation(self.hidden(x))
        return self.out(x) * 10.0


class DeterministicActor(nn.Module):
    """TD3 actor with sequential backbone."""

    def __init__(
        self,
        *,
        backbone: str,
        state_dim: int,
        action_dim: int,
        history_length: int,
    ) -> None:
        super().__init__()
        self.encoder = build_encoder(
            backbone=backbone,
            state_dim=state_dim,
            history_length=history_length,
        )
        self.policy = nn.Linear(96, action_dim, bias=False)
        nn.init.uniform_(self.policy.weight, -0.003, 0.003)
        self.output = nn.Tanh()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(state)
        return self.output(self.policy(encoded))


class StochasticActor(nn.Module):
    """SAC actor with squashed Gaussian policy."""

    def __init__(
        self,
        *,
        backbone: str,
        state_dim: int,
        action_dim: int,
        history_length: int,
    ) -> None:
        super().__init__()
        self.encoder = build_encoder(
            backbone=backbone,
            state_dim=state_dim,
            history_length=history_length,
        )
        self.mean_head = nn.Linear(96, action_dim, bias=False)
        self.log_std_head = nn.Linear(96, action_dim, bias=False)
        nn.init.uniform_(self.mean_head.weight, -0.003, 0.003)
        nn.init.uniform_(self.log_std_head.weight, -0.003, 0.003)
        self.output = nn.Tanh()

    def forward(self, state: torch.Tensor, *, explore: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(state)
        means = self.mean_head(encoded)
        log_stds = torch.clamp(self.log_std_head(encoded), min=-10.0, max=2.0)
        stds = log_stds.exp()
        dist = Normal(means, stds)
        pre_tanh = dist.rsample() if explore else means
        action = self.output(pre_tanh)
        log_prob = dist.log_prob(pre_tanh) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=1, keepdim=True)
