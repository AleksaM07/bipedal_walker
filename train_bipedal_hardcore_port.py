"""Self-contained custom hardcore training script with SAC/TD3 and sequential models."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import deque, namedtuple
from dataclasses import dataclass
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch
from loguru import logger
from torch import nn
from torch.distributions import Normal


CHECKPOINT_ALIASES = {"best_raw", "best_shaped", "last"}
DEFAULT_OUTPUT_DIR = Path("artifacts") / "runs" / "hardcore"
DEFAULT_FRAME_SKIP = 2
DEFAULT_FALL_PENALTY = -10.0
DEFAULT_LR = 4e-4
DEFAULT_BATCH_SIZE = 64
DEFAULT_GAMMA = 0.98
DEFAULT_TAU = 0.01
DEFAULT_ALPHA = 0.01
DEFAULT_DEVICE = "auto"
DEFAULT_STALL_CHECK_WINDOW = 40
DEFAULT_STALL_GRACE_STEPS = 80
DEFAULT_STALL_MIN_PROGRESS = 0.35
DEFAULT_STALL_PATIENCE = 2
DEFAULT_STALL_PENALTY = 0.0
LEGACY_HELPER_STALL_PENALTY = -20.0


class ObservationHistoryWrapper(gym.ObservationWrapper):
    """Stacks the last N observations into a `(history, obs_dim)` tensor."""

    def __init__(self, env: gym.Env, history_length: int = 12) -> None:
        super().__init__(env)
        self.history_length = max(int(history_length), 1)
        self._history: deque[np.ndarray] = deque(maxlen=self.history_length)

        base_low = np.asarray(self.observation_space.low, dtype=np.float32)
        base_high = np.asarray(self.observation_space.high, dtype=np.float32)
        stacked_low = np.repeat(base_low[np.newaxis, ...], self.history_length, axis=0)
        stacked_high = np.repeat(base_high[np.newaxis, ...], self.history_length, axis=0)
        self.observation_space = gym.spaces.Box(
            low=stacked_low,
            high=stacked_high,
            dtype=np.float32,
        )

    def _stack(self) -> np.ndarray:
        return np.stack(list(self._history), axis=0).astype(np.float32)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        self._history.append(obs.copy())
        return self._stack()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.env.reset(seed=seed, options=options)
        obs = np.asarray(observation, dtype=np.float32)
        self._history.clear()
        for _ in range(self.history_length):
            self._history.append(obs.copy())
        return self._stack(), info


class HardcoreWalkerWrapper(gym.Wrapper):
    """Hardcore helper wrapper with frame-skip, fall shaping, and optional anti-stall."""

    def __init__(
        self,
        env: gym.Env,
        *,
        frame_skip: int = DEFAULT_FRAME_SKIP,
        fall_penalty: float = DEFAULT_FALL_PENALTY,
        anti_stall: bool = False,
        stall_check_window: int = DEFAULT_STALL_CHECK_WINDOW,
        stall_grace_steps: int = DEFAULT_STALL_GRACE_STEPS,
        stall_min_progress: float = DEFAULT_STALL_MIN_PROGRESS,
        stall_patience: int = DEFAULT_STALL_PATIENCE,
        stall_penalty: float = LEGACY_HELPER_STALL_PENALTY,
    ) -> None:
        super().__init__(env)
        self.frame_skip = max(int(frame_skip), 1)
        self.fall_penalty = float(fall_penalty)
        self.anti_stall = bool(anti_stall)
        self.stall_check_window = max(int(stall_check_window), 1)
        self.stall_grace_steps = max(int(stall_grace_steps), 0)
        self.stall_min_progress = float(stall_min_progress)
        self.stall_patience = max(int(stall_patience), 1)
        self.stall_penalty = float(stall_penalty)
        self._episode_step = 0
        self._last_progress_x = 0.0
        self._stall_fail_count = 0

    def _get_hull_x(self) -> float:
        hull = getattr(self.env.unwrapped, "hull", None)
        if hull is None:
            return 0.0
        position = getattr(hull, "position", None)
        if position is None:
            return 0.0
        return float(getattr(position, "x", 0.0))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.env.reset(seed=seed, options=options)
        self._episode_step = 0
        self._stall_fail_count = 0
        self._last_progress_x = self._get_hull_x()
        reset_info = dict(info)
        reset_info["hull_x"] = float(self._last_progress_x)
        return observation, reset_info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        total_shaped_reward = 0.0
        total_raw_reward = 0.0
        terminated = False
        truncated = False
        dead = False
        stalled = False
        observation: np.ndarray | None = None
        info: dict[str, Any] = {}

        for _ in range(self.frame_skip):
            observation, reward, terminated, truncated, info = self.env.step(action)
            raw_reward = float(reward)
            total_raw_reward += raw_reward

            dead = bool(getattr(self.env.unwrapped, "game_over", False))
            shaped_reward = self.fall_penalty if dead else raw_reward
            total_shaped_reward += shaped_reward

            if terminated or truncated:
                break

        self._episode_step += 1
        current_x = self._get_hull_x()
        progress_delta = float(current_x - self._last_progress_x)

        check_stall = (
            self.anti_stall
            and not dead
            and not terminated
            and not truncated
            and self._episode_step >= self.stall_grace_steps
            and self._episode_step % self.stall_check_window == 0
        )
        if check_stall:
            if progress_delta < self.stall_min_progress:
                self._stall_fail_count += 1
            else:
                self._stall_fail_count = 0
            self._last_progress_x = current_x
            if self._stall_fail_count >= self.stall_patience:
                total_shaped_reward += self.stall_penalty
                terminated = True
                stalled = True

        step_info = dict(info)
        step_info["dead"] = dead
        step_info["stalled"] = stalled
        step_info["stall_fail_count"] = int(self._stall_fail_count)
        step_info["hull_x"] = float(current_x)
        step_info["progress_delta"] = float(progress_delta)
        step_info["raw_reward"] = float(total_raw_reward)
        step_info["shaped_reward"] = float(total_shaped_reward)
        return observation, total_shaped_reward, terminated, truncated, step_info


def make_hardcore_env(
    *,
    env_id: str = "BipedalWalkerHardcore-v3",
    history_length: int = 12,
    frame_skip: int = DEFAULT_FRAME_SKIP,
    fall_penalty: float = DEFAULT_FALL_PENALTY,
    anti_stall: bool = False,
    stall_check_window: int = DEFAULT_STALL_CHECK_WINDOW,
    stall_grace_steps: int = DEFAULT_STALL_GRACE_STEPS,
    stall_min_progress: float = DEFAULT_STALL_MIN_PROGRESS,
    stall_patience: int = DEFAULT_STALL_PATIENCE,
    stall_penalty: float = LEGACY_HELPER_STALL_PENALTY,
    render_mode: str | None = None,
) -> gym.Env:
    """Creates the custom hardcore environment used by the trainer."""
    env_kwargs: dict[str, Any] = {}
    if render_mode is not None:
        env_kwargs["render_mode"] = render_mode

    env = gym.make(env_id, **env_kwargs)
    env = HardcoreWalkerWrapper(
        env,
        frame_skip=frame_skip,
        fall_penalty=fall_penalty,
        anti_stall=anti_stall,
        stall_check_window=stall_check_window,
        stall_grace_steps=stall_grace_steps,
        stall_min_progress=stall_min_progress,
        stall_patience=stall_patience,
        stall_penalty=stall_penalty,
    )
    if history_length > 1:
        env = ObservationHistoryWrapper(env, history_length=history_length)
    return env


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


class ReplayBuffer:
    """Simple replay buffer that supports sequence-shaped states."""

    def __init__(self, *, buffer_size: int, batch_size: int, device: torch.device) -> None:
        self.memory = deque(maxlen=int(buffer_size))
        self.batch_size = int(batch_size)
        self.device = device
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        experience = self.experience(
            np.asarray(state, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        )
        self.memory.append(experience)

    def sample(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(
            np.stack([exp.state for exp in experiences], axis=0)
        ).float().to(self.device)
        actions = torch.from_numpy(
            np.stack([exp.action for exp in experiences], axis=0)
        ).float().to(self.device)
        rewards = torch.from_numpy(
            np.asarray([exp.reward for exp in experiences], dtype=np.float32).reshape(-1, 1)
        ).float().to(self.device)
        next_states = torch.from_numpy(
            np.stack([exp.next_state for exp in experiences], axis=0)
        ).float().to(self.device)
        dones = torch.from_numpy(
            np.asarray([exp.done for exp in experiences], dtype=np.float32).reshape(-1, 1)
        ).float().to(self.device)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.memory)


class GaussianNoise:
    """Clipped Gaussian noise."""

    def __init__(self, *, mu: np.ndarray, sigma: float, clip: float | None = None) -> None:
        self.mu = np.asarray(mu, dtype=np.float32)
        self.sigma = float(sigma)
        self.clip = None if clip is None else float(clip)

    def __call__(self) -> np.ndarray:
        sample = np.random.normal(self.mu, self.sigma).astype(np.float32)
        if self.clip is not None:
            sample = np.clip(sample, -self.clip, self.clip)
        return sample


class DecayingOrnsteinUhlenbeckNoise:
    """OU exploration process with per-episode sigma decay."""

    def __init__(
        self,
        *,
        mu: np.ndarray,
        theta: float = 4.0,
        sigma: float = 1.2,
        dt: float = 0.04,
        sigma_decay: float = 0.9995,
    ) -> None:
        self.mu = np.asarray(mu, dtype=np.float32)
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.dt = float(dt)
        self.sigma_decay = float(sigma_decay)
        self.state = self.mu.copy()

    def __call__(self) -> np.ndarray:
        noise = np.random.normal(size=self.mu.shape).astype(np.float32)
        self.state = (
            self.state
            + self.theta * (self.mu - self.state) * self.dt
            + self.sigma * np.sqrt(self.dt) * noise
        ).astype(np.float32)
        return self.state.copy()

    def step_end(self) -> None:
        return None

    def episode_end(self) -> None:
        self.sigma *= self.sigma_decay
        self.state = self.mu.copy()


def resolve_device(device: str | None) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA je trazen, ali nije dostupan. Proveri PyTorch CUDA instalaciju i GPU okruzenje.")
    return torch.device(device)


def soft_update(local_model: nn.Module, target_model: nn.Module, tau: float) -> None:
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


def hard_update(local_model: nn.Module, target_model: nn.Module) -> None:
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(local_param.data)


@dataclass
class AgentMetadata:
    algorithm: str
    backbone: str
    history_length: int
    episode: int | None = None
    eval_mean_reward: float | None = None
    eval_mean_shaped_reward: float | None = None


class SACAgent:
    """Custom SAC agent closely following the ugur training recipe."""

    algorithm = "sac"

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        backbone: str,
        history_length: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        lr: float = DEFAULT_LR,
        weight_decay: float = 0.0,
        gamma: float = DEFAULT_GAMMA,
        alpha: float = DEFAULT_ALPHA,
        tau: float = DEFAULT_TAU,
        batch_size: int = DEFAULT_BATCH_SIZE,
        buffer_size: int = 500_000,
        update_freq: int = 1,
        device: str | None = None,
    ) -> None:
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.backbone = backbone
        self.history_length = int(history_length)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.tau = float(tau)
        self.batch_size = int(batch_size)
        self.update_freq = int(update_freq)
        self.learn_call = 0
        self.device = resolve_device(device)
        self.action_low = torch.as_tensor(action_low, dtype=torch.float32, device=self.device)
        self.action_high = torch.as_tensor(action_high, dtype=torch.float32, device=self.device)

        self.actor = StochasticActor(
            backbone=backbone,
            state_dim=state_dim,
            action_dim=action_dim,
            history_length=history_length,
        ).to(self.device)
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            amsgrad=True,
        )

        self.critic_1 = CriticNetwork(
            backbone=backbone,
            state_dim=state_dim,
            action_dim=action_dim,
            history_length=history_length,
        ).to(self.device)
        self.target_critic_1 = CriticNetwork(
            backbone=backbone,
            state_dim=state_dim,
            action_dim=action_dim,
            history_length=history_length,
        ).to(self.device)
        hard_update(self.critic_1, self.target_critic_1)
        self.critic_1_optimizer = torch.optim.AdamW(
            self.critic_1.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            amsgrad=True,
        )

        self.critic_2 = CriticNetwork(
            backbone=backbone,
            state_dim=state_dim,
            action_dim=action_dim,
            history_length=history_length,
        ).to(self.device)
        self.target_critic_2 = CriticNetwork(
            backbone=backbone,
            state_dim=state_dim,
            action_dim=action_dim,
            history_length=history_length,
        ).to(self.device)
        hard_update(self.critic_2, self.target_critic_2)
        self.critic_2_optimizer = torch.optim.AdamW(
            self.critic_2.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            amsgrad=True,
        )

        self.memory = ReplayBuffer(
            buffer_size=buffer_size,
            batch_size=batch_size,
            device=self.device,
        )
        self.mse_loss = nn.MSELoss()

    def train_mode(self) -> None:
        self.actor.train()
        self.critic_1.train()
        self.critic_2.train()

    def eval_mode(self) -> None:
        self.actor.eval()
        self.critic_1.eval()
        self.critic_2.eval()

    def freeze_networks(self) -> None:
        for parameter in chain(
            self.actor.parameters(),
            self.critic_1.parameters(),
            self.critic_2.parameters(),
        ):
            parameter.requires_grad = False

    @torch.no_grad()
    def get_action(self, state: np.ndarray, *, explore: bool = True) -> np.ndarray:
        state_tensor = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0).to(self.device)
        action, _ = self.actor(state_tensor, explore=explore)
        action = torch.max(torch.min(action, self.action_high), self.action_low)
        return action.cpu().numpy()[0]

    def observe(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.memory.add(state, action, reward, next_state, done)

    def learn_one_step(self) -> dict[str, float] | None:
        if len(self.memory) <= self.batch_size:
            return None
        experiences = self.memory.sample()
        return self.learn(experiences)

    def learn(
        self,
        experiences: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> dict[str, float]:
        self.learn_call += 1
        states, actions, rewards, next_states, dones = experiences

        with torch.no_grad():
            next_actions, next_log_probs = self.actor(next_states, explore=True)
            target_q1 = self.target_critic_1(next_states, next_actions)
            target_q2 = self.target_critic_2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            q_targets = rewards + self.gamma * target_q * (1.0 - dones)

        expected_q1 = self.critic_1(states, actions)
        critic_1_loss = self.mse_loss(expected_q1, q_targets)
        self.critic_1_optimizer.zero_grad(set_to_none=True)
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        expected_q2 = self.critic_2(states, actions)
        critic_2_loss = self.mse_loss(expected_q2, q_targets)
        self.critic_2_optimizer.zero_grad(set_to_none=True)
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        sampled_actions, log_probs = self.actor(states, explore=True)
        q_pi = torch.min(
            self.critic_1(states, sampled_actions),
            self.critic_2(states, sampled_actions),
        )
        actor_loss = -(q_pi - self.alpha * log_probs).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.learn_call % self.update_freq == 0:
            self.learn_call = 0
            soft_update(self.critic_1, self.target_critic_1, self.tau)
            soft_update(self.critic_2, self.target_critic_2, self.tau)

        return {
            "actor_loss": float(actor_loss.item()),
            "critic_1_loss": float(critic_1_loss.item()),
            "critic_2_loss": float(critic_2_loss.item()),
        }

    def step_end(self) -> None:
        return None

    def episode_end(self) -> None:
        return None

    def save_checkpoint(self, path: Path, *, metadata: AgentMetadata | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": self.algorithm,
                "backbone": self.backbone,
                "history_length": self.history_length,
                "metadata": None if metadata is None else metadata.__dict__,
                "actor": self.actor.state_dict(),
                "critic_1": self.critic_1.state_dict(),
                "critic_2": self.critic_2.state_dict(),
                "target_critic_1": self.target_critic_1.state_dict(),
                "target_critic_2": self.target_critic_2.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
                "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic_1.load_state_dict(checkpoint["critic_1"])
        self.critic_2.load_state_dict(checkpoint["critic_2"])
        self.target_critic_1.load_state_dict(checkpoint["target_critic_1"])
        self.target_critic_2.load_state_dict(checkpoint["target_critic_2"])
        if "actor_optimizer" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        if "critic_1_optimizer" in checkpoint:
            self.critic_1_optimizer.load_state_dict(checkpoint["critic_1_optimizer"])
        if "critic_2_optimizer" in checkpoint:
            self.critic_2_optimizer.load_state_dict(checkpoint["critic_2_optimizer"])
        return checkpoint


class TD3Agent:
    """Custom TD3 agent closely following the ugur training recipe."""

    algorithm = "td3"

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        backbone: str,
        history_length: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        lr: float = DEFAULT_LR,
        weight_decay: float = 0.0,
        gamma: float = DEFAULT_GAMMA,
        tau: float = DEFAULT_TAU,
        batch_size: int = DEFAULT_BATCH_SIZE,
        buffer_size: int = 500_000,
        update_freq: int = 2,
        device: str | None = None,
    ) -> None:
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.backbone = backbone
        self.history_length = int(history_length)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.batch_size = int(batch_size)
        self.update_freq = int(update_freq)
        self.learn_call = 0
        self.device = resolve_device(device)
        self.action_low = torch.as_tensor(action_low, dtype=torch.float32, device=self.device)
        self.action_high = torch.as_tensor(action_high, dtype=torch.float32, device=self.device)

        self.actor = DeterministicActor(
            backbone=backbone,
            state_dim=state_dim,
            action_dim=action_dim,
            history_length=history_length,
        ).to(self.device)
        self.target_actor = DeterministicActor(
            backbone=backbone,
            state_dim=state_dim,
            action_dim=action_dim,
            history_length=history_length,
        ).to(self.device)
        hard_update(self.actor, self.target_actor)
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            amsgrad=True,
        )

        self.critic_1 = CriticNetwork(
            backbone=backbone,
            state_dim=state_dim,
            action_dim=action_dim,
            history_length=history_length,
        ).to(self.device)
        self.target_critic_1 = CriticNetwork(
            backbone=backbone,
            state_dim=state_dim,
            action_dim=action_dim,
            history_length=history_length,
        ).to(self.device)
        hard_update(self.critic_1, self.target_critic_1)
        self.critic_1_optimizer = torch.optim.AdamW(
            self.critic_1.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            amsgrad=True,
        )

        self.critic_2 = CriticNetwork(
            backbone=backbone,
            state_dim=state_dim,
            action_dim=action_dim,
            history_length=history_length,
        ).to(self.device)
        self.target_critic_2 = CriticNetwork(
            backbone=backbone,
            state_dim=state_dim,
            action_dim=action_dim,
            history_length=history_length,
        ).to(self.device)
        hard_update(self.critic_2, self.target_critic_2)
        self.critic_2_optimizer = torch.optim.AdamW(
            self.critic_2.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            amsgrad=True,
        )

        self.noise_generator = DecayingOrnsteinUhlenbeckNoise(
            mu=np.zeros(action_dim, dtype=np.float32),
            theta=4.0,
            sigma=1.2,
            dt=0.04,
            sigma_decay=0.9995,
        )
        self.target_noise = GaussianNoise(
            mu=np.zeros(action_dim, dtype=np.float32),
            sigma=0.2,
            clip=0.4,
        )
        self.memory = ReplayBuffer(
            buffer_size=buffer_size,
            batch_size=batch_size,
            device=self.device,
        )
        self.mse_loss = nn.MSELoss()

    def train_mode(self) -> None:
        self.actor.train()
        self.critic_1.train()
        self.critic_2.train()

    def eval_mode(self) -> None:
        self.actor.eval()
        self.critic_1.eval()
        self.critic_2.eval()

    def freeze_networks(self) -> None:
        for parameter in chain(
            self.actor.parameters(),
            self.critic_1.parameters(),
            self.critic_2.parameters(),
        ):
            parameter.requires_grad = False

    @torch.no_grad()
    def get_action(self, state: np.ndarray, *, explore: bool = False) -> np.ndarray:
        state_tensor = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0).to(self.device)
        action = self.actor(state_tensor)[0]
        if explore:
            action = action + torch.as_tensor(self.noise_generator(), dtype=torch.float32, device=self.device)
        action = torch.max(torch.min(action, self.action_high), self.action_low)
        return action.cpu().numpy()

    def observe(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.memory.add(state, action, reward, next_state, done)

    def learn_one_step(self) -> dict[str, float] | None:
        if len(self.memory) <= self.batch_size:
            return None
        experiences = self.memory.sample()
        return self.learn(experiences)

    def learn(
        self,
        experiences: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> dict[str, float]:
        self.learn_call += 1
        states, actions, rewards, next_states, dones = experiences

        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            next_actions = next_actions + torch.as_tensor(
                self.target_noise(),
                dtype=torch.float32,
                device=self.device,
            )
            next_actions = torch.max(torch.min(next_actions, self.action_high), self.action_low)
            target_q1 = self.target_critic_1(next_states, next_actions)
            target_q2 = self.target_critic_2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            q_targets = rewards + self.gamma * target_q * (1.0 - dones)

        expected_q1 = self.critic_1(states, actions)
        critic_1_loss = self.mse_loss(expected_q1, q_targets)
        self.critic_1_optimizer.zero_grad(set_to_none=True)
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        expected_q2 = self.critic_2(states, actions)
        critic_2_loss = self.mse_loss(expected_q2, q_targets)
        self.critic_2_optimizer.zero_grad(set_to_none=True)
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        actor_loss_value = 0.0
        if self.learn_call % self.update_freq == 0:
            self.learn_call = 0
            predicted_actions = self.actor(states)
            actor_loss = -self.critic_1(states, predicted_actions).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            actor_loss_value = float(actor_loss.item())

            soft_update(self.actor, self.target_actor, self.tau)
            soft_update(self.critic_1, self.target_critic_1, self.tau)
            soft_update(self.critic_2, self.target_critic_2, self.tau)

        return {
            "actor_loss": actor_loss_value,
            "critic_1_loss": float(critic_1_loss.item()),
            "critic_2_loss": float(critic_2_loss.item()),
        }

    def step_end(self) -> None:
        self.noise_generator.step_end()

    def episode_end(self) -> None:
        self.noise_generator.episode_end()

    def save_checkpoint(self, path: Path, *, metadata: AgentMetadata | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": self.algorithm,
                "backbone": self.backbone,
                "history_length": self.history_length,
                "metadata": None if metadata is None else metadata.__dict__,
                "actor": self.actor.state_dict(),
                "target_actor": self.target_actor.state_dict(),
                "critic_1": self.critic_1.state_dict(),
                "critic_2": self.critic_2.state_dict(),
                "target_critic_1": self.target_critic_1.state_dict(),
                "target_critic_2": self.target_critic_2.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
                "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.target_actor.load_state_dict(checkpoint["target_actor"])
        self.critic_1.load_state_dict(checkpoint["critic_1"])
        self.critic_2.load_state_dict(checkpoint["critic_2"])
        self.target_critic_1.load_state_dict(checkpoint["target_critic_1"])
        self.target_critic_2.load_state_dict(checkpoint["target_critic_2"])
        if "actor_optimizer" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        if "critic_1_optimizer" in checkpoint:
            self.critic_1_optimizer.load_state_dict(checkpoint["critic_1_optimizer"])
        if "critic_2_optimizer" in checkpoint:
            self.critic_2_optimizer.load_state_dict(checkpoint["critic_2_optimizer"])
        return checkpoint


Agent = SACAgent | TD3Agent
EnvFactory = Callable[..., gym.Env]


def evaluate_agent(
    env_factory: EnvFactory,
    agent: Agent,
    *,
    episodes: int = 20,
    max_steps: int = 750,
    seed_start: int = 10_000,
    render: bool = False,
    video_folder: str | Path | None = None,
    video_prefix: str = "checkpoint_eval",
    video_episodes: int = 0,
) -> dict[str, Any]:
    """Runs deterministic evaluation and returns both raw and shaped metrics."""
    rewards: list[float] = []
    shaped_rewards: list[float] = []
    lengths: list[int] = []
    episode_details: list[dict[str, Any]] = []
    video_files: list[str] = []
    video_root = None if video_folder is None else Path(video_folder)

    agent.eval_mode()
    for episode_index in range(episodes):
        episode_seed = seed_start + episode_index
        env = env_factory()
        should_record = video_root is not None and episode_index < max(int(video_episodes), 0)
        if should_record:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=str(video_root),
                episode_trigger=lambda current_episode: current_episode == 0,
                name_prefix=f"{video_prefix}_ep{episode_index + 1}_seed{episode_seed}",
                disable_logger=True,
            )
        try:
            observation, _ = env.reset(seed=episode_seed)
            done = False
            episode_reward = 0.0
            episode_shaped_reward = 0.0
            steps = 0

            while not done and steps < max_steps:
                action = agent.get_action(observation, explore=False)
                clipped_action = np.clip(action, env.action_space.low, env.action_space.high)
                observation, shaped_reward, terminated, truncated, info = env.step(clipped_action)
                raw_reward = float(info.get("raw_reward", shaped_reward))
                episode_reward += raw_reward
                episode_shaped_reward += float(shaped_reward)
                steps += 1
                done = terminated or truncated
                if render:
                    env.render()

            rewards.append(float(episode_reward))
            shaped_rewards.append(float(episode_shaped_reward))
            lengths.append(int(steps))
            episode_details.append(
                {
                    "index": int(episode_index + 1),
                    "seed": int(episode_seed),
                    "reward": float(episode_reward),
                    "shaped_reward": float(episode_shaped_reward),
                    "length": int(steps),
                }
            )
        finally:
            env.close()
        if should_record and video_root is not None:
            matching_videos = sorted(
                video_root.glob(f"{video_prefix}_ep{episode_index + 1}_seed{episode_seed}*.mp4"),
                key=lambda path: path.stat().st_mtime,
            )
            if matching_videos:
                video_files.append(str(matching_videos[-1]))
            else:
                logger.warning(
                    "Video nije pronadjen posle evaluacije | ep={} | seed={} | folder={}",
                    episode_index + 1,
                    episode_seed,
                    video_root,
                )

    best_episode = max(episode_details, key=lambda item: float(item["reward"]))
    worst_episode = min(episode_details, key=lambda item: float(item["reward"]))
    result = {
        "episodes": int(episodes),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_shaped_reward": float(np.mean(shaped_rewards)),
        "std_shaped_reward": float(np.std(shaped_rewards)),
        "mean_length": float(np.mean(lengths)),
        "rewards": rewards,
        "shaped_rewards": shaped_rewards,
        "lengths": lengths,
        "episode_details": episode_details,
        "best_episode": dict(best_episode),
        "worst_episode": dict(worst_episode),
    }
    if video_root is not None:
        result["video_folder"] = str(video_root)
        result["video_files"] = video_files
    return result


def train_agent(
    env_factory: EnvFactory,
    agent: Agent,
    *,
    evaluation_env_factory: EnvFactory | None = None,
    episodes: int = 8_000,
    explore_episodes: int = 50,
    eval_frequency: int = 200,
    eval_episodes: int = 20,
    max_steps: int = 750,
    score_limit: float = 300.0,
    checkpoint_dir: str | Path = DEFAULT_OUTPUT_DIR,
    seed: int = 42,
    episode_offset: int = 0,
) -> dict[str, Any]:
    """Episode-based training loop modeled after the ugur repo."""
    train_env = env_factory()
    eval_env = evaluation_env_factory or env_factory
    checkpoint_root = Path(checkpoint_dir)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    scores_window: deque[float] = deque(maxlen=100)
    raw_window: deque[float] = deque(maxlen=100)
    training_history: list[dict[str, Any]] = []
    evaluation_history: list[dict[str, Any]] = []
    best_raw_checkpoint = checkpoint_root / "best_raw.pt"
    best_shaped_checkpoint = checkpoint_root / "best_shaped.pt"
    last_checkpoint = checkpoint_root / "last.pt"

    best_raw_mean = -np.inf
    if best_raw_checkpoint.exists():
        checkpoint = torch.load(best_raw_checkpoint, map_location="cpu")
        metadata = checkpoint.get("metadata") or {}
        if metadata.get("eval_mean_reward") is not None:
            best_raw_mean = float(metadata["eval_mean_reward"])

    best_shaped_mean = -np.inf
    if best_shaped_checkpoint.exists():
        checkpoint = torch.load(best_shaped_checkpoint, map_location="cpu")
        metadata = checkpoint.get("metadata") or {}
        if metadata.get("eval_mean_shaped_reward") is not None:
            best_shaped_mean = float(metadata["eval_mean_shaped_reward"])

    logger.info(
        "Custom hardcore trening | episodes={} | episode_offset={} | explore_ep={} | eval_every={} | eval_ep={} | max_steps={}",
        episodes,
        episode_offset,
        explore_episodes,
        eval_frequency,
        eval_episodes,
        max_steps,
    )

    try:
        for local_episode in range(1, int(episodes) + 1):
            episode = int(episode_offset + local_episode)
            observation, _ = train_env.reset(seed=seed + episode - 1)
            done = False
            steps = 0
            score = 0.0
            raw_score = 0.0
            last_losses: dict[str, float] | None = None
            agent.train_mode()

            while not done and steps < max_steps:
                action = agent.get_action(observation, explore=True)
                clipped_action = np.clip(action, train_env.action_space.low, train_env.action_space.high)
                next_observation, shaped_reward, terminated, truncated, info = train_env.step(clipped_action)
                transition_done = bool(terminated or truncated)
                raw_reward = float(info.get("raw_reward", shaped_reward))
                agent.observe(
                    observation,
                    clipped_action,
                    float(shaped_reward),
                    next_observation,
                    transition_done,
                )

                observation = next_observation
                score += float(shaped_reward)
                raw_score += raw_reward
                steps += 1
                done = terminated or truncated
                agent.step_end()

            if episode > int(explore_episodes):
                agent.episode_end()
                for _ in range(steps):
                    losses = agent.learn_one_step()
                    if losses is not None:
                        last_losses = losses

            scores_window.append(score)
            raw_window.append(raw_score)
            average_score_100 = float(np.mean(scores_window))
            average_raw_score_100 = float(np.mean(raw_window))
            episode_summary = {
                "episode": int(episode),
                "reward": float(raw_score),
                "shaped_reward": float(score),
                "rolling_mean_reward": average_raw_score_100,
                "rolling_mean_shaped_reward": average_score_100,
                "steps": int(steps),
                "losses": {} if last_losses is None else dict(last_losses),
            }
            training_history.append(episode_summary)
            logger.info(
                "Episode {} | raw={:.2f} | shaped={:.2f} | avg100_raw={:.2f} | steps={}",
                episode,
                raw_score,
                score,
                average_raw_score_100,
                steps,
            )

            if episode % int(eval_frequency) == 0 or average_score_100 > float(score_limit):
                evaluation = evaluate_agent(
                    eval_env,
                    agent,
                    episodes=eval_episodes,
                    max_steps=max_steps,
                    seed_start=seed + 100_000 + episode * eval_episodes,
                )
                evaluation["episode"] = int(episode)
                evaluation_history.append(evaluation)
                checkpoint_path = checkpoint_root / f"ep{episode}.pt"

                metadata = AgentMetadata(
                    algorithm=agent.algorithm,
                    backbone=agent.backbone,
                    history_length=agent.history_length,
                    episode=episode,
                    eval_mean_reward=float(evaluation["mean_reward"]),
                    eval_mean_shaped_reward=float(evaluation["mean_shaped_reward"]),
                )
                agent.save_checkpoint(checkpoint_path, metadata=metadata)
                agent.save_checkpoint(last_checkpoint, metadata=metadata)
                logger.info(
                    "Eval @ episode {} | raw_mean={:.2f} | shaped_mean={:.2f} | ckpt={}",
                    episode,
                    evaluation["mean_reward"],
                    evaluation["mean_shaped_reward"],
                    checkpoint_path,
                )

                if float(evaluation["mean_reward"]) > best_raw_mean:
                    best_raw_mean = float(evaluation["mean_reward"])
                    agent.save_checkpoint(best_raw_checkpoint, metadata=metadata)
                    logger.info("Novi best_raw checkpoint | {:.2f}", best_raw_mean)

                if float(evaluation["mean_shaped_reward"]) > best_shaped_mean:
                    best_shaped_mean = float(evaluation["mean_shaped_reward"])
                    agent.save_checkpoint(best_shaped_checkpoint, metadata=metadata)
                    logger.info("Novi best_shaped checkpoint | {:.2f}", best_shaped_mean)

                if average_score_100 > float(score_limit):
                    logger.info("Dosegnut score_limit na episode {}. Zavrsavam trening.", episode)
                    break

        if not last_checkpoint.exists():
            metadata = AgentMetadata(
                algorithm=agent.algorithm,
                backbone=agent.backbone,
                history_length=agent.history_length,
                episode=episode_offset + len(training_history) if training_history else episode_offset or None,
            )
            agent.save_checkpoint(last_checkpoint, metadata=metadata)

        return {
            "episodes_completed": int(episode_offset + len(training_history)),
            "episodes_ran_this_session": int(len(training_history)),
            "checkpoint_dir": str(checkpoint_root),
            "best_raw_checkpoint": str(best_raw_checkpoint) if best_raw_checkpoint.exists() else None,
            "best_shaped_checkpoint": str(best_shaped_checkpoint) if best_shaped_checkpoint.exists() else None,
            "last_checkpoint": str(last_checkpoint) if last_checkpoint.exists() else None,
            "best_eval_mean_reward": None if best_raw_mean == -np.inf else float(best_raw_mean),
            "best_eval_mean_shaped_reward": None if best_shaped_mean == -np.inf else float(best_shaped_mean),
            "training_history": training_history,
            "evaluation_history": evaluation_history,
        }
    finally:
        train_env.close()


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def checkpoint_root(self) -> Path:
        return self.root

    @property
    def training_history(self) -> Path:
        return self.root / "training_history.json"

    @property
    def evaluation_history(self) -> Path:
        return self.root / "evaluation_history.json"

    @property
    def videos_dir(self) -> Path:
        return self.root / "videos"

    def log_path(self, mode: str) -> Path:
        return self.root / f"{mode_token(mode)}.log"

    def summary_path(self, mode: str) -> Path:
        return self.root / f"{mode_token(mode)}_summary.json"


def mode_token(mode: str) -> str:
    """Builds a short filesystem-friendly token for each CLI mode."""
    return "test100" if mode == "test-100" else mode


def is_run_dir(path: Path) -> bool:
    """Detects whether a directory already looks like a resolved run root."""
    direct_markers = {
        "best_raw.pt",
        "best_shaped.pt",
        "last.pt",
        "train.log",
        "test.log",
        "test100.log",
        "train_summary.json",
        "test_summary.json",
        "test100_summary.json",
        "training_history.json",
        "evaluation_history.json",
    }
    if not path.exists():
        return False
    if (path / "checkpoints").exists():
        return True
    if (path / "videos").exists():
        return True
    return any((path / marker).exists() for marker in direct_markers)


def build_run_paths(root: Path) -> RunPaths:
    """Returns the canonical file layout for a single run folder."""
    return RunPaths(root=root)


def resolve_run_output_dir(base_output_dir: Path, run_name: str) -> Path:
    """Accepts either a base experiments dir or an already-resolved run dir."""
    if is_run_dir(base_output_dir):
        return base_output_dir
    return base_output_dir / run_name


def find_checkpoint_candidates(base_output_dir: Path, checkpoint: str) -> list[Path]:
    """Searches recursively for alias checkpoints when the run dir is ambiguous."""
    if checkpoint not in CHECKPOINT_ALIASES or not base_output_dir.exists():
        return []
    checkpoint_name = f"{checkpoint}.pt"
    return sorted(path.resolve() for path in base_output_dir.rglob(checkpoint_name))


def resolve_checkpoint_path(output_dir: Path, checkpoint: str, *, search_root: Path | None = None) -> Path:
    """Resolves checkpoint aliases into actual files."""
    if checkpoint in CHECKPOINT_ALIASES:
        alias_candidates = [
            output_dir / f"{checkpoint}.pt",
            output_dir / "checkpoints" / f"{checkpoint}.pt",
        ]
        for direct_path in alias_candidates:
            if direct_path.exists():
                return direct_path
        direct_path = alias_candidates[0]
        if search_root is None:
            return direct_path
        candidates = find_checkpoint_candidates(search_root, checkpoint)
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            formatted = "\n".join(f"- {candidate}" for candidate in candidates[:10])
            raise FileNotFoundError(
                "Pronasao sam vise checkpoint kandidata za alias "
                f"'{checkpoint}' ispod '{search_root}'. Prosledi pun --checkpoint put ili "
                f"precizan --output-dir do run foldera.\n{formatted}"
            )
        return direct_path
    return Path(checkpoint)


def resolve_history_length(algo: str, history_length: int | None) -> int:
    """Uses ugur-like defaults when history length is not explicitly set."""
    if history_length is not None:
        return max(int(history_length), 1)
    return 12 if algo == "sac" else 6


def format_run_value(value: float) -> str:
    """Formats floats into short filesystem-friendly tokens."""
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def resolve_video_label(checkpoint: str) -> str:
    """Builds a filesystem-friendly checkpoint label for video outputs."""
    if checkpoint in {"best_raw", "best_shaped", "last"}:
        return checkpoint
    return Path(checkpoint).stem or "checkpoint"


def build_env_factory(
    *,
    env_id: str,
    history_length: int,
    frame_skip: int,
    fall_penalty: float,
    anti_stall: bool,
    stall_check_window: int,
    stall_grace_steps: int,
    stall_min_progress: float,
    stall_patience: int,
    stall_penalty: float,
    render_mode: str | None = None,
):
    """Creates a reusable env factory closure."""
    return lambda: make_hardcore_env(
        env_id=env_id,
        history_length=history_length,
        frame_skip=frame_skip,
        fall_penalty=fall_penalty,
        anti_stall=anti_stall,
        stall_check_window=stall_check_window,
        stall_grace_steps=stall_grace_steps,
        stall_min_progress=stall_min_progress,
        stall_patience=stall_patience,
        stall_penalty=stall_penalty,
        render_mode=render_mode,
    )


def build_agent(args: argparse.Namespace, *, state_dim: int, action_dim: int, action_low, action_high):
    """Builds the requested custom agent."""
    agent_kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "backbone": args.backbone,
        "history_length": args.history_length,
        "action_low": action_low,
        "action_high": action_high,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "gamma": args.gamma,
        "batch_size": args.batch_size,
        "buffer_size": args.buffer_size,
        "device": args.device,
    }
    if args.algo == "sac":
        return SACAgent(
            alpha=args.alpha,
            tau=args.tau,
            update_freq=args.update_freq,
            **agent_kwargs,
        )
    return TD3Agent(
        tau=args.tau,
        update_freq=args.update_freq,
        **agent_kwargs,
    )


def build_run_name(args: argparse.Namespace) -> str:
    """Creates a run name that keeps different experiment settings separate."""
    parts = [
        args.algo,
        args.backbone,
        f"h{args.history_length}",
        f"s{args.seed}",
    ]
    if float(args.lr) != DEFAULT_LR:
        parts.append(f"lr{format_run_value(args.lr)}")
    if int(args.batch_size) != DEFAULT_BATCH_SIZE:
        parts.append(f"bs{int(args.batch_size)}")
    if int(args.frame_skip) != DEFAULT_FRAME_SKIP:
        parts.append(f"fs{int(args.frame_skip)}")
    if float(args.fall_penalty) != DEFAULT_FALL_PENALTY:
        parts.append(f"fp{format_run_value(args.fall_penalty)}")
    if float(args.gamma) != DEFAULT_GAMMA:
        parts.append(f"g{format_run_value(args.gamma)}")
    if float(args.tau) != DEFAULT_TAU:
        parts.append(f"tau{format_run_value(args.tau)}")
    if args.algo == "sac" and float(args.alpha) != DEFAULT_ALPHA:
        parts.append(f"a{format_run_value(args.alpha)}")
    if args.anti_stall:
        parts.append("as")
        if int(args.stall_grace_steps) != DEFAULT_STALL_GRACE_STEPS:
            parts.append(f"g{int(args.stall_grace_steps)}")
        if int(args.stall_check_window) != DEFAULT_STALL_CHECK_WINDOW:
            parts.append(f"w{int(args.stall_check_window)}")
        if float(args.stall_min_progress) != DEFAULT_STALL_MIN_PROGRESS:
            parts.append(f"mp{format_run_value(args.stall_min_progress)}")
        if int(args.stall_patience) != DEFAULT_STALL_PATIENCE:
            parts.append(f"p{int(args.stall_patience)}")
        if float(args.stall_penalty) != DEFAULT_STALL_PENALTY:
            parts.append(f"sp{format_run_value(args.stall_penalty)}")
    return "_".join(parts)


def format_summary(summary: dict[str, Any]) -> str:
    """Builds a concise terminal summary."""
    lines = [
        "",
        "===== Custom Hardcore Port =====",
        f"Algoritam: {summary['algorithm'].upper()} | backbone: {summary['backbone'].upper()}",
        f"Okruzenje: {summary['env_id']}",
        f"Device: {summary.get('resolved_device', summary.get('device', 'unknown'))}",
        f"History: {summary['history_length']} | frame_skip: {summary['frame_skip']} | fall_penalty: {summary['fall_penalty']}",
    ]
    if summary.get("anti_stall"):
        lines.append(
            "Anti-stall: on"
            f" | grace={summary['stall_grace_steps']}"
            f" | window={summary['stall_check_window']}"
            f" | min_progress={summary['stall_min_progress']}"
            f" | patience={summary['stall_patience']}"
            f" | penalty={summary['stall_penalty']}"
        )
    if summary.get("anti_stall") and not summary.get("checkpoint_eval_anti_stall", summary.get("anti_stall")):
        lines.append("Checkpoint evaluacija: raw hardcore env bez anti-stall-a")
    if summary["mode"] == "train":
        lines.extend(
            [
                f"Epizode: {summary['episodes_completed']} | max_steps: {summary['max_steps']}",
                f"Best raw ckpt: {summary.get('best_raw_checkpoint')}",
                f"Best shaped ckpt: {summary.get('best_shaped_checkpoint')}",
                f"Last ckpt: {summary.get('last_checkpoint')}",
            ]
        )
        if summary.get("resume_from") is not None:
            lines.append(
                f"Resume: {summary['resume_from']} | start_episode={summary.get('resume_episode_offset', 0)} | session_episodes={summary.get('episodes_ran_this_session', 0)}"
            )
        if summary.get("final_eval") is not None:
            final_eval = summary["final_eval"]
            lines.extend(
                [
                    "",
                    "Finalna evaluacija izabranog checkpoint-a:",
                    f"- mean raw reward: {float(final_eval['mean_reward']):.2f}",
                    f"- mean shaped reward: {float(final_eval['mean_shaped_reward']):.2f}",
                    f"- std raw reward: {float(final_eval['std_reward']):.2f}",
                    f"- prosecan broj koraka: {float(final_eval['mean_length']):.1f}",
                    f"- best ep: reward={float(final_eval['best_episode']['reward']):.2f} | seed={int(final_eval['best_episode']['seed'])}",
                    f"- worst ep: reward={float(final_eval['worst_episode']['reward']):.2f} | seed={int(final_eval['worst_episode']['seed'])}",
                ]
            )
    else:
        evaluation = summary["evaluation"]
        lines.extend(
            [
                f"Checkpoint: {summary['checkpoint_path']}",
                "",
                "Evaluacija:",
                f"- mean raw reward: {float(evaluation['mean_reward']):.2f}",
                f"- mean shaped reward: {float(evaluation['mean_shaped_reward']):.2f}",
                f"- std raw reward: {float(evaluation['std_reward']):.2f}",
                f"- prosecan broj koraka: {float(evaluation['mean_length']):.1f}",
            ]
        )
        if evaluation.get("video_files"):
            lines.append(f"- video fajlova: {len(evaluation['video_files'])}")
    lines.extend(
        [
            "",
            "Artefakti:",
            f"- log: {summary['log_file']}",
            f"- summary: {summary['summary_file']}",
        ]
    )
    if summary.get("video_dir") is not None:
        lines.append(f"- video dir: {summary['video_dir']}")
    return "\n".join(lines)


def main() -> None:
    """Main CLI entrypoint."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="{time:HH:mm:ss} | {level:<8} | {message}",
        level="INFO",
    )

    parser = argparse.ArgumentParser(
        description="Faithful-ish ugur-style custom SAC/TD3 port for BipedalWalkerHardcore-v3.",
    )
    parser.add_argument("--mode", choices=("train", "test", "test-100"), default="train")
    parser.add_argument("--algo", choices=("sac", "td3"), default="sac")
    parser.add_argument("--backbone", choices=("lstm", "transformer"), default="lstm")
    parser.add_argument("--env-id", default="BipedalWalkerHardcore-v3")
    parser.add_argument("--history-length", type=int, default=None)
    parser.add_argument("--frame-skip", type=int, default=DEFAULT_FRAME_SKIP)
    parser.add_argument("--fall-penalty", type=float, default=DEFAULT_FALL_PENALTY)
    parser.add_argument("--anti-stall", action="store_true")
    parser.add_argument("--stall-check-window", type=int, default=DEFAULT_STALL_CHECK_WINDOW)
    parser.add_argument("--stall-grace-steps", type=int, default=DEFAULT_STALL_GRACE_STEPS)
    parser.add_argument("--stall-min-progress", type=float, default=DEFAULT_STALL_MIN_PROGRESS)
    parser.add_argument("--stall-patience", type=int, default=DEFAULT_STALL_PATIENCE)
    parser.add_argument("--stall-penalty", type=float, default=DEFAULT_STALL_PENALTY)
    parser.add_argument("--episodes", type=int, default=8_000)
    parser.add_argument("--explore-episodes", type=int, default=50)
    parser.add_argument("--eval-frequency", type=int, default=200)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--final-eval-episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=750)
    parser.add_argument("--score-limit", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=DEFAULT_DEVICE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--update-freq", type=int, default=None)
    parser.add_argument("--checkpoint", default="best_raw")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-episodes", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    args = parser.parse_args()

    if args.update_freq is None:
        args.update_freq = 1 if args.algo == "sac" else 2
    args.history_length = resolve_history_length(args.algo, args.history_length)

    run_name = build_run_name(args)
    base_output_dir = args.output_dir
    output_dir = resolve_run_output_dir(base_output_dir, run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_paths = build_run_paths(output_dir)
    log_path = run_paths.log_path(args.mode)
    file_sink_id = logger.add(
        log_path,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
        level="INFO",
        encoding="utf-8",
        mode="w",
    )

    probe_env = make_hardcore_env(
        env_id=args.env_id,
        history_length=args.history_length,
        frame_skip=args.frame_skip,
        fall_penalty=args.fall_penalty,
        anti_stall=args.anti_stall,
        stall_check_window=args.stall_check_window,
        stall_grace_steps=args.stall_grace_steps,
        stall_min_progress=args.stall_min_progress,
        stall_patience=args.stall_patience,
        stall_penalty=args.stall_penalty,
    )
    try:
        state_dim = int(probe_env.observation_space.shape[-1])
        action_dim = int(probe_env.action_space.shape[-1])
        action_low = probe_env.action_space.low
        action_high = probe_env.action_space.high
    finally:
        probe_env.close()

    agent = build_agent(
        args,
        state_dim=state_dim,
        action_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
    )
    resolved_device = str(agent.device)
    resume_episode_offset = 0
    resume_checkpoint_path = None

    try:
        logger.info(
            "Pokretanje custom hardcore porta | mode={} | algo={} | backbone={} | history={} | frame_skip={} | fall_penalty={} | lr={} | batch={} | gamma={} | alpha={} | tau={} | requested_device={} | resolved_device={}",
            args.mode,
            args.algo,
            args.backbone,
            args.history_length,
            args.frame_skip,
            args.fall_penalty,
            args.lr,
            args.batch_size,
            args.gamma,
            args.alpha,
            args.tau,
            args.device,
            resolved_device,
        )

        train_history_path = run_paths.training_history
        eval_history_path = run_paths.evaluation_history

        env_factory = build_env_factory(
            env_id=args.env_id,
            history_length=args.history_length,
            frame_skip=args.frame_skip,
            fall_penalty=args.fall_penalty,
            anti_stall=args.anti_stall,
            stall_check_window=args.stall_check_window,
            stall_grace_steps=args.stall_grace_steps,
            stall_min_progress=args.stall_min_progress,
            stall_patience=args.stall_patience,
            stall_penalty=args.stall_penalty,
        )
        checkpoint_eval_anti_stall = False
        checkpoint_eval_env_factory = build_env_factory(
            env_id=args.env_id,
            history_length=args.history_length,
            frame_skip=args.frame_skip,
            fall_penalty=args.fall_penalty,
            anti_stall=checkpoint_eval_anti_stall,
            stall_check_window=args.stall_check_window,
            stall_grace_steps=args.stall_grace_steps,
            stall_min_progress=args.stall_min_progress,
            stall_patience=args.stall_patience,
            stall_penalty=args.stall_penalty,
        )
        test_eval_env_factory = build_env_factory(
            env_id=args.env_id,
            history_length=args.history_length,
            frame_skip=args.frame_skip,
            fall_penalty=args.fall_penalty,
            anti_stall=args.anti_stall,
            stall_check_window=args.stall_check_window,
            stall_grace_steps=args.stall_grace_steps,
            stall_min_progress=args.stall_min_progress,
            stall_patience=args.stall_patience,
            stall_penalty=args.stall_penalty,
            render_mode="rgb_array" if args.record_video else None,
        )
        final_eval_env_factory = build_env_factory(
            env_id=args.env_id,
            history_length=args.history_length,
            frame_skip=args.frame_skip,
            fall_penalty=args.fall_penalty,
            anti_stall=checkpoint_eval_anti_stall,
            stall_check_window=args.stall_check_window,
            stall_grace_steps=args.stall_grace_steps,
            stall_min_progress=args.stall_min_progress,
            stall_patience=args.stall_patience,
            stall_penalty=args.stall_penalty,
            render_mode="rgb_array" if args.record_video else None,
        )
        if args.anti_stall:
            logger.info(
                "Trening koristi anti-stall, ali checkpoint/final evaluacija ide bez anti-stall-a da raw reward ostane uporediv."
            )
        if args.mode == "train" and args.resume_from:
            resume_checkpoint_path = resolve_checkpoint_path(
                output_dir,
                args.resume_from,
                search_root=base_output_dir,
            )
            if not resume_checkpoint_path.exists():
                raise FileNotFoundError(f"Resume checkpoint nije pronadjen: {resume_checkpoint_path}")
            resume_state = agent.load_checkpoint(resume_checkpoint_path)
            resume_metadata = resume_state.get("metadata") or {}
            resume_episode_offset = int(resume_metadata.get("episode") or 0)
            logger.info(
                "Warm-start trening iz checkpoint-a | checkpoint={} | previous_episode={}",
                resume_checkpoint_path,
                resume_episode_offset,
            )
        checkpoint_label = resolve_video_label(args.checkpoint)
        video_dir = (
            run_paths.videos_dir / f"{mode_token(args.mode)}_{checkpoint_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if args.record_video
            else None
        )

        if args.mode == "train":
            training_result = train_agent(
                env_factory,
                agent,
                evaluation_env_factory=checkpoint_eval_env_factory,
                episodes=args.episodes,
                explore_episodes=args.explore_episodes,
                eval_frequency=args.eval_frequency,
                eval_episodes=args.eval_episodes,
                max_steps=args.max_steps,
                score_limit=args.score_limit,
                checkpoint_dir=run_paths.checkpoint_root,
                seed=args.seed,
                episode_offset=resume_episode_offset,
            )
            train_history_path.write_text(
                json.dumps(training_result["training_history"], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            eval_history_path.write_text(
                json.dumps(training_result["evaluation_history"], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            final_eval = None
            checkpoint_path = resolve_checkpoint_path(output_dir, args.checkpoint, search_root=base_output_dir)
            if checkpoint_path.exists():
                eval_agent = build_agent(
                    args,
                    state_dim=state_dim,
                    action_dim=action_dim,
                    action_low=action_low,
                    action_high=action_high,
                )
                eval_agent.load_checkpoint(checkpoint_path)
                final_eval = evaluate_agent(
                    final_eval_env_factory,
                    eval_agent,
                    episodes=args.final_eval_episodes,
                    max_steps=args.max_steps,
                    seed_start=args.seed + 500_000,
                    video_folder=video_dir,
                    video_prefix=f"{args.algo}_{args.backbone}_{checkpoint_label}",
                    video_episodes=args.video_episodes,
                )
            else:
                logger.info("Trazeni checkpoint za finalnu evaluaciju ne postoji | {}", checkpoint_path)

            summary = {
                "mode": "train",
                "algorithm": args.algo,
                "backbone": args.backbone,
                "env_id": args.env_id,
                "requested_device": args.device,
                "resolved_device": resolved_device,
                "history_length": int(args.history_length),
                "frame_skip": int(args.frame_skip),
                "fall_penalty": float(args.fall_penalty),
                "anti_stall": bool(args.anti_stall),
                "resume_from": None if resume_checkpoint_path is None else str(resume_checkpoint_path),
                "resume_episode_offset": int(resume_episode_offset),
                "checkpoint_eval_anti_stall": bool(checkpoint_eval_anti_stall),
                "final_eval_anti_stall": bool(checkpoint_eval_anti_stall),
                "stall_check_window": int(args.stall_check_window),
                "stall_grace_steps": int(args.stall_grace_steps),
                "stall_min_progress": float(args.stall_min_progress),
                "stall_patience": int(args.stall_patience),
                "stall_penalty": float(args.stall_penalty),
                "episodes_completed": int(training_result["episodes_completed"]),
                "episodes_ran_this_session": int(training_result.get("episodes_ran_this_session", 0)),
                "max_steps": int(args.max_steps),
                "best_raw_checkpoint": training_result["best_raw_checkpoint"],
                "best_shaped_checkpoint": training_result["best_shaped_checkpoint"],
                "last_checkpoint": training_result["last_checkpoint"],
                "best_eval_mean_reward": training_result["best_eval_mean_reward"],
                "best_eval_mean_shaped_reward": training_result["best_eval_mean_shaped_reward"],
                "training_history_file": str(train_history_path),
                "evaluation_history_file": str(eval_history_path),
                "selected_checkpoint": str(
                    resolve_checkpoint_path(output_dir, args.checkpoint, search_root=base_output_dir)
                ),
                "final_eval": final_eval,
                "log_file": str(log_path),
                "video_dir": None if final_eval is None else final_eval.get("video_folder"),
            }
        else:
            checkpoint_path = resolve_checkpoint_path(output_dir, args.checkpoint, search_root=base_output_dir)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint nije pronadjen: {checkpoint_path}")
            agent.load_checkpoint(checkpoint_path)
            eval_episodes = args.eval_episodes if args.mode == "test" else args.final_eval_episodes
            evaluation = evaluate_agent(
                test_eval_env_factory,
                agent,
                episodes=eval_episodes,
                max_steps=args.max_steps,
                seed_start=args.seed + 700_000,
                video_folder=video_dir,
                video_prefix=f"{args.algo}_{args.backbone}_{checkpoint_label}",
                video_episodes=args.video_episodes,
            )
            summary = {
                "mode": args.mode,
                "algorithm": args.algo,
                "backbone": args.backbone,
                "env_id": args.env_id,
                "requested_device": args.device,
                "resolved_device": resolved_device,
                "history_length": int(args.history_length),
                "frame_skip": int(args.frame_skip),
                "fall_penalty": float(args.fall_penalty),
                "anti_stall": bool(args.anti_stall),
                "stall_check_window": int(args.stall_check_window),
                "stall_grace_steps": int(args.stall_grace_steps),
                "stall_min_progress": float(args.stall_min_progress),
                "stall_patience": int(args.stall_patience),
                "stall_penalty": float(args.stall_penalty),
                "checkpoint_path": str(checkpoint_path),
                "evaluation": evaluation,
                "log_file": str(log_path),
                "video_dir": evaluation.get("video_folder"),
            }

        summary_path = run_paths.summary_path(args.mode)
        summary["summary_file"] = str(summary_path)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(format_summary(summary), file=sys.stderr, flush=True)
    finally:
        logger.remove(file_sink_id)


if __name__ == "__main__":
    main()
