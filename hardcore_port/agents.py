"""Custom SAC and TD3 agents for the dedicated hardcore port."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .models import CriticNetwork, DeterministicActor, StochasticActor
from .noise import DecayingOrnsteinUhlenbeckNoise, GaussianNoise
from .replay import ReplayBuffer


def resolve_device(device: str | None) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        lr: float = 4e-4,
        weight_decay: float = 0.0,
        gamma: float = 0.98,
        alpha: float = 0.01,
        tau: float = 0.01,
        batch_size: int = 64,
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
        lr: float = 4e-4,
        weight_decay: float = 0.0,
        gamma: float = 0.98,
        tau: float = 0.01,
        batch_size: int = 64,
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
