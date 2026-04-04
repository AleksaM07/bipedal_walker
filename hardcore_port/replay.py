"""Replay buffer helpers for custom off-policy agents."""

from __future__ import annotations

import random
from collections import deque, namedtuple

import numpy as np
import torch


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
