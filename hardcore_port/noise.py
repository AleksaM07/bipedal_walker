"""Exploration noise processes used by custom TD3."""

from __future__ import annotations

import numpy as np


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
