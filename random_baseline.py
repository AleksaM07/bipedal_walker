"""Educational random baseline for BipedalWalker-v3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class BaselineStats:
    label: str
    mean_reward: float
    std_reward: float
    rewards: list[float]


def run_episode(env, policy_fn: Callable[[np.ndarray], np.ndarray], seed: int | None = None) -> float:
    """Run one episode and return the cumulative reward."""
    observation, _ = env.reset(seed=seed)
    done = False
    total_reward = 0.0

    while not done:
        action = np.asarray(policy_fn(observation), dtype=np.float32)
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        done = terminated or truncated

    return total_reward


def manual_random_policy(env):
    """Manual baseline: explicitly sample a uniformly random action."""

    def policy(_observation: np.ndarray) -> np.ndarray:
        low = env.action_space.low
        high = env.action_space.high

        # Uniform sampling is the simplest "manual" baseline:
        # each motor command is drawn independently inside the valid range.
        return np.random.uniform(low=low, high=high).astype(np.float32)

    return policy


def gym_random_policy(env):
    """Library baseline: rely on Gymnasium's official action-space sampler."""

    def policy(_observation: np.ndarray) -> np.ndarray:
        return env.action_space.sample().astype(np.float32)

    return policy


def evaluate_policy(env_factory, policy_builder, episodes: int = 5, seed_start: int = 0, label: str = "") -> BaselineStats:
    rewards: list[float] = []

    for episode_idx in range(episodes):
        env = env_factory()
        try:
            policy_fn = policy_builder(env)
            reward = run_episode(env, policy_fn=policy_fn, seed=seed_start + episode_idx)
            rewards.append(reward)
        finally:
            env.close()

    return BaselineStats(
        label=label,
        mean_reward=float(np.mean(rewards)),
        std_reward=float(np.std(rewards)),
        rewards=rewards,
    )


def compare_random_baselines(env_factory, episodes: int = 5, seed_start: int = 0) -> dict[str, BaselineStats]:
    """Compare the explicit manual sampler against Gymnasium's built-in sampler."""
    manual = evaluate_policy(
        env_factory=env_factory,
        policy_builder=manual_random_policy,
        episodes=episodes,
        seed_start=seed_start,
        label="manual_random",
    )
    library = evaluate_policy(
        env_factory=env_factory,
        policy_builder=gym_random_policy,
        episodes=episodes,
        seed_start=seed_start,
        label="gymnasium_random",
    )
    return {"manual": manual, "library": library}
