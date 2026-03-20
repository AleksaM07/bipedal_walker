"""Najprostiji baseline: pusti nasumicne akcije i vidi koliko je lose."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class BaselineStats:
    # label = ime baseline varijante
    # mean_reward/std_reward = prosecni rezultat
    # rewards = sirovi rezultati po epizodama
    label: str
    mean_reward: float
    std_reward: float
    rewards: list[float]


def run_episode(env, policy_fn: Callable[[np.ndarray], np.ndarray], seed: int | None = None) -> float:
    """Pokreni jednu epizodu i vrati ukupan reward."""
    observation, _ = env.reset(seed=seed)
    done = False
    total_reward = 0.0

    while not done:
        # policy_fn kaze koju akciju zelimo za trenutno stanje.
        action = np.asarray(policy_fn(observation), dtype=np.float32)
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        done = terminated or truncated

    return total_reward


def manual_random_policy(env):
    """Rucni random baseline: mi sami uzorkujemo akciju."""

    def policy(_observation: np.ndarray) -> np.ndarray:
        low = env.action_space.low
        high = env.action_space.high

        # Najprostija moguca ideja:
        # za svaku komponentu akcije biramo slucajan broj iz dozvoljenog opsega.
        return np.random.uniform(low=low, high=high).astype(np.float32)

    return policy


def gym_random_policy(env):
    """Library random baseline: prepusti Gymnasium-u da bira random akciju."""

    def policy(_observation: np.ndarray) -> np.ndarray:
        return env.action_space.sample().astype(np.float32)

    return policy


def evaluate_policy(env_factory, policy_builder, episodes: int = 5, seed_start: int = 0, label: str = "") -> BaselineStats:
    # Ova funkcija vrti vise epizoda i pravi statistiku.
    rewards: list[float] = []

    for episode_idx in range(episodes):
        env = env_factory()
        try:
            # policy_builder prima env i vraca funkciju koja bira akcije.
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
    """Uporedi nasa random akcije vs Gymnasium random akcije."""

    # "manual" = mi sami uzorkujemo iz action range.
    manual = evaluate_policy(
        env_factory=env_factory,
        policy_builder=manual_random_policy,
        episodes=episodes,
        seed_start=seed_start,
        label="manual_random",
    )

    # "library" = koristimo env.action_space.sample().
    library = evaluate_policy(
        env_factory=env_factory,
        policy_builder=gym_random_policy,
        episodes=episodes,
        seed_start=seed_start,
        label="gymnasium_random",
    )
    return {"manual": manual, "library": library}
