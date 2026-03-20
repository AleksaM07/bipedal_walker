"""Small Stable-Baselines3 helpers for the real training script."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np


def make_env(env_id: str, *, hardcore: bool = False, render_mode: str | None = None) -> gym.Env:
    env_kwargs: dict[str, Any] = {}
    if hardcore:
        env_kwargs["hardcore"] = True
    if render_mode is not None:
        env_kwargs["render_mode"] = render_mode
    return gym.make(env_id, **env_kwargs)


def evaluate_model(
    model: Any,
    env_id: str,
    *,
    episodes: int = 5,
    seed: int = 0,
    hardcore: bool = False,
) -> dict[str, object]:
    if episodes < 1:
        raise ValueError("episodes must be at least 1.")

    rewards: list[float] = []
    episode_lengths: list[int] = []

    for episode_index in range(episodes):
        env = make_env(env_id, hardcore=hardcore)
        try:
            observation, _ = env.reset(seed=seed + episode_index)
            done = False
            total_reward = 0.0
            episode_length = 0

            while not done:
                action, _ = model.predict(observation, deterministic=True)
                action = np.asarray(action, dtype=np.float32)
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                episode_length += 1
                done = terminated or truncated

            rewards.append(total_reward)
            episode_lengths.append(episode_length)
        finally:
            env.close()

    return {
        "eval_episodes": int(episodes),
        "eval_deterministic": True,
        "eval_mean_reward": float(np.mean(rewards)),
        "eval_std_reward": float(np.std(rewards)),
        "eval_rewards": rewards,
        "eval_episode_lengths": episode_lengths,
        "eval_mean_episode_length": float(np.mean(episode_lengths)),
    }


def record_video(
    model: Any,
    env_id: str,
    video_folder: str | Path,
    *,
    name_prefix: str,
    episodes: int = 1,
    seed: int = 0,
    hardcore: bool = False,
) -> list[str]:
    if episodes < 1:
        raise ValueError("episodes must be at least 1.")

    video_path = Path(video_folder)
    video_path.mkdir(parents=True, exist_ok=True)

    env = gym.wrappers.RecordVideo(
        make_env(env_id, hardcore=hardcore, render_mode="rgb_array"),
        video_folder=str(video_path),
        episode_trigger=lambda episode_index: episode_index < episodes,
        name_prefix=name_prefix,
        disable_logger=True,
    )

    try:
        for episode_index in range(episodes):
            observation, _ = env.reset(seed=seed + episode_index)
            done = False

            while not done:
                action, _ = model.predict(observation, deterministic=True)
                action = np.asarray(action, dtype=np.float32)
                observation, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
    finally:
        env.close()

    return sorted(str(path) for path in video_path.glob("*.mp4"))


def train_and_evaluate_sb3(
    algorithm_name: str,
    algorithm_cls: Any,
    env_id: str,
    *,
    total_timesteps: int,
    seed: int = 0,
    save_path: str | Path | None = None,
    eval_episodes: int = 5,
    progress_bar: bool = False,
    hardcore: bool = False,
    video_folder: str | Path | None = None,
    video_episodes: int = 1,
) -> dict[str, object]:
    if total_timesteps < 1:
        raise ValueError("total_timesteps must be at least 1.")

    training_env = make_env(env_id, hardcore=hardcore)
    try:
        model = algorithm_cls("MlpPolicy", training_env, verbose=0, seed=seed)
        model.learn(total_timesteps=total_timesteps, progress_bar=progress_bar)
    finally:
        training_env.close()

    model_path = Path(save_path or f"artifacts/models/{algorithm_name}_bipedalwalker_seed{seed}").with_suffix("")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))

    summary = {
        "algorithm": algorithm_name,
        "env_id": env_id,
        "seed": seed,
        "total_timesteps": int(total_timesteps),
        "saved_model_path": str(model_path.with_suffix(".zip")),
    }
    summary.update(
        evaluate_model(
            model=model,
            env_id=env_id,
            episodes=eval_episodes,
            seed=seed,
            hardcore=hardcore,
        )
    )

    video_files: list[str] = []
    video_error: str | None = None
    if video_folder is not None and video_episodes > 0:
        try:
            video_files = record_video(
                model=model,
                env_id=env_id,
                video_folder=video_folder,
                name_prefix=f"{algorithm_name}_bipedalwalker_v3",
                episodes=video_episodes,
                seed=seed,
                hardcore=hardcore,
            )
        except Exception as error:
            video_error = str(error)

    summary["video_files"] = video_files
    summary["video_error"] = video_error
    return summary
