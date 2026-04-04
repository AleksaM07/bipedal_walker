"""Training and evaluation loops for the dedicated hardcore port."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from loguru import logger

from .agents import AgentMetadata, SACAgent, TD3Agent

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
    episodes: int = 8_000,
    explore_episodes: int = 50,
    eval_frequency: int = 200,
    eval_episodes: int = 20,
    max_steps: int = 750,
    score_limit: float = 300.0,
    checkpoint_dir: str | Path = Path("artifacts") / "custom_hardcore" / "checkpoints",
    seed: int = 42,
) -> dict[str, Any]:
    """Episode-based training loop modeled after the ugur repo."""
    train_env = env_factory()
    checkpoint_root = Path(checkpoint_dir)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    scores_window: deque[float] = deque(maxlen=100)
    raw_window: deque[float] = deque(maxlen=100)
    training_history: list[dict[str, Any]] = []
    evaluation_history: list[dict[str, Any]] = []
    best_raw_mean = -np.inf
    best_shaped_mean = -np.inf

    best_raw_checkpoint = checkpoint_root / "best_raw.pt"
    best_shaped_checkpoint = checkpoint_root / "best_shaped.pt"
    last_checkpoint = checkpoint_root / "last.pt"

    logger.info(
        "Custom hardcore trening | episodes={} | explore_ep={} | eval_every={} | eval_ep={} | max_steps={}",
        episodes,
        explore_episodes,
        eval_frequency,
        eval_episodes,
        max_steps,
    )

    try:
        for episode in range(1, int(episodes) + 1):
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
                dead = bool(info.get("dead", False))
                raw_reward = float(info.get("raw_reward", shaped_reward))
                agent.observe(
                    observation,
                    clipped_action,
                    float(shaped_reward),
                    next_observation,
                    dead,
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
                    env_factory,
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
                episode=len(training_history) if training_history else None,
            )
            agent.save_checkpoint(last_checkpoint, metadata=metadata)

        return {
            "episodes_completed": int(len(training_history)),
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
