"""Environment helpers for the dedicated hardcore port."""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np


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
    """Faithful-ish port of the ugur hardcore helper wrapper."""

    def __init__(
        self,
        env: gym.Env,
        *,
        frame_skip: int = 2,
        fall_penalty: float = -10.0,
        anti_stall: bool = False,
        stall_check_window: int = 40,
        stall_grace_steps: int = 80,
        stall_min_progress: float = 0.35,
        stall_patience: int = 2,
        stall_penalty: float = -20.0,
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
    frame_skip: int = 2,
    fall_penalty: float = -10.0,
    anti_stall: bool = False,
    stall_check_window: int = 40,
    stall_grace_steps: int = 80,
    stall_min_progress: float = 0.35,
    stall_patience: int = 2,
    stall_penalty: float = -20.0,
    render_mode: str | None = None,
) -> gym.Env:
    """Creates the custom hardcore environment used by the port."""
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
