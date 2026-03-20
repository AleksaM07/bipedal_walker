"""Educational TD3 implementation plus a simple Stable-Baselines3 workflow."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch import nn

from sb3_workflow import train_and_evaluate_sb3

try:
    from stable_baselines3 import TD3
except Exception:  # pragma: no cover - optional dependency
    TD3 = None


def build_mlp(input_dim: int, output_dim: int, hidden_dim: int = 128) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class DeterministicActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = build_mlp(obs_dim, act_dim, hidden_dim)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.network(observation))


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = build_mlp(obs_dim + act_dim, 1, hidden_dim)

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([observation, action], dim=-1)).squeeze(-1)


@dataclass
class ReplayBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1.0 - tau).add_(tau * source_param.data)


def collect_random_batch(env, batch_size: int, seed: int | None = None) -> ReplayBatch:
    observation, _ = env.reset(seed=seed)

    observations = []
    actions = []
    rewards = []
    next_observations = []
    dones = []

    for _ in range(batch_size):
        action = env.action_space.sample().astype(np.float32)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        observations.append(observation)
        actions.append(action)
        rewards.append(float(reward))
        next_observations.append(next_observation)
        dones.append(float(done))

        if done:
            observation, _ = env.reset()
        else:
            observation = next_observation

    return ReplayBatch(
        observations=torch.as_tensor(np.asarray(observations), dtype=torch.float32),
        actions=torch.as_tensor(np.asarray(actions), dtype=torch.float32),
        rewards=torch.as_tensor(np.asarray(rewards), dtype=torch.float32),
        next_observations=torch.as_tensor(np.asarray(next_observations), dtype=torch.float32),
        dones=torch.as_tensor(np.asarray(dones), dtype=torch.float32),
    )


def td3_update(
    actor: DeterministicActor,
    critic_1: Critic,
    critic_2: Critic,
    target_actor: DeterministicActor,
    target_critic_1: Critic,
    target_critic_2: Critic,
    batch: ReplayBatch,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    gamma: float = 0.99,
    tau: float = 0.005,
    policy_noise: float = 0.2,
    noise_clip: float = 0.5,
    update_actor: bool = True,
) -> dict[str, float]:
    with torch.no_grad():
        noise = torch.randn_like(batch.actions) * policy_noise
        noise = torch.clamp(noise, -noise_clip, noise_clip)
        next_actions = torch.clamp(target_actor(batch.next_observations) + noise, -1.0, 1.0)

        target_q1 = target_critic_1(batch.next_observations, next_actions)
        target_q2 = target_critic_2(batch.next_observations, next_actions)
        target_q = torch.min(target_q1, target_q2)
        td_target = batch.rewards + gamma * (1.0 - batch.dones) * target_q

    current_q1 = critic_1(batch.observations, batch.actions)
    current_q2 = critic_2(batch.observations, batch.actions)
    critic_loss = torch.mean((current_q1 - td_target) ** 2) + torch.mean((current_q2 - td_target) ** 2)

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    actor_loss_value = float("nan")
    if update_actor:
        predicted_actions = actor(batch.observations)

        # TD3 actor update is deterministic: choose actions that maximize Q1.
        actor_loss = -critic_1(batch.observations, predicted_actions).mean()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        actor_loss_value = float(actor_loss.item())

        soft_update(target_actor, actor, tau=tau)
        soft_update(target_critic_1, critic_1, tau=tau)
        soft_update(target_critic_2, critic_2, tau=tau)

    return {
        "critic_loss": float(critic_loss.item()),
        "actor_loss": actor_loss_value,
        "target_q_mean": float(td_target.mean().item()),
    }


def run_manual_td3_demo(env_factory, batch_size: int = 256, learning_rate: float = 3e-4, seed: int = 0) -> dict[str, float]:
    """Run one random-batch/update pair as an educational TD3 demo."""

    env = env_factory()
    try:
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        torch.manual_seed(seed)
        np.random.seed(seed)

        actor = DeterministicActor(obs_dim=obs_dim, act_dim=act_dim)
        critic_1 = Critic(obs_dim=obs_dim, act_dim=act_dim)
        critic_2 = Critic(obs_dim=obs_dim, act_dim=act_dim)
        target_actor = DeterministicActor(obs_dim=obs_dim, act_dim=act_dim)
        target_critic_1 = Critic(obs_dim=obs_dim, act_dim=act_dim)
        target_critic_2 = Critic(obs_dim=obs_dim, act_dim=act_dim)
        target_actor.load_state_dict(actor.state_dict())
        target_critic_1.load_state_dict(critic_1.state_dict())
        target_critic_2.load_state_dict(critic_2.state_dict())

        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
        critic_optimizer = torch.optim.Adam(list(critic_1.parameters()) + list(critic_2.parameters()), lr=learning_rate)

        batch = collect_random_batch(env=env, batch_size=batch_size, seed=seed)
        metrics = td3_update(
            actor=actor,
            critic_1=critic_1,
            critic_2=critic_2,
            target_actor=target_actor,
            target_critic_1=target_critic_1,
            target_critic_2=target_critic_2,
            batch=batch,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            update_actor=True,
        )
        metrics["reward_mean"] = float(batch.rewards.mean().item())
        return metrics
    finally:
        env.close()


def run_library_td3(
    env_id: str,
    *,
    total_timesteps: int = 50_000,
    save_path: str | None = None,
    seed: int = 0,
    eval_episodes: int = 5,
    progress_bar: bool = False,
    hardcore: bool = False,
    video_folder: str | None = None,
    video_episodes: int = 1,
) -> dict[str, object]:
    if TD3 is None:
        raise ImportError("stable_baselines3 nije instaliran. Instaliraj stable-baselines3 da pokrenes library TD3.")

    return train_and_evaluate_sb3(
        algorithm_name="td3",
        algorithm_cls=TD3,
        env_id=env_id,
        total_timesteps=total_timesteps,
        save_path=save_path,
        seed=seed,
        eval_episodes=eval_episodes,
        progress_bar=progress_bar,
        hardcore=hardcore,
        video_folder=video_folder,
        video_episodes=video_episodes,
    )
