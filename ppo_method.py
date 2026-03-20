"""Educational PPO implementation plus a simple Stable-Baselines3 workflow."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from sb3_workflow import train_and_evaluate_sb3

try:
    from stable_baselines3 import PPO
except Exception:  # pragma: no cover - optional dependency
    PPO = None


def build_mlp(input_dim: int, output_dim: int, hidden_dim: int = 64) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, output_dim),
    )


class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.actor_mean = build_mlp(obs_dim, act_dim, hidden_dim)
        self.critic = build_mlp(obs_dim, 1, hidden_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, observation: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        mean = self.actor_mean(observation)
        std = torch.exp(self.log_std).expand_as(mean)
        distribution = Normal(mean, std)
        value = self.critic(observation).squeeze(-1)
        return distribution, value

    def act(self, observation: np.ndarray) -> np.ndarray:
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            distribution, _ = self.forward(observation_tensor)

            # PPO usually samples from a Gaussian policy during training.
            # tanh keeps actions inside [-1, 1], which matches BipedalWalker.
            action = torch.tanh(distribution.sample())

        return action.squeeze(0).cpu().numpy().astype(np.float32)


@dataclass
class PPORollout:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    next_value: torch.Tensor


def gaussian_log_prob(distribution: Normal, pre_tanh_action: torch.Tensor) -> torch.Tensor:
    return distribution.log_prob(pre_tanh_action).sum(dim=-1)


def collect_rollout(env, model: PPOActorCritic, rollout_steps: int, seed: int | None = None) -> PPORollout:
    observation, _ = env.reset(seed=seed)

    observations = []
    actions = []
    rewards = []
    dones = []
    log_probs = []
    values = []

    for _ in range(rollout_steps):
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            distribution, value = model(observation_tensor)
            pre_tanh_action = distribution.sample()
            action = torch.tanh(pre_tanh_action)
            log_prob = gaussian_log_prob(distribution, pre_tanh_action)

        next_observation, reward, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
        done = terminated or truncated

        observations.append(observation)
        actions.append(action.squeeze(0).cpu().numpy())
        rewards.append(float(reward))
        dones.append(float(done))
        log_probs.append(float(log_prob.item()))
        values.append(float(value.item()))

        if done:
            observation, _ = env.reset()
        else:
            observation = next_observation

    with torch.no_grad():
        next_value = model(torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0))[1]

    return PPORollout(
        observations=torch.as_tensor(np.asarray(observations), dtype=torch.float32),
        actions=torch.as_tensor(np.asarray(actions), dtype=torch.float32),
        rewards=torch.as_tensor(np.asarray(rewards), dtype=torch.float32),
        dones=torch.as_tensor(np.asarray(dones), dtype=torch.float32),
        log_probs=torch.as_tensor(np.asarray(log_probs), dtype=torch.float32),
        values=torch.as_tensor(np.asarray(values), dtype=torch.float32),
        next_value=next_value.squeeze(0),
    )


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.tensor(0.0, dtype=torch.float32)
    next_values = torch.cat([values[1:], next_value.unsqueeze(0)])

    for t in reversed(range(len(rewards))):
        mask = 1.0 - dones[t]

        # delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        td_error = rewards[t] + gamma * next_values[t] * mask - values[t]

        # GAE recursively accumulates future TD errors with decay lambda.
        last_advantage = td_error + gamma * gae_lambda * mask * last_advantage
        advantages[t] = last_advantage

    returns = advantages + values
    return advantages, returns


def ppo_update(
    model: PPOActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: PPORollout,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.0,
) -> dict[str, float]:
    advantages, returns = compute_gae(
        rewards=rollout.rewards,
        values=rollout.values,
        dones=rollout.dones,
        next_value=rollout.next_value,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    distribution, predicted_values = model(rollout.observations)

    # atanh reverses the tanh squashing so we can evaluate Gaussian log-probability.
    clipped_actions = torch.clamp(rollout.actions, -0.999, 0.999)
    pre_tanh_actions = torch.atanh(clipped_actions)
    new_log_probs = gaussian_log_prob(distribution, pre_tanh_actions)
    ratio = torch.exp(new_log_probs - rollout.log_probs)

    surrogate_1 = ratio * advantages
    surrogate_2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages

    policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
    value_loss = torch.mean((returns - predicted_values) ** 2)
    entropy = distribution.entropy().sum(dim=-1).mean()
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(entropy.item()),
        "mean_advantage": float(advantages.mean().item()),
    }


def run_manual_ppo_demo(env_factory, rollout_steps: int = 512, learning_rate: float = 3e-4, seed: int = 0) -> dict[str, float]:
    """Run one rollout/update pair as an educational PPO demo."""

    env = env_factory()
    try:
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = PPOActorCritic(obs_dim=obs_dim, act_dim=act_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        rollout = collect_rollout(env=env, model=model, rollout_steps=rollout_steps, seed=seed)
        metrics = ppo_update(model=model, optimizer=optimizer, rollout=rollout)
        metrics["reward_mean"] = float(rollout.rewards.mean().item())
        metrics["reward_sum"] = float(rollout.rewards.sum().item())
        return metrics
    finally:
        env.close()


def run_library_ppo(
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
    if PPO is None:
        raise ImportError("stable_baselines3 nije instaliran. Instaliraj stable-baselines3 da pokrenes library PPO.")

    return train_and_evaluate_sb3(
        algorithm_name="ppo",
        algorithm_cls=PPO,
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
