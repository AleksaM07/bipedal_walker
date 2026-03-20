"""SAC primer: rucna mini-verzija zbog ucenja + SB3 verzija za pravi trening."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from sb3_workflow import train_and_evaluate_sb3

try:
    from stable_baselines3 import SAC
except Exception:  # pragma: no cover - optional dependency
    SAC = None


def build_mlp(input_dim: int, output_dim: int, hidden_dim: int = 128) -> nn.Sequential:
    # Obicna neuronska mreza sa dva skrivena sloja.
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class GaussianActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()

        # Backbone prvo obradi observation u "skrivene" feature-e.
        self.backbone = build_mlp(obs_dim, hidden_dim, hidden_dim)

        # Mean i log_std opisuju Gaussovu raspodelu akcija.
        self.mean = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Linear(hidden_dim, act_dim)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(observation)
        mean = self.mean(hidden)

        # Clamp sprecava da std ode u totalni haos.
        log_std = torch.clamp(self.log_std(hidden), min=-5.0, max=2.0)
        return mean, log_std

    def sample(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # SAC koristi stohasticku politiku, pa uzorkujemo akciju.
        mean, log_std = self.forward(observation)
        std = log_std.exp()
        distribution = Normal(mean, std)

        # rsample omogucava backprop kroz uzorkovanje.
        pre_tanh_action = distribution.rsample()
        action = torch.tanh(pre_tanh_action)

        # Posle tanh-a raspodela vise nije "cista" Gaussova,
        # pa moramo da dodamo correction term za log_prob.
        log_prob = distribution.log_prob(pre_tanh_action) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()

        # Critic gleda i stanje i akciju, pa im zbirno ulazimo u mrezu.
        self.network = build_mlp(obs_dim + act_dim, 1, hidden_dim)

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        critic_input = torch.cat([observation, action], dim=-1)
        return self.network(critic_input).squeeze(-1)


@dataclass
class ReplayBatch:
    # Batch podataka koji izgleda kao mali isecek replay buffer-a.
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    # target <- malo staro + malo novo
    # Ovo drzi target mreze stabilnijim od direktnog kopiranja na svaki korak.
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1.0 - tau).add_(tau * source_param.data)


def collect_random_batch(env, batch_size: int, seed: int | None = None) -> ReplayBatch:
    # Za edukativni demo ovde NE treniramo dugo.
    # Samo skupimo random korake i posle nad njima pokazemo jedan SAC update.
    observation, _ = env.reset(seed=seed)

    observations = []
    actions = []
    rewards = []
    next_observations = []
    dones = []

    for _ in range(batch_size):
        # Random akcija, cisto da imamo neki batch tranzicija.
        action = env.action_space.sample().astype(np.float32)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        observations.append(observation)
        actions.append(action)
        rewards.append(float(reward))
        next_observations.append(next_observation)
        dones.append(float(done))

        if done:
            # Ako je epizoda pukla ili zavrsila, pocinjemo novu.
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


def sac_update(
    actor: GaussianActor,
    critic_1: Critic,
    critic_2: Critic,
    target_critic_1: Critic,
    target_critic_2: Critic,
    batch: ReplayBatch,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    gamma: float = 0.99,
    alpha: float = 0.2,
    tau: float = 0.005,
) -> dict[str, float]:
    with torch.no_grad():
        # 1. Iz sledeceg stanja uzorkujemo sledecu akciju.
        next_actions, next_log_probs = actor.sample(batch.next_observations)
        target_q1 = target_critic_1(batch.next_observations, next_actions)
        target_q2 = target_critic_2(batch.next_observations, next_actions)

        # SAC voli akcije koje su i dobre po Q i dovoljno "raznovrsne".
        # Zato se pojavljuje - alpha * log_prob.
        target_q = torch.min(target_q1, target_q2) - alpha * next_log_probs
        td_target = batch.rewards + gamma * (1.0 - batch.dones) * target_q

    # 2. Critic mreze treba da pogode taj td_target.
    current_q1 = critic_1(batch.observations, batch.actions)
    current_q2 = critic_2(batch.observations, batch.actions)
    critic_loss = torch.mean((current_q1 - td_target) ** 2) + torch.mean((current_q2 - td_target) ** 2)

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # 3. Sada biramo akcije po trenutnom actor-u.
    sampled_actions, log_probs = actor.sample(batch.observations)
    q1_pi = critic_1(batch.observations, sampled_actions)
    q2_pi = critic_2(batch.observations, sampled_actions)
    min_q_pi = torch.min(q1_pi, q2_pi)

    # Actor zeli:
    # - veliku Q vrednost
    # - dovoljno entropije / istrazivanja
    # U loss formi to ispadne alpha * log_prob - Q.
    actor_loss = torch.mean(alpha * log_probs - min_q_pi)

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    # 4. Polako pomeramo target critic mreze ka glavnim critic mrezama.
    soft_update(target_critic_1, critic_1, tau=tau)
    soft_update(target_critic_2, critic_2, tau=tau)

    return {
        "critic_loss": float(critic_loss.item()),
        "actor_loss": float(actor_loss.item()),
        "target_q_mean": float(td_target.mean().item()),
        "log_prob_mean": float(log_probs.mean().item()),
    }


def run_manual_sac_demo(env_factory, batch_size: int = 256, learning_rate: float = 3e-4, seed: int = 0) -> dict[str, float]:
    """Mali SAC demo: skupi random batch i uradi jedan update."""

    env = env_factory()
    try:
        # Koliko brojeva ulazi i izlazi iz mreze.
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Glavne mreze.
        actor = GaussianActor(obs_dim=obs_dim, act_dim=act_dim)
        critic_1 = Critic(obs_dim=obs_dim, act_dim=act_dim)
        critic_2 = Critic(obs_dim=obs_dim, act_dim=act_dim)

        # Target mreze se koriste za stabilniji racun td_target-a.
        target_critic_1 = Critic(obs_dim=obs_dim, act_dim=act_dim)
        target_critic_2 = Critic(obs_dim=obs_dim, act_dim=act_dim)
        target_critic_1.load_state_dict(critic_1.state_dict())
        target_critic_2.load_state_dict(critic_2.state_dict())

        # Jedan optimizer za actor, jedan za oba critic-a.
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
        critic_optimizer = torch.optim.Adam(list(critic_1.parameters()) + list(critic_2.parameters()), lr=learning_rate)

        # Skupimo batch i odradimo jedan SAC update.
        batch = collect_random_batch(env=env, batch_size=batch_size, seed=seed)
        metrics = sac_update(
            actor=actor,
            critic_1=critic_1,
            critic_2=critic_2,
            target_critic_1=target_critic_1,
            target_critic_2=target_critic_2,
            batch=batch,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
        )
        metrics["reward_mean"] = float(batch.rewards.mean().item())
        return metrics
    finally:
        env.close()


def run_library_sac(
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
    if SAC is None:
        raise ImportError("stable_baselines3 nije instaliran. Instaliraj stable-baselines3 da pokrenes library SAC.")

    # Ovo je normalna, prakticna SAC putanja preko gotove biblioteke.
    return train_and_evaluate_sb3(
        algorithm_name="sac",
        algorithm_cls=SAC,
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
