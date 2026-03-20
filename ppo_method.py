"""PPO primer: rucna mini-verzija zbog ucenja + SB3 verzija za pravi trening."""

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
    # MLP = obicna fully-connected mreza.
    # Ovde pravimo mali "mozak" za aktora ili kriticara.
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

        # Actor kaze koje akcije bi bile dobre.
        self.actor_mean = build_mlp(obs_dim, act_dim, hidden_dim)

        # Critic procenjuje koliko je neko stanje "dobro".
        self.critic = build_mlp(obs_dim, 1, hidden_dim)

        # PPO cesto koristi Gaussovu raspodelu nad akcijama.
        # log_std govori koliko je ta raspodela siroka.
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, observation: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        # Actor vraca srednju vrednost akcije.
        mean = self.actor_mean(observation)

        # std mora biti pozitivan, zato iz log_std idemo preko exp.
        std = torch.exp(self.log_std).expand_as(mean)
        distribution = Normal(mean, std)

        # Critic vraca V(s), tj. vrednost stanja.
        value = self.critic(observation).squeeze(-1)
        return distribution, value

    def act(self, observation: np.ndarray) -> np.ndarray:
        # Pretvaramo numpy observation u torch tensor i dodajemo batch dimenziju.
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            distribution, _ = self.forward(observation_tensor)

            # PPO tokom treninga uzorkuje akciju iz raspodele.
            # tanh je bitan jer BipedalWalker ocekuje akcije u opsegu [-1, 1].
            action = torch.tanh(distribution.sample())

        return action.squeeze(0).cpu().numpy().astype(np.float32)


@dataclass
class PPORollout:
    # Ovo je samo paket podataka koje smo skupili iz env-a.
    # Posle ih sve zajedno koristimo u PPO update koraku.
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    next_value: torch.Tensor


def gaussian_log_prob(distribution: Normal, pre_tanh_action: torch.Tensor) -> torch.Tensor:
    # log_prob govori koliko je neka akcija "verovatna" po trenutnoj politici.
    # sum(dim=-1) sabira doprinos svih komponenti akcije.
    return distribution.log_prob(pre_tanh_action).sum(dim=-1)


def collect_rollout(env, model: PPOActorCritic, rollout_steps: int, seed: int | None = None) -> PPORollout:
    # Rollout = samo odigramo vise koraka i zapamtimo sta se desilo.
    observation, _ = env.reset(seed=seed)

    # Sve skupljamo u Python liste pa ih na kraju pretvaramo u tensore.
    observations = []
    actions = []
    rewards = []
    dones = []
    log_probs = []
    values = []

    for _ in range(rollout_steps):
        # Opet dodajemo batch dimenziju jer mreza ocekuje oblik [batch, features].
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            distribution, value = model(observation_tensor)

            # Uzorkujemo akciju pre tanh-a.
            pre_tanh_action = distribution.sample()

            # Posle tanh-a akcija upada u validan opseg.
            action = torch.tanh(pre_tanh_action)
            log_prob = gaussian_log_prob(distribution, pre_tanh_action)

        # Ovaj action ide u env kao numpy niz.
        next_observation, reward, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
        done = terminated or truncated

        # Cuvamo sve sto ce nam trebati za PPO update.
        observations.append(observation)
        actions.append(action.squeeze(0).cpu().numpy())
        rewards.append(float(reward))
        dones.append(float(done))
        log_probs.append(float(log_prob.item()))
        values.append(float(value.item()))

        if done:
            # Ako je epizoda gotova, odmah krenemo novu.
            observation, _ = env.reset()
        else:
            observation = next_observation

    with torch.no_grad():
        # PPO koristi i procenu vrednosti za "sledece" stanje.
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
    # advantages = koliko je akcija bila bolja ili gora od onoga sto je critic ocekivao
    # returns = ciljna vrednost za critic mrezu
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.tensor(0.0, dtype=torch.float32)
    next_values = torch.cat([values[1:], next_value.unsqueeze(0)])

    for t in reversed(range(len(rewards))):
        # Ako je epizoda gotova, ne gledamo dalje u buducnost.
        mask = 1.0 - dones[t]

        # TD error = koliko se reward + procena buducnosti razlikuju od V(s_t).
        td_error = rewards[t] + gamma * next_values[t] * mask - values[t]

        # GAE sabira vise buducih TD gresaka, ali ih polako slabi sa gamma i lambda.
        # Ukratko: stabilnija procena advantages nego potpuno "sirov" reward.
        last_advantage = td_error + gamma * gae_lambda * mask * last_advantage
        advantages[t] = last_advantage

    # return = ono sto zelimo da critic pogodi.
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
    # 1. Iz rolloutu pravimo advantages i returns.
    advantages, returns = compute_gae(
        rewards=rollout.rewards,
        values=rollout.values,
        dones=rollout.dones,
        next_value=rollout.next_value,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    # Normalizacija cesto pomaze da PPO trening bude stabilniji.
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Pustimo sve observation-e ponovo kroz mrezu.
    distribution, predicted_values = model(rollout.observations)

    # Ranije smo sacuvali akcije posle tanh-a.
    # Da bismo dobili log_prob po Gaussovoj raspodeli, moramo da se vratimo pre tanh-a.
    clipped_actions = torch.clamp(rollout.actions, -0.999, 0.999)
    pre_tanh_actions = torch.atanh(clipped_actions)
    new_log_probs = gaussian_log_prob(distribution, pre_tanh_actions)

    # ratio govori koliko se nova politika razlikuje od stare.
    ratio = torch.exp(new_log_probs - rollout.log_probs)

    # PPO pravi dve verzije istog cilja i uzima "sigurniju" manju vrednost.
    surrogate_1 = ratio * advantages
    surrogate_2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages

    # policy_loss tera aktora da bira bolje akcije.
    policy_loss = -torch.min(surrogate_1, surrogate_2).mean()

    # value_loss tera kriticara da bolje pogadja returns.
    value_loss = torch.mean((returns - predicted_values) ** 2)

    # Entropy nagradjuje malo istrazivanja / raznovrsnosti.
    entropy = distribution.entropy().sum(dim=-1).mean()

    # Ukupan loss je kombinacija svega.
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

    # Klasicni PyTorch update korak.
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
    """Mali PPO demo: skupi rollout i uradi jedan update."""

    env = env_factory()
    try:
        # obs_dim = koliko brojeva dobijamo iz env-a
        # act_dim = koliko brojeva saljemo kao akciju
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Napravimo model i optimizer.
        model = PPOActorCritic(obs_dim=obs_dim, act_dim=act_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Skupimo iskustvo i odradimo jedan PPO update.
        rollout = collect_rollout(env=env, model=model, rollout_steps=rollout_steps, seed=seed)
        metrics = ppo_update(model=model, optimizer=optimizer, rollout=rollout)

        # Dodajemo jos par korisnih brojeva da lakse vidimo sta se desilo.
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

    # Ovo je "prava" PPO putanja za normalan trening.
    # Rucni PPO gore je samo za ucenje ideje.
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
