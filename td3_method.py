"""TD3 primer: rucna mini-verzija zbog ucenja + SB3 verzija za pravi trening."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from sb3_workflow import train_and_evaluate_sb3
from stable_baselines3 import TD3



def build_mlp(input_dim: int, output_dim: int, hidden_dim: int = 128) -> nn.Sequential:
    """Pravi malu fully-connected mrezu za TD3 actor i critic deo.

    Koristimo je kao zajednicki helper da ne ponavljamo isti kod za konstrukciju
    mreze na vise mesta.

    Akademski pregled:
    I ovde koristimo MLP kao nelinearni aproksimator funkcije f_theta(x):
    f_theta(x) = W3 * ReLU(W2 * ReLU(W1 * x + b1) + b2) + b3
    Sto je standardan izbor za kontinuiranu kontrolu sa vektorskim ulazima.
    """
    # Obicna neuronska mreza koju koristimo i za actor i za critic.
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class DeterministicActor(nn.Module):
    """TD3 actor koji implementira deterministicku politiku.

    Akademski pregled:
    TD3 koristi deterministicku politiku oblika:
    a = mu_theta(s)
    za razliku od PPO i SAC, ovde se ne modeluje cela raspodela akcija nego
    direktna mapa iz stanja u akciju.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128) -> None:
        """Pravi TD3 actor koji za stanje bira jednu konkretnu akciju.

        Akademski pregled:
        Cilj je da mreza nauci aproksimaciju funkcije mu_theta(s) koja vraca
        akciju visoke Q-vrednosti za svako stanje.
        """
        super().__init__()
        self.network = build_mlp(obs_dim, act_dim, hidden_dim)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Pretvara observation u akciju ogranicenu na validan opseg.

        Akademski pregled:
        Formalno:
        a = tanh(mu_theta(s))
        gde tanh obezbedjuje da izlaz ostane u kontrolisanom opsegu akcija.
        """
        # TD3 koristi deterministicku politiku:
        # za dato stanje, mreza bira jednu konkretnu akciju.
        return torch.tanh(self.network(observation))


class Critic(nn.Module):
    """TD3 critic koji aproksimira Q-funkciju nad stanjem i akcijom.

    Akademski pregled:
    Kao i kod DDPG/TD3 porodice metoda, critic uci funkciju Q_theta(s, a) koja
    meri kvalitet konkretne akcije u konkretnom stanju.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128) -> None:
        """Pravi TD3 critic mrezu koja ocenjuje par stanje-akcija.

        Akademski pregled:
        Ulaz je konkatenacija [s, a], a izlaz je skalarna procena Q_theta(s, a).
        """
        super().__init__()

        # Critic opet gleda stanje + akciju zajedno.
        self.network = build_mlp(obs_dim + act_dim, 1, hidden_dim)

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Vraca Q-vrednost za dati observation i action.

        Akademski pregled:
        Ovde aproksimiramo akcijsko-vrednosnu funkciju:
        Q_theta(s_t, a_t) ~= E[sum_{k=0}^inf gamma^k * r_{t+k} | s_t, a_t]
        """
        return self.network(torch.cat([observation, action], dim=-1)).squeeze(-1)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polako pomera target mrezu ka glavnoj mrezi.

    Ovo radimo da target mreze ne bi skakale previse naglo iz koraka u korak.

    Akademski pregled:
    Kao i u drugim off-policy metodama, koristimo:
    theta_target <- (1 - tau) * theta_target + tau * theta_source
    da bismo dobili sporije i stabilnije target procene.
    """
    # Target mrezu pomeramo polako, ne naglo.
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1.0 - tau).add_(tau * source_param.data)


def collect_random_batch(env, batch_size: int, seed: int | None = None) -> dict[str, torch.Tensor]:
    """Skuplja jedan batch tranzicija sa random akcijama za TD3 demo.

    Funkcija ne pokusava da igra dobro, nego samo da skupi dovoljno podataka
    da bismo mogli da pokazemo jedan TD3 update korak.

    Akademski pregled:
    Kao i kod SAC-a, skupljamo batch tranzicija:
    B = {(s_t, a_t, r_t, s_{t+1}, d_t)}_{t=1}^N
    nad kojim kasnije racunamo bootstrapped target vrednosti.
    """
    # Kao i kod SAC demo-a, ovde samo skupimo random podatke
    # da pokazemo kako izgleda jedan TD3 update.
    observation, _ = env.reset(seed=seed)

    observations = []
    actions = []
    rewards = []
    next_observations = []
    dones = []
    sample_action = env.action_space.sample

    for _ in range(batch_size):
        # Random akcija cisto za demo.
        action = np.asarray(sample_action(), dtype=np.float32)
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

    # Vracamo obican recnik umesto posebne dataclass klase.
    return {
        "observations": torch.as_tensor(np.asarray(observations), dtype=torch.float32),
        "actions": torch.as_tensor(np.asarray(actions), dtype=torch.float32),
        "rewards": torch.as_tensor(np.asarray(rewards), dtype=torch.float32),
        "next_observations": torch.as_tensor(np.asarray(next_observations), dtype=torch.float32),
        "dones": torch.as_tensor(np.asarray(dones), dtype=torch.float32),
    }


def td3_update(
    actor: DeterministicActor,
    critic_1: Critic,
    critic_2: Critic,
    target_actor: DeterministicActor,
    target_critic_1: Critic,
    target_critic_2: Critic,
    batch: dict[str, torch.Tensor],
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    gamma: float = 0.99,
    tau: float = 0.005,
    policy_noise: float = 0.2,
    noise_clip: float = 0.5,
    update_actor: bool = True,
) -> dict[str, float]:
    """Radi jedan rucni TD3 update korak.

    Funkcija prvo racuna target Q vrednost preko target mreza, zatim trenira
    oba critic-a i po potrebi actor. Posle toga osvezava target mreze.

    Vracene metrike sluze za pracenje tog jednog update koraka.

    Akademski pregled:
    TD3 target koristi target policy smoothing:
    a' = clip(mu_target(s') + epsilon, -1, 1)
    gde je epsilon ~ clip(N(0, sigma^2), -c, c)
    Zatim:
    y = r + gamma * (1 - d) * min(Q'_1(s', a'), Q'_2(s', a'))
    Critic minimizuje:
    L_Q = E[(Q_1(s, a) - y)^2 + (Q_2(s, a) - y)^2]
    a actor optimizuje:
    J_mu = -E[Q_1(s, mu_theta(s))]
    uz odlozen update actor-a, sto je i dalo ime Twin Delayed DDPG.
    """
    with torch.no_grad():
        # TD3 dodaje malo buke na target akciju.
        # To pomaze da politika ne postane previse "krhka".
        noise = torch.randn_like(batch["actions"]) * policy_noise
        noise = torch.clamp(noise, -noise_clip, noise_clip)
        next_actions = torch.clamp(target_actor(batch["next_observations"]) + noise, -1.0, 1.0)

        # TD3 ima dva critic-a i uzima manju Q vrednost.
        # Ideja: manje preoptimisticna procena.
        target_q1 = target_critic_1(batch["next_observations"], next_actions)
        target_q2 = target_critic_2(batch["next_observations"], next_actions)
        target_q = torch.min(target_q1, target_q2)
        td_target = batch["rewards"] + gamma * (1.0 - batch["dones"]) * target_q

    # Critic-i pokusavaju da pogode td_target.
    current_q1 = critic_1(batch["observations"], batch["actions"])
    current_q2 = critic_2(batch["observations"], batch["actions"])
    critic_loss = torch.mean((current_q1 - td_target) ** 2) + torch.mean((current_q2 - td_target) ** 2)

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    actor_loss_value = float("nan")
    if update_actor:
        # Actor bira akcije za trenutna stanja.
        predicted_actions = actor(batch["observations"])

        # Zelimo akcije koje critic_1 smatra sto boljim.
        actor_loss = -critic_1(batch["observations"], predicted_actions).mean()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        actor_loss_value = float(actor_loss.item())

        # Posle actor update-a osvezavamo target mreze.
        soft_update(target_actor, actor, tau=tau)
        soft_update(target_critic_1, critic_1, tau=tau)
        soft_update(target_critic_2, critic_2, tau=tau)

    return {
        "critic_loss": float(critic_loss.item()),
        "actor_loss": actor_loss_value,
        "target_q_mean": float(td_target.mean().item()),
    }


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
    """Pokrece gotovu Stable-Baselines3 TD3 implementaciju.

    Ovo je prakticna funkcija za pravi trening i samo prosledjuje argumente u
    zajednicki SB3 workflow koji radi trening, evaluaciju i random baseline.

    Akademski pregled:
    SB3 verzija koristi istu osnovnu TD3 ideju: dva critic-a, target policy
    smoothing i delayed policy updates, ali u punoj biblioteckoj implementaciji.
    """
    if TD3 is None:
        raise ImportError("stable_baselines3 nije instaliran. Instaliraj stable-baselines3 da pokrenes library TD3.")

    # Ovo je realna TD3 putanja preko biblioteke.
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
