"""SAC primer: rucna mini-verzija zbog ucenja + SB3 verzija za pravi trening."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from sb3_workflow import train_and_evaluate_sb3
from stable_baselines3 import SAC


def build_mlp(input_dim: int, output_dim: int, hidden_dim: int = 128) -> nn.Sequential:
    """Pravi malu fully-connected mrezu za SAC komponente.

    Funkcija sluzi kao zajednicki gradivni blok za actor i critic mreze, da ne
    dupliramo isti kod svaki put kada pravimo novu mrezu.

    Akademski pregled:
    Kao i kod drugih dubokih RL metoda, ovde koristimo MLP kao aproksimator
    nelinearne funkcije f_theta(x). U najjednostavnijem obliku:
    f_theta(x) = W3 * ReLU(W2 * ReLU(W1 * x + b1) + b2) + b3
    gde theta oznacava skup svih parametara mreze.
    """
    # Obicna neuronska mreza sa dva skrivena sloja.
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class GaussianActor(nn.Module):
    """SAC actor koji modeluje stohasticku politiku u kontinualnom prostoru.

    Akademski pregled:
    Soft Actor-Critic koristi politiku pi_theta(a|s) koja maksimizuje i reward
    i entropiju. Tipicna forma je:
    pi_theta(a|s) = N(mu_theta(s), diag(sigma_theta(s)^2))
    a cilj metode ukljucuje i entropijski clan alpha * H(pi(.|s)).
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128) -> None:
        """Pravi SAC actor koji opisuje Gaussovu raspodelu akcija.

        Ovaj model ne vraca jednu fiksnu akciju, nego raspodelu iz koje akcije
        mogu da se uzorkuju. To je tipicno za SAC, jer metoda voli i dobru
        akciju i dovoljno istrazivanja.

        Akademski pregled:
        Actor parametrize srednju vrednost mu_theta(s) i log-standardnu
        devijaciju log sigma_theta(s), pa zatim iz njih gradi Gaussovu
        raspodelu nad akcijama.
        """
        super().__init__()

        # Backbone prvo obradi observation u "skrivene" feature-e.
        self.backbone = build_mlp(obs_dim, hidden_dim, hidden_dim)

        # Mean i log_std opisuju Gaussovu raspodelu akcija.
        self.mean = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Linear(hidden_dim, act_dim)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Za dato stanje vraca mean i log_std akcione raspodele.

        Ove dve vrednosti zajedno definisu Gaussovu raspodelu iz koje SAC posle
        moze da uzorkuje akciju.

        Akademski pregled:
        Za dato stanje s racunamo:
        mu = mu_theta(s)
        log sigma = log sigma_theta(s)
        pa je raspodela:
        pi_theta(.|s) = N(mu, sigma^2)
        """
        hidden = self.backbone(observation)
        mean = self.mean(hidden)

        # Clamp sprecava da std ode u totalni haos.
        log_std = torch.clamp(self.log_std(hidden), min=-5.0, max=2.0)
        return mean, log_std

    def sample(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Uzorkuje akciju iz actor raspodele i vraca i njenu log-verovatnocu.

        SAC-u trebaju i sama akcija i log_prob te akcije, jer se u loss-u
        pojavljuje i kvalitet akcije i entropijski deo koji tera istrazivanje.

        Akademski pregled:
        SAC tipicno koristi reparametrizaciju:
        u = mu + sigma * epsilon, epsilon ~ N(0, I)
        a = tanh(u)
        a log-verovatnoca posle tanh transformacije dobija korekciju:
        log pi(a|s) = log N(u; mu, sigma^2) - sum_i log(1 - a_i^2)
        """
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
    """SAC critic mreza koja aproksimira akcijsko-vrednosnu funkciju.

    Akademski pregled:
    Critic pokusava da nauci Q(s, a), tj. ocekivani diskontovani povrat ako u
    stanju s odigramo akciju a, a zatim pratimo politiku dalje.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128) -> None:
        """Pravi SAC critic mrezu koja ocenjuje par stanje-akcija.

        Akademski pregled:
        U SAC-u je standardno da critic prima spoj [s, a] i vraca skalar
        Q_theta(s, a).
        """
        super().__init__()

        # Critic gleda i stanje i akciju, pa im zbirno ulazimo u mrezu.
        self.network = build_mlp(obs_dim + act_dim, 1, hidden_dim)

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Vraca Q-vrednost za dati observation i action.

        Akademski pregled:
        Formalno, ovde aproksimiramo:
        Q_theta(s_t, a_t) ~= E[sum_{k=0}^inf gamma^k * r_{t+k} | s_t, a_t]
        """
        critic_input = torch.cat([observation, action], dim=-1)
        return self.network(critic_input).squeeze(-1)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polako pomera target mrezu ka source mrezi.

    Umesto da target mrezu odmah prepisemo, ovde pravimo blagu interpolaciju.
    To pomaze da trening bude stabilniji.

    Akademski pregled:
    Ovo je Polyak averaging:
    theta_target <- (1 - tau) * theta_target + tau * theta_source
    za malo tau, target mreza se menja sporije i stabilizuje bootstrapping.
    """
    # target <- malo staro + malo novo
    # Ovo drzi target mreze stabilnijim od direktnog kopiranja na svaki korak.
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1.0 - tau).add_(tau * source_param.data)


def collect_random_batch(env, batch_size: int, seed: int | None = None) -> dict[str, torch.Tensor]:
    """Skuplja jedan batch tranzicija koristeci random akcije.

    Ovo je edukativni helper za rucni SAC demo. Ideja nije da bude pametan,
    nego samo da nam obezbedi podatke nad kojima mozemo da pokazemo jedan SAC
    update korak.

    Akademski pregled:
    Batch je skup tranzicija oblika:
    B = {(s_t, a_t, r_t, s_{t+1}, d_t)}_{t=1}^N
    Ovaj deo nije "off-policy replay buffer" u punom smislu, ali simulira isti
    tip podataka nad kojim SAC radi update.
    """
    # Za edukativni demo ovde NE treniramo dugo.
    # Samo skupimo random korake i posle nad njima pokazemo jedan SAC update.
    observation, _ = env.reset(seed=seed)

    observations = []
    actions = []
    rewards = []
    next_observations = []
    dones = []
    sample_action = env.action_space.sample

    for _ in range(batch_size):
        # Random akcija, cisto da imamo neki batch tranzicija.
        action = np.asarray(sample_action(), dtype=np.float32)
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

    # Vracamo obican recnik umesto posebne dataclass klase.
    return {
        "observations": torch.as_tensor(np.asarray(observations), dtype=torch.float32),
        "actions": torch.as_tensor(np.asarray(actions), dtype=torch.float32),
        "rewards": torch.as_tensor(np.asarray(rewards), dtype=torch.float32),
        "next_observations": torch.as_tensor(np.asarray(next_observations), dtype=torch.float32),
        "dones": torch.as_tensor(np.asarray(dones), dtype=torch.float32),
    }


def sac_update(
    actor: GaussianActor,
    critic_1: Critic,
    critic_2: Critic,
    target_critic_1: Critic,
    target_critic_2: Critic,
    batch: dict[str, torch.Tensor],
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    gamma: float = 0.99,
    alpha: float = 0.2,
    tau: float = 0.005,
) -> dict[str, float]:
    """Radi jedan rucni SAC update korak nad jednim batch-em podataka.

    Funkcija racuna target Q vrednosti, trenira oba critic-a, zatim trenira
    actor i na kraju blago osvezava target mreze. Vracene metrike sluze samo
    da mozemo da vidimo kako je prosao taj jedan update.

    Akademski pregled:
    SAC target za critic je:
    y = r + gamma * (1 - d) * (min(Q'_1(s', a'), Q'_2(s', a')) - alpha * log pi(a'|s'))
    Critic minimizuje:
    L_Q = E[(Q_1(s, a) - y)^2 + (Q_2(s, a) - y)^2]
    Actor minimizuje:
    J_pi = E[alpha * log pi(a|s) - min(Q_1(s, a), Q_2(s, a))]
    gde alpha kontrolise kompromis izmedju kvaliteta akcije i entropije.
    """
    with torch.no_grad():
        # 1. Iz sledeceg stanja uzorkujemo sledecu akciju.
        next_actions, next_log_probs = actor.sample(batch["next_observations"])
        target_q1 = target_critic_1(batch["next_observations"], next_actions)
        target_q2 = target_critic_2(batch["next_observations"], next_actions)

        # SAC voli akcije koje su i dobre po Q i dovoljno "raznovrsne".
        # Zato se pojavljuje - alpha * log_prob.
        target_q = torch.min(target_q1, target_q2) - alpha * next_log_probs
        td_target = batch["rewards"] + gamma * (1.0 - batch["dones"]) * target_q

    # 2. Critic mreze treba da pogode taj td_target.
    current_q1 = critic_1(batch["observations"], batch["actions"])
    current_q2 = critic_2(batch["observations"], batch["actions"])
    critic_loss = torch.mean((current_q1 - td_target) ** 2) + torch.mean((current_q2 - td_target) ** 2)

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # 3. Sada biramo akcije po trenutnom actor-u.
    sampled_actions, log_probs = actor.sample(batch["observations"])
    q1_pi = critic_1(batch["observations"], sampled_actions)
    q2_pi = critic_2(batch["observations"], sampled_actions)
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
    """Pokrece gotovu Stable-Baselines3 SAC implementaciju.

    Ova funkcija je praktican ulaz za pravi trening i prosledjuje parametre u
    zajednicki SB3 workflow koji radi trening, evaluaciju i random baseline.

    Akademski pregled:
    Bibliotecka verzija zadrzava istu osnovnu SAC ideju: maksimalni ocekivani
    reward uz entropijski regularizovan cilj, samo sa kompletnijim replay,
    target i optimizacionim mehanizmima.
    """
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
