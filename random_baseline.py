"""Najprostiji baseline: pusti nasumicne akcije i vidi koliko je lose."""

from __future__ import annotations

from typing import Callable

import numpy as np
from loguru import logger


def run_episode(env, policy_fn: Callable[[np.ndarray], np.ndarray], seed: int | None = None) -> float:
    """Pokrece jednu celu epizodu i vraca ukupan reward.

    Funkcija resetuje okruzenje, zatim u petlji poziva policy_fn da dobije
    sledecu akciju i salje tu akciju u env. Sve reward vrednosti sabira dok
    epizoda ne stigne do kraja.

    Ovo je najosnovnija funkcija u baseline delu, jer predstavlja jedno
    kompletno "odigravanje" okruzenja.

    Akademski pregled:
    Ukupan povrat epizode je:
    G = sum_{t=0}^{T-1} r_t
    U ovom baseline-u ne radimo ucenje, nego samo merimo kakav rezultat daje
    odabrana politika kada se pusti kroz celo okruzenje.
    """
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
    """Pravi policy funkciju koja sama rucno bira random akcije.

    Umesto da koristimo gotovu Gymnasium random akciju, ovde mi sami citamo
    dozvoljeni raspon akcija iz env-a i vracamo novu funkciju koja uvek bira
    slucajne vrednosti iz tog raspona.

    Akademski pregled:
    Za svaku komponentu akcije a_i biramo vrednost iz uniformne raspodele:
    a_i ~ U(low_i, high_i)
    To je jednostavna kontrolna politika koja ne koristi informaciju o stanju.
    """

    low = env.action_space.low
    high = env.action_space.high

    def policy(_observation: np.ndarray) -> np.ndarray:
        """Vraca jednu nasumicnu akciju u validnom opsegu."""
        # Najprostija moguca ideja:
        # za svaku komponentu akcije biramo slucajan broj iz dozvoljenog opsega.
        return np.random.uniform(low=low, high=high).astype(np.float32)

    return policy


def gym_random_policy(env):
    """Pravi policy funkciju koja koristi ugadjeni Gymnasium random sampler.

    Ovaj pristup je kraci i oslanja se na env.action_space.sample(), pa nam
    sluzi kao "sluzbena" random varijanta za poredjenje.

    Akademski pregled:
    Ovo je standardni referentni sampler iz definicije action space-a. Ideja je
    ista kao i kod rucne random politike: politika ne zavisi od stanja s_t.
    """

    sample_action = env.action_space.sample

    def policy(_observation: np.ndarray) -> np.ndarray:
        """Vraca jednu random akciju pomocu Gymnasium action space samplera."""
        return np.asarray(sample_action(), dtype=np.float32)

    return policy


def evaluate_policy(env_factory, policy_builder, episodes: int = 5, seed_start: int = 0, label: str = "") -> dict[str, object]:
    """Vrti vise epizoda za dati policy i racuna osnovnu statistiku.

    env_factory pravi novo okruzenje za svaku epizodu, a policy_builder od tog
    env-a pravi funkciju koja zna da vrati akciju. Tako mozemo istu evaluaciju
    da primenimo i na rucni random policy i na Gymnasium random policy.

    Na kraju vracamo recnik sa imenom baseline-a, prosecnim reward-om,
    standardnom devijacijom i pojedinacnim rezultatima po epizodama.

    Akademski pregled:
    Ovde radimo Monte Carlo procenu performansi politike. Za epizodne povrate
    G_1, ..., G_N racunamo:
    mean = (1 / N) * sum_i G_i
    std = sqrt((1 / N) * sum_i (G_i - mean)^2)
    """
    # Ova funkcija vrti vise epizoda i pravi statistiku.
    rewards: list[float] = []

    for episode_idx in range(episodes):
        env = env_factory()
        try:
            # policy_builder prima env i vraca funkciju koja bira akcije.
            policy_fn = policy_builder(env)
            reward = run_episode(env, policy_fn=policy_fn, seed=seed_start + episode_idx)
            rewards.append(reward)
            logger.info(
                "Random baseline [{}] | epizoda {}/{} | reward={:.2f}",
                label or "baseline",
                episode_idx + 1,
                episodes,
                reward,
            )
        finally:
            env.close()

    # I ovde vracamo obican recnik umesto posebne dataclass klase.
    return {
        "label": label,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "rewards": rewards,
    }


def compare_random_baselines(env_factory, episodes: int = 5, seed_start: int = 0) -> dict[str, dict[str, object]]:
    """Uporedjuje dve random baseline varijante na istom okruzenju.

    Prva varijanta je rucna, gde sami uzorkujemo akcije iz opsega action
    space-a. Druga varijanta koristi Gymnasium-ov ugradjeni sample metod.

    Rezultat je recnik sa obe statistike, tako da lako mozemo da vidimo koliko
    je random igranje lose i da li istrenirani model uspeva da ga pobedi.

    Akademski pregled:
    Ovo je kontrolni eksperiment: poredimo dva stohasticka baseline-a da bismo
    dobili referentni nivo performansi bez ucenja i bez parametarske politike.
    """

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
