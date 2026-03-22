"""Zajednicki helperi za trening, evaluaciju i algoritme u projektu."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from loguru import logger
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback


class TrainingProgressCallback(BaseCallback):
    """Jednostavan callback koji javlja napredak treninga.

    Akademski pregled:
    Ovaj callback ne menja optimizacioni algoritam, nego samo meri tok
    eksperimenta. Osnovna velicina koju prati je relativni napredak:
    p = t / T
    gde je t broj odradjenih koraka, a T ukupan broj planiranih koraka.
    """

    def __init__(self, total_timesteps: int) -> None:
        """Inicijalizuje pracenje progresa treninga.

        Akademski pregled:
        U logici callback-a koristimo aproksimaciju "svakih 10%" tako sto je
        interval logovanja:
        log_every = max(T / 10, 1)
        """
        super().__init__()
        self.total_timesteps = max(int(total_timesteps), 1)
        self.log_every = max(self.total_timesteps // 10, 1)
        self.next_log_step = self.log_every
        self.last_logged_step = 0
        self.start_time = 0.0

    def _on_training_start(self) -> None:
        """Pamti vreme pocetka treninga.

        Akademski pregled:
        Kasnije iz ovoga racunamo proteklo vreme:
        elapsed = t_now - t_start
        """
        self.start_time = time.perf_counter()
        logger.info("Trening | 0/{} koraka (0.0%)", self.total_timesteps)

    def _on_step(self) -> bool:
        """Loguje procenat zavrsenog treninga tokom ucenja.

        Akademski pregled:
        Glavna izvedena metrika je:
        progress_percent = 100 * num_timesteps / total_timesteps
        """
        current_step = min(int(self.num_timesteps), self.total_timesteps)
        if current_step > self.last_logged_step and (
            current_step >= self.next_log_step or current_step >= self.total_timesteps
        ):
            elapsed = time.perf_counter() - self.start_time
            progress = (current_step / self.total_timesteps) * 100.0
            logger.info(
                "Trening | {}/{} koraka ({:.1f}%) | {:.1f}s",
                current_step,
                self.total_timesteps,
                progress,
                elapsed,
            )
            self.last_logged_step = current_step
            while self.next_log_step <= current_step:
                self.next_log_step += self.log_every
        return True

    def _on_training_end(self) -> None:
        """Loguje ukupno trajanje treninga na kraju.

        Akademski pregled:
        Ovde samo sumarizujemo eksperiment kroz ukupno proteklo vreme, bez
        menjanja parametara modela.
        """
        elapsed = time.perf_counter() - self.start_time
        logger.info("Trening zavrsen | {}/{} koraka | {:.1f}s", self.total_timesteps, self.total_timesteps, elapsed)


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
    manual = evaluate_policy(
        env_factory=env_factory,
        policy_builder=manual_random_policy,
        episodes=episodes,
        seed_start=seed_start,
        label="manual_random",
    )
    library = evaluate_policy(
        env_factory=env_factory,
        policy_builder=gym_random_policy,
        episodes=episodes,
        seed_start=seed_start,
        label="gymnasium_random",
    )
    return {"manual": manual, "library": library}


def make_env(env_id: str, *, hardcore: bool = False, render_mode: str | None = None) -> gym.Env:
    """Pravi jedno Gymnasium okruzenje sa opcijama koje trazimo.

    env_id govori koje okruzenje otvaramo, na primer BipedalWalker-v3.
    Ako je hardcore=True, pravimo tezu verziju staze. Ako je prosledjen
    render_mode, okruzenje ce umeti da vraca slike, sto nam treba za video.

    Funkcija vraca potpuno spreman env objekat koji posle mozemo da koristimo
    za trening, evaluaciju ili snimanje.

    Akademski pregled:
    U RL terminima ovde instanciramo MDP/POMDP simulaciju, tj. objekat koji
    definise prelaze i reward kroz nepoznate funkcije P(s'|s,a) i R(s,a).
    Parametar `hardcore` menja tezinu zadatka, a time i distribuciju iskustava.
    """
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
    """Pusta istrenirani model kroz vise test epizoda i pravi statistiku.

    Model ovde vise ne treniramo, nego samo proveravamo kako se ponasa.
    Za svaku epizodu pustamo model da bira akcije, skupljamo ukupan reward i
    broj koraka do kraja epizode.

    Na kraju funkcija vraca recnik sa prosecnim reward-om, standardnom
    devijacijom, pojedinacnim reward-ima i duzinama epizoda.

    Akademski pregled:
    Evaluacija je Monte Carlo procena performansi deterministicke politike.
    Za epizodne povrate G_1, ..., G_N racunamo:
    mean_reward = (1 / N) * sum_i G_i
    std_reward = sqrt((1 / N) * sum_i (G_i - mean_reward)^2)
    a za duzine epizoda analogno racunamo prosecan broj koraka.
    """
    if episodes < 1:
        raise ValueError("episodes must be at least 1.")

    rewards: list[float] = []
    episode_lengths: list[int] = []

    logger.info("Evaluacija modela | {} epizoda", episodes)
    predict = model.predict

    for episode_index in range(episodes):
        env = make_env(env_id, hardcore=hardcore)
        try:
            observation, _ = env.reset(seed=seed + episode_index)
            done = False
            total_reward = 0.0
            episode_length = 0

            while not done:
                action, _ = predict(observation, deterministic=True)
                action = np.asarray(action, dtype=np.float32)
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                episode_length += 1
                done = terminated or truncated

            rewards.append(total_reward)
            episode_lengths.append(episode_length)
            logger.info(
                "Evaluacija modela | epizoda {}/{} | reward={:.2f} | duzina={}",
                episode_index + 1,
                episodes,
                total_reward,
                episode_length,
            )
        finally:
            env.close()

    summary = {
        "eval_episodes": int(episodes),
        "eval_deterministic": True,
        "eval_mean_reward": float(np.mean(rewards)),
        "eval_std_reward": float(np.std(rewards)),
        "eval_rewards": rewards,
        "eval_episode_lengths": episode_lengths,
        "eval_mean_episode_length": float(np.mean(episode_lengths)),
    }
    logger.info(
        "Evaluacija modela zavrsena | mean_reward={:.2f} | std={:.2f}",
        summary["eval_mean_reward"],
        summary["eval_std_reward"],
    )
    return summary


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
    """Snima video jednog ili vise pokretanja istreniranog modela.

    Funkcija pravi video folder, pokrece env u modu koji vraca slike i onda
    pusta model da igra zadati broj epizoda. Gym wrapper automatski pretvara
    te frejmove u mp4 fajlove.

    Rezultat je lista putanja do napravljenih video fajlova.

    Akademski pregled:
    Ovaj deo nema novu optimizacionu matematiku, nego samo belezi vizuelnu
    trajektoriju politike tau = {(s_t, a_t)} radi kvalitativne analize.
    """
    if episodes < 1:
        raise ValueError("episodes must be at least 1.")

    video_path = Path(video_folder)
    video_path.mkdir(parents=True, exist_ok=True)

    logger.info("Snimanje videa | {} epizoda | folder={}", episodes, video_path)
    predict = model.predict

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
                action, _ = predict(observation, deterministic=True)
                action = np.asarray(action, dtype=np.float32)
                observation, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            logger.info("Snimanje videa | epizoda {}/{} gotova", episode_index + 1, episodes)
    finally:
        env.close()

    video_files = sorted(str(path) for path in video_path.glob("*.mp4"))
    logger.info("Snimanje videa zavrseno | {} fajlova", len(video_files))
    return video_files


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
    """Pokrece ceo SB3 tok: trening, cuvanje, evaluaciju i random baseline.

    Ovo je glavna helper funkcija za "pravi" rad projekta. Prvo pravi env,
    zatim instancira trazeni Stable-Baselines3 algoritam i pokrece trening.
    Posle toga cuva model na disk, evaluira ga kroz vise epizoda i poredi ga
    sa random baseline-om.

    Ako je trazeno snimanje videa, na kraju pokusava da napravi i mp4 fajl.
    Funkcija vraca jedan summary recnik sa svim rezultatima.

    Akademski pregled:
    Ovo je eksperimentalni pipeline: trening aproksimira politiku, evaluacija
    procenjuje njen ocekivani povrat, a random baseline daje kontrolnu tacku.
    Jedna od izvedenih metrika u summary-ju je:
    improvement_vs_random = mean_reward_model - mean_reward_random
    """
    if total_timesteps < 1:
        raise ValueError("total_timesteps must be at least 1.")

    training_env = make_env(env_id, hardcore=hardcore)
    try:
        logger.info("Pravljenje SB3 modela | algoritam={}", algorithm_name)
        model = algorithm_cls("MlpPolicy", training_env, verbose=0, seed=seed)
        model.learn(
            total_timesteps=total_timesteps,
            progress_bar=progress_bar,
            callback=TrainingProgressCallback(total_timesteps),
        )
    finally:
        training_env.close()

    model_path = Path(save_path or f"artifacts/models/{algorithm_name}_bipedalwalker_seed{seed}").with_suffix("")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    logger.info("Model sacuvan | {}", model_path.with_suffix(".zip"))

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

    logger.info("Pokretanje random baseline-a")
    random_baseline = compare_random_baselines(
        env_factory=lambda: make_env(env_id, hardcore=hardcore),
        episodes=eval_episodes,
        seed_start=seed,
    )
    summary["random_baseline"] = random_baseline
    summary["beats_random_baseline"] = summary["eval_mean_reward"] > random_baseline["library"]["mean_reward"]
    summary["improvement_vs_random"] = summary["eval_mean_reward"] - random_baseline["library"]["mean_reward"]
    logger.info(
        "Random baseline zavrsen | random_mean={:.2f} | improvement={:.2f}",
        random_baseline["library"]["mean_reward"],
        summary["improvement_vs_random"],
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
    """Pokrece gotovu Stable-Baselines3 PPO implementaciju.

    Ovo je tanak wrapper oko zajednickog workflow-a za pravi trening.

    Akademski pregled:
    Stable-Baselines3 implementira PPO, tj. optimizaciju clipped surrogate
    cilja nad stohastickom politikom i critic mrezom.
    """
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

    Ovo je tanak wrapper oko zajednickog workflow-a za pravi trening.

    Akademski pregled:
    Stable-Baselines3 implementira SAC, tj. off-policy ucenje sa entropijski
    regularizovanim ciljem i stohastickom politikom.
    """
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

    Ovo je tanak wrapper oko zajednickog workflow-a za pravi trening.

    Akademski pregled:
    Stable-Baselines3 implementira TD3, tj. off-policy ucenje sa dva critic-a,
    target policy smoothing-om i delayed policy update-ima.
    """
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
