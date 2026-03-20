"""Jednostavne helper funkcije za pravi trening preko Stable-Baselines3."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback

from random_baseline import compare_random_baselines


class TrainingProgressCallback(BaseCallback):
    """Jednostavan callback koji javlja napredak treninga."""

    def __init__(self, total_timesteps: int) -> None:
        super().__init__()
        self.total_timesteps = max(int(total_timesteps), 1)
        self.log_every = max(self.total_timesteps // 10, 1)
        self.next_log_step = self.log_every
        self.last_logged_step = 0
        self.start_time = 0.0

    def _on_training_start(self) -> None:
        self.start_time = time.perf_counter()
        logger.info("Trening | 0/{} koraka (0.0%)", self.total_timesteps)

    def _on_step(self) -> bool:
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
        elapsed = time.perf_counter() - self.start_time
        logger.info("Trening zavrsen | {}/{} koraka | {:.1f}s", self.total_timesteps, self.total_timesteps, elapsed)


def make_env(env_id: str, *, hardcore: bool = False, render_mode: str | None = None) -> gym.Env:
    """Pravi jedno Gymnasium okruzenje sa opcijama koje trazimo.

    env_id govori koje okruzenje otvaramo, na primer BipedalWalker-v3.
    Ako je hardcore=True, pravimo tezu verziju staze. Ako je prosledjen
    render_mode, okruzenje ce umeti da vraca slike, sto nam treba za video.

    Funkcija vraca potpuno spreman env objekat koji posle mozemo da koristimo
    za trening, evaluaciju ili snimanje.
    """
    # Sve opcije za gym.make skupljamo na jedno mesto.
    # Tako nam ostatak koda bude cistiji.
    env_kwargs: dict[str, Any] = {}

    # Hardcore=True znaci teza staza.
    if hardcore:
        env_kwargs["hardcore"] = True

    # render_mode koristimo kada hocemo video.
    if render_mode is not None:
        env_kwargs["render_mode"] = render_mode

    # Na kraju stvarno pravimo okruzenje.
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
    """
    if episodes < 1:
        raise ValueError("episodes must be at least 1.")

    # Ovde skupljamo rezultat svake test epizode.
    rewards: list[float] = []
    episode_lengths: list[int] = []

    logger.info("Evaluacija modela | {} epizoda", episodes)

    for episode_index in range(episodes):
        # Za svaku epizodu pravimo novo okruzenje da sve krene cisto.
        env = make_env(env_id, hardcore=hardcore)
        try:
            observation, _ = env.reset(seed=seed + episode_index)
            done = False
            total_reward = 0.0
            episode_length = 0

            while not done:
                # model.predict vraca akciju koju trenutni model zeli da odigra.
                # deterministic=True znaci: bez dodatne slucajnosti u testu.
                action, _ = model.predict(observation, deterministic=True)
                action = np.asarray(action, dtype=np.float32)

                # Jedan korak simulacije.
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                episode_length += 1

                # Epizoda se zavrsava ako je env javio terminated ili truncated.
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
            # Uvek zatvaramo env, cak i ako nesto pukne.
            env.close()

    # Vracamo jednostavan recnik da kasnije lako odstampamo JSON summary.
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
    """
    if episodes < 1:
        raise ValueError("episodes must be at least 1.")

    # Ovo je folder gde ce gym wrapper ubaciti mp4 fajlove.
    video_path = Path(video_folder)
    video_path.mkdir(parents=True, exist_ok=True)

    logger.info("Snimanje videa | {} epizoda | folder={}", episodes, video_path)

    # RecordVideo radi samo ako env vraca slike.
    # Zato ovde trazimo render_mode="rgb_array".
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
                # Tokom snimanja samo pustamo model da igra.
                action, _ = model.predict(observation, deterministic=True)
                action = np.asarray(action, dtype=np.float32)
                observation, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            logger.info("Snimanje videa | epizoda {}/{} gotova", episode_index + 1, episodes)
    finally:
        env.close()

    # Vracamo listu svih mp4 fajlova koje je wrapper napravio.
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
    """
    if total_timesteps < 1:
        raise ValueError("total_timesteps must be at least 1.")

    # 1. Napravimo training env.
    training_env = make_env(env_id, hardcore=hardcore)
    try:
        logger.info("Pravljenje SB3 modela | algoritam={}", algorithm_name)
        # 2. Napravimo SB3 model.
        # "MlpPolicy" znaci obicna fully-connected neuronska mreza.
        model = algorithm_cls("MlpPolicy", training_env, verbose=0, seed=seed)

        # 3. Pokrenemo trening.
        model.learn(
            total_timesteps=total_timesteps,
            progress_bar=progress_bar,
            callback=TrainingProgressCallback(total_timesteps),
        )
    finally:
        training_env.close()

    # 4. Sacuvamo istrenirani model na disk.
    model_path = Path(save_path or f"artifacts/models/{algorithm_name}_bipedalwalker_seed{seed}").with_suffix("")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    logger.info("Model sacuvan | {}", model_path.with_suffix(".zip"))

    # 5. Pravimo osnovni summary.
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

    # 6. Pokrenemo i random baseline da imamo glupo-prostu referencu.
    # Ako model ne pobedi random baseline, to je znak da nije naucio mnogo.
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

    # 7. Video je opcionalan.
    # Ako korisnik nije trazio video, ova lista ostaje prazna.
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
            # Ne rusimo ceo trening ako video nije uspeo.
            # Samo sacuvamo poruku o gresci.
            video_error = str(error)

    summary["video_files"] = video_files
    summary["video_error"] = video_error
    return summary
