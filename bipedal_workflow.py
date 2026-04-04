"""Zajednicki helperi za trening, evaluaciju i algoritme u projektu."""

from __future__ import annotations

import time
from collections import deque
from functools import partial
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch
from loguru import logger
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecNormalize, VecVideoRecorder


def resolve_env_id(env_id: str, *, hardcore: bool = False) -> str:
    """Vraca stvarni Gymnasium env ID koji treba koristiti.

    Za BipedalWalker koristimo registrovani hardcore env ID kada je trazen
    hardcore mod, jer on ima i odgovarajuci TimeLimit iz registry-ja.
    """
    if hardcore and env_id == "BipedalWalker-v3":
        return "BipedalWalkerHardcore-v3"
    return env_id


def resolve_device(device: str) -> str:
    """Vraca stvarni PyTorch/SB3 device koji treba koristiti."""
    requested_device = device.lower()
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        logger.info("CUDA je trazen, ali nije dostupan. Prelazim na CPU.")
        return "cpu"
    return requested_device


class ObservationHistoryWrapper(gym.ObservationWrapper):
    """Pretvara jedno observation stanje u kratku istoriju poslednjih stanja."""

    def __init__(self, env: gym.Env, history_length: int = 4) -> None:
        super().__init__(env)
        self.history_length = max(int(history_length), 1)
        self._observation_history: deque[np.ndarray] = deque(maxlen=self.history_length)

        base_low = np.asarray(self.observation_space.low, dtype=np.float32)
        base_high = np.asarray(self.observation_space.high, dtype=np.float32)
        stacked_low = np.repeat(base_low[np.newaxis, ...], self.history_length, axis=0)
        stacked_high = np.repeat(base_high[np.newaxis, ...], self.history_length, axis=0)
        self.observation_space = gym.spaces.Box(
            low=stacked_low.reshape(-1),
            high=stacked_high.reshape(-1),
            dtype=np.float32,
        )

    def _stack_history(self) -> np.ndarray:
        return np.concatenate(list(self._observation_history), axis=0).astype(np.float32)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        self._observation_history.append(obs.copy())
        return self._stack_history()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.env.reset(seed=seed, options=options)
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        self._observation_history.clear()
        for _ in range(self.history_length):
            self._observation_history.append(obs.copy())
        return self._stack_history(), info


class HardcoreBipedalWrapper(gym.Wrapper):
    """Hardcore helper: frame skip + blaze kaznjavanje terminalnog pada."""

    def __init__(
        self,
        env: gym.Env,
        *,
        frame_skip: int = 2,
        fall_penalty: float = -10.0,
        failure_reward_threshold: float = -50.0,
    ) -> None:
        super().__init__(env)
        self.frame_skip = max(int(frame_skip), 1)
        self.fall_penalty = float(fall_penalty)
        self.failure_reward_threshold = float(failure_reward_threshold)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        total_shaped_reward = 0.0
        total_raw_reward = 0.0
        observation: np.ndarray | None = None
        terminated = False
        truncated = False
        info: dict[str, Any] = {}

        for _ in range(self.frame_skip):
            observation, reward, terminated, truncated, info = self.env.step(action)
            raw_reward = float(reward)
            total_raw_reward += raw_reward

            game_over = bool(getattr(self.env.unwrapped, "game_over", False))
            fell = game_over or (terminated and not truncated and raw_reward <= self.failure_reward_threshold)
            shaped_reward = self.fall_penalty if fell else raw_reward

            total_shaped_reward += shaped_reward
            if terminated or truncated:
                break

        info = dict(info)
        info["raw_reward"] = float(total_raw_reward)
        info["shaped_reward"] = float(total_shaped_reward)
        info["dead"] = bool(getattr(self.env.unwrapped, "game_over", False))
        return observation, total_shaped_reward, terminated, truncated, info


def apply_bipedal_wrappers(
    env: gym.Env,
    *,
    frame_skip: int = 1,
    observation_history: int = 1,
    fall_penalty: float | None = None,
    failure_reward_threshold: float = -50.0,
) -> gym.Env:
    """Primeni opciono hardcore-specifik wrapper stack na env."""
    wrapped_env = env

    if frame_skip > 1 or fall_penalty is not None:
        wrapped_env = HardcoreBipedalWrapper(
            wrapped_env,
            frame_skip=frame_skip,
            fall_penalty=-10.0 if fall_penalty is None else fall_penalty,
            failure_reward_threshold=failure_reward_threshold,
        )

    if observation_history > 1:
        wrapped_env = ObservationHistoryWrapper(
            wrapped_env,
            history_length=observation_history,
        )

    return wrapped_env


def build_algorithm_config(
    algorithm_name: str,
    *,
    preset: str,
    hardcore: bool,
    frame_skip: int | None = None,
    observation_history: int | None = None,
    fall_penalty: float | None = None,
) -> dict[str, object]:
    """Pravi konfiguraciju modela i env wrapper-e prema izabranom preset-u."""
    requested_preset = preset.lower()
    effective_preset = requested_preset
    if hardcore and requested_preset == "default":
        effective_preset = "hardcore"

    model_kwargs: dict[str, Any] = {}
    normalize_observations = False
    normalize_rewards = False
    normalization_gamma = 0.99
    frame_skip_value = 1
    observation_history_value = 1
    fall_penalty_value: float | None = None
    failure_reward_threshold = -50.0
    action_noise_sigma: float | None = None
    notes: list[str] = []

    if algorithm_name == "ppo":
        model_kwargs.update(
            learning_rate=3e-4,
            batch_size=256,
        )
        if effective_preset == "fast":
            model_kwargs.update(
                n_steps=1024,
                n_epochs=5,
                gamma=0.99,
                gae_lambda=0.95,
            )
            notes.append(
                "Fast preset za PPO koristi kraci rollout i manje epoha radi brzeg feedback loop-a."
            )
        elif effective_preset == "hardcore":
            model_kwargs.update(
                n_steps=4096,
                batch_size=512,
                n_epochs=10,
                gamma=0.999,
                gae_lambda=0.98,
                ent_coef=0.001,
                use_sde=True,
                sde_sample_freq=4,
                policy_kwargs={
                    "net_arch": {
                        "pi": [256, 256, 128],
                        "vf": [256, 256, 128],
                    }
                },
            )
            normalize_observations = True
            normalize_rewards = True
            normalization_gamma = 0.999
            frame_skip_value = 2
            observation_history_value = 4
            fall_penalty_value = -10.0
            notes.append(
                "Hardcore PPO preset ukljucuje VecNormalize, frame skip i kratku observation istoriju."
            )
        else:
            model_kwargs.update(
                n_steps=2048,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                use_sde=True,
                sde_sample_freq=4,
            )
    elif algorithm_name == "sac":
        model_kwargs.update(
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=10_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            use_sde=True,
            sde_sample_freq=32,
            policy_kwargs={"net_arch": [256, 256]},
        )
        if effective_preset == "fast":
            model_kwargs.update(
                learning_starts=2_000,
                batch_size=128,
                buffer_size=200_000,
            )
            notes.append(
                "Fast preset za SAC smanjuje warmup i replay buffer radi brze probe konfiguracije."
            )
        elif effective_preset == "hardcore":
            model_kwargs.update(
                learning_rate=3e-4,
                buffer_size=1_500_000,
                learning_starts=20_000,
                batch_size=512,
                gamma=0.995,
                ent_coef="auto_0.1",
                policy_kwargs={"net_arch": [512, 512, 256]},
            )
            normalize_observations = True
            frame_skip_value = 2
            observation_history_value = 6
            fall_penalty_value = -10.0
            notes.append(
                "Hardcore SAC preset prati uspesne pokusaje: frame skip, blazi fall penalty i history ulaz."
            )
    elif algorithm_name == "td3":
        model_kwargs.update(
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=10_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            policy_kwargs={"net_arch": [400, 300]},
        )
        action_noise_sigma = 0.1
        if effective_preset == "fast":
            model_kwargs.update(
                learning_starts=2_000,
                batch_size=128,
                buffer_size=200_000,
            )
            action_noise_sigma = 0.15
            notes.append(
                "Fast preset za TD3 ubrzava probe kroz manji replay buffer i raniji start ucenja."
            )
        elif effective_preset == "hardcore":
            model_kwargs.update(
                buffer_size=1_500_000,
                learning_starts=25_000,
                batch_size=512,
                gamma=0.995,
                policy_kwargs={"net_arch": [512, 512, 256]},
            )
            normalize_observations = True
            frame_skip_value = 2
            observation_history_value = 6
            fall_penalty_value = -10.0
            action_noise_sigma = 0.2
            notes.append(
                "Hardcore TD3 preset dodaje jacu mrezu, sporiji horizon i istu env obradu kao uspesni hardcore radovi."
            )
    else:
        raise ValueError(f"Nepodrzan algoritam: {algorithm_name}")

    if frame_skip is not None:
        frame_skip_value = max(int(frame_skip), 1)
    if observation_history is not None:
        observation_history_value = max(int(observation_history), 1)
    if fall_penalty is not None:
        fall_penalty_value = float(fall_penalty)

    return {
        "requested_preset": requested_preset,
        "effective_preset": effective_preset,
        "model_kwargs": model_kwargs,
        "normalize_observations": normalize_observations,
        "normalize_rewards": normalize_rewards,
        "normalization_gamma": normalization_gamma,
        "frame_skip": frame_skip_value,
        "observation_history": observation_history_value,
        "fall_penalty": fall_penalty_value,
        "failure_reward_threshold": failure_reward_threshold,
        "action_noise_sigma": action_noise_sigma,
        "uses_hardcore_helpers": bool(
            frame_skip_value > 1 or observation_history_value > 1 or fall_penalty_value is not None
        ),
        "notes": notes,
    }


class TrainingProgressCallback(BaseCallback):
    """Jednostavan callback koji javlja napredak treninga.

    Ovaj callback ne menja optimizacioni algoritam, nego samo meri tok
    eksperimenta. Osnovna velicina koju prati je relativni napredak:
    p = t / T
    gde je t broj odradjenih koraka, a T ukupan broj planiranih koraka.
    """

    def __init__(self, total_timesteps: int) -> None:
        """Inicijalizuje pracenje progresa treninga.

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

        Kasnije iz ovoga racunamo proteklo vreme:
        elapsed = t_now - t_start
        """
        self.start_time = time.perf_counter()
        logger.info("Trening | 0/{} koraka (0.0%)", self.total_timesteps)

    def _on_step(self) -> bool:
        """Loguje procenat zavrsenog treninga tokom ucenja.

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

        Ovde samo sumarizujemo eksperiment kroz ukupno proteklo vreme, bez
        menjanja parametara modela.
        """
        elapsed = time.perf_counter() - self.start_time
        logger.info("Trening zavrsen | {}/{} koraka | {:.1f}s", self.total_timesteps, self.total_timesteps, elapsed)


def run_episode(env, policy_fn: Callable[[gym.Env, np.ndarray], np.ndarray], seed: int | None = None) -> float:
    """Pokrece jednu celu epizodu i vraca ukupan reward.

    Funkcija resetuje okruzenje, zatim u petlji poziva policy_fn da dobije
    sledecu akciju i salje tu akciju u env. Sve reward vrednosti sabira dok
    epizoda ne stigne do kraja.

    Ovo je najosnovnija funkcija u baseline delu, jer predstavlja jedno
    kompletno "odigravanje" okruzenja.

    Ukupan povrat epizode je:
    G = sum_{t=0}^{T-1} r_t
    U ovom baseline-u ne radimo ucenje, nego samo merimo kakav rezultat daje
    odabrana politika kada se pusti kroz celo okruzenje.
    """
    observation, _ = env.reset(seed=seed)
    if seed is not None and hasattr(env, "action_space"):
        env.action_space.seed(seed)
    done = False
    total_reward = 0.0

    while not done:
        # policy_fn kaze koju akciju zelimo za trenutno stanje.
        action = np.asarray(policy_fn(env, observation), dtype=np.float32)
        observation, reward, terminated, truncated, info = env.step(action)
        raw_reward = float(info.get("raw_reward", reward))
        total_reward += raw_reward
        done = terminated or truncated

    return total_reward


def manual_random_policy(env, _observation: np.ndarray) -> np.ndarray:
    """Vraca rucno nasumicnu akciju u validnom opsegu.

    Umesto da koristimo gotovu Gymnasium random akciju, ovde mi sami citamo
    dozvoljeni raspon akcija iz env-a i direktno biramo slucajne vrednosti iz
    tog raspona.

    Za svaku komponentu akcije a_i biramo vrednost iz uniformne raspodele:
    a_i ~ U(low_i, high_i)
    To je jednostavna kontrolna politika koja ne koristi informaciju o stanju.
    """
    low = env.action_space.low
    high = env.action_space.high
    rng = getattr(env.action_space, "np_random", None)
    if rng is None:
        return np.random.uniform(low=low, high=high).astype(np.float32)
    # za svaku komponentu akcije biramo slucajan broj iz dozvoljenog opsega.
    return rng.uniform(low=low, high=high, size=low.shape).astype(np.float32)


def gym_random_policy(env, _observation: np.ndarray) -> np.ndarray:
    """Vraca random akciju pomocu Gymnasium action space samplera.

    Ovaj pristup je kraci i oslanja se na env.action_space.sample(), pa nam
    sluzi kao "sluzbena" random varijanta za poredjenje.

    Ovo je standardni referentni sampler iz definicije action space-a. Ideja je
    ista kao i kod rucne random politike: politika ne zavisi od stanja s_t.
    """
    return np.asarray(env.action_space.sample(), dtype=np.float32)


def evaluate_policy(
    env_factory,
    policy_fn: Callable[[gym.Env, np.ndarray], np.ndarray],
    episodes: int = 5,
    seed_start: int = 0,
    label: str = "",
) -> dict[str, object]:
    """Vrti vise epizoda za dati policy i racuna osnovnu statistiku.

    env_factory pravi novo okruzenje za svaku epizodu, a policy_fn je funkcija
    koja prima env i observation i vraca akciju. Tako mozemo istu evaluaciju da
    primenimo i na rucni random policy i na Gymnasium random policy.

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
        policy_fn=manual_random_policy,
        episodes=episodes,
        seed_start=seed_start,
        label="manual_random",
    )
    library = evaluate_policy(
        env_factory=env_factory,
        policy_fn=gym_random_policy,
        episodes=episodes,
        seed_start=seed_start,
        label="gymnasium_random",
    )
    return {"manual": manual, "library": library}


def make_env(
    env_id: str,
    *,
    hardcore: bool = False,
    render_mode: str | None = None,
    frame_skip: int = 1,
    observation_history: int = 1,
    fall_penalty: float | None = None,
    failure_reward_threshold: float = -50.0,
) -> gym.Env:
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
    resolved_env_id = resolve_env_id(env_id, hardcore=hardcore)
    env_kwargs: dict[str, Any] = {}

    if render_mode is not None:
        env_kwargs["render_mode"] = render_mode

    env = gym.make(resolved_env_id, **env_kwargs)
    return apply_bipedal_wrappers(
        env,
        frame_skip=frame_skip,
        observation_history=observation_history,
        fall_penalty=fall_penalty,
        failure_reward_threshold=failure_reward_threshold,
    )


def make_single_vec_env(
    env_id: str,
    *,
    hardcore: bool = False,
    render_mode: str | None = None,
    frame_skip: int = 1,
    observation_history: int = 1,
    fall_penalty: float | None = None,
    failure_reward_threshold: float = -50.0,
) -> DummyVecEnv:
    """Pravi DummyVecEnv sa jednim env-om za evaluaciju/video/normalizaciju."""
    return DummyVecEnv(
        [
            lambda: make_env(
                env_id,
                hardcore=hardcore,
                render_mode=render_mode,
                frame_skip=frame_skip,
                observation_history=observation_history,
                fall_penalty=fall_penalty,
                failure_reward_threshold=failure_reward_threshold,
            )
        ]
    )


def make_evaluation_env(
    env_id: str,
    *,
    hardcore: bool = False,
    render_mode: str | None = None,
    vecnormalize_path: str | Path | None = None,
    frame_skip: int = 1,
    observation_history: int = 1,
    fall_penalty: float | None = None,
    failure_reward_threshold: float = -50.0,
) -> gym.Env | VecEnv:
    """Pravi env za evaluaciju, sa ili bez ucitanih VecNormalize statistika."""
    if vecnormalize_path is None:
        return make_env(
            env_id,
            hardcore=hardcore,
            render_mode=render_mode,
            frame_skip=frame_skip,
            observation_history=observation_history,
            fall_penalty=fall_penalty,
            failure_reward_threshold=failure_reward_threshold,
        )

    base_env = make_single_vec_env(
        env_id,
        hardcore=hardcore,
        render_mode=render_mode,
        frame_skip=frame_skip,
        observation_history=observation_history,
        fall_penalty=fall_penalty,
        failure_reward_threshold=failure_reward_threshold,
    )
    normalized_env = VecNormalize.load(str(vecnormalize_path), base_env)
    normalized_env.training = False
    normalized_env.norm_reward = False
    return normalized_env


def make_training_env(
    env_id: str,
    *,
    algorithm_name: str,
    hardcore: bool = False,
    seed: int = 0,
    train_envs: int = 1,
    normalize_observations: bool = False,
    normalize_rewards: bool = False,
    normalization_gamma: float = 0.99,
    frame_skip: int = 1,
    observation_history: int = 1,
    fall_penalty: float | None = None,
    failure_reward_threshold: float = -50.0,
) -> gym.Env | VecEnv:
    """Pravi trening env, po potrebi paralelizovan za PPO."""
    effective_train_envs = max(int(train_envs), 1)

    if algorithm_name != "ppo" and effective_train_envs > 1:
        logger.info(
            "Paralelni env-ovi su trenutno ukljuceni samo za PPO, pa za {} koristim 1 env.",
            algorithm_name,
        )
        effective_train_envs = 1

    if effective_train_envs == 1 and not (normalize_observations or normalize_rewards):
        return make_env(
            env_id,
            hardcore=hardcore,
            frame_skip=frame_skip,
            observation_history=observation_history,
            fall_penalty=fall_penalty,
            failure_reward_threshold=failure_reward_threshold,
        )

    resolved_env_id = resolve_env_id(env_id, hardcore=hardcore)
    env_kwargs: dict[str, Any] = {}
    wrapper_class = partial(
        apply_bipedal_wrappers,
        frame_skip=frame_skip,
        observation_history=observation_history,
        fall_penalty=fall_penalty,
        failure_reward_threshold=failure_reward_threshold,
    )
    training_env: gym.Env | VecEnv

    if effective_train_envs == 1:
        training_env = make_vec_env(
            resolved_env_id,
            n_envs=1,
            seed=seed,
            env_kwargs=env_kwargs,
            vec_env_cls=DummyVecEnv,
            wrapper_class=wrapper_class,
        )
    else:
        logger.info("Paralelni trening | algoritam={} | env-ova={}", algorithm_name, effective_train_envs)
        try:
            training_env = make_vec_env(
                resolved_env_id,
                n_envs=effective_train_envs,
                seed=seed,
                env_kwargs=env_kwargs,
                vec_env_cls=SubprocVecEnv,
                wrapper_class=wrapper_class,
            )
        except (PermissionError, OSError) as error:
            logger.info(
                "SubprocVecEnv nije dostupan u ovom okruzenju ({}), pa prelazim na DummyVecEnv.",
                error,
            )
            training_env = make_vec_env(
                resolved_env_id,
                n_envs=effective_train_envs,
                seed=seed,
                env_kwargs=env_kwargs,
                vec_env_cls=DummyVecEnv,
                wrapper_class=wrapper_class,
            )

    if normalize_observations or normalize_rewards:
        logger.info(
            "Normalizacija trening env-a | norm_obs={} | norm_reward={}",
            normalize_observations,
            normalize_rewards,
        )
        return VecNormalize(
            training_env,
            norm_obs=normalize_observations,
            norm_reward=normalize_rewards,
            clip_obs=10.0,
            gamma=normalization_gamma,
        )

    return training_env


def rollout_model_episode(
    model: Any,
    env: gym.Env | VecEnv,
    *,
    seed: int,
    deterministic: bool = True,
) -> dict[str, object]:
    """Pokrece jednu celu epizodu istreniranog modela.

    Ovaj helper koristimo i za evaluaciju i za video, tako da ista logika
    biranja akcija i sabiranja reward-a bude na jednom mestu.

    Akademski pregled:
    Za fiksnu politiku pi(a|s) i pocetni seed ovde realizujemo jednu
    trajektoriju tau = (s_0, a_0, r_0, ..., s_T) i merimo njen povrat G.
    """
    if isinstance(env, VecEnv):
        if hasattr(env, "seed"):
            env.seed(seed)
        observation = env.reset()
        done = False
        total_reward = 0.0
        total_shaped_reward = 0.0
        episode_length = 0
        predict = model.predict

        while not done:
            action, _ = predict(observation, deterministic=deterministic)
            action = np.asarray(action, dtype=np.float32)
            observation, reward, terminated, infos = env.step(action)
            shaped_reward = float(np.asarray(reward).reshape(-1)[0])
            info = infos[0] if infos else {}
            raw_reward = float(info.get("raw_reward", shaped_reward))
            total_reward += raw_reward
            total_shaped_reward += shaped_reward
            episode_length += 1
            done = bool(np.asarray(terminated).reshape(-1)[0])

        return {
            "seed": int(seed),
            "reward": float(total_reward),
            "shaped_reward": float(total_shaped_reward),
            "length": int(episode_length),
        }

    observation, _ = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    total_shaped_reward = 0.0
    episode_length = 0
    predict = model.predict

    while not done:
        action, _ = predict(observation, deterministic=deterministic)
        action = np.asarray(action, dtype=np.float32)
        observation, reward, terminated, truncated, info = env.step(action)
        shaped_reward = float(reward)
        raw_reward = float(info.get("raw_reward", shaped_reward))
        total_reward += raw_reward
        total_shaped_reward += shaped_reward
        episode_length += 1
        done = terminated or truncated

    return {
        "seed": int(seed),
        "reward": float(total_reward),
        "shaped_reward": float(total_shaped_reward),
        "length": int(episode_length),
    }


def get_env_max_episode_steps(
    env_id: str,
    *,
    hardcore: bool = False,
    frame_skip: int = 1,
) -> int | None:
    """Vraca maksimalan broj koraka po epizodi ako je poznat."""
    env = make_env(env_id, hardcore=hardcore)
    try:
        max_episode_steps = getattr(getattr(env, "spec", None), "max_episode_steps", None)
    finally:
        env.close()

    if max_episode_steps is None:
        return None
    adjusted_steps = int(max_episode_steps)
    if frame_skip > 1:
        adjusted_steps = max(int(np.ceil(adjusted_steps / frame_skip)), 1)
    return adjusted_steps


def build_policy_diagnostics(
    env_id: str,
    *,
    hardcore: bool,
    eval_mean_reward: float,
    episode_lengths: list[int],
    best_episode_reward: float,
    frame_skip: int = 1,
) -> list[str]:
    """Pravi kratke tekstualne napomene o kvalitetu naucene politike."""
    diagnostics: list[str] = []
    max_episode_steps = get_env_max_episode_steps(
        env_id,
        hardcore=hardcore,
        frame_skip=frame_skip,
    )

    if max_episode_steps is not None and episode_lengths:
        all_hit_time_limit = all(length >= max_episode_steps for length in episode_lengths)
        if all_hit_time_limit and eval_mean_reward < 0.0:
            diagnostics.append(
                "Agent uglavnom dozivi vremenski limit epizode bez dobrog napretka. "
                "To obicno znaci da politika nije pukla, ali nije naucila korisno hodanje."
            )

    if best_episode_reward < 0.0:
        diagnostics.append(
            "Ni najbolja evaluaciona epizoda nema pozitivan reward, pa je politika "
            "trenutno ispod praga koji bismo smatrali upotrebljivim hodanjem."
        )

    if hardcore and eval_mean_reward < 0.0:
        diagnostics.append(
            "Hardcore mod je znacajno tezi od obicnog okruzenja, pa negativan reward "
            "sa podrazumevanim SB3 podesavanjima nije neuobicajen."
        )

    return diagnostics


def evaluate_model(
    model: Any,
    env_id: str,
    *,
    episodes: int = 5,
    seed: int = 0,
    hardcore: bool = False,
    vecnormalize_path: str | Path | None = None,
    frame_skip: int = 1,
    observation_history: int = 1,
    fall_penalty: float | None = None,
    failure_reward_threshold: float = -50.0,
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
    shaped_rewards: list[float] = []
    episode_lengths: list[int] = []
    evaluation_episodes: list[dict[str, object]] = []

    logger.info("Evaluacija modela | {} epizoda", episodes)

    for episode_index in range(episodes):
        env = make_evaluation_env(
            env_id,
            hardcore=hardcore,
            vecnormalize_path=vecnormalize_path,
            frame_skip=frame_skip,
            observation_history=observation_history,
            fall_penalty=fall_penalty,
            failure_reward_threshold=failure_reward_threshold,
        )
        try:
            episode_summary = rollout_model_episode(
                model=model,
                env=env,
                seed=seed + episode_index,
                deterministic=True,
            )
            episode_summary["index"] = int(episode_index + 1)
            evaluation_episodes.append(episode_summary)
            rewards.append(float(episode_summary["reward"]))
            shaped_rewards.append(float(episode_summary["shaped_reward"]))
            episode_lengths.append(int(episode_summary["length"]))
            if abs(float(episode_summary["reward"]) - float(episode_summary["shaped_reward"])) > 1e-6:
                logger.info(
                    "Evaluacija modela | epizoda {}/{} | raw_reward={:.2f} | shaped_reward={:.2f} | duzina={}",
                    episode_index + 1,
                    episodes,
                    episode_summary["reward"],
                    episode_summary["shaped_reward"],
                    episode_summary["length"],
                )
            else:
                logger.info(
                    "Evaluacija modela | epizoda {}/{} | reward={:.2f} | duzina={}",
                    episode_index + 1,
                    episodes,
                    episode_summary["reward"],
                    episode_summary["length"],
                )
        finally:
            env.close()

    best_episode = max(evaluation_episodes, key=lambda episode: float(episode["reward"]))
    worst_episode = min(evaluation_episodes, key=lambda episode: float(episode["reward"]))
    summary = {
        "eval_episodes": int(episodes),
        "eval_deterministic": True,
        "eval_mean_reward": float(np.mean(rewards)),
        "eval_std_reward": float(np.std(rewards)),
        "eval_rewards": rewards,
        "eval_mean_shaped_reward": float(np.mean(shaped_rewards)),
        "eval_std_shaped_reward": float(np.std(shaped_rewards)),
        "eval_shaped_rewards": shaped_rewards,
        "eval_episode_lengths": episode_lengths,
        "eval_mean_episode_length": float(np.mean(episode_lengths)),
        "evaluation_episodes": evaluation_episodes,
        "best_eval_episode": dict(best_episode),
        "worst_eval_episode": dict(worst_episode),
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
    evaluation_episodes: list[dict[str, object]],
    episodes: int = 1,
    seed: int = 0,
    hardcore: bool = False,
    vecnormalize_path: str | Path | None = None,
    frame_skip: int = 1,
    observation_history: int = 1,
    fall_penalty: float | None = None,
    failure_reward_threshold: float = -50.0,
) -> dict[str, object]:
    """Snima best/worst evaluaciju i po potrebi dodatne epizode.

    Podrazumevana ideja je da korisnik dobije jedan "success" i jedan
    "failure" video bez dodatnog razmisljanja. Zato kada je episodes=1,
    snimamo najbolju i najgoru evaluacionu epizodu. Ako je episodes > 1,
    pored njih snimamo jos tacno toliko dodatnih epizoda sa novim seed-ovima.

    Rezultat je recnik sa razdvojenim putanjama za best, worst i dodatne
    epizode, plus jednom objedinjavanom listom svih video fajlova.

    Akademski pregled:
    Ovaj deo nema novu optimizacionu matematiku, nego samo belezi vizuelnu
    trajektoriju politike tau = {(s_t, a_t)} radi kvalitativne analize.
    """
    if episodes < 1:
        raise ValueError("episodes must be at least 1.")

    video_path = Path(video_folder)
    video_path.mkdir(parents=True, exist_ok=True)

    session_folder = video_path / f"run_seed{seed}_{int(time.time())}"
    session_folder.mkdir(parents=True, exist_ok=True)
    max_episode_steps = get_env_max_episode_steps(
        env_id,
        hardcore=hardcore,
        frame_skip=frame_skip,
    ) or 2_000

    def record_single_video(label: str, episode_seed: int, target_folder: Path, prefix_suffix: str) -> str:
        target_folder.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Snimanje videa | {} | seed={} | folder={}", label, episode_seed, target_folder)
        if vecnormalize_path is None:
            env: gym.Env | VecEnv = gym.wrappers.RecordVideo(
                make_env(
                    env_id,
                    hardcore=hardcore,
                    render_mode="rgb_array",
                    frame_skip=frame_skip,
                    observation_history=observation_history,
                    fall_penalty=fall_penalty,
                    failure_reward_threshold=failure_reward_threshold,
                ),
                video_folder=str(target_folder),
                episode_trigger=lambda episode_index: episode_index == 0,
                name_prefix=f"{name_prefix}_{prefix_suffix}_seed{episode_seed}",
                disable_logger=True,
            )
        else:
            base_env = make_single_vec_env(
                env_id,
                hardcore=hardcore,
                render_mode="rgb_array",
                frame_skip=frame_skip,
                observation_history=observation_history,
                fall_penalty=fall_penalty,
                failure_reward_threshold=failure_reward_threshold,
            )
            normalized_env = VecNormalize.load(str(vecnormalize_path), base_env)
            normalized_env.training = False
            normalized_env.norm_reward = False
            env = VecVideoRecorder(
                normalized_env,
                video_folder=str(target_folder),
                record_video_trigger=lambda step_index: step_index == 0,
                video_length=max_episode_steps,
                name_prefix=f"{name_prefix}_{prefix_suffix}_seed{episode_seed}",
            )

        try:
            rollout_model_episode(
                model=model,
                env=env,
                seed=episode_seed,
                deterministic=True,
            )
        finally:
            env.close()

        video_files = sorted(target_folder.glob("*.mp4"), key=lambda path: path.stat().st_mtime)
        if not video_files:
            raise RuntimeError(f"Video fajl nije napravljen za seed {episode_seed}.")
        return str(video_files[-1])

    if not evaluation_episodes:
        raise ValueError("evaluation_episodes must not be empty when recording videos.")

    best_episode = max(evaluation_episodes, key=lambda episode: float(episode["reward"]))
    worst_episode = min(evaluation_episodes, key=lambda episode: float(episode["reward"]))

    logger.info(
        "Snimanje videa | best + worst + {} dodatnih epizoda | folder={}",
        episodes if episodes > 1 else 0,
        session_folder,
    )

    best_file = record_single_video(
        label="najbolja evaluaciona epizoda",
        episode_seed=int(best_episode["seed"]),
        target_folder=session_folder / "best",
        prefix_suffix="best",
    )

    if int(worst_episode["seed"]) == int(best_episode["seed"]):
        worst_file = best_file
    else:
        worst_file = record_single_video(
            label="najgora evaluaciona epizoda",
            episode_seed=int(worst_episode["seed"]),
            target_folder=session_folder / "worst",
            prefix_suffix="worst",
        )

    extra_episodes: list[dict[str, object]] = []
    extra_count = episodes if episodes > 1 else 0
    extra_seed_start = seed + len(evaluation_episodes)

    for extra_index in range(extra_count):
        extra_seed = int(extra_seed_start + extra_index)
        extra_file = record_single_video(
            label=f"dodatna epizoda {extra_index + 1}/{extra_count}",
            episode_seed=extra_seed,
            target_folder=session_folder / "extras" / f"seed_{extra_seed}",
            prefix_suffix=f"extra_{extra_index + 1:02d}",
        )
        extra_episodes.append(
            {
                "index": int(extra_index + 1),
                "seed": extra_seed,
                "file": extra_file,
            }
        )

    files = [best_file]
    if worst_file not in files:
        files.append(worst_file)
    files.extend(extra_episode["file"] for extra_episode in extra_episodes)

    video_summary = {
        "session_folder": str(session_folder),
        "requested_video_episodes": int(episodes),
        "recorded_files_count": int(len(files)),
        "best_episode": {
            **best_episode,
            "file": best_file,
        },
        "worst_episode": {
            **worst_episode,
            "file": worst_file,
        },
        "extra_episodes": extra_episodes,
        "files": files,
    }
    logger.info("Snimanje videa zavrseno | {} fajlova", len(files))
    return video_summary


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
    train_envs: int = 1,
    skip_random_baseline: bool = False,
    video_folder: str | Path | None = None,
    video_episodes: int = 1,
    device: str = "auto",
    preset: str = "default",
    frame_skip: int | None = None,
    observation_history: int | None = None,
    fall_penalty: float | None = None,
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

    effective_train_envs = max(int(train_envs), 1)
    if algorithm_name != "ppo":
        effective_train_envs = 1

    resolved_device = resolve_device(device)
    algorithm_config = build_algorithm_config(
        algorithm_name,
        preset=preset,
        hardcore=hardcore,
        frame_skip=frame_skip,
        observation_history=observation_history,
        fall_penalty=fall_penalty,
    )
    if algorithm_config["effective_preset"] != algorithm_config["requested_preset"]:
        logger.info(
            "Trening preset | trazen={} | efektivni={}",
            algorithm_config["requested_preset"],
            algorithm_config["effective_preset"],
        )
    else:
        logger.info("Trening preset | {}", algorithm_config["effective_preset"])
    logger.info("SB3 device | {}", resolved_device)
    logger.info(
        "Env helper-i | frame_skip={} | history={} | fall_penalty={}",
        algorithm_config["frame_skip"],
        algorithm_config["observation_history"],
        algorithm_config["fall_penalty"],
    )
    for note in algorithm_config["notes"]:
        logger.info(note)
    if algorithm_name in {"sac", "td3"} and resolved_device == "cpu":
        logger.info(
            "{} na CPU-u je cesto znatno sporiji od PPO-a za ovaj tip zadatka.",
            algorithm_name.upper(),
        )

    model_path = Path(save_path or f"artifacts/models/{algorithm_name}_bipedalwalker_seed{seed}").with_suffix("")
    vecnormalize_path: str | None = None
    training_env = make_training_env(
        env_id,
        algorithm_name=algorithm_name,
        hardcore=hardcore,
        seed=seed,
        train_envs=effective_train_envs,
        normalize_observations=bool(algorithm_config["normalize_observations"]),
        normalize_rewards=bool(algorithm_config["normalize_rewards"]),
        normalization_gamma=float(algorithm_config["normalization_gamma"]),
        frame_skip=int(algorithm_config["frame_skip"]),
        observation_history=int(algorithm_config["observation_history"]),
        fall_penalty=algorithm_config["fall_penalty"],
        failure_reward_threshold=float(algorithm_config["failure_reward_threshold"]),
    )
    try:
        model_kwargs = dict(algorithm_config["model_kwargs"])
        action_noise_sigma = algorithm_config["action_noise_sigma"]
        if algorithm_name == "td3" and action_noise_sigma is not None:
            action_shape = training_env.action_space.shape
            if action_shape is None:
                raise ValueError("TD3 zahteva Box action space sa poznatim oblikom.")
            model_kwargs["action_noise"] = NormalActionNoise(
                mean=np.zeros(action_shape, dtype=np.float32),
                sigma=np.full(action_shape, float(action_noise_sigma), dtype=np.float32),
            )
        logger.info("Pravljenje SB3 modela | algoritam={}", algorithm_name)
        model = algorithm_cls(
            "MlpPolicy",
            training_env,
            verbose=0,
            seed=seed,
            device=resolved_device,
            **model_kwargs,
        )
        model.learn(
            total_timesteps=total_timesteps,
            progress_bar=progress_bar,
            callback=TrainingProgressCallback(total_timesteps),
        )
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))
        logger.info("Model sacuvan | {}", model_path.with_suffix(".zip"))
        if isinstance(training_env, VecNormalize):
            vecnormalize_path = str(model_path.with_suffix(".vecnormalize.pkl"))
            training_env.save(vecnormalize_path)
            logger.info("VecNormalize statistike sacuvane | {}", vecnormalize_path)
    finally:
        training_env.close()

    summary = {
        "algorithm": algorithm_name,
        "env_id": resolve_env_id(env_id, hardcore=hardcore),
        "requested_env_id": env_id,
        "hardcore": bool(hardcore),
        "seed": seed,
        "device": resolved_device,
        "requested_preset": str(algorithm_config["requested_preset"]),
        "training_preset": str(algorithm_config["effective_preset"]),
        "algorithm_hyperparameters": dict(algorithm_config["model_kwargs"]),
        "vecnormalize_path": vecnormalize_path,
        "uses_vecnormalize": vecnormalize_path is not None,
        "uses_hardcore_helpers": bool(algorithm_config["uses_hardcore_helpers"]),
        "frame_skip": int(algorithm_config["frame_skip"]),
        "observation_history": int(algorithm_config["observation_history"]),
        "fall_penalty": algorithm_config["fall_penalty"],
        "training_notes": list(algorithm_config["notes"]),
        "total_timesteps": int(total_timesteps),
        "train_envs": int(effective_train_envs),
        "saved_model_path": str(model_path.with_suffix(".zip")),
    }
    summary.update(
        evaluate_model(
            model=model,
            env_id=env_id,
            episodes=eval_episodes,
            seed=seed,
            hardcore=hardcore,
            vecnormalize_path=vecnormalize_path,
            frame_skip=int(algorithm_config["frame_skip"]),
            observation_history=int(algorithm_config["observation_history"]),
            fall_penalty=algorithm_config["fall_penalty"],
            failure_reward_threshold=float(algorithm_config["failure_reward_threshold"]),
        )
    )

    random_baseline: dict[str, dict[str, object]] | None = None
    beats_random_baseline: bool | None = None
    improvement_vs_random: float | None = None
    if skip_random_baseline:
        logger.info("Random baseline preskocen radi brzeg eksperimenta.")
    else:
        logger.info("Pokretanje random baseline-a")
        random_baseline = compare_random_baselines(
            env_factory=lambda: make_env(
                env_id,
                hardcore=hardcore,
                frame_skip=int(algorithm_config["frame_skip"]),
                observation_history=int(algorithm_config["observation_history"]),
                fall_penalty=algorithm_config["fall_penalty"],
                failure_reward_threshold=float(algorithm_config["failure_reward_threshold"]),
            ),
            episodes=eval_episodes,
            seed_start=seed,
        )
        beats_random_baseline = summary["eval_mean_reward"] > random_baseline["library"]["mean_reward"]
        improvement_vs_random = summary["eval_mean_reward"] - random_baseline["library"]["mean_reward"]
        logger.info(
            "Random baseline zavrsen | random_mean={:.2f} | improvement={:.2f}",
            random_baseline["library"]["mean_reward"],
            improvement_vs_random,
        )

    summary["random_baseline"] = random_baseline
    summary["beats_random_baseline"] = beats_random_baseline
    summary["improvement_vs_random"] = improvement_vs_random
    summary["diagnostics"] = build_policy_diagnostics(
        env_id=env_id,
        hardcore=hardcore,
        eval_mean_reward=float(summary["eval_mean_reward"]),
        episode_lengths=[int(length) for length in summary["eval_episode_lengths"]],
        best_episode_reward=float(summary["best_eval_episode"]["reward"]),
        frame_skip=int(algorithm_config["frame_skip"]),
    )

    video_summary: dict[str, object] = {
        "session_folder": None,
        "requested_video_episodes": int(video_episodes),
        "recorded_files_count": 0,
        "best_episode": None,
        "worst_episode": None,
        "extra_episodes": [],
        "files": [],
    }
    video_error: str | None = None
    if video_folder is not None and video_episodes > 0:
        try:
            video_summary = record_video(
                model=model,
                env_id=env_id,
                video_folder=video_folder,
                name_prefix=f"{algorithm_name}_bipedalwalker_v3",
                evaluation_episodes=list(summary["evaluation_episodes"]),
                episodes=video_episodes,
                seed=seed,
                hardcore=hardcore,
                vecnormalize_path=vecnormalize_path,
                frame_skip=int(algorithm_config["frame_skip"]),
                observation_history=int(algorithm_config["observation_history"]),
                fall_penalty=algorithm_config["fall_penalty"],
                failure_reward_threshold=float(algorithm_config["failure_reward_threshold"]),
            )
        except Exception as error:
            video_error = str(error)

    summary["videos"] = video_summary
    summary["video_files"] = list(video_summary["files"])
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
    train_envs: int = 1,
    skip_random_baseline: bool = False,
    video_folder: str | None = None,
    video_episodes: int = 1,
    device: str = "auto",
    preset: str = "default",
    frame_skip: int | None = None,
    observation_history: int | None = None,
    fall_penalty: float | None = None,
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
        train_envs=train_envs,
        skip_random_baseline=skip_random_baseline,
        video_folder=video_folder,
        video_episodes=video_episodes,
        device=device,
        preset=preset,
        frame_skip=frame_skip,
        observation_history=observation_history,
        fall_penalty=fall_penalty,
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
    train_envs: int = 1,
    skip_random_baseline: bool = False,
    video_folder: str | None = None,
    video_episodes: int = 1,
    device: str = "auto",
    preset: str = "default",
    frame_skip: int | None = None,
    observation_history: int | None = None,
    fall_penalty: float | None = None,
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
        train_envs=train_envs,
        skip_random_baseline=skip_random_baseline,
        video_folder=video_folder,
        video_episodes=video_episodes,
        device=device,
        preset=preset,
        frame_skip=frame_skip,
        observation_history=observation_history,
        fall_penalty=fall_penalty,
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
    train_envs: int = 1,
    skip_random_baseline: bool = False,
    video_folder: str | None = None,
    video_episodes: int = 1,
    device: str = "auto",
    preset: str = "default",
    frame_skip: int | None = None,
    observation_history: int | None = None,
    fall_penalty: float | None = None,
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
        train_envs=train_envs,
        skip_random_baseline=skip_random_baseline,
        video_folder=video_folder,
        video_episodes=video_episodes,
        device=device,
        preset=preset,
        frame_skip=frame_skip,
        observation_history=observation_history,
        fall_penalty=fall_penalty,
    )
