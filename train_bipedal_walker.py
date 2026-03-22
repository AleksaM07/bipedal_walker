"""Glavna skripta: treniraj model, testiraj ga i po zelji snimi video."""

from __future__ import annotations # omogucava nam da koristimo tipove koji su definisani kasnije u kodu

import argparse
import json
import sys
from pathlib import Path # standardna biblioteka za rad sa fajlovima

from loguru import logger

from ppo_method import run_library_ppo
from sac_method import run_library_sac
from td3_method import run_library_td3


# Ovo je ime Gymnasium okruzenja koje koristimo svuda u projektu.
ENV_ID = "BipedalWalker-v3"


def main() -> None:
    """Glavna ulazna tacka skripte.

    Ova funkcija cita argumente iz terminala, bira koji RL algoritam zelimo
    da koristimo, priprema putanje za cuvanje modela i videa i na kraju
    pokrece trening preko odgovarajuce helper funkcije.

    Kada se trening zavrsi, funkcija stampa JSON summary sa najbitnijim
    rezultatima, kao sto su reward, duzina epizoda, random baseline i
    putanja do sacuvanog modela.

    Akademski pregled:
    Ova funkcija ne uvodi novu RL matematiku, nego definise eksperimentalni
    protokol: izbor algoritma, broj trening koraka, broj evaluacionih epizoda,
    seed i izlazne artefakte. U tom smislu ona orkestrira merenje performansi,
    na primer prosecan reward:
    mean_reward = (1 / N) * sum_i G_i
    koji se kasnije pojavljuje u summary-ju.
    """
    # argparse cita argumente iz terminala, npr:
    # python train_bipedal_walker.py --algo ppo --timesteps 50000
    logger.remove()
    logger.add(
        sys.stderr,
        format="{time:HH:mm:ss} | {level:<8} | {message}",
        level="INFO",
    )

    parser = argparse.ArgumentParser(
        description="Train and evaluate a Stable-Baselines3 agent on BipedalWalker-v3.",
    )
    # Biramo koji algoritam hocemo da koristimo.
    parser.add_argument("--algo", choices=("ppo", "sac", "td3"), default="ppo")
    # Koliko ukupno koraka treninga zelimo.
    parser.add_argument("--timesteps", type=int, default=50_000)
    # Koliko punih epizoda zelimo za test posle treninga.
    parser.add_argument("--eval-episodes", type=int, default=5)
    # Seed sluzi da rezultati budu ponovljivi koliko je moguce.
    parser.add_argument("--seed", type=int, default=42)
    # Ovde ce ici modeli i video fajlovi.
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    # Ako dodamo ovu zastavicu, skripta ce snimiti video posle treninga.
    parser.add_argument("--record-video", action="store_true")
    # Koliko epizoda zelimo da snimimo.
    parser.add_argument("--video-episodes", type=int, default=1)
    # Stable-Baselines3 moze da prikaze progress bar tokom treninga.
    parser.add_argument("--progress-bar", action="store_true")
    # Hardcore je teza verzija istog okruzenja.
    parser.add_argument("--hardcore", action="store_true")
    args = parser.parse_args()
    log_dir = args.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{args.algo}_bipedalwalker_seed{args.seed}.log"
    file_sink_id = logger.add(
        log_path,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
        level="INFO",
        encoding="utf-8",
        mode="w",
    )

    # Mapa: ime algoritma -> funkcija koja zna da ga pokrene.
    runners = {
        "ppo": run_library_ppo,
        "sac": run_library_sac,
        "td3": run_library_td3,
    }

    # Gde snimamo istrenirani model.
    model_save_path = args.output_dir / "models" / f"{args.algo}_bipedalwalker_seed{args.seed}"

    # Video folder pravimo samo ako je korisnik trazio snimanje.
    video_folder = None
    if args.record_video:
        video_folder = args.output_dir / "videos" / f"{args.algo}_bipedalwalker_seed{args.seed}"

    try:
        logger.info(
            "Pokretanje treninga | algo={} | timesteps={} | eval_ep={} | seed={} | video={}",
            args.algo,
            args.timesteps,
            args.eval_episodes,
            args.seed,
            args.record_video,
        )

        # Pozivamo izabrani algoritam sa svim opcijama koje je korisnik zadao.
        summary = runners[args.algo](
            env_id=ENV_ID,
            total_timesteps=args.timesteps,
            save_path=str(model_save_path),
            seed=args.seed,
            eval_episodes=args.eval_episodes,
            progress_bar=args.progress_bar,
            hardcore=args.hardcore,
            video_folder=str(video_folder) if video_folder is not None else None,
            video_episodes=args.video_episodes,
        )
        summary["log_file"] = str(log_path)

        logger.info(
            "Gotovo | eval_mean_reward={:.2f} | random_mean={:.2f} | model={}",
            summary["eval_mean_reward"],
            summary["random_baseline"]["library"]["mean_reward"],
            summary["saved_model_path"],
        )
        logger.info("Log sacuvan | {}", log_path)

        # Na kraju stampamo JSON da lepo vidimo sta se desilo:
        # reward, duzine epizoda, putanja do modela i eventualno video fajlovi.
        print(json.dumps(summary, indent=2))
    finally:
        logger.remove(file_sink_id)


if __name__ == "__main__":
    main()
