"""Glavna skripta: treniraj model, testiraj ga i po zelji snimi video."""

from __future__ import annotations # omogucava nam da koristimo tipove koji su definisani kasnije u kodu

import argparse
import json
import sys
from pathlib import Path # standardna biblioteka za rad sa fajlovima

from loguru import logger

from bipedal_workflow import run_library_ppo, run_library_sac, run_library_td3


# Ovo je ime Gymnasium okruzenja koje koristimo svuda u projektu.
ENV_ID = "BipedalWalker-v3"


def format_episode_summary(label: str, episode: dict[str, object] | None) -> str:
    """Pravi kratku tekstualnu liniju za jednu epizodu iz summary-ja."""
    if not episode:
        return f"- {label}: nema podataka"
    return (
        f"- {label}: reward={float(episode['reward']):.2f} | "
        f"duzina={int(episode['length'])} | seed={int(episode['seed'])}"
    )


def format_terminal_summary(summary: dict[str, object]) -> str:
    """Vraca pregledan zavrsni rezime za terminal."""
    env_label = str(summary["env_id"])
    if bool(summary.get("hardcore")) and "Hardcore" not in env_label:
        env_label += " (hardcore)"

    random_baseline = None
    if summary.get("random_baseline") is not None:
        random_baseline = summary["random_baseline"]["library"]
    videos = summary.get("videos", {})
    extra_videos = videos.get("extra_episodes", [])
    diagnostics = list(summary.get("diagnostics", []))

    lines = [
        "",
        "===== Rezime eksperimenta =====",
        f"Algoritam: {str(summary['algorithm']).upper()}",
        f"Okruzenje: {env_label}",
        f"Preset: {str(summary.get('training_preset', 'default'))} | device: {str(summary.get('device', 'cpu'))}",
        (
            "Env helper-i: "
            f"frame_skip={int(summary.get('frame_skip', 1))} | "
            f"history={int(summary.get('observation_history', 1))} | "
            f"fall_penalty={summary.get('fall_penalty')}"
        ),
        f"Timesteps: {int(summary['total_timesteps'])} | seed: {int(summary['seed'])} | train envs: {int(summary.get('train_envs', 1))}",
        "",
        "Evaluacija:",
        f"- mean reward: {float(summary['eval_mean_reward']):.2f}",
        f"- std reward: {float(summary['eval_std_reward']):.2f}",
        format_episode_summary("najbolja epizoda", summary.get("best_eval_episode")),
        format_episode_summary("najgora epizoda", summary.get("worst_eval_episode")),
        f"- prosecan broj koraka: {float(summary['eval_mean_episode_length']):.1f}",
        "",
        "Random baseline:",
    ]
    if summary.get("eval_mean_shaped_reward") is not None:
        shaped_mean = float(summary["eval_mean_shaped_reward"])
        if abs(shaped_mean - float(summary["eval_mean_reward"])) > 1e-6:
            lines.insert(10, f"- shaped mean reward: {shaped_mean:.2f}")
    if random_baseline is None:
        lines.append("- preskocen")
    else:
        lines.extend(
            [
                f"- gymnasium random mean: {float(random_baseline['mean_reward']):.2f}",
                f"- improvement vs random: {float(summary['improvement_vs_random']):.2f}",
                f"- pobedjuje random: {'da' if bool(summary['beats_random_baseline']) else 'ne'}",
            ]
        )

    if diagnostics:
        lines.extend(
            [
                "",
                "Dijagnostika:",
                *[f"- {message}" for message in diagnostics],
            ]
        )

    lines.append("")
    lines.append("Video:")
    if summary.get("video_error") is not None:
        lines.append(f"- greska: {summary['video_error']}")
    elif videos.get("files"):
        if videos.get("best_episode") is not None:
            lines.append(f"- best: {videos['best_episode']['file']}")
        if videos.get("worst_episode") is not None:
            lines.append(f"- worst: {videos['worst_episode']['file']}")
        lines.append(f"- dodatne epizode: {len(extra_videos)}")
    else:
        lines.append("- video nije trazen")

    lines.extend(
        [
            "",
            "Artefakti:",
            f"- model: {summary['saved_model_path']}",
            f"- log: {summary['log_file']}",
            f"- summary json: {summary['summary_file']}",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Glavna ulazna tacka skripte.

    Ova funkcija cita argumente iz terminala, bira koji RL algoritam zelimo
    da koristimo, priprema putanje za cuvanje modela i videa i na kraju
    pokrece trening preko odgovarajuce helper funkcije.

    Kada se trening zavrsi, funkcija stampa JSON summary sa najbitnijim
    rezultatima, kao sto su reward, duzina epizoda, random baseline i
    putanja do sacuvanog modela.

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
    # Broj paralelnih trening env-ova. Najvise pomaze PPO-u.
    parser.add_argument("--train-envs", type=int, default=1)
    # Seed sluzi da rezultati budu ponovljivi koliko je moguce.
    parser.add_argument("--seed", type=int, default=42)
    # Ovde ce ici modeli i video fajlovi.
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    # Ako dodamo ovu zastavicu, skripta ce snimiti video posle treninga.
    parser.add_argument("--record-video", action="store_true")
    # Kada je 1, snimamo best i worst evaluacionu epizodu.
    # Kada je >1, pored njih snimamo jos toliko dodatnih epizoda.
    parser.add_argument("--video-episodes", type=int, default=1)
    # Stable-Baselines3 moze da prikaze progress bar tokom treninga.
    parser.add_argument("--progress-bar", action="store_true")
    # Preset menja defaultni trening setup bez potrebe da kucamo gomilu parametara.
    parser.add_argument("--preset", choices=("default", "fast", "hardcore"), default="default")
    # Device govori gde PyTorch/SB3 pokusava da trenira model.
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    # Hardcore je teza verzija istog okruzenja.
    parser.add_argument("--hardcore", action="store_true")
    # Koliko puta ponavljamo istu akciju u wrapper-u.
    parser.add_argument("--frame-skip", type=int, default=None)
    # Koliko observation frejmova spajamo u jedan ulaz.
    parser.add_argument("--history-length", type=int, default=None)
    # Ako je postavljeno, menja terminalnu kaznu pri padu.
    parser.add_argument("--fall-penalty", type=float, default=None)
    # Ako hocemo sto brzi cycle, mozemo da preskocimo random baseline.
    parser.add_argument("--skip-random-baseline", action="store_true")
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
            "Pokretanje treninga | algo={} | timesteps={} | eval_ep={} | train_envs={} | seed={} | preset={} | device={} | video={} | hardcore={} | frame_skip={} | history={} | fall_penalty={} | skip_random={}",
            args.algo,
            args.timesteps,
            args.eval_episodes,
            args.train_envs,
            args.seed,
            args.preset,
            args.device,
            args.record_video,
            args.hardcore,
            args.frame_skip,
            args.history_length,
            args.fall_penalty,
            args.skip_random_baseline,
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
            train_envs=args.train_envs,
            skip_random_baseline=args.skip_random_baseline,
            video_folder=str(video_folder) if video_folder is not None else None,
            video_episodes=args.video_episodes,
            device=args.device,
            preset=args.preset,
            frame_skip=args.frame_skip,
            observation_history=args.history_length,
            fall_penalty=args.fall_penalty,
        )
        summary["log_file"] = str(log_path)
        summary_dir = args.output_dir / "summaries"
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / f"{args.algo}_bipedalwalker_seed{args.seed}.json"
        summary["summary_file"] = str(summary_path)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        if summary.get("random_baseline") is None:
            logger.info(
                "Gotovo | eval_mean_reward={:.2f} | model={}",
                summary["eval_mean_reward"],
                summary["saved_model_path"],
            )
        else:
            logger.info(
                "Gotovo | eval_mean_reward={:.2f} | random_mean={:.2f} | model={}",
                summary["eval_mean_reward"],
                summary["random_baseline"]["library"]["mean_reward"],
                summary["saved_model_path"],
            )
        logger.info("Log sacuvan | {}", log_path)
        logger.info("Summary sacuvan | {}", summary_path)

        # Na kraju stampamo kratak, pregledan rezime, a puni JSON summary
        # ostavljamo u zasebnom fajlu.
        print(format_terminal_summary(summary), file=sys.stderr, flush=True)
    finally:
        logger.remove(file_sink_id)


if __name__ == "__main__":
    main()
