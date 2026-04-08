"""Namenska skripta za BipedalWalkerHardcore-v3 eksperimente."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

from bipedal_workflow import DEFAULT_DEVICE, run_library_ppo, run_library_sac, run_library_td3
from train_bipedal_walker import ENV_ID, format_terminal_summary


def resolve_default_history_length(algo: str, history_length: int | None) -> int:
    """Vraca razumnu default istoriju observation-a za dati algoritam."""
    if history_length is not None:
        return max(int(history_length), 1)
    if algo == "ppo":
        return 4
    return 6


def main() -> None:
    """Pokrece hardcore-specifican trening sa smislenim defaultima."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="{time:HH:mm:ss} | {level:<8} | {message}",
        level="INFO",
    )

    parser = argparse.ArgumentParser(
        description="Train and evaluate a hardcore-focused Stable-Baselines3 agent on BipedalWalkerHardcore-v3.",
    )
    parser.add_argument("--algo", choices=("ppo", "sac", "td3"), default="sac")
    parser.add_argument("--timesteps", type=int, default=1_500_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--train-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts") / "hardcore")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-episodes", type=int, default=1)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=DEFAULT_DEVICE)
    parser.add_argument("--preset", choices=("default", "fast", "hardcore"), default="hardcore")
    parser.add_argument("--frame-skip", type=int, default=2)
    parser.add_argument("--history-length", type=int, default=None)
    parser.add_argument("--fall-penalty", type=float, default=-10.0)
    parser.add_argument("--skip-random-baseline", action="store_true")
    args = parser.parse_args()

    history_length = resolve_default_history_length(args.algo, args.history_length)

    log_dir = args.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"hardcore_{args.algo}_seed{args.seed}.log"
    file_sink_id = logger.add(
        log_path,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
        level="INFO",
        encoding="utf-8",
        mode="w",
    )

    runners = {
        "ppo": run_library_ppo,
        "sac": run_library_sac,
        "td3": run_library_td3,
    }
    model_save_path = args.output_dir / "models" / f"hardcore_{args.algo}_seed{args.seed}"
    video_folder = None
    if args.record_video:
        video_folder = args.output_dir / "videos" / f"hardcore_{args.algo}_seed{args.seed}"

    try:
        logger.info(
            "Pokretanje hardcore treninga | algo={} | timesteps={} | eval_ep={} | train_envs={} | seed={} | preset={} | frame_skip={} | history={} | fall_penalty={} | device={} | video={} | skip_random={}",
            args.algo,
            args.timesteps,
            args.eval_episodes,
            args.train_envs,
            args.seed,
            args.preset,
            args.frame_skip,
            history_length,
            args.fall_penalty,
            args.device,
            args.record_video,
            args.skip_random_baseline,
        )

        summary = runners[args.algo](
            env_id=ENV_ID,
            total_timesteps=args.timesteps,
            save_path=str(model_save_path),
            seed=args.seed,
            eval_episodes=args.eval_episodes,
            progress_bar=args.progress_bar,
            hardcore=True,
            train_envs=args.train_envs,
            skip_random_baseline=args.skip_random_baseline,
            video_folder=str(video_folder) if video_folder is not None else None,
            video_episodes=args.video_episodes,
            device=args.device,
            preset=args.preset,
            frame_skip=args.frame_skip,
            observation_history=history_length,
            fall_penalty=args.fall_penalty,
        )
        summary["log_file"] = str(log_path)
        summary_dir = args.output_dir / "summaries"
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / f"hardcore_{args.algo}_seed{args.seed}.json"
        summary["summary_file"] = str(summary_path)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        logger.info(
            "Hardcore trening gotov | eval_mean_reward={:.2f} | model={}",
            summary["eval_mean_reward"],
            summary["saved_model_path"],
        )
        logger.info("Log sacuvan | {}", log_path)
        logger.info("Summary sacuvan | {}", summary_path)
        print(format_terminal_summary(summary), file=sys.stderr, flush=True)
    finally:
        logger.remove(file_sink_id)


if __name__ == "__main__":
    main()
