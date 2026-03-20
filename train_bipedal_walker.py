"""Train, evaluate, and optionally record a BipedalWalker-v3 policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ppo_method import run_library_ppo
from sac_method import run_library_sac
from td3_method import run_library_td3


ENV_ID = "BipedalWalker-v3"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a Stable-Baselines3 agent on BipedalWalker-v3.",
    )
    parser.add_argument("--algo", choices=("ppo", "sac", "td3"), default="ppo")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-episodes", type=int, default=1)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--hardcore", action="store_true")
    args = parser.parse_args()

    runners = {
        "ppo": run_library_ppo,
        "sac": run_library_sac,
        "td3": run_library_td3,
    }

    model_save_path = args.output_dir / "models" / f"{args.algo}_bipedalwalker_seed{args.seed}"
    video_folder = None
    if args.record_video:
        video_folder = args.output_dir / "videos" / f"{args.algo}_bipedalwalker_seed{args.seed}"

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

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
