"""Dedicated ugur-style hardcore port with custom SAC/TD3 and sequential models."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from hardcore_port import SACAgent, TD3Agent, evaluate_agent, make_hardcore_env, train_agent


def resolve_checkpoint_path(output_dir: Path, checkpoint: str) -> Path:
    """Resolves checkpoint aliases into actual files."""
    alias_map = {
        "best_raw": output_dir / "checkpoints" / "best_raw.pt",
        "best_shaped": output_dir / "checkpoints" / "best_shaped.pt",
        "last": output_dir / "checkpoints" / "last.pt",
    }
    if checkpoint in alias_map:
        return alias_map[checkpoint]
    return Path(checkpoint)


def resolve_history_length(algo: str, history_length: int | None) -> int:
    """Uses ugur-like defaults when history length is not explicitly set."""
    if history_length is not None:
        return max(int(history_length), 1)
    return 12 if algo == "sac" else 6


def format_run_value(value: float) -> str:
    """Formats floats into short filesystem-friendly tokens."""
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def resolve_video_label(checkpoint: str) -> str:
    """Builds a filesystem-friendly checkpoint label for video outputs."""
    if checkpoint in {"best_raw", "best_shaped", "last"}:
        return checkpoint
    return Path(checkpoint).stem or "checkpoint"


def build_env_factory(
    *,
    env_id: str,
    history_length: int,
    frame_skip: int,
    fall_penalty: float,
    anti_stall: bool,
    stall_check_window: int,
    stall_grace_steps: int,
    stall_min_progress: float,
    stall_patience: int,
    stall_penalty: float,
    render_mode: str | None = None,
):
    """Creates a reusable env factory closure."""
    return lambda: make_hardcore_env(
        env_id=env_id,
        history_length=history_length,
        frame_skip=frame_skip,
        fall_penalty=fall_penalty,
        anti_stall=anti_stall,
        stall_check_window=stall_check_window,
        stall_grace_steps=stall_grace_steps,
        stall_min_progress=stall_min_progress,
        stall_patience=stall_patience,
        stall_penalty=stall_penalty,
        render_mode=render_mode,
    )


def build_agent(args: argparse.Namespace, *, state_dim: int, action_dim: int, action_low, action_high):
    """Builds the requested custom agent."""
    agent_kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "backbone": args.backbone,
        "history_length": args.history_length,
        "action_low": action_low,
        "action_high": action_high,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "gamma": args.gamma,
        "batch_size": args.batch_size,
        "buffer_size": args.buffer_size,
        "device": args.device,
    }
    if args.algo == "sac":
        return SACAgent(
            alpha=args.alpha,
            tau=args.tau,
            update_freq=args.update_freq,
            **agent_kwargs,
        )
    return TD3Agent(
        tau=args.tau,
        update_freq=args.update_freq,
        **agent_kwargs,
    )


def build_run_name(args: argparse.Namespace) -> str:
    """Creates a run name that keeps different experiment settings separate."""
    parts = [
        args.algo,
        args.backbone,
        f"h{args.history_length}",
        f"seed{args.seed}",
        f"lr{format_run_value(args.lr)}",
        f"bs{int(args.batch_size)}",
        f"fs{int(args.frame_skip)}",
        f"fp{format_run_value(args.fall_penalty)}",
    ]
    if args.algo == "sac":
        parts.append(f"a{format_run_value(args.alpha)}")
    if args.anti_stall:
        parts.extend(
            [
                "astall",
                f"gw{int(args.stall_grace_steps)}",
                f"cw{int(args.stall_check_window)}",
                f"mp{format_run_value(args.stall_min_progress)}",
                f"pt{int(args.stall_patience)}",
                f"sp{format_run_value(args.stall_penalty)}",
            ]
        )
    return "_".join(parts)


def format_summary(summary: dict[str, Any]) -> str:
    """Builds a concise terminal summary."""
    lines = [
        "",
        "===== Custom Hardcore Port =====",
        f"Algoritam: {summary['algorithm'].upper()} | backbone: {summary['backbone'].upper()}",
        f"Okruzenje: {summary['env_id']}",
        f"History: {summary['history_length']} | frame_skip: {summary['frame_skip']} | fall_penalty: {summary['fall_penalty']}",
    ]
    if summary.get("anti_stall"):
        lines.append(
            "Anti-stall: on"
            f" | grace={summary['stall_grace_steps']}"
            f" | window={summary['stall_check_window']}"
            f" | min_progress={summary['stall_min_progress']}"
            f" | patience={summary['stall_patience']}"
            f" | penalty={summary['stall_penalty']}"
        )
    if summary["mode"] == "train":
        lines.extend(
            [
                f"Epizode: {summary['episodes_completed']} | max_steps: {summary['max_steps']}",
                f"Best raw ckpt: {summary.get('best_raw_checkpoint')}",
                f"Best shaped ckpt: {summary.get('best_shaped_checkpoint')}",
                f"Last ckpt: {summary.get('last_checkpoint')}",
            ]
        )
        if summary.get("final_eval") is not None:
            final_eval = summary["final_eval"]
            lines.extend(
                [
                    "",
                    "Finalna evaluacija izabranog checkpoint-a:",
                    f"- mean raw reward: {float(final_eval['mean_reward']):.2f}",
                    f"- mean shaped reward: {float(final_eval['mean_shaped_reward']):.2f}",
                    f"- std raw reward: {float(final_eval['std_reward']):.2f}",
                    f"- prosecan broj koraka: {float(final_eval['mean_length']):.1f}",
                    f"- best ep: reward={float(final_eval['best_episode']['reward']):.2f} | seed={int(final_eval['best_episode']['seed'])}",
                    f"- worst ep: reward={float(final_eval['worst_episode']['reward']):.2f} | seed={int(final_eval['worst_episode']['seed'])}",
                ]
            )
    else:
        evaluation = summary["evaluation"]
        lines.extend(
            [
                f"Checkpoint: {summary['checkpoint_path']}",
                "",
                "Evaluacija:",
                f"- mean raw reward: {float(evaluation['mean_reward']):.2f}",
                f"- mean shaped reward: {float(evaluation['mean_shaped_reward']):.2f}",
                f"- std raw reward: {float(evaluation['std_reward']):.2f}",
                f"- prosecan broj koraka: {float(evaluation['mean_length']):.1f}",
            ]
        )
        if evaluation.get("video_files"):
            lines.append(f"- video fajlova: {len(evaluation['video_files'])}")
    lines.extend(
        [
            "",
            "Artefakti:",
            f"- log: {summary['log_file']}",
            f"- summary: {summary['summary_file']}",
        ]
    )
    if summary.get("video_dir") is not None:
        lines.append(f"- video dir: {summary['video_dir']}")
    return "\n".join(lines)


def main() -> None:
    """Main CLI entrypoint."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="{time:HH:mm:ss} | {level:<8} | {message}",
        level="INFO",
    )

    parser = argparse.ArgumentParser(
        description="Faithful-ish ugur-style custom SAC/TD3 port for BipedalWalkerHardcore-v3.",
    )
    parser.add_argument("--mode", choices=("train", "test", "test-100"), default="train")
    parser.add_argument("--algo", choices=("sac", "td3"), default="sac")
    parser.add_argument("--backbone", choices=("lstm", "transformer"), default="lstm")
    parser.add_argument("--env-id", default="BipedalWalkerHardcore-v3")
    parser.add_argument("--history-length", type=int, default=None)
    parser.add_argument("--frame-skip", type=int, default=2)
    parser.add_argument("--fall-penalty", type=float, default=-10.0)
    parser.add_argument("--anti-stall", action="store_true")
    parser.add_argument("--stall-check-window", type=int, default=40)
    parser.add_argument("--stall-grace-steps", type=int, default=80)
    parser.add_argument("--stall-min-progress", type=float, default=0.35)
    parser.add_argument("--stall-patience", type=int, default=2)
    parser.add_argument("--stall-penalty", type=float, default=-20.0)
    parser.add_argument("--episodes", type=int, default=8_000)
    parser.add_argument("--explore-episodes", type=int, default=50)
    parser.add_argument("--eval-frequency", type=int, default=200)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--final-eval-episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=750)
    parser.add_argument("--score-limit", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--update-freq", type=int, default=None)
    parser.add_argument("--checkpoint", default="best_raw")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-episodes", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "hardcore_port",
    )
    args = parser.parse_args()

    if args.update_freq is None:
        args.update_freq = 1 if args.algo == "sac" else 2
    args.history_length = resolve_history_length(args.algo, args.history_length)

    run_name = build_run_name(args)
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{args.mode}.log"
    file_sink_id = logger.add(
        log_path,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
        level="INFO",
        encoding="utf-8",
        mode="w",
    )

    probe_env = make_hardcore_env(
        env_id=args.env_id,
        history_length=args.history_length,
        frame_skip=args.frame_skip,
        fall_penalty=args.fall_penalty,
        anti_stall=args.anti_stall,
        stall_check_window=args.stall_check_window,
        stall_grace_steps=args.stall_grace_steps,
        stall_min_progress=args.stall_min_progress,
        stall_patience=args.stall_patience,
        stall_penalty=args.stall_penalty,
    )
    try:
        state_dim = int(probe_env.observation_space.shape[-1])
        action_dim = int(probe_env.action_space.shape[-1])
        action_low = probe_env.action_space.low
        action_high = probe_env.action_space.high
    finally:
        probe_env.close()

    agent = build_agent(
        args,
        state_dim=state_dim,
        action_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
    )

    try:
        logger.info(
            "Pokretanje custom hardcore porta | mode={} | algo={} | backbone={} | history={} | frame_skip={} | fall_penalty={} | lr={} | batch={} | gamma={} | alpha={} | tau={} | device={}",
            args.mode,
            args.algo,
            args.backbone,
            args.history_length,
            args.frame_skip,
            args.fall_penalty,
            args.lr,
            args.batch_size,
            args.gamma,
            args.alpha,
            args.tau,
            args.device,
        )

        summary_dir = output_dir / "summaries"
        summary_dir.mkdir(parents=True, exist_ok=True)
        train_history_path = output_dir / "training_history.json"
        eval_history_path = output_dir / "evaluation_history.json"

        env_factory = build_env_factory(
            env_id=args.env_id,
            history_length=args.history_length,
            frame_skip=args.frame_skip,
            fall_penalty=args.fall_penalty,
            anti_stall=args.anti_stall,
            stall_check_window=args.stall_check_window,
            stall_grace_steps=args.stall_grace_steps,
            stall_min_progress=args.stall_min_progress,
            stall_patience=args.stall_patience,
            stall_penalty=args.stall_penalty,
        )
        eval_env_factory = build_env_factory(
            env_id=args.env_id,
            history_length=args.history_length,
            frame_skip=args.frame_skip,
            fall_penalty=args.fall_penalty,
            anti_stall=args.anti_stall,
            stall_check_window=args.stall_check_window,
            stall_grace_steps=args.stall_grace_steps,
            stall_min_progress=args.stall_min_progress,
            stall_patience=args.stall_patience,
            stall_penalty=args.stall_penalty,
            render_mode="rgb_array" if args.record_video else None,
        )
        checkpoint_label = resolve_video_label(args.checkpoint)
        video_dir = (
            output_dir / "videos" / f"{args.mode}_{checkpoint_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if args.record_video
            else None
        )

        if args.mode == "train":
            training_result = train_agent(
                env_factory,
                agent,
                episodes=args.episodes,
                explore_episodes=args.explore_episodes,
                eval_frequency=args.eval_frequency,
                eval_episodes=args.eval_episodes,
                max_steps=args.max_steps,
                score_limit=args.score_limit,
                checkpoint_dir=output_dir / "checkpoints",
                seed=args.seed,
            )
            train_history_path.write_text(
                json.dumps(training_result["training_history"], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            eval_history_path.write_text(
                json.dumps(training_result["evaluation_history"], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            final_eval = None
            checkpoint_path = resolve_checkpoint_path(output_dir, args.checkpoint)
            if checkpoint_path.exists():
                eval_agent = build_agent(
                    args,
                    state_dim=state_dim,
                    action_dim=action_dim,
                    action_low=action_low,
                    action_high=action_high,
                )
                eval_agent.load_checkpoint(checkpoint_path)
                final_eval = evaluate_agent(
                    eval_env_factory,
                    eval_agent,
                    episodes=args.final_eval_episodes,
                    max_steps=args.max_steps,
                    seed_start=args.seed + 500_000,
                    video_folder=video_dir,
                    video_prefix=f"{args.algo}_{args.backbone}_{checkpoint_label}",
                    video_episodes=args.video_episodes,
                )
            else:
                logger.info("Trazeni checkpoint za finalnu evaluaciju ne postoji | {}", checkpoint_path)

            summary = {
                "mode": "train",
                "algorithm": args.algo,
                "backbone": args.backbone,
                "env_id": args.env_id,
                "history_length": int(args.history_length),
                "frame_skip": int(args.frame_skip),
                "fall_penalty": float(args.fall_penalty),
                "anti_stall": bool(args.anti_stall),
                "stall_check_window": int(args.stall_check_window),
                "stall_grace_steps": int(args.stall_grace_steps),
                "stall_min_progress": float(args.stall_min_progress),
                "stall_patience": int(args.stall_patience),
                "stall_penalty": float(args.stall_penalty),
                "episodes_completed": int(training_result["episodes_completed"]),
                "max_steps": int(args.max_steps),
                "best_raw_checkpoint": training_result["best_raw_checkpoint"],
                "best_shaped_checkpoint": training_result["best_shaped_checkpoint"],
                "last_checkpoint": training_result["last_checkpoint"],
                "best_eval_mean_reward": training_result["best_eval_mean_reward"],
                "best_eval_mean_shaped_reward": training_result["best_eval_mean_shaped_reward"],
                "training_history_file": str(train_history_path),
                "evaluation_history_file": str(eval_history_path),
                "selected_checkpoint": str(resolve_checkpoint_path(output_dir, args.checkpoint)),
                "final_eval": final_eval,
                "log_file": str(log_path),
                "video_dir": None if final_eval is None else final_eval.get("video_folder"),
            }
        else:
            checkpoint_path = resolve_checkpoint_path(output_dir, args.checkpoint)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint nije pronadjen: {checkpoint_path}")
            agent.load_checkpoint(checkpoint_path)
            eval_episodes = args.eval_episodes if args.mode == "test" else args.final_eval_episodes
            evaluation = evaluate_agent(
                eval_env_factory,
                agent,
                episodes=eval_episodes,
                max_steps=args.max_steps,
                seed_start=args.seed + 700_000,
                video_folder=video_dir,
                video_prefix=f"{args.algo}_{args.backbone}_{checkpoint_label}",
                video_episodes=args.video_episodes,
            )
            summary = {
                "mode": args.mode,
                "algorithm": args.algo,
                "backbone": args.backbone,
                "env_id": args.env_id,
                "history_length": int(args.history_length),
                "frame_skip": int(args.frame_skip),
                "fall_penalty": float(args.fall_penalty),
                "anti_stall": bool(args.anti_stall),
                "stall_check_window": int(args.stall_check_window),
                "stall_grace_steps": int(args.stall_grace_steps),
                "stall_min_progress": float(args.stall_min_progress),
                "stall_patience": int(args.stall_patience),
                "stall_penalty": float(args.stall_penalty),
                "checkpoint_path": str(checkpoint_path),
                "evaluation": evaluation,
                "log_file": str(log_path),
                "video_dir": evaluation.get("video_folder"),
            }

        summary_path = summary_dir / f"{args.mode}_summary.json"
        summary["summary_file"] = str(summary_path)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(format_summary(summary), file=sys.stderr, flush=True)
    finally:
        logger.remove(file_sink_id)


if __name__ == "__main__":
    main()
