from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "artifacts" / "archive" / "reports" / "runs"


@dataclass(frozen=True)
class ExperimentSpec:
    slug: str
    title: str
    category: str
    objective: str
    source_summary: str
    status: str
    verdict: str
    takeaway: str
    policy_paths: tuple[str, ...] = ()
    video_paths: tuple[str, ...] = ()
    extra_logs: tuple[str, ...] = ()
    train_log: str | None = None


EXPERIMENTS: tuple[ExperimentSpec, ...] = (
    ExperimentSpec(
        slug="01_sb3_ppo_hardcore_baseline",
        title="SB3 PPO Hardcore Baseline",
        category="legacy_baseline",
        objective="Proveriti koliko daleko standardni SB3 PPO moze da dogura na Hardcore modu bez custom sekvencijalnih modela.",
        source_summary="artifacts/summaries/ppo_bipedalwalker_seed42.json",
        status="partial_baseline",
        verdict="Poboljsava random baseline, ali i dalje ne daje upotrebljivo hodanje na Hardcore zadatku.",
        takeaway="Koristan kao referentni baseline i dokaz da sam standardni PPO nije dovoljan.",
    ),
    ExperimentSpec(
        slug="02_sb3_td3_hardcore_baseline",
        title="SB3 TD3 Hardcore Baseline",
        category="legacy_baseline",
        objective="Uporediti TD3 baseline sa PPO-om na istom Hardcore zadatku.",
        source_summary="artifacts/summaries/td3_bipedalwalker_seed42.json",
        status="failed_baseline",
        verdict="Politika uglavnom prezivljava do time-limit-a bez korisnog napretka i ostaje duboko negativna.",
        takeaway="Dobar negativan primer koji opravdava potrebu za custom portom i checkpoint evaluacijom.",
        video_paths=(
            "artifacts/archive/reports/sources/td3_hardcore_baseline_videos/td3_hardcore_baseline_best_seed46-episode-0.mp4",
            "artifacts/archive/reports/sources/td3_hardcore_baseline_videos/td3_hardcore_baseline_worst_seed44-episode-0.mp4",
        ),
    ),
    ExperimentSpec(
        slug="03_custom_sac_lstm_best_raw",
        title="Custom SAC + LSTM Best Raw Checkpoint",
        category="custom_port",
        objective="Izmeriti najbolji vanilla custom SAC+LSTM checkpoint na cistom Hardcore env-u.",
        source_summary="artifacts/archive/reports/eval/sac_lstm_h12_seed42_lr0p0004_bs64_fs2_fpm10_a0p01/summaries/test_summary.json",
        status="strong_partial",
        verdict="Najjaci rezultat u trenutnom repou: veoma stabilan checkpoint koji ne pada, ali i dalje ostaje ispod nule.",
        takeaway="Custom SAC + LSTM sa history ulazom pravi veliki pomak u odnosu na legacy baseline i predstavlja glavni kandidat za dalji rad.",
        train_log="artifacts/runs/hardcore/legacy_sac_lstm_h12_s42/train.log",
    ),
    ExperimentSpec(
        slug="04_custom_sac_lstm_antistall_transfer",
        title="Custom SAC + LSTM Anti-Stall Transfer Check",
        category="custom_port_diagnostic",
        objective="Proveriti da li anti-stall shaping samo popravlja trening dinamiku ili i realno transferuje na cist Hardcore env.",
        source_summary="artifacts/archive/reports/eval/sac_lstm_h12_seed42_lr0p0004_bs64_fs2_fpm10_a0p05/summaries/test_summary.json",
        status="diagnostic_only",
        verdict="Anti-stall pomaze treningu, ali checkpoint sa ep400 slabo generalizuje kada se meri bez anti-stall pravila.",
        takeaway="Koristan kao alat za razbijanje lokalnog minimuma, ali ne kao finalni kriterijum uspeha.",
        train_log="artifacts/archive/hardcore_misc/sac_lstm_h12_s42_a005_as_g80_w40_mp0p35_p2_spm20/train.log",
    ),
)


EVAL_RE = re.compile(
    r"Eval @ episode (?P<episode>\d+) \| raw_mean=(?P<raw>-?\d+(?:\.\d+)?) \| shaped_mean=(?P<shaped>-?\d+(?:\.\d+)?)"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    return path if path.is_absolute() else ROOT / path


def parse_eval_log(log_path: Path | None) -> list[dict[str, Any]]:
    if log_path is None or not log_path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = EVAL_RE.search(line)
        if not match:
            continue
        entries.append(
            {
                "episode": int(match.group("episode")),
                "raw_mean": float(match.group("raw")),
                "shaped_mean": float(match.group("shaped")),
            }
        )
    return entries


def extract_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    if "evaluation" in summary:
        evaluation = summary["evaluation"]
        return {
            "episodes": int(evaluation["episodes"]),
            "mean_reward": float(evaluation["mean_reward"]),
            "std_reward": float(evaluation["std_reward"]),
            "mean_shaped_reward": float(evaluation["mean_shaped_reward"]),
            "mean_length": float(evaluation["mean_length"]),
            "best_episode": evaluation.get("best_episode"),
            "worst_episode": evaluation.get("worst_episode"),
        }
    return {
        "episodes": int(summary.get("eval_episodes", 0)),
        "mean_reward": float(summary.get("eval_mean_reward", 0.0)),
        "std_reward": float(summary.get("eval_std_reward", 0.0)),
        "mean_shaped_reward": None,
        "mean_length": float(summary.get("eval_mean_episode_length", 0.0)),
        "best_episode": summary.get("best_eval_episode"),
        "worst_episode": summary.get("worst_eval_episode"),
    }


def infer_policy_files(summary: dict[str, Any], spec: ExperimentSpec) -> list[Path]:
    candidates = [resolve_path(raw_path) for raw_path in spec.policy_paths]
    if not candidates:
        for key in ("saved_model_path", "vecnormalize_path", "checkpoint_path"):
            candidates.append(resolve_path(summary.get(key)))
    return [path for path in candidates if path is not None and path.exists()]


def infer_video_files(summary: dict[str, Any], spec: ExperimentSpec) -> list[Path]:
    candidates = [resolve_path(raw_path) for raw_path in spec.video_paths]
    if not candidates:
        if "evaluation" in summary:
            candidates.extend(resolve_path(raw_path) for raw_path in summary["evaluation"].get("video_files", []))
        else:
            videos = summary.get("videos", {})
            best = videos.get("best_episode") or {}
            worst = videos.get("worst_episode") or {}
            candidates.extend([resolve_path(best.get("file")), resolve_path(worst.get("file"))])
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path is None or not path.exists() or path in seen:
            continue
        deduped.append(path)
        seen.add(path)
    return deduped


def infer_log_files(summary: dict[str, Any], spec: ExperimentSpec) -> list[Path]:
    candidates = [resolve_path(summary.get("log_file"))]
    candidates.extend(resolve_path(raw_path) for raw_path in spec.extra_logs)
    if spec.train_log is not None:
        candidates.append(resolve_path(spec.train_log))
    return [path for path in candidates if path is not None and path.exists()]


def safe_copy(src: Path, dest_dir: Path) -> str:
    dest_dir.mkdir(parents=True, exist_ok=True)
    destination = dest_dir / src.name
    if destination.resolve() != src.resolve():
        shutil.copy2(src, destination)
    return destination.name


def format_episode(episode: dict[str, Any] | None) -> str:
    if not episode:
        return "n/a"
    reward = episode.get("reward")
    reward_text = f"{float(reward):.2f}" if isinstance(reward, (int, float)) else "n/a"
    return f"reward={reward_text}, seed={episode.get('seed', 'n/a')}, length={episode.get('length', 'n/a')}"


def build_analysis_markdown(
    spec: ExperimentSpec,
    metrics: dict[str, Any],
    copied_policy_files: list[str],
    copied_video_files: list[str],
    copied_log_files: list[str],
    eval_entries: list[dict[str, Any]],
) -> str:
    lines = [
        f"# {spec.title}",
        "",
        "## Objective",
        spec.objective,
        "",
        "## Verdict",
        f"- Status: `{spec.status}`",
        f"- Zakljucak: {spec.verdict}",
        f"- Zasto je bitno: {spec.takeaway}",
        "",
        "## Key Metrics",
        f"- Evaluated episodes: {metrics['episodes']}",
        f"- Mean raw reward: {metrics['mean_reward']:.2f}",
        f"- Std raw reward: {metrics['std_reward']:.2f}",
        f"- Mean episode length: {metrics['mean_length']:.1f}",
    ]
    if metrics.get("mean_shaped_reward") is not None:
        lines.append(f"- Mean shaped reward: {metrics['mean_shaped_reward']:.2f}")
    lines.extend(
        [
            f"- Best episode: {format_episode(metrics.get('best_episode'))}",
            f"- Worst episode: {format_episode(metrics.get('worst_episode'))}",
            "",
        ]
    )
    if eval_entries:
        best_eval = max(eval_entries, key=lambda item: item["raw_mean"])
        latest_eval = eval_entries[-1]
        lines.extend(
            [
                "## Training Eval Highlights",
                f"- Best periodic eval: episode {best_eval['episode']} | raw_mean={best_eval['raw_mean']:.2f} | shaped_mean={best_eval['shaped_mean']:.2f}",
                f"- Latest periodic eval found in log: episode {latest_eval['episode']} | raw_mean={latest_eval['raw_mean']:.2f} | shaped_mean={latest_eval['shaped_mean']:.2f}",
                "",
            ]
        )
    lines.extend(
        [
            "## Bundled Artifacts",
            f"- Policy files: {len(copied_policy_files)}",
            f"- Video files: {len(copied_video_files)}",
            f"- Log files: {len(copied_log_files)}",
            "",
        ]
    )
    return "\n".join(lines)


def build_report_summary(
    spec: ExperimentSpec,
    source_summary_path: Path,
    metrics: dict[str, Any],
    copied_policy_files: list[str],
    copied_video_files: list[str],
    copied_log_files: list[str],
    eval_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    best_eval = max(eval_entries, key=lambda item: item["raw_mean"]) if eval_entries else None
    latest_eval = eval_entries[-1] if eval_entries else None
    return {
        "slug": spec.slug,
        "title": spec.title,
        "category": spec.category,
        "status": spec.status,
        "objective": spec.objective,
        "verdict": spec.verdict,
        "takeaway": spec.takeaway,
        "source_summary": str(source_summary_path.relative_to(ROOT)),
        "metrics": metrics,
        "best_periodic_eval": best_eval,
        "latest_periodic_eval": latest_eval,
        "policy_files": copied_policy_files,
        "video_files": copied_video_files,
        "log_files": copied_log_files,
    }


def build_index(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Report Artifact Bundle",
        "",
        "Ovaj folder okuplja glavne eksperimente u kratkim, stabilnim i citljivim report paketima.",
        "",
        "| Experiment | Status | Mean raw reward | Folder |",
        "|---|---|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['title']} | `{row['status']}` | {row['mean_reward']:.2f} | `{row['folder']}` |"
        )
    lines.extend(
        [
            "",
            "Svaki eksperiment sadrzi podfoldere `policy/`, `videos/`, `summary/` i fajl `analysis.md`.",
            "",
            "Bundle se regenerise komandom:",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe scripts\\curate_report_runs.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []

    for spec in EXPERIMENTS:
        run_dir = REPORT_ROOT / spec.slug
        if run_dir.exists():
            shutil.rmtree(run_dir)

        source_summary_path = ROOT / spec.source_summary
        if not source_summary_path.exists():
            raise FileNotFoundError(f"Source summary not found: {source_summary_path}")

        summary = read_json(source_summary_path)
        metrics = extract_metrics(summary)
        policy_files = infer_policy_files(summary, spec)
        video_files = infer_video_files(summary, spec)
        log_files = infer_log_files(summary, spec)
        eval_entries = parse_eval_log(resolve_path(spec.train_log))

        copied_policy_files = [safe_copy(path, run_dir / "policy") for path in policy_files]
        copied_video_files = [safe_copy(path, run_dir / "videos") for path in video_files]
        copied_log_files = [safe_copy(path, run_dir / "logs") for path in log_files]

        summary_dir = run_dir / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_summary_path, summary_dir / "source_summary.json")

        report_summary = build_report_summary(
            spec,
            source_summary_path,
            metrics,
            copied_policy_files,
            copied_video_files,
            copied_log_files,
            eval_entries,
        )
        (summary_dir / "report_summary.json").write_text(
            json.dumps(report_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (run_dir / "analysis.md").write_text(
            build_analysis_markdown(
                spec,
                metrics,
                copied_policy_files,
                copied_video_files,
                copied_log_files,
                eval_entries,
            ),
            encoding="utf-8",
        )

        manifest.append(report_summary)
        index_rows.append(
            {
                "title": spec.title,
                "status": spec.status,
                "mean_reward": metrics["mean_reward"],
                "folder": str(run_dir.relative_to(ROOT)),
            }
        )

    (REPORT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (REPORT_ROOT / "INDEX.md").write_text(build_index(index_rows), encoding="utf-8")


if __name__ == "__main__":
    main()
