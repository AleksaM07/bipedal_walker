"""Generate reproducible figures for the BipedalWalker project report.

All machine-readable metrics are loaded from files in the repository. The
three standard-environment values are the only exception: they currently
exist in the report and presentation, but not in matching JSON result files.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent
FIGURE_DIR = PROJECT_ROOT / "report_figures"

PPO_HARDCORE_PATH = (
    PROJECT_ROOT
    / "artifacts/runs/standard/ppo_bipedalwalker_seed42/test_summary.json"
)
TD3_HARDCORE_PATH = (
    PROJECT_ROOT
    / "artifacts/runs/standard/td3_bipedalwalker_seed42/test_summary.json"
)
LEGACY_SAC_LSTM_PATH = (
    PROJECT_ROOT
    / "artifacts/runs/hardcore/legacy_sac_lstm_h12_s42/test_summary.json"
)
BREAKTHROUGH_PATH = (
    PROJECT_ROOT / "artifacts/runs/hardcore/fix_eval_best_raw/test_summary.json"
)
RESUMED_EVAL_PATH = (
    PROJECT_ROOT / "artifacts/runs/hardcore/res_eval_best_raw/test_summary.json"
)
EP4600_EVAL_PATH = (
    PROJECT_ROOT
    / "historical_videos/"
    "sac_lstm_h12_seed42_lr0p0004_bs64_fs2_fpm10_a0p01/"
    "summaries/test_summary.json"
)
FIX_TRAIN_LOG = (
    PROJECT_ROOT / "artifacts/runs/hardcore/fix_train_a001_as/train.log"
)
RESUMED_TRAIN_LOG = (
    PROJECT_ROOT / "artifacts/runs/hardcore/res_train_a001_as/train.log"
)

TRAIN_PATTERN = re.compile(
    r"Episode (?P<episode>\d+) \| raw=(?P<raw>-?\d+(?:\.\d+)?) "
    r"\| shaped=(?P<shaped>-?\d+(?:\.\d+)?) "
    r"\| avg100_raw=(?P<average>-?\d+(?:\.\d+)?) "
    r"\| steps=(?P<steps>\d+)"
)
EVAL_PATTERN = re.compile(
    r"Eval @ episode (?P<episode>\d+) "
    r"\| raw_mean=(?P<raw>-?\d+(?:\.\d+)?) "
    r"\| shaped_mean=(?P<shaped>-?\d+(?:\.\d+)?)"
)


@dataclass(frozen=True)
class Evaluation:
    """Normalized evaluation metrics loaded from a JSON summary."""

    mean: float
    std: float
    rewards: list[float]
    lengths: list[int]
    episodes: int
    environment: str
    algorithm: str


def require_files(paths: list[Path]) -> None:
    """Raise a clear error if any required input file is missing."""
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in paths if not path.is_file()]
    if missing:
        joined = "\n  - ".join(missing)
        raise FileNotFoundError(f"Nedostaju ulazni fajlovi:\n  - {joined}")


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from the repository."""
    try:
        with path.open(encoding="utf-8") as input_file:
            data = json.load(input_file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Neispravan JSON fajl: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Očekivan je JSON objekat: {path}")
    return data


def load_evaluation(path: Path) -> Evaluation:
    """Load either an SB3 or custom SAC evaluation summary."""
    data = load_json(path)
    evaluation = data.get("evaluation", data)

    rewards = evaluation.get("rewards", evaluation.get("eval_rewards"))
    lengths = evaluation.get("lengths", evaluation.get("eval_episode_lengths", []))
    mean = evaluation.get("mean_reward", evaluation.get("eval_mean_reward"))
    std = evaluation.get("std_reward", evaluation.get("eval_std_reward"))
    episodes = evaluation.get("episodes", evaluation.get("eval_episodes"))

    if rewards is None or mean is None or std is None or episodes is None:
        raise ValueError(f"Nepotpuna evaluacija u fajlu: {path}")

    parsed_rewards = [float(value) for value in rewards]
    if len(parsed_rewards) != int(episodes):
        raise ValueError(
            f"Broj nagrada ne odgovara broju epizoda u fajlu: {path}"
        )

    return Evaluation(
        mean=float(mean),
        std=float(std),
        rewards=parsed_rewards,
        lengths=[int(value) for value in lengths],
        episodes=int(episodes),
        environment=str(data.get("env_id", "")),
        algorithm=str(data.get("algorithm", "")),
    )


def save_figure(figure: plt.Figure, filename: str) -> None:
    """Save a report figure with consistent print quality."""
    output_path = FIGURE_DIR / filename
    figure.savefig(output_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    LOGGER.info("Kreirano: %s", output_path.relative_to(PROJECT_ROOT))


def configure_plot_style() -> None:
    """Set a readable style suitable for an A4 report."""
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 15,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def capture_environment_frame(environment_id: str, seed: int) -> np.ndarray:
    """Render a deterministic initial frame from a Gymnasium environment."""
    environment = gym.make(environment_id, render_mode="rgb_array")
    try:
        environment.reset(seed=seed)
        frame = environment.render()
    finally:
        environment.close()
    if frame is None:
        raise RuntimeError(f"Okruženje nije vratilo kadar: {environment_id}")
    return np.asarray(frame)


def plot_environment_comparison() -> None:
    """Compare deterministic frames of the normal and hardcore terrain."""
    standard = capture_environment_frame("BipedalWalker-v3", seed=42)
    hardcore = capture_environment_frame("BipedalWalkerHardcore-v3", seed=42)

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    panels = [
        (axes[0], standard, "(a) BipedalWalker-v3"),
        (axes[1], hardcore, "(b) BipedalWalkerHardcore-v3"),
    ]
    for axis, frame, title in panels:
        axis.imshow(frame)
        axis.set_title(title, fontweight="bold")
        axis.axis("off")
    figure.suptitle("Poređenje okruženja, seed=42", fontweight="bold")
    figure.tight_layout()
    save_figure(figure, "environment_comparison.png")


def add_box(
    axis: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    color: str,
) -> None:
    """Add a rounded diagram box."""
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.6,
        edgecolor="#263238",
        facecolor=color,
    )
    axis.add_patch(patch)
    axis.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )


def add_arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#37474f",
) -> None:
    """Add an arrow to the architecture diagram."""
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.6,
            color=color,
        )
    )


def plot_observation_action_diagram() -> None:
    """Draw the reproducible observation-history-actor-critic flow."""
    figure, axis = plt.subplots(figsize=(12, 5.8))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    boxes = [
        ((0.03, 0.57), 0.16, 0.18, "24 vrednosti\nopservacije", "#e3f2fd"),
        ((0.23, 0.57), 0.18, 0.18, "Istorija poslednjih\n12 opservacija", "#e8eaf6"),
        ((0.45, 0.57), 0.14, 0.18, "LSTM\nenkoder", "#ede7f6"),
        ((0.64, 0.57), 0.13, 0.18, "SAC\nactor", "#e8f5e9"),
        ((0.82, 0.57), 0.15, 0.18, "4 kontinualne\nakcije", "#fff3e0"),
        ((0.82, 0.18), 0.15, 0.16, "Motori kukova\ni kolena", "#fbe9e7"),
        ((0.52, 0.18), 0.18, 0.16, "Dva critic modela\n(samo trening)", "#f3e5f5"),
        ((0.26, 0.18), 0.17, 0.16, "Replay buffer\n(s, a, r, s')", "#eceff1"),
    ]
    for xy, width, height, text, color in boxes:
        add_box(axis, xy, width, height, text, color)

    for start, end in [
        ((0.19, 0.66), (0.23, 0.66)),
        ((0.41, 0.66), (0.45, 0.66)),
        ((0.59, 0.66), (0.64, 0.66)),
        ((0.77, 0.66), (0.82, 0.66)),
        ((0.895, 0.57), (0.895, 0.34)),
        ((0.43, 0.26), (0.52, 0.26)),
    ]:
        add_arrow(axis, start, end)

    add_arrow(axis, (0.705, 0.57), (0.63, 0.34), "#7b1fa2")
    add_arrow(axis, (0.82, 0.60), (0.70, 0.32), "#7b1fa2")
    axis.text(
        0.70,
        0.43,
        "akcija i latentno stanje",
        ha="center",
        color="#6a1b9a",
        fontsize=9,
    )
    axis.text(
        0.5,
        0.92,
        "Tok odluke SAC--LSTM agenta",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
    )
    save_figure(figure, "observation_action_diagram.png")


def plot_standard_algorithms() -> None:
    """Plot standard-environment values that exist only in report materials."""
    algorithms = ["PPO", "SAC", "TD3"]

    # Vrednost je preuzeta iz postojećeg izveštaja i nije potvrđena JSON fajlom.
    means = np.array([125.28, 282.08, 299.54])
    # Vrednost je preuzeta iz postojećeg izveštaja i nije potvrđena JSON fajlom.
    standard_deviations = np.array([131.29, 0.91, 0.46])

    figure, axis = plt.subplots(figsize=(8.6, 5.4))
    colors = ["#78909c", "#26a69a", "#1565c0"]
    bars = axis.bar(
        algorithms,
        means,
        yerr=standard_deviations,
        capsize=6,
        color=colors,
        edgecolor="#263238",
        linewidth=0.8,
    )
    axis.axhline(300, color="#c62828", linestyle="--", linewidth=1.4)
    axis.text(-0.45, 305, "približan prag: 300", color="#b71c1c", ha="left")
    axis.set_ylabel("Srednja evaluaciona nagrada")
    axis.set_title("Standardno okruženje: prijavljeni rezultati")
    axis.grid(axis="y", alpha=0.25)
    axis.set_ylim(-25, 350)
    for bar, value in zip(bars, means, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 8,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    axis.text(
        0.01,
        0.01,
        "Izvor: postojeći izveštaj/prezentacija; nema odgovarajućih JSON potvrda.",
        transform=axis.transAxes,
        fontsize=9,
        color="#455a64",
    )
    save_figure(figure, "standard_algorithms_rewards.png")


def get_random_baseline(data: dict[str, Any]) -> Evaluation:
    """Extract the random baseline embedded in the PPO summary."""
    baseline = data.get("random_baseline", {}).get("manual")
    if not baseline:
        raise ValueError(f"Nedostaje random baseline u fajlu: {PPO_HARDCORE_PATH}")
    rewards = [float(value) for value in baseline["rewards"]]
    return Evaluation(
        mean=float(baseline["mean_reward"]),
        std=float(baseline["std_reward"]),
        rewards=rewards,
        lengths=[],
        episodes=len(rewards),
        environment="BipedalWalkerHardcore-v3",
        algorithm="random",
    )


def plot_hardcore_progression() -> None:
    """Plot the progression using only repository JSON evaluations."""
    ppo_data = load_json(PPO_HARDCORE_PATH)
    random_eval = get_random_baseline(ppo_data)
    evaluations = [
        random_eval,
        load_evaluation(PPO_HARDCORE_PATH),
        load_evaluation(TD3_HARDCORE_PATH),
        load_evaluation(LEGACY_SAC_LSTM_PATH),
        load_evaluation(BREAKTHROUGH_PATH),
        load_evaluation(RESUMED_EVAL_PATH),
        load_evaluation(EP4600_EVAL_PATH),
    ]
    labels = [
        "Random",
        "PPO",
        "TD3",
        "Legacy\nSAC--LSTM",
        "Prvi proboj\n+ anti-stall",
        "Best-raw\ncheckpoint",
        "Checkpoint\nep4600",
    ]

    means = np.array([evaluation.mean for evaluation in evaluations])
    errors = np.array([evaluation.std for evaluation in evaluations])
    counts = [evaluation.episodes for evaluation in evaluations]
    colors = [
        "#b0bec5",
        "#90a4ae",
        "#78909c",
        "#9575cd",
        "#5c6bc0",
        "#26a69a",
        "#00897b",
    ]

    figure, axis = plt.subplots(figsize=(12.2, 6.3))
    x_positions = np.arange(len(labels))
    bars = axis.bar(
        x_positions,
        means,
        yerr=errors,
        capsize=5,
        color=colors,
        edgecolor="#263238",
        linewidth=0.8,
    )
    axis.axhline(0, color="#263238", linewidth=0.9)
    axis.axhline(300, color="#c62828", linestyle="--", linewidth=1.5)
    axis.text(
        len(labels) - 0.05,
        306,
        "približan prag rešavanja: 300",
        color="#b71c1c",
        ha="right",
    )
    axis.set_xticks(x_positions, labels)
    axis.set_ylabel("Srednja sirova evaluaciona nagrada")
    axis.set_title("Napredak potvrđenih evaluacija na hardcore zadatku")
    axis.grid(axis="y", alpha=0.25)
    axis.set_ylim(-175, 350)

    for bar, value, count in zip(bars, means, counts, strict=True):
        offset = 10 if value >= 0 else -16
        vertical_alignment = "bottom" if value >= 0 else "top"
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:.2f}\n(n={count})",
            ha="center",
            va=vertical_alignment,
            fontsize=9,
            fontweight="bold",
        )
    save_figure(figure, "hardcore_model_progression.png")


def plot_best_evaluation_episodes() -> None:
    """Plot individual rewards from the strongest confirmed five-episode test."""
    evaluation = load_evaluation(EP4600_EVAL_PATH)
    if evaluation.episodes != 5:
        raise ValueError(
            "Očekivano je pet epizoda u najjačem potvrđenom evaluacionom paketu."
        )

    x_positions = np.arange(1, evaluation.episodes + 1)
    figure, axis = plt.subplots(figsize=(9.2, 5.4))
    bars = axis.bar(
        x_positions,
        evaluation.rewards,
        color=["#00796b", "#80cbc4", "#26a69a", "#00796b", "#4db6ac"],
        edgecolor="#263238",
        linewidth=0.8,
    )
    axis.axhline(
        evaluation.mean,
        color="#1565c0",
        linewidth=1.6,
        label=f"Srednja vrednost = {evaluation.mean:.2f}",
    )
    axis.axhline(
        300,
        color="#c62828",
        linestyle="--",
        linewidth=1.5,
        label="Približan prag = 300",
    )
    axis.set_xlabel("Evaluaciona epizoda")
    axis.set_ylabel("Sirova nagrada")
    axis.set_title("Checkpoint ep4600: pojedinačne epizode")
    axis.set_xticks(x_positions)
    axis.set_ylim(0, 330)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="lower right")
    for bar, value in zip(bars, evaluation.rewards, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 6,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    save_figure(figure, "best_evaluation_episodes.png")


def parse_training_log(path: Path) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Parse per-episode and periodic-evaluation entries from a training log."""
    episodes: list[dict[str, float]] = []
    evaluations: list[dict[str, float]] = []
    with path.open(encoding="utf-8") as log_file:
        for line in log_file:
            train_match = TRAIN_PATTERN.search(line)
            if train_match:
                episodes.append(
                    {
                        key: float(value)
                        for key, value in train_match.groupdict().items()
                    }
                )
                continue
            evaluation_match = EVAL_PATTERN.search(line)
            if evaluation_match:
                evaluations.append(
                    {
                        key: float(value)
                        for key, value in evaluation_match.groupdict().items()
                    }
                )
    if not episodes:
        raise ValueError(f"Nema epizodnih podataka u logu: {path}")
    return episodes, evaluations


def plot_training_curve() -> None:
    """Plot the anti-stall training and its explicitly logged continuation."""
    fix_episodes, fix_evaluations = parse_training_log(FIX_TRAIN_LOG)
    resumed_episodes, resumed_evaluations = parse_training_log(RESUMED_TRAIN_LOG)

    # The resumed log explicitly states warm-start from episode 3200.
    fix_episodes = [entry for entry in fix_episodes if entry["episode"] <= 3200]
    fix_evaluations = [
        entry for entry in fix_evaluations if entry["episode"] <= 3200
    ]
    resumed_episodes = [
        entry for entry in resumed_episodes if entry["episode"] >= 3201
    ]
    combined_episodes = fix_episodes + resumed_episodes

    steps_by_episode: dict[int, float] = {}
    cumulative_steps = 0.0
    raw_rewards: list[float] = []
    moving_average: list[float] = []
    step_values: list[float] = []
    rolling_window: list[float] = []
    for entry in combined_episodes:
        cumulative_steps += entry["steps"]
        episode = int(entry["episode"])
        steps_by_episode[episode] = cumulative_steps
        raw_rewards.append(entry["raw"])
        rolling_window.append(entry["raw"])
        if len(rolling_window) > 100:
            rolling_window.pop(0)
        moving_average.append(float(np.mean(rolling_window)))
        step_values.append(cumulative_steps)

    all_evaluations = fix_evaluations + resumed_evaluations
    evaluation_steps: list[float] = []
    evaluation_rewards: list[float] = []
    for entry in all_evaluations:
        episode = int(entry["episode"])
        if episode in steps_by_episode:
            evaluation_steps.append(steps_by_episode[episode])
            evaluation_rewards.append(entry["raw"])

    continuation_step = steps_by_episode.get(3200)
    if continuation_step is None:
        raise ValueError("Nije moguće pronaći granicu nastavka treninga.")

    figure, axis = plt.subplots(figsize=(11.6, 6.2))
    axis.scatter(
        step_values,
        raw_rewards,
        s=5,
        alpha=0.12,
        color="#607d8b",
        label="Sirova nagrada po trening epizodi",
    )
    axis.plot(
        step_values,
        moving_average,
        linewidth=2.1,
        color="#1565c0",
        label="Pokretni prosek (100 epizoda)",
    )
    axis.plot(
        evaluation_steps,
        evaluation_rewards,
        marker="o",
        markersize=4,
        linewidth=1.3,
        color="#d84315",
        label="Periodična čista evaluacija (20 epizoda)",
    )
    axis.axvline(
        continuation_step,
        color="#6a1b9a",
        linestyle="--",
        linewidth=1.5,
        label="Nastavak iz checkpoint-a, epizoda 3200",
    )
    axis.axhline(300, color="#c62828", linestyle=":", linewidth=1.3)
    axis.set_xlabel("Kumulativni koraci politike (frame_skip=2)")
    axis.set_ylabel("Sirova nagrada")
    axis.set_title("SAC--LSTM anti-stall trening i nastavljena linija")
    axis.grid(alpha=0.2)
    axis.legend(loc="upper left")
    save_figure(figure, "training_curve_sac_lstm.png")


def validate_environment_names() -> None:
    """Ensure all JSON-backed hardcore figures use the expected environment."""
    paths = [
        PPO_HARDCORE_PATH,
        TD3_HARDCORE_PATH,
        LEGACY_SAC_LSTM_PATH,
        BREAKTHROUGH_PATH,
        RESUMED_EVAL_PATH,
        EP4600_EVAL_PATH,
    ]
    for path in paths:
        evaluation = load_evaluation(path)
        if evaluation.environment != "BipedalWalkerHardcore-v3":
            raise ValueError(
                f"Neočekivano okruženje {evaluation.environment!r} u fajlu {path}"
            )


def main() -> None:
    """Generate every data-driven report figure."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    required_paths = [
        PPO_HARDCORE_PATH,
        TD3_HARDCORE_PATH,
        LEGACY_SAC_LSTM_PATH,
        BREAKTHROUGH_PATH,
        RESUMED_EVAL_PATH,
        EP4600_EVAL_PATH,
        FIX_TRAIN_LOG,
        RESUMED_TRAIN_LOG,
    ]
    require_files(required_paths)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    configure_plot_style()
    validate_environment_names()

    plot_environment_comparison()
    plot_observation_action_diagram()
    plot_standard_algorithms()
    plot_hardcore_progression()
    plot_best_evaluation_episodes()
    plot_training_curve()


if __name__ == "__main__":
    main()
