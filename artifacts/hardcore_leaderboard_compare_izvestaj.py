"""Plot comparable public and local BipedalWalkerHardcore-v3 results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
OUTPUT_PATH = PROJECT_ROOT / "report_figures" / "hardcore_vs_leaderboard_reference.png"


def load_custom_evaluation(path: Path) -> tuple[float, float]:
    """Load mean and standard deviation from a custom-agent summary."""
    data = json.loads(path.read_text(encoding="utf-8"))
    evaluation = data["evaluation"]
    return float(evaluation["mean_reward"]), float(evaluation["std_reward"])


def load_sb3_evaluation(path: Path) -> tuple[float, float]:
    """Load mean and standard deviation from an SB3 evaluation summary."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data["eval_mean_reward"]), float(data["eval_std_reward"])


breakthrough_mean, breakthrough_std = load_custom_evaluation(
    ARTIFACTS_DIR / "runs/hardcore/fix_eval_best_raw/test_summary.json"
)
continued_mean, continued_std = load_custom_evaluation(
    ARTIFACTS_DIR / "runs/hardcore/res_eval_best_raw/test_summary.json"
)
initial_mean, initial_std = load_custom_evaluation(
    ARTIFACTS_DIR / "runs/hardcore/legacy_sac_lstm_h12_s42/test_summary.json"
)
ppo_mean, ppo_std = load_sb3_evaluation(
    ARTIFACTS_DIR / "runs/standard/ppo_bipedalwalker_seed42/test_summary.json"
)
td3_mean, td3_std = load_sb3_evaluation(
    ARTIFACTS_DIR / "runs/standard/td3_bipedalwalker_seed42/test_summary.json"
)

# Public v3 values are transcribed from the OpenAI Gym leaderboard referenced
# in the report. Only Nick Kaparinos reports a standard deviation there.
public_results = [
    {"label": "Alister Maguire / PPO", "mean": 313.00, "std": 0.0, "color": "#d9d9d9"},
    {"label": "honghaow / TD3-FORK", "mean": 312.10, "std": 0.0, "color": "#d9d9d9"},
    {
        "label": "Nick Kaparinos / SAC",
        "mean": 305.40,
        "std": 21.35,
        "color": "#d9d9d9",
    },
]

local_results = [
    {
        "label": "SAC-LSTM + anti-stall, ep4600 (n=5)",
        "mean": 216.84,
        "std": 76.06,
        "color": "#1b9e77",
    },
    {
        "label": "Nastavljeni SAC-LSTM (n=5)",
        "mean": continued_mean,
        "std": continued_std,
        "color": "#2ca02c",
    },
    {
        "label": "Anti-stall breakthrough (n=20)",
        "mean": breakthrough_mean,
        "std": breakthrough_std,
        "color": "#ff7f0e",
    },
    {
        "label": "Pocetni SAC-LSTM (n=1)",
        "mean": initial_mean,
        "std": initial_std,
        "color": "#9467bd",
    },
    {"label": "SB3 PPO (n=5)", "mean": ppo_mean, "std": ppo_std, "color": "#d62728"},
    {"label": "SB3 TD3 (n=5)", "mean": td3_mean, "std": td3_std, "color": "#8c564b"},
]

all_results = sorted(public_results + local_results, key=lambda row: row["mean"])
local_results_sorted = sorted(local_results, key=lambda row: row["mean"])

plt.style.use("seaborn-v0_8-whitegrid")
figure, (overview_axis, detail_axis) = plt.subplots(
    2,
    1,
    figsize=(14, 12),
    gridspec_kw={"height_ratios": [1.15, 1.0]},
)

overview_axis.barh(
    [row["label"] for row in all_results],
    [row["mean"] for row in all_results],
    xerr=[row["std"] for row in all_results],
    color=[row["color"] for row in all_results],
    alpha=0.95,
)
overview_axis.axvline(300, linestyle="--", linewidth=1.2, color="black")
overview_axis.axvline(0, linestyle=":", linewidth=1.0, color="#666666")
overview_axis.set_title(
    "BipedalWalkerHardcore-v3: javni rezultati i eksperimenti ovog rada",
    fontsize=15,
    weight="bold",
)
overview_axis.set_xlabel("Srednja nagrada")
overview_axis.legend(
    handles=[
        Patch(facecolor="#d9d9d9", label="OpenAI Gym leaderboard, v3"),
        Patch(facecolor="#1b9e77", label="Eksperimenti ovog rada"),
        Line2D([0], [0], color="black", lw=1.2, label="Crna linija: +/- 1 std."),
    ],
    loc="lower right",
)

detail_axis.barh(
    [row["label"] for row in local_results_sorted],
    [row["mean"] for row in local_results_sorted],
    xerr=[row["std"] for row in local_results_sorted],
    color=[row["color"] for row in local_results_sorted],
    alpha=0.95,
)
detail_axis.axvline(0, linestyle=":", linewidth=1.0, color="#666666")
detail_axis.set_title(
    "Detaljni prikaz eksperimenata ovog rada",
    fontsize=13,
    weight="bold",
)
detail_axis.set_xlabel("Srednja nagrada")

min_x = min(row["mean"] - row["std"] for row in local_results_sorted) - 25
max_x = max(row["mean"] + row["std"] for row in local_results_sorted) + 35
detail_axis.set_xlim(min_x, max_x)

for row in local_results_sorted:
    mean = row["mean"]
    std = row["std"]
    x_position = mean + std + 6 if mean >= 0 else mean - std - 6
    detail_axis.text(
        x_position,
        row["label"],
        f"{mean:.2f}",
        va="center",
        ha="left" if mean >= 0 else "right",
        fontsize=9,
        weight="bold",
    )

figure.text(
    0.01,
    0.01,
    (
        "Stubici prikazuju srednju nagradu, a crne linije jednu standardnu "
        "devijaciju kada je dostupna. Javne prijave nemaju potpuno ujednacen "
        "evaluacioni protokol, pa poredenje sluzi kao referentni okvir."
    ),
    ha="left",
    fontsize=9,
)

figure.tight_layout(rect=(0, 0.035, 1, 1))
figure.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")
plt.close(figure)

print(f"Sacuvan grafikon: {OUTPUT_PATH}")
