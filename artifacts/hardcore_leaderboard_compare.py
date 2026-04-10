from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = PLOTS_DIR / "hardcore_vs_leaderboard_reference.png"


def short_label(text: str, max_len: int = 42) -> str:
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def first_existing_path(candidates: list[Path], label: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find {label}. Looked at: {searched}")


def load_leaderboard(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append(
                {
                    "ranking": int(row["Ranking"]),
                    "model": row["Model"].strip(),
                    "mean": float(row["Mean Reward"]),
                    "std": float(row["Std Reward"]),
                }
            )
    return rows


def load_custom_eval(path: Path) -> tuple[float, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    evaluation = data["evaluation"]
    return float(evaluation["mean_reward"]), float(evaluation["std_reward"])


def load_sb3_eval(path: Path) -> tuple[float, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data["eval_mean_reward"]), float(data["eval_std_reward"])


def load_best_periodic_eval(path: Path) -> tuple[float, int]:
    best_mean: float | None = None
    best_episode: int | None = None
    pattern = re.compile(r"Eval @ episode (\d+) \| raw_mean=([-0-9.]+)")

    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        episode = int(match.group(1))
        mean = float(match.group(2))
        if best_mean is None or mean > best_mean:
            best_mean = mean
            best_episode = episode

    if best_mean is None or best_episode is None:
        raise ValueError(f"Could not parse periodic clean evals from {path}")

    return best_mean, best_episode


def approximate_rank(rows: list[dict[str, object]], mean: float) -> int:
    return 1 + sum(1 for row in rows if float(row["mean"]) > mean)


LEADERBOARD_PATH = first_existing_path(
    [
        ARTIFACTS / "bipedalwalker_leaderboard.csv",
        ROOT / "bipedalwalker_leaderboard.csv",
    ],
    "leaderboard CSV",
)

leaderboard_rows = load_leaderboard(LEADERBOARD_PATH)
top_n = 10
top_rows = sorted(leaderboard_rows, key=lambda row: int(row["ranking"]))[:top_n]

u8_peak_mean, u8_peak_episode = load_best_periodic_eval(
    ARTIFACTS / "runs" / "hardcore" / "res_train_a001_as" / "train.log"
)
u7_eval_mean, u7_eval_std = load_custom_eval(
    ARTIFACTS / "runs" / "hardcore" / "fix_eval_best_raw" / "test_summary.json"
)
u8_eval_mean, u8_eval_std = load_custom_eval(
    ARTIFACTS / "runs" / "hardcore" / "res_eval_best_raw" / "test_summary.json"
)
u5_eval_mean, u5_eval_std = load_custom_eval(
    ARTIFACTS / "runs" / "hardcore" / "legacy_sac_lstm_h12_s42" / "test_summary.json"
)
u3_eval_mean, u3_eval_std = load_sb3_eval(
    ARTIFACTS / "runs" / "standard" / "ppo_bipedalwalker_seed42" / "test_summary.json"
)
u4_eval_mean, u4_eval_std = load_sb3_eval(
    ARTIFACTS / "runs" / "standard" / "td3_bipedalwalker_seed42" / "test_summary.json"
)

my_results = [
    {
        "label": "SAC-LSTM + anti-stall (best episode)",
        "mean": u8_peak_mean,
        "std": 0.0,
        "kind": "Mine",
        "color": "#1b9e77",
    },
    {
        "label": "SAC-LSTM + anti-stall (overall)",
        "mean": u8_eval_mean,
        "std": u8_eval_std,
        "kind": "Mine",
        "color": "#2ca02c",
    },
    {
        "label": "SAC-LSTM + anti-stall (breakthrough)",
        "mean": u7_eval_mean,
        "std": u7_eval_std,
        "kind": "Mine",
        "color": "#ff7f0e",
    },
    {
        "label": "SAC-LSTM",
        "mean": u5_eval_mean,
        "std": u5_eval_std,
        "kind": "Mine",
        "color": "#9467bd",
    },
    {
        "label": "SB3 PPO",
        "mean": u3_eval_mean,
        "std": u3_eval_std,
        "kind": "Mine",
        "color": "#d62728",
    },
    {
        "label": "SB3 TD3",
        "mean": u4_eval_mean,
        "std": u4_eval_std,
        "kind": "Mine",
        "color": "#8c564b",
    },
]

for result in my_results:
    result["approx_rank"] = approximate_rank(leaderboard_rows, float(result["mean"]))
    result["plot_label"] = f"{result['label']} (~ #{result['approx_rank']})"

leaderboard_plot_rows = [
    {
        "plot_label": f"#{int(row['ranking'])} {short_label(str(row['model']))}",
        "mean": float(row["mean"]),
        "std": float(row["std"]),
        "kind": "Leaderboard",
        "color": "#d9d9d9",
    }
    for row in top_rows
]

combined_rows = leaderboard_plot_rows + my_results
combined_rows = sorted(combined_rows, key=lambda row: float(row["mean"]))

my_rows_sorted = sorted(my_results, key=lambda row: float(row["mean"]))

plt.style.use("seaborn-v0_8-whitegrid")
fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(15, 14),
    gridspec_kw={"height_ratios": [1.5, 1.0]},
)

ax1.barh(
    [row["plot_label"] for row in combined_rows],
    [float(row["mean"]) for row in combined_rows],
    xerr=[float(row["std"]) for row in combined_rows],
    color=[str(row["color"]) for row in combined_rows],
    alpha=0.95,
)
ax1.axvline(300, linestyle="--", linewidth=1.2, color="black")
ax1.axvline(0, linestyle=":", linewidth=1.0, color="#666666")
ax1.set_title("My Hardcore Results vs Public BipedalWalker Leaderboard", fontsize=16, weight="bold")
ax1.set_xlabel("Mean reward")
ax1.set_ylabel("")
ax1.tick_params(axis="y", labelsize=9)
ax1.legend(
    handles=[
        Patch(facecolor="#d9d9d9", label=f"Leaderboard top {top_n}"),
        Patch(facecolor="#1b9e77", label="My hardcore runs"),
        Line2D([0], [0], color="black", lw=1.2, label="Black whiskers = +/- 1 std"),
    ],
    loc="lower right",
)

ax2.barh(
    [str(row["label"]) for row in my_rows_sorted],
    [float(row["mean"]) for row in my_rows_sorted],
    xerr=[float(row["std"]) for row in my_rows_sorted],
    color=[str(row["color"]) for row in my_rows_sorted],
    alpha=0.95,
)
ax2.axvline(0, linestyle=":", linewidth=1.0, color="#666666")
ax2.set_title("Zoom: My Best Hardcore Results", fontsize=14, weight="bold")
ax2.set_xlabel("Mean reward")
ax2.set_ylabel("")

min_x = min(float(row["mean"]) - float(row["std"]) for row in my_rows_sorted) - 20
max_x = max(float(row["mean"]) + float(row["std"]) for row in my_rows_sorted) + 30
ax2.set_xlim(min_x, max_x)

for row in my_rows_sorted:
    mean = float(row["mean"])
    std = float(row["std"])
    x_pos = mean + std + 6 if mean >= 0 else mean - std - 6
    ax2.text(
        x_pos,
        str(row["label"]),
        f"{mean:.2f}",
        va="center",
        ha="left" if mean >= 0 else "right",
        fontsize=10,
        weight="bold",
    )

fig.text(
    0.01,
    0.01,
    (
        "Bars show mean reward. Black whiskers show +/- 1 standard deviation across evaluation episodes.\n"
        "Reference note: the public CSV used in result_analsys.ipynb is for BipedalWalker-v3, "
        "not a dedicated BipedalWalkerHardcore-v3 leaderboard. This chart is therefore a "
        "presentation-friendly external reference comparison, not a strict apples-to-apples ranking."
    ),
    ha="left",
    fontsize=10,
)

plt.tight_layout(rect=(0, 0.03, 1, 1))
plt.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")
plt.close(fig)

print(f"Saved chart: {OUTPUT_PATH}")
print()
print("Approximate positions vs public BipedalWalker-v3 leaderboard by mean reward:")
for row in sorted(my_results, key=lambda item: float(item["mean"]), reverse=True):
    extra = f", source_episode={u8_peak_episode}" if row["label"] == "SAC-LSTM + anti-stall (best episode)" else ""
    print(
        f"- {row['label']}: mean={float(row['mean']):.2f}, "
        f"approx_rank={int(row['approx_rank'])}{extra}"
    )
