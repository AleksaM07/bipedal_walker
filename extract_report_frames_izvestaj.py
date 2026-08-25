"""Extract documented representative frames from repository MP4 files.

The selected moments were reviewed manually. They are tied to JSON-backed
evaluations and are intended for qualitative, not quantitative, analysis.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent
FIGURE_DIR = PROJECT_ROOT / "report_figures"


@dataclass(frozen=True)
class FrameSelection:
    """A reproducible video-frame selection."""

    source: Path
    timestamp_seconds: float
    output_name: str
    panel_title: str
    provenance: str


SELECTIONS = [
    FrameSelection(
        source=PROJECT_ROOT
        / "artifacts/archive/reports/runs/01_sb3_ppo_hardcore_baseline/videos/"
        "ppo_bipedalwalker_v3_worst_seed44-step-0-to-step-2000.mp4",
        timestamp_seconds=4.10,
        output_name="early_fall_example.png",
        panel_title="(a) Rani pad",
        provenance="PPO, najgora evaluaciona epizoda, seed=44",
    ),
    FrameSelection(
        source=PROJECT_ROOT
        / "artifacts/archive/reports/runs/02_sb3_td3_hardcore_baseline/videos/"
        "td3_hardcore_baseline_best_seed46-episode-0.mp4",
        timestamp_seconds=30.00,
        output_name="stagnation_example.png",
        panel_title="(b) Stagnacija",
        provenance="TD3, najbolja evaluaciona epizoda, seed=46",
    ),
    FrameSelection(
        source=PROJECT_ROOT
        / "artifacts/runs/hardcore/res_eval_best_raw/videos/"
        "sac_lstm_best_raw_ep1_seed700042-episode-0.mp4",
        timestamp_seconds=7.00,
        output_name="successful_episode.png",
        panel_title="(c) Najuspešniji zabeleženi pokušaj",
        provenance="SAC--LSTM, nagrada=293.44, seed=700042",
    ),
]


def require_inputs() -> str:
    """Return the ffmpeg executable and validate every source video."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError(
            "Program ffmpeg nije pronađen na PATH promenljivoj."
        )
    missing = [
        str(selection.source.relative_to(PROJECT_ROOT))
        for selection in SELECTIONS
        if not selection.source.is_file()
    ]
    if missing:
        joined = "\n  - ".join(missing)
        raise FileNotFoundError(f"Nedostaju video-snimci:\n  - {joined}")
    return ffmpeg


def extract_frame(ffmpeg: str, selection: FrameSelection) -> Path:
    """Extract one exact timestamp with ffmpeg."""
    output_path = FIGURE_DIR / selection.output_name
    command = [
        ffmpeg,
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{selection.timestamp_seconds:.2f}",
        "-i",
        str(selection.source),
        "-frames:v",
        "1",
        "-update",
        "1",
        str(output_path),
    ]
    subprocess.run(command, check=True)
    LOGGER.info(
        "%s <- %s, t=%.2fs",
        output_path.relative_to(PROJECT_ROOT),
        selection.source.relative_to(PROJECT_ROOT),
        selection.timestamp_seconds,
    )
    return output_path


def add_captioned_frame(
    axis: plt.Axes, image_path: Path, selection: FrameSelection
) -> None:
    """Add one source-preserving frame to the combined panel."""
    with Image.open(image_path) as image:
        axis.imshow(image.copy())
    axis.set_title(selection.panel_title, fontsize=12, fontweight="bold")
    axis.text(
        0.5,
        -0.04,
        selection.provenance,
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
    )
    axis.axis("off")


def create_combined_figure(
    image_paths: list[Path], selections: list[FrameSelection]
) -> None:
    """Create a three-panel qualitative comparison."""
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    for axis, image_path, selection in zip(
        axes, image_paths, selections, strict=True
    ):
        add_captioned_frame(axis, image_path, selection)
    figure.suptitle(
        "Reprezentativni režimi ponašanja iz sačuvanih evaluacija",
        fontsize=15,
        fontweight="bold",
    )
    figure.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.15, wspace=0.04)
    output_path = FIGURE_DIR / "failure_modes.png"
    figure.savefig(output_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    LOGGER.info("Kreirano: %s", output_path.relative_to(PROJECT_ROOT))


def create_stagnation_comparison(ffmpeg: str) -> None:
    """Show that the TD3 pose and position barely change over 15 seconds."""
    selection = SELECTIONS[1]
    temporary_paths: list[Path] = []
    for label, timestamp in [("t15", 15.0), ("t30", 30.0)]:
        temporary_selection = FrameSelection(
            source=selection.source,
            timestamp_seconds=timestamp,
            output_name=f"_stagnation_{label}.png",
            panel_title="",
            provenance="",
        )
        temporary_paths.append(extract_frame(ffmpeg, temporary_selection))

    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))
    for axis, image_path, timestamp in zip(
        axes, temporary_paths, [15.0, 30.0], strict=True
    ):
        with Image.open(image_path) as image:
            axis.imshow(image.copy())
        axis.set_title(f"t = {timestamp:.0f} s", fontweight="bold")
        axis.axis("off")
    figure.suptitle(
        "TD3: gotovo nepromenjena pozicija tokom evaluacije",
        fontsize=13,
        fontweight="bold",
    )
    figure.tight_layout()
    output_path = FIGURE_DIR / "stagnation_example.png"
    figure.savefig(output_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    for temporary_path in temporary_paths:
        temporary_path.unlink()
    LOGGER.info("Kreirano: %s", output_path.relative_to(PROJECT_ROOT))


def main() -> None:
    """Extract individual frames and their combined report panel."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    ffmpeg = require_inputs()
    image_paths = [extract_frame(ffmpeg, selection) for selection in SELECTIONS]
    create_combined_figure(image_paths, SELECTIONS)
    create_stagnation_comparison(ffmpeg)


if __name__ == "__main__":
    main()
