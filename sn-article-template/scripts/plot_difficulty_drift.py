#!/usr/bin/env python3
"""Plot Figure 2: policy-conditioned solvability drift during GRPO training."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-grpo-b")

import matplotlib.pyplot as plt
import numpy as np


TEMPLATE_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = TEMPLATE_DIR.parent
FIG_DIR = TEMPLATE_DIR / "figures"
META_DIR = FIG_DIR / "metadata"
TRACK_DIR = REPO_DIR / "outputs" / "difficulty_drift_eval_pdist20_seed42"

CHECKPOINTS = [0, 10, 20, 30, 40, 50, 100, 250, 500, 1000, 1500, 2000, 4000, 7473]
REVERSAL_DELTA = 0.125


def checkpoint_path(checkpoint: int) -> Path:
    if checkpoint == 0:
        return TRACK_DIR / "gsm8k_p_scores_final_no_truncation_2_sampled_20pct.jsonl"
    return TRACK_DIR / f"gsm8k_p_scores_sample20_checkpoint-{checkpoint}.jsonl"


def read_jsonl_scores(path: Path) -> dict[str, float]:
    rows: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            uid = str(row["uid"])
            if uid in rows:
                raise ValueError(f"Duplicate uid={uid} in {path}:{line_no}")
            rows[uid] = float(row["p"])
    return rows


def load_tracking_matrix() -> tuple[list[str], np.ndarray, dict[int, str]]:
    """Load the common 1495 tracked problems across all checkpoints."""
    score_by_checkpoint = {}
    source_files = {}
    for checkpoint in CHECKPOINTS:
        path = checkpoint_path(checkpoint)
        source_files[checkpoint] = str(path)
        score_by_checkpoint[checkpoint] = read_jsonl_scores(path)

    uid_sets = [set(scores) for scores in score_by_checkpoint.values()]
    common_uids = sorted(set.intersection(*uid_sets), key=lambda value: int(value))
    if not common_uids:
        raise ValueError("No common tracked uid set found across checkpoints.")

    matrix = np.array(
        [[score_by_checkpoint[checkpoint][uid] for checkpoint in CHECKPOINTS] for uid in common_uids],
        dtype=float,
    )
    if matrix.shape != (1495, 14):
        raise ValueError(f"Expected a 1495x14 tracking matrix, got {matrix.shape}.")
    return common_uids, matrix, source_files


def meaningful_step_signs(values: np.ndarray, delta: float = REVERSAL_DELTA) -> list[int]:
    signs = []
    for diff in np.diff(values):
        if abs(diff) >= delta - 1e-12:
            signs.append(1 if diff > 0 else -1)
    return signs


def reversal_count(values: np.ndarray, delta: float = REVERSAL_DELTA) -> int:
    signs = meaningful_step_signs(values, delta)
    return sum(left != right for left, right in zip(signs, signs[1:]))


def reversal_positions(values: np.ndarray, delta: float = REVERSAL_DELTA) -> list[int]:
    """Return checkpoint indices where the meaningful step direction changes."""
    signs = []
    positions = []
    for index, diff in enumerate(np.diff(values), start=1):
        if abs(diff) >= delta - 1e-12:
            signs.append(1 if diff > 0 else -1)
            positions.append(index)
    return [positions[idx] for idx in range(1, len(signs)) if signs[idx] != signs[idx - 1]]


def trajectory_metrics(values: np.ndarray) -> dict[str, float | int]:
    span = float(np.max(values) - np.min(values))
    total_variation = float(np.sum(np.abs(np.diff(values))))
    return {
        "p0": float(values[0]),
        "pfinal": float(values[-1]),
        "delta_final": float(values[-1] - values[0]),
        "span": span,
        "total_variation": total_variation,
        "extra_variation": total_variation - span,
        "reversals": reversal_count(values),
    }


def select_representative_samples(uids: list[str], matrix: np.ndarray) -> list[dict[str, object]]:
    """Select representative trajectories by deterministic metric-based rules."""
    rows = []
    for idx, uid in enumerate(uids):
        values = matrix[idx]
        metrics = trajectory_metrics(values)
        rows.append({"uid": uid, "index": idx, "values": values, **metrics})

    upward_pool = [row for row in rows if row["delta_final"] >= 0.5]
    if not upward_pool:
        upward_pool = rows
    upward = max(upward_pool, key=lambda row: (row["delta_final"], row["span"], -int(row["uid"])))

    downward_pool = [row for row in rows if row["p0"] >= 0.625 and row["delta_final"] <= -0.5]
    if not downward_pool:
        downward_pool = rows
    downward = min(downward_pool, key=lambda row: (row["delta_final"], -row["p0"], int(row["uid"])))

    clear_pool = [row for row in rows if row["reversals"] == 1]
    if not clear_pool:
        clear_pool = [row for row in rows if row["reversals"] >= 1]
    clear = max(clear_pool, key=lambda row: (row["span"], abs(row["delta_final"]), -int(row["uid"])))

    repeated_pool = [row for row in rows if row["reversals"] >= 3]
    if not repeated_pool:
        repeated_pool = rows
    repeated = max(repeated_pool, key=lambda row: (row["extra_variation"], row["reversals"], row["span"], -int(row["uid"])))

    selected = [
        ("Increasing solvability", upward),
        ("Decreasing solvability", downward),
        ("Clear reversal", clear),
        ("Repeated non-monotonic changes", repeated),
    ]
    return [
        {
            "panel": chr(ord("c") + panel_idx),
            "type": name,
            **row,
            "reversal_positions": reversal_positions(row["values"]),
        }
        for panel_idx, (name, row) in enumerate(selected)
    ]


def write_selection_metadata(selected: list[dict[str, object]]) -> None:
    META_DIR.mkdir(parents=True, exist_ok=True)
    path = META_DIR / "fig2_difficulty_drift_selected_samples.csv"
    fieldnames = [
        "panel",
        "type",
        "uid",
        "p0",
        "pfinal",
        "delta_final",
        "span",
        "total_variation",
        "extra_variation",
        "reversals",
        "reversal_positions",
        *[f"P_{checkpoint}" for checkpoint in CHECKPOINTS],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            values = row["values"]
            writer.writerow(
                {
                    "panel": row["panel"],
                    "type": row["type"],
                    "uid": row["uid"],
                    "p0": f"{row['p0']:.5f}",
                    "pfinal": f"{row['pfinal']:.5f}",
                    "delta_final": f"{row['delta_final']:.5f}",
                    "span": f"{row['span']:.5f}",
                    "total_variation": f"{row['total_variation']:.5f}",
                    "extra_variation": f"{row['extra_variation']:.5f}",
                    "reversals": row["reversals"],
                    "reversal_positions": " ".join(str(CHECKPOINTS[pos]) for pos in row["reversal_positions"]),
                    **{f"P_{checkpoint}": f"{value:.5f}" for checkpoint, value in zip(CHECKPOINTS, values)},
                }
            )


def style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)
        spine.set_color("#3a3a3a")
    ax.tick_params(axis="both", width=0.65, length=3, labelsize=7, colors="#333333")


def validate_reported_statistics(matrix: np.ndarray) -> None:
    spans = matrix.max(axis=1) - matrix.min(axis=1)
    checks = [
        ("mean span", float(spans.mean()), 0.2770, 5e-5),
        ("R_i >= 0.25", float(np.mean(spans >= 0.25 - 1e-12) * 100), 54.72, 0.005),
    ]
    for delta, expected_one, expected_two in [
        (0.0625, 87.02, 78.33),
        (0.125, 53.38, 36.79),
        (0.25, 4.95, 1.14),
    ]:
        counts = np.array([reversal_count(row, delta) for row in matrix])
        checks.extend(
            [
                (f"delta={delta} >=1 reversal", float(np.mean(counts >= 1) * 100), expected_one, 0.005),
                (f"delta={delta} >=2 reversals", float(np.mean(counts >= 2) * 100), expected_two, 0.005),
            ]
        )
    mismatches = [
        f"{name}: computed {value:.6f}, expected {expected:.4f}"
        for name, value, expected, tolerance in checks
        if abs(value - expected) > tolerance
    ]
    if mismatches:
        raise ValueError("Reported statistics mismatch:\n" + "\n".join(mismatches))


def plot_figure(uids: list[str], matrix: np.ndarray, selected: list[dict[str, object]]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.1,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    spans = matrix.max(axis=1) - matrix.min(axis=1)
    mean_span = float(spans.mean())
    span_fraction = float(np.mean(spans >= 0.25 - 1e-12) * 100)
    reversal_thresholds = [0.0625, 0.125, 0.25]
    reversal_one = []
    reversal_two = []
    for delta in reversal_thresholds:
        counts = np.array([reversal_count(row, delta) for row in matrix])
        reversal_one.append(float(np.mean(counts >= 1) * 100))
        reversal_two.append(float(np.mean(counts >= 2) * 100))

    fig = plt.figure(figsize=(7.25, 5.70), dpi=300)
    gridspec = fig.add_gridspec(
        2,
        4,
        height_ratios=[1.25, 1.00],
        hspace=0.88,
        wspace=0.42,
        left=0.075,
        right=0.985,
        bottom=0.105,
        top=0.94,
    )

    ax_span = fig.add_subplot(gridspec[0, :2])
    bins = np.arange(0, 1.0001, 0.0625)
    ax_span.hist(spans, bins=bins, color="#5d7fa3", edgecolor="white", linewidth=0.45)
    ax_span.axvspan(0.25, 1.0, color="#5d7fa3", alpha=0.11, lw=0)
    ax_span.axvline(0.25, color="#444444", linestyle=(0, (3.2, 2.2)), linewidth=1.0)
    ax_span.axvline(mean_span, color="#8c2d3e", linewidth=1.05)
    ax_span.text(
        0.257,
        ax_span.get_ylim()[1] * 0.88,
        f"{span_fraction:.2f}% $\\geq 0.25$",
        fontsize=7.3,
        color="#333333",
        va="top",
    )
    ax_span.text(
        mean_span + 0.012,
        ax_span.get_ylim()[1] * 0.62,
        f"Mean $R_i$={mean_span:.4f}",
        fontsize=7.3,
        color="#6b1f2e",
        va="center",
    )
    ax_span.set_title("(a) Solvability variation across tracked problems", loc="left", pad=6, fontweight="bold")
    ax_span.set_xlabel("Training-period solvability span $R_i$")
    ax_span.set_ylabel("Number of tracked problems")
    ax_span.set_xlim(0, 1.0)
    ax_span.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_span.grid(axis="y", color="#e1e1e1", linewidth=0.45)
    style_axes(ax_span)

    ax_rev = fig.add_subplot(gridspec[0, 2:])
    x = np.arange(len(reversal_thresholds))
    width = 0.32
    colors_bar = ["#6f8fb4", "#c78a72"]
    bars_one = ax_rev.bar(x - width / 2, reversal_one, width, label="At least one", color=colors_bar[0], edgecolor="#333333", linewidth=0.35)
    bars_two = ax_rev.bar(x + width / 2, reversal_two, width, label="At least two", color=colors_bar[1], edgecolor="#333333", linewidth=0.35)
    ax_rev.axvspan(0.5, 1.5, color="#efefef", alpha=0.75, zorder=0)
    for bars in [bars_one, bars_two]:
        for bar in bars:
            ax_rev.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2.0,
                f"{bar.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=6.7,
                color="#333333",
            )
    ax_rev.set_title("(b) Prevalence of non-monotonic reversals", loc="left", pad=6, fontweight="bold")
    ax_rev.set_ylabel("Percentage of tracked problems (%)")
    ax_rev.set_xticks(x)
    ax_rev.set_xticklabels([r"$\delta=0.0625$", r"$\delta=0.125$", r"$\delta=0.25$"])
    ax_rev.set_ylim(0, 100)
    ax_rev.set_yticks([0, 25, 50, 75, 100])
    ax_rev.grid(axis="y", color="#e1e1e1", linewidth=0.45)
    ax_rev.legend(frameon=False, loc="upper right", fontsize=7, handlelength=1.2, borderaxespad=0.2)
    ax_rev.text(1, 95, "primary", ha="center", va="top", fontsize=6.8, color="#555555")
    style_axes(ax_rev)

    x_positions = np.arange(len(CHECKPOINTS))
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    for panel_idx, row in enumerate(selected):
        ax = fig.add_subplot(gridspec[1, panel_idx])
        values = row["values"]
        ax.plot(
            x_positions,
            values,
            color=colors[panel_idx],
            linewidth=1.35,
            marker="o",
            markersize=2.9,
            markerfacecolor="white",
            markeredgewidth=0.8,
        )
        if row["panel"] in {"e", "f"}:
            for pos in row["reversal_positions"]:
                ax.scatter(
                    [pos],
                    [values[pos]],
                    s=28,
                    marker="^",
                    facecolors="none",
                    edgecolors="#222222",
                    linewidths=0.75,
                    zorder=4,
                )
        title = f"({row['panel']}) {row['type']}"
        title_size = 7.25
        if row["panel"] == "f":
            title = "(f) Repeated\nnon-monotonic\nchanges"
            title_size = 6.4
        ax.set_title(title, loc="left", pad=5, fontweight="bold", fontsize=title_size, y=1.34)
        ax.text(
            0.0,
            1.04,
            f"ID {row['uid']}\n$P_0$={row['p0']:.3f}, $P_T$={row['pfinal']:.3f}\nReversals = {row['reversals']}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=5.65,
            color="#333333",
            clip_on=False,
        )
        ax.set_xlim(-0.35, len(CHECKPOINTS) - 0.65)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks([0, 5, 7, 9, 13])
        ax.set_xticklabels(["0", "50", "250", "1000", "7473"], fontsize=5.6, rotation=35, ha="right")
        ax.set_yticks([0, 0.5, 1.0])
        if panel_idx == 0:
            ax.set_ylabel("$P_t(x)$")
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("Tracked checkpoint", fontsize=7.1)
        ax.grid(axis="y", color="#dedede", linewidth=0.42, alpha=0.82)
        style_axes(ax)

    fig.savefig(FIG_DIR / "fig2_difficulty_drift.pdf")
    fig.savefig(FIG_DIR / "fig2_difficulty_drift.png", dpi=300)
    plt.close(fig)


def main() -> None:
    uids, matrix, _ = load_tracking_matrix()
    validate_reported_statistics(matrix)
    selected = select_representative_samples(uids, matrix)
    write_selection_metadata(selected)
    plot_figure(uids, matrix, selected)
    for row in selected:
        print(
            f"({row['panel']}) {row['type']}: uid={row['uid']}, "
            f"P0={row['p0']:.5f}, Pfinal={row['pfinal']:.5f}, "
            f"span={row['span']:.5f}, reversals={row['reversals']}, "
            f"E={row['extra_variation']:.5f}"
        )


if __name__ == "__main__":
    main()
