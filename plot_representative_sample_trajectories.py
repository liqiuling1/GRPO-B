import argparse
import csv
import html
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple


CHECKPOINTS = [0, 10, 20, 30, 40, 50, 100, 250, 500, 1000, 1500, 2000, 4000, 7473]
DEFAULT_STATS_DIR = Path(
    "outputs/observation_sets_gsm8k_p_intervals_seed42/"
    "difficulty_drift_eval_pdist20_seed42/difficulty_drift_statistics"
)
DEFAULT_ORIGINAL_FILE = Path(
    "outputs/observation_sets_gsm8k_p_intervals_seed42/"
    "difficulty_drift_eval_pdist20_seed42/gsm8k_p_scores_final_no_truncation_2_sampled_20pct.jsonl"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select and plot representative single-sample p-score trajectories."
    )
    parser.add_argument("--per_sample_csv", type=Path, default=DEFAULT_STATS_DIR / "per_sample_drift.csv")
    parser.add_argument("--original_file", type=Path, default=DEFAULT_ORIGINAL_FILE)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_STATS_DIR / "representative_sample_trajectories",
    )
    parser.add_argument("--samples_per_category", type=int, default=3)
    return parser.parse_args()


def load_questions(path: Path) -> Dict[str, str]:
    questions: Dict[str, str] = {}
    if not path.is_file():
        return questions
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            questions[str(row.get("uid"))] = str(row.get("question", ""))
    return questions


def load_per_sample(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        sample_idx = header.index("sample_id")
        p_indices = [header.index(f"P_{checkpoint}") for checkpoint in CHECKPOINTS]
        range_idx = header.index("Range")
        max_dev_idx = header.index("MaxInitialDeviation")
        for raw in reader:
            trajectory = [float(raw[idx]) for idx in p_indices]
            rows.append(
                {
                    "sample_id": raw[sample_idx],
                    "trajectory": trajectory,
                    "range": float(raw[range_idx]),
                    "max_initial_deviation": float(raw[max_dev_idx]),
                }
            )
    return rows


def slope(values: List[float]) -> float:
    n = len(values)
    x_mean = (n - 1) / 2
    y_mean = mean(values)
    denom = sum((i - x_mean) ** 2 for i in range(n))
    return sum((i - x_mean) * (value - y_mean) for i, value in enumerate(values)) / denom


def sign_changes(values: List[float]) -> int:
    diffs = []
    for left, right in zip(values, values[1:]):
        diff = right - left
        if abs(diff) >= 1 / 32 - 1e-9:
            diffs.append(1 if diff > 0 else -1)
    return sum(a != b for a, b in zip(diffs, diffs[1:]))


def max_drawdown(values: List[float]) -> float:
    best = values[0]
    drawdown = 0.0
    for value in values:
        best = max(best, value)
        drawdown = max(drawdown, best - value)
    return drawdown


def max_runup(values: List[float]) -> float:
    worst = values[0]
    runup = 0.0
    for value in values:
        worst = min(worst, value)
        runup = max(runup, value - worst)
    return runup


def add_features(rows: List[Dict[str, object]]) -> None:
    for row in rows:
        values = row["trajectory"]
        row["p0"] = values[0]
        row["p_final"] = values[-1]
        row["delta_final"] = values[-1] - values[0]
        row["slope"] = slope(values)
        row["sign_changes"] = sign_changes(values)
        row["max_drawdown"] = max_drawdown(values)
        row["max_runup"] = max_runup(values)
        row["std_like"] = (sum((v - mean(values)) ** 2 for v in values) / len(values)) ** 0.5


def pick_unique(candidates: List[Dict[str, object]], selected_ids: set, n: int) -> List[Dict[str, object]]:
    picked = []
    for row in candidates:
        if row["sample_id"] in selected_ids:
            continue
        picked.append(row)
        selected_ids.add(row["sample_id"])
        if len(picked) == n:
            break
    return picked


def select_representatives(rows: List[Dict[str, object]], samples_per_category: int) -> List[Dict[str, object]]:
    selected_ids = set()
    selected: List[Dict[str, object]] = []

    easier = [
        row
        for row in rows
        if row["delta_final"] >= 0.25 and row["slope"] > 0 and row["range"] >= 0.25
    ]
    easier.sort(key=lambda row: (row["delta_final"], row["slope"], -row["sign_changes"]), reverse=True)
    for row in pick_unique(easier, selected_ids, samples_per_category):
        row["category"] = "明显持续变易"
        selected.append(row)

    harder = [
        row
        for row in rows
        if row["delta_final"] <= -0.25 and row["slope"] < 0 and row["range"] >= 0.25
    ]
    harder.sort(key=lambda row: (-row["delta_final"], -row["slope"], -row["sign_changes"]), reverse=True)
    for row in pick_unique(harder, selected_ids, samples_per_category):
        row["category"] = "明显变难"
        selected.append(row)

    oscillating = [
        row
        for row in rows
        if row["range"] >= 0.5 and row["sign_changes"] >= 4
    ]
    oscillating.sort(key=lambda row: (row["range"], row["sign_changes"], row["std_like"]), reverse=True)
    for row in pick_unique(oscillating, selected_ids, samples_per_category):
        row["category"] = "上下往返"
        selected.append(row)

    stable = [row for row in rows if row["range"] <= 0.0625]
    stable.sort(key=lambda row: (row["range"], row["std_like"], abs(row["delta_final"])))
    for row in pick_unique(stable, selected_ids, samples_per_category):
        row["category"] = "相对稳定"
        selected.append(row)

    return selected


def write_selected_csv(path: Path, selected: List[Dict[str, object]], questions: Dict[str, str]) -> None:
    columns = [
        "category",
        "sample_id",
        "P0",
        "P_final",
        "Delta_final",
        "Range",
        "MaxInitialDeviation",
        "Slope",
        "SignChanges",
        *[f"P_{checkpoint}" for checkpoint in CHECKPOINTS],
        "question",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in selected:
            out = {
                "category": row["category"],
                "sample_id": row["sample_id"],
                "P0": row["p0"],
                "P_final": row["p_final"],
                "Delta_final": row["delta_final"],
                "Range": row["range"],
                "MaxInitialDeviation": row["max_initial_deviation"],
                "Slope": row["slope"],
                "SignChanges": row["sign_changes"],
                "question": questions.get(str(row["sample_id"]), ""),
            }
            for checkpoint, value in zip(CHECKPOINTS, row["trajectory"]):
                out[f"P_{checkpoint}"] = value
            writer.writerow(out)


def write_selected_markdown(path: Path, selected: List[Dict[str, object]]) -> None:
    columns = ["category", "sample_id", "P0", "P7473", "Delta_final", "Range", "MaxInitialDeviation", "SignChanges"]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in selected:
        values = [
            row["category"],
            row["sample_id"],
            f"{row['p0']:.5f}".rstrip("0").rstrip("."),
            f"{row['p_final']:.5f}".rstrip("0").rstrip("."),
            f"{row['delta_final']:.5f}".rstrip("0").rstrip("."),
            f"{row['range']:.5f}".rstrip("0").rstrip("."),
            f"{row['max_initial_deviation']:.5f}".rstrip("0").rstrip("."),
            str(row["sign_changes"]),
        ]
        lines.append("| " + " | ".join(map(str, values)) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_svg(path: Path, selected: List[Dict[str, object]], title: str, subtitle: str) -> None:
    width, height = 1260, 760
    margin_left, margin_right, margin_top, margin_bottom = 92, 285, 58, 92
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    colors = {
        "明显持续变易": "#1b9e77",
        "明显变难": "#d95f02",
        "上下往返": "#7570b3",
        "相对稳定": "#4d4d4d",
    }
    x_positions = [
        margin_left + i * plot_w / (len(CHECKPOINTS) - 1)
        for i in range(len(CHECKPOINTS))
    ]

    def y(value: float) -> float:
        return margin_top + (1 - value) * plot_h

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{0}" height="{1}" viewBox="0 0 {0} {1}">'.format(width, height),
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="24" y="34" font-family="Arial" font-size="22" font-weight="700">{html.escape(title)}</text>',
        f'<text x="24" y="57" font-family="Arial" font-size="13" fill="#555">{html.escape(subtitle)}</text>',
    ]
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        yy = y(tick)
        lines.append(f'<line x1="{margin_left}" y1="{yy:.2f}" x2="{margin_left + plot_w}" y2="{yy:.2f}" stroke="#e5e5e5"/>')
        lines.append(f'<text x="{margin_left - 12}" y="{yy + 4:.2f}" text-anchor="end" font-family="Arial" font-size="12" fill="#333">{tick:.2f}</text>')
    lines.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#333"/>')
    lines.append(f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="#333"/>')
    lines.append(f'<text x="20" y="{margin_top + plot_h / 2}" transform="rotate(-90 20 {margin_top + plot_h / 2})" text-anchor="middle" font-family="Arial" font-size="15" fill="#111">P_t</text>')

    for x, checkpoint in zip(x_positions, CHECKPOINTS):
        lines.append(f'<line x1="{x:.2f}" y1="{margin_top + plot_h}" x2="{x:.2f}" y2="{margin_top + plot_h + 6}" stroke="#333"/>')
        rotate = -45 if checkpoint >= 1000 else 0
        if rotate:
            lines.append(f'<text x="{x:.2f}" y="{margin_top + plot_h + 28}" transform="rotate({rotate} {x:.2f} {margin_top + plot_h + 28})" text-anchor="end" font-family="Arial" font-size="12" fill="#333">{checkpoint}</text>')
        else:
            lines.append(f'<text x="{x:.2f}" y="{margin_top + plot_h + 22}" text-anchor="middle" font-family="Arial" font-size="12" fill="#333">{checkpoint}</text>')
    lines.append(f'<text x="{margin_left + plot_w / 2}" y="{height - 18}" text-anchor="middle" font-family="Arial" font-size="15" fill="#111">Checkpoint</text>')

    for row in selected:
        color = colors[row["category"]]
        points = " ".join(
            f"{x:.2f},{y(value):.2f}" for x, value in zip(x_positions, row["trajectory"])
        )
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.6" stroke-linejoin="round" stroke-linecap="round" opacity="0.9"/>')
        for x, value in zip(x_positions, row["trajectory"]):
            lines.append(f'<circle cx="{x:.2f}" cy="{y(value):.2f}" r="3" fill="{color}" stroke="white" stroke-width="1"/>')

    legend_x = margin_left + plot_w + 32
    legend_y = margin_top + 8
    lines.append(f'<text x="{legend_x}" y="{legend_y}" font-family="Arial" font-size="14" font-weight="700">Selected samples</text>')
    y_cursor = legend_y + 24
    categories = ["明显持续变易", "明显变难", "上下往返", "相对稳定"]
    categories = [category for category in categories if any(row["category"] == category for row in selected)]
    for category in categories:
        lines.append(f'<text x="{legend_x}" y="{y_cursor}" font-family="Arial" font-size="13" font-weight="700" fill="{colors[category]}">{html.escape(category)}</text>')
        y_cursor += 18
        for row in [r for r in selected if r["category"] == category]:
            label = (
                f"id={row['sample_id']}  "
                f"P0={row['p0']:.2f}, Pend={row['p_final']:.2f}, R={row['range']:.2f}"
            )
            lines.append(f'<text x="{legend_x + 10}" y="{y_cursor}" font-family="Arial" font-size="12" fill="#333">{html.escape(label)}</text>')
            y_cursor += 16
        y_cursor += 8
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    rows = load_per_sample(args.per_sample_csv)
    add_features(rows)
    selected = select_representatives(rows, args.samples_per_category)
    questions = load_questions(args.original_file)

    if len(selected) < args.samples_per_category * 4:
        raise RuntimeError(f"Only selected {len(selected)} samples; expected {args.samples_per_category * 4}.")

    out_dir = args.output_dir
    csv_path = out_dir / "representative_sample_trajectories.csv"
    md_path = out_dir / "representative_sample_trajectories.md"
    svg_path = out_dir / "representative_sample_trajectories.svg"
    write_selected_csv(csv_path, selected, questions)
    write_selected_markdown(md_path, selected)
    plot_svg(
        svg_path,
        selected,
        title="Representative Single-Sample Difficulty Trajectories",
        subtitle="P-score over checkpoints; 12 selected samples grouped by drift pattern",
    )
    category_slug = {
        "明显持续变易": "easier",
        "明显变难": "harder",
        "上下往返": "oscillating",
        "相对稳定": "stable",
    }
    category_paths = []
    for category, slug in category_slug.items():
        category_rows = [row for row in selected if row["category"] == category]
        category_path = out_dir / f"representative_sample_trajectories_{slug}.svg"
        plot_svg(
            category_path,
            category_rows,
            title=f"{category} Single-Sample Trajectories",
            subtitle="P-score over checkpoints; one drift pattern per figure",
        )
        category_paths.append(category_path)

    print(f"Wrote selected CSV: {csv_path}")
    print(f"Wrote selected Markdown: {md_path}")
    print(f"Wrote SVG plot: {svg_path}")
    for category_path in category_paths:
        print(f"Wrote category SVG plot: {category_path}")
    print("Selected samples:")
    for row in selected:
        print(
            f"{row['category']}\tuid={row['sample_id']}\t"
            f"P0={row['p0']:.5f}\tP7473={row['p_final']:.5f}\t"
            f"Range={row['range']:.5f}\tMaxDev={row['max_initial_deviation']:.5f}\t"
            f"SignChanges={row['sign_changes']}"
        )


if __name__ == "__main__":
    main()
