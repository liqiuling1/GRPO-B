import argparse
import csv
import json
import math
import re
import struct
import zlib
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from analyze_vanilla_grpo_difficulty_drift import write_xlsx


CHECKPOINTS = [0, 10, 20, 30, 40, 50, 100, 250, 500, 1000, 1500, 2000, 4000, 7473]
DEFAULT_INPUT_DIR = Path(
    "outputs/observation_sets_gsm8k_p_intervals_seed42/"
    "difficulty_drift_eval_pdist20_seed42"
)
DEFAULT_ORIGINAL_NAME = "gsm8k_p_scores_final_no_truncation_2_sampled_20pct.jsonl"
STAGES = [
    ("B1", "[0.625,0.875]", 0.625, 0.875),
    ("B2", "[0.500,0.750]", 0.500, 0.750),
    ("B3", "[0.375,0.625]", 0.375, 0.625),
    ("B4", "[0.250,0.500]", 0.250, 0.500),
    ("B5", "[0.125,0.375]", 0.125, 0.375),
]
COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]
EPS = 1e-9


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze target interval membership drift across Vanilla GRPO checkpoints."
    )
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--original_file", type=Path, default=None)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Defaults to <input_dir>/difficulty_membership_drift",
    )
    parser.add_argument("--sample_id_field", default="uid")
    parser.add_argument("--p_field", default="p")
    parser.add_argument("--rollout_field", default="_num_answers")
    parser.add_argument("--expected_rollouts", type=int, default=32)
    parser.add_argument("--expected_n", type=int, default=1495)
    return parser.parse_args()


def checkpoint_file(input_dir: Path, original_file: Path, checkpoint: int) -> Path:
    if checkpoint == 0:
        return original_file
    candidates = sorted(input_dir.glob(f"*checkpoint-{checkpoint}.jsonl"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected exactly one checkpoint-{checkpoint} file, got {len(candidates)}")
    return candidates[0]


def pct(value: float) -> str:
    if math.isnan(value):
        return "NaN"
    return f"{value * 100:.2f}%"


def fmt(value: float) -> str:
    if math.isnan(value):
        return "NaN"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def read_checkpoint(
    path: Path,
    checkpoint: int,
    sample_id_field: str,
    p_field: str,
    rollout_field: str,
    expected_rollouts: int,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    values: Dict[str, float] = {}
    duplicate_count = 0
    missing_id_count = 0
    missing_p_count = 0
    bad_rollout_count = 0
    abnormal_p_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            row = json.loads(line)
            if sample_id_field not in row:
                missing_id_count += 1
                continue
            if p_field not in row:
                missing_p_count += 1
                continue
            sample_id = str(row[sample_id_field])
            if sample_id in values:
                duplicate_count += 1
                continue
            p = float(row[p_field])
            if row.get(rollout_field) != expected_rollouts:
                bad_rollout_count += 1
            correct_count = round(p * expected_rollouts)
            if not (
                -EPS <= p <= 1 + EPS
                and 0 <= correct_count <= expected_rollouts
                and abs(p - correct_count / expected_rollouts) <= 1e-8
            ):
                abnormal_p_count += 1
            values[sample_id] = p
    return values, {
        "checkpoint": checkpoint,
        "file": str(path),
        "sample_count": len(values),
        "duplicate_count": duplicate_count,
        "missing_id_count": missing_id_count,
        "missing_p_count": missing_p_count,
        "bad_rollout_count": bad_rollout_count,
        "abnormal_success_rate_count": abnormal_p_count,
    }


def load_all(args) -> Tuple[Dict[int, Dict[str, float]], List[Dict[str, object]], List[str], List[Path]]:
    original_file = args.original_file or (args.input_dir / DEFAULT_ORIGINAL_NAME)
    files = [checkpoint_file(args.input_dir, original_file, checkpoint) for checkpoint in CHECKPOINTS]
    data: Dict[int, Dict[str, float]] = {}
    integrity: List[Dict[str, object]] = []
    for checkpoint, path in zip(CHECKPOINTS, files):
        values, row = read_checkpoint(
            path,
            checkpoint,
            args.sample_id_field,
            args.p_field,
            args.rollout_field,
            args.expected_rollouts,
        )
        data[checkpoint] = values
        integrity.append(row)
    id_sets = [set(data[checkpoint]) for checkpoint in CHECKPOINTS]
    common_ids = sorted(set.intersection(*id_sets), key=lambda value: int(value) if value.isdigit() else value)
    union_ids = set.union(*id_sets)
    for row in integrity:
        ids = set(data[int(row["checkpoint"])])
        row["missing_vs_union"] = len(union_ids - ids)
        row["effective_aligned_n"] = len(common_ids)
    return data, integrity, common_ids, files


def in_interval(p: float, lower: float, upper: float) -> bool:
    return lower - EPS <= p <= upper + EPS


def build_sets(data: Dict[int, Dict[str, float]], sample_ids: Sequence[str]) -> Dict[str, Dict[int, Set[str]]]:
    stage_sets: Dict[str, Dict[int, Set[str]]] = {}
    for stage, _interval, lower, upper in STAGES:
        stage_sets[stage] = {}
        for checkpoint in CHECKPOINTS:
            stage_sets[stage][checkpoint] = {
                sample_id
                for sample_id in sample_ids
                if in_interval(data[checkpoint][sample_id], lower, upper)
            }
    return stage_sets


def safe_rate(num: int, den: int) -> float:
    return math.nan if den == 0 else num / den


def build_membership_rows(stage_sets: Dict[str, Dict[int, Set[str]]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    interval_by_stage = {stage: interval for stage, interval, _lower, _upper in STAGES}
    for stage, _interval, _lower, _upper in STAGES:
        initial = stage_sets[stage][0]
        initial_size = len(initial)
        for checkpoint in CHECKPOINTS:
            current = stage_sets[stage][checkpoint]
            intersection = initial & current
            outflow = initial - current
            inflow = current - initial
            union = initial | current
            size_change = len(current) - initial_size
            rows.append(
                {
                    "Stage": stage,
                    "Interval": interval_by_stage[stage],
                    "Checkpoint": checkpoint,
                    "Initial_Size": initial_size,
                    "Current_Size": len(current),
                    "Size_Change": size_change,
                    "Size_Change_Rate": pct(safe_rate(size_change, initial_size)),
                    "Intersection_Count": len(intersection),
                    "Retention_Rate": pct(safe_rate(len(intersection), initial_size)),
                    "Outflow_Count": len(outflow),
                    "Outflow_Rate": pct(safe_rate(len(outflow), initial_size)),
                    "Inflow_Count": len(inflow),
                    "Inflow_Rate": pct(safe_rate(len(inflow), len(current))),
                    "Jaccard": fmt(safe_rate(len(intersection), len(union))),
                }
            )
    return rows


def build_adjacent_rows(stage_sets: Dict[str, Dict[int, Set[str]]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for stage, _interval, _lower, _upper in STAGES:
        for prev, current_ckpt in zip(CHECKPOINTS, CHECKPOINTS[1:]):
            previous_set = stage_sets[stage][prev]
            current_set = stage_sets[stage][current_ckpt]
            intersection = previous_set & current_set
            outflow = previous_set - current_set
            inflow = current_set - previous_set
            union = previous_set | current_set
            rows.append(
                {
                    "Stage": stage,
                    "Previous_Checkpoint": prev,
                    "Current_Checkpoint": current_ckpt,
                    "Previous_Size": len(previous_set),
                    "Current_Size": len(current_set),
                    "Adjacent_Retention": pct(safe_rate(len(intersection), len(previous_set))),
                    "Adjacent_Outflow": pct(safe_rate(len(outflow), len(previous_set))),
                    "Adjacent_Inflow": pct(safe_rate(len(inflow), len(current_set))),
                    "Adjacent_Jaccard": fmt(safe_rate(len(intersection), len(union))),
                }
            )
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def hex_to_rgb(color: str) -> Tuple[int, int, int]:
    color = color.lstrip("#")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)


def write_png(path: Path, series: List[Tuple[str, List[float], str]], y_min: float, y_max: float) -> None:
    width, height = 1500, 950
    left, right, top, bottom = 105, 65, 80, 115
    canvas = bytearray([255, 255, 255] * width * height)

    def put(x: int, y: int, color: Tuple[int, int, int]) -> None:
        if 0 <= x < width and 0 <= y < height:
            i = (y * width + x) * 3
            canvas[i:i + 3] = bytes(color)

    def line(x0: float, y0: float, x1: float, y1: float, color: Tuple[int, int, int], thickness: int = 2) -> None:
        x0, y0, x1, y1 = map(round, [x0, y0, x1, y1])
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            for ox in range(-thickness, thickness + 1):
                for oy in range(-thickness, thickness + 1):
                    put(x0 + ox, y0 + oy, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def circle(cx: float, cy: float, r: int, color: Tuple[int, int, int]) -> None:
        cx, cy = round(cx), round(cy)
        for x in range(cx - r, cx + r + 1):
            for y in range(cy - r, cy + r + 1):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                    put(x, y, color)

    def x_at(i: int) -> float:
        return left + i * (width - left - right) / (len(CHECKPOINTS) - 1)

    def y_at(v: float) -> float:
        return top + (y_max - v) * (height - top - bottom) / (y_max - y_min)

    axis = (40, 40, 40)
    grid = (225, 225, 225)
    for k in range(6):
        y = top + k * (height - top - bottom) / 5
        line(left, y, width - right, y, grid, 1)
    line(left, top, left, height - bottom, axis, 2)
    line(left, height - bottom, width - right, height - bottom, axis, 2)
    for _label, values, color_hex in series:
        color = hex_to_rgb(color_hex)
        points = [(x_at(i), y_at(value)) for i, value in enumerate(values)]
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            line(x0, y0, x1, y1, color, 3)
        for x, y in points:
            circle(x, y, 5, color)

    raw_rows = []
    for y in range(height):
        row = bytes(canvas[(y * width) * 3:((y + 1) * width) * 3])
        raw_rows.append(b"\x00" + row)
    raw = b"".join(raw_rows)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"pHYs", struct.pack(">IIB", 11811, 11811, 1))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def write_svg(path: Path, title: str, series: List[Tuple[str, List[float], str]], y_min: float, y_max: float, y_label: str) -> None:
    width, height = 1120, 680
    left, right, top, bottom = 90, 210, 60, 90
    plot_w, plot_h = width - left - right, height - top - bottom

    def x_at(i: int) -> float:
        return left + i * plot_w / (len(CHECKPOINTS) - 1)

    def y_at(v: float) -> float:
        return top + (y_max - v) * plot_h / (y_max - y_min)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="24" y="34" font-family="Arial" font-size="20" font-weight="700">{title}</text>',
    ]
    for k in range(6):
        value = y_min + (y_max - y_min) * k / 5
        y = y_at(value)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e5e5"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{fmt(value)}</text>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>')
    lines.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>')
    for i, checkpoint in enumerate(CHECKPOINTS):
        x = x_at(i)
        rotate = -45 if checkpoint >= 1000 else 0
        if rotate:
            lines.append(f'<text x="{x:.1f}" y="{top + plot_h + 30}" transform="rotate(-45 {x:.1f} {top + plot_h + 30})" text-anchor="end" font-family="Arial" font-size="12">{checkpoint}</text>')
        else:
            lines.append(f'<text x="{x:.1f}" y="{top + plot_h + 22}" text-anchor="middle" font-family="Arial" font-size="12">{checkpoint}</text>')
    lines.append(f'<text x="18" y="{top + plot_h / 2}" transform="rotate(-90 18 {top + plot_h / 2})" text-anchor="middle" font-family="Arial" font-size="14">{y_label}</text>')
    for label, values, color in series:
        pts = " ".join(f"{x_at(i):.1f},{y_at(v):.1f}" for i, v in enumerate(values))
        lines.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for i, v in enumerate(values):
            lines.append(f'<circle cx="{x_at(i):.1f}" cy="{y_at(v):.1f}" r="3.5" fill="{color}"/>')
    legend_x, legend_y = left + plot_w + 26, top + 20
    for idx, (label, _values, color) in enumerate(series):
        y = legend_y + idx * 24
        lines.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 24}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{legend_x + 32}" y="{y + 4}" font-family="Arial" font-size="13">{label}</text>')
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def numeric_percent(value: str) -> float:
    return float(value.rstrip("%"))


def make_plots(out_dir: Path, all_rows: List[Dict[str, object]]) -> List[Path]:
    paths: List[Path] = []
    rows_by_stage = {stage: [r for r in all_rows if r["Stage"] == stage] for stage, *_ in STAGES}
    retention_series = []
    jaccard_series = []
    for idx, (stage, interval, _lower, _upper) in enumerate(STAGES):
        rows = rows_by_stage[stage]
        retention_series.append((f"{stage} {interval}", [numeric_percent(r["Retention_Rate"]) for r in rows], COLORS[idx]))
        jaccard_series.append((f"{stage} {interval}", [float(r["Jaccard"]) for r in rows], COLORS[idx]))
    for stem, title, series, y_min, y_max, y_label in [
        ("retention_curves", "Retention by Course Interval", retention_series, 0, 100, "Retention (%)"),
        ("jaccard_curves", "Jaccard Similarity by Course Interval", jaccard_series, 0, 1, "Jaccard"),
    ]:
        png = out_dir / f"{stem}.png"
        svg = out_dir / f"{stem}.svg"
        write_png(png, series, y_min, y_max)
        write_svg(svg, title, series, y_min, y_max, y_label)
        paths.extend([png, svg])

    b3 = rows_by_stage["B3"]
    b3_series_defs = [
        ("B3_current_size", "B3 Current Size", [("Current_Size", [float(r["Current_Size"]) for r in b3], COLORS[0])], 0, max(float(r["Current_Size"]) for r in b3) * 1.1, "Count"),
        ("B3_retention", "B3 Retention", [("Retention", [numeric_percent(r["Retention_Rate"]) for r in b3], COLORS[1])], 0, 100, "Retention (%)"),
        ("B3_inflow", "B3 Inflow", [("Inflow", [numeric_percent(r["Inflow_Rate"]) for r in b3], COLORS[2])], 0, 100, "Inflow (%)"),
        ("B3_jaccard", "B3 Jaccard", [("Jaccard", [float(r["Jaccard"]) for r in b3], COLORS[3])], 0, 1, "Jaccard"),
    ]
    for stem, title, series, y_min, y_max, y_label in b3_series_defs:
        png = out_dir / f"{stem}.png"
        svg = out_dir / f"{stem}.svg"
        write_png(png, series, y_min, y_max)
        write_svg(svg, title, series, y_min, y_max, y_label)
        paths.extend([png, svg])
    return paths


def print_integrity(integrity: Sequence[Dict[str, object]], expected_n: int) -> None:
    print("==== Data Integrity Check ====")
    print("checkpoint\tsample_count\tmissing_vs_union\tduplicate_count\tbad_rollout_count\tabnormal_success_rate_count")
    for row in integrity:
        print(
            f"{row['checkpoint']}\t{row['sample_count']}\t{row['missing_vs_union']}\t"
            f"{row['duplicate_count']}\t{row['bad_rollout_count']}\t{row['abnormal_success_rate_count']}"
        )
    counts = {int(row["sample_count"]) for row in integrity}
    if counts != {expected_n}:
        print(f"WARNING: sample counts differ from expected_n={expected_n}: {sorted(counts)}")


def main():
    args = parse_args()
    out_dir = args.output_dir or (args.input_dir / "difficulty_membership_drift")
    out_dir.mkdir(parents=True, exist_ok=True)
    data, integrity, sample_ids, files = load_all(args)
    print_integrity(integrity, args.expected_n)
    print(f"Aligned effective N: {len(sample_ids)}")

    stage_sets = build_sets(data, sample_ids)
    all_rows = build_membership_rows(stage_sets)
    adjacent_rows = build_adjacent_rows(stage_sets)

    all_columns = [
        "Stage", "Interval", "Checkpoint", "Initial_Size", "Current_Size", "Size_Change",
        "Size_Change_Rate", "Intersection_Count", "Retention_Rate", "Outflow_Count",
        "Outflow_Rate", "Inflow_Count", "Inflow_Rate", "Jaccard",
    ]
    adjacent_columns = [
        "Stage", "Previous_Checkpoint", "Current_Checkpoint", "Previous_Size", "Current_Size",
        "Adjacent_Retention", "Adjacent_Outflow", "Adjacent_Inflow", "Adjacent_Jaccard",
    ]
    all_path = out_dir / "membership_drift_all_stages.csv"
    adjacent_path = out_dir / "adjacent_membership_drift.csv"
    integrity_path = out_dir / "data_integrity_check.csv"
    metadata_path = out_dir / "metadata.json"
    write_csv(all_path, all_rows, all_columns)
    write_csv(adjacent_path, adjacent_rows, adjacent_columns)
    write_csv(
        integrity_path,
        integrity,
        ["checkpoint", "file", "sample_count", "missing_vs_union", "duplicate_count", "missing_id_count", "missing_p_count", "bad_rollout_count", "abnormal_success_rate_count", "effective_aligned_n"],
    )

    stage_paths: Dict[str, Path] = {}
    sheets = [("All_Stages", all_rows, all_columns)]
    for stage, _interval, _lower, _upper in STAGES:
        rows = [row for row in all_rows if row["Stage"] == stage]
        path = out_dir / f"{stage}_membership_drift.csv"
        write_csv(path, rows, all_columns)
        stage_paths[stage] = path
        sheets.append((stage, rows, all_columns))
    sheets.append(("Adjacent_Drift", adjacent_rows, adjacent_columns))
    xlsx_path = out_dir / "membership_drift_statistics.xlsx"
    write_xlsx(xlsx_path, sheets)
    plot_paths = make_plots(out_dir, all_rows)

    metadata = {
        "input_files": [str(path) for path in files],
        "effective_n": len(sample_ids),
        "checkpoints": CHECKPOINTS,
        "stages": [{"stage": s, "interval": interval, "lower": lo, "upper": hi} for s, interval, lo, hi in STAGES],
        "definitions": {
            "Retention": "|S0 intersect St| / |S0|",
            "Outflow": "|S0 - St| / |S0|",
            "Inflow": "|St - S0| / |St|",
            "Jaccard": "|S0 intersect St| / |S0 union St|",
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("==== Initial Sizes ====")
    for stage, interval, _lower, _upper in STAGES:
        print(f"{stage} {interval}: {len(stage_sets[stage][0])}")

    print("==== Key Checkpoints ====")
    key_rows = [
        row for row in all_rows
        if int(row["Checkpoint"]) in {100, 500, 1000, 7473}
    ]
    print("Stage\tCheckpoint\tCurrent_Size\tRetention\tInflow\tJaccard")
    for stage, *_ in STAGES:
        for row in [r for r in key_rows if r["Stage"] == stage]:
            print(f"{stage}\t{row['Checkpoint']}\t{row['Current_Size']}\t{row['Retention_Rate']}\t{row['Inflow_Rate']}\t{row['Jaccard']}")

    final_rows = [row for row in all_rows if int(row["Checkpoint"]) == 7473]
    most_changed = min(final_rows, key=lambda row: float(row["Jaccard"]))
    print("==== Summary ====")
    print(f"Most changed interval by final Jaccard: {most_changed['Stage']} {most_changed['Interval']} Jaccard={most_changed['Jaccard']}")
    for row in final_rows:
        size_rate = abs(numeric_percent(row["Size_Change_Rate"]))
        retention = numeric_percent(row["Retention_Rate"])
        jaccard = float(row["Jaccard"])
        if size_rate <= 10 and (retention <= 60 or jaccard <= 0.5):
            print(
                "Stable size but changed membership: "
                f"{row['Stage']} size_change={row['Size_Change_Rate']} "
                f"retention={row['Retention_Rate']} jaccard={row['Jaccard']}"
            )
    print("==== Outputs ====")
    for path in [all_path, *[stage_paths[s] for s, *_ in STAGES], adjacent_path, xlsx_path, integrity_path, metadata_path, *plot_paths]:
        print(path.resolve())


if __name__ == "__main__":
    main()
