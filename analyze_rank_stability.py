import argparse
import csv
import json
import math
import struct
import zlib
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from analyze_vanilla_grpo_difficulty_drift import write_xlsx


CHECKPOINTS = [0, 10, 20, 30, 40, 50, 100, 250, 500, 1000, 1500, 2000, 4000, 7473]
SCATTER_CHECKPOINTS = [100, 500, 1000, 7473]
DEFAULT_INPUT_DIR = Path(
    "outputs/observation_sets_gsm8k_p_intervals_seed42/"
    "difficulty_drift_eval_pdist20_seed42"
)
DEFAULT_ORIGINAL_NAME = "gsm8k_p_scores_final_no_truncation_2_sampled_20pct.jsonl"
EPS = 1e-9


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze whether initial p-score difficulty ranking remains stable over checkpoints."
    )
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--original_file", type=Path, default=None)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Defaults to <input_dir>/difficulty_rank_stability",
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
    matches = sorted(input_dir.glob(f"*checkpoint-{checkpoint}.jsonl"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one checkpoint-{checkpoint} JSONL in {input_dir}, got {len(matches)}")
    return matches[0]


def fmt(value: float, digits: int = 6) -> str:
    if math.isnan(value):
        return "NaN"
    return f"{value:.{digits}f}"


def fmt_p(value: float) -> str:
    if math.isnan(value):
        return "NaN"
    if value == 0:
        return "0.000e+00"
    return f"{value:.3e}"


def pct(value: float) -> str:
    if math.isnan(value):
        return "NaN"
    return f"{value * 100:.2f}%"


def normal_two_sided_p(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


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
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
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
        "raw_sample_count": len(values),
        "duplicate_count": duplicate_count,
        "missing_id_count": missing_id_count,
        "missing_p_count": missing_p_count,
        "bad_rollout_count": bad_rollout_count,
        "abnormal_p_count": abnormal_p_count,
    }


def load_all(args) -> Tuple[Dict[int, Dict[str, float]], List[Dict[str, object]], List[str], List[Path]]:
    original_file = args.original_file or (args.input_dir / DEFAULT_ORIGINAL_NAME)
    files = [checkpoint_file(args.input_dir, original_file, ckpt) for ckpt in CHECKPOINTS]
    data: Dict[int, Dict[str, float]] = {}
    integrity: List[Dict[str, object]] = []
    for ckpt, path in zip(CHECKPOINTS, files):
        values, row = read_checkpoint(
            path, ckpt, args.sample_id_field, args.p_field, args.rollout_field, args.expected_rollouts
        )
        data[ckpt] = values
        integrity.append(row)
    id_sets = [set(data[ckpt]) for ckpt in CHECKPOINTS]
    common = sorted(set.intersection(*id_sets), key=lambda x: int(x) if x.isdigit() else x)
    base_ids = set(data[0])
    for row in integrity:
        ids = set(data[int(row["checkpoint"])])
        row["missing_vs_checkpoint0"] = len(base_ids - ids)
        row["effective_n_vs_checkpoint0"] = len(base_ids & ids)
        row["effective_aligned_n_all"] = len(common)
    return data, integrity, common, files


def average_ranks(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and abs(indexed[j][1] - indexed[i][1]) <= EPS:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    sx = sum((v - mx) ** 2 for v in x)
    sy = sum((v - my) ** 2 for v in y)
    if sx <= 0 or sy <= 0:
        return math.nan
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(sx * sy)


def spearman_with_ties(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    rho = pearson(average_ranks(x), average_ranks(y))
    if math.isnan(rho) or abs(rho) >= 1:
        return rho, 0.0 if not math.isnan(rho) else math.nan
    # Large-N normal approximation via the usual t statistic; scipy is unavailable in this environment.
    t = rho * math.sqrt((len(x) - 2) / max(1e-15, 1 - rho * rho))
    return rho, normal_two_sided_p(t)


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    concordant = discordant = ties_x = ties_y = 0
    n = len(x)
    for i in range(n - 1):
        xi = x[i]
        yi = y[i]
        for j in range(i + 1, n):
            dx = (xi > x[j]) - (xi < x[j])
            dy = (yi > y[j]) - (yi < y[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    tau = math.nan if denom == 0 else (concordant - discordant) / denom
    if math.isnan(tau):
        return tau, math.nan
    # Asymptotic normal approximation; tau-b numerator/denominator handles ties.
    var = 2 * (2 * n + 5) / (9 * n * (n - 1))
    z = tau / math.sqrt(var) if var > 0 else math.inf
    return tau, normal_two_sided_p(z)


def tie_stats(values: Sequence[float]) -> Tuple[int, int, float]:
    counts = Counter(values)
    largest = max(counts.values())
    return len(counts), largest, largest / len(values)


def build_statistics(
    data: Dict[int, Dict[str, float]], integrity: List[Dict[str, object]]
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    stats_rows: List[Dict[str, object]] = []
    tie_rows: List[Dict[str, object]] = []
    detail_rows: List[Dict[str, object]] = []
    base_ids = set(data[0])

    for ckpt in CHECKPOINTS:
        ids = sorted(base_ids & set(data[ckpt]), key=lambda x: int(x) if x.isdigit() else x)
        p0 = [data[0][sid] for sid in ids]
        pt = [data[ckpt][sid] for sid in ids]
        rho, rho_p = spearman_with_ties(p0, pt)
        tau, tau_p = kendall_tau_b(p0, pt)
        unique_count, largest_tie, largest_rate = tie_stats(pt)
        if ckpt == 0:
            rho, rho_p, tau, tau_p = 1.0, 0.0, 1.0, 0.0
        stats_rows.append(
            {
                "Checkpoint": ckpt,
                "N": len(ids),
                "Unique_P_Count": unique_count,
                "Largest_Tie_Group": largest_tie,
                "Largest_Tie_Group_Rate": pct(largest_rate),
                "Spearman_rho": fmt(rho),
                "Spearman_pvalue": fmt_p(rho_p),
                "Kendall_tau_b": fmt(tau),
                "Kendall_pvalue": fmt_p(tau_p),
            }
        )
        tie_rows.append(
            {
                "Checkpoint": ckpt,
                "N": len(ids),
                "Unique_P_Count": unique_count,
                "Largest_Tie_Group": largest_tie,
                "Largest_Tie_Group_Rate": pct(largest_rate),
            }
        )
        rank0 = average_ranks(p0)
        rankt = average_ranks(pt)
        for sid, a, b, r0, rt in zip(ids, p0, pt, rank0, rankt):
            detail_rows.append(
                {
                    "Checkpoint": ckpt,
                    "sample_id": sid,
                    "P0": a,
                    "Pt": b,
                    "rank_P0": fmt(r0),
                    "rank_Pt": fmt(rt),
                }
            )
        for row in integrity:
            if int(row["checkpoint"]) == ckpt:
                row["final_effective_n"] = len(ids)
    return stats_rows, tie_rows, detail_rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def hex_to_rgb(color: str) -> Tuple[int, int, int]:
    color = color.lstrip("#")
    return int(color[:2], 16), int(color[2:4], 16), int(color[4:6], 16)


def png_canvas(width: int, height: int, bg=(255, 255, 255)) -> bytearray:
    return bytearray(bg * width * height)


def put(canvas: bytearray, width: int, height: int, x: int, y: int, color: Tuple[int, int, int]) -> None:
    if 0 <= x < width and 0 <= y < height:
        i = (y * width + x) * 3
        canvas[i:i + 3] = bytes(color)


def line(canvas: bytearray, width: int, height: int, x0: float, y0: float, x1: float, y1: float, color, thickness=2):
    x0, y0, x1, y1 = map(round, [x0, y0, x1, y1])
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        for ox in range(-thickness, thickness + 1):
            for oy in range(-thickness, thickness + 1):
                put(canvas, width, height, x0 + ox, y0 + oy, color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def circle(canvas: bytearray, width: int, height: int, cx: float, cy: float, r: int, color) -> None:
    cx, cy = round(cx), round(cy)
    for x in range(cx - r, cx + r + 1):
        for y in range(cy - r, cy + r + 1):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                put(canvas, width, height, x, y, color)


def write_png(path: Path, width: int, height: int, canvas: bytearray) -> None:
    rows = []
    for y in range(height):
        rows.append(b"\x00" + bytes(canvas[(y * width) * 3:((y + 1) * width) * 3]))
    raw = b"".join(rows)

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


def plot_correlation_png(path: Path, stats_rows: Sequence[Dict[str, object]]) -> None:
    width, height = 1500, 950
    left, right, top, bottom = 110, 70, 90, 120
    canvas = png_canvas(width, height)
    grid, axis = (225, 225, 225), (35, 35, 35)

    def x_at(ckpt: int) -> float:
        return left + math.log10(ckpt + 1) / math.log10(7473 + 1) * (width - left - right)

    def y_at(v: float) -> float:
        return top + (1 - v) * (height - top - bottom)

    for k in range(6):
        y = top + k * (height - top - bottom) / 5
        line(canvas, width, height, left, y, width - right, y, grid, 1)
    line(canvas, width, height, left, top, left, height - bottom, axis, 2)
    line(canvas, width, height, left, height - bottom, width - right, height - bottom, axis, 2)
    series = [
        ("Spearman_rho", (31, 119, 180)),
        ("Kendall_tau_b", (227, 74, 51)),
    ]
    for key, color in series:
        pts = [(x_at(int(r["Checkpoint"])), y_at(float(r[key]))) for r in stats_rows]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            line(canvas, width, height, x0, y0, x1, y1, color, 3)
        for x, y in pts:
            circle(canvas, width, height, x, y, 6, color)
    write_png(path, width, height, canvas)


def plot_count_matrix_png(path: Path, p0: Sequence[float], pt: Sequence[float]) -> None:
    width, height = 1050, 1050
    left, right, top, bottom = 100, 70, 70, 100
    canvas = png_canvas(width, height)
    matrix = [[0 for _ in range(33)] for _ in range(33)]
    for a, b in zip(p0, pt):
        x = round(a * 32)
        y = round(b * 32)
        matrix[y][x] += 1
    max_count = max(max(row) for row in matrix)
    cell_w = (width - left - right) / 33
    cell_h = (height - top - bottom) / 33
    for yi in range(33):
        for xi in range(33):
            c = matrix[yi][xi]
            if c == 0:
                color = (250, 250, 250)
            else:
                t = math.log1p(c) / math.log1p(max_count)
                color = (round(255 - 220 * t), round(245 - 145 * t), round(235 - 20 * t))
            x0 = round(left + xi * cell_w)
            x1 = round(left + (xi + 1) * cell_w)
            y0 = round(top + (32 - yi) * cell_h)
            y1 = round(top + (33 - yi) * cell_h)
            for x in range(x0, x1):
                for y in range(y0, y1):
                    put(canvas, width, height, x, y, color)
    grid = (220, 220, 220)
    axis = (30, 30, 30)
    for i in range(34):
        x = left + i * cell_w
        y = top + i * cell_h
        line(canvas, width, height, x, top, x, height - bottom, grid, 1)
        line(canvas, width, height, left, y, width - right, y, grid, 1)
    line(canvas, width, height, left, height - bottom, width - right, top, (20, 20, 20), 2)
    line(canvas, width, height, left, top, left, height - bottom, axis, 2)
    line(canvas, width, height, left, height - bottom, width - right, height - bottom, axis, 2)
    write_png(path, width, height, canvas)


def write_correlation_svg(path: Path, stats_rows: Sequence[Dict[str, object]]) -> None:
    width, height = 1120, 680
    left, right, top, bottom = 90, 190, 60, 95
    plot_w, plot_h = width - left - right, height - top - bottom

    def x_at(ckpt: int) -> float:
        return left + math.log10(ckpt + 1) / math.log10(7473 + 1) * plot_w

    def y_at(v: float) -> float:
        return top + (1 - v) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="34" font-family="Arial" font-size="20" font-weight="700">Difficulty Rank Correlation vs Checkpoint</text>',
    ]
    for k in range(6):
        v = k / 5
        y = y_at(v)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e5e5"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{v:.1f}</text>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>')
    lines.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>')
    for ckpt in CHECKPOINTS:
        x = x_at(ckpt)
        lines.append(f'<text x="{x:.1f}" y="{top + plot_h + 30}" transform="rotate(-45 {x:.1f} {top + plot_h + 30})" text-anchor="end" font-family="Arial" font-size="11">{ckpt}</text>')
    specs = [("Spearman_rho", "#1f77b4"), ("Kendall_tau_b", "#d62728")]
    for key, color in specs:
        pts = " ".join(f'{x_at(int(r["Checkpoint"])):.1f},{y_at(float(r[key])):.1f}' for r in stats_rows)
        lines.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.6"/>')
        for r in stats_rows:
            lines.append(f'<circle cx="{x_at(int(r["Checkpoint"])):.1f}" cy="{y_at(float(r[key])):.1f}" r="3.5" fill="{color}"/>')
    lx, ly = left + plot_w + 24, top + 25
    for i, (label, color) in enumerate(specs):
        y = ly + i * 26
        lines.append(f'<line x1="{lx}" y1="{y}" x2="{lx + 24}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{lx + 32}" y="{y + 4}" font-family="Arial" font-size="13">{label}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_count_matrix_svg(path: Path, ckpt: int, p0: Sequence[float], pt: Sequence[float]) -> None:
    width, height = 820, 820
    left, right, top, bottom = 85, 40, 55, 85
    plot_w, plot_h = width - left - right, height - top - bottom
    matrix = [[0 for _ in range(33)] for _ in range(33)]
    for a, b in zip(p0, pt):
        matrix[round(b * 32)][round(a * 32)] += 1
    max_count = max(max(row) for row in matrix)
    cell_w, cell_h = plot_w / 33, plot_h / 33
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="24" y="32" font-family="Arial" font-size="18" font-weight="700">P0 vs P{ckpt} Count Matrix</text>',
    ]
    for yi in range(33):
        for xi in range(33):
            c = matrix[yi][xi]
            t = 0 if c == 0 else math.log1p(c) / math.log1p(max_count)
            color = f'rgb({round(255 - 220*t)},{round(245 - 145*t)},{round(235 - 20*t)})'
            x = left + xi * cell_w
            y = top + (32 - yi) * cell_h
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" fill="{color}" stroke="#eee" stroke-width="0.4"/>')
    lines.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top}" stroke="#111" stroke-width="2"/>')
    for v in [0, .25, .5, .75, 1.0]:
        x = left + v * plot_w
        y = top + (1 - v) * plot_h
        lines.append(f'<text x="{x:.1f}" y="{top + plot_h + 24}" text-anchor="middle" font-family="Arial" font-size="11">{v:.2f}</text>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{v:.2f}</text>')
    lines.append(f'<text x="{left + plot_w/2}" y="{height - 20}" text-anchor="middle" font-family="Arial" font-size="13">P0</text>')
    lines.append(f'<text x="18" y="{top + plot_h/2}" transform="rotate(-90 18 {top + plot_h/2})" text-anchor="middle" font-family="Arial" font-size="13">P{ckpt}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    out_dir = args.output_dir or (args.input_dir / "difficulty_rank_stability")
    out_dir.mkdir(parents=True, exist_ok=True)
    data, integrity, common_ids, files = load_all(args)
    print("==== Data Integrity Check ====")
    print("checkpoint\traw_sample_count\tduplicate_count\tmissing_vs_checkpoint0\tfinal_effective_N\tabnormal_P")
    for row in integrity:
        print(f"{row['checkpoint']}\t{row['raw_sample_count']}\t{row['duplicate_count']}\t{row['missing_vs_checkpoint0']}\t{row['effective_n_vs_checkpoint0']}\t{row['abnormal_p_count']}")

    stats_rows, tie_rows, detail_rows = build_statistics(data, integrity)
    stat_cols = [
        "Checkpoint", "N", "Unique_P_Count", "Largest_Tie_Group", "Largest_Tie_Group_Rate",
        "Spearman_rho", "Spearman_pvalue", "Kendall_tau_b", "Kendall_pvalue",
    ]
    tie_cols = ["Checkpoint", "N", "Unique_P_Count", "Largest_Tie_Group", "Largest_Tie_Group_Rate"]
    detail_cols = ["Checkpoint", "sample_id", "P0", "Pt", "rank_P0", "rank_Pt"]
    integrity_cols = [
        "checkpoint", "file", "raw_sample_count", "duplicate_count", "missing_id_count", "missing_p_count",
        "missing_vs_checkpoint0", "effective_n_vs_checkpoint0", "bad_rollout_count", "abnormal_p_count", "final_effective_n",
    ]

    stats_path = out_dir / "rank_correlation_statistics.csv"
    tie_path = out_dir / "tie_statistics.csv"
    detail_path = out_dir / "rank_detail_by_checkpoint.csv"
    integrity_path = out_dir / "data_integrity_check.csv"
    xlsx_path = out_dir / "rank_correlation_statistics.xlsx"
    meta_path = out_dir / "metadata.json"
    write_csv(stats_path, stats_rows, stat_cols)
    write_csv(tie_path, tie_rows, tie_cols)
    write_csv(detail_path, detail_rows, detail_cols)
    write_csv(integrity_path, integrity, integrity_cols)
    write_xlsx(xlsx_path, [("Rank_Correlation", stats_rows, stat_cols), ("Tie_Statistics", tie_rows, tie_cols)])

    corr_png = out_dir / "difficulty_rank_correlation.png"
    corr_svg = out_dir / "difficulty_rank_correlation.svg"
    plot_correlation_png(corr_png, stats_rows)
    write_correlation_svg(corr_svg, stats_rows)
    base_ids = sorted(set(data[0]), key=lambda x: int(x) if x.isdigit() else x)
    plot_paths = [corr_png, corr_svg]
    for ckpt in SCATTER_CHECKPOINTS:
        ids = [sid for sid in base_ids if sid in data[ckpt]]
        p0 = [data[0][sid] for sid in ids]
        pt = [data[ckpt][sid] for sid in ids]
        png = out_dir / f"p0_vs_p{ckpt}.png"
        svg = out_dir / f"p0_vs_p{ckpt}.svg"
        plot_count_matrix_png(png, p0, pt)
        write_count_matrix_svg(svg, ckpt, p0, pt)
        plot_paths.extend([png, svg])

    metadata = {
        "input_files": [str(path) for path in files],
        "effective_aligned_n_all": len(common_ids),
        "method": {
            "spearman": "Pearson correlation of average ranks; ties receive average ranks. p-value uses large-N normal approximation because scipy is unavailable.",
            "kendall_tau_b": "Pairwise concordant/discordant counts with ties in x/y handled in tau-b denominator. p-value uses large-N normal approximation because scipy is unavailable.",
        },
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    key = {int(r["Checkpoint"]): r for r in stats_rows}
    min_s = min(stats_rows, key=lambda r: float(r["Spearman_rho"]))
    min_k = min(stats_rows, key=lambda r: float(r["Kendall_tau_b"]))
    largest_tie = max(tie_rows, key=lambda r: int(r["Largest_Tie_Group"]))
    print("==== Summary ====")
    print(f"Input files: {len(files)}")
    for path in files:
        print(f"  {path}")
    print(f"Effective aligned N: {len(common_ids)}")
    for ckpt in SCATTER_CHECKPOINTS:
        print(
            f"checkpoint {ckpt}: Spearman={key[ckpt]['Spearman_rho']}, "
            f"Kendall_tau_b={key[ckpt]['Kendall_tau_b']}"
        )
    print(f"Lowest Spearman: checkpoint {min_s['Checkpoint']} rho={min_s['Spearman_rho']}")
    print(f"Lowest Kendall tau-b: checkpoint {min_k['Checkpoint']} tau={min_k['Kendall_tau_b']}")
    print(
        "Largest tie group: "
        f"checkpoint {largest_tie['Checkpoint']} size={largest_tie['Largest_Tie_Group']} "
        f"rate={largest_tie['Largest_Tie_Group_Rate']}"
    )
    print("==== Outputs ====")
    for path in [stats_path, tie_path, detail_path, integrity_path, xlsx_path, meta_path, *plot_paths]:
        print(path.resolve())


if __name__ == "__main__":
    main()
