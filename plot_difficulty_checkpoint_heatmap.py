import argparse
import csv
import html
from pathlib import Path
from typing import List, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


DEFAULT_BIN_COLUMNS = [
    "P=0",
    "(0,0.125)",
    "[0.125,0.25)",
    "[0.25,0.375)",
    "[0.375,0.5)",
    "[0.5,0.625)",
    "[0.625,0.75)",
    "[0.75,0.875)",
    "[0.875,1)",
    "P=1",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot a checkpoint-by-difficulty-bin heatmap from difficulty_checkpoint_bin_table.csv."
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=Path(
            "outputs/observation_sets_gsm8k_p_intervals_seed42/"
            "difficulty_drift_eval_pdist20_seed42/drift_analysis/"
            "difficulty_checkpoint_bin_table.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to <input_csv stem>_<value>_heatmap.svg.",
    )
    parser.add_argument("--value", choices=["count", "fraction"], default="fraction")
    parser.add_argument(
        "--bins",
        nargs="*",
        default=DEFAULT_BIN_COLUMNS,
        help="Difficulty-bin columns to plot, in y-axis order.",
    )
    parser.add_argument("--title", default="")
    parser.add_argument("--cmap", default="YlGnBu")
    parser.add_argument(
        "--palette",
        choices=["orangered", "viridis", "bluegreen"],
        default="orangered",
        help="Color palette used by the dependency-free SVG renderer.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Draw cell values. Useful for small tables; can clutter large tables.",
    )
    parser.add_argument(
        "--fig_width",
        type=float,
        default=13.0,
    )
    parser.add_argument(
        "--fig_height",
        type=float,
        default=6.2,
    )
    return parser.parse_args()


def read_table(path: Path, bins: List[str]) -> Tuple[List[str], List[List[float]], List[int]]:
    checkpoints: List[str] = []
    totals: List[int] = []
    columns = {name: [] for name in bins}

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in ["checkpoint", *bins] if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing columns in {path}: {missing}")
        for row in reader:
            checkpoints.append(str(row["checkpoint"]))
            row_total = int(row.get("total") or sum(int(row[column]) for column in bins))
            totals.append(row_total)
            for column in bins:
                columns[column].append(int(row[column]))

    data = [[float(value) for value in columns[column]] for column in bins]
    return checkpoints, data, totals


def color_hex(value: float, vmin: float, vmax: float, palette: str) -> str:
    if vmax <= vmin:
        t = 0.0
    else:
        t = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
    palettes = {
        # Strong contrast for count/fraction tables: white -> yellow -> orange -> red -> purple.
        "orangered": [
            (255, 255, 255),
            (255, 245, 204),
            (254, 217, 118),
            (253, 141, 60),
            (227, 26, 28),
            (128, 0, 128),
        ],
        "viridis": [
            (68, 1, 84),
            (59, 82, 139),
            (33, 145, 140),
            (94, 201, 98),
            (253, 231, 37),
        ],
        "bluegreen": [
            (247, 252, 253),
            (204, 236, 230),
            (127, 205, 187),
            (65, 182, 196),
            (44, 127, 184),
            (37, 52, 148),
        ],
    }
    stops = palettes[palette]
    pos = t * (len(stops) - 1)
    idx = int(pos)
    frac = pos - idx
    if idx >= len(stops) - 1:
        rgb = stops[-1]
    else:
        a = stops[idx]
        b = stops[idx + 1]
        rgb = tuple(round(a[i] + (b[i] - a[i]) * frac) for i in range(3))
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def draw_svg(
    output: Path,
    checkpoints: List[str],
    bins: List[str],
    values: List[List[float]],
    title: str,
    value_kind: str,
    annotate: bool,
    palette: str,
) -> None:
    cell_w = 62
    cell_h = 34
    left = 112
    top = 64
    right = 36
    bottom = 70
    width = left + len(checkpoints) * cell_w + right
    height = top + len(bins) * cell_h + bottom
    flat_values = [value for row in values for value in row]
    vmin = min(flat_values) if flat_values else 0.0
    vmax = max(flat_values) if flat_values else 1.0
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">{html.escape(title)}</text>',
    ]
    for y, label in enumerate(bins):
        cy = top + y * cell_h + cell_h / 2 + 5
        lines.append(
            f'<text x="{left - 8}" y="{cy:.1f}" text-anchor="end" font-family="Arial" font-size="12">{html.escape(label)}</text>'
        )
    for x, label in enumerate(checkpoints):
        cx = left + x * cell_w + cell_w / 2
        lines.append(
            f'<text x="{cx:.1f}" y="{top + len(bins) * cell_h + 22}" text-anchor="middle" font-family="Arial" font-size="12" transform="rotate(-35 {cx:.1f} {top + len(bins) * cell_h + 22})">{html.escape(label)}</text>'
        )

    for y, row in enumerate(values):
        for x, val in enumerate(row):
            fill = color_hex(val, vmin, vmax, palette=palette)
            px = left + x * cell_w
            py = top + y * cell_h
            lines.append(
                f'<rect x="{px}" y="{py}" width="{cell_w}" height="{cell_h}" fill="{fill}" stroke="#ffffff" stroke-width="1"/>'
            )
            if annotate:
                text = f"{val:.2f}" if value_kind == "fraction" else str(int(round(val)))
                lines.append(
                    f'<text x="{px + cell_w / 2:.1f}" y="{py + cell_h / 2 + 4:.1f}" text-anchor="middle" font-family="Arial" font-size="11" fill="black">{html.escape(text)}</text>'
                )

    lines.extend(
        [
            f'<text x="{left + len(checkpoints) * cell_w / 2:.1f}" y="{height - 8}" text-anchor="middle" font-family="Arial" font-size="13">checkpoint</text>',
            f'<text x="16" y="{top + len(bins) * cell_h / 2:.1f}" text-anchor="middle" font-family="Arial" font-size="13" transform="rotate(-90 16 {top + len(bins) * cell_h / 2:.1f})">success-rate interval</text>',
            f'<text x="{width - right}" y="48" text-anchor="end" font-family="Arial" font-size="11">min={vmin:.3g}, max={vmax:.3g}</text>',
            "</svg>",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_heatmap(args) -> Path:
    checkpoints, counts, totals = read_table(args.input_csv, args.bins)
    if args.value == "fraction":
        values = []
        for row in counts:
            values.append([
                value / (totals[idx] if totals[idx] else 1)
                for idx, value in enumerate(row)
            ])
        colorbar_label = "fraction of samples"
        default_title = "Difficulty-bin Fraction by Checkpoint"
    else:
        values = counts
        colorbar_label = "sample count"
        default_title = "Difficulty-bin Count by Checkpoint"

    output = args.output
    if output is None:
        output = args.input_csv.with_name(f"{args.input_csv.stem}_{args.value}_heatmap.svg")
    output.parent.mkdir(parents=True, exist_ok=True)

    if plt is None or output.suffix.lower() == ".svg":
        if output.suffix.lower() != ".svg":
            output = output.with_suffix(".svg")
        draw_svg(
            output=output,
            checkpoints=checkpoints,
            bins=args.bins,
            values=values,
            title=args.title or default_title,
            value_kind=args.value,
            annotate=args.annotate,
            palette=args.palette,
        )
        return output

    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height), constrained_layout=True)
    image = ax.imshow(values, aspect="auto", cmap=args.cmap)

    ax.set_xticks(list(range(len(checkpoints))))
    ax.set_xticklabels(checkpoints, rotation=45, ha="right")
    ax.set_yticks(list(range(len(args.bins))))
    ax.set_yticklabels(args.bins)
    ax.set_xlabel("checkpoint")
    ax.set_ylabel("success-rate interval")
    ax.set_title(args.title or default_title)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(colorbar_label)

    if args.annotate:
        for y, row in enumerate(values):
            for x, val in enumerate(row):
                if args.value == "fraction":
                    text = f"{val:.2f}"
                else:
                    text = f"{int(val)}"
                ax.text(x, y, text, ha="center", va="center", fontsize=7, color="black")

    fig.savefig(output, dpi=args.dpi)
    plt.close(fig)
    return output


def main():
    args = parse_args()
    output = make_heatmap(args)
    print(f"Wrote heatmap: {output}")


if __name__ == "__main__":
    main()
