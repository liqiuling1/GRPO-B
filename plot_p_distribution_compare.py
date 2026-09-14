import argparse
import json
import math
import os
import tempfile
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Counter as CounterType, Dict, List, Optional, Tuple


def parse_decimal(value) -> Optional[Decimal]:
    if value is None:
        return None
    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None
    if not number.is_finite():
        return None
    return number


def format_decimal(value: Decimal) -> str:
    normalized = value.normalize()
    text = format(normalized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def load_p_counts(path: Path, p_field: str) -> Tuple[CounterType[Decimal], int, int]:
    counts: CounterType[Decimal] = Counter()
    total_rows = 0
    skipped_rows = 0

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            total_rows += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON: {exc}") from exc

            p_value = parse_decimal(record.get(p_field))
            if p_value is None:
                skipped_rows += 1
                continue
            counts[p_value] += 1

    return counts, total_rows, skipped_rows


def default_label(path: Path) -> str:
    return path.stem


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare sample counts at each p difficulty value across jsonl datasets."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Jsonl files to compare. Each row should contain the p field.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional display labels. Must have the same count as --inputs.",
    )
    parser.add_argument("--output", default="plots/p_distribution_compare.png", help="Output image path.")
    parser.add_argument("--p_field", default="p", help="Difficulty field name in each jsonl row.")
    parser.add_argument("--title", default="GSM8K P Difficulty Distribution", help="Plot title.")
    parser.add_argument("--ylabel", default="Count", help="Y axis label.")
    parser.add_argument("--dpi", type=int, default=180, help="Output image DPI.")
    parser.add_argument(
        "--show_values",
        action="store_true",
        help="Draw count labels above bars. Useful for small numbers of p values.",
    )
    parser.add_argument(
        "--max_xtick_labels",
        type=int,
        default=40,
        help="Show every x tick label when there are at most this many p values; otherwise thin labels.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_paths = [Path(name) for name in args.inputs]
    labels = args.labels if args.labels else [default_label(path) for path in input_paths]
    if len(labels) != len(input_paths):
        raise SystemExit(f"--labels count ({len(labels)}) must match --inputs count ({len(input_paths)}).")

    series: List[Tuple[Path, str, CounterType[Decimal], int, int]] = []
    all_p_values = set()
    for path, label in zip(input_paths, labels):
        counts, total_rows, skipped_rows = load_p_counts(path, args.p_field)
        if not counts:
            raise SystemExit(f"No valid `{args.p_field}` values found in {path}.")
        series.append((path, label, counts, total_rows, skipped_rows))
        all_p_values.update(counts.keys())

    p_values = sorted(all_p_values)
    x_labels = [format_decimal(value) for value in p_values]

    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="matplotlib-", dir="/tmp"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dataset_count = len(series)
    x_positions = list(range(len(p_values)))
    group_width = 0.82
    bar_width = group_width / max(1, dataset_count)
    figure_width = max(11.0, min(28.0, 0.42 * len(p_values) + 2.8 * dataset_count))

    fig, axis = plt.subplots(figsize=(figure_width, 6.2))
    colors = plt.get_cmap("tab10").colors

    for dataset_index, (_path, label, counts, _total_rows, _skipped_rows) in enumerate(series):
        offset = (dataset_index - (dataset_count - 1) / 2) * bar_width
        ys = [counts.get(p_value, 0) for p_value in p_values]
        bars = axis.bar(
            [x + offset for x in x_positions],
            ys,
            width=bar_width * 0.92,
            label=label,
            color=colors[dataset_index % len(colors)],
            edgecolor="black",
            linewidth=0.35,
        )
        if args.show_values:
            axis.bar_label(bars, labels=[str(value) if value else "" for value in ys], fontsize=7, padding=2)

    axis.set_title(args.title)
    axis.set_xlabel(args.p_field)
    axis.set_ylabel(args.ylabel)
    axis.grid(axis="y", alpha=0.25, linestyle="--")
    axis.set_axisbelow(True)

    if len(x_labels) <= args.max_xtick_labels:
        tick_positions = x_positions
        tick_labels = x_labels
    else:
        step = max(1, math.ceil(len(x_labels) / args.max_xtick_labels))
        tick_positions = x_positions[::step]
        tick_labels = x_labels[::step]

    axis.set_xticks(tick_positions)
    axis.set_xticklabels(tick_labels, rotation=45, ha="right")
    axis.legend(fontsize=8)
    fig.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi)
    plt.close(fig)

    print(f"Saved plot: {output_path}")
    for path, label, counts, total_rows, skipped_rows in series:
        print(
            f"{label}: file={path}, rows={total_rows}, valid_p={sum(counts.values())}, "
            f"unique_p={len(counts)}, skipped={skipped_rows}"
        )


if __name__ == "__main__":
    main()
