import argparse
import json
import os
import tempfile
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Counter as CounterType, List, Optional, Tuple


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


def parse_p_bin_edges(text: str) -> List[Decimal]:
    if not text.strip():
        raise ValueError("--p_bin_edges cannot be empty.")

    try:
        edges = [Decimal(part.strip()) for part in text.split(",") if part.strip()]
    except InvalidOperation as exc:
        raise ValueError(f"Invalid --p_bin_edges={text!r}. Expected comma-separated numbers.") from exc

    if len(edges) < 2:
        raise ValueError("--p_bin_edges must contain at least two values.")
    if any(not edge.is_finite() for edge in edges):
        raise ValueError("--p_bin_edges values must be finite numbers.")
    if any(edges[idx] >= edges[idx + 1] for idx in range(len(edges) - 1)):
        raise ValueError("--p_bin_edges must be strictly increasing.")
    return edges


def format_bin_label(left: Decimal, right: Decimal, is_last: bool) -> str:
    closing = "]" if is_last else ")"
    return f"[{format_decimal(left)}, {format_decimal(right)}{closing}"


def make_bin_labels(edges: List[Decimal]) -> List[str]:
    return [
        format_bin_label(edges[idx], edges[idx + 1], is_last=(idx == len(edges) - 2))
        for idx in range(len(edges) - 1)
    ]


def bucket_for_p(p_value: Decimal, edges: List[Decimal]) -> str:
    for idx in range(len(edges) - 1):
        left = edges[idx]
        right = edges[idx + 1]
        is_last = idx == len(edges) - 2
        if left <= p_value <= right if is_last else left <= p_value < right:
            return format_bin_label(left, right, is_last=is_last)

    raise ValueError(
        f"Found p={format_decimal(p_value)} outside the provided --p_bin_edges range "
        f"[{format_decimal(edges[0])}, {format_decimal(edges[-1])}]."
    )


def load_bin_counts(path: Path, p_field: str, edges: List[Decimal]) -> Tuple[CounterType[str], int, int]:
    counts: CounterType[str] = Counter()
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
            counts[bucket_for_p(p_value, edges)] += 1

    return counts, total_rows, skipped_rows


def default_label(path: Path) -> str:
    return path.stem


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare sample counts across manually configured p difficulty intervals."
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
    parser.add_argument(
        "--p_bin_edges",
        required=True,
        help=(
            "Comma-separated interval edges, for example '0,0.25,0.5,0.75,1'. "
            "Bins are [left, right), except the final bin is [left, right]."
        ),
    )
    parser.add_argument("--output", default="plots/p_distribution_compare_bins.png", help="Output image path.")
    parser.add_argument("--p_field", default="p", help="Difficulty field name in each jsonl row.")
    parser.add_argument("--title", default="GSM8K P Difficulty Distribution by Interval", help="Plot title.")
    parser.add_argument("--ylabel", default="Count", help="Y axis label.")
    parser.add_argument("--dpi", type=int, default=180, help="Output image DPI.")
    parser.add_argument(
        "--show_values",
        action="store_true",
        help="Draw count labels above bars.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    edges = parse_p_bin_edges(args.p_bin_edges)
    bin_labels = make_bin_labels(edges)
    input_paths = [Path(name) for name in args.inputs]
    labels = args.labels if args.labels else [default_label(path) for path in input_paths]
    if len(labels) != len(input_paths):
        raise SystemExit(f"--labels count ({len(labels)}) must match --inputs count ({len(input_paths)}).")

    series = []
    for path, label in zip(input_paths, labels):
        counts, total_rows, skipped_rows = load_bin_counts(path, args.p_field, edges)
        if not counts:
            raise SystemExit(f"No valid `{args.p_field}` values found in {path}.")
        series.append((path, label, counts, total_rows, skipped_rows))

    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="matplotlib-", dir="/tmp"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dataset_count = len(series)
    x_positions = list(range(len(bin_labels)))
    group_width = 0.82
    bar_width = group_width / max(1, dataset_count)
    figure_width = max(9.5, min(22.0, 1.25 * len(bin_labels) + 2.8 * dataset_count))

    fig, axis = plt.subplots(figsize=(figure_width, 6.2))
    colors = plt.get_cmap("tab10").colors

    for dataset_index, (_path, label, counts, _total_rows, _skipped_rows) in enumerate(series):
        offset = (dataset_index - (dataset_count - 1) / 2) * bar_width
        ys = [counts.get(bin_label, 0) for bin_label in bin_labels]
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
            axis.bar_label(bars, labels=[str(value) if value else "" for value in ys], fontsize=8, padding=2)

    axis.set_title(args.title)
    axis.set_xlabel(args.p_field + " interval")
    axis.set_ylabel(args.ylabel)
    axis.grid(axis="y", alpha=0.25, linestyle="--")
    axis.set_axisbelow(True)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(bin_labels, rotation=30, ha="right")
    axis.legend(fontsize=8)
    fig.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi)
    plt.close(fig)

    print(f"Saved plot: {output_path}")
    print(f"P bin edges: {', '.join(format_decimal(edge) for edge in edges)}")
    for path, label, counts, total_rows, skipped_rows in series:
        print(
            f"{label}: file={path}, rows={total_rows}, valid_p={sum(counts.values())}, "
            f"bins={len(bin_labels)}, skipped={skipped_rows}"
        )
        for bin_label in bin_labels:
            print(f"  {bin_label}: {counts.get(bin_label, 0)}")


if __name__ == "__main__":
    main()
