import argparse
import csv
import json
import math
import os
import re
import statistics
import tempfile
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze p-score difficulty drift for the same uid set across multiple JSONL score files."
        )
    )
    parser.add_argument(
        "--input_dir",
        default="outputs/difficulty_drift_eval_pdist20_seed42",
        help="Directory containing the original sampled JSONL and checkpoint JSONL files.",
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=None,
        help="Optional explicit JSONL file list. If omitted, all JSONL files in input_dir are used.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels matching --inputs. If omitted, labels are inferred from filenames.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory. Defaults to <input_dir>/drift_analysis.",
    )
    parser.add_argument("--uid_field", default="uid")
    parser.add_argument("--p_field", default="p")
    parser.add_argument(
        "--bin_edges",
        default="0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1",
        help="Comma-separated p bin edges. Bins are [left,right), except the final bin is [left,right].",
    )
    parser.add_argument(
        "--stable_delta",
        type=float,
        default=0.125,
        help="Absolute p change below this threshold is treated as stable for direction labels.",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Only write CSV/JSON outputs; skip matplotlib plots.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> Dict[str, dict]:
    rows = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "uid" not in row:
                raise ValueError(f"Missing uid in {path}:{line_no}")
            if "p" not in row:
                raise ValueError(f"Missing p in {path}:{line_no}")
            uid = str(row["uid"])
            if uid in rows:
                raise ValueError(f"Duplicate uid={uid} in {path}:{line_no}")
            rows[uid] = row
    return rows


def uid_sort_key(uid: str):
    try:
        return (0, int(uid))
    except ValueError:
        return (1, uid)


def infer_label(path: Path) -> str:
    name = path.stem
    match = CHECKPOINT_RE.search(name)
    if match:
        return f"checkpoint-{match.group(1)}"
    if "sampled" in name or "final_no_truncation" in name:
        return "original"
    return name


def file_sort_key(path: Path):
    label = infer_label(path)
    if label == "original":
        return (-1, label)
    match = CHECKPOINT_RE.search(label)
    if match:
        return (int(match.group(1)), label)
    return (10**9, label)


def parse_bin_edges(text: str) -> List[Decimal]:
    edges = [Decimal(part.strip()) for part in text.split(",") if part.strip()]
    if len(edges) < 2:
        raise ValueError("--bin_edges must contain at least two values.")
    if any(edges[idx] >= edges[idx + 1] for idx in range(len(edges) - 1)):
        raise ValueError("--bin_edges must be strictly increasing.")
    return edges


def decimal_p(value) -> Decimal:
    return Decimal(str(value))


def bin_label_for_p(p_value, edges: List[Decimal]) -> str:
    p = decimal_p(p_value)
    for idx in range(len(edges) - 1):
        left = edges[idx]
        right = edges[idx + 1]
        is_last = idx == len(edges) - 2
        if left <= p < right or (is_last and left <= p <= right):
            close = "]" if is_last else ")"
            return f"[{left}, {right}{close}"
    return "out_of_range"


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def std(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def median(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.median(values) if values else 0.0


def direction(delta: float, threshold: float) -> str:
    if delta >= threshold:
        return "p_increased_easier"
    if delta <= -threshold:
        return "p_decreased_harder"
    return "stable"


def safe_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label)


def write_csv(path: Path, fieldnames: List[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def stage_number(label: str, fallback: int) -> int:
    if label == "original":
        return 0
    match = CHECKPOINT_RE.search(label)
    return int(match.group(1)) if match else fallback


def build_series(paths: List[Path], labels: List[str]):
    stage_rows = [read_jsonl(path) for path in paths]
    uid_sets = [set(rows) for rows in stage_rows]
    common_uids = set.intersection(*uid_sets) if uid_sets else set()
    all_uids = set.union(*uid_sets) if uid_sets else set()
    missing_by_label = {
        label: len(all_uids - set(rows))
        for label, rows in zip(labels, stage_rows)
    }
    return stage_rows, sorted(common_uids, key=uid_sort_key), missing_by_label


def make_outputs(paths: List[Path], labels: List[str], args) -> None:
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.input_dir) / "drift_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    edges = parse_bin_edges(args.bin_edges)
    stage_rows, uids, missing_by_label = build_series(paths, labels)

    p_by_stage: List[Dict[str, float]] = []
    bin_by_stage: List[Dict[str, str]] = []
    trunc_by_stage: List[Dict[str, float]] = []
    for rows in stage_rows:
        p_by_stage.append({uid: float(rows[uid][args.p_field]) for uid in uids})
        bin_by_stage.append({uid: bin_label_for_p(rows[uid][args.p_field], edges) for uid in uids})
        trunc_by_stage.append(
            {uid: float(rows[uid].get("_sample_likely_truncated", 0) or 0) for uid in uids}
        )

    bin_labels = [bin_label_for_p(edges[idx], edges) for idx in range(len(edges) - 1)]
    bin_labels = []
    for idx in range(len(edges) - 1):
        close = "]" if idx == len(edges) - 2 else ")"
        bin_labels.append(f"[{edges[idx]}, {edges[idx + 1]}{close}")

    summary_rows = []
    for stage_idx, label in enumerate(labels):
        values = [p_by_stage[stage_idx][uid] for uid in uids]
        baseline_values = [p_by_stage[0][uid] for uid in uids]
        deltas = [value - base for value, base in zip(values, baseline_values)]
        summary_rows.append(
            {
                "label": label,
                "step": stage_number(label, stage_idx),
                "num_common_uids": len(uids),
                "missing_uids": missing_by_label[label],
                "mean_p": mean(values),
                "median_p": median(values),
                "std_p": std(values),
                "min_p": min(values) if values else "",
                "max_p": max(values) if values else "",
                "mean_delta_from_original": mean(deltas),
                "median_delta_from_original": median(deltas),
                "mean_abs_delta_from_original": mean(abs(delta) for delta in deltas),
                "truncated_completion_mean": mean(trunc_by_stage[stage_idx].values()),
            }
        )
    write_csv(
        output_dir / "difficulty_drift_overall_summary.csv",
        list(summary_rows[0].keys()) if summary_rows else [],
        summary_rows,
    )

    trajectory_rows = []
    for uid in uids:
        p_values = [p_by_stage[idx][uid] for idx in range(len(labels))]
        deltas_from_original = [value - p_values[0] for value in p_values]
        step_deltas = [
            p_values[idx] - p_values[idx - 1]
            for idx in range(1, len(p_values))
        ]
        row = {
            "uid": uid,
            "p_min": min(p_values),
            "p_max": max(p_values),
            "p_range": max(p_values) - min(p_values),
            "p_mean": mean(p_values),
            "p_std": std(p_values),
            "net_delta_original_to_last": p_values[-1] - p_values[0],
            "mean_abs_step_delta": mean(abs(delta) for delta in step_deltas),
            "direction_original_to_last": direction(
                p_values[-1] - p_values[0],
                args.stable_delta,
            ),
        }
        for label, value in zip(labels, p_values):
            row[f"p_{label}"] = value
            row[f"bin_{label}"] = bin_by_stage[labels.index(label)][uid]
        for label, delta_value in zip(labels[1:], deltas_from_original[1:]):
            row[f"delta_original_to_{label}"] = delta_value
        for idx in range(1, len(labels)):
            row[f"delta_{labels[idx - 1]}_to_{labels[idx]}"] = step_deltas[idx - 1]
        trajectory_rows.append(row)

    trajectory_fields = list(trajectory_rows[0].keys()) if trajectory_rows else []
    write_csv(output_dir / "difficulty_drift_uid_trajectories.csv", trajectory_fields, trajectory_rows)

    bin_count_rows = []
    for stage_idx, label in enumerate(labels):
        counts = Counter(bin_by_stage[stage_idx].values())
        for bin_label in bin_labels + ["out_of_range"]:
            count = counts.get(bin_label, 0)
            bin_uids = [uid for uid in uids if bin_by_stage[stage_idx][uid] == bin_label]
            p_values = [p_by_stage[stage_idx][uid] for uid in bin_uids]
            bin_count_rows.append(
                {
                    "label": label,
                    "step": stage_number(label, stage_idx),
                    "bin": bin_label,
                    "count": count,
                    "fraction": ratio(count, len(uids)),
                    "mean_p_in_bin": mean(p_values),
                }
            )
    write_csv(
        output_dir / "difficulty_drift_bin_counts.csv",
        ["label", "step", "bin", "count", "fraction", "mean_p_in_bin"],
        bin_count_rows,
    )

    pool_rows = []
    baseline_bins = bin_by_stage[0]
    for stage_idx in range(1, len(labels)):
        current_bins = bin_by_stage[stage_idx]
        for bin_label in bin_labels:
            before_uids = {uid for uid in uids if baseline_bins[uid] == bin_label}
            after_uids = {uid for uid in uids if current_bins[uid] == bin_label}
            retained = before_uids & after_uids
            outflow = before_uids - after_uids
            inflow = after_uids - before_uids
            union = before_uids | after_uids
            pool_rows.append(
                {
                    "before_label": labels[0],
                    "after_label": labels[stage_idx],
                    "after_step": stage_number(labels[stage_idx], stage_idx),
                    "bin": bin_label,
                    "before_count": len(before_uids),
                    "after_count": len(after_uids),
                    "retained_count": len(retained),
                    "outflow_count": len(outflow),
                    "inflow_count": len(inflow),
                    "retention_ratio": ratio(len(retained), len(before_uids)),
                    "outflow_ratio_of_before": ratio(len(outflow), len(before_uids)),
                    "inflow_ratio_of_after": ratio(len(inflow), len(after_uids)),
                    "jaccard_overlap": ratio(len(retained), len(union)),
                    "mean_delta_retained": mean(
                        p_by_stage[stage_idx][uid] - p_by_stage[0][uid] for uid in retained
                    ),
                    "mean_delta_outflow": mean(
                        p_by_stage[stage_idx][uid] - p_by_stage[0][uid] for uid in outflow
                    ),
                    "mean_delta_inflow": mean(
                        p_by_stage[stage_idx][uid] - p_by_stage[0][uid] for uid in inflow
                    ),
                }
            )
    write_csv(
        output_dir / "difficulty_drift_pool_stability_vs_original.csv",
        [
            "before_label",
            "after_label",
            "after_step",
            "bin",
            "before_count",
            "after_count",
            "retained_count",
            "outflow_count",
            "inflow_count",
            "retention_ratio",
            "outflow_ratio_of_before",
            "inflow_ratio_of_after",
            "jaccard_overlap",
            "mean_delta_retained",
            "mean_delta_outflow",
            "mean_delta_inflow",
        ],
        pool_rows,
    )

    change_rows = []
    for stage_idx in range(1, len(labels)):
        for uid in uids:
            from_bin = bin_by_stage[0][uid]
            to_bin = bin_by_stage[stage_idx][uid]
            delta_value = p_by_stage[stage_idx][uid] - p_by_stage[0][uid]
            change_rows.append(
                {
                    "uid": uid,
                    "from_label": labels[0],
                    "to_label": labels[stage_idx],
                    "to_step": stage_number(labels[stage_idx], stage_idx),
                    "p_from": p_by_stage[0][uid],
                    "p_to": p_by_stage[stage_idx][uid],
                    "delta": delta_value,
                    "from_bin": from_bin,
                    "to_bin": to_bin,
                    "bin_change": "retained" if from_bin == to_bin else "moved",
                    "direction": direction(delta_value, args.stable_delta),
                }
            )
    write_csv(
        output_dir / "difficulty_drift_uid_bin_changes_vs_original.csv",
        [
            "uid",
            "from_label",
            "to_label",
            "to_step",
            "p_from",
            "p_to",
            "delta",
            "from_bin",
            "to_bin",
            "bin_change",
            "direction",
        ],
        change_rows,
    )

    transition_dir = output_dir / "transition_matrices"
    transition_dir.mkdir(exist_ok=True)
    for from_idx, to_idx, prefix in (
        [(0, idx, "baseline") for idx in range(1, len(labels))]
        + [(idx - 1, idx, "consecutive") for idx in range(1, len(labels))]
    ):
        from_counts = Counter(bin_by_stage[from_idx].values())
        transition_counts = Counter(
            (bin_by_stage[from_idx][uid], bin_by_stage[to_idx][uid]) for uid in uids
        )
        rows = []
        for from_bin in bin_labels:
            for to_bin in bin_labels:
                count = transition_counts.get((from_bin, to_bin), 0)
                rows.append(
                    {
                        "from_label": labels[from_idx],
                        "to_label": labels[to_idx],
                        "from_bin": from_bin,
                        "to_bin": to_bin,
                        "count": count,
                        "fraction_of_from_bin": ratio(count, from_counts.get(from_bin, 0)),
                        "fraction_of_all": ratio(count, len(uids)),
                    }
                )
        write_csv(
            transition_dir / f"{prefix}_{safe_label(labels[from_idx])}_to_{safe_label(labels[to_idx])}.csv",
            [
                "from_label",
                "to_label",
                "from_bin",
                "to_bin",
                "count",
                "fraction_of_from_bin",
                "fraction_of_all",
            ],
            rows,
        )

    metadata = {
        "inputs": [str(path) for path in paths],
        "labels": labels,
        "output_dir": str(output_dir),
        "num_common_uids": len(uids),
        "missing_by_label": missing_by_label,
        "bin_edges": [str(edge) for edge in edges],
        "stable_delta": args.stable_delta,
    }
    with (output_dir / "difficulty_drift_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    if not args.no_plots:
        write_plots(output_dir, labels, bin_labels, bin_count_rows, summary_rows, trajectory_rows)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "num_common_uids": len(uids),
                "files": [
                    "difficulty_drift_uid_trajectories.csv",
                    "difficulty_drift_bin_counts.csv",
                    "difficulty_drift_pool_stability_vs_original.csv",
                    "difficulty_drift_uid_bin_changes_vs_original.csv",
                    "difficulty_drift_overall_summary.csv",
                    "transition_matrices/*.csv",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def write_plots(
    output_dir: Path,
    labels: List[str],
    bin_labels: List[str],
    bin_count_rows: List[dict],
    summary_rows: List[dict],
    trajectory_rows: List[dict],
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="matplotlib-", dir="/tmp"))
    try:
        import matplotlib
    except ModuleNotFoundError:
        print("matplotlib is not installed; skipped plot generation.")
        return

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [row["step"] for row in summary_rows]
    mean_ps = [row["mean_p"] for row in summary_rows]
    plt.figure(figsize=(9, 5))
    plt.plot(steps, mean_ps, marker="o", linewidth=2)
    plt.xlabel("Checkpoint step")
    plt.ylabel("Mean p")
    plt.title("Mean Difficulty p Over Checkpoints")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "difficulty_drift_mean_p_over_steps.png", dpi=180)
    plt.close()

    counts_by_label = defaultdict(dict)
    for row in bin_count_rows:
        counts_by_label[row["label"]][row["bin"]] = row["count"]

    x = list(range(len(bin_labels)))
    plt.figure(figsize=(12, 6))
    for label in labels:
        y = [counts_by_label[label].get(bin_label, 0) for bin_label in bin_labels]
        plt.plot(x, y, marker="o", linewidth=1.8, label=label)
    plt.xticks(x, bin_labels, rotation=35, ha="right")
    plt.xlabel("p bin")
    plt.ylabel("Count")
    plt.title("Difficulty Distribution by p Interval")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "difficulty_drift_bin_distribution_counts.png", dpi=180)
    plt.close()

    deltas = [row["net_delta_original_to_last"] for row in trajectory_rows]
    if deltas:
        plt.figure(figsize=(9, 5))
        plt.hist(deltas, bins=33, edgecolor="white")
        plt.xlabel("p(last) - p(original)")
        plt.ylabel("UID count")
        plt.title("Per-UID Net p Change")
        plt.grid(axis="y", alpha=0.2)
        plt.tight_layout()
        plt.savefig(output_dir / "difficulty_drift_uid_net_delta_hist.png", dpi=180)
        plt.close()


def main():
    args = parse_args()
    if args.inputs:
        paths = [Path(path) for path in args.inputs]
    else:
        paths = sorted(Path(args.input_dir).glob("*.jsonl"), key=file_sort_key)
    if not paths:
        raise ValueError("No input JSONL files found.")

    if args.labels:
        if len(args.labels) != len(paths):
            raise ValueError("--labels must have the same length as --inputs/files.")
        labels = args.labels
    else:
        labels = [infer_label(path) for path in paths]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Labels must be unique. Got: {labels}")

    make_outputs(paths, labels, args)


if __name__ == "__main__":
    main()
