# 这是一个 根据P分成抽样得代码

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stratified sampling by exact p values. "
            "Produces sampled and remaining JSONL datasets while preserving the original p distribution as closely as possible."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file containing a numeric 'p' field.",
    )
    parser.add_argument(
        "--sample_percent",
        type=float,
        required=True,
        help="Target sample size as a percentage of the whole dataset, in [0, 100].",
    )
    parser.add_argument(
        "--sampled_output",
        type=str,
        default="",
        help="Output JSONL path for sampled rows. Defaults to <input>_sampled_<pct>.jsonl",
    )
    parser.add_argument(
        "--remaining_output",
        type=str,
        default="",
        help="Output JSONL path for remaining rows. Defaults to <input>_remaining_<pct>.jsonl",
    )
    parser.add_argument(
        "--sampled_summary",
        type=str,
        default="",
        help="Summary JSON path for sampled rows. Defaults to <sampled_output>_summary.json",
    )
    parser.add_argument(
        "--remaining_summary",
        type=str,
        default="",
        help="Summary JSON path for remaining rows. Defaults to <remaining_output>_summary.json",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for stratified sampling within each p bucket.",
    )
    parser.add_argument(
        "--p_bin_edges",
        type=str,
        default="",
        help=(
            "Optional comma-separated p bin edges for interval-based sampling, "
            "for example '0,0.25,0.5,0.75,1'. "
            "If omitted, the script stratifies by exact p values."
        ),
    )
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
            if "p" not in row:
                raise ValueError(f"Row {line_no} of {path} is missing required field 'p'.")
            try:
                row["p"] = float(row["p"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Row {line_no} of {path} has non-numeric p={row['p']!r}.") from exc
            rows.append(row)
    return rows


def percent_tag(sample_percent: float) -> str:
    text = f"{sample_percent:.6f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def default_output_paths(input_path: Path, sample_percent: float) -> Tuple[Path, Path, Path, Path]:
    tag = percent_tag(sample_percent)
    sampled_output = input_path.with_name(f"{input_path.stem}_sampled_{tag}pct.jsonl")
    remaining_output = input_path.with_name(f"{input_path.stem}_remaining_{tag}pct.jsonl")
    sampled_summary = sampled_output.with_name(f"{sampled_output.stem}_summary.json")
    remaining_summary = remaining_output.with_name(f"{remaining_output.stem}_summary.json")
    return sampled_output, remaining_output, sampled_summary, remaining_summary


def parse_p_bin_edges(text: str) -> List[float]:
    if not text.strip():
        return []

    try:
        edges = [float(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid --p_bin_edges={text!r}. Expected comma-separated floats.") from exc

    if len(edges) < 2:
        raise ValueError("--p_bin_edges must contain at least two values.")
    if any(edge < 0.0 or edge > 1.0 for edge in edges):
        raise ValueError("--p_bin_edges values must be within [0, 1].")
    if any(edges[idx] >= edges[idx + 1] for idx in range(len(edges) - 1)):
        raise ValueError("--p_bin_edges must be strictly increasing.")
    return edges


def format_bin_label(left: float, right: float, is_last: bool) -> str:
    closing = "]" if is_last else ")"
    return f"[{left:g}, {right:g}{closing}"


def bucket_key_for_p(p_value: float, p_bin_edges: List[float]) -> str:
    if not p_bin_edges:
        return f"p={p_value:g}"

    for idx in range(len(p_bin_edges) - 1):
        left = p_bin_edges[idx]
        right = p_bin_edges[idx + 1]
        is_last = idx == len(p_bin_edges) - 2
        if left <= p_value <= right if is_last else left <= p_value < right:
            return format_bin_label(left, right, is_last=is_last)

    raise ValueError(
        f"Found p={p_value} outside the provided --p_bin_edges range "
        f"[{p_bin_edges[0]}, {p_bin_edges[-1]}]."
    )


def group_row_indices(rows: List[Dict], p_bin_edges: List[float]) -> Dict[str, List[int]]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[bucket_key_for_p(row["p"], p_bin_edges)].append(idx)
    return dict(grouped)


def allocate_sample_counts(grouped_indices: Dict[str, List[int]], target_total: int, rng: random.Random) -> Dict[str, int]:
    total_rows = sum(len(indices) for indices in grouped_indices.values())
    if total_rows == 0:
        return {bucket: 0 for bucket in grouped_indices}

    quotas = {}
    allocated = {}
    for bucket_name, indices in grouped_indices.items():
        quota = len(indices) * target_total / total_rows
        quotas[bucket_name] = quota
        allocated[bucket_name] = min(len(indices), math.floor(quota))

    remaining = target_total - sum(allocated.values())
    if remaining <= 0:
        return allocated

    candidates = []
    for bucket_name, indices in grouped_indices.items():
        if allocated[bucket_name] < len(indices):
            candidates.append((quotas[bucket_name] - allocated[bucket_name], rng.random(), bucket_name))
    candidates.sort(key=lambda item: (-item[0], item[1]))

    candidate_idx = 0
    while remaining > 0 and candidate_idx < len(candidates):
        _, _, bucket_name = candidates[candidate_idx]
        if allocated[bucket_name] < len(grouped_indices[bucket_name]):
            allocated[bucket_name] += 1
            remaining -= 1
        candidate_idx += 1

    return allocated


def stratified_sample_indices(
    rows: List[Dict],
    sample_percent: float,
    seed: int,
    p_bin_edges: List[float],
) -> Tuple[set[int], Dict[str, int]]:
    rng = random.Random(seed)
    total_rows = len(rows)
    target_total = round(total_rows * sample_percent / 100.0)
    grouped_indices = group_row_indices(rows, p_bin_edges)
    allocated_counts = allocate_sample_counts(grouped_indices, target_total, rng)

    sampled_indices: set[int] = set()
    for bucket_name, indices in grouped_indices.items():
        shuffled = list(indices)
        rng.shuffle(shuffled)
        sampled_indices.update(shuffled[:allocated_counts[bucket_name]])

    return sampled_indices, allocated_counts


def build_summary(
    rows: List[Dict],
    source_input: Path,
    sample_percent: float,
    seed: int,
    dataset_role: str,
    p_bin_edges: List[float],
) -> Dict:
    total_rows = len(rows)
    grouped_indices = group_row_indices(rows, p_bin_edges)
    exact_p_indices = group_row_indices(rows, [])
    p_distribution = []
    for difficulty_rank, p_value in enumerate(sorted({row["p"] for row in rows}), start=1):
        count = len(exact_p_indices[f"p={p_value:g}"])
        p_distribution.append(
            {
                "difficulty_rank_hard_to_easy": difficulty_rank,
                "p": p_value,
                "count": count,
                "ratio": (count / total_rows) if total_rows else 0.0,
            }
        )

    bucket_distribution = []
    bucket_order = list(grouped_indices.keys())
    if p_bin_edges:
        bucket_order = [
            format_bin_label(
                p_bin_edges[idx],
                p_bin_edges[idx + 1],
                is_last=(idx == len(p_bin_edges) - 2),
            )
            for idx in range(len(p_bin_edges) - 1)
        ]
    else:
        bucket_order = sorted(grouped_indices.keys(), key=lambda item: float(item.split("=", 1)[1]))

    for difficulty_rank, bucket_name in enumerate(bucket_order, start=1):
        count = len(grouped_indices.get(bucket_name, []))
        bucket_distribution.append(
            {
                "difficulty_rank_hard_to_easy": difficulty_rank,
                "bucket": bucket_name,
                "count": count,
                "ratio": (count / total_rows) if total_rows else 0.0,
            }
        )

    return {
        "source_input": str(source_input),
        "dataset_role": dataset_role,
        "sample_percent": sample_percent,
        "seed": seed,
        "num_rows": total_rows,
        "sampling_mode": "p_intervals" if p_bin_edges else "exact_p_values",
        "p_bin_edges": p_bin_edges,
        "num_p_buckets": len(p_distribution),
        "num_sampling_buckets": len(bucket_distribution),
        "difficulty_distribution_by_sampling_bucket": bucket_distribution,
        "difficulty_distribution_by_p": p_distribution,
    }


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.sample_percent <= 100.0:
        raise ValueError("--sample_percent must be within [0, 100].")

    input_path = Path(args.input)
    rows = load_rows(input_path)
    p_bin_edges = parse_p_bin_edges(args.p_bin_edges)

    default_sampled, default_remaining, default_sampled_summary, default_remaining_summary = default_output_paths(
        input_path=input_path,
        sample_percent=args.sample_percent,
    )
    sampled_output = Path(args.sampled_output) if args.sampled_output else default_sampled
    remaining_output = Path(args.remaining_output) if args.remaining_output else default_remaining
    sampled_summary = Path(args.sampled_summary) if args.sampled_summary else default_sampled_summary
    remaining_summary = Path(args.remaining_summary) if args.remaining_summary else default_remaining_summary

    sampled_indices, _ = stratified_sample_indices(
        rows=rows,
        sample_percent=args.sample_percent,
        seed=args.seed,
        p_bin_edges=p_bin_edges,
    )

    sampled_rows = [row for idx, row in enumerate(rows) if idx in sampled_indices]
    remaining_rows = [row for idx, row in enumerate(rows) if idx not in sampled_indices]

    sampled_payload = build_summary(
        rows=sampled_rows,
        source_input=input_path,
        sample_percent=args.sample_percent,
        seed=args.seed,
        dataset_role="sampled",
        p_bin_edges=p_bin_edges,
    )
    remaining_payload = build_summary(
        rows=remaining_rows,
        source_input=input_path,
        sample_percent=args.sample_percent,
        seed=args.seed,
        dataset_role="remaining",
        p_bin_edges=p_bin_edges,
    )

    write_jsonl(sampled_output, sampled_rows)
    write_jsonl(remaining_output, remaining_rows)
    write_json(sampled_summary, sampled_payload)
    write_json(remaining_summary, remaining_payload)

    print(f"Input rows: {len(rows)}")
    print(f"Sampled rows: {len(sampled_rows)}")
    print(f"Remaining rows: {len(remaining_rows)}")
    if p_bin_edges:
        print(f"Sampling mode: p_intervals ({', '.join(str(edge) for edge in p_bin_edges)})")
    else:
        print("Sampling mode: exact_p_values")
    print(f"Saved sampled dataset to: {sampled_output}")
    print(f"Saved remaining dataset to: {remaining_output}")
    print(f"Saved sampled summary to: {sampled_summary}")
    print(f"Saved remaining summary to: {remaining_summary}")


if __name__ == "__main__":
    main()
