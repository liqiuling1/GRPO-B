import argparse
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, Optional, Set, Tuple


def parse_decimal(value, field_name: str) -> Decimal:
    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Invalid {field_name} value: {value!r}") from exc
    if not number.is_finite():
        raise ValueError(f"Invalid {field_name} value: {value!r}")
    return number


def format_decimal(value: Decimal) -> str:
    text = format(value.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def parse_p_range(value: str) -> Tuple[Decimal, Decimal]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError("--p_range must be two comma-separated values, for example 0.375,0.65625")

    left = parse_decimal(parts[0], "p_range left")
    right = parse_decimal(parts[1], "p_range right")
    if left >= right:
        raise ValueError("--p_range left edge must be smaller than right edge")
    return left, right


def format_p_range(p_range: Tuple[Decimal, Decimal], right_closed: bool) -> str:
    left, right = p_range
    closing = "]" if right_closed else ")"
    return f"[{format_decimal(left)}, {format_decimal(right)}{closing}"


def uid_sort_key(uid: str):
    try:
        return (0, int(uid))
    except ValueError:
        return (1, uid)


def is_in_range(p_value: Decimal, p_range: Tuple[Decimal, Decimal], right_closed: bool) -> bool:
    left, right = p_range
    if right_closed:
        return left <= p_value <= right
    return left <= p_value < right


def load_target_uids(
    path: Path,
    uid_field: str,
    p_field: str,
    p_range: Tuple[Decimal, Decimal],
    right_closed: bool,
) -> Tuple[Set[str], int, Dict[str, Decimal]]:
    target_uids: Set[str] = set()
    all_p_by_uid: Dict[str, Decimal] = {}
    total_rows = 0

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            total_rows += 1

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON: {exc}") from exc

            if uid_field not in row:
                raise ValueError(f"{path}:{line_number} missing uid field `{uid_field}`")
            if p_field not in row:
                raise ValueError(f"{path}:{line_number} missing p field `{p_field}`")

            uid = str(row[uid_field])
            if uid in all_p_by_uid:
                raise ValueError(f"{path}:{line_number} duplicate uid: {uid}")

            p_value = parse_decimal(row[p_field], p_field)
            all_p_by_uid[uid] = p_value
            if is_in_range(p_value, p_range, right_closed):
                target_uids.add(uid)

    return target_uids, total_rows, all_p_by_uid


def ratio(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def format_ratio(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{value:.6f}"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare two p-scored jsonl files and summarize target difficulty pool stability "
            "for a configured p interval."
        )
    )
    parser.add_argument("--before", required=True, help="First jsonl file, usually checkpoint 0 or baseline.")
    parser.add_argument("--after", required=True, help="Second jsonl file, usually a later checkpoint.")
    parser.add_argument("--p_range", required=True, help="Target p interval as left,right. Default is [left, right).")
    parser.add_argument(
        "--p_range_right_closed",
        action="store_true",
        help="Use [left, right] for --p_range instead of the default [left, right).",
    )
    parser.add_argument("--uid_field", default="uid", help="UID field used to identify examples.")
    parser.add_argument("--p_field", default="p", help="Difficulty p field name.")
    parser.add_argument(
        "--output_prefix",
        default="outputs/p_pool_stability",
        help="Prefix for summary/metrics outputs.",
    )
    parser.add_argument(
        "--write_uid_lists",
        action="store_true",
        help="Also write retained, inflow, and outflow uid lists.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    before_path = Path(args.before)
    after_path = Path(args.after)
    p_range = parse_p_range(args.p_range)
    p_range_text = format_p_range(p_range, args.p_range_right_closed)

    before_target_uids, before_rows, before_p_by_uid = load_target_uids(
        before_path, args.uid_field, args.p_field, p_range, args.p_range_right_closed
    )
    after_target_uids, after_rows, after_p_by_uid = load_target_uids(
        after_path, args.uid_field, args.p_field, p_range, args.p_range_right_closed
    )

    before_uids = set(before_p_by_uid)
    after_uids = set(after_p_by_uid)
    common_uids = before_uids & after_uids
    only_before_uids = before_uids - after_uids
    only_after_uids = after_uids - before_uids

    retained_uids = before_target_uids & after_target_uids
    outflow_uids = before_target_uids - after_target_uids
    inflow_uids = after_target_uids - before_target_uids
    union_uids = before_target_uids | after_target_uids

    before_target_count = len(before_target_uids)
    after_target_count = len(after_target_uids)
    retained_count = len(retained_uids)
    outflow_count = len(outflow_uids)
    inflow_count = len(inflow_uids)
    union_count = len(union_uids)

    retention_ratio = ratio(retained_count, before_target_count)
    outflow_ratio = ratio(outflow_count, before_target_count)
    inflow_ratio = ratio(inflow_count, after_target_count)
    jaccard_overlap = ratio(retained_count, union_count)

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_json = output_prefix.with_name(output_prefix.name + "_summary.json")
    metrics_tsv = output_prefix.with_name(output_prefix.name + "_metrics.tsv")

    summary = {
        "before": str(before_path),
        "after": str(after_path),
        "p_range": [format_decimal(value) for value in p_range],
        "p_range_right_closed": args.p_range_right_closed,
        "target_interval": p_range_text,
        "before_rows": before_rows,
        "after_rows": after_rows,
        "common_uids": len(common_uids),
        "only_before_uids": len(only_before_uids),
        "only_after_uids": len(only_after_uids),
        "before_target_count": before_target_count,
        "after_target_count": after_target_count,
        "retained_count": retained_count,
        "outflow_count": outflow_count,
        "inflow_count": inflow_count,
        "union_count": union_count,
        "retention_ratio": retention_ratio,
        "outflow_ratio_of_before_target": outflow_ratio,
        "inflow_ratio_of_after_target": inflow_ratio,
        "jaccard_overlap": jaccard_overlap,
    }

    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    metric_fields = [
        "before",
        "after",
        "target_interval",
        "before_rows",
        "after_rows",
        "common_uids",
        "only_before_uids",
        "only_after_uids",
        "before_target_count",
        "after_target_count",
        "retained_count",
        "outflow_count",
        "inflow_count",
        "union_count",
        "retention_ratio",
        "outflow_ratio_of_before_target",
        "inflow_ratio_of_after_target",
        "jaccard_overlap",
    ]
    with metrics_tsv.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(metric_fields) + "\n")
        handle.write(
            "\t".join(
                [
                    str(before_path),
                    str(after_path),
                    p_range_text,
                    str(before_rows),
                    str(after_rows),
                    str(len(common_uids)),
                    str(len(only_before_uids)),
                    str(len(only_after_uids)),
                    str(before_target_count),
                    str(after_target_count),
                    str(retained_count),
                    str(outflow_count),
                    str(inflow_count),
                    str(union_count),
                    format_ratio(retention_ratio),
                    format_ratio(outflow_ratio),
                    format_ratio(inflow_ratio),
                    format_ratio(jaccard_overlap),
                ]
            )
            + "\n"
        )

    if args.write_uid_lists:
        uid_json = output_prefix.with_name(output_prefix.name + "_uid_lists.json")
        with uid_json.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "retained_uids": sorted(retained_uids, key=uid_sort_key),
                    "outflow_uids": sorted(outflow_uids, key=uid_sort_key),
                    "inflow_uids": sorted(inflow_uids, key=uid_sort_key),
                    "before_target_uids": sorted(before_target_uids, key=uid_sort_key),
                    "after_target_uids": sorted(after_target_uids, key=uid_sort_key),
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
            handle.write("\n")

    print("==== P Pool Stability Summary ====")
    print(f"Before: {before_path}")
    print(f"After: {after_path}")
    print(f"Target interval: {p_range_text}")
    print(f"Before rows: {before_rows}")
    print(f"After rows: {after_rows}")
    print(f"Common uid: {len(common_uids)}")
    print(f"Only before uid: {len(only_before_uids)}")
    print(f"Only after uid: {len(only_after_uids)}")
    print()
    print(f"|S0| before target count: {before_target_count}")
    print(f"|St| after target count: {after_target_count}")
    print(f"|S0 & St| retained count: {retained_count}")
    print(f"|S0 - St| outflow count: {outflow_count}")
    print(f"|St - S0| inflow count: {inflow_count}")
    print(f"|S0 union St| union count: {union_count}")
    print(f"Retention ratio: {format_ratio(retention_ratio)}")
    print(f"Outflow ratio of before target: {format_ratio(outflow_ratio)}")
    print(f"Inflow ratio of after target: {format_ratio(inflow_ratio)}")
    print(f"Jaccard overlap: {format_ratio(jaccard_overlap)}")
    print()
    print(f"Wrote: {summary_json}")
    print(f"Wrote: {metrics_tsv}")
    if args.write_uid_lists:
        print(f"Wrote: {output_prefix.with_name(output_prefix.name + '_uid_lists.json')}")


if __name__ == "__main__":
    main()
