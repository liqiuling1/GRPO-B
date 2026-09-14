import argparse
import json
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Counter as CounterType, Dict, List, Optional, Tuple


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


def parse_observe_range(value: str) -> Tuple[Decimal, Decimal]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError("--observe_p_range must be two comma-separated values, for example 0.375,0.65625")
    left = parse_decimal(parts[0], "observe_p_range left")
    right = parse_decimal(parts[1], "observe_p_range right")
    if left >= right:
        raise ValueError("--observe_p_range left edge must be smaller than right edge")
    return left, right


def observe_label(
    observe_p: Optional[Decimal],
    observe_range: Optional[Tuple[Decimal, Decimal]],
    range_right_closed: bool,
) -> str:
    if observe_range is None:
        assert observe_p is not None
        return format_decimal(observe_p)

    left, right = observe_range
    closing = "]" if range_right_closed else ")"
    return f"[{format_decimal(left)}, {format_decimal(right)}{closing}"


def observe_file_label(
    observe_p: Optional[Decimal],
    observe_range: Optional[Tuple[Decimal, Decimal]],
    range_right_closed: bool,
) -> str:
    if observe_range is None:
        assert observe_p is not None
        return format_decimal(observe_p)

    left, right = observe_range
    closing = "closed" if range_right_closed else "open"
    return f"{format_decimal(left)}_to_{format_decimal(right)}_{closing}"


def p_matches_observe(
    p_value: Decimal,
    observe_p: Optional[Decimal],
    observe_range: Optional[Tuple[Decimal, Decimal]],
    range_right_closed: bool,
) -> bool:
    if observe_range is None:
        return p_value == observe_p

    left, right = observe_range
    if range_right_closed:
        return left <= p_value <= right
    return left <= p_value < right


def load_jsonl_by_uid(path: Path, uid_field: str, p_field: str) -> Tuple[Dict[str, dict], CounterType[Decimal]]:
    rows_by_uid: Dict[str, dict] = {}
    p_counts: CounterType[Decimal] = Counter()

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON: {exc}") from exc

            if uid_field not in row:
                raise ValueError(f"{path}:{line_number} missing uid field `{uid_field}`")
            if p_field not in row:
                raise ValueError(f"{path}:{line_number} missing p field `{p_field}`")

            uid = str(row[uid_field])
            if uid in rows_by_uid:
                raise ValueError(f"{path}:{line_number} duplicate uid: {uid}")

            p_value = parse_decimal(row[p_field], p_field)
            row["_parsed_p"] = p_value
            rows_by_uid[uid] = row
            p_counts[p_value] += 1

    return rows_by_uid, p_counts


def sorted_counter_items(counter: CounterType[Decimal]) -> List[Tuple[Decimal, int]]:
    return [(key, counter[key]) for key in sorted(counter)]


def write_counter_table(path: Path, header: List[str], rows: List[List[str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(row) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare two jsonl score files by uid and summarize p difficulty transitions "
            "for one observed p value or p range."
        )
    )
    parser.add_argument("--before", required=True, help="First jsonl file, before training/evaluation.")
    parser.add_argument("--after", required=True, help="Second jsonl file, after training/evaluation.")
    parser.add_argument(
        "--observe_p",
        default=None,
        help="Exact difficulty p value to observe, for example 0.5. Mutually exclusive with --observe_p_range.",
    )
    parser.add_argument(
        "--observe_p_range",
        default=None,
        help="Difficulty p interval to observe as left,right. Default interval is [left, right).",
    )
    parser.add_argument(
        "--observe_p_range_right_closed",
        action="store_true",
        help="Use [left, right] for --observe_p_range instead of the default [left, right).",
    )
    parser.add_argument("--uid_field", default="uid", help="UID field used to align rows.")
    parser.add_argument("--p_field", default="p", help="Difficulty field name.")
    parser.add_argument(
        "--output_prefix",
        default="outputs/p_transition_observe",
        help="Prefix for optional TSV/JSON outputs.",
    )
    parser.add_argument(
        "--write_uid_lists",
        action="store_true",
        help="Also write uid lists for each transition involving observe_p.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    before_path = Path(args.before)
    after_path = Path(args.after)
    if bool(args.observe_p) == bool(args.observe_p_range):
        raise SystemExit("Set exactly one of --observe_p or --observe_p_range.")

    observe_p = parse_decimal(args.observe_p, "observe_p") if args.observe_p else None
    observe_range = parse_observe_range(args.observe_p_range) if args.observe_p_range else None
    observe_text = observe_label(observe_p, observe_range, args.observe_p_range_right_closed)
    observe_file_text = observe_file_label(observe_p, observe_range, args.observe_p_range_right_closed)

    before_rows, before_p_counts = load_jsonl_by_uid(before_path, args.uid_field, args.p_field)
    after_rows, after_p_counts = load_jsonl_by_uid(after_path, args.uid_field, args.p_field)
    del before_p_counts, after_p_counts

    before_uids = set(before_rows)
    after_uids = set(after_rows)
    common_uids = before_uids & after_uids
    only_before = before_uids - after_uids
    only_after = after_uids - before_uids

    before_observe_to_after: CounterType[Decimal] = Counter()
    after_observe_from_before: CounterType[Decimal] = Counter()
    transition_counts: CounterType[Tuple[Decimal, Decimal]] = Counter()
    before_observe_uid_lists: Dict[Decimal, List[str]] = {}
    after_observe_uid_lists: Dict[Decimal, List[str]] = {}
    before_observe_count = 0
    after_observe_count = 0

    for row in before_rows.values():
        if p_matches_observe(
            row["_parsed_p"], observe_p, observe_range, args.observe_p_range_right_closed
        ):
            before_observe_count += 1
    for row in after_rows.values():
        if p_matches_observe(
            row["_parsed_p"], observe_p, observe_range, args.observe_p_range_right_closed
        ):
            after_observe_count += 1

    for uid in common_uids:
        before_p = before_rows[uid]["_parsed_p"]
        after_p = after_rows[uid]["_parsed_p"]
        transition_counts[(before_p, after_p)] += 1

        if p_matches_observe(before_p, observe_p, observe_range, args.observe_p_range_right_closed):
            before_observe_to_after[after_p] += 1
            before_observe_uid_lists.setdefault(after_p, []).append(uid)
        if p_matches_observe(after_p, observe_p, observe_range, args.observe_p_range_right_closed):
            after_observe_from_before[before_p] += 1
            after_observe_uid_lists.setdefault(before_p, []).append(uid)

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    before_to_after_tsv = output_prefix.with_name(output_prefix.name + "_before_observe_to_after.tsv")
    after_from_before_tsv = output_prefix.with_name(output_prefix.name + "_after_observe_from_before.tsv")
    summary_json = output_prefix.with_name(output_prefix.name + "_summary.json")

    before_to_after_rows = [
        [
            format_decimal(after_p),
            str(count),
            f"{count / before_observe_count * 100:.4f}%" if before_observe_count else "0.0000%",
        ]
        for after_p, count in sorted_counter_items(before_observe_to_after)
    ]
    write_counter_table(
        before_to_after_tsv,
        [f"after_{args.p_field}", "count", f"percent_of_before_observe_{observe_file_text}"],
        before_to_after_rows,
    )

    after_from_before_rows = [
        [
            format_decimal(before_p),
            str(count),
            f"{count / after_observe_count * 100:.4f}%" if after_observe_count else "0.0000%",
        ]
        for before_p, count in sorted_counter_items(after_observe_from_before)
    ]
    write_counter_table(
        after_from_before_tsv,
        [f"before_{args.p_field}", "count", f"percent_of_after_observe_{observe_file_text}"],
        after_from_before_rows,
    )

    summary = {
        "before": str(before_path),
        "after": str(after_path),
        "observe_p": format_decimal(observe_p) if observe_p is not None else None,
        "observe_p_range": [format_decimal(value) for value in observe_range] if observe_range else None,
        "observe_p_range_right_closed": args.observe_p_range_right_closed if observe_range else None,
        "observe": observe_text,
        "before_rows": len(before_rows),
        "after_rows": len(after_rows),
        "common_uids": len(common_uids),
        "only_before": len(only_before),
        "only_after": len(only_after),
        "before_observe_count": before_observe_count,
        "after_observe_count": after_observe_count,
        "before_observe_to_after": {
            format_decimal(key): value for key, value in sorted_counter_items(before_observe_to_after)
        },
        "after_observe_from_before": {
            format_decimal(key): value for key, value in sorted_counter_items(after_observe_from_before)
        },
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    if args.write_uid_lists:
        uid_json = output_prefix.with_name(output_prefix.name + "_uid_lists.json")
        with uid_json.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "before_observe_to_after": {
                        format_decimal(key): sorted(values) for key, values in before_observe_uid_lists.items()
                    },
                    "after_observe_from_before": {
                        format_decimal(key): sorted(values) for key, values in after_observe_uid_lists.items()
                    },
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
            f.write("\n")

    print("==== P Transition Summary ====")
    print(f"Before: {before_path}")
    print(f"After: {after_path}")
    print(f"Observe p selector: {observe_text}")
    print(f"Before rows: {len(before_rows)}")
    print(f"After rows: {len(after_rows)}")
    print(f"Common uid: {len(common_uids)}")
    print(f"Only before uid: {len(only_before)}")
    print(f"Only after uid: {len(only_after)}")
    print()

    print(f"1) Before p in {observe_text} -> After p distribution")
    print(f"Before observe count: {before_observe_count}")
    print("after_p\tcount\tpercent")
    for row in before_to_after_rows:
        print("\t".join(row))
    print()

    print(f"2) After p in {observe_text} <- Before p distribution")
    print(f"After observe count: {after_observe_count}")
    print("before_p\tcount\tpercent")
    for row in after_from_before_rows:
        print("\t".join(row))
    print()

    print(f"Wrote: {before_to_after_tsv}")
    print(f"Wrote: {after_from_before_tsv}")
    print(f"Wrote: {summary_json}")
    if args.write_uid_lists:
        print(f"Wrote: {output_prefix.with_name(output_prefix.name + '_uid_lists.json')}")


if __name__ == "__main__":
    main()
