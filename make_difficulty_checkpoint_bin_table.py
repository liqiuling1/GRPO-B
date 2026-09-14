import argparse
import csv
import json
import re
from collections import OrderedDict
from decimal import Decimal
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)")

BIN_COLUMNS: List[str] = [
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
        description=(
            "Build a wide checkpoint-by-difficulty-bin count table from GSM8K p-score JSONL files."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("outputs/difficulty_drift_eval_pdist20_seed42"),
        help="Directory containing original and checkpoint p-score JSONL files.",
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=None,
        help="Optional explicit JSONL file list. If omitted, all *.jsonl files in --input_dir are used.",
    )
    parser.add_argument(
        "--output_prefix",
        type=Path,
        default=None,
        help=(
            "Output prefix without extension. Defaults to "
            "<input_dir>/drift_analysis/difficulty_checkpoint_bin_table"
        ),
    )
    parser.add_argument("--p_field", default="p")
    parser.add_argument("--uid_field", default="uid")
    parser.add_argument(
        "--include_total",
        action="store_true",
        help="Append total and out_of_range columns to the CSV/Markdown table.",
    )
    return parser.parse_args()


def decimal_p(value) -> Decimal:
    return Decimal(str(value))


def bin_for_p(value) -> str:
    p = decimal_p(value)
    if p == Decimal("0"):
        return "P=0"
    if Decimal("0") < p < Decimal("0.125"):
        return "(0,0.125)"
    if Decimal("0.125") <= p < Decimal("0.25"):
        return "[0.125,0.25)"
    if Decimal("0.25") <= p < Decimal("0.375"):
        return "[0.25,0.375)"
    if Decimal("0.375") <= p < Decimal("0.5"):
        return "[0.375,0.5)"
    if Decimal("0.5") <= p < Decimal("0.625"):
        return "[0.5,0.625)"
    if Decimal("0.625") <= p < Decimal("0.75"):
        return "[0.625,0.75)"
    if Decimal("0.75") <= p < Decimal("0.875"):
        return "[0.75,0.875)"
    if Decimal("0.875") <= p < Decimal("1"):
        return "[0.875,1)"
    if p == Decimal("1"):
        return "P=1"
    return "out_of_range"


def infer_checkpoint(path: Path) -> int:
    match = CHECKPOINT_RE.search(path.name)
    if match:
        return int(match.group(1))
    return 0


def file_sort_key(path: Path) -> Tuple[int, str]:
    return (infer_checkpoint(path), path.name)


def discover_inputs(input_dir: Path) -> List[Path]:
    paths = sorted(input_dir.glob("*.jsonl"), key=file_sort_key)
    if not paths:
        raise FileNotFoundError(f"No *.jsonl files found in {input_dir}")
    return paths


def count_bins(path: Path, p_field: str, uid_field: str) -> Dict[str, int]:
    counts: Dict[str, int] = OrderedDict((column, 0) for column in BIN_COLUMNS)
    counts["out_of_range"] = 0
    seen_uids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if uid_field not in row:
                raise ValueError(f"Missing {uid_field!r} in {path}:{line_no}")
            if p_field not in row:
                raise ValueError(f"Missing {p_field!r} in {path}:{line_no}")
            uid = str(row[uid_field])
            if uid in seen_uids:
                raise ValueError(f"Duplicate uid={uid} in {path}:{line_no}")
            seen_uids.add(uid)
            counts[bin_for_p(row[p_field])] += 1
    counts["total"] = len(seen_uids)
    return counts


def build_rows(paths: Iterable[Path], p_field: str, uid_field: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in paths:
        checkpoint = infer_checkpoint(path)
        counts = count_bins(path, p_field=p_field, uid_field=uid_field)
        row: Dict[str, object] = OrderedDict()
        row["checkpoint"] = checkpoint
        for column in BIN_COLUMNS:
            row[column] = counts[column]
        row["total"] = counts["total"]
        row["out_of_range"] = counts["out_of_range"]
        row["source_file"] = str(path)
        rows.append(row)
    return sorted(rows, key=lambda row: (int(row["checkpoint"]), str(row["source_file"])))


def write_csv(path: Path, rows: List[Dict[str, object]], include_total: bool) -> None:
    columns = ["checkpoint", *BIN_COLUMNS]
    if include_total:
        columns.extend(["total", "out_of_range"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in columns})


def write_markdown(path: Path, rows: List[Dict[str, object]], include_total: bool) -> None:
    columns = ["checkpoint", *BIN_COLUMNS]
    if include_total:
        columns.extend(["total", "out_of_range"])
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_xlsx_if_available(path: Path, rows: List[Dict[str, object]], include_total: bool) -> bool:
    try:
        import pandas as pd
    except ImportError:
        return False

    columns = ["checkpoint", *BIN_COLUMNS]
    if include_total:
        columns.extend(["total", "out_of_range"])
    df = pd.DataFrame([{column: row[column] for column in columns} for row in rows])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=False)
    return True


def main():
    args = parse_args()
    paths = [Path(path) for path in args.inputs] if args.inputs else discover_inputs(args.input_dir)
    output_prefix = args.output_prefix or (
        args.input_dir / "drift_analysis" / "difficulty_checkpoint_bin_table"
    )
    rows = build_rows(paths, p_field=args.p_field, uid_field=args.uid_field)

    csv_path = output_prefix.with_suffix(".csv")
    md_path = output_prefix.with_suffix(".md")
    xlsx_path = output_prefix.with_suffix(".xlsx")
    meta_path = output_prefix.with_suffix(".metadata.json")

    write_csv(csv_path, rows, include_total=args.include_total)
    write_markdown(md_path, rows, include_total=args.include_total)
    wrote_xlsx = write_xlsx_if_available(xlsx_path, rows, include_total=args.include_total)

    metadata = {
        "inputs": [str(path) for path in paths],
        "output_csv": str(csv_path),
        "output_markdown": str(md_path),
        "output_xlsx": str(xlsx_path) if wrote_xlsx else None,
        "bin_columns": BIN_COLUMNS,
        "include_total": args.include_total,
        "num_rows": len(rows),
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote Markdown: {md_path}")
    if wrote_xlsx:
        print(f"Wrote Excel: {xlsx_path}")
    else:
        print("Skipped Excel: pandas/openpyxl is not available")
    print(f"Wrote metadata: {meta_path}")


if __name__ == "__main__":
    main()
