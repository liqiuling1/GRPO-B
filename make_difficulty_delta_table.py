import argparse
import csv
import json
import re
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Tuple


CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)")
DEFAULT_INPUT_DIR = Path(
    "outputs/observation_sets_gsm8k_p_intervals_seed42/"
    "difficulty_drift_eval_pdist20_seed42"
)
DEFAULT_ORIGINAL_NAME = "gsm8k_p_scores_final_no_truncation_2_sampled_20pct.jsonl"
THRESHOLDS = [0.125, 0.25, 0.375, 0.5]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute checkpoint-level p-score drift statistics against the original p-score file."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing the original JSONL and checkpoint JSONL files.",
    )
    parser.add_argument(
        "--original_file",
        type=Path,
        default=None,
        help=(
            "Original p-score JSONL. Defaults to "
            "<input_dir>/gsm8k_p_scores_final_no_truncation_2_sampled_20pct.jsonl."
        ),
    )
    parser.add_argument(
        "--checkpoint_files",
        nargs="*",
        default=None,
        help="Optional explicit checkpoint JSONL files. Defaults to *checkpoint-*.jsonl in --input_dir.",
    )
    parser.add_argument(
        "--output_prefix",
        type=Path,
        default=None,
        help=(
            "Output prefix without extension. Defaults to "
            "<input_dir>/drift_analysis/difficulty_checkpoint_delta_table."
        ),
    )
    parser.add_argument("--p_field", default="p")
    parser.add_argument("--uid_field", default="uid")
    parser.add_argument(
        "--no_checkpoint0",
        action="store_true",
        help="Do not include the original file as checkpoint 0.",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=6,
        help="Decimal digits for mean/median values in CSV/Markdown.",
    )
    parser.add_argument(
        "--percent_digits",
        type=int,
        default=2,
        help="Decimal digits for percentage columns in Markdown.",
    )
    return parser.parse_args()


def infer_checkpoint(path: Path) -> int:
    match = CHECKPOINT_RE.search(path.name)
    if match:
        return int(match.group(1))
    return 0


def checkpoint_sort_key(path: Path) -> Tuple[int, str]:
    return (infer_checkpoint(path), path.name)


def load_p_by_uid(path: Path, p_field: str, uid_field: str) -> Dict[str, float]:
    values: Dict[str, float] = {}
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
            if uid in values:
                raise ValueError(f"Duplicate uid={uid} in {path}:{line_no}")
            values[uid] = float(row[p_field])
    return values


def discover_checkpoint_files(input_dir: Path, original_file: Path) -> List[Path]:
    paths = sorted(input_dir.glob("*checkpoint-*.jsonl"), key=checkpoint_sort_key)
    paths = [path for path in paths if path.resolve() != original_file.resolve()]
    if not paths:
        raise FileNotFoundError(f"No *checkpoint-*.jsonl files found in {input_dir}")
    return paths


def format_float(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".rstrip("0").rstrip(".") if digits > 0 else str(round(value))


def format_percent(value: float, digits: int) -> str:
    return f"{value * 100:.{digits}f}%"


def summarize_delta(
    checkpoint: int,
    current: Dict[str, float],
    original: Dict[str, float],
) -> Dict[str, object]:
    current_uids = set(current)
    original_uids = set(original)
    if current_uids != original_uids:
        missing_from_current = sorted(original_uids - current_uids)[:5]
        missing_from_original = sorted(current_uids - original_uids)[:5]
        raise ValueError(
            "UID mismatch for checkpoint "
            f"{checkpoint}: missing_from_current={missing_from_current}, "
            f"missing_from_original={missing_from_original}"
        )

    deltas = [current[uid] - original[uid] for uid in sorted(original)]
    abs_deltas = [abs(value) for value in deltas]
    n = len(deltas)
    if n == 0:
        raise ValueError(f"No rows for checkpoint {checkpoint}")

    row: Dict[str, object] = {
        "Checkpoint": checkpoint,
        "Mean ΔP": sum(deltas) / n,
        "Mean |ΔP|": sum(abs_deltas) / n,
        "Median |ΔP|": median(abs_deltas),
        "N": n,
    }
    for threshold in THRESHOLDS:
        row[f"≥{threshold}"] = sum(value >= threshold for value in abs_deltas) / n
    return row


def build_rows(
    original_file: Path,
    checkpoint_files: Iterable[Path],
    p_field: str,
    uid_field: str,
    include_checkpoint0: bool,
) -> List[Dict[str, object]]:
    original = load_p_by_uid(original_file, p_field=p_field, uid_field=uid_field)
    rows: List[Dict[str, object]] = []
    if include_checkpoint0:
        rows.append(summarize_delta(0, original, original))

    for path in sorted(checkpoint_files, key=checkpoint_sort_key):
        checkpoint = infer_checkpoint(path)
        current = load_p_by_uid(path, p_field=p_field, uid_field=uid_field)
        rows.append(summarize_delta(checkpoint, current, original))

    return sorted(rows, key=lambda row: int(row["Checkpoint"]))


def write_csv(path: Path, rows: List[Dict[str, object]], digits: int) -> None:
    columns = [
        "Checkpoint",
        "Mean ΔP",
        "Mean |ΔP|",
        "Median |ΔP|",
        "≥0.125",
        "≥0.25",
        "≥0.375",
        "≥0.5",
        "N",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            for column in columns:
                if column not in {"Checkpoint", "N"}:
                    formatted[column] = format_float(float(row[column]), digits)
            writer.writerow({column: formatted[column] for column in columns})


def write_markdown(
    path: Path,
    rows: List[Dict[str, object]],
    digits: int,
    percent_digits: int,
) -> None:
    columns = [
        "Checkpoint",
        "Mean ΔP",
        "Mean |ΔP|",
        "Median |ΔP|",
        "≥0.125",
        "≥0.25",
        "≥0.375",
        "≥0.5",
        "N",
    ]
    display_columns = [column.replace("|", r"\|") for column in columns]
    lines = [
        "| " + " | ".join(display_columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            if column in {"Checkpoint", "N"}:
                values.append(str(row[column]))
            elif column.startswith("≥"):
                values.append(format_percent(float(row[column]), percent_digits))
            else:
                values.append(format_float(float(row[column]), digits))
        lines.append("| " + " | ".join(values) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metadata(
    path: Path,
    rows: List[Dict[str, object]],
    original_file: Path,
    checkpoint_files: Iterable[Path],
) -> None:
    metadata = {
        "definition": "ΔP = p_checkpoint - p_original, thresholds use |ΔP|.",
        "original_file": str(original_file),
        "checkpoint_files": [str(path) for path in checkpoint_files],
        "thresholds": THRESHOLDS,
        "num_rows": len(rows),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    original_file = args.original_file or (args.input_dir / DEFAULT_ORIGINAL_NAME)
    if not original_file.is_file():
        raise FileNotFoundError(f"Original file not found: {original_file}")

    checkpoint_files = (
        [Path(path) for path in args.checkpoint_files]
        if args.checkpoint_files
        else discover_checkpoint_files(args.input_dir, original_file)
    )
    output_prefix = args.output_prefix or (
        args.input_dir / "drift_analysis" / "difficulty_checkpoint_delta_table"
    )

    rows = build_rows(
        original_file=original_file,
        checkpoint_files=checkpoint_files,
        p_field=args.p_field,
        uid_field=args.uid_field,
        include_checkpoint0=not args.no_checkpoint0,
    )

    csv_path = output_prefix.with_suffix(".csv")
    md_path = output_prefix.with_suffix(".md")
    meta_path = output_prefix.with_suffix(".metadata.json")

    write_csv(csv_path, rows, digits=args.digits)
    write_markdown(
        md_path,
        rows,
        digits=args.digits,
        percent_digits=args.percent_digits,
    )
    write_metadata(meta_path, rows, original_file, checkpoint_files)

    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote Markdown: {md_path}")
    print(f"Wrote metadata: {meta_path}")


if __name__ == "__main__":
    main()
