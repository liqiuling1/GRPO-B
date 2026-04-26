import argparse
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sort score rows by p*(1-p) descending, add uid1 as the new rank index, "
            "and keep the original uid as the second field."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output JSONL file. Defaults to <input>_pvar_sorted.jsonl",
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
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
    return rows


def build_output_path(input_path: Path, output_arg: str) -> Path:
    if output_arg:
        return Path(output_arg)
    return input_path.with_name(f"{input_path.stem}_pvar_sorted.jsonl")


def get_p_times_one_minus_p(row: Dict) -> float:
    if "p_times_one_minus_p" in row:
        return float(row["p_times_one_minus_p"])
    p = float(row.get("p", 0.0))
    return p * (1.0 - p)


def get_zero_p_priority(row: Dict, p_times_one_minus_p: float) -> int:
    if p_times_one_minus_p != 0.0:
        return 0

    p = float(row.get("p", 0.0))
    if p == 0.0:
        return 0
    if p == 1.0:
        return 1
    return 2


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = build_output_path(input_path, args.output)

    rows = load_rows(input_path)
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            -get_p_times_one_minus_p(row),
            get_zero_p_priority(row, get_p_times_one_minus_p(row)),
        ),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(sorted_rows):
            new_row = {"uid1": str(idx)}
            for key, value in row.items():
                new_row[key] = value
            handle.write(json.dumps(new_row, ensure_ascii=False) + "\n")

    print(f"Saved sorted rows to: {output_path}")
    print(f"Rows written: {len(sorted_rows)}")


if __name__ == "__main__":
    main()
