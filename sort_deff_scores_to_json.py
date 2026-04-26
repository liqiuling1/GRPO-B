import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sort DEFF score rows by deff desc, p*(1-p) desc, "
            "then random within exact ties, and save as a JSON array."
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
        help="Output JSON file. Defaults to <input>_deff_sorted.json",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to shuffle exact ties.",
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
    return input_path.with_name(f"{input_path.stem}_deff_sorted.json")


def get_p_times_one_minus_p(row: Dict) -> float:
    if "p_times_one_minus_p" in row:
        return float(row["p_times_one_minus_p"])
    p = float(row.get("p", 0.0))
    return p * (1.0 - p)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = build_output_path(input_path, args.output)

    rows = load_rows(input_path)
    rng = random.Random(args.seed)

    decorated = []
    for row in rows:
        deff = float(row.get("deff", 0.0))
        p_times_one_minus_p = get_p_times_one_minus_p(row)
        decorated.append(((-deff, -p_times_one_minus_p, rng.random()), row))

    decorated.sort(key=lambda item: item[0])

    sorted_rows: List[Dict] = []
    for idx, (_, row) in enumerate(decorated):
        new_row = {"uid2": str(idx)}
        for key, value in row.items():
            new_row[key] = value
        sorted_rows.append(new_row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(sorted_rows, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"Saved sorted JSON to: {output_path}")
    print(f"Rows written: {len(sorted_rows)}")


if __name__ == "__main__":
    main()
