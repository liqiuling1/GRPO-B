import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sort DEFF score rows by p desc, deff asc, random within exact ties."
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
        help="Output JSONL file. Defaults to <input>_sorted.jsonl",
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
    return input_path.with_name(f"{input_path.stem}_sorted.jsonl")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = build_output_path(input_path, args.output)

    rows = load_rows(input_path)
    rng = random.Random(args.seed)

    decorated = []
    for row in rows:
        p = float(row.get("p", 0.0))
        deff = float(row.get("deff", 0.0))
        decorated.append(((-p, deff, rng.random()), row))

    decorated.sort(key=lambda item: item[0])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for idx, (_, row) in enumerate(decorated):
            new_row = {"uid1": str(idx)}
            for key, value in row.items():
                new_row[key] = value
            handle.write(json.dumps(new_row, ensure_ascii=False) + "\n")

    print(f"Saved sorted rows to: {output_path}")
    print(f"Rows written: {len(decorated)}")


if __name__ == "__main__":
    main()
