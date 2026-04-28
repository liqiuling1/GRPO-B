import argparse
import json
import os
from typing import Dict


def uid_sort_key(uid: str):
    try:
        return (0, int(uid))
    except ValueError:
        return (1, uid)


def parse_args():
    parser = argparse.ArgumentParser(description="Merge scored JSONL files by uid.")
    parser.add_argument("--score_files", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected_count", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    rows_by_uid: Dict[str, dict] = {}

    for path in args.score_files:
        with open(path, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "uid" not in row:
                    raise ValueError(f"Missing uid on line {line_no}: {path}")
                if "p" not in row:
                    raise ValueError(f"Missing p on line {line_no}: {path}")

                uid = str(row["uid"])
                if uid in rows_by_uid:
                    raise ValueError(f"Duplicate uid={uid}; first seen earlier, repeated in {path}:{line_no}")
                rows_by_uid[uid] = row

    if args.expected_count > 0 and len(rows_by_uid) != args.expected_count:
        raise ValueError(
            f"Merged {len(rows_by_uid)} rows, expected {args.expected_count}. "
            "A later truncation UID file may still be non-empty, or an input score file is missing."
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as out_handle:
        for uid in sorted(rows_by_uid, key=uid_sort_key):
            out_handle.write(json.dumps(rows_by_uid[uid], ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "out": args.out,
                "score_files": args.score_files,
                "merged_rows": len(rows_by_uid),
                "expected_count": args.expected_count or None,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
