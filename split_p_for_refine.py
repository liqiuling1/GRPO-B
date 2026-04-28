import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split a scored JSONL file into kept p==1 rows and p<1 UID rows."
    )
    parser.add_argument("--scores_file", required=True, help="Input scored JSONL file.")
    parser.add_argument("--keep_scores_out", required=True, help="Rows with p >= threshold.")
    parser.add_argument("--refine_uids_out", required=True, help="UID-only rows with p < threshold.")
    parser.add_argument("--threshold", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.keep_scores_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.refine_uids_out) or ".", exist_ok=True)

    total = 0
    kept = 0
    refine = 0

    with open(args.scores_file, "r", encoding="utf-8") as src, open(
        args.keep_scores_out, "w", encoding="utf-8"
    ) as keep_handle, open(args.refine_uids_out, "w", encoding="utf-8") as uid_handle:
        for line_no, line in enumerate(src, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "uid" not in row:
                raise ValueError(f"Missing uid on line {line_no}: {args.scores_file}")
            if "p" not in row:
                raise ValueError(f"Missing p on line {line_no}: {args.scores_file}")

            total += 1
            if float(row["p"]) >= args.threshold:
                keep_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
            else:
                uid_handle.write(json.dumps({"uid": str(row["uid"])}, ensure_ascii=False) + "\n")
                refine += 1

    print(
        json.dumps(
            {
                "scores_file": args.scores_file,
                "keep_scores_out": args.keep_scores_out,
                "refine_uids_out": args.refine_uids_out,
                "threshold": args.threshold,
                "total": total,
                "kept_scores": kept,
                "refine_uids": refine,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
