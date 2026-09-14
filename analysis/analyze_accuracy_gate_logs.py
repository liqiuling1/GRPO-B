import ast
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


LOG_RE = re.compile(r"train_accuracy_gate_0\.5_(\d+)-(\d+)\.log$")
DICT_RE = re.compile(r"\{.*\}")
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis"


def as_float(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def log_sort_key(path):
    match = LOG_RE.search(path.name)
    if not match:
        return (10**9, 10**9, path.name)
    return (int(match.group(1)), int(match.group(2)), path.name)


def iter_rows(path):
    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        match = DICT_RE.search(line)
        if not match or "accuracy_gate/" not in match.group(0):
            continue
        try:
            row = ast.literal_eval(match.group(0))
        except (SyntaxError, ValueError):
            continue
        yield line_no, row


def main():
    logs = sorted(ROOT.glob("train_accuracy_gate_0.5_*.log"), key=log_sort_key)
    records = []
    per_file = []

    for path in logs:
        file_records = []
        for line_no, row in iter_rows(path):
            group_count = as_float(row, "accuracy_gate/group_count")
            accepted = as_float(row, "accuracy_gate/accepted_groups")
            rejected = group_count - accepted
            record = {
                "file": path.name,
                "line": line_no,
                "accepted": accepted,
                "rejected": rejected,
                "group_count": group_count,
                "accepted_update_steps": as_float(row, "accuracy_gate/accepted_update_steps"),
                "rejected_update_steps": as_float(row, "accuracy_gate/rejected_update_steps"),
                "correct_rate": as_float(row, "accuracy_gate/correct_rate_mean"),
                "loss": as_float(row, "loss"),
                "grad_norm": as_float(row, "grad_norm"),
                "update_proxy": as_float(row, "update_proxy"),
                "learning_rate": as_float(row, "learning_rate"),
                "epoch": as_float(row, "epoch"),
            }
            file_records.append(record)
            records.append(record)

        total = sum(r["group_count"] for r in file_records)
        accepted = sum(r["accepted"] for r in file_records)
        rejected = sum(r["rejected"] for r in file_records)
        per_file.append(
            {
                "file": path.name,
                "rows": len(file_records),
                "total_groups": int(total),
                "accepted_groups": int(accepted),
                "rejected_groups": int(rejected),
                "accepted_rate": accepted / total if total else 0.0,
                "groups_per_accepted": total / accepted if accepted else 0.0,
                "start_accepted_step": int(file_records[0]["accepted_update_steps"]) if file_records else None,
                "end_accepted_step": int(file_records[-1]["accepted_update_steps"]) if file_records else None,
            }
        )

    with (OUT_DIR / "accuracy_gate_0.5_file_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_file[0].keys()))
        writer.writeheader()
        writer.writerows(per_file)

    indexed_records = []
    for idx, record in enumerate(records, 1):
        indexed = {"attempt_index": idx, **record}
        indexed_records.append(indexed)

    with (OUT_DIR / "accuracy_gate_0.5_records.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(indexed_records[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(indexed_records)

    window = 200
    rolling_x = []
    rolling_rate = []
    rolling_per_one = []
    for end in range(window, len(indexed_records) + 1):
        chunk = indexed_records[end - window : end]
        total = sum(r["group_count"] for r in chunk)
        accepted = sum(r["accepted"] for r in chunk)
        rolling_x.append(chunk[-1]["attempt_index"])
        rolling_rate.append(accepted / total if total else 0.0)
        rolling_per_one.append(total / accepted if accepted else float("nan"))

    bucket_size = 50
    buckets = defaultdict(lambda: {"total": 0.0, "accepted": 0.0})
    for record in indexed_records:
        bucket = int(record["accepted_update_steps"] // bucket_size) * bucket_size
        buckets[bucket]["total"] += record["group_count"]
        buckets[bucket]["accepted"] += record["accepted"]

    bucket_rows = []
    for bucket in sorted(buckets):
        total = buckets[bucket]["total"]
        accepted = buckets[bucket]["accepted"]
        bucket_rows.append(
            {
                "bucket": f"{bucket}-{bucket + bucket_size}",
                "total_groups": int(total),
                "accepted_groups": int(accepted),
                "rejected_groups": int(total - accepted),
                "accepted_rate": accepted / total if total else 0.0,
                "groups_per_accepted": total / accepted if accepted else 0.0,
            }
        )

    with (OUT_DIR / "accuracy_gate_0.5_bucket_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bucket_rows[0].keys()))
        writer.writeheader()
        writer.writerows(bucket_rows)

    suspicious = [
        r
        for r in indexed_records
        if r["accepted"] == 0 and (abs(r["loss"]) > 1e-12 or r["grad_norm"] > 1e-12 or r["update_proxy"] > 1e-12)
    ]
    with (OUT_DIR / "accuracy_gate_0.5_suspicious_zero_accepted_updates.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = list(indexed_records[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(suspicious)

    total = sum(r["group_count"] for r in indexed_records)
    accepted = sum(r["accepted"] for r in indexed_records)
    rejected = total - accepted

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    axes[0].plot(rolling_x, rolling_rate, color="#1f77b4", linewidth=1.6)
    axes[0].set_title("Accuracy gate=0.5 accepted fraction, rolling 200 attempts")
    axes[0].set_xlabel("Attempted groups")
    axes[0].set_ylabel("Accepted fraction")
    axes[0].grid(True, alpha=0.25)

    bucket_x = [int(row["bucket"].split("-", 1)[0]) for row in bucket_rows]
    bucket_rate = [row["accepted_rate"] for row in bucket_rows]
    axes[1].bar(bucket_x, bucket_rate, width=bucket_size * 0.8, color="#ff7f0e", alpha=0.85)
    axes[1].set_title("Accepted fraction by accepted-update bucket")
    axes[1].set_xlabel("Accepted update step bucket")
    axes[1].set_ylabel("Accepted fraction")
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        f"gate=0.5: total={int(total)}, accepted={int(accepted)}, rejected={int(rejected)}, "
        f"overall={accepted / total:.2%}, attempts/accepted={total / accepted:.2f}"
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "accuracy_gate_0.5_trend.png", dpi=160)

    print(f"logs: {len(logs)}")
    print(f"records/groups: {int(total)}")
    print(f"accepted: {int(accepted)}")
    print(f"rejected: {int(rejected)}")
    print(f"accepted_rate: {accepted / total:.6f}")
    print(f"groups_per_accepted: {total / accepted:.3f}")
    print(f"suspicious_zero_accepted_updates: {len(suspicious)}")
    print(f"wrote: {OUT_DIR / 'accuracy_gate_0.5_trend.png'}")


if __name__ == "__main__":
    main()
