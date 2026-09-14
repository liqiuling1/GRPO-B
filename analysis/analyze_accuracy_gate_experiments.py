import ast
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis"
DICT_RE = re.compile(r"\{.*\}")
GATE_RE = re.compile(
    r"train_accuracy_gate_(?P<gate>[0-9.]+(?:-[0-9.]+)?)(?:_(?P<run>1st|2nd))?_(?P<start>\d+)-(?P<end>\d+)\.log$"
)


EXPERIMENTS = {
    "0.5": ["train_accuracy_gate_0.5_*.log"],
    "0.375-0.625": ["train_accuracy_gate_0.375-0.625_*.log"],
    "0.4375-0.5625_1st": ["train_accuracy_gate_0.4375-0.5625_1st_*.log"],
    "0.4375-0.5625_2nd": ["train_accuracy_gate_0.4375-0.5625_2nd_*.log"],
}


def as_float(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def log_sort_key(path):
    match = GATE_RE.search(path.name)
    if not match:
        return (10**9, 10**9, path.name)
    return (int(match.group("start")), int(match.group("end")), path.name)


def iter_metric_rows(path):
    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        match = DICT_RE.search(line)
        if not match or "accuracy_gate/" not in match.group(0):
            continue
        try:
            row = ast.literal_eval(match.group(0))
        except (SyntaxError, ValueError):
            continue
        yield line_no, row


def read_header(path):
    header = {
        "resume_checkpoint": "",
        "initial_accepted_update_steps": "",
        "resume_attempt_step": "",
        "trainer_max_steps": "",
        "min_correct_rate": "",
        "max_correct_rate": "",
    }
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[:120]:
        gate_match = re.search(r"min_correct_rate=([0-9.]+), max_correct_rate=([0-9.]+)", line)
        if gate_match:
            header["min_correct_rate"] = gate_match.group(1)
            header["max_correct_rate"] = gate_match.group(2)
        resume_match = re.search(
            r"从 checkpoint 续训: ([^;]+); initial_accepted_update_steps=(\d+); "
            r"resume_attempt_step=(\d+); trainer max_steps=(\d+)",
            line,
        )
        if resume_match:
            header["resume_checkpoint"] = resume_match.group(1)
            header["initial_accepted_update_steps"] = resume_match.group(2)
            header["resume_attempt_step"] = resume_match.group(3)
            header["trainer_max_steps"] = resume_match.group(4)
    return header


def collect_experiment(name, patterns):
    logs = []
    for pattern in patterns:
        logs.extend(ROOT.glob(pattern))
    logs = sorted(set(logs), key=log_sort_key)

    records = []
    file_rows = []
    for path in logs:
        header = read_header(path)
        file_records = []
        last_step = None
        truncated_at_line = ""
        truncated_from_step = ""
        truncated_to_step = ""
        for line_no, row in iter_metric_rows(path):
            group_count = as_float(row, "accuracy_gate/group_count")
            accepted = as_float(row, "accuracy_gate/accepted_groups")
            correct_rate = as_float(row, "accuracy_gate/correct_rate_mean")
            accepted_update_steps = as_float(row, "accuracy_gate/accepted_update_steps")
            if last_step is not None and accepted_update_steps < last_step:
                truncated_at_line = line_no
                truncated_from_step = int(last_step)
                truncated_to_step = int(accepted_update_steps)
                break
            last_step = accepted_update_steps
            record = {
                "experiment": name,
                "file": path.name,
                "line": line_no,
                "group_count": group_count,
                "accepted": accepted,
                "rejected": group_count - accepted,
                "accepted_update_steps": accepted_update_steps,
                "rejected_update_steps": as_float(row, "accuracy_gate/rejected_update_steps"),
                "correct_rate": correct_rate,
                "correct_count_of_16": int(round(correct_rate * 16)),
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
        first_step = int(file_records[0]["accepted_update_steps"]) if file_records else None
        last_step = int(file_records[-1]["accepted_update_steps"]) if file_records else None
        file_rows.append(
            {
                "experiment": name,
                "file": path.name,
                **header,
                "rows": len(file_records),
                "start_accepted_step_in_log": first_step,
                "end_accepted_step_in_log": last_step,
                "truncated_at_line": truncated_at_line,
                "truncated_from_step": truncated_from_step,
                "truncated_to_step": truncated_to_step,
                "total_groups": int(total),
                "accepted_groups": int(accepted),
                "rejected_groups": int(total - accepted),
                "accepted_rate": accepted / total if total else 0.0,
                "groups_per_accepted": total / accepted if accepted else 0.0,
            }
        )

    indexed = []
    for idx, record in enumerate(records, 1):
        indexed.append({"experiment_attempt_index": idx, **record})
    return logs, indexed, file_rows


def summarize_buckets(records, bucket_size=50):
    buckets = defaultdict(lambda: {"total": 0.0, "accepted": 0.0, "difficulty": Counter(), "accepted_difficulty": Counter()})
    for record in records:
        step = int(record["accepted_update_steps"])
        bucket = (step // bucket_size) * bucket_size
        data = buckets[bucket]
        data["total"] += record["group_count"]
        data["accepted"] += record["accepted"]
        data["difficulty"][record["correct_count_of_16"]] += record["group_count"]
        if record["accepted"]:
            data["accepted_difficulty"][record["correct_count_of_16"]] += record["accepted"]

    rows = []
    for bucket in sorted(buckets):
        data = buckets[bucket]
        total = data["total"]
        accepted = data["accepted"]
        row = {
            "bucket": f"{bucket}-{bucket + bucket_size}",
            "total_groups": int(total),
            "accepted_groups": int(accepted),
            "rejected_groups": int(total - accepted),
            "accepted_rate": accepted / total if total else 0.0,
            "groups_per_accepted": total / accepted if accepted else 0.0,
        }
        for correct_count in range(17):
            row[f"all_k{correct_count}"] = int(data["difficulty"][correct_count])
        for correct_count in range(17):
            row[f"accepted_k{correct_count}"] = int(data["accepted_difficulty"][correct_count])
        rows.append(row)
    return rows


def summarize_buckets_by_file(records, bucket_size=50):
    by_file = defaultdict(list)
    for record in records:
        by_file[record["file"]].append(record)

    rows = []
    for file_name in sorted(by_file):
        for row in summarize_buckets(by_file[file_name], bucket_size=bucket_size):
            rows.append({"file": file_name, **row})
    return rows


def compact_difficulty_rows(name, rows):
    compact = []
    for row in rows:
        accepted = row["accepted_groups"]
        compact_row = {
            "experiment": name,
            **({"file": row["file"]} if "file" in row else {}),
            "bucket": row["bucket"],
            "total_groups": row["total_groups"],
            "accepted_groups": accepted,
            "rejected_groups": row["rejected_groups"],
            "accepted_rate": row["accepted_rate"],
            "groups_per_accepted": row["groups_per_accepted"],
        }
        for correct_count in range(17):
            count = row.get(f"accepted_k{correct_count}", 0)
            if count:
                compact_row[f"accepted_k{correct_count}_of_16"] = count
                compact_row[f"accepted_k{correct_count}_share"] = count / accepted if accepted else 0.0
        compact.append(compact_row)
    return compact


def merge_records_prefer_later_segments(records, file_rows):
    records_by_file = defaultdict(list)
    for record in records:
        records_by_file[record["file"]].append(record)

    merged = []
    for index, file_row in enumerate(file_rows):
        file_name = file_row["file"]
        file_records = records_by_file[file_name]
        cutoff = None
        if index + 1 < len(file_rows):
            next_initial = file_rows[index + 1].get("initial_accepted_update_steps")
            if next_initial != "":
                cutoff = int(next_initial)
            elif file_rows[index + 1].get("start_accepted_step_in_log") is not None:
                cutoff = int(file_rows[index + 1]["start_accepted_step_in_log"])

        for record in file_records:
            if cutoff is not None and int(record["accepted_update_steps"]) > cutoff:
                continue
            merged.append(record)

    return [
        {"experiment_attempt_index": index, **{k: v for k, v in record.items() if k != "experiment_attempt_index"}}
        for index, record in enumerate(merged, 1)
    ]


def summarize_records(records):
    total = sum(record["group_count"] for record in records)
    accepted = sum(record["accepted"] for record in records)
    return {
        "total_groups": int(total),
        "accepted_groups": int(accepted),
        "rejected_groups": int(total - accepted),
        "accepted_rate": accepted / total if total else 0.0,
        "groups_per_accepted": total / accepted if accepted else 0.0,
        "start_accepted_step": int(records[0]["accepted_update_steps"]) if records else "",
        "end_accepted_step": int(records[-1]["accepted_update_steps"]) if records else "",
    }


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_bucket_rates(all_bucket_rows, output_name):
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {
        "0.5": "#1f77b4",
        "0.375-0.625": "#ff7f0e",
        "0.4375-0.5625_1st": "#2ca02c",
        "0.4375-0.5625_2nd": "#d62728",
    }
    for name, rows in all_bucket_rows.items():
        x = [int(row["bucket"].split("-", 1)[0]) for row in rows]
        y = [row["accepted_rate"] for row in rows]
        ax.plot(x, y, marker="o", linewidth=1.8, markersize=3.5, label=name, color=colors.get(name))
    ax.set_title("Accepted data fraction by 50 accepted-update buckets")
    ax.set_xlabel("Accepted update bucket")
    ax.set_ylabel("Accepted fraction")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / output_name, dpi=160)


def main():
    OUT_DIR.mkdir(exist_ok=True)

    all_file_rows = []
    all_compact_rows = []
    all_merged_summary_rows = []
    all_bucket_rows = {}
    all_merged_bucket_rows = {}
    all_records = []
    all_merged_records = []
    for name, patterns in EXPERIMENTS.items():
        logs, records, file_rows = collect_experiment(name, patterns)
        if not logs:
            continue
        merged_records = merge_records_prefer_later_segments(records, file_rows)
        bucket_rows = summarize_buckets(records)
        merged_bucket_rows = summarize_buckets(merged_records)
        file_bucket_rows = summarize_buckets_by_file(records)
        compact_rows = compact_difficulty_rows(name, bucket_rows)
        merged_compact_rows = compact_difficulty_rows(name, merged_bucket_rows)
        file_compact_rows = []
        for row in file_bucket_rows:
            file_compact_rows.extend(compact_difficulty_rows(name, [row]))
        write_csv(OUT_DIR / f"accuracy_gate_{name}_file_summary.csv", file_rows)
        write_csv(OUT_DIR / f"accuracy_gate_{name}_bucket_difficulty_summary.csv", bucket_rows)
        write_csv(OUT_DIR / f"accuracy_gate_{name}_bucket_accepted_difficulty_compact.csv", compact_rows)
        write_csv(OUT_DIR / f"accuracy_gate_{name}_merged_bucket_difficulty_summary.csv", merged_bucket_rows)
        write_csv(OUT_DIR / f"accuracy_gate_{name}_merged_bucket_accepted_difficulty_compact.csv", merged_compact_rows)
        write_csv(OUT_DIR / f"accuracy_gate_{name}_file_bucket_accepted_difficulty_compact.csv", file_compact_rows)
        write_csv(OUT_DIR / f"accuracy_gate_{name}_records.csv", records)
        write_csv(OUT_DIR / f"accuracy_gate_{name}_merged_records.csv", merged_records)
        all_file_rows.extend(file_rows)
        all_compact_rows.extend(compact_rows)
        all_merged_summary_rows.append({"experiment": name, **summarize_records(merged_records)})
        all_records.extend(records)
        all_merged_records.extend(merged_records)
        all_bucket_rows[name] = bucket_rows
        all_merged_bucket_rows[name] = merged_bucket_rows

    write_csv(OUT_DIR / "accuracy_gate_all_file_summary.csv", all_file_rows)
    write_csv(OUT_DIR / "accuracy_gate_all_bucket_accepted_difficulty_compact.csv", all_compact_rows)
    write_csv(OUT_DIR / "accuracy_gate_all_merged_summary.csv", all_merged_summary_rows)
    write_csv(OUT_DIR / "accuracy_gate_all_records.csv", all_records)
    write_csv(OUT_DIR / "accuracy_gate_all_merged_records.csv", all_merged_records)
    plot_bucket_rates(all_bucket_rows, "accuracy_gate_all_bucket_rates.png")
    plot_bucket_rates(all_merged_bucket_rows, "accuracy_gate_all_merged_bucket_rates.png")

    for row in all_file_rows:
        print(
            row["experiment"],
            row["file"],
            "rows",
            row["rows"],
            "accepted",
            row["accepted_groups"],
            "rate",
            f"{row['accepted_rate']:.4f}",
            "steps",
            row["start_accepted_step_in_log"],
            row["end_accepted_step_in_log"],
            "resume",
            row["initial_accepted_update_steps"] or "NO",
        )
    print(f"wrote: {OUT_DIR / 'accuracy_gate_all_bucket_rates.png'}")
    print(f"wrote: {OUT_DIR / 'accuracy_gate_all_merged_bucket_rates.png'}")


if __name__ == "__main__":
    main()
