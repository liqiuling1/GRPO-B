import argparse
import csv
import json
import math
import re
import zipfile
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Dict, Iterable, List, Sequence, Tuple
from xml.sax.saxutils import escape


CHECKPOINTS = [0, 10, 20, 30, 40, 50, 100, 250, 500, 1000, 1500, 2000, 4000, 7473]
CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)")
DEFAULT_INPUT_DIR = Path(
    "outputs/observation_sets_gsm8k_p_intervals_seed42/"
    "difficulty_drift_eval_pdist20_seed42"
)
DEFAULT_ORIGINAL_NAME = "gsm8k_p_scores_final_no_truncation_2_sampled_20pct.jsonl"
RANGE_THRESHOLDS = [0.125, 0.25, 0.375, 0.50, 0.75]
ABS_DELTA_THRESHOLDS = [0.125, 0.25, 0.375, 0.50]
ROLLOUTS = 32
EPS = 1e-9


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze single-sample difficulty drift over a Vanilla GRPO checkpoint trajectory."
    )
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--original_file",
        type=Path,
        default=None,
        help=f"Defaults to <input_dir>/{DEFAULT_ORIGINAL_NAME}",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Defaults to <input_dir>/difficulty_drift_statistics",
    )
    parser.add_argument("--sample_id_field", default="uid")
    parser.add_argument("--p_field", default="p")
    parser.add_argument("--rollout_field", default="_num_answers")
    parser.add_argument("--expected_n", type=int, default=1495)
    parser.add_argument("--expected_rollouts", type=int, default=ROLLOUTS)
    return parser.parse_args()


def checkpoint_file(input_dir: Path, original_file: Path, checkpoint: int) -> Path:
    if checkpoint == 0:
        return original_file
    candidates = sorted(input_dir.glob(f"*checkpoint-{checkpoint}.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"Missing checkpoint-{checkpoint} JSONL in {input_dir}")
    if len(candidates) > 1:
        names = [str(path) for path in candidates]
        raise ValueError(f"Multiple files matched checkpoint-{checkpoint}: {names}")
    return candidates[0]


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def fmt(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def threshold_label(value: float) -> str:
    return f">={value:.3f}"


def read_checkpoint(
    path: Path,
    checkpoint: int,
    sample_id_field: str,
    p_field: str,
    rollout_field: str,
    expected_rollouts: int,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    p_by_id: Dict[str, float] = {}
    duplicate_count = 0
    abnormal_success_rate_count = 0
    bad_rollout_count = 0
    missing_id_count = 0
    missing_p_count = 0

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if sample_id_field not in row:
                missing_id_count += 1
                continue
            if p_field not in row:
                missing_p_count += 1
                continue

            sample_id = str(row[sample_id_field])
            if sample_id in p_by_id:
                duplicate_count += 1
                continue

            p_value = float(row[p_field])
            rollouts = row.get(rollout_field)
            if rollouts != expected_rollouts:
                bad_rollout_count += 1

            inferred_correct = round(p_value * expected_rollouts)
            is_valid_p = (
                -EPS <= p_value <= 1 + EPS
                and 0 <= inferred_correct <= expected_rollouts
                and abs(p_value - inferred_correct / expected_rollouts) <= 1e-8
            )
            if not is_valid_p:
                abnormal_success_rate_count += 1

            p_by_id[sample_id] = p_value

    info = {
        "checkpoint": checkpoint,
        "file": str(path),
        "sample_count": len(p_by_id),
        "duplicate_count": duplicate_count,
        "missing_id_count": missing_id_count,
        "missing_p_count": missing_p_count,
        "bad_rollout_count": bad_rollout_count,
        "abnormal_success_rate_count": abnormal_success_rate_count,
    }
    return p_by_id, info


def build_aligned_data(args) -> Tuple[Dict[int, Dict[str, float]], List[Dict[str, object]], List[str], List[Path]]:
    original_file = args.original_file or (args.input_dir / DEFAULT_ORIGINAL_NAME)
    if not original_file.is_file():
        raise FileNotFoundError(f"Original file not found: {original_file}")

    files = [checkpoint_file(args.input_dir, original_file, checkpoint) for checkpoint in CHECKPOINTS]
    data: Dict[int, Dict[str, float]] = {}
    integrity_rows: List[Dict[str, object]] = []

    for checkpoint, path in zip(CHECKPOINTS, files):
        p_by_id, info = read_checkpoint(
            path=path,
            checkpoint=checkpoint,
            sample_id_field=args.sample_id_field,
            p_field=args.p_field,
            rollout_field=args.rollout_field,
            expected_rollouts=args.expected_rollouts,
        )
        data[checkpoint] = p_by_id
        integrity_rows.append(info)

    all_ids = [set(data[checkpoint]) for checkpoint in CHECKPOINTS]
    common_ids = sorted(set.intersection(*all_ids), key=lambda value: int(value) if value.isdigit() else value)
    union_ids = set.union(*all_ids)

    for row in integrity_rows:
        checkpoint = int(row["checkpoint"])
        ids = set(data[checkpoint])
        row["missing_vs_union"] = len(union_ids - ids)
        row["extra_vs_common"] = len(ids - set(common_ids))
        row["effective_aligned_n"] = len(common_ids)

    return data, integrity_rows, common_ids, files


def compute_per_sample_rows(data: Dict[int, Dict[str, float]], sample_ids: Sequence[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for sample_id in sample_ids:
        trajectory = {checkpoint: data[checkpoint][sample_id] for checkpoint in CHECKPOINTS}
        p0 = trajectory[0]
        min_checkpoint, p_min = min(trajectory.items(), key=lambda item: (item[1], item[0]))
        max_checkpoint, p_max = max(trajectory.items(), key=lambda item: (item[1], -item[0]))
        max_initial_deviation = max(abs(trajectory[checkpoint] - p0) for checkpoint in CHECKPOINTS)

        row: Dict[str, object] = {"sample_id": sample_id}
        for checkpoint in CHECKPOINTS:
            row[f"P_{checkpoint}"] = trajectory[checkpoint]
        row.update(
            {
                "P_0": p0,
                "P_min": p_min,
                "P_min_checkpoint": min_checkpoint,
                "P_max": p_max,
                "P_max_checkpoint": max_checkpoint,
                "Range": p_max - p_min,
                "MaxInitialDeviation": max_initial_deviation,
            }
        )
        rows.append(row)
    return rows


def count_rate(values: Sequence[float], threshold: float) -> Tuple[int, float]:
    count = sum(value >= threshold - EPS for value in values)
    return count, count / len(values) if values else math.nan


def build_range_statistics(per_sample_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    ranges = [float(row["Range"]) for row in per_sample_rows]
    rows: List[Dict[str, object]] = [
        {"Metric": "N", "Value": len(ranges), "Threshold": "", "Count": "", "Rate(%)": ""},
        {"Metric": "Mean Range", "Value": fmt(mean(ranges)), "Threshold": "", "Count": "", "Rate(%)": ""},
        {"Metric": "Median Range", "Value": fmt(median(ranges)), "Threshold": "", "Count": "", "Rate(%)": ""},
        {"Metric": "Std Range", "Value": fmt(pstdev(ranges)), "Threshold": "", "Count": "", "Rate(%)": ""},
        {"Metric": "Max Range", "Value": fmt(max(ranges)), "Threshold": "", "Count": "", "Rate(%)": ""},
        {"Metric": "Min Range", "Value": fmt(min(ranges)), "Threshold": "", "Count": "", "Rate(%)": ""},
    ]
    for threshold in RANGE_THRESHOLDS:
        count, rate = count_rate(ranges, threshold)
        rows.append(
            {
                "Metric": "Range threshold",
                "Value": "",
                "Threshold": threshold_label(threshold),
                "Count": count,
                "Rate(%)": pct(rate),
            }
        )
    return rows


def build_checkpoint_initial_deviation_statistics(
    data: Dict[int, Dict[str, float]], sample_ids: Sequence[str]
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for checkpoint in CHECKPOINTS:
        if checkpoint == 0:
            continue
        deltas = [data[checkpoint][sample_id] - data[0][sample_id] for sample_id in sample_ids]
        abs_deltas = [abs(value) for value in deltas]
        row: Dict[str, object] = {
            "Checkpoint": checkpoint,
            "Mean_DeltaP": fmt(mean(deltas)),
            "Mean_AbsDeltaP": fmt(mean(abs_deltas)),
            "Median_AbsDeltaP": fmt(median(abs_deltas)),
            "Std_AbsDeltaP": fmt(pstdev(abs_deltas)),
        }
        for threshold in ABS_DELTA_THRESHOLDS:
            count, rate = count_rate(abs_deltas, threshold)
            label = str(threshold).rstrip("0").rstrip(".")
            row[f"AbsDelta_ge_{label}_Count"] = count
            row[f"AbsDelta_ge_{label}_Rate"] = pct(rate)
        easier_count = sum(delta >= 0.25 - EPS for delta in deltas)
        harder_count = sum(delta <= -0.25 + EPS for delta in deltas)
        n = len(deltas)
        row["Easier_ge_0.25_Count"] = easier_count
        row["Easier_ge_0.25_Rate"] = pct(easier_count / n)
        row["Harder_ge_0.25_Count"] = harder_count
        row["Harder_ge_0.25_Rate"] = pct(harder_count / n)
        rows.append(row)
    return rows


def build_max_initial_deviation_statistics(per_sample_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    values = [float(row["MaxInitialDeviation"]) for row in per_sample_rows]
    rows: List[Dict[str, object]] = [
        {"Metric": "N", "Value": len(values), "Threshold": "", "Count": "", "Rate(%)": ""},
        {"Metric": "Mean MaxInitialDeviation", "Value": fmt(mean(values)), "Threshold": "", "Count": "", "Rate(%)": ""},
        {"Metric": "Median MaxInitialDeviation", "Value": fmt(median(values)), "Threshold": "", "Count": "", "Rate(%)": ""},
        {"Metric": "Std MaxInitialDeviation", "Value": fmt(pstdev(values)), "Threshold": "", "Count": "", "Rate(%)": ""},
        {"Metric": "Max MaxInitialDeviation", "Value": fmt(max(values)), "Threshold": "", "Count": "", "Rate(%)": ""},
        {"Metric": "Min MaxInitialDeviation", "Value": fmt(min(values)), "Threshold": "", "Count": "", "Rate(%)": ""},
    ]
    for threshold in RANGE_THRESHOLDS:
        count, rate = count_rate(values, threshold)
        rows.append(
            {
                "Metric": "MaxInitialDeviation threshold",
                "Value": "",
                "Threshold": threshold_label(threshold),
                "Count": count,
                "Rate(%)": pct(rate),
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def xlsx_cell(value: object) -> str:
    if value is None:
        return "<c/>"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"<c><v>{value}</v></c>"
    text = escape(str(value))
    return f'<c t="inlineStr"><is><t>{text}</t></is></c>'


def xlsx_sheet_xml(rows: Sequence[Sequence[object]]) -> str:
    sheet_rows = []
    for row_idx, row in enumerate(rows, start=1):
        cells = "".join(xlsx_cell(value) for value in row)
        sheet_rows.append(f'<row r="{row_idx}">{cells}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{''.join(sheet_rows)}</sheetData>"
        "</worksheet>"
    )


def write_xlsx(path: Path, sheets: Sequence[Tuple[str, Sequence[Dict[str, object]], Sequence[str]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
    ]
    workbook_sheets = []
    workbook_rels = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
    ]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            "</Relationships>",
        )
        for idx, (sheet_name, rows, columns) in enumerate(sheets, start=1):
            content_types.append(
                f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
                'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            )
            workbook_sheets.append(f'<sheet name="{escape(sheet_name)}" sheetId="{idx}" r:id="rId{idx}"/>')
            workbook_rels.append(
                f'<Relationship Id="rId{idx}" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
                f'Target="worksheets/sheet{idx}.xml"/>'
            )
            sheet_rows = [list(columns)]
            sheet_rows.extend([[row.get(column, "") for column in columns] for row in rows])
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", xlsx_sheet_xml(sheet_rows))
        content_types.append("</Types>")
        workbook_rels.append("</Relationships>")
        zf.writestr("[Content_Types].xml", "".join(content_types))
        zf.writestr("xl/_rels/workbook.xml.rels", "".join(workbook_rels))
        zf.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f"<sheets>{''.join(workbook_sheets)}</sheets>"
            "</workbook>",
        )


def print_integrity(integrity_rows: Sequence[Dict[str, object]], expected_n: int) -> None:
    print("==== Data Integrity Check ====")
    print(
        "checkpoint\tsample_count\tmissing_vs_union\tduplicate_count\t"
        "bad_rollout_count\tabnormal_success_rate_count"
    )
    for row in integrity_rows:
        print(
            f"{row['checkpoint']}\t{row['sample_count']}\t{row['missing_vs_union']}\t"
            f"{row['duplicate_count']}\t{row['bad_rollout_count']}\t"
            f"{row['abnormal_success_rate_count']}"
        )
    counts = {int(row["sample_count"]) for row in integrity_rows}
    if counts != {expected_n}:
        print(f"WARNING: sample counts differ from expected_n={expected_n}: {sorted(counts)}")


def main():
    args = parse_args()
    output_dir = args.output_dir or (args.input_dir / "difficulty_drift_statistics")
    output_dir.mkdir(parents=True, exist_ok=True)

    data, integrity_rows, sample_ids, files = build_aligned_data(args)
    print_integrity(integrity_rows, args.expected_n)
    print(f"Aligned effective N: {len(sample_ids)}")

    per_sample_rows = compute_per_sample_rows(data, sample_ids)
    range_stats = build_range_statistics(per_sample_rows)
    checkpoint_stats = build_checkpoint_initial_deviation_statistics(data, sample_ids)
    max_initial_stats = build_max_initial_deviation_statistics(per_sample_rows)

    per_sample_columns = (
        ["sample_id", *[f"P_{checkpoint}" for checkpoint in CHECKPOINTS]]
        + ["P_0", "P_min", "P_min_checkpoint", "P_max", "P_max_checkpoint", "Range", "MaxInitialDeviation"]
    )
    range_columns = ["Metric", "Value", "Threshold", "Count", "Rate(%)"]
    checkpoint_columns = [
        "Checkpoint",
        "Mean_DeltaP",
        "Mean_AbsDeltaP",
        "Median_AbsDeltaP",
        "Std_AbsDeltaP",
        "AbsDelta_ge_0.125_Count",
        "AbsDelta_ge_0.125_Rate",
        "AbsDelta_ge_0.25_Count",
        "AbsDelta_ge_0.25_Rate",
        "AbsDelta_ge_0.375_Count",
        "AbsDelta_ge_0.375_Rate",
        "AbsDelta_ge_0.5_Count",
        "AbsDelta_ge_0.5_Rate",
        "Easier_ge_0.25_Count",
        "Easier_ge_0.25_Rate",
        "Harder_ge_0.25_Count",
        "Harder_ge_0.25_Rate",
    ]
    max_initial_columns = ["Metric", "Value", "Threshold", "Count", "Rate(%)"]

    per_sample_path = output_dir / "per_sample_drift.csv"
    range_path = output_dir / "range_statistics.csv"
    checkpoint_path = output_dir / "checkpoint_initial_deviation_statistics.csv"
    max_initial_path = output_dir / "max_initial_deviation_statistics.csv"
    xlsx_path = output_dir / "difficulty_drift_statistics.xlsx"
    integrity_path = output_dir / "data_integrity_check.csv"
    metadata_path = output_dir / "metadata.json"

    write_csv(per_sample_path, per_sample_rows, per_sample_columns)
    write_csv(range_path, range_stats, range_columns)
    write_csv(checkpoint_path, checkpoint_stats, checkpoint_columns)
    write_csv(max_initial_path, max_initial_stats, max_initial_columns)
    write_csv(
        integrity_path,
        integrity_rows,
        [
            "checkpoint",
            "file",
            "sample_count",
            "missing_vs_union",
            "duplicate_count",
            "missing_id_count",
            "missing_p_count",
            "bad_rollout_count",
            "abnormal_success_rate_count",
            "effective_aligned_n",
        ],
    )
    write_xlsx(
        xlsx_path,
        [
            ("Range_Summary", range_stats, range_columns),
            ("Initial_Deviation_By_Checkpoint", checkpoint_stats, checkpoint_columns),
            ("Max_Initial_Deviation", max_initial_stats, max_initial_columns),
            ("Per_Sample_Details", per_sample_rows, per_sample_columns),
        ],
    )
    metadata = {
        "input_files": [str(path) for path in files],
        "checkpoints": CHECKPOINTS,
        "sample_id_field": args.sample_id_field,
        "p_field": args.p_field,
        "rollout_field": args.rollout_field,
        "expected_rollouts": args.expected_rollouts,
        "effective_n": len(sample_ids),
        "definition_range": "Range_i = max_t P_i,t - min_t P_i,t",
        "definition_delta": "DeltaP_i,t = P_i,t - P_i,0",
        "definition_max_initial_deviation": "max_t |P_i,t - P_i,0|",
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    ranges = [float(row["Range"]) for row in per_sample_rows]
    max_devs = [float(row["MaxInitialDeviation"]) for row in per_sample_rows]
    r025 = count_rate(ranges, 0.25)[1]
    r050 = count_rate(ranges, 0.50)[1]
    m025 = count_rate(max_devs, 0.25)[1]
    m050 = count_rate(max_devs, 0.50)[1]

    print("==== Summary ====")
    print(f"Input files: {len(files)}")
    for path in files:
        print(f"  {path}")
    print(f"Effective N: {len(sample_ids)}")
    print(f"Mean Range: {fmt(mean(ranges))}")
    print(f"Median Range: {fmt(median(ranges))}")
    print(f"Range>=0.25 Rate: {pct(r025)}")
    print(f"Range>=0.50 Rate: {pct(r050)}")
    print(f"MaxInitialDeviation>=0.25 Rate: {pct(m025)}")
    print(f"MaxInitialDeviation>=0.50 Rate: {pct(m050)}")
    print("==== Outputs ====")
    for path in [per_sample_path, range_path, checkpoint_path, max_initial_path, xlsx_path, integrity_path, metadata_path]:
        print(path.resolve())


if __name__ == "__main__":
    main()
