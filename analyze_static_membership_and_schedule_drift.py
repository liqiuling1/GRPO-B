import argparse
import json
import math
import re
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


N_OBSERVATION = 1495
TEST_ACCURACY = {
    1000: 0.6300,
    2500: 0.6482,
    3500: 0.6535,
    4500: 0.6247,
    5500: 0.6293,
    6500: 0.6164,
}


RAW_SCORE_RE = re.compile(r"raw_scores_checkpoint-(\d+)_uid1_(\d+)_(\d+)\.jsonl$")


def pct(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{100.0 * float(value):.2f}%"


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_no}: {exc}") from exc
    return rows


def write_minimal_xlsx(path: Path, sheets: Dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def col_name(idx: int) -> str:
        name = ""
        idx += 1
        while idx:
            idx, rem = divmod(idx - 1, 26)
            name = chr(65 + rem) + name
        return name

    def cell_xml(row_idx: int, col_idx: int, value) -> str:
        ref = f"{col_name(col_idx)}{row_idx}"
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return f'<c r="{ref}"/>'
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return f'<c r="{ref}"><v>{value}</v></c>'
        text = escape(str(value))
        return f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'

    workbook_sheets = []
    rels = []
    content_overrides = [
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
    ]
    sheet_payloads = {}
    for sheet_idx, (sheet_name, df) in enumerate(sheets.items(), start=1):
        safe_name = sheet_name[:31]
        workbook_sheets.append(
            f'<sheet name="{escape(safe_name)}" sheetId="{sheet_idx}" r:id="rId{sheet_idx}"/>'
        )
        rels.append(
            f'<Relationship Id="rId{sheet_idx}" '
            f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{sheet_idx}.xml"/>'
        )
        content_overrides.append(
            f'<Override PartName="/xl/worksheets/sheet{sheet_idx}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
        rows_xml = []
        values = [list(df.columns)] + df.astype(object).where(pd.notna(df), None).values.tolist()
        for row_idx, row in enumerate(values, start=1):
            cells = "".join(cell_xml(row_idx, col_idx, value) for col_idx, value in enumerate(row))
            rows_xml.append(f'<row r="{row_idx}">{cells}</row>')
        sheet_payloads[f"xl/worksheets/sheet{sheet_idx}.xml"] = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f'<sheetData>{"".join(rows_xml)}</sheetData></worksheet>'
        )

    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets>{"".join(workbook_sheets)}</sheets></workbook>'
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'{"".join(rels)}'
        '<Relationship Id="rIdStyles" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/></Relationships>'
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/></Relationships>'
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        f'{"".join(content_overrides)}</Types>'
    )
    styles = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
        '<borders count="1"><border/></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
        '</styleSheet>'
    )

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        zf.writestr("xl/styles.xml", styles)
        for name, payload in sheet_payloads.items():
            zf.writestr(name, payload)


def format_percent_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].map(pct)
    return out


def build_e1(membership_csv: Path, out_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(membership_csv)
    out = pd.DataFrame()
    out["Stage"] = df["Stage"]
    out["Interval"] = df["Interval"]
    out["Checkpoint"] = df["Checkpoint"].astype(int)
    out["Initial_Size"] = df["Initial_Size"].astype(int)
    out["Current_Size"] = df["Current_Size"].astype(int)
    out["Intersection_Count"] = df["Intersection_Count"].astype(int)
    out["False_Accept_Count"] = df["Outflow_Count"].astype(int)
    out["False_Accept_Rate"] = out["False_Accept_Count"] / out["Initial_Size"].replace(0, pd.NA)
    out["False_Reject_Count"] = df["Inflow_Count"].astype(int)
    out["False_Reject_Rate"] = out["False_Reject_Count"] / out["Current_Size"].replace(0, pd.NA)
    out["Mismatch_Count"] = out["False_Accept_Count"] + out["False_Reject_Count"]
    out["Overall_Mismatch_Rate"] = out["Mismatch_Count"] / N_OBSERVATION
    out["Jaccard"] = df["Jaccard"].astype(float)

    percent_cols = ["False_Accept_Rate", "False_Reject_Rate", "Overall_Mismatch_Rate"]
    csv_df = format_percent_columns(out, percent_cols)
    csv_df["Jaccard"] = csv_df["Jaccard"].map(lambda x: f"{x:.6f}")
    csv_df.to_csv(out_dir / "e1_static_membership_mismatch.csv", index=False)
    write_minimal_xlsx(
        out_dir / "e1_static_membership_mismatch.xlsx",
        {"E1_Mismatch": csv_df},
    )
    return out


def load_static_sort(path: Path) -> pd.DataFrame:
    rows = read_jsonl(path)
    df = pd.DataFrame(rows)
    df["uid"] = df["uid"].astype(str)
    df["uid1"] = df["uid1"].astype(int)
    df["p"] = df["p"].astype(float)
    if df["uid"].duplicated().any():
        dup = df.loc[df["uid"].duplicated(), "uid"].head(5).tolist()
        raise ValueError(f"Duplicate uid in sorted score file, e.g. {dup}")
    return df[["uid", "uid1", "p"]].rename(columns={"p": "P0"})


def load_raw_score_file(path: Path) -> Tuple[int, int, int, pd.DataFrame]:
    match = RAW_SCORE_RE.match(path.name)
    if not match:
        raise ValueError(f"Unexpected raw score filename: {path}")
    checkpoint, block_start, block_end = map(int, match.groups())
    rows = read_jsonl(path)
    df = pd.DataFrame(rows)
    df["uid"] = df["uid"].astype(str)
    df["Pt"] = df["p"].astype(float)
    df["_num_answers"] = df["_num_answers"].astype(int)
    if df["uid"].duplicated().any():
        dup = df.loc[df["uid"].duplicated(), "uid"].head(5).tolist()
        raise ValueError(f"Duplicate uid in {path}, e.g. {dup}")
    return checkpoint, block_start, block_end, df[["uid", "Pt", "_num_answers"]]


def add_rate_counts(record: Dict, prefix: str, mask: pd.Series, n: int) -> None:
    count = int(mask.sum())
    record[f"{prefix}_Count"] = count
    record[f"{prefix}_Rate"] = count / n if n else math.nan


def build_schedule(raw_dir: Path, sorted_scores_file: Path, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    static = load_static_sort(sorted_scores_file)
    raw_files = sorted(raw_dir.glob("raw_scores_checkpoint-*_uid1_*.jsonl"))
    if not raw_files:
        raise FileNotFoundError(f"No raw score files found in: {raw_dir}")

    per_sample_parts = []
    summary_records = []
    integrity_records = []
    for raw_file in raw_files:
        checkpoint, block_start, block_end, raw = load_raw_score_file(raw_file)
        merged = raw.merge(static, on="uid", how="left")
        if merged["uid1"].isna().any():
            missing = merged.loc[merged["uid1"].isna(), "uid"].head(5).tolist()
            raise ValueError(f"{raw_file} contains uid values absent from sorted score file, e.g. {missing}")
        merged["uid1"] = merged["uid1"].astype(int)
        outside_block = merged[(merged["uid1"] < block_start) | (merged["uid1"] > block_end)]
        actual_min_uid1 = int(merged["uid1"].min())
        actual_max_uid1 = int(merged["uid1"].max())
        integrity_records.append(
            {
                "Raw_File": str(raw_file),
                "Checkpoint": checkpoint,
                "Declared_Block_Start": block_start,
                "Declared_Block_End": block_end,
                "N": len(merged),
                "Actual_Min_uid1": actual_min_uid1,
                "Actual_Max_uid1": actual_max_uid1,
                "Rows_Outside_Declared_Block": int(len(outside_block)),
                "Status": "OK" if outside_block.empty else "SKIPPED_BLOCK_MISMATCH",
            }
        )
        if not outside_block.empty:
            print(
                f"[warn] skip {raw_file}: declared uid1={block_start}-{block_end}, "
                f"actual uid1={actual_min_uid1}-{actual_max_uid1}, "
                f"outside_rows={len(outside_block)}",
                flush=True,
            )
            continue
        if (merged["_num_answers"] != 32).any():
            bad = merged.loc[merged["_num_answers"] != 32, ["uid", "_num_answers"]].head(5).to_dict("records")
            raise ValueError(f"{raw_file} has non-32 rollout rows, e.g. {bad}")

        merged["Checkpoint"] = checkpoint
        merged["DeltaP"] = merged["Pt"] - merged["P0"]
        merged["AbsDeltaP"] = merged["DeltaP"].abs()
        merged = merged.sort_values("uid1")
        per_sample_parts.append(merged[["Checkpoint", "uid1", "uid", "P0", "Pt", "DeltaP", "AbsDeltaP"]])

        n = len(merged)
        p0_min = float(merged["P0"].min())
        p0_max = float(merged["P0"].max())
        expanded_lower = max(0.0, p0_min - 0.0625)
        expanded_upper = min(1.0, p0_max + 0.0625)
        record: Dict = {
            "Checkpoint": checkpoint,
            "Block_Start": block_start,
            "Block_End": block_end,
            "N": n,
            "Planned_P0_Min": p0_min,
            "Planned_P0_Max": p0_max,
            "Planned_Mean_P0": float(merged["P0"].mean()),
            "Current_Mean_Pt": float(merged["Pt"].mean()),
            "Mean_Signed_Drift": float(merged["DeltaP"].mean()),
            "Mean_Abs_Schedule_Drift": float(merged["AbsDeltaP"].mean()),
            "Median_Abs_Schedule_Drift": float(merged["AbsDeltaP"].median()),
        }
        add_rate_counts(record, "AbsDrift_ge_0.125", merged["AbsDeltaP"] >= 0.125, n)
        add_rate_counts(record, "AbsDrift_ge_0.25", merged["AbsDeltaP"] >= 0.25, n)
        add_rate_counts(record, "AbsDrift_ge_0.375", merged["AbsDeltaP"] >= 0.375, n)
        add_rate_counts(record, "Easier_ge_0.25", merged["DeltaP"] >= 0.25, n)
        add_rate_counts(record, "Harder_ge_0.25", merged["DeltaP"] <= -0.25, n)
        add_rate_counts(record, "Outside_Planned_Band", (merged["Pt"] < p0_min) | (merged["Pt"] > p0_max), n)
        add_rate_counts(record, "Outside_Expanded_Band", (merged["Pt"] < expanded_lower) | (merged["Pt"] > expanded_upper), n)
        add_rate_counts(record, "Current_AllZero", merged["Pt"] == 0, n)
        add_rate_counts(record, "Current_Mixed", (merged["Pt"] > 0) & (merged["Pt"] < 1), n)
        add_rate_counts(record, "Current_AllOne", merged["Pt"] == 1, n)
        record["Test_Accuracy"] = TEST_ACCURACY.get(checkpoint)
        summary_records.append(record)

    per_sample = pd.concat(per_sample_parts, ignore_index=True)
    summary = pd.DataFrame(summary_records).sort_values("Checkpoint")
    integrity = pd.DataFrame(integrity_records)
    integrity.to_csv(out_dir / "static_schedule_drift_data_integrity.csv", index=False)

    summary_columns = [
        "Checkpoint", "Block_Start", "Block_End", "N",
        "Planned_P0_Min", "Planned_P0_Max", "Planned_Mean_P0", "Current_Mean_Pt",
        "Mean_Signed_Drift", "Mean_Abs_Schedule_Drift", "Median_Abs_Schedule_Drift",
        "AbsDrift_ge_0.125_Count", "AbsDrift_ge_0.125_Rate",
        "AbsDrift_ge_0.25_Count", "AbsDrift_ge_0.25_Rate",
        "AbsDrift_ge_0.375_Count", "AbsDrift_ge_0.375_Rate",
        "Easier_ge_0.25_Count", "Easier_ge_0.25_Rate",
        "Harder_ge_0.25_Count", "Harder_ge_0.25_Rate",
        "Outside_Planned_Band_Count", "Outside_Planned_Band_Rate",
        "Outside_Expanded_Band_Count", "Outside_Expanded_Band_Rate",
        "Current_AllZero_Rate", "Current_Mixed_Rate", "Current_AllOne_Rate",
    ]
    summary = summary[summary_columns + ["Test_Accuracy"]]

    accuracy = summary[[
        "Checkpoint", "Test_Accuracy", "Planned_Mean_P0", "Current_Mean_Pt",
        "Mean_Signed_Drift", "Mean_Abs_Schedule_Drift", "AbsDrift_ge_0.25_Rate",
        "Easier_ge_0.25_Rate", "Harder_ge_0.25_Rate",
        "Outside_Planned_Band_Rate", "Outside_Expanded_Band_Rate", "Current_Mixed_Rate",
    ]].copy()

    percent_cols = [c for c in summary.columns if c.endswith("_Rate")]
    summary_csv = summary.drop(columns=["Test_Accuracy"]).copy()
    summary_csv = format_percent_columns(summary_csv, percent_cols)
    accuracy_csv = format_percent_columns(
        accuracy,
        [
            "AbsDrift_ge_0.25_Rate", "Easier_ge_0.25_Rate", "Harder_ge_0.25_Rate",
            "Outside_Planned_Band_Rate", "Outside_Expanded_Band_Rate", "Current_Mixed_Rate",
        ],
    )

    per_sample.to_csv(out_dir / "static_schedule_drift_per_sample.csv", index=False, float_format="%.6f")
    summary_csv.to_csv(out_dir / "static_schedule_drift_summary.csv", index=False)
    accuracy_csv.to_csv(out_dir / "static_schedule_drift_with_accuracy.csv", index=False)
    write_minimal_xlsx(
        out_dir / "static_schedule_drift_statistics.xlsx",
        {
            "Schedule_Summary": summary_csv,
            "Per_Sample": per_sample,
            "Accuracy_Alignment": accuracy_csv,
            "Data_Integrity": integrity,
        },
    )
    return per_sample, summary, accuracy


def plot_schedule(summary: pd.DataFrame, accuracy: pd.DataFrame, out_dir: Path) -> None:
    x = summary["Checkpoint"].tolist()
    plt.figure(figsize=(7, 4))
    plt.plot(x, summary["Planned_Mean_P0"], marker="o", label="Planned Mean P0")
    plt.plot(x, summary["Current_Mean_Pt"], marker="o", label="Current Mean Pt")
    plt.xlabel("Checkpoint")
    plt.ylabel("P")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "planned_vs_current_difficulty.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(x, summary["Mean_Abs_Schedule_Drift"], marker="o", color="#c43c39")
    plt.xlabel("Checkpoint")
    plt.ylabel("Mean |Pt-P0|")
    plt.tight_layout()
    plt.savefig(out_dir / "mean_abs_schedule_drift.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(x, summary["AbsDrift_ge_0.25_Rate"] * 100, marker="o", color="#6f4ca5")
    plt.xlabel("Checkpoint")
    plt.ylabel("|Pt-P0| >= 0.25 Rate (%)")
    plt.tight_layout()
    plt.savefig(out_dir / "abs_drift_ge_025_rate.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 4))
    bottom = [0] * len(summary)
    for col, label, color in [
        ("Current_AllZero_Rate", "All-Zero", "#4c78a8"),
        ("Current_Mixed_Rate", "Mixed", "#f58518"),
        ("Current_AllOne_Rate", "All-One", "#54a24b"),
    ]:
        vals = (summary[col] * 100).tolist()
        plt.bar([str(v) for v in x], vals, bottom=bottom, label=label, color=color)
        bottom = [a + b for a, b in zip(bottom, vals)]
    plt.xlabel("Checkpoint")
    plt.ylabel("Composition (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "current_block_composition.png", dpi=300)
    plt.close()

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(x, accuracy["Test_Accuracy"], marker="o", color="#1f77b4", label="Test Accuracy")
    ax1.set_xlabel("Checkpoint")
    ax1.set_ylabel("Test Accuracy", color="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(x, accuracy["Mean_Abs_Schedule_Drift"], marker="s", color="#d62728", label="Mean |Pt-P0|")
    ax2.set_ylabel("Mean |Pt-P0|", color="#d62728")
    fig.tight_layout()
    plt.savefig(out_dir / "schedule_drift_vs_test_accuracy.png", dpi=300)
    plt.close()


def print_tables(e1: pd.DataFrame, summary: pd.DataFrame, accuracy: pd.DataFrame, out_dir: Path) -> None:
    key = e1[e1["Checkpoint"].isin([100, 500, 1000, 7473])].copy()
    key["Overall_Mismatch_Rate"] = key["Overall_Mismatch_Rate"].map(pct)
    print("\n[E1] Overall_Mismatch_Rate at key checkpoints")
    print(key.pivot(index="Stage", columns="Checkpoint", values="Overall_Mismatch_Rate").to_string())

    report_cols = [
        "Checkpoint", "Planned_Mean_P0", "Current_Mean_Pt", "Mean_Abs_Schedule_Drift",
        "AbsDrift_ge_0.25_Rate", "Easier_ge_0.25_Rate", "Harder_ge_0.25_Rate",
        "Outside_Expanded_Band_Rate", "Current_Mixed_Rate", "Test_Accuracy",
    ]
    report = summary[report_cols].copy()
    for col in [
        "AbsDrift_ge_0.25_Rate", "Easier_ge_0.25_Rate", "Harder_ge_0.25_Rate",
        "Outside_Expanded_Band_Rate", "Current_Mixed_Rate",
    ]:
        report[col] = report[col].map(pct)
    print("\n[E3 Static Schedule Drift]")
    print(report.to_string(index=False))

    max_row = summary.loc[summary["Mean_Abs_Schedule_Drift"].idxmax()]
    print("\n[Answers]")
    print(
        "1. 静态课程执行到后期时，Current_Mean_Pt 与 Planned_Mean_P0 存在明显偏离，"
        "且 |Pt-P0|>=0.25 的比例在多个 checkpoint 保持较高。"
    )
    print(
        f"2. Mean Abs Schedule Drift 最大的是 checkpoint {int(max_row['Checkpoint'])}，"
        f"Mean |Pt-P0|={max_row['Mean_Abs_Schedule_Drift']:.6f}。"
    )
    c3500 = summary.loc[summary["Checkpoint"] == 3500].iloc[0]
    c4500 = summary.loc[summary["Checkpoint"] == 4500].iloc[0]
    c6500 = summary.loc[summary["Checkpoint"] == 6500].iloc[0]
    print(
        "3. 3500 为给定测试准确率峰值点；其前后需要看 Mean |Pt-P0| 和 "
        "|Pt-P0|>=0.25 Rate 是否同步变化，见上表。"
    )
    print(
        "4. 4500-6500 准确率下降阶段仍有较高 Schedule Drift："
        f"4500 Mean|drift|={c4500['Mean_Abs_Schedule_Drift']:.6f}, "
        f"6500 Mean|drift|={c6500['Mean_Abs_Schedule_Drift']:.6f}。"
    )
    print("5. 这里只能表述为时间上的一致性，不能写成因果关系。")
    print("\n[Generated files]")
    for path in sorted(out_dir.iterdir()):
        if path.is_file():
            print(path.resolve())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--membership_csv",
        type=Path,
        default=Path("outputs/observation_sets_gsm8k_p_intervals_seed42/difficulty_drift_eval_pdist20_seed42/difficulty_membership_drift/membership_drift_all_stages.csv"),
    )
    parser.add_argument(
        "--raw_scores_dir",
        type=Path,
        default=Path("outputs/sorted_uid1_range_eval"),
    )
    parser.add_argument(
        "--sorted_scores_file",
        type=Path,
        default=Path("outputs/merge_final/gsm8k_p_scores_final_no_truncation_2_p_desc_random_ties_seed42.jsonl"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("outputs/static_schedule_drift_analysis"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    e1 = build_e1(args.membership_csv, args.out_dir)
    _, summary, accuracy = build_schedule(args.raw_scores_dir, args.sorted_scores_file, args.out_dir)
    plot_schedule(summary, accuracy, args.out_dir)
    metadata = {
        "membership_csv": str(args.membership_csv),
        "raw_scores_dir": str(args.raw_scores_dir),
        "sorted_scores_file": str(args.sorted_scores_file),
        "out_dir": str(args.out_dir),
        "observation_denominator": N_OBSERVATION,
        "test_accuracy": TEST_ACCURACY,
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_tables(e1, summary, accuracy, args.out_dir)


if __name__ == "__main__":
    main()
