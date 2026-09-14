import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Sequence, Tuple

from analyze_rank_stability import circle, line, png_canvas, put, write_png
from analyze_vanilla_grpo_difficulty_drift import write_xlsx


CHECKPOINTS = [0, 10, 20, 30, 40, 50, 100, 250, 500, 1000, 1500, 2000, 4000, 7473]
THRESHOLDS = [0.0625, 0.125, 0.25]
MAIN_DELTA = 0.125
DEFAULT_INPUT_DIR = Path(
    "outputs/observation_sets_gsm8k_p_intervals_seed42/"
    "difficulty_drift_eval_pdist20_seed42"
)
DEFAULT_ORIGINAL_NAME = "gsm8k_p_scores_final_no_truncation_2_sampled_20pct.jsonl"
STAGES = [
    ("B1", "[0.625,0.875]", 0.625, 0.875),
    ("B2", "[0.500,0.750]", 0.500, 0.750),
    ("B3", "[0.375,0.625]", 0.375, 0.625),
    ("B4", "[0.250,0.500]", 0.250, 0.500),
    ("B5", "[0.125,0.375]", 0.125, 0.375),
]
EPS = 1e-9


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze non-monotonic p-score drift, direction reversals, and interval re-entry."
    )
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--original_file", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--sample_id_field", default="uid")
    parser.add_argument("--p_field", default="p")
    parser.add_argument("--rollout_field", default="_num_answers")
    parser.add_argument("--expected_rollouts", type=int, default=32)
    parser.add_argument("--expected_n", type=int, default=1495)
    return parser.parse_args()


def checkpoint_file(input_dir: Path, original_file: Path, checkpoint: int) -> Path:
    if checkpoint == 0:
        return original_file
    matches = sorted(input_dir.glob(f"*checkpoint-{checkpoint}.jsonl"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one checkpoint-{checkpoint} file, got {len(matches)}")
    return matches[0]


def fmt(value: float) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def pct(value: float) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    return f"{value * 100:.2f}%"


def read_checkpoint(path: Path, checkpoint: int, args) -> Tuple[Dict[str, float], Dict[str, object]]:
    values: Dict[str, float] = {}
    duplicate = missing_id = missing_p = bad_rollout = abnormal_p = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if args.sample_id_field not in row:
                missing_id += 1
                continue
            if args.p_field not in row:
                missing_p += 1
                continue
            sid = str(row[args.sample_id_field])
            if sid in values:
                duplicate += 1
                continue
            p = float(row[args.p_field])
            if row.get(args.rollout_field) != args.expected_rollouts:
                bad_rollout += 1
            correct = round(p * args.expected_rollouts)
            if not (-EPS <= p <= 1 + EPS and abs(p - correct / args.expected_rollouts) <= 1e-8):
                abnormal_p += 1
            values[sid] = p
    return values, {
        "checkpoint": checkpoint,
        "file": str(path),
        "sample_count": len(values),
        "duplicate_count": duplicate,
        "missing_id_count": missing_id,
        "missing_p_count": missing_p,
        "bad_rollout_count": bad_rollout,
        "abnormal_p_count": abnormal_p,
    }


def load_all(args):
    original = args.original_file or (args.input_dir / DEFAULT_ORIGINAL_NAME)
    files = [checkpoint_file(args.input_dir, original, ckpt) for ckpt in CHECKPOINTS]
    data, integrity = {}, []
    for ckpt, path in zip(CHECKPOINTS, files):
        values, row = read_checkpoint(path, ckpt, args)
        data[ckpt] = values
        integrity.append(row)
    id_sets = [set(data[ckpt]) for ckpt in CHECKPOINTS]
    common = sorted(set.intersection(*id_sets), key=lambda x: int(x) if x.isdigit() else x)
    union = set.union(*id_sets)
    for row in integrity:
        ids = set(data[int(row["checkpoint"])])
        row["missing_vs_union"] = len(union - ids)
        row["effective_aligned_n"] = len(common)
    return data, integrity, common, files


def directions(values: Sequence[float], delta: float) -> List[int]:
    out = []
    for left, right in zip(values, values[1:]):
        d = right - left
        if d >= delta - EPS:
            out.append(1)
        elif d <= -delta + EPS:
            out.append(-1)
        else:
            out.append(0)
    return out


def reversal_count(values: Sequence[float], delta: float) -> int:
    signs = [d for d in directions(values, delta) if d != 0]
    return sum(a != b for a, b in zip(signs, signs[1:]))


def in_interval(p: float, lower: float, upper: float) -> bool:
    return lower - EPS <= p <= upper + EPS


def reentry_count(membership: Sequence[int]) -> int:
    count = 0
    was_in = bool(membership[0])
    exited_after_in = False
    for m in membership[1:]:
        if was_in and not m:
            exited_after_in = True
        elif exited_after_in and m:
            count += 1
            exited_after_in = False
            was_in = True
        elif m:
            was_in = True
    return count


def has_initial_exit_and_reentry(membership: Sequence[int]) -> Tuple[bool, bool]:
    if not membership[0]:
        return False, False
    exited = False
    reentered = False
    for m in membership[1:]:
        if not m:
            exited = True
        elif exited and m:
            reentered = True
            break
    return exited, reentered


def build_per_sample(data, sample_ids):
    rows = []
    for sid in sample_ids:
        vals = [data[ckpt][sid] for ckpt in CHECKPOINTS]
        r = max(vals) - min(vals)
        tv = sum(abs(b - a) for a, b in zip(vals, vals[1:]))
        excess = max(0.0, tv - r)
        ratio = math.nan if r <= EPS else tv / r
        row = {
            "sample_id": sid,
            "Range": r,
            "Total_Variation": tv,
            "Oscillation_Excess": excess,
            "Oscillation_Ratio": ratio,
        }
        for delta in THRESHOLDS:
            rc = reversal_count(vals, delta)
            label = str(delta).rstrip("0").rstrip(".")
            row[f"Reversal_Count_delta_{label}"] = rc
            row[f"NonMonotonic_delta_{label}"] = rc >= 1
        for ckpt, val in zip(CHECKPOINTS, vals):
            row[f"P_{ckpt}"] = val
        rows.append(row)
    return rows


def summarize_reversals(per_rows, delta: float):
    label = str(delta).rstrip("0").rstrip(".")
    key = f"Reversal_Count_delta_{label}"
    counts = {"0": 0, "1": 0, "2": 0, "3": 0, ">=4": 0}
    for row in per_rows:
        rc = int(row[key])
        counts[str(rc) if rc < 4 else ">=4"] += 1
    n = len(per_rows)
    rows = []
    for bucket in ["0", "1", "2", "3", ">=4"]:
        rows.append({"Delta": delta, "Reversal_Count_Bucket": bucket, "Count": counts[bucket], "Rate": pct(counts[bucket] / n)})
    return rows


def threshold_sensitivity(per_rows):
    rows = []
    n = len(per_rows)
    for delta in THRESHOLDS:
        label = str(delta).rstrip("0").rstrip(".")
        count = sum(int(row[f"Reversal_Count_delta_{label}"]) >= 1 for row in per_rows)
        rows.append({"Delta": delta, "NonMonotonic_Count": count, "NonMonotonicRate": pct(count / n)})
    return rows


def oscillation_summary(per_rows):
    ranges = [float(r["Range"]) for r in per_rows]
    tvs = [float(r["Total_Variation"]) for r in per_rows]
    ex = [float(r["Oscillation_Excess"]) for r in per_rows]
    ratios = [float(r["Oscillation_Ratio"]) for r in per_rows if not math.isnan(float(r["Oscillation_Ratio"]))]
    rows = [
        {"Metric": "N", "Value": len(per_rows), "Threshold": "", "Count": "", "Rate": ""},
        {"Metric": "Mean Range", "Value": fmt(mean(ranges)), "Threshold": "", "Count": "", "Rate": ""},
        {"Metric": "Median Range", "Value": fmt(median(ranges)), "Threshold": "", "Count": "", "Rate": ""},
        {"Metric": "Mean TV", "Value": fmt(mean(tvs)), "Threshold": "", "Count": "", "Rate": ""},
        {"Metric": "Median TV", "Value": fmt(median(tvs)), "Threshold": "", "Count": "", "Rate": ""},
        {"Metric": "Mean OscillationExcess", "Value": fmt(mean(ex)), "Threshold": "", "Count": "", "Rate": ""},
        {"Metric": "Median OscillationExcess", "Value": fmt(median(ex)), "Threshold": "", "Count": "", "Rate": ""},
        {"Metric": "Mean OscillationRatio", "Value": fmt(mean(ratios)), "Threshold": "", "Count": "", "Rate": ""},
        {"Metric": "Median OscillationRatio", "Value": fmt(median(ratios)), "Threshold": "", "Count": "", "Rate": ""},
    ]
    for th in [0.125, 0.25, 0.5]:
        count = sum(v >= th - EPS for v in ex)
        rows.append({"Metric": "OscillationExcess threshold", "Value": "", "Threshold": f">={th}", "Count": count, "Rate": pct(count / len(ex))})
    return rows


def reentry_stats(data, sample_ids):
    initial_rows, general_rows, detail_rows = [], [], []
    for stage, interval, lo, hi in STAGES:
        initial_size = ever_exited = reentered_initial = 0
        ever_in = general_reentered = 0
        count_buckets = {"0": 0, "1": 0, "2": 0, ">=3": 0}
        for sid in sample_ids:
            vals = [data[ckpt][sid] for ckpt in CHECKPOINTS]
            mem = [1 if in_interval(v, lo, hi) else 0 for v in vals]
            rc = reentry_count(mem)
            bucket = str(rc) if rc < 3 else ">=3"
            count_buckets[bucket] += 1
            if any(mem):
                ever_in += 1
            if rc >= 1:
                general_reentered += 1
            if mem[0]:
                initial_size += 1
                exited, reentered = has_initial_exit_and_reentry(mem)
                if exited:
                    ever_exited += 1
                if reentered:
                    reentered_initial += 1
            detail_rows.append({
                "Stage": stage,
                "Interval": interval,
                "sample_id": sid,
                "Initial_Member": bool(mem[0]),
                "Ever_In_Interval": bool(any(mem)),
                "Reentry_Count": rc,
                "General_Reentry": rc >= 1,
                "Membership_Sequence": "".join(map(str, mem)),
            })
        initial_rows.append({
            "Stage": stage,
            "Interval": interval,
            "Initial_Size": initial_size,
            "Ever_Exited_Count": ever_exited,
            "Ever_Exited_Rate": pct(ever_exited / initial_size) if initial_size else "NaN",
            "Reentered_Count": reentered_initial,
            "Reentry_Rate": pct(reentered_initial / ever_exited) if ever_exited else "NaN",
        })
        grow = {
            "Stage": stage,
            "Interval": interval,
            "Ever_In_Interval_Count": ever_in,
            "General_Reentry_Count": general_reentered,
            "General_Reentry_Rate": pct(general_reentered / ever_in) if ever_in else "NaN",
        }
        for bucket in ["0", "1", "2", ">=3"]:
            grow[f"Reentry_Count_{bucket}"] = count_buckets[bucket]
            grow[f"Reentry_Rate_{bucket}"] = pct(count_buckets[bucket] / len(sample_ids))
        general_rows.append(grow)
    return initial_rows, general_rows, detail_rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]], cols: Sequence[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cols), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def draw_bar_png(path: Path, labels: Sequence[str], values: Sequence[float], ymax: float = 100):
    w, h = 1200, 850
    c = png_canvas(w, h)
    left, right, top, bottom = 95, 55, 65, 105
    axis = (35, 35, 35)
    grid = (225, 225, 225)
    for k in range(6):
        y = top + k * (h - top - bottom) / 5
        line(c, w, h, left, y, w - right, y, grid, 1)
    line(c, w, h, left, top, left, h - bottom, axis, 2)
    line(c, w, h, left, h - bottom, w - right, h - bottom, axis, 2)
    slot = (w - left - right) / len(values)
    bar_w = slot * 0.55
    color = (49, 130, 189)
    for i, v in enumerate(values):
        x0 = round(left + i * slot + (slot - bar_w) / 2)
        x1 = round(x0 + bar_w)
        y0 = round(top + (1 - min(v, ymax) / ymax) * (h - top - bottom))
        for x in range(x0, x1):
            for y in range(y0, h - bottom):
                put(c, w, h, x, y, color)
    write_png(path, w, h, c)


def draw_scatter_png(path: Path, xs: Sequence[float], ys: Sequence[float]):
    w, h = 1050, 900
    c = png_canvas(w, h)
    left, right, top, bottom = 90, 60, 60, 90
    max_v = max(max(xs), max(ys), 1e-9)
    def x_at(v): return left + v / max_v * (w - left - right)
    def y_at(v): return top + (1 - v / max_v) * (h - top - bottom)
    line(c, w, h, left, h - bottom, w - right, top, (30, 30, 30), 2)
    line(c, w, h, left, top, left, h - bottom, (35, 35, 35), 2)
    line(c, w, h, left, h - bottom, w - right, h - bottom, (35, 35, 35), 2)
    for x, y in zip(xs, ys):
        circle(c, w, h, x_at(x), y_at(y), 3, (214, 96, 77))
    write_png(path, w, h, c)


def draw_trajectories_png(path: Path, reps: Sequence[Dict[str, object]]):
    w, h = 1400, 900
    c = png_canvas(w, h)
    left, right, top, bottom = 90, 70, 60, 90
    colors = [(27, 158, 119), (217, 95, 2), (117, 112, 179)]
    def x_at(i): return left + i * (w-left-right)/(len(CHECKPOINTS)-1)
    def y_at(v): return top + (1-v)*(h-top-bottom)
    line(c,w,h,left,top,left,h-bottom,(35,35,35),2)
    line(c,w,h,left,h-bottom,w-right,h-bottom,(35,35,35),2)
    for idx,row in enumerate(reps):
        vals=[float(row[f"P_{ckpt}"]) for ckpt in CHECKPOINTS]
        color=colors[idx//4]
        pts=[(x_at(i),y_at(v)) for i,v in enumerate(vals)]
        for (x0,y0),(x1,y1) in zip(pts,pts[1:]):
            line(c,w,h,x0,y0,x1,y1,color,2)
        for x,y in pts:
            circle(c,w,h,x,y,4,color)
    write_png(path,w,h,c)


def representative_samples(per_rows):
    chosen_ids = set()
    reps = []
    def add(group, reason, candidates):
        for row in candidates:
            if row["sample_id"] in chosen_ids:
                continue
            out = dict(row)
            out["Group"] = group
            out["Selection_Reason"] = reason
            reps.append(out)
            chosen_ids.add(row["sample_id"])
            if sum(r["Group"] == group for r in reps) >= 4:
                break
    add("A_high_nonmonotonic", "max reversal_count_delta_0.125, then max Oscillation_Excess",
        sorted(per_rows, key=lambda r: (int(r["Reversal_Count_delta_0.125"]), float(r["Oscillation_Excess"]), float(r["Range"]), r["sample_id"]), reverse=True))
    add("B_large_range_monotonic", "Reversal_Count_delta_0.125=0, then max Range",
        sorted([r for r in per_rows if int(r["Reversal_Count_delta_0.125"]) == 0], key=lambda r: (float(r["Range"]), float(r["Total_Variation"]), r["sample_id"]), reverse=True))
    add("C_stable", "min Range",
        sorted(per_rows, key=lambda r: (float(r["Range"]), float(r["Total_Variation"]), r["sample_id"])))
    return reps


def main():
    args = parse_args()
    out_dir = args.output_dir or (args.input_dir / "difficulty_nonmonotonic_drift")
    out_dir.mkdir(parents=True, exist_ok=True)
    data, integrity, sample_ids, files = load_all(args)
    print("==== Data Integrity Check ====")
    print("checkpoint\tsample_count\tmissing_vs_union\tduplicate_count\tbad_rollout_count\tabnormal_p_count")
    for row in integrity:
        print(f"{row['checkpoint']}\t{row['sample_count']}\t{row['missing_vs_union']}\t{row['duplicate_count']}\t{row['bad_rollout_count']}\t{row['abnormal_p_count']}")
    print(f"Aligned effective N: {len(sample_ids)}")

    per_rows = build_per_sample(data, sample_ids)
    reversal_rows = []
    for delta in THRESHOLDS:
        reversal_rows.extend(summarize_reversals(per_rows, delta))
    sensitivity_rows = threshold_sensitivity(per_rows)
    osc_rows = oscillation_summary(per_rows)
    initial_rows, general_rows, detail_rows = reentry_stats(data, sample_ids)
    reps = representative_samples(per_rows)

    per_cols = ["sample_id","Range","Total_Variation","Oscillation_Excess","Oscillation_Ratio",
                "Reversal_Count_delta_0.0625","Reversal_Count_delta_0.125","Reversal_Count_delta_0.25",
                "NonMonotonic_delta_0.0625","NonMonotonic_delta_0.125","NonMonotonic_delta_0.25",
                *[f"P_{ckpt}" for ckpt in CHECKPOINTS]]
    reversal_cols = ["Delta","Reversal_Count_Bucket","Count","Rate"]
    sensitivity_cols = ["Delta","NonMonotonic_Count","NonMonotonicRate"]
    osc_cols = ["Metric","Value","Threshold","Count","Rate"]
    initial_cols = ["Stage","Interval","Initial_Size","Ever_Exited_Count","Ever_Exited_Rate","Reentered_Count","Reentry_Rate"]
    general_cols = ["Stage","Interval","Ever_In_Interval_Count","General_Reentry_Count","General_Reentry_Rate",
                    "Reentry_Count_0","Reentry_Rate_0","Reentry_Count_1","Reentry_Rate_1","Reentry_Count_2","Reentry_Rate_2","Reentry_Count_>=3","Reentry_Rate_>=3"]
    detail_cols = ["Stage","Interval","sample_id","Initial_Member","Ever_In_Interval","Reentry_Count","General_Reentry","Membership_Sequence"]
    rep_cols = ["Group","Selection_Reason",*per_cols]
    integrity_cols = ["checkpoint","file","sample_count","missing_vs_union","duplicate_count","missing_id_count","missing_p_count","bad_rollout_count","abnormal_p_count","effective_aligned_n"]

    paths = {
        "per": out_dir/"per_sample_nonmonotonic_drift.csv",
        "rev": out_dir/"reversal_statistics.csv",
        "sens": out_dir/"nonmonotonic_threshold_sensitivity.csv",
        "osc": out_dir/"oscillation_statistics.csv",
        "init": out_dir/"initial_reentry_statistics.csv",
        "gen": out_dir/"general_reentry_statistics.csv",
        "detail": out_dir/"per_sample_reentry_details.csv",
        "rep": out_dir/"representative_samples.csv",
        "integrity": out_dir/"data_integrity_check.csv",
        "xlsx": out_dir/"nonmonotonic_drift_statistics.xlsx",
    }
    write_csv(paths["per"], per_rows, per_cols)
    write_csv(paths["rev"], reversal_rows, reversal_cols)
    write_csv(paths["sens"], sensitivity_rows, sensitivity_cols)
    write_csv(paths["osc"], osc_rows, osc_cols)
    write_csv(paths["init"], initial_rows, initial_cols)
    write_csv(paths["gen"], general_rows, general_cols)
    write_csv(paths["detail"], detail_rows, detail_cols)
    write_csv(paths["rep"], reps, rep_cols)
    write_csv(paths["integrity"], integrity, integrity_cols)
    write_xlsx(paths["xlsx"], [
        ("Overall_Summary", osc_rows, osc_cols),
        ("Per_Sample", per_rows, per_cols),
        ("Reversal_Statistics", reversal_rows, reversal_cols),
        ("Threshold_Sensitivity", sensitivity_rows, sensitivity_cols),
        ("Initial_Reentry", initial_rows, initial_cols),
        ("General_Reentry", general_rows, general_cols),
        ("Reentry_Details", detail_rows, detail_cols),
        ("Representative_Samples", reps, rep_cols),
    ])

    main_rev = [r for r in reversal_rows if abs(float(r["Delta"]) - MAIN_DELTA) < EPS]
    draw_bar_png(out_dir/"direction_reversal_distribution.png", [r["Reversal_Count_Bucket"] for r in main_rev], [float(r["Rate"].rstrip("%")) for r in main_rev])
    draw_bar_png(out_dir/"nonmonotonic_threshold_sensitivity.png", [str(r["Delta"]) for r in sensitivity_rows], [float(r["NonMonotonicRate"].rstrip("%")) for r in sensitivity_rows])
    draw_scatter_png(out_dir/"range_vs_total_variation.png", [float(r["Range"]) for r in per_rows], [float(r["Total_Variation"]) for r in per_rows])
    draw_bar_png(out_dir/"initial_reentry_rate.png", [r["Stage"] for r in initial_rows], [0 if r["Reentry_Rate"]=="NaN" else float(r["Reentry_Rate"].rstrip("%")) for r in initial_rows])
    draw_trajectories_png(out_dir/"representative_sample_trajectories.png", reps)

    metadata = {
        "input_files": [str(p) for p in files],
        "effective_n": len(sample_ids),
        "checkpoints": CHECKPOINTS,
        "thresholds": THRESHOLDS,
        "stages": STAGES,
    }
    meta_path = out_dir/"metadata.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    sens = {float(r["Delta"]): r for r in sensitivity_rows}
    osc_map = {r["Metric"]: r["Value"] for r in osc_rows}
    ex025 = next(r for r in osc_rows if r["Metric"] == "OscillationExcess threshold" and r["Threshold"] == ">=0.25")
    rev_ge2 = sum(int(r["Reversal_Count_delta_0.125"]) >= 2 for r in per_rows)
    max_reentry = max(initial_rows, key=lambda r: -1 if r["Reentry_Rate"] == "NaN" else float(r["Reentry_Rate"].rstrip("%")))
    print("==== Summary ====")
    print(f"Effective N: {len(sample_ids)}")
    for delta in THRESHOLDS:
        print(f"NonMonotonicRate delta={delta}: {sens[delta]['NonMonotonicRate']}")
    print(f"Mean Range: {osc_map['Mean Range']}")
    print(f"Median Range: {osc_map['Median Range']}")
    print(f"Mean Total Variation: {osc_map['Mean TV']}")
    print(f"Median Total Variation: {osc_map['Median TV']}")
    print(f"Mean OscillationExcess: {osc_map['Mean OscillationExcess']}")
    print(f"Median OscillationExcess: {osc_map['Median OscillationExcess']}")
    print(f"OscillationExcess>=0.25 Rate: {ex025['Rate']}")
    print("Initial Re-entry Rates:")
    for row in initial_rows:
        print(f"  {row['Stage']} {row['Interval']}: {row['Reentry_Rate']}")
    print(f"Most obvious initial re-entry stage: {max_reentry['Stage']} {max_reentry['Interval']} {max_reentry['Reentry_Rate']}")
    print(f"Samples with Reversal_Count_delta_0.125 >= 2: {rev_ge2} ({pct(rev_ge2 / len(sample_ids))})")
    print("==== Outputs ====")
    for p in [*paths.values(), meta_path, out_dir/"direction_reversal_distribution.png", out_dir/"nonmonotonic_threshold_sensitivity.png", out_dir/"range_vs_total_variation.png", out_dir/"initial_reentry_rate.png", out_dir/"representative_sample_trajectories.png"]:
        print(p.resolve())


if __name__ == "__main__":
    main()
