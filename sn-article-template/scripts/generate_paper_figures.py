#!/usr/bin/env python3
"""Generate manuscript figures from local experiment data and reported tables."""

import csv
import json
import math
from pathlib import Path


TEMPLATE_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = TEMPLATE_DIR.parent
FIG_DIR = TEMPLATE_DIR / "figures"
META_DIR = FIG_DIR / "metadata"
TRACK_DIR = REPO_DIR / "outputs" / "difficulty_drift_eval_pdist20_seed42"

CHECKPOINTS = [0, 10, 20, 30, 40, 50, 100, 250, 500, 1000, 1500, 2000, 4000, 7473]

COLORS = {
    "blue": (0.000, 0.365, 0.655),
    "orange": (0.835, 0.369, 0.000),
    "green": (0.000, 0.520, 0.390),
    "purple": (0.455, 0.220, 0.620),
    "vermillion": (0.720, 0.185, 0.130),
    "sky": (0.290, 0.610, 0.770),
    "black": (0.090, 0.090, 0.090),
    "gray": (0.430, 0.430, 0.430),
    "lightgray": (0.875, 0.875, 0.875),
    "panel": (0.965, 0.970, 0.975),
    "panel2": (0.985, 0.985, 0.985),
}

FS_AXIS = 8.0
FS_TICK = 7.0
FS_PANEL = 9.0
LW_AXIS = 0.65
LW_GRID = 0.30
LW_LINE = 1.05
MS = 2.1


def esc(text):
    return str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


class PDF:
    def __init__(self, path, width=520, height=340):
        self.path = Path(path)
        self.w = width
        self.h = height
        self.ops = []

    def color(self, rgb, stroke=True):
        op = "RG" if stroke else "rg"
        self.ops.append(f"{rgb[0]:.3f} {rgb[1]:.3f} {rgb[2]:.3f} {op}")

    def line_width(self, value):
        self.ops.append(f"{value:.3f} w")

    def line(self, x1, y1, x2, y2, rgb=COLORS["black"], width=0.8):
        self.color(rgb, True)
        self.line_width(width)
        self.ops.append(f"{x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S")

    def dashed_line(self, x1, y1, x2, y2, rgb=COLORS["gray"], width=0.8, dash="3 3"):
        self.color(rgb, True)
        self.line_width(width)
        self.ops.append(f"[{dash}] 0 d")
        self.ops.append(f"{x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S")
        self.ops.append("[] 0 d")

    def rect(self, x, y, w, h, stroke=COLORS["black"], fill=None, width=0.8):
        if fill is not None:
            self.color(fill, False)
            self.ops.append(f"{x:.2f} {y:.2f} {w:.2f} {h:.2f} re f")
        if stroke is not None:
            self.color(stroke, True)
            self.line_width(width)
            self.ops.append(f"{x:.2f} {y:.2f} {w:.2f} {h:.2f} re S")

    def circle(self, x, y, r, stroke=COLORS["black"], fill=None, width=0.8):
        k = 0.5522847498
        c = r * k
        self.ops.append(f"{x + r:.2f} {y:.2f} m")
        self.ops.append(
            f"{x + r:.2f} {y + c:.2f} {x + c:.2f} {y + r:.2f} {x:.2f} {y + r:.2f} c"
        )
        self.ops.append(
            f"{x - c:.2f} {y + r:.2f} {x - r:.2f} {y + c:.2f} {x - r:.2f} {y:.2f} c"
        )
        self.ops.append(
            f"{x - r:.2f} {y - c:.2f} {x - c:.2f} {y - r:.2f} {x:.2f} {y - r:.2f} c"
        )
        self.ops.append(
            f"{x + c:.2f} {y - r:.2f} {x + r:.2f} {y - c:.2f} {x + r:.2f} {y:.2f} c"
        )
        if fill is not None:
            self.color(fill, False)
            self.ops.append("f")
        if stroke is not None:
            self.color(stroke, True)
            self.line_width(width)
            self.ops.append("S")

    def square_marker(self, x, y, r, stroke=COLORS["black"], fill=None, width=0.8):
        self.rect(x - r, y - r, 2 * r, 2 * r, stroke=stroke, fill=fill, width=width)

    def polyline(self, pts, rgb=COLORS["blue"], width=1.0, dash=None):
        if len(pts) < 2:
            return
        self.color(rgb, True)
        self.line_width(width)
        self.ops.append("[] 0 d" if dash is None else f"[{dash}] 0 d")
        self.ops.append(f"{pts[0][0]:.2f} {pts[0][1]:.2f} m")
        for x, y in pts[1:]:
            self.ops.append(f"{x:.2f} {y:.2f} l")
        self.ops.append("S")
        self.ops.append("[] 0 d")

    def text(self, x, y, text, size=9, rgb=COLORS["black"], align="left"):
        width = len(str(text)) * size * 0.47
        if align == "center":
            x -= width / 2
        elif align == "right":
            x -= width
        self.color(rgb, False)
        self.ops.append(f"BT /F1 {size:.1f} Tf {x:.2f} {y:.2f} Td ({esc(text)}) Tj ET")

    def arrow(self, x1, y1, x2, y2, rgb=COLORS["gray"], width=0.8):
        self.line(x1, y1, x2, y2, rgb, width)
        ang = math.atan2(y2 - y1, x2 - x1)
        for a in (ang + 2.55, ang - 2.55):
            self.line(x2, y2, x2 + 7 * math.cos(a), y2 + 7 * math.sin(a), rgb, width)

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        content = "\n".join(self.ops).encode("latin-1", "replace")
        objs = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.w} {self.h}] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>".encode(),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        ]
        out = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
        xref = [0]
        for i, obj in enumerate(objs, 1):
            xref.append(len(out))
            out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
        start = len(out)
        out += f"xref\n0 {len(objs)+1}\n0000000000 65535 f \n".encode()
        for off in xref[1:]:
            out += f"{off:010d} 00000 n \n".encode()
        out += f"trailer << /Size {len(objs)+1} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n".encode()
        self.path.write_bytes(out)


def checkpoint_path(ckpt):
    if ckpt == 0:
        return TRACK_DIR / "gsm8k_p_scores_final_no_truncation_2_sampled_20pct.jsonl"
    return TRACK_DIR / f"gsm8k_p_scores_sample20_checkpoint-{ckpt}.jsonl"


def load_tracking():
    data = {}
    files = {}
    for ckpt in CHECKPOINTS:
        path = checkpoint_path(ckpt)
        files[ckpt] = str(path)
        vals = {}
        with path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                vals[str(row["uid"])] = float(row["p"])
        data[ckpt] = vals
    common = sorted(set.intersection(*(set(v) for v in data.values())), key=lambda x: int(x))
    return data, common, files


def meaningful_reversals(vals, delta=0.125):
    signs = []
    for a, b in zip(vals, vals[1:]):
        d = b - a
        if abs(d) >= delta - 1e-12:
            signs.append(1 if d > 0 else -1)
    return sum(a != b for a, b in zip(signs, signs[1:]))


def select_representative_samples(data, uids):
    bins = [(0.0, 0.375), (0.375, 0.625), (0.625, 1.001)]
    selected = []
    used = set()
    for left, right in bins:
        rows = []
        for uid in uids:
            vals = [data[c][uid] for c in CHECKPOINTS]
            rev = meaningful_reversals(vals)
            span = max(vals) - min(vals)
            if left <= vals[0] < right and rev >= 1:
                rows.append((span, rev, uid, vals))
        rows.sort(key=lambda x: (-x[0], -x[1], int(x[2])))
        for span, rev, uid, vals in rows[:3]:
            if uid not in used:
                selected.append({"uid": uid, "p0": vals[0], "span": span, "reversals": rev, "values": vals})
                used.add(uid)
    # Keep two high-span reversed trajectories from each initial-solvability bin.
    return selected[:2] + selected[3:5] + selected[6:8]


def draw_axes(pdf, x0, y0, w, h, xticks, yticks, xlim, ylim, xlabel, ylabel, xticklabels=None):
    for yv in yticks:
        y = y0 + (yv - ylim[0]) / (ylim[1] - ylim[0]) * h
        pdf.line(x0, y, x0 + w, y, COLORS["lightgray"], LW_GRID)
        pdf.text(x0 - 6, y - 3, f"{yv:.2f}" if ylim[1] <= 1.1 else f"{yv:.0f}", FS_TICK, align="right")
    pdf.line(x0, y0, x0 + w, y0, COLORS["black"], LW_AXIS)
    pdf.line(x0, y0, x0, y0 + h, COLORS["black"], LW_AXIS)
    labels = xticklabels if xticklabels is not None else [str(x) for x in xticks]
    for xv, label in zip(xticks, labels):
        x = x0 + (xv - xlim[0]) / (xlim[1] - xlim[0]) * w
        pdf.line(x, y0, x, y0 - 3, COLORS["black"], 0.45)
        pdf.text(x, y0 - 13, label, FS_TICK, align="center")
    pdf.text(x0 + w / 2, y0 - 28, xlabel, FS_AXIS, align="center")
    if ylabel:
        pdf.text(x0, y0 + h + 10, ylabel, FS_AXIS, align="left")

    def scale(xv, yv):
        x = x0 + (xv - xlim[0]) / (xlim[1] - xlim[0]) * w
        y = y0 + (yv - ylim[0]) / (ylim[1] - ylim[0]) * h
        return x, y

    return scale


def figure1():
    pdf = PDF(FIG_DIR / "fig1_framework.pdf", 760, 330)

    def group(x, w, title):
        pdf.rect(x, 42, w, 248, COLORS["lightgray"], COLORS["panel2"], 0.55)
        pdf.text(x + 12, 273, title, 9.5)

    def box(x, y, w, h, text, fill=COLORS["panel"]):
        pdf.rect(x, y, w, h, COLORS["gray"], fill, 0.65)
        for i, line in enumerate(text.split("\n")):
            pdf.text(x + w / 2, y + h - 15 - 10 * i, line, 7.8, align="center")

    group(24, 210, "I. Difficulty Characterization")
    group(264, 270, "II. Online Curricula")
    group(564, 172, "III. Empirical Findings")

    box(60, 228, 100, 28, "Initial policy")
    box(60, 178, 100, 28, "Estimate P0(x)")
    box(48, 120, 124, 36, "Policy-dependent\ndifficulty dynamics")
    box(48, 62, 124, 36, "Static-curriculum\ndifficulty mismatch")
    pdf.arrow(110, 228, 110, 206)
    pdf.arrow(110, 178, 110, 156)
    pdf.arrow(110, 120, 110, 98)

    box(342, 206, 116, 32, "Estimate current\nPt(x)")
    box(292, 122, 112, 36, "Fixed-medium\nB* = B3")
    box(414, 122, 112, 36, "Staged curriculum\nB1 -> ... -> B5")
    pdf.arrow(400, 206, 348, 158)
    pdf.arrow(400, 206, 470, 158)

    findings = [
        ("policy-dependent", "difficulty dynamics"),
        ("persistent medium targeting", "is insufficient"),
        ("moderate B4 gives the", "strongest observed result"),
        ("further B5 progression", "reduces performance"),
    ]
    for i, text in enumerate(findings):
        y = 225 - i * 43
        pdf.circle(590, y + 10, 2.0, stroke=COLORS["blue"], fill=COLORS["blue"], width=0.4)
        pdf.text(602, y + 10, text[0], 7.2)
        pdf.text(602, y, text[1], 7.2)
    pdf.arrow(172, 138, 342, 222)
    pdf.arrow(458, 222, 578, 226)
    pdf.save()


def figure_motivation():
    import os
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-grpo-b")

    import matplotlib.pyplot as plt
    from matplotlib import patches

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 7.2,
        "axes.titlesize": 6.6,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })

    blue = "#5f8fb3"
    blue_light = "#e8f1f7"
    teal = "#75a88e"
    teal_light = "#e8f3ee"
    orange = "#c58b67"
    orange_light = "#f4ebe5"
    gray = "#6b7280"
    gray_light = "#f2f4f5"
    panel_edge = "#cfd8df"
    dark = "#111827"
    correct = "#4d8f82"
    wrong = "#c78362"

    def add_box(ax, xy, w, h, text, fc, ec=panel_edge, lw=0.75, fs=7.0,
                weight="normal", radius=0.018, color=dark):
        box = patches.FancyBboxPatch(
            xy, w, h,
            boxstyle=f"round,pad=0.012,rounding_size={radius}",
            linewidth=lw, edgecolor=ec, facecolor=fc,
            transform=ax.transAxes, clip_on=False,
        )
        ax.add_patch(box)
        ax.text(
            xy[0] + w / 2, xy[1] + h / 2, text,
            ha="center", va="center", fontsize=fs, fontweight=weight,
            color=color, transform=ax.transAxes, linespacing=1.18,
        )
        return box

    def add_arrow(ax, start, end, color=gray, lw=0.9, style="-|>", rad=0.0):
        arrow = patches.FancyArrowPatch(
            start, end, arrowstyle=style, mutation_scale=8.5,
            linewidth=lw, color=color, connectionstyle=f"arc3,rad={rad}",
            transform=ax.transAxes, clip_on=False,
        )
        ax.add_patch(arrow)
        return arrow

    fig, axes = plt.subplots(
        1, 3, figsize=(7.35, 3.55), dpi=300,
        gridspec_kw={"width_ratios": [1.38, 0.90, 1.42]},
    )
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    ax = axes[0]
    ax.set_title("(a) Learning signal and solvability", loc="left", pad=5)
    regions = [
        ("Easy", "weak\nsignal", 5, blue_light, blue),
        ("Medium", "informative\nsignal", 3, teal_light, teal),
        ("Hard", "limited\nsignal", 1, orange_light, orange),
    ]
    box_w = 0.325
    xs = [0.002, 0.3375, 0.673]
    for x, (title, note, n_correct, fc, ec) in zip(xs, regions):
        cx0 = x + box_w / 2
        add_box(ax, (x, 0.225), box_w, 0.625, "", fc, ec, lw=0.8)
        ax.text(cx0, 0.755, title, ha="center", va="center",
                fontsize=7.5, fontweight="bold", color=dark, transform=ax.transAxes)
        for i in range(6):
            cx = cx0 - 0.078 + (i % 3) * 0.078
            cy = 0.585 - (i // 3) * 0.090
            fill = correct if i < n_correct else wrong
            ax.scatter([cx], [cy], s=38, marker="o", facecolor=fill,
                       edgecolor="white", linewidth=0.55, transform=ax.transAxes, zorder=3)
            ax.text(cx, cy - 0.003, r"$\checkmark$" if i < n_correct else r"$\times$",
                    ha="center", va="center", fontsize=6.1, color="white",
                    transform=ax.transAxes, zorder=4)
        ax.text(cx0, 0.355, note, ha="center", va="center",
                fontsize=5.45, color=gray, transform=ax.transAxes, linespacing=1.00)

    ax = axes[1]
    ax.text(0.47, 1.035, "(b) Difficulty changes with the policy",
            ha="center", va="bottom", fontsize=6.6, fontweight="bold",
            color=dark, transform=ax.transAxes, clip_on=False)
    xpts = [0.12, 0.38, 0.62, 0.88]
    ypts = [0.535, 0.715, 0.455, 0.535]
    ckpt_labels = ["ckpt 0", "later", "later", "final"]
    region_labels = [r"$B_3$", r"$B_2$", r"$B_4$", r"$B_3$"]
    state_w, state_h = 0.16, 0.130
    for x, y, ckl, regl in zip(xpts, ypts, ckpt_labels, region_labels):
        add_box(ax, (x - state_w / 2, y - state_h / 2), state_w, state_h, regl,
                teal_light if regl == r"$B_3$" else gray_light,
                teal if regl == r"$B_3$" else panel_edge, fs=8.2, weight="bold")
        ax.text(x, 0.170, ckl, ha="center", va="center", fontsize=6.05,
                color=gray, transform=ax.transAxes)
    arrow_specs = [
        ((xpts[0] + state_w / 2 + 0.010, ypts[0] + 0.010),
         (xpts[1] - state_w / 2 - 0.010, ypts[1] - 0.030), blue, 0.13),
        ((xpts[1] + state_w / 2 + 0.010, ypts[1] - 0.025),
         (xpts[2] - state_w / 2 - 0.010, ypts[2] + 0.030), orange, -0.15),
        ((xpts[2] + state_w / 2 + 0.010, ypts[2] + 0.030),
         (xpts[3] - state_w / 2 - 0.010, ypts[3] + 0.010), blue, 0.13),
    ]
    for start, end, arrow_color, rad in arrow_specs:
        add_arrow(ax, start, end, color=arrow_color, lw=1.15, rad=rad)
    ax.text(0.50, 0.290, "Same problem changes difficulty as the policy evolves",
            ha="center", va="center", fontsize=6.25, color=dark,
            transform=ax.transAxes)
    add_arrow(ax, (0.16, 0.105), (0.84, 0.105), color=panel_edge, lw=0.75)
    for x in xpts:
        ax.plot([x, x], [0.096, 0.114], color=panel_edge, linewidth=0.55,
                transform=ax.transAxes, clip_on=False)

    ax = axes[2]
    ax.set_title("(c) Research questions and tests", loc="left", pad=5)
    add_box(
        ax, (0.050, 0.790), 0.90, 0.125,
        "Q1. Do precomputed\ndifficulty estimates remain\naligned during training?",
        blue_light, blue, fs=5.45, weight="bold",
    )
    add_box(
        ax, (0.050, 0.600), 0.425, 0.145,
        "Difficulty tracking\nStatic curriculum",
        blue_light, blue, fs=6.0,
    )
    add_box(
        ax, (0.525, 0.600), 0.425, 0.145,
        "Intermediate-solvability\nsamples are often\nprioritized for\ninformative signals",
        teal_light, teal, fs=4.75, color=dark,
    )
    add_box(
        ax, (0.050, 0.365), 0.90, 0.170,
        "Q2. Is persistent medium\ntargeting sufficient, or should\nthe curriculum move to\nharder regions?",
        teal_light, teal, fs=5.25, weight="bold",
    )
    test_boxes = [
        ((0.190, 0.185), "Fixed\nmedium", teal_light, teal),
        ((0.550, 0.185), "Staged\n$B_1 \\rightarrow B_5$", orange_light, orange),
    ]
    for xy, text, fc, ec in test_boxes:
        add_box(ax, xy, 0.26, 0.118, text, fc, ec, fs=6.45)
    add_arrow(ax, (0.265, 0.790), (0.265, 0.745), color=blue, lw=0.82)
    add_arrow(ax, (0.735, 0.600), (0.690, 0.535), color=teal, lw=0.82)
    add_arrow(ax, (0.405, 0.365), (0.325, 0.303), color=teal, lw=0.82)
    add_arrow(ax, (0.595, 0.365), (0.680, 0.303), color=orange, lw=0.82)
    add_box(
        ax, (0.085, 0.030), 0.83, 0.100,
        "Precomputed difficulty can become stale;\nthe benefit of increasing difficulty is non-monotonic",
        gray_light, panel_edge, fs=5.05, color=dark,
    )

    fig.subplots_adjust(left=0.012, right=0.992, bottom=0.055, top=0.90, wspace=0.46)
    fig.savefig(FIG_DIR / "fig0_motivation.pdf", bbox_inches="tight", pad_inches=0.035)
    fig.savefig(FIG_DIR / "fig0_motivation.png", dpi=450, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def plot_line_panel(pdf, x0, y0, w, h, series, xlim, ylim, xticks, yticks, xlabel, ylabel, legend_at=None, xticklabels=None):
    scale = draw_axes(pdf, x0, y0, w, h, xticks, yticks, xlim, ylim, xlabel, ylabel, xticklabels)
    for item in series:
        pts = [scale(x, y) for x, y in zip(item["x"], item["y"])]
        pdf.polyline(pts, item["color"], item.get("width", 1.0), item.get("dash"))
        for x, y in pts:
            pdf.circle(x, y, item.get("marker", 2.0), stroke=item["color"], fill=(1, 1, 1), width=0.6)
    if legend_at:
        lx, ly = legend_at
        for i, item in enumerate(series):
            yy = ly - i * 12
            pdf.line(lx, yy + 3, lx + 14, yy + 3, item["color"], 1.0)
            pdf.circle(lx + 7, yy + 3, 2, stroke=item["color"], fill=(1, 1, 1), width=0.6)
            pdf.text(lx + 19, yy, item["label"], 7.5)
    return scale


def figure2():
    data, uids, files = load_tracking()
    selected = select_representative_samples(data, uids)
    with (META_DIR / "fig2_selected_samples.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["uid", "p0", "span", "meaningful_reversals_delta_0.125", *[f"P_{c}" for c in CHECKPOINTS]])
        for row in selected:
            writer.writerow([row["uid"], row["p0"], row["span"], row["reversals"], *row["values"]])
    pdf = PDF(FIG_DIR / "fig2_difficulty_trajectories.pdf", 620, 390)
    x_positions = list(range(len(CHECKPOINTS)))
    tick_steps = [0, 50, 250, 1000, 7473]
    tick_pos = [CHECKPOINTS.index(v) for v in tick_steps]
    panel_w, panel_h = 160, 118
    starts = [(58, 225), (238, 225), (418, 225), (58, 62), (238, 62), (418, 62)]
    colors = [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["purple"], COLORS["sky"], COLORS["vermillion"]]
    pdf.text(58, 368, "Empirical solvability", FS_AXIS)
    for idx, row in enumerate(selected):
        x0, y0 = starts[idx]
        xlabel = "Checkpoint" if idx >= 3 else ""
        ylabel = ""
        scale = draw_axes(
            pdf, x0, y0, panel_w, panel_h, tick_pos, [0, .5, 1.0], (0, 13), (0, 1),
            xlabel, ylabel, [str(v) for v in tick_steps]
        )
        pts = [scale(x, y) for x, y in zip(x_positions, row["values"])]
        color = colors[idx]
        pdf.polyline(pts, color, LW_LINE)
        for x, y in pts:
            pdf.circle(x, y, MS, stroke=color, fill=(1, 1, 1), width=0.55)
        pdf.text(x0 + 4, y0 + panel_h + 9, f"uid {row['uid']}, P0={row['p0']:.3f}", 7.7)
        pdf.text(x0 + panel_w - 4, y0 + 6, f"rev={row['reversals']}", 7.2, COLORS["gray"], align="right")
    pdf.save()
    return files, selected


def figure3():
    import os
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-grpo-b")

    import matplotlib.pyplot as plt
    from matplotlib import patches
    import numpy as np

    data, uids, _ = load_tracking()
    p0 = np.array([data[0][uid] for uid in uids], dtype=float)
    p_final = np.array([data[7473][uid] for uid in uids], dtype=float)

    def average_ranks(values):
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=float)
        sorted_values = values[order]
        i = 0
        while i < len(values):
            j = i + 1
            while j < len(values) and sorted_values[j] == sorted_values[i]:
                j += 1
            ranks[order[i:j]] = (i + 1 + j) / 2.0
            i = j
        return ranks

    r0 = average_ranks(p0)
    rf = average_ranks(p_final)
    spearman = float(np.corrcoef(r0, rf)[0, 1])
    if abs(spearman - 0.809217) > 5e-4:
        raise ValueError(f"Unexpected final Spearman correlation: {spearman:.6f}")

    b3_left, b3_right = 0.375, 0.625
    initial_b3 = (p0 >= b3_left) & (p0 <= b3_right)
    final_b3 = (p_final >= b3_left) & (p_final <= b3_right)
    retained = int(np.sum(initial_b3 & final_b3))
    initial_size = int(np.sum(initial_b3))
    final_size = int(np.sum(final_b3))
    left = initial_size - retained
    entered = final_size - retained
    union = int(np.sum(initial_b3 | final_b3))
    retention = retained / initial_size * 100
    inflow = entered / final_size * 100
    jaccard = retained / union
    expected = (270, 160, 56, 214, 104, 20.74, 65.00, 0.1497)
    observed = (initial_size, final_size, retained, left, entered, retention, inflow, jaccard)
    if (
        observed[:5] != expected[:5]
        or abs(retention - expected[5]) > 0.005
        or abs(inflow - expected[6]) > 0.005
        or abs(jaccard - expected[7]) > 0.00005
    ):
        raise ValueError(f"Unexpected B3 turnover values: {observed}")

    if len(uids) != 1495:
        raise ValueError(f"Unexpected number of tracked samples: {len(uids)}")

    rank_by_checkpoint = {
        ckpt: average_ranks(np.array([data[ckpt][uid] for uid in uids], dtype=float))
        for ckpt in CHECKPOINTS
    }
    spearman_curve = [float(np.corrcoef(rank_by_checkpoint[0], rank_by_checkpoint[ckpt])[0, 1]) for ckpt in CHECKPOINTS]
    expected_rhos = {100: 0.926240, 500: 0.854797, 1000: 0.842897, 7473: 0.809217}
    for ckpt, expected_rho in expected_rhos.items():
        observed_rho = spearman_curve[CHECKPOINTS.index(ckpt)]
        if abs(observed_rho - expected_rho) > 5e-7:
            raise ValueError(f"Unexpected Spearman at checkpoint {ckpt}: {observed_rho:.6f}")
    if abs(spearman_curve[0] - 1.0) > 1e-12:
        raise ValueError(f"Unexpected initial Spearman: {spearman_curve[0]:.12f}")

    with (META_DIR / "fig3_spearman_curve.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["checkpoint", "spearman_with_p0"])
        for ckpt, rho in zip(CHECKPOINTS, spearman_curve):
            writer.writerow([ckpt, f"{rho:.6f}"])

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.titlesize": 8.0,
        "axes.labelsize": 8.1,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
    })
    fig = plt.figure(figsize=(7.25, 3.15), dpi=300, constrained_layout=False)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 0.95], wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_axis_on()

    blue = "#4e7fa9"
    orange = "#c78362"
    green = "#3b8f73"
    gray = "#4b5563"
    light_gray = "#eef1f4"
    light_blue = "#e8f0f7"
    light_orange = "#f5ebe5"
    light_green = "#e7f3ee"

    x_positions = np.arange(len(CHECKPOINTS))
    ax.plot(x_positions, spearman_curve, color=blue, linewidth=1.15, marker="o",
            markersize=3.0, markerfacecolor="white", markeredgewidth=0.75)
    ax.axhline(0.8, color="#c7cdd4", linewidth=0.65, linestyle=(0, (3, 2)), zorder=0)
    ax.set_xlim(-0.35, len(CHECKPOINTS) - 0.65)
    ax.set_ylim(0.75, 1.01)
    tick_indices = [0, 5, 7, 9, 13]
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([str(CHECKPOINTS[i]) for i in tick_indices])
    ax.set_yticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    ax.set_xlabel("Tracked checkpoint")
    ax.set_ylabel(r"Spearman rank correlation with $P_0$")
    ax.set_title("(a) Global rank consistency over training", loc="left", fontweight="bold", pad=5)
    ax.grid(axis="y", color="#d7dde3", linestyle="-", linewidth=0.45, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.annotate(r"$\rho=1.0000$", xy=(0, spearman_curve[0]), xytext=(0.7, 0.996),
                textcoords="data", fontsize=6.9, color="#111827",
                arrowprops=dict(arrowstyle="-", color="#6b7280", lw=0.55))
    ax.annotate(r"$\rho=0.8092$", xy=(len(CHECKPOINTS) - 1, spearman_curve[-1]),
                xytext=(len(CHECKPOINTS) - 4.2, 0.823),
                textcoords="data", fontsize=7.1, color="#111827",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#d1d5db", linewidth=0.55),
                arrowprops=dict(arrowstyle="-", color="#6b7280", lw=0.55))
    ax.text(0.02, 0.06, "Equal horizontal spacing", transform=ax.transAxes,
            ha="left", va="center", fontsize=6.7, color="#4b5563")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_axis_off()
    ax2.set_title(r"(b) Local $B_3$ membership turnover", loc="left", fontweight="bold", pad=5)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    retained_color = "#6fa88f"
    left_color = "#d8dde3"
    entered_color = "#d79a72"

    left_donut = ax2.inset_axes([0.02, 0.42, 0.41, 0.46])
    right_donut = ax2.inset_axes([0.57, 0.42, 0.41, 0.46])
    for donut_ax, values, colors, title, center in [
        (left_donut, [retained, left], [retained_color, left_color], r"Initial $B_3$", "$n=270$"),
        (right_donut, [retained, entered], [retained_color, entered_color], r"Final $B_3$", "$n=160$"),
    ]:
        donut_ax.pie(
            values,
            colors=colors,
            startangle=90,
            counterclock=False,
            wedgeprops=dict(width=0.33, edgecolor="white", linewidth=1.0),
        )
        donut_ax.text(0, 0.05, title, ha="center", va="center",
                      fontsize=7.4, fontweight="bold", color="#111827")
        donut_ax.text(0, -0.16, center, ha="center", va="center",
                      fontsize=7.1, color="#374151")
        donut_ax.set_aspect("equal")
        donut_ax.set_axis_off()

    ax2.annotate("", xy=(0.56, 0.66), xytext=(0.44, 0.66),
                 arrowprops=dict(arrowstyle="-|>", color=green, lw=0.9, mutation_scale=8))
    ax2.text(0.50, 0.715, "56 shared\nproblems", ha="center", va="center",
             linespacing=1.0,
             fontsize=7.0, color="#286b57", fontweight="bold")
    ax2.text(0.205, 0.355, "Retained: 20.74% (n=56)", ha="center", va="center",
             fontsize=6.3, color="#286b57")
    ax2.text(0.205, 0.292, "Left: 79.26% (n=214)", ha="center", va="center",
             fontsize=6.3, color="#4b5563")
    ax2.text(0.775, 0.355, "Retained: 35.00% (n=56)", ha="center", va="center",
             fontsize=6.3, color="#286b57")
    ax2.text(0.775, 0.292, "Entered: 65.00% (n=104)", ha="center", va="center",
             fontsize=6.3, color="#835334")
    ax2.plot([0.30, 0.70], [0.208, 0.208], color="#d1d5db", linewidth=0.55)
    ax2.text(0.50, 0.150, r"$B_3=[0.375,0.625]$" "\n" "Jaccard = 0.1497",
             ha="center", va="center", fontsize=7.3, color="#111827", linespacing=1.25)

    fig.savefig(FIG_DIR / "fig3_global_local_drift.pdf", bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def figure_b_initial_final_rank_consistency():
    import os
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-grpo-b")

    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, PowerNorm
    import numpy as np

    data, uids, _ = load_tracking()
    if len(uids) != 1495:
        raise ValueError(f"Unexpected number of tracked samples: {len(uids)}")

    p0 = np.array([data[0][uid] for uid in uids], dtype=float)
    p_final = np.array([data[7473][uid] for uid in uids], dtype=float)

    def average_ranks(values):
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=float)
        sorted_values = values[order]
        i = 0
        while i < len(values):
            j = i + 1
            while j < len(values) and sorted_values[j] == sorted_values[i]:
                j += 1
            ranks[order[i:j]] = (i + 1 + j) / 2.0
            i = j
        return ranks

    r0 = average_ranks(p0)
    rf = average_ranks(p_final)
    spearman = float(np.corrcoef(r0, rf)[0, 1])
    if abs(spearman - 0.809217) > 5e-7:
        raise ValueError(f"Unexpected final Spearman correlation: {spearman:.6f}")

    initial_rank_percentile = (r0 - 1.0) / (len(uids) - 1) * 100.0
    final_rank_percentile = (rf - 1.0) / (len(uids) - 1) * 100.0

    with (META_DIR / "figB_initial_final_rank_percentiles.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["uid", "p0", "p_final", "initial_rank_percentile", "final_rank_percentile"])
        for uid, p0_value, pf_value, initial_percentile, final_percentile in sorted(
            zip(uids, p0, p_final, initial_rank_percentile, final_rank_percentile),
            key=lambda row: int(row[0])
        ):
            writer.writerow([
                uid,
                f"{p0_value:.6f}",
                f"{pf_value:.6f}",
                f"{initial_percentile:.6f}",
                f"{final_percentile:.6f}",
            ])

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
    })
    cmap = LinearSegmentedColormap.from_list(
        "rank_count_bluegray",
        ["#f7fafc", "#d9e6ef", "#93b5cc", "#4f85aa", "#24577d"],
    )
    fig, ax = plt.subplots(figsize=(4.55, 3.72), dpi=300)
    counts, xedges, yedges, image = ax.hist2d(
        initial_rank_percentile,
        final_rank_percentile,
        bins=np.linspace(0, 100, 26),
        cmap=cmap,
        norm=PowerNorm(gamma=0.62),
        cmin=1,
    )
    ax.plot([0, 100], [0, 100], color="#6b7280", linewidth=0.75, linestyle=(0, (4, 3)))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Initial rank percentile")
    ax.set_ylabel("Final rank percentile")
    ax.set_title("Initial--final global rank consistency", loc="left", fontweight="bold", pad=6)
    ax.grid(color="#e5e7eb", linewidth=0.30, alpha=0.50)
    ax.set_axisbelow(True)
    ax.text(0.04, 0.95, r"Spearman $\rho=0.8092$", transform=ax.transAxes,
            ha="left", va="top", fontsize=7.3,
            bbox=dict(boxstyle="round,pad=0.18", facecolor=(1, 1, 1, 0.82), edgecolor="#d1d5db", linewidth=0.45))
    ax.text(0.985, 0.02, r"$y=x$", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.8, color="#4b5563")
    ax.text(0.00, -0.14, "harder", transform=ax.transAxes, ha="left", va="top",
            fontsize=6.7, color="#4b5563")
    ax.text(1.00, -0.14, "easier", transform=ax.transAxes, ha="right", va="top",
            fontsize=6.7, color="#4b5563")
    ax.text(-0.15, 0.00, "harder", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.7, color="#4b5563", rotation=90)
    ax.text(-0.15, 1.00, "easier", transform=ax.transAxes, ha="right", va="top",
            fontsize=6.7, color="#4b5563", rotation=90)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label("Number of problems", fontsize=7.4)
    cbar.outline.set_linewidth(0.55)
    cbar.ax.tick_params(labelsize=6.8, width=0.55, length=2.5)
    fig.savefig(FIG_DIR / "figB_initial_final_rank_consistency.pdf", bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def figure4():
    import os
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-grpo-b")

    import matplotlib.pyplot as plt
    import numpy as np

    staged_x = [400, 450, 500, 550, 600, 650, 700, 750, 800]
    staged_y = [0.6331, 0.6270, 0.6171, 0.6399, 0.6444, 0.6505, 0.6558, 0.6391, 0.6353]
    fixed_x = [400, 450, 500, 550, 600, 650]
    fixed_y = [0.6376, 0.6459, 0.6452, 0.6217, 0.6217, 0.6247]

    expected_staged = [
        (400, 0.6331), (450, 0.6270), (500, 0.6171), (550, 0.6399),
        (600, 0.6444), (650, 0.6505), (700, 0.6558), (750, 0.6391), (800, 0.6353),
    ]
    expected_fixed = [
        (400, 0.6376), (450, 0.6459), (500, 0.6452),
        (550, 0.6217), (600, 0.6217), (650, 0.6247),
    ]
    if list(zip(staged_x, staged_y)) != expected_staged or list(zip(fixed_x, fixed_y)) != expected_fixed:
        raise ValueError("Figure 4 trajectory values do not match Appendix D.")

    categories = [
        "Standard\nGRPO",
        "Static\ncurriculum",
        "Fixed\nmedium",
        "Staged\n$B_4$",
        "Staged\n$B_5$+100",
    ]
    seeds_by_category = np.array([
        [0.6459, 0.6414, 0.6459],
        [0.6535, 0.6497, 0.6535],
        [0.6459, 0.6422, 0.6467],
        [0.6558, 0.6535, 0.6679],
        [0.6353, 0.6404, 0.6458],
    ])
    means = np.array([0.6444, 0.6522, 0.6449, 0.6591, 0.6405])
    stds = np.array([0.0026, 0.0022, 0.0024, 0.0077, 0.0053])
    if (
        np.max(np.abs(seeds_by_category.mean(axis=1) - means)) > 5e-5
        or np.max(np.abs(seeds_by_category.std(axis=1, ddof=1) - stds)) > 5e-5
    ):
        raise ValueError("Figure 4 multi-seed values do not match Table 6.")

    with (META_DIR / "fig4_performance_points.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["panel", "series", "x", "test_accuracy", "note"])
        for x, y in zip(staged_x, staged_y):
            note = {
                400: "selected B2", 550: "selected B3", 700: "selected B4",
                750: "B5 + 50", 800: "B5 + 100",
            }.get(x, "")
            writer.writerow(["a", "staged curriculum", x, f"{y:.4f}", note])
        for x, y in zip(fixed_x, fixed_y):
            writer.writerow(["a", "fixed medium B3", x, f"{y:.4f}", ""])
        for category, values, mean, std in zip(categories, seeds_by_category, means, stds):
            clean_category = category.replace("\n", " ").replace("$", "")
            for seed_idx, value in enumerate(values, start=1):
                writer.writerow(["b", clean_category, f"seed {seed_idx}", f"{value:.4f}", ""])
            writer.writerow(["b", clean_category, "mean", f"{mean:.4f}", f"std={std:.4f}"])

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.1,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
    })

    blue = "#4e7fa9"
    orange = "#c78362"
    green = "#3b8f73"
    vermillion = "#b45a4a"
    gray = "#6b7280"

    fig = plt.figure(figsize=(7.35, 3.25), dpi=300, constrained_layout=False)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.10, 0.90], wspace=0.34)

    ax = fig.add_subplot(gs[0, 0])
    for start, stop, label, color in [
        (450, 550, "$B_3$", "#e8f0f7"),
        (600, 700, "$B_4$", "#e7f3ee"),
        (750, 800, "$B_5$", "#f5ebe5"),
    ]:
        ax.axvspan(start, stop, color=color, alpha=0.82, zorder=0)
        ax.text((start + stop) / 2, 0.6670, label, ha="center", va="top",
                fontsize=7.1, color="#374151")
    ax.plot(staged_x, staged_y, color=blue, linewidth=1.25, marker="o",
            markersize=3.2, markerfacecolor="white", markeredgewidth=0.75,
            label="Staged curriculum")
    ax.plot(fixed_x, fixed_y, color=orange, linewidth=1.05, marker="s",
            markersize=3.0, markerfacecolor="white", markeredgewidth=0.70,
            linestyle=(0, (3, 2)), label=r"Fixed medium ($B_3$)")
    ax.scatter([550], [0.6399], s=35, marker="o", facecolor="white",
               edgecolor=green, linewidth=1.0, zorder=5)
    ax.scatter([700], [0.6558], s=70, marker="*", facecolor=green,
               edgecolor="#1f4f40", linewidth=0.55, zorder=6)
    ax.text(548, 0.6424, "selected $B_3$", ha="right", va="bottom",
            fontsize=6.8, color="#374151")
    ax.text(700, 0.6592, "selected $B_4$", ha="center", va="bottom",
            fontsize=6.9, color="#1f4f40", fontweight="bold")
    ax.axvline(750, color=gray, linewidth=0.7, linestyle=(0, (3, 2)))
    ax.text(756, 0.6188, "$B_5$ entry", ha="left", va="bottom",
            fontsize=6.8, color="#4b5563")
    ax.annotate("-2.05 pp", xy=(800, 0.6353), xytext=(720, 0.6462),
                ha="left", va="center", fontsize=7.0, color=vermillion,
                arrowprops=dict(arrowstyle="->", color=vermillion, lw=0.75,
                                shrinkA=2, shrinkB=2))
    ax.set_xlim(382, 818)
    ax.set_ylim(0.615, 0.668)
    ax.set_xticks([400, 500, 600, 700, 800])
    ax.set_yticks([0.62, 0.63, 0.64, 0.65, 0.66])
    ax.set_xlabel("Cumulative effective updates")
    ax.set_ylabel("GSM8K test accuracy")
    ax.set_title("(a) Performance during staged curriculum training",
                 loc="left", fontweight="bold", pad=5)
    ax.grid(axis="y", color="#d7dde3", linewidth=0.45, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=True, framealpha=0.88, edgecolor="#d1d5db",
              handlelength=2.2, borderpad=0.35, labelspacing=0.35)

    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(categories))
    bar_colors = ["#dbe5ef", "#cbd9e6", "#dbe5ef", "#6fa88f", "#e6d8cf"]
    edge_colors = ["#8aa4bc", "#7f9ab2", "#8aa4bc", green, "#b88d78"]
    bars = ax2.bar(x, means, width=0.58, color=bar_colors, edgecolor=edge_colors,
                   linewidth=0.85, zorder=3)
    for bar, mean in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width() / 2, mean + 0.0009, f"{mean:.4f}",
                 ha="center", va="bottom", fontsize=6.7, color="#111827")
    ax2.set_xlim(-0.45, 4.45)
    ax2.set_ylim(0.638, 0.663)
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.set_yticks([0.64, 0.645, 0.65, 0.655, 0.66])
    ax2.set_ylabel("GSM8K test accuracy")
    ax2.set_title("(b) Multi-seed performance comparison",
                  loc="left", fontweight="bold", pad=5)
    ax2.grid(axis="y", color="#d7dde3", linewidth=0.45, alpha=0.85)
    ax2.set_axisbelow(True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.savefig(FIG_DIR / "fig4_performance_comparison.pdf", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(FIG_DIR / "fig4_performance_comparison.png", dpi=450, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


DIAG = {
    "Seed 1": {
        "B1": ([0, 50, 100, 150, 200, 250], [0.785889, 0.788086, 0.801758, 0.834961, 0.830078, 0.835938], 150),
        "B2": ([0, 50, 100, 150, 200, 250, 300, 350], [0.701172, 0.691406, 0.767578, 0.791992, 0.802734, 0.817383, 0.815430, 0.801718], 250),
        "B3": ([0, 50, 100, 150, 200, 250], [0.690430, 0.716797, 0.707031, 0.743164, 0.743164, 0.712891], 150),
        "B4": ([0, 50, 100, 150, 200, 250], [0.619141, 0.596680, 0.621094, 0.643555, 0.563477, 0.551477], 150),
        "B5": ([0, 50, 100], [0.521484, 0.497070, 0.451172], None),
    },
    "Seed 2": {
        "B1": ([0, 50, 100, 150, 200, 250, 300, 350, 400], [0.784356, 0.813477, 0.843750, 0.855469, 0.863281, 0.893555, 0.911133, 0.897461, 0.903320], 300),
        "B2": ([0, 50, 100, 150, 200, 250], [0.774332, 0.795898, 0.821289, 0.823242, 0.805664, 0.812500], 150),
        "B3": ([0, 50, 100, 150, 200, 250, 300], [0.713462, 0.742188, 0.721680, 0.736328, 0.771484, 0.767578, 0.733398], 200),
        "B4": ([0, 50, 100, 150, 200], [0.573463, 0.589844, 0.617188, 0.616211, 0.612305], 100),
        "B5": ([0, 50, 100], [0.512695, 0.508555, 0.452148], None),
    },
    "Seed 3": {
        "B1": ([0, 50, 100, 150, 200, 250, 300, 350], [0.790565, 0.813547, 0.835444, 0.866211, 0.899414, 0.920898, 0.920898, 0.913086], 250),
        "B2": ([0, 50, 100, 150, 200, 250, 300, 350, 400], [0.784536, 0.812346, 0.827891, 0.838835, 0.831055, 0.840844, 0.843750, 0.820313, 0.837891], 300),
        "B3": ([0, 50, 100, 150], [0.757813, 0.766602, 0.753906, 0.747070], 50),
        "B4": ([0, 50, 100, 150, 200], [0.635489, 0.666992, 0.680664, 0.668945, 0.556641], 100),
        "B5": ([0, 50, 100], [0.574453, 0.571289, 0.527344], None),
    },
}


def figure_c1():
    import os
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-grpo-b")

    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt

    expected_selected = {
        "Seed 1": {"B1": (150, 0.834961), "B2": (250, 0.817383), "B3": (150, 0.743164), "B4": (150, 0.643555)},
        "Seed 2": {"B1": (300, 0.911133), "B2": (150, 0.823242), "B3": (200, 0.771484), "B4": (100, 0.617188)},
        "Seed 3": {"B1": (250, 0.920898), "B2": (300, 0.843750), "B3": (50, 0.766602), "B4": (100, 0.680664)},
    }
    expected_b5 = {
        "Seed 1": [0.521484, 0.497070, 0.451172],
        "Seed 2": [0.512695, 0.508555, 0.452148],
        "Seed 3": [0.574453, 0.571289, 0.527344],
    }
    for seed, stages in expected_selected.items():
        for stage, (sel_x, sel_y) in stages.items():
            xs, ys, sel = DIAG[seed][stage]
            if sel != sel_x or abs(ys[xs.index(sel_x)] - sel_y) > 5e-7:
                raise ValueError(f"Figure C1 selected checkpoint mismatch: {seed} {stage}")
        b5_xs, b5_ys, b5_sel = DIAG[seed]["B5"]
        if b5_sel is not None or len(b5_xs) != 3 or any(abs(a - b) > 5e-7 for a, b in zip(b5_ys, expected_b5[seed])):
            raise ValueError(f"Figure C1 B5 trajectory mismatch: {seed}")

    with (META_DIR / "figC1_diagnostic_trajectories.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "stage", "local_effective_updates", "diagnostic_accuracy", "selected"])
        for seed, stages in DIAG.items():
            for stage, (xs, ys, sel) in stages.items():
                for x, y in zip(xs, ys):
                    writer.writerow([seed, stage, x, f"{y:.6f}", int(sel == x)])

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.titlesize": 8.2,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
    })

    stage_colors = {
        "B1": "#4e7fa9",
        "B2": "#5b9a93",
        "B3": "#6fa88f",
        "B4": "#c78362",
        "B5": "#b45a4a",
    }

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.72), dpi=300, sharey=True)
    for idx, (ax, seed) in enumerate(zip(axes, ["Seed 1", "Seed 2", "Seed 3"])):
        for stage in ["B1", "B2", "B3", "B4", "B5"]:
            xs, ys, sel = DIAG[seed][stage]
            color = stage_colors[stage]
            ax.plot(xs, ys, color=color, linewidth=1.05, marker="o",
                    markersize=2.7, markerfacecolor="white",
                    markeredgewidth=0.65, zorder=3)
            if sel is not None:
                j = xs.index(sel)
                ax.scatter([xs[j]], [ys[j]], s=34, marker="D",
                           facecolor="white", edgecolor="#111827",
                           linewidth=0.9, zorder=5)
                ax.scatter([xs[j]], [ys[j]], s=14, marker="D",
                           facecolor=color, edgecolor=color,
                           linewidth=0.45, zorder=6)
        ax.set_title(f"({chr(97 + idx)}) {seed}", loc="left",
                     fontweight="bold", pad=5)
        ax.set_xlim(-12, 412)
        ax.set_ylim(0.43, 0.94)
        ax.set_xticks([0, 100, 200, 300, 400])
        ax.set_yticks([0.45, 0.60, 0.75, 0.90])
        ax.grid(axis="y", color="#d7dde3", linewidth=0.45, alpha=0.85)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx == 0:
            ax.set_ylabel("Diagnostic accuracy")
        ax.set_xlabel("Local effective updates")

    stage_handles = [
        mlines.Line2D([], [], color=stage_colors[stage], marker="o",
                      markerfacecolor="white", markeredgewidth=0.65,
                      linewidth=1.05, markersize=3.2, label=stage)
        for stage in ["B1", "B2", "B3", "B4", "B5"]
    ]
    selected_handle = mlines.Line2D([], [], color="#111827", marker="D",
                                    markerfacecolor="white", markeredgewidth=0.9,
                                    linestyle="None", markersize=4.2,
                                    label="Selected checkpoint")
    fig.legend(handles=stage_handles + [selected_handle], loc="upper center",
               bbox_to_anchor=(0.5, 1.02), ncol=6, frameon=False,
               handlelength=1.7, columnspacing=1.1)
    fig.text(0.5, 0.018, "Selection tolerance: 0.0107", ha="center",
             va="bottom", fontsize=7.0, color="#4b5563")
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.20, top=0.82, wspace=0.20)
    fig.savefig(FIG_DIR / "figC1_diagnostic_trajectories.pdf", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(FIG_DIR / "figC1_diagnostic_trajectories.png", dpi=450, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def figure_d1():
    import os
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-grpo-b")

    import matplotlib.pyplot as plt
    import numpy as np

    steps = [1000, 2500, 3500, 4500, 5500, 6500]
    planned = [0.9531, 0.8523, 0.7528, 0.6098, 0.3986, 0.1357]
    current = [0.9796, 0.9422, 0.8384, 0.7275, 0.5573, 0.1544]
    mad = [0.0396, 0.1090, 0.1562, 0.1464, 0.1863, 0.0713]
    expected = [
        (1000, 0.9531, 0.9796, 0.0396),
        (2500, 0.8523, 0.9422, 0.1090),
        (3500, 0.7528, 0.8384, 0.1562),
        (4500, 0.6098, 0.7275, 0.1464),
        (5500, 0.3986, 0.5573, 0.1863),
        (6500, 0.1357, 0.1544, 0.0713),
    ]
    if list(zip(steps, planned, current, mad)) != expected:
        raise ValueError("Figure D1 values do not match Table D10.")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.titlesize": 8.2,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
    })

    blue = "#4e7fa9"
    orange = "#c78362"
    teal = "#8ab8ad"
    gray = "#374151"

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.25, 2.72), dpi=300, gridspec_kw={"wspace": 0.26}
    )

    ax1.plot(steps, planned, color=blue, linewidth=1.15, marker="o",
             markersize=3.0, markerfacecolor="white", markeredgewidth=0.7,
             label=r"Planned mean $P_0$")
    ax1.plot(steps, current, color=orange, linewidth=1.15, marker="o",
             markersize=3.0, markerfacecolor="white", markeredgewidth=0.7,
             label=r"Current mean $P_t$")
    ax1.set_title("(a) Planned and current solvability", loc="left",
                  fontweight="bold", pad=5)
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Mean solvability")
    ax1.set_xlim(800, 6700)
    ax1.set_ylim(0.0, 1.04)
    ax1.set_xticks(steps)
    ax1.set_xticklabels([str(s) for s in steps], rotation=0)
    ax1.set_yticks([0.00, 0.25, 0.50, 0.75, 1.00])
    ax1.grid(axis="y", color="#d7dde3", linewidth=0.45, alpha=0.85)
    ax1.set_axisbelow(True)
    ax1.legend(loc="upper right", frameon=True, framealpha=0.88,
               edgecolor="#d1d5db", borderpad=0.35, labelspacing=0.32,
               handlelength=1.8)

    x = np.arange(len(steps))
    bar_colors = [teal] * len(steps)
    edge_colors = ["#5f9f94"] * len(steps)
    ax2.bar(x, mad, width=0.58, color=bar_colors, edgecolor=edge_colors,
            linewidth=0.8, zorder=3)
    max_idx = steps.index(5500)
    ax2.text(max_idx, mad[max_idx] + 0.006, "0.1863",
             ha="center", va="bottom", fontsize=7.0, color=gray)
    ax2.set_title("(b) Difficulty misalignment magnitude", loc="left",
                  fontweight="bold", pad=5)
    ax2.set_xlabel("Training step")
    ax2.set_ylabel("Mean abs. deviation")
    ax2.set_xlim(-0.55, len(steps) - 0.45)
    ax2.set_ylim(0.0, 0.21)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(s) for s in steps], rotation=0)
    ax2.set_yticks([0.00, 0.05, 0.10, 0.15, 0.20])
    ax2.grid(axis="y", color="#d7dde3", linewidth=0.45, alpha=0.85)
    ax2.set_axisbelow(True)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.20, top=0.88, wspace=0.26)
    fig.savefig(FIG_DIR / "figD1_static_mismatch.pdf", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(FIG_DIR / "figD1_static_mismatch.png", dpi=450, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def update_training_paradigms_terms():
    import os
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-grpo-b")

    import matplotlib.pyplot as plt
    from matplotlib import patches

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })

    blue_light = "#d9ecfb"
    green_light = "#dff1df"
    orange_light = "#f8d0a0"
    gray_light = "#eeeeee"
    border = "#222222"

    def add_panel(ax, title):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(patches.Rectangle((0.02, 0.04), 0.96, 0.88, fill=False,
                                       edgecolor=border, linewidth=0.9))
        ax.text(0.05, 0.87, title, ha="left", va="center",
                fontsize=9.7, fontweight="bold")

    def add_box(ax, cx, cy, w, h, text, fc):
        ax.add_patch(patches.FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            facecolor=fc, edgecolor=border, linewidth=0.65))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=8.6,
                linespacing=1.15)

    def add_arrow(ax, x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=border, lw=0.75,
                                    shrinkA=0, shrinkB=0, mutation_scale=8))

    fig, axs = plt.subplots(2, 2, figsize=(7.35, 5.45), dpi=320)
    fig.suptitle("Comparison of Training Paradigms", fontsize=13.2,
                 fontweight="bold", y=0.985)

    add_panel(axs[0, 0], "(a) Standard GRPO")
    yvals = [0.68, 0.46, 0.25]
    add_box(axs[0, 0], 0.50, yvals[0], 0.44, 0.15, "GSM8K\nTraining Set", gray_light)
    add_box(axs[0, 0], 0.50, yvals[1], 0.44, 0.15, "All Samples", green_light)
    add_box(axs[0, 0], 0.50, yvals[2], 0.44, 0.15, "GRPO Training", orange_light)
    add_arrow(axs[0, 0], 0.50, 0.605, 0.50, 0.535)
    add_arrow(axs[0, 0], 0.50, 0.385, 0.50, 0.325)
    axs[0, 0].text(0.50, 0.105, "No difficulty control.", ha="center", fontsize=7.1)

    add_panel(axs[0, 1], "(b) Static Curriculum")
    yvals = [0.74, 0.57, 0.40, 0.23]
    add_box(axs[0, 1], 0.50, yvals[0], 0.44, 0.13, "GSM8K\nTraining Set", gray_light)
    add_box(axs[0, 1], 0.50, yvals[1], 0.44, 0.13, r"Estimate $P_0(x)$", blue_light)
    add_box(axs[0, 1], 0.50, yvals[2], 0.44, 0.13, "Fixed Easy-to-Hard Order", green_light)
    add_box(axs[0, 1], 0.50, yvals[3], 0.44, 0.13, "GRPO Training", orange_light)
    for y0, y1 in [(0.675, 0.635), (0.505, 0.465), (0.335, 0.295)]:
        add_arrow(axs[0, 1], 0.50, y0, 0.50, y1)
    axs[0, 1].text(0.50, 0.105, "Fixed difficulty / fixed schedule.", ha="center", fontsize=7.1)

    add_panel(axs[1, 0], "(c) Fixed-Medium")
    yvals = [0.76, 0.60, 0.43, 0.23]
    add_box(axs[1, 0], 0.50, yvals[0], 0.46, 0.13, "Current Policy", gray_light)
    add_box(axs[1, 0], 0.50, yvals[1], 0.46, 0.13, r"Estimate $P_t(x)$", blue_light)
    add_box(axs[1, 0], 0.50, yvals[2], 0.46, 0.15,
            r"Select $P_t(x)\in B_3$" + "\n" + r"$B_3=[0.375,0.625]$", green_light)
    add_box(axs[1, 0], 0.50, yvals[3], 0.46, 0.13, "GRPO Training", orange_light)
    for y0, y1 in [(0.695, 0.665), (0.535, 0.505), (0.355, 0.295)]:
        add_arrow(axs[1, 0], 0.50, y0, 0.50, y1)
    axs[1, 0].text(0.50, 0.085, "Online sample selection.\nFixed target region.",
                   ha="center", va="center", fontsize=6.5, linespacing=1.00)

    add_panel(axs[1, 1], "(d) Staged Curriculum")
    yvals = [0.72, 0.585, 0.45, 0.315, 0.18]
    add_box(axs[1, 1], 0.50, yvals[0], 0.46, 0.105, "Current Policy", gray_light)
    add_box(axs[1, 1], 0.50, yvals[1], 0.46, 0.105, r"Estimate $P_t(x)$", blue_light)
    add_box(axs[1, 1], 0.50, yvals[2], 0.46, 0.105, r"Select $P_t(x)\in B_s$", green_light)
    add_box(axs[1, 1], 0.50, yvals[3], 0.46, 0.105,
            r"$B_1 \rightarrow B_2 \rightarrow B_3 \rightarrow B_4 \rightarrow B_5$", green_light)
    add_box(axs[1, 1], 0.50, yvals[4], 0.46, 0.105, "GRPO Training", orange_light)
    for y0, y1 in [(0.667, 0.638), (0.532, 0.503), (0.397, 0.368), (0.262, 0.233)]:
        add_arrow(axs[1, 1], 0.50, y0, 0.50, y1)
    axs[1, 1].text(0.50, 0.075, "Online sample selection.\nStaged curriculum schedule.",
                   ha="center", va="center", fontsize=6.1, linespacing=1.00)

    fig.subplots_adjust(left=0.025, right=0.985, bottom=0.035, top=0.93,
                        wspace=0.04, hspace=0.07)
    fig.savefig(FIG_DIR / "fig1_training_paradigms.png", dpi=320,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def write_metadata(fig2_files, selected):
    metadata = {
        "fig0_motivation.pdf": {"source": "Conceptual motivation figure derived from the Introduction; no experimental datapoints."},
        "fig1_framework.pdf": {"source": "Conceptual workflow derived from main.tex study design; no experimental datapoints."},
        "fig2_difficulty_trajectories.pdf": {
            "source_files": fig2_files,
            "selection_rule": "Use all 1,495 common tracked samples; compute meaningful reversals from consecutive P_t changes with |delta| >= 0.125; within initial-solvability bins [0,0.375), [0.375,0.625), and [0.625,1], sort candidates by descending training-period span, descending reversal count, then ascending uid; select up to three from each bin.",
            "selected_uids": [row["uid"] for row in selected],
        },
        "fig3_global_local_drift.pdf": {
            "source": "All 14 tracked-sample JSONL files under outputs/difficulty_drift_eval_pdist20_seed42; B3 turnover values validated against main.tex Appendix B.",
            "spearman_curve_metadata": "figures/metadata/fig3_spearman_curve.csv",
        },
        "figB_initial_final_rank_consistency.pdf": {
            "source": "Checkpoint-0 and checkpoint-7473 tracked-sample JSONL files under outputs/difficulty_drift_eval_pdist20_seed42.",
            "rank_percentile_metadata": "figures/metadata/figB_initial_final_rank_percentiles.csv",
        },
        "fig4_performance_comparison.pdf": {
            "source": "Seed 1 staged trajectory and fixed-medium trajectory from Appendix D; multi-seed comparison from Table 6 in main.tex.",
            "performance_points_metadata": "figures/metadata/fig4_performance_points.csv"
        },
        "figC1_diagnostic_trajectories.pdf": {
            "source": "Tables C6-C9 values in main.tex.",
            "diagnostic_trajectory_metadata": "figures/metadata/figC1_diagnostic_trajectories.csv"
        },
        "figD1_static_mismatch.pdf": {"source": "Table D10 values in main.tex."},
    }
    (META_DIR / "figure_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main():
    FIG_DIR.mkdir(exist_ok=True)
    META_DIR.mkdir(exist_ok=True)
    figure_motivation()
    figure1()
    update_training_paradigms_terms()
    fig2_files, selected = figure2()
    figure3()
    figure_b_initial_final_rank_consistency()
    figure4()
    figure_c1()
    figure_d1()
    write_metadata(fig2_files, selected)


if __name__ == "__main__":
    main()
