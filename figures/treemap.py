"""
treemap.py
Custom implementation.

Author: Alexandre Devaux Rivière
Project: NPM3D
Date: 20/03/2026
"""

import io
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch


def _worst_ratio(row, w):
    if not row:
        return float("inf")
    s = sum(row)
    return max(max(w * w * v / (s * s), s * s / (w * w * v)) for v in row)


def _squarify(sizes, x, y, dx, dy):
    if not sizes:
        return []
    if len(sizes) == 1:
        return [{"x": x, "y": y, "dx": dx, "dy": dy}]

    # normalise so total area == dx * dy
    total = sum(sizes)
    sizes = [s / total * dx * dy for s in sizes]

    w = min(dx, dy)
    row, rest = [], list(sizes)

    while rest:
        candidate = row + [rest[0]]
        if not row or _worst_ratio(candidate, w) <= _worst_ratio(row, w):
            row, rest = candidate, rest[1:]
        else:
            break # next item would worsen the aspect ratios

    row_sum = sum(row)
    rects = []
    if dx >= dy: # lay row along x-axis
        rw = row_sum / dy if dy else 0
        ry = y
        for v in row:
            rh = v / row_sum * dy if row_sum else 0
            rects.append({"x": x, "y": ry, "dx": rw, "dy": rh})
            ry += rh
        rects += _squarify(rest, x + rw, y, dx - rw, dy)
    else: # lay row along y-axis
        rh = row_sum / dx if dx else 0
        rx = x
        for v in row:
            rw = v / row_sum * dx if row_sum else 0
            rects.append({"x": rx, "y": y, "dx": rw, "dy": rh})
            rx += rw
        rects += _squarify(rest, x, y + rh, dx, dy - rh)

    return rects


def squarify_layout(values, x, y, w, h):
    total = sum(values)
    norm = [v / total * w * h for v in values]
    return _squarify(norm, x, y, w, h)


CSV_PATH = "figures/csv/inference_latency_ms.csv"
df = pd.read_csv(CSV_PATH)

OPS = ["QKV encoding", "KNN query", "Relative positional encoding",
          "Relation & weight encoding", "Value aggregation", "FPS & downsample"]
STAGES = ["enc1", "enc2", "enc3", "enc4", "enc5"]

epoch_means = df.drop(columns="epoch").mean()
op_sums = {op: float(np.sum([epoch_means[f"{s}/{op}"] for s in STAGES])) for op in OPS}

total = sum(op_sums.values())
labels_sorted = sorted(op_sums, key=op_sums.get, reverse=True)
values_sorted = [op_sums[op] for op in labels_sorted]

PALETTE = {
    "KNN query": "#2C6FAC",
    "Relation & weight encoding": "#B84A28",
    "FPS & downsample": "#4A8F52",
    "Relative positional encoding": "#7048A0",
    "Value aggregation": "#A83858",
    "QKV encoding": "#6B6A62",
}
FIG_W, FIG_H = 7.0, 4.0
BG_COLOR = "#606060" # gris sombre

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)
ax.axis("off")

rects = squarify_layout(values_sorted, 0, 0, FIG_W, FIG_H)
PAD = 0.022

for rect, op in zip(rects, labels_sorted):
    x, y, w, h = rect["x"], rect["y"], rect["dx"], rect["dy"]
    val = op_sums[op]
    pct = val / total * 100

    ax.add_patch(FancyBboxPatch(
        (x + PAD, y + PAD), max(0, w - 2*PAD), max(0, h - 2*PAD),
        boxstyle="round,pad=0.005",
        linewidth=0.6, edgecolor="white",
        facecolor=PALETTE[op], zorder=2,
    ))

    cx, cy = x + w / 2, y + h / 2
    fs_main = min(12, w * 11.8, h * 16)
    fs_sub = min(10, w * 10.5,  h * 14)
    show_name = w > 0.50 and h > 0.25
    show_ms = w > 0.55 and h > 0.42
    show_pct = w > 0.60 and h > 0.60

    n_lines = sum([show_name, show_ms, show_pct])
    line_gap = min(0.17, h * 0.25)
    yy = cy + (n_lines - 1) * line_gap / 2

    W = "white"
    WD = (1.0, 1.0, 1.0, 0.78)

    if show_name:
        LABEL_OVERRIDE = {
            "Relation & weight encoding": "Relation &\nweight encoding",
            "FPS & downsample": "FPS &\ndownsample",
            "Value aggregation":            "Value\naggregation",
            "Relative positional encoding": "Relative\npositional\nencoding",
        }
        name = LABEL_OVERRIDE.get(op, op)
        n_label_lines = name.count("\n") + 1
        ax.text(cx, yy, name, ha="center", va="center", zorder=3,
                fontsize=fs_main, fontweight="bold", color=W, clip_on=True,
                multialignment="center", linespacing=1.3)
        yy -= line_gap * n_label_lines
    if show_ms:
        ax.text(cx, yy, f"{val:.1f} ms", ha="center", va="center", zorder=3,
                fontsize=fs_sub, fontweight="demibold", color=WD, clip_on=True)
        yy -= line_gap
    if show_pct:
        ax.text(cx, yy, f"({pct:.1f}%)", ha="center", va="center", zorder=3,
                fontsize=fs_sub - 0.5, fontweight="demibold", color=WD, clip_on=True)

ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.set_aspect("equal")

plt.tight_layout(pad=0.2)

OUT = "figures/generated_figures/latency_treemap.pdf"
fig.savefig(OUT, format="pdf", bbox_inches="tight")