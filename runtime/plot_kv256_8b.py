#!/usr/bin/env python3
"""BENCHMARK_RESULTS §8b figure — what FBGEMM dev PR #18 (Blackwell head_dim-256 CuTe backward) buys.

left : fused-layer fwd/bwd/e2e — TFLOPS bars with the MFU% annotated per bar (no separate MFU line:
       every arm shares the 2500 TF peak, so MFU is the same ranking), §3 Dark2 palette, dotted y-grid.
       Four arms — Triton d256 (pre-#18), CUTLASS d256 (PR #18), d128·h4, d128·h8 (same shape as d256·4).
right: e2e training MFU vs batch at the upstream-default shape (kv256, heads=4) — the pair that isolates
       the kernel — plus the d128·h8 same-shape arm as a dashed secondary reference. Same arm colours.

Data sources: layer = kv256micro-* branches + historical §3 v26.06 row for Triton d256 (all measured with
hstu_layer_benchmark.py, --iters 100 --warmup-iters 50, 1L·SL4096·B32, bf16, --full-sequence);
e2e = kvab-* branches, mean of n=3 medians.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures", "v2606")
os.makedirs(FIG, exist_ok=True)

PEAK = 2500.0
phases = ["fwd", "bwd", "e2e"]

# arm -> (bar colour, darker MFU-line colour, line style) — §3 Dark2 family: orange = before, teal = after
ARMS = [
    ("Triton d256 (pre-#18)",                "#d95f02", "#a34702", "o-"),
    ("CUTLASS d256 (PR #18)",                "#1b9e77", "#0b6e4f", "s--"),
    ("d128 · heads=4",                       "#7570b3", "#4a4680", "^-."),
    ("d128 · heads=8 (same shape N·D=1024)", "#e7298a", "#a81b63", "D:"),
]

# fused-layer TFLOPS per phase, arm order as ARMS; MFU derived = TF / 2500
LAYER = {
    "fwd": [708, 979, 751, 954],
    "bwd": [325, 510, 647, 808],
    "e2e": [391, 602, 675, 848],
}
# e2e training MFU (mean of n=3 medians), keyed by arm name (d128·h4 has no e2e arm)
E2E = {
    "Triton d256 (pre-#18)": {32: 9.67, 256: 13.84},
    "CUTLASS d256 (PR #18)": {32: 19.17, 128: 22.52, 256: 22.78},
    "d128 · heads=8 (same shape N·D=1024)": {32: 25.69, 128: 30.27},
}

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.6), gridspec_kw=dict(width_ratios=[1.3, 1]))

x = np.arange(len(phases)); w = 0.20
for i, (name, col, lcol, mk) in enumerate(ARMS):
    tf = [LAYER[p][i] for p in phases]
    mfu = [100 * t / PEAK for t in tf]
    xo = x + (i - 1.5) * w
    bars = ax.bar(xo, tf, w, label=name, color=col)
    for b, t, m in zip(bars, tf, mfu):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 8, f"{t:.0f}\n{m:.0f}%",
                ha="center", va="bottom", fontsize=8, linespacing=1.15)
ax.set_xticks(x); ax.set_xticklabels(phases, fontsize=11)
ax.set_title("fused layer — bf16, 1L · SL4096 · B32, full sequences, heads as labeled (§3 setup)", fontsize=10.5)
ax.grid(axis="y", ls=":", alpha=0.4); ax.set_axisbelow(True); ax.set_ylim(0, 1300)
ax.set_ylabel("TFLOPS  (labels: TFLOPS + MFU% on the shared 2500 peak)", fontsize=10.5)
ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9, ncol=2, columnspacing=0.9)

for name, col, lcol, mk in ARMS:
    if name not in E2E:
        continue
    pts = E2E[name]; bs = sorted(pts)
    ls = "--" if "heads=8" in name else "-"
    ax2.plot(bs, [pts[b] for b in bs], ls, marker=mk[0], ms=7, lw=2, color=col, label=name)
    for b in bs:
        dy = -13 if "Triton" in name else 7      # triton labels below the line (the arrow sits above it)
        ax2.annotate(f"{pts[b]:.1f}", (b, pts[b]), textcoords="offset points", xytext=(0, dy),
                     ha="center", fontsize=8, color=col)
ax2.annotate("", xy=(32, 19.17), xytext=(32, 9.67),
             arrowprops=dict(arrowstyle="->", color="#222", lw=1.4))
ax2.text(36, 14.2, "×1.98 step\nspeedup", fontsize=8.5, color="#222")
ax2.text(150, 31.9, "kv128 bs256 OOMs (both d256/d128 arms\nsit ~272 GB at N·D=1024)", fontsize=7.5, color="#666")
ax2.set_xscale("log", base=2); ax2.set_xticks([32, 128, 256]); ax2.set_xticklabels(["bs32", "bs128", "bs256"])
ax2.set_xlim(24, 400)
ax2.set_ylim(5, 34)
ax2.set_ylabel("e2e training MFU (%)", fontsize=10.5)
ax2.set_title("E2E — upstream-default shape (kv256, heads=4)", fontsize=10.5)
ax2.legend(fontsize=8, frameon=False, loc="lower right")
ax2.grid(ls=":", alpha=0.4); ax2.set_axisbelow(True)

fig.suptitle("GB300 §8b — head_dim-256 on Blackwell: FBGEMM dev PR #18 vs the Triton fallback "
             "(kv256 image, 1B rows, 8 GPU, lognormal/4096)   ·   bars = TFLOPS, annotated with MFU% (peak 2500)",
             fontsize=11.5)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIG, f"v2606_kv256_8b.{ext}"), dpi=200, bbox_inches="tight")
print("wrote figures/v2606/v2606_kv256_8b.{png,pdf}")
