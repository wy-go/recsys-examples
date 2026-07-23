#!/usr/bin/env python3
"""§3 (v26.06 doc) HSTU fused-layer sweep — v26.05 vs v26.06 image A/B, per kernel.
Two subplots: fused CUTLASS d128 and fused Triton d256. Left axis: TFLOPS bars (v26.05 vs v26.06).
Right axis: MFU% line per image (both on the 2500 bf16 peak). Same visual language as BENCHMARK_RESULTS §3
(plot_layer_compare.py), but the two bars are the two images instead of two platforms.
Both arms measured identically: hstu_layer_benchmark.py, --iters 100 --warmup-iters 50, 1L·4h·SL4096·B32,
bf16, --full-sequence. v26.05 numbers from BENCHMARK_RESULTS.md §3 (GB300); v26.06 measured on figs-v2606-s3-layer."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures", "v2606"); os.makedirs(FIG, exist_ok=True)
PEAK = 2500.0
A, B = "#d95f02", "#1b9e77"      # bar colours: v26.05 orange, v26.06 teal
AL, BL = "#a34702", "#0b6e4f"    # darker line colours for MFU
phases = ["fwd", "bwd", "e2e"]

# fwd, bwd, e2e TFLOPS (measured, matched iters). MFU derived = TF / 2500.
data = {
    "fused CUTLASS · d128": {"v26.05": [727.1, 687.4, 699.0], "v26.06": [754.8, 650.2, 678.7]},
    "fused Triton · d256":  {"v26.05": [706.0, 322.3, 388.0], "v26.06": [707.9, 325.1, 390.7]},
}

x = np.arange(len(phases)); w = 0.38
fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
for ax, (title, d) in zip(axes, data.items()):
    a_tf, b_tf = d["v26.05"], d["v26.06"]
    a_mfu = [100*t/PEAK for t in a_tf]; b_mfu = [100*t/PEAK for t in b_tf]
    b1 = ax.bar(x - w/2, a_tf, w, label="v26.05  TFLOPS", color=A)
    b2 = ax.bar(x + w/2, b_tf, w, label="v26.06  TFLOPS", color=B)
    for bars, tfs in ((b1, a_tf), (b2, b_tf)):
        for bar, t in zip(bars, tfs):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+6, f"{t:.0f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(phases, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", ls=":", alpha=0.4); ax.set_axisbelow(True); ax.margins(y=0.20)
    ax2 = ax.twinx()
    ax2.plot(x - w/2, a_mfu, "o-",  color=AL, lw=2, ms=7, label="v26.05  MFU")
    ax2.plot(x + w/2, b_mfu, "s--", color=BL, lw=2, ms=7, label="v26.06  MFU")
    for xi, m in zip(x - w/2, a_mfu): ax2.text(xi, m + 1.0, f"{m:.0f}%", ha="center", va="bottom", fontsize=8, color=AL)
    for xi, m in zip(x + w/2, b_mfu): ax2.text(xi, m + 1.0, f"{m:.0f}%", ha="center", va="bottom", fontsize=8, color=BL)
    ax2.set_ylim(0, 40); ax2.set_ylabel("MFU % (peak 2500)", fontsize=10.5)
    ax._ax2 = ax2

axes[0].set_ylabel("TFLOPS  (↑ higher is better)", fontsize=10.5)
h1, la1 = axes[0].get_legend_handles_labels()
h2, la2 = axes[0]._ax2.get_legend_handles_labels()
axes[0].legend(h1 + h2, la1 + la2, fontsize=9, loc="upper right", framealpha=0.9)
fig.suptitle("HSTU fused-layer — v26.05 vs v26.06 (bf16, 1L · 4h · SL4096 · B32, iters 100/warmup 50)   ·   bars = TFLOPS, line = MFU% (peak 2500)", fontsize=11)
fig.tight_layout()
base = os.path.join(FIG, "s3_layer")
fig.savefig(base + ".png", dpi=200, bbox_inches="tight")
fig.savefig(base + ".pdf", bbox_inches="tight")
plt.close(fig)
print("wrote", base + ".png / .pdf")
