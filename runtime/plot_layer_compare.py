#!/usr/bin/env python3
"""§3 HSTU fused-layer GB300-vs-H100 comparison, per head_dim.
Left axis: TFLOPS bars (GB300 vs H100). Right axis: MFU% as a line per platform. Output PNG (inline) + PDF.
Usage: python3 runtime/plot_layer_compare.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIGROOT = os.path.join(HERE, "..", "figures")
GB_PEAK, H_PEAK = 2500.0, 989.0                 # bf16 dense peaks
GB, H = "#1b9e77", "#d95f02"                    # bar colours (GB300 teal, H100 orange)
GBL, HL = "#0b6e4f", "#a34702"                  # darker line colours for MFU
phases = ["fwd", "bwd", "e2e"]

# FUSED layer TFLOPS (measured); MFU derived = TF / peak
data = {
    "d128  (GB300 CUTLASS · H100 Hopper CUTLASS)": {"GB300": [727.1, 687.4, 699.0], "H100": [361.5, 365.8, 360.6]},
    "d256  (GB300 triton · H100 Hopper CUTLASS)":  {"GB300": [706.0, 322.3, 388.0], "H100": [454.8, 381.8, 400.2]},
}

x = np.arange(len(phases)); w = 0.38
fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
for ax, (title, d) in zip(axes, data.items()):
    gb_tf, h_tf = d["GB300"], d["H100"]
    gb_mfu = [100*t/GB_PEAK for t in gb_tf]; h_mfu = [100*t/H_PEAK for t in h_tf]
    # left axis: TFLOPS bars
    b1 = ax.bar(x - w/2, gb_tf, w, label="GB300  TFLOPS", color=GB)
    b2 = ax.bar(x + w/2, h_tf, w, label="H100  TFLOPS", color=H)
    for bars, tfs in ((b1, gb_tf), (b2, h_tf)):
        for b, t in zip(bars, tfs):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+6, f"{t:.0f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(phases, fontsize=11)
    ax.set_title(title, fontsize=10.5)
    ax.grid(axis="y", ls=":", alpha=0.4); ax.set_axisbelow(True); ax.margins(y=0.20)
    # right axis: MFU% line per platform
    ax2 = ax.twinx()
    l1, = ax2.plot(x - w/2, gb_mfu, "o-",  color=GBL, lw=2, ms=7, label="GB300  MFU")
    l2, = ax2.plot(x + w/2, h_mfu, "s--", color=HL, lw=2, ms=7, label="H100  MFU")
    for xi, m in zip(x - w/2, gb_mfu): ax2.text(xi, m + 1.4, f"{m:.0f}%", ha="center", va="bottom", fontsize=8, color=GBL)
    for xi, m in zip(x + w/2, h_mfu): ax2.text(xi, m + 1.4, f"{m:.0f}%", ha="center", va="bottom", fontsize=8, color=HL)
    ax2.set_ylim(0, 62); ax2.set_ylabel("MFU %", fontsize=10.5)
    ax._ax2 = ax2   # keep ref

axes[0].set_ylabel("TFLOPS  (↑ higher is better)", fontsize=10.5)
# one combined legend (bars + MFU lines)
h1, la1 = axes[0].get_legend_handles_labels()
h2, la2 = axes[0]._ax2.get_legend_handles_labels()
axes[0].legend(h1 + h2, la1 + la2, fontsize=9, loc="upper left", framealpha=0.9)
fig.suptitle("HSTU fused-layer — GB300 vs H100 (bf16, 1L · 4h · SL4096 · B32)   ·   bars = TFLOPS, line = MFU%   ·   ↑ higher is better", fontsize=12)
fig.tight_layout()
base = os.path.join(FIGROOT, "layer_gb300_vs_h100")
fig.savefig(base + ".png", dpi=200, bbox_inches="tight")
fig.savefig(base + ".pdf", bbox_inches="tight")
plt.close(fig)
print("wrote", base + ".png / .pdf")
