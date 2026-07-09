#!/usr/bin/env python3
"""§2.2 RAW kernel-time-sum — 4 cases (H100/GB300 x nj/jag), ABSOLUTE per-step rank0 busy-ms.
Each kernel's full GPU busy-time summed by category (overlaps double-counted, so the bar TOTAL exceeds the step
wall-time and NCCL is over-counted). Absolute (not %-normalized) so magnitudes compare directly across cases —
e.g. H100's NCCL GPU-time dwarfs GB300's. Numbers from kernsum_categorize.py on cuda_gpu_kern_sum (/GPUs /20 steps)."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

CASES = ["H100 nj", "H100 jagged", "GB300 nj", "GB300 jagged"]
CATS  = ["Attention", "NCCL", "GEMM", "Eltwise", "Embedding", "Other"]
DATA  = {  # per-step rank0 busy-time (ms), summed (overlaps double-counted)
    "H100 nj":      [78.4, 236.8, 43.6, 42.1, 3.0, 4.1],
    "H100 jagged":  [18.0,  75.4, 10.9, 12.4, 1.2, 1.3],
    "GB300 nj":     [14.4,  10.2, 10.2, 27.9, 2.7, 0.2],
    "GB300 jagged": [ 4.5,  14.3,  2.7,  9.0, 1.1, 0.1],
}
COL = {"Attention":"#e6194B","NCCL":"#4363d8","GEMM":"#3cb44b","Eltwise":"#f58231","Embedding":"#911eb4","Other":"#9A9A9A"}

fig, ax = plt.subplots(figsize=(8.8, 5.0))
y = range(len(CASES))
left = [0.0]*len(CASES)
for ci, cat in enumerate(CATS):
    vals = [DATA[c][ci] for c in CASES]
    ax.barh(list(y), vals, left=left, color=COL[cat], label=cat, edgecolor="white", height=0.62)
    for i, (v, l) in enumerate(zip(vals, left)):
        if v >= 12: ax.text(l+v/2, i, f"{v:.0f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    left = [l+v for l, v in zip(left, vals)]
WALL = {"H100 nj": 387, "H100 jagged": 121, "GB300 nj": 80, "GB300 jagged": 63}  # clean per-step wall (nvtx :step median)
for i, c in enumerate(CASES):
    ax.text(sum(DATA[c])+4, i, f"Σ {sum(DATA[c]):.0f}", va="center", fontsize=8.5, color="#333", fontweight="bold")
    w = WALL[c]
    ax.plot([w, w], [i-0.36, i+0.36], color="black", lw=2.0, zorder=5)  # clean wall marker
    ax.text(w, i-0.44, f"wall {w}", ha="center", va="bottom", fontsize=7.5, color="black")
ax.set_yticks(list(y)); ax.set_yticklabels(CASES, fontsize=10); ax.invert_yaxis()
ax.set_xlim(0, 460); ax.set_xlabel("per-step rank0 GPU busy-time (ms), summed — overlaps double-counted (Σ ≠ step wall-time)")
ax.set_title("Raw kernel-time-sum by category, absolute — magnitudes compare directly\n"
             "H100-nj NCCL 237 ms vs GB300-nj 10 ms = 23× (2-node IB SendRecv vs coherent NVLink)", fontsize=10.5)
ax.legend(ncol=6, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.20), frameon=False, columnspacing=1.2)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(rect=[0, 0.04, 1, 1])
base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "perf_kernsum_raw4")
fig.savefig(base+".png", dpi=200, bbox_inches="tight"); fig.savefig(base+".pdf", bbox_inches="tight")
print("wrote", base+".png/.pdf")
