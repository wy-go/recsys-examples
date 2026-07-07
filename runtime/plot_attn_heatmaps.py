#!/usr/bin/env python3
"""Render the §2 HSTU attention-kernel heatmaps from the raw benchmark tables in figures/attn_data/.
Each cell shows TFLOPS with (MFU%) beneath. Batch axis: small-at-top -> big-at-bottom (matches upstream).
Usage:  python3 runtime/plot_attn_heatmaps.py            # writes into figures/gb300 and figures/h100
"""
import re, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "figures", "attn_data")
FIGROOT = os.path.join(HERE, "..", "figures")
ROW = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s*\|(.*)$")

def parse(path):
    cells, bss, sls = {}, set(), set()
    for line in open(path):
        m = ROW.match(line)
        if not m:
            continue
        bs, sl, tok, rest = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
        if tok != bs * sl:          # skip non-data rows (tokens must equal bs*sl)
            continue
        bss.add(bs); sls.add(sl)
        mk = "OVF" if "OVERFLOW" in rest else ("OOM" if "OOM" in rest else None)
        if mk:
            cells[(bs, sl)] = (np.nan, np.nan, np.nan, mk); continue
        n = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", rest.replace("%", ""))][:9]
        cells[(bs, sl)] = (n[1], n[4], n[7], None)   # fwd_tf, bwd_tf, (fwd+bwd)_tf
    return cells, sorted(bss), sorted(sls)

def heatmap(ax, cells, bss, sls, idx, title, peak):
    M = np.full((len(bss), len(sls)), np.nan); marks = {}
    for i, b in enumerate(bss):
        for j, s in enumerate(sls):
            v = cells.get((b, s))
            if v is None: continue
            if v[3] is not None: marks[(i, j)] = v[3]
            else: M[i, j] = v[idx]
    vmax = np.nanmax(M) if np.isfinite(M).any() else 1.0
    vmin = max(np.nanmin(M[M > 0]) if (M > 0).any() else 0.1, 0.05)
    # origin="upper" -> row 0 (smallest batch) at TOP, largest batch at bottom (matches upstream table order)
    # interpolation="nearest" -> crisp cell blocks when embedded in the vector SVG/PDF
    im = ax.imshow(M, aspect="auto", origin="upper", cmap="viridis", norm=LogNorm(vmin=vmin, vmax=vmax),
                   interpolation="nearest")
    ax.set_xticks(range(len(sls))); ax.set_xticklabels(sls, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(bss))); ax.set_yticklabels(bss, fontsize=9)
    ax.set_xlabel("SeqLen", fontsize=11); ax.set_ylabel("Batch", fontsize=11)
    ax.set_title(title, fontsize=12)
    # top-2 finite cells in this panel -> highlight the peak (red) and runner-up (blue)
    fin = sorted(((M[i, j], i, j) for i in range(len(bss)) for j in range(len(sls))
                  if np.isfinite(M[i, j])), reverse=True)
    top1 = fin[0][1:] if fin else (-1, -1)
    top2 = fin[1][1:] if len(fin) > 1 else (-1, -1)
    for i in range(len(bss)):
        for j in range(len(sls)):
            if (i, j) in marks:
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, color="0.82"))
                ax.text(j, i, marks[(i, j)], ha="center", va="center", fontsize=8, color="0.35")
            elif np.isfinite(M[i, j]):
                t = M[i, j]; mfu = 100.0 * t / peak
                lbl = f"{t:.0f}" if t >= 10 else f"{t:.1f}"
                mlbl = f"({mfu:.0f}%)" if mfu >= 10 else f"({mfu:.1f}%)"
                if (i, j) == top1:   col, wt, edge = "#e6194B", "bold", "#e6194B"   # max  -> red
                elif (i, j) == top2: col, wt, edge = "#1f6feb", "bold", "#1f6feb"   # 2nd  -> blue
                else:                col, wt, edge = ("white" if t < vmax * 0.5 else "black"), "normal", None
                ax.text(j, i, f"{lbl}\n{mlbl}", ha="center", va="center", fontsize=7,
                        linespacing=0.95, color=col, fontweight=wt)
                if edge:
                    ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor=edge, lw=1.8))
    return im

def figure(src, out, suptitle, peak):
    cells, bss, sls = parse(os.path.join(DATA, src))
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.4))
    for ax, idx, nm in zip(axes, (0, 1, 2), ("fwd", "bwd", "fwd+bwd")):
        im = heatmap(ax, cells, bss, sls, idx, f"{nm} TFLOPS  (MFU%)", peak)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    best = max(((v[2], b, s) for (b, s), v in cells.items() if v[3] is None and np.isfinite(v[2])), default=(0, 0, 0))
    fig.suptitle(f"{suptitle}   |   peak fwd+bwd {best[0]:.0f} TFLOPS @ BS={best[1]} SeqLen={best[2]}", fontsize=14, y=1.02)
    fig.text(0.5, -0.015, "per panel: red = highest cell · blue = 2nd-highest", ha="center", fontsize=10, color="0.4")
    fig.tight_layout()
    base = os.path.join(FIGROOT, out[:-4]); os.makedirs(os.path.dirname(base), exist_ok=True)
    fig.savefig(base + ".png", dpi=200, bbox_inches="tight")   # sharp PNG — the only format GitLab renders inline in .md
    fig.savefig(base + ".pdf", bbox_inches="tight")            # vector PDF — download / print / zoom
    plt.close(fig)
    print(f"  {out[:-4]:30s}.svg/.pdf  {len(bss)}x{len(sls)} grid, peak f+b {best[0]:.0f}TF@BS{best[1]}/SL{best[2]}")

GB, HP = 2500.0, 989.0   # bf16 dense peaks (GB300 per NVL72 Quick Start Guide; H100)
JOBS = [
    ("gb300_cut128.txt",     "gb300/attn_cutlass_kv128.png", "GB300 — Blackwell CUTLASS, kv128 (head_dim 128)", GB),
    ("gb300_tri128.txt",     "gb300/attn_triton_kv128.png",  "GB300 — Triton, kv128 (head_dim 128)", GB),
    ("gb300_tri256.txt",     "gb300/attn_triton_kv256.png",  "GB300 — Triton, kv256 (head_dim 256)", GB),
    ("gb300_tri256_ext.txt", "gb300/attn_triton_kv256_ext.png", "GB300 — Triton kv256, EXTENDED grid (BS≤256 / SeqLen≤32768)", GB),
    ("h100_orig.txt",        "h100/attn_cutlass_kv256.png",  "H100 — Hopper CUTLASS, kv256 (head_dim 256)", HP),
    ("h100_grid128.txt",     "h100/attn_cutlass_kv128.png",  "H100 — Hopper CUTLASS, kv128 (head_dim 128)", HP),
    ("h100_c64.txt",         "h100/attn_cutlass_kv64.png",   "H100 — Hopper CUTLASS, kv64 (head_dim 64)", HP),
    ("h100_t128.txt",        "h100/attn_triton_kv128.png",   "H100 — Triton, kv128 (head_dim 128)", HP),
    ("h100_t256.txt",        "h100/attn_triton_kv256.png",   "H100 — Triton, kv256 (head_dim 256)", HP),
]
for src, out, title, peak in JOBS:
    if os.path.exists(os.path.join(DATA, src)): figure(src, out, title, peak)
    else: print(f"  SKIP (missing) {src}")
print("done")
