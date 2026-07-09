#!/usr/bin/env python3
"""§4c GPU-time breakdown — 3-way (upstream-H100 / our-H100 / our-GB300), reproducing PERF_ANALYSIS §2.2.
Buckets nsys `cuda_gpu_kern_sum` kernels into categories and renders a stacked composition bar.
Usage: python3 runtime/plot_perf_breakdown.py   (reads scratchpad CSVs baked as constants below if absent)
"""
import os, re, csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=os.path.dirname(os.path.abspath(__file__)); FIG=os.path.join(HERE,"..","figures")
DATA=os.environ.get("NSYSDATA","/tmp/claude-1000/-opt-tiger-rl-gr/78428c37-f4a4-465d-9f70-adcb7f1b5af8/scratchpad/nsysdata")

# category -> regex (first match wins, in this order)
CATS=[("Attention (HSTU)", r"flash::compute_attn|HSTUAttention|Hstu_fwd|Hstu_bwd|hstu.*attn"),
      ("NCCL (comms)",      r"nccl"),
      ("GEMM (dense)",      r"nvjet|cublas|ampere_|sm\d+_gemm|cutlass_\w*gemm"),
      ("Norm/Act/Eltwise",  r"layer_norm|_ln_mul|silu|dropout|elementwise|reduce_kernel|FillFunctor|CatArray|clamp|Functor"),
      ("Embedding/sparse",  r"embedding|dyn_emb|dynamicemb|fbgemm|bucketize|permute|radix_sort|DeviceScan|DeviceCompact|DeviceSelect|RunLength|linearize_index"),
      ("Copy/other",        r".*")]

def bucket(csvpath):
    tot={c[0]:0.0 for c in CATS}
    with open(csvpath) as f:
        for row in csv.reader(f):
            if not row or not re.match(r"^[0-9.]+$", row[0].strip()): continue
            pct=float(row[0]); name=row[-1]
            for cat,rx in CATS:
                if re.search(rx,name): tot[cat]+=pct; break
    return tot

# our measured runs
h100=bucket(os.path.join(DATA,"h100_cuda_gpu_kern_sum.csv"))
gb300=bucket(os.path.join(DATA,"gb300_cuda_gpu_kern_sum.csv"))
# upstream from PERF_ANALYSIS.md §2.2 (H100 16-GPU rank0, their category names mapped to ours)
upstream={"Attention (HSTU)":43.09, "NCCL (comms)":2.04, "GEMM (dense)":23.68,
          "Norm/Act/Eltwise":21.31, "Embedding/sparse":5.74, "Copy/other":4.14}

# upstream dropped here: this figure is RAW kernel-time-sum, a different accounting than upstream's exposed §2.2 (see the
# exposed sunburst for the upstream-comparable, method-matched figure).
runs=[("Our H100\n(D256)",h100), ("Our GB300\n(D128)",gb300)]
names=[c[0] for c in CATS]
colors={"Attention (HSTU)":"#e6194B","NCCL (comms)":"#4363d8","GEMM (dense)":"#3cb44b",
        "Norm/Act/Eltwise":"#f58231","Embedding/sparse":"#911eb4","Copy/other":"#9A9A9A"}

# horizontal stacked bars, one per platform
fig,ax=plt.subplots(figsize=(10,3.6))
for i,(lbl,d) in enumerate(runs):
    left=0
    for cat in names:
        v=d.get(cat,0.0)
        ax.barh(i,v,left=left,color=colors[cat],edgecolor="white",height=0.62)
        if v>=4: ax.text(left+v/2,i,f"{v:.0f}",ha="center",va="center",fontsize=9,color="white",fontweight="bold")
        left+=v
ax.set_yticks(range(len(runs))); ax.set_yticklabels([r[0].replace("\n"," ") for r in runs],fontsize=10)
ax.set_xlim(0,100); ax.invert_yaxis()
ax.set_xlabel("% of GPU-kernel time (RAW cuda_gpu_kern_sum, exp4_caching_hr 16-GPU)",fontsize=10)
ax.set_title("HSTU GPU-time composition — our H100 vs our GB300 (raw kernel-time-sum)",fontsize=12)
handles=[plt.Rectangle((0,0),1,1,color=colors[c]) for c in names]
ax.legend(handles,names,ncol=6,fontsize=8.5,loc="upper center",bbox_to_anchor=(0.5,-0.22),frameon=False)
fig.tight_layout()
base=os.path.join(FIG,"perf_breakdown_3way"); os.makedirs(FIG,exist_ok=True)
fig.savefig(base+".png",dpi=200,bbox_inches="tight"); fig.savefig(base+".pdf",bbox_inches="tight")
plt.close(fig)
# print the category table
print(f"{'category':22s} {'upstream':>9s} {'ourH100':>9s} {'ourGB300':>9s}")
for c in names:
    print(f"{c:22s} {upstream.get(c,0):8.1f}% {h100.get(c,0):8.1f}% {gb300.get(c,0):8.1f}%")
print("wrote", base+".png/.pdf")
