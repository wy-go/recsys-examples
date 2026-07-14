#!/usr/bin/env python3
"""§8 step-variability, real-time view: per-step exposed breakdown in ABSOLUTE ms, in run order (step 1..80). Bar
HEIGHT = that step's wall-clock; stacked bands = where the time went. Shows what the boxplot/sunburst can't — the
duration<->composition CORRELATION: the tall (slow) steps are tall because the NCCL(exposed) band balloons, i.e. slow
steps are slow because the collective stalls the GPU. Same colours as the sunbursts. Data: nsys_repro/scaleup_perstep/."""
import os, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE=os.path.dirname(os.path.abspath(__file__)); FIG=os.path.join(HERE,"..","figures")
DDIR=os.path.join(HERE,"nsys_repro","scaleup_perstep")
CFG=[("exp5","exp5 · 50M/2048"),("zipf","zipf · 1B/4096"),("logn","lognormal · 1B/4096")]
D={lab:json.load(open(os.path.join(DDIR,f"{lab}.json"))) for lab,_ in CFG}
# stack order (bottom->top) + colours; 'other' folds nccl-overlap/others/overlapped
STACK=[("GPU idle","IDLE","#9A9A9A"),("nccl(exposed)","NCCL(exp)","#4363d8"),
       ("hstu fwd/bwd (attention)","HSTU","#e6194B"),("gemm","GEMM","#3cb44b"),
       ("elementwise","ELEM","#f58231"),("embedding op","EMB","#911eb4"),("other","OTH","#bcbd22")]
def series(a,key):
    if key=="gemm": return np.array(a["gemm / uvqk"])+np.array(a["gemm / projection"])+np.array(a["gemm / others"])
    if key=="other": return np.array(a["nccl(overlap)"])+np.array(a["others"])+np.array(a["overlapped"])
    return np.array(a[key])

fig,axes=plt.subplots(3,1,figsize=(12,9.2),sharex=True)
for ax,(lab,title) in zip(axes,CFG):
    d=D[lab]; ms=np.array(d["step_ms"]); n=len(ms); x=np.arange(1,n+1); a=d["arrays"]
    bottom=np.zeros(n)
    for key,_,col in STACK:
        h=series(a,key)/100.0*ms   # % of step -> absolute ms
        ax.bar(x,h,bottom=bottom,width=1.0,color=col,edgecolor="none",label=_ if ax is axes[0] else None)
        bottom+=h
    ax.axhline(d["fastest_ms"],ls=":",lw=1,color="#444")
    ax.text(n,d["fastest_ms"],f" fastest {d['fastest_ms']:.0f}ms",fontsize=7.5,color="#444",va="center")
    ax.set_ylabel("step time (ms)",fontsize=10); ax.set_title(title,fontsize=10.5,loc="left")
    ax.set_xlim(0.5,n+0.5); ax.margins(x=0)
axes[0].legend(ncol=7,fontsize=8,loc="upper center",bbox_to_anchor=(0.5,1.42),frameon=False,columnspacing=1.1,handlelength=1.2)
axes[-1].set_xlabel("step (run order)",fontsize=11)
fig.suptitle("§8 jagged — per-step GPU-time in absolute ms, in run order (GB300 16-GPU)\n"
             "bar height = step wall-clock · the compute bands (HSTU/GEMM/ELEM) are steady; step-to-step variation is in the idle + NCCL(exposed) bands (host gaps / comms stalls)",
             fontsize=11,y=0.995)
fig.tight_layout(rect=[0,0,1,0.95])
base=os.path.join(FIG,"perf_scaleup_realtime"); os.makedirs(FIG,exist_ok=True)
fig.savefig(base+".png",dpi=200,bbox_inches="tight"); fig.savefig(base+".pdf",bbox_inches="tight")
print("wrote",base+".png/.pdf")
