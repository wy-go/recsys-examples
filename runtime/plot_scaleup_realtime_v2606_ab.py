#!/usr/bin/env python3
"""§8a v26.05 -> v26.06 per-step real-time view: per-step exposed breakdown in ABSOLUTE ms, in run order (step 1..80),
on the MATCHED 1B/4096/lognormal config (8-GPU). Two panels (v26.05 | v26.06), SHARED y-axis so the v26.06 speedup is
visible as shorter bars. Bar height = that step's wall-clock; stacked bands = where the time went. Same colours/stack as
the §8 realtime figure (plot_scaleup_realtime.py). Data: nsys_repro/scaleup_perstep/{v2605,v2606}.json (exposed_perstep.py)."""
import os, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE=os.path.dirname(os.path.abspath(__file__)); FIG=os.path.join(HERE,"..","figures")
DDIR=os.path.join(HERE,"nsys_repro","scaleup_perstep")
CFG=[("v2605","v26.05 · median 94.0 ms/step · MFU ~16%"),("v2606","v26.06 · median 77.5 ms/step · MFU ~19%")]
D={lab:json.load(open(os.path.join(DDIR,f"{lab}.json"))) for lab,_ in CFG}
STACK=[("GPU idle","IDLE","#9A9A9A"),("nccl(exposed)","NCCL(exp)","#4363d8"),
       ("hstu fwd/bwd (attention)","HSTU","#e6194B"),("gemm","GEMM","#3cb44b"),
       ("elementwise","ELEM","#f58231"),("embedding op","EMB","#911eb4"),("other","OTH","#bcbd22")]
def series(a,key):
    if key=="gemm": return np.array(a["gemm / uvqk"])+np.array(a["gemm / projection"])+np.array(a["gemm / others"])
    if key=="other": return np.array(a["nccl(overlap)"])+np.array(a["others"])+np.array(a["overlapped"])
    return np.array(a[key])

fig,axes=plt.subplots(2,1,figsize=(12,7.0),sharex=True,sharey=True)
for ax,(lab,title) in zip(axes,CFG):
    d=D[lab]; ms=np.array(d["step_ms"]); n=len(ms); x=np.arange(1,n+1); a=d["arrays"]
    bottom=np.zeros(n)
    for key,leg,col in STACK:
        h=series(a,key)/100.0*ms
        ax.bar(x,h,bottom=bottom,width=0.82,color=col,edgecolor="none",label=leg if ax is axes[0] else None)
        bottom+=h
    ax.axhline(d["fastest_ms"],ls=":",lw=1,color="#444")
    ax.text(n,d["fastest_ms"],f" fastest {d['fastest_ms']:.0f}ms",fontsize=7.5,color="#444",va="center")
    ax.axhline(d["median_ms"],ls="--",lw=1,color="#111")
    ax.text(0.7,d["median_ms"],f"median {d['median_ms']:.0f}ms",fontsize=7.5,color="#111",va="bottom")
    ax.set_ylabel("step time (ms)",fontsize=10); ax.set_title(title,fontsize=10.5,loc="left")
    ax.set_xlim(0.5,n+0.5); ax.margins(x=0)
axes[0].legend(ncol=7,fontsize=8,loc="upper center",bbox_to_anchor=(0.5,1.34),frameon=False,columnspacing=1.1,handlelength=1.2)
axes[-1].set_xlabel("step (run order)",fontsize=11)
fig.suptitle("§8a v26.05 → v26.06 — per-step GPU-time in absolute ms, run order (matched 1B/4096/lognormal, GB300 8-GPU, shared y-axis)\n"
             "every v26.06 bar is shorter (−17.6% median step, −18.9% total); the shrink is in the NCCL(exposed) + compute bands — the grey IDLE band is ~unchanged step-for-step",
             fontsize=11,y=0.995)
fig.tight_layout(rect=[0,0,1,0.93])
base=os.path.join(FIG,"perf_scaleup_realtime_v2606_ab"); os.makedirs(FIG,exist_ok=True)
fig.savefig(base+".png",dpi=200,bbox_inches="tight"); fig.savefig(base+".pdf",bbox_inches="tight")
print("wrote",base+".png/.pdf")
