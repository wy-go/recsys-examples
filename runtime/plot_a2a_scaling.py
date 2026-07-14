#!/usr/bin/env python3
"""§6b: NVLink all-to-all bus bandwidth vs GPU count, per message size. Single runs each — the ≥36-GPU points are
placement noise (which physical nodes the gang lands on), NOT an N-scaling law: 8/16 sit at ~660 @512MB, 36-56 drop to
147-225, then 64 jumps back to 613. x is categorical (uneven GPU counts) so the non-monotonicity reads directly."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=os.path.dirname(os.path.abspath(__file__)); FIG=os.path.join(HERE,"..","figures")
GPUS=[8,16,36,48,56,64]
BW={"@16MB":[274,181,88,104,120,94], "@128MB":[577,559,141,178,208,479], "@512MB":[668,659,147,190,225,613]}
COL={"@16MB":"#9ecae1","@128MB":"#4292c6","@512MB":"#08519c"}
x=list(range(len(GPUS)))

fig,ax=plt.subplots(figsize=(8,4.8))
# shade the placement-noise region (>=36 GPU) — everything right of the 16-GPU point
ax.axvspan(1.5,5.5,color="#f0f0f0",zorder=0)
ax.text(3.5,690,"≥36 GPU: single runs — placement noise\n(same full-NVLink topology, 4× spread)",
        ha="center",va="top",fontsize=8.5,color="#888",style="italic")
ax.text(0.5,690,"8 / 16 GPU:\nreliable ~660 @512MB",ha="center",va="top",fontsize=8.5,color="#08519c")
for lbl,ys in BW.items():
    ax.plot(x,ys,"o-",color=COL[lbl],lw=2,ms=7,label="a2a "+lbl)
    for xi,yi in zip(x,ys):
        if lbl=="@512MB": ax.annotate(f"{yi}",(xi,yi),xytext=(0,7),textcoords="offset points",ha="center",fontsize=8,color=COL[lbl],fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([f"{g}" for g in GPUS],fontsize=10)
ax.set_xlabel("GPUs in the NVLink supernode",fontsize=11)
ax.set_ylabel("all-to-all bus bandwidth (GB/s)",fontsize=11)
ax.set_ylim(0,720)
ax.set_title("NVLink all-to-all is non-monotonic beyond 16 GPU — placement noise, not an N-scaling law",fontsize=11)
ax.legend(loc="center right",fontsize=9,frameon=False)
ax.grid(axis="y",alpha=0.3)
fig.tight_layout()
base=os.path.join(FIG,"perf_a2a_scaling"); os.makedirs(FIG,exist_ok=True)
fig.savefig(base+".png",dpi=200,bbox_inches="tight"); fig.savefig(base+".pdf",bbox_inches="tight")
print("wrote",base+".png/.pdf")
