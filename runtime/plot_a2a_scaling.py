#!/usr/bin/env python3
"""§6b: NVLink all-to-all bus bandwidth vs GPU count (= 4-GPU worker nodes x 4), per message size. x is the TRUE GPU
count (to scale). The picture: full ~610-670 GB/s @512MB at POWER-OF-2 rank counts (8/16/32/64 = 2/4/8/16 nodes), a
sharp drop to ~150-230 at 36/48/56 (9/12/14 nodes, NOT power of 2). Every gang was verified SINGLE-RACK full NVLink
(one nvidia-smi Fabric ClusterUUID each; 36/48/56/64 even share the SAME rack) and the dip REPRODUCED on re-run, so it
is not placement/cross-rack -- it's an NCCL all_to_all algorithm effect: pairwise-exchange is bandwidth-optimal only for
power-of-2 ranks, and 36/48/56 fall to a slower schedule. 8=power-of-2 workers=fast; 36=non-power-of-2=slow."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=os.path.dirname(os.path.abspath(__file__)); FIG=os.path.join(HERE,"..","figures")
# all single-rack full NVLink (ClusterUUID verified). 8/16/64 earlier runs; 32 (new) + 36/48/56 (re-run) this batch.
GPUS=[8,16,32,36,48,56,64]
BW={"@16MB":[274,181,137.5,91.1,114.7,123.8,94],
    "@128MB":[577,559,549.5,140.9,184.7,211.8,479],
    "@512MB":[668,659,651.8,153.3,189.9,228.2,613]}
POW2={8,16,32,64}   # power-of-2 rank counts (2/4/8/16 nodes) -> full-bandwidth all-to-all
COL={"@16MB":"#9ecae1","@128MB":"#4292c6","@512MB":"#08519c"}
RDMA={8:83,16:99.4,32:90.6,36:90.8,48:89.7,56:86.1,64:87.5}   # @512MB RDMA/IB baseline (flat ~83-99; IB spans 4-7 racks)
x=GPUS   # numeric x — position ∝ GPU count

fig,ax=plt.subplots(figsize=(8,4.8))
# shade the non-power-of-2 region (36-56)
ax.axvspan(34,58,color="#f0f0f0",zorder=0)
ax.text(46,700,"36 / 48 / 56 GPU (9/12/14 nodes):\nNON-power-of-2 → slower all-to-all\n(single-rack full NVLink, reproduced)",
        ha="center",va="top",fontsize=8,color="#888",style="italic")
ax.text(6.5,455,"8 / 16 / 32 / 64 GPU (2/4/8/16 nodes):\npower-of-2 → full ~610–670 @512MB",ha="left",va="center",fontsize=8.5,color="#08519c")
for lbl,ys in BW.items():
    ax.plot(x,ys,"o-",color=COL[lbl],lw=2,ms=7,label="a2a "+lbl)
    for xi,yi in zip(x,ys):
        if lbl=="@512MB":
            ax.annotate(f"{yi:.0f}",(xi,yi),xytext=(0,7),textcoords="offset points",ha="center",fontsize=8,
                        color=COL[lbl],fontweight="bold")
            ax.plot(xi,yi,marker="*" if xi in POW2 else "o",ms=13 if xi in POW2 else 7,
                    color="#08519c",zorder=5)
# RDMA @512MB baseline (IB / default scene) — the crossover floor
rx=[g for g in GPUS if RDMA.get(g) is not None]; ry=[RDMA[g] for g in rx]
ax.plot(rx,ry,"s--",color="#d95f0e",lw=2,ms=6,label="RDMA @512MB (IB)")
ax.annotate("RDMA (IB) ~75–99 flat",(rx[-1],ry[-1]),xytext=(6,-4),textcoords="offset points",
            fontsize=8,color="#d95f0e",fontweight="bold",va="top")
ax.set_xticks(GPUS); ax.set_xticklabels([f"{g}" for g in GPUS],fontsize=10)
ax.set_xlim(4,68)
ax.set_xlabel("GPUs in the NVLink supernode (x to scale; ★ = power-of-2 rank count)",fontsize=11)
ax.set_ylabel("all-to-all bus bandwidth (GB/s)",fontsize=11)
ax.set_ylim(0,720)
ax.set_title("NVLink all-to-all: full ~650 GB/s at power-of-2 rank counts (8/16/32/64), a slower schedule at 36/48/56",fontsize=10)
ax.legend(loc="center right",fontsize=9,frameon=False)
ax.grid(axis="y",alpha=0.3)
fig.tight_layout()
base=os.path.join(FIG,"perf_a2a_scaling"); os.makedirs(FIG,exist_ok=True)
fig.savefig(base+".png",dpi=200,bbox_inches="tight"); fig.savefig(base+".pdf",bbox_inches="tight")
print("wrote",base+".png/.pdf")
