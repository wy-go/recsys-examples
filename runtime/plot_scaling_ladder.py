#!/usr/bin/env python3
"""§6a: e2e scaling efficiency vs GPU count — H100 (IB) vs GB300 (NVL72 NVLink). Efficiency = per-GPU throughput
retained vs the 4-GPU baseline (ideal linear scaling = 100%). H100 holds within a node then drops to ~74% crossing to
2 IB nodes at 16; GB300 stays ~99% flat to 32 on one NVL72 domain. Plotting *efficiency* (not raw MFU) makes it
apples-to-apples across the two platforms' very different absolute levels. Raw global TFLOPS annotated on each point."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=os.path.dirname(os.path.abspath(__file__)); FIG=os.path.join(HERE,"..","figures")
# (GPUs, global TFLOPS); efficiency computed as (TFLOPS/GPU)/(TFLOPS/GPU at 4) * 100
H100=[(4,1315.65),(8,2662.53),(16,3922.64)]
GB300=[(4,762),(8,1520),(16,3009),(32,6027)]
def eff(rows):
    base=rows[0][1]/rows[0][0]
    return [(g, tf, (tf/g)/base*100) for g,tf in rows]
H,G=eff(H100),eff(GB300)
GPUS=[4,8,16,32]; xof={g:i for i,g in enumerate(GPUS)}

fig,ax=plt.subplots(figsize=(7.6,4.8))
ax.axhline(100,ls="--",lw=1.2,color="#bbb",zorder=0)
ax.text(3.05,100,"ideal linear",fontsize=8,color="#999",va="center")
for rows,lab,col,mk in [(H,"H100 · IB (2 DGX at 16)","#f58231","o"),(G,"GB300 · one NVL72 NVLink","#12a594","s")]:
    xs=[xof[g] for g,_,_ in rows]; ys=[e for _,_,e in rows]
    ax.plot(xs,ys,mk+"-",color=col,lw=2.4,ms=8,label=lab)
    for (g,tf,e) in rows:
        ax.annotate(f"{tf:.0f} TF\n{e:.0f}%",(xof[g],e),xytext=(0,9 if col=="#12a594" else -20),
                    textcoords="offset points",ha="center",fontsize=8,color=col,fontweight="bold",linespacing=0.9)
# highlight the H100 drop
ax.annotate("→ 75%\n(−25% vs ideal:\nIB 1→2 nodes)",(xof[16],H[2][2]),xytext=(xof[16]-0.05,55),ha="center",fontsize=8.5,
            color="#f58231",arrowprops=dict(arrowstyle="->",color="#f58231",lw=1.3))
ax.set_xticks(list(xof.values())); ax.set_xticklabels([str(g) for g in GPUS],fontsize=10)
ax.set_xlabel("GPUs",fontsize=11); ax.set_ylabel("scaling efficiency\n(per-GPU throughput vs 4-GPU baseline)",fontsize=10)
ax.set_ylim(45,110); ax.set_xlim(-0.3,3.4)
ax.set_title("E2E scaling: GB300 stays flat over NVLink; H100 drops to 75% at 16 GPU (2 IB nodes)",fontsize=10.5)
ax.legend(loc="lower left",fontsize=9,frameon=False); ax.grid(axis="y",alpha=0.3)
fig.tight_layout()
base=os.path.join(FIG,"perf_scaling_ladder"); os.makedirs(FIG,exist_ok=True)
fig.savefig(base+".png",dpi=200,bbox_inches="tight"); fig.savefig(base+".pdf",bbox_inches="tight")
print("wrote",base+".png/.pdf")
