#!/usr/bin/env python3
"""§8 companion: exposed GPU-time breakdown across the scale-study sweep, as stacked bars (one bar per config).
Same exposed accounting + colour scheme as the §5.2 sunbursts (exposed_faststep.py + exposed_gemm_split.py, % of the
fastest step). Reads left-to-right as the §8 story: the §5.2(B) exp4 reference, then the prefetch ablation (exp5), then
the 1B sweep over zipf and lognormal sequence lengths. IDLE (grey, bottom) collapses 55%->20% as sequences lengthen and
the step becomes compute/elementwise-bound; exposed NCCL (blue) shrinks as a % as the useful work grows around the fixed
all-reduce. Numbers match the §8 exposed table exactly."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=os.path.dirname(os.path.abspath(__file__)); FIG=os.path.join(HERE,"..","figures")

CONFIGS=["§5.2(B)\n50M/2048 exp4\n(reference)","50M/2048\nexp5","1B/4096\nzipf","1B/4096\nlognormal"]
# stacking order (bottom->top) with the sunburst grouping colours; nccl split exposed/overlap
SEG=[  # (label, colour, [exp4, exp5, zipf, logn])
  ("GPU idle","#9A9A9A",             [55.5,62.5,47.4,20.4]),
  ("nccl (exposed)","#4363d8",       [19.1,11.5, 9.7, 6.6]),
  ("nccl (overlap)","#9fb0e8",       [ 0.4, 0.1, 0.6, 0.7]),
  ("hstu fwd/bwd","#e6194B",         [ 6.4, 6.6,15.0,21.4]),
  ("gemm","#3cb44b",                 [ 3.8, 3.9, 5.6,12.3]),
  ("elementwise","#f58231",          [12.7,13.0,18.5,34.6]),
  ("embedding","#911eb4",            [ 1.3, 1.5, 2.2, 2.6]),
  ("others","#bcbd22",               [ 0.6, 0.6, 1.0, 1.1]),
  ("overlapped","#42d4f4",           [ 0.1, 1.5, 0.7, 2.1]),
]
x=list(range(len(CONFIGS)))
fig,ax=plt.subplots(figsize=(8.4,6.0))
bottom=[0]*len(CONFIGS)
for lab,col,vals in SEG:
    ax.bar(x,vals,0.62,bottom=bottom,color=col,edgecolor="white",linewidth=0.6,label=lab)
    for xi,(v,b) in enumerate(zip(vals,bottom)):
        # annotate the story segments (idle / exposed nccl / attention / elementwise) when big enough
        if lab in ("GPU idle","nccl (exposed)","hstu fwd/bwd","elementwise") and v>=4:
            ax.text(xi,b+v/2,f"{v:.0f}",ha="center",va="center",fontsize=8.5,
                    color="white",fontweight="bold")
    bottom=[b+v for b,v in zip(bottom,vals)]
ax.set_xticks(x); ax.set_xticklabels(CONFIGS,fontsize=8.5)
ax.set_ylabel("exposed GPU-time — % of the fastest step",fontsize=11)
ax.set_ylim(0,104)
ax.set_title("§8 scale-study GPU-time breakdown — idle collapses 55→20% as sequences lengthen;\n"
             "the step goes idle-bound → elementwise/compute-bound (exposed accounting, fastest step)",fontsize=10.5)
ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.09),ncol=5,fontsize=8.5,frameon=False,columnspacing=1.2)
ax.grid(axis="y",alpha=0.3)
fig.tight_layout()
base=os.path.join(FIG,"perf_scaleup_breakdown"); os.makedirs(FIG,exist_ok=True)
fig.savefig(base+".png",dpi=200,bbox_inches="tight"); fig.savefig(base+".pdf",bbox_inches="tight")
print("wrote",base+".png/.pdf")
