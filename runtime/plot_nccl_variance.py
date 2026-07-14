#!/usr/bin/env python3
"""§5.2 companion: exposed-NCCL variability across ALL steps of a capture (not just the fastest).

Per-step exposed NCCL % (sole-active NCCL / step, same sweep as exposed_nccl_split.py, applied to every 'step N'
NVTX window) for our H100 and GB300 non-jagged captures. Shows the §5.2 sunbursts use the FASTEST step — the best
case — while the typical step exposes far more, and that exposure is NOT flat on either platform (it tracks step
duration: slow steps are slow *because* the collective stalls the GPU). Values from figs-{h100,gb300}-nj traces.
"""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=os.path.dirname(os.path.abspath(__file__)); FIG=os.path.join(HERE,"..","figures")

# per-step exposed NCCL % (nccl_perstep sweep on the fastest-to-slowest step windows)
H100=[21.86,21.87,21.87,21.87,21.87,21.87,21.87,21.88,22.53,22.53,22.55,22.55,22.55,22.56,22.57,22.57,22.7,22.7,22.7,22.71,22.71,22.71,22.72,22.72,23.38,23.38,23.39,23.39,23.39,23.42,23.42,23.44,24.21,24.22,24.22,24.22,24.23,24.23,24.23,24.23,24.23,24.23,24.24,24.24,24.24,24.25,24.25,24.25,24.38,24.38,24.39,24.4,24.4,24.4,24.41,24.44,24.64,24.64,24.64,24.64,24.64,24.65,24.65,24.65,25.0,25.05,25.06,25.09,25.13,25.14,25.14,25.17,25.26,25.27,25.3,25.32,25.32,25.33,25.34,25.34,25.35,25.35,25.36,25.36,25.36,25.36,25.37,25.44,25.61,25.62,25.62,25.63,25.63,25.63,25.63,25.63,25.98,25.98,25.98,25.99,25.99,26.0,26.0,26.09,26.62,26.63,26.63,26.64,26.64,26.64,26.65,26.67,30.37,30.38,30.38,30.38,30.38,30.39,30.39,30.39,34.83,34.84,34.85,34.85,34.85,34.85,34.88,34.88,55.63,55.63,55.65,55.66,55.66,55.66,55.69,55.69,55.86,55.86,55.87,55.88,55.88,55.88,55.88,55.88,58.9,59.0,59.01,59.05,59.07,59.1,59.13,59.13,59.24,59.27,59.28,59.29,59.31,59.32,59.32,59.33]
GB300=[5.53,6.13,6.14,6.16,6.18,6.92,7.08,7.73,7.82,8.77,9.07,10.7,11.04,11.06,11.43,11.92,11.92,11.97,12.0,12.54,12.95,13.01,13.1,13.13,13.13,13.15,13.51,13.63,13.75,13.94,14.73,15.05,15.07,15.07,15.27,16.92,17.04,17.1,17.18,18.33,18.65,18.82,19.0,20.61,21.07,21.25,21.26,21.3,21.66,22.33,22.33,22.44,22.91,22.97,23.71,23.91,24.47,25.1,25.64,25.7,25.7,25.9,26.62,26.67,26.81,26.86,26.87,27.08,29.98,30.16,30.41,30.44,30.46,30.68,30.7,31.17,33.35,33.54,33.54,34.83]
UPSTREAM=1.6   # upstream's published fastest-step exposed NCCL (single number; no trace to distribute)

series=[("Our H100\n(2 DGX · IB)",H100,"#f58231"),("Our GB300\n(1 NVL72 · NVLink)",GB300,"#12a594")]
fig,ax=plt.subplots(figsize=(7.2,5.2))
med=lambda v: sorted(v)[len(v)//2]
for i,(lab,v,col) in enumerate(series,1):
    ax.boxplot([v],positions=[i],widths=0.5,whis=(0,100),patch_artist=True,
               boxprops=dict(facecolor=col,alpha=0.28,edgecolor=col),
               medianprops=dict(color=col,linewidth=2.2),
               whiskerprops=dict(color=col),capprops=dict(color=col),showfliers=False)
    xs=[i-0.28+0.56*((k*0.61803)%1) for k in range(len(v))]   # deterministic jitter
    ax.scatter(xs,v,s=9,color=col,alpha=0.55,zorder=3,edgecolors="none")
    ax.annotate(f"median {med(v):.0f}%",(i,med(v)),xytext=(i+0.34,med(v)),fontsize=9,color=col,fontweight="bold",va="center")
    ax.annotate(f"fastest {min(v):.0f}%\n(= §5.2 sunburst)",(i,min(v)),xytext=(i+0.30,min(v)-3),fontsize=8,color="#555",va="top")
ax.axhline(UPSTREAM,ls="--",lw=1.3,color="#888")
ax.annotate(f"upstream fastest step {UPSTREAM}%",(1.5,UPSTREAM),xytext=(0.6,UPSTREAM+2.5),fontsize=8.5,color="#666")
ax.set_xticks([1,2]); ax.set_xticklabels([s[0] for s in series],fontsize=10)
ax.set_ylabel("exposed NCCL — % of step",fontsize=11); ax.set_ylim(0,63)
ax.set_title("Exposed NCCL is variable, not flat — the §5.2 fastest step is the best case\n"
             "(per-step, non-jagged; slow steps are slow because the collective stalls the GPU)",fontsize=11)
ax.grid(axis="y",alpha=0.3)
fig.tight_layout()
base=os.path.join(FIG,"perf_nccl_variance"); os.makedirs(FIG,exist_ok=True)
fig.savefig(base+".png",dpi=200,bbox_inches="tight"); fig.savefig(base+".pdf",bbox_inches="tight")
print("wrote",base+".png/.pdf")
