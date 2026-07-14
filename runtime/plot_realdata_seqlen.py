#!/usr/bin/env python3
"""§8 reference: per-user sequence-length distributions of the REAL recsys datasets used by the HSTU / generative-
recommenders paper (MovieLens ml-1m/ml-20m, KuaiRand Pure/1K, Amazon Books), computed from the raw interaction logs
on the pod (groupby user -> #interactions, matching the HSTU preprocessor). Overlaid on log-x with the benchmark's
SYNTHETIC operating points (§5 zipf, §8 1B zipf/lognormal) so the reader can see the synthetic sequence lengths sit
inside the real range. Amazon Books is the raw/unfiltered McAuley-2014 ratings-only file (heavy single-review tail,
median 1); the paper additionally k-core-filters it. Data: runtime/realdata_seqlen.json
(a compact binned summary — aggregate counts only, no user-level data). Regenerate the summary with
scratchpad/compute_realdata_seqlen.py against the raw logs."""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=os.path.dirname(os.path.abspath(__file__)); FIG=os.path.join(HERE,"..","figures")
r=json.load(open(os.path.join(HERE,"realdata_seqlen.json")))
bins=np.array(r["bins"]); ctr=np.sqrt(bins[:-1]*bins[1:])   # geometric bin centres
COL={"MovieLens ml-20m":"#e6194B","MovieLens ml-1m":"#f58231","KuaiRand-Pure":"#12a594","KuaiRand-1K":"#4363d8","Amazon Books":"#911eb4"}

fig,ax=plt.subplots(figsize=(9,5.4))
for d in r["datasets"]:
    c=np.array(d["hist_counts"],float); frac=c/c.sum()   # PMF over log bins
    ax.plot(ctr,frac,"-",color=COL[d["name"]],lw=2.2,
            label=f"{d['name']}  (n={d['n_users']:,}, mean {d['mean']:.0f}, median {d['median']:.0f})")
    ax.axvline(d["mean"],color=COL[d["name"]],ls=":",lw=1.0,alpha=0.6)

# synthetic benchmark operating points (means)
SYN=[("§5 zipf\n50M/2048",491),("§8 1B zipf",869),("§8 1B\nlognormal",1963)]
for lab,m in SYN:
    ax.axvline(m,color="#333",ls="--",lw=1.3,alpha=0.8)
    ax.text(m,0.075,lab,rotation=90,va="top",ha="right",fontsize=7.5,color="#333")

ax.set_xscale("log"); ax.set_xlim(1,2e5)
ax.set_xlabel("per-user sequence length (# interactions, log scale)",fontsize=11)
ax.set_ylabel("fraction of users (per log-bin)",fontsize=11)
ax.set_ylim(0,0.155)
ax.set_title("Real recsys per-user sequence lengths are heavy-tailed, spanning ~1 to 10^5\n"
             "(dotted = each dataset's mean · dashed = the benchmark's synthetic operating points — all inside the real range)",
             fontsize=10.5)
ax.legend(loc="upper right",fontsize=8.5,frameon=False)
ax.grid(alpha=0.3)
fig.tight_layout()
base=os.path.join(FIG,"perf_realdata_seqlen"); os.makedirs(FIG,exist_ok=True)
fig.savefig(base+".png",dpi=200,bbox_inches="tight"); fig.savefig(base+".pdf",bbox_inches="tight")
print("wrote",base+".png/.pdf")
