#!/usr/bin/env python3
"""§5 §2.2(A) NON-JAGGED sunburst — exposed accounting on rank0's fastest step (upstream PERF_ANALYSIS §2.2 method).
Upstream H100 (its published §2.2 table, fine gemm/nccl split) vs our matched-config runs (exposed_faststep.py; coarse
gemm/nccl — no sub-split, shown as single leaves). Our-H100 panel added when figs-h100-exposed lands.
"""
import os, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); FIG=os.path.join(HERE,"..","figures")

DATA={
  # nccl = N-E(exposed) + N-O(overlap); overlap = generic multi-active minus N-O (same convention as upstream / §2.2(B)).
  "Upstream H100 · step 162 · D256 (nj)": {
    "attention":43.09,"gemm":23.69,"elem":21.31,"embedding":5.74,"idle":3.34,"nccl":2.04,"other":0.66,"overlap":0.13,
    "gemm_sub":{"UVQK":17.85,"PROJ":4.80,"G-O":1.04},"nccl_sub":{"N-E":1.58,"N-O":0.46}},
  # EXACT split from figs-h100-nj-timeline sqlite (genuinely-fastest 235ms step; exposed_gemm_split.py).
  # A 2nd capture's fastest step (323ms) had nccl 43% — IB collective exposure is highly variable step-to-step.
  "Our H100 · step 159 · 235.6 ms · D256 (nj)": {
    "attention":33.73,"gemm":18.75,"elem":18.10,"embedding":0.88,"idle":3.80,"nccl":22.29,"other":2.37,"overlap":0.08,
    "gemm_sub":{"UVQK":14.75,"PROJ":2.58,"G-O":1.43},"nccl_sub":{"N-Ed":15.75,"N-Es":6.13,"N-Od":0.04,"N-Os":0.37},   # exposed_nccl_split.py: exposed 72% dense; overlap ~all sparse
    "idle_sub":{"I-launch":1.69,"I-host":1.40,"I-sync":0.22,"I-copy":0.32,"I-oth":0.17}},   # exposed_idle_split.py: idle = CPU launch+host bubble
  "Our GB300 · step 153 · 78.0 ms · D128 (nj)": {   # run figs-gb300-nj2048-exp2; gemm EXACT per-kernel innermost-NVTX from sqlite (UVQK 70.5/PROJ 21.6/G-O[mlp] 8.0); nccl split from exposed_nccl_split.py
    "attention":18.52,"gemm":13.01,"elem":36.11,"embedding":2.71,"idle":20.47,"nccl":7.28,"other":1.24,"overlap":0.54,
    "gemm_sub":{"UVQK":9.17,"PROJ":2.81,"G-O":1.04},"nccl_sub":{"N-Ed":6.51,"N-Es":0.57,"N-Od":0.00,"N-Os":0.20},   # exposed 92% dense; overlap all sparse
    "idle_sub":{"I-launch":7.64,"I-host":8.12,"I-sync":2.43,"I-copy":1.66,"I-oth":0.62}},   # idle = CPU launch+host bubble (fast D128 kernels starve the host pipeline)
}
ORDER=["attention","gemm","elem","embedding","idle","nccl","other","overlap"]
LBL ={"attention":"HSTU","gemm":"GEMM","elem":"ELEM","embedding":"EMB","idle":"IDLE","nccl":"NCCL","other":"OTH","overlap":"OVL"}
COL ={"attention":"#e6194B","gemm":"#3cb44b","elem":"#f58231","embedding":"#911eb4",
      "idle":"#9A9A9A","nccl":"#4363d8","other":"#bcbd22","overlap":"#42d4f4"}
def lighten(h,f=0.5):
    r,g,b=int(h[1:3],16),int(h[3:5],16),int(h[5:7],16); return "#%02x%02x%02x"%(int(r+(255-r)*f),int(g+(255-g)*f),int(b+(255-b)*f))
GSH={"UVQK":0.28,"PROJ":0.5,"G-O":0.72}; NSH={"N-E":0.3,"N-Ed":0.20,"N-Es":0.45,"N-Od":0.60,"N-Os":0.78}
ISH={"I-launch":0.20,"I-host":0.42,"I-sync":0.58,"I-copy":0.72,"I-oth":0.85}
SUBSH={"gemm":GSH,"nccl":NSH,"idle":ISH}

def sunburst(ax,d,title):
    inner_v=[d.get(c,0) for c in ORDER]; inner_c=[COL[c] for c in ORDER]
    ov,oc,olbl=[],[],[]
    for c in ORDER:
        v=d.get(c,0); sub=d.get(c+"_sub") if c in ("gemm","nccl","idle") else None
        if sub:
            SH=SUBSH[c]
            for s,sv in sub.items(): ov.append(sv);oc.append(lighten(COL[c],SH.get(s,.5)));olbl.append(s)
        else:
            ov.append(v);oc.append(COL[c]);olbl.append("")   # single leaf (labelled once in inner ring)
    ws=ax.pie(inner_v,radius=0.70,colors=inner_c,startangle=90,counterclock=False,
              wedgeprops=dict(width=0.40,edgecolor="white",linewidth=1.0))[0]
    for wc,c,v in zip(ws,ORDER,inner_v):
        if v<1.6: continue
        a=math.radians((wc.theta1+wc.theta2)/2); ax.text(math.cos(a)*0.50,math.sin(a)*0.50,f"{LBL[c]}\n{v:.0f}",
            ha="center",va="center",fontsize=7,color="white",fontweight="bold",linespacing=0.9)
    w2,_=ax.pie(ov,radius=1.0,colors=oc,startangle=90,counterclock=False,
                wedgeprops=dict(width=0.28,edgecolor="white",linewidth=0.8))
    for wc,lb,v in zip(w2,olbl,ov):
        if not lb or v<1.0: continue
        a=math.radians((wc.theta1+wc.theta2)/2); ax.text(math.cos(a)*0.86,math.sin(a)*0.86,f"{lb} {v:.0f}" if v>=2.5 else lb,
            ha="center",va="center",fontsize=6.3,color="#222",fontweight="bold")
    ax.set_title(title,fontsize=11)

fig,axes=plt.subplots(1,len(DATA),figsize=(5.5*len(DATA),5.8))
if len(DATA)==1: axes=[axes]
for ax,(t,d) in zip(axes,DATA.items()): sunburst(ax,d,t)
fig.suptitle("HSTU GPU-time sunburst — NON-JAGGED (matched config), exposed accounting, fastest step (upstream §2.2 method)\ninner ring = grouping · outer ring = leaves · % of the step",fontsize=12)
KEY=[("HSTU",COL["attention"],"hstu fwd/bwd"),("UVQK",lighten(COL["gemm"],GSH["UVQK"]),"gemm / uvqk"),
     ("PROJ",lighten(COL["gemm"],GSH["PROJ"]),"gemm / projection"),("G-O",lighten(COL["gemm"],GSH["G-O"]),"gemm / others"),
     ("GEMM",COL["gemm"],"gemm (ours: not sub-split)"),("ELEM",COL["elem"],"elementwise"),("EMB",COL["embedding"],"embedding op"),
     ("N-Ed",lighten(COL["nccl"],NSH["N-Ed"]),"nccl DENSE/all-reduce (exposed)"),("N-Es",lighten(COL["nccl"],NSH["N-Es"]),"nccl SPARSE/all-to-all (exposed)"),
     ("N-Os",lighten(COL["nccl"],NSH["N-Os"]),"nccl sparse (overlap)"),("N-E",lighten(COL["nccl"],NSH["N-E"]),"nccl exposed (upstream, unsplit)"),
     ("I-launch",lighten(COL["idle"],ISH["I-launch"]),"idle: kernel-launch (CPU dispatch)"),("I-host",lighten(COL["idle"],ISH["I-host"]),"idle: host/python gap"),
     ("I-sync",lighten(COL["idle"],ISH["I-sync"]),"idle: sync wait"),("I-copy",lighten(COL["idle"],ISH["I-copy"]),"idle: H↔D copy"),
     ("OTH",COL["other"],"others"),("OVL",COL["overlap"],"overlapped")]
handles=[plt.Rectangle((0,0),1,1,color=c) for _,c,_ in KEY]
fig.legend(handles,[f"{ab} = {full}" for ab,_,full in KEY],ncol=6,fontsize=8,loc="lower center",frameon=False,bbox_to_anchor=(0.5,-0.02),columnspacing=1.4,handlelength=1.1)
fig.tight_layout(rect=[0,0.08,1,0.94])
base=os.path.join(FIG,"perf_sunburst_exposed_nj"); os.makedirs(FIG,exist_ok=True)
fig.savefig(base+".png",dpi=200,bbox_inches="tight"); fig.savefig(base+".pdf",bbox_inches="tight")
print("wrote",base+".png/.pdf")
