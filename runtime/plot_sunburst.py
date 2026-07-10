#!/usr/bin/env python3
"""§4c 2-ring sunburst matching upstream PERF_ANALYSIS §2.2 exactly (exposed accounting, FASTEST step).
Inner ring = 8 groupings (HSTU/GEMM/ELEM/EMB/IDLE/NCCL/OTH/OVL); outer ring = leaves:
GEMM -> UVQK/PROJ/G-O, NCCL -> N-E(exposed)/N-O(overlap), the rest repeat as labeled leaves.  % of the fastest step.
Upstream from its published §2.2 table; ours via runtime/nsys_repro/exposed_faststep.py.
"""
import os, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=os.path.dirname(os.path.abspath(__file__)); FIG=os.path.join(HERE,"..","figures")

# inner value per grouping + outer leaf splits (gemm_sub, nccl_sub). % of fastest step.
# Jagged runs only (H100 + GB300); upstream panel intentionally omitted — see §2.2 (A) for the upstream-matched comparison.
DATA={
  "Our H100 · step 153 · D256 (jagged)": {   # figs-h100-jag-timeline sqlite; EXACT gemm split; comms-dominated on the short step (NCCL 43%)
    "attention":17.14,"gemm":10.25,"elem":11.70,"embedding":0.98,"idle":14.60,"nccl":43.58,"other":1.61,"overlap":0.06,
    "gemm_sub":{"UVQK":7.53,"PROJ":2.24,"G-O":0.48},"nccl_sub":{"N-E":43.32,"N-O":0.26}},
  "Our GB300 · step 155 · D128 (jagged)": {  # exp4_caching_hr jagged capture (branch mislabeled 'exp2'); gemm EXACT per-kernel innermost-NVTX from sqlite (short seq -> MLP dominates the tiny exposed gemm)
    "attention":6.44,"gemm":3.80,"elem":12.71,"embedding":1.32,"idle":55.54,"nccl":19.50,"other":0.58,"overlap":0.11,
    "gemm_sub":{"UVQK":1.22,"PROJ":0.48,"G-O":2.10},"nccl_sub":{"N-E":19.06,"N-O":0.44}},
}

ORDER=["attention","gemm","elem","embedding","idle","nccl","other","overlap"]
LBL ={"attention":"HSTU","gemm":"GEMM","elem":"ELEM","embedding":"EMB","idle":"IDLE","nccl":"NCCL","other":"OTH","overlap":"OVL"}
LEAF={"attention":"HSTU","elem":"ELEM","embedding":"EMB","idle":"IDLE","other":"OTH","overlap":"OVL"}  # single-leaf groupings
COL ={"attention":"#e6194B","gemm":"#3cb44b","elem":"#f58231","embedding":"#911eb4",
      "idle":"#9A9A9A","nccl":"#4363d8","other":"#bcbd22","overlap":"#42d4f4"}
def lighten(hexc,f=0.5):
    r=int(hexc[1:3],16);g=int(hexc[3:5],16);b=int(hexc[5:7],16)
    return "#%02x%02x%02x"%(int(r+(255-r)*f),int(g+(255-g)*f),int(b+(255-b)*f))
GSH={"UVQK":0.28,"PROJ":0.5,"G-O":0.72}; NSH={"N-E":0.3,"N-O":0.6}

def sunburst(ax,d,title):
    inner_v=[d.get(c,0) for c in ORDER]; inner_c=[COL[c] for c in ORDER]
    ov,oc,olbl=[],[],[]
    for c in ORDER:
        v=d.get(c,0)
        if c=="gemm":
            for sub,sv in d.get("gemm_sub",{}).items(): ov.append(sv);oc.append(lighten(COL["gemm"],GSH.get(sub,.5)));olbl.append(sub)
        elif c=="nccl":
            for sub,sv in d.get("nccl_sub",{}).items(): ov.append(sv);oc.append(lighten(COL["nccl"],NSH.get(sub,.5)));olbl.append(sub)
        else:
            ov.append(v);oc.append(COL[c]);olbl.append("")   # leaf: same colour (a continuation), labelled ONCE in the inner ring
    w,_=ax.pie(inner_v,radius=0.70,colors=inner_c,startangle=90,counterclock=False,
               wedgeprops=dict(width=0.40,edgecolor="white",linewidth=1.0))
    for wc,c,v in zip(w,ORDER,inner_v):
        if v<1.6: continue   # label every grouping once (GEMM/NCCL show their subtotal; tiny OTH/OVL -> legend)
        a=math.radians((wc.theta1+wc.theta2)/2); ax.text(math.cos(a)*0.50,math.sin(a)*0.50,f"{LBL[c]}\n{v:.0f}",
            ha="center",va="center",fontsize=7,color="white",fontweight="bold",linespacing=0.9)
    w2,_=ax.pie(ov,radius=1.0,colors=oc,startangle=90,counterclock=False,
                wedgeprops=dict(width=0.28,edgecolor="white",linewidth=0.8))
    for wc,lb,v in zip(w2,olbl,ov):
        if not lb or v<1.0: continue
        a=math.radians((wc.theta1+wc.theta2)/2)
        txt=f"{lb} {v:.0f}" if v>=2.5 else lb
        ax.text(math.cos(a)*0.86,math.sin(a)*0.86,txt,ha="center",va="center",fontsize=6.3,color="#222",fontweight="bold")
    ax.set_title(title,fontsize=11)

fig,axes=plt.subplots(1,2,figsize=(11,5.8))
for ax,(t,d) in zip(axes,DATA.items()): sunburst(ax,d,t)
fig.suptitle("HSTU GPU-time sunburst — jagged runs, exposed accounting, fastest step (upstream PERF_ANALYSIS §2.2 method)\ninner ring = grouping · outer ring = leaves · % of the step",fontsize=12)
# legend keyed like upstream's: every leaf = abbreviation + full name (inner grouping colour, outer shade for children)
KEY=[("HSTU",COL["attention"],"hstu fwd/bwd"),
     ("UVQK",lighten(COL["gemm"],GSH["UVQK"]),"gemm / uvqk"),
     ("PROJ",lighten(COL["gemm"],GSH["PROJ"]),"gemm / projection"),
     ("G-O",lighten(COL["gemm"],GSH["G-O"]),"gemm / others"),
     ("ELEM",COL["elem"],"elementwise"),
     ("EMB",COL["embedding"],"embedding op"),
     ("N-E",lighten(COL["nccl"],NSH["N-E"]),"nccl (exposed)"),
     ("N-O",lighten(COL["nccl"],NSH["N-O"]),"nccl (overlap)"),
     ("IDLE",COL["idle"],"GPU idle"),
     ("OTH",COL["other"],"others"),
     ("OVL",COL["overlap"],"overlapped")]
handles=[plt.Rectangle((0,0),1,1,color=c) for _,c,_ in KEY]
fig.legend(handles,[f"{ab} = {full}" for ab,_,full in KEY],ncol=6,fontsize=8,loc="lower center",
           frameon=False,bbox_to_anchor=(0.5,-0.02),columnspacing=1.4,handlelength=1.1)
fig.tight_layout(rect=[0,0.08,1,0.94])
base=os.path.join(FIG,"perf_sunburst_exposed"); os.makedirs(FIG,exist_ok=True)
fig.savefig(base+".png",dpi=200,bbox_inches="tight"); fig.savefig(base+".pdf",bbox_inches="tight")
print("wrote",base+".png/.pdf")
