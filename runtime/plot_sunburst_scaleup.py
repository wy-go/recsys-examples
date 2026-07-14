#!/usr/bin/env python3
"""§8 scale-study GPU-time sunburst — exposed accounting on rank0's fastest step, SAME rings/colours/legend as the
§5.2 sunbursts (plot_sunburst.py / plot_sunburst_nj.py). One panel per scale-study config: the §5.2(B) exp4 reference,
the prefetch ablation (exp5), and the 1B sweep over zipf / lognormal sequence lengths. GB300, 16 GPU, jagged.
gemm split into UVQK/PROJ/G-O (exposed_gemm_split.py); nccl split into dense/sparse × exposed/overlap
(N-Ed/N-Es/N-Od/N-Os, exposed_perstep_subsplit.py — dense = gradient all-reduce, sparse = embedding all-to-all); idle
split into CPU causes (I-launch/I-host/I-sync/I-copy/I-oth) — same leaves + shading as the §5.2 sunbursts. 80-step aggregate.
"""
import os, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); FIG=os.path.join(HERE,"..","figures")

# inner value per grouping + outer leaf splits. % of step, TIME-WEIGHTED AGGREGATE over all 80 steady steps
# (exposed_perstep.py on the archived figs-gb300-scaleup-* sqlite; exp4 = §5.2(B) fastest-step ref). Each column ~100.
DATA={
  "exp4 (ref, fastest) · 50M/2048": {
    "attention":6.4,"gemm":3.8,"elem":12.7,"embedding":1.3,"idle":55.5,"nccl":19.5,"other":0.6,"overlap":0.1,
    "gemm_sub":{"UVQK":1.2,"PROJ":0.5,"G-O":2.1},"nccl_sub":{"N-Ed":9.91,"N-Es":9.15,"N-Od":0.0,"N-Os":0.44},
    "idle_sub":{"I-launch":21.01,"I-host":28.07,"I-sync":3.04,"I-copy":2.33,"I-oth":1.09}},
  "exp5 · 50M/2048 (80-step agg)": {
    "attention":6.3,"gemm":3.7,"elem":12.6,"embedding":1.4,"idle":63.2,"nccl":11.9,"other":0.7,"overlap":0.0,
    "gemm_sub":{"UVQK":1.9,"PROJ":0.7,"G-O":1.1},"nccl_sub":{"N-Ed":8.63,"N-Es":3.03,"N-Od":0.01,"N-Os":0.21},
    "idle_sub":{"I-launch":23.76,"I-host":30.78,"I-sync":3.38,"I-copy":3.57,"I-oth":1.79}},
  "zipf · 1B/4096 (80-step agg)": {
    "attention":15.7,"gemm":6.2,"elem":19.5,"embedding":2.2,"idle":45.4,"nccl":9.9,"other":1.0,"overlap":0.1,
    "gemm_sub":{"UVQK":3.7,"PROJ":1.3,"G-O":1.2},"nccl_sub":{"N-Ed":7.58,"N-Es":2.04,"N-Od":0.01,"N-Os":0.27},
    "idle_sub":{"I-launch":15.88,"I-host":21.83,"I-sync":3.01,"I-copy":3.38,"I-oth":1.40}},
  "lognormal · 1B/4096 (80-step agg)": {
    "attention":19.8,"gemm":11.7,"elem":32.6,"embedding":2.7,"idle":23.8,"nccl":7.9,"other":1.2,"overlap":0.2,
    "gemm_sub":{"UVQK":5.3,"PROJ":1.6,"G-O":4.8},"nccl_sub":{"N-Ed":5.10,"N-Es":2.14,"N-Od":0.13,"N-Os":0.51},
    "idle_sub":{"I-launch":8.98,"I-host":9.57,"I-sync":2.39,"I-copy":2.00,"I-oth":0.88}},
}
ORDER=["attention","gemm","elem","embedding","idle","nccl","other","overlap"]
LBL ={"attention":"HSTU","gemm":"GEMM","elem":"ELEM","embedding":"EMB","idle":"IDLE","nccl":"NCCL","other":"OTH","overlap":"OVL"}
COL ={"attention":"#e6194B","gemm":"#3cb44b","elem":"#f58231","embedding":"#911eb4",
      "idle":"#9A9A9A","nccl":"#4363d8","other":"#bcbd22","overlap":"#42d4f4"}
def lighten(h,f=0.5):
    r,g,b=int(h[1:3],16),int(h[3:5],16),int(h[5:7],16); return "#%02x%02x%02x"%(int(r+(255-r)*f),int(g+(255-g)*f),int(b+(255-b)*f))
GSH={"UVQK":0.28,"PROJ":0.5,"G-O":0.72}; NSH={"N-Ed":0.20,"N-Es":0.45,"N-Od":0.60,"N-Os":0.78}
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
    ax.set_title(title,fontsize=10.5)

fig,axes=plt.subplots(1,len(DATA),figsize=(5.5*len(DATA),5.8))
for ax,(t,d) in zip(axes,DATA.items()): sunburst(ax,d,t)
fig.suptitle("§8 scale-study GPU-time sunburst — GB300 16-GPU, exposed accounting, time-weighted aggregate over 80 steps\n"
             "inner ring = grouping · outer ring = leaves · % of the step (exp4 = §5.2(B) fastest-step ref)",fontsize=12)
KEY=[("HSTU",COL["attention"],"hstu fwd/bwd"),("UVQK",lighten(COL["gemm"],GSH["UVQK"]),"gemm / uvqk"),
     ("PROJ",lighten(COL["gemm"],GSH["PROJ"]),"gemm / projection"),("G-O",lighten(COL["gemm"],GSH["G-O"]),"gemm / others"),
     ("ELEM",COL["elem"],"elementwise"),("EMB",COL["embedding"],"embedding op"),
     ("N-Ed",lighten(COL["nccl"],NSH["N-Ed"]),"nccl DENSE/all-reduce (exposed)"),("N-Es",lighten(COL["nccl"],NSH["N-Es"]),"nccl SPARSE/all-to-all (exposed)"),
     ("N-Os",lighten(COL["nccl"],NSH["N-Os"]),"nccl sparse (overlap)"),
     ("I-launch",lighten(COL["idle"],ISH["I-launch"]),"idle: kernel-launch (CPU dispatch)"),("I-host",lighten(COL["idle"],ISH["I-host"]),"idle: host/python gap"),
     ("I-sync",lighten(COL["idle"],ISH["I-sync"]),"idle: sync wait"),("I-copy",lighten(COL["idle"],ISH["I-copy"]),"idle: H↔D copy"),
     ("OTH",COL["other"],"others"),("OVL",COL["overlap"],"overlapped")]
handles=[plt.Rectangle((0,0),1,1,color=c) for _,c,_ in KEY]
fig.legend(handles,[f"{ab} = {full}" for ab,_,full in KEY],ncol=6,fontsize=8,loc="lower center",frameon=False,bbox_to_anchor=(0.5,-0.02),columnspacing=1.4,handlelength=1.1)
fig.tight_layout(rect=[0,0.08,1,0.94])
base=os.path.join(FIG,"perf_sunburst_scaleup"); os.makedirs(FIG,exist_ok=True)
fig.savefig(base+".png",dpi=200,bbox_inches="tight"); fig.savefig(base+".pdf",bbox_inches="tight")
print("wrote",base+".png/.pdf")
