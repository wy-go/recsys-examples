#!/usr/bin/env python3
"""§8a v26.05 -> v26.06 image A/B GPU-time sunburst — SAME rings/colours/legend as the §8 / §5.2 sunbursts
(plot_sunburst_scaleup.py), two panels (v26.05 | v26.06) on the MATCHED 1B/4096/lognormal config (8-GPU, jagged,
80-step time-weighted aggregate; exposed accounting via exposed_perstep.py + exposed_perstep_subsplit.py).

KEY: the two disks are drawn AREA-PROPORTIONAL to total step time (v26.06 radius = sqrt(6192.6/7637.2) x v26.05), so a
wedge's AREA is proportional to ABSOLUTE ms, not % of step. That way "idle flat / step shrank" is visible: GPU idle keeps
~the same area across the two disks (its % grows only because the disk shrank), while exposed NCCL visibly collapses.

gemm split UVQK/PROJ/G-O; nccl split dense/sparse x exposed/overlap (N-Ed/N-Es/N-Od/N-Os, dense=grad all-reduce,
sparse=embedding all-to-all); idle split into CPU causes (I-launch/I-host/I-sync/I-copy/I-oth). Same leaves + shading.
"""
import os, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); FIG=os.path.join(HERE,"..","figures")

# % of step, 80-step time-weighted aggregate (exposed_perstep.py / exposed_perstep_subsplit.py on the matched
# figs-gb300-scaleup-r1b-ratio01-n8-bs32-logn sqlite pair). sigma_step = total ms over 80 steps -> sets disk area.
DATA={
  "v26.05  (median 94.0 ms/step · MFU ~16%)": {
    "sigma":7637.2,
    "attention":17.29,"gemm":9.95,"elem":28.87,"embedding":2.32,"idle":23.13,"nccl":17.22,"other":1.03,"overlap":0.13,
    "gemm_sub":{"UVQK":5.93,"PROJ":1.94,"G-O":2.08},"nccl_sub":{"N-Ed":10.55,"N-Es":5.70,"N-Od":0.02,"N-Os":0.96},
    "idle_sub":{"I-launch":9.08,"I-host":9.22,"I-sync":2.07,"I-copy":2.04,"I-oth":0.75}},
  "v26.06  (median 77.5 ms/step · MFU ~19%)": {
    "sigma":6192.6,
    "attention":21.35,"gemm":12.40,"elem":24.68,"embedding":2.90,"idle":28.45,"nccl":8.60,"other":1.34,"overlap":0.22,
    "gemm_sub":{"UVQK":5.41,"PROJ":1.83,"G-O":5.16},"nccl_sub":{"N-Ed":4.87,"N-Es":2.91,"N-Od":0.02,"N-Os":0.80},
    "idle_sub":{"I-launch":11.03,"I-host":11.18,"I-sync":2.68,"I-copy":2.61,"I-oth":0.97}},
}
SIG_REF=max(d["sigma"] for d in DATA.values())   # largest disk = radius 1.0

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
    R=math.sqrt(d["sigma"]/SIG_REF)          # area-proportional radius scale
    r_in, r_out, w_in, w_out = 0.70*R, 1.0*R, 0.40*R, 0.28*R
    inner_v=[d.get(c,0) for c in ORDER]; inner_c=[COL[c] for c in ORDER]
    ov,oc,olbl=[],[],[]
    for c in ORDER:
        v=d.get(c,0); sub=d.get(c+"_sub") if c in ("gemm","nccl","idle") else None
        if sub:
            SH=SUBSH[c]
            for s,sv in sub.items(): ov.append(sv);oc.append(lighten(COL[c],SH.get(s,.5)));olbl.append(s)
        else:
            ov.append(v);oc.append(COL[c]);olbl.append("")
    ws=ax.pie(inner_v,radius=r_in,colors=inner_c,startangle=90,counterclock=False,
              wedgeprops=dict(width=w_in,edgecolor="white",linewidth=1.0))[0]
    for wc,c,v in zip(ws,ORDER,inner_v):
        if v<1.6: continue
        a=math.radians((wc.theta1+wc.theta2)/2); rr=r_in-w_in/2
        ax.text(math.cos(a)*rr,math.sin(a)*rr,f"{LBL[c]}\n{v:.0f}",
            ha="center",va="center",fontsize=7,color="white",fontweight="bold",linespacing=0.9)
    w2,_=ax.pie(ov,radius=r_out,colors=oc,startangle=90,counterclock=False,
                wedgeprops=dict(width=w_out,edgecolor="white",linewidth=0.8))
    for wc,lb,v in zip(w2,olbl,ov):
        if not lb or v<1.0: continue
        a=math.radians((wc.theta1+wc.theta2)/2); rr=r_out-w_out/2
        ax.text(math.cos(a)*rr,math.sin(a)*rr,f"{lb} {v:.0f}" if v>=2.5 else lb,
            ha="center",va="center",fontsize=6.3,color="#222",fontweight="bold")
    ax.set_title(title,fontsize=10.5); ax.set_xlim(-1.05,1.05); ax.set_ylim(-1.05,1.05); ax.set_aspect("equal")

fig,axes=plt.subplots(1,len(DATA),figsize=(5.7*len(DATA),6.0))
for ax,(t,d) in zip(axes,DATA.items()): sunburst(ax,d,t)
fig.suptitle("§8a  v26.05 → v26.06 image A/B — GPU-time sunburst, matched 1B/4096/lognormal (8-GPU, 80-step aggregate)\n"
             "inner ring = grouping · outer ring = leaves · % of step; DISK AREA ∝ step time (v26.06 = 0.81× area) → equal area = equal absolute ms\n"
             "step −18.9% ·  idle area ~flat (launch+host unchanged)  ·  exposed NCCL ~halves  ·  elementwise shrinks",fontsize=11)
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
fig.tight_layout(rect=[0,0.08,1,0.90])
base=os.path.join(FIG,"perf_sunburst_v2606_ab"); os.makedirs(FIG,exist_ok=True)
fig.savefig(base+".png",dpi=200,bbox_inches="tight"); fig.savefig(base+".pdf",bbox_inches="tight")
print("wrote",base+".png/.pdf")
