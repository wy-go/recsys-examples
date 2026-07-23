#!/usr/bin/env python3
"""§8b kv256·heads4 vs kv128·heads8 GPU-time sunburst — SAME rings/colours/legend as the §8a A/B sunburst
(plot_sunburst_v2606_ab.py, itself matching the §8 / §5.2 sunbursts), two panels on the same-shape pair
(N·D=1024; 8-GPU, 1B/4096/lognormal, bs32, jagged, 80-step time-weighted aggregate; exposed accounting via
exposed_perstep.py + exposed_perstep_subsplit.py on the kv256sun/kv128sun captures).

KEY: the two disks are drawn AREA-PROPORTIONAL to total step time (kv128h8 radius = sqrt(9198.1/10898.0) x kv256h4),
so a wedge's AREA is proportional to ABSOLUTE ms, not % of step. The gap between the disks is the red HSTU wedge:
non-attention compute areas are ~equal, while the faster d128·8 arm exposes MORE NCCL and idle (%).

gemm split UVQK/PROJ/G-O; nccl split dense/sparse x exposed/overlap (N-Ed/N-Es/N-Od/N-Os, dense=grad all-reduce,
sparse=embedding all-to-all); idle split into CPU causes (I-launch/I-host/I-sync/I-copy/I-oth). Same leaves + shading.
"""
import os, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); FIG=os.path.join(HERE,"..","figures","v2606")

# % of step, 80-step time-weighted aggregate (exposed_perstep.py / exposed_perstep_subsplit.py on the
# figs-gb300-kv256sun / -kv128sun sqlite pair). sigma_step = total ms over 80 steps -> sets disk area.
DATA={
  "CUTLASS d256 · heads=4 (PR #18)\nmedian 136.1 ms/step · MFU 19.2%": {
    "sigma":10898.0,
    "attention":42.13,"gemm":12.80,"elem":25.93,"embedding":1.66,"idle":11.52,"nccl":4.95,"other":0.64,"overlap":0.32,
    "gemm_sub":{"UVQK":9.52,"PROJ":2.71,"G-O":0.57},"nccl_sub":{"N-Ed":2.09,"N-Es":1.75,"N-Od":0.61,"N-Os":0.50},
    "idle_sub":{"I-launch":3.94,"I-host":5.20,"I-sync":1.31,"I-copy":0.80,"I-oth":0.30}},
  "d128 · heads=8 (same shape N·D=1024)\nmedian 114.5 ms/step · MFU 25.7%": {
    "sigma":9198.1,
    "attention":26.43,"gemm":14.46,"elem":25.72,"embedding":1.93,"idle":16.18,"nccl":14.24,"other":0.75,"overlap":0.23,
    "gemm_sub":{"UVQK":10.71,"PROJ":3.07,"G-O":0.68},"nccl_sub":{"N-Ed":3.71,"N-Es":5.66,"N-Od":4.07,"N-Os":0.80},
    "idle_sub":{"I-launch":6.45,"I-host":6.23,"I-sync":1.72,"I-copy":1.41,"I-oth":0.40}},
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
fig.suptitle("§8b  head_dim 256·4 vs 128·8, same shape (N·D=1024) — GPU-time sunburst, 1B/4096/lognormal bs32 (GB300 8-GPU, 80-step aggregate)\n"
             "inner ring = grouping · outer ring = leaves · % of step; DISK AREA ∝ step time (d128·8 = 0.84× area) → equal area = equal absolute ms\n"
             "step −15.9% ·  the gap is the red HSTU wedge (57 → 30 ms)  ·  both configs move the same bytes — the faster d128·8 config just hides less of its NCCL under compute",fontsize=11)
KEY=[("HSTU",COL["attention"],"hstu fwd/bwd"),("UVQK",lighten(COL["gemm"],GSH["UVQK"]),"gemm / uvqk"),
     ("PROJ",lighten(COL["gemm"],GSH["PROJ"]),"gemm / projection"),("G-O",lighten(COL["gemm"],GSH["G-O"]),"gemm / others"),
     ("ELEM",COL["elem"],"elementwise"),("EMB",COL["embedding"],"embedding op"),
     ("N-Ed",lighten(COL["nccl"],NSH["N-Ed"]),"nccl DENSE/all-reduce (exposed)"),("N-Es",lighten(COL["nccl"],NSH["N-Es"]),"nccl SPARSE/all-to-all (exposed)"),
     ("N-Od",lighten(COL["nccl"],NSH["N-Od"]),"nccl dense (overlap)"),("N-Os",lighten(COL["nccl"],NSH["N-Os"]),"nccl sparse (overlap)"),
     ("I-launch",lighten(COL["idle"],ISH["I-launch"]),"idle: kernel-launch (CPU dispatch)"),("I-host",lighten(COL["idle"],ISH["I-host"]),"idle: host/python gap"),
     ("I-sync",lighten(COL["idle"],ISH["I-sync"]),"idle: sync wait"),("I-copy",lighten(COL["idle"],ISH["I-copy"]),"idle: H↔D copy"),
     ("OTH",COL["other"],"others"),("OVL",COL["overlap"],"overlapped")]
handles=[plt.Rectangle((0,0),1,1,color=c) for _,c,_ in KEY]
fig.legend(handles,[f"{ab} = {full}" for ab,_,full in KEY],ncol=6,fontsize=8,loc="lower center",frameon=False,bbox_to_anchor=(0.5,-0.02),columnspacing=1.4,handlelength=1.1)
fig.tight_layout(rect=[0,0.08,1,0.90])
base=os.path.join(FIG,"kv256_sunburst"); os.makedirs(FIG,exist_ok=True)
fig.savefig(base+".png",dpi=200,bbox_inches="tight"); fig.savefig(base+".pdf",bbox_inches="tight")
print("wrote",base+".png/.pdf")
