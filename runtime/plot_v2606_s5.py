#!/usr/bin/env python3
"""New-doc §5 (GB300 v26.06 lognormal) detailed-perf figures, as a v26.05-vs-v26.06 pairing for convenience.
SUNBURST: 1x2 — v26.06 (exp4_caching_hr, 50M/4096) next to the v26.05 lognormal 1B/4096 twin.
PER-STEP: 2x1 stacked (shared y) — v26.06 (50M/4096) above v26.05 lognormal (1B/4096).
Same rings/colours/stack as BENCHMARK_RESULTS §8 (plot_sunburst_scaleup.py / plot_scaleup_realtime.py).
v26.06 leaves: exposed_perstep.py AGG + subsplit; v26.05 logn: from plot_sunburst_scaleup.py DATA (80-step agg)."""
import os, math, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); FIG=os.path.join(HERE,"..","figures","v2606")
os.makedirs(FIG,exist_ok=True)

# ---- exposed % of step (80-step aggregate). Two configs. ----
# NOTE: these sunburst dicts (and the title strings below) are HARDCODED from the subsplit analysis and must be
# re-synced by hand if the per-step jsons are regenerated — the per-step panel reads the jsons live, the sunburst does not.
V2606={"attention":20.54,"gemm":11.82,"elem":23.99,"embedding":2.84,"idle":33.41,"nccl":5.82,"other":1.19,"overlap":0.16,
   "gemm_sub":{"UVQK":7.94,"PROJ":2.72,"G-O":1.16},"nccl_sub":{"N-Ed":3.89,"N-Es":1.33,"N-Od":0.00,"N-Os":0.60},
   "idle_sub":{"I-launch":12.52,"I-host":14.88,"I-sync":2.66,"I-copy":2.48,"I-oth":0.97}}
# v26.05 lognormal 1B/4096 (BENCHMARK_RESULTS §8 / plot_sunburst_scaleup.py)
V2605={"attention":19.8,"gemm":11.7,"elem":32.6,"embedding":2.7,"idle":23.8,"nccl":7.9,"other":1.2,"overlap":0.2,
   "gemm_sub":{"UVQK":5.3,"PROJ":1.6,"G-O":4.8},"nccl_sub":{"N-Ed":5.10,"N-Es":2.14,"N-Od":0.13,"N-Os":0.51},
   "idle_sub":{"I-launch":8.98,"I-host":9.57,"I-sync":2.39,"I-copy":2.00,"I-oth":0.88}}

ORDER=["attention","gemm","elem","embedding","idle","nccl","other","overlap"]
LBL={"attention":"HSTU","gemm":"GEMM","elem":"ELEM","embedding":"EMB","idle":"IDLE","nccl":"NCCL","other":"OTH","overlap":"OVL"}
COL={"attention":"#e6194B","gemm":"#3cb44b","elem":"#f58231","embedding":"#911eb4",
     "idle":"#9A9A9A","nccl":"#4363d8","other":"#bcbd22","overlap":"#42d4f4"}
def lighten(h,f=0.5):
    r,g,b=int(h[1:3],16),int(h[3:5],16),int(h[5:7],16); return "#%02x%02x%02x"%(int(r+(255-r)*f),int(g+(255-g)*f),int(b+(255-b)*f))
GSH={"UVQK":0.28,"PROJ":0.5,"G-O":0.72}; NSH={"N-Ed":0.20,"N-Es":0.45,"N-Od":0.60,"N-Os":0.78}
ISH={"I-launch":0.20,"I-host":0.42,"I-sync":0.58,"I-copy":0.72,"I-oth":0.85}; SUBSH={"gemm":GSH,"nccl":NSH,"idle":ISH}

def sunburst(ax,D,title):
    inner_v=[D.get(c,0) for c in ORDER]; inner_c=[COL[c] for c in ORDER]
    ov,oc,olbl=[],[],[]
    for c in ORDER:
        sub=D.get(c+"_sub") if c in ("gemm","nccl","idle") else None
        if sub:
            for s,sv in sub.items(): ov.append(sv);oc.append(lighten(COL[c],SUBSH[c].get(s,.5)));olbl.append(s)
        else: ov.append(D.get(c,0));oc.append(COL[c]);olbl.append("")
    ws=ax.pie(inner_v,radius=0.70,colors=inner_c,startangle=90,counterclock=False,
              wedgeprops=dict(width=0.40,edgecolor="white",linewidth=1.0))[0]
    for wc,c,v in zip(ws,ORDER,inner_v):
        if v<1.6: continue
        a=math.radians((wc.theta1+wc.theta2)/2); ax.text(math.cos(a)*0.50,math.sin(a)*0.50,f"{LBL[c]}\n{v:.0f}",
            ha="center",va="center",fontsize=8,color="white",fontweight="bold",linespacing=0.9)
    w2=ax.pie(ov,radius=1.0,colors=oc,startangle=90,counterclock=False,
              wedgeprops=dict(width=0.28,edgecolor="white",linewidth=0.8))[0]
    for wc,lb,v in zip(w2,olbl,ov):
        if not lb or v<1.0: continue
        a=math.radians((wc.theta1+wc.theta2)/2); ax.text(math.cos(a)*0.86,math.sin(a)*0.86,f"{lb} {v:.0f}" if v>=2.5 else lb,
            ha="center",va="center",fontsize=6.6,color="#222",fontweight="bold")
    ax.set_title(title,fontsize=9.5)

# ---------- sunburst 1x2 (v26.06 | v26.05 lognormal 1B/4096) ----------
fig,axes=plt.subplots(1,2,figsize=(13,6.8))
sunburst(axes[0],V2606,"v26.06 · exp4_caching_hr · 50M/4096\nmedian step 81 ms · idle 33% · MFU ~16.7% (peak 2500)")
sunburst(axes[1],V2605,"v26.05 · lognormal · 1B/4096\nmedian step 84 ms · idle 24% · MFU ~12.9% (peak 2500)")
KEY=[("HSTU",COL["attention"],"hstu fwd/bwd"),("UVQK",lighten(COL["gemm"],GSH["UVQK"]),"gemm/uvqk"),
     ("PROJ",lighten(COL["gemm"],GSH["PROJ"]),"gemm/projection"),("G-O",lighten(COL["gemm"],GSH["G-O"]),"gemm/others"),
     ("ELEM",COL["elem"],"elementwise"),("EMB",COL["embedding"],"embedding op"),
     ("N-Ed",lighten(COL["nccl"],NSH["N-Ed"]),"nccl dense/all-reduce (exp)"),("N-Es",lighten(COL["nccl"],NSH["N-Es"]),"nccl sparse/a2a (exp)"),
     ("I-launch",lighten(COL["idle"],ISH["I-launch"]),"idle: kernel-launch"),("I-host",lighten(COL["idle"],ISH["I-host"]),"idle: host/python gap"),
     ("I-sync",lighten(COL["idle"],ISH["I-sync"]),"idle: sync"),("I-copy",lighten(COL["idle"],ISH["I-copy"]),"idle: H↔D copy"),
     ("OTH",COL["other"],"others")]
handles=[plt.Rectangle((0,0),1,1,color=c) for _,c,_ in KEY]
fig.legend(handles,[f"{ab} = {full}" for ab,_,full in KEY],ncol=7,fontsize=7.3,loc="lower center",frameon=False,bbox_to_anchor=(0.5,-0.02),columnspacing=1.1,handlelength=1.1)
fig.suptitle("Exposed GPU-time — v26.06 (50M/4096) vs v26.05 lognormal (1B/4096) · GB300 16-GPU · 80-step aggregate",fontsize=11,y=1.0)
fig.tight_layout(rect=[0,0.07,1,0.98])
fig.savefig(os.path.join(FIG,"s5_sunburst.png"),dpi=200,bbox_inches="tight"); fig.savefig(os.path.join(FIG,"s5_sunburst.pdf"),bbox_inches="tight")
print("wrote s5_sunburst (2-panel)")

# ---------- per-step bars 2x1 (v26.06 above v26.05), shared y ----------
STACK=[("GPU idle","IDLE","#9A9A9A"),("nccl(exposed)","NCCL(exp)","#4363d8"),("hstu fwd/bwd (attention)","HSTU","#e6194B"),
       ("gemm","GEMM","#3cb44b"),("elementwise","ELEM","#f58231"),("embedding op","EMB","#911eb4"),("other","OTH","#bcbd22")]
def series(a,key):
    if key=="gemm": return np.array(a["gemm / uvqk"])+np.array(a["gemm / projection"])+np.array(a["gemm / others"])
    if key=="other": return np.array(a["nccl(overlap)"])+np.array(a["others"])+np.array(a["overlapped"])
    return np.array(a[key])
PS=[("v26.06 · exp4_caching_hr · 50M/4096", json.load(open(os.path.join(HERE,"nsys_repro","v2606_perstep","s5.json")))),
    ("v26.05 · lognormal · 1B/4096",        json.load(open(os.path.join(HERE,"nsys_repro","scaleup_perstep","logn.json"))))]
fig2,axs=plt.subplots(2,1,figsize=(12,7.4),sharey=True)
for ax2,(lab,d) in zip(axs,PS):
    ms=np.array(d["step_ms"]); n=len(ms); x=np.arange(1,n+1); a=d["arrays"]; bottom=np.zeros(n)
    for key,leg,col in STACK:
        h=series(a,key)/100.0*ms; ax2.bar(x,h,bottom=bottom,width=0.82,color=col,edgecolor="none",label=leg); bottom+=h
    ax2.axhline(d["median_ms"],ls="--",lw=1,color="#111"); ax2.text(0.7,d["median_ms"],f"median {d['median_ms']:.0f}ms",fontsize=7.5,va="bottom")
    ax2.set_ylabel("step time (ms)"); ax2.set_xlim(0.5,n+0.5); ax2.margins(x=0); ax2.set_title(lab,fontsize=9.5)
axs[1].set_xlabel("step (run order)")
axs[0].legend(ncol=7,fontsize=8,loc="upper center",bbox_to_anchor=(0.5,1.22),frameon=False,columnspacing=1.1,handlelength=1.2)
fig2.suptitle("Per-step GPU-time (absolute ms, run order) — v26.06 vs v26.05 lognormal · GB300 16-GPU · shared y-axis",fontsize=10,y=1.0)
fig2.tight_layout(rect=[0,0,1,0.98])
fig2.savefig(os.path.join(FIG,"s5_perstep.png"),dpi=200,bbox_inches="tight"); fig2.savefig(os.path.join(FIG,"s5_perstep.pdf"),bbox_inches="tight")
print("wrote s5_perstep (2-panel)")
