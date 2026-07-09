#!/usr/bin/env python3
"""Exposed GPU-time breakdown on the FASTEST single step (matches upstream PERF_ANALYSIS §2.2: "rank0, step 162, the
fastest selected GPU window"). Finds the shortest 'step N' NVTX range from the sqlite, then sweep-lines rank0's kernels
clipped to that window -> per-category exposed %. Usage: exposed_faststep.py <sqlite> <cuda_gpu_trace.csv> <label>"""
import sqlite3, csv, sys, re
from collections import Counter, defaultdict

sq, csvp, label = sys.argv[1], sys.argv[2], sys.argv[3]
con=sqlite3.connect(sq); cur=con.cursor()
try:
    rows=cur.execute("SELECT n.start,n.end FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id "
                     "WHERE COALESCE(s.value,n.text) LIKE 'step %' AND n.end>n.start").fetchall()
except Exception:
    rows=cur.execute("SELECT start,end FROM NVTX_EVENTS WHERE text LIKE 'step %' AND end>start").fetchall()
S,E=min(rows,key=lambda r:r[1]-r[0])   # fastest (shortest) step window
print("fastest-step window [%d,%d] dur=%.2f ms (of %d step ranges)"%(S,E,(E-S)/1e6,len(rows)))

CATS=[("attention",r"flash::compute_attn|HSTUAttention|Hstu_fwd|Hstu_bwd|hstu.*attn"),("nccl",r"nccl"),
      ("gemm",r"nvjet|cublas|ampere_|sm\d+_gemm|cutlass_\w*gemm"),
      ("elem",r"layer_norm|_ln_mul|silu|dropout|elementwise|reduce_kernel|FillFunctor|CatArray|clamp|Functor"),
      ("embedding",r"embedding|dyn_emb|dynamicemb|fbgemm|bucketize|permute|radix_sort|DeviceScan|DeviceCompact|DeviceSelect|RunLength|linearize_index"),
      ("other",r".*")]
def cat(n):
    for c,rx in CATS:
        if re.search(rx,n): return c
    return "other"
rows=[]
with open(csvp) as f:
    r=csv.reader(f); h=next(r)
    def col(s):
        for i,x in enumerate(h):
            if s.lower() in x.strip().lower(): return i
    si,di,dv,nm=col("Start"),col("Duration"),col("Device"),col("Name")
    for row in r:
        if not row or len(row)<=max(si,di,nm): continue
        try: st=float(row[si].replace(",","")); du=float(row[di].replace(",",""))
        except: continue
        rows.append((st,du,row[dv] if dv is not None else "0",row[nm]))
dev0=Counter(d for _,_,d,_ in rows).most_common(1)[0][0]
ev=[]
for st,du,d,n in rows:
    if d!=dev0 or du<=0: continue
    a=max(st,S); b=min(st+du,E)
    if b<=a: continue
    ev.append((a,1,cat(n))); ev.append((b,-1,cat(n)))
ev.sort(key=lambda e:(e[0],e[1]))
active=defaultdict(int); buck=defaultdict(float); tp=None
for t,d,c in ev:
    if tp is not None and t>tp:
        seg=t-tp; n=sum(active.values())
        if n==0: buck["idle"]+=seg
        elif n==1: buck["exp_"+[k for k,v in active.items() if v>0][0]]+=seg
        else: buck["overlap"]+=seg
    active[c]+=d; tp=t
tot=E-S
print("=== %s FASTEST-STEP exposed breakdown (%% of step) device=%s ==="%(label,dev0))
for k in sorted(buck,key=lambda x:-buck[x]): print("  %-16s %6.2f%%"%(k,buck[k]/tot*100))
print("SUNBURST_FAST %s %s"%(label,",".join("%s=%.3f"%(k,buck[k]/tot*100) for k in buck)))
