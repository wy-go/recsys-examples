#!/usr/bin/env python3
"""EXACT §2.2 nccl sub-split — exposed/overlap × sparse/dense.

Splits the exposed NCCL leaf two ways, on rank0's fastest step (same window + sweep as exposed_faststep.py):
  - exposed (sole-active) vs overlap (concurrent with compute)  -- as before
  - sparse (embedding all-to-all = ncclDevKernel_SendRecv) vs dense (gradient collectives =
    AllReduce / AllGather / ReduceScatter / Broadcast)  -- from the kernel name (no NVTX needed).
The all-to-all-v embedding dispatch/combine is point-to-point SendRecv; the dense-model DDP gradient sync is
AllReduce. AllGather/ReduceScatter (small) are lumped with dense (dense-side collectives); noted where non-trivial.

Usage: exposed_nccl_split.py <rep.sqlite> <cuda_gpu_trace.csv> <label>
"""
import sqlite3, csv, sys, re
from collections import Counter, defaultdict

sq, csvp, label = sys.argv[1], sys.argv[2], sys.argv[3]
con = sqlite3.connect(sq); cur = con.cursor()
try:
    rows = cur.execute("SELECT n.start,n.end FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id "
                       "WHERE COALESCE(s.value,n.text) LIKE 'step %' AND n.end>n.start").fetchall()
except Exception:
    rows = cur.execute("SELECT start,end FROM NVTX_EVENTS WHERE text LIKE 'step %' AND end>start").fetchall()
con.close()
S, E = min(rows, key=lambda r: r[1] - r[0])
W = E - S
print("fastest-step window dur=%.2f ms (of %d step ranges)" % (W / 1e6, len(rows)))

SPARSE = re.compile(r"SendRecv|AllToAll|alltoall", re.I)
DENSE = re.compile(r"AllReduce|AllGather|ReduceScatter|Broadcast", re.I)
def nccl_kind(n):
    if SPARSE.search(n): return "sparse"
    if DENSE.search(n): return "dense"
    return "dense"   # any other nccl collective -> dense side

def cat(n):
    if re.search(r"nccl", n, re.I): return "nccl:" + nccl_kind(n)
    if re.search(r"flash::compute_attn|HSTUAttention|Hstu_fwd|Hstu_bwd|hstu.*attn|blackwellhstu", n, re.I): return "attention"
    if re.search(r"nvjet|cublas|ampere_|sm\d+_gemm|cutlass_\w*gemm", n, re.I): return "gemm"
    if re.search(r"layer_norm|_ln_mul|silu|dropout|elementwise|reduce_kernel|FillFunctor|CatArray|clamp|Functor", n, re.I): return "elem"
    if re.search(r"embedding|dyn_emb|dynamicemb|fbgemm|bucketize|permute|radix|DeviceScan|DeviceSelect|linearize_index", n, re.I): return "embedding"
    return "other"

rows = []
with open(csvp) as f:
    r = csv.reader(f); h = next(r)
    def col(s):
        for i, x in enumerate(h):
            if s.lower() in x.strip().lower(): return i
    si, di, dv, nm = col("Start"), col("Duration"), col("Device"), col("Name")
    for row in r:
        if not row or len(row) <= max(si, di, nm): continue
        try: st = float(row[si].replace(",", "")); du = float(row[di].replace(",", ""))
        except: continue
        rows.append((st, du, row[dv] if dv is not None else "0", row[nm]))
dev0 = Counter(d for _, _, d, _ in rows).most_common(1)[0][0]
ev = []
for st, du, d, n in rows:
    if d != dev0 or du <= 0: continue
    a = max(st, S); b = min(st + du, E)
    if b <= a: continue
    ev.append((a, 1, cat(n))); ev.append((b, -1, cat(n)))
ev.sort(key=lambda e: (e[0], e[1]))
active = defaultdict(int); buck = defaultdict(float); tp = None
for t, d, c in ev:
    if tp is not None and t > tp:
        seg = t - tp; on = [k for k, v in active.items() if v > 0]
        if len(on) == 1:
            k = on[0]
            if k.startswith("nccl:"): buck["exposed_" + k.split(":")[1]] += seg     # sole-active nccl -> exposed
            # (non-nccl sole-active handled elsewhere; we only care about nccl here)
        else:
            nk = [k for k in on if k.startswith("nccl:")]
            if nk: buck["overlap_" + nk[0].split(":")[1]] += seg                     # nccl concurrent -> overlap
    active[c] += d; tp = t

pct = lambda k: buck.get(k, 0) / W * 100
es, ed = pct("exposed_sparse"), pct("exposed_dense")
os_, od = pct("overlap_sparse"), pct("overlap_dense")
print("=== %s NCCL split (%% of fastest step) device=%s ===" % (label, dev0))
print("  exposed:  sparse(a2a)=%.3f  dense(allreduce)=%.3f   [N-E total=%.3f]" % (es, ed, es + ed))
print("  overlap:  sparse=%.3f      dense=%.3f               [N-O total=%.3f]" % (os_, od, os_ + od))
print("SUNBURST_NCCL %s NEs=%.3f NEd=%.3f NOs=%.3f NOd=%.3f" % (label, es, ed, os_, od))
