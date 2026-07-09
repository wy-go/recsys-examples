#!/usr/bin/env python3
"""EXACT §2.2 gemm sub-split (uvqk / projection / others) — upstream's method.

The busy-proportion split is an approximation; this is the real thing: classify each *sole-active* gemm kernel
in the exposed sweep by its **innermost NVTX op range**, which needs the sqlite (kernel names alone are ambiguous —
3 nvjet shapes are shared across UVQK/PROJ/MLP). Then the exposed gemm total splits by where the exposed time
actually lands.

Pipeline:
  1. sqlite: for each gemm kernel INSTANCE, find its innermost op range via launch-time correlation
     (kernel.correlationId -> RUNTIME.start = CPU launch -> NVTX range containing it on that thread).
     UVQK = ln+linear_bias+silu / ln_linear_silu ; PROJ = linear_residual ; G-O = mlp (or any other).
     -> map {kernel GPU start -> op}.
  2. gputrace CSV: sweep rank0's kernels in the fastest-step window; when a gemm kernel is the SOLE active
     category, add its dt to that kernel's op bucket (looked up by GPU start).
  -> exact exposed uvqk / proj / G-O (they sum to the coarse exposed gemm).

Usage: exposed_gemm_split.py <rep.sqlite> <cuda_gpu_trace.csv> <S_ns> <E_ns> [device_substr]
"""
import sqlite3, csv, sys, re
from collections import defaultdict

sq, csvp, S, E = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
devsub = sys.argv[5] if len(sys.argv) > 5 else None

OP_PAT = [("UVQK", r"ln\+linear_bias\+silu|ln_linear_silu"), ("PROJ", r"linear_residual"), ("G-O", r"\bmlp\b")]
def op_of(text):
    for op, rx in OP_PAT:
        if re.search(rx, text, re.I): return op
    return None

con = sqlite3.connect(sq); cur = con.cursor()
def has(t):
    return cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (t,)).fetchone() is not None

# --- 1. op ranges (start,end,globalTid,op) ---
def strjoin(col):  # NVTX text is either inline (.text) or via .textId -> StringIds
    return f"COALESCE((SELECT value FROM StringIds WHERE id=n.textId), n.text)"
rows = cur.execute(f"SELECT n.start, n.end, n.globalTid, {strjoin('')} AS t FROM NVTX_EVENTS n "
                   f"WHERE n.end > n.start").fetchall()
op_by_tid = defaultdict(list)
for st, en, tid, t in rows:
    if not t: continue
    op = op_of(t)
    if op: op_by_tid[tid].append((st, en, op))
for tid in op_by_tid: op_by_tid[tid].sort()

# --- 2. gemm kernels -> launch time -> innermost op ---
kname = "COALESCE((SELECT value FROM StringIds WHERE id=k.demangledName),(SELECT value FROM StringIds WHERE id=k.shortName))"
q = (f"SELECT k.start, {kname} AS nm, r.start AS launch, r.globalTid "
     f"FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON k.correlationId=r.correlationId")
start2op = {}
import bisect
for kstart, nm, launch, tid in cur.execute(q).fetchall():
    if not nm or not re.search(r"nvjet|cublas", nm, re.I): continue
    ivs = op_by_tid.get(tid, [])
    op = None
    # innermost op range containing the launch time
    lo = bisect.bisect_right([iv[0] for iv in ivs], launch) - 1
    for j in range(lo, -1, -1):
        st, en, o = ivs[j]
        if st <= launch <= en: op = o; break
        if en < launch and lo - j > 50: break
    start2op[kstart] = op or "G-O"
con.close()

# --- 3. exposed sweep on the gputrace, sub-splitting gemm by op ---
CATS = [("attention", r"flash::compute_attn|HSTUAttention|Hstu_fwd|Hstu_bwd|hstu.*attn|blackwellhstu"),
        ("nccl", r"nccl"), ("gemm", r"nvjet|cublas"),
        ("elem", r"layer_norm|_ln_mul|silu|dropout|elementwise|reduce_kernel|_weighted_layer_norm|multi_tensor|scan"),
        ("embedding", r"embedding|dyn_emb|dynamicemb|fbgemm|bucketize|permute|radix"), ("other", r".*")]
def cat(n):
    for c, rx in CATS:
        if re.search(rx, n, re.I): return c
    return "other"
rows = list(csv.DictReader(open(csvp)))
dev = None
if devsub:
    dev = next((r["Device"] for r in rows if devsub in r["Device"]), None)
else:
    from collections import Counter
    dev = Counter(r["Device"] for r in rows).most_common(1)[0][0]
ev = []
for r in rows:
    if r["Device"] != dev: continue
    st = float(r["Start (ns)"]); du = float(r["Duration (ns)"])
    a = max(st, S); b = min(st + du, E)
    if b <= a: continue
    c = cat(r["Name"] or "")
    lab = c if c != "gemm" else "gemm:" + start2op.get(int(st), "G-O")
    ev.append((a, 1, lab)); ev.append((b, -1, lab))
ev.sort(key=lambda e: (e[0], e[1]))
active = defaultdict(int); buck = defaultdict(float); tp = None
for t, d, lab in ev:
    if tp is not None and t > tp:
        seg = t - tp; on = [k for k, v in active.items() if v > 0]
        if len(on) == 1: buck[on[0]] += seg               # sole-active -> exposed
        elif "nccl" in on: buck["nccl_overlap"] += seg    # nccl overlapping compute -> N-O
    active[lab] += d; tp = t
W = E - S
print(f"device={dev}  window={W/1e6:.2f}ms")
g = {k: v for k, v in buck.items() if k.startswith("gemm:")}
tot_g = sum(g.values())
for k in sorted(g, key=lambda x: -g[x]):
    print(f"  {k:12s} {g[k]/W*100:6.3f}% of step   ({g[k]/tot_g*100:5.1f}% of exposed gemm)")
print(f"  gemm total (sole-active) = {tot_g/W*100:.3f}% of step")
# nccl exposed/overlap (the windowed sweep behind §2.2's nccl leaves)
print(f"  nccl  N-E(exposed) = {buck.get('nccl',0)/W*100:.3f}%   N-O(overlap) = {buck.get('nccl_overlap',0)/W*100:.3f}%")
