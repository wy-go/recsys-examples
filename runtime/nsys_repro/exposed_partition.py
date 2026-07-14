#!/usr/bin/env python3
"""EXACT single-sweep 'exposed GPU-time' partition of the fastest step (11 leaves that TILE the step).

This is the exclusive-partition version of the §5.2(B) / §8 breakdown. Unlike the coarse method (independent
per-bucket sweeps in exposed_faststep.py + exposed_nccl_split.py + exposed_gemm_split.py, whose overlap slivers
double-count and push the column over 100), this walks rank0's GPU timeline ONCE and charges every instant to
exactly one leaf, so the leaves sum to ~100.

Rule per timeline segment (from a start/end sweep of the fastest-step window on the gputrace CSV):
  n==0 active kernels            -> GPU idle
  n==1 (sole active kernel)      -> that kernel's leaf, categorized by its innermost NVTX op:
        attention                -> hstu fwd/bwd (attention)
        gemm (nvjet|cublas)      -> gemm/uvqk | gemm/projection | gemm/others  (by innermost NVTX op range)
        elementwise              -> elementwise
        embedding                -> embedding op
        nccl                     -> nccl(exposed)
        other                    -> others
  n>1  (concurrent kernels)      -> nccl(overlap) if any active kernel is nccl, else overlapped

Categorization regexes + the innermost-NVTX gemm split are copied verbatim from exposed_gemm_split.py so the
leaf assignment is identical to the EXACT gemm/nccl sub-splits.

Usage: exposed_partition.py <rep.sqlite> <cuda_gpu_trace.csv> <S_ns> <E_ns> [device_substr] [label]
"""
import sqlite3, csv, sys, re, bisect
from collections import defaultdict, Counter

sq, csvp, S, E = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
devsub = sys.argv[5] if len(sys.argv) > 5 else None
label = sys.argv[6] if len(sys.argv) > 6 else ""

# --- gemm innermost-NVTX op split (verbatim from exposed_gemm_split.py) ---
OP_PAT = [("UVQK", r"ln\+linear_bias\+silu|ln_linear_silu"), ("PROJ", r"linear_residual"), ("G-O", r"\bmlp\b")]
def op_of(text):
    for op, rx in OP_PAT:
        if re.search(rx, text, re.I): return op
    return None

con = sqlite3.connect(sq); cur = con.cursor()
rows = cur.execute("SELECT n.start, n.end, n.globalTid, "
                   "COALESCE((SELECT value FROM StringIds WHERE id=n.textId), n.text) AS t "
                   "FROM NVTX_EVENTS n WHERE n.end > n.start").fetchall()
op_by_tid = defaultdict(list)
for st, en, tid, t in rows:
    if not t: continue
    op = op_of(t)
    if op: op_by_tid[tid].append((st, en, op))
for tid in op_by_tid: op_by_tid[tid].sort()

kname = ("COALESCE((SELECT value FROM StringIds WHERE id=k.demangledName),"
         "(SELECT value FROM StringIds WHERE id=k.shortName))")
q = (f"SELECT k.start, {kname} AS nm, r.start AS launch, r.globalTid "
     f"FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON k.correlationId=r.correlationId")
start2op = {}
for kstart, nm, launch, tid in cur.execute(q).fetchall():
    if not nm or not re.search(r"nvjet|cublas", nm, re.I): continue
    ivs = op_by_tid.get(tid, [])
    op = None
    lo = bisect.bisect_right([iv[0] for iv in ivs], launch) - 1
    for j in range(lo, -1, -1):
        st, en, o = ivs[j]
        if st <= launch <= en: op = o; break
        if en < launch and lo - j > 50: break
    start2op[kstart] = op or "G-O"
con.close()

# --- kernel -> category (verbatim regexes from exposed_gemm_split.py) ---
CATS = [("attention", r"flash::compute_attn|HSTUAttention|Hstu_fwd|Hstu_bwd|hstu.*attn|blackwellhstu"),
        ("nccl", r"nccl"), ("gemm", r"nvjet|cublas"),
        ("elem", r"layer_norm|_ln_mul|silu|dropout|elementwise|reduce_kernel|_weighted_layer_norm|multi_tensor|scan"),
        ("embedding", r"embedding|dyn_emb|dynamicemb|fbgemm|bucketize|permute|radix"), ("other", r".*")]
def cat(n):
    for c, rx in CATS:
        if re.search(rx, n, re.I): return c
    return "other"

rows = list(csv.DictReader(open(csvp)))
if devsub:
    dev = next((r["Device"] for r in rows if devsub in r["Device"]), None)
else:
    dev = Counter(r["Device"] for r in rows).most_common(1)[0][0]

# --- single exclusive sweep ---
ev = []
for r in rows:
    if r["Device"] != dev: continue
    st = float(r["Start (ns)"]); du = float(r["Duration (ns)"])
    if du <= 0: continue
    a = max(st, S); b = min(st + du, E)
    if b <= a: continue
    c = cat(r["Name"] or "")
    lab = c if c != "gemm" else "gemm:" + start2op.get(int(st), "G-O")
    ev.append((a, 1, lab)); ev.append((b, -1, lab))
ev.sort(key=lambda e: (e[0], e[1]))

# leaf names
LEAF = {"attention": "hstu fwd/bwd (attention)", "gemm:UVQK": "gemm / uvqk",
        "gemm:PROJ": "gemm / projection", "gemm:G-O": "gemm / others",
        "elem": "elementwise", "embedding": "embedding op", "nccl": "nccl(exposed)", "other": "others"}
buck = defaultdict(float)
active = defaultdict(int); tp = None
for t, d, lab in ev:
    if tp is not None and t > tp:
        seg = t - tp
        on = [k for k, v in active.items() if v > 0]
        if not on:
            buck["GPU idle"] += seg
        elif len(on) == 1:
            buck[LEAF.get(on[0], "others")] += seg
        else:
            if any(k == "nccl" for k in on):
                buck["nccl(overlap)"] += seg
            else:
                buck["overlapped"] += seg
    active[lab] += d; tp = t

W = E - S
ORDER = ["hstu fwd/bwd (attention)", "gemm / uvqk", "gemm / projection", "gemm / others",
         "elementwise", "embedding op", "GPU idle", "nccl(exposed)", "nccl(overlap)", "others", "overlapped"]
print(f"=== {label} EXACT single-sweep partition  device={dev}  window={W/1e6:.3f}ms  [S={S},E={E}] ===")
tot = 0.0
for k in ORDER:
    v = buck.get(k, 0.0) / W * 100
    tot += v
    print(f"  {k:28s} {v:6.2f}%")
print(f"  {'Total':28s} {tot:6.2f}%")
print("PARTITION %s %s Total=%.3f" % (label, ",".join("%s=%.3f" % (k, buck.get(k, 0)/W*100) for k in ORDER), tot))
