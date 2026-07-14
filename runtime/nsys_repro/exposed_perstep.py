#!/usr/bin/env python3
"""PER-STEP exposed-GPU-time partition over EVERY steady 'step N' window in a capture.

Reuses the EXACT exclusive single-sweep partition of exposed_partition.py (same NVTX gemm-op split,
same kernel->category regexes, same 11 leaves that tile each step to ~100%), but applies it to every
'step N' NVTX window instead of only the fastest one. Emits, per capture:
  * n_steps and the fastest / median / slowest step durations (ms)
  * the full 11-leaf breakdown for the fastest, median and slowest step
  * the time-weighted AGGREGATE 11-leaf breakdown = sum(leaf_time)/sum(step_time)*100
  * per-step arrays (list across steps) for every leaf, for boxplots

Slowest = max-duration steady step, UNLESS that max is >3x the median (obvious outlier), in which case
the ~p90 step is used instead (and flagged).

Usage: exposed_perstep.py <rep.sqlite> <cuda_gpu_trace.csv> [device_substr] [label]
Prints a human summary plus JSON (marked with 'JSON_PERSTEP <label> {...}') on the last line.
"""
import sqlite3, csv, sys, re, bisect, json, statistics
from collections import defaultdict, Counter

sq, csvp = sys.argv[1], sys.argv[2]
devsub = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None
label = sys.argv[4] if len(sys.argv) > 4 else ""

# --- gemm innermost-NVTX op split (verbatim from exposed_partition.py) ---
OP_PAT = [("UVQK", r"ln\+linear_bias\+silu|ln_linear_silu"), ("PROJ", r"linear_residual"), ("G-O", r"\bmlp\b")]
def op_of(text):
    for op, rx in OP_PAT:
        if re.search(rx, text, re.I): return op
    return None

con = sqlite3.connect(sq); cur = con.cursor()

# steady 'step N' windows
try:
    steps = cur.execute("SELECT n.start,n.end FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id "
                        "WHERE COALESCE(s.value,n.text) LIKE 'step %' AND n.end>n.start").fetchall()
except Exception:
    steps = cur.execute("SELECT start,end FROM NVTX_EVENTS WHERE text LIKE 'step %' AND end>start").fetchall()
steps.sort(key=lambda r: r[0])

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

# --- kernel -> category (verbatim from exposed_partition.py) ---
CATS = [("attention", r"flash::compute_attn|HSTUAttention|Hstu_fwd|Hstu_bwd|hstu.*attn|blackwellhstu"),
        ("nccl", r"nccl"), ("gemm", r"nvjet|cublas"),
        ("elem", r"layer_norm|_ln_mul|silu|dropout|elementwise|reduce_kernel|_weighted_layer_norm|multi_tensor|scan"),
        ("embedding", r"embedding|dyn_emb|dynamicemb|fbgemm|bucketize|permute|radix"), ("other", r".*")]
def cat(n):
    for c, rx in CATS:
        if re.search(rx, n, re.I): return c
    return "other"

allrows = list(csv.DictReader(open(csvp)))
if devsub:
    dev = next((r["Device"] for r in allrows if devsub in r["Device"]), None)
else:
    dev = Counter(r["Device"] for r in allrows).most_common(1)[0][0]

# pre-parse kernels on the target device once: (start, end, leaf-label)
KRN = []
for r in allrows:
    if r["Device"] != dev: continue
    st = float(r["Start (ns)"]); du = float(r["Duration (ns)"])
    if du <= 0: continue
    c = cat(r["Name"] or "")
    lab = c if c != "gemm" else "gemm:" + start2op.get(int(st), "G-O")
    KRN.append((st, st + du, lab))
KRN.sort()
KSTARTS = [k[0] for k in KRN]
MAXDUR = max((en - st for st, en, _ in KRN), default=0.0)  # longest kernel, for safe window lower-bound

LEAF = {"attention": "hstu fwd/bwd (attention)", "gemm:UVQK": "gemm / uvqk",
        "gemm:PROJ": "gemm / projection", "gemm:G-O": "gemm / others",
        "elem": "elementwise", "embedding": "embedding op", "nccl": "nccl(exposed)", "other": "others"}
ORDER = ["hstu fwd/bwd (attention)", "gemm / uvqk", "gemm / projection", "gemm / others",
         "elementwise", "embedding op", "GPU idle", "nccl(exposed)", "nccl(overlap)", "others", "overlapped"]

def partition(S, E):
    """exclusive single sweep of window [S,E] -> dict leaf-> ns time."""
    ev = []
    # any kernel overlapping [S,E] has start in [S-MAXDUR, E); MAXDUR bounds how far back to look.
    i = bisect.bisect_left(KSTARTS, S - MAXDUR)
    for j in range(i, len(KRN)):
        st, en, lab = KRN[j]
        if st >= E: break
        a = max(st, S); b = min(en, E)
        if b <= a: continue
        ev.append((a, 1, lab)); ev.append((b, -1, lab))
    ev.sort(key=lambda e: (e[0], e[1]))
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
    return buck

# run partition for every step
per = []   # list of dicts: {S,E,dur_ns, leaves:{leaf:ns}}
for S, E in steps:
    b = partition(S, E)
    per.append({"S": S, "E": E, "dur": E - S, "leaves": {k: b.get(k, 0.0) for k in ORDER}})

n = len(per)
durs = sorted(range(n), key=lambda i: per[i]["dur"])
i_fast = durs[0]
i_slow = durs[-1]
# median index by duration
med_val = statistics.median(p["dur"] for p in per)
i_med = min(range(n), key=lambda i: abs(per[i]["dur"] - med_val))
# outlier check on slowest
slow_note = ""
if per[i_slow]["dur"] > 3 * med_val:
    p90i = durs[int(round(0.9 * (n - 1)))]
    slow_note = "MAX %0.2fms >3x median -> using p90 step %.2fms instead" % (
        per[i_slow]["dur"]/1e6, per[p90i]["dur"]/1e6)
    i_slow = p90i

def pct(p):
    W = p["dur"]
    return {k: p["leaves"][k] / W * 100 for k in ORDER}

def col(p):
    d = pct(p); d["Total"] = sum(d[k] for k in ORDER); return d

# aggregate: sum(leaf)/sum(dur)
sum_leaf = {k: sum(p["leaves"][k] for p in per) for k in ORDER}
sum_dur = sum(p["dur"] for p in per)
agg = {k: sum_leaf[k] / sum_dur * 100 for k in ORDER}
agg["Total"] = sum(agg[k] for k in ORDER)

fast, med, slow = col(per[i_fast]), col(per[i_med]), col(per[i_slow])

print(f"=== {label} PER-STEP partition  device={dev}  n_steps={n} ===")
print(f"  fastest={per[i_fast]['dur']/1e6:.2f}ms  median={per[i_med]['dur']/1e6:.2f}ms  slowest={per[i_slow]['dur']/1e6:.2f}ms")
if slow_note: print("  " + slow_note)
print(f"  {'leaf':28s} {'fast':>8s} {'median':>8s} {'slow':>8s} {'AGG':>8s}")
for k in ORDER + ["Total"]:
    print(f"  {k:28s} {fast[k]:7.2f}% {med[k]:7.2f}% {slow[k]:7.2f}% {agg[k]:7.2f}%")

# per-step arrays for boxplots
arrays = {k: [round(pct(p)[k], 3) for p in per] for k in ORDER}
step_ms = [round(p["dur"]/1e6, 3) for p in per]
out = {"label": label, "n_steps": n, "device": dev,
       "fastest_ms": round(per[i_fast]["dur"]/1e6, 3),
       "median_ms": round(per[i_med]["dur"]/1e6, 3),
       "slowest_ms": round(per[i_slow]["dur"]/1e6, 3),
       "slow_note": slow_note,
       "table": {"fast": {k: round(fast[k], 2) for k in ORDER + ["Total"]},
                 "median": {k: round(med[k], 2) for k in ORDER + ["Total"]},
                 "slow": {k: round(slow[k], 2) for k in ORDER + ["Total"]},
                 "aggregate": {k: round(agg[k], 2) for k in ORDER + ["Total"]}},
       "step_ms": step_ms,
       "arrays": arrays}
print("JSON_PERSTEP " + label + " " + json.dumps(out))
