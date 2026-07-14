#!/usr/bin/env python3
"""PER-STEP NCCL and IDLE sub-splits, time-weighted aggregated over EVERY steady 'step N' window.

Generalizes exposed_nccl_split.py and exposed_idle_split.py (which each operate on the single
fastest step only) to run over ALL 80 steady steps and produce a time-weighted aggregate
(Sigma leaf_ns / Sigma step_ns * 100), matching exposed_perstep.py's aggregate convention.

Sub-leaves emitted (all as % of step):
  NCCL:  N-Ed (dense exposed), N-Es (sparse exposed), N-Od (dense overlap), N-Os (sparse overlap)
         dense  = AllReduce/AllGather/ReduceScatter/Broadcast  (gradient collectives)
         sparse = SendRecv/AllToAll                            (embedding all-to-all)
         exposed = nccl sole-active;  overlap = nccl concurrent with compute
  IDLE:  I-launch (cudaLaunchKernel), I-host (no CUDA API), I-sync (Synchronize/EventQuery),
         I-copy (Memcpy/Memset), I-oth (any other API) -- exclusive precedence sweep per idle gap.

The NCCL sub-leaves sum to exposed_perstep's nccl(exposed)+nccl(overlap); the IDLE sub-leaves
sum to exposed_perstep's GPU idle -- the window sweep + device are identical.

Usage: exposed_perstep_subsplit.py <rep.sqlite> <cuda_gpu_trace.csv> [device_substr] [label]
Prints per-capture sub-leaf table (% of step) + group totals for reconciliation, plus a JSON line.
"""
import sqlite3, csv, sys, re, bisect, json
from collections import Counter, defaultdict

sq, csvp = sys.argv[1], sys.argv[2]
devsub = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None
label = sys.argv[4] if len(sys.argv) > 4 else ""

# --- steady 'step N' windows (verbatim from exposed_perstep.py) ---
con = sqlite3.connect(sq); cur = con.cursor()
try:
    steps = cur.execute("SELECT n.start,n.end FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id "
                        "WHERE COALESCE(s.value,n.text) LIKE 'step %' AND n.end>n.start").fetchall()
except Exception:
    steps = cur.execute("SELECT start,end FROM NVTX_EVENTS WHERE text LIKE 'step %' AND end>start").fetchall()
steps.sort(key=lambda r: r[0])

# --- gputrace CSV -> per-device kernel rows ---
allrows = []
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
        allrows.append((st, du, row[dv] if dv is not None else "0", row[nm]))

if devsub:
    dev = next((d for _, _, d, _ in allrows if devsub in d), None)
else:
    dev = Counter(d for _, _, d, _ in allrows).most_common(1)[0][0]

# --- NCCL categorization (verbatim from exposed_nccl_split.py) ---
SPARSE = re.compile(r"SendRecv|AllToAll|alltoall", re.I)
DENSE = re.compile(r"AllReduce|AllGather|ReduceScatter|Broadcast", re.I)
def nccl_kind(n):
    if SPARSE.search(n): return "sparse"
    if DENSE.search(n): return "dense"
    return "dense"
def ncat(n):
    if re.search(r"nccl", n, re.I): return "nccl:" + nccl_kind(n)
    if re.search(r"flash::compute_attn|HSTUAttention|Hstu_fwd|Hstu_bwd|hstu.*attn|blackwellhstu", n, re.I): return "attention"
    if re.search(r"nvjet|cublas|ampere_|sm\d+_gemm|cutlass_\w*gemm", n, re.I): return "gemm"
    if re.search(r"layer_norm|_ln_mul|silu|dropout|elementwise|reduce_kernel|FillFunctor|CatArray|clamp|Functor", n, re.I): return "elem"
    if re.search(r"embedding|dyn_emb|dynamicemb|fbgemm|bucketize|permute|radix|DeviceScan|DeviceSelect|linearize_index", n, re.I): return "embedding"
    return "other"

# pre-parse target-device kernels once: (start, end, ncat-label)
DEVK = []
for st, du, d, n in allrows:
    if d != dev or du <= 0: continue
    DEVK.append((st, st + du, ncat(n)))
DEVK.sort()
DSTARTS = [k[0] for k in DEVK]
DMAX = max((en - st for st, en, _ in DEVK), default=0.0)

def nccl_window(S, E):
    """sweep window [S,E]; return ns bucket for exposed_/overlap_ sparse/dense."""
    ev = []
    i = bisect.bisect_left(DSTARTS, S - DMAX)
    for j in range(i, len(DEVK)):
        st, en, c = DEVK[j]
        if st >= E: break
        a = max(st, S); b = min(en, E)
        if b <= a: continue
        ev.append((a, 1, c)); ev.append((b, -1, c))
    ev.sort(key=lambda e: (e[0], e[1]))
    active = defaultdict(int); buck = defaultdict(float); tp = None
    for t, d, c in ev:
        if tp is not None and t > tp:
            seg = t - tp; on = [k for k, v in active.items() if v > 0]
            # Match exposed_perstep's boundary: nccl treated as ONE leaf for exposed/overlap.
            #   exposed = nccl active with NO non-nccl compute active (nccl sole-active, incl.
            #             two nccl KINDS running together -> still exposed, not overlap).
            #   overlap = nccl active concurrently with >=1 non-nccl compute kernel.
            # Sub-split by kind; a segment with BOTH sparse & dense nccl is split 50/50 (unbiased;
            # sparse/dense sub-values aren't constrained by the group-total consistency check).
            kinds = set(k.split(":")[1] for k in on if k.startswith("nccl:"))
            if kinds:
                grp = "overlap" if any(not k.startswith("nccl:") for k in on) else "exposed"
                if len(kinds) == 1:
                    buck[grp + "_" + next(iter(kinds))] += seg
                else:
                    buck[grp + "_sparse"] += seg / 2.0
                    buck[grp + "_dense"] += seg / 2.0
        active[c] += d; tp = t
    return buck

# --- IDLE: idle intervals (n==0 sweep on all device kernels) + CPU-API attribution ---
def idle_intervals(S, E):
    ev = []
    i = bisect.bisect_left(DSTARTS, S - DMAX)
    for j in range(i, len(DEVK)):
        st, en, _ = DEVK[j]
        if st >= E: break
        a = max(st, S); b = min(en, E)
        if b <= a: continue
        ev.append((a, 1)); ev.append((b, -1))
    ev.sort()
    active = 0; tp = None; idle = []
    for t, d in ev:
        if tp is not None and t > tp and active == 0: idle.append((tp, t))
        active += d; tp = t
    if tp is not None and tp < E and active == 0: idle.append((tp, E))
    elif tp is None: idle.append((S, E))  # no kernel in window -> fully idle
    return idle

def kind(nm):
    if re.search(r"LaunchKernel", nm): return "I-launch"
    if re.search(r"Synchronize|EventQuery", nm): return "I-sync"
    if re.search(r"Memcpy|Memset", nm): return "I-copy"
    return "I-oth"
PRIO = {"I-launch": 4, "I-sync": 3, "I-copy": 2, "I-oth": 1}

def idle_window(S, E):
    idle = idle_intervals(S, E)
    api = cur.execute("SELECT r.start,r.end,COALESCE(s.value,'?') FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
                      "LEFT JOIN StringIds s ON r.nameId=s.id WHERE r.end>? AND r.start<?", (S, E)).fetchall()
    api.sort(); starts = [a[0] for a in api]
    buck = defaultdict(float)
    for ga, gb in idle:
        i = max(0, bisect.bisect_left(starts, ga) - 3)
        segs = []
        j = i
        while j < len(api) and api[j][0] < gb:
            a, b, nm = api[j]; a2, b2 = max(a, ga), min(b, gb)
            if b2 > a2: segs.append((a2, 1, kind(nm))); segs.append((b2, -1, kind(nm)))
            j += 1
        segs.sort()
        active_k = defaultdict(int); tp2 = ga
        for t, d, k in segs:
            if t > tp2:
                on = [kk for kk, vv in active_k.items() if vv > 0]
                if on: buck[max(on, key=lambda x: PRIO[x])] += t - tp2
                else: buck["I-host"] += t - tp2
            active_k[k] += d; tp2 = t
        if tp2 < gb: buck["I-host"] += gb - tp2
    return buck

# --- accumulate over all steps (ns), then time-weighted aggregate ---
NKEYS = ["exposed_dense", "exposed_sparse", "overlap_dense", "overlap_sparse"]
IKEYS = ["I-launch", "I-host", "I-sync", "I-copy", "I-oth"]
sum_n = defaultdict(float); sum_i = defaultdict(float); sum_dur = 0.0
for S, E in steps:
    nb = nccl_window(S, E); ib = idle_window(S, E)
    for k in NKEYS: sum_n[k] += nb.get(k, 0.0)
    for k in IKEYS: sum_i[k] += ib.get(k, 0.0)
    sum_dur += (E - S)
con.close()

def p(x): return x / sum_dur * 100
NEd, NEs, NOd, NOs = p(sum_n["exposed_dense"]), p(sum_n["exposed_sparse"]), p(sum_n["overlap_dense"]), p(sum_n["overlap_sparse"])
I = {k: p(sum_i[k]) for k in IKEYS}

print("=== %s subsplit  device=%s  n_steps=%d  Sigma_step=%.1f ms ===" % (label, dev, len(steps), sum_dur/1e6))
print("  NCCL leaves (%% of step):")
print("    N-Ed (dense exposed)  = %6.2f%%" % NEd)
print("    N-Es (sparse exposed) = %6.2f%%" % NEs)
print("    N-Od (dense overlap)  = %6.2f%%" % NOd)
print("    N-Os (sparse overlap) = %6.2f%%" % NOs)
print("    -> nccl(exposed)=N-Ed+N-Es = %.2f%% ;  nccl(overlap)=N-Od+N-Os = %.2f%%" % (NEd+NEs, NOd+NOs))
print("  IDLE leaves (%% of step):")
for k in IKEYS: print("    %-8s = %6.2f%%" % (k, I[k]))
print("    -> GPU idle = %.2f%%" % sum(I.values()))
out = {"label": label, "device": dev, "n_steps": len(steps),
       "N-Ed": round(NEd, 2), "N-Es": round(NEs, 2), "N-Od": round(NOd, 2), "N-Os": round(NOs, 2),
       "nccl_exposed": round(NEd+NEs, 2), "nccl_overlap": round(NOd+NOs, 2),
       "I-launch": round(I["I-launch"], 2), "I-host": round(I["I-host"], 2), "I-sync": round(I["I-sync"], 2),
       "I-copy": round(I["I-copy"], 2), "I-oth": round(I["I-oth"], 2), "GPU_idle": round(sum(I.values()), 2)}
print("JSON_SUBSPLIT " + label + " " + json.dumps(out))
