"""Split the exposed 'GPU idle' bucket by WHAT THE HOST IS DOING during each GPU-idle gap.

Idle = instants with no categorized GPU kernel active (same n==0 sweep as exposed_faststep.py, on the gputrace CSV,
so the sub-buckets sum to the sunburst's idle). Each idle gap is then attributed to the CPU CUDA-runtime API spanning it
(from the sqlite CUPTI_ACTIVITY_KIND_RUNTIME) -> why the GPU is starved:
  I-launch  cudaLaunchKernel / cuLaunchKernel*      (CPU dispatching kernels - launch overhead)
  I-host    no CUDA API in the gap                  (pure host/python compute between launches)
  I-sync    cudaStreamSynchronize/DeviceSync/EventSync/EventQuery  (CPU blocked waiting on the GPU)
  I-copy    cudaMemcpy*/cudaMemset*                 (host<->device transfer)
  I-oth     any other API (events, etc.)
Usage: exposed_idle_split.py <rep.sqlite> <cuda_gpu_trace.csv> <label>
"""
import sqlite3, csv, sys, re, bisect
from collections import Counter, defaultdict

sq, csvp, label = sys.argv[1], sys.argv[2], sys.argv[3]
con = sqlite3.connect(sq); cur = con.cursor()
try:
    rows = cur.execute("SELECT n.start,n.end FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id "
                       "WHERE COALESCE(s.value,n.text) LIKE 'step %' AND n.end>n.start").fetchall()
except Exception:
    rows = cur.execute("SELECT start,end FROM NVTX_EVENTS WHERE text LIKE 'step %' AND end>start").fetchall()
S, E = min(rows, key=lambda r: r[1] - r[0]); W = E - S

# --- idle intervals: n==0 categorized-kernel sweep on the gputrace CSV (matches the sunburst) ---
CATS = [r"flash::compute_attn|HSTUAttention|Hstu_fwd|Hstu_bwd|hstu.*attn|blackwellhstu", r"nccl",
        r"nvjet|cublas|ampere_|sm\d+_gemm|cutlass_\w*gemm",
        r"layer_norm|_ln_mul|silu|dropout|elementwise|reduce_kernel|FillFunctor|CatArray|clamp|Functor",
        r"embedding|dyn_emb|dynamicemb|fbgemm|bucketize|permute|radix|DeviceScan|DeviceSelect|linearize_index"]
def is_kernel(n):  # a categorized GPU kernel (attention/nccl/gemm/elem/embedding/other) — everything counts as active
    return True
gp = []
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
        gp.append((st, du, row[dv] if dv is not None else "0"))
dev0 = Counter(d for _, _, d in gp).most_common(1)[0][0]
ev = []
for st, du, d in gp:
    if d != dev0 or du <= 0: continue
    a = max(st, S); b = min(st + du, E)
    if b <= a: continue
    ev.append((a, 1)); ev.append((b, -1))
ev.sort()
active = 0; tp = None; idle = []
for t, d in ev:
    if tp is not None and t > tp and active == 0: idle.append((tp, t))
    active += d; tp = t
if tp is not None and tp < E and active == 0: idle.append((tp, E))
tot_idle = sum(b - a for a, b in idle)

# --- attribute each idle gap to the CPU CUDA-runtime API spanning it ---
api = cur.execute("SELECT r.start,r.end,COALESCE(s.value,'?') FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
                  "LEFT JOIN StringIds s ON r.nameId=s.id WHERE r.end>? AND r.start<?", (S, E)).fetchall()
api.sort(); starts = [a[0] for a in api]
def kind(nm):
    if re.search(r"LaunchKernel", nm): return "I-launch"
    if re.search(r"Synchronize|EventQuery", nm): return "I-sync"
    if re.search(r"Memcpy|Memset", nm): return "I-copy"
    return "I-oth"
# EXCLUSIVE attribution: a gap can have concurrent APIs on multiple CPU threads, so sweep each gap and give every
# instant to the single highest-precedence active API kind (launch>sync>copy>oth); uncovered instants = I-host.
PRIO = {"I-launch": 4, "I-sync": 3, "I-copy": 2, "I-oth": 1}
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
con.close()

print("=== %s IDLE split (%% of fastest step, window %.1f ms) dev=%s ===" % (label, W / 1e6, dev0))
print("  idle total = %.2f%%" % (tot_idle / W * 100))
for k in ("I-launch", "I-host", "I-sync", "I-copy", "I-oth"):
    print("  %-9s %5.2f%%" % (k, buck.get(k, 0) / W * 100))
print("SUNBURST_IDLE %s %s" % (label, ",".join("%s=%.3f" % (k, buck.get(k, 0) / W * 100) for k in
      ("I-launch", "I-host", "I-sync", "I-copy", "I-oth"))))
