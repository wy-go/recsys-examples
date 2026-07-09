#!/usr/bin/env python3
"""§2.3/§2.4 per-op MFU reconstruction — ONE tested tool (replaces hand-calc).

Numbers come from two exact sources, never from nsys "projected" time (that under-counts overlapped
multi-stream GEMM -> non-physical MFU >peak; see exec-log §2.4 caveat):

  TIME  = actual kernel busy-time from `nvtx_kern_sum`, per op (attention=flash/HSTUAtt, GEMM=nvjet/cublas),
          de-duplicated by GPU-count (the report aggregates all node0 GPUs -> one row per (kernel,GPU); rank0
          = sum distinct kernels / GPU-count), /20 profiled steps. Per-step compute busy-time is per-step-stable
          (fastest vs median 0.3% on the gputrace), so this equals upstream's rank0 fastest-step value.
  FLOP  = recsys `commons/utils/perf.py:cal_hstu_flops` formula (GEMM linear in Σseqlens, attention quadratic
          in the real per-sequence lengths). Non-jagged: closed form (all seqlens=T). Jagged: GEMM/other are
          linear -> exact from Σseqlens; attention = total - gemm - other, where `total` is the trainer's exact
          cal_hstu_flops output (calibrated from the run log: flops_interval / (20 steps * 16 GPU)).

Self-test: the non-jagged path must reproduce the published doc numbers, and every MFU is peak-checked (<100%).

Usage: reconstruct_2324.py <dir-with-{plat}-{nj,jag}_kern.csv-and-.log>
"""
import csv, re, sys, statistics
from collections import defaultdict

# ---- config (upstream exp4; GB300 forced D128/non-ctx) ----
CFG = {
    "h100":  dict(heads=4, dim=256, hidden=1024, C=3, layers=8, B=32, peak=989,  gpus=8),
    "gb300": dict(heads=4, dim=128, hidden=1024, C=0, layers=8, B=32, peak=2500, gpus=4),
}
# published non-jagged per-op FLOP (T), the self-test target (from the doc / upstream §1.6)
DOC_NJ = {
    "h100":  dict(attn=30.83, uvqk=26.41, proj=6.60),
    "gb300": dict(attn=15.39, uvqk=13.19, proj=3.30),
}
NVTX = dict(attn=(":hstu attn fwd", ":hstu attn bwd", ("flash", "hstuatt")),
            uvqk=(":hstu ln+linear_bias+silu fwd", ":ln_linear_silu bwd", ("nvjet", "cublas")),
            proj=(":hstu linear_residual fwd", ":hstu linear_residual bwd", ("nvjet", "cublas")))

def busy(csvf, rng, kmatch, gpus):
    """rank0 per-step busy-time (ms) for one op: sum distinct kernels / gpu-count / 20 steps."""
    bn, cnt = defaultdict(float), defaultdict(int)
    for r in csv.DictReader(open(csvf)):
        if (r["NVTX Range"] or "").strip() == rng and any(m in (r["Kernel Name"] or "").lower() for m in kmatch):
            bn[r["Kernel Name"]] += float(r["Total Time (ns)"]); cnt[r["Kernel Name"]] += 1
    return sum(v / cnt[k] for k, v in bn.items()) / 20 / 1e6

def gemm_per_token(c):   # cal_hstu_flops, fwd, per token per layer
    uvqk = 2 * 4 * c["heads"] * c["dim"] * c["hidden"]   # qkvu proj
    proj = 2 * c["heads"] * c["hidden"] * c["dim"]        # output proj
    return uvqk, proj
def other_per_token(c):  # mul (x2 bwd) + residual add
    return 2 * c["heads"] * c["dim"] + c["heads"] * c["hidden"]
def attn_flop_closed(c, seqlens):  # cal_hstu_flops attention term, fwd+bwd (x3.5), summed over seqs, x layers
    tot = 0.0
    for L in seqlens:
        hist = L - c["C"]
        f = 4 * c["heads"] * L * (c["C"] + hist) * c["dim"] - 2 * c["heads"] * hist * hist * c["dim"]
        tot += f
    return tot * 3.5 * c["layers"]

def sigma_seqlens(c, tokens_logged):   # post-doubling + contextual:  Σ(2·L_item + C)
    return 2 * tokens_logged + c["B"] * c["C"]

def flops_interval(logf):              # trainer's exact cal_hstu_flops output for the interval (timing-independent)
    v = [float(a) * float(b) / 1000 for a, b in re.findall(
         r"achieved FLOPS ([\d.]+) TFLOPS.*?", open(logf).read()) or []] if False else []
    v = []
    for ln in open(logf):
        m = re.search(r"achieved FLOPS ([\d.]+) TFLOPS", ln); e = re.search(r"elapsed_time ([\d.]+) ms", ln)
        if m and e: v.append(float(m.group(1)) * float(e.group(1)) / 1000)
    return statistics.median(v)
def tokens_logged(logf):
    t = sorted(set(int(x) for x in re.findall(r"tokens (\d+)", open(logf).read())))
    return t[-1] / (20 * 16)           # global interval tokens / (20 steps * 16 GPU)

NORM = 20 * 16   # log_interval steps * world_size (validated: flops_interval_nj / total_nj = 320 on both platforms)

def perop_flop(c, seqlens_sigma, total=None):
    """Exact per-op FLOP (T), fwd+bwd. GEMM/other linear (exact); attention closed-form if total is None else total-gemm-other."""
    uvqk_ft, proj_ft = gemm_per_token(c); oth_ft = other_per_token(c)
    L = c["layers"]
    uvqk = uvqk_ft * 3 * L * seqlens_sigma / 1e12     # gemm x3 (fwd + 2x bwd)
    proj = proj_ft * 3 * L * seqlens_sigma / 1e12
    other = oth_ft * L * seqlens_sigma / 1e12
    if total is None:                                  # non-jagged self-test: closed-form attention
        attn = attn_flop_closed(c, [seqlens_sigma / c["B"]] * c["B"]) / 1e12
    else:
        attn = total - uvqk - proj - other             # jagged: exact attention = total - linear terms
    return dict(attn=attn, uvqk=uvqk, proj=proj, other=other)

def report(plat, d):
    c = CFG[plat]; base = f"{d}/{plat}"
    # ---- FLOP ----
    sig_nj = sigma_seqlens(c, 65536)                   # non-jagged tokens_logged = 32*2048
    fl_nj = perop_flop(c, sig_nj)
    tok_jg = tokens_logged(f"{base}-jag.log"); sig_jg = sigma_seqlens(c, tok_jg)
    tot_jg = flops_interval(f"{base}-jag.log") / NORM
    fl_jg = perop_flop(c, sig_jg, total=tot_jg)
    # ---- self-test: nj FLOP must match the doc ----
    for op in ("attn", "uvqk", "proj"):
        got, want = fl_nj[op], DOC_NJ[plat][op]
        assert abs(got - want) / want < 0.01, f"SELF-TEST FAIL {plat} {op}: {got:.3f} != doc {want}"
    print(f"===== {plat.upper()}  (peak {c['peak']} TFLOPS; nj self-test PASSED) =====")
    print(f"  jagged Σseqlens/GPU/step={sig_jg:.0f}  total/GPU/step={tot_jg:.4f}T  "
          f"(attn {fl_jg['attn']:.3f} / uvqk {fl_jg['uvqk']:.3f} / proj {fl_jg['proj']:.3f}T)")
    # ---- MFU per op (fwd+bwd), TIME from busy() ----
    RB = dict(attn=2.5, uvqk=2.0, proj=2.0)             # bwd/fwd FLOP ratio
    for op in ("attn", "uvqk", "proj"):
        frng, brng, km = NVTX[op]
        tf = busy(f"{base}-jag_kern.csv", frng, km, c["gpus"])
        tb = busy(f"{base}-jag_kern.csv", brng, km, c["gpus"])
        ff = fl_jg[op] / (1 + RB[op]); fb = fl_jg[op] - ff
        for ph, t, fl in (("FWD", tf, ff), ("BWD", tb, fb), ("F+B", tf + tb, fl_jg[op])):
            tflops = fl / t * 1000; mfu = tflops / c["peak"] * 100
            assert mfu < 100, f"NON-PHYSICAL {plat} {op} {ph}: {mfu:.1f}% > peak"
            print(f"    {op:5s} {ph:3s}  busy={t:7.3f}ms  FLOP={fl:6.3f}T  {tflops:7.0f} TFLOPS  MFU={mfu:6.2f}%")

if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    for plat in ("h100", "gb300"):
        report(plat, d)
