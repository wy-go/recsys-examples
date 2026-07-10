#!/usr/bin/env python3
"""§2.2 RAW kernel-time-sum — categorize `cuda_gpu_kern_sum` into HSTU/NCCL/GEMM/Eltwise/Embedding/Other.
Each kernel's full GPU busy-time summed by category, as % of the sum (overlaps double-counted -> NCCL over-counted;
that contrast with the exposed view is the point).

Usage: kernsum_categorize.py <cuda_gpu_kern_sum.csv> [<csv2> ...]   # one per case
"""
import csv, sys, re

CATS = [("Attention", r"flash::compute_attn|HSTUAttention|Hstu_fwd|Hstu_bwd|hstu.*attn|blackwellhstu"),
        ("NCCL",      r"nccl"),
        ("GEMM",      r"nvjet|cublas|ampere_|sm\d+_gemm|cutlass"),
        ("Eltwise",   r"layer_norm|_ln_mul|silu|dropout|elementwise|reduce_kernel|_weighted_layer_norm|multi_tensor|scan"),
        ("Embedding", r"embedding|dyn_emb|dynamicemb|fbgemm|bucketize|permute|radix|DeviceSelect|DeviceCompact"),
        ("Other",     r".*")]
def cat(n):
    for c, rx in CATS:
        if re.search(rx, n, re.I): return c
    return "Other"

def one(path):
    agg = {c: 0.0 for c, _ in CATS}
    for r in csv.DictReader(open(path)):
        agg[cat(r["Name"] or "")] += float(r["Total Time (ns)"])
    tot = sum(agg.values()) or 1.0
    return {c: agg[c] / tot * 100 for c, _ in CATS}

if __name__ == "__main__":
    print(f"{'file':40} " + " ".join(f"{c:>10}" for c, _ in CATS))
    for p in sys.argv[1:]:
        r = one(p)
        print(f"{p.split('/')[-1]:40} " + " ".join(f"{r[c]:9.1f}%" for c, _ in CATS))
