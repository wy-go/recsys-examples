#!/usr/bin/env python3
"""§2.4 isolated GEMM micro-benchmark — reproduces upstream PERF_ANALYSIS §2.4 UVQK/projection GEMM TFLOPS+MFU.
Times the EXACT GEMM shapes (per GPU step, 8 layers) standalone in bf16 with CUDA events, fwd + bwd (dgrad+wgrad).
The training capture can't attribute per-op GEMM time (kernels launch via generic cublasGemmEx); this measures the
kernel capability at upstream's shapes directly. Run on H100 (peak 989) and GB300 (peak 2500).
Usage: python3 gemm_microbench.py --peak 989   # H100 ;  --peak 2500 for GB300
"""
import argparse, torch

ap = argparse.ArgumentParser()
ap.add_argument("--peak", type=float, required=True, help="bf16 dense TFLOPS peak (H100 989, GB300 2500)")
ap.add_argument("--iters", type=int, default=200)
ap.add_argument("--warmup", type=int, default=50)
# upstream shapes (per GPU step, aggregate over 8 layers is applied via L)
ap.add_argument("-B", type=int, default=32); ap.add_argument("-T", type=int, default=4099)
ap.add_argument("-H", type=int, default=1024); ap.add_argument("-N", type=int, default=4)
ap.add_argument("-D", type=int, default=256); ap.add_argument("-L", type=int, default=8)
a = ap.parse_args()
dev = "cuda"; dt = torch.bfloat16
M = a.B * a.T                       # 131,168
UV_N = 4 * a.N * a.D                # UVQK output width (4*N*D)
PROJ_K = a.N * a.D                  # projection input width (N*D)
torch.backends.cuda.matmul.allow_tf32 = False

def time_ms(fn):
    for _ in range(a.warmup): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(a.iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / a.iters   # ms per call

def bench(name, m, n, k, flop):
    # fwd: [m,k]@[k,n]; dgrad: [m,n]@[n,k]; wgrad: [k,m]@[m,n]
    A = torch.randn(m, k, device=dev, dtype=dt); W = torch.randn(k, n, device=dev, dtype=dt)
    dY = torch.randn(m, n, device=dev, dtype=dt)
    fwd = time_ms(lambda: torch.matmul(A, W))
    dgrad = time_ms(lambda: torch.matmul(dY, W.t()))
    wgrad = time_ms(lambda: torch.matmul(A.t(), dY))
    bwd = dgrad + wgrad
    for ph, ms, fl in [("FWD", fwd, flop), ("BWD", bwd, 2 * flop), ("FWD+BWD", fwd + bwd, 3 * flop)]:
        tf = fl / (ms / 1e3) / 1e12
        print(f"  {name:11s} {ph:8s} {fl/1e12*a.L:6.2f}T  {ms*a.L:8.3f} ms  {tf:7.1f} TFLOPS  {tf/a.peak*100:5.1f}% MFU")

print(f"GEMM micro-bench  M={M} UVQK(N={UV_N},K={a.H}) proj(N={a.H},K={PROJ_K})  bf16  peak={a.peak}  (x{a.L} layers)")
print(f"  {'op':11s} {'phase':8s} {'FLOPs':>6s}  {'time':>10s}  {'TFLOPS':>8s}  {'MFU':>7s}")
bench("UVQK",       M, UV_N, a.H, 2 * M * UV_N * a.H)      # 131168 x 4096 x 1024
bench("Projection", M, a.H, PROJ_K, 2 * M * a.H * PROJ_K)  # 131168 x 1024 x 1024
print(f"  [{torch.cuda.get_device_name(0)}, torch {torch.__version__}]")
