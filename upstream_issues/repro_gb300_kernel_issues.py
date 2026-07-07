#!/usr/bin/env python3
"""Self-contained reproduction of three recsys-examples HSTU issues on Blackwell (GB300 / sm_103).

HOW TO RUN
----------
1. Copy this file into the upstream repo at  recsys-examples/examples/hstu/
2. From that directory (so `configs` and `modules` import), on a Blackwell GPU:
       python3 repro_gb300_kernel_issues.py
   (or from anywhere:  HSTU_DIR=/path/to/recsys-examples/examples/hstu python3 repro_gb300_kernel_issues.py)

It runs three cases; each prints REPRODUCED (the bug fired) or DID-NOT-REPRODUCE, and the exact runtime message.
Issue 3 (contextual) is backend-only (any GPU). Issues 1 and 2 require a Blackwell GPU (CUTLASS -> FusedHSTUAttention on sm 10).
See GB300_KERNEL_ISSUES.md for the write-up.
"""
import os
import sys
import traceback

# --- make `configs` / `modules` importable no matter the CWD ---
HSTU_DIR = os.path.abspath(os.environ.get("HSTU_DIR", os.path.dirname(__file__) or "."))
for p in (HSTU_DIR, os.path.dirname(HSTU_DIR)):     # hstu dir + examples dir (for `commons`)
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from configs.hstu_config import KernelBackend
from modules.hstu_attention import create_hstu_attention

DT = torch.bfloat16
NUM_HEADS = 4


def uniform_offsets(batch, seqlen, device):
    """Contiguous jagged offsets for `batch` sequences each of length `seqlen`: [0, L, 2L, ...]."""
    return (torch.arange(0, batch + 1, dtype=torch.int32, device=device) * seqlen)


def make_qkv(batch, seqlen, head_dim, device, requires_grad=False):
    T = batch * seqlen
    mk = lambda: torch.randn(T, NUM_HEADS * head_dim, dtype=DT, device=device, requires_grad=requires_grad)
    return mk(), mk(), mk()


def build(backend, head_dim):
    attn = create_hstu_attention(
        kernel_backend=backend, num_heads=NUM_HEADS,
        attention_dim=head_dim, linear_dim=head_dim, is_causal=True,
    )
    return attn.to(DT).cuda().eval()


def banner(n, title):
    print("\n" + "=" * 88 + f"\n[Issue {n}] {title}\n" + "=" * 88)


def issue1_contextual():
    banner(3, "Triton attention rejects a per-sequence contextual-length tensor")
    dev = torch.cuda.current_device()
    batch, seqlen, head_dim = 2, 16, 128
    attn = build(KernelBackend.TRITON, head_dim)
    tq, tk, tv = make_qkv(batch, seqlen, head_dim, dev)
    off = uniform_offsets(batch, seqlen, dev)
    # per-sequence contextual lengths as a tensor of shape (batch,) — exactly what the docstring says is allowed
    num_ctx = torch.tensor([1, 1], dtype=torch.int32, device=dev)
    print(f"  config: TRITON, batch={batch}, seqlen={seqlen}, head_dim={head_dim}, "
          f"num_contextuals=tensor{tuple(num_ctx.tolist())}")
    try:
        attn(tq, tk, tv, off, seqlen, seqlen, num_contextuals=num_ctx)
        print("  DID-NOT-REPRODUCE: forward accepted the tensor (already fixed on this build).")
    except AssertionError as e:
        print(f"  REPRODUCED (AssertionError): {e}")
    except Exception as e:
        print(f"  OTHER error: {type(e).__name__}: {e}")


def issue2_headdim256():
    banner(1, "Blackwell CUTLASS head_dim (kv_channels) 256 — test forward AND backward separately")
    dev = torch.cuda.current_device()
    batch, seqlen = 2, 64
    for head_dim in (128, 256):     # 128 is the control; 256 is under test
        # --- forward ---
        try:
            attn = build(KernelBackend.CUTLASS, head_dim)
            tq, tk, tv = make_qkv(batch, seqlen, head_dim, dev, requires_grad=True)
            off = uniform_offsets(batch, seqlen, dev)
            g = torch.randn(batch * seqlen, NUM_HEADS * head_dim, dtype=DT, device=dev)
            out = attn(tq, tk, tv, off, seqlen, seqlen)
            torch.cuda.synchronize()
            fwd = "OK"
        except Exception as e:
            print(f"  head_dim={head_dim}: forward FAILS ({type(e).__name__}): {str(e).splitlines()[0]}")
            continue
        # --- backward (training path) ---
        try:
            out.backward(g)
            torch.cuda.synchronize()
            bwd = "OK"
        except Exception as e:
            bwd = f"FAILS ({type(e).__name__}): {str(e).splitlines()[0]}"
        print(f"  head_dim={head_dim}: forward {fwd} | backward {bwd}")


def issue3_int32_overflow():
    banner(2, "INT32 overflow in the Blackwell CUTLASS backward at large batch × seqlen")
    dev = torch.cuda.current_device()
    head_dim = 128
    # (batch, seqlen, expected):  control below threshold, then a cell at/above 2^20 tokens
    cases = [(32, 16384, "runs"), (64, 16384, "overflow")]
    for batch, seqlen, expect in cases:
        tokens = batch * seqlen
        try:
            attn = build(KernelBackend.CUTLASS, head_dim)
            tq, tk, tv = make_qkv(batch, seqlen, head_dim, dev, requires_grad=True)
            off = uniform_offsets(batch, seqlen, dev)
            g = torch.randn(batch * seqlen, NUM_HEADS * head_dim, dtype=DT, device=dev)
            out = attn(tq, tk, tv, off, seqlen, seqlen)   # forward
            out.backward(g)                                # backward is where it overflows
            torch.cuda.synchronize()
            print(f"  batch={batch} seqlen={seqlen} (tokens={tokens:,}): fwd+bwd OK  (expected: {expect})")
        except OverflowError as e:
            print(f"  batch={batch} seqlen={seqlen} (tokens={tokens:,}): REPRODUCED (OverflowError): {e}")
        except torch.cuda.OutOfMemoryError:
            print(f"  batch={batch} seqlen={seqlen} (tokens={tokens:,}): SKIPPED (OOM — needs more HBM to reach this cell)")
        except Exception as e:
            print(f"  batch={batch} seqlen={seqlen} (tokens={tokens:,}): {type(e).__name__}: {str(e).splitlines()[0]}")
        finally:
            torch.cuda.empty_cache()


def main():
    if not torch.cuda.is_available():
        print("no CUDA device"); return
    p = torch.cuda.get_device_properties(0)
    print(f"device: {p.name}  sm_{p.major}{p.minor}  torch {torch.__version__}  cuda {torch.version.cuda}")
    if p.major != 10:
        print("NOTE: not a Blackwell GPU (sm_10x) — Issues 2 & 3 need Blackwell; CUTLASS falls back otherwise.")
    for fn in (issue2_headdim256, issue3_int32_overflow, issue1_contextual):
        try:
            fn()
        except Exception:
            traceback.print_exc()


if __name__ == "__main__":
    main()
