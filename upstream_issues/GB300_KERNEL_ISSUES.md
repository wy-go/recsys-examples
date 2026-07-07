# recsys-examples HSTU on Blackwell (GB300 / sm_103) — reproducible issues

Problems hit while benchmarking the HSTU attention kernels in `recsys-examples` (v26.05) on **GB300
(NVIDIA Blackwell, compute capability 10.3 / “sm_103”, CUDA 13, torch 2.9.1, bf16)**. Reproducible with the
self-contained `repro_gb300_kernel_issues.py` in this directory (drop it into `examples/hstu/` of the upstream repo
and run it on a Blackwell GPU).

Line numbers below are against `examples/hstu/modules/hstu_attention.py` as of v26.05. **Verified on GB300**
(NVIDIA GB300, sm_103, torch 2.9.1 / cu13) — status noted per issue.

---

## Issue 1 — Blackwell CUTLASS head_dim (kv_channels) 256: forward runs, **training path fails**

**Where:** the CUTLASS backend → `FusedHSTUAttention.forward` (line 253) → `hstu_attn_varlen_func`
(the `fbgemm_gpu_hstu` Blackwell CuTe-DSL kernel).

**Status (verified on GB300):** ✅ confirmed, and it's the **backward**, not the forward. head_dim 256 **forward
runs**; the **backward raises** `AssertionError: Only support head_dim 64 and 128`. head_dim 128 runs fwd+bwd. So
inference/forward at 256 is fine, but **training (backward) at head_dim 256 is unsupported** on the Blackwell CUTLASS
kernel — which is why an e2e training run with `--kernel_backend cutlass --kv_channels 256` fails on GB300.

**Trigger:** `kernel_backend=CUTLASS`, `kv_channels=256`, **backward pass**, on Blackwell (sm_103). `kv_channels` 64
and 128 run fwd+bwd; 256 forward runs, 256 backward asserts.

**Context (why it matters):** the end-to-end training benchmark with `--kernel_backend cutlass --kv_channels 256`
does not run on GB300, so head_dim 256 must fall back to the Triton backend. The corelib
`README.md` lists 256 among supported head dims, and Hopper (H100) CUTLASS runs 256 fwd+bwd — so this is a gap in the
Blackwell kernel path, not a model-config error. (Note: the Hopper **backward** launch template only instantiates
`Headdim <= 128` in `corelib/hstu/hopper/hstu_bwd_launch_template.h:193`, so a head-dim-256 backward limit is plausible
kernel-side.)

---

## Issue 2 — INT32 indexing overflow in the Blackwell CUTLASS backward (implementation limit, not precision)

**Where:** CUTLASS backward → Blackwell CuTe-DSL kernel constructing an int32 memref descriptor.

**Status (verified on GB300):** ✅ reproduced. `batch=32 × seqlen=16384` runs; `batch=64 × seqlen=16384` raises
`OverflowError: Value overflow: 2147483648 exceeds range of ...` — 2147483648 is exactly `2**31`.

**Symptom:** the **backward** raises a Python-level `OverflowError` from CuTe-DSL building a memref descriptor whose
int32 index exceeds `2**31 - 1`. The forward of the same config succeeds; the backward overflows. If unhandled, the
`OverflowError` escapes and aborts the whole benchmark sweep — not just the offending cell.

**Trigger (empirical, num_heads=4, head_dim=128, bf16):** the backward overflows once
`batch × seqlen ≥ 1,048,576` (= 2²⁰). At that point one backward buffer has
`batch × seqlen × num_heads × head_dim = 2²⁰ × 4 × 128 = 2²⁹` elements, and the overflowing int32 is a **byte
offset** into the backward's **fp32** dQ accumulator (4 bytes/element): `2²⁹ × 4 B = 2³¹ B = 2,147,483,648` — exactly
the value in the error, one past `INT32_MAX` (2³¹−1). (The bf16 q/k/v are 2 B, but dQ is accumulated in fp32; that
×4 is what turns a 2²⁹-element buffer into a 2³¹ byte offset — element count alone, 2²⁹ < 2³¹, would not overflow.)

| batch × seqlen | tokens | backward |
|---|---|---|
| 32 × 16384 | 524,288 (2¹⁹) | ✅ runs |
| 128 × 4096 | 524,288 (2¹⁹) | ✅ runs |
| **64 × 16384** | **1,048,576 (2²⁰)** | ❌ `OverflowError` (verified) |
| **128 × 8192** | **1,048,576 (2²⁰)** | ❌ `OverflowError` |
| **128 × 16384** | **2,097,152 (2²¹)** | ❌ `OverflowError` |

(These are exactly the 3 “OVF” cells in an 8×8 batch∈{1..128} × seqlen∈{128..16384} CUTLASS-kv128 sweep.)
The threshold scales with `num_heads × head_dim`, so it moves for other model shapes.

**Nature — indexing, not precision.** This is **not** a numerical/precision limitation (nothing about bf16 range or
accuracy). It is a pure **addressing** limit: an integer *offset* into global memory exceeds `2**31 - 1`. The result
would be bit-for-bit correct with a wider index. So it's an **implementation limitation** — a 32-bit index in the
Blackwell CuTe-DSL path — that surfaces as an ungraceful `OverflowError` rather than a 64-bit path or a clear
"size unsupported" message.

**Scope — Hopper already does it right (64-bit indexing).** The mature Hopper/C++ HSTU kernel declares its index type
as **`int64_t`** and computes every stride/offset in 64-bit:

```cpp
// corelib/hstu/csrc/hstu_attn/src/hstu.h:40
using index_t = int64_t;                                            // offset type is 64-bit
// corelib/hstu/csrc/hstu_attn/src/block_info.h:42
index_t q_offset(index_t row_stride) { return uint32_t(sum_s_q) * row_stride; }   // uint32 token idx * int64 stride -> int64
// corelib/hstu/csrc/hstu_attn/src/hstu_bwd.h:184
make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.dq_accum_ptr) + row_offset_dq_accum1);  // float* + int64 offset
```

**How H100 avoids it:** the offset variable is `int64_t`, and it holds an **element** offset (not a byte offset),
computed as `uint32_t(token_index) * int64_t(row_stride)` → the multiply is promoted to 64-bit (`block_info.h:42`).
It's then applied as `ElementAccum* (=float*) + int64_offset` (`hstu_bwd.h:184`), so the **×4 byte scaling for fp32
is done by 64-bit pointer arithmetic** — the int64 only ever holds ~2²⁹ (the element offset), never the 2³¹ byte
value, so nothing narrows to 32 bits. (The one 32-bit quantity on H100 is the *token index* `uint32_t(sum_s_q)`,
which caps total tokens at 2³² ≈ 4.29 B — a ceiling ~2000× higher than Blackwell's int32 *byte-offset* wall.)

The Blackwell CuTe-DSL kernel is a separate, newer JIT path whose memref descriptor pins that offset to **int32** and
(per the observed value) as a **byte** offset — so `2²⁹ elems × 4 B = 2³¹ B` overflows. Same math, 32-bit container.

**Fix / workaround.** Real fix: use a 64-bit index in the Blackwell memref-descriptor path (as Hopper's `index_t =
int64_t` already does). Benchmark-side workaround: wrap the per-config backward in `try/except OverflowError` and mark
the cell (distinct from `torch.cuda.OutOfMemoryError`) so one overflowing cell doesn't abort the whole sweep.

---

## Issue 3 — Triton attention rejects per-sequence contextual lengths (contradicts its own API)

**Where:** `TritonHSTUAttention.forward` (class at line 141), body at lines ~205–208:

```python
if num_contextuals is None:
    num_contextuals = 0
assert isinstance(
    num_contextuals, int
), "num_contextuals must be an integer in TritonHSTUAttention"
```

**Status (verified on GB300):** ✅ reproduced — `AssertionError: num_contextuals must be an integer in
TritonHSTUAttention` when passing a `(batch,)` tensor. (Backend-level; not Blackwell-specific — any GPU.)

**Trigger:** pass `num_contextuals` as a `torch.Tensor` of shape `(batch_size,)` — i.e. real data in which different
sequences have different numbers of contextual tokens (the standard case for ranking datasets such as MovieLens /
KuaiRand that carry contextual features).

**Why it's a bug (not a config error):** the method's own signature and docstring advertise the tensor form —
`num_contextuals: Optional[Union[int, torch.Tensor]]` … *“could be a single integer **or a tensor of shape
(batch_size,)** when different sequences have different number of contextuals.”* The two sibling backends accept it:
`TorchHSTUAttention.forward` and `FusedHSTUAttention.forward` both do `num_contextuals.to(torch.int32)`. Only the
**Triton** path hard-asserts `int`. The underlying `triton_hstu_mha` accepts a per-sequence tensor, so the assert is
simply over-restrictive. Net effect: **the Triton backend cannot train on real contextual data**, even though the API
says it can. (On GB300 this bites in practice because head_dim-256 real data can't use CUTLASS — see Issue 1 — so it
falls back to Triton, straight into this assert.)

**Minimal fix (what we used):** accept a uniform contextual tensor by collapsing it to the scalar when all entries are
equal (and otherwise keep the tensor and let `triton_hstu_mha` handle it):

```python
if isinstance(num_contextuals, torch.Tensor):
    assert int(num_contextuals.min()) == int(num_contextuals.max()), "uniform contextual length required"
    num_contextuals = int(num_contextuals.max().item())
```
