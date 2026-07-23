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

> **UPDATE 2026-07-23 — the KERNEL gap is fixed upstream-fork-side; the remaining blocker is recsys-examples
> itself.** `jiayus-nvidia/FBGEMM` `dev` landed **PR #18 "Hstu blackwell dim256"**
> (`ab8bfc6f57b630081459c76de1da25668be0bb6c`, 2026-07-22): a pure-Python CuTe-DSL **backward** for head_dim 256
> on SM100 (`src/hstu_blackwell/hstu_bwd_256_cute*.py`). The original assert is gone at that commit. What still
> blocks 256 in recsys-examples (verified live on `main`, which pins pre-#18 `647f0f57`):
> 1. **the framework guard** — `examples/hstu/ops/fused_hstu_op.py` raises
>    `"Blackwell fwd only supports head_dim in (64, 128)"` and the backward dispatch falls through to the slow
>    path for 256 (fwd `:372`, bwd `:763` in the v26.05/06 lineage);
> 2. **the submodule pin** — no recsys-examples ref consumes #18 (newest tag v26.06);
> 3. **packaging** — the v26.06 wrapper's module-level `import hstu.hstu_ops_gpu` no longer resolves against
>    FBGEMM dev, which installs that module at `hstu.hstu_blackwell.hstu_ops_gpu`.
> **Our workaround** (measured in [`BENCHMARK_RESULTS.md` §8b](../BENCHMARK_RESULTS.md)): image
> `Dockerfile.gb300.recsys.v2606kv256.py312` advances the FBGEMM submodule to #18; `KVDIM=256` in
> `runtime/scaleup/scaleup_entry.sh` relaxes the guard and makes the import tolerant at launch. Result: d256
> attention e2e **1020 TF/40.8%** (3.0× the Triton fallback's 337 TF/13.5%), fused layer **602 TF/24.1%** (+54%
> over Triton's 391). The d256 *backward* is still slower than d128's (878 vs 1298 TF), so Issue 1's practical
> gap is narrowed, not closed.

---

## Issue 2 — INT32 indexing overflow in the Blackwell CUTLASS backward (implementation limit, not precision)

> **UPDATE 2026-07-23 — FIXED as of v26.06.** The FBGEMM pinned by recsys-examples **v26.06** carries the Blackwell
> backward int32-overflow fix: the same CUTLASS-kv128 sweep now runs **0 OVF over the full extended grid** (through
> SL65536, deep past the old `BS×SL ≥ 2²⁰` wall), and the peak moves into the previously-overflowing corner
> (fwd+bwd 1454 TF / 58% @ BS1·SL65536). The grid's remaining grey corner (**≥ 2²³ ≈ 8.4M tokens**) is a genuine
> **HBM OOM staircase** (CUDA illegal-access on the backward workspace), *not* an int32 overflow. FBGEMM has since
> also landed `9e50261` ("Fix Blackwell HSTU **dQ workspace offset overflow**", 2026-07-10; included in the PR-#18
> image of Issue 1) which plausibly addresses that workspace illegal-access — not yet re-measured. The text below is
> preserved as the v26.05 record.

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
**Triton** path (in `hstu_attention.py`) hard-asserts `int`. The underlying `triton_hstu_mha` accepts a per-sequence
tensor, so the assert is simply over-restrictive. Net effect: **the Triton backend cannot train on real contextual data**,
even though the API says it can. (On GB300 this bites in practice because head_dim-256 real data can't use CUTLASS — see
Issue 1 — so it falls back to Triton, straight into this assert.)

> **Scope note:** this Issue is *only* about the module-level `TritonHSTUAttention` in `hstu_attention.py`, and it is
> fixable (collapse to scalar). It is **not** the whole contextual story on Blackwell. The e2e training benchmark uses the
> *fused* HSTU layer (`ops/fused_hstu_op.py`), where the roles invert: the Triton branch handles contextual fine, but the
> **Blackwell CUTLASS branch rejects contextual entirely** — see **Issue 4**. So the earlier framing "only Triton rejects
> contextual" is incorrect for the e2e CUTLASS path.

**Minimal fix (what we used):** accept a uniform contextual tensor by collapsing it to the scalar when all entries are
equal (and otherwise keep the tensor and let `triton_hstu_mha` handle it):

```python
if isinstance(num_contextuals, torch.Tensor):
    assert int(num_contextuals.min()) == int(num_contextuals.max()), "uniform contextual length required"
    num_contextuals = int(num_contextuals.max().item())
```

---

## Issue 4 — Blackwell **CUTLASS** (fused HSTU op) rejects contextual tokens entirely (the e2e training path)

**Where:** `ops/fused_hstu_op.py` — helper `_blackwell_num_contexts_or_none` (line ~61), called from the fused
forward (line ~377) and backward (line ~762) inside the `elif sm_major_version == 10:` (Blackwell) branch:

```python
def _blackwell_num_contexts_or_none(num_contexts):
    if num_contexts is None:
        return None
    if isinstance(num_contexts, int):
        if num_contexts == 0:
            return None
        raise ValueError("Blackwell fused_hstu_op does not support contextual tokens")
    if torch.count_nonzero(num_contexts).item() == 0:
        return None
    raise ValueError("Blackwell fused_hstu_op does not support contextual tokens")
```

**Status (verified on GB300):** ✅ reproduced — an e2e run with `--kernel_backend cutlass --include-contextual` aborts
with `ValueError: Blackwell fused_hstu_op does not support contextual tokens`.

**Trigger:** the **fused** HSTU layer (which the benchmark selects automatically whenever
`tensor_model_parallel_size == 1` — `training/trainer/utils.py:75-76`) **+ `kernel_backend=cutlass` + any nonzero
contextual**, on Blackwell (sm_10x). This rejects contextual *outright* — even a **uniform int** is refused (line ~67) —
which is strictly stronger than Issue 3 (Triton rejects only the *per-sequence tensor*, but runs fine with a uniform int).

**Why it matters (and how it differs from Issue 3):** these are two **distinct** restrictions in two **different** code
paths, and only together do they explain the GB300 contextual situation:

| | code path | rejects | fixable? | affects |
|---|---|---|---|---|
| **Issue 3** | `hstu_attention.py` `TritonHSTUAttention` | per-sequence *tensor* (accepts uniform int) | yes (collapse to scalar) | DEBUG/NATIVE-layer real-data runs |
| **Issue 4** | `fused_hstu_op.py` Blackwell CUTLASS branch | **all** contextual (even uniform int) | **no** (hard `raise`, no in-op fallback) | e2e FUSED-layer CUTLASS runs |

The fused op's **Triton** branch (`fused_hstu_op.py:177`) never reaches the guard, so **Triton runs contextual on
Blackwell**; it is specifically the **CUTLASS** kernel that cannot. Net: on GB300 the fast Blackwell CUTLASS kernel is
usable **only without contextual**, so any run that wants both the CUTLASS kernel *and* contextual is impossible — you
must give up one. This is why the §4c CUTLASS perf-analysis (and the §4b B-ladder) is **forced** non-contextual: to
profile the fast kernel we drop contextual; to keep contextual we'd have to switch to the (much slower) Triton kernel.

**Contrast Hopper (H100):** the `sm_major_version == 9` branch does **not** call `_blackwell_num_contexts_or_none`, so
Hopper CUTLASS runs contextual normally — which is why our H100 runs (and upstream's) are contextual and GB300's CUTLASS
runs are not.
