# §5 reproduction pipeline — turn an nsys capture into the PERF_ANALYSIS numbers

These scripts reproduce **BENCHMARK_RESULTS.md §5.1–§5.4** (our reproduction of upstream's
[`PERF_ANALYSIS.md`](https://github.com/NVIDIA/recsys-examples/blob/main/examples/hstu/training/benchmark/PERF_ANALYSIS.md))
from a single Nsight Systems capture of the upstream `exp4_caching_hr` e2e benchmark. Two steps: **capture**
(run the upstream benchmark under nsys on your own machines) then **analyze** (the python here). Everything is
plain `torchrun` + `nsys` + python — no scheduler or cluster infra required.

## 1. Capture (on your machines)

Run upstream's e2e benchmark under nsys, on the rank-0 node. `--nsys` makes the upstream script wrap the run;
`--nproc` is GPUs per node; for multi-node, replace upstream's `--standalone` in
`training/benchmark/scripts/run_single_experiment_local.sh` with your torchrun rendezvous flags.

```bash
cd examples/hstu
RS=training/benchmark/scripts/run_single_experiment_local.sh
# H100 (D256, contextual). GB300: drop --include-contextual and add --kv_channels 128 (Blackwell CUTLASS
# can't do head_dim 256 → upstream_issues Issue #1 / #4). --non_jagged = upstream's method.
EXA="--include-contextual --balanced_shuffler --kernel_backend cutlass --caching --ratio 0.1 \
     --dist_type hash_roundrobin --value_dist zipf --value_dist_alpha 1.05 \
     --kv_channels 256 --max_sequence_length 2048 --non_jagged"
bash $RS exp4_caching_hr --benchmark-type=e2e --nproc=8 --nsys --output-dir=./out --exp-args="$EXA"

# one capture → all four reports (do NOT split into separate runs)
REP=$(find ./out -name '*.nsys-rep' | head -1)
nsys stats --report nvtx_kern_sum     --format csv --output out/nvtx     "$REP"   # §5.3/§5.4 per-op busy-time
nsys stats --report nvtx_sum          --format csv --output out/nvtxsum  "$REP"   # per-range times (fastest step)
nsys stats --report cuda_gpu_kern_sum --format csv --output out/kern     "$REP"   # §5.2 raw kern-sum categories
nsys stats --report cuda_gpu_trace    --format csv --output out/gputrace "$REP"   # §5.2 exposed sweep timeline
nsys export --type sqlite --force-overwrite true --output out/rep.sqlite "$REP"   # step windows + kernel→NVTX map
```

## 2. Analyze (this directory)

| Doc section | What | Script | Inputs |
|---|---|---|---|
| §5.1 e2e MFU | median MFU over the steady-state window (iters 199–999) | (from the run log) | `run.log` |
| §5.1(B) jagged FLOP | exact `cal_hstu_flops` total, calibrated NORM = log_interval × world | `reconstruct_2324.py` | `run.log` |
| §5.2 exposed (coarse) | 8-category exposed % on rank0's fastest step | `exposed_faststep.py <sqlite> <gputrace.csv> <label>` | sqlite + gputrace |
| §5.2 gemm leaves (exact) | UVQK/PROJ/G-O by innermost NVTX range + nccl exposed/overlap | `exposed_gemm_split.py <sqlite> <gputrace.csv> <startNs> <endNs>` | sqlite + gputrace + fastest-step window |
| §5.2 raw kern-sum | HSTU/NCCL/GEMM/Eltwise/Embedding/Other, per-step ms | `kernsum_categorize.py <cuda_gpu_kern_sum.csv> …` | `kern` CSV |
| §5.3/§5.4 per-op MFU | attention + UVQK/proj TFLOPS & MFU, nj + jagged | `reconstruct_2324.py <dir>` | `nvtx` CSV + `run.log` |
| figures | sunbursts + raw-kernsum bars | `../plot_sunburst_nj.py`, `../plot_sunburst.py`, `../plot_kernsum_raw4.py` | numbers baked from the above |

`reconstruct_2324.py` carries both platform configs (H100: peak 989, D256, C3, 8 GPU; GB300: peak 2500, D128,
C0, 4 GPU) and self-tests the non-jagged path against the published doc numbers; every MFU is peak-checked.

## Method invariants (why these, not the obvious-but-wrong alternatives)

- **Time = actual kernel busy-time** (`nvtx_kern_sum` / `cuda_gpu_trace`), **never** `nvtx_gpu_proj_sum` — its
  projected span collapses overlapped multi-stream GEMM and under-counts it → non-physical MFU **> peak**.
- **De-dup to rank0**: `nvtx_kern_sum` aggregates all node-0 GPUs (one row per (kernel, GPU)); rank0 =
  Σ distinct kernels ÷ GPU-count ÷ steps.
- **FLOP = `cal_hstu_flops`** on the real per-sequence lengths (attention ∝ ΣLᵢ² quadratic, GEMM linear in
  tokens); for jagged, attention = the trainer's exact total − the linear GEMM/other terms (no effective-`T`
  approximation).
- **`.rep` → `.sqlite`** export must run on the same arch that produced the `.rep` (x86 nsys can hang on arm
  reps), but the resulting sqlite is arch-portable, so `exposed_gemm_split.py` can run anywhere.

MFU is on each GPU's bf16 dense peak (H100 989, GB300 2500 TFLOPS), so it is comparable across platforms; raw
absolute TFLOPS are not. See BENCHMARK_RESULTS.md §5 for the full write-up.

## Cross-checks (independent of the nsys pipeline)

- `gemm_microbench.py` — times the exact UVQK/projection GEMM shapes standalone with CUDA events (fwd + dgrad +
  wgrad). The training capture can't attribute per-op GEMM time (kernels launch via generic `cublasGemmEx`);
  this isolates it, cross-checking §5.4.
- `torch_profile_entry.sh` — runs the FUSED HSTU layer under `torch.profiler` (framework-attributed, in-context)
  to cross-check the §5.3/§5.4 per-op split without nsys.

## §1–§4 (not nsys-based)

The correctness + kernel/layer/e2e sweeps in BENCHMARK_RESULTS.md §1–§4 are captured by `../run_benchmarks.sh`
(`train` / `attn` / `layer` / `e2e` — plain torchrun, no cluster infra) and plotted by `../plot_attn_heatmaps.py`
(§2) and `../plot_layer_compare.py` (§3).
