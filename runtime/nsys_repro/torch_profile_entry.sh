#!/usr/bin/env bash
# §2.3/§2.4 clean per-op reproduction via torch.profiler (framework-attributed, in-context).
# Injects a torch.profiler block into upstream hstu_layer_benchmark.py (reusing its model construction),
# runs the FUSED HSTU layer fwd+bwd under the profiler, and emits:
#   (1) a Perfetto/Chrome timeline trace (op-labeled) -> torchprof_*.json
#   (2) key_averages(group_by_input_shape) -> UVQK vs projection GEMM + attention, per-op CUDA time + FLOPs
# Env: DPH (dim-per-head: 256 H100 / 128 GB300), OUT (output dir), EXROOT (recsys-examples root).
set -uo pipefail
DPH=${DPH:-256}; OUT=${OUT:-/tmp/torchprof}; EXROOT=${EXROOT:-/opt/recsys-examples}
mkdir -p "$OUT"
cd "$EXROOT/examples/hstu" || { echo "no recsys examples at $EXROOT"; exit 2; }
export PYTHONPATH="$EXROOT/examples:$PYTHONPATH"; export TRITON_PTXAS_PATH=${TRITON_PTXAS_PATH:-/usr/local/cuda/bin/ptxas}
BM=training/benchmark/scripts/hstu_layer_benchmark.py

# --- inject the torch.profiler block right before the '# nsys' section (idempotent) ---
python3 - "$BM" <<'PYX'
import sys,io
f=sys.argv[1]; s=open(f).read()
if "TORCH_PROFILE" in s:
    print("already patched"); sys.exit(0)
anchor="    # nsys — only when --profile True"
block='''    # torch.profiler — clean per-op attribution for BENCHMARK_RESULTS §2.3/§2.4 (env-gated)
    if os.environ.get("TORCH_PROFILE"):
        import torch.profiler as _tp
        _out = output_dir or "."
        os.makedirs(_out, exist_ok=True)
        _reset_grads()
        with _tp.profile(
            activities=[_tp.ProfilerActivity.CPU, _tp.ProfilerActivity.CUDA],
            record_shapes=True, with_flops=True, with_stack=False,
            schedule=_tp.schedule(wait=1, warmup=3, active=10, repeat=1),
        ) as _prof:
            for _ in range(15):
                _reset_grads()
                _r = _fwd(); _r.values.backward(grad_output)
                torch.cuda.synchronize(); _prof.step()
        _tr = os.path.join(_out, f"torchprof_{log_layer_type}_d{dim_per_head}.json")
        _prof.export_chrome_trace(_tr); print(f"TORCHPROF_TRACE {_tr}")
        print("===TORCHPROF_TABLE_BEGIN===")
        print(_prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=60))
        print("===TORCHPROF_TABLE_END===")

'''
s=s.replace(anchor, block+anchor, 1)
open(f,"w").write(s); print("patched torch.profiler block into", f)
PYX

# --- run the fused layer (D256 cutlass on H100; D128 cutlass on GB300 blackwell) under torch.profiler ---
echo "### torch.profiler run: fused/cutlass D=$DPH, 8 layers, B32, seqlen $([ $DPH -ge 256 ] && echo 4099 || echo 4096) ###"
SEQ=$([ "$DPH" -ge 256 ] && echo 4099 || echo 4096)
TORCH_PROFILE=1 python3 "$BM" run \
  --iters 30 --warmup-iters 20 --layer-type fused --kernel-backend cutlass \
  --dim-per-head "$DPH" --num-heads 4 --num-layers 8 --dtype bfloat16 \
  --max-seqlen "$SEQ" --batchsize 32 --full-sequence True --output-dir "$OUT" 2>&1 \
  | grep -aE "TORCHPROF|===|aten::|hstu|linear|silu|attn|mm|bmm|addmm|cuda_time|Name|Self CUDA|MFU|peak|Error|Traceback" | tail -80
echo "### trace(s): $(ls -la $OUT/torchprof_*.json 2>/dev/null) ###"
