#!/usr/bin/env bash
# run_benchmarks.sh — reproduce the HSTU v26.05 H100-vs-GB300 benchmark INSIDE a built image
# (docker/Dockerfile.recsys.v2605 on x86/H100, docker/Dockerfile.gb300.recsys.v2605 on arm/GB300).
# No bridge / eu_launch / dulwich — plain torchrun + the upstream benchmark scripts + our patches.
#
#   EX        = path to recsys-examples/examples (in-image or vendored)
#   RUNTIME   = path to this runtime/ dir
# Usage:
#   ./run_benchmarks.sh setup                       # shims + patches (run once)
#   ./run_benchmarks.sh train  <dataset> <gin> <backend> [nproc]   # real-data e2e training
#   ./run_benchmarks.sh attn   <backend> <kv>       # benchmark type 1: attention kernel fwd/bwd
#   ./run_benchmarks.sh layer  <ltype> <backend> <dim>   # benchmark type 2: HSTU layer fwd/bwd
#   ./run_benchmarks.sh e2e    [nproc]              # benchmark type 3: end-to-end exp0..5
set -uo pipefail
RUNTIME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${EX:=/opt/recsys-examples/examples}"          # adjust to your in-image layout
export HOME="${HOME:-/tmp/trh}"; mkdir -p "$HOME"

setup_runtime() {
  # numpy-2.0 / register_fake shims (sitecustomize) + the .pth that beats byted-wandb's auto-import.
  export PYTHONPATH="$RUNTIME:${PYTHONPATH:-}"
  SP="$(python3 -c 'import site; print(site.getusersitepackages())')"; mkdir -p "$SP"
  cp -f "$RUNTIME/00_npshim.pth" "$SP/00_npshim.pth"
  cp -f "$RUNTIME/wandb_hook.py" "$EX/hstu/training/wandb_hook.py" 2>/dev/null || true
  # byted-wandb tracking ON for every run (set TK_HOST to your region: EU=ml-tracking.tiktoke.org).
  export WANDB_PROJECT="${WANDB_PROJECT:-recsys-hstu-gb300}" WANDB_MODE="${WANDB_MODE:-online}" WANDB_DISABLE_STATS=1
  export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
  python3 -c "import numpy as n; print('npshim OK np.float_=', n.float_)"
}

cmd="${1:?see header for usage}"; shift || true
case "$cmd" in
  setup)
    setup_runtime
    bash "$RUNTIME/patches.sh" "$EX" ${1:+--debug-layer}
    ;;
  train)   # real-data training (kuairand-27k/pure, movielen). For movielen use backend=pytorch.
    DS="${1:?dataset}"; GIN="${2:?gin}"; BACKEND="${3:?backend}"; NPROC="${4:-1}"
    setup_runtime
    cd "$EX/hstu"
    G="training/configs/$GIN"
    printf '\nNetworkArgs.kernel_backend = %s\nTrainerArgs.pipeline_type = %s\n' "'$BACKEND'" "'none'" >> "$G"
    torchrun --standalone --nproc_per_node "$NPROC" training/wandb_hook.py --gin-config-file "$G"
    ;;
  attn)    # benchmark type 1 — attention kernel fwd+bwd, FULL upstream batch x seqlen grid (CUTLASS kv<=128, TRITON 128/256)
    BACKEND="${1:?backend}"; KV="${2:?kv}"
    # default = upstream 8x8 grid; override args 3/4 to extend (GB300 capacity headroom), e.g. attn triton 256 1,2,...,256 128,...,32768
    BS="${3:-1,2,4,8,16,32,64,128}"; SL="${4:-128,256,512,1024,2048,4096,8192,16384}"
    setup_runtime
    cd "$EX/hstu"
    G="/tmp/attn_${BACKEND}_${KV}.gin"; cp training/configs/benchmark_ranking.gin "$G"
    printf '\nNetworkArgs.kernel_backend = %s\nNetworkArgs.kv_channels = %s\n' "'$BACKEND'" "$KV" >> "$G"
    python3 training/benchmark/scripts/hstu_attn_kernel_benchmark.py --gin-config-file "$G" \
      --batch-sizes "$BS" --seqlens "$SL" --warmup-iters 10 --bench-iters 50 --output-dir "/tmp/attn_out"
    ;;
  layer)   # benchmark type 2 — HSTU layer fwd/bwd (fused/debug; no TE-native; cutlass<=128, triton-256)
    LT="${1:?ltype}"; BACKEND="${2:?backend}"; DIM="${3:?dim}"
    setup_runtime
    cd "$EX/hstu"
    python3 training/benchmark/scripts/hstu_layer_benchmark.py run \
      --iters 50 --warmup-iters 20 --layer-type "$LT" --kernel-backend "$BACKEND" \
      --dim-per-head "$DIM" --num-heads 4 --num-layers 1 --max-seqlen 4096 --batchsize 32 \
      --full-sequence True --dtype bfloat16 --dump-memory-snapshot False
    ;;
  e2e)     # benchmark type 3 — end-to-end exp0..5 (synthetic zipf). NOTE: exp2+ use cutlass; on GB300
    NPROC="${1:-1}"   # the e2e gin must be kv<=128 for cutlass (256 raises on Blackwell).
    setup_runtime
    cd "$EX/hstu"
    ./training/benchmark/scripts/run_all_experiments_local.sh \
      --exp-file=training/benchmark/experiments.txt --nproc="$NPROC"
    ;;
  *) echo "unknown command: $cmd"; exit 2 ;;
esac
echo "run_benchmarks.sh: $cmd done"
