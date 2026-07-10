#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Arch-agnostic HSTU nsys benchmark entrypoint — recsys-examples PERF_ANALYSIS §2.1–§2.4.
#
# Runs the upstream `exp4_caching_hr` e2e benchmark under Nsight Systems on node0 (which owns rank0),
# joins the torchrun rendezvous on every other node, then emits the RAW nsys artifacts into $OUT. This
# script only CAPTURES — all per-op MFU / exposed / raw-kernsum analysis is done afterward, off-node, by
# the shipped python (exposed_faststep.py, reconstruct_perop_mfu.py, kernsum_categorize.py,
# exposed_gemm_split.py; see README.md). The `.rep`→`.sqlite` export is done here on the (arm) node
# because x86 nsys hangs on arm reps; the sqlite is arch-portable for local analysis.
#
# The SAME script runs on both platforms — everything platform-specific is auto-detected from `uname -m`
# and overridable by env, so an H100 job and a GB300 job share one capture path:
#
#   PLATFORM     gb300 | h100          default: aarch64→gb300, x86_64→h100
#   KV           head_dim              default: gb300 128, h100 256   (Blackwell CUTLASS can't do 256 → Issue #1)
#   CTX          0 | 1  contextual     default: gb300 0,   h100 1     (Blackwell CUTLASS rejects ctx  → Issue #4)
#   JAGGED       0 | 1                 default 0 (= --non_jagged, upstream's method)
#   EXP          experiment name       default exp4_caching_hr
#   MAXSEQ       --max_sequence_length default 2048
#   NG           gpus / node           default $GPUS_PER_NODE (or 8)
#   SRC          source tree           default: first of ./examples,/opt/recsys-2605-examples,/opt/recsys-examples that has hstu/
#   PIP_IDX      pip index             default https://pypi.org/simple/
#   NSYS_URL     optional nsys tarball  fetched only if `nsys` is not already on PATH (e.g. arm builds)
#
# Multi-node topology (set by your launcher / scheduler; single-node defaults shown):
#   NNODES=1  NODE_RANK=0  GPUS_PER_NODE=8  MASTER_ADDR=localhost  MASTER_PORT=29500
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
export HOME=/tmp/trh; mkdir -p "$HOME" /tmp/tp
exec > >(tee /tmp/tp/run.log) 2>&1

# ── node / topology ──
NODE=${NODE_RANK:-0}; NW=${NNODES:-1}; NG=${NG:-${GPUS_PER_NODE:-8}}
ARCH=$(uname -m)
PLATFORM=${PLATFORM:-$([ "$ARCH" = aarch64 ] && echo gb300 || echo h100)}

# ── per-platform defaults (all overridable by env) ──
if [ "$PLATFORM" = gb300 ]; then
  KV=${KV:-128}; CTX=${CTX:-0}
else
  KV=${KV:-256}; CTX=${CTX:-1}
fi
PIP_IDX=${PIP_IDX:-https://pypi.org/simple/}
JAGGED=${JAGGED:-0}; EXP=${EXP:-exp4_caching_hr}; MAXSEQ=${MAXSEQ:-2048}
TAG=$([ "$JAGGED" = 1 ] && echo jag || echo nj)
echo "=== $PLATFORM ($ARCH) ${TAG} kv$KV ctx$CTX exp=$EXP  node=$NODE/$NW gpu=$NG (16 = ${NG}x${NW})  $(date -u +%FT%TZ) $(hostname) ==="
nvidia-smi -L | head -4

# ── deps (retry; overridable index) ──
pipr(){ for t in 1 2 3; do pip install --user --no-warn-script-location "$@" -i "$PIP_IDX" 2>&1|tail -1 && return 0; sleep 4; done; }
pipr pandas rich nvtx tqdm einops matplotlib click
pipr --no-deps gin-config torchmetrics==1.0.3 lightning-utilities dulwich urllib3
export PYTHONPATH="$(python3 -c 'import sys,os;print(os.path.expanduser("~/.local/lib/python%d.%d/site-packages"%sys.version_info[:2]))'):${PYTHONPATH:-}"

# ── writable copy of the source tree ──
SRC=${SRC:-$(for d in ./examples /opt/recsys-2605-examples /opt/recsys-examples; do [ -d "$d/hstu" ] && { echo "$d"; break; }; done)}
[ -n "$SRC" ] || { echo "!! no source tree (set SRC=)"; exit 2; }
EX=$HOME/ex; cp -r "$SRC" "$EX"; cd "$EX/hstu" || { echo "!! no $EX/hstu"; exit 2; }
export PYTHONPATH="$EX:$EX/hstu:/opt/megatron-lm:$PYTHONPATH"; export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# ── GB300 only: register the bf16 dense peak (2500) so MFU is on the right denominator. H100 (989) ships in perf.py. ──
if [ "$PLATFORM" = gb300 ]; then
  python3 - "$EX/commons/utils/perf.py" <<'PPY'
import sys; f=sys.argv[1]; s=open(f).read(); a='_KNOWN_GPU_SPECS: Dict[str, Dict[str, float]] = {'
if "GB300" not in s and a in s:
    s=s.replace(a, a+'\n    "GB300": {"tf32":2500,"fp16":2500,"bf16":2500,"fp8":10000,"int8":307,"fp4":15000},'); open(f,"w").write(s); print("perf+GB300")
PPY
fi

# ── multi-node torchrun rendezvous: rewrite upstream's --standalone → nnodes/node_rank/master ──
RS=training/benchmark/scripts/run_single_experiment_local.sh
sed -i 's|^\([[:space:]]*\)--standalone \\$|\1--nnodes='"$NW"' --node_rank='"$NODE"' --master_addr=${MASTER_ADDR:-localhost} --master_port=${MASTER_PORT:-29500} \\|' "$RS"

# ── nsys binary: x86 nvcr images already ship it; on arm you may need a fetched build.
#    If nsys isn't on PATH and NSYS_URL is set, fetch + extract it (node0 only). ──
if [ "$NODE" = "0" ] && ! command -v nsys >/dev/null 2>&1 && [ -n "${NSYS_URL:-}" ]; then
  for t in 1 2 3; do timeout 300 curl -fsSL "$NSYS_URL" -o /tmp/nsys.tgz 2>&1|tail -2; [ -s /tmp/nsys.tgz ] && break; sleep 5; done
  NSYSDIR=$HOME/nsys; mkdir -p "$NSYSDIR"; tar xzf /tmp/nsys.tgz -C "$NSYSDIR" --exclude='._*' --warning=no-unknown-keyword 2>&1|grep -avE "Ignoring unknown|LIBARCHIVE"|tail -1
  export PATH="$(dirname "$(find "$NSYSDIR" -name nsys -type f 2>/dev/null|head -1)"):$PATH"
fi
if [ "$NODE" = "0" ]; then
  echo "nsys=$(command -v nsys)"; nsys --version 2>&1|head -1
  nsys --version >/dev/null 2>&1 || { echo "!! nsys does not run — put it on PATH or set NSYS_URL — abort"; exit 3; }
fi

# ── experiment args ──
CTXFLAG=$([ "$CTX" = 1 ] && echo "--include-contextual" || echo "")
NJFLAG=$([ "$JAGGED" = 1 ] && echo "" || echo "--non_jagged")
EXA="$CTXFLAG --balanced_shuffler --kernel_backend cutlass --caching --ratio 0.1 --dist_type hash_roundrobin --value_dist zipf --value_dist_alpha 1.05 --kv_channels $KV --max_sequence_length $MAXSEQ $NJFLAG"
OUT=$EX/hstu/training/benchmark/results/s${MAXSEQ}; mkdir -p "$OUT"
set +e

# ── other nodes: join the rendezvous (no nsys) and exit ──
if [ "$NODE" != "0" ]; then
  echo "##### node$NODE: join rendezvous (no nsys) #####"
  bash "$RS" "$EXP" --benchmark-type=e2e --nproc="$NG" --output-dir="$OUT" --exp-args="$EXA" 2>&1 | tail -8
  echo "RC=${PIPESTATUS[0]}  node$NODE done"; exit 0
fi

# ── node0: run under nsys (owns rank0) ──
echo "##### node0: $EXP e2e ${TAG} 16-GPU (${NG}x${NW}) UNDER nsys #####"
bash "$RS" "$EXP" --benchmark-type=e2e --nproc="$NG" --nsys --output-dir="$OUT" --exp-args="$EXA" 2>&1 \
  | grep -aE "achieved FLOPS|MFU|tokens|Running Experiment|nsys|Collecting|Processing|Generated|\.nsys-rep|Error|Traceback|OOM|assert|is_jagged" | tail -50
echo "RC=${PIPESTATUS[0]}"
REP=$(find "$OUT" -name '*.nsys-rep' 2>/dev/null|head -1); echo "REP=$REP"; ls -la "$REP" 2>/dev/null
[ -n "$REP" ] && [ -f "$REP" ] || { echo "!! no .nsys-rep produced"; exit 4; }

# ── ONE capture → all four reports (do not split into separate runs):
#    nvtx_kern_sum = per-op kernel busy-time (§2.3/§2.4)   nvtx_sum = per-range instance times (fastest-step pick)
#    cuda_gpu_kern_sum = category totals (raw kern-sum)     cuda_gpu_trace = per-kernel timeline (§2.2 exposed sweep)
#    sqlite = step-window NVTX + kernel→launch→NVTX map (fastest-step window + exact gemm sub-split). ──
nsys stats --report nvtx_kern_sum     --format csv --output "$OUT/nvtx"     "$REP" 2>&1|tail -1
nsys stats --report nvtx_sum          --format csv --output "$OUT/nvtxsum"  "$REP" 2>&1|tail -1
nsys stats --report cuda_gpu_kern_sum --format csv --output "$OUT/kern"     "$REP" 2>&1|tail -1
nsys stats --report cuda_gpu_trace    --format csv --output "$OUT/gputrace" "$REP" 2>&1|tail -1
nsys export --type sqlite --force-overwrite true --output "$OUT/rep.sqlite" "$REP" 2>&1|tail -1
gzip -f "$OUT/rep.sqlite" 2>&1|tail -1   # rep.sqlite.gz (portable to x86 for exposed_gemm_split.py)
# always keep the raw .nsys-rep (the Nsight Systems GUI timeline) — this is an nsys capture, so the trace is a deliverable
gzip -c "$REP" > "$OUT/$(basename "$REP").gz"; echo "kept .nsys-rep.gz=$(ls -la $OUT/*.nsys-rep.gz)"
ls -la "$OUT"/*.csv "$OUT"/rep.sqlite.gz 2>/dev/null
set -e

# ── raw artifacts (run.log, four CSVs, rep.sqlite.gz, .nsys-rep.gz) are now in $OUT.
#    Copy them off-node and run the Layer-3 analysis scripts on them (see README). ──
echo "ARTIFACTS in $OUT :"; ls -1 "$OUT"/*.csv "$OUT"/rep.sqlite.gz "$OUT"/*.nsys-rep.gz /tmp/tp/run.log 2>/dev/null
echo "NSYS_BENCHMARK_DONE ($PLATFORM $TAG)"
