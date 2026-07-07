#!/usr/bin/env bash
# patches.sh — apply the GB300/v26.05 source patches needed to TRAIN HSTU on Blackwell (sm_103).
# These are minimal, surgical edits to the vendored upstream source; each is idempotent / no-op if
# the pattern is gone. Run once against the examples/ tree before launching any benchmark.
#
# Usage:  ./patches.sh <EX>            where <EX> = path to recsys-examples/examples
#         ./patches.sh <EX> --debug-layer   also force HSTULayerType -> DEBUG (real-data triton path)
set -uo pipefail
EX="${1:?usage: patches.sh <recsys-examples/examples> [--debug-layer]}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# (1) perf.py: register a GB300 entry in _KNOWN_GPU_SPECS so step-time logging doesn't crash on sm_103.
#     NOTE: these peak numbers are PLACEHOLDERS — TFLOPS/MFU derived from them are DIAGNOSTIC ONLY,
#     never a headline hardware figure (the FLOP count itself is modeled: HSTU bwd = fwd*2.5).
python3 - "$EX/commons/utils/perf.py" <<'PYX'
import sys
f=sys.argv[1]; s=open(f).read()
a="_KNOWN_GPU_SPECS: Dict[str, Dict[str, float]] = {"
if "GB300" not in s and a in s:
    s=s.replace(a, a+chr(10)+'    "GB300": {"fp64":1300,"fp32":80,"tf32":2500,"fp16":2500,"bf16":2500,"fp8":10000,"int8":307,"fp4":15000},  # GB300 bf16 peak = 2500 (GB300 NVL72 Quick Start Guide)')
    open(f,"w").write(s); print("perf.py: +GB300 (bf16=2500)")
elif '"GB300"' in s and '"bf16": 5000' in s.replace(" ",""):
    s=s.replace('"bf16":5000','"bf16":2500').replace('"bf16": 5000','"bf16": 2500'); open(f,"w").write(s); print("perf.py: GB300 bf16 5000->2500")
else: print("perf.py: GB300 already 2500 / skipped")
PYX

# (2) hstu_attention.py: accept a uniform per-sequence contextual-length tensor on the triton path.
python3 "$HERE/ctxfix.py" "$EX/hstu/modules/hstu_attention.py"

# (3) dmp_to_tp.py: label all-gather uses keyed_jagged_tensor_allgather (gatherv path mismatches dims on TP).
sed -i "s/output_batch.labels = gatherv_along_first_dim(batch.labels, tp_pg)/output_batch.labels = keyed_jagged_tensor_allgather(batch.labels, tp_pg)/" \
  "$EX/commons/distributed/dmp_to_tp.py" 2>/dev/null && echo "dmp_to_tp.py: patched" || echo "dmp_to_tp.py: skipped"

# (4) hstu_sequence_dataset.py: floor (not ceil) #batches so the last partial batch is dropped (drop_last).
sed -i "s#math.ceil(self._num_samples / self._global_batch_size)#math.floor(self._num_samples / self._global_batch_size)#" \
  "$EX/commons/datasets/hstu_sequence_dataset.py" 2>/dev/null && echo "hstu_sequence_dataset.py: floor" || echo "hstu_sequence_dataset.py: skipped"

# (5) optional: force every HSTU layer to the DEBUG reference (plain torch + triton attention).
#     Use for the real-data triton path; OMIT to benchmark the FUSED/CUTLASS Blackwell kernel.
if [ "${2:-}" = "--debug-layer" ]; then
  sed -i "s/layer_type = HSTULayerType.FUSED/layer_type = HSTULayerType.DEBUG/g; s/layer_type = HSTULayerType.NATIVE/layer_type = HSTULayerType.DEBUG/g" \
    "$EX/hstu/training/trainer/utils.py" && echo "trainer/utils.py: layer_type -> DEBUG"
fi
# (6) py3.11: hstu_attn_kernel_benchmark.py:238 puts a backslash inside an f-string (py3.12-only syntax).
#     Our GB300 base is py3.11 (reused arm FBGEMM base) -> backslash->slash in that table header.
python3 - "$EX/hstu/training/benchmark/scripts/hstu_attn_kernel_benchmark.py" <<'PYK'
import sys
f=sys.argv[1]; s=open(f).read()
s2=s.replace("'BS \\\\ SeqLen'","'BS / SeqLen'")
if s!=s2: open(f,"w").write(s2); print("attn-bench: py3.11 f-string fixed")
else: print("attn-bench: skipped")
PYK
# (7) attn-kernel benchmark: its per-config loop skips OOM configs (marks them, breaks), but the Blackwell
#     BWD kernel raises OverflowError (CuTe int32 memref desc, >=2^31 elems) which escapes -> whole sweep dies.
#     Teach the skip-handler to also catch OverflowError, labeled INT32OVF (distinct from OOM). Default grid stays.
python3 - "$EX/hstu/training/benchmark/scripts/hstu_attn_kernel_benchmark.py" <<'PYO'
import sys
f=sys.argv[1]; s=open(f).read()
s=s.replace("            except torch.cuda.OutOfMemoryError:",
            "            except (torch.cuda.OutOfMemoryError, OverflowError) as _e:\n                _mk = \"OOM\" if isinstance(_e, torch.cuda.OutOfMemoryError) else \"INT32OVF\"")
s=s.replace("                    f\"{'OOM':>9} {'---':>12} {'---':>9} | \"\n                    f\"{'OOM':>9} {'---':>12} {'---':>9} | \"\n                    f\"{'OOM':>9} {'---':>12} {'---':>9}\"",
            "                    f\"{_mk:>9} {'---':>12} {'---':>9} | \"\n                    f\"{_mk:>9} {'---':>12} {'---':>9} | \"\n                    f\"{_mk:>9} {'---':>12} {'---':>9}\"")
open(f,"w").write(s); print("attn-bench: OverflowError skip-handler added")
PYO
echo "patches.sh: done"
