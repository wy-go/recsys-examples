"""byted-wandb tracking hook for recsys-examples HSTU — ADDITIVE, no upstream edits.

Dropped into examples/hstu/training/ at container runtime and run instead of
pretrain_gr_ranking.py. It monkeypatches `trainer.training.print_rank_0` (the single
metric-reporting funnel, present & same-shape in v26.01 AND v26.04) to forward the
[train]/[eval] lines to byted-wandb, then calls the real pretrain main(). Nothing in
upstream is modified, so it survives a v26.04 migration / git pull.

Run:  torchrun --standalone --nproc_per_node N training/wandb_hook.py --gin-config-file <gin>
Env:  WANDB_PROJECT, WANDB_NAME, WANDB_MODE(online|offline|disabled), WANDB_RUN_ID(+resume), TK_HOST,
      WANDB_DISABLE_STATS=1 (turn off the automatic wandb `system/*` GPU/CPU/runtime metrics)
"""
# --- NumPy 2.0 compat: byted-wandb 0.13.x references removed aliases (np.float_, np.NaN, ...).
#     Re-add them BEFORE wandb is imported, else `import wandb` raises AttributeError on NumPy>=2.0
#     (the v26.05 image ships NumPy 2.x). Runs at module load -> before _init_wandb's `import wandb`. ---
import numpy as _np
for _o, _n in [("float_", "float64"), ("complex_", "complex128"), ("unicode_", "str_"),
               ("string_", "bytes_"), ("bool8", "bool_"), ("object0", "object_"),
               ("int0", "intp"), ("NaN", "nan"), ("Inf", "inf"), ("infty", "inf")]:
    try:
        if not hasattr(_np, _o) and hasattr(_np, _n):
            setattr(_np, _o, getattr(_np, _n))
    except Exception:
        pass

import os
import re

RANK = int(os.environ.get("RANK", "0"))
WORLD = int(os.environ.get("WORLD_SIZE", "1"))
_WANDB = False  # set True iff init succeeded on the logging rank


def _init_wandb():
    global _WANDB
    if RANK != 0:  # only global rank 0 owns the single run
        return
    try:
        import wandb
        run_id = os.environ.get("WANDB_RUN_ID") or None
        # Settings built defensively: console="off" always; _disable_stats only when asked.
        # If the kwarg is unknown in this byted-wandb version, fall back (must NOT break init).
        try:
            _skw = {"console": "off"}  # avoid console-capture / dropped points
            if os.environ.get("WANDB_DISABLE_STATS", "0") == "1":
                _skw["_disable_stats"] = True  # turn OFF the automatic system/* metrics
            settings = wandb.Settings(**_skw)
        except Exception:
            settings = wandb.Settings(console="off")
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "recsys-hstu"),
            name=os.environ.get("WANDB_NAME", f"world{WORLD}"),
            mode=os.environ.get("WANDB_MODE", "online"),
            id=run_id,
            resume=("allow" if run_id else None),
            settings=settings,
        )
        _WANDB = True  # init succeeded -> logging is on; nothing below may turn it off
        print(f"[wandb_hook] tracking ON (rank0): project={os.environ.get('WANDB_PROJECT','recsys-hstu')}", flush=True)
    except Exception as e:  # never let tracking break training
        print(f"[wandb_hook] tracking DISABLED (init failed): {e}", flush=True)
        return
    # custom x-axis (best-effort; glob/step_metric may be unsupported in byted-wandb 0.13.x).
    # We log WITHOUT a positional step anyway, so logging works even if this whole block no-ops.
    try:
        wandb.define_metric("global_step")
        for _m in ("train/loss", "train/tokens", "train/tflops", "eval/auc"):
            wandb.define_metric(_m, step_metric="global_step")
    except Exception as e:
        print(f"[wandb_hook] define_metric skipped (non-fatal): {e}", flush=True)


_TRAIN = re.compile(r"\[train\].*?iter\s+(\d+).*?loss\s+([0-9.eE+-]+)", re.S)
_TOK = re.compile(r"tokens\s+(\d+)")
_TFLOPS = re.compile(r"([0-9.]+)\s*TFLOPS")
_AUC = re.compile(r"auc[^0-9]*([0-9]*\.[0-9]+)", re.I)
_last_step = {"v": 0}


def _patch_logger():
    try:
        import wandb
        import trainer.training as T
    except Exception as e:
        print(f"[wandb_hook] patch skipped: {e}", flush=True)
        return
    _orig = T.print_rank_0

    def _patched(message, *a, **k):
        if _WANDB:
            try:
                s = str(message)
                mt = _TRAIN.search(s)
                if mt:
                    step = int(mt.group(1))
                    _last_step["v"] = step
                    d = {"train/loss": float(mt.group(2)), "global_step": step}
                    tok = _TOK.search(s)
                    if tok:
                        d["train/tokens"] = int(tok.group(1))
                    tf = _TFLOPS.search(s)
                    if tf:
                        d["train/tflops"] = float(tf.group(1))
                    wandb.log(d)  # no positional step -> no monotonic-drop risk
                elif "[eval]" in s:
                    aucs = [float(x) for x in _AUC.findall(s)]
                    if aucs:
                        # eval/auc stays = first AUC (task0) for continuity with prior
                        # runs/dashboards & best_auc. For multi-task (num_tasks>1) configs
                        # also log each head as eval/auc/taskN. Single log call -> one
                        # history row per eval (same as before), keyed to global_step.
                        d = {"eval/auc": aucs[0], "global_step": _last_step["v"]}
                        if len(aucs) > 1:
                            for i, v in enumerate(aucs):
                                d[f"eval/auc/task{i}"] = v
                        wandb.log(d)
                        best = wandb.summary.get("best_auc", 0.0) or 0.0
                        if aucs[0] > best:
                            wandb.summary["best_auc"] = aucs[0]
            except Exception:
                pass  # parsing/logging must never crash training
        return _orig(message, *a, **k)

    T.print_rank_0 = _patched
    print("[wandb_hook] print_rank_0 patched -> forwarding [train]/[eval] to wandb", flush=True)


def _finish_wandb():
    if _WANDB:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    import atexit
    _init_wandb()
    atexit.register(_finish_wandb)  # also fires on preemption/SIGTERM-driven exit
    _patch_logger()
    import pretrain_gr_ranking  # same dir; imports trainer.training (already patched)
    try:
        pretrain_gr_ranking.main()
    finally:
        _finish_wandb()
