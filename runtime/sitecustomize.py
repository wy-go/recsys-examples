# sitecustomize.py — runs at every Python startup (when on PYTHONPATH). Two ADDITIVE compat shims:
#  (1) NumPy 2.0: re-add aliases byted-wandb 0.13.x references (np.float_, np.NaN, ...).
#  (2) torch.library.register_fake passthrough: the arm v26.05 image builds HSTU PURE-PYTHON (no compiled
#      fbgemm .so), so the model import chain (ops/fused_hstu_op.py -> hstu.hstu_ops_gpu) tries to
#      register_fake "fbgemm::hstu_varlen_fwd_80" etc. which DON'T EXIST. Those ops are never CALLED in the
#      DEBUG layer (pure torch). We swallow ONLY "does not exist" (re-raise anything else) so the import
#      succeeds. NOTE: the clean fix is to build arm HSTU compiled (HSTU_ARCH_LIST="8.0 9.0 10.0") so the
#      ops exist; this shim is the no-rebuild unblock and matches the validated 0194 path.
import numpy as _np
for _o, _n in [("float_", "float64"), ("complex_", "complex128"), ("unicode_", "str_"),
               ("string_", "bytes_"), ("bool8", "bool_"), ("object0", "object_"),
               ("int0", "intp"), ("NaN", "nan"), ("Inf", "inf"), ("infty", "inf")]:
    try:
        if not hasattr(_np, _o) and hasattr(_np, _n):
            setattr(_np, _o, getattr(_np, _n))
    except Exception:
        pass

try:
    import torch as _t
    _orig_rf = _t.library.register_fake

    def _rf_shim(name, *a, **k):
        try:
            r = _orig_rf(name, *a, **k)
        except RuntimeError as e:
            if "does not exist" in str(e):
                return (lambda fn: fn)   # op absent (pure-Python HSTU) -> no-op decorator
            raise
        if callable(r):
            def _wrap(fn):
                try:
                    return r(fn)
                except RuntimeError as e:
                    if "does not exist" in str(e):
                        return fn
                    raise
            return _wrap
        return r

    _t.library.register_fake = _rf_shim
except Exception:
    pass
