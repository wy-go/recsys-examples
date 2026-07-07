#!/usr/bin/env python3
# ctxfix — TritonHSTUAttention asserts num_contextuals is a Python int, but real-data
# loaders pass a per-sequence torch.Tensor. When the contextual length is uniform across
# the batch (the real-data case), collapse it to that int so the triton path accepts it.
# Usage: python3 ctxfix.py <path-to>/examples/hstu/modules/hstu_attention.py
import sys

f = sys.argv[1]
s = open(f).read()
old = '''        if num_contextuals is None:
            num_contextuals = 0
        assert isinstance(
            num_contextuals, int
        ), "num_contextuals must be an integer in TritonHSTUAttention"'''
new = '''        if num_contextuals is None:
            num_contextuals = 0
        if isinstance(num_contextuals, torch.Tensor):
            assert int(num_contextuals.min().item()) == int(num_contextuals.max().item()), "uniform contextual length required"
            num_contextuals = int(num_contextuals.max().item())
        assert isinstance(
            num_contextuals, int
        ), "num_contextuals must be an integer in TritonHSTUAttention"'''
if old in s:
    open(f, "w").write(s.replace(old, new))
    print("ctxfix: applied")
else:
    print("ctxfix: skipped (pattern not found — already patched or upstream changed)")
