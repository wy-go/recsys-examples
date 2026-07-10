#!/usr/bin/env python3
"""Full exp-ladder stats table (avg + peak), reproducing upstream E2E_BENCHMARK's columns.

Upstream's own `analyze_results.py` reports only the PEAK (max over iters) FLOPS/MFU. Its
published E2E_BENCHMARK table, however, has both averages and peaks:

    Exp | Name | Avg TFLOPS/GPU | Avg MFU (%) | Peak TFLOPS/GPU | Peak MFU (%) | Speedup vs Baseline | Notes

This script reconstructs that full table from the same per-iter log lines
("achieved FLOPS <x> TFLOPS, MFU <y>%") that analyze_results.py parses, adding the mean.

The logged FLOPS is the GLOBAL (all-rank) throughput and MFU is the per-GPU utilization
(FLOPS_global / (world_size * peak_per_gpu)); so per-GPU TFLOPS = global / world_size, while
MFU needs no rescaling. world_size is auto-detected from the log ("nRanks N" / "world_size N"),
overridable with --world-size. Speedup is on avg per-GPU TFLOPS vs the first (baseline) rung.

Usage:
    python e2e_stats_table.py <results_dir> [--world-size 16] [--skip-warmup 1] [--md out.md]
"""
import argparse
import os
import re
import sys
from pathlib import Path

FLOPS_RE = re.compile(r"achieved FLOPS\s+([\d.]+)\s+TFLOPS")
MFU_RE = re.compile(r"MFU\s+([\d.]+)%")
WS_RE = re.compile(r"(?:nRanks|world_size)[ =:]+(\d+)")


def extract_series(log_path):
    """Return (flops_list, mfu_list, world_size_or_None) for one exp log."""
    flops, mfu, ws = [], [], None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            fm, mm = FLOPS_RE.search(line), MFU_RE.search(line)
            if fm and mm:
                flops.append(float(fm.group(1)))
                mfu.append(float(mm.group(1)))
            wm = WS_RE.search(line)
            if wm:
                ws = max(ws or 0, int(wm.group(1)))
    return flops, mfu, ws


def find_logs(results_dir):
    """{exp_name: log_path}, one log per exp subdir, sorted by name (rung order)."""
    out = {}
    for d in sorted(Path(results_dir).iterdir()):
        if not d.is_dir():
            continue
        logs = sorted(d.glob("*.log"))
        if logs:
            out[d.name] = str(logs[0])
    return out


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir")
    ap.add_argument("--world-size", type=int, default=None,
                    help="GPUs (per-GPU = global/world_size). Default: auto-detect, else 16.")
    ap.add_argument("--skip-warmup", type=int, default=1,
                    help="Drop the first N logged points per exp (warmup). Default 1.")
    ap.add_argument("--md", default=None, help="Also write a Markdown table to this path.")
    args = ap.parse_args()

    logs = find_logs(args.results_dir)
    if not logs:
        print(f"No exp logs under {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    rows, baseline_avg_tf = [], None
    detected_ws = None
    for name, path in logs.items():
        flops, mfu, ws = extract_series(path)
        detected_ws = detected_ws or ws
        if not flops:
            print(f"  {name:22s} | no per-iter FLOPS/MFU lines", file=sys.stderr)
            continue
        if args.skip_warmup and len(flops) > args.skip_warmup:
            flops, mfu = flops[args.skip_warmup:], mfu[args.skip_warmup:]
        world = args.world_size or ws or detected_ws or 16
        avg_tf, peak_tf = mean(flops) / world, max(flops) / world
        avg_mfu, peak_mfu = mean(mfu), max(mfu)
        if baseline_avg_tf is None:
            baseline_avg_tf = avg_tf
        rows.append({
            "name": name, "n": len(flops), "world": world,
            "avg_tf": avg_tf, "peak_tf": peak_tf,
            "avg_mfu": avg_mfu, "peak_mfu": peak_mfu,
            "speedup": avg_tf / baseline_avg_tf if baseline_avg_tf else float("nan"),
        })

    ws_note = args.world_size or detected_ws or 16
    hdr = ("Exp", "Avg TF/GPU", "Avg MFU%", "Peak TF/GPU", "Peak MFU%", "Speedup", "n")
    print(f"\nE2E ladder stats  (world_size={ws_note}, "
          f"per-GPU = global/{ws_note}, skip_warmup={args.skip_warmup})")
    print("-" * 82)
    print("{:<22} {:>11} {:>9} {:>12} {:>10} {:>8} {:>4}".format(*hdr))
    print("-" * 82)
    for r in rows:
        print("{name:<22} {avg_tf:>11.1f} {avg_mfu:>8.2f}% {peak_tf:>12.1f} "
              "{peak_mfu:>9.2f}% {speedup:>7.2f}x {n:>4}".format(**r))
    print("-" * 82)

    if args.md:
        lines = ["| Exp | Avg TFLOPS/GPU | Avg MFU (%) | Peak TFLOPS/GPU | Peak MFU (%) | Speedup vs Baseline |",
                 "|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(f"| {r['name']} | {r['avg_tf']:.1f} | {r['avg_mfu']:.2f} | "
                         f"{r['peak_tf']:.1f} | {r['peak_mfu']:.2f} | {r['speedup']:.2f}x |")
        Path(args.md).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {args.md}")


if __name__ == "__main__":
    main()
