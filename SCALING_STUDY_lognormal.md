# GB300 Sparse-Embedding Scaling Study — capacity × throughput frontier (lognormal)

Companion to [`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md) (§6a e2e GPU-count ladder, §8 seqlen-dist perf-study, §9 1B-model
scale-up). This study maps the **HBM-capacity feasibility frontier and throughput** of the HSTU sparse-embedding table as
four axes are swept together — **rows** (table size) × **ratio** (HBM-cache fraction) × **GPUs** × **batch** — under a
**realistic (lognormal) sequence-length distribution** (mean 2000, std 1000), the §8-recommended stand-in for production
traffic. (A conservative-default **zipf** companion — short-sequence-dominated, which §8 shows *understates* GB300
utilization — carries the identical grid for reference.)

**Config (all 42 runs):** GB300 (288 GB HBM, 2500 TFLOPS bf16 dense peak/GPU), non-contextual, **kv128 CUTLASS**, adam,
`--max_sequence_length 4096`, **lognormal** sequence-length distribution (mean 2000 / std 1000), 4-GPU workers on one NVL72.
Throughput = median achieved TFLOPS over the steady window (cold/warmup iters dropped); MFU = global/(N×2500) (matches the
trainer's own MFU field). Peak HBM = runtime `nvidia-smi` max on FIT runs. Tabulated by
[`runtime/scaleup/tabulate_scaleup.py`](runtime/scaleup/tabulate_scaleup.py); grid from `runtime/scaleup/feasibility.py`,
launched via `SEQDIST=lognormal runtime/scaleup/launch_scaleup.sh`.

---

## 1. Feasibility frontier — governed by **rows/GPU**, and **seqlen-distribution-independent**

**21 FIT / 21 OOM** — **identical, cell-for-cell, to the zipf grid**: no config changes fit/OOM status between the two
distributions. Feasibility is a clean step function of the per-GPU vocab shard (`rows ÷ GPUs`):

| rows/GPU | outcome |
|---|---|
| **125 M** | **FIT** — all 21 runs |
| **250 M** | **OOM** — all 11 runs |
| 500 M / 1 B / 2 B | OOM — all 10 runs |

A table fits iff **rows/GPU ≤ ~125M** (192 GB/GPU under adam); 250M/GPU → 384 GB > 288 GB → OOM. You buy capacity by
**adding GPUs** (1B fits at **8**, 2B at **16**, 4B at **32**, 8B needs **>32**) — **not** by lowering the ratio:

> **`item_vocab_gpu_capacity_ratio` does NOT move the feasibility frontier.** OOM configs requested low ratios (0.05–0.25)
> yet still OOM — the DynamicEmb `cuMemCreate` cache-storage reservation is sized by the **full logical shard (rows/N)**,
> independent of the cache ratio. Ratio only sets the resident footprint of runs that already fit. (Same mechanism as §9's
> H100 1B OOM, where offload/ratio didn't help.)

**Why the frontier is seqlen-independent:** capacity is set by the embedding-table shard (rows × 128 × 4 B × 3 for weight +
adam m,v, ÷N), which has nothing to do with sequence length. Lognormal *does* raise **activation** HBM (longer sequences —
e.g. 1B/8-GPU/bs128 hits **275 GB** vs zipf's 208 GB, near the 288 GB ceiling), but **no config crosses from FIT to OOM**.

---

## 2. Throughput — lognormal fills the GPU (MFU up ~2.6×, peak 21.3%)

Under realistic (lognormal) traffic, **every FIT config's MFU rises** vs the short-sequence-dominated zipf default —
**mean +8.2 pp, ~2.6×** (range 1.5–3.5×). The whole-grid MFU band moves from zipf **3.5–14.3%** to lognormal
**10.4–21.3%**. This is a bigger lift than §8's single 1B/16-GPU point (8.4%→12.9%) because the steady-state token count
roughly **triples** (e.g. 1B/8-GPU/bs32: 2.5M → **8.3M** tok/step) — the mean-2000 lognormal sequences pack the GPU far
more than zipf's short-sequence tail.

- **Batch is a lever, but no longer dominant.** Under zipf, bs32→128 roughly *tripled* MFU (launch/comms-starved at bs32);
  under lognormal the same sweep is much flatter — 1B/8: 13.2→19.4→**21.3**, 2B/16: 12.9→18.4→**21.1**, 4B/32:
  10.4→17.7→**20.8** (only ~1.6–2.0×) — because the long sequences already do most of the filling at bs32. The realistic
  regime is far less host/comms-bound, so the marginal value of a large batch shrinks (and bs128 pushes activation HBM to
  ~275 GB, near the 288 GB device limit).
- **Ratio is a throughput no-op** (~10.4–15.1% band at fixed batch, mostly noise), same as zipf.
- **Weak scaling near-flat** — holding rows/GPU=125M, ratio 0.1, bs32: 8-GPU 13.2%, 16-GPU 12.9%, 32-GPU 10.4% MFU (~20%
  per-GPU erosion 8→32, added cross-worker comms), not a cliff.

### zipf → lognormal MFU (21 matched FIT points)

| rows | ratio | GPUs | bs | zipf MFU% | **logn MFU%** | Δ (pp) | logn/zipf |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.05 | 8 | 32 | 5.39 | **12.40** | +7.01 | 2.30× |
| 1 | 0.10 | 8 | 32 | 4.73 | **13.15** | +8.42 | 2.78× |
| 1 | 0.10 | 8 | 64 | 10.18 | **19.35** | +9.17 | 1.90× |
| 1 | 0.10 | 8 | 128 | 14.20 | **21.30** | +7.10 | 1.50× |
| 1 | 0.25 | 8 | 32 | 5.70 | **13.86** | +8.16 | 2.43× |
| 1 | 0.50 | 8 | 32 | 4.65 | **12.19** | +7.54 | 2.62× |
| 1 | 1.00 | 8 | 32 | 4.42 | **15.11** | +10.69 | 3.42× |
| 2 | 0.05 | 16 | 32 | 4.23 | **14.27** | +10.04 | 3.37× |
| 2 | 0.10 | 16 | 32 | 4.29 | **12.93** | +8.64 | 3.01× |
| 2 | 0.10 | 16 | 64 | 7.09 | **18.42** | +11.33 | 2.60× |
| 2 | 0.10 | 16 | 128 | 14.25 | **21.11** | +6.86 | 1.48× |
| 2 | 0.25 | 16 | 32 | 5.58 | **12.59** | +7.01 | 2.26× |
| 2 | 0.50 | 16 | 32 | 4.11 | **10.73** | +6.62 | 2.61× |
| 2 | 1.00 | 16 | 32 | 4.01 | **11.79** | +7.78 | 2.94× |
| 4 | 0.05 | 32 | 32 | 3.65 | **11.68** | +8.03 | 3.20× |
| 4 | 0.10 | 32 | 32 | 3.62 | **10.41** | +6.79 | 2.88× |
| 4 | 0.10 | 32 | 64 | 7.87 | **17.73** | +9.86 | 2.25× |
| 4 | 0.10 | 32 | 128 | 13.42 | **20.82** | +7.40 | 1.55× |
| 4 | 0.25 | 32 | 32 | 4.53 | **12.86** | +8.33 | 2.84× |
| 4 | 0.50 | 32 | 32 | 3.46 | **12.16** | +8.70 | 3.51× |
| 4 | 1.00 | 32 | 32 | 4.67 | **11.74** | +7.07 | 2.51× |

---

## 3. Master table (42 runs, lognormal)

| rows(B) | ratio | GPUs | bs | fit? | TFLOPS/GPU | MFU% | peak HBM/GPU (GB) | step ms | tok/step |
|---:|---:|---:|---:|:--|---:|---:|---:|---:|---:|
| 1 | 0.05 | 4 | 32 | **OOM** | – | – | 151.7 | – | – |
| 1 | 0.05 | 8 | 32 | FIT | 310.0 | 12.40 | 127.0 | 1522.0 | 8,342,033 |
| 1 | 0.10 | 4 | 32 | **OOM** | – | – | 5.8 | – | – |
| 1 | 0.10 | 4 | 64 | **OOM** | – | – | 5.8 | – | – |
| 1 | 0.10 | 4 | 128 | **OOM** | – | – | 5.8 | – | – |
| 1 | 0.10 | 8 | 32 | FIT | 328.8 | 13.15 | 199.5 | 1434.9 | 8,342,033 |
| 1 | 0.10 | 8 | 64 | FIT | 483.7 | 19.35 | 227.4 | 1957.9 | 16,733,123 |
| 1 | 0.10 | 8 | 128 | FIT | 532.5 | 21.30 | 275.3 | 3570.6 | 33,562,624 |
| 1 | 0.25 | 4 | 32 | **OOM** | – | – | 6.7 | – | – |
| 1 | 0.25 | 8 | 32 | FIT | 346.5 | 13.86 | 233.3 | 1361.5 | 8,342,033 |
| 1 | 0.50 | 8 | 32 | FIT | 304.7 | 12.19 | 233.3 | 1548.3 | 8,342,033 |
| 1 | 1.00 | 8 | 32 | FIT | 377.9 | 15.11 | 233.3 | 1248.4 | 8,342,033 |
| 2 | 0.05 | 4 | 32 | **OOM** | – | – | 5.8 | – | – |
| 2 | 0.05 | 16 | 32 | FIT | 356.7 | 14.27 | 112.9 | 1336.3 | 16,817,716 |
| 2 | 0.10 | 4 | 32 | **OOM** | – | – | 9.7 | – | – |
| 2 | 0.10 | 4 | 64 | **OOM** | – | – | 9.7 | – | – |
| 2 | 0.10 | 4 | 128 | **OOM** | – | – | 9.7 | – | – |
| 2 | 0.10 | 16 | 32 | FIT | 323.3 | 12.93 | 185.4 | 1474.5 | 16,817,716 |
| 2 | 0.10 | 16 | 64 | FIT | 460.4 | 18.42 | 200.2 | 2075.5 | 33,696,184 |
| 2 | 0.10 | 16 | 128 | FIT | 527.9 | 21.11 | 222.5 | 3617.2 | 67,337,816 |
| 2 | 0.25 | 4 | 32 | **OOM** | – | – | 11.6 | – | – |
| 2 | 0.25 | 16 | 32 | FIT | 314.7 | 12.59 | 219.1 | 1514.8 | 16,817,716 |
| 2 | 0.50 | 16 | 32 | FIT | 268.3 | 10.73 | 219.1 | 1776.6 | 16,817,716 |
| 2 | 1.00 | 16 | 32 | FIT | 294.8 | 11.79 | 219.1 | 1617.2 | 16,817,716 |
| 4 | 0.05 | 4 | 32 | **OOM** | – | – | 9.7 | – | – |
| 4 | 0.05 | 32 | 32 | FIT | 292.1 | 11.68 | 127.6 | 1628.1 | 33,570,436 |
| 4 | 0.10 | 4 | 32 | **OOM** | – | – | 17.5 | – | – |
| 4 | 0.10 | 4 | 64 | **OOM** | – | – | 17.5 | – | – |
| 4 | 0.10 | 4 | 128 | **OOM** | – | – | 17.5 | – | – |
| 4 | 0.10 | 32 | 32 | FIT | 260.3 | 10.41 | 200.1 | 1827.0 | 33,570,436 |
| 4 | 0.10 | 32 | 64 | FIT | 443.4 | 17.73 | 214.8 | 2149.5 | 67,247,648 |
| 4 | 0.10 | 32 | 128 | FIT | 520.5 | 20.82 | 274.8 | 3667.9 | 134,649,248 |
| 4 | 0.25 | 32 | 32 | FIT | 321.5 | 12.86 | 233.8 | 1478.9 | 33,570,436 |
| 4 | 0.50 | 32 | 32 | FIT | 304.0 | 12.16 | 233.8 | 1564.1 | 33,570,436 |
| 4 | 1.00 | 32 | 32 | FIT | 293.5 | 11.74 | 233.8 | 1619.8 | 33,570,436 |
| 8 | 0.05 | 4 | 32 | **OOM** | – | – | 17.5 | – | – |
| 8 | 0.05 | 32 | 32 | **OOM** | – | – | 152.7 | – | – |
| 8 | 0.10 | 32 | 32 | **OOM** | – | – | 6.7 | – | – |
| 8 | 0.10 | 32 | 64 | **OOM** | – | – | 6.7 | – | – |
| 8 | 0.10 | 32 | 128 | **OOM** | – | – | 6.7 | – | – |
| 8 | 0.25 | 32 | 32 | **OOM** | – | – | 7.7 | – | – |
| 8 | 0.50 | 32 | 32 | **OOM** | – | – | 7.7 | – | – |

All OOMs are `cuMemCreate: out of memory` in `DynamicEmbCache._create_cache_storage`, RC=1, zero iters (250M rows/GPU).
The launcher's trailing "predicted HBM" line is non-monotonic in ratio and does not predict OOM — trust the runtime
`nvidia-smi` max on FIT runs.

---

## 4. Headline

**GB300 sparse-embedding scaling is a rows/GPU capacity problem that a realistic workload runs *efficiently*.** A table fits
iff **rows/GPU ≤ ~125M** (buy capacity with GPUs, not the cache ratio — the VMM allocator ignores it) — a frontier that is
**identical under zipf and lognormal** (capacity is seqlen-independent). Once it fits, **realistic (lognormal) sequence
lengths fill the GPU**: MFU rises ~**2.6× over the zipf default** (mean +8.2 pp, **peak 21.3%**), and because the long
sequences already saturate the pipeline, **batch size stops being the dominant lever** (its ~3× zipf effect shrinks to
~1.6–2×) — the workload is compute/activation-bound rather than launch/comms-starved, with bs128 pushing HBM near the
288 GB ceiling. The short-sequence-dominated **zipf default materially understates GB300 utilization** vs realistic traffic.
