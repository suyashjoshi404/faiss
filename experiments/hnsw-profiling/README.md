# HNSW profiling

This experiment builds a FAISS HNSW index, then measures query latency, recall,
distance computations, and hops by graph layer for SIFT1M or Enron.

New users should complete [BUILD.md](BUILD.md) before running the commands
below.

For reproducible build and search measurements, follow
[BENCHMARKING.md](BENCHMARKING.md). In particular, keep the thread count fixed;
use a quiet machine, CPU affinity, warm-up, and repeated runs when comparing
latency or build time.

## Requirements

- Run from the FAISS repository root.
- Use Python 3 with `faiss` built from this checkout and `numpy` installed.
- The imported FAISS build must expose per-layer HNSW statistics, including
  `hnsw_stats.get_nhops_at_level` and `hnsw_stats.get_ndis_at_level`.
- Dataset files must be present under `data/sift/` and `data/enron/`.

## Run SIFT1M

Build the index, then profile it:

```bash
python3 experiments/hnsw-profiling/scripts/build_sift_index.py --dataset sift --M 32 --ef-construction 128 --threads 1
python3 experiments/hnsw-profiling/scripts/profile_sift_query.py --dataset sift --M 32 --k 10 --ef-search 64 --threads 1
```

The builder writes `indexes/sift/sift_hnsw_M32.index`. The profiler writes
`results/sift_hnsw_M32_ef64_layer_stats.csv` and prints aggregate timing and
recall statistics.

## Run Enron

```bash
python3 experiments/hnsw-profiling/scripts/build_sift_index.py --dataset enron --M 32 --ef-construction 128 --threads 1
python3 experiments/hnsw-profiling/scripts/profile_sift_query.py --dataset enron --M 32 --k 10 --ef-search 64 --threads 1
```

The builder writes `indexes/enron/enron_hnsw_M32.index`; the profiler writes
`results/enron_hnsw_M32_ef64_layer_stats.csv`.

Change `--ef-search` to profile another search setting. The profiler's defaults
are `k=10`, `ef-search=64`, `threads=1`, and `warmup=500`. The builder defaults
to `M=32`, `ef-construction=128`, and `threads=1`. Use `--index-file` or
`--output` on the profiler to select different paths; use `--data-file` or
`--index-file` on the builder to override its input or output path.
