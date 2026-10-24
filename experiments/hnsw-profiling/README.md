# HNSW and SHG profiling

This experiment builds a FAISS HNSW index, then measures query latency, recall,
distance computations, and hops by graph layer for SIFT1M or Enron.

The matching SHG experiment uses its own build and search scripts; see
[shg/README.md](shg/README.md).

For HNSW, complete [BUILD.md](BUILD.md) before running the commands below.

For reproducible build and search measurements, follow
[BENCHMARKING.md](BENCHMARKING.md). In particular, keep the thread count fixed;
use a quiet machine, CPU affinity, warm-up, and repeated runs when comparing
latency or build time.

## HNSW requirements

- Run from the FAISS repository root.
- Use Python 3 with `faiss` built from this checkout and `numpy` installed.
- The imported FAISS build must expose per-layer HNSW statistics, including
  `hnsw_stats.get_nhops_at_level` and `hnsw_stats.get_ndis_at_level`.
- Dataset files must be present under `data/sift/` and `data/enron/`.

## Run SIFT1M

Build the index, then profile it:

```bash
python3 experiments/hnsw-profiling/scripts/build_hnsw_index.py --dataset sift --M 32 --ef-construction 128 --threads 1
python3 experiments/hnsw-profiling/scripts/profile_hnsw_query.py --dataset sift --M 32 --k 10 --ef-search 64 --threads 1
```

The builder writes `indexes/sift/sift_hnsw_M32.index`. The profiler writes
`results/sift_hnsw_M32_ef64_layer_stats.csv` and prints aggregate timing and
recall statistics. Build and search timings are appended to
`observations/sift_hnsw_performance.csv`; per-layer average hops and distance
computations, mean recall@k, and search QPS are appended to
`observations/sift_hnsw_layer_averages.csv`. The performance CSV also includes
search recall@k and QPS.

The HNSW build timer covers loading the base vectors, index allocation, and
graph insertion; writing the index file is excluded. The search timer starts
after warm-up and covers one search pass over all queries. Recall calculation
and per-query layer-stat collection are outside that interval. SHG uses the
same phase boundaries and a monotonic wall clock; its shortcut build is timed
separately. Search is batched in FAISS HNSW and issued one query at a time by
the SHG implementation, so the timing intervals match in scope while retaining
each implementation's search API.

## Run Enron

```bash
python3 experiments/hnsw-profiling/scripts/build_hnsw_index.py --dataset enron --M 32 --ef-construction 128 --threads 1
python3 experiments/hnsw-profiling/scripts/profile_hnsw_query.py --dataset enron --M 32 --k 10 --ef-search 64 --threads 1
```

The builder writes `indexes/enron/enron_hnsw_M32.index`; the profiler writes
`results/enron_hnsw_M32_ef64_layer_stats.csv`. Build and search timings are
appended to `observations/enron_hnsw_performance.csv`.
Per-layer averages are appended to
`observations/enron_hnsw_layer_averages.csv` with recall@k and QPS; the
performance CSV includes both metrics for its search row.

Change `--ef-search` to profile another search setting. The profiler's defaults
are `k=10`, `ef-search=64`, `threads=1`, and `warmup=500`. The builder defaults
to `M=32`, `ef-construction=128`, and `threads=1`. Use `--index-file` or
`--output` on the profiler to select different paths; use `--data-file` or
`--index-file` on the builder to override its input or output path. Use
`--performance-output` on either script to select a different performance CSV.
Use `--layer-averages-output` on the query profiler to select a different
per-layer averages CSV. Both observation files append so repeated runs remain
available.

## Run SHG

Compile the SHG runner as described in [shg/README.md](shg/README.md), then
use the corresponding scripts:

```bash
python3 experiments/hnsw-profiling/scripts/build_shg_index.py --dataset sift --M 32 --ef-construction 128
python3 experiments/hnsw-profiling/scripts/profile_shg_query.py --dataset sift --M 32 --k 10 --ef-search 64
```

Use `--dataset enron` for Enron. SHG writes an `.index` graph file and a
companion `.index.shg` file containing the search state. Keep both files for
later searches.
