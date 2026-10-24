# SHG profiling

This experiment ports [SHG-Index](https://github.com/gzyhkust/SHG-Index) at
commit `adadbc1a348ba94dbbb442308c964fbe816f11a7` to the SIFT1M and Enron
datasets used by the HNSW profiler. Run commands from the FAISS repository root.

## Build the runner

```bash
cmake -S experiments/hnsw-profiling/shg -B build/shg-profile -DCMAKE_BUILD_TYPE=Release
cmake --build build/shg-profile -j2
```

Rebuild the runner after updating the SHG C++ source; the Python scripts use
the executable at `build/shg-profile/shg_profile`.

## Build and search

```bash
python3 experiments/hnsw-profiling/scripts/build_shg_index.py --dataset sift --M 32 --ef-construction 128
python3 experiments/hnsw-profiling/scripts/profile_shg_query.py --dataset sift --M 32 --k 10 --ef-search 64
```

Use `--dataset enron` for Enron. The build script saves
`indexes/<dataset>/<dataset>_shg_M32.index` and a companion `.index.shg` file.
Both are required for search. An existing index requires `--force` to rebuild.
Search loads the saved index without rebuilding its graph or shortcuts.

The search script writes `results/<dataset>_shg_M32_ef64_layer_stats.csv`.
Both scripts append timings to `observations/<dataset>_shg_performance.csv`;
search also appends per-layer averages to
`observations/<dataset>_shg_layer_averages.csv`. Build and shortcut times
exclude saving; search time excludes warmup, recall, statistics collection, and
CSV output. SHG runs on one thread.

Use `--index-file` to select another index path, `--performance-output` to
select another timing CSV, or `--executable` if the runner was built elsewhere.
The build script accepts `--data-file`; search accepts `--query-file`,
`--groundtruth-file`, `--output`, `--layer-averages-output`, and `--warmup`.

The vendored SHG and PGM code is under `vendor/`; its licenses are preserved
there.
