# Build and prerequisites

This experiment requires the FAISS Python package built from this checkout.
Installing a released `faiss-cpu` package will not provide the custom per-layer
HNSW statistics used by the profiler.

Run every command below from the FAISS repository root.

## 1. Install system prerequisites

The build requires Python 3.10 or newer, a C++20 compiler, CMake 3.24 or newer,
SWIG, and a BLAS implementation.

On Ubuntu or Debian:

```bash
sudo apt update
sudo apt install build-essential libopenblas-dev python3-dev python3-pip python3-venv
```

Confirm that Python is new enough:

```bash
python3 --version
```

The Python build process installs its compatible SWIG build dependency
automatically.

## 2. Create an isolated environment

```bash
python3 -m venv faiss
source faiss/bin/activate
python -m pip install --upgrade pip 'cmake>=3.24'
```

## 3. Build and install this checkout

```bash
python -m pip install .
```

The project configuration builds a CPU-only release package with Python
bindings. Re-run the command with `--force-reinstall` after changing FAISS C++
sources:

```bash
python -m pip install --force-reinstall --no-deps .
```

## 4. Verify the custom build

```bash
python - <<'PY'
import faiss

stats = faiss.cvar.hnsw_stats
assert hasattr(stats, "get_nhops_at_level")
assert hasattr(stats, "get_ndis_at_level")
print("FAISS module:", faiss.__file__)
print("Per-layer HNSW statistics: available")
PY
```

Both assertions must pass. The printed module path should point inside
`faiss`, which confirms that another FAISS installation is not
being imported.

## 5. Check the datasets

The experiment expects this layout:

```text
experiments/hnsw-profiling/data/
├── enron/
│   ├── enron.data_new
│   ├── enron_query.fvecs
│   └── enron_groundtruth.ivecs
└── sift/
    ├── sift_base.fvecs
    ├── sift_query.fvecs
    ├── sift_learn.fvecs
    └── sift_groundtruth.ivecs
```

The SIFT1M build uses `sift_base.fvecs`; profiling uses `sift_query.fvecs` and
`sift_groundtruth.ivecs`. `sift_learn.fvecs` belongs to the standard SIFT1M
dataset but is not used by these HNSW scripts.

Continue with [README.md](README.md) to build an index and run the profiler.
