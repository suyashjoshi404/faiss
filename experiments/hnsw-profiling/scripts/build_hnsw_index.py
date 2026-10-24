#!/usr/bin/env python3
"""Build a vanilla FAISS HNSW index for SIFT or Enron."""

import argparse
import csv
import os
import time
from pathlib import Path

import faiss
import numpy as np


EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
DATASETS = {
    "sift": {
        "data": str(EXPERIMENT_DIR / "data/sift/sift_base.fvecs"),
        "index": str(EXPERIMENT_DIR / "indexes/sift/sift_hnsw_M{M}.index"),
    },
    "enron": {
        "data": str(EXPERIMENT_DIR / "data/enron/enron.data_new"),
        "index": str(EXPERIMENT_DIR / "indexes/enron/enron_hnsw_M{M}.index"),
    },
}


def load_data_new(path):
    with open(path, "rb") as handle:
        header = np.fromfile(handle, dtype="<u4", count=3)
        if len(header) != 3:
            raise ValueError(f"Invalid data_new header in {path}")
        element_bytes, count, dimension = map(int, header)
        if element_bytes != 4:
            raise ValueError(
                f"Expected float32 data in {path}, got {element_bytes}-byte elements"
            )
        vectors = np.fromfile(handle, dtype="<f4", count=count * dimension)
    if vectors.size != count * dimension:
        raise ValueError(
            f"Truncated data file: expected {count * dimension} values, "
            f"found {vectors.size}"
        )
    return np.ascontiguousarray(vectors.reshape(count, dimension))


def load_base_vectors(dataset, path):
    if dataset == "sift":
        raw = np.fromfile(path, dtype="<f4")
        if raw.size == 0:
            raise ValueError(f"Empty fvecs file: {path}")
        dimension = int(raw[:1].view("<i4")[0])
        if dimension <= 0 or raw.size % (dimension + 1):
            raise ValueError(f"Invalid fvecs record length in {path}")
        records = raw.reshape(-1, dimension + 1)
        dimensions = records[:, 0].copy().view("<i4")
        if not np.all(dimensions == dimension):
            raise ValueError(f"Inconsistent vector dimensions in {path}")
        return np.ascontiguousarray(records[:, 1:], dtype=np.float32)
    return load_data_new(path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, default="sift")
    parser.add_argument("--M", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=128)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--data-file")
    parser.add_argument("--index-file")
    parser.add_argument("--performance-output")
    parser.add_argument(
        "--force", action="store_true", help="Replace an existing index file"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = DATASETS[args.dataset]
    data_file = args.data_file or config["data"]
    index_file = args.index_file or config["index"].format(M=args.M)

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Dataset not found: {data_file}")
    if os.path.exists(index_file) and not args.force:
        print(f"{index_file} already exists. Use --force to rebuild it.")
        return

    faiss.omp_set_num_threads(args.threads)
    print(f"Loading {args.dataset} base vectors from {data_file}...")
    # Match the SHG runner: build timing covers loading the base vectors,
    # index allocation, and graph insertion. Saving the index is excluded.
    start = time.perf_counter()
    xb = load_base_vectors(args.dataset, data_file)
    count, dimension = xb.shape

    print(
        f"Building vanilla HNSW "
        f"(n={count:,}, d={dimension}, M={args.M}, "
        f"efConstruction={args.ef_construction}, threads={args.threads})..."
    )
    index = faiss.IndexHNSWFlat(dimension, args.M, faiss.METRIC_L2)
    index.hnsw.efConstruction = args.ef_construction
    index.add(xb)
    elapsed = time.perf_counter() - start

    os.makedirs(os.path.dirname(os.path.abspath(index_file)), exist_ok=True)
    faiss.write_index(index, index_file)

    performance_file = args.performance_output or str(
        EXPERIMENT_DIR / "observations" / f"{args.dataset}_hnsw_performance.csv"
    )
    os.makedirs(os.path.dirname(os.path.abspath(performance_file)), exist_ok=True)
    write_header = not os.path.exists(performance_file) or os.path.getsize(performance_file) == 0
    with open(performance_file, "a", newline="") as handle:
        fields = ["dataset", "algorithm", "phase", "count", "dimension", "M",
                  "ef_construction", "ef_search", "k", "threads", "seconds",
                  "average_latency_us", "queries_per_second", "recall_at_k",
                  "shortcut_build_seconds"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow({"dataset": args.dataset, "algorithm": "hnsw", "phase": "build",
                         "count": count, "dimension": dimension, "M": args.M,
                         "ef_construction": args.ef_construction, "threads": args.threads,
                         "seconds": f"{elapsed:.8f}"})

    levels = faiss.vector_to_array(index.hnsw.levels)
    num_layers = int(index.hnsw.max_level) + 1
    print(f"Index built in {elapsed:.2f}s and saved to {index_file}")
    print(f"Build performance recorded in {performance_file}")
    print(f"Entry point: {index.hnsw.entry_point}")
    print(f"Number of layers: {num_layers}")
    for layer in range(num_layers):
        print(f"  layer {layer}: {np.count_nonzero(levels > layer):,} nodes")


if __name__ == "__main__":
    main()
