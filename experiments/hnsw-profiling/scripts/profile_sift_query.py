#!/usr/bin/env python3
"""Profile vanilla FAISS HNSW hops and distance computations by layer."""

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
        "query": str(EXPERIMENT_DIR / "data/sift/sift_query.fvecs"),
        "groundtruth": str(EXPERIMENT_DIR / "data/sift/sift_groundtruth.ivecs"),
        "index": str(EXPERIMENT_DIR / "indexes/sift/sift_hnsw_M{M}.index"),
    },
    "enron": {
        "query": str(EXPERIMENT_DIR / "data/enron/enron_query.fvecs"),
        "groundtruth": str(EXPERIMENT_DIR / "data/enron/enron_groundtruth.ivecs"),
        "index": str(EXPERIMENT_DIR / "indexes/enron/enron_hnsw_M{M}.index"),
    },
}


def load_fvecs(path):
    raw = np.fromfile(path, dtype="<f4")
    if raw.size == 0:
        raise ValueError(f"Empty fvecs file: {path}")
    dimension = int(raw[:1].view("<i4")[0])
    if raw.size % (dimension + 1):
        raise ValueError(f"Invalid fvecs record length in {path}")
    records = raw.reshape(-1, dimension + 1)
    dimensions = records[:, 0].copy().view("<i4")
    if not np.all(dimensions == dimension):
        raise ValueError(f"Inconsistent vector dimensions in {path}")
    return np.ascontiguousarray(records[:, 1:], dtype=np.float32)


def load_ivecs(path):
    raw = np.fromfile(path, dtype="<i4")
    if raw.size == 0:
        raise ValueError(f"Empty ivecs file: {path}")
    width = int(raw[0])
    if raw.size % (width + 1):
        raise ValueError(f"Invalid ivecs record length in {path}")
    records = raw.reshape(-1, width + 1)
    if not np.all(records[:, 0] == width):
        raise ValueError(f"Inconsistent record widths in {path}")
    return np.ascontiguousarray(records[:, 1:])


def load_queries_and_groundtruth(dataset, config):
    if dataset == "sift":
        return load_fvecs(config["query"]), load_ivecs(config["groundtruth"])
    return load_fvecs(config["query"]), load_ivecs(config["groundtruth"])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, default="sift")
    parser.add_argument("--M", type=int, default=32)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=64)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--index-file")
    parser.add_argument("--output")
    parser.add_argument("--warmup", type=int, default=500)
    return parser.parse_args()


def recall_at_k(labels, groundtruth, k):
    labels = np.asarray(labels).reshape(-1)
    groundtruth = np.asarray(groundtruth).reshape(-1)
    truth_k = min(k, groundtruth.size)
    if truth_k == 0:
        raise ValueError("Cannot compute recall with empty ground truth")
    return len(set(labels[:truth_k]) & set(groundtruth[:truth_k])) / truth_k


def main():
    args = parse_args()
    config = DATASETS[args.dataset]
    index_file = args.index_file or config["index"].format(M=args.M)
    output_file = args.output or str(
        EXPERIMENT_DIR
        / "results"
        / f"{args.dataset}_hnsw_M{args.M}_ef{args.ef_search}_layer_stats.csv"
    )

    if not os.path.exists(index_file):
        raise FileNotFoundError(f"Index not found: {index_file}")

    faiss.omp_set_num_threads(args.threads)
    index = faiss.read_index(index_file)
    if not isinstance(index, faiss.IndexHNSW):
        raise TypeError(f"Expected HNSW index, got {type(index)}")
    if not hasattr(faiss.cvar.hnsw_stats, "get_nhops_at_level"):
        raise RuntimeError(
            "This FAISS Python module lacks per-level HNSW statistics. "
            "Rebuild and reinstall the modified /home/suyash/faiss source first."
        )

    index.hnsw.efSearch = args.ef_search
    queries, groundtruth = load_queries_and_groundtruth(args.dataset, config)
    if queries.ndim != 2 or queries.shape[1] != index.d:
        raise ValueError(
            f"Query shape {queries.shape} is incompatible with index dimension {index.d}"
        )
    if len(queries) != len(groundtruth):
        raise ValueError("Query and ground-truth counts differ")

    num_queries = len(queries)
    max_level = int(index.hnsw.max_level)
    num_layers = max_level + 1
    levels = faiss.vector_to_array(index.hnsw.levels)
    nodes_per_layer = [
        int(np.count_nonzero(levels > layer)) for layer in range(num_layers)
    ]

    print(f"Dataset:      {args.dataset}")
    print(f"Queries:      {num_queries:,}")
    print(f"Dimension:    {index.d}")
    print(f"k:            {args.k}")
    print(f"efSearch:     {args.ef_search}")
    print(f"Threads:      {args.threads}")
    print(f"Entry point:  {index.hnsw.entry_point}")
    print(f"Layers:       {num_layers} (0 through {max_level})")
    for layer, count in enumerate(nodes_per_layer):
        print(f"  layer {layer}: {count:,} nodes")

    warmup_size = min(args.warmup, num_queries)
    if warmup_size:
        index.search(queries[:warmup_size], args.k)

    # Time one batched search. Per-query instrumentation below is deliberately
    # excluded because repeated Python calls distort latency.
    faiss.cvar.hnsw_stats.reset()
    start = time.perf_counter()
    _, batch_labels = index.search(queries, args.k)
    elapsed = time.perf_counter() - start
    batch_hops = int(faiss.cvar.hnsw_stats.nhops)
    batch_ndis = int(faiss.cvar.hnsw_stats.ndis)

    fieldnames = [
        "dataset",
        "query_id",
        "layer",
        "nodes_on_layer",
        "max_level",
        "num_layers",
        "M",
        "ef_search",
        "k",
        "hops",
        "distance_computations",
        "total_query_hops",
        "total_query_distance_computations",
        "recall_at_k",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    layer_hop_totals = np.zeros(num_layers, dtype=np.int64)
    layer_ndis_totals = np.zeros(num_layers, dtype=np.int64)
    with open(output_file, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for query_id, query in enumerate(queries):
            faiss.cvar.hnsw_stats.reset()
            _, labels = index.search(query.reshape(1, -1), args.k)
            stats = faiss.cvar.hnsw_stats
            total_hops = int(stats.nhops)
            total_ndis = int(stats.ndis)
            recall = recall_at_k(labels[0], groundtruth[query_id], args.k)

            for layer in range(num_layers):
                hops = int(stats.get_nhops_at_level(layer))
                ndis = int(stats.get_ndis_at_level(layer))
                layer_hop_totals[layer] += hops
                layer_ndis_totals[layer] += ndis
                writer.writerow(
                    {
                        "dataset": args.dataset,
                        "query_id": query_id,
                        "layer": layer,
                        "nodes_on_layer": nodes_per_layer[layer],
                        "max_level": max_level,
                        "num_layers": num_layers,
                        "M": args.M,
                        "ef_search": args.ef_search,
                        "k": args.k,
                        "hops": hops,
                        "distance_computations": ndis,
                        "total_query_hops": total_hops,
                        "total_query_distance_computations": total_ndis,
                        "recall_at_k": f"{recall:.8f}",
                    }
                )

    batch_recall = np.mean(
        [
            recall_at_k(batch_labels[i], groundtruth[i], args.k)
            for i in range(num_queries)
        ]
    )
    print("=" * 64)
    print(f"Batch search time:          {elapsed:.6f} s")
    print(f"Average latency:            {elapsed / num_queries * 1e6:.2f} us/query")
    print(f"Queries/sec:                {num_queries / elapsed:.2f}")
    print(f"Average total hops:         {batch_hops / num_queries:.2f}")
    print(f"Average distance comps:     {batch_ndis / num_queries:.2f}")
    print(f"Recall@{args.k}:                 {batch_recall:.6f}")
    print("Per-layer averages:")
    for layer in range(num_layers):
        print(
            f"  layer {layer}: hops={layer_hop_totals[layer] / num_queries:.2f}, "
            f"distance_computations={layer_ndis_totals[layer] / num_queries:.2f}"
        )
    print(f"Per-query/per-layer CSV:    {output_file}")
    print("=" * 64)


if __name__ == "__main__":
    main()
