"""
Main benchmark script for TANNS experiments.

Reproduces the key experiments from Section VI of the paper:
  - Search performance (QPS vs Recall Rate)
  - Index construction performance (Update Throughput, Memory Usage)
  - Scalability test

Usage:
    python experiments/benchmark.py [--n 10000] [--dim 128] [--pattern uniform]
                                     [--n_queries 100] [--k 10] [--metric euclidean]
"""

import sys
import os
import time
import argparse
import numpy as np
from typing import List, Dict

# Allow importing from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tanns.data_types import Vector, TANNSQuery
from tanns.distance import euclidean_distance, cosine_distance, get_distance_fn
from tanns.timestamp_graph import TimestampGraph
from tanns.compressed_timestamp_graph import CompressedTimestampGraph
from experiments.data_generator import (
    generate_dataset,
    generate_queries,
    compute_ground_truth,
    compute_recall,
    build_events,
)
from experiments.baselines import PreFiltering, PostFilteringHNSW


def run_benchmark(
    n: int = 5000,
    dim: int = 64,
    pattern: str = "uniform",
    n_queries: int = 200,
    k: int = 10,
    metric: str = "euclidean",
    M: int = 16,
    M_prime: int = 200,
    ef_search_values: List[int] = None,
    verbose: bool = True,
):
    """Run a full benchmark experiment."""
    if ef_search_values is None:
        ef_search_values = [k, 2 * k, 5 * k, 10 * k, 20 * k, 50 * k]

    dist_fn = get_distance_fn(metric)

    print(f"\n{'='*60}")
    print(f"TANNS Benchmark")
    print(f"  n={n}, dim={dim}, pattern={pattern}, k={k}, metric={metric}, M={M}")
    print(f"{'='*60}")

    # ----------------------------------------------------------------
    # 1. Generate dataset
    # ----------------------------------------------------------------
    print("\n[1/4] Generating dataset...")
    vectors, T = generate_dataset(n=n, dim=dim, pattern=pattern, metric=metric)
    queries = generate_queries(T=T, n_queries=n_queries, k=k, dim=dim, metric=metric)

    # ----------------------------------------------------------------
    # 2. Compute ground truth
    # ----------------------------------------------------------------
    print("[2/4] Computing ground truth...")
    ground_truth = compute_ground_truth(vectors, queries, distance_fn=dist_fn)
    valid_queries = [i for i, gt in enumerate(ground_truth) if len(gt) > 0]
    print(f"  Valid queries: {len(valid_queries)}/{n_queries}")

    # ----------------------------------------------------------------
    # 3. Build indexes
    # ----------------------------------------------------------------
    print("\n[3/4] Building indexes...")
    events = build_events(vectors)

    # --- Timestamp Graph ---
    print("  Building Timestamp Graph...")
    tg = TimestampGraph(M=M, M_prime=M_prime, distance_fn=dist_fn)
    tg._expired = {}
    t0 = time.time()
    tg.build_from_stream(events)
    tg_build_time = time.time() - t0
    print(f"  TG build time: {tg_build_time:.2f}s")

    # --- Compressed Timestamp Graph ---
    print("  Building Compressed Timestamp Graph...")
    ctg = CompressedTimestampGraph(M=M, M_prime=M_prime, distance_fn=dist_fn)
    t0 = time.time()
    ctg.build_from_stream(events)
    ctg_build_time = time.time() - t0
    print(f"  CTG build time: {ctg_build_time:.2f}s")

    # --- Pre-Filtering baseline ---
    print("  Building Pre-Filtering baseline...")
    pf = PreFiltering(distance_fn=dist_fn)
    pf.build(vectors)

    # --- Post-Filtering HNSW baseline ---
    print("  Building Post-Filtering (HNSW) baseline...")
    pfh = PostFilteringHNSW(M=M, ef_construction=M_prime, distance_fn=dist_fn)
    pfh.build(vectors)

    # ----------------------------------------------------------------
    # 4. Evaluate search performance
    # ----------------------------------------------------------------
    print("\n[4/4] Evaluating search performance (QPS vs Recall)...")
    print(f"\n{'Method':<30} {'ef/param':<10} {'Recall':<10} {'QPS':<12}")
    print("-" * 62)

    results = {}

    # Pre-Filtering (exact)
    preds_pf, t_pf = _run_search(pf, queries, method_type="pre")
    recall_pf = compute_recall(preds_pf, ground_truth)
    qps_pf = n_queries / max(t_pf, 1e-9)
    results["Pre-Filtering"] = [{"ef": "N/A", "recall": recall_pf, "qps": qps_pf}]
    print(f"{'Pre-Filtering':<30} {'N/A':<10} {recall_pf:.4f}     {qps_pf:.1f}")

    # Post-Filtering HNSW
    pfh_points = []
    for ef in ef_search_values:
        preds, t = _run_search_pfh(pfh, queries, ef=ef)
        recall = compute_recall(preds, ground_truth)
        qps = n_queries / max(t, 1e-9)
        pfh_points.append({"ef": ef, "recall": recall, "qps": qps})
        print(f"{'Post-Filtering (HNSW)':<30} {ef:<10} {recall:.4f}     {qps:.1f}")
    results["Post-Filtering (HNSW)"] = pfh_points

    # Timestamp Graph
    tg_points = []
    for ef in ef_search_values:
        preds, t = _run_search_tg(tg, queries, ef=ef)
        recall = compute_recall(preds, ground_truth)
        qps = n_queries / max(t, 1e-9)
        tg_points.append({"ef": ef, "recall": recall, "qps": qps})
        print(f"{'Timestamp Graph':<30} {ef:<10} {recall:.4f}     {qps:.1f}")
    results["Timestamp Graph"] = tg_points

    # Compressed Timestamp Graph
    ctg_points = []
    for ef in ef_search_values:
        preds, t = _run_search_ctg(ctg, queries, ef=ef)
        recall = compute_recall(preds, ground_truth)
        qps = n_queries / max(t, 1e-9)
        ctg_points.append({"ef": ef, "recall": recall, "qps": qps})
        print(f"{'Compressed TG':<30} {ef:<10} {recall:.4f}     {qps:.1f}")
    results["Compressed Timestamp Graph"] = ctg_points

    # Memory analysis
    print("\n--- Memory Usage (approx. neighbor references) ---")
    mem = ctg.memory_usage()
    for k_m, v_m in mem.items():
        print(f"  {k_m}: {v_m}")

    print(f"\n  TG build time:  {tg_build_time:.2f}s")
    print(f"  CTG build time: {ctg_build_time:.2f}s")

    return results


# ------------------------------------------------------------------
# Search helpers
# ------------------------------------------------------------------

def _run_search(method, queries, method_type="pre"):
    preds = []
    t0 = time.time()
    for q in queries:
        res = method.search(q)
        preds.append([vid for _, vid in res])
    return preds, time.time() - t0


def _run_search_pfh(method, queries, ef=100):
    preds = []
    t0 = time.time()
    for q in queries:
        res = method.search(q, multiplier=max(1, ef // q.k))
        preds.append([vid for _, vid in res])
    return preds, time.time() - t0


def _run_search_tg(tg: TimestampGraph, queries, ef=50):
    preds = []
    t0 = time.time()
    for q in queries:
        res = tg.search_at(q, ef=ef)
        preds.append([vid for _, vid in res])
    return preds, time.time() - t0


def _run_search_ctg(ctg: CompressedTimestampGraph, queries, ef=50):
    preds = []
    t0 = time.time()
    for q in queries:
        res = ctg.search_at(q, ef=ef)
        preds.append([vid for _, vid in res])
    return preds, time.time() - t0


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TANNS Benchmark")
    parser.add_argument("--n", type=int, default=5000, help="Number of vectors")
    parser.add_argument("--dim", type=int, default=64, help="Vector dimensionality")
    parser.add_argument("--pattern", type=str, default="uniform",
                        choices=["short", "long", "mixed", "uniform"],
                        help="Temporal data pattern")
    parser.add_argument("--n_queries", type=int, default=200, help="Number of queries")
    parser.add_argument("--k", type=int, default=10, help="Number of nearest neighbors")
    parser.add_argument("--metric", type=str, default="euclidean",
                        choices=["euclidean", "cosine"],
                        help="Distance metric")
    parser.add_argument("--M", type=int, default=16, help="Neighbor count in graph")
    parser.add_argument("--M_prime", type=int, default=200, help="Candidate count in construction")

    args = parser.parse_args()

    run_benchmark(
        n=args.n,
        dim=args.dim,
        pattern=args.pattern,
        n_queries=args.n_queries,
        k=args.k,
        metric=args.metric,
        M=args.M,
        M_prime=args.M_prime,
    )
