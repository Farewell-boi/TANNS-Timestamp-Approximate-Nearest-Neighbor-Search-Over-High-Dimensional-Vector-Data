"""
Demo script: quick end-to-end demonstration of TANNS.

Run:  python demo.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from tanns.data_types import Vector, TANNSQuery
from tanns.timestamp_graph import TimestampGraph
from tanns.compressed_timestamp_graph import CompressedTimestampGraph
from experiments.data_generator import (
    generate_dataset, generate_queries, compute_ground_truth,
    compute_recall, build_events,
)
from tanns.distance import euclidean_distance


def demo():
    print("=" * 60)
    print("TANNS Demo: Timestamp ANN Search")
    print("=" * 60)

    # ── Parameters ──────────────────────────────────────────────────
    N = 2000        # number of vectors
    DIM = 64        # dimensionality
    PATTERN = "mixed"   # short / long / mixed / uniform
    N_QUERIES = 100
    K = 10
    M = 16
    M_PRIME = 100
    EF = 80

    # ── Generate data ──────────────────────────────────────────────
    print(f"\n[1] Generating {N} vectors (dim={DIM}, pattern={PATTERN})...")
    vectors, T = generate_dataset(n=N, dim=DIM, pattern=PATTERN)
    queries = generate_queries(T=T, n_queries=N_QUERIES, k=K, dim=DIM)
    print(f"    T = {T}, queries = {N_QUERIES}")

    # ── Ground truth ───────────────────────────────────────────────
    print("[2] Computing exact ground truth...")
    gt = compute_ground_truth(vectors, queries)
    valid_cnt = sum(1 for g in gt if len(g) > 0)
    print(f"    Valid queries: {valid_cnt}/{N_QUERIES}")

    # ── Build Timestamp Graph ──────────────────────────────────────
    print("\n[3] Building Timestamp Graph (TG)...")
    tg = TimestampGraph(M=M, M_prime=M_PRIME)
    tg._expired = {}
    events = build_events(vectors)

    import time
    t0 = time.time()
    tg.build_from_stream(events)
    t_tg = time.time() - t0
    print(f"    Build time: {t_tg:.2f}s")

    # ── Build Compressed Timestamp Graph ──────────────────────────
    print("[4] Building Compressed Timestamp Graph (CTG)...")
    ctg = CompressedTimestampGraph(M=M, M_prime=M_PRIME, mu=8)
    t0 = time.time()
    ctg.build_from_stream(events)
    t_ctg = time.time() - t0
    print(f"    Build time: {t_ctg:.2f}s")

    # ── Search & Evaluate ─────────────────────────────────────────
    print("\n[5] Searching and evaluating...")

    # Timestamp Graph
    preds_tg = []
    t0 = time.time()
    for q in queries:
        res = tg.search_at(q, ef=EF)
        preds_tg.append([vid for _, vid in res])
    t_search_tg = time.time() - t0
    recall_tg = compute_recall(preds_tg, gt)
    qps_tg = N_QUERIES / max(t_search_tg, 1e-9)

    # Compressed Timestamp Graph
    preds_ctg = []
    t0 = time.time()
    for q in queries:
        res = ctg.search_at(q, ef=EF)
        preds_ctg.append([vid for _, vid in res])
    t_search_ctg = time.time() - t0
    recall_ctg = compute_recall(preds_ctg, gt)
    qps_ctg = N_QUERIES / max(t_search_ctg, 1e-9)

    # Brute-force baseline (pre-filtering)
    from experiments.baselines import PreFiltering
    pf = PreFiltering(distance_fn=euclidean_distance)
    pf.build(vectors)
    preds_pf = []
    t0 = time.time()
    for q in queries:
        res = pf.search(q)
        preds_pf.append([vid for _, vid in res])
    t_search_pf = time.time() - t0
    recall_pf = compute_recall(preds_pf, gt)
    qps_pf = N_QUERIES / max(t_search_pf, 1e-9)

    # ── Results ──────────────────────────────────────────────────
    print(f"\n{'Method':<35} {'Recall':<10} {'QPS':<12} {'Build(s)':<10}")
    print("-" * 70)
    print(f"{'Pre-Filtering (exact baseline)':<35} {recall_pf:.4f}     {qps_pf:<12.1f} -")
    print(f"{'Timestamp Graph':<35} {recall_tg:.4f}     {qps_tg:<12.1f} {t_tg:.2f}")
    print(f"{'Compressed Timestamp Graph':<35} {recall_ctg:.4f}     {qps_ctg:<12.1f} {t_ctg:.2f}")

    # Memory usage
    mem = ctg.memory_usage()
    print(f"\nCTG Memory breakdown (neighbor refs):")
    for k_m, v_m in mem.items():
        print(f"  {k_m:25s}: {v_m}")

    print("\nDemo complete.")


if __name__ == "__main__":
    demo()
