"""
Baseline methods for TANNS comparison.

As described in Section VI-A of the paper:
  1. Pre-Filtering:      Filter valid vectors first, then linear scan.
  2. Post-Filtering (HNSW): Full HNSW search first, then filter by timestamp.
  3. ACORN-like:         Graph index with predicate-agnostic construction.
  4. SeRF-like:          Segment graph with start time as attribute + post-filter on end time.
"""

import heapq
import numpy as np
from typing import List, Tuple, Dict, Optional, Callable, Set

from tanns.data_types import Vector, TANNSQuery
from tanns.distance import euclidean_distance
from tanns.hnsw import HNSW


# =====================================================================
# Baseline 1: Pre-Filtering
# =====================================================================

class PreFiltering:
    """
    Pre-Filtering baseline:
    1. Retrieve all valid vectors at ts.
    2. Linear scan to find k nearest neighbors.
    
    Always exact. Slow for large datasets.
    """

    def __init__(self, distance_fn: Callable = euclidean_distance):
        self.distance_fn = distance_fn
        self.vectors: List[Vector] = []

    def build(self, vectors: List[Vector]) -> None:
        self.vectors = list(vectors)

    def update(self, vec: Vector, event_type: str) -> None:
        if event_type == "insert":
            self.vectors.append(vec)
        elif event_type == "expire":
            pass  # We keep all vectors; filter at query time

    def search(self, query: TANNSQuery) -> List[Tuple[float, int]]:
        ts = query.timestamp
        k = query.k
        q = query.query_vector

        valid = [v for v in self.vectors if v.is_valid_at(ts)]
        if not valid:
            return []

        dists = [(self.distance_fn(q, v.data), v.id) for v in valid]
        dists.sort(key=lambda x: x[0])
        return dists[:k]


# =====================================================================
# Baseline 2: Post-Filtering (HNSW)
# =====================================================================

class PostFilteringHNSW:
    """
    Post-Filtering with HNSW:
    1. Build HNSW over ALL vectors (ignoring timestamps).
    2. At query time: search HNSW with large candidate set, then filter by validity.
    
    Approximate. May miss valid vectors with aggressive pruning.
    """

    def __init__(
        self,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 200,
        distance_fn: Callable = euclidean_distance,
    ):
        self.M = M
        self.ef_search = ef_search
        self.distance_fn = distance_fn
        self.hnsw = HNSW(M=M, ef_construction=ef_construction,
                         ef_search=ef_search, distance_fn=distance_fn)
        self.vectors: Dict[int, Vector] = {}

    def build(self, vectors: List[Vector]) -> None:
        """Build HNSW over all vectors."""
        for v in vectors:
            self.hnsw.add(v)
            self.vectors[v.id] = v

    def update(self, vec: Vector, event_type: str) -> None:
        if event_type == "insert":
            self.hnsw.add(vec)
            self.vectors[vec.id] = vec
        # Note: HNSW doesn't support deletion; we keep expired vectors
        # and handle them at query time via filtering

    def search(self, query: TANNSQuery, multiplier: int = 10) -> List[Tuple[float, int]]:
        """
        Search HNSW with ef = k * multiplier, then filter by validity at ts.
        """
        ts = query.timestamp
        k = query.k
        q = query.query_vector
        ef = max(self.ef_search, k * multiplier)

        candidates = self.hnsw.search(q, k=min(len(self.hnsw.nodes), ef), ef=ef)

        # Filter by validity
        results = []
        for d, vid in candidates:
            v = self.vectors.get(vid)
            if v is not None and v.is_valid_at(ts):
                results.append((d, vid))
            if len(results) >= k:
                break

        return results[:k]


# =====================================================================
# Baseline 3: Naive Graph-based TANNS (Section III-B)
# =====================================================================

class NaiveGraphTANNS:
    """
    Naive Graph-based TANNS (Section III-B of the paper):
    Build a separate HNSW index for each timestamp.
    Memory O(MN²), Update O(MNlogN).
    
    For tractability, we build HNSW lazily per unique timestamp.
    """

    def __init__(
        self,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        distance_fn: Callable = euclidean_distance,
    ):
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.distance_fn = distance_fn
        self.all_vectors: List[Vector] = []
        # Cache: timestamp -> HNSW index
        self._cache: Dict[int, HNSW] = {}

    def build(self, vectors: List[Vector]) -> None:
        self.all_vectors = list(vectors)

    def update(self, vec: Vector, event_type: str) -> None:
        if event_type == "insert":
            self.all_vectors.append(vec)
        self._cache.clear()  # Invalidate cache

    def search(self, query: TANNSQuery) -> List[Tuple[float, int]]:
        ts = query.timestamp
        k = query.k
        q = query.query_vector

        if ts not in self._cache:
            # Build HNSW for timestamp ts
            valid = [v for v in self.all_vectors if v.is_valid_at(ts)]
            idx = HNSW(M=self.M, ef_construction=self.ef_construction,
                       ef_search=self.ef_search, distance_fn=self.distance_fn)
            for v in valid:
                idx.add(v)
            self._cache[ts] = idx

        idx = self._cache[ts]
        if len(idx.nodes) == 0:
            return []
        return idx.search(q, k=k, ef=max(self.ef_search, k))


# =====================================================================
# Utility: compute recall rate
# =====================================================================

def evaluate_methods(
    methods: Dict[str, object],
    queries: List[TANNSQuery],
    ground_truth: List[List[int]],
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate multiple methods on recall rate.
    
    Returns dict: {method_name: {'recall': float, 'n_queries': int}}
    """
    import time
    from experiments.data_generator import compute_recall

    results_all = {}
    for name, method in methods.items():
        preds = []
        start = time.time()
        for q in queries:
            res = method.search(q)
            preds.append([vid for _, vid in res])
        elapsed = time.time() - start
        recall = compute_recall(preds, ground_truth)
        qps = len(queries) / max(elapsed, 1e-9)
        results_all[name] = {
            "recall": recall,
            "qps": qps,
            "n_queries": len(queries),
            "elapsed_s": elapsed,
        }
    return results_all
