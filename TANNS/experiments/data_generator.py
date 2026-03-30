"""
Data generators for TANNS experiments.

Generates synthetic high-dimensional datasets with temporal patterns:
  - Short:   valid time ranges < 0.05T
  - Long:    valid time ranges > 0.4T
  - Mixed:   half short, half long
  - Uniform: uniformly distributed valid time ranges in [1, T]

Each vector is assigned a random start/end timestamp.
"""

import numpy as np
import random
from typing import List, Tuple, Optional

from tanns.data_types import Vector


def generate_dataset(
    n: int = 1000,
    dim: int = 128,
    pattern: str = "uniform",
    seed: int = 42,
    metric: str = "euclidean",
) -> Tuple[List[Vector], int]:
    """
    Generate a synthetic dataset of n high-dimensional vectors with temporal patterns.
    
    Parameters
    ----------
    n : int
        Number of vectors. Default: 1000.
    dim : int
        Dimensionality of each vector. Default: 128.
    pattern : str
        Temporal pattern: 'short', 'long', 'mixed', or 'uniform'. Default: 'uniform'.
    seed : int
        Random seed. Default: 42.
    metric : str
        'euclidean' or 'cosine'. Default: 'euclidean'.
    
    Returns
    -------
    vectors : List[Vector]
        Generated vectors with start/end timestamps.
    T : int
        Maximum timestamp (= 2*n per paper, since each vector inserts once and expires once).
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    # T = 2*n (each vector enters at start, expires at end → 2N events)
    T = 2 * n

    # Generate vector data
    if metric == "cosine":
        # Normalize for cosine similarity
        raw = rng.standard_normal((n, dim)).astype(np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        data = raw / np.maximum(norms, 1e-9)
    else:
        data = rng.standard_normal((n, dim)).astype(np.float32)

    # Assign start/end timestamps based on pattern
    vectors = []
    for i in range(n):
        s, e = _sample_timestamps(i, n, T, pattern, rng)
        vec = Vector(id=i, data=data[i], start=s, end=e)
        vectors.append(vec)

    return vectors, T


def _sample_timestamps(i: int, n: int, T: int, pattern: str, rng) -> Tuple[int, int]:
    """Sample start and end timestamps for a vector."""
    max_range = T

    if pattern == "short":
        # Valid range < 0.05T
        range_len = max(1, int(rng.uniform(1, 0.05 * T)))
    elif pattern == "long":
        # Valid range > 0.4T
        range_len = max(1, int(rng.uniform(0.4 * T, 0.7 * T)))
    elif pattern == "mixed":
        # Half short, half long
        if i < n // 2:
            range_len = max(1, int(rng.uniform(1, 0.05 * T)))
        else:
            range_len = max(1, int(rng.uniform(0.4 * T, 0.7 * T)))
    else:  # uniform
        range_len = max(1, int(rng.uniform(1, T)))

    start = max(1, int(rng.uniform(1, max(1, T - range_len))))
    end = min(T, start + range_len)
    return start, end


def generate_queries(
    T: int,
    n_queries: int = 100,
    k: int = 10,
    seed: int = 123,
    dim: int = 128,
    metric: str = "euclidean",
) -> List:
    """
    Generate random TANNS queries.
    
    Returns list of TANNSQuery objects with random query vectors and timestamps.
    """
    from tanns.data_types import TANNSQuery
    rng = np.random.default_rng(seed)

    queries = []
    for _ in range(n_queries):
        ts = int(rng.integers(1, T + 1))
        if metric == "cosine":
            q = rng.standard_normal(dim).astype(np.float32)
            q = q / max(np.linalg.norm(q), 1e-9)
        else:
            q = rng.standard_normal(dim).astype(np.float32)
        queries.append(TANNSQuery(query_vector=q, timestamp=ts, k=k))

    return queries


def compute_ground_truth(
    vectors: List[Vector],
    queries: List,
    distance_fn=None,
) -> List[List[int]]:
    """
    Compute exact kNN ground truth for each TANNS query.
    
    For each query, finds k nearest neighbors from the valid vectors at ts.
    Returns list of lists of vector IDs (sorted by distance).
    """
    from tanns.distance import euclidean_distance
    if distance_fn is None:
        distance_fn = euclidean_distance

    ground_truth = []
    for q in queries:
        ts = q.timestamp
        k = q.k
        valid = [v for v in vectors if v.is_valid_at(ts)]
        if not valid:
            ground_truth.append([])
            continue
        dists = [(distance_fn(q.query_vector, v.data), v.id) for v in valid]
        dists.sort(key=lambda x: x[0])
        ground_truth.append([vid for _, vid in dists[:k]])

    return ground_truth


def compute_recall(
    results: List[List[int]],
    ground_truth: List[List[int]],
) -> float:
    """
    Compute average recall rate: |r ∩ r*| / k
    """
    if not results or not ground_truth:
        return 0.0

    total = 0.0
    for r, gt in zip(results, ground_truth):
        if not gt:
            continue
        k = len(gt)
        hits = len(set(r) & set(gt))
        total += hits / k

    return total / len([gt for gt in ground_truth if gt])


def build_events(vectors: List[Vector]) -> List[Tuple[int, str, Vector]]:
    """
    Convert a list of vectors to a stream of (timestamp, event_type, vector) events.
    Returns events sorted by timestamp.
    """
    events = []
    for vec in vectors:
        events.append((vec.start, "insert", vec))
        if vec.end is not None:
            events.append((vec.end, "expire", vec))
    events.sort(key=lambda x: x[0])
    return events
