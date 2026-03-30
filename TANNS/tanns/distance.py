"""
Distance functions for high-dimensional vectors.
"""
import numpy as np
from typing import Callable


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Euclidean (L2) distance between two vectors."""
    diff = a - b
    return float(np.dot(diff, diff))  # squared L2 (faster, same ordering)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance (1 - cosine similarity) between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / (norm_a * norm_b))


def get_distance_fn(metric: str = "euclidean") -> Callable:
    """Get a distance function by name."""
    if metric in ("euclidean", "l2"):
        return euclidean_distance
    elif metric in ("cosine",):
        return cosine_distance
    else:
        raise ValueError(f"Unknown metric: {metric}. Supported: 'euclidean', 'cosine'")
