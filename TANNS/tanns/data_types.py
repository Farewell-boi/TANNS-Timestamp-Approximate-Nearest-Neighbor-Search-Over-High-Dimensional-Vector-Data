"""
Data type definitions for TANNS.
Each vector has a start timestamp (s) and end timestamp (e).
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Vector:
    """
    A high-dimensional vector with temporal validity.
    
    Attributes:
        id: Unique identifier for the vector.
        data: The high-dimensional vector data (numpy array).
        start: The timestamp when this vector becomes valid.
        end: The timestamp when this vector expires (None if still valid).
    """
    id: int
    data: np.ndarray
    start: int
    end: Optional[int] = None  # None means still valid (not expired)

    def is_valid_at(self, ts: int) -> bool:
        """Check if this vector is valid at timestamp ts."""
        if self.end is None:
            return self.start <= ts
        return self.start <= ts < self.end

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Vector):
            return self.id == other.id
        return False

    def __repr__(self):
        return f"Vector(id={self.id}, start={self.start}, end={self.end})"


@dataclass
class TANNSQuery:
    """
    A TANNS query: find k approximate nearest neighbors valid at timestamp ts.
    
    Attributes:
        query_vector: The query vector data.
        timestamp: The query timestamp ts.
        k: Number of nearest neighbors to retrieve.
    """
    query_vector: np.ndarray
    timestamp: int
    k: int = 10
