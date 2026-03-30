"""
HNSW (Hierarchical Navigable Small World) graph index.

Reference:
  Malkov, Y. A., & Yashunin, D. A. (2018).
  "Efficient and robust approximate nearest neighbor search using
   hierarchical navigable small world graphs."
  IEEE TPAMI, 2018.

This implementation covers the base-layer operations used in TANNS.
For TANNS, we use only the base (flat) layer from HNSW, i.e., NSW.
The multi-layer extension is provided for completeness.

Algorithm 1: HNSW Search
Algorithm 2: HNSW Construction  (includes Select-Nbrs)
"""
import heapq
import math
import random
import numpy as np
from typing import List, Set, Dict, Tuple, Optional, Callable

from .data_types import Vector
from .distance import euclidean_distance


class HNSWNode:
    """A node in the HNSW graph."""

    def __init__(self, vec: Vector, max_layer: int):
        self.vec = vec
        # neighbors[layer] = list of Vector IDs
        self.neighbors: List[List[int]] = [[] for _ in range(max_layer + 1)]

    def __repr__(self):
        return f"HNSWNode(id={self.vec.id})"


class HNSW:
    """
    Hierarchical Navigable Small World (HNSW) graph index.

    Parameters
    ----------
    M : int
        Maximum number of neighbors per node (per layer). Default: 16.
    M_max0 : int
        Maximum neighbors at layer 0. Default: 2*M.
    ef_construction : int
        Size of candidate set during construction (M' in paper). Default: 200.
    ef_search : int
        Size of candidate set during search (k' in paper). Default: 50.
    distance_fn : Callable
        Distance function. Default: squared Euclidean.
    """

    def __init__(
        self,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        distance_fn: Callable = euclidean_distance,
    ):
        self.M = M
        self.M_max0 = 2 * M  # layer-0 allows more neighbors
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.distance_fn = distance_fn

        # Nodes indexed by vector id
        self.nodes: Dict[int, HNSWNode] = {}
        self.entry_point: Optional[int] = None  # vector id of entry point
        self.max_layer: int = 0

        # Probability factor for layer assignment
        self._ml = 1.0 / math.log(max(M, 2))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, vec: Vector) -> None:
        """Insert a vector into the HNSW index (Algorithm 2)."""
        l = self._random_level()
        if self.max_layer < l:
            self.max_layer = l
        node = HNSWNode(vec, l)
        self.nodes[vec.id] = node

        if self.entry_point is None:
            self.entry_point = vec.id
            return

        ep = [self.entry_point]
        # Traverse from top layer down to l+1 (greedy descend, ef=1)
        for layer in range(self.max_layer, l, -1):
            # Make sure the entry point node exists at this layer
            ep_filtered = [eid for eid in ep if eid in self.nodes and layer < len(self.nodes[eid].neighbors)]
            if not ep_filtered:
                break
            # _search_layer returns List[HNSWNode]; extract ids for next iteration
            ep = [n.vec.id for n in self._search_layer(vec.data, ep_filtered, 1, layer)]

        # Insert into layers l down to 0
        for layer in range(min(l, self.max_layer), -1, -1):
            M_cur = self.M_max0 if layer == 0 else self.M
            # ep is List[int] here
            candidates = self._search_layer(vec.data, ep, self.ef_construction, layer)
            neighbors = self._select_neighbors(vec, candidates, M_cur, layer)
            node.neighbors[layer] = [n.vec.id for n in neighbors]
            # Bidirectional connection
            for nb in neighbors:
                nb_node = self.nodes[nb.vec.id]
                # Extend neighbors list if this node was inserted at a lower level
                while len(nb_node.neighbors) <= layer:
                    nb_node.neighbors.append([])
                nb_node.neighbors[layer].append(vec.id)
                M_nb = self.M_max0 if layer == 0 else self.M
                if len(nb_node.neighbors[layer]) > M_nb:
                    # Shrink by selecting best M neighbors
                    cands = [self.nodes[nid] for nid in nb_node.neighbors[layer]]
                    selected = self._select_neighbors(nb.vec, cands, M_nb, layer)
                    nb_node.neighbors[layer] = [s.vec.id for s in selected]
            ep = [n.vec.id for n in candidates]  # update ep as ids for next layer

        # Update entry point if new node has higher layer
        if l > self.max_layer or self.entry_point is None:
            self.entry_point = vec.id

    def search(self, query: np.ndarray, k: int, ef: Optional[int] = None) -> List[Tuple[float, int]]:
        """
        Search for k approximate nearest neighbors (Algorithm 1).

        Returns list of (distance, vector_id) sorted ascending by distance.
        """
        if self.entry_point is None:
            return []
        if ef is None:
            ef = max(self.ef_search, k)

        ep = [self.entry_point]
        for layer in range(self.max_layer, 0, -1):
            result_nodes = self._search_layer(query, ep, 1, layer)
            ep = [n.vec.id for n in result_nodes]

        candidates = self._search_layer(query, ep, ef, 0)

        # Return top-k
        results = [(self.distance_fn(query, self.nodes[n.vec.id].vec.data), n.vec.id)
                   for n in candidates]
        results.sort(key=lambda x: x[0])
        return results[:k]

    def get_neighbors(self, vid: int, layer: int = 0) -> List[int]:
        """Get neighbor ids of node vid at given layer."""
        node = self.nodes.get(vid)
        if node is None:
            return []
        if layer >= len(node.neighbors):
            return []
        return list(node.neighbors[layer])

    def __len__(self):
        return len(self.nodes)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _random_level(self) -> int:
        """Sample insertion layer."""
        level = 0
        while random.random() < (1.0 / math.e) and level < 32:
            level += 1
        return level

    def _search_layer(
        self,
        query: np.ndarray,
        entry_points: List[int],
        ef: int,
        layer: int,
    ) -> List["HNSWNode"]:
        """
        Greedy search within a single layer (Algorithm 1 base).
        Returns ef nearest nodes found (as HNSWNode objects).
        """
        visited: Set[int] = set(entry_points)
        # pool: min-heap of (dist, id) — candidates to expand
        # ann:  max-heap of (-dist, id) — current best ef results
        pool: List[Tuple[float, int]] = []
        ann: List[Tuple[float, int]] = []  # stored as (-dist, id)

        for ep_id in entry_points:
            d = self.distance_fn(query, self.nodes[ep_id].vec.data)
            heapq.heappush(pool, (d, ep_id))
            heapq.heappush(ann, (-d, ep_id))

        while pool:
            d_u, u_id = heapq.heappop(pool)
            # Stopping criterion: closest candidate is farther than worst in ann
            worst_d = -ann[0][0] if ann else float("inf")
            if d_u > worst_d:
                break

            # Expand neighbors
            u_node = self.nodes.get(u_id)
            if u_node is None or layer >= len(u_node.neighbors):
                continue
            for nb_id in u_node.neighbors[layer]:
                if nb_id not in visited and nb_id in self.nodes:
                    visited.add(nb_id)
                    d_nb = self.distance_fn(query, self.nodes[nb_id].vec.data)
                    worst_d = -ann[0][0] if ann else float("inf")
                    if len(ann) < ef or d_nb < worst_d:
                        heapq.heappush(pool, (d_nb, nb_id))
                        heapq.heappush(ann, (-d_nb, nb_id))
                        if len(ann) > ef:
                            heapq.heappop(ann)

        return [self.nodes[vid] for _, vid in ann if vid in self.nodes]

    def _select_neighbors(
        self,
        origin: Vector,
        candidates: List["HNSWNode"],
        M: int,
        layer: int,
        extend_candidates: bool = False,
        keep_pruned: bool = False,
    ) -> List["HNSWNode"]:
        """
        Heuristic neighbor selection (Algorithm 2, lines 9-16 in paper).
        
        Prioritizes diverse neighbors; a candidate u is dominated by v if:
            dis(origin, v) < dis(origin, u)  AND  dis(u, v) < dis(origin, u)
        """
        # Sort candidates by distance to origin
        scored = sorted(
            candidates,
            key=lambda node: self.distance_fn(origin.data, node.vec.data)
        )
        selected: List[HNSWNode] = []
        discarded: List[HNSWNode] = []

        for node in scored:
            if len(selected) >= M:
                break
            dominated = False
            d_origin_u = self.distance_fn(origin.data, node.vec.data)
            for s in selected:
                d_origin_s = self.distance_fn(origin.data, s.vec.data)
                d_u_s = self.distance_fn(node.vec.data, s.vec.data)
                # u is dominated by s if s is closer to origin AND s is closer to u than origin
                if d_origin_s < d_origin_u and d_u_s < d_origin_u:
                    dominated = True
                    break
            if not dominated:
                selected.append(node)
            else:
                discarded.append(node)

        # Optionally keep pruned to fill up to M
        if keep_pruned:
            for node in discarded:
                if len(selected) >= M:
                    break
                selected.append(node)

        return selected[:M]
