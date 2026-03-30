"""
Timestamp Graph (TG) for TANNS queries.

Section IV of the paper:
  "Timestamp Approximate Nearest Neighbor Search over High-Dimensional Vector Data"
  ICDE 2025.

Key ideas:
  - A single graph index managing valid vectors across ALL historical timestamps.
  - Each node stores PRIMARY neighbors TG[u] (M neighbors) and BACKUP neighbors B[u] (M backup).
  - Historic neighbor lists are versioned per timestamp to enable reconstruction.
  - Point Insertion  → Algorithm 3
  - Point Expiration → Algorithm 4
  - TANNS Search     → binary search on historic neighbor list + greedy routing

Complexity:
  - Search:  O(log²N)
  - Update:  O(log²N)
  - Space:   O(M²N)
"""

import heapq
import bisect
import numpy as np
from typing import List, Dict, Set, Optional, Tuple, Callable

from .data_types import Vector, TANNSQuery
from .distance import euclidean_distance


class NeighborListVersion:
    """
    A versioned snapshot of a node's neighbor list at a given timestamp.
    """
    def __init__(self, timestamp: int, neighbors: List[int]):
        self.timestamp = timestamp
        self.neighbors = list(neighbors)

    def __repr__(self):
        return f"NLV(t={self.timestamp}, nbrs={self.neighbors})"


class TimestampGraph:
    """
    Timestamp Graph: a single HNSW-like proximity graph that tracks neighbor
    lists across all historical timestamps via versioned neighbor lists.

    Parameters
    ----------
    M : int
        Number of primary neighbors per node. Default: 16.
    M_prime : int
        Candidate neighbor count during search/construction (ef_construction). Default: 200.
    distance_fn : Callable
        Distance function. Default: squared Euclidean.
    """

    def __init__(
        self,
        M: int = 16,
        M_prime: int = 200,
        distance_fn: Callable = euclidean_distance,
    ):
        self.M = M
        self.M_prime = M_prime
        self.distance_fn = distance_fn

        # Primary neighbor lists: TG[u] = list of currently-valid neighbor ids
        self.TG: Dict[int, List[int]] = {}
        # Backup neighbor lists: B[u] = list of backup neighbor ids (up to M)
        self.B: Dict[int, List[int]] = {}
        # Historic neighbor lists: hist[u] = list of NeighborListVersion, sorted by timestamp
        self.hist: Dict[int, List[NeighborListVersion]] = {}
        # ALL vectors (current + expired): needed for historical queries
        self._all_vectors: Dict[int, Vector] = {}
        # Currently active (not-yet-expired) vector ids
        self._active: Set[int] = set()
        # Currently valid points (id -> Vector) — references _all_vectors
        self.vectors: Dict[int, Vector] = {}

        # Entry point for graph traversal
        self._entry_point: Optional[int] = None
        # Expired archive (set externally or populated by build_from_stream)
        self._expired: Dict[int, Vector] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def insert(self, vec: Vector) -> None:
        """
        Insert a new point into the timestamp graph at vec.start (Algorithm 3).
        """
        t = vec.start
        vid = vec.id
        self._all_vectors[vid] = vec
        self.vectors[vid] = vec
        self._active.add(vid)
        self.TG[vid] = []
        self.B[vid] = []
        self.hist[vid] = []

        if self._entry_point is None:
            self._entry_point = vid
            self.hist[vid].append(NeighborListVersion(t, []))
            return

        # Step 1: Search for M' candidate neighbors
        cand_nodes = self._search(vec.data, self.M_prime)
        # cand_nodes is list of (dist, vid) sorted by dist

        # Step 2: Select 2M from candidates
        all_cands = [self._all_vectors[nid] for _, nid in cand_nodes if nid in self._all_vectors and nid in self._active]
        selected_2M = self._select_nbrs(vec, all_cands, 2 * self.M)

        # M primary + M backup
        primary = selected_2M[:self.M]
        backup = selected_2M[self.M:]
        self.TG[vid] = [n.id for n in primary]
        self.B[vid] = [n.id for n in backup]

        # Record initial historic neighbor list
        self.hist[vid].append(NeighborListVersion(t, self.TG[vid]))

        # Step 3: Update neighbors of primary neighbors (Algorithm 3, lines 4-12)
        for nbr_vec in primary:
            uid = nbr_vec.id
            TG_u = self.TG[uid]
            B_u = self.B[uid]

            # Check if o (vec) should be added to TG[u]
            if self._should_add_primary(uid, vid):
                # Move furthest primary to backup, add vid as primary
                if len(TG_u) >= self.M:
                    furthest = self._furthest(uid, TG_u)
                    TG_u.remove(furthest)
                    B_u.append(furthest)
                    # Trim backup to M
                    if len(B_u) > self.M:
                        B_u.remove(self._furthest(uid, B_u))
                TG_u.append(vid)
                # Record new version
                self.hist[uid].append(NeighborListVersion(t, list(TG_u)))
            else:
                # Add to backup
                B_u.append(vid)
                if len(B_u) > self.M:
                    B_u.remove(self._furthest(uid, B_u))

        # Update entry point (prefer the earliest inserted stable point)
        # Keep entry point as is (first inserted) for simplicity.

    def expire(self, vec: Vector) -> None:
        """
        Expire a point from the timestamp graph at vec.end (Algorithm 4).
        """
        t = vec.end
        vid = vec.id

        if vid not in self.TG:
            return

        # Move to expired (keep data for historical queries)
        self._expired[vid] = vec
        if vid in self.vectors:
            del self.vectors[vid]
        self._active.discard(vid)

        # Remove from all neighbors
        primary_nbrs = list(self.TG.get(vid, []))

        for uid in primary_nbrs:
            if uid not in self.TG:
                continue
            TG_u = self.TG[uid]
            B_u = self.B[uid]

            if vid in B_u:
                # Case 1: o is a backup neighbor → just discard
                B_u.remove(vid)
            elif vid in TG_u:
                # Case 2: o is a primary neighbor → replace with backup
                TG_u.remove(vid)
                if B_u:
                    # Select best M from TG[u] ∪ B[u]
                    cands = list(TG_u) + list(B_u)
                    cand_vecs = [self._all_vectors[x] for x in cands if x in self._all_vectors and x in self._active]
                    u_vec = self._all_vectors.get(uid)
                    if u_vec is None:
                        continue
                    new_TG = self._select_nbrs(u_vec, cand_vecs, self.M)
                    new_TG_ids = [n.id for n in new_TG]
                    self.TG[uid] = new_TG_ids
                    # Update backup: remove all that are now in TG
                    self.B[uid] = [x for x in B_u if x not in new_TG_ids and x in self._active]
                    # Record new version
                    self.hist[uid].append(NeighborListVersion(t, list(new_TG_ids)))
                else:
                    # Backup empty: re-search from scratch
                    u_vec = self._all_vectors.get(uid)
                    if u_vec is None:
                        continue
                    cand_nodes = self._search(u_vec.data, self.M_prime, exclude={vid})
                    all_cands = [self._all_vectors[nid] for _, nid in cand_nodes
                                 if nid in self._all_vectors and nid in self._active and nid != vid]
                    selected_2M = self._select_nbrs(u_vec, all_cands, 2 * self.M)
                    new_TG = selected_2M[:self.M]
                    new_B = selected_2M[self.M:]
                    self.TG[uid] = [n.id for n in new_TG]
                    self.B[uid] = [n.id for n in new_B]
                    self.hist[uid].append(NeighborListVersion(t, list(self.TG[uid])))

        # Clean up TG/B (keep hist for historical queries)
        del self.TG[vid]
        del self.B[vid]

        # Update entry point if expired
        if self._entry_point == vid:
            if self._active:
                self._entry_point = next(iter(self._active))
            else:
                self._entry_point = None

    def search_at(self, query: TANNSQuery, ef: Optional[int] = None) -> List[Tuple[float, int]]:
        """
        TANNS query: find k approximate nearest neighbors valid at query.timestamp.
        
        This is Algorithm 1 adapted for timestamp graph:
        - When visiting a node u, its neighbors are determined by the timestamp.
        - Binary search on historic neighbor list to find neighbors at ts.
        
        Returns list of (distance, vector_id) sorted ascending.
        """
        ts = query.timestamp
        k = query.k
        q = query.query_vector
        if ef is None:
            ef = max(50, k)

        if self._entry_point is None and not self._all_vectors:
            return []

        # Find a valid entry point at timestamp ts
        ep = self._find_entry_at(ts)
        if ep is None:
            return []

        visited: Set[int] = {ep}
        pool: List[Tuple[float, int]] = []
        ann: List[Tuple[float, int]] = []  # max-heap as (-dist, id)

        ep_data = self._all_vectors[ep].data
        d_ep = self.distance_fn(q, ep_data)
        heapq.heappush(pool, (d_ep, ep))
        heapq.heappush(ann, (-d_ep, ep))

        while pool:
            d_u, u_id = heapq.heappop(pool)
            worst_d = -ann[0][0] if ann else float("inf")
            if d_u > worst_d:
                break

            # Get neighbors of u at timestamp ts
            nbrs = self._get_neighbors_at(u_id, ts)
            for nb_id in nbrs:
                if nb_id in visited:
                    continue
                # Check validity at ts
                nb_vec = self._all_vectors.get(nb_id)
                if nb_vec is None or not nb_vec.is_valid_at(ts):
                    continue
                visited.add(nb_id)
                d_nb = self.distance_fn(q, nb_vec.data)
                worst_d = -ann[0][0] if ann else float("inf")
                if len(ann) < ef or d_nb < worst_d:
                    heapq.heappush(pool, (d_nb, nb_id))
                    heapq.heappush(ann, (-d_nb, nb_id))
                    if len(ann) > ef:
                        heapq.heappop(ann)

        # Filter to only valid vectors and return top-k
        results = []
        for neg_d, vid in sorted(ann, key=lambda x: -x[0]):
            v = self._all_vectors.get(vid)
            if v is not None and v.is_valid_at(ts):
                results.append((-neg_d, vid))
        results.sort(key=lambda x: x[0])
        return results[:k]

    def get_neighbors_at(self, vid: int, ts: int) -> List[int]:
        """Public method to get neighbors of vid at timestamp ts (for testing)."""
        return self._get_neighbors_at(vid, ts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _search(
        self,
        query: np.ndarray,
        ef: int,
        exclude: Optional[Set[int]] = None,
    ) -> List[Tuple[float, int]]:
        """
        Greedy search in the CURRENT timestamp graph (for construction).
        Returns list of (dist, vid) sorted by dist.
        """
        if self._entry_point is None or not self._active:
            return []
        if exclude is None:
            exclude = set()

        ep = self._entry_point
        if ep in exclude or ep not in self._active:
            # Find any valid ep
            found = False
            for vid in self._active:
                if vid not in exclude:
                    ep = vid
                    found = True
                    break
            if not found:
                return []

        visited: Set[int] = {ep}
        pool: List[Tuple[float, int]] = []
        ann: List[Tuple[float, int]] = []

        d_ep = self.distance_fn(query, self._all_vectors[ep].data)
        heapq.heappush(pool, (d_ep, ep))
        heapq.heappush(ann, (-d_ep, ep))

        while pool:
            d_u, u_id = heapq.heappop(pool)
            worst_d = -ann[0][0] if ann else float("inf")
            if d_u > worst_d:
                break

            for nb_id in self.TG.get(u_id, []):
                if nb_id in visited or nb_id in exclude or nb_id not in self._active:
                    continue
                visited.add(nb_id)
                d_nb = self.distance_fn(query, self._all_vectors[nb_id].data)
                worst_d = -ann[0][0] if ann else float("inf")
                if len(ann) < ef or d_nb < worst_d:
                    heapq.heappush(pool, (d_nb, nb_id))
                    heapq.heappush(ann, (-d_nb, nb_id))
                    if len(ann) > ef:
                        heapq.heappop(ann)

        results = [(-neg_d, vid) for neg_d, vid in ann]
        results.sort(key=lambda x: x[0])
        return results

    def _select_nbrs(
        self,
        origin: Vector,
        candidates: List[Vector],
        M: int,
    ) -> List[Vector]:
        """
        Heuristic neighbor selection (Select-Nbrs, Algorithm 2 lines 10-17).
        Excludes dominated candidates.
        """
        scored = sorted(candidates, key=lambda v: self.distance_fn(origin.data, v.data))
        selected: List[Vector] = []

        for cand in scored:
            if len(selected) >= M:
                break
            d_oc = self.distance_fn(origin.data, cand.data)
            dominated = False
            for sel in selected:
                d_os = self.distance_fn(origin.data, sel.data)
                d_cs = self.distance_fn(cand.data, sel.data)
                if d_os < d_oc and d_cs < d_oc:
                    dominated = True
                    break
            if not dominated:
                selected.append(cand)

        return selected[:M]

    def _should_add_primary(self, uid: int, new_vid: int) -> bool:
        u_vec = self._all_vectors.get(uid)
        new_vec = self._all_vectors.get(new_vid)
        if u_vec is None or new_vec is None:
            return False

        TG_u = self.TG.get(uid, [])
        d_new = self.distance_fn(u_vec.data, new_vec.data)

        if len(TG_u) < self.M:
            return True  # Room available

        furthest = self._furthest(uid, TG_u)
        if furthest is None:
            return True
        d_furthest = self.distance_fn(u_vec.data, self._all_vectors[furthest].data)
        if d_new >= d_furthest:
            return False

        # Check domination
        for sel_id in TG_u:
            sel_vec = self._all_vectors.get(sel_id)
            if sel_vec is None:
                continue
            d_os = self.distance_fn(u_vec.data, sel_vec.data)
            d_cs = self.distance_fn(new_vec.data, sel_vec.data)
            if d_os < d_new and d_cs < d_new:
                return False

        return True

    def _furthest(self, origin_id: int, nbr_ids: List[int]) -> Optional[int]:
        origin_vec = self._all_vectors.get(origin_id)
        if origin_vec is None or not nbr_ids:
            return None
        worst_d = -1.0
        worst_id = None
        for nid in nbr_ids:
            v = self._all_vectors.get(nid)
            if v is None:
                continue
            d = self.distance_fn(origin_vec.data, v.data)
            if d > worst_d:
                worst_d = d
                worst_id = nid
        return worst_id

    def _get_neighbors_at(self, vid: int, ts: int) -> List[int]:
        """
        Get the neighbor list of node vid at timestamp ts via binary search.
        O(log N).
        """
        hist = self.hist.get(vid, [])
        if not hist:
            return []

        lo, hi = 0, len(hist) - 1
        result_idx = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if hist[mid].timestamp <= ts:
                result_idx = mid
                lo = mid + 1
            else:
                hi = mid - 1

        if result_idx == -1:
            return []
        return list(hist[result_idx].neighbors)

    def _find_entry_at(self, ts: int) -> Optional[int]:
        """Find a valid entry point at timestamp ts."""
        # Scan all historical vectors for one valid at ts with a known history
        for vid, vec in self._all_vectors.items():
            if vec.is_valid_at(ts) and vid in self.hist and self.hist[vid]:
                return vid
        return None

    def build_from_stream(self, events: List[Tuple[int, str, Vector]]) -> None:
        """
        Build the timestamp graph from a stream of events.
        
        Parameters
        ----------
        events : list of (timestamp, event_type, vector)
            event_type is 'insert' or 'expire'
            Events must be sorted by timestamp.
        """
        for ts, etype, vec in sorted(events, key=lambda x: x[0]):
            if etype == "insert":
                self.insert(vec)
            elif etype == "expire":
                self.expire(vec)
