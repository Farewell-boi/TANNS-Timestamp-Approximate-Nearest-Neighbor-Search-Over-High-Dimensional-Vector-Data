"""
Compressed Timestamp Graph (Section V of the paper).

Integrates the Historic Neighbor Tree (HNT) into the Timestamp Graph to
compress neighbor lists to O(MN) space (same as a single-timestamp HNSW).

Key difference from TimestampGraph:
  - Instead of storing ALL versioned neighbor lists per point,
    we only keep L_now (current list) + a balanced HNT for each point.
  - Reconstructing L_t at query time via Algorithm 5 (O(log n + M_r)).
  - Appending neighbor changes via Algorithm 6 (O(log n) per update).

Space complexity: O(MN)
Time complexity for search: O(log²N)
Time complexity for update: O(log²N)
"""

import heapq
import numpy as np
from typing import List, Dict, Set, Optional, Tuple, Callable

from .data_types import Vector, TANNSQuery
from .distance import euclidean_distance
from .historic_neighbor_tree import HistoricNeighborTree


class CompressedTimestampGraph:
    """
    Compressed Timestamp Graph using Historic Neighbor Trees.

    For each point u, instead of maintaining a full list of versioned
    neighbor snapshots, we maintain:
      - TG[u]: current primary neighbor list
      - B[u]: current backup neighbor list
      - HNT[u]: historic neighbor tree for compressed history
      - L_now[u]: current neighbor list (reference into HNT)

    Parameters
    ----------
    M : int
        Primary neighbor count per node. Default: 16.
    M_prime : int
        Candidate count during construction. Default: 200.
    mu : int
        Leaf node size for HNT. Default: 8.
    distance_fn : Callable
        Distance function. Default: squared Euclidean.
    """

    def __init__(
        self,
        M: int = 16,
        M_prime: int = 200,
        mu: int = 8,
        distance_fn: Callable = euclidean_distance,
    ):
        self.M = M
        self.M_prime = M_prime
        self.mu = mu
        self.distance_fn = distance_fn

        # Current primary/backup neighbor lists
        self.TG: Dict[int, List[int]] = {}
        self.B: Dict[int, List[int]] = {}
        # Historic Neighbor Tree per point
        self.HNT: Dict[int, HistoricNeighborTree] = {}
        # Current neighbor list per point (L_now for HNT)
        self._L_now: Dict[int, List[Vector]] = {}
        # ALL vectors (current + expired): needed for historical queries
        self._all_vectors: Dict[int, Vector] = {}
        # Active vector ids
        self._active: Set[int] = set()
        # Current active vectors (alias for compatibility)
        self.vectors: Dict[int, Vector] = {}
        # Archived expired vectors
        self._expired: Dict[int, Vector] = {}
        # Entry point
        self._entry_point: Optional[int] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def insert(self, vec: Vector) -> None:
        """Insert a vector (Algorithm 3 + HNT initialisation)."""
        t = vec.start
        vid = vec.id
        self._all_vectors[vid] = vec
        self.vectors[vid] = vec
        self._active.add(vid)
        self.TG[vid] = []
        self.B[vid] = []
        self.HNT[vid] = HistoricNeighborTree(mu=self.mu)
        self._L_now[vid] = []

        if self._entry_point is None:
            self._entry_point = vid
            return

        # Search for candidates (use active vectors only)
        cand_nodes = self._search(vec.data, self.M_prime)
        all_cands = [self._all_vectors[nid] for _, nid in cand_nodes
                     if nid in self._all_vectors and nid in self._active]
        selected_2M = self._select_nbrs(vec, all_cands, 2 * self.M)

        primary = selected_2M[:self.M]
        backup = selected_2M[self.M:]
        self.TG[vid] = [n.id for n in primary]
        self.B[vid] = [n.id for n in backup]

        # Initialise HNT with initial neighbor list
        self._update_hnt(vid, [self._all_vectors[nid] for nid in self.TG[vid] if nid in self._all_vectors])

        # Update primary neighbors
        for nbr_vec in primary:
            uid = nbr_vec.id
            if self._should_add_primary(uid, vid):
                if len(self.TG[uid]) >= self.M:
                    furthest = self._furthest(uid, self.TG[uid])
                    if furthest is not None:
                        self.TG[uid].remove(furthest)
                        self.B[uid].append(furthest)
                        if len(self.B[uid]) > self.M:
                            self.B[uid].remove(self._furthest(uid, self.B[uid]))
                self.TG[uid].append(vid)
                self._update_hnt(uid, [self._all_vectors[nid] for nid in self.TG[uid] if nid in self._all_vectors])
            else:
                self.B[uid].append(vid)
                if len(self.B[uid]) > self.M:
                    self.B[uid].remove(self._furthest(uid, self.B[uid]))

    def expire(self, vec: Vector) -> None:
        """Expire a vector (Algorithm 4 + HNT update)."""
        t = vec.end
        vid = vec.id

        if vid not in self.TG:
            return

        primary_nbrs = list(self.TG.get(vid, []))
        self._expired[vid] = vec
        if vid in self.vectors:
            del self.vectors[vid]
        self._active.discard(vid)

        for uid in primary_nbrs:
            if uid not in self.TG:
                continue
            TG_u = self.TG[uid]
            B_u = self.B[uid]

            if vid in B_u:
                B_u.remove(vid)
            elif vid in TG_u:
                TG_u.remove(vid)
                if B_u:
                    cands = [x for x in list(TG_u) + list(B_u) if x in self._active]
                    cand_vecs = [self._all_vectors[x] for x in cands if x in self._all_vectors]
                    u_vec = self._all_vectors.get(uid)
                    if u_vec is None:
                        continue
                    new_TG = self._select_nbrs(u_vec, cand_vecs, self.M)
                    new_TG_ids = [n.id for n in new_TG]
                    self.TG[uid] = new_TG_ids
                    self.B[uid] = [x for x in B_u if x not in new_TG_ids and x in self._active]
                    self._update_hnt(uid, [self._all_vectors[nid] for nid in new_TG_ids if nid in self._all_vectors])
                else:
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
                    self._update_hnt(uid, [self._all_vectors[nid] for nid in self.TG[uid] if nid in self._all_vectors])

        del self.TG[vid]
        del self.B[vid]

        if self._entry_point == vid:
            self._entry_point = next(iter(self._active)) if self._active else None

    def search_at(self, query: TANNSQuery, ef: Optional[int] = None) -> List[Tuple[float, int]]:
        """
        TANNS search using the compressed timestamp graph.
        Neighbor retrieval uses HNT reconstruction (Algorithm 5).
        """
        ts = query.timestamp
        k = query.k
        q = query.query_vector
        if ef is None:
            ef = max(50, k)

        if self._entry_point is None and not self._all_vectors:
            return []

        ep = self._find_entry_at(ts)
        if ep is None:
            return []

        visited: Set[int] = {ep}
        pool: List[Tuple[float, int]] = []
        ann: List[Tuple[float, int]] = []

        ep_data = self._get_vec_data(ep)
        if ep_data is None:
            return []
        d_ep = self.distance_fn(q, ep_data)
        heapq.heappush(pool, (d_ep, ep))
        heapq.heappush(ann, (-d_ep, ep))

        while pool:
            d_u, u_id = heapq.heappop(pool)
            worst_d = -ann[0][0] if ann else float("inf")
            if d_u > worst_d:
                break

            # Get neighbors of u at ts via HNT reconstruction
            nbrs = self._get_neighbors_at(u_id, ts)
            for nb_id in nbrs:
                if nb_id in visited:
                    continue
                nb_data = self._get_vec_data(nb_id)
                if nb_data is None:
                    continue
                nb_vec = self._get_vector(nb_id)
                if nb_vec is None or not nb_vec.is_valid_at(ts):
                    continue
                visited.add(nb_id)
                d_nb = self.distance_fn(q, nb_data)
                worst_d = -ann[0][0] if ann else float("inf")
                if len(ann) < ef or d_nb < worst_d:
                    heapq.heappush(pool, (d_nb, nb_id))
                    heapq.heappush(ann, (-d_nb, nb_id))
                    if len(ann) > ef:
                        heapq.heappop(ann)

        results = []
        for neg_d, vid in ann:
            vec = self._get_vector(vid)
            if vec is not None and vec.is_valid_at(ts):
                results.append((-neg_d, vid))
        results.sort(key=lambda x: x[0])
        return results[:k]

    def build_from_stream(self, events: List[Tuple[int, str, Vector]]) -> None:
        """Build from a stream of (timestamp, event_type, vector) events."""
        for ts, etype, vec in sorted(events, key=lambda x: x[0]):
            if etype == "insert":
                self.insert(vec)
            elif etype == "expire":
                self.expire(vec)

    def memory_usage(self) -> Dict[str, int]:
        """
        Estimate memory usage (number of vector references stored).
        Returns dict with breakdown.
        """
        current_nbrs = sum(len(v) for v in self.TG.values())
        current_backup = sum(len(v) for v in self.B.values())
        hnt_total = sum(hnt.total_points() for hnt in self.HNT.values())
        l_now_total = sum(len(l) for l in self._L_now.values())
        return {
            "current_primary": current_nbrs,
            "current_backup": current_backup,
            "hnt_stored": hnt_total,
            "l_now": l_now_total,
            "total": current_nbrs + current_backup + hnt_total,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_hnt(self, vid: int, new_nbr_vecs: List[Vector]) -> None:
        """Update the HNT for point vid with the new neighbor list."""
        if vid not in self.HNT:
            self.HNT[vid] = HistoricNeighborTree(mu=self.mu)
        self.HNT[vid].append(new_nbr_vecs)
        self._L_now[vid] = list(new_nbr_vecs)

    def _get_neighbors_at(self, vid: int, ts: int) -> List[int]:
        """Get neighbor list at ts: use HNT reconstruction (Algorithm 5)."""
        hnt = self.HNT.get(vid)
        if hnt is None:
            return self.TG.get(vid, [])
        return hnt.reconstruct(ts)

    def _get_vector(self, vid: int) -> Optional[Vector]:
        return self._all_vectors.get(vid)

    def _get_vec_data(self, vid: int) -> Optional[np.ndarray]:
        v = self._all_vectors.get(vid)
        return v.data if v is not None else None

    def _find_entry_at(self, ts: int) -> Optional[int]:
        # Try current entry point
        if self._entry_point is not None:
            v = self._all_vectors.get(self._entry_point)
            if v is not None and v.is_valid_at(ts):
                return self._entry_point
        # Fallback: scan all vectors
        for vid, v in self._all_vectors.items():
            if v.is_valid_at(ts) and vid in self.HNT:
                return vid
        return None

    def _search(
        self,
        query: np.ndarray,
        ef: int,
        exclude: Optional[Set[int]] = None,
    ) -> List[Tuple[float, int]]:
        """Greedy search in current graph for construction."""
        if self._entry_point is None or not self._active:
            return []
        if exclude is None:
            exclude = set()

        ep = self._entry_point
        if ep in exclude or ep not in self._active:
            ep = None
            for vid in self._active:
                if vid not in exclude:
                    ep = vid
                    break
            if ep is None:
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

    def _select_nbrs(self, origin: Vector, candidates: List[Vector], M: int) -> List[Vector]:
        """Heuristic neighbor selection (same as in TimestampGraph)."""
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
            return True
        furthest = self._furthest(uid, TG_u)
        if furthest is None:
            return True
        d_furthest = self.distance_fn(u_vec.data, self._all_vectors[furthest].data)
        if d_new >= d_furthest:
            return False
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
