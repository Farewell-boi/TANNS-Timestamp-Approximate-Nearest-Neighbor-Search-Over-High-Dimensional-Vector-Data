"""
Historic Neighbor Tree (HNT) for compressing neighbor lists in the timestamp graph.

Section V of the paper:
  "Timestamp Approximate Nearest Neighbor Search over High-Dimensional Vector Data"
  ICDE 2025.

Key ideas:
  - Inspired by interval trees but uses bottom-up balanced construction.
  - Each internal node has a timestamp t and stores points valid at t that
    are NOT valid at any ancestor's timestamp.
  - Leaf nodes store recent points (size bounded by µ).
  - L_now: current neighbor list (sorted by start time).
  - A point is stored in the highest ancestor where it is valid.

Algorithms:
  - Algorithm 5: Neighbor List Reconstruction (search at timestamp t)
  - Algorithm 6: Append Neighbor List (dynamic construction from new L_t)

Space complexity: O(n) where n = total number of points ever in the neighbor list.
Reconstruction time: O(log n + M_r) where M_r = size of result.
"""

from typing import List, Optional, Dict, Set, Tuple
from .data_types import Vector


class HNTNode:
    """
    A node in the Historic Neighbor Tree.
    
    Each node stores:
      - timestamp: the timestamp associated with this internal node (None for leaf)
      - points: sorted lists of (start_time, point_id) and (end_time, point_id)
      - left, right child pointers
      - is_leaf flag
    """
    def __init__(self, timestamp: Optional[int] = None, is_leaf: bool = False):
        self.timestamp: Optional[int] = timestamp
        self.is_leaf: bool = is_leaf
        self.left: Optional["HNTNode"] = None
        self.right: Optional["HNTNode"] = None
        # Points stored in this node: dict {point_id -> Vector}
        self.points: Dict[int, Vector] = {}
        # Sorted by start time (ascending): list of (start, pid)
        self.points_by_start: List[Tuple[int, int]] = []
        # Sorted by end time (ascending): list of (end, pid)
        self.points_by_end: List[Tuple[int, int]] = []
        # Parent node reference (for lifting during adjustment)
        self.parent: Optional["HNTNode"] = None

    def add_point(self, vec: Vector) -> None:
        """Add a point to this node's storage."""
        if vec.id in self.points:
            return
        self.points[vec.id] = vec
        # Insert into start-sorted list (keep sorted)
        import bisect
        bisect.insort(self.points_by_start, (vec.start, vec.id))
        if vec.end is not None:
            bisect.insort(self.points_by_end, (vec.end, vec.id))

    def remove_point(self, pid: int) -> Optional[Vector]:
        """Remove a point from this node."""
        if pid not in self.points:
            return None
        import bisect
        vec = self.points.pop(pid)
        # Remove from start-sorted list
        idx = bisect.bisect_left(self.points_by_start, (vec.start, pid))
        while idx < len(self.points_by_start) and self.points_by_start[idx] != (vec.start, pid):
            idx += 1
        if idx < len(self.points_by_start):
            self.points_by_start.pop(idx)
        # Remove from end-sorted list
        if vec.end is not None:
            idx = bisect.bisect_left(self.points_by_end, (vec.end, pid))
            while idx < len(self.points_by_end) and self.points_by_end[idx] != (vec.end, pid):
                idx += 1
            if idx < len(self.points_by_end):
                self.points_by_end.pop(idx)
        return vec

    def get_valid_at(self, ts: int) -> List[int]:
        """
        Return IDs of all points in this node valid at timestamp ts.
        Uses the sorted lists to stop early (O(valid + 1) per node).
        """
        import bisect
        valid = []
        if ts < (self.timestamp or float("inf")):
            # t < n.t: scan by start time
            # Valid points have start <= ts AND (end > ts or end is None)
            # Points sorted by start — stop at first point with start > ts
            for start, pid in self.points_by_start:
                if start > ts:
                    break
                vec = self.points[pid]
                if vec.end is None or vec.end > ts:
                    valid.append(pid)
        else:
            # t > n.t: scan by end time
            # Valid points have start <= ts AND (end > ts or end is None)
            for end, pid in self.points_by_end:
                if end <= ts:
                    continue
                vec = self.points[pid]
                if vec.start <= ts:
                    valid.append(pid)
            # Also include points with end=None
            for pid, vec in self.points.items():
                if vec.end is None and vec.start <= ts:
                    valid.append(pid)
            valid = list(set(valid))
        return valid

    def size(self) -> int:
        return len(self.points)

    def __repr__(self):
        return (f"HNTNode(t={self.timestamp}, leaf={self.is_leaf}, "
                f"n_points={len(self.points)})")


class HistoricNeighborTree:
    """
    A balanced binary tree for compressing neighbor lists.

    For each point o in the timestamp graph, we maintain:
      - L_now: current neighbor list (points not yet in HNT)
      - root: root of the HNT

    Parameters
    ----------
    mu : int
        Leaf node size threshold. Default: 8 (paper default).
    """

    def __init__(self, mu: int = 8):
        self.mu = mu
        self.root: Optional[HNTNode] = None
        # Current neighbor list (sorted by start time)
        self.L_now: List[Vector] = []  # sorted by start time ascending
        # Map from point id to the node it lives in
        self._node_of: Dict[int, HNTNode] = {}

    # ------------------------------------------------------------------
    # Algorithm 5: Neighbor List Reconstruction
    # ------------------------------------------------------------------

    def reconstruct(self, ts: int) -> List[int]:
        """
        Reconstruct the neighbor list at timestamp ts (Algorithm 5).
        
        Returns list of point IDs valid at ts.
        Time complexity: O(log n + M_r) where n = total points ever, M_r = result size.
        """
        L_t: List[int] = []

        # Step 1: Scan L_now (sorted by start time ascending)
        for vec in self.L_now:
            if vec.start > ts:
                break  # All subsequent have start > ts as well
            if vec.is_valid_at(ts):
                L_t.append(vec.id)

        if self.root is None:
            return L_t

        # Step 2: Traverse HNT from root
        node = self.root
        while node is not None:
            # Add valid points at this node
            for pid in node.get_valid_at(ts):
                if pid not in set(L_t):
                    L_t.append(pid)

            if node.is_leaf:
                break
            if node.timestamp is None:
                break
            if ts == node.timestamp:
                break  # All lower nodes have no valid points at ts
            elif ts < node.timestamp:
                node = node.left
            else:
                node = node.right

        return L_t

    # ------------------------------------------------------------------
    # Algorithm 6: Append Neighbor List
    # ------------------------------------------------------------------

    def append(self, L_t: List[Vector]) -> None:
        """
        Update the HNT with the latest neighbor list L_t (Algorithm 6).
        
        L_t is the CURRENT (new) neighbor list from the timestamp graph.
        Points in L_now but NOT in L_t have just been removed → process them.
        """
        L_t_ids = {v.id for v in L_t}
        L_now_ids = {v.id for v in self.L_now}

        # Points removed from current neighbor list (L_now \ L_t)
        removed = [v for v in self.L_now if v.id not in L_t_ids]

        for vec in removed:
            pid = vec.id
            if pid not in self._node_of:
                # New point (not yet in HNT): add it
                self._add_new_point(vec)
            else:
                # Existing point: adjust its position
                self._adjust_point(vec)

        # Update L_now ← L_t (sorted by start time)
        self.L_now = sorted(L_t, key=lambda v: v.start)

    def _add_new_point(self, vec: Vector) -> None:
        """
        Add a new point to the HNT (Algorithm 6, lines 3-12).
        Insert into the newest leaf node on the active path,
        as high as possible where the point is still valid.
        """
        if self.root is None:
            # Create first leaf node
            leaf = HNTNode(is_leaf=True)
            leaf.add_point(vec)
            self._node_of[vec.id] = leaf
            self.root = leaf
            return

        # Find newest leaf node (rightmost leaf)
        n_c = self._newest_leaf()
        # Try to place in the highest node on the path from root to n_c
        path = self._path_to(n_c)
        placed = False
        for node in path:
            if node is n_c or (node.timestamp is not None and vec.is_valid_at(node.timestamp)):
                node.add_point(vec)
                self._node_of[vec.id] = node
                placed = True
                break

        if not placed:
            # Place in leaf node n_c
            n_c.add_point(vec)
            self._node_of[vec.id] = n_c

        # Check if leaf node exceeds µ
        if n_c.size() > self.mu:
            self._split_leaf(n_c)

    def _adjust_point(self, vec: Vector) -> None:
        """
        Adjust an existing point: try to lift it as high as possible (Algorithm 6, lines 14-17).
        """
        pid = vec.id
        n_p = self._node_of.get(pid)
        if n_p is None:
            self._add_new_point(vec)
            return

        # Try to lift p from n_p toward root
        path = self._path_to(n_p)
        for node in path:
            if node is n_p:
                continue  # Skip current position
            if node.timestamp is not None and vec.is_valid_at(node.timestamp):
                # Move p to this higher node
                n_p.remove_point(pid)
                node.add_point(vec)
                self._node_of[pid] = node
                break
            # Also: if we passed a valid ancestor, stop
        # If no adjustment possible, leave it where it is

    def _split_leaf(self, leaf: HNTNode) -> None:
        """
        When a leaf exceeds µ points, split:
        - Find the maximal complete binary subtree containing leaf
        - Create an internal node with that subtree as left child
        - Create a new leaf as right sibling (Algorithm 6, lines 7-12)
        """
        # Find the maximal complete binary subtree containing leaf
        subtree_root = self._max_complete_subtree(leaf)
        parent_of_subtree = subtree_root.parent

        # The internal node's timestamp = the timestamp of the last inserted
        # point in the subtree (the highest start time in subtree's points)
        ts = self._max_start_in_subtree(subtree_root)
        if ts is None:
            ts = 0

        n_i = HNTNode(timestamp=ts, is_leaf=False)
        n_l = HNTNode(is_leaf=True)

        n_i.left = subtree_root
        n_i.right = n_l
        subtree_root.parent = n_i
        n_l.parent = n_i

        if parent_of_subtree is None:
            # subtree_root was the root
            self.root = n_i
            n_i.parent = None
        else:
            # Replace subtree_root in parent
            if parent_of_subtree.left is subtree_root:
                parent_of_subtree.left = n_i
            else:
                parent_of_subtree.right = n_i
            n_i.parent = parent_of_subtree

    def _newest_leaf(self) -> HNTNode:
        """Find the newest (rightmost) leaf node."""
        if self.root is None:
            leaf = HNTNode(is_leaf=True)
            self.root = leaf
            return leaf
        node = self.root
        while not node.is_leaf:
            if node.right is not None:
                node = node.right
            else:
                break
        return node

    def _path_to(self, target: HNTNode) -> List[HNTNode]:
        """Return path from root to target (inclusive), root first."""
        path = []
        path_set: Set[int] = set()

        def dfs(node: Optional[HNTNode]) -> bool:
            if node is None:
                return False
            path.append(node)
            if node is target:
                return True
            if dfs(node.left) or dfs(node.right):
                return True
            path.pop()
            return False

        dfs(self.root)
        return path

    def _max_complete_subtree(self, leaf: HNTNode) -> HNTNode:
        """
        Find the maximal complete binary subtree that contains `leaf`
        and is a left subtree of its parent (per Algorithm 6, line 8).
        """
        node = leaf
        while node.parent is not None:
            if not self._is_complete_subtree(node):
                return node
            parent = node.parent
            if parent.left is node:
                # Check if the left subtree is complete
                if self._is_complete_subtree(node):
                    node = parent
                else:
                    return node
            else:
                return node
        return node

    def _is_complete_subtree(self, node: HNTNode) -> bool:
        """Check if the subtree rooted at node is a complete binary tree."""
        if node is None:
            return True
        if node.is_leaf:
            return True
        left_h = self._height(node.left)
        right_h = self._height(node.right)
        if node.left is None or node.right is None:
            return False
        if not self._is_complete_subtree(node.left):
            return False
        if left_h == right_h:
            return self._is_complete_subtree(node.right)
        elif left_h == right_h + 1:
            # Left is one level deeper, left must be complete, right must be full
            return self._is_full_subtree(node.right)
        return False

    def _is_full_subtree(self, node: Optional[HNTNode]) -> bool:
        """Check if subtree is a full (perfect) binary tree."""
        if node is None:
            return True
        if node.is_leaf:
            return True
        if node.left is None and node.right is None:
            return True
        if node.left is None or node.right is None:
            return False
        return (self._height(node.left) == self._height(node.right)
                and self._is_full_subtree(node.left)
                and self._is_full_subtree(node.right))

    def _height(self, node: Optional[HNTNode]) -> int:
        """Height of subtree (0 for None, 1 for leaf)."""
        if node is None:
            return 0
        if node.is_leaf:
            return 1
        return 1 + max(self._height(node.left), self._height(node.right))

    def _max_start_in_subtree(self, node: Optional[HNTNode]) -> Optional[int]:
        """Find the maximum start timestamp of any point in the subtree."""
        if node is None:
            return None
        best = None
        for pid, vec in node.points.items():
            if best is None or vec.start > best:
                best = vec.start
        left_best = self._max_start_in_subtree(node.left)
        right_best = self._max_start_in_subtree(node.right)
        for v in [left_best, right_best]:
            if v is not None and (best is None or v > best):
                best = v
        return best

    def total_points(self) -> int:
        """Total number of points stored in the HNT (for space analysis)."""
        return self._count_subtree(self.root)

    def _count_subtree(self, node: Optional[HNTNode]) -> int:
        if node is None:
            return 0
        return len(node.points) + self._count_subtree(node.left) + self._count_subtree(node.right)
