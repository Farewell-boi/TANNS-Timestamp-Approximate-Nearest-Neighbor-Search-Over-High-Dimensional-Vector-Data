"""
Unit tests for TANNS components.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
import numpy as np
from tanns.data_types import Vector, TANNSQuery
from tanns.distance import euclidean_distance, cosine_distance
from tanns.hnsw import HNSW
from tanns.timestamp_graph import TimestampGraph
from tanns.historic_neighbor_tree import HistoricNeighborTree
from tanns.compressed_timestamp_graph import CompressedTimestampGraph
from experiments.data_generator import (
    generate_dataset, generate_queries, compute_ground_truth,
    compute_recall, build_events,
)


class TestDistance(unittest.TestCase):
    def test_euclidean(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        d = euclidean_distance(a, b)
        self.assertAlmostEqual(d, 2.0)  # squared L2 = 1+1 = 2

    def test_cosine(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        d = cosine_distance(a, b)
        self.assertAlmostEqual(d, 1.0)  # orthogonal → cosine dist = 1

    def test_cosine_identical(self):
        a = np.array([1.0, 1.0])
        d = cosine_distance(a, a)
        self.assertAlmostEqual(d, 0.0)


class TestVector(unittest.TestCase):
    def test_validity(self):
        v = Vector(id=0, data=np.zeros(4), start=3, end=7)
        self.assertFalse(v.is_valid_at(2))
        self.assertTrue(v.is_valid_at(3))
        self.assertTrue(v.is_valid_at(6))
        self.assertFalse(v.is_valid_at(7))
        self.assertFalse(v.is_valid_at(10))

    def test_validity_open_ended(self):
        v = Vector(id=1, data=np.zeros(4), start=5, end=None)
        self.assertFalse(v.is_valid_at(4))
        self.assertTrue(v.is_valid_at(5))
        self.assertTrue(v.is_valid_at(1000))


class TestHNSW(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.dim = 16
        self.n = 200
        self.hnsw = HNSW(M=8, ef_construction=50, ef_search=20)
        data = np.random.randn(self.n, self.dim).astype(np.float32)
        self.vecs = [Vector(id=i, data=data[i], start=1, end=None) for i in range(self.n)]
        for v in self.vecs:
            self.hnsw.add(v)

    def test_search_returns_k_results(self):
        q = np.random.randn(self.dim).astype(np.float32)
        results = self.hnsw.search(q, k=10)
        self.assertEqual(len(results), 10)

    def test_search_accuracy(self):
        """Recall should be reasonably high for synthetic data."""
        from experiments.data_generator import compute_recall
        k = 5
        n_test = 20
        recalls = []
        for _ in range(n_test):
            q = np.random.randn(self.dim).astype(np.float32)
            # Ground truth: brute force
            dists = [(euclidean_distance(q, v.data), v.id) for v in self.vecs]
            dists.sort()
            gt = [vid for _, vid in dists[:k]]
            # HNSW result
            res = self.hnsw.search(q, k=k, ef=50)
            pred = [vid for _, vid in res]
            hits = len(set(pred) & set(gt))
            recalls.append(hits / k)
        avg_recall = sum(recalls) / len(recalls)
        self.assertGreater(avg_recall, 0.7, f"HNSW recall too low: {avg_recall:.3f}")


class TestHistoricNeighborTree(unittest.TestCase):
    def _make_vec(self, vid, start, end=None):
        return Vector(id=vid, data=np.zeros(4), start=start, end=end)

    def test_reconstruct_empty(self):
        hnt = HistoricNeighborTree(mu=4)
        result = hnt.reconstruct(ts=5)
        self.assertEqual(result, [])

    def test_append_and_reconstruct(self):
        hnt = HistoricNeighborTree(mu=4)
        v1 = self._make_vec(1, start=1, end=10)
        v2 = self._make_vec(2, start=2, end=10)
        v3 = self._make_vec(3, start=3, end=10)

        # Append initial list
        hnt.append([v1, v2, v3])

        # Reconstruct should return all valid at ts=5
        result = hnt.reconstruct(ts=5)
        self.assertIn(1, result)
        self.assertIn(2, result)
        self.assertIn(3, result)

    def test_reconstruct_after_removal(self):
        hnt = HistoricNeighborTree(mu=4)
        v1 = self._make_vec(1, start=1, end=10)
        v2 = self._make_vec(2, start=2, end=10)
        v3 = self._make_vec(3, start=3, end=10)

        hnt.append([v1, v2, v3])
        # Remove v1 from current list (simulate expiry)
        hnt.append([v2, v3])

        # At ts=5: all three were valid → should reconstruct all from HNT
        result = hnt.reconstruct(ts=5)
        # v1 should be in HNT (removed from L_now)
        self.assertIn(1, result)


class TestTimestampGraph(unittest.TestCase):
    def _build_simple_tg(self):
        """Build a small timestamp graph for testing."""
        np.random.seed(42)
        dim = 16
        n = 50
        T = 2 * n

        rng = np.random.default_rng(42)
        data = rng.standard_normal((n, dim)).astype(np.float32)

        vectors = []
        for i in range(n):
            s = int(rng.integers(1, T // 2))
            e = int(rng.integers(T // 2, T))
            vectors.append(Vector(id=i, data=data[i], start=s, end=e))

        tg = TimestampGraph(M=8, M_prime=50)
        tg._expired = {}
        events = build_events(vectors)
        tg.build_from_stream(events)
        return tg, vectors, T

    def test_search_returns_results(self):
        tg, vectors, T = self._build_simple_tg()
        q_data = np.random.randn(16).astype(np.float32)
        query = TANNSQuery(query_vector=q_data, timestamp=T // 2, k=5)
        results = tg.search_at(query, ef=20)
        self.assertIsInstance(results, list)
        # Results may be empty if no valid vectors at that ts

    def test_recall_reasonable(self):
        tg, vectors, T = self._build_simple_tg()
        queries = generate_queries(T=T, n_queries=50, k=5, dim=16)
        gt = compute_ground_truth(vectors, queries)
        preds = []
        for q in queries:
            res = tg.search_at(q, ef=50)
            preds.append([vid for _, vid in res])
        recall = compute_recall(preds, gt)
        print(f"\n  TimestampGraph recall: {recall:.3f}")
        self.assertGreater(recall, 0.5, f"TG recall too low: {recall:.3f}")


class TestCompressedTimestampGraph(unittest.TestCase):
    def _build_small_ctg(self):
        np.random.seed(42)
        dim = 16
        n = 50
        T = 2 * n

        rng = np.random.default_rng(42)
        data = rng.standard_normal((n, dim)).astype(np.float32)

        vectors = []
        for i in range(n):
            s = int(rng.integers(1, T // 2))
            e = int(rng.integers(T // 2, T))
            vectors.append(Vector(id=i, data=data[i], start=s, end=e))

        ctg = CompressedTimestampGraph(M=8, M_prime=50, mu=4)
        events = build_events(vectors)
        ctg.build_from_stream(events)
        return ctg, vectors, T

    def test_search_returns_results(self):
        ctg, vectors, T = self._build_small_ctg()
        q_data = np.random.randn(16).astype(np.float32)
        query = TANNSQuery(query_vector=q_data, timestamp=T // 2, k=5)
        results = ctg.search_at(query, ef=20)
        self.assertIsInstance(results, list)

    def test_recall_reasonable(self):
        ctg, vectors, T = self._build_small_ctg()
        queries = generate_queries(T=T, n_queries=50, k=5, dim=16)
        gt = compute_ground_truth(vectors, queries)
        preds = []
        for q in queries:
            res = ctg.search_at(q, ef=50)
            preds.append([vid for _, vid in res])
        recall = compute_recall(preds, gt)
        print(f"\n  CompressedTimestampGraph recall: {recall:.3f}")
        self.assertGreater(recall, 0.4, f"CTG recall too low: {recall:.3f}")

    def test_memory_usage(self):
        ctg, vectors, T = self._build_small_ctg()
        mem = ctg.memory_usage()
        self.assertIn("total", mem)
        self.assertGreaterEqual(mem["total"], 0)
        print(f"\n  CTG memory: {mem}")


class TestDataGenerator(unittest.TestCase):
    def test_generate_dataset_shapes(self):
        vectors, T = generate_dataset(n=100, dim=32, pattern="uniform", seed=0)
        self.assertEqual(len(vectors), 100)
        self.assertEqual(vectors[0].data.shape, (32,))
        self.assertGreater(T, 0)

    def test_all_patterns(self):
        for pattern in ["short", "long", "mixed", "uniform"]:
            vectors, T = generate_dataset(n=50, dim=8, pattern=pattern, seed=0)
            self.assertEqual(len(vectors), 50)

    def test_validity_constraints(self):
        vectors, T = generate_dataset(n=100, dim=16, seed=0)
        for v in vectors:
            self.assertGreaterEqual(v.start, 1)
            self.assertLessEqual(v.start, T)
            if v.end is not None:
                self.assertGreater(v.end, v.start)
                self.assertLessEqual(v.end, T)


if __name__ == "__main__":
    unittest.main(verbosity=2)
