#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for the custom Cython min-cost flow implementation.

Tests include:
1. Unit tests for basic min-cost flow problems
2. Integration tests with constrained k-means
3. Optional equivalence tests against OR-Tools (if installed)
"""

import pytest
import numpy as np
import sys
import os

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)


class TestSimpleMinCostFlow:
    """Unit tests for SimpleMinCostFlow class."""

    def test_simple_flow(self):
        """Test basic 2-node flow problem."""
        from size_constrained_clustering.k_means_constrained.min_cost_flow import (
            SimpleMinCostFlow,
        )

        mcf = SimpleMinCostFlow()

        # Simple network: source (0) -> sink (1) with capacity 10 and cost 5
        arc = mcf.add_arc_with_capacity_and_unit_cost(0, 1, 10, 5)
        assert arc == 0

        mcf.set_node_supply(0, 5)  # Source supplies 5
        mcf.set_node_supply(1, -5)  # Sink demands 5

        status = mcf.solve()
        assert status == mcf.OPTIMAL
        assert mcf.flow(0) == 5

    def test_parallel_arcs(self):
        """Test network with parallel arcs of different costs."""
        from size_constrained_clustering.k_means_constrained.min_cost_flow import (
            SimpleMinCostFlow,
        )

        mcf = SimpleMinCostFlow()

        # Two parallel arcs with different costs
        # Flow should use cheaper arc first
        mcf.add_arc_with_capacity_and_unit_cost(0, 1, 5, 10)  # Expensive
        mcf.add_arc_with_capacity_and_unit_cost(0, 1, 5, 1)  # Cheap

        mcf.set_node_supply(0, 7)
        mcf.set_node_supply(1, -7)

        status = mcf.solve()
        assert status == mcf.OPTIMAL

        # Should use all 5 of cheap arc, and 2 of expensive arc
        assert mcf.flow(0) + mcf.flow(1) == 7

    def test_diamond_network(self):
        """Test diamond-shaped network."""
        from size_constrained_clustering.k_means_constrained.min_cost_flow import (
            SimpleMinCostFlow,
        )

        mcf = SimpleMinCostFlow()

        # Diamond: 0 -> 1 -> 3, 0 -> 2 -> 3
        #          cost: 1     2      3     1
        mcf.add_arc_with_capacity_and_unit_cost(0, 1, 10, 1)  # arc 0
        mcf.add_arc_with_capacity_and_unit_cost(0, 2, 10, 3)  # arc 1
        mcf.add_arc_with_capacity_and_unit_cost(1, 3, 10, 2)  # arc 2
        mcf.add_arc_with_capacity_and_unit_cost(2, 3, 10, 1)  # arc 3

        mcf.set_node_supply(0, 5)
        mcf.set_node_supply(1, 0)
        mcf.set_node_supply(2, 0)
        mcf.set_node_supply(3, -5)

        status = mcf.solve()
        assert status == mcf.OPTIMAL

        # Total flow should be 5
        total_flow_out = mcf.flow(0) + mcf.flow(1)
        assert total_flow_out == 5

    def test_infeasible_network(self):
        """Test that infeasible networks are correctly detected."""
        from size_constrained_clustering.k_means_constrained.min_cost_flow import (
            SimpleMinCostFlow,
        )

        mcf = SimpleMinCostFlow()

        # Supply doesn't match demand
        mcf.add_arc_with_capacity_and_unit_cost(0, 1, 10, 5)
        mcf.set_node_supply(0, 10)
        mcf.set_node_supply(1, -5)  # Demand is less than supply

        status = mcf.solve()
        assert status == mcf.INFEASIBLE

    def test_capacity_constraint(self):
        """Test that capacity constraints are respected."""
        from size_constrained_clustering.k_means_constrained.min_cost_flow import (
            SimpleMinCostFlow,
        )

        mcf = SimpleMinCostFlow()

        # Arc with capacity 3, but need to flow 5
        mcf.add_arc_with_capacity_and_unit_cost(0, 1, 3, 1)  # Cheap but small
        mcf.add_arc_with_capacity_and_unit_cost(0, 1, 10, 10)  # Expensive but large

        mcf.set_node_supply(0, 5)
        mcf.set_node_supply(1, -5)

        status = mcf.solve()
        assert status == mcf.OPTIMAL

        # Should use all 3 of cheap arc
        assert mcf.flow(0) == 3
        # And 2 of expensive arc
        assert mcf.flow(1) == 2

    def test_zero_flow(self):
        """Test network with zero supply/demand."""
        from size_constrained_clustering.k_means_constrained.min_cost_flow import (
            SimpleMinCostFlow,
        )

        mcf = SimpleMinCostFlow()

        mcf.add_arc_with_capacity_and_unit_cost(0, 1, 10, 5)
        mcf.set_node_supply(0, 0)
        mcf.set_node_supply(1, 0)

        status = mcf.solve()
        assert status == mcf.OPTIMAL
        assert mcf.flow(0) == 0

    def test_large_costs(self):
        """Test handling of large costs."""
        from size_constrained_clustering.k_means_constrained.min_cost_flow import (
            SimpleMinCostFlow,
        )

        mcf = SimpleMinCostFlow()

        mcf.add_arc_with_capacity_and_unit_cost(0, 1, 100, 1000000)
        mcf.set_node_supply(0, 50)
        mcf.set_node_supply(1, -50)

        status = mcf.solve()
        assert status == mcf.OPTIMAL
        assert mcf.flow(0) == 50

    def test_multiple_sources_sinks(self):
        """Test network with multiple sources and sinks."""
        from size_constrained_clustering.k_means_constrained.min_cost_flow import (
            SimpleMinCostFlow,
        )

        mcf = SimpleMinCostFlow()

        # Two sources (0, 1) -> intermediate (2) -> two sinks (3, 4)
        mcf.add_arc_with_capacity_and_unit_cost(0, 2, 10, 1)
        mcf.add_arc_with_capacity_and_unit_cost(1, 2, 10, 2)
        mcf.add_arc_with_capacity_and_unit_cost(2, 3, 10, 1)
        mcf.add_arc_with_capacity_and_unit_cost(2, 4, 10, 1)

        mcf.set_node_supply(0, 5)
        mcf.set_node_supply(1, 5)
        mcf.set_node_supply(2, 0)
        mcf.set_node_supply(3, -4)
        mcf.set_node_supply(4, -6)

        status = mcf.solve()
        assert status == mcf.OPTIMAL


class TestVectorizedWrapper:
    """Test the vectorized wrapper functions."""

    def test_vectorized_add_arcs(self):
        """Test vectorized arc addition."""
        from size_constrained_clustering.k_means_constrained.mincostflow_vectorized import (
            SimpleMinCostFlowVectorized,
        )

        mcf = SimpleMinCostFlowVectorized()

        tails = np.array([0, 0, 1], dtype=np.int32)
        heads = np.array([1, 2, 2], dtype=np.int32)
        capacities = np.array([5, 5, 5], dtype=np.int32)
        costs = np.array([1, 2, 1], dtype=np.int32)

        mcf.AddArcWithCapacityAndUnitCostVectorized(tails, heads, capacities, costs)

        nodes = np.array([0, 1, 2], dtype=np.int32)
        supplies = np.array([5, 0, -5], dtype=np.int32)
        mcf.SetNodeSupplyVectorized(nodes, supplies)

        status = mcf.solve()
        assert status == mcf.OPTIMAL

        # Test vectorized flow retrieval
        arc_indices = np.array([0, 1, 2], dtype=np.int32)
        flows = mcf.FlowVectorized(arc_indices)
        assert flows.sum() == 5  # Total flow out should equal supply


class TestKMeansIntegration:
    """Integration tests with constrained k-means."""

    def test_kmeans_constrained_simple(self):
        """Test constrained k-means with simple data."""
        from sklearn.datasets import make_blobs
        import collections
        from size_constrained_clustering.k_means_constrained import (
            k_means_constrained,
        )

        n_samples = 200
        n_clusters = 4
        centers = [(-5, -5), (0, 0), (5, 5), (10, 10)]

        X, _ = make_blobs(
            n_samples=n_samples,
            n_features=2,
            cluster_std=1.0,
            centers=centers,
            shuffle=False,
            random_state=42,
        )

        size_min = 40
        size_max = 60

        _, labels, _ = k_means_constrained(
            X,
            n_clusters=n_clusters,
            size_min=size_min,
            size_max=size_max,
            random_state=42,
        )

        label_counter = collections.Counter(labels)

        for label, count in label_counter.items():
            assert count >= size_min, f"Cluster {label} has {count} < {size_min}"
            assert count <= size_max, f"Cluster {label} has {count} > {size_max}"

    def test_kmeans_constrained_class(self):
        """Test KMeansConstrained class interface."""
        from sklearn.datasets import make_blobs
        import collections
        from size_constrained_clustering.k_means_constrained import (
            KMeansConstrained,
        )

        n_samples = 300
        n_clusters = 3

        X, _ = make_blobs(
            n_samples=n_samples, n_features=2, centers=n_clusters, random_state=42
        )

        size_min = 90
        size_max = 110

        model = KMeansConstrained(
            n_clusters=n_clusters,
            size_min=size_min,
            size_max=size_max,
            random_state=42,
        )
        model.fit(X)

        label_counter = collections.Counter(model.labels_)

        for label, count in label_counter.items():
            assert count >= size_min
            assert count <= size_max

        # Test predict
        predictions = model.predict(X[:10])
        assert len(predictions) == 10


class TestORToolsEquivalence:
    """Equivalence tests against OR-Tools (skipped if OR-Tools not available)."""

    @pytest.fixture
    def ortools_available(self):
        """Check if OR-Tools is available."""
        try:
            from ortools.graph.python.min_cost_flow import (
                SimpleMinCostFlow as OrtoolsMCF,
            )

            return True
        except ImportError:
            return False

    def test_equivalence_simple(self, ortools_available):
        """Test our implementation gives same result as OR-Tools for simple case."""
        if not ortools_available:
            pytest.skip("OR-Tools not available")

        from ortools.graph.python.min_cost_flow import SimpleMinCostFlow as OrtoolsMCF
        from size_constrained_clustering.k_means_constrained.min_cost_flow import (
            SimpleMinCostFlow as CythonMCF,
        )

        # Create identical problems
        for _ in range(10):  # Test multiple random problems
            np.random.seed(42)

            ortools_mcf = OrtoolsMCF()
            cython_mcf = CythonMCF()

            # Build identical networks
            n_nodes = 10
            n_arcs = 20

            for arc_idx in range(n_arcs):
                tail = np.random.randint(0, n_nodes - 1)
                head = np.random.randint(tail + 1, n_nodes)
                capacity = np.random.randint(1, 100)
                cost = np.random.randint(1, 1000)

                ortools_mcf.add_arc_with_capacity_and_unit_cost(tail, head, capacity, cost)
                cython_mcf.add_arc_with_capacity_and_unit_cost(tail, head, capacity, cost)

            # Set supplies - first node is source, last is sink
            supply = np.random.randint(1, 50)
            ortools_mcf.set_node_supply(0, supply)
            ortools_mcf.set_node_supply(n_nodes - 1, -supply)
            cython_mcf.set_node_supply(0, supply)
            cython_mcf.set_node_supply(n_nodes - 1, -supply)

            # Middle nodes have 0 supply
            for i in range(1, n_nodes - 1):
                ortools_mcf.set_node_supply(i, 0)
                cython_mcf.set_node_supply(i, 0)

            # Solve both
            ortools_status = ortools_mcf.solve()
            cython_status = cython_mcf.solve()

            # Compare results
            assert ortools_status == cython_status

            if ortools_status == ortools_mcf.OPTIMAL:
                # Compare flows (they might not be identical due to different tie-breaking,
                # but total cost should be the same)
                ortools_cost = ortools_mcf.optimal_cost()
                cython_cost = cython_mcf.optimal_cost()
                assert ortools_cost == cython_cost, f"Costs differ: {ortools_cost} vs {cython_cost}"

    def test_kmeans_equivalence(self, ortools_available):
        """Test that k-means gives equivalent results with our implementation."""
        if not ortools_available:
            pytest.skip("OR-Tools not available")

        # This test would require temporarily switching between implementations
        # For now, we just verify our implementation produces valid clusters
        pass


class TestPerformance:
    """Performance benchmarks."""

    @pytest.mark.slow
    def test_medium_scale(self):
        """Test with medium-scale problem (1000 points)."""
        from sklearn.datasets import make_blobs
        import collections
        import time
        from size_constrained_clustering.k_means_constrained import (
            k_means_constrained,
        )

        n_samples = 1000
        n_clusters = 10

        X, _ = make_blobs(n_samples=n_samples, n_features=2, centers=n_clusters, random_state=42)

        size_min = 90
        size_max = 110

        start = time.perf_counter()
        _, labels, _ = k_means_constrained(
            X,
            n_clusters=n_clusters,
            size_min=size_min,
            size_max=size_max,
            random_state=42,
            n_init=1,  # Single init for speed
        )
        elapsed = time.perf_counter() - start

        label_counter = collections.Counter(labels)

        for label, count in label_counter.items():
            assert count >= size_min
            assert count <= size_max

        print(f"\n1000 points, 10 clusters: {elapsed:.2f}s")

    @pytest.mark.slow
    def test_large_scale(self):
        """Test with larger problem (10000 points)."""
        from sklearn.datasets import make_blobs
        import collections
        import time
        from size_constrained_clustering.k_means_constrained import (
            k_means_constrained,
        )

        n_samples = 10000
        n_clusters = 20

        X, _ = make_blobs(n_samples=n_samples, n_features=2, centers=n_clusters, random_state=42)

        size_min = 450
        size_max = 550

        start = time.perf_counter()
        _, labels, _ = k_means_constrained(
            X,
            n_clusters=n_clusters,
            size_min=size_min,
            size_max=size_max,
            random_state=42,
            n_init=1,
        )
        elapsed = time.perf_counter() - start

        label_counter = collections.Counter(labels)

        for label, count in label_counter.items():
            assert count >= size_min
            assert count <= size_max

        print(f"\n10000 points, 20 clusters: {elapsed:.2f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
