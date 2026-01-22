#!/usr/bin/env python

import numpy as np

# Use our custom Cython implementation instead of OR-Tools
# This provides identical API to ortools.graph.python.min_cost_flow.SimpleMinCostFlow
# but works with Python 3.14t (free-threaded) and removes the OR-Tools dependency
from size_constrained_clustering.k_means_constrained.min_cost_flow import SimpleMinCostFlow

from size_constrained_clustering.k_means_constrained.mincostflow_vectorized_ import (
    SimpleMinCostFlow_AddArcWithCapacityAndUnitCostVectorized,
    SimpleMinCostFlow_SetNodeSupplyVectorized,
    SimpleMinCostFlow_FlowVectorized,
)


class SimpleMinCostFlowVectorized(SimpleMinCostFlow):
    def AddArcWithCapacityAndUnitCostVectorized(self, tail, head, capacity, unit_cost):
        return SimpleMinCostFlow_AddArcWithCapacityAndUnitCostVectorized(
            self, tail, head, capacity, unit_cost
        )

    def SetNodeSupplyVectorized(self, node, supply):
        return SimpleMinCostFlow_SetNodeSupplyVectorized(self, node, supply)

    def FlowVectorized(self, arc):
        return SimpleMinCostFlow_FlowVectorized(self, arc)
