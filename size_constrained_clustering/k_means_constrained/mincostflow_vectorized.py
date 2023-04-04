#!/usr/bin/env python

import numpy as np

# Cython paths must be fully qualified
from ortools.graph.python.min_cost_flow import SimpleMinCostFlow

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
