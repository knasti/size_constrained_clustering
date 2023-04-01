#!/usr/bin/env python

import numpy as np
from ortools.graph.python.min_cost_flow import SimpleMinCostFlow

# Cython paths must be fully qualified
from k_means_size_constrained.mincostflow_vectorized_ import (
    SimpleMinCostFlow_FlowVectorized,
    SimpleMinCostFlow_SetNodeSupplyVectorized,
    SimpleMinCostFlow_AddArcWithCapacityAndUnitCostVectorized,
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
