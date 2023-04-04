
import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.int32
ctypedef np.int32_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
def SimpleMinCostFlow_AddArcWithCapacityAndUnitCostVectorized(
        self,
        np.ndarray[DTYPE_t, ndim=1] tail,
        np.ndarray[DTYPE_t, ndim=1] head,
        np.ndarray[DTYPE_t, ndim=1] capacity,
        np.ndarray[DTYPE_t, ndim=1] unit_cost):

    cdef int len = tail.shape[0]

    assert tail.dtype == DTYPE
    assert head.dtype == DTYPE
    assert capacity.dtype == DTYPE
    assert unit_cost.dtype == DTYPE
    assert head.shape[0] == len
    assert capacity.shape[0] == len
    assert unit_cost.shape[0] == len

    for i in range(len):
        self.add_arc_with_capacity_and_unit_cost(tail[i], head[i], capacity[i], unit_cost[i])


@cython.boundscheck(False)
@cython.wraparound(False)
def SimpleMinCostFlow_SetNodeSupplyVectorized(self,
                                              np.ndarray[DTYPE_t, ndim=1] node,
                                              np.ndarray[DTYPE_t, ndim=1] supply):
    cdef int len = node.shape[0]

    assert node.dtype == DTYPE
    assert supply.dtype == DTYPE
    assert supply.shape[0] == len

    for i in range(len):
        self.set_node_supply(node[i], supply[i])

@cython.boundscheck(False)
@cython.wraparound(False)
def SimpleMinCostFlow_FlowVectorized(self,
                                     np.ndarray[DTYPE_t, ndim=1] arc):

    cdef int len = arc.shape[0]

    assert arc.dtype == DTYPE

    cdef np.ndarray flow = np.zeros(len, dtype=DTYPE)

    for i in range(len):
        flow[i] = self.flow(arc[i])

    return flow

