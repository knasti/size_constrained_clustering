# cython: language_level=3
# cython: freethreading_compatible=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
"""
Min-Cost Flow implementation using Successive Shortest Paths algorithm.

This is a drop-in replacement for ortools.graph.python.min_cost_flow.SimpleMinCostFlow
designed to work with Python 3.14t (free-threaded) without the OR-Tools dependency.

Algorithm: Successive Shortest Paths with Bellman-Ford
- Simple and correct implementation
- Time complexity: O(V * E * flow) with Bellman-Ford

For the constrained k-means use case, this is efficient enough since:
- The graph has O(n_points * n_clusters) arcs
- Total flow is n_points (one unit per data point)
"""

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free, realloc
from libc.string cimport memset
from libc.limits cimport LONG_MAX, INT_MAX

# NumPy type definitions
DTYPE_INT = np.int32
DTYPE_LONG = np.int64
ctypedef np.int32_t DTYPE_INT_t
ctypedef np.int64_t DTYPE_LONG_t

# Status codes matching OR-Tools API exactly
# See: https://developers.google.com/optimization/reference/graph/min_cost_flow/SimpleMinCostFlow
DEF STATUS_NOT_SOLVED = 0
DEF STATUS_OPTIMAL = 1
DEF STATUS_FEASIBLE = 2
DEF STATUS_INFEASIBLE = 3
DEF STATUS_UNBALANCED = 4
DEF STATUS_BAD_RESULT = 5
DEF STATUS_BAD_COST_RANGE = 6

# Initial capacity for dynamic arrays
DEF INITIAL_CAPACITY = 1024


cdef class SimpleMinCostFlow:
    """
    Min-cost flow solver compatible with OR-Tools SimpleMinCostFlow API.

    Uses Successive Shortest Paths algorithm with Bellman-Ford for path finding.
    """

    # Status constants (class attributes for API compatibility)
    OPTIMAL = STATUS_OPTIMAL
    INFEASIBLE = STATUS_INFEASIBLE
    NOT_SOLVED = STATUS_NOT_SOLVED
    BAD_COST_RANGE = STATUS_BAD_COST_RANGE

    # Graph structure - stores forward arcs only
    cdef int _num_arcs          # Number of user arcs (forward only)
    cdef int _arc_capacity_allocated
    cdef int _max_node_id

    # Arc data arrays (2 * _num_arcs total: forward + reverse)
    cdef int* _arc_tail
    cdef int* _arc_head
    cdef long* _arc_capacity
    cdef long* _arc_unit_cost
    cdef long* _arc_flow

    # Node data
    cdef int _num_nodes
    cdef int _node_capacity_allocated
    cdef long* _node_supply

    # For Bellman-Ford
    cdef long* _dist           # Distance from source
    cdef int* _parent_arc      # Parent arc in shortest path tree

    # Solver state
    cdef int _status
    cdef bint _solved
    cdef long _total_cost

    def __cinit__(self):
        """Initialize data structures."""
        self._num_arcs = 0
        self._arc_capacity_allocated = INITIAL_CAPACITY * 2  # Space for forward + reverse
        self._max_node_id = -1

        # Allocate arc arrays
        self._arc_tail = <int*>malloc(self._arc_capacity_allocated * sizeof(int))
        self._arc_head = <int*>malloc(self._arc_capacity_allocated * sizeof(int))
        self._arc_capacity = <long*>malloc(self._arc_capacity_allocated * sizeof(long))
        self._arc_unit_cost = <long*>malloc(self._arc_capacity_allocated * sizeof(long))
        self._arc_flow = <long*>malloc(self._arc_capacity_allocated * sizeof(long))

        # Node arrays start empty
        self._num_nodes = 0
        self._node_capacity_allocated = 0
        self._node_supply = NULL
        self._dist = NULL
        self._parent_arc = NULL

        self._status = STATUS_NOT_SOLVED
        self._solved = False
        self._total_cost = 0

        if (self._arc_tail == NULL or self._arc_head == NULL or
            self._arc_capacity == NULL or self._arc_unit_cost == NULL or
            self._arc_flow == NULL):
            raise MemoryError("Failed to allocate memory for min-cost flow solver")

    def __dealloc__(self):
        """Free allocated memory."""
        if self._arc_tail != NULL:
            free(self._arc_tail)
        if self._arc_head != NULL:
            free(self._arc_head)
        if self._arc_capacity != NULL:
            free(self._arc_capacity)
        if self._arc_unit_cost != NULL:
            free(self._arc_unit_cost)
        if self._arc_flow != NULL:
            free(self._arc_flow)
        if self._node_supply != NULL:
            free(self._node_supply)
        if self._dist != NULL:
            free(self._dist)
        if self._parent_arc != NULL:
            free(self._parent_arc)

    cdef int _ensure_arc_capacity(self, int required) except -1:
        """Ensure arc arrays can hold at least 'required' arcs."""
        cdef int new_capacity
        cdef void* tmp

        if required <= self._arc_capacity_allocated:
            return 0

        new_capacity = self._arc_capacity_allocated
        while new_capacity < required:
            new_capacity *= 2

        tmp = realloc(self._arc_tail, new_capacity * sizeof(int))
        if tmp == NULL:
            raise MemoryError("Failed to reallocate arc_tail")
        self._arc_tail = <int*>tmp

        tmp = realloc(self._arc_head, new_capacity * sizeof(int))
        if tmp == NULL:
            raise MemoryError("Failed to reallocate arc_head")
        self._arc_head = <int*>tmp

        tmp = realloc(self._arc_capacity, new_capacity * sizeof(long))
        if tmp == NULL:
            raise MemoryError("Failed to reallocate arc_capacity")
        self._arc_capacity = <long*>tmp

        tmp = realloc(self._arc_unit_cost, new_capacity * sizeof(long))
        if tmp == NULL:
            raise MemoryError("Failed to reallocate arc_unit_cost")
        self._arc_unit_cost = <long*>tmp

        tmp = realloc(self._arc_flow, new_capacity * sizeof(long))
        if tmp == NULL:
            raise MemoryError("Failed to reallocate arc_flow")
        self._arc_flow = <long*>tmp

        self._arc_capacity_allocated = new_capacity
        return 0

    cdef int _ensure_node_capacity(self, int required) except -1:
        """Ensure node arrays can hold at least 'required' nodes."""
        cdef int new_capacity, old_capacity
        cdef void* tmp
        cdef int i

        if required <= self._node_capacity_allocated:
            return 0

        old_capacity = self._node_capacity_allocated
        new_capacity = max(INITIAL_CAPACITY, old_capacity)
        while new_capacity < required:
            new_capacity *= 2

        if self._node_supply == NULL:
            self._node_supply = <long*>malloc(new_capacity * sizeof(long))
        else:
            tmp = realloc(self._node_supply, new_capacity * sizeof(long))
            if tmp == NULL:
                raise MemoryError("Failed to reallocate node_supply")
            self._node_supply = <long*>tmp

        if self._dist == NULL:
            self._dist = <long*>malloc(new_capacity * sizeof(long))
        else:
            tmp = realloc(self._dist, new_capacity * sizeof(long))
            if tmp == NULL:
                raise MemoryError("Failed to reallocate dist")
            self._dist = <long*>tmp

        if self._parent_arc == NULL:
            self._parent_arc = <int*>malloc(new_capacity * sizeof(int))
        else:
            tmp = realloc(self._parent_arc, new_capacity * sizeof(int))
            if tmp == NULL:
                raise MemoryError("Failed to reallocate parent_arc")
            self._parent_arc = <int*>tmp

        # Initialize new elements
        for i in range(old_capacity, new_capacity):
            self._node_supply[i] = 0

        self._node_capacity_allocated = new_capacity
        return 0

    def add_arc_with_capacity_and_unit_cost(self, int tail, int head,
                                            long capacity, long unit_cost) -> int:
        """
        Add an arc from tail to head with given capacity and unit cost.
        Returns the arc index.
        """
        cdef int arc_index = self._num_arcs
        cdef int total_arcs = (self._num_arcs + 1) * 2  # forward + reverse for all arcs

        self._ensure_arc_capacity(total_arcs)

        if tail > self._max_node_id:
            self._max_node_id = tail
        if head > self._max_node_id:
            self._max_node_id = head

        # Store forward arc at index arc_index * 2
        cdef int fwd_idx = arc_index * 2
        cdef int rev_idx = arc_index * 2 + 1

        self._arc_tail[fwd_idx] = tail
        self._arc_head[fwd_idx] = head
        self._arc_capacity[fwd_idx] = capacity
        self._arc_unit_cost[fwd_idx] = unit_cost
        self._arc_flow[fwd_idx] = 0

        # Store reverse arc at index arc_index * 2 + 1
        self._arc_tail[rev_idx] = head
        self._arc_head[rev_idx] = tail
        self._arc_capacity[rev_idx] = 0  # Reverse starts with 0 capacity
        self._arc_unit_cost[rev_idx] = -unit_cost
        self._arc_flow[rev_idx] = 0

        self._num_arcs += 1
        self._solved = False

        return arc_index

    def set_node_supply(self, int node, long supply) -> None:
        """Set the supply (positive) or demand (negative) for a node."""
        if node > self._max_node_id:
            self._max_node_id = node

        self._ensure_node_capacity(node + 1)
        self._node_supply[node] = supply
        self._solved = False

    cdef inline long _residual(self, int arc_idx) noexcept nogil:
        """Get residual capacity of arc."""
        return self._arc_capacity[arc_idx] - self._arc_flow[arc_idx]

    cdef bint _bellman_ford(self, int source) noexcept nogil:
        """
        Run Bellman-Ford from source using residual graph.
        Returns True if any sink (negative excess) is reachable.
        """
        cdef int i, j, arc, u, v
        cdef long cost, residual, new_dist
        cdef int total_arcs = self._num_arcs * 2
        cdef bint changed

        # Initialize distances
        for i in range(self._num_nodes):
            self._dist[i] = LONG_MAX
            self._parent_arc[i] = -1
        self._dist[source] = 0

        # Relax edges n-1 times (or until no change)
        for j in range(self._num_nodes):
            changed = False
            for arc in range(total_arcs):
                residual = self._residual(arc)
                if residual > 0:
                    u = self._arc_tail[arc]
                    v = self._arc_head[arc]
                    cost = self._arc_unit_cost[arc]
                    if self._dist[u] < LONG_MAX:
                        new_dist = self._dist[u] + cost
                        if new_dist < self._dist[v]:
                            self._dist[v] = new_dist
                            self._parent_arc[v] = arc
                            changed = True
            if not changed:
                break

        return True  # Path existence checked by caller

    cdef int _find_sink(self, long* excess) noexcept nogil:
        """Find reachable node with negative excess (demand), return -1 if none."""
        cdef int i
        cdef int best = -1
        cdef long best_dist = LONG_MAX

        for i in range(self._num_nodes):
            if excess[i] < 0 and self._dist[i] < LONG_MAX:
                if self._dist[i] < best_dist:
                    best_dist = self._dist[i]
                    best = i
        return best

    cdef long _augment(self, int sink, long* excess) noexcept nogil:
        """
        Augment flow along path from source to sink.
        Returns amount of flow pushed.
        """
        cdef int node = sink
        cdef int arc
        cdef long bottleneck = LONG_MAX
        cdef long residual

        # Find bottleneck capacity
        while self._parent_arc[node] != -1:
            arc = self._parent_arc[node]
            residual = self._residual(arc)
            if residual < bottleneck:
                bottleneck = residual
            node = self._arc_tail[arc]

        # Also limited by how much we can push/absorb
        cdef int source = node  # We've backtracked to the source
        if excess[source] < bottleneck:
            bottleneck = excess[source]
        if -excess[sink] < bottleneck:
            bottleneck = -excess[sink]

        if bottleneck <= 0:
            return 0

        # Push flow along the path
        node = sink
        while self._parent_arc[node] != -1:
            arc = self._parent_arc[node]
            self._arc_flow[arc] += bottleneck
            # Update reverse arc: reverse of arc i is at i^1 (XOR with 1)
            self._arc_flow[arc ^ 1] -= bottleneck
            node = self._arc_tail[arc]

        # Update excess
        excess[source] -= bottleneck
        excess[sink] += bottleneck

        return bottleneck

    cdef int _ssp_solve(self) noexcept nogil:
        """
        Solve min-cost flow using Successive Shortest Paths.
        Returns status code.
        """
        cdef long* excess
        cdef int i, source, sink
        cdef long total_supply, pushed
        cdef int total_arcs = self._num_arcs * 2
        cdef int max_iterations

        # Check balance
        total_supply = 0
        for i in range(self._num_nodes):
            total_supply += self._node_supply[i]
        if total_supply != 0:
            return STATUS_INFEASIBLE

        # Initialize flows to 0
        for i in range(total_arcs):
            self._arc_flow[i] = 0

        # Create excess array (copy of supply)
        excess = <long*>malloc(self._num_nodes * sizeof(long))
        if excess == NULL:
            return STATUS_INFEASIBLE

        for i in range(self._num_nodes):
            excess[i] = self._node_supply[i]

        # Limit iterations to prevent infinite loops
        max_iterations = self._num_nodes * self._num_arcs + 1000000

        # Main loop: find shortest augmenting paths
        for _ in range(max_iterations):
            # Find a source (node with positive excess)
            source = -1
            for i in range(self._num_nodes):
                if excess[i] > 0:
                    source = i
                    break

            if source == -1:
                # No more excess, we're done
                break

            # Run Bellman-Ford from source
            self._bellman_ford(source)

            # Find reachable sink
            sink = self._find_sink(excess)
            if sink == -1:
                # No reachable sink - infeasible
                free(excess)
                return STATUS_INFEASIBLE

            # Augment along the path
            pushed = self._augment(sink, excess)
            if pushed == 0:
                # Couldn't push any flow - should not happen
                free(excess)
                return STATUS_INFEASIBLE

        # Check all excess is zero
        for i in range(self._num_nodes):
            if excess[i] != 0:
                free(excess)
                return STATUS_INFEASIBLE

        free(excess)
        return STATUS_OPTIMAL

    def solve(self) -> int:
        """Solve the min-cost flow problem. Returns status code."""
        cdef int i

        self._num_nodes = self._max_node_id + 1
        if self._num_nodes <= 0:
            self._status = STATUS_OPTIMAL
            self._solved = True
            return STATUS_OPTIMAL

        self._ensure_node_capacity(self._num_nodes)

        self._status = self._ssp_solve()

        # Calculate total cost
        if self._status == STATUS_OPTIMAL:
            self._total_cost = 0
            # Sum cost of forward arcs only (even indices)
            for i in range(self._num_arcs):
                self._total_cost += self._arc_flow[i * 2] * self._arc_unit_cost[i * 2]

        self._solved = True
        return self._status

    def flow(self, int arc) -> long:
        """Get flow on an arc after solving."""
        if not self._solved:
            return 0
        if arc < 0 or arc >= self._num_arcs:
            return 0
        # Flow is stored at arc * 2 (forward arc index)
        return self._arc_flow[arc * 2]

    def optimal_cost(self) -> long:
        """Get total cost of optimal flow."""
        return self._total_cost

    def num_arcs(self) -> int:
        """Return number of user arcs."""
        return self._num_arcs

    def num_nodes(self) -> int:
        """Return number of nodes."""
        return self._num_nodes

    def tail(self, int arc) -> int:
        """Get tail (source) node of an arc."""
        if arc < 0 or arc >= self._num_arcs:
            return -1
        return self._arc_tail[arc * 2]

    def head(self, int arc) -> int:
        """Get head (destination) node of an arc."""
        if arc < 0 or arc >= self._num_arcs:
            return -1
        return self._arc_head[arc * 2]

    def capacity(self, int arc) -> long:
        """Get capacity of an arc."""
        if arc < 0 or arc >= self._num_arcs:
            return 0
        return self._arc_capacity[arc * 2]

    def unit_cost(self, int arc) -> long:
        """Get unit cost of an arc."""
        if arc < 0 or arc >= self._num_arcs:
            return 0
        return self._arc_unit_cost[arc * 2]
