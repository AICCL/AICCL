"""Problem instance and topology information."""

import os

import numpy as np


COPY_ONLY_COLLECTIVES = frozenset({
    "allgather",
    "alltoall",
    "alltoallv",
    "broadcast",
})
SUPPORTED_COLLECTIVES = COPY_ONLY_COLLECTIVES | {"allreduce"}


def normalize_collective_type(collective_type):
    """Normalize collective type names used across scenarios and scripts."""
    normalized = (collective_type or "allgather").strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "allgather": "allgather",
        "alltoall": "alltoall",
        "alltoallv": "alltoallv",
        "alltoallvariable": "alltoallv",
        "broadcast": "broadcast",
        "allreduce": "allreduce",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported collective_type={collective_type!r}. "
            f"Supported: {sorted(SUPPORTED_COLLECTIVES)}"
        )
    return aliases[normalized]


class ProblemInstance:
    """Problem instance for collective communication scheduling.
    
    Attributes:
        V: Number of nodes
        C: Number of chunks
        E: Number of edges
        T: Time limit (steps)
        capacities: Edge capacities, shape (E,)
        topology: Edge list, shape (E, 2)
        D: Destination matrix, shape (E, V)
        demands: Demand matrix, shape (C, V)
        initial_state: Initial chunk distribution, shape (C, V)
        shared_constraints: List of (edge_indices, limit) tuples
        topology_info: TopologyInfo object
    """
    
    def __init__(
        self,
        num_nodes,
        num_chunks,
        num_edges,
        time_limit,
        capacities,
        topology,
        demands,
        initial_state,
        shared_constraints=None,
        topology_info=None,
        collective_type="allgather",
    ):
        self.V = num_nodes
        self.C = num_chunks
        self.E = num_edges
        self.T = time_limit
        self.capacities = capacities
        self.k = np.min(capacities) if len(capacities) > 0 else 0
        self.collective_type = normalize_collective_type(collective_type)
        
        self.topology = topology
        
        # NOTE: D matrix removed to save memory at scale.
        # Use compute_received_chunks() or edge_dst indexing instead.
        # D[e, v] = 1 iff edge e points to node v, which is just edge_dst[e] == v
            
        self.demands = demands
        self.initial_state = initial_state
        self.shared_constraints = shared_constraints if shared_constraints is not None else []
        
        if topology_info is not None:
            self.topology_info = topology_info
        else:
            self.topology_info = TopologyInfo(
                self.V, self.E, np.array(topology), 
                np.array(capacities), self.shared_constraints
            )
    
    def to_dict(self):
        """Convert ProblemInstance to dictionary for JSON serialization."""
        return {
            'V': int(self.V),
            'C': int(self.C),
            'E': int(self.E),
            'T': int(self.T),
            'collective_type': self.collective_type,
            'capacities': self.capacities.tolist() if isinstance(self.capacities, np.ndarray) else list(self.capacities),
            'topology': self.topology.tolist() if isinstance(self.topology, np.ndarray) else list(self.topology),
            'demands': self.demands.tolist() if isinstance(self.demands, np.ndarray) else self.demands.tolist(),
            'initial_state': self.initial_state.tolist() if isinstance(self.initial_state, np.ndarray) else self.initial_state.tolist(),
            'shared_constraints': self.shared_constraints,
            'topology_info': self.topology_info.to_dict() if hasattr(self.topology_info, 'to_dict') else None,
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create ProblemInstance from dictionary."""
        capacities = np.array(data['capacities'])
        topology = [tuple(edge) for edge in data['topology']]
        demands = np.array(data['demands'])
        initial_state = np.array(data['initial_state'])
        shared_constraints = data.get('shared_constraints', [])
        
        # Reconstruct topology_info
        topology_info_data = data.get('topology_info')
        topology_info = None
        if topology_info_data:
            topology_info = TopologyInfo.from_dict(topology_info_data)
        
        return cls(
            num_nodes=data['V'],
            num_chunks=data['C'],
            num_edges=data['E'],
            time_limit=data['T'],
            capacities=capacities,
            topology=topology,
            demands=demands,
            initial_state=initial_state,
            shared_constraints=shared_constraints,
            topology_info=topology_info,
            collective_type=data.get('collective_type', 'allgather'),
        )


class TopologyInfo:
    """Topology information shared across scenarios.
    
    Attributes:
        V: Number of nodes
        E: Number of edges
        edges: Edge list, shape (E, 2)
        capacities: Edge capacities, shape (E,)
        shared_constraints: List of (edge_indices, limit)
        edge_src: Source nodes, shape (E,)
        edge_dst: Destination nodes, shape (E,)
        D: Destination matrix, shape (E, V)
        dist_matrix: Shortest path matrix, shape (V, V)
        group_map: Dict mapping edge to constraint groups
    """
    
    def __init__(self, V, E, edges, capacities, shared_constraints, 
                 cache_dir=None, name=None):
        self.V = V
        self.E = E
        self.edges = edges
        self.capacities = capacities
        self.shared_constraints = shared_constraints
        self.cache_dir = cache_dir
        self.name = name
        
        self.edge_src = edges[:, 0]
        self.edge_dst = edges[:, 1]
        
        # NOTE: Dense D matrix (E, V) removed to save O(E*V) = O(V^3) memory.
        # Use compute_received_chunks() for Y_t @ D equivalent.
            
        # Shortest paths
        self.dist_matrix = self._compute_shortest_paths()
        
        # Group constraints
        self.group_map = {}
        for g_idx, (edge_indices, limit) in enumerate(shared_constraints):
            for e in edge_indices:
                if e not in self.group_map:
                    self.group_map[e] = []
                self.group_map[e].append((g_idx, limit))
    
    def _compute_shortest_paths(self):
        """Compute shortest paths using Floyd-Warshall."""
        if self.cache_dir is not None:
            cache_file = os.path.join(self.cache_dir, 'shortest_paths.npy')
            if os.path.exists(cache_file):
                try:
                    dist = np.load(cache_file)
                    if dist.shape == (self.V, self.V):
                        return dist
                except Exception:
                    pass
        
        inf = float(self.V + 1)
        dist = np.full((self.V, self.V), inf, dtype=np.float32)
        np.fill_diagonal(dist, 0)
        for u, v in self.edges:
            dist[u, v] = 1.0
        
        for k in range(self.V):
            dist = np.minimum(dist, dist[:, [k]] + dist[[k], :])
        
        if self.cache_dir is not None:
            try:
                os.makedirs(self.cache_dir, exist_ok=True)
                cache_file = os.path.join(self.cache_dir, 'shortest_paths.npy')
                np.save(cache_file, dist)
            except Exception:
                pass
        
        return dist
    
    def to_dict(self):
        """Convert TopologyInfo to dictionary for JSON serialization."""
        return {
            'V': int(self.V),
            'E': int(self.E),
            'edges': self.edges.tolist() if isinstance(self.edges, np.ndarray) else list(self.edges),
            'capacities': np.asarray(getattr(self, 'base_capacities', self.capacities)).tolist(),
            'shared_constraints': getattr(self, 'base_shared_constraints', self.shared_constraints),
            'timing': self.timing.to_dict() if getattr(self, 'timing', None) else None,
            'cache_dir': self.cache_dir,
            'name': self.name,
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create TopologyInfo from dictionary."""
        edges = np.array(data['edges'])
        capacities = np.array(data['capacities'])
        shared_constraints = data.get('shared_constraints', [])
        cache_dir = data.get('cache_dir')
        name = data.get('name')
        
        topo = cls(
            V=data['V'],
            E=data['E'],
            edges=edges,
            capacities=capacities,
            shared_constraints=shared_constraints,
            cache_dir=cache_dir,
            name=name,
        )
        spec = data.get("timing")
        if spec:
            from .timing import configure_timing
            configure_timing(topo, spec["mode"], spec["refinement"], spec["chunk_bytes"], spec["profile"], timed=True)
        return topo


def compute_received_chunks(Y_t, edge_dst, V):
    """Compute which chunks are received at each node from schedule Y_t.
    
    This replaces the dense matrix operation: N_t = (Y_t @ D > 0)
    where D is the (E, V) destination incidence matrix.
    
    The key insight: D[e, v] = 1 iff edge_dst[e] == v
    So (Y_t @ D)[c, v] > 0 iff there exists edge e where Y_t[c, e] = 1 and edge_dst[e] = v
    
    Args:
        Y_t: Schedule matrix, shape (C, E), binary
        edge_dst: Destination node for each edge, shape (E,)
        V: Number of nodes
        
    Returns:
        N_t: Received chunks matrix, shape (C, V), binary
             N_t[c, v] = 1 if chunk c is received at node v in this slot
    """
    C, E = Y_t.shape
    N_t = np.zeros((C, V), dtype=np.int32)
    
    # Find all (chunk, edge) pairs where Y_t[c, e] = 1
    chunk_indices, edge_indices = np.where(Y_t)
    
    if len(chunk_indices) > 0:
        # Get destination nodes for these edges
        dst_nodes = edge_dst[edge_indices]
        # Mark these (chunk, node) pairs as received
        N_t[chunk_indices, dst_nodes] = 1
    
    return N_t
