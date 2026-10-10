"""Generate train/test traffic on the supplied communication topologies."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .config import DATA_DIR
from .envs.problem import ProblemInstance, TopologyInfo


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def regular(ct, n, cf):
    if ct == 'allreduce':
        return np.ones((cf, n), dtype=int), np.ones((cf, n), dtype=int)
    c = n * cf if ct == 'allgather' else n * n * cf
    initial = np.zeros((c, n), dtype=int)
    demands = np.zeros_like(initial)
    for q in range(c):
        src = q // cf if ct == 'allgather' else q // (n * cf)
        initial[q, src] = 1
        if ct == 'allgather':
            demands[q] = 1
            demands[q, src] = 0
        else:
            dst = q // cf % n
            if dst != src:
                demands[q, dst] = 1
    return initial, demands

def variable(n, cf, distribution, rng):
    pairs = np.array([(s, d) for s in range(n) for d in range(n) if s != d])
    volume = len(pairs) * cf
    if distribution == 'iid':
        weights = np.ones(len(pairs))
    elif distribution == 'zipf':
        weights = (rng.permutation(len(pairs)) + 1).astype(float) ** -1.5
    elif distribution == 'hotspot':
        hot = pairs[:, 1] == rng.integers(n)
        weights = np.where(hot, .9 / hot.sum(), .1 / (~hot).sum())
    elif distribution == 'sparse':
        permutation = rng.permutation(n)
        while np.any(permutation == np.arange(n)):
            permutation = rng.permutation(n)
        weights = (pairs[:, 1] == permutation[pairs[:, 0]]).astype(float)
    else:
        raise ValueError(distribution)
    counts = rng.multinomial(volume, weights / weights.sum())
    sources, targets = np.repeat(pairs[:, 0], counts), np.repeat(pairs[:, 1], counts)
    initial = np.zeros((volume, n), dtype=int)
    demands = np.zeros_like(initial)
    initial[np.arange(volume), sources] = 1
    demands[np.arange(volume), targets] = 1
    return initial, demands

def load_dataset(split='test', data_dir=DATA_DIR, topology='all',
                 collective='all', chunk_factor='all', horizon=40):
    if split not in {'train', 'test'} or horizon < 1:
        raise ValueError('Choose train/test and a positive horizon')
    data = json.loads((Path(data_dir) / (split + '.json')).read_text())
    definitions = data['topologies']
    if topology != 'all' and topology not in definitions:
        raise ValueError('Unknown topology: ' + topology)
    kinds = ('allgather', 'allreduce', 'alltoall', 'alltoallv')
    if collective != 'all' and collective not in kinds:
        raise ValueError('Unknown collective: ' + collective)
    settings = data['traffic_generator']
    factors = settings['chunk_factors']
    if chunk_factor != 'all' and int(chunk_factor) not in factors:
        raise ValueError('Unknown chunk factor: ' + str(chunk_factor))
    rng = np.random.default_rng()
    rng.bit_generator.state = settings['rng_state']
    instances = {name: TopologyInfo.from_dict({**t, 'cache_dir': None})
                 for name, t in definitions.items()}
    seen, problems = set(), []
    # Generate in the stored order before filtering, preserving every traffic draw.
    for family, topo in instances.items():
        t = definitions[family]
        topo_hash = digest({k: v for k, v in t.items() if k not in ('name', 'cache_dir')})
        for cf in factors:
            for kind in kinds:
                distributions = settings['distributions'] if kind == 'alltoallv' else ['regular']
                for distribution in distributions:
                    target = settings['samples'] if kind == 'alltoallv' else 1
                    accepted = draws = 0
                    while accepted < target:
                        draws += 1
                        if draws > 10000:
                            raise ValueError('Unable to generate distinct traffic cases')
                        initial, demands = (variable(topo.V, cf, distribution, rng)
                                            if kind == 'alltoallv' else regular(kind, topo.V, cf))
                        semantic = dict(collective=kind, topology_sha256=topo_hash,
                            horizon=horizon, initial_state=initial.tolist(), demands=demands.tolist(),
                            layout='variable pair-major' if kind == 'alltoallv' else 'standard source/destination-major',
                            value_type='float32', uniform_chunk_units=1)
                        semantic_hash = digest(semantic)
                        if semantic_hash in seen:
                            if kind != 'alltoallv':
                                raise ValueError('Duplicate regular traffic case')
                            continue
                        seen.add(semantic_hash)
                        identifier = f'{family}-cf{cf}-{kind}-{distribution}-{accepted:03d}'
                        accepted += 1
                        if (topology not in ('all', family) or collective not in ('all', kind)
                                or (chunk_factor != 'all' and int(chunk_factor) != cf)):
                            continue
                        problem = ProblemInstance(topo.V, len(initial), topo.E, horizon,
                            topo.capacities, topo.edges, demands.astype(bool), initial.astype(bool),
                            topo.shared_constraints, topo, kind)
                        problem.topology_family = family
                        problems.append((identifier, problem))
    if not problems:
        raise ValueError('No cases match the requested topology/collective/chunk factor')
    return problems
