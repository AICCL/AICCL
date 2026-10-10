"""Slot-start arrival/weighted-distance features; never alter feasibility.

Pending values remain unavailable as sources. Shortest physical service paths
are actor hints, not proofs of feasible routing under concurrent reservations.
"""
import numpy as np


def service_distances(topo):
    timing = topo.timing
    if timing is None:
        raise ValueError('Arrival features require timed topology')
    distance = np.full((topo.V, topo.V), np.inf, dtype=np.float64)
    np.fill_diagonal(distance, 0.)
    for e, (src, dst) in enumerate(topo.edges):
        cost = max([timing.copy_cost[e]] + [timing.group_cost[g]
            for g, members in enumerate(timing.groups) if e in members])
        distance[src, dst] = min(distance[src, dst], cost)
    for k in range(topo.V):
        distance = np.minimum(distance, distance[:, k, None] + distance[None, k, :])
    if not np.isfinite(distance).all():
        raise ValueError('Arrival feature topology must be strongly connected')
    return distance


def slot_arrival_context(state, topo, distance):
    from .candidate_control import outstanding_targets
    targets = outstanding_targets(state)
    count = len(targets)
    target_distance = np.zeros((count, topo.V))
    for c in range(count):
        if targets[c].any():
            target_distance[c] = distance[:, targets[c]].min(axis=1)
    singleton = np.zeros(count, dtype=bool)
    wait = np.zeros(count)
    partner_distance = np.zeros((count, topo.V))
    if state['collective_type'] == 'allreduce':
        resident_count = state['partial'].any(axis=2).sum(axis=1)
        for c in range(count):
            partners = [p for p in state.get('pending', []) if p['chunk'] == c and p['kind'] == 'partial']
            if partners:
                singleton[c] = state['mode'][c] == 0 and resident_count[c] == 1
                wait[c] = min(p['arrival'] - state['slot'] for p in partners) / topo.timing.refinement
                partner_distance[c] = distance[:, [p['dst'] for p in partners]].min(axis=1)
    return dict(singleton=singleton, wait=wait, targets=target_distance, partners=partner_distance)


def candidate_arrival_features(context, chunks, edges, topo, remaining_slots, delays):
    chunks, edges = np.asarray(chunks), np.asarray(edges)
    src, dst = topo.edge_src[edges], topo.edge_dst[edges]
    reference = topo.timing.reference_time
    before = context['targets'][chunks, src];after = context['targets'][chunks, dst]
    advance = np.clip((before - after) / reference, -1., 1.)
    slack = np.clip((remaining_slots - delays - after / topo.timing.delta)
                    / max(1, remaining_slots), -1., 1.)
    partner_advance = np.clip((context['partners'][chunks, src] - context['partners'][chunks, dst])
                             / reference, -1., 1.)
    return np.column_stack((context['singleton'][chunks], context['wait'][chunks], advance, slack,
                            partner_advance)).astype(np.float32)
