"""Soft AllReduce rendezvous hint from resident, pending and selected landings.

Only the neural residual sees this hint. It neither changes resource masks nor
executes a rescue/fallback. A selected landing is a scoring projection, never
an available source. Each candidate excludes its own resident contribution.
"""
import numpy as np


def meeting_advance(state, chunks, edges, topo, distance, selected_moves):
    result = np.zeros(len(chunks), dtype=np.float32)
    if state['collective_type'] != 'allreduce':
        return result
    for c in np.unique(chunks):
        if state['mode'][c] != 0:
            continue
        indices = np.flatnonzero(chunks == c)
        holders = np.flatnonzero(state['partial'][c].any(axis=1))
        for i in indices:
            src, dst = topo.edge_src[edges[i]], topo.edge_dst[edges[i]]
            partners = [int(selected_moves[c, v]) if selected_moves[c, v] >= 0 else int(v)
                        for v in holders if v != src]
            partners += [p['dst'] for p in state.get('pending', [])
                         if p['chunk'] == c and p['kind'] == 'partial']
            if partners:
                before = min(distance[src, v] for v in partners)
                after = min(distance[dst, v] for v in partners)
                result[i] = np.clip((before-after)/topo.timing.reference_time, -1., 1.)
    return result
