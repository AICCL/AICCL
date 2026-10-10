"""NumPy-only reservation and candidate-coverage helpers."""
import numpy as np


def outstanding_targets(state, selected_arrivals=None):
    """Unsatisfied targets without a final value already pending/reserved.

    Reservation suppresses further work for a fully covered copy shard; it
    does not make the pending value resident or available as a source. Partial
    AllReduce tokens are not final target deliveries.
    """
    targets = np.asarray(state['demands'], dtype=bool).copy()
    for event in state.get('pending', []):
        if event['kind'] in {'copy', 'full'}:
            targets[event['chunk'], event['dst']] = False
    if selected_arrivals is not None:
        selected_arrivals = np.asarray(selected_arrivals, dtype=bool)
        if selected_arrivals.shape != targets.shape:
            raise ValueError('Selected arrival reservation shape mismatch')
        if state['collective_type'] == 'allreduce':
            # Decoder records partial reservations in the same matrix. Only
            # full-mode rows may be treated as final-result reservations.
            targets[state['mode'] == 1] &= ~selected_arrivals[state['mode'] == 1]
        else:
            targets &= ~selected_arrivals
    return targets


def balanced_topk(chunks, scores, budget, rotation=0):
    """Keep the strong score pool with a bounded rotating coverage quota.

    Reserve at most one eighth of the pool for currently omitted shards.
    Replacing only that many low-score candidates avoids imposing equal shard
    quotas on resource-conflicting actions. On a static eligible set, every
    omitted shard gets a proposal within a bounded number of micro-steps.
    Pool order follows the legacy argsort, including its NumPy tie rule; store
    the complete pool in each PPO rollout rather than reconstructing ties.
    """
    chunks, scores = np.asarray(chunks), np.asarray(scores)
    if chunks.ndim != 1 or scores.shape != chunks.shape:
        raise ValueError('Candidate chunks/scores must be matching vectors')
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
        raise ValueError('Candidate budget must be a positive integer')
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite candidate score')
    if len(chunks) <= budget:
        return np.arange(len(chunks), dtype=np.int64)
    global_order = np.argsort(scores)
    selected = global_order[-budget:]
    # Vectorized best proposal per shard, without a Python sort per shard.
    shard_order = np.lexsort((np.arange(len(chunks)), -scores, chunks))
    ordered_chunks = chunks[shard_order]
    starts = np.flatnonzero(np.r_[True, ordered_chunks[1:] != ordered_chunks[:-1]])
    unique_chunks = ordered_chunks[starts]
    omitted = ~np.isin(unique_chunks, chunks[selected])
    proposals = shard_order[starts[omitted]]
    if not len(proposals):
        return selected
    quota = max(1, budget // 8)
    # Advance by a full quota so coverage does not move only one shard at a
    # time. This guarantee concerns eligible proposals, not chosen actions.
    offset = int(rotation) * quota % len(proposals)
    proposals = np.roll(proposals, -offset)[:min(quota, len(proposals))]
    # Keep one of the original best proposals for each globally covered shard
    # when duplicate proposals supply enough eviction slots. Otherwise rotate
    # anchor eviction as well, so coverage cannot permanently displace one
    # original shard while rotating only newly added shards.
    _, reversed_anchors = np.unique(chunks[selected[::-1]], return_index=True)
    anchor_positions = budget - 1 - reversed_anchors
    anchors = np.zeros(budget, dtype=bool)
    anchors[anchor_positions] = True
    redundant = selected[~anchors]
    count = len(proposals)
    evicted = redundant[:count]
    if len(evicted) < count:
        protected = selected[anchors]
        extra = count - len(evicted)
        start = int(rotation) * extra % len(protected)
        evicted = np.r_[evicted, np.roll(protected, -start)[:extra]]
    chosen = np.zeros(len(chunks), dtype=bool)
    chosen[selected] = True
    chosen[evicted] = False
    chosen[proposals] = True
    return global_order[chosen[global_order]]


def candidate_relations(state, chunks, edges, topo, initial_state, selected_arrivals=None):
    """Expose exact contributor/target identities to topology-aware pooling.

    Inputs remain resident/pending separately. These masks are actor features,
    not substitutes for runtime provenance validation or source feasibility.
    """
    chunks, edges = np.asarray(chunks), np.asarray(edges)
    src, dst = topo.edge_src[edges], topo.edge_dst[edges]
    n = topo.V
    if state['collective_type'] == 'allreduce':
        source = state['partial'][chunks, src].copy()
        destination = state['partial'][chunks, dst].copy()
        source[state['full'][chunks, src]] = True
        destination[state['full'][chunks, dst]] = True
    else:
        source = np.asarray(initial_state, dtype=bool)[chunks].copy()
        destination = source & state['state'][chunks, dst, None]
    targets = outstanding_targets(state, selected_arrivals)[chunks]
    pending = []
    for event in state.get('pending', []):
        c = event['chunk']
        if event['kind'] == 'partial':
            contributions = np.asarray(event['payload'], dtype=bool).copy()
        elif state['collective_type'] == 'allreduce':
            contributions = np.ones(n, dtype=bool)
        else:
            contributions = np.asarray(initial_state[c], dtype=bool).copy()
        pending.append(dict(chunk=c, dst=event['dst'], contributions=contributions,
                            wait=event['arrival'] - state['slot'],
                            wait_reference=(event['arrival'] - state['slot']) / topo.timing.refinement,
                            partial=event['kind'] == 'partial'))
    return dict(source=source, missing=source & ~destination, targets=targets,
                pending=pending, chunks_count=len(state['demands']))
