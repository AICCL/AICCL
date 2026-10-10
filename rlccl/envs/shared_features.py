"""State features for a single shared policy; no reference trajectories."""
import numpy as np
from .resource_potential import resource_route_features
from .candidate_control import outstanding_targets

def delay_distances(topo):
    d=np.full((topo.V,topo.V),np.inf);np.fill_diagonal(d,0)
    for e,(u,v) in enumerate(topo.edges):d[u,v]=min(d[u,v],topo.timing.copy_delay[e])
    for k in range(topo.V):d=np.minimum(d,d[:,k,None]+d[None,k,:])
    return d

def shared_relation_features(state,chunks,edges,topo,selected):
    route=copy_route_features(state,chunks,edges,topo,selected)
    targets=outstanding_targets(state,selected)
    resource=(resource_route_features(state,chunks,edges,topo,targets)
              if state['collective_type']!='allreduce' else np.zeros((len(chunks),4),np.float32))
    # Saturation is a feature scale only, never an action or hard-mask rule.
    route=np.tanh(np.column_stack((route,resource))/4).astype(np.float32)
    task=np.zeros((len(chunks),4),np.float32)
    task[:,['allgather','allreduce','alltoall','alltoallv'].index(state['collective_type'])]=1
    return dict(route=route,task=task)

def progress_potential(problem,state):
    demands=state['demands'];total=max(1,problem.demands.sum())
    satisfied=1-float(demands.sum())/total
    if state['collective_type']=='allreduce':
        from .runtime_state import allreduce_merge_potential
        return 4*allreduce_merge_potential(state)/problem.C+4*satisfied
    topo=problem.topology_info
    if not hasattr(topo,'_progress_distance'):topo._progress_distance=delay_distances(topo)
    d=topo._progress_distance;resident=state['state']
    best=np.where(resident[:,:,None],d[None],np.inf).min(axis=1)
    for event in state.get('pending',[]):
        c=event['chunk'];best[c]=np.minimum(best[c],event['arrival']-state['slot']+d[event['dst']])
    deficit=np.where(demands,np.minimum(best,problem.T),0).sum()/total/max(1,topo.timing.refinement)
    return 8*satisfied-2*float(deficit)

def copy_route_features(state, chunks, edges, topo, selected_arrivals=None):
    """Vectorized marginal future-distance gain; hints never change masks."""
    features = np.zeros((len(chunks), 12), dtype=np.float32)
    if state['collective_type'] == 'allreduce' or not len(chunks):
        return features
    from .shared_features import delay_distances
    if not hasattr(topo, '_copy_delay_distances'):
        topo._copy_delay_distances = delay_distances(topo)
    distance = topo._copy_delay_distances
    resident = state['state']
    timing = topo.timing
    delay = timing.copy_delay
    reference = max(1, timing.refinement)
    cache = getattr(topo, '_copy_route_context', None)
    if cache is None or cache['state'] is not state:
        best = np.where(resident[:, :, None], distance[None], np.inf).min(axis=1)
        for event in state.get('pending', []):
            if event['kind'] in ('copy', 'full'):
                c = event['chunk']
                best[c] = np.minimum(best[c], event['arrival']-state['slot']+distance[event['dst']])
        incoming_delay = np.where(resident[:, topo.edge_src], delay[None], np.inf)
        nearest_incoming = np.full((len(resident), topo.V), np.inf)
        for v in range(topo.V):
            es = np.flatnonzero(topo.edge_dst == v)
            if len(es):
                nearest_incoming[:, v] = incoming_delay[:, es].min(axis=1)
        work = timing.copy_work.copy()
        for g, members in enumerate(timing.groups):
            work[members] += timing.group_pattern[:, g].sum()
        cache = dict(state=state, best=best, incoming=nearest_incoming, work=work)
        topo._copy_route_context = cache
    targets = outstanding_targets(state, selected_arrivals)
    count = targets.sum(axis=1)
    mask = targets[chunks]
    denominators = np.maximum(1, count[chunks])
    best = cache['best'][chunks]
    if selected_arrivals is not None:
        selected = selected_arrivals[chunks]
        reserve = np.where(selected[:, :, None],
            cache['incoming'][chunks, :, None] + distance[None], np.inf).min(axis=1)
        best = np.minimum(best, reserve)
    source, destination = topo.edge_src[edges], topo.edge_dst[edges]
    after_all = distance[destination]
    before_all = distance[source]
    gain = np.where(mask, np.maximum(0, best-after_all), 0)
    after = np.where(mask, after_all, np.inf).min(axis=1)
    before = np.where(mask, before_all, np.inf).min(axis=1)
    endpoint_need = targets.sum(axis=0) / max(1, len(targets))
    source_traffic = (resident & targets.any(axis=1)[:, None]).sum(axis=0) / max(1, len(targets))
    features[:, 0] = gain.max(axis=1) / reference
    features[:, 1] = gain.sum(axis=1) / denominators / reference
    features[:, 2] = (gain>0).sum(axis=1) / denominators
    features[:, 3] = (before-after) / reference
    features[:, 4] = delay[edges] / reference
    features[:, 5] = after / reference
    features[:, 6] = resident[chunks].sum(axis=1) / max(1, topo.V)
    features[:, 7] = targets[chunks, destination]
    features[:, 8] = endpoint_need[destination]
    features[:, 9] = source_traffic[source]
    features[:, 10] = np.where(mask, best, 0).max(axis=1) / reference
    features[:, 11] = cache['work'][edges]
    features[count[chunks]==0] = 0
    return np.clip(features, -40, 40)


