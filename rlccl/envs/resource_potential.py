"""Numerically equivalent static-topology feature caching prototype; no active source mutation."""
import numpy as np
def resource_route_features(state,chunks,edges,topo,targets):
 out=np.zeros((len(chunks),4),dtype=np.float32)
 G=len(topo.timing.groups);V=topo.V
 if not G or not len(chunks):return out
 cached=getattr(topo,'_resource_route_static',None)
 if cached is None:
  pattern=np.zeros((len(topo.edge_src),G),dtype=np.float64)
  for g,members in enumerate(topo.timing.groups):pattern[members,g]=topo.timing.group_work[g]
  if not G or not len(chunks):return out
  # A single NIC group can have zero required work if another NIC bypasses it.
  # Pool normalized service across each node's send and receive groups.
  # These are state summaries, not new hard resource constraints or action labels.
  composite=[]
  for direction in (topo.edge_src,topo.edge_dst):
   for node in range(V):
    gs=[g for g,members in enumerate(topo.timing.groups)
        if len(members) and np.all(direction[members]==node)]
    if gs:composite.append(pattern[:,gs].sum(axis=1))
  if composite:pattern=np.column_stack((pattern,*composite))
  topo._resource_route_static=pattern
 else:
  pattern=cached
 G=pattern.shape[1]
 D=getattr(topo,'_resource_route_distances',None)
 if D is None:
  D=np.full((G,V,V),np.inf)
  D[:,np.arange(V),np.arange(V)]=0
  for e,(s,d) in enumerate(zip(topo.edge_src,topo.edge_dst)):D[:,s,d]=np.minimum(D[:,s,d],pattern[e])
  for v in range(V):D=np.minimum(D,D[:,:,v,None]+D[:,None,v,:])
  topo._resource_route_distances=D
 cache=getattr(topo,'_resource_route_context',None)
 if cache is None or cache['state'] is not state:
  resident=state['state'];best=np.full((G,len(resident),V),np.inf)
  for v in range(V):
   rows=np.flatnonzero(resident[:,v])
   if len(rows):best[:,rows,:]=np.minimum(best[:,rows,:],D[:,v,None,:])
  for event in state.get('pending',[]):
   if event['kind']=='copy':
    c,d=event['chunk'],event['dst'];best[:,c,:]=np.minimum(best[:,c,:],D[:,d,:])
  cache=dict(state=state,best=best);topo._resource_route_context=cache
 best=cache['best'];ref=max(1,topo.timing.refinement)
 # State is immutable within a slot. Only target rows changed by selected
 # arrivals need another resource-distance reduction; keep canonical C order
 # for the final sum, rather than incremental floating-point pressure updates.
 previous=cache.get('target_snapshot')
 if previous is None:
  before=np.where(targets[None],best,0).max(axis=2)
  before=np.nan_to_num(before,nan=0,posinf=40*ref)
  cache['target_before']=before
 else:
  before=cache['target_before']
  changed=np.flatnonzero(np.any(previous!=targets,axis=1))
  if len(changed):
   values=np.where(targets[changed][None],best[:,changed,:],0).max(axis=2)
   before[:,changed]=np.nan_to_num(values,nan=0,posinf=40*ref)
 cache['target_snapshot']=targets.copy()
 pressure=before.sum(axis=1)
 future=np.minimum(best[:,chunks,:],D[:,topo.edge_dst[edges],:])
 after=np.where(targets[chunks][None],future,0).max(axis=2)
 after=np.nan_to_num(after,nan=0,posinf=40*ref)
 gain=np.maximum(0,before[:,chunks]-after)
 cost=pattern[edges].T;waste=np.maximum(0,cost-gain)
 out[:,0]=gain.max(axis=0)/ref
 out[:,1]=((ref+pressure[:,None])*waste).sum(axis=0)/(ref*ref)
 out[:,2]=(pressure[:,None]*gain).sum(axis=0)/(ref*ref)
 out[:,3]=np.where(cost>0,pressure[:,None],0).max(axis=0)/ref
 out[~targets[chunks].any(axis=1)]=0
 return np.clip(out,0,40)
