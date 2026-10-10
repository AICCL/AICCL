"""Current-slot planned arrivals: actor observation only, never physical state."""
import math
import numpy as np
def planned_arrival(problem,state,chunk,edge,topo):
    c,e=int(chunk),int(edge);src,dst=int(topo.edge_src[e]),int(topo.edge_dst[e])
    if state['slot']<0 or not (0<=c<problem.C and 0<=e<problem.E):raise ValueError('Invalid prefix action')
    red=state['collective_type']=='allreduce' and state['mode'][c]==0
    aggregate=bool(red and state['partial'][c,dst].any())
    if red:
        contributions=state['partial'][c,src].copy()
        if not contributions.any():raise ValueError('Planned source is not resident')
        if aggregate:
            partner=state['partial'][c,dst]
            if np.any(contributions&partner):raise ValueError('Planned aggregate overlaps')
            contributions|=partner
        kind='partial'
    elif state['collective_type']=='allreduce':
        if not state['full'][c,src]:raise ValueError('Planned full source is not resident')
        contributions=np.ones(topo.V,dtype=bool);kind='full'
    else:
        if not state['state'][c,src]:raise ValueError('Planned copy source is not resident')
        contributions=np.asarray(problem.initial_state[c],dtype=bool).copy();kind='copy'
    timing=topo.timing
    cost=float(timing.aggregate_cost[e] if aggregate else timing.copy_cost[e])
    work=[cost/timing.delta]+[float(timing.group_cost[g])/timing.delta for g,m in enumerate(timing.groups) if e in m]
    delay=max(max(1,math.ceil(w-1e-10)) for w in work)
    expected=int(timing.aggregate_delay[e] if aggregate else timing.copy_delay[e])
    if delay!=expected:raise ValueError('Prefix observation delay differs from physical service rule')
    if state['slot']+delay>problem.T:raise ValueError('Planned arrival outside actual horizon')
    contributions.setflags(write=False)
    return dict(chunk=c,dst=dst,contributions=contributions,wait=delay,
                wait_reference=delay/timing.refinement,partial=kind=='partial',
                kind=kind,arrival=state['slot']+delay,edge=e,planned=True)

def pending_snapshot(real_pending,planned):
    """Own tuple/event/array copies; later plan append cannot mutate old snapshots."""
    result=[]
    for event in tuple(real_pending)+tuple(planned):
        copied={**event};copied['contributions']=np.asarray(event['contributions'],dtype=bool).copy()
        copied['contributions'].setflags(write=False);result.append(copied)
    return tuple(result)

def recompute_prefix_stop(model,h_v,h_e,h_c,g_ctx,edge_src,edge_dst,micro_actions,device,
                          entropy_mode,stop_entropy_weight):
    """One exact pending pool per decision, including motion logits for STOP."""
    import torch
    from torch.distributions import Categorical
    from .policy_entropy import factorized_stop_entropy
    logps=[];entropies=[]
    for i,action in enumerate(micro_actions):
        size=len(action['cand_e']);chosen=int(action['action_idx'])
        if not action.get('has_stop') or size<1 or not 0<=chosen<=size or bool(action.get('stop'))!=(chosen==size):
            raise ValueError('Malformed prefix STOP rollout')
        if action.get('stop') and i!=len(micro_actions)-1:raise ValueError('Actions after STOP')
        if chosen<size and (int(action['cand_c'][chosen])!=action['selected_c'] or int(action['cand_e'][chosen])!=action['selected_e']):
            raise ValueError('Selected prefix candidate mismatch')
        if 'relations' not in action or action.get('plan_observation_schema')!='slot_prefix_future_v1':
            raise ValueError('Exact prefix observation missing')
        ce=torch.as_tensor(action['cand_e'],dtype=torch.long,device=device)
        cc=torch.as_tensor(action['cand_c'],dtype=torch.long,device=device)
        dyn=torch.as_tensor(action['cand_dyn_feats'],dtype=torch.float32,device=device)
        motion=model.get_candidate_logits(h_v,h_e,h_c,g_ctx,edge_src,edge_dst,ce,cc,dyn,relations=action['relations'])
        stop=model.get_stop_logit(g_ctx,torch.as_tensor(action['stop_feats'],dtype=torch.float32,device=device),size,dyn)
        logits=torch.cat((motion,stop.reshape(1)));dist=Categorical(logits=logits)
        logps.append(dist.log_prob(torch.as_tensor(chosen,dtype=torch.long,device=device)))
        entropies.append(factorized_stop_entropy(logits,stop_entropy_weight) if entropy_mode=='factorized_stop' else dist.entropy())
    return torch.stack(logps),torch.stack(entropies)
