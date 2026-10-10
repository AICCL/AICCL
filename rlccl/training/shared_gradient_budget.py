"""Route own PPO losses through one shared trainable encoder."""
import math
import torch
PREFIXES=('node_encoder','edge_encoder','chunk_encoder','layer1','layer2','global_pool','pending_encoder')
def shared_parameters(model):return [p for n,p in model.named_parameters() if n.startswith(PREFIXES)]
def budget_vectors(policy,total,ratio,project):
    if not math.isfinite(float(ratio)) or ratio<0:raise ValueError('Gradient ratio must be finite/nonnegative')
    value=total-policy;pn=torch.linalg.vector_norm(policy);vn=torch.linalg.vector_norm(value);dot=torch.dot(policy,value);projected=False
    if project and pn>1e-12 and dot<0:value=value-dot/(pn*pn)*policy;projected=True
    vafter=torch.linalg.vector_norm(value);scale=min(1.,float(ratio*pn/vafter)) if pn>1e-12 and vafter>1e-12 else 1.
    mixed=policy+scale*value
    if not torch.isfinite(mixed).all():raise FloatingPointError('Nonfinite routed PPO gradient')
    return mixed,dict(policy_norm=float(pn),weighted_value_norm_before=float(vn),weighted_value_norm_after=float(vafter)*scale,value_scale=scale,projected_conflict=projected,policy_value_dot_before=float(dot),policy_value_dot_after=float(torch.dot(policy,value))*scale)
def apply_shared_budget(params,policy_grads,ratio,project):
    ps=[torch.zeros_like(p).flatten() if g is None else g.detach().flatten() for p,g in zip(params,policy_grads)]
    totals=[torch.zeros_like(p).flatten() if p.grad is None else p.grad.detach().flatten() for p in params]
    mixed,info=budget_vectors(torch.cat(ps),torch.cat(totals),ratio,project);start=0
    for p in params:
        stop=start+p.numel();p.grad=mixed[start:stop].view_as(p).clone();start=stop
    return info
