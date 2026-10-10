"""Optional independent norm caps for policy/encoder and value-head gradients."""
import math
import torch


def clip_policy_and_value_gradients(model, max_norm, separate=False):
    if type(separate) is not bool:
        raise ValueError('Separate clipping flag must be boolean')
    if not math.isfinite(max_norm) or max_norm <= 0:
        raise ValueError('Gradient norm cap must be finite and positive')
    if not separate:
        return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, error_if_nonfinite=True)
    policy, value = [], []
    for name, parameter in model.named_parameters():
        (value if name.startswith('critic.') else policy).append(parameter)
    if not policy or not value:
        raise ValueError('Separate clipping requires both policy and critic parameter groups')
    # Validate both before clipping either, so NaNs cannot leave partial mutation.
    for p in policy + value:
        if p.grad is not None and not torch.isfinite(p.grad).all():
            raise FloatingPointError('Nonfinite gradient')
    actor_norm = torch.nn.utils.clip_grad_norm_(policy, max_norm, error_if_nonfinite=True)
    critic_norm = torch.nn.utils.clip_grad_norm_(value, max_norm, error_if_nonfinite=True)
    return torch.sqrt(actor_norm.square() + critic_norm.square())
