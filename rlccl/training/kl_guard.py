"""Backtrack an entire PPO update using its sampled full-rollout joint KL.

This bounds the measured Monte Carlo k3 mean on the collected rollout, not
the exact KL of the complete policy or its out-of-sample generalization.
"""

import copy
import math

import torch

from .checkpoint import capture_rng_state, restore_rng_state


def guarded_ppo_update(model, optimizer, update, measure, target,
                       max_backtracks=4, factor=0.5):
    """Retry the same rollout and minibatch permutations at smaller step sizes.

    Rejection restores parameters, buffers, Adam moments/step counts and all
    RNG states. Collection is never repeated and its episode budget is intact.
    A successful retry retains its optimizer state but restores the base LR.
    If every trial is rejected, the original model and optimizer are retained.
    """
    if target is None or not math.isfinite(target) or target <= 0:
        raise ValueError('Post-update guard requires a finite positive KL target')
    if isinstance(max_backtracks, bool) or not isinstance(max_backtracks, int) or max_backtracks < 0:
        raise ValueError('max_backtracks must be a nonnegative integer')
    if not math.isfinite(factor) or not 0 < factor < 1:
        raise ValueError('Backtrack factor must be strictly between zero and one')
    base_lrs = [group['lr'] for group in optimizer.param_groups]
    if any(not math.isfinite(lr) or lr <= 0 for lr in base_lrs):
        raise ValueError('Optimizer learning rates must be finite and positive')
    model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    rng_state = capture_rng_state()
    attempts = []

    def restore():
        model.load_state_dict(model_state)
        # load_state_dict can retain tensor references, so each retry must get
        # fresh copies or an attempted Adam step would corrupt the snapshot.
        optimizer.load_state_dict(copy.deepcopy(optimizer_state))
        optimizer.zero_grad(set_to_none=True)
        restore_rng_state(rng_state)

    for trial in range(max_backtracks + 1):
        if trial:
            restore()
        scale = factor ** trial
        for group, lr in zip(optimizer.param_groups, base_lrs):
            group['lr'] = lr * scale
        record = dict(trial=trial, learning_rate_scale=scale, accepted=False)
        try:
            stats = update()
            record['optimizer_steps'] = stats['optimizer_steps']
            kls = measure()
            if not kls or not all(math.isfinite(x) for x in kls):
                raise FloatingPointError('Nonfinite or empty post-update rollout KL')
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError('Nonfinite post-update parameter')
            record.update(mean_k3=sum(kls) / len(kls), max_sample_k3=max(kls))
            record['accepted'] = record['mean_k3'] <= target
        except FloatingPointError as error:
            record['error'] = str(error)
        except Exception:
            restore()
            raise
        attempts.append(record)
        if record['accepted']:
            for group, lr in zip(optimizer.param_groups, base_lrs):
                group['lr'] = lr
            return stats, kls, dict(enabled=True, target_mean_k3=target,
                                    accepted=True, attempts=attempts,
                                    accepted_learning_rate_scale=scale,
                                    rejected_updates=trial)
    restore()
    kls = measure()
    if not kls or not all(math.isfinite(x) for x in kls) or sum(kls) / len(kls) > target:
        raise FloatingPointError('Restored rollout does not satisfy the KL target')
    return None, kls, dict(enabled=True, target_mean_k3=target, accepted=False,
                          attempts=attempts, accepted_learning_rate_scale=0.,
                          rejected_updates=len(attempts))
