"""PPO trainer for collective communication optimization."""

import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from .buffer import SlotBuffer
from .kl_guard import guarded_ppo_update
from .gradient_control import clip_policy_and_value_gradients
from ..envs.decoder import SlotDecoder, recompute_logp_slot
from ..envs.evaluator import evaluate_schedule
from ..envs.runtime_state import (
    allreduce_merge_potential,
    apply_slot_schedule,
    init_runtime_state,
    is_completed,
    remaining_demand,
)


def _disable_tqdm():
    """Keep tqdm enabled by default unless explicitly disabled."""
    override = os.environ.get("RLCCL_TQDM")
    if override is not None:
        return override.strip() in {"0", "false", "False", "off", "OFF"}
    return False


def compute_gae_advantages(rewards, values, dones, gamma, gae_lambda, durations=None):
    """Compute Generalized Advantage Estimation.
    
    Args:
        rewards: Tensor of rewards, shape (T,)
        values: Tensor of value estimates, shape (T,)
        dones: Tensor of done flags, shape (T,)
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        
    Returns:
        advantages: GAE advantages, shape (T,)
        returns: Returns (advantages + values), shape (T,)
    """
    advantages = []
    gae = 0.0
    
    for t in reversed(range(len(rewards))):
        if dones[t]:
            gae = 0.0
            next_val = 0.0
        else:
            next_val = values[t + 1].item() if t + 1 < len(values) else 0.0
        
        duration = durations[t] if durations is not None else 1.0
        discount = gamma ** duration
        delta = rewards[t] + discount * next_val - values[t]
        gae = delta + discount * gae_lambda ** duration * gae
        advantages.insert(0, gae)
    
    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = advantages + values
    
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages, returns


def train_epoch(model, train_problems, device, config, epoch, total_epochs,
                optimizer=None, metrics=None, episode_indices=None):
    """Train for one epoch.
    
    Args:
        model: SlotLevelPolicy model
        train_problems: List of (scenario_id, ProblemInstance)
        device: torch device
        config: Training configuration dict
        epoch: Current epoch number (0-indexed)
        total_epochs: Total number of epochs
        
    Returns:
        avg_policy_loss: Average policy loss
        avg_value_loss: Average value loss
        avg_entropy: Average configured exploration regularizer (full entropy by default)
        total_reward: Total average reward
    """
    model.train()
    decision_ppo = bool(config.get("decision_ppo", False))
    entropy_mode=config.get('policy_entropy','categorical')
    stop_entropy_weight=float(config.get('stop_entropy_weight',1.0))
    if entropy_mode not in ('categorical','factorized_stop'):
        raise ValueError('Unknown policy entropy regularizer')
    if entropy_mode=='factorized_stop':
        if not getattr(model,'use_stop',False) or not np.isfinite(stop_entropy_weight) or stop_entropy_weight<=0:
            raise ValueError('Factorized STOP entropy requires STOP and a finite positive weight')
    return_scale = float(config.get("return_scale", 1.0))
    critic_encoder_grad = config.get('critic_encoder_grad', True)
    separate_clipping = config.get('separate_gradient_clipping', False)
    if type(critic_encoder_grad) is not bool or type(separate_clipping) is not bool:
        raise ValueError('Gradient routing and clipping flags must be boolean')
    model.critic_encoder_grad = critic_encoder_grad
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    started = time.perf_counter()
    episodes = []
    slot_buffer = SlotBuffer()
    
    # Collect experience
    episode_target = config.get('episodes_per_update')
    if episode_indices is not None:
        indices = np.asarray(episode_indices)
        if indices.ndim != 1 or indices.dtype.kind not in 'iu' or len(indices) != episode_target:
            raise ValueError('Frozen episode order must match the explicit episode budget')
        if np.any(indices < 0) or np.any(indices >= len(train_problems)):
            raise ValueError('Frozen episode index outside the training corpus')
    elif episode_target is not None:
        from .sampling import balanced_episode_order
        indices = balanced_episode_order(train_problems, episode_target)
    else:
        indices = np.random.permutation(len(train_problems))
    slots_collected = 0
    micro_actions_collected = 0
    sampled_stops_collected = 0
    batch_target = config['batch_target']
    
    pbar = tqdm(indices, desc=f"Epoch {epoch+1}/{total_epochs} - Collecting", disable=_disable_tqdm())
    
    for idx in pbar:
        if episode_target is None and slots_collected >= batch_target:
            break
        
        scenario_id, problem = train_problems[idx]
        if np.sum(problem.demands) == 0:
            if episode_target is not None:
                raise ValueError('Cannot skip a zero-demand episode in a fixed budget')
            continue
        
        topo_info = getattr(problem, 'topology_info', None)
        if topo_info is None:
            if episode_target is not None:
                raise ValueError('Cannot skip a missing-topology episode in a fixed budget')
            continue
        
        decoder = SlotDecoder(topo_info)
        runtime_state = init_runtime_state(problem)
        initial_total_demands = max(1.0, float(np.sum(problem.demands)))
        timing = getattr(topo_info, "timing", None)
        duration = 1 / timing.refinement if timing else 1.0
        shaping_coef = float(config.get('allreduce_shaping_coef', 0.0))
        if not np.isfinite(shaping_coef) or shaping_coef < 0:
            raise ValueError('AllReduce shaping coefficient must be finite and nonnegative')
        
        # Get static info once per problem (shared across all slots of this problem)
        static_info = decoder.get_static_info()
        
        for t in range(problem.T):
            prev_merge_potential = allreduce_merge_potential(runtime_state) / problem.C
            if config.get('shared_progress_shaping', False):
                from ..envs.shared_features import progress_potential
                previous_progress = progress_potential(problem, runtime_state)
            Y_t, logp_slot, entropy_slot, value, state_info, micro_actions = decoder.decode_slot(
                model, problem, runtime_state, t, problem.T, train=True
            )

            runtime_state, _ = apply_slot_schedule(
                problem,
                runtime_state,
                Y_t,
                topology_info=topo_info,
                validate=False,
            )

            remaining = float(remaining_demand(problem, runtime_state))
            slot_reward = -duration if timing else -remaining / initial_total_demands
            terminal_deficit = remaining / initial_total_demands
            if problem.collective_type == "allreduce":
                from .own_advantages import integer_allreduce_consolidation
                terminal_deficit = .5*(terminal_deficit + 1.0 - integer_allreduce_consolidation(runtime_state))
            episode_success = is_completed(problem, runtime_state)
            episode_timeout = (t == problem.T - 1)
            episode_end = episode_success or episode_timeout
            if shaping_coef > 0.0 and problem.collective_type == "allreduce":
                # Each shard contributes at most one unit, independent of chunk
                # factor. In-flight partials are included by merge_potential.
                # Finite-horizon terminal potential is zero on both completion
                # and timeout, so the discounted sum telescopes to -Phi(s0).
                next_merge_potential = (0.0 if episode_end else
                                        allreduce_merge_potential(runtime_state) / problem.C)
                slot_reward += shaping_coef * (
                    config['gamma'] ** duration * next_merge_potential - prev_merge_potential
                )
            if config.get('shared_progress_shaping', False):
                next_progress = 0.0 if episode_end else progress_potential(problem, runtime_state)
                slot_reward += config['gamma'] ** duration * next_progress - previous_progress
            if timing and episode_timeout and not episode_success:
                slot_reward -= (2.0 if decision_ppo else problem.T * duration + 2.0)
                slot_reward -= float(config.get('training_failure_deficit_coef', 0.0))*terminal_deficit
            
            slot_buffer.add(
                state_info=state_info,
                actions=micro_actions,
                logprob_slot=logp_slot.detach().cpu(),
                value=value.detach().cpu() * return_scale,
                reward=slot_reward,
                done=episode_end,
                static_info=static_info, duration=duration  # Shared reference, not copied
            )
            
            slots_collected += 1
            micro_actions_collected += sum(not action.get('stop', False) for action in micro_actions)
            sampled_stops_collected += sum(action.get('stop', False) for action in micro_actions)
            pbar.set_postfix({'slots': slots_collected, 'target': batch_target})
            
            if episode_end:
                episodes.append({"id": scenario_id, "collective": problem.collective_type,
                                 "topology_family": getattr(problem, 'topology_family', None),
                                 "success": bool(episode_success), "slots": t + 1,
                                 "remaining": int(remaining), "reference_time": (t + 1) * duration,
                                 "own_environment_return": -(t+1)*duration - (0.0 if episode_success else 2.0+float(config.get("training_failure_deficit_coef",0.0))*terminal_deficit)})
                break
    
    if len(slot_buffer) == 0:
        print("  No slots collected, skipping update")
        return 0.0, 0.0, 0.0, 0.0
    collection_seconds = time.perf_counter() - started
    
    # Compute advantages
    rewards = torch.tensor(slot_buffer.slot_rewards, dtype=torch.float32)
    values = torch.tensor([v.item() for v in slot_buffer.slot_values], dtype=torch.float32)
    dones = torch.tensor(slot_buffer.slot_dones, dtype=torch.float32)
    
    advantages, returns = compute_gae_advantages(
        rewards, values, dones, config['gamma'], config['gae_lambda'], slot_buffer.slot_durations
    )
    
    if config.get('own_input_rloo', False):
        from .own_advantages import own_input_advantages
        ends=np.flatnonzero(slot_buffer.slot_dones)
        starts=np.r_[0,ends[:-1]+1]
        advantages=own_input_advantages([row['id'] for row in episodes],
            [row['own_environment_return'] for row in episodes], starts,ends,len(slot_buffer))

    target_kl = config.get('target_joint_kl')
    if target_kl is not None and (not np.isfinite(target_kl) or target_kl <= 0):
        raise ValueError('target_joint_kl must be finite and positive')
    buffer_size = len(slot_buffer)
    total_reward = rewards.mean().item()

    def optimize_rollout():
        # PPO Update
        buffer_size = len(slot_buffer)
        indices_ppo = np.arange(buffer_size)
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0
        kl_sum = 0.0
        kl_samples = 0
        clipped_sum = 0.0
        gradient_norms = []
        optimizer_steps = 0
        kl_early_stopped = False
        maximum_minibatch_kl = 0.0
        shared_budget_records=[]
        
        for ppo_epoch in range(config['ppo_epochs']):
            np.random.shuffle(indices_ppo)
            mb_size = config['mini_batch_size']
            
            pbar_ppo = tqdm(
                range(0, buffer_size, mb_size),
                desc=f"  PPO Epoch {ppo_epoch+1}/{config['ppo_epochs']}",
                disable=_disable_tqdm(),
            )
            
            for start in pbar_ppo:
                end = min(start + mb_size, buffer_size)
                batch_indices = indices_ppo[start:end]
                
                optimizer.zero_grad()
                loss_accum = 0.0
                policy_and_entropy_accum = 0.0
                policy_loss_accum = 0.0
                value_loss_accum = 0.0
                entropy_accum = 0.0
                minibatch_kl = []
                
                for idx_sample in batch_indices:
                    state_info = slot_buffer.slot_states[idx_sample]
                    micro_actions = slot_buffer.slot_actions[idx_sample]
                    static_info = slot_buffer.slot_static_infos[idx_sample]
                    old_logp = slot_buffer.slot_logprobs[idx_sample].to(device)
                    advantage = advantages[idx_sample].to(device)
                    ret = returns[idx_sample].to(device)
                    
                    logp_new, entropy_new, value_new = recompute_logp_slot(
                        model, state_info, micro_actions, device, static_info, return_decisions=decision_ppo,
                        entropy_mode=entropy_mode, stop_entropy_weight=stop_entropy_weight
                    )
                    
                    if decision_ppo:
                        old_logp = torch.as_tensor([a['old_logprob'] for a in micro_actions], device=device)
                        assert old_logp.shape == logp_new.shape
                    log_ratio = logp_new - old_logp
                    ratio = torch.exp(log_ratio)
                    sample_kl = float((ratio - 1 - log_ratio).detach().mean()) if log_ratio.numel() else 0.0
                    if not np.isfinite(sample_kl):
                        raise FloatingPointError('Nonfinite joint-slot KL')
                    minibatch_kl.append(sample_kl)
                    kl_sum += sample_kl
                    kl_samples += 1
                    clipped_sum += float((torch.abs(ratio - 1) > config['clip_eps']).float().mean().detach()) if ratio.numel() else 0.0
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1.0 - config['clip_eps'], 
                                       1.0 + config['clip_eps']) * advantage
                    policy_loss = -torch.min(surr1, surr2)
                    if decision_ppo:
                        policy_loss = policy_loss.mean() if policy_loss.numel() else value_new.sum()*0
                        entropy_new = entropy_new.mean() if entropy_new.numel() else value_new.sum()*0
                    
                    value_loss = F.mse_loss(value_new.squeeze(), ret / return_scale)
                    loss = (policy_loss + config['value_coef'] * value_loss 
                           - config['entropy_coef'] * entropy_new)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite PPO loss; refusing to save a corrupted model")
                    loss_accum += loss
                    policy_and_entropy_accum += policy_loss-config['entropy_coef']*entropy_new
                    
                    policy_loss_accum += policy_loss.item()
                    value_loss_accum += value_loss.item()
                    entropy_accum += entropy_new.item()
                
                current_kl = float(np.mean(minibatch_kl))
                maximum_minibatch_kl = max(maximum_minibatch_kl, current_kl)
                if target_kl is not None and current_kl > target_kl:
                    kl_early_stopped = True
                    break  # do not apply this minibatch's update
                loss_accum = loss_accum / len(batch_indices)
                ratio_budget=config.get('shared_value_gradient_ratio')
                if ratio_budget is not None:
                    from .shared_gradient_budget import shared_parameters,apply_shared_budget
                    shared=shared_parameters(model)
                    policy_grads=torch.autograd.grad(policy_and_entropy_accum/len(batch_indices),shared,retain_graph=True,allow_unused=True)
                loss_accum.backward()
                if ratio_budget is not None:
                    shared_budget_records.append(apply_shared_budget(shared,policy_grads,ratio_budget,config.get('project_conflicting_value_gradient',False)))
                
                if 'max_grad_norm' in config:
                    norm = clip_policy_and_value_gradients(model, config['max_grad_norm'],
                                                           separate=separate_clipping)
                    gradient_norms.append(float(norm))
                
                optimizer.step()
                optimizer_steps += 1
                if any(not torch.isfinite(p).all() for p in model.parameters()):
                    raise FloatingPointError("Non-finite model parameter after PPO update")
                
                total_policy_loss += policy_loss_accum
                total_value_loss += value_loss_accum
                total_entropy += entropy_accum
                num_updates += len(batch_indices)
                
                pbar_ppo.set_postfix({
                    'p_loss': policy_loss_accum / len(batch_indices),
                    'v_loss': value_loss_accum / len(batch_indices),
                })
            if kl_early_stopped:
                break
        
        return dict(total_policy_loss=total_policy_loss, total_value_loss=total_value_loss,
                    total_entropy=total_entropy, num_updates=num_updates, kl_sum=kl_sum,
                    kl_samples=kl_samples, clipped_sum=clipped_sum, gradient_norms=gradient_norms,
                    optimizer_steps=optimizer_steps, kl_early_stopped=kl_early_stopped,
                    maximum_minibatch_kl=maximum_minibatch_kl,shared_gradient_budget=shared_budget_records)

    def measure_rollout_kl():
        final_joint_kls = []
        if target_kl is not None:
            with torch.no_grad():
                for i in range(buffer_size):
                    new_logp, _, _ = recompute_logp_slot(
                        model, slot_buffer.slot_states[i], slot_buffer.slot_actions[i],
                        device, slot_buffer.slot_static_infos[i], return_decisions=decision_ppo)
                    old = (torch.as_tensor([a['old_logprob'] for a in slot_buffer.slot_actions[i]], device=device) if decision_ppo else slot_buffer.slot_logprobs[i].to(device))
                    difference = new_logp - old
                    if decision_ppo:
                        final_joint_kls.extend((torch.expm1(difference)-difference).cpu().tolist())
                    else:
                        final_joint_kls.append(float(torch.expm1(difference) - difference))
            if not np.isfinite(final_joint_kls).all():
                raise FloatingPointError('Nonfinite final rollout joint-slot KL')
        return final_joint_kls

    guard = bool(config.get('post_update_kl_guard', False))
    guard_metrics = {'enabled': guard}
    if guard:
        update_stats, final_joint_kls, guard_metrics = guarded_ppo_update(
            model, optimizer, optimize_rollout, measure_rollout_kl, target_kl,
            max_backtracks=config.get('joint_kl_max_backtracks', 4),
            factor=config.get('joint_kl_backtrack_factor', 0.5))
    else:
        update_stats = optimize_rollout()
        final_joint_kls = measure_rollout_kl()
    if update_stats is None:
        update_stats = dict(total_policy_loss=0., total_value_loss=0., total_entropy=0.,
                            num_updates=0, kl_sum=0., kl_samples=0, clipped_sum=0.,
                            gradient_norms=[], optimizer_steps=0, kl_early_stopped=True,
                            maximum_minibatch_kl=0.)
    num_updates = update_stats['num_updates']
    kl_sum, kl_samples = update_stats['kl_sum'], update_stats['kl_samples']
    clipped_sum = update_stats['clipped_sum']
    gradient_norms = update_stats['gradient_norms']
    optimizer_steps = update_stats['optimizer_steps']
    kl_early_stopped = update_stats['kl_early_stopped']
    maximum_minibatch_kl = update_stats['maximum_minibatch_kl']
    avg_policy_loss = update_stats['total_policy_loss'] / max(num_updates, 1)
    avg_value_loss = update_stats['total_value_loss'] / max(num_updates, 1)
    avg_entropy = update_stats['total_entropy'] / max(num_updates, 1)
    if metrics is not None:
        metrics.update(policy_entropy_regularizer=entropy_mode, stop_entropy_weight=stop_entropy_weight, kl_unit='autoregressive_decision' if decision_ppo else 'joint_slot', return_scale=return_scale, all_parameters_trainable=all(p.requires_grad for p in model.parameters()), slots_collected=slots_collected, episodes=len(episodes),
                       episode_records=episodes, micro_actions=micro_actions_collected,
                       sampled_stops=sampled_stops_collected, collection_seconds=collection_seconds,
                       optimization_seconds=time.perf_counter() - started - collection_seconds,
                       rollout_success_rate=float(np.mean([x['success'] for x in episodes])) if episodes else None,
                       rollout_mean_reference_time=float(np.mean([x['reference_time'] for x in episodes])) if episodes else None,
                       rollout_by_collective={ct: {"count": sum(x['collective'] == ct for x in episodes),
                           "success_rate": float(np.mean([x['success'] for x in episodes if x['collective'] == ct]))}
                           for ct in sorted({x['collective'] for x in episodes})},
                       approx_kl=kl_sum / max(kl_samples, 1), clip_fraction=clipped_sum / max(kl_samples, 1),
                       mean_gradient_norm=float(np.mean(gradient_norms)) if gradient_norms else None,
                       max_gradient_norm=float(max(gradient_norms)) if gradient_norms else None,
                       optimizer_steps=optimizer_steps, kl_early_stopped=kl_early_stopped,
                       post_update_kl_guard=guard_metrics,
                       max_minibatch_joint_kl=maximum_minibatch_kl,
                       final_joint_kl_mean=float(np.mean(final_joint_kls)) if final_joint_kls else None,
                       final_joint_kl_max=float(max(final_joint_kls)) if final_joint_kls else None,
                       shared_gradient_budget=update_stats.get('shared_gradient_budget',[]),
                       episode_budget=episode_target,
                       train_seconds=time.perf_counter() - started)
    
    return avg_policy_loss, avg_value_loss, avg_entropy, total_reward


def evaluate_model(model, test_problems, device, metrics=None):
    """Evaluate model on test set.
    
    Args:
        model: SlotLevelPolicy model
        test_problems: List of (scenario_id, ProblemInstance)
        device: torch device
        
    Returns:
        avg_score: Average evaluation score
        avg_steps: Average completion steps
    """
    model.eval()
    scores = []
    completion_steps = []
    records = []
    
    with torch.no_grad():
        for scenario_id, problem in tqdm(test_problems, desc="Evaluating", disable=_disable_tqdm()):
            topo_info = getattr(problem, 'topology_info', None)
            if topo_info is None:
                continue
            
            decoder = SlotDecoder(topo_info)
            runtime_state = init_runtime_state(problem)
            schedule = []
            
            for t in range(problem.T):
                Y_t, _, _, _, _, _ = decoder.decode_slot(
                    model, problem, runtime_state, t, problem.T, train=False
                )
                schedule.append(Y_t)

                runtime_state, _ = apply_slot_schedule(
                    problem,
                    runtime_state,
                    Y_t,
                    topology_info=topo_info,
                    validate=False,
                )

                if is_completed(problem, runtime_state):
                    completion_steps.append(t + 1)
                    break
            else:
                completion_steps.append(problem.T)
            
            while len(schedule) < problem.T:
                schedule.append(np.zeros((problem.C, problem.E), dtype=int))
            
            score, error = evaluate_schedule(schedule, problem)
            timing = getattr(topo_info, "timing", None)
            records.append({"id": scenario_id, "collective": problem.collective_type,
                            "success": error == "", "score": float(score),
                            "slots": completion_steps[-1],
                            "reference_time": completion_steps[-1] / timing.refinement if timing else completion_steps[-1]})
            if error == "":  # Empty string means success, not None
                scores.append(score)
            else:
                if getattr(topo_info, "timing", None):
                    scores.append(score)  # Failed timed episodes must influence model selection.
                # Log the error for debugging
                print(f"  Warning: Problem {scenario_id} evaluation failed: {error}")
    
    if scores:
        avg_score = np.mean(scores)
        avg_steps = np.mean(completion_steps)
    else:
        # Return a very poor score instead of 0.0, since score is negative (closer to 0 is better)
        # Use -(T_max + 10) as default poor score
        avg_score = -1000.0
        avg_steps = 0.0
    
    if metrics is not None:
        metrics.update(count=len(records), success_rate=float(np.mean([x['success'] for x in records])) if records else None,
                       by_collective={ct: {"count": sum(x['collective'] == ct for x in records),
                           "success_rate": float(np.mean([x['success'] for x in records if x['collective'] == ct])),
                           "mean_score": float(np.mean([x['score'] for x in records if x['collective'] == ct]))}
                           for ct in sorted({x['collective'] for x in records})}, records=records)
    return avg_score, avg_steps
