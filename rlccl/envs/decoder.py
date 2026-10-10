"""Slot decoder for autoregressive schedule generation."""

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from .problem import COPY_ONLY_COLLECTIVES, normalize_collective_type
from .policy_entropy import factorized_stop_entropy
from .runtime_state import (
    MODE_FULL,
    MODE_REDUCE,
    apply_slot_schedule,
    clone_runtime_state,
    get_runtime_demands,
    get_runtime_resident_matrix,
    init_runtime_state,
    is_completed,
)


def deterministic_action(logits, use_stop=False, selector='learned'):
    """Choose the binary STOP/continue decision, then a continuation action.

    STOP is one category while continuation is the union of all transmission
    categories. Comparing STOP with only the largest individual category can
    select STOP even when its total probability is tiny. The fixed 1/2 binary
    decision uses the same categorical logits as rollout and PPO replay; only
    deterministic decoding changes. The strong greedy and legacy decisions
    retain their existing semantics.
    """
    if not use_stop or selector == 'greedy':
        return torch.argmax(logits)
    if logits.ndim != 1 or logits.numel() < 2:
        raise ValueError('STOP decision requires a nonempty continuation pool')
    if bool(logits[-1] >= torch.logsumexp(logits[:-1], dim=0)):
        return logits.new_tensor(logits.numel() - 1, dtype=torch.long)
    return torch.argmax(logits[:-1])


class SlotDecoder:
    """Autoregressively construct one slot schedule Y_t."""

    def __init__(self, topology_info):
        self.topo = topology_info
        self.timing = getattr(topology_info, "timing", None)
        self.V = topology_info.V
        self.E = topology_info.E

        self.edge_src = np.asarray(topology_info.edge_src, dtype=np.int64)
        self.edge_dst = np.asarray(topology_info.edge_dst, dtype=np.int64)
        self.capacities = np.asarray(topology_info.capacities, dtype=np.float32)
        self.dist_matrix = np.asarray(topology_info.dist_matrix, dtype=np.float32)

        self.group_to_edges = []
        self.edge_to_groups = [[] for _ in range(self.E)]
        self.group_limits = []

        if getattr(self.topo, "shared_constraints", None):
            for g_idx, (edges, limit) in enumerate(self.topo.shared_constraints):
                self.group_to_edges.append(np.asarray(edges, dtype=np.int64))
                self.group_limits.append(float(limit))
                for e in edges:
                    self.edge_to_groups[e].append(g_idx)

        self.group_limits = (
            np.asarray(self.group_limits, dtype=np.float32)
            if self.group_limits else np.array([], dtype=np.float32)
        )
        self.num_groups = len(self.group_limits)

        self.out_deg = np.zeros(self.V, dtype=np.float32)
        self.in_deg = np.zeros(self.V, dtype=np.float32)
        np.add.at(self.out_deg, self.edge_src, 1)
        np.add.at(self.in_deg, self.edge_dst, 1)
        self.f_out = self.out_deg / max(1, self.E)
        self.f_in = self.in_deg / max(1, self.E)

        self.max_steps = int(self.capacities.sum()) + 5
        self.edge_src_t = torch.tensor(self.edge_src, dtype=torch.long)
        self.edge_dst_t = torch.tensor(self.edge_dst, dtype=torch.long)
        self._static_info_cpu = {
            "edge_src_t": self.edge_src_t.clone(),
            "edge_dst_t": self.edge_dst_t.clone(),
            "capacities": self.capacities.copy(),
            "max_steps": self.max_steps,
            "group_limits": self.group_limits.copy(),
            "edge_to_groups": self.edge_to_groups,
            "num_groups": self.num_groups,
            "V": self.V,
            "E": self.E,
        }

    def get_static_info(self):
        """Return static topology info shared across all slots."""
        return self._static_info_cpu

    def compute_dist_to_demand(self, demands):
        """Distance from every node to the nearest remaining demander for each shard."""
        C, V = demands.shape
        inf_dist = V + 1.0
        dist_to_demand = np.full((C, V), inf_dist, dtype=np.float32)
        for v in range(V):
            cs = np.where(demands[:, v])[0]
            if len(cs) > 0:
                dists = self.dist_matrix[:, v]
                dist_to_demand[cs, :] = np.minimum(dist_to_demand[cs, :], dists)
        return dist_to_demand

    def _get_node_features(self, runtime_state, t, T):
        resident = get_runtime_resident_matrix(runtime_state).astype(np.float32)
        demands = get_runtime_demands(runtime_state).astype(np.float32)
        C = resident.shape[0]
        f_held = resident.sum(axis=0) / max(1, C)
        f_need = demands.sum(axis=0) / max(1, C)
        f_time = np.full(self.V, t / max(1, T), dtype=np.float32)
        return np.stack([f_held, f_need, self.f_out, self.f_in, f_time], axis=1).astype(np.float32)

    def _get_chunk_features(self, collective_type, runtime_state):
        resident = get_runtime_resident_matrix(runtime_state).astype(np.float32)
        demands = get_runtime_demands(runtime_state).astype(np.float32)
        token_count_fraction = resident.sum(axis=1) / max(1, self.V)
        full_need_fraction = demands.sum(axis=1) / max(1, self.V)

        if collective_type == "allreduce":
            partial_mass = runtime_state["partial"].sum(axis=2).astype(np.float32)
            reduce_mass = partial_mass.max(axis=1) / max(1, self.V)
            full_mass = np.where(runtime_state["full"].any(axis=1), 1.0, 0.0).astype(np.float32)
            max_mass = np.where(runtime_state["mode"] == MODE_REDUCE, reduce_mass, full_mass)
        else:
            max_mass = np.where(resident.sum(axis=1) > 0, 1.0, 0.0).astype(np.float32)

        return np.stack(
            [token_count_fraction, full_need_fraction, max_mass],
            axis=1,
        ).astype(np.float32)

    def _get_edge_features(self, edge_usage):
        f_cap_rem = (self.capacities - edge_usage) / np.maximum(self.capacities, 1e-6)
        f_cap_static = self.capacities / max(np.max(self.capacities), 1e-6)
        return np.stack([f_cap_rem, f_cap_static], axis=1).astype(np.float32)

    def _build_initial_candidates(self, collective_type, runtime_state):
        demands = get_runtime_demands(runtime_state)

        if collective_type in COPY_ONLY_COLLECTIVES:
            resident = runtime_state["state"]
            chunk_has_demand = demands.sum(axis=1) > 0
            src_has = resident[:, self.edge_src]
            dst_lacks = ~resident[:, self.edge_dst]
            valid_matrix = src_has & dst_lacks & chunk_has_demand[:, np.newaxis]
            return np.where(valid_matrix)

        if collective_type != "allreduce":
            raise ValueError(f"Unsupported collective_type={collective_type!r}")

        partial_holders = runtime_state["partial"].any(axis=2)
        full_holders = runtime_state["full"]
        mode = runtime_state["mode"]

        reduce_valid = (mode[:, np.newaxis] == MODE_REDUCE) & partial_holders[:, self.edge_src]
        full_valid = (
            (mode[:, np.newaxis] == MODE_FULL)
            & full_holders[:, self.edge_src]
            & ~full_holders[:, self.edge_dst]
        )
        valid_matrix = reduce_valid | full_valid
        return np.where(valid_matrix)

    def _compute_group_mask(self, edge_indices, group_usage):
        mask_group = np.ones(len(edge_indices), dtype=bool)
        if self.num_groups == 0:
            return mask_group

        full_groups = group_usage >= self.group_limits
        if np.any(full_groups):
            blocked_edges = np.zeros(self.E, dtype=bool)
            for g in np.where(full_groups)[0]:
                blocked_edges[self.group_to_edges[g]] = True
            mask_group = ~blocked_edges[edge_indices]
        return mask_group

    def _compute_revisit_mask(
        self,
        runtime_state,
        all_c_idxs,
        all_e_idxs,
        current_mask,
        reduce_source_used,
        reduce_dst_incoming,
    ):
        visited = runtime_state.get("visited")
        if visited is None:
            return current_mask

        reduce_mask = current_mask & (runtime_state["mode"][all_c_idxs] == MODE_REDUCE)
        reduce_indices = np.where(reduce_mask)[0]
        if reduce_indices.size == 0:
            return current_mask

        partial_holders = runtime_state["partial"].any(axis=2)
        filtered_mask = current_mask.copy()

        for chunk_id in np.unique(all_c_idxs[reduce_indices]):
            chunk_indices = reduce_indices[all_c_idxs[reduce_indices] == chunk_id]
            chunk_edges = all_e_idxs[chunk_indices]
            chunk_src = self.edge_src[chunk_edges]
            chunk_dst = self.edge_dst[chunk_edges]

            for local_idx, global_idx in enumerate(chunk_indices):
                src = int(chunk_src[local_idx])
                dst = int(chunk_dst[local_idx])
                if not visited[chunk_id, src, dst]:
                    continue

                resident_partner = partial_holders[chunk_id, dst] and not reduce_source_used[chunk_id, dst]
                incoming_partner = reduce_dst_incoming[chunk_id, dst] > 0
                peer_partner = np.any((chunk_dst == dst) & (chunk_src != src))
                if not (resident_partner or incoming_partner or peer_partner):
                    filtered_mask[global_idx] = False

        return filtered_mask

    def _compute_current_mask(
        self,
        collective_type,
        runtime_state,
        all_c_idxs,
        all_e_idxs,
        edge_usage,
        group_usage,
        reduce_source_used,
        reduce_dst_incoming,
        full_dst_received,
        active_mask,
    ):
        mask_edge = edge_usage[all_e_idxs] < self.capacities[all_e_idxs]
        mask_group = self._compute_group_mask(all_e_idxs, group_usage)

        if collective_type in COPY_ONLY_COLLECTIVES:
            dst = self.edge_dst[all_e_idxs]
            mask_mode = ~full_dst_received[all_c_idxs, dst]
        else:
            src = self.edge_src[all_e_idxs]
            dst = self.edge_dst[all_e_idxs]
            chunk_mode = runtime_state["mode"][all_c_idxs]
            reduce_mask = chunk_mode == MODE_REDUCE
            mask_mode = np.ones(len(all_c_idxs), dtype=bool)
            if np.any(reduce_mask):
                reduce_idx = np.where(reduce_mask)[0]
                mask_mode[reduce_idx] = ~reduce_source_used[
                    all_c_idxs[reduce_idx], src[reduce_idx]
                ]
            if np.any(~reduce_mask):
                full_idx = np.where(~reduce_mask)[0]
                mask_mode[full_idx] = ~full_dst_received[
                    all_c_idxs[full_idx], dst[full_idx]
                ]
        current_mask = active_mask & mask_edge & mask_group & mask_mode
        if collective_type == "allreduce" and not getattr(self, '_soft_revisit', False):
            current_mask = self._compute_revisit_mask(
                runtime_state,
                all_c_idxs,
                all_e_idxs,
                current_mask,
                reduce_source_used,
                reduce_dst_incoming,
            )
        if self.timing:
            from .timed_state import timed_candidate_mask
            current_mask &= timed_candidate_mask(
                runtime_state, all_c_idxs, all_e_idxs, self.topo, self._timed_loads,
                reduce_source_used, full_dst_received, self._remaining_slots)
        if getattr(self, '_target_reservations', False):
            from .candidate_control import outstanding_targets
            outstanding = outstanding_targets(runtime_state, full_dst_received)
            active_chunks = outstanding.any(axis=1)
            if collective_type == 'allreduce':
                # Partial reservations cannot be mistaken for full results.
                active_chunks |= runtime_state['mode'] == MODE_REDUCE
            current_mask &= active_chunks[all_c_idxs]
        return current_mask

    def _get_candidate_dynamic_features(
        self,
        collective_type,
        runtime_state,
        cand_c,
        cand_e,
        dist_to_demand,
        edge_usage,
        group_usage,
        step,
        max_steps,
    ):
        demands = get_runtime_demands(runtime_state)
        cand_src = self.edge_src[cand_e]
        cand_dst = self.edge_dst[cand_e]

        f_dst_need = demands[cand_c, cand_dst].astype(np.float32)
        d_src = dist_to_demand[cand_c, cand_src]
        d_dst = dist_to_demand[cand_c, cand_dst]
        f_dist_red = (d_src - d_dst) / max(1.0, self.V)

        edge_rem = self.capacities[cand_e] - edge_usage[cand_e]
        f_edge_rem = edge_rem / np.maximum(self.capacities[cand_e], 1e-6)

        f_group_rem = np.ones(len(cand_e), dtype=np.float32)
        if self.num_groups > 0:
            for i, e in enumerate(cand_e):
                groups = self.edge_to_groups[e]
                if groups:
                    min_rem = 1.0
                    for g in groups:
                        rem = (self.group_limits[g] - group_usage[g]) / max(self.group_limits[g], 1e-6)
                        min_rem = min(min_rem, rem)
                    f_group_rem[i] = min_rem

        f_step_progress = np.full(len(cand_e), step / max(max_steps, 1), dtype=np.float32)
        f_dst_has_partner = np.zeros(len(cand_e), dtype=np.float32)
        f_src_mass = np.ones(len(cand_e), dtype=np.float32)
        f_dst_mass = np.zeros(len(cand_e), dtype=np.float32)
        f_merged_mass = np.ones(len(cand_e), dtype=np.float32)
        f_would_complete = np.ones(len(cand_e), dtype=np.float32)
        f_mode_is_full = np.ones(len(cand_e), dtype=np.float32)

        if collective_type == "allreduce":
            chunk_mode = runtime_state["mode"][cand_c]
            reduce_idx = np.where(chunk_mode == MODE_REDUCE)[0]
            if reduce_idx.size > 0:
                src_sets = runtime_state["partial"][cand_c[reduce_idx], cand_src[reduce_idx]]
                dst_sets = runtime_state["partial"][cand_c[reduce_idx], cand_dst[reduce_idx]]
                src_mass = src_sets.sum(axis=1) / max(1, self.V)
                dst_mass = dst_sets.sum(axis=1) / max(1, self.V)
                merged_mass = np.logical_or(src_sets, dst_sets).sum(axis=1) / max(1, self.V)

                f_mode_is_full[reduce_idx] = 0.0
                f_dst_has_partner[reduce_idx] = (dst_mass > 0).astype(np.float32)
                f_src_mass[reduce_idx] = src_mass.astype(np.float32)
                f_dst_mass[reduce_idx] = dst_mass.astype(np.float32)
                f_merged_mass[reduce_idx] = merged_mass.astype(np.float32)
                f_would_complete[reduce_idx] = np.isclose(merged_mass, 1.0).astype(np.float32)

        else:
            resident = runtime_state["state"]
            f_src_mass = resident[cand_c, cand_src].astype(np.float32)
            f_dst_mass = resident[cand_c, cand_dst].astype(np.float32)
            f_merged_mass = np.maximum(f_src_mass, f_dst_mass)

        return np.stack(
            [
                f_dst_need,
                f_dst_has_partner,
                f_src_mass,
                f_dst_mass,
                f_merged_mass,
                f_would_complete,
                f_edge_rem,
                f_group_rem,
                f_step_progress,
                f_mode_is_full,
                f_dist_red,
            ],
            axis=1,
        ).astype(np.float32)

    def _prune_candidates(
        self,
        collective_type,
        runtime_state,
        active_indices,
        all_c_idxs,
        all_e_idxs,
        dist_to_demand,
        edge_usage,
        group_usage,
        step,
        train,
    ):
        max_cands = 256 if train else 512
        if len(active_indices) <= max_cands:
            cand_c = all_c_idxs[active_indices]
            cand_e = all_e_idxs[active_indices]
            cand_dyn_feats = self._get_candidate_dynamic_features(
                collective_type,
                runtime_state,
                cand_c,
                cand_e,
                dist_to_demand,
                edge_usage,
                group_usage,
                step,
                self.max_steps,
            )
            return active_indices, cand_c, cand_e, cand_dyn_feats

        cand_c_full = all_c_idxs[active_indices]
        cand_e_full = all_e_idxs[active_indices]
        if collective_type in COPY_ONLY_COLLECTIVES:
            # Rank with only the features used by the heuristic. Computing all
            # dynamic features here repeats shared-group traversal for every
            # active (chunk, edge) pair, before the candidate bound is applied.
            cand_src = self.edge_src[cand_e_full]
            cand_dst = self.edge_dst[cand_e_full]
            demands = get_runtime_demands(runtime_state)
            resident = runtime_state["state"]
            dst_need = demands[cand_c_full, cand_dst].astype(np.float32)
            merged_mass = np.maximum(
                resident[cand_c_full, cand_src],
                resident[cand_c_full, cand_dst],
            ).astype(np.float32)
            dist_red = (
                dist_to_demand[cand_c_full, cand_src]
                - dist_to_demand[cand_c_full, cand_dst]
            ) / max(1.0, self.V)
            # Keep the existing score, float32 arithmetic and argsort order.
            heuristic_scores = 8.0 + 4.0 * dst_need + 2.0 * merged_mass + dist_red
            if getattr(self, '_candidate_coverage', False):
                from .candidate_control import balanced_topk
                top_k_local = balanced_topk(cand_c_full, heuristic_scores, max_cands, self._slot_index + step)
            else:
                top_k_local = np.argsort(heuristic_scores)[-max_cands:]
            pruned_indices = active_indices[top_k_local]
            cand_c = cand_c_full[top_k_local]
            cand_e = cand_e_full[top_k_local]
            cand_dyn_feats = self._get_candidate_dynamic_features(
                collective_type,
                runtime_state,
                cand_c,
                cand_e,
                dist_to_demand,
                edge_usage,
                group_usage,
                step,
                self.max_steps,
            )
            return pruned_indices, cand_c, cand_e, cand_dyn_feats

        cand_dyn_feats_full = self._get_candidate_dynamic_features(
            collective_type,
            runtime_state,
            cand_c_full,
            cand_e_full,
            dist_to_demand,
            edge_usage,
            group_usage,
            step,
            self.max_steps,
        )
        heuristic_scores = (
            8.0 * cand_dyn_feats_full[:, 5]
            + 4.0 * cand_dyn_feats_full[:, 0]
            + 2.0 * cand_dyn_feats_full[:, 4]
            + cand_dyn_feats_full[:, 10]
        )
        if getattr(self, '_candidate_coverage', False):
            from .candidate_control import balanced_topk
            top_k_local = balanced_topk(cand_c_full, heuristic_scores, max_cands, self._slot_index + step)
        else:
            top_k_local = np.argsort(heuristic_scores)[-max_cands:]
        pruned_indices = active_indices[top_k_local]
        return (
            pruned_indices,
            cand_c_full[top_k_local],
            cand_e_full[top_k_local],
            cand_dyn_feats_full[top_k_local],
        )

    def decode_slot(self, model, problem, runtime_state, t, T, train=True,
                    selector='learned', candidate_observer=None, policy_options=None):
        """Decode a single slot from slot-start runtime state."""
        if selector not in {'learned', 'greedy'} or (selector == 'greedy' and train):
            raise ValueError('Greedy selection is inference-only; unknown selector rejected')
        collective_type = normalize_collective_type(problem.collective_type)
        plan_observation = bool(getattr(model, 'current_slot_plan_observation', False)) if selector == 'learned' else False
        if plan_observation and (not self.timing or not getattr(model, 'relational_features', False) or not getattr(model, 'use_stop', False)):
            raise ValueError('Slot plan observation requires timed relational STOP policy')
        device = next(model.parameters()).device if selector == 'learned' else torch.device('cpu')
        C = problem.C
        E = self.E
        options = {key: bool(getattr(model, key, False)) for key in
                   ('use_stop', 'target_reservations', 'candidate_coverage', 'relational_features')}
        if getattr(model, 'shared_redesign', False) or (policy_options and 'shared_redesign' in policy_options):
            options['shared_redesign'] = bool(getattr(model, 'shared_redesign', False))
        if getattr(model, 'arrival_aware', False) or (policy_options and 'arrival_aware' in policy_options):
            options['arrival_aware'] = bool(getattr(model, 'arrival_aware', False))
        if policy_options is not None:
            from rlccl.config import get_model_kwargs
            get_model_kwargs(timed=bool(self.timing), policy_options=policy_options)
            if selector == 'learned' and options != policy_options:
                raise ValueError('Decode flags differ from the learned policy schema')
            options = policy_options
        self._soft_revisit = bool(options.get('arrival_aware', False))
        self._target_reservations = options['target_reservations']
        self._candidate_coverage = options['candidate_coverage']
        self._slot_index = t
        use_stop = options['use_stop']

        if self.edge_src_t.device != device:
            self.edge_src_t = self.edge_src_t.to(device)
            self.edge_dst_t = self.edge_dst_t.to(device)

        if self.timing:
            self._timed_loads = runtime_state["resource_loads"].copy()
            self._remaining_slots = T - t
        arrival_context = None
        if options.get('arrival_aware', False):
            from .arrival_features import service_distances, slot_arrival_context
            if not hasattr(self, '_service_distances'):
                self._service_distances = service_distances(self.topo)
            arrival_context = slot_arrival_context(runtime_state, self.topo, self._service_distances)
        demands = get_runtime_demands(runtime_state)
        dist_to_demand = self.compute_dist_to_demand(demands)
        node_feats_np = self._get_node_features(runtime_state, t, T)
        chunk_feats_np = self._get_chunk_features(collective_type, runtime_state)
        edge_feats_np = self._get_edge_features(np.zeros(E, dtype=np.float32))

        if self.timing:
            pending_count = np.zeros(self.V, dtype=np.float32)
            pending_wait = np.zeros(self.V, dtype=np.float32)
            chunk_pending = np.zeros(C, dtype=np.float32)
            for event in runtime_state["pending"]:
                dst, c = event["dst"], event["chunk"]
                pending_count[dst] += 1
                pending_wait[dst] = max(pending_wait[dst], (event["arrival"] - t) / self.timing.refinement)
                chunk_pending[c] += 1 / max(1, self.V)
            node_feats_np = np.column_stack((node_feats_np, pending_count / max(1, C), pending_wait))
            chunk_feats_np = np.column_stack((chunk_feats_np, chunk_pending))
            edge_feats_np[:, 0] = 1 - self._timed_loads[0, :E]
            busy = self._timed_loads[:, :E].sum(axis=0) / self.timing.refinement
            edge_feats_np = np.column_stack((edge_feats_np, self.timing.copy_cost / self.timing.reference_time,
                                            self.timing.aggregate_cost / self.timing.reference_time, busy))
        node_feats = torch.tensor(node_feats_np, dtype=torch.float32, device=device)
        chunk_feats = torch.tensor(chunk_feats_np, dtype=torch.float32, device=device)
        edge_feats = torch.tensor(edge_feats_np, dtype=torch.float32, device=device)

        with torch.no_grad():
            if selector == 'learned':
                h_v, h_e, h_c, g_ctx = model.encode_state(
                    node_feats, edge_feats, self.edge_src_t, self.edge_dst_t, chunk_feats)
                value = model.get_value(g_ctx)
            else:
                value = torch.tensor(0., device=device)

        state_info = {
            "collective_type": collective_type,
            "runtime_state": clone_runtime_state(runtime_state),
            "node_feats": node_feats.cpu(),
            "edge_feats": edge_feats.cpu(),
            "chunk_feats": chunk_feats.cpu(),
            "dist_to_demand": dist_to_demand.copy(),
            "t": t,
            "T": T,
            "current_slot_plan_observation": plan_observation,
            "policy_schema": ('stop_v1' if use_stop else 'legacy') if selector == 'learned' else 'greedy',
        }

        all_c_idxs, all_e_idxs = self._build_initial_candidates(collective_type, runtime_state)
        if len(all_c_idxs) == 0:
            Y_t = np.zeros((C, E), dtype=np.int32)
            zero = torch.tensor(0.0, device=device)
            return Y_t, zero, zero, value, state_info, []

        Y_t = np.zeros((C, E), dtype=np.int32)
        edge_usage = np.zeros(E, dtype=np.float32)
        group_usage = np.zeros(self.num_groups, dtype=np.float32) if self.num_groups > 0 else np.array([], dtype=np.float32)
        reduce_source_used = np.zeros((C, self.V), dtype=bool)
        reduce_dst_incoming = np.zeros((C, self.V), dtype=np.int32)
        full_dst_received = np.zeros((C, self.V), dtype=bool)
        active_mask = np.ones(len(all_c_idxs), dtype=bool)
        selected_moves = np.full((C, self.V), -1, dtype=np.int64)

        micro_actions = []
        plan_events = []
        logp_list = []
        entropy_list = []

        for step in range(self.max_steps):
            current_mask = self._compute_current_mask(
                collective_type,
                runtime_state,
                all_c_idxs,
                all_e_idxs,
                edge_usage,
                group_usage,
                reduce_source_used,
                reduce_dst_incoming,
                full_dst_received,
                active_mask,
            )
            if not np.any(current_mask):
                break

            active_indices = np.where(current_mask)[0]
            active_indices, cand_c, cand_e, cand_dyn_feats_np = self._prune_candidates(
                collective_type,
                runtime_state,
                active_indices,
                all_c_idxs,
                all_e_idxs,
                dist_to_demand,
                edge_usage,
                group_usage,
                step,
                train,
            )

            if self.timing:
                from .timed_state import aggregate_flags
                agg = aggregate_flags(runtime_state, cand_c, cand_e, self.topo)
                cost = np.where(agg, self.timing.aggregate_cost[cand_e], self.timing.copy_cost[cand_e])
                delay = np.where(agg, self.timing.aggregate_delay[cand_e], self.timing.copy_delay[cand_e])
                cand_dyn_feats_np = np.column_stack((
                    cand_dyn_feats_np, cost / self.timing.reference_time,
                    delay / self.timing.refinement,
                    np.full(len(cand_e), 1 / self.timing.refinement),
                    1 - self._timed_loads[0, cand_e])).astype(np.float32)
                if arrival_context is not None:
                    from .arrival_features import candidate_arrival_features
                    extra = candidate_arrival_features(arrival_context, cand_c, cand_e, self.topo, T - t, delay)
                    cand_dyn_feats_np = np.column_stack((cand_dyn_feats_np, extra)).astype(np.float32)
                    from .meeting_features import meeting_advance
                    cand_dyn_feats_np[:, 19] = meeting_advance(
                        runtime_state, cand_c, cand_e, self.topo,
                        self._service_distances, selected_moves)
                    if collective_type == 'allreduce':
                        partial = runtime_state['mode'][cand_c] == MODE_REDUCE
                        visits = runtime_state['visited'][cand_c, self.edge_src[cand_e], self.edge_dst[cand_e]]
                        cand_dyn_feats_np[:, 19] -= (partial & visits).astype(np.float32)
            if candidate_observer is not None:
                candidate_observer(t, step, cand_c.copy(), cand_e.copy(), cand_dyn_feats_np.copy())
            with torch.no_grad():
                cand_dyn_feats = torch.tensor(cand_dyn_feats_np, dtype=torch.float32, device=device)
                cand_e_t = torch.tensor(cand_e, dtype=torch.long, device=device)
                cand_c_t = torch.tensor(cand_c, dtype=torch.long, device=device)
                relation_kwargs = {}
                if selector == 'learned' and options['relational_features']:
                    from .candidate_control import candidate_relations
                    relations = candidate_relations(runtime_state, cand_c, cand_e, self.topo,
                                                    problem.initial_state, full_dst_received)
                    if options.get('shared_redesign', False):
                        from .shared_features import shared_relation_features
                        relations.update(shared_relation_features(runtime_state, cand_c, cand_e, self.topo, full_dst_received))
                    if plan_observation:
                        from .plan_observation import pending_snapshot
                        relations['pending'] = pending_snapshot(relations['pending'], plan_events)
                    relation_kwargs['relations'] = relations
                if selector == 'greedy':
                    logits = torch.as_tensor(greedy_candidate_scores(cand_dyn_feats_np), device=device)
                else:
                    logits = model.get_candidate_logits(
                        h_v, h_e, h_c, g_ctx, self.edge_src_t, self.edge_dst_t,
                        cand_e_t, cand_c_t, cand_dyn_feats, **relation_kwargs)
                if use_stop:
                    from .candidate_control import outstanding_targets
                    outstanding = outstanding_targets(runtime_state, full_dst_received)
                    pending = runtime_state.get('pending', [])
                    wait = max((event['arrival'] - t for event in pending), default=0)
                    free = (1 - self._timed_loads[0]).mean() if self.timing else 1.0
                    stop_feats_np = np.array([
                        step / max(1, self.max_steps),
                        full_dst_received.sum() / max(1, C * self.V),
                        demands.sum() / max(1, C * self.V),
                        outstanding.sum() / max(1, C * self.V),
                        free, len(cand_e) / 512.,
                        wait / self.timing.refinement if self.timing else 0.,
                        (T - t) / max(1, T),
                    ], dtype=np.float32)
                    stop_feats = torch.as_tensor(stop_feats_np, device=device)
                    # Fixed strong greedy retains its original positive
                    # demand/completion/mass score and uses STOP utility 0.
                    stop_logit = (torch.tensor(0., device=device) if selector == 'greedy'
                                  else model.get_stop_logit(g_ctx, stop_feats, len(cand_e), cand_dyn_feats))
                    logits = torch.cat((logits, stop_logit.reshape(1)))
                dist = (Categorical(logits=logits) if use_stop
                        else Categorical(F.softmax(logits, dim=0)))
                action_idx = (dist.sample() if train else
                              deterministic_action(logits, use_stop, selector))
                log_prob = dist.log_prob(action_idx)
                entropy = dist.entropy()
                if selector == 'greedy':
                    log_prob = entropy = torch.tensor(0., device=device)

            logp_list.append(log_prob)
            entropy_list.append(entropy)

            idx = action_idx.item()
            if use_stop:
                stop = idx == len(cand_e)
                micro_actions.append({
                    'action_idx': idx, 'cand_c': cand_c.copy(), 'cand_e': cand_e.copy(),
                    'cand_dyn_feats': cand_dyn_feats_np.copy(), 'stop_feats': stop_feats_np.copy(),
                    'selected_c': None if stop else int(cand_c[idx]),
                    'selected_e': None if stop else int(cand_e[idx]), 'step': step,
                    'has_stop': True, 'stop': stop, 'old_logprob': float(log_prob),
                })
                if relation_kwargs:
                    micro_actions[-1]['relations'] = relations
                    if plan_observation:
                        micro_actions[-1]['plan_observation_schema'] = 'slot_prefix_future_v1'
                if stop:
                    break  # finish this slot; pending operations continue normally
            best_global_idx = active_indices[idx]
            c = int(all_c_idxs[best_global_idx])
            e = int(all_e_idxs[best_global_idx])
            src = int(self.edge_src[e])
            dst = int(self.edge_dst[e])

            if not use_stop:
                micro_actions.append({
                    "action_idx": idx,
                    "cand_c": cand_c.copy(),
                    "cand_e": cand_e.copy(),
                    "selected_c": c,
                    "selected_e": e,
                    "step": step,
                })

            if self.timing:
                micro_actions[-1]["cand_dyn_feats"] = cand_dyn_feats_np.copy()
                agg_selected = bool(aggregate_flags(runtime_state, np.array([c]), np.array([e]), self.topo)[0])
                self.timing.reserve(self._timed_loads, e, agg_selected)
                full_dst_received[c, dst] = True
                if agg_selected:
                    reduce_source_used[c, dst] = True
            if plan_observation:
                from .plan_observation import planned_arrival
                plan_events.append(planned_arrival(problem, runtime_state, c, e, self.topo))
            Y_t[c, e] = 1
            active_mask[best_global_idx] = False
            edge_usage[e] += 1
            for g in self.edge_to_groups[e]:
                group_usage[g] += 1

            if collective_type == "allreduce" and runtime_state["mode"][c] == MODE_REDUCE:
                reduce_source_used[c, src] = True
                reduce_dst_incoming[c, dst] += 1
                selected_moves[c, src] = dst
            else:
                full_dst_received[c, dst] = True

        if logp_list:
            logp_slot = torch.stack(logp_list).sum()
            entropy_slot = torch.stack(entropy_list).mean()
        else:
            logp_slot = torch.tensor(0.0, device=device)
            entropy_slot = torch.tensor(0.0, device=device)

        return Y_t, logp_slot, entropy_slot, value, state_info, micro_actions


def recompute_logp_slot(model, state_info, micro_actions, device, static_info, return_decisions=False,
                        entropy_mode='categorical', stop_entropy_weight=1.0):
    """Recompute slot log-probability and entropy from stored micro-actions."""
    if state_info.get('policy_schema') == 'greedy':
        raise ValueError('Greedy output is not a sampled policy rollout')
    if entropy_mode not in ('categorical','factorized_stop'):
        raise ValueError('Unknown policy entropy regularizer')
    if entropy_mode=='factorized_stop' and state_info.get('policy_schema')!='stop_v1':
        raise ValueError('Factorized STOP entropy requires a stored STOP policy rollout')
    node_feats = state_info["node_feats"].to(device)
    edge_feats = state_info["edge_feats"].to(device)
    chunk_feats = state_info["chunk_feats"].to(device)
    dist_to_demand = state_info["dist_to_demand"]
    collective_type = state_info["collective_type"]
    runtime_state = clone_runtime_state(state_info["runtime_state"])

    edge_src_t = static_info["edge_src_t"].to(device)
    edge_dst_t = static_info["edge_dst_t"].to(device)
    capacities = static_info["capacities"]
    group_limits = static_info["group_limits"]
    edge_to_groups = static_info["edge_to_groups"]
    num_groups = static_info["num_groups"]
    V = static_info["V"]
    E = static_info["E"]
    max_steps = static_info["max_steps"]

    h_v, h_e, h_c, g_ctx = model.encode_state(
        node_feats,
        edge_feats,
        edge_src_t,
        edge_dst_t,
        chunk_feats,
    )
    value_new = model.get_value(g_ctx)
    observed_plan = bool(state_info.get('current_slot_plan_observation', False))
    if observed_plan != bool(getattr(model, 'current_slot_plan_observation', False)):
        raise ValueError('Sampling and recompute plan observation flags differ')

    if len(micro_actions) == 0:
        zero = torch.tensor(0.0, device=device)
        if return_decisions:
            empty = value_new.new_empty(0)
            return empty, empty, value_new
        return zero, zero, value_new

    if state_info.get('policy_schema') == 'stop_v1':
        if not getattr(model, 'use_stop', False):
            raise ValueError('STOP rollout cannot be recomputed by a legacy policy')
        if observed_plan:
            from .plan_observation import recompute_prefix_stop
            logps, entropies = recompute_prefix_stop(
                model,h_v,h_e,h_c,g_ctx,edge_src_t,edge_dst_t,micro_actions,device,
                entropy_mode,stop_entropy_weight)
            if return_decisions:return logps,entropies,value_new
            return logps.sum(),entropies.mean(),value_new
        # Reuse the EXACT candidate pool and dynamic features from rollout.
        # Joint-slot log probability includes every action and sampled STOP.
        all_e = torch.as_tensor(np.concatenate([a['cand_e'] for a in micro_actions]),
                                dtype=torch.long, device=device)
        all_c = torch.as_tensor(np.concatenate([a['cand_c'] for a in micro_actions]),
                                dtype=torch.long, device=device)
        all_dyn = torch.as_tensor(np.concatenate([a['cand_dyn_feats'] for a in micro_actions]),
                                  dtype=torch.float32, device=device)
        relation_kwargs = {}
        if getattr(model, 'relational_features', False):
            if any('relations' not in action for action in micro_actions):
                raise ValueError('Relational features missing from rollout')
            first = micro_actions[0]['relations']
            relation_kwargs['relations'] = dict(
                **{key: np.concatenate([a['relations'][key] for a in micro_actions])
                   for key in ('source', 'missing', 'targets', 'route', 'task') if key in first},
                pending=first['pending'], chunks_count=first['chunks_count'])
        flat_logits = model.get_candidate_logits(h_v, h_e, h_c, g_ctx, edge_src_t, edge_dst_t,
                                                all_e, all_c, all_dyn, **relation_kwargs)
        offset, logps, entropies = 0, [], []
        for i, action in enumerate(micro_actions):
            size = len(action['cand_e'])
            if not action.get('has_stop') or size < 1:
                raise ValueError('Malformed STOP rollout candidate pool')
            if action.get('stop') and i != len(micro_actions) - 1:
                raise ValueError('Actions follow a sampled STOP')
            chosen = int(action['action_idx'])
            if not 0 <= chosen <= size or bool(action.get('stop')) != (chosen == size):
                raise ValueError('STOP action index disagrees with rollout')
            if chosen < size and (int(action['cand_c'][chosen]) != action['selected_c']
                                  or int(action['cand_e'][chosen]) != action['selected_e']):
                raise ValueError('Selected candidate disagrees with rollout pool')
            stop_feats = torch.as_tensor(action['stop_feats'], dtype=torch.float32, device=device)
            stop_logit = model.get_stop_logit(g_ctx, stop_feats, size, all_dyn[offset:offset + size])
            logits = torch.cat((flat_logits[offset:offset + size], stop_logit.reshape(1)))
            offset += size
            dist = Categorical(logits=logits)
            chosen_t = torch.as_tensor(chosen, dtype=torch.long, device=device)
            logps.append(dist.log_prob(chosen_t))
            entropies.append(factorized_stop_entropy(logits,stop_entropy_weight)
                             if entropy_mode=='factorized_stop' else dist.entropy())
        if return_decisions:
            return torch.stack(logps), torch.stack(entropies), value_new
        return torch.stack(logps).sum(), torch.stack(entropies).mean(), value_new

    edge_src_np = edge_src_t.cpu().numpy()
    edge_dst_np = edge_dst_t.cpu().numpy()
    demands = get_runtime_demands(runtime_state)

    edge_usage = np.zeros(E, dtype=np.float32)
    group_usage = np.zeros(num_groups, dtype=np.float32) if num_groups > 0 else np.array([], dtype=np.float32)
    reduce_source_used = np.zeros((demands.shape[0], V), dtype=bool)
    full_dst_received = np.zeros((demands.shape[0], V), dtype=bool)

    all_cand_e = []
    all_cand_c = []
    all_dyn_feats = []
    segment_sizes = []
    actions_taken = []

    for ma in micro_actions:
        cand_c = ma["cand_c"]
        cand_e = ma["cand_e"]
        step = ma["step"]
        selected_c = ma["selected_c"]
        selected_e = ma["selected_e"]

        cand_src = edge_src_np[cand_e]
        cand_dst = edge_dst_np[cand_e]

        f_dst_need = demands[cand_c, cand_dst].astype(np.float32)
        d_src = dist_to_demand[cand_c, cand_src]
        d_dst = dist_to_demand[cand_c, cand_dst]
        f_dist_red = (d_src - d_dst) / max(1.0, V)

        edge_rem = capacities[cand_e] - edge_usage[cand_e]
        f_edge_rem = edge_rem / np.maximum(capacities[cand_e], 1e-6)

        f_group_rem = np.ones(len(cand_e), dtype=np.float32)
        if num_groups > 0:
            for i, e in enumerate(cand_e):
                groups = edge_to_groups[e]
                if groups:
                    min_rem = 1.0
                    for g in groups:
                        rem = (group_limits[g] - group_usage[g]) / max(group_limits[g], 1e-6)
                        min_rem = min(min_rem, rem)
                    f_group_rem[i] = min_rem

        f_step_progress = np.full(len(cand_e), step / max(max_steps, 1), dtype=np.float32)
        f_dst_has_partner = np.zeros(len(cand_e), dtype=np.float32)
        f_src_mass = np.ones(len(cand_e), dtype=np.float32)
        f_dst_mass = np.zeros(len(cand_e), dtype=np.float32)
        f_merged_mass = np.ones(len(cand_e), dtype=np.float32)
        f_would_complete = np.ones(len(cand_e), dtype=np.float32)
        f_mode_is_full = np.ones(len(cand_e), dtype=np.float32)

        if collective_type == "allreduce":
            chunk_mode = runtime_state["mode"][cand_c]
            reduce_idx = np.where(chunk_mode == MODE_REDUCE)[0]
            if reduce_idx.size > 0:
                src_sets = runtime_state["partial"][cand_c[reduce_idx], cand_src[reduce_idx]]
                dst_sets = runtime_state["partial"][cand_c[reduce_idx], cand_dst[reduce_idx]]
                src_mass = src_sets.sum(axis=1) / max(1, V)
                dst_mass = dst_sets.sum(axis=1) / max(1, V)
                merged_mass = np.logical_or(src_sets, dst_sets).sum(axis=1) / max(1, V)

                f_mode_is_full[reduce_idx] = 0.0
                f_dst_has_partner[reduce_idx] = (dst_mass > 0).astype(np.float32)
                f_src_mass[reduce_idx] = src_mass.astype(np.float32)
                f_dst_mass[reduce_idx] = dst_mass.astype(np.float32)
                f_merged_mass[reduce_idx] = merged_mass.astype(np.float32)
                f_would_complete[reduce_idx] = np.isclose(merged_mass, 1.0).astype(np.float32)

        else:
            resident = runtime_state["state"]
            f_src_mass = resident[cand_c, cand_src].astype(np.float32)
            f_dst_mass = resident[cand_c, cand_dst].astype(np.float32)
            f_merged_mass = np.maximum(f_src_mass, f_dst_mass)

        cand_dyn_feats_np = np.stack(
            [
                f_dst_need,
                f_dst_has_partner,
                f_src_mass,
                f_dst_mass,
                f_merged_mass,
                f_would_complete,
                f_edge_rem,
                f_group_rem,
                f_step_progress,
                f_mode_is_full,
                f_dist_red,
            ],
            axis=1,
        ).astype(np.float32)

        if "cand_dyn_feats" in ma:
            cand_dyn_feats_np = ma["cand_dyn_feats"]
        all_cand_e.append(cand_e)
        all_cand_c.append(cand_c)
        all_dyn_feats.append(cand_dyn_feats_np)
        segment_sizes.append(len(cand_e))
        actions_taken.append(ma["action_idx"])

        edge_usage[selected_e] += 1
        for g in edge_to_groups[selected_e]:
            group_usage[g] += 1

        src = int(edge_src_np[selected_e])
        dst = int(edge_dst_np[selected_e])
        if collective_type == "allreduce" and runtime_state["mode"][selected_c] == MODE_REDUCE:
            reduce_source_used[selected_c, src] = True
        else:
            full_dst_received[selected_c, dst] = True

    flat_cand_e = torch.tensor(np.concatenate(all_cand_e), dtype=torch.long, device=device)
    flat_cand_c = torch.tensor(np.concatenate(all_cand_c), dtype=torch.long, device=device)
    flat_dyn_feats = torch.tensor(np.concatenate(all_dyn_feats), dtype=torch.float32, device=device)

    all_logits = model.get_candidate_logits(
        h_v,
        h_e,
        h_c,
        g_ctx,
        edge_src_t,
        edge_dst_t,
        flat_cand_e,
        flat_cand_c,
        flat_dyn_feats,
    )

    logp_list = []
    entropy_list = []
    current_offset = 0
    for i, size in enumerate(segment_sizes):
        segment_logits = all_logits[current_offset: current_offset + size]
        current_offset += size
        probs = F.softmax(segment_logits, dim=0)
        dist = Categorical(probs)
        action_t = torch.tensor(actions_taken[i], dtype=torch.long, device=device)
        logp_list.append(dist.log_prob(action_t))
        entropy_list.append(dist.entropy())

    logp_slot_new = torch.stack(logp_list).sum()
    entropy_slot_new = torch.stack(entropy_list).mean()
    return logp_slot_new, entropy_slot_new, value_new


def greedy_candidate_scores(features):
    """Frozen strong completion/demand/mass/hop score; no fitted lookahead."""
    scores = 8. * features[:, 5] + 4. * features[:, 0] + 2. * features[:, 4] + features[:, 10]
    if features.shape[1] == 20:
        scores = scores - 8. * features[:, 15]
    return scores


def solve_with_model(problem, model, topology_info, selector='learned',
                     candidate_observer=None, policy_options=None):
    """Run greedy decoding over a full horizon."""
    if model is not None:
        model.eval()
    topo = getattr(problem, "topology_info", topology_info)
    decoder = SlotDecoder(topo)
    runtime_state = init_runtime_state(problem)
    schedule = []

    with torch.no_grad():
        for t in range(problem.T):
            Y_t, _, _, _, _, _ = decoder.decode_slot(
                model,
                problem,
                runtime_state,
                t,
                problem.T,
                train=False,
                selector=selector, candidate_observer=candidate_observer,
                policy_options=policy_options,
            )
            schedule.append(Y_t)
            runtime_state, _ = apply_slot_schedule(
                problem,
                runtime_state,
                Y_t,
                topology_info=topo,
                validate=False,
            )
            if is_completed(problem, runtime_state):
                break

    while len(schedule) < problem.T:
        schedule.append(np.zeros((problem.C, problem.E), dtype=np.int32))

    return schedule
