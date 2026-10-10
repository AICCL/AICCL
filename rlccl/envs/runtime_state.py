"""Shared runtime state and slot transitions for RLCCL collectives."""

from __future__ import annotations

from typing import Dict, Tuple

import copy

import numpy as np

from .problem import COPY_ONLY_COLLECTIVES, normalize_collective_type

MODE_REDUCE = 0
MODE_FULL = 1


def _clone_array_dict(runtime_state: Dict[str, object]) -> Dict[str, object]:
    cloned = {}
    for key, value in runtime_state.items():
        if isinstance(value, np.ndarray):
            cloned[key] = value.copy()
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def clone_runtime_state(runtime_state: Dict[str, object]) -> Dict[str, object]:
    """Deep-copy numpy-backed runtime state."""
    return _clone_array_dict(runtime_state)


def get_runtime_demands(runtime_state: Dict[str, object]) -> np.ndarray:
    return runtime_state["demands"]


def get_runtime_resident_matrix(runtime_state: Dict[str, object]) -> np.ndarray:
    collective_type = normalize_collective_type(runtime_state["collective_type"])
    if collective_type == "allreduce":
        partial_holders = runtime_state["partial"].any(axis=2)
        return partial_holders | runtime_state["full"]
    return runtime_state["state"]


def _init_resident_state(problem) -> Dict[str, object]:
    """Build slot-start runtime state from a problem instance."""
    collective_type = normalize_collective_type(problem.collective_type)
    base_demands = np.asarray(problem.demands, dtype=bool).copy()

    if collective_type in COPY_ONLY_COLLECTIVES:
        return {
            "collective_type": collective_type,
            "state": np.asarray(problem.initial_state, dtype=bool).copy(),
            "demands": base_demands,
        }

    if collective_type != "allreduce":
        raise ValueError(f"Unsupported collective_type={collective_type!r}")

    C = problem.C
    V = problem.V
    initial_state = np.asarray(problem.initial_state, dtype=bool)
    partial = np.zeros((C, V, V), dtype=bool)
    visited = np.zeros((C, V, V), dtype=bool)
    for c in range(C):
        owners = np.where(initial_state[c])[0]
        for owner in owners:
            partial[c, owner, owner] = True
            visited[c, owner, owner] = True

    mode = np.full(C, MODE_REDUCE, dtype=np.int8)
    full = np.zeros((C, V), dtype=bool)
    demands = base_demands.copy()

    for c in range(C):
        masses = partial[c].sum(axis=1)
        complete_nodes = np.where(masses == V)[0]
        if complete_nodes.size > 0:
            dst = int(complete_nodes[0])
            mode[c] = MODE_FULL
            partial[c] = False
            full[c, dst] = True
            demands[c, dst] = False

    return {
        "collective_type": collective_type,
        "mode": mode,
        "partial": partial,
        "full": full,
        "visited": visited,
        "demands": demands,
    }


def init_runtime_state(problem):
    state = _init_resident_state(problem)
    if getattr(problem.topology_info, "timing", None):
        from .timed_state import init_timed_state
        return init_timed_state(state, problem.topology_info)
    return state


def remaining_demand(problem, runtime_state: Dict[str, object]) -> int:
    """Count unsatisfied demands under the current runtime semantics."""
    return int(np.asarray(runtime_state["demands"], dtype=np.int32).sum())


def is_completed(problem, runtime_state: Dict[str, object]) -> bool:
    """Return whether all demands have been satisfied."""
    return remaining_demand(problem, runtime_state) == 0 and not runtime_state.get("pending")


def allreduce_merge_potential(runtime_state: Dict[str, object]) -> float:
    """Potential that grows with partial consolidation and is stable under relocation."""
    collective_type = normalize_collective_type(runtime_state["collective_type"])
    if collective_type != "allreduce":
        return 0.0

    partial = np.asarray(runtime_state["partial"], dtype=bool)
    mode = np.asarray(runtime_state["mode"], dtype=np.int8)
    V = partial.shape[1]
    if V == 0:
        return 0.0

    masses = partial.sum(axis=2).astype(np.float32) / float(V)
    reduce_scores = np.sum(masses * masses, axis=1)
    for event in runtime_state.get("pending", []):
        if event["kind"] == "partial":
            reduce_scores[event["chunk"]] += (event["payload"].sum() / float(V)) ** 2
    full_scores = np.ones_like(reduce_scores, dtype=np.float32)
    return float(np.where(mode == MODE_REDUCE, reduce_scores, full_scores).sum())


def _validate_schedule_common(problem, Y_t: np.ndarray, topology_info) -> np.ndarray:
    if Y_t.shape != (problem.C, problem.E):
        raise ValueError(
            f"Invalid schedule shape {tuple(Y_t.shape)}; expected ({problem.C}, {problem.E})"
        )

    if np.any((Y_t != 0) & (Y_t != 1)):
        raise ValueError("Schedule matrix must be binary")

    edge_loads = np.sum(Y_t, axis=0)
    if np.any(edge_loads > problem.capacities + 1e-6):
        idx = int(np.where(edge_loads > problem.capacities + 1e-6)[0][0])
        raise ValueError(
            f"Bandwidth constraint violated on edge {idx} "
            f"({edge_loads[idx]} > {problem.capacities[idx]})"
        )

    for indices, limit in problem.shared_constraints:
        group_load = np.sum(edge_loads[indices])
        if group_load > limit + 1e-6:
            raise ValueError(
                f"Shared bandwidth constraint violated (load {group_load} > {limit})"
            )

    return edge_loads


def _apply_copy_only_slot(problem, runtime_state, Y_t, topology_info, validate):
    state = np.asarray(runtime_state["state"], dtype=bool)
    demands = np.asarray(runtime_state["demands"], dtype=bool)

    if validate:
        chunk_indices, edge_indices = np.where(Y_t > 0)
        for c, e in zip(chunk_indices, edge_indices):
            src = int(topology_info.edge_src[e])
            if not state[c, src]:
                raise ValueError(
                    f"Feasibility violated: node {src} sends chunk {c} but does not hold it"
                )

    arrivals = np.zeros((problem.C, problem.V), dtype=bool)
    chunk_indices, edge_indices = np.where(Y_t > 0)
    if chunk_indices.size > 0:
        dst_nodes = topology_info.edge_dst[edge_indices]
        arrivals[chunk_indices, dst_nodes] = True

    satisfied = arrivals & demands
    next_state = state | arrivals
    next_demands = demands & ~arrivals
    return {
        "collective_type": runtime_state["collective_type"],
        "state": next_state,
        "demands": next_demands,
    }, {
        "newly_satisfied": satisfied,
        "newly_satisfied_count": int(satisfied.sum()),
    }


def _apply_allreduce_slot(problem, runtime_state, Y_t, topology_info, validate):
    C = problem.C
    V = problem.V

    mode = np.asarray(runtime_state["mode"], dtype=np.int8)
    partial = np.asarray(runtime_state["partial"], dtype=bool)
    full = np.asarray(runtime_state["full"], dtype=bool)
    visited = np.asarray(runtime_state["visited"], dtype=bool)
    demands = np.asarray(runtime_state["demands"], dtype=bool)

    next_mode = mode.copy()
    next_partial = np.zeros_like(partial)
    next_full = full.copy()
    next_visited = np.zeros_like(visited)
    next_demands = demands.copy()
    satisfied = np.zeros_like(demands)

    for c in range(C):
        scheduled_edges = np.where(Y_t[c] > 0)[0]
        if mode[c] == MODE_REDUCE:
            src_used = np.zeros(V, dtype=bool)
            token_rows = partial[c]
            visit_rows = visited[c]

            for e in scheduled_edges:
                src = int(topology_info.edge_src[e])
                if not token_rows[src].any():
                    raise ValueError(
                        f"Feasibility violated: reduce-mode source node {src} "
                        f"does not hold shard {c}"
                    )
                if src_used[src]:
                    raise ValueError(
                        "reduce-mode source token was used more than once in the same slot"
                    )
                src_used[src] = True

            merged_tokens = np.zeros((V, V), dtype=bool)
            merged_visited = np.zeros((V, V), dtype=bool)
            resident_mask = token_rows.any(axis=1)
            keep_nodes = np.where(resident_mask & ~src_used)[0]
            if keep_nodes.size > 0:
                merged_tokens[keep_nodes] = token_rows[keep_nodes]
                merged_visited[keep_nodes] = visit_rows[keep_nodes]

            incoming_sources = [[] for _ in range(V)]

            for e in scheduled_edges:
                src = int(topology_info.edge_src[e])
                dst = int(topology_info.edge_dst[e])
                merged_tokens[dst] |= token_rows[src]
                incoming_sources[dst].append(src)

            for dst in range(V):
                participant_count = len(incoming_sources[dst]) + int(dst in keep_nodes)
                if participant_count == 0:
                    continue
                if participant_count == 1 and dst not in keep_nodes:
                    src = incoming_sources[dst][0]
                    merged_visited[dst] = visit_rows[src]
                    merged_visited[dst, dst] = True
                elif participant_count >= 2:
                    merged_visited[dst] = False
                    merged_visited[dst, dst] = True

            copies_per_contributor = merged_tokens.sum(axis=0)
            if np.any(copies_per_contributor != 1):
                raise ValueError(
                    f"AllReduce provenance invariant violated for shard {c}: "
                    f"copies per contributor={copies_per_contributor.tolist()}"
                )

            masses = merged_tokens.sum(axis=1)
            complete_nodes = np.where(masses == V)[0]
            if complete_nodes.size > 1:
                raise ValueError(
                    f"AllReduce shard {c} produced multiple full tokens in one slot"
                )

            if complete_nodes.size == 1:
                dst = int(complete_nodes[0])
                next_mode[c] = MODE_FULL
                next_partial[c] = False
                next_full[c] = False
                next_full[c, dst] = True
                next_visited[c] = False
                if next_demands[c, dst]:
                    satisfied[c, dst] = True
                next_demands[c, dst] = False
            else:
                next_mode[c] = MODE_REDUCE
                next_partial[c] = merged_tokens
                next_full[c] = False
                next_visited[c] = merged_visited
        else:
            dst_received = np.zeros(V, dtype=bool)
            for e in scheduled_edges:
                src = int(topology_info.edge_src[e])
                dst = int(topology_info.edge_dst[e])
                if not full[c, src]:
                    raise ValueError(
                        f"Feasibility violated: full-mode source node {src} "
                        f"does not hold full shard {c}"
                    )
                if full[c, dst]:
                    raise ValueError(
                        f"Redundant full delivery: node {dst} already holds shard {c}"
                    )
                if dst_received[dst]:
                    raise ValueError(
                        f"Duplicate full delivery in one slot for shard {c} to node {dst}"
                    )
                dst_received[dst] = True

            if scheduled_edges.size > 0:
                dst_nodes = topology_info.edge_dst[scheduled_edges]
                next_full[c, dst_nodes] = True
                newly_satisfied = demands[c, dst_nodes]
                satisfied[c, dst_nodes] = newly_satisfied
                next_demands[c, dst_nodes] = False
            next_partial[c] = False
            next_mode[c] = MODE_FULL
            next_visited[c] = visited[c]

    return {
        "collective_type": runtime_state["collective_type"],
        "mode": next_mode,
        "partial": next_partial,
        "full": next_full,
        "visited": next_visited,
        "demands": next_demands,
    }, {
        "newly_satisfied": satisfied,
        "newly_satisfied_count": int(satisfied.sum()),
    }


def apply_slot_schedule(
    problem,
    runtime_state: Dict[str, object],
    Y_t: np.ndarray,
    topology_info=None,
    validate: bool = False,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Apply one slot schedule and return next runtime state plus transition info."""
    topo = topology_info or getattr(problem, "topology_info", None)
    if topo is None:
        raise ValueError("topology_info is required to apply a slot schedule")

    if getattr(topo, "timing", None):
        if Y_t.shape != (problem.C, problem.E) or np.any((Y_t != 0) & (Y_t != 1)):
            raise ValueError("timed schedule must be a binary (C,E) matrix")
        from .timed_state import apply_timed_slot
        return apply_timed_slot(problem, runtime_state, Y_t, topo)
    _validate_schedule_common(problem, Y_t, topo)
    collective_type = normalize_collective_type(runtime_state["collective_type"])

    if collective_type in COPY_ONLY_COLLECTIVES:
        return _apply_copy_only_slot(problem, runtime_state, Y_t, topo, validate)
    if collective_type == "allreduce":
        return _apply_allreduce_slot(problem, runtime_state, Y_t, topo, validate)
    raise ValueError(f"Unsupported collective_type={collective_type!r}")
