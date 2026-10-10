"""Slot service budgets. Times are seconds in alpha_beta, reference units in capacity.

This is a conservative synthesis model, not a packet or GPU simulator. Each
operation starts at a slot boundary, reserves service in successive slots and
becomes usable only at its completion boundary. Copy and aggregate share budgets.
"""
import hashlib
import json
import math

import numpy as np


def ceil_slots(value):
    return np.maximum(1, np.ceil(np.asarray(value) - 1e-10).astype(np.int64))


class SlotTiming:
    def __init__(self, topo, mode="capacity", refinement=1.0, chunk_bytes=None, profile=None):
        if mode not in {"capacity", "alpha_beta"}:
            raise ValueError("normalization must be capacity or alpha_beta")
        if not np.isfinite(refinement) or refinement <= 0:
            raise ValueError("slot_refinement must be finite and positive")
        self.mode = mode
        self.refinement = float(refinement)
        self.chunk_bytes = chunk_bytes
        self.profile = profile
        self.edge_order = np.asarray(topo.edges).tolist()
        E, V = topo.E, topo.V
        self.groups = [np.asarray(es, dtype=np.int64) for es, _ in topo.shared_constraints]
        self.edge_groups = [[] for _ in range(E)]
        for g, es in enumerate(self.groups):
            for e in es:
                self.edge_groups[e].append(g)
        if mode == "capacity":
            caps = np.asarray(topo.capacities, dtype=np.float64)
            group_caps = np.asarray([lim for _, lim in topo.shared_constraints], dtype=np.float64)
            if np.any(caps <= 0) or np.any(group_caps <= 0):
                raise ValueError("timed capacity mode requires positive edge and shared capacities")
            self.copy_cost = 1.0 / caps
            self.aggregate_cost = self.copy_cost.copy()
            self.group_cost = 1.0 / group_caps
            self.reference_time = 1.0
        else:
            if not isinstance(chunk_bytes, int) or isinstance(chunk_bytes, bool) or chunk_bytes <= 0:
                raise ValueError("alpha_beta requires positive integer --chunk_bytes")
            if not isinstance(profile, dict) or profile.get("units") != "seconds_bytes":
                raise ValueError("alpha_beta requires a profile with units=seconds_bytes")
            if not profile.get("provenance"):
                raise ValueError("profile must state calibration provenance (or explicitly synthetic)")
            def vector(key, size, positive=False):
                if key not in profile:
                    raise ValueError(f"profile is missing {key}; unknown gamma cannot be inferred")
                a = np.asarray(profile[key], dtype=np.float64)
                if a.ndim == 0:
                    a = np.full(size, float(a))
                if a.shape != (size,) or not np.isfinite(a).all() or np.any(a < 0):
                    raise ValueError(f"invalid profile {key}: expected scalar or {size} finite nonnegative values")
                if positive and np.any(a <= 0):
                    raise ValueError(f"{key} must be positive")
                return a
            alpha = vector("alpha_s", E)
            beta = vector("beta_s_per_byte", E, positive=True)
            gamma = vector("gamma_s_per_byte", V)
            self.copy_cost = alpha + chunk_bytes * beta
            self.aggregate_cost = self.copy_cost + chunk_bytes * gamma[topo.edge_dst]
            self.group_cost = chunk_bytes * vector("shared_beta_s_per_byte", len(self.groups))
            if np.any(self.group_cost <= 0):
                raise ValueError("shared_beta_s_per_byte must be positive for every shared group")
            # One fixed reference for both phases, including the slowest shared resource.
            self.reference_time = float(max(self.aggregate_cost.max(), self.group_cost.max(initial=0)))
        self.delta = self.reference_time / self.refinement
        self.copy_work = self.copy_cost / self.delta
        self.aggregate_work = self.aggregate_cost / self.delta
        self.group_work = self.group_cost / self.delta
        self.height = int(ceil_slots(max(self.aggregate_work.max(), self.group_work.max(initial=0))))
        offsets = np.arange(self.height)[:, None]
        self.copy_pattern = np.clip(self.copy_work[None, :] - offsets, 0, 1)
        self.aggregate_pattern = np.clip(self.aggregate_work[None, :] - offsets, 0, 1)
        self.group_pattern = np.clip(self.group_work[None, :] - offsets, 0, 1)
        self.copy_delay = ceil_slots(self.copy_work)
        self.aggregate_delay = ceil_slots(self.aggregate_work)
        for e, gs in enumerate(self.edge_groups):
            if gs:
                group_delay = int(ceil_slots(self.group_work[gs].max()))
                self.copy_delay[e] = max(self.copy_delay[e], group_delay)
                self.aggregate_delay[e] = max(self.aggregate_delay[e], group_delay)
        self.signature = hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    def to_dict(self):
        return {"mode": self.mode, "refinement": self.refinement,
                "chunk_bytes": self.chunk_bytes, "profile": self.profile,
                "reference_time": self.reference_time, "slot_duration": self.delta,
                "edge_order": self.edge_order,
                "shared_groups": [es.tolist() for es in self.groups],
                "copy_cost": self.copy_cost.tolist(),
                "aggregate_cost": self.aggregate_cost.tolist(),
                "shared_cost": self.group_cost.tolist(),
                "service_model": "boundary_service_reservation_v1"}

    def empty_loads(self):
        return np.zeros((self.height, len(self.copy_cost) + len(self.groups)), dtype=np.float64)

    def feasible_edges(self, loads, aggregate=False):
        E = len(self.copy_cost)
        pattern = self.aggregate_pattern if aggregate else self.copy_pattern
        valid = np.all(loads[:, :E] + pattern <= 1 + 1e-9, axis=0)
        group_ok = np.all(loads[:, E:] + self.group_pattern <= 1 + 1e-9, axis=0)
        for g in np.where(~group_ok)[0]:
            valid[self.groups[g]] = False
        return valid

    def reserve(self, loads, e, aggregate=False):
        pattern = self.aggregate_pattern if aggregate else self.copy_pattern
        if np.any(loads[:, e] + pattern[:, e] > 1 + 1e-9):
            raise ValueError(f"cross-slot service budget exceeded on edge {e}")
        for g in self.edge_groups[e]:
            if np.any(loads[:, len(self.copy_cost) + g] + self.group_pattern[:, g] > 1 + 1e-9):
                raise ValueError(f"cross-slot shared service budget exceeded in group {g}")
        loads[:, e] += pattern[:, e]
        for g in self.edge_groups[e]:
            loads[:, len(self.copy_cost) + g] += self.group_pattern[:, g]


def configure_timing(topo, mode="capacity", refinement=1.0, chunk_bytes=None, profile=None, timed=False):
    """Leave the default capacity/refinement=1 path byte-for-byte compatible."""
    if mode == "capacity" and profile is not None:
        raise ValueError("cost_profile applies to alpha_beta normalization only")
    if mode == "capacity" and refinement == 1 and not timed:
        topo.timing = None
        return topo
    timing = SlotTiming(topo, mode, refinement, chunk_bytes, profile)
    topo.base_capacities = np.asarray(topo.capacities).copy()
    topo.base_shared_constraints = [(list(es), float(lim)) for es, lim in topo.shared_constraints]
    topo.timing = timing
    # Integer bounds limit candidate loop size; the service calendar enforces actual capacity.
    topo.capacities = ceil_slots(1.0 / timing.copy_work)
    topo.shared_constraints = [(es.tolist(), int(ceil_slots(1.0 / timing.group_work[g])))
                               for g, es in enumerate(timing.groups)]
    return topo


def add_timing_arguments(parser):
    parser.add_argument("--normalization", choices=["capacity", "alpha_beta"], default="capacity")
    parser.add_argument("--slot_refinement", type=float, default=1.0,
                        help="Delta=reference/refinement; 2 halves the slot duration")
    parser.add_argument("--timed_slots", action="store_true", help="use service reservations even at refinement=1")
    parser.add_argument("--chunk_bytes", type=int, default=None)
    parser.add_argument("--cost_profile", default=None, help="explicit JSON costs in edge/node/shared-group order")


def configure_from_args(topo, args):
    profile = None
    if args.cost_profile:
        with open(args.cost_profile) as f:
            profile = json.load(f)
        # A multi-topology file must name each topology; never reuse indexed vectors silently.
        if "topologies" in profile:
            profile = profile["topologies"][topo.name]
    return configure_timing(topo, args.normalization, args.slot_refinement,
                            args.chunk_bytes, profile, args.timed_slots)


def timed_horizon(topo, reference_slots):
    timing = getattr(topo, "timing", None)
    if not timing:
        return reference_slots
    horizon = int(math.floor(reference_slots * timing.refinement + 1e-10))
    if horizon < 1:
        raise ValueError("physical time budget is shorter than one slot")
    return horizon


def attach_timing_to_problems(named_problems, args):
    topologies = {}
    metadata = {}
    for scenario_id, problem in named_problems:
        # Scenario IDs include the topology; callers also attach its exact name.
        name = getattr(problem, "topology_name", None) or problem.topology_info.name
        if name is None:
            raise ValueError("problem must carry its topology name for timing configuration")
        if name not in topologies:
            topo = problem.topology_info
            topo.name = name
            topologies[name] = configure_from_args(topo, args)
            timing = getattr(topo, "timing", None)
            if timing:
                metadata[name] = timing.to_dict()
        topo = topologies[name]
        problem.topology_info = topo
        problem.capacities = topo.capacities
        problem.shared_constraints = topo.shared_constraints
        problem.T = timed_horizon(topo, problem.T)
    return metadata


def validate_timing_checkpoint(checkpoint, metadata):
    if metadata and checkpoint.get("timing_config") != metadata:
        raise ValueError("checkpoint timing configuration differs: use matching normalization, S and Delta, or train a new model")


def evaluate_timed_schedule(schedule, problem):
    from .runtime_state import init_runtime_state, apply_slot_schedule, is_completed
    timing = problem.topology_info.timing
    state = init_runtime_state(problem)
    result = {"success": False, "completion_slots": None, "model_time": None,
              "time_unit": "seconds" if timing.mode == "alpha_beta" else "capacity_reference",
              "timing": timing.to_dict(), "error": ""}
    if len(schedule) > problem.T:
        result["error"] = "Schedule length exceeds time limit"
        return result
    for t, Y in enumerate(schedule):
        try:
            state, _ = apply_slot_schedule(problem, state, np.asarray(Y), validate=True)
        except ValueError as exc:
            result["error"] = f"Invalid schedule at step {t}: {exc}"
            return result
        if is_completed(problem, state):
            result.update(success=True, completion_slots=t + 1, model_time=(t + 1) * timing.delta)
            return result
    result["error"] = "Unfinished demands or in-flight operations at end of schedule"
    return result
