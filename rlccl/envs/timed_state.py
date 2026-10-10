"""Delayed arrivals and token consumption for the timed synthesis model."""
import numpy as np

from .runtime_state import clone_runtime_state, MODE_REDUCE, MODE_FULL


def init_timed_state(state, topo):
    timing = topo.timing
    state.update(pending=[], resource_loads=timing.empty_loads(), slot=0,
                 timing_signature=timing.signature)
    return state


def pending_destinations(state):
    reserved = np.zeros_like(state["demands"], dtype=bool)
    for event in state.get("pending", []):
        reserved[event["chunk"], event["dst"]] = True
    return reserved


def aggregate_flags(state, chunks, edges, topo):
    if state["collective_type"] != "allreduce":
        return np.zeros(len(chunks), dtype=bool)
    return ((state["mode"][chunks] == MODE_REDUCE)
            & state["partial"][chunks, topo.edge_dst[edges]].any(axis=1))


def timed_candidate_mask(state, chunks, edges, topo, loads, used, arrivals, remaining_slots):
    timing = topo.timing
    src, dst = topo.edge_src[edges], topo.edge_dst[edges]
    agg = aggregate_flags(state, chunks, edges, topo)
    ok = np.where(agg, timing.feasible_edges(loads, True)[edges],
                  timing.feasible_edges(loads)[edges])
    delay = np.where(agg, timing.aggregate_delay[edges], timing.copy_delay[edges])
    ok &= delay <= remaining_slots
    ok &= ~pending_destinations(state)[chunks, dst] & ~arrivals[chunks, dst]
    if state["collective_type"] == "allreduce":
        red = state["mode"][chunks] == MODE_REDUCE
        # Fixed slot-start tokens: an aggregate consumes BOTH inputs, and an
        # incoming token cannot also serve as a source within the same slot.
        ok &= ~(red & (used[chunks, src] | used[chunks, dst] | arrivals[chunks, src]))
    return ok


def validate_provenance(state, V):
    if state["collective_type"] != "allreduce":
        return
    counts = state["partial"].sum(axis=1).astype(np.int64)
    for event in state["pending"]:
        if event["kind"] == "partial":
            counts[event["chunk"]] += event["payload"]
    red = state["mode"] == MODE_REDUCE
    if np.any(counts[red] != 1):
        raise ValueError("AllReduce provenance must be conserved across resident and in-flight tokens")


def apply_timed_slot(problem, state, Y, topo):
    timing = topo.timing
    if state.get("timing_signature") != timing.signature:
        raise ValueError("runtime state and timing profile do not match")
    nxt = clone_runtime_state(state)
    used = np.zeros((problem.C, problem.V), dtype=bool)
    arrivals = pending_destinations(state)
    residents = state["partial"].any(axis=2) | state["full"] if state["collective_type"] == "allreduce" else state["state"]
    now = state["slot"]
    for c, e in zip(*np.where(Y)):
        c, e = int(c), int(e)
        src, dst = int(topo.edge_src[e]), int(topo.edge_dst[e])
        if not residents[c, src] or src == dst:
            raise ValueError("timed action needs a resident source and distinct destination")
        if arrivals[c, dst]:
            raise ValueError("duplicate or conflicting in-flight destination")
        red = state["collective_type"] == "allreduce" and state["mode"][c] == MODE_REDUCE
        agg = bool(red and state["partial"][c, dst].any())
        if red:
            if used[c, src] or used[c, dst] or arrivals[c, src]:
                raise ValueError("source or aggregate partner consumed more than once")
            payload = state["partial"][c, src].copy()
            visits = state["visited"][c, src].copy()
            used[c, src] = True
            nxt["partial"][c, src] = False
            nxt["visited"][c, src] = False
            if agg:
                partner = state["partial"][c, dst]
                if np.any(payload & partner):
                    raise ValueError("aggregate input contributions overlap")
                payload |= partner
                used[c, dst] = True
                nxt["partial"][c, dst] = False
                nxt["visited"][c, dst] = False
                visits[:] = False
            visits[dst] = True
            kind = "partial"
        else:
            if residents[c, dst]:
                raise ValueError("copy destination already holds the value")
            payload, visits, kind = None, None, "full" if state["collective_type"] == "allreduce" else "copy"
        timing.reserve(nxt["resource_loads"], e, agg)
        delay = int(timing.aggregate_delay[e] if agg else timing.copy_delay[e])
        if now + delay > problem.T:
            raise ValueError("action would arrive outside the synthesis horizon")
        nxt["pending"].append({"chunk": c, "dst": dst, "edge": e, "kind": kind,
                               "aggregate": agg, "payload": payload, "visited": visits,
                               "arrival": now + delay})
        arrivals[c, dst] = True
    nxt["slot"] = now + 1
    satisfied = np.zeros_like(state["demands"], dtype=bool)
    waiting = []
    for event in nxt["pending"]:
        if event["arrival"] > nxt["slot"]:
            waiting.append(event)
            continue
        c, dst = event["chunk"], event["dst"]
        if event["kind"] == "partial":
            if nxt["partial"][c, dst].any():
                raise ValueError("arrival conflicts with a resident token")
            nxt["partial"][c, dst] = event["payload"]
            nxt["visited"][c, dst] = event["visited"]
            if event["payload"].sum() == problem.V:
                nxt["mode"][c] = MODE_FULL
                nxt["partial"][c] = False
                nxt["visited"][c] = False
                nxt["full"][c, dst] = True
                satisfied[c, dst] = nxt["demands"][c, dst]
                nxt["demands"][c, dst] = False
        else:
            key = "full" if event["kind"] == "full" else "state"
            nxt[key][c, dst] = True
            satisfied[c, dst] = nxt["demands"][c, dst]
            nxt["demands"][c, dst] = False
    nxt["pending"] = waiting
    loads = nxt["resource_loads"]
    loads[:-1] = loads[1:]
    loads[-1] = 0
    validate_provenance(nxt, problem.V)
    return nxt, {"newly_satisfied": satisfied, "newly_satisfied_count": int(satisfied.sum()),
                 "elapsed_time": nxt["slot"] * timing.delta}
