"""Collective communication problems, decoder and runtime state."""
from .problem import ProblemInstance, TopologyInfo
from .evaluator import evaluate_schedule
from .runtime_state import apply_slot_schedule, init_runtime_state, is_completed, remaining_demand


def __getattr__(name):
    if name in {'SlotDecoder', 'solve_with_model'}:
        from . import decoder
        return getattr(decoder, name)
    raise AttributeError(name)
