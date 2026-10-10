"""Score generated collective communication schedules."""
import numpy as np
from .runtime_state import init_runtime_state, apply_slot_schedule, is_completed

def evaluate_schedule(schedule, problem):
    """
    Evaluates a schedule.
    schedule: List of T matrices, each (C, E) representing Y_t
    
    Returns:
        score: Combined score where:
            - Primary: negative completion step (fewer steps = higher score)
            - Secondary: weighted satisfaction as tiebreaker
            Score = -completion_step + 0.001 * weighted_score
            
        Error score: -(T + 2) to ensure errors rank below any valid completion
    """
    timing = getattr(problem.topology_info, "timing", None)
    if timing:
        from .timing import evaluate_timed_schedule
        metrics = evaluate_timed_schedule(schedule, problem)
        score = -metrics["model_time"] / timing.reference_time if metrics["success"] else -(problem.T / timing.refinement + 2)
        return score, metrics["error"]
    # Error score: worse than any valid completion (which is at most -T)
    error_score = -(problem.T + 2)
    
    if len(schedule) > problem.T:
        return error_score, "Schedule length exceeds time limit"

    runtime_state = init_runtime_state(problem)
    
    weighted_score = 0.0  # Tiebreaker score
    completion_step = len(schedule)  # Default to max if not completed
    
    for t, Y_t in enumerate(schedule):
        try:
            runtime_state, transition_info = apply_slot_schedule(
                problem,
                runtime_state,
                np.asarray(Y_t, dtype=np.int32),
                topology_info=getattr(problem, "topology_info", None),
                validate=True,
            )
        except ValueError as exc:
            return error_score, f"Invalid schedule at step {t}: {exc}"

        weighted_score += (1.0 / (t + 1)) * transition_info["newly_satisfied_count"]

        if is_completed(problem, runtime_state):
            completion_step = t + 1  # Record completion step (1-indexed)
            break
    
    # Final score: primary = -completion_step, secondary = weighted_score as tiebreaker
    # Fewer steps = higher score (less negative)
    # Same steps: higher weighted_score = higher total score
    final_score = -completion_step + 0.001 * weighted_score
            
    return final_score, ""
