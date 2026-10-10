#!/usr/bin/env python3
"""Generate communication schedules using an AICCL checkpoint."""
import argparse
import json
from pathlib import Path
import signal
import time
import numpy as np
import torch
from rlccl.config import DATA_DIR, POLICY_OPTIONS, PROJECT_ROOT
from rlccl.dataset import load_dataset
from rlccl.envs.decoder import SlotDecoder
from rlccl.envs.runtime_state import init_runtime_state, apply_slot_schedule, is_completed
from rlccl.envs.timing import evaluate_timed_schedule
from rlccl.runner import load_model, save_json, save_schedule
from rlccl.training.checkpoint import seed_training


class SolveTimeout(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=PROJECT_ROOT / 'checkpoints/D512.pth')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--topology', default='all')
    parser.add_argument('--collective', choices=['all', 'allgather', 'allreduce', 'alltoall', 'alltoallv'], default='all')
    parser.add_argument('--chunk-factor', choices=['all', '1', '2', '4'], default='all')
    parser.add_argument('--limit', type=int, help='Only run the first N selected inputs')
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--out', type=Path, default=Path('outputs/test'))
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('limit must be positive')
    if not 0 < args.timeout < float('inf'):
        parser.error('timeout must be finite and positive')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    seed_training(42)
    model = load_model(args.checkpoint, args.device)
    problems = load_dataset('test', args.data_dir, args.topology, args.collective, args.chunk_factor)
    if args.limit is not None:
        problems = problems[:args.limit]
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'schedules').mkdir()
    rows = []
    previous_handler = signal.signal(signal.SIGALRM,
        lambda *_: (_ for _ in ()).throw(SolveTimeout()))
    try:
        with torch.no_grad(), (args.out / 'results.jsonl').open('w') as output:
            for identifier, problem in problems:
                state = init_runtime_state(problem)
                decoder = SlotDecoder(problem.topology_info)
                actions, schedule = [], []
                timed_out = False
                started = time.perf_counter()
                try:
                    signal.setitimer(signal.ITIMER_REAL, args.timeout)
                    for slot in range(problem.T):
                        y, *_ = decoder.decode_slot(model, problem, state, slot, problem.T,
                            train=False, selector='learned', policy_options=POLICY_OPTIONS)
                        actions.extend((slot, int(c), int(e)) for c, e in zip(*np.nonzero(y)))
                        schedule.append(y)
                        state, _ = apply_slot_schedule(problem, state, y, validate=True)
                        if is_completed(problem, state):
                            break
                except SolveTimeout:
                    timed_out = True
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                seconds = time.perf_counter() - started
                evaluation = evaluate_timed_schedule(schedule, problem)
                success = bool(evaluation['success'] and not timed_out)
                row = dict(id=identifier, collective=problem.collective_type,
                    success=success, timeout=timed_out,
                    completion_slots=len(schedule) if success else None,
                    model_time=len(schedule) * problem.topology_info.timing.delta if success else None,
                    actions=len(actions), synthesis_seconds=seconds)
                save_schedule(args.out / 'schedules' / (identifier + '.npz'), problem, sorted(actions))
                output.write(json.dumps(row) + '\n')
                output.flush()
                rows.append(row)
                print(f'{identifier}: success={success}, slots={row["completion_slots"]}, '
                      f'synthesis={seconds:.3f}s', flush=True)
    finally:
        signal.signal(signal.SIGALRM, previous_handler)
    summary = dict(inputs=len(rows), success=sum(r['success'] for r in rows),
                   failed=sum(not r['success'] for r in rows), timeouts=sum(r['timeout'] for r in rows))
    save_json(args.out / 'summary.json', summary)
    print(summary)


if __name__ == '__main__':
    main()
