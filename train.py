#!/usr/bin/env python3
"""Train AICCL with PPO or resume the same training run."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('RLCCL_TQDM', '0')
import torch
from rlccl.config import DATA_DIR, POLICY_OPTIONS, get_config, get_model_feature_dims
from rlccl.dataset import load_dataset
from rlccl.runner import make_model
from rlccl.training.checkpoint import atomic_save, capture_rng_state, restore_rng_state, seed_training
from rlccl.training.ppo_trainer import train_epoch
from rlccl.training.sampling import joint_collective_repeated_order


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--out', type=Path, default=Path('checkpoints/training'))
    parser.add_argument('--episodes', type=int, default=6000)
    parser.add_argument('--until', type=int, help='Stop at this episode while retaining the full episode plan')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    until = args.episodes if args.until is None else args.until
    if args.episodes < 8 or args.episodes % 8 or not 0 <= until <= args.episodes or until % 8:
        parser.error('episodes and until must be multiples of eight, with 0 <= until <= episodes')
    if not 0 < args.lr < float('inf'):
        parser.error('lr must be finite and positive')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    seed_training(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; use --device cpu')
    problems = load_dataset('train', args.data_dir)
    plan = joint_collective_repeated_order(problems, args.episodes)
    model = make_model(device)
    config = get_config()
    config['lr'] = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    signature = dict(episodes=args.episodes, seed=args.seed, hidden_dim=64,
        config=config, policy_options=POLICY_OPTIONS,
        data_sha256=hashlib.sha256((args.data_dir / 'train.json').read_bytes()).hexdigest(),
        episode_ids=[problems[int(i)][0] for i in plan])
    args.out.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.out / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    metrics_file = args.out / 'metrics.jsonl'
    completed = updates = 0
    if args.resume:
        checkpoint = torch.load(args.out / 'latest.pth', map_location=device, weights_only=False)
        if checkpoint['run_config'] != signature:
            raise ValueError('Resume requires the same data, episode budget, seed and configuration')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        restore_rng_state(checkpoint['rng_state'])
        completed, updates = checkpoint['completed_episodes'], checkpoint['updates']
        if until < completed:
            raise ValueError('until precedes the saved checkpoint')
        content = metrics_file.read_bytes()
        length = checkpoint['metrics_bytes']
        if hashlib.sha256(content[:length]).hexdigest() != checkpoint['metrics_sha256']:
            raise ValueError('Saved training metrics do not match the checkpoint')
        if len(content) > length:
            backup = args.out / f'uncommitted-metrics-{os.getpid()}.jsonl'
            backup.write_bytes(content[length:])
            with metrics_file.open('r+b') as output:
                output.truncate(length)
    else:
        metrics_file.touch(exist_ok=False)

    def save_checkpoint():
        checkpoint = dict(model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(), rng_state=capture_rng_state(),
            completed_episodes=completed, updates=updates, run_config=signature,
            hidden_dim=64, model_version=7, model_feature_dims=get_model_feature_dims(True, True),
            policy_options=POLICY_OPTIONS, current_slot_plan_observation=True,
            metrics_bytes=metrics_file.stat().st_size,
            metrics_sha256=hashlib.sha256(metrics_file.read_bytes()).hexdigest())
        atomic_save(checkpoint, args.out / 'latest.pth')
        if completed in (0, 64, 256, 512, 1000, 2000, 4000, 5000, 6000) or completed == until:
            atomic_save(checkpoint, args.out / f'checkpoint-ep{completed:04d}.pth')

    if not args.resume:
        save_checkpoint()
    while completed < until:
        metrics = {}
        losses = train_epoch(model, problems, device, config, updates, args.episodes // 8,
            optimizer=optimizer, metrics=metrics, episode_indices=plan[completed:completed + 8])
        completed += 8
        updates += 1
        metrics['decision_kl_mean'] = metrics.pop('final_joint_kl_mean')
        metrics['decision_kl_max'] = metrics.pop('final_joint_kl_max')
        with metrics_file.open('a') as output:
            output.write(json.dumps(dict(completed_episodes=completed, metrics=metrics,
                                         losses=list(losses)), allow_nan=False) + '\n')
            output.flush()
            os.fsync(output.fileno())
        save_checkpoint()
        print(f'Episodes {completed}/{args.episodes}: success={metrics["rollout_success_rate"]:.3f}, '
              f'loss={losses[0]:.6f}', flush=True)
    print(f'Checkpoint: {args.out / "latest.pth"}')


if __name__ == '__main__':
    main()
