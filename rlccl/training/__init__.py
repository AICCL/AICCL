"""Training module for PPO-based optimization."""

from .buffer import SlotBuffer
from .ppo_trainer import train_epoch, evaluate_model, compute_gae_advantages

__all__ = ['SlotBuffer', 'train_epoch', 'evaluate_model', 'compute_gae_advantages']
