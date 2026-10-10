"""AICCL collective communication scheduler."""
__version__ = '1.0.0'
from .config import get_config


def __getattr__(name):
    if name in {'SlotLevelPolicy', 'ECDUGNNLayer'}:
        from . import models
        return getattr(models, name)
    raise AttributeError(name)
