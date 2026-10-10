"""Common model and checkpoint helpers for train.py and test.py."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .config import POLICY_OPTIONS, get_model_kwargs, validate_checkpoint_compatibility
from .models.slot_policy import SlotLevelPolicy

D512_SHA256 = '2fe0c47c950744bd878c90f59dec42033b9e307675a1261dca6200063eb99471'


def make_model(device='cpu', hidden_dim=64):
    model = SlotLevelPolicy(**get_model_kwargs(hidden_dim, timed=True,
                                               policy_options=POLICY_OPTIONS)).to(device)
    model.current_slot_plan_observation = True
    return model


def load_model(checkpoint, device='cpu'):
    checkpoint = Path(checkpoint)
    if checkpoint.name == 'D512.pth':
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != D512_SHA256:
            raise ValueError('D512 checkpoint checksum does not match')
    data = torch.load(checkpoint, map_location=device, weights_only=False)
    validate_checkpoint_compatibility(data, timed=True, policy_options=POLICY_OPTIONS)
    hidden_dim = data.get('hidden_dim', data.get('spec', {}).get('hidden_dim', 64))
    model = make_model(device, hidden_dim)
    model.load_state_dict(data['model_state_dict'], strict=True)
    model.eval()
    return model


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def save_schedule(path, problem, actions):
    indices = np.asarray(actions, dtype=np.int64).reshape(-1, 3)
    np.savez_compressed(path, shape=np.asarray([problem.T, problem.C, problem.E]),
                        slots=indices[:, 0], chunks=indices[:, 1], edges=indices[:, 2])
