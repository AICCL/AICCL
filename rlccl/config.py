"""AICCL model dimensions and PPO hyperparameters."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'Data'
CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'
OUTPUT_DIR = PROJECT_ROOT / 'outputs'

POLICY_OPTIONS = {'use_stop': True,
 'target_reservations': True,
 'candidate_coverage': True,
 'relational_features': True,
 'arrival_aware': True,
 'shared_redesign': True}

DEFAULT_TRAIN_CONFIG = {'epochs': 30,
 'batch_target': 500,
 'hidden_dim': 64,
 'lr': 0.0001,
 'episodes_per_update': 8,
 'ppo_epochs': 2,
 'mini_batch_size': 32,
 'gamma': 1.0,
 'gae_lambda': 0.95,
 'clip_eps': 0.2,
 'entropy_coef': 0.002,
 'value_coef': 0.5,
 'max_grad_norm': 0.5,
 'return_scale': 22.0,
 'decision_ppo': True,
 'shared_progress_shaping': True,
 'allreduce_shaping_coef': 0.0,
 'critic_encoder_grad': True,
 'separate_gradient_clipping': True,
 'target_joint_kl': 0.015,
 'post_update_kl_guard': True,
 'joint_kl_max_backtracks': 4,
 'joint_kl_backtrack_factor': 0.5,
 'own_input_rloo': True,
 'training_failure_deficit_coef': 4.0,
 'policy_entropy': 'factorized_stop',
 'stop_entropy_weight': 10.0,
 'shared_value_gradient_ratio': 0.5,
 'project_conflicting_value_gradient': True}

MODEL_VERSION = 2
MODEL_FEATURE_DIMS = {
    'node_feat_dim': 5,
    'edge_feat_dim': 2,
    'cand_feat_dim': 11,
    'chunk_feat_dim': 3,
}

# Available topologies
AVAILABLE_TOPOLOGIES = ['regular-4', 'regular-8', 'regular-16',
    'heterogeneous-3-5', 'heterogeneous-4-8', 'heterogeneous-4-4-8',
    'sparse-ring-6', 'sparse-two-clique-8']

# Traffic patterns
TRAFFIC_PATTERNS = {
    'AllGather': 0.3,
    'All-to-All-V': 0.5,
    'AllToAll': 0.2,
}

def get_config():
    """Get default configuration."""
    return DEFAULT_TRAIN_CONFIG.copy()


TIMED_MODEL_VERSION = 3
OPTIMIZED_MODEL_VERSION = 5
ARRIVAL_MODEL_VERSION = 6
SHARED_MODEL_VERSION = 7
OPTIMIZED_POLICY_OPTIONS = dict(use_stop=True, target_reservations=True,
                                candidate_coverage=True, relational_features=True)
TIMED_FEATURE_DIMS = dict(node_feat_dim=7, edge_feat_dim=5, cand_feat_dim=15, chunk_feat_dim=4)


def get_model_feature_dims(timed=False, arrival_aware=False):
    """Get the canonical feature dimensions for SlotLevelPolicy."""
    if arrival_aware and not timed:
        raise ValueError('Arrival features require timed semantics')
    dims = (TIMED_FEATURE_DIMS if timed else MODEL_FEATURE_DIMS).copy()
    if arrival_aware:
        dims['cand_feat_dim'] = 20
    return dims


def get_model_kwargs(hidden_dim=None, timed=False, policy_options=None, actor_prior_scale=0.0):
    """Build keyword arguments for SlotLevelPolicy."""
    arrival_aware = bool(policy_options and policy_options.get('arrival_aware', False))
    kwargs = get_model_feature_dims(timed, arrival_aware)
    if hidden_dim is not None:
        kwargs['hidden_dim'] = hidden_dim
    if policy_options is not None:
        if not timed:
            raise ValueError('Optimized policy requires explicit timed semantics')
        if set(policy_options) not in (set(OPTIMIZED_POLICY_OPTIONS), set(OPTIMIZED_POLICY_OPTIONS) | {'arrival_aware'}, set(OPTIMIZED_POLICY_OPTIONS) | {'arrival_aware', 'shared_redesign'}) or any(
                type(value) is not bool for value in policy_options.values()):
            raise ValueError('Policy options must explicitly bind boolean optimization flags')
        kwargs.update(policy_options)
    if actor_prior_scale:
        if policy_options is None or not policy_options['use_stop']:
            raise ValueError('Actor prior requires optimized STOP options')
        kwargs['actor_prior_scale'] = actor_prior_scale
    return kwargs


def validate_checkpoint_compatibility(checkpoint, timed=False, policy_options=None):
    """Raise a clear error when a checkpoint uses an old model spec."""
    arrival_aware = bool(policy_options and policy_options.get('arrival_aware', False))
    expected_dims = get_model_feature_dims(timed, arrival_aware)
    expected_version = TIMED_MODEL_VERSION if timed else MODEL_VERSION
    if policy_options is not None:
        get_model_kwargs(timed=timed, policy_options=policy_options)
        expected_version = (SHARED_MODEL_VERSION if policy_options.get('shared_redesign', False) else
                            ARRIVAL_MODEL_VERSION if arrival_aware else OPTIMIZED_MODEL_VERSION)
        if checkpoint.get('policy_options') != policy_options:
            raise ValueError('Checkpoint optimization flags differ from the selected policy')
    checkpoint_version = checkpoint.get('model_version')
    checkpoint_dims = checkpoint.get('model_feature_dims')

    if checkpoint_version != expected_version:
        raise ValueError(
            f"Incompatible checkpoint model_version={checkpoint_version!r}; "
            f"expected {expected_version}. Retrain for the selected feature schema/slot model."
        )

    if checkpoint_dims != expected_dims:
        raise ValueError(
            f"Incompatible checkpoint feature dims={checkpoint_dims!r}; "
            f"expected {expected_dims!r}."
        )
