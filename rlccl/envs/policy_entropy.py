"""Exploration regularizers for a categorical move pool plus one STOP choice."""
import math
import torch
from torch.distributions import Bernoulli,Categorical

def factorized_stop_entropy(logits, stop_weight=1.0):
    """H(move | continue) + weight H(STOP / continue).

    This is an exploration regularizer, not the entropy of the full categorical
    action distribution. Sampling and action log probabilities remain unchanged.
    Removing the (1-p_STOP) factor prevents the conditional move entropy from
    suppressing STOP exploration as the candidate pool grows.
    """
    weight=float(stop_weight)
    if not math.isfinite(weight) or weight<=0:
        raise ValueError("STOP entropy weight must be finite and positive")
    if logits.ndim!=1 or logits.numel()<2:
        raise ValueError("Factorized STOP entropy requires a nonempty move pool")
    moves=logits[:-1]
    conditional=Categorical(logits=moves).entropy()
    stop_mass_logit=logits[-1]-torch.logsumexp(moves,dim=0)
    binary=Bernoulli(logits=stop_mass_logit).entropy()
    return conditional+weight*binary
