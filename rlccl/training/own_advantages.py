"""Own-input environmental RLOO only; exact ties never become training signal."""
import numpy as np
import torch

def own_input_advantages(group_ids, scores, starts, ends, slots):
    result=torch.zeros(slots,dtype=torch.float32)
    for key in set(group_ids):
        group=[j for j,k in enumerate(group_ids) if k==key]
        if len(group)<2:raise ValueError('Own-input RLOO requires repeated identical inputs')
        values=np.asarray([scores[j] for j in group],dtype=np.float64)
        if not np.isfinite(values).all():raise FloatingPointError('Nonfinite environmental return')
        if np.ptp(values)<=1e-10:continue
        for j in group:
            others=[scores[k] for k in group if k!=j]
            advantage=float(scores[j]-np.mean(others))
            result[int(starts[j]):int(ends[j])+1]=advantage
    scale=float(result.std()) if slots>1 else 0.
    if scale>1e-6:result=result/scale
    else:result.zero_()
    return result


def integer_allreduce_consolidation(state):
    """Node-order invariant own progress; integer counts until the final divide."""
    partial=np.asarray(state['partial'],dtype=bool)
    C,V,_=partial.shape
    counts=partial.sum(axis=2,dtype=np.int64)
    squares=(counts*counts).sum(axis=1,dtype=np.int64)
    for event in state.get('pending',[]):
        if event['kind']=='partial':
            n=int(np.count_nonzero(event['payload']))
            squares[event['chunk']]+=n*n
    squares=np.where(np.asarray(state['mode'])==0,squares,V*V)
    return int(squares.sum(dtype=np.int64))/float(C*V*V)
