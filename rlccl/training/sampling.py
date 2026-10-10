"""Episode-budget sampling with explicit collective/topology-family balance."""
from collections import defaultdict

import numpy as np


def balanced_episode_order(problems, episodes):
    if not isinstance(episodes, int) or isinstance(episodes, bool) or episodes < 1:
        raise ValueError('episodes_per_update must be a positive integer')
    groups = defaultdict(lambda: defaultdict(list))
    for index, (_, problem) in enumerate(problems):
        if not np.asarray(problem.demands).any():
            raise ValueError('Balanced episode corpus contains a zero-demand case')
        family = getattr(problem, 'topology_family', None)
        if not family:
            raise ValueError('Balanced sampling requires an explicit topology_family per problem')
        groups[problem.collective_type][family].append(index)
    if set(groups) != {'allgather', 'allreduce', 'alltoall', 'alltoallv'}:
        raise ValueError('Balanced corpus must contain all four collectives')
    family_sets = [set(cells) for cells in groups.values()]
    if any(families != family_sets[0] for families in family_sets):
        raise ValueError('Each collective must cover the same topology families')
    pools = {(ct, family): np.random.permutation(indices).tolist()
             for ct, cells in groups.items() for family, indices in cells.items()}
    positions = defaultdict(int)
    collectives = sorted(groups)
    families = sorted(family_sets[0])
    family_orders = {ct: np.random.permutation(families).tolist() for ct in collectives}
    selected = []
    round_index = 0
    while len(selected) < episodes:
        for ct in np.random.permutation(collectives):
            family = family_orders[ct][round_index % len(families)]
            key = ct, family
            if positions[key] == len(pools[key]):
                pools[key] = np.random.permutation(groups[ct][family]).tolist()
                positions[key] = 0
            selected.append(pools[key][positions[key]])
            positions[key] += 1
            if len(selected) == episodes:
                break
        round_index += 1
    return np.asarray(selected, dtype=np.int64)

def joint_collective_repeated_order(problems, episodes):
    """Two own rollouts of each collective per eight-episode PPO update."""
    if not isinstance(episodes,int) or isinstance(episodes,bool) or episodes<8 or episodes%8:
        raise ValueError('Joint four-collective own-pair plan requires an episode multiple of eight')
    base=balanced_episode_order(problems,episodes//2)
    plan=np.repeat(base,2)
    for begin in range(0,episodes,8):
        block=plan[begin:begin+8]
        kinds=[problems[int(i)][1].collective_type for i in block]
        if any(kinds.count(ct)!=2 for ct in ('allgather','allreduce','alltoall','alltoallv')):
            raise AssertionError('Every PPO update must include two own rollouts per collective')
    return plan
