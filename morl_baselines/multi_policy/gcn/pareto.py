import numpy as np

def get_non_pareto_dominated(solutions: np.ndarray):
    is_efficient = np.ones(solutions.shape[0], dtype=bool)
    for i, c in enumerate(solutions):
        if is_efficient[i]:
            # Remove dominated points, will also remove itself
            is_efficient[is_efficient] = np.any(solutions[is_efficient] > c, axis=1)
            # keep this solution as non-dominated
            is_efficient[i] = 1

    return is_efficient, solutions[is_efficient]

def pareto_l2( returns, sma, params ):
    non_dominated_i, non_dominated = get_non_pareto_dominated(returns)

    # we will compute distance of each point with each non-dominated point,
    # duplicate each point with number of non_dominated to compute respective distance
    returns_exp = np.tile(np.expand_dims(returns, 1), (1, len(non_dominated), 1))
    # distance to closest non_dominated point
    l2 = np.min(np.linalg.norm(returns_exp - non_dominated, axis=-1), axis=-1) * -1

    # all points that are too close together (crowding distance < threshold) get a penalty
    non_dominated_i = np.nonzero(non_dominated_i)[0]
    _, unique_i = np.unique(non_dominated, axis=0, return_index=True)
    unique_i = non_dominated_i[unique_i]
    duplicates = np.ones(len(l2), dtype=bool)
    duplicates[unique_i] = False
    l2[duplicates] -= 1e-5
    l2[sma] *= 2
 
    return l2

