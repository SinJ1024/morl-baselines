import numpy as np

def get_nash_score(solutions: np.ndarray):
    scores = np.zeros(solutions.shape[0], dtype=float)
    for i, c in enumerate(solutions):
        scores[i] = solutions.prod()

    return scores

def nash_l2( returns, sma, params ):
    scores = get_nash_score(returns)

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

