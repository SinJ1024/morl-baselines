import numpy as np
from morl_baselines.common.performance_indicators import gini

######################################################################
# Dominance Functions
######################################################################

def get_non_pareto_dominated(solutions: np.ndarray, params=None):
    is_efficient = np.ones(solutions.shape[0], dtype=bool)
    for i, c in enumerate(solutions):
        if is_efficient[i]:
            # Remove dominated points, will also remove itself
            is_efficient[is_efficient] = np.any(solutions[is_efficient] > c, axis=1)
            # keep this solution as non-dominated
            is_efficient[i] = 1

    return is_efficient, solutions[is_efficient]

def get_nash_score(solutions: np.ndarray, shift=0.0, epsilon=1e-10):
    shifted = solutions + shift
    return np.mean(np.log(np.maximum(shifted, epsilon)), axis=1)

def get_nash_dominated(solutions: np.ndarray, params):
    """
    """
    params = params or {}
    mode = params.get('mode', 'pareto_filter')
    shift = params.get('shift', 0.0)
    epsilon = params.get('epsilon', 1e-10)

    scores = get_nash_score(solutions, shift=shift, epsilon=epsilon)
    pareto_mask, _ = get_non_pareto_dominated(solutions)
    pareto_size = int(pareto_mask.sum())

    if mode == 'pareto_filter':
        top_k = int(params.get('top_k', pareto_size))
        top_k = max(1, min(top_k, pareto_size))
        pf_indices = np.flatnonzero(pareto_mask)
        pf_scores = scores[pareto_mask]
        if top_k < len(pf_scores):
            pf_top_k = np.argpartition(pf_scores, -top_k)[-top_k:]
            pf_indices = pf_indices[pf_top_k]
 
    elif mode == 'pareto_sized':
        if pareto_size >= len(scores):
            mask = np.ones(len(scores), dtype=bool)
            return mask, solutions[mask]
        pf_indices = np.argpartition(scores, -pareto_size)[-pareto_size:]

    mask = np.zeros(len(solutions), dtype=bool)
    mask[pf_indices] = True
    return mask, solutions[mask]

######################################################################
# Utility Functions
######################################################################

def lorenz_vector(points, proportional=False):
    """Compute the Lorenz vector of a set of points."""
    # sort points per dimension
    sorted = np.sort(points, axis=1)
    lv = np.cumsum(sorted, axis=1)
    if proportional:
        lv = lv / np.sum(points, axis=1, keepdims=True)
 
    return lv
def penalize_crowding(non_dominated_i, non_dominated, l2, sma):
    """Penalize crowding if points by setting a penalty to all points that are too close together (crowding distance < threshold)
    Args:
        non_dominated_i: the indices of non-dominated strategies in the original front tensor
        non_dominated: the vectors of non-dominated strategies
        l2: the actual l2 norm
        sma: indices of crowded points found earlier through some crowding distance function
    """
    non_dominated_i = np.nonzero(non_dominated_i)[0]
    _, unique_i = np.unique(non_dominated, axis=0, return_index=True)
    unique_i = non_dominated_i[unique_i]
    duplicates = np.ones(len(l2), dtype=bool)
    duplicates[unique_i] = False
    l2[duplicates] -= 1e-5
    l2[sma] *= 2
    return l2

######################################################################
# L2 distance functions
######################################################################

def pareto_l2( returns, sma, params ):
    non_dominated_i, non_dominated = get_non_pareto_dominated(returns)

    # we will compute distance of each point with each non-dominated point,
    # duplicate each point with number of non_dominated to compute respective distance
    returns_exp = np.tile(np.expand_dims(returns, 1), (1, len(non_dominated), 1))
    # distance to closest non_dominated point
    l2 = np.min(np.linalg.norm(returns_exp - non_dominated, axis=-1), axis=-1) * -1

    l2 = penalize_crowding(non_dominated_i, non_dominated, l2, sma)
    return l2

def lorenz_l2( returns, sma, params ):
    distance_ref = params['distance_ref']
    lcn_lambda = params['lcn_lambda']

    if distance_ref == 'nondominated':
        lv = lorenz_vector(np.array(returns))
        return pareto_l2(lv, sma, None)

    if distance_ref == 'nondominated_mean':
        lv = lorenz_vector(np.array(returns))
        non_dominated_i, _ = get_non_pareto_dominated(lv)
        non_dominated = returns[non_dominated_i]

        optimal = np.full_like(returns, non_dominated.mean(axis=0))
        l2 = np.linalg.norm(returns - optimal, axis=-1) * -1

        l2 = penalize_crowding( non_dominated_i, non_dominated, l2, sma )
        return l2

    if distance_ref == 'optimal_max':
        optimal = np.full_like(returns, returns[returns.sum(axis=1).argmax()].mean())
        l2 = np.linalg.norm(returns - optimal, axis=-1) * -1

        # all points that are too close together (crowding distance < threshold) get a penalty
        _, unique_i = np.unique(returns, axis=0, return_index=True)
        duplicates = np.ones(len(l2), dtype=bool)
        duplicates[unique_i] = False
        l2[duplicates] -= 1e-5
        l2[sma] *= 2
        return l2
 
    spatial_alpha = params.get('spatial_alpha', 0.0)
    route_contexts = params.get('route_contexts', None)
    assert lcn_lambda is not None, "lcn_lambda must be set when using distance_ref='interpolate(2/3)'"
    if distance_ref == 'interpolate':
        lv = lorenz_vector(np.array(returns))
        non_dominated_i, _ = get_non_pareto_dominated(lv)
        non_dominated = returns[non_dominated_i]
        ginis = gini(non_dominated, normalized=True)
        # If no solution is left after filtering, take the ones with the lowest gini
        if spatial_alpha > 0 and route_contexts is not None:
            nd_contexts = route_contexts[np.nonzero(non_dominated_i)[0]]
            effective_lambdas_nd = np.maximum(lcn_lambda, spatial_alpha * nd_contexts)
            non_dominated_i = ginis <= effective_lambdas_nd
        else:
            non_dominated_i = ginis <= lcn_lambda

        if non_dominated_i.sum() == 0:
            threshold = np.min(ginis)
            non_dominated_i = ginis <= threshold

        non_dominated = non_dominated[non_dominated_i]
        non_dominated_i = np.zeros(len(returns), dtype=bool)
        non_dominated_i[]

    elif distance_ref == 'interpolate2':
        lv = lorenz_vector(np.array(returns))

        if spatial_alpha > 0 and route_contexts is not None:
            effective_lambdas = np.maximum(lcn_lambda, spatial_alpha * route_contexts)
            fv = effective_lambdas[:, np.newaxis] * returns + (1 - effective_lambdas[:, np.newaxis]) * lv
        else:
            fv = lcn_lambda * returns + (1 - lcn_lambda) * lv

        non_dominated_i, non_dominated = get_non_pareto_dominated(fv)

    elif distance_ref == 'interpolate3':
        # sort returns in increasing order
        returns = np.sort(returns, axis=1)

        lv = lorenz_vector(np.array(returns))
        # The final vector is a weighted average of the lorenz vector and the full returns
        if spatial_alpha > 0 and route_contexts is not None:
            effective_lambdas = np.maximum(lcn_lambda, spatial_alpha * route_contexts)
            fv = effective_lambdas[:, np.newaxis] * returns + (1 - effective_lambdas[:, np.newaxis]) * lv
        else:
            fv = lcn_lambda * returns + (1 - lcn_lambda) * lv

        non_dominated_i, non_dominated = get_non_pareto_dominated(fv)

    # we will compute distance of each point with each non-dominated point,
    # duplicate each point with number of non_dominated to compute respective distance
    returns_exp = np.tile(np.expand_dims(returns, 1), (1, len(non_dominated), 1))

    # distance to closest non_dominated point
    l2 = np.min(np.linalg.norm(returns_exp - non_dominated, axis=-1), axis=-1) * -1
    l2 = penalize_crowding( non_dominated_i, non_dominated, l2, sma )

    return l2

def nash_l2( returns, sma, params ):
    non_dominated_i, non_dominated = get_nash_dominated(returns, params)

    # we will compute distance of each point with each non-dominated point,
    # duplicate each point with number of non_dominated to compute respective distance
    returns_exp = np.tile(np.expand_dims(returns, 1), (1, len(non_dominated), 1))
    # distance to closest non_dominated point
    l2 = np.min(np.linalg.norm(returns_exp - non_dominated, axis=-1), axis=-1) * -1

    l2 = penalize_crowding( non_dominated_i, non_dominated, l2, sma )
 
    return l2

