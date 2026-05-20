import numpy as np
from morl_baselines.multi_policy.gcn.pareto import pareto_l2
from morl_baselines.multi_policy.gcn.pareto import get_non_dominated_inds
from morl_baselines.common.performance_indicators import gini

def lorenz_vector(points, proportional=False):
    """Compute the Lorenz vector of a set of points."""
    # sort points per dimension
    sorted = np.sort(points, axis=1)
    lv = np.cumsum(sorted, axis=1)
    if proportional:
        lv = lv / np.sum(points, axis=1, keepdims=True)
 
    return lv

def lorenz_l2(returns, sma, params):
    distance_ref = params['distance_ref']
    lcn_lambda = params['lcn_lambda']

    if distance_ref == 'nondominated':
        lv = lorenz_vector(np.array(returns))
        return pareto_l2(lv, sma, None)

    elif distance_ref == 'nondominated_mean':
        lv = lorenz_vector(np.array(returns))
        non_dominated_i = get_non_dominated_inds(lv)
        non_dominated = returns[non_dominated_i]

        optimal = np.full_like(returns, non_dominated.mean(axis=0))
        l2 = np.linalg.norm(returns - optimal, axis=-1) * -1

        # all points that are too close together (crowding distance < threshold) get a penalty
        non_dominated_i = np.nonzero(non_dominated_i)[0]
        _, unique_i = np.unique(non_dominated, axis=0, return_index=True)
        unique_i = non_dominated_i[unique_i]
        duplicates = np.ones(len(l2), dtype=bool)
        duplicates[unique_i] = False
        l2[duplicates] -= 1e-5
        l2[sma] *= 2

    elif distance_ref == 'optimal_max':
        optimal = np.full_like(returns, returns[returns.sum(axis=1).argmax()].mean())
        l2 = np.linalg.norm(returns - optimal, axis=-1) * -1

        # all points that are too close together (crowding distance < threshold) get a penalty
        _, unique_i = np.unique(returns, axis=0, return_index=True)
        duplicates = np.ones(len(l2), dtype=bool)
        duplicates[unique_i] = False
        l2[duplicates] -= 1e-5
        l2[sma] *= 2
    elif distance_ref == 'interpolate':
        assert lcn_lambda is not None, "lcn_lambda must be set when using distance_ref='interpolate'"
        non_dominated_i = get_non_dominated_inds(returns)
        non_dominated = returns[non_dominated_i]
        ginis = gini(non_dominated, normalized=True)
        # Filter out the ND points whose gini is > lamda (or the min gini)
        non_dominated_i = ginis <= lcn_lambda
        # If no solution is left after filtering, take the ones with the lowest gini
        if sum(non_dominated_i) == 0:
            threshold = np.min(ginis)
            non_dominated_i = ginis <= threshold
        non_dominated = non_dominated[non_dominated_i]

        # we will compute distance of each point with each non-dominated point,
        # duplicate each point with number of non_dominated to compute respective distance
        returns_exp = np.tile(np.expand_dims(returns, 1), (1, len(non_dominated), 1))
        # distance to closest non_dominated point
        l2 = np.min(np.linalg.norm(returns_exp - non_dominated, axis=-1), axis=-1) * -1

        # all points that are too close together (crowding distance < cd_threshold) get a penalty
        non_dominated_i = np.nonzero(non_dominated_i)[0]
        _, unique_i = np.unique(non_dominated, axis=0, return_index=True)
        unique_i = non_dominated_i[unique_i]
        duplicates = np.ones(len(l2), dtype=bool)
        duplicates[unique_i] = False
        l2[duplicates] -= 1e-5
        l2[sma] *= 2
    elif distance_ref == 'interpolate2':
        assert lcn_lambda is not None, "lcn_lambda must be set when using distance_ref='interpolate2'"

        lv = lorenz_vector(np.array(returns))
        # The final vector is a weighted average of the lorenz vector and the full returns
        fv = lcn_lambda * returns + (1 - lcn_lambda) * lv

        non_dominated_i = get_non_dominated_inds(fv)
        non_dominated = returns[non_dominated_i]

        # we will compute distance of each point with each non-dominated point,
        # duplicate each point with number of non_dominated to compute respective distance
        returns_exp = np.tile(np.expand_dims(returns, 1), (1, len(non_dominated), 1))
        # distance to closest non_dominated point
        l2 = np.min(np.linalg.norm(returns_exp - non_dominated, axis=-1), axis=-1) * -1

        # all points that are too close together (crowding distance < cd_threshold) get a penalty
        non_dominated_i = np.nonzero(non_dominated_i)[0]
        _, unique_i = np.unique(non_dominated, axis=0, return_index=True)
        unique_i = non_dominated_i[unique_i]
        duplicates = np.ones(len(l2), dtype=bool)
        duplicates[unique_i] = False
        l2[duplicates] -= 1e-5
        l2[sma] *= 2
    elif distance_ref == 'interpolate3':
        assert lcn_lambda is not None, "lcn_lambda must be set when using distance_ref='interpolate2'"

        # sort returns in increasing order
        returns = np.sort(returns, axis=1)

        lv = lorenz_vector(np.array(returns))
        # The final vector is a weighted average of the lorenz vector and the full returns
        fv = lcn_lambda * returns + (1 - lcn_lambda) * lv

        non_dominated_i = get_non_dominated_inds(fv)
        non_dominated = returns[non_dominated_i]

        # we will compute distance of each point with each non-dominated point,
        # duplicate each point with number of non_dominated to compute respective distance
        returns_exp = np.tile(np.expand_dims(returns, 1), (1, len(non_dominated), 1))
        # distance to closest non_dominated point
        l2 = np.min(np.linalg.norm(returns_exp - non_dominated, axis=-1), axis=-1) * -1

        # all points that are too close together (crowding distance < cd_threshold) get a penalty
        non_dominated_i = np.nonzero(non_dominated_i)[0]
        _, unique_i = np.unique(non_dominated, axis=0, return_index=True)
        unique_i = non_dominated_i[unique_i]
        duplicates = np.ones(len(l2), dtype=bool)
        duplicates[unique_i] = False
        l2[duplicates] -= 1e-5
        l2[sma] *= 2

    return l2
