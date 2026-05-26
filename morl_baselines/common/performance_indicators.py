"""Performance indicators for multi-objective RL algorithms.

We mostly rely on pymoo for the computation of axiomatic indicators (HV and IGD), but some are customly made.
"""

from copy import deepcopy
from typing import Callable, List

import numpy as np
import numpy.typing as npt
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD


def hypervolume(ref_point: np.ndarray, points: List[npt.ArrayLike]) -> float:
    """Computes the hypervolume metric for a set of points (value vectors) and a reference point (from Pymoo).

    Args:
        ref_point (np.ndarray): Reference point
        points (List[np.ndarray]): List of value vectors

    Returns:
        float: Hypervolume metric
    """
    return HV(ref_point=ref_point * -1)(np.array(points) * -1)


def igd(known_front: List[np.ndarray], current_estimate: List[np.ndarray]) -> float:
    """Inverted generational distance metric. Requires to know the optimal front.

    Args:
        known_front: known pareto front for the problem
        current_estimate: current pareto front

    Return:
        a float stating the average distance between a point in current_estimate and its nearest point in known_front
    """
    ind = IGD(np.array(known_front))
    return ind(np.array(current_estimate))


def sparsity(front: List[np.ndarray]) -> float:
    """Sparsity metric from PGMORL.

    (!) This metric only considers the points from the PF identified by the algorithm, not the full objective space.
    Therefore, it is misleading (e.g. learning only one point is considered good) and we recommend not using it when comparing algorithms.

    Basically, the sparsity is the average distance between each point in the front.

    Args:
        front: current pareto front to compute the sparsity on

    Returns:
        float: sparsity metric
    """
    if len(front) < 2:
        return 0.0

    sparsity_value = 0.0
    m = len(front[0])
    front = np.array(front)
    for dim in range(m):
        objs_i = np.sort(deepcopy(front.T[dim]))
        for i in range(1, len(objs_i)):
            sparsity_value += np.square(objs_i[i] - objs_i[i - 1])
    sparsity_value /= len(front) - 1

    return sparsity_value


def expected_utility(front: List[np.ndarray], weights_set: List[np.ndarray], utility: Callable = np.dot) -> float:
    """Expected Utility Metric.

    Expected utility of the policies on the PF for various weights.
    Similar to R-Metrics in MOO. But only needs one PF approximation.
    Paper: L. M. Zintgraf, T. V. Kanters, D. M. Roijers, F. A. Oliehoek, and P. Beau, “Quality Assessment of MORL Algorithms: A Utility-Based Approach,” 2015.

    Args:
        front: current pareto front to compute the eum on
        weights_set: weights to use for the utility computation
        utility: utility function to use (default: dot product)

    Returns:
        float: eum metric
    """
    maxs = []
    for weights in weights_set:
        scalarized_front = np.array([utility(weights, point) for point in front])
        maxs.append(np.max(scalarized_front))

    return np.mean(np.array(maxs), axis=0)


def cardinality(front: List[np.ndarray]) -> float:
    """Cardinality Metric.

    Cardinality of the Pareto front approximation.

    Args:
        front: current pareto front to compute the cardinality on

    Returns:
        float: cardinality metric
    """
    return len(front)


def maximum_utility_loss(
    front: List[np.ndarray], reference_set: List[np.ndarray], weights_set: np.ndarray, utility: Callable = np.dot
) -> float:
    """Maximum Utility Loss Metric.

    Maximum utility loss of the policies on the PF for various weights.
    Paper: L. M. Zintgraf, T. V. Kanters, D. M. Roijers, F. A. Oliehoek, and P. Beau, “Quality Assessment of MORL Algorithms: A Utility-Based Approach,” 2015.

    Args:
        front: current pareto front to compute the mul on
        reference_set: reference set (e.g. true Pareto front) to compute the mul on
        weights_set: weights to use for the utility computation
        utility: utility function to use (default: dot product)

    Returns:
        float: mul metric
    """
    max_scalarized_values_ref = [np.max([utility(weight, point) for point in reference_set]) for weight in weights_set]
    max_scalarized_values = [np.max([utility(weight, point) for point in front]) for weight in weights_set]
    utility_losses = [max_scalarized_values_ref[i] - max_scalarized_values[i] for i in range(len(max_scalarized_values))]
    return np.max(utility_losses)

def gini(x, normalized=True):
    """Compute the Gini index of a given numpy array.
    TODO: make it work for all-dimensional arrays

    Args:
        x (np.array): array of values (e.g. rewards)
        normalized (bool, optional): whether to normalize the Gini index. Defaults to True.

    Returns:
        float: Gini index
    """
    sorted_x = np.sort(x, axis=1)
    n = x.shape[1]
    cum_x = np.cumsum(sorted_x, axis=1, dtype=float)
    gi = (n + 1 - 2 * np.sum(cum_x, axis=1) / cum_x[:, -1]) / n
    if normalized:
        gi = gi * (n / (n - 1))
    return gi


def max_min_satisfaction_floor(cell_satisfaction_rates: np.ndarray, cell_demands: np.ndarray) -> np.ndarray:
    """Rawlsian Max-Min Satisfaction Floor: min satisfaction rate across cells with nonzero demand.

    Args:
        cell_satisfaction_rates: shape (n_lines, grid_size) — per-cell satisfaction per evaluated line.
        cell_demands: shape (grid_size,) — per-cell total demand.

    Returns:
        np.ndarray: shape (n_lines,) — floor metric for each evaluated line.
    """
    has_demand = cell_demands > 0
    if not np.any(has_demand):
        return np.zeros(cell_satisfaction_rates.shape[0])
    return np.min(cell_satisfaction_rates[:, has_demand], axis=1)


def spatial_sen_welfare(
    cell_satisfaction_rates: np.ndarray,
    cell_demands: np.ndarray,
    agg_od_by_cell: np.ndarray,
) -> tuple:
    """Spatial Sen Welfare: split cells into high/low demand regions, compute SW per region.

    SW_k = E_k * (1 - G_k) where E_k = sum of satisfied demand, G_k = Gini of satisfaction rates.

    Args:
        cell_satisfaction_rates: shape (n_lines, grid_size).
        cell_demands: shape (grid_size,).
        agg_od_by_cell: shape (grid_size,) — aggregated OD demand per cell for region splitting.

    Returns:
        (sw_high, sw_low): each shape (n_lines,).
    """
    has_demand = cell_demands > 0
    if not np.any(has_demand):
        n = cell_satisfaction_rates.shape[0]
        return np.zeros(n), np.zeros(n)

    threshold = np.median(agg_od_by_cell[has_demand])
    high_mask = has_demand & (agg_od_by_cell >= threshold)
    low_mask = has_demand & (agg_od_by_cell < threshold)

    def _region_welfare(mask):
        n_lines = cell_satisfaction_rates.shape[0]
        n_cells = np.sum(mask)
        if n_cells < 2:
            satisfied = np.sum(cell_satisfaction_rates[:, mask] * cell_demands[mask], axis=1)
            return satisfied

        region_sr = cell_satisfaction_rates[:, mask]
        satisfied = np.sum(region_sr * cell_demands[mask], axis=1)
        gi = gini(region_sr, normalized=True)
        gi = np.clip(gi, 0.0, 1.0)
        return satisfied * (1 - gi)

    return _region_welfare(high_mask), _region_welfare(low_mask)
