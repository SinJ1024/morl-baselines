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
    x = np.asarray(x, dtype=float)
    sorted_x = np.sort(x, axis=1)
    n = x.shape[1]
    cum_x = np.cumsum(sorted_x, axis=1, dtype=float)
    total = cum_x[:, -1]
    # guard against all-zero rows (total == 0): gini is undefined -> treat as 0 (no inequality)
    safe_total = np.where(total == 0, 1.0, total)
    gi = (n + 1 - 2 * np.sum(cum_x, axis=1) / safe_total) / n
    gi = np.where(total == 0, 0.0, gi)
    if normalized and n > 1:
        gi = gi * (n / (n - 1))
    return gi


def served_floor(cell_satisfaction_rates: np.ndarray, cell_demands: np.ndarray) -> np.ndarray:
    """Worst service quality among cells the line actually reaches.

    min satisfaction among demand cells with satisfaction > 0. Unlike a Rawlsian
    floor over ALL demand cells (which is always 0 when a single 20-station line
    covers only ~3% of cells), this conditions on served cells so it is non-degenerate.

    NOTE: must be read together with demand_coverage — a route serving a single cell
    perfectly scores 1.0 here. High served_floor + high demand_coverage = genuinely good.

    Args:
        cell_satisfaction_rates: shape (n_lines, grid_size).
        cell_demands: shape (grid_size,).

    Returns:
        np.ndarray: shape (n_lines,) — served floor for each line.
    """
    has_demand = cell_demands > 0
    n_lines = cell_satisfaction_rates.shape[0]
    result = np.zeros(n_lines)
    for i in range(n_lines):
        served_mask = has_demand & (cell_satisfaction_rates[i] > 0)
        if np.any(served_mask):
            result[i] = np.min(cell_satisfaction_rates[i, served_mask])
    return result


def demand_coverage(cell_satisfaction_rates: np.ndarray, cell_demands: np.ndarray) -> np.ndarray:
    """Fraction of total demand VOLUME that is served (demand-weighted reach).

    Unlike a raw cell-count reach metric (~pinned by the station budget), this weights
    by demand so reaching a high-demand cell counts more than a low-demand one.

    Args:
        cell_satisfaction_rates: shape (n_lines, grid_size).
        cell_demands: shape (grid_size,).

    Returns:
        np.ndarray: shape (n_lines,) — served demand fraction in [0, 1] for each line.
    """
    has_demand = cell_demands > 0
    total_demand = cell_demands[has_demand].sum()
    if total_demand == 0:
        return np.zeros(cell_satisfaction_rates.shape[0])
    served_demand = np.sum(
        (cell_satisfaction_rates[:, has_demand] > 0).astype(float) * cell_demands[has_demand],
        axis=1,
    )
    return served_demand / total_demand


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
    n_lines = cell_satisfaction_rates.shape[0]

    def _region_welfare(mask):
        if np.sum(mask) == 0:
            return np.zeros(n_lines)
        region_sr = cell_satisfaction_rates[:, mask]          # (n_lines, n_region_cells)
        w = cell_demands[mask]
        total_w = w.sum()
        # demand-weighted MEAN satisfaction in region, normalized to [0, 1]
        # (so high/low regions are comparable regardless of their demand mass)
        mean_sat = (region_sr * w).sum(axis=1) / (total_w + 1e-12)
        welfare = np.zeros(n_lines)
        for i in range(n_lines):
            served = region_sr[i] > 0
            # inequality among SERVED cells only — avoids gini saturating to 1 on the
            # mostly-zero (unserved) cells, which previously crushed welfare to ~0
            if served.sum() >= 2:
                gi = float(gini(region_sr[i][served][None, :], normalized=True)[0])
                gi = np.clip(gi, 0.0, 1.0)
            else:
                gi = 0.0
            welfare[i] = mean_sat[i] * (1.0 - gi)
        return welfare

    return _region_welfare(high_mask), _region_welfare(low_mask)


def price_equity(
    cell_satisfaction_rates: np.ndarray,
    cell_demands: np.ndarray,
    house_prices: np.ndarray,
) -> tuple:
    """Housing-price equity: transit service in affordable vs. expensive areas.

    Splits cells (with both demand and price data) into low/high price by median.
    For each group computes demand-weighted MEAN satisfaction (in [0, 1]).
    equity_ratio = sat_low / sat_high  (>1 = pro-affordable, the fair direction).

    Args:
        cell_satisfaction_rates: shape (n_lines, grid_size).
        cell_demands: shape (grid_size,).
        house_prices: shape (grid_size,) — 0 = no price data.

    Returns:
        (sat_low, sat_high, equity_ratio): each shape (n_lines,).
    """
    n_lines = cell_satisfaction_rates.shape[0]
    zeros = np.zeros(n_lines)

    valid = (cell_demands > 0) & (house_prices > 0)
    if np.sum(valid) < 4:
        return zeros, zeros, np.ones(n_lines)

    median_price = np.median(house_prices[valid])
    low_price_mask = valid & (house_prices < median_price)
    high_price_mask = valid & (house_prices >= median_price)

    def _weighted_mean_sat(mask):
        if np.sum(mask) == 0:
            return zeros
        weights = cell_demands[mask]
        total_w = np.sum(weights)
        if total_w == 0:
            return zeros
        return np.sum(cell_satisfaction_rates[:, mask] * weights, axis=1) / total_w

    sat_low = _weighted_mean_sat(low_price_mask)
    sat_high = _weighted_mean_sat(high_price_mask)

    # FIX: when a line serves no expensive cells, the ratio is undefined.
    # Set it to 1.0 (neutral) instead of dividing by ~1e-10 (which exploded to ~1e10).
    ratio = np.ones(n_lines)
    nz = sat_high > 0
    ratio[nz] = sat_low[nz] / sat_high[nz]
    return sat_low, sat_high, ratio
