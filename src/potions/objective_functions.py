"""Objective functions for evaluating model outputs against measurements.

Faithful port of the reference ``objective_functions.py`` (pandas semantics:
Pearson/Spearman correlations and ddof-1 standard deviations agree exactly with
the golden Rust-backed runs).  Every function takes measured and simulated
``pandas.Series`` and returns a scalar ``float``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from pandas import Series


def kge(y_obs: Series, y_pred: Series) -> float:
    """Kling-Gupta efficiency.

    ``1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)`` with ``alpha =
    std(sim)/std(obs)`` and ``beta = mean(sim)/mean(obs)``.
    """
    r: float = float(y_pred.corr(y_obs, method="pearson"))
    alpha: float = float(y_pred.std() / y_obs.std())
    beta: float = float(y_pred.mean() / y_obs.mean())
    return 1.0 - ((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2) ** 0.5


def log_kge(y_obs: Series, y_pred: Series) -> float:
    """Kling-Gupta efficiency of the log10(x + 0.1)-transformed data."""
    y_obs = y_obs.map(lambda x: np.log10(x + 0.1))
    y_pred = y_pred.map(lambda x: np.log10(x + 0.1))
    r: float = float(y_pred.corr(y_obs, method="pearson"))
    alpha: float = float(y_pred.std() / y_obs.std())
    beta: float = float(y_pred.mean() / y_obs.mean())
    return 1.0 - ((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2) ** 0.5


def nse(y_obs: Series, y_pred: Series) -> float:
    """Nash-Sutcliffe efficiency."""
    numer: float = float(((y_pred - y_obs) ** 2).sum())
    denom: float = float(((y_obs - y_obs.mean()) ** 2).sum())
    return 1.0 - numer / denom


def log_nse(y_obs: Series, y_pred: Series) -> float:
    """Nash-Sutcliffe efficiency of the log10(x + 0.1)-transformed data."""
    y_obs = y_obs.map(lambda x: np.log10(x + 0.1))
    y_pred = y_pred.map(lambda x: np.log10(x + 0.1))
    numer: float = float(((y_pred - y_obs) ** 2).sum())
    denom: float = float(((y_obs - y_obs.mean()) ** 2).sum())
    return 1.0 - numer / denom


def objective_low_flow(
    obj_func: Callable[[Series, Series], float],
    low_flow_pctl: float,
    meas: Series,
    sim: Series,
) -> float:
    """Evaluate ``obj_func`` on only the bottom ``low_flow_pctl`` of the data."""
    if not (0 < low_flow_pctl <= 1.0):
        raise ValueError(f"Percentile must be between 0 and 1, not {low_flow_pctl}")
    thresh: float = float(meas.quantile(low_flow_pctl))
    mask: Series = meas <= thresh
    return obj_func(meas[mask], sim[mask])


def objective_high_flow(
    obj_func: Callable[[Series, Series], float],
    high_flow_pctl: float,
    meas: Series,
    sim: Series,
) -> float:
    """Evaluate ``obj_func`` on only the top ``high_flow_pctl`` of the data."""
    if not (0 < high_flow_pctl <= 1.0):
        raise ValueError(f"Percentile must be between 0 and 1, not {high_flow_pctl}")
    thresh: float = float(meas.quantile(high_flow_pctl))
    mask: Series = meas >= thresh
    return obj_func(meas[mask], sim[mask])


def bias(y_obs: Series, y_pred: Series) -> float:
    """Mean relative bias of the model: ``mean(sim - obs) / mean(obs)``."""
    return float((y_pred - y_obs).mean() / y_obs.mean())


def r_squared(y_obs: Series, y_pred: Series) -> float:
    """Pearson correlation squared."""
    return float(y_obs.corr(y_pred, method="pearson") ** 2)


def spearman_rho(y_obs: Series, y_pred: Series) -> float:
    """Spearman rank correlation."""
    return float(y_obs.corr(y_pred, method="spearman"))


DEFAULT_OBJECTIVE_FUNCTIONS: list[tuple[str, Callable[[Series, Series], float]]] = [
    ("kge", kge),
    ("log_kge", log_kge),
    ("nse", nse),
    ("log_nse", log_nse),
    ("bias", bias),
    ("spearman_rho", spearman_rho),
    ("r_squared", r_squared),
]