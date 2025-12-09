from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, LinearRegression


def _prepare_xy(
    stock_returns: pd.Series,
    market_returns: pd.Series,
) -> Tuple[np.ndarray, np.ndarray]:
    """Align two return series and return (X, y) arrays for regression."""
    joined = pd.concat(
        [stock_returns.rename("stock"), market_returns.rename("market")],
        axis=1,
    ).dropna()

    x = joined["market"].to_numpy().reshape(-1, 1)
    y = joined["stock"].to_numpy()
    return x, y


def calculate_beta_ols(
    stock_returns: pd.Series,
    market_returns: pd.Series,
) -> float:
    """Plain OLS beta: stock = alpha + beta * market + epsilon."""
    x, y = _prepare_xy(stock_returns, market_returns)
    if len(y) < 5:
        raise ValueError("Not enough overlapping observations to estimate beta.")
    model = LinearRegression()
    model.fit(x, y)
    return float(model.coef_[0])


def calculate_beta_huber(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    epsilon: float = 1.35,
) -> float:
    """Robust beta using Huber regression."""
    x, y = _prepare_xy(stock_returns, market_returns)
    if len(y) < 5:
        raise ValueError("Not enough overlapping observations to estimate beta.")
    model = HuberRegressor(epsilon=epsilon)
    model.fit(x, y)
    return float(model.coef_[0])


def calculate_beta_clip_pct10(
    stock_returns: pd.Series,
    market_returns: pd.Series,
) -> float:
    """
    Beta estimated after clipping stock and market returns at 10th / 90th percentiles.
    """
    x_raw, y_raw = _prepare_xy(stock_returns, market_returns)

    x = x_raw.copy()
    y = y_raw.copy()

    for arr in (x, y):
        lo = np.nanpercentile(arr, 10)
        hi = np.nanpercentile(arr, 90)
        np.clip(arr, lo, hi, out=arr)

    model = LinearRegression()
    model.fit(x, y)
    return float(model.coef_[0])