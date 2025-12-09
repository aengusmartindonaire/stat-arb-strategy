from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curve(
    returns: pd.Series,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
) -> plt.Axes:
    """Plot cumulative equity curve from a return series."""
    if ax is None:
        _, ax = plt.subplots()

    equity = (1 + returns.fillna(0.0)).cumprod()
    ax.plot(equity.index, equity.values, label=label or "Strategy")

    ax.set_title("Equity Curve")
    ax.set_ylabel("Cumulative Value")
    ax.set_xlabel("Date")

    if label is not None:
        ax.legend()

    return ax


def plot_ic_timeseries(
    ic: pd.Series,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot time series of rank-IC."""
    if ax is None:
        _, ax = plt.subplots()

    ic = ic.dropna()
    ax.plot(ic.index, ic.values)
    ax.axhline(0.0, linestyle="--")

    ax.set_title("Rank-IC over time")
    ax.set_ylabel("IC")
    ax.set_xlabel("Date")

    return ax


def plot_ic_by_year(
    ic: pd.Series,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Bar chart of average rank-IC by calendar year."""
    if ax is None:
        _, ax = plt.subplots()

    ic = ic.dropna()
    if ic.empty:
        return ax

    year_ic = ic.groupby(ic.index.year).mean()
    ax.bar(year_ic.index.astype(str), year_ic.values)

    ax.set_title("Average Rank-IC by Year")
    ax.set_ylabel("Mean IC")
    ax.set_xlabel("Year")

    return ax