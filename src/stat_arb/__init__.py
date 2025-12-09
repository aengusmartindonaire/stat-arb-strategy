"""
Statistical Arbitrage Strategy Development package.

This bundles the main building blocks:

- config:     config loading, paths
- data_pipeline: universe + returns + risk-free handling
- factors:    beta / hedge-factor utilities
- strategies: signal construction (reversal, momentum, hedged)
- backtest:   generic backtest helpers (you can extend later)
- performance: IC / trade-level summary utilities
- plotting:   standard plots for performance diagnostics
"""

from . import config
from . import data_pipeline
from . import factors
from . import strategies
from . import backtest
from . import performance
from . import plotting

__all__ = [
    "config",
    "data_pipeline",
    "factors",
    "strategies",
    "backtest",
    "performance",
    "plotting",
]

__version__ = "0.1.0"