# Statistical Arbitrage Strategy Framework

![Status](https://img.shields.io/badge/Status-In-Progress-blue)
![Python](https://img.shields.io/badge/Python-3.11-green)
![Focus](https://img.shields.io/badge/Focus-StatArb-orange)

## Overview

This repository implements a modular, event-driven backtesting framework for US Equities Statistical Arbitrage. The project is structured as a Python package (`stat_arb`) designed to handle the end-to-end quantitative research workflow: from survivorship-bias-free data ingestion to factor-neutral signal generation and performance attribution.

The current implementation benchmarks three distinct strategies over a 10-year horizon (2016–2026), focusing on the efficacy of hedging systematic risk (SPY) to isolate idiosyncratic alpha.

## Key Features

* **Custom Data Pipeline**: Ingests Bloomberg annual portfolio files to construct a survivorship-bias-free universe and computes daily excess returns.
* **Factor Hedging Engine**: dynamic calculation of rolling betas against hedge instruments (e.g., SPY) to compute residual returns ($r_{hedged}$).
* **Vectorized Backtester**: High-performance execution using `pandas`/`numpy` for signal generation and rank-based portfolio construction.
* **Performance Analytics**: Automated calculation of Information Coefficients (Rank IC), Sharpe Ratios, and Drawdowns.

## Repository Structure

The core logic is encapsulated in the `src/stat_arb` package, keeping research notebooks lightweight and reproducible.

```text
├── README.md
├── configs
│   ├── factor_hedged.yaml
│   ├── paths.yaml
│   └── unhedged_reversal.yaml
├── environment.yml
├── notebooks
│   ├── 01_data_processing.ipynb
│   ├── 02_unhedged_reversal.ipynb
│   ├── 03_factor_hedged_strategies.ipynb
│   └── 04_performance_analysis.ipynb
├── requirements.txt
├── scripts
│   ├── run_backtest_hedged.py
│   ├── run_backtest_unhedged.py
│   └── run_prepare_data.py
├── src
│   ├── data
│   │   ├── processed
│   │   │   ├── betas_spy.parquet
│   │   │   ├── daily_excess_returns.parquet
│   │   │   ├── spy_hedged_momentum
│   │   │   │   ├── portfolio_returns.parquet
│   │   │   │   ├── rank_ic.parquet
│   │   │   │   └── weights.parquet
│   │   │   ├── spy_hedged_reversal
│   │   │   │   ├── portfolio_returns.parquet
│   │   │   │   ├── rank_ic.parquet
│   │   │   │   └── weights.parquet
│   │   │   └── unhedged_reversal
│   │   │       ├── portfolio_returns.parquet
│   │   │       ├── rank_ic.parquet
│   │   │       └── weights.parquet
│   │   └── raw
│   │       ├── 20200101_US_Port.xlsx
│   │       ├── 20210101_US_Port.xlsx
│   │       ├── 20220101_US_Port.xlsx
│   │       ├── 20230101_US_Port.xlsx
│   │       ├── 20240101_US_Port.xlsx
│   │       └── 20250101_US_Port.xlsx
│   └── stat_arb
│       ├── __init__.py
│       ├── __pycache__
│       │   ├── __init__.cpython-311.pyc
│       │   ├── __init__.cpython-38.pyc
│       │   ├── backtest.cpython-311.pyc
│       │   ├── config.cpython-311.pyc
│       │   ├── config.cpython-38.pyc
│       │   ├── data_pipeline.cpython-311.pyc
│       │   ├── data_pipeline.cpython-38.pyc
│       │   ├── factors.cpython-311.pyc
│       │   ├── performance.cpython-311.pyc
│       │   ├── plotting.cpython-311.pyc
│       │   ├── strategies.cpython-311.pyc
│       │   └── utils.cpython-311.pyc
│       ├── backtest.py
│       ├── config.py
│       ├── data_pipeline.py
│       ├── factors.py
│       ├── performance.py
│       ├── plotting.py
│       ├── strategies.py
│       └── utils.py
└── tests
    ├── __init__.py
    ├── test_backtest.py
    ├── test_data_pipeline.py
    └── test_performance.py
```
---
## Strategies & Methodology

### 1. Unhedged Daily Reversal

A baseline mean-reversion strategy capitalizing on short-term liquidity provision.

- **Signal**: Short the stock if the absolute daily return exceeds a threshold (e.g., 2%).
- **Logic**:  
  $\text{Signal}_t = -\text{Return}_t \text{ if } |\text{Return}_t| > 0.02$.

### 2. SPY-Hedged Reversal

Attempts to isolate idiosyncratic mean reversion by stripping out market beta.

- **Hedged Return**:  
  $R_{\text{hedged}} = R_{\text{stock}} - \beta \times R_{\text{SPY}}$.
- **Signal**: Short the hedged return if it exceeds the threshold.

### 3. SPY-Hedged Residual Momentum (12M–1M)

A cross-sectional momentum strategy applied to residual (idiosyncratic) returns.

- **Lookback**: 252 days (12 months) with a 21-day (1 month) lag to avoid short-term reversals.
- **Ranking**: Long top decile / Short bottom decile of cumulative hedged returns.

---

## Performance Results

The following performance metrics cover the backtest period from **Jan 2016 to Dec 2025**.

| Strategy                | Mean IC | IC Hit Rate | Cumulative Return | Sharpe Ratio | Max Drawdown |
|-------------------------|:-------:|:-----------:|:-----------------:|:------------:|:------------:|
| **Unhedged Reversal**   | 0.0110  |   52.8%     |      -98.2%       |    -1.01     |    -99.6%    |
| **SPY-Hedged Reversal** | 0.0129  |   53.2%     |      -97.5%       |    -0.88     |    -99.5%    |
| **SPY-Hedged Momentum** | **0.0168** | **56.7%** | **+721.8%**      |  **0.48**    |  **-39.1%**  |

Source: [`04_performance_analysis.ipynb`](https://github.com/aengusmartindonaire/stat-arb-strategy/blob/main/notebooks/04_performance_analysis.ipynb).

### Analysis

While the daily reversal strategies (both unhedged and hedged) suffered from alpha decay and negative drift in this regime, the **Factor-Hedged Momentum** strategy demonstrated robust performance, generating a **721% cumulative return** with a positive Sharpe ratio of **0.48**.

---
## Data & Confidentiality

This repository used but currently does not track the following data files:

- `src/data/raw/*.xlsx` – raw Bloomberg portfolio extracts  
- `src/data/processed/*.parquet` – intermediate and backtest outputs

This is due to the following:

- Bloomberg data and many course datasets are subject to licensing and usage restrictions.
- In many classes and research settings, **we are not allowed to publish raw Bloomberg files** in a public repository.
- Even processed files (`.parquet`) may still contain information that should not be shared openly.

Thus, these are not included from this repository.

---

## Getting Started

### Prerequisites

- Python 3.11+
- `pandas`, `numpy`, `matplotlib`, `pyarrow` (see `requirements.txt`)

### Installation

```bash
git clone https://github.com/aengusmartindonaire/stat-arb-strategy.git
pip install -r requirements.txt
```

or

```bash
  conda env create -f environment.yml
  conda activate stat-arb-strategy
  python -m ipykernel install --user --name stat-arb-strategy
```
---
## Acknowledgement

Thanks to Dr. Low for the instructions and dataset.

