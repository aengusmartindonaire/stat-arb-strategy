# Statistical Arbitrage Strategy Framework

![Status](https://img.shields.io/badge/Status-Completed-blue)
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

## Configuration

Strategy parameters are decoupled from the source code and defined in `yaml` files within the `configs/` directory. This allows for easy experimentation without modifying the codebase:

* **`configs/paths.yaml`**: Defines input/output directories and data filenames.
* **Strategy Configs** (e.g., `unhedged_reversal.yaml`):
    * `threshold`: The absolute return hurdle for trade entry (default: 0.02).
    * `top_frac` / `bottom_frac`: Percentile of universe to trade.
    * `lookback_window`: Rolling window for beta or momentum calculations.

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
    ├── test_performance.py
    └── test_scripts_smoke.py  
```

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

*Note:* Performance metrics assume 0bps transaction costs to isolate signal efficacy. Real-world implementation would require accounting for slippage and commission.


## Data & Confidentiality

This repository used but currently does not track the following data files:

- `src/data/raw/*.xlsx` – raw Bloomberg portfolio extracts  
- `src/data/processed/*.parquet` – intermediate and backtest outputs

Please take note of the following: 

* **Universe Definition**: Uses Bloomberg annual portfolio extracts (`.xlsx`) to identify the constituent list for each year, ensuring survivorship bias is minimized by including delisted stocks.
* **Market Data**: Daily adjusted close prices and risk-free rates are fetched dynamically using `yfinance` and FRED (Federal Reserve Economic Data).
* **Confidentiality**: The raw Bloomberg Excel files are **not** included in this repository due to licensing restrictions. Users must provide their own universe files in `src/data/raw/` to replicate the full pipeline.


## Getting Started

### Prerequisites

- Python 3.11+
- Key Libraries: `pandas`, `numpy`, `yfinance` (data), `pyarrow` (storage), `matplotlib` (viz)

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

### Usage & Workflow

The framework is designed to be run sequentially. You can execute the pipeline via the command line scripts or explore interactively using the notebooks.

### 1. Data Preparation
First, generate the survivorship-bias-free data panel. This script ingests the raw Bloomberg universe files (Excel) and fetches daily adjusted close prices via `yfinance`.

```bash
python scripts/run_prepare_data.py
```
Output: `src/data/processed/daily_excess_returns.parquet`

### 2. Run backtest 
Execute the strategy backtests. These scripts load the processed data, generate signals, and compute performance metrics.
```bash
# Run baseline unhedged strategy
python scripts/run_backtest_unhedged.py

# Run SPY-hedged strategy (calculates rolling betas automatically)
python scripts/run_backtest_hedged.py
```

### 3. Analysis
Launch the Jupyter notebooks to visualize equity curves, IC decay, and turnover.

```bash
jupyter lab notebooks/04_performance_analysis.ipynb
```

### Testing

This project includes a comprehensive test suite using `pytest` to validate data pipelines, financial calculations, and full integration.

To run the full suite (including smoke tests for scripts):

```bash
# Add src and scripts to PYTHONPATH to ensure correct imports

PYTHONPATH=src:scripts pytest tests
```



## Acknowledgement

Thanks to Professor Low for the instruction materials, guidance, and dataset.

