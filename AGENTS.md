# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the analysis
python btc_returns_analysis.py
```

## Architecture

Single-file quantitative analysis tool (`btc_returns_analysis.py`) with a linear data pipeline:

1. **Data Fetching** (`get_data`) — Downloads OHLCV from Yahoo Finance via `yfinance`
2. **Return Calculation** (`calculate_returns`) — Adds columns: `daily_return`, `log_return`, `cumulative_return`, `drawdown`, `rolling_vol_30d`
3. **Statistics** (`compute_stats`) — Computes performance metrics (Sharpe, Sortino, CAGR, etc.)
4. **Visualization** (`plot_analysis`) — 4-panel matplotlib chart saved to `btc_analysis.png`
5. **Output** (`print_stats`) — Prints formatted stats table to console

## Key Conventions

- Crypto assets use 365 trading days for annualization (not 252 like equities)
- The `risk_free_rate` parameter defaults to 0.05 (5%)
- Ticker symbols follow Yahoo Finance format: `BTC-USD`, `ETH-USD`, `SPY`
- All return calculations operate on the `close` column (adjusted close prices)
