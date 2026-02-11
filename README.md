# BTC Price & Returns Analysis

A beginner-friendly quantitative finance project for analyzing Bitcoin price data and returns.

## Features

1. **Pull Real BTC Price Data** - Downloads historical OHLCV data from Yahoo Finance
2. **Calculate Daily Returns** - Simple and log returns, cumulative returns
3. **Visualize Price & Returns** - 4-panel analysis chart
4. **Analyze Return Distribution** - Compare actual returns vs normal distribution
5. **Rolling Volatility** - 30-day annualized volatility tracking
6. **Key Performance Stats** - Sharpe Ratio, Sortino Ratio, Max Drawdown, Win Rate, CAGR

## Installation

```bash
pip install yfinance pandas numpy matplotlib scipy
```

Or using the requirements file:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python btc_returns_analysis.py
```

## Output

The script will:
- Print a performance summary table with key metrics
- Generate a 4-panel visualization saved as `btc_analysis.png`

## Customization

Edit the constants in the `__main__` block to analyze different assets:

```python
TICKER = 'BTC-USD'   # Try 'ETH-USD', 'SPY', 'ES=F', etc.
PERIOD = '2y'        # Options: '1y', '2y', '5y', 'max'
```

## Metrics Explained

| Metric | Description |
|--------|-------------|
| Sharpe Ratio | Return per unit of risk (>1 is good, >2 is excellent) |
| Sortino Ratio | Like Sharpe but only penalizes downside volatility |
| Max Drawdown | Worst peak-to-trough decline |
| Win Rate | Percentage of days with positive returns |
| CAGR | Compound Annual Growth Rate |
| Skewness | Negative = more crash risk |
| Kurtosis | Positive = fat tails (more extreme days) |
