"""
=============================================================
  BTC Price & Returns Analysis — Quant Starter Project
=============================================================
  A beginner-friendly quant project that covers:
    1. Pulling real BTC price data
    2. Calculating daily returns
    3. Visualizing price and cumulative returns
    4. Analysing return distribution
    5. Rolling volatility
    6. Key performance stats (Sharpe, Max Drawdown, Win Rate)

  Requirements:
    pip install yfinance pandas numpy matplotlib scipy

  Run:
    python btc_returns_analysis.py
=============================================================
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. PULL DATA
# ─────────────────────────────────────────────

def get_data(ticker='BTC-USD', period='2y', interval='1d'):
    """
    Download historical OHLCV data from Yahoo Finance.

    ticker  : asset symbol  (BTC-USD, ETH-USD, SPY, etc.)
    period  : how far back  ('1y', '2y', '5y', 'max')
    interval: bar size      ('1d', '1wk', '1mo')
    """
    print(f"Downloading {ticker} data ({period}, {interval} bars)...")
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check your ticker symbol.")

    # Keep only Close price and flatten multi-level columns if present
    df = df[['Close']].copy()
    df.columns = ['close']
    df.dropna(inplace=True)

    print(f"  Got {len(df)} rows | {df.index[0].date()} → {df.index[-1].date()}")
    return df


# ─────────────────────────────────────────────
# 2. CALCULATE RETURNS & STATS
# ─────────────────────────────────────────────

def calculate_returns(df):
    """
    Add return columns to the dataframe.

    daily_return     : % change day over day  (the core building block)
    log_return       : ln(P_t / P_{t-1})      (useful for stats — normally distributed)
    cumulative_return: how $1 invested grew over time
    drawdown         : % decline from the rolling peak (risk metric)
    """
    df = df.copy()

    # Simple daily return: (today - yesterday) / yesterday
    df['daily_return'] = df['close'].pct_change()

    # Log return: mathematically cleaner for statistics
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # Cumulative return: multiply (1 + daily_return) across all days
    df['cumulative_return'] = (1 + df['daily_return']).cumprod()

    # Drawdown: how far are we below the all-time high at each point?
    rolling_max = df['cumulative_return'].cummax()
    df['drawdown'] = (df['cumulative_return'] - rolling_max) / rolling_max

    # Rolling 30-day annualised volatility
    df['rolling_vol_30d'] = df['daily_return'].rolling(30).std() * np.sqrt(365)

    df.dropna(inplace=True)
    return df


def compute_stats(df, risk_free_rate=0.05):
    """
    Compute key quant performance metrics.

    Sharpe Ratio   : return per unit of risk (higher = better, >1 is good)
    Sortino Ratio  : like Sharpe but only penalises downside volatility
    Max Drawdown   : worst peak-to-trough decline (risk of ruin metric)
    Win Rate       : % of days with positive returns
    CAGR           : Compound Annual Growth Rate
    """
    returns = df['daily_return'].dropna()
    n_days  = len(returns)
    n_years = n_days / 365

    # Total return
    total_return = df['cumulative_return'].iloc[-1] - 1

    # CAGR
    cagr = (1 + total_return) ** (1 / n_years) - 1

    # Annualised volatility (crypto uses 365 days)
    ann_vol = returns.std() * np.sqrt(365)

    # Sharpe Ratio (annualised)
    daily_rf = risk_free_rate / 365
    excess_returns = returns - daily_rf
    sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(365)

    # Sortino Ratio — only downside deviation in the denominator
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(365)
    sortino = (cagr - risk_free_rate) / downside_std if downside_std > 0 else np.nan

    # Max Drawdown
    max_dd = df['drawdown'].min()

    # Win Rate
    win_rate = (returns > 0).mean()

    # Skewness & Kurtosis (from your stats coursework)
    skewness = stats.skew(returns)
    kurt     = stats.kurtosis(returns)   # excess kurtosis (normal = 0)

    # Best / Worst single day
    best_day  = returns.max()
    worst_day = returns.min()

    return {
        'Total Return'      : f"{total_return:.1%}",
        'CAGR'              : f"{cagr:.1%}",
        'Ann. Volatility'   : f"{ann_vol:.1%}",
        'Sharpe Ratio'      : f"{sharpe:.2f}",
        'Sortino Ratio'     : f"{sortino:.2f}",
        'Max Drawdown'      : f"{max_dd:.1%}",
        'Win Rate'          : f"{win_rate:.1%}",
        'Best Day'          : f"{best_day:.1%}",
        'Worst Day'         : f"{worst_day:.1%}",
        'Skewness'          : f"{skewness:.2f}",
        'Excess Kurtosis'   : f"{kurt:.2f}  (normal=0, fat tails = high +ve)",
    }


# ─────────────────────────────────────────────
# 3. VISUALISE
# ─────────────────────────────────────────────

def plot_analysis(df, ticker='BTC-USD'):
    """
    4-panel chart:
      Top-left    : Price history
      Top-right   : Daily returns (bar chart)
      Bottom-left : Return distribution vs normal curve
      Bottom-right: Rolling 30d volatility + drawdown
    """
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'{ticker} — Price & Returns Analysis', fontsize=16, fontweight='bold', y=1.01)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    # ── Panel 1: Price ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df.index, df['close'], color='#F7931A', linewidth=1.5, label='Price')
    ax1.fill_between(df.index, df['close'], alpha=0.1, color='#F7931A')
    ax1.set_title('Price History (USD)', fontweight='bold')
    ax1.set_ylabel('Price ($)')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Daily Returns ───────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    colors = ['#2ECC71' if r >= 0 else '#E74C3C' for r in df['daily_return']]
    ax2.bar(df.index, df['daily_return'] * 100, color=colors, width=1, alpha=0.7)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_title('Daily Returns (%)', fontweight='bold')
    ax2.set_ylabel('Return (%)')
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Return Distribution ────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    returns = df['daily_return'].dropna() * 100

    # Histogram of actual returns
    ax3.hist(returns, bins=80, density=True, color='#3498DB', alpha=0.6, label='Actual Returns')

    # Overlay a normal distribution with same mean/std for comparison
    # KEY INSIGHT: if the actual histogram has fatter tails than the curve,
    # that's excess kurtosis — common in crypto and means more extreme days
    x = np.linspace(returns.min(), returns.max(), 300)
    normal_curve = stats.norm.pdf(x, returns.mean(), returns.std())
    ax3.plot(x, normal_curve, color='red', linewidth=2, label='Normal Distribution')

    ax3.axvline(returns.mean(), color='orange', linestyle='--', linewidth=1.5, label=f'Mean: {returns.mean():.2f}%')
    ax3.set_title('Return Distribution vs Normal', fontweight='bold')
    ax3.set_xlabel('Daily Return (%)')
    ax3.set_ylabel('Density')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ── Panel 4: Rolling Vol + Drawdown ─────────────────
    ax4 = fig.add_subplot(gs[1, 1])

    # Rolling volatility on primary axis
    ax4.plot(df.index, df['rolling_vol_30d'] * 100, color='purple', linewidth=1.5, label='30d Ann. Vol (%)')
    ax4.set_ylabel('Annualised Volatility (%)', color='purple')
    ax4.tick_params(axis='y', labelcolor='purple')

    # Drawdown on secondary axis
    ax4b = ax4.twinx()
    ax4b.fill_between(df.index, df['drawdown'] * 100, 0, alpha=0.3, color='red', label='Drawdown (%)')
    ax4b.set_ylabel('Drawdown (%)', color='red')
    ax4b.tick_params(axis='y', labelcolor='red')

    ax4.set_title('Rolling Volatility & Drawdown', fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # Combined legend
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4b.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')

    plt.tight_layout()
    plt.savefig('btc_analysis.png', dpi=150, bbox_inches='tight')
    print("\nChart saved to btc_analysis.png")
    plt.show()


# ─────────────────────────────────────────────
# 4. PRINT STATS TABLE
# ─────────────────────────────────────────────

def print_stats(stats_dict, ticker):
    print(f"\n{'='*45}")
    print(f"  Performance Summary — {ticker}")
    print(f"{'='*45}")
    for k, v in stats_dict.items():
        print(f"  {k:<22} {v}")
    print(f"{'='*45}\n")

    # Teach the reader what the numbers mean
    print("📘 Quick Notes:")
    print("  Sharpe > 1.0  → good risk-adjusted return")
    print("  Sharpe > 2.0  → excellent (rare in practice)")
    print("  Skewness < 0  → more negative tail (crash risk)")
    print("  Kurtosis > 0  → fat tails (more extreme days than normal)")
    print("  Max Drawdown  → the worst loss from a peak you'd have suffered\n")


# ─────────────────────────────────────────────
# 5. MAIN — run everything
# ─────────────────────────────────────────────

if __name__ == '__main__':

    TICKER = 'BTC-USD'   # swap to 'ETH-USD', 'SPY', 'ES=F' etc.
    PERIOD = '2y'        # '1y', '2y', '5y', 'max'

    # Step 1: Get data
    df = get_data(ticker=TICKER, period=PERIOD)

    # Step 2: Calculate returns
    df = calculate_returns(df)

    # Step 3: Compute stats
    perf = compute_stats(df, risk_free_rate=0.05)

    # Step 4: Print stats
    print_stats(perf, TICKER)

    # Step 5: Plot
    plot_analysis(df, ticker=TICKER)

    # ── BONUS: Quick experiment ──────────────────────────
    # Uncomment to compare BTC vs ETH side by side
    #
    # df_eth = get_data('ETH-USD', period=PERIOD)
    # df_eth = calculate_returns(df_eth)
    # perf_eth = compute_stats(df_eth)
    # print_stats(perf_eth, 'ETH-USD')
    #
    # correlation = df['daily_return'].corr(df_eth['daily_return'])
    # print(f"BTC/ETH daily return correlation: {correlation:.2f}")
