"""
alpha_utils.py
The 7-step alpha template as reusable functions:
Signal -> Rank -> Position -> Shift -> Returns -> Combine -> Evaluate
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def rank_signal(signal_df):
    """Cross-sectional rank, per day (axis=1). Higher signal = higher rank."""
    return signal_df.rank(axis=1, method="first")


def rank_to_position(rank_df, n_stocks=None, long_pct=0.1, short_pct=0.1):
    """
    Convert rank into +1 / -1 / 0 positions.
    Top long_pct = long (+1). Bottom short_pct = short (-1).
    Uses per-day valid stock count (not a fixed n_stocks) so cutoffs
    stay correct even when some days have NaNs (e.g. early rolling-window rows).
    """
    valid_count = rank_df.notna().sum(axis=1)  # per-day count, Series
    long_cutoff = valid_count * (1 - long_pct)
    short_cutoff = valid_count * short_pct

    # broadcast per-day cutoffs across columns
    long_cutoff_df = pd.DataFrame(
        np.tile(long_cutoff.values.reshape(-1, 1), rank_df.shape[1]),
        index=rank_df.index, columns=rank_df.columns
    )
    short_cutoff_df = pd.DataFrame(
        np.tile(short_cutoff.values.reshape(-1, 1), rank_df.shape[1]),
        index=rank_df.index, columns=rank_df.columns
    )

    position = np.where(rank_df > long_cutoff_df, 1,
                np.where(rank_df <= short_cutoff_df, -1, 0))
    return pd.DataFrame(position, index=rank_df.index, columns=rank_df.columns)


def shift_positions(position_df, lag=1):
    """Shift positions forward to avoid lookahead bias."""
    return position_df.shift(lag)


def compute_returns(close_df):
    """Daily simple returns from close prices."""
    return close_df.pct_change(fill_method=None)


def strategy_returns(pos_shifted, daily_returns):
    """Combine: equal-weighted portfolio return per day."""
    n_active = (pos_shifted != 0).sum(axis=1)
    return (pos_shifted * daily_returns).sum(axis=1) / n_active.replace(0, np.nan)


def sharpe_ratio(returns, periods_per_year=252):
    """Annualised Sharpe ratio."""
    return returns.mean() / returns.std() * np.sqrt(periods_per_year)


def compute_ic(signal_df, forward_returns_df):
    """
    Daily Information Coefficient: Spearman correlation between
    today's signal and tomorrow's returns, computed cross-sectionally.
    """
    ic_list = []
    for date in signal_df.index:
        s = signal_df.loc[date]
        r = forward_returns_df.loc[date]
        valid = s.notna() & r.notna()
        if valid.sum() > 1:
            ic, _ = spearmanr(s[valid], r[valid])
            ic_list.append(ic)
        else:
            ic_list.append(np.nan)
    return pd.Series(ic_list, index=signal_df.index)


def icir(ic_series):
    """ICIR = mean(IC) / std(IC). Primary alpha quality metric."""
    return ic_series.mean() / ic_series.std()


def evaluate_alpha(signal_df, close_df, n_stocks, long_pct=0.1, short_pct=0.1):
    """
    Full pipeline: signal -> rank -> position -> shift -> returns -> combine -> evaluate.
    Returns a dict of results.
    """
    rank_df = rank_signal(signal_df)
    position_df = rank_to_position(rank_df, n_stocks, long_pct, short_pct)
    pos_shifted = shift_positions(position_df)

    daily_returns = compute_returns(close_df)
    strat_returns = strategy_returns(pos_shifted, daily_returns)

    forward_returns = daily_returns.shift(-1)
    ic_series = compute_ic(signal_df, forward_returns)

    return {
        "strategy_returns": strat_returns,
        "sharpe": sharpe_ratio(strat_returns),
        "ic_series": ic_series,
        "ic_mean": ic_series.mean(),
        "icir": icir(ic_series),
    }

def ts_rank(df, window):
    """Rolling rank of the most recent window of the column"""
    return df.rolling(window).apply(lambda x: pd.Series(x).rank().iloc[-1])

def rolling_corr(df1, df2, window):
    '''Rolling correlation between two dataframes, column by column'''
    return df1.rolling(window).corr(df2)
