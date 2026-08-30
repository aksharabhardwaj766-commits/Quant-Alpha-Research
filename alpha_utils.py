"""
alpha_utils.py
The 7-step alpha template as reusable functions:
Signal -> Rank -> Position -> Shift -> Returns -> Combine -> Evaluate
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SPLIT_DATE = pd.Timestamp('2025-02-06')

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

def signal_to_position(signal_df, long_pct=0.1, short_pct=0.1):
    '''signal strength weighted positions, within the long/short bucket - stronger signal = bigger position
    positions are dollar-neutral (longs sum to +1, shorts sum to -1)'''
    rank_df = signal_df.rank(axis=1, pct=True) # percentile rank, 0 to 1, per day
    long_mask = rank_df > (1 - long_pct)
    short_mask = rank_df <= short_pct

    # keep raw signal only inside long/short buckets, zero elsewhere
    long_signal = signal_df.where(long_mask, 0)
    short_signal = signal_df.where(short_mask, 0).abs() #use magnitude for short weighing

    # normalize each side to sum to 1 (perday), so total long = +1, total short = -1
    long_weights = long_signal.div(long_signal.sum(axis=1), axis=0).fillna(0)
    short_weights = short_signal.div(short_signal.sum(axis=1), axis=0). fillna(0)

    position = long_weights - short_weights
    return position 


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


def evaluate_alpha(signal_df, close_df, n_stocks, long_pct=0.1, short_pct=0.1, neutralize='None', weighting='rank', split_date=SPLIT_DATE):
    """
    Full pipeline: signal -> rank -> position -> shift -> returns -> combine -> evaluate.
    Returns a dict of results.
    neutralisation : None (No Neutralisation), neutralisation : 'market' (subtract cross sectional mean every day)
    weighing = rank (existing +1/-1 buckets) or signal - strength weighted
    """
    if neutralize == 'market':
        signal_df = signal_df.sub(signal_df.mean(axis=1), axis=0)

    if weighting == 'signal':
        position_df = signal_to_position(signal_df, long_pct, short_pct)
        pos_shifted = shift_positions(position_df)

    else:        
        rank_df = rank_signal(signal_df)
        position_df = rank_to_position(rank_df, n_stocks, long_pct, short_pct)
        pos_shifted = shift_positions(position_df)

    daily_returns = compute_returns(close_df)
    strat_returns = strategy_returns(pos_shifted, daily_returns)

    forward_returns = daily_returns.shift(-1)
    ic_series = compute_ic(signal_df, forward_returns)

    train_returns = strat_returns[strat_returns.index < split_date]
    test_returns = strat_returns[strat_returns.index >= split_date]
    train_ic = ic_series[ic_series.index < split_date]
    test_ic = ic_series[ic_series.index >= split_date]

    return {
        "strategy_returns": strat_returns,
        "sharpe": sharpe_ratio(strat_returns),
        "ic_series": ic_series,
        "ic_mean": ic_series.mean(),
        "icir": icir(ic_series),
        "train_sharpe": sharpe_ratio(train_returns),
        "test_sharpe": sharpe_ratio(test_returns),
        "train_ic": train_ic.mean(),
        "test_ic": test_ic.mean(),
    }

def ts_rank(df, window):
    """Rolling rank of the most recent window of the column"""
    return df.rolling(window).apply(lambda x: pd.Series(x).rank().iloc[-1])

def rolling_corr(df1, df2, window):
    '''Rolling correlation between two dataframes, column by column'''
    return df1.rolling(window).corr(df2)

def ts_min(series, window):
    '''Rolling min over past window days'''
    return series.rolling(window).min()

def ts_max(series, window):
    '''Rolling max over past window days'''
    return series.rolling(window).max()

def linear_decay(df, window):
    '''Vectorised linear decay weighted moving average, most recent days get high weight '''
    weights = np.arange(1, window+1)
    weight_sum = weights.sum()

    values = df.values # shape : (n_dates, n_stocks)
    n_dates, n_stocks = values.shape
    result = np.full_like(values, np.nan, dtype=float)

    for i in range(window - 1, n_dates):
        window_slice = values[i - window + 1 : i+1, :]
        result[i, :] = np.dot(weights, window_slice)/weight_sum
    return pd.DataFrame(result, index=df.index, columns=df.columns)    

