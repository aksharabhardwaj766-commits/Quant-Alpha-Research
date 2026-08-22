"""
data_loader.py
Downloads S&P 500 OHLCV data via yfinance, saves to parquet.
Run once, then reuse the cached file.
"""

import pandas as pd
import yfinance as yf
import os

DATA_PATH = "data/sp500_5y.parquet"


def get_sp500_tickers():
    """Fetch current S&P 500 ticker list from GitHub."""
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
    df = pd.read_csv(url)
    return df["Symbol"].tolist()


def download_data(period="5y"):
    """Download OHLCV for all S&P 500 tickers. Saves to parquet."""
    tickers = get_sp500_tickers()
    print(f"Downloading {len(tickers)} tickers...")

    data = yf.download(tickers, period=period, auto_adjust=True, group_by="ticker")

    os.makedirs("data", exist_ok=True)
    data.to_parquet(DATA_PATH)
    print(f"Saved to {DATA_PATH}, shape: {data.shape}")
    return data


def load_data():
    """Load cached data if it exists, else download fresh."""
    if os.path.exists(DATA_PATH):
        print(f"Loading cached data from {DATA_PATH}")
        return pd.read_parquet(DATA_PATH)
    return download_data()


def get_close_prices(data):
    """Extract just the Close price panel: dates x tickers."""
    return data.xs("Close", axis=1, level="Price")

def get_open_prices(data):
    '''Extract Open price panel: dates x tickers'''
    return data.xs('Open', axis=1, level='Price')

def get_volume(data):
    """Extract just the Volume panel: dates x tickers."""
    return data.xs("Volume", axis=1, level="Price")


if __name__ == "__main__":
    df = load_data()
    close = get_close_prices(df)
    print(close.tail())
