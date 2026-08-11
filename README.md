# Quant Alpha Research

101 Formulaic Alphas project (Kakushadze 2016).

## Setup
```
pip install -r requirements.txt
```

## Structure
- `data_loader.py` — downloads/caches S&P 500 OHLCV data
- `alpha_utils.py` — 7-step template: Signal -> Rank -> Position -> Shift -> Returns -> Combine -> Evaluate
- `main.ipynb` — notebook that defines signals, calls the pipeline, plots results

## Usage
Open `main.ipynb` in VS Code (Jupyter extension). First run downloads data to `data/sp500_5y.parquet` and caches it.
