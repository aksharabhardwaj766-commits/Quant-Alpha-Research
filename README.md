# Quant Alpha Research

## Setup 

pip install -r requiremnts.txt

## Structure
- data_loader.py : downloads/caches S&P 500 OHLCV data
- alpha_utils.py : 7-step template : Signal -> Rank -> Position -> shift -> Returns -> Combine ->      Evaluate
- main.ipynb : notebook that defines signal, calls the pipeline, plot the output

## Simulated Alphas

1. 12-1 Month Momentum (Jagdeesh-Titman): classic cross-sectional momentum ✅

Signal : 11 month returns (252-day lookback), skipping the most recent month (21-days to avoid short-term reversal contamination)

Universe : Top 100 stocks (large-cap within S&P 500)

Weighting : rank strength

Train : Sharpe = 0.9735 | IC Mean = 0.0248

Test OOS : Sharpe = 0.8041 | IC mean = 0.0267

2. Short Term Reversal, 1 week/1 month (Jagdeesh-Titman): Opposite sign of reversal ✅

Signal : Past returns over short window (ex: 5 trade days, 21 trade days)

Skip-day test: skip=1 removes next-days bid-ask bounce.

5d train sharpe drops 1.25 (skip=0) -> 0.77 (skip=1)

confirming raw signal is partly-bounce noise not pure overeaction.

Only (5d, skip 1) tested OOS

Universe : Bottom 200 stocks (small-cap within S&P 500)

Weighting : rank strength

Train : Sharpe = 0.766 | IC Mean = 0.011

Test OOS : Sharpe = 0.190 | IC mean = 0.006

IC is cyclic (bursts ~0.05, flat near 0) in both train and test signal works intermittently, not uniformly over time

3. Earning Surprise Proxy ✅

4. Amihud Alpha ✅

- In search of more ideas

5. 

6. 

7. 

8. 

9. 

10.

## Requirements
pandas

numpy 

yfinance

scipy

matplotlib

seaborn

jupyter