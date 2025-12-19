# Solar Activity and Market Risk

This repository contains an empirical analysis exploring whether solar activity
(Sunspot Numbers) is statistically associated with the market risk environment
of the S&P 500 index.

Key choices:
- Risk proxy: monthly realized volatility (std of daily log-returns)
- Two alignments: retrospective (same-month) vs predictive (anti-lookahead, M → M+1)
- Robust inference: HAC / Newey–West standard errors
- Stability check: split sample into three subperiods

## Structure
- `code/`: main Jupyter notebook (analysis + results)
- `code/solar_market_risk.py`: reusable functions used by the notebook
- `requirements.txt`: dependencies
- `data/`: placeholder (no raw data committed)

## Run
1. Install dependencies: `pip install -r requirements.txt`
2. Open and run the notebook in `code/`

## Disclaimer
For research and educational purposes only. Not investment advice.
