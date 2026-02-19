# mpt-app

Streamlit Modern Portfolio Theory app using Financial Modeling Prep (FMP) data.

## Features (v1)
- Universe: current S&P 500 constituents (strict full-history filter)
- Prices: daily EOD `adjClose`
- Optimization: random sampling of equal-weight portfolios, maximizing Sharpe
- Two tabs:
  - Single Simulation
  - Walk-forward Backtest (step = holding period)
- Efficient frontier chart (sampled for UI performance), always includes key portfolios:
  - min vol, max vol, min return, max return, max Sharpe

## Setup (Codespaces / local)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env`:
```bash
cp .env.example .env
# edit .env and set your key
```

Run:
```bash
streamlit run app.py
```

## Notes
- This app caches downloads in Streamlit's cache (not a database).
- Strict full-history mode keeps only symbols with complete history on the intersection calendar.
