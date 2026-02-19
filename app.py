from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd

import config
from mpt.fmp_client import FMPClient, FMPConfig
from mpt.data_prep import build_price_panel_on_calendar
from mpt.optimizer import simulate_random_equal_weight
from mpt.frontier import approximate_efficient_frontier
from mpt.plotting import plot_frontier, plot_equity_curve
from mpt.backtest import walk_forward_backtest
from mpt.metrics import cagr, max_drawdown
from mpt.utils import clamp_int

st.set_page_config(page_title="MPT App (Equal-Weight Random Search)", layout="wide")

st.title("Modern Portfolio Theory — Equal-Weight Random Search")
st.caption("S&P 500 universe (current constituents), adjClose returns, random equal-weight portfolios, max Sharpe.")

st.markdown(
    """<style>
      div[data-testid="stMetricValue"] { font-size: 1.35rem; }
    </style>""",
    unsafe_allow_html=True,
)


def _make_client() -> FMPClient:
    cfg = FMPConfig(
        base_url=config.FMP_BASE_URL,
        api_key=config.FMP_API_KEY,
        timeout_sec=config.HTTP_TIMEOUT_SEC,
        max_retries=config.HTTP_MAX_RETRIES,
        max_calls_per_min=config.MAX_CALLS_PER_MIN,
    )
    return FMPClient(cfg)


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_data_cached(api_key: str) -> dict:
    client = _make_client()
    symbols = client.sp500_constituents()

    spy_df = client.historical_price_eod_full("SPY").set_index("date")["adjClose"].astype(float).sort_index()
    calendar = spy_df.index

    panel = build_price_panel_on_calendar(
        client=client,
        symbols=symbols,
        calendar=calendar,
        max_workers=config.MAX_WORKERS,
        progress_cb=None,
    )

    return {
        "symbols_requested": symbols,
        "symbols_kept": panel.symbols,
        "prices": panel.prices,
        "start_date": panel.start_date,
        "end_date": panel.end_date,
        "spy_prices": spy_df,
    }


def load_data_with_progress() -> dict:
    client = _make_client()
    symbols = client.sp500_constituents()

    with st.spinner("Downloading SPY price history..."):
        spy_df = client.historical_price_eod_full("SPY").set_index("date")["adjClose"].astype(float).sort_index()
        calendar = spy_df.index

    prog = st.progress(0.0, text="Downloading S&P 500 price history...")
    status = st.empty()

    def cb(done, total):
        prog.progress(done / total, text=f"Downloading {done:,} / {total:,} symbols...")
        status.write(f"Downloaded {done:,} / {total:,} symbols")

    panel = build_price_panel_on_calendar(
        client=client,
        symbols=symbols,
        calendar=calendar,
        max_workers=config.MAX_WORKERS,
        progress_cb=cb,
    )

    prog.empty()
    status.empty()

    return {
        "symbols_requested": symbols,
        "symbols_kept": panel.symbols,
        "prices": panel.prices,
        "start_date": panel.start_date,
        "end_date": panel.end_date,
        "spy_prices": spy_df,
    }


with st.sidebar:
    st.header("Setup")
    if not config.FMP_API_KEY:
        st.error("Missing FMP_API_KEY. Create a .env file (see .env.example) and restart Streamlit.")
        st.stop()

    st.write("**Universe:** Current S&P 500 constituents.")
    st.write("**Eligibility:** Filtered dynamically by your lookback/holding settings.")
    st.write(f"**Rate limit:** {config.MAX_CALLS_PER_MIN}/min (buffered).")

    use_cache = st.checkbox("Use cached data (recommended)", value=True)

    if st.button("Load / Refresh Data"):
        if use_cache:
            with st.spinner("Loading data (cached when available)..."):
                data = load_data_cached(config.FMP_API_KEY)
        else:
            data = load_data_with_progress()
        st.session_state["data"] = data

data = st.session_state.get("data")
if data is None:
    st.info("Click **Load / Refresh Data** in the sidebar to download and prepare the dataset.")
    st.stop()

symbols_kept = data["symbols_kept"]
prices = data["prices"].copy()
spy_prices = data["spy_prices"].copy()

colA, colB, colC = st.columns([1, 1, 2])
colA.metric("Symbols requested", f"{len(data['symbols_requested']):,}")
colB.metric("Symbols with any data", f"{len(symbols_kept):,}")
colC.markdown(
    f"""<div style="font-size:0.95rem; padding-top:0.35rem;">
    <b>History range</b><br>{data['start_date'].date()} → {data['end_date'].date()}
    </div>""",
    unsafe_allow_html=True,
)

tabs = st.tabs(["Single Simulation", "Walk-Forward Backtest"])

# ----------------
# Single Simulation
# ----------------
with tabs[0]:
    st.subheader("Single Simulation (one lookback window)")
    st.caption("Randomly samples equal-weight portfolios and selects the best Sharpe on the chosen lookback.")

    max_lookback = max(63, len(prices) - 2)  # need lookback+1 prices
    c1, c2, c3, c4 = st.columns(4)

    lookback_days = c1.slider(
        "Lookback (trading days)",
        min_value=63,
        max_value=max_lookback,
        value=clamp_int(config.DEFAULT_LOOKBACK_DAYS, 63, max_lookback),
        step=21,
    )

    k_max = min(config.MAX_K, len(symbols_kept))
    k = c2.slider("Portfolio size (k)", min_value=2, max_value=k_max, value=min(config.DEFAULT_K, k_max), step=1)

    trials = c3.number_input(
        "Trials",
        min_value=1_000,
        max_value=config.MAX_TRIALS_CAP,
        value=min(config.DEFAULT_TRIALS, config.MAX_TRIALS_CAP),
        step=5_000,
    )

    rf = c4.number_input(
        "Risk-free rate (annual, decimal)",
        min_value=0.0,
        max_value=0.20,
        value=float(config.DEFAULT_RF),
        step=0.005,
        format="%.3f",
    )

    seed = st.number_input("Random seed", min_value=0, max_value=10_000_000, value=42, step=1)

    run = st.button("Run Simulation", type="primary")

    if run:
        lb_prices = prices.iloc[-(lookback_days + 1):]
        eligible_mask = lb_prices.notna().all(axis=0)
        eligible = lb_prices.columns[eligible_mask].tolist()

        if len(eligible) < k:
            st.error(
                f"Only {len(eligible)} symbols have complete price history for the selected lookback. "
                f"Shorten lookback or reduce k."
            )
            st.stop()

        lb_rets = lb_prices[eligible].pct_change().iloc[1:]

        mu = lb_rets.mean(axis=0).to_numpy()
        cov = lb_rets.cov().to_numpy()
        mu_ann = mu * config.TRADING_DAYS_PER_YEAR
        cov_ann = cov * config.TRADING_DAYS_PER_YEAR

        st.info(f"Eligible symbols for this lookback: {len(eligible):,}")

        prog = st.progress(0.0, text="Running simulations...")
        status = st.empty()

        def sim_cb(done, total):
            prog.progress(done / total, text=f"Running simulations... {done:,} / {total:,}")
            status.write(f"Processed {done:,} / {total:,} trials")

        res = simulate_random_equal_weight(
            mu_ann=mu_ann,
            cov_ann=cov_ann,
            tickers=eligible,
            k=int(k),
            trials=int(trials),
            rf=float(rf),
            batch_size=int(config.BATCH_SIZE),
            seed=int(seed),
            plot_sample_size=int(config.PLOT_SAMPLE_SIZE),
            top_sharpe_to_plot=int(config.TOP_SHARPE_TO_PLOT),
            progress_cb=sim_cb,
        )

        prog.empty()
        status.empty()

        # Row 1: best + key table
        left, right = st.columns([1, 2])
        with left:
            st.markdown("### Best Sharpe Portfolio")
            st.write(", ".join(res.best_sharpe.tickers))
            st.metric("Sharpe", f"{res.best_sharpe.sharpe:.3f}")
            st.metric("Ann. Return", f"{res.best_sharpe.ann_return:.2%}")
            st.metric("Ann. Vol", f"{res.best_sharpe.ann_vol:.2%}")

        with right:
            st.markdown("### Key Portfolios")
            key_rows = []
            for name, p in res.key_points.items():
                key_rows.append({
                    "key": name,
                    "sharpe": p.sharpe,
                    "ann_return": p.ann_return,
                    "ann_vol": p.ann_vol,
                    "tickers": ", ".join(p.tickers),
                })
            st.dataframe(pd.DataFrame(key_rows).sort_values("key"), use_container_width=True, height=300)

        # Row 2: full-width chart
        st.markdown("### Efficient Frontier (Sampled)")
        df_points = res.plot_points.copy()
        frontier = approximate_efficient_frontier(df_points[df_points["kind"].isin(["sample", "top", "key"])])
        fig = plot_frontier(df_points, frontier)
        st.plotly_chart(fig, use_container_width=True)

# ----------------
# Backtest
# ----------------
with tabs[1]:
    st.subheader("Walk-Forward Backtest (step = holding period)")
    st.caption("For each window: optimize on lookback (random trials), then hold for the holding period. Compare to SPY.")

    c1, c2, c3, c4 = st.columns(4)

    hold_days = c2.slider(
        "Holding period (trading days)",
        min_value=21,
        max_value=252,
        value=clamp_int(config.DEFAULT_HOLD_DAYS, 21, 252),
        step=21,
        key="bt_hold",
    )

    max_lookback = max(63, len(prices) - hold_days - 2)
    lookback_days = c1.slider(
        "Lookback (trading days)",
        min_value=63,
        max_value=max_lookback,
        value=clamp_int(config.DEFAULT_LOOKBACK_DAYS, 63, max_lookback),
        step=21,
        key="bt_lb",
    )

    k_max = min(config.MAX_K, len(symbols_kept))
    k = c3.slider("Portfolio size (k)", min_value=2, max_value=k_max, value=min(config.DEFAULT_K, k_max), step=1, key="bt_k")

    trials = c4.number_input(
        "Trials per window",
        min_value=1_000,
        max_value=config.MAX_TRIALS_CAP,
        value=20_000,
        step=5_000,
        key="bt_trials",
    )

    seed = st.number_input("Base random seed", min_value=0, max_value=10_000_000, value=42, step=1, key="bt_seed")

    run_bt = st.button("Run Backtest", type="primary", key="bt_run")

    if run_bt:
        prog = st.progress(0.0, text="Running walk-forward backtest...")
        status = st.empty()

        def cb(done, total):
            if total <= 0:
                return
            prog.progress(min(done / total, 1.0), text=f"Backtest windows: {done} / {total}")
            status.write(f"Completed {done} / {total} windows")

        bt = walk_forward_backtest(
            prices=prices,
            spy_prices=spy_prices,
            lookback_days=int(lookback_days),
            hold_days=int(hold_days),
            trials=int(trials),
            k=int(k),
            rf=0.0,
            batch_size=int(config.BATCH_SIZE),
            seed=int(seed),
            progress_cb=cb,
        )

        prog.empty()
        status.empty()

        if bt.equity_strategy.empty or bt.equity_spy.empty:
            st.warning("Backtest produced no completed windows (likely due to insufficient eligible symbols for the chosen k).")
            st.stop()

        strat = bt.equity_strategy / bt.equity_strategy.iloc[0]
        bench = bt.equity_spy / bt.equity_spy.iloc[0]

        fig = plot_equity_curve(strat, bench)
        st.plotly_chart(fig, use_container_width=True)

        summ1, summ2, summ3, summ4 = st.columns(4)
        summ1.metric("Strategy CAGR", f"{cagr(strat):.2%}")
        summ2.metric("SPY CAGR", f"{cagr(bench):.2%}")
        summ3.metric("Strategy Max Drawdown", f"{max_drawdown(strat):.2%}")
        summ4.metric("SPY Max Drawdown", f"{max_drawdown(bench):.2%}")

        st.markdown("### Window Log")
        st.dataframe(bt.windows, use_container_width=True, height=420)

        csv = bt.windows.to_csv(index=False).encode("utf-8")
        st.download_button("Download window log (CSV)", data=csv, file_name="walk_forward_windows.csv", mime="text/csv")
