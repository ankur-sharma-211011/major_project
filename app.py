import streamlit as st
import pandas as pd
import numpy as np
from datetime import date

import main5
from machine_learning_strategies_revised import (
    generate_investor_views,
    download_stock_data,
)

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Intelligent Portfolio Strategy Terminal",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# STYLING
# =========================================================
st.markdown(
    """
<style>

html, body, [class*="css"] {
    font-family: Inter;
}

.main {
    background: linear-gradient(180deg,#08111f,#111827);
}

.block-container {
    padding-top:1.5rem;
    padding-bottom:2rem;
}

h1,h2,h3 {
    color:white;
}

div[data-testid="metric-container"]{
    background: #111827;
    border: 1px solid #374151;
    padding:18px;
    border-radius:18px;
    box-shadow: 0px 2px 12px rgba(0,0,0,.35);
}

.strategy-box{
    background:#111827;
    padding:22px;
    border-radius:18px;
    border:1px solid #374151;
    margin-bottom:15px;
}

.big-number{
    font-size:34px;
    font-weight:700;
    color:#60a5fa;
}

.small-label{
    font-size:14px;
    color:#9ca3af;
}

hr {
    border:1px solid #374151;
}

</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# HEADER
# =========================================================
st.title("📊 Intelligent Portfolio Strategy Terminal")
st.caption(
    "Machine Learning Views • Black-Litterman Optimization • Backtested Allocation Decisions"
)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.header("⚙ Configuration")

    tickers_input = st.text_input(
        "Portfolio Universe", "DE, AGCO, ADM, BG, CF, FMC, MOO, DBA, GLD, TLT"
    )

    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    market_rep = st.text_input("Benchmark", "MOO")

    st.divider()

    st.subheader("Dates")

    start_date = st.date_input("Training Start", date(2014, 1, 1))

    end_date = st.date_input("Training End", date(2026, 2, 1))

    backtest_start = st.date_input("Backtest Start", date(2026, 2, 1))

    backtest_end = st.date_input("Backtest End", date(2026, 4, 1))

    st.divider()

    st.subheader("Methodology")

    st.info("Risk-Free Rate fixed internally")
    st.info("Target Volatility selected via volatility sweep")

    st.divider()

    forward_days = st.number_input("Forward Days", value=20)

    model_type = st.selectbox(
        "ML Model", ["XGBoost", "Random Forest", "Linear Regression"]
    )

    run_views = st.button("🔍 Generate ML Views", use_container_width=True)

    run_pipeline = st.button("🚀 Run Full Pipeline", use_container_width=True)

# =========================================================
# TABS
# =========================================================
tab1, tab2 = st.tabs(["📈 ML Insights", "🧠 Strategy Analysis"])

# =========================================================
# TAB 1
# =========================================================
with tab1:

    if run_views:

        st.subheader("Predicted Returns")

        ml_results = []

        for ticker in tickers:

            try:
                pred, conf = generate_investor_views(
                    ticker,
                    str(start_date),
                    str(end_date),
                    model_type=model_type,
                    forward_days=forward_days,
                )

                pred = float(pred)
                conf = float(conf)

            except:
                pred = np.nan
                conf = 0

            prices = download_stock_data(ticker, str(start_date), str(end_date))

            left, right = st.columns([2, 1])

            with left:
                if "Adj Close" in prices.columns:
                    st.line_chart(prices["Adj Close"])

            with right:
                st.metric(ticker, f"{pred*100:.2f}%", f"Confidence {conf:.2f}")

            ml_results.append(
                {"Ticker": ticker, "Predicted Return": pred, "Confidence": conf}
            )

        df = pd.DataFrame(ml_results).sort_values("Predicted Return", ascending=False)

        st.subheader("Opportunity Ranking")

        st.dataframe(df)

        st.bar_chart(df.set_index("Ticker")["Predicted Return"])

# =========================================================
# TAB 2
# =========================================================
with tab2:

    if run_pipeline:

        with st.spinner("Running optimization..."):

            results = main5.full_pipeline(
                tickers=tickers,
                allocations={t: 1 / len(tickers) for t in tickers},
                market_rep=[market_rep],
                start_date=str(start_date),
                end_date=str(end_date),
                backtest_start=str(backtest_start),
                backtest_end=str(backtest_end),
                post_bt_end=str(end_date),
                forward_days=forward_days,
                model_type=model_type,
            )

        st.success("Pipeline Complete")

        weights = results["portfolio_weights"]
        metrics = results["performance_metrics"]

        best = max(metrics, key=lambda x: metrics[x]["sharpe_ratio"])

        selected_vol = results["selected_target_volatility"]

        bl_return = metrics["ML Black-Litterman"]["total_return_percent"]
        mvo_return = metrics["MVO"]["total_return_percent"]

        excess_return = bl_return - mvo_return

        # ========================================
        # DASHBOARD TOP CARDS
        # ========================================
        st.markdown("## Portfolio Summary")

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric("Best Strategy", best)

        c2.metric("Sharpe", f"{metrics[best]['sharpe_ratio']:.2f}")

        c3.metric("Return", f"{metrics[best]['total_return_percent']:.2f}%")

        c4.metric("Target Vol", f"{selected_vol*100:.1f}%")

        c5.metric("BL Excess vs MVO", f"{excess_return:.2f}%")

        st.divider()

        # ========================================
        # STRATEGY TABLE
        # ========================================
        st.subheader("Strategy Comparison")

        metric_df = pd.DataFrame(metrics).T

        st.dataframe(
            metric_df.style.highlight_max(
                subset=[
                    "sharpe_ratio",
                    "sortino_ratio",
                    "calmar_ratio",
                    "total_return_percent",
                ],
                axis=0,
            )
        )

        # ========================================
        # BACKTEST CHART
        # ========================================
        st.subheader("Backtest Growth")

        chart_df = results["chart_data"]

        if chart_df is not None:

            chart_df = (chart_df / chart_df.iloc[0] - 1) * 100

            st.line_chart(chart_df)

        # ========================================
        # ALLOCATIONS
        # ========================================
        st.subheader("Portfolio Weights")

        cols = st.columns(3)

        strategies = list(weights.keys())

        for i, strat in enumerate(strategies):

            with cols[i]:

                st.markdown(f"### {strat.upper()}")

                dfw = pd.DataFrame.from_dict(
                    weights[strat], orient="index", columns=["Weight"]
                )

                st.bar_chart(dfw)

        # ========================================
        # RECOMMENDATION PANEL
        # ========================================
        st.divider()

        st.subheader("🚀 Recommendation")

        if excess_return > 0:
            message = "Black-Litterman is outperforming MVO."
        else:
            message = "MVO currently exceeds Black-Litterman."

        st.success(
            f"""
Recommended Strategy: {best}

Reasoning:
• Highest risk-adjusted performance

• Selected via volatility sweep

• BL vs MVO Excess Return:
{excess_return:.2f}%

• {message}
"""
        )

        # ========================================
        # RAW MODEL
        # ========================================
        with st.expander("View Raw Model Output"):
            st.json(results)
