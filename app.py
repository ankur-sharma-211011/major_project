import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
import os
from PIL import Image

import main5  # your final backend
from machine_learning_strategies_revised import (
    generate_investor_views,
    download_stock_data,
)

st.set_page_config(page_title="AI Portfolio Dashboard", layout="wide")

# ===================== HEADER =====================
st.title("📊 AI-Powered Portfolio Dashboard")
st.markdown("ML Views → Strategy Comparison → Final Investment Decision")

# ===================== SIDEBAR =====================
with st.sidebar:
    st.header("⚙️ Configuration")

    tickers_input = st.text_input("Tickers", "AAPL,MSFT,GOOGL")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    market_rep = st.text_input("Market Benchmark", "SPY")

    start_date = st.date_input("Start Date", date(2018, 1, 1))
    end_date = st.date_input("End Date", date(2023, 12, 31))

    backtest_start = st.date_input("Backtest Start", date(2020, 1, 1))
    backtest_end = st.date_input("Backtest End", date(2022, 12, 31))

    risk_free_rate = st.number_input("Risk Free Rate", value=0.04)
    target_volatility = st.number_input("Target Volatility", value=0.3)

    min_weight = st.number_input("Min Weight", value=0.01)
    max_weight = st.number_input("Max Weight", value=0.4)

    forward_days = st.number_input("Forward Days", value=20)
    model_type = st.selectbox(
        "Model", ["XGBoost", "Random Forest", "Linear Regression"]
    )

    run_views = st.button("🔍 Generate ML Views")
    run_pipeline = st.button("🚀 Run Full Pipeline")

# ===================== TABS =====================
tab1, tab2 = st.tabs(["📊 ML Insights", "📈 Strategy Analysis"])

# ===================== TAB 1: ML VIEWS =====================
with tab1:
    if run_views:
        st.subheader("ML Predicted Returns & Confidence")

        results = []

        for t in tickers:
            try:
                pred, conf = generate_investor_views(
                    t,
                    str(start_date),
                    str(end_date),
                    model_type=model_type,
                    forward_days=forward_days,
                )
                pred, conf = float(pred), float(conf)
            except:
                pred, conf = np.nan, 0

            df = download_stock_data(t, str(start_date), str(end_date))

            col1, col2 = st.columns([2, 1])
            with col1:
                st.line_chart(df["Adj Close"])
            with col2:
                st.metric(f"{t}", f"{pred*100:.2f}%", f"Conf: {conf:.2f}")

            results.append({"Ticker": t, "Predicted Return": pred, "Confidence": conf})

        df = pd.DataFrame(results)
        st.dataframe(df)
        st.bar_chart(df.set_index("Ticker")["Predicted Return"])

# ===================== TAB 2: FULL PIPELINE =====================
with tab2:
    if run_pipeline:
        st.subheader("Strategy Comparison & Backtest Results")

        with st.spinner("Running full pipeline..."):
            results = main5.full_pipeline(
                tickers=tickers,
                allocations={t: 1 / len(tickers) for t in tickers},
                market_rep=[market_rep],
                start_date=str(start_date),
                end_date=str(end_date),
                backtest_start=str(backtest_start),
                backtest_end=str(backtest_end),
                post_bt_end=str(end_date),
                risk_free_rate=risk_free_rate,
                target_volatility=target_volatility,
                min_weight=min_weight,
                max_weight=max_weight,
                forward_days=forward_days,
                model_type=model_type,
            )

        st.success("Pipeline completed")

        # ================= WEIGHTS =================
        st.subheader("⚖️ Portfolio Allocations")
        weights = results.get("portfolio_weights", {})

        for strategy, w in weights.items():
            st.markdown(f"### {strategy.upper()}")
            st.bar_chart(pd.DataFrame.from_dict(w, orient="index", columns=["Weight"]))

        # ================= METRICS =================
        st.subheader("📊 Strategy Performance Comparison")
        metrics = results.get("performance_metrics", {})

        if metrics:
            df_metrics = pd.DataFrame(metrics).T
            st.dataframe(df_metrics)
            st.bar_chart(df_metrics[["sharpe_ratio", "sortino_ratio", "calmar_ratio"]])

            # BEST STRATEGY
            best = max(metrics, key=lambda x: metrics[x]["sharpe_ratio"])
            st.success(f"🏆 Best Performing Strategy: {best}")

        # ================= CHART =================
        st.subheader("📈 Backtest Performance")
        if os.path.exists("portfolio_comparison.png"):
            st.image(Image.open("portfolio_comparison.png"))

        # ================= FINAL DECISION =================
        st.subheader("🚀 Final Investment Decision")
        bl_weights = weights.get("ml_black_litterman", {})

        st.success("Recommended Strategy: ML Black-Litterman")
        st.write(
            "This strategy incorporates ML predictions + market equilibrium for optimal allocation."
        )
        st.bar_chart(
            pd.DataFrame.from_dict(bl_weights, orient="index", columns=["Allocation"])
        )

        # ================= RAW =================
        st.subheader("📄 Raw Output")
        st.json(results)
