# This main5 is different from original main5 for serialization

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from pypfopt import EfficientFrontier, risk_models, expected_returns, black_litterman
from pypfopt.black_litterman import market_implied_risk_aversion
import yfinance as yf
import portfolio_statistics as ps
import machine_learning_strategies_revised as mls
import warnings
import json

warnings.filterwarnings("ignore")


# === JSON SERIALIZATION FIX ===
def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj


# === Data Acquisition ===
def fetch_price_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)[
        "Adj Close"
    ]
    return data if isinstance(data, pd.DataFrame) else data.to_frame()


# === ML-Based View Generation ===
def generate_views(
    tickers, start_date, end_date, model_type="XGBoost", forward_days=20
):
    views, confidences = {}, {}
    for ticker in tickers:
        view, confidence = mls.generate_investor_views(
            ticker, start_date, end_date, model_type, forward_days
        )
        views[ticker] = view
        confidences[ticker] = confidence
    return views, confidences


# === Optimization: Mean-Variance ===
def optimize_mvo(mu, S, min_weight, max_weight, target_volatility):
    ef = EfficientFrontier(mu, S)
    ef.add_constraint(lambda w: w >= min_weight)
    ef.add_constraint(lambda w: w <= max_weight)
    ef.efficient_risk(target_volatility=target_volatility)
    return ef.clean_weights()


# === Optimization: Black-Litterman ===
def optimize_black_litterman(
    S,
    pi,
    investor_views,
    omega,
    market_weights,
    delta,
    min_weight,
    max_weight,
    target_volatility,
):
    bl = black_litterman.BlackLittermanModel(
        cov_matrix=S,
        pi=pi,
        absolute_views=investor_views,
        omega=omega,
        market_weights=market_weights,
        risk_aversion=delta,
    )
    bl_mu = bl.bl_returns()
    bl_cov = bl.bl_cov()
    ef = EfficientFrontier(bl_mu, bl_cov)
    ef.add_constraint(lambda w: w >= min_weight)
    ef.add_constraint(lambda w: w <= max_weight)
    ef.efficient_risk(target_volatility=target_volatility)
    return ef.clean_weights(), bl_mu


# === Backtesting ===
def run_backtest(data, weight_dict, tickers):
    weights_array = np.array([weight_dict[t] for t in tickers])
    daily_returns = data.pct_change()
    strat_return = daily_returns.dot(weights_array)
    cumulative = (1 + strat_return).cumprod()
    return strat_return, cumulative


# === Metrics ===
def extract(value):
    return float(value.values[0]) if isinstance(value, pd.Series) else float(value)


def compute_metrics(returns, labels, risk_free_rate):
    metrics = {}
    for name, ret in zip(labels, returns):
        metrics[name] = {
            "sharpe_ratio": round(extract(ps.sharpe_ratio(ret, risk_free_rate)), 4),
            "sortino_ratio": round(extract(ps.sortino_ratio(ret, risk_free_rate)), 4),
            "calmar_ratio": round(extract(ps.calmar_ratio(ret, risk_free_rate)), 4),
            "total_return_percent": round(extract((1 + ret).prod() - 1) * 100, 2),
        }
    return metrics


# === Visualization ===
def plot_and_save_cumulative(
    cums,
    labels,
    metrics,
    benchmark=None,
    benchmark_label="Benchmark (MOO)",
    filename="portfolio_comparison.png",
):
    colors = sns.color_palette(
        "bright", n_colors=len(labels) + (1 if benchmark is not None else 0)
    )
    strategy_colors = {label: colors[i] for i, label in enumerate(labels)}
    if benchmark is not None:
        strategy_colors[benchmark_label] = colors[len(labels)]

    fig, ax = plt.subplots(figsize=(16, 9), constrained_layout=True)
    fig.set_facecolor("black")
    ax.set_facecolor("black")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.2f}%".format(y)))

    for label, line in zip(labels, cums):
        ax.plot((line - 1) * 100, label=label, color=strategy_colors[label])
    if benchmark is not None:
        ax.plot(
            (benchmark - 1) * 100,
            label=benchmark_label,
            color=strategy_colors[benchmark_label],
        )

    for i, label in enumerate(labels):
        m = metrics[label]
        y_pos = 0.78 - i * 0.1
        text = f"{label}:\nSharpe: {m['sharpe_ratio']:.2f}\nSortino: {m['sortino_ratio']:.2f}\nCalmar: {m['calmar_ratio']:.2f}\nReturn: {m['total_return_percent']:.2f}%"
        fig.text(
            0.07,
            y_pos,
            text,
            transform=fig.transFigure,
            fontsize=10,
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.3",
                edgecolor=strategy_colors[label],
                facecolor="black",
            ),
        )

    ax.set_title("Cumulative Portfolio Returns", color="white")
    ax.set_xlabel("Date")
    ax.set_ylabel("Gain (%)")
    ax.legend(loc="best")
    ax.grid(True)

    plt.savefig(filename, format="png", dpi=300, bbox_inches="tight", facecolor="black")
    plt.close(fig)


# === Full Pipeline ===
def full_pipeline(
    tickers,
    allocations,
    market_rep,
    start_date,
    end_date,
    backtest_start,
    backtest_end,
    post_bt_end,
    risk_free_rate=0.04,
    target_volatility=0.3,
    min_weight=0.01,
    max_weight=0.2,
    forward_days=20,
    model_type="XGBoost",
):
    price_data = fetch_price_data(tickers, start_date, end_date)
    market_data = fetch_price_data(market_rep, start_date, end_date)

    mu = expected_returns.mean_historical_return(price_data)
    S = risk_models.sample_cov(price_data)
    delta = market_implied_risk_aversion(market_data)
    pi = expected_returns.capm_return(price_data, market_prices=market_data)

    investor_views, view_confidences = generate_views(
        tickers, start_date, end_date, model_type, forward_days
    )
    omega_diag = [
        1.0 / view_confidences[t] if view_confidences[t] != 0 else 1e-4
        for t in investor_views
    ]
    omega = np.diag(omega_diag)
    market_weights = {ticker: 1 / len(tickers) for ticker in tickers}

    weights_mvo = optimize_mvo(mu, S, min_weight, max_weight, target_volatility)
    weights_bl, bl_mu = optimize_black_litterman(
        S,
        pi,
        investor_views,
        omega,
        market_weights,
        delta,
        min_weight,
        max_weight,
        target_volatility,
    )
    weights_unopt = allocations

    data_bt = fetch_price_data(tickers, backtest_start, backtest_end)
    ret_mvo, cum_mvo = run_backtest(data_bt, weights_mvo, tickers)
    ret_bl, cum_bl = run_backtest(data_bt, weights_bl, tickers)
    ret_unopt, cum_unopt = run_backtest(data_bt, weights_unopt, tickers)

    benchmark = fetch_price_data(market_rep, backtest_start, backtest_end)
    benchmark_cum = (1 + benchmark.pct_change()).cumprod()

    strategy_labels = ["ML Black-Litterman", "MVO", "Unoptimized"]
    returns = [ret_bl, ret_mvo, ret_unopt]
    cums = [cum_bl, cum_mvo, cum_unopt]
    metrics = compute_metrics(returns, strategy_labels, risk_free_rate)

    plot_and_save_cumulative(
        cums,
        strategy_labels,
        metrics,
        benchmark_cum[market_rep[0]],  # ✅ FIXED
        benchmark_label=market_rep[0],
        filename="portfolio_comparison.png",
    )

    results = {
        "portfolio_weights": {
            "unoptimized": weights_unopt,
            "mvo": weights_mvo,
            "ml_black_litterman": weights_bl,
        },
        "performance_metrics": metrics,
        "views": investor_views,
        "view_confidences": view_confidences,
        "expected_returns_bl": bl_mu.to_dict(),
    }

    # ✅ FIX APPLIED HERE
    with open("portfolio_results.json", "w") as f:
        json.dump(convert_to_serializable(results), f, indent=4)

    print("Saved chart and JSON successfully (no serialization errors)")
    return results
