# main5 file that has volatility sweep method

import numpy as np
import pandas as pd
import yfinance as yf
import warnings

from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import black_litterman
from pypfopt.black_litterman import market_implied_risk_aversion

import portfolio_statistics as ps
import machine_learning_strategies_revised as mls

warnings.filterwarnings("ignore")

# ===============================
# FIXED BACKEND PARAMETERS
# ===============================
RISK_FREE_RATE = 0.0446
MIN_WEIGHT = 0.01
MAX_WEIGHT = 0.20
VOL_RANGE = np.linspace(0.05, 0.30, 21)


# ===============================
# DATA
# ===============================
def fetch_price_data(tickers, start_date, end_date):
    data = yf.download(
        tickers, start=start_date, end=end_date, auto_adjust=False, progress=False
    )["Adj Close"]

    if isinstance(data, pd.Series):
        data = data.to_frame()

    return data


# ===============================
# ML VIEWS
# ===============================
def generate_views(
    tickers, start_date, end_date, model_type="XGBoost", forward_days=20
):

    views = {}
    confidences = {}

    for ticker in tickers:
        view, conf = mls.generate_investor_views(
            ticker, start_date, end_date, model_type, forward_days
        )

        views[ticker] = view
        confidences[ticker] = conf

    return views, confidences


# ===============================
# HELPERS
# ===============================
def extract(value):
    return float(value.values[0]) if isinstance(value, pd.Series) else float(value)


def compute_metrics(returns, labels):

    metrics = {}

    for name, ret in zip(labels, returns):

        metrics[name] = {
            "sharpe_ratio": round(extract(ps.sharpe_ratio(ret, RISK_FREE_RATE)), 4),
            "sortino_ratio": round(extract(ps.sortino_ratio(ret, RISK_FREE_RATE)), 4),
            "calmar_ratio": round(extract(ps.calmar_ratio(ret, RISK_FREE_RATE)), 4),
            "total_return_percent": round(extract((1 + ret).prod() - 1) * 100, 2),
        }

    return metrics


def run_backtest(data, weights_dict, tickers):

    w = np.array([weights_dict[t] for t in tickers])

    returns = data.pct_change().dropna()

    strat_return = returns.dot(w)

    cumulative = (1 + strat_return).cumprod()

    return strat_return, cumulative


def try_optimize(mu, cov, target_vol):

    try:
        ef = EfficientFrontier(mu, cov)

        ef.add_constraint(lambda w: w >= MIN_WEIGHT)

        ef.add_constraint(lambda w: w <= MAX_WEIGHT)

        ef.efficient_risk(target_volatility=target_vol)

        return ef.clean_weights()

    except:
        return None


# ===============================
# VOLATILITY SWEEP
# ===============================
def run_volatility_sweep(
    tickers, mu, S, pi, investor_views, omega, market_weights, delta, data_bt
):

    results = []

    returns_bt = data_bt.pct_change().dropna()

    for vol in VOL_RANGE:

        try:
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

        except:
            continue

        weights = try_optimize(bl_mu, bl_cov, vol)

        if weights is None:
            continue

        # w = np.array([weights[t] for t in tickers])

        # port_ret = returns_bt.dot(w)

        # total_return = float((1 + port_ret).prod() - 1)

        # results.append({"vol": vol, "weights": weights, "return": total_return})

        # Upgraded volatility sweep

        # BL return
        w_bl = np.array([weights[t] for t in tickers])

        ret_bl = returns_bt.dot(w_bl)

        bl_return = float((1 + ret_bl).prod() - 1)

        # MVO at same volatility
        weights_mvo = try_optimize(mu, S, vol)

        if weights_mvo is None:
            continue

        w_mvo = np.array([weights_mvo[t] for t in tickers])

        ret_mvo = returns_bt.dot(w_mvo)

        mvo_return = float((1 + ret_mvo).prod() - 1)

        # objective = excess return over MVO
        excess_over_mvo = bl_return - mvo_return

        results.append(
            {
                "vol": vol,
                "weights": weights,
                "bl_return": bl_return,
                "mvo_return": mvo_return,
                "excess_over_mvo": excess_over_mvo,
            }
        )

    if len(results) == 0:
        raise RuntimeError("No feasible volatility found")

    # best = max(results, key=lambda x: x["return"])

    best = max(results, key=lambda x: x["excess_over_mvo"])

    return best


def compute_post_backtest_allocation(
    tickers,
    market_rep,
    start_date,
    post_bt_end,
    selected_vol,
    forward_days=20,
    model_type="XGBoost",
):
    # import numpy as np
    # from pypfopt import expected_returns, risk_models, black_litterman
    # from pypfopt.black_litterman import market_implied_risk_aversion

    # ===============================
    # DATA
    # ===============================
    price_data = fetch_price_data(tickers, start_date, post_bt_end)
    market_data = fetch_price_data(market_rep, start_date, post_bt_end)

    # ===============================
    # CORE INPUTS
    # ===============================
    mu = expected_returns.mean_historical_return(price_data)
    S = risk_models.sample_cov(price_data)
    delta = market_implied_risk_aversion(market_data)

    pi = expected_returns.capm_return(price_data, market_prices=market_data)

    # ===============================
    # ML VIEWS
    # ===============================
    investor_views, view_conf = generate_views(
        tickers, start_date, post_bt_end, model_type, forward_days
    )

    omega_diag = [
        1.0 / view_conf[t] if view_conf[t] != 0 else 1e-4 for t in investor_views
    ]
    omega = np.diag(omega_diag)

    market_weights = {t: 1 / len(tickers) for t in tickers}

    # ===============================
    # BLACK-LITTERMAN
    # ===============================
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

    # ===============================
    # OPTIMIZATION
    # ===============================
    weights = try_optimize(bl_mu, bl_cov, selected_vol)

    if weights is None:
        ef = EfficientFrontier(bl_mu, bl_cov)
        ef.add_constraint(lambda w: w >= MIN_WEIGHT)
        ef.add_constraint(lambda w: w <= MAX_WEIGHT)
        ef.max_sharpe()
        weights = ef.clean_weights()

    return weights


# ===============================
# MAIN PIPELINE
# ===============================
def full_pipeline(
    tickers,
    allocations,
    market_rep,
    start_date,
    end_date,
    backtest_start,
    backtest_end,
    post_bt_end,
    forward_days=20,
    model_type="XGBoost",
):

    # TRAINING DATA
    price_data = fetch_price_data(tickers, start_date, end_date)

    market_data = fetch_price_data(market_rep, start_date, end_date)

    mu = expected_returns.mean_historical_return(price_data)

    S = risk_models.sample_cov(price_data)

    delta = market_implied_risk_aversion(market_data)

    pi = expected_returns.capm_return(price_data, market_prices=market_data)

    # ML VIEWS
    investor_views, view_confidences = generate_views(
        tickers, start_date, end_date, model_type, forward_days
    )

    omega_diag = [
        1.0 / view_confidences[t] if view_confidences[t] != 0 else 1e-4
        for t in investor_views
    ]

    omega = np.diag(omega_diag)

    market_weights = {t: 1 / len(tickers) for t in tickers}

    # BACKTEST DATA
    data_bt = fetch_price_data(tickers, backtest_start, backtest_end)

    # VOLATILITY SWEEP
    best = run_volatility_sweep(
        tickers, mu, S, pi, investor_views, omega, market_weights, delta, data_bt
    )

    selected_vol = best["vol"]
    weights_bl = best["weights"]

    # MVO uses SAME chosen vol
    weights_mvo = try_optimize(mu, S, selected_vol)

    if weights_mvo is None:

        ef = EfficientFrontier(mu, S)

        ef.add_constraint(lambda w: w >= MIN_WEIGHT)

        ef.add_constraint(lambda w: w <= MAX_WEIGHT)

        ef.max_sharpe()

        weights_mvo = ef.clean_weights()

    weights_unopt = allocations

    # BACKTEST
    ret_bl, cum_bl = run_backtest(data_bt, weights_bl, tickers)

    ret_mvo, cum_mvo = run_backtest(data_bt, weights_mvo, tickers)

    ret_un, cum_un = run_backtest(data_bt, weights_unopt, tickers)

    benchmark = fetch_price_data(market_rep, backtest_start, backtest_end)

    benchmark_cum = (1 + benchmark.pct_change()).cumprod()

    # CHART DATA
    chart_df = pd.DataFrame(
        {
            "ML Black-Litterman": cum_bl,
            "MVO": cum_mvo,
            "Unoptimized": cum_un,
            market_rep[0]: benchmark_cum[market_rep[0]],
        }
    ).dropna()

    # METRICS
    labels = ["ML Black-Litterman", "MVO", "Unoptimized"]

    metrics = compute_metrics([ret_bl, ret_mvo, ret_un], labels)

    # RESULTS
    # results = {
    #     "selected_target_volatility": float(selected_vol),
    #     "fixed_risk_free_rate": RISK_FREE_RATE,
    #     "portfolio_weights": {
    #         "unoptimized": weights_unopt,
    #         "mvo": weights_mvo,
    #         "ml_black_litterman": weights_bl,
    #     },
    #     "performance_metrics": metrics,
    #     "chart_data": chart_df,
    #     "views": investor_views,
    #     "view_confidences": view_confidences,
    # }

    post_weights = compute_post_backtest_allocation(
        tickers,
        market_rep,
        start_date,
        post_bt_end,
        selected_vol,
        forward_days,
        model_type,
    )

    results = {
        "portfolio_weights": {
            "unoptimized": weights_unopt,
            "mvo": weights_mvo,
            "ml_black_litterman": weights_bl,
        },
        "post_backtest_weights": post_weights,
        "performance_metrics": metrics,
        "selected_target_volatility": float(selected_vol),
        "fixed_risk_free_rate": RISK_FREE_RATE,
        "chart_data": chart_df,
        "views": investor_views,
        "view_confidences": view_confidences,
    }

    return results
