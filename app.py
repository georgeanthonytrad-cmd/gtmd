import os
import math
import requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta

# -----------------------------
# SETUP
# -----------------------------
st.set_page_config(page_title="Stock Scoring Dashboard", layout="wide")

# Add your API keys in Streamlit secrets:
# FMP_API_KEY = "your_key"
# TRADIER_TOKEN = "your_token"

FMP_API_KEY = st.secrets.get("FMP_API_KEY", os.getenv("FMP_API_KEY", ""))
TRADIER_TOKEN = st.secrets.get("TRADIER_TOKEN", os.getenv("TRADIER_TOKEN", ""))

BASE_FMP = "https://financialmodelingprep.com/stable"
TRADIER_BASE = "https://api.tradier.com/v1"

# -----------------------------
# HELPERS
# -----------------------------
def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def fmp_get(endpoint, params=None):
    if not FMP_API_KEY:
        return None
    params = params or {}
    params["apikey"] = FMP_API_KEY
    url = f"{BASE_FMP}/{endpoint}"
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def tradier_get(endpoint, params=None):
    if not TRADIER_TOKEN:
        return None
    headers = {
        "Authorization": f"Bearer {TRADIER_TOKEN}",
        "Accept": "application/json"
    }
    try:
        r = requests.get(f"{TRADIER_BASE}/{endpoint}", headers=headers, params=params or {}, timeout=15)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def score_range(value, rules):
    for condition, points in rules:
        try:
            if condition(value):
                return points
        except Exception:
            pass
    return 0

# -----------------------------
# DATA FUNCTIONS
# -----------------------------
def get_price_history(symbol):
    data = fmp_get("historical-price-eod/full", {"symbol": symbol})
    if not data or not isinstance(data, list):
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if df.empty or "date" not in df or "close" not in df:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.dropna(subset=["close"])


def get_quote(symbol):
    data = fmp_get("quote", {"symbol": symbol})
    if isinstance(data, list) and data:
        return data[0]
    return {}


def get_ratios(symbol):
    data = fmp_get("ratios-ttm", {"symbol": symbol})
    if isinstance(data, list) and data:
        return data[0]
    return {}


def get_key_metrics(symbol):
    data = fmp_get("key-metrics-ttm", {"symbol": symbol})
    if isinstance(data, list) and data:
        return data[0]
    return {}


def get_income_growth(symbol):
    data = fmp_get("income-statement-growth", {"symbol": symbol, "period": "annual", "limit": 1})
    if isinstance(data, list) and data:
        return data[0]
    return {}


def get_analyst_target(symbol):
    data = fmp_get("price-target-summary", {"symbol": symbol})
    if isinstance(data, list) and data:
        return data[0]
    if isinstance(data, dict):
        return data
    return {}


def get_earnings(symbol):
    today = datetime.utcnow().date()
    start = today - timedelta(days=30)
    end = today + timedelta(days=90)
    data = fmp_get("earnings-calendar", {
        "symbol": symbol,
        "from": start.isoformat(),
        "to": end.isoformat()
    })
    if isinstance(data, list) and data:
        df = pd.DataFrame(data)
        if "date" in df:
            df["date"] = pd.to_datetime(df["date"]).dt.date
            future = df[df["date"] >= today].sort_values("date")
            if not future.empty:
                return future.iloc[0].to_dict()
            return df.sort_values("date", ascending=False).iloc[0].to_dict()
    return {}


def get_options_summary(symbol):
    expirations = tradier_get("markets/options/expirations", {
        "symbol": symbol,
        "includeAllRoots": "true",
        "strikes": "false"
    })
    result = {"put_call_ratio": np.nan, "avg_iv": np.nan, "gamma_exposure_proxy": np.nan}
    try:
        dates = expirations["expirations"]["date"]
        if not dates:
            return result

        exp = dates[0] if isinstance(dates, list) else dates

        chain = tradier_get("markets/options/chains", {
            "symbol": symbol,
            "expiration": exp,
            "greeks": "true"
        })

        options = chain.get("options", {}).get("option", [])
        if isinstance(options, dict):
            options = [options]

        df = pd.DataFrame(options)
        if df.empty:
            return result

        df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0)

        puts = df[df["option_type"] == "put"]["volume"].sum()
        calls = df[df["option_type"] == "call"]["volume"].sum()
        result["put_call_ratio"] = puts / calls if calls > 0 else np.nan

        ivs = []
        gamma_proxy = 0

        for _, row in df.iterrows():
            greeks = row.get("greeks", {})
            if isinstance(greeks, dict):
                iv = safe_float(greeks.get("mid_iv", greeks.get("smv_vol")))
                gamma = safe_float(greeks.get("gamma"), 0)
                oi = safe_float(row.get("open_interest"), 0)

                if not np.isnan(iv):
                    ivs.append(iv)

                gamma_proxy += gamma * oi * 100

        result["avg_iv"] = np.nanmean(ivs) if ivs else np.nan
        result["gamma_exposure_proxy"] = gamma_proxy

    except Exception:
        pass

    return result

# -----------------------------
# SCORING
# -----------------------------
def build_metrics(symbol, tsp_yes, sr_yes):
    quote = get_quote(symbol)
    ratios = get_ratios(symbol)
    metrics = get_key_metrics(symbol)
    growth = get_income_growth(symbol)
    target = get_analyst_target(symbol)
    earnings = get_earnings(symbol)
    options = get_options_summary(symbol)
    prices = get_price_history(symbol)

    current_price = safe_float(quote.get("price"))
    if np.isnan(current_price) and not prices.empty:
        current_price = prices["close"].iloc[-1]

    if not prices.empty:
        prices["rsi"] = calculate_rsi(prices["close"])
        prices["sma50"] = prices["close"].rolling(50).mean()
        prices["sma200"] = prices["close"].rolling(200).mean()

        last = prices.iloc[-1]
        close = last["close"]

        change_5d = (close / prices["close"].iloc[-6] - 1) * 100 if len(prices) > 6 else np.nan
        change_1m = (close / prices["close"].iloc[-22] - 1) * 100 if len(prices) > 22 else np.nan
        rsi = last["rsi"]
        dist_50 = (close / last["sma50"] - 1) * 100 if not np.isnan(last["sma50"]) else np.nan
        dist_200 = (close / last["sma200"] - 1) * 100 if not np.isnan(last["sma200"]) else np.nan
    else:
        change_5d = change_1m = rsi = dist_50 = dist_200 = np.nan

    ttm_pe = safe_float(quote.get("pe", ratios.get("priceEarningsRatioTTM")))
    forward_pe = safe_float(quote.get("forwardPE", metrics.get("forwardPE")))
    peg = safe_float(ratios.get("priceEarningsToGrowthRatioTTM", metrics.get("pegRatioTTM")))

    eps_raw = safe_float(growth.get("growthEPS"))
    rev_raw = safe_float(growth.get("growthRevenue"))

    eps_growth = eps_raw * 100 if abs(eps_raw) < 5 else eps_raw
    revenue_growth = rev_raw * 100 if abs(rev_raw) < 5 else rev_raw

    avg_target = safe_float(target.get("priceTargetAverage", target.get("targetMeanPrice")))
    analyst_upside = ((avg_target / current_price) - 1) * 100 if current_price and avg_target and not np.isnan(avg_target) else np.nan

    earnings_date = earnings.get("date", "N/A")
    days_to_earnings = None
    try:
        days_to_earnings = (pd.to_datetime(earnings_date).date() - datetime.utcnow().date()).days
    except Exception:
        pass

    values = {
        "Current Price": current_price,
        "TTM P/E": ttm_pe,
        "Forward P/E": forward_pe,
        "Forward P/E < TTM P/E": bool(forward_pe < ttm_pe) if not np.isnan(forward_pe) and not np.isnan(ttm_pe) else False,
        "PEG": peg,
        "PEG < 1.2": bool(peg < 1.2) if not np.isnan(peg) else False,
        "EPS Growth YoY %": eps_growth,
        "Revenue Growth YoY %": revenue_growth,
        "5D Price Change %": change_5d,
        "1M Price Change %": change_1m,
        "RSI": rsi,
        "Distance from 50D MA %": dist_50,
        "Distance from 200D MA %": dist_200,
        "Put/Call Ratio": options["put_call_ratio"],
        "Average IV": options["avg_iv"],
        "Gamma Exposure Proxy": options["gamma_exposure_proxy"],
        "Analyst Target Upside %": analyst_upside,
        "Earnings Date": earnings_date,
        "Days to Earnings": days_to_earnings,
        "TSP": tsp_yes,
        "SR": sr_yes,
    }

    score = 0
    details = []

    # Fundamentals: 30
    pts = 10 if eps_growth > 15 else 0
    score += pts
    details.append(("EPS growth YoY > 15%", pts, eps_growth))

    pts = 8 if revenue_growth > 10 else 4 if revenue_growth > 0 else 0
    score += pts
    details.append(("Revenue growth YoY", pts, revenue_growth))

    pts = 6 if analyst_upside > 10 else 3 if analyst_upside > 0 else 0
    score += pts
    details.append(("Analyst target upside", pts, analyst_upside))

    pts = 6 if eps_growth > 0 and revenue_growth > 0 else 0
    score += pts
    details.append(("Positive earnings/revenue trend", pts, None))

    # Valuation: 20
    pts = 6 if values["Forward P/E < TTM P/E"] else 0
    score += pts
    details.append(("Forward P/E < TTM P/E", pts, None))

    pts = 8 if values["PEG < 1.2"] else 4 if not np.isnan(peg) and peg < 1.8 else 0
    score += pts
    details.append(("PEG < 1.2", pts, peg))

    pts = 6 if not np.isnan(ttm_pe) and 0 < ttm_pe < 35 else 3 if not np.isnan(ttm_pe) and 35 <= ttm_pe < 60 else 0
    score += pts
    details.append(("P/E reasonable", pts, ttm_pe))

    # Momentum/Technicals: 20
    pts = score_range(change_5d, [
        (lambda x: -8 <= x <= -3, 4),
        (lambda x: -3 < x <= 3, 2),
        (lambda x: x > 8, -2),
    ])
    score += pts
    details.append(("5-day price change", pts, change_5d))

    pts = 4 if not np.isnan(change_1m) and change_1m > 0 else 2 if not np.isnan(change_1m) and -5 <= change_1m <= 0 else 0
    score += pts
    details.append(("1-month price change", pts, change_1m))

    pts = score_range(rsi, [
        (lambda x: 30 <= x <= 45, 4),
        (lambda x: 45 < x <= 60, 3),
        (lambda x: x < 30, 2),
        (lambda x: x > 70, -4),
    ])
    score += pts
    details.append(("RSI", pts, rsi))

    pts = 4 if not np.isnan(dist_50) and -5 <= dist_50 <= 2 else 2 if not np.isnan(dist_50) and -10 <= dist_50 < -5 else 0
    score += pts
    details.append(("Distance from 50D MA", pts, dist_50))

    pts = 4 if not np.isnan(dist_200) and dist_200 > 0 else -6 if not np.isnan(dist_200) and dist_200 < 0 else 0
    score += pts
    details.append(("Distance from 200D MA", pts, dist_200))

    # Sentiment/Options: 20
    pcr = options["put_call_ratio"]
    pts = 5 if not np.isnan(pcr) and pcr > 1.1 else -5 if not np.isnan(pcr) and pcr < 0.6 else 0
    score += pts
    details.append(("Put/Call Ratio", pts, pcr))

    iv = options["avg_iv"]
    pts = 5 if not np.isnan(iv) and iv > 0.60 else 3 if not np.isnan(iv) and iv < 0.30 else 0
    score += pts
    details.append(("Implied Volatility", pts, iv))

    gex = options["gamma_exposure_proxy"]
    pts = 4 if not np.isnan(gex) and gex > 0 else -4 if not np.isnan(gex) and gex < 0 else 0
    score += pts
    details.append(("Gamma exposure proxy", pts, gex))

    pts = 6 if not np.isnan(iv) and iv > 0.60 and not np.isnan(change_5d) and change_5d < -3 else 0
    score += pts
    details.append(("IV spike + price drop", pts, None))

    # Event risk: 10
    if days_to_earnings is not None and 0 <= days_to_earnings <= 7:
        pts = -6
    elif days_to_earnings is not None and -7 <= days_to_earnings < 0:
        pts = 6
    else:
        pts = 4
    score += pts
    details.append(("Earnings timing", pts, days_to_earnings))

    # TSP/SR: 10
    pts = 5 if tsp_yes else 0
    score += pts
    details.append(("TSP", pts, tsp_yes))

    pts = 5 if sr_yes else 0
    score += pts
    details.append(("SR", pts, sr_yes))

    return values, score, pd.DataFrame(details, columns=["Factor", "Points", "Value"]), prices


def rating(score):
    if score >= 90:
        return "STRONG BUY ZONE"
    if score >= 75:
        return "GOOD SETUP"
    if score >= 60:
        return "WATCHLIST / WAIT"
    return "AVOID / WEAK SETUP"

# -----------------------------
# UI
# -----------------------------
st.title("Live Stock Scoring Dashboard")
st.caption("Educational screening tool only. Not financial advice.")

with st.sidebar:
    st.header("Input")
    symbol = st.text_input("Ticker", value="META").upper().strip()
    tsp_yes = st.checkbox("TSP = Yes", value=False)
    sr_yes = st.checkbox("SR = Yes", value=False)
    run = st.button("Analyze")

if run and symbol:
    values, total_score, detail_df, prices = build_metrics(symbol, tsp_yes, sr_yes)

    col1, col2, col3 = st.columns(3)
    col1.metric("Ticker", symbol)
    col2.metric("Score / 110", f"{total_score:.0f}")
    col3.metric("Rating", rating(total_score))

    st.subheader("Key Metrics")
    metric_df = pd.DataFrame([values]).T.reset_index()
    metric_df.columns = ["Metric", "Value"]
    st.dataframe(metric_df, use_container_width=True)

    st.subheader("Score Breakdown")
    st.dataframe(detail_df, use_container_width=True)

    if not prices.empty:
        st.subheader("Price Chart")
        chart_df = prices.tail(250).set_index("date")[["close"]]
        st.line_chart(chart_df)

    st.subheader("Decision Logic")
    if total_score >= 90:
        st.success("High-conviction setup. Consider stock entry or selling puts depending on IV and earnings timing.")
    elif total_score >= 75:
        st.info("Good setup. Look for confirmation with TSP/SR and avoid chasing before earnings.")
    elif total_score >= 60:
        st.warning("Watchlist only. Wait for better technical confirmation or valuation improvement.")
    else:
        st.error("Weak setup. Avoid unless there is a specific catalyst or reversal confirmation.")

else:
    st.info("Enter a ticker and click Analyze.")
