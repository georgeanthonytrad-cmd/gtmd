import os
import requests
import pandas as pd
import numpy as np
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None

# -----------------------------
# SETUP
# -----------------------------
st.set_page_config(page_title="Live Stock Scoring Dashboard", layout="wide")

# Add your FMP API key in Streamlit Secrets:
# FMP_API_KEY = "your_key"
FMP_API_KEY = st.secrets.get("FMP_API_KEY", os.getenv("FMP_API_KEY", ""))

FMP_V3 = "https://financialmodelingprep.com/api/v3"


# -----------------------------
# HELPERS
# -----------------------------
def safe_float(x, default=np.nan):
    try:
        if x is None or x == "":
            return default
        if isinstance(x, str):
            x = x.replace("%", "").replace(",", "").strip()
        return float(x)
    except Exception:
        return default


def fmp_get(path, params=None):
    if not FMP_API_KEY:
        return None

    params = params or {}
    params["apikey"] = FMP_API_KEY

    try:
        response = requests.get(f"{FMP_V3}/{path}", params=params, timeout=20)
        if response.status_code != 200:
            st.sidebar.warning(f"FMP error {response.status_code} on {path}")
            return None
        return response.json()
    except Exception as e:
        st.sidebar.warning(f"FMP request failed: {e}")
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
def get_fmp_quote(symbol):
    data = fmp_get(f"quote/{symbol}")
    if isinstance(data, list) and len(data) > 0:
        return data[0]
    return {}


def get_fmp_history(symbol):
    data = fmp_get(f"historical-price-full/{symbol}", {"serietype": "line"})
    hist = data.get("historical") if isinstance(data, dict) else None

    if not hist:
        return pd.DataFrame()

    df = pd.DataFrame(hist)

    if df.empty or "date" not in df or "close" not in df:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    return df.sort_values("date").dropna(subset=["close"])


def get_fmp_ratios(symbol):
    data = fmp_get(f"ratios-ttm/{symbol}")
    if isinstance(data, list) and data:
        return data[0]
    return {}


def get_fmp_metrics(symbol):
    data = fmp_get(f"key-metrics-ttm/{symbol}")
    if isinstance(data, list) and data:
        return data[0]
    return {}


def get_fmp_growth(symbol):
    data = fmp_get(f"income-statement-growth/{symbol}", {
        "period": "annual",
        "limit": 1
    })
    if isinstance(data, list) and data:
        return data[0]
    return {}


def get_fmp_analyst_target(symbol):
    data = fmp_get(f"price-target-summary/{symbol}")
    if isinstance(data, list) and data:
        return data[0]
    if isinstance(data, dict):
        return data
    return {}


def get_yfinance_backup(symbol):
    if yf is None:
        return {}, pd.DataFrame()

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.get_info() or {}
        hist = ticker.history(period="1y", interval="1d")

        if hist is None or hist.empty:
            return info, pd.DataFrame()

        hist = hist.reset_index()
        hist.columns = [str(c).lower().replace(" ", "_") for c in hist.columns]

        if "date" not in hist.columns:
            hist.rename(columns={hist.columns[0]: "date"}, inplace=True)

        if "close" not in hist.columns:
            return info, pd.DataFrame()

        hist["date"] = pd.to_datetime(hist["date"])
        return info, hist[["date", "close"]].copy()

    except Exception:
        return {}, pd.DataFrame()


# -----------------------------
# SCORING MODEL — 100 POINTS
# -----------------------------
def build_metrics(symbol, tsp_yes, sr_yes):
    quote = get_fmp_quote(symbol)
    prices = get_fmp_history(symbol)
    yf_info = {}

    # Yahoo fallback if FMP price/history fails.
    if prices.empty or not quote:
        yf_info, yf_prices = get_yfinance_backup(symbol)
        if prices.empty:
            prices = yf_prices

    ratios = get_fmp_ratios(symbol)
    metrics = get_fmp_metrics(symbol)
    growth = get_fmp_growth(symbol)
    target = get_fmp_analyst_target(symbol)

    current_price = safe_float(
        quote.get("price"),
        safe_float(yf_info.get("currentPrice"), safe_float(yf_info.get("regularMarketPrice")))
    )

    if np.isnan(current_price) and not prices.empty:
        current_price = safe_float(prices["close"].iloc[-1])

    if not prices.empty:
        prices = prices.copy()
        prices["rsi"] = calculate_rsi(prices["close"])
        prices["sma50"] = prices["close"].rolling(50).mean()
        prices["sma200"] = prices["close"].rolling(200).mean()

        last = prices.iloc[-1]
        close = safe_float(last["close"])

        change_5d = (close / prices["close"].iloc[-6] - 1) * 100 if len(prices) > 6 else np.nan
        change_1m = (close / prices["close"].iloc[-22] - 1) * 100 if len(prices) > 22 else np.nan
        rsi = safe_float(last["rsi"])
        dist_50 = (close / last["sma50"] - 1) * 100 if not np.isnan(safe_float(last["sma50"])) else np.nan
        dist_200 = (close / last["sma200"] - 1) * 100 if not np.isnan(safe_float(last["sma200"])) else np.nan
    else:
        change_5d = change_1m = rsi = dist_50 = dist_200 = np.nan

    ttm_pe = safe_float(quote.get("pe"), safe_float(yf_info.get("trailingPE")))
    forward_pe = safe_float(yf_info.get("forwardPE"), safe_float(metrics.get("forwardPE")))
    peg = safe_float(yf_info.get("pegRatio"), safe_float(ratios.get("priceEarningsToGrowthRatioTTM")))

    eps_raw = safe_float(growth.get("growthEPS"))
    rev_raw = safe_float(growth.get("growthRevenue"))

    eps_growth = eps_raw * 100 if not np.isnan(eps_raw) and abs(eps_raw) < 5 else eps_raw
    revenue_growth = rev_raw * 100 if not np.isnan(rev_raw) and abs(rev_raw) < 5 else rev_raw

    if np.isnan(eps_growth):
        eps_growth = safe_float(yf_info.get("earningsQuarterlyGrowth")) * 100

    if np.isnan(revenue_growth):
        revenue_growth = safe_float(yf_info.get("revenueGrowth")) * 100

    avg_target = safe_float(
        target.get("priceTargetAverage"),
        safe_float(yf_info.get("targetMeanPrice"))
    )

    analyst_upside = (
        ((avg_target / current_price) - 1) * 100
        if current_price and avg_target and not np.isnan(avg_target)
        else np.nan
    )

    forward_pe_less_than_ttm = (
        bool(forward_pe < ttm_pe)
        if not np.isnan(forward_pe) and not np.isnan(ttm_pe)
        else False
    )

    peg_under_1_2 = bool(peg < 1.2) if not np.isnan(peg) else False

    values = {
        "Current Price": current_price,
        "TTM P/E": ttm_pe,
        "Forward P/E": forward_pe,
        "Forward P/E < TTM P/E": forward_pe_less_than_ttm,
        "PEG": peg,
        "PEG < 1.2": peg_under_1_2,
        "EPS Growth YoY %": eps_growth,
        "Revenue Growth YoY %": revenue_growth,
        "5D Price Change %": change_5d,
        "1M Price Change %": change_1m,
        "RSI": rsi,
        "Distance from 50D MA %": dist_50,
        "Distance from 200D MA %": dist_200,
        "Analyst Target Upside %": analyst_upside,
        "TSP": tsp_yes,
        "SR": sr_yes,
        "Data Source": "FMP first, Yahoo fallback",
    }

    score = 0
    details = []

    # -----------------------------
    # FUNDAMENTALS — 30 POINTS
    # -----------------------------
    pts = 12 if eps_growth > 15 else 6 if eps_growth > 0 else 0
    score += pts
    details.append(("EPS growth YoY", pts, eps_growth, "Max 12"))

    pts = 10 if revenue_growth > 10 else 5 if revenue_growth > 0 else 0
    score += pts
    details.append(("Revenue growth YoY", pts, revenue_growth, "Max 10"))

    pts = 8 if analyst_upside > 10 else 4 if analyst_upside > 0 else 0
    score += pts
    details.append(("Analyst target upside", pts, analyst_upside, "Max 8"))

    # -----------------------------
    # VALUATION — 25 POINTS
    # -----------------------------
    pts = 8 if forward_pe_less_than_ttm else 0
    score += pts
    details.append(("Forward P/E < TTM P/E", pts, forward_pe_less_than_ttm, "Max 8"))

    pts = 10 if peg_under_1_2 else 5 if not np.isnan(peg) and peg < 1.8 else 0
    score += pts
    details.append(("PEG", pts, peg, "Max 10"))

    pts = 7 if not np.isnan(ttm_pe) and 0 < ttm_pe < 35 else 3 if not np.isnan(ttm_pe) and 35 <= ttm_pe < 60 else 0
    score += pts
    details.append(("P/E ratio reasonableness", pts, ttm_pe, "Max 7"))

    # -----------------------------
    # MOMENTUM / TECHNICALS — 25 POINTS
    # -----------------------------
    pts = score_range(change_5d, [
        (lambda x: -8 <= x <= -3, 5),   # dip-buy zone
        (lambda x: -3 < x <= 3, 3),     # stable
        (lambda x: x > 8, -2),          # possibly extended
    ])
    score += pts
    details.append(("5-day price change", pts, change_5d, "Max 5"))

    pts = 5 if not np.isnan(change_1m) and change_1m > 0 else 3 if not np.isnan(change_1m) and -5 <= change_1m <= 0 else 0
    score += pts
    details.append(("1-month price change", pts, change_1m, "Max 5"))

    pts = score_range(rsi, [
        (lambda x: 30 <= x <= 45, 6),
        (lambda x: 45 < x <= 60, 4),
        (lambda x: x < 30, 3),
        (lambda x: x > 70, -4),
    ])
    score += pts
    details.append(("RSI", pts, rsi, "Max 6"))

    pts = 5 if not np.isnan(dist_50) and -5 <= dist_50 <= 2 else 3 if not np.isnan(dist_50) and -10 <= dist_50 < -5 else 0
    score += pts
    details.append(("Distance from 50D MA", pts, dist_50, "Max 5"))

    pts = 4 if not np.isnan(dist_200) and dist_200 > 0 else -6 if not np.isnan(dist_200) and dist_200 < 0 else 0
    score += pts
    details.append(("Distance from 200D MA", pts, dist_200, "Max 4"))

    # -----------------------------
    # SETUP CONFIRMATION — 20 POINTS
    # -----------------------------
    pts = 10 if tsp_yes else 0
    score += pts
    details.append(("TSP", pts, tsp_yes, "Max 10"))

    pts = 10 if sr_yes else 0
    score += pts
    details.append(("SR", pts, sr_yes, "Max 10"))

    # Cap score between 0 and 100 because some technical penalties can subtract points.
    score = max(0, min(100, score))

    detail_df = pd.DataFrame(details, columns=["Factor", "Points", "Value", "Max Points"])

    return values, score, detail_df, prices


def rating(score):
    if score >= 85:
        return "STRONG SETUP"
    if score >= 70:
        return "GOOD SETUP"
    if score >= 55:
        return "WATCHLIST / WAIT"
    return "AVOID / WEAK SETUP"


# -----------------------------
# UI
# -----------------------------
st.title("Live Stock Scoring Dashboard")
st.caption("Educational stock scoring tool only. Not financial advice.")

with st.sidebar:
    st.header("Input")
    symbol = st.text_input("Ticker", value="META").upper().strip()
    tsp_yes = st.checkbox("TSP = Yes", value=False)
    sr_yes = st.checkbox("SR = Yes", value=False)
    run = st.button("Analyze")

    st.divider()
    st.caption("API Key Status")
    st.write("FMP:", "✅ Connected" if FMP_API_KEY else "❌ Missing")
    st.write("Yahoo backup:", "✅ Available" if yf is not None else "❌ Missing yfinance")

if run and symbol:
    with st.spinner(f"Pulling live data for {symbol}..."):
        values, total_score, detail_df, prices = build_metrics(symbol, tsp_yes, sr_yes)

    col1, col2, col3 = st.columns(3)
    col1.metric("Ticker", symbol)
    col2.metric("Score / 100", f"{total_score:.0f}")
    col3.metric("Rating", rating(total_score))

    if np.isnan(safe_float(values["Current Price"])):
        st.error("No price data loaded. Check ticker spelling, FMP key, or Yahoo backup.")
        st.info('Streamlit → App settings → Secrets → add: FMP_API_KEY = "your_key"')
    else:
        st.success("Price and technical indicators loaded.")

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
    if total_score >= 85:
        st.success("Strong setup. Fundamentals, valuation, technicals, and setup confirmation are aligned.")
    elif total_score >= 70:
        st.info("Good setup. Consider waiting for clean entry confirmation if TSP/SR are not both checked.")
    elif total_score >= 55:
        st.warning("Watchlist only. Some factors are positive, but the full setup is not strong yet.")
    else:
        st.error("Weak setup. Avoid unless there is a separate catalyst or reversal confirmation.")
else:
    st.info("Enter a ticker and click Analyze.")
