import os
import requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta

try:
    import yfinance as yf
except Exception:
    yf = None

st.set_page_config(page_title="Live Stock Scoring Dashboard", layout="wide")

# -----------------------------
# API KEYS
# Add these in Streamlit Secrets:
# FMP_API_KEY = "your_fmp_key"
# BARCHART_API_KEY = "your_barchart_key"
# TRADIER_TOKEN = "your_tradier_token"
# -----------------------------
FMP_API_KEY = st.secrets.get("FMP_API_KEY", os.getenv("FMP_API_KEY", ""))
BARCHART_API_KEY = st.secrets.get("BARCHART_API_KEY", os.getenv("BARCHART_API_KEY", ""))
TRADIER_TOKEN = st.secrets.get("TRADIER_TOKEN", os.getenv("TRADIER_TOKEN", ""))

FMP_V3 = "https://financialmodelingprep.com/api/v3"
BARCHART_BASE = "https://ondemand.websol.barchart.com"
TRADIER_BASE = "https://api.tradier.com/v1"


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


def find_first_number(data, possible_keys):
    """Search nested dict/list for first numeric value matching any possible key."""
    if isinstance(data, dict):
        lower_map = {str(k).lower(): k for k in data.keys()}
        for key in possible_keys:
            real_key = lower_map.get(key.lower())
            if real_key is not None:
                val = safe_float(data.get(real_key))
                if not np.isnan(val):
                    return val
        for value in data.values():
            found = find_first_number(value, possible_keys)
            if not np.isnan(found):
                return found
    elif isinstance(data, list):
        for item in data:
            found = find_first_number(item, possible_keys)
            if not np.isnan(found):
                return found
    return np.nan


def find_first_text(data, possible_keys):
    """Search nested dict/list for first text value matching any possible key."""
    if isinstance(data, dict):
        lower_map = {str(k).lower(): k for k in data.keys()}
        for key in possible_keys:
            real_key = lower_map.get(key.lower())
            if real_key is not None and data.get(real_key) not in [None, ""]:
                return str(data.get(real_key))
        for value in data.values():
            found = find_first_text(value, possible_keys)
            if found:
                return found
    elif isinstance(data, list):
        for item in data:
            found = find_first_text(item, possible_keys)
            if found:
                return found
    return ""


def fmp_get(path, params=None):
    if not FMP_API_KEY:
        return None
    params = params or {}
    params["apikey"] = FMP_API_KEY
    try:
        r = requests.get(f"{FMP_V3}/{path}", params=params, timeout=20)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def barchart_get(endpoint, params=None):
    """
    Barchart OnDemand endpoint example:
    https://ondemand.websol.barchart.com/getEquityOptionsOverviewSummary.json?apikey=KEY&symbols=META
    """
    if not BARCHART_API_KEY:
        return None

    params = params or {}
    params["apikey"] = BARCHART_API_KEY

    # endpoint should NOT include .json
    endpoint = endpoint.replace(".json", "")
    url = f"{BARCHART_BASE}/{endpoint}.json"

    try:
        r = requests.get(url, params=params, timeout=20)
        if r.status_code != 200:
            return {"_error": f"Barchart HTTP {r.status_code}: {r.text[:200]}"}
        return r.json()
    except Exception as e:
        return {"_error": f"Barchart request failed: {e}"}


def tradier_get(endpoint, params=None):
    if not TRADIER_TOKEN:
        return None
    headers = {"Authorization": f"Bearer {TRADIER_TOKEN}", "Accept": "application/json"}
    try:
        r = requests.get(f"{TRADIER_BASE}/{endpoint}", headers=headers, params=params or {}, timeout=20)
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
# PRICE / FUNDAMENTAL DATA
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
    data = fmp_get(f"income-statement-growth/{symbol}", {"period": "annual", "limit": 1})
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


def get_fmp_earnings(symbol):
    today = datetime.utcnow().date()
    start = today - timedelta(days=30)
    end = today + timedelta(days=120)
    data = fmp_get("earning_calendar", {
        "symbol": symbol,
        "from": start.isoformat(),
        "to": end.isoformat(),
    })
    if isinstance(data, list) and data:
        df = pd.DataFrame(data)
        if "date" in df:
            df["date"] = pd.to_datetime(df["date"]).dt.date
            future = df[df["date"] >= today].sort_values("date")
            if not future.empty:
                row = future.iloc[0].to_dict()
                row["_source"] = "FMP"
                return row
            row = df.sort_values("date", ascending=False).iloc[0].to_dict()
            row["_source"] = "FMP"
            return row
    return {}


def get_yfinance_backup(symbol):
    if yf is None:
        return {}, pd.DataFrame()
    try:
        t = yf.Ticker(symbol)
        info = t.get_info() or {}
        hist = t.history(period="1y", interval="1d")
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
# BARCHART DATA
# -----------------------------
def get_barchart_options_overview(symbol):
    """
    Attempts Barchart OnDemand options overview.
    Expected to cover put/call ratio, weighted IV, IV rank/percentile.
    Field names can vary by subscription/response, so this parser is flexible.
    """
    result = {
        "put_call_ratio": np.nan,
        "avg_iv": np.nan,
        "iv_rank": np.nan,
        "iv_percentile": np.nan,
        "gamma_exposure_proxy": np.nan,
        "status": "Barchart key missing or options overview unavailable",
        "raw_error": "",
    }

    if not BARCHART_API_KEY:
        return result

    data = barchart_get("getEquityOptionsOverviewSummary", {"symbols": symbol})

    if not data:
        result["status"] = "No Barchart response"
        return result

    if isinstance(data, dict) and data.get("_error"):
        result["status"] = "Barchart options request failed"
        result["raw_error"] = data.get("_error")
        return result

    # Many Barchart responses have status + results.
    if isinstance(data, dict) and data.get("status", {}).get("code") not in [None, 200]:
        result["status"] = f"Barchart status: {data.get('status')}"
        return result

    result["put_call_ratio"] = find_first_number(data, [
        "putCallRatio",
        "put_call_ratio",
        "putCallVolumeRatio",
        "volumePutCallRatio",
        "putCallOpenInterestRatio",
        "openInterestPutCallRatio",
        "pcRatio",
        "putCall",
    ])

    iv = find_first_number(data, [
        "weightedImpliedVolatility",
        "weighted_iv",
        "impliedVolatility",
        "iv",
        "avgIv",
        "averageIv",
        "averageImpliedVolatility",
    ])

    # If Barchart returns IV as 34.5, convert to decimal 0.345 for scoring consistency.
    if not np.isnan(iv) and iv > 3:
        iv = iv / 100.0
    result["avg_iv"] = iv

    result["iv_rank"] = find_first_number(data, [
        "ivRank",
        "impliedVolatilityRank",
        "volatilityRank",
    ])

    result["iv_percentile"] = find_first_number(data, [
        "ivPercentile",
        "impliedVolatilityPercentile",
        "volatilityPercentile",
    ])

    # True GEX may not be available through this endpoint. This remains a placeholder unless field exists.
    result["gamma_exposure_proxy"] = find_first_number(data, [
        "gammaExposure",
        "gamma_exposure",
        "gex",
        "gammaExposureProxy",
    ])

    result["status"] = "Barchart options overview loaded"
    return result


def get_barchart_earnings(symbol):
    """
    Barchart getEarningsCalendar works by date range/calendar.
    This loops upcoming dates and finds the requested symbol.
    """
    result = {
        "date": "N/A",
        "source": "",
        "status": "Barchart key missing or earnings unavailable",
    }

    if not BARCHART_API_KEY:
        return result

    today = datetime.utcnow().date()

    # Search next 120 days in chunks. This avoids relying on undocumented symbol filtering.
    for offset in range(0, 121, 7):
        start_date = today + timedelta(days=offset)
        data = barchart_get("getEarningsCalendar", {
            "type": "earnings",
            "startDate": start_date.isoformat(),
        })

        if not data or (isinstance(data, dict) and data.get("_error")):
            continue

        # Barchart commonly puts data in "results", but parser is flexible.
        rows = []
        if isinstance(data, dict):
            if isinstance(data.get("results"), list):
                rows = data.get("results")
            elif isinstance(data.get("data"), list):
                rows = data.get("data")
            else:
                # Try any list in response
                for v in data.values():
                    if isinstance(v, list):
                        rows = v
                        break
        elif isinstance(data, list):
            rows = data

        for row in rows:
            if not isinstance(row, dict):
                continue

            row_symbol = str(row.get("symbol", row.get("symbolName", row.get("ticker", "")))).upper()
            if row_symbol == symbol.upper():
                date_txt = find_first_text(row, ["date", "earningsDate", "reportDate"])
                if date_txt:
                    result["date"] = str(pd.to_datetime(date_txt).date())
                    result["source"] = "Barchart"
                    result["status"] = "Barchart earnings loaded"
                    return result

    return result


# -----------------------------
# TRADIER OPTIONS BACKUP
# -----------------------------
def get_tradier_options_summary(symbol):
    result = {
        "put_call_ratio": np.nan,
        "avg_iv": np.nan,
        "gamma_exposure_proxy": np.nan,
        "status": "Tradier key missing or options data unavailable",
    }
    if not TRADIER_TOKEN:
        return result

    try:
        expirations = tradier_get("markets/options/expirations", {
            "symbol": symbol,
            "includeAllRoots": "true",
            "strikes": "false",
        })
        dates = expirations.get("expirations", {}).get("date", []) if expirations else []
        if not dates:
            return result

        exp = dates[0] if isinstance(dates, list) else dates
        chain = tradier_get("markets/options/chains", {
            "symbol": symbol,
            "expiration": exp,
            "greeks": "true",
        })
        options = chain.get("options", {}).get("option", []) if chain else []
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

                # This is a simple proxy, not professional dealer GEX.
                gamma_proxy += gamma * oi * 100

        result["avg_iv"] = np.nanmean(ivs) if ivs else np.nan
        result["gamma_exposure_proxy"] = gamma_proxy
        result["status"] = "Tradier options chain loaded"
    except Exception as e:
        result["status"] = f"Tradier options unavailable: {e}"

    return result


def get_options_summary(symbol):
    """
    Barchart first, Tradier backup.
    """
    bc = get_barchart_options_overview(symbol)
    tr = get_tradier_options_summary(symbol)

    result = {
        "put_call_ratio": bc["put_call_ratio"],
        "avg_iv": bc["avg_iv"],
        "iv_rank": bc["iv_rank"],
        "iv_percentile": bc["iv_percentile"],
        "gamma_exposure_proxy": bc["gamma_exposure_proxy"],
        "options_status": bc["status"],
    }

    # Fill missing values from Tradier backup.
    if np.isnan(result["put_call_ratio"]):
        result["put_call_ratio"] = tr["put_call_ratio"]
    if np.isnan(result["avg_iv"]):
        result["avg_iv"] = tr["avg_iv"]
    if np.isnan(result["gamma_exposure_proxy"]):
        result["gamma_exposure_proxy"] = tr["gamma_exposure_proxy"]

    if "loaded" not in result["options_status"].lower() and "loaded" in tr["status"].lower():
        result["options_status"] = tr["status"]

    if bc.get("raw_error"):
        result["options_status"] += f" | {bc['raw_error']}"

    return result


# -----------------------------
# MAIN BUILD + SCORE
# -----------------------------
def build_metrics(symbol, tsp_yes, sr_yes):
    quote = get_fmp_quote(symbol)
    prices = get_fmp_history(symbol)
    yf_info = {}

    # Fallback to Yahoo if FMP does not load prices.
    if prices.empty or not quote:
        yf_info, yf_prices = get_yfinance_backup(symbol)
        if prices.empty:
            prices = yf_prices

    ratios = get_fmp_ratios(symbol)
    metrics = get_fmp_metrics(symbol)
    growth = get_fmp_growth(symbol)
    target = get_fmp_analyst_target(symbol)

    bc_earnings = get_barchart_earnings(symbol)
    fmp_earnings = get_fmp_earnings(symbol)
    options = get_options_summary(symbol)

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

    avg_target = safe_float(target.get("priceTargetAverage"), safe_float(yf_info.get("targetMeanPrice")))
    analyst_upside = ((avg_target / current_price) - 1) * 100 if current_price and avg_target and not np.isnan(avg_target) else np.nan

    # Earnings: Barchart first, FMP backup.
    earnings_date = bc_earnings.get("date", "N/A")
    earnings_source = bc_earnings.get("source", "")

    if earnings_date == "N/A":
        earnings_date = fmp_earnings.get("date", "N/A")
        earnings_source = fmp_earnings.get("_source", "FMP" if earnings_date != "N/A" else "")

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
        "IV Rank": options["iv_rank"],
        "IV Percentile": options["iv_percentile"],
        "Gamma Exposure Proxy": options["gamma_exposure_proxy"],
        "Analyst Target Upside %": analyst_upside,
        "Earnings Date": earnings_date,
        "Earnings Source": earnings_source,
        "Days to Earnings": days_to_earnings,
        "TSP": tsp_yes,
        "SR": sr_yes,
        "Options Data Status": options["options_status"],
        "Data Source": "FMP + Barchart first, Yahoo/Tradier backup",
    }

    score = 0
    details = []

    # Fundamentals: 30
    pts = 10 if eps_growth > 15 else 0
    score += pts; details.append(("EPS growth YoY > 15%", pts, eps_growth))

    pts = 8 if revenue_growth > 10 else 4 if revenue_growth > 0 else 0
    score += pts; details.append(("Revenue growth YoY", pts, revenue_growth))

    pts = 6 if analyst_upside > 10 else 3 if analyst_upside > 0 else 0
    score += pts; details.append(("Analyst target upside", pts, analyst_upside))

    pts = 6 if eps_growth > 0 and revenue_growth > 0 else 0
    score += pts; details.append(("Positive earnings/revenue trend", pts, None))

    # Valuation: 20
    pts = 6 if values["Forward P/E < TTM P/E"] else 0
    score += pts; details.append(("Forward P/E < TTM P/E", pts, None))

    pts = 8 if values["PEG < 1.2"] else 4 if not np.isnan(peg) and peg < 1.8 else 0
    score += pts; details.append(("PEG < 1.2", pts, peg))

    pts = 6 if not np.isnan(ttm_pe) and 0 < ttm_pe < 35 else 3 if not np.isnan(ttm_pe) and 35 <= ttm_pe < 60 else 0
    score += pts; details.append(("P/E reasonable", pts, ttm_pe))

    # Momentum/Technicals: 20
    pts = score_range(change_5d, [
        (lambda x: -8 <= x <= -3, 4),
        (lambda x: -3 < x <= 3, 2),
        (lambda x: x > 8, -2),
    ])
    score += pts; details.append(("5-day price change", pts, change_5d))

    pts = 4 if not np.isnan(change_1m) and change_1m > 0 else 2 if not np.isnan(change_1m) and -5 <= change_1m <= 0 else 0
    score += pts; details.append(("1-month price change", pts, change_1m))

    pts = score_range(rsi, [
        (lambda x: 30 <= x <= 45, 4),
        (lambda x: 45 < x <= 60, 3),
        (lambda x: x < 30, 2),
        (lambda x: x > 70, -4),
    ])
    score += pts; details.append(("RSI", pts, rsi))

    pts = 4 if not np.isnan(dist_50) and -5 <= dist_50 <= 2 else 2 if not np.isnan(dist_50) and -10 <= dist_50 < -5 else 0
    score += pts; details.append(("Distance from 50D MA", pts, dist_50))

    pts = 4 if not np.isnan(dist_200) and dist_200 > 0 else -6 if not np.isnan(dist_200) and dist_200 < 0 else 0
    score += pts; details.append(("Distance from 200D MA", pts, dist_200))

    # Sentiment/Options: 20
    pcr = options["put_call_ratio"]
    pts = 5 if not np.isnan(pcr) and pcr > 1.1 else -5 if not np.isnan(pcr) and pcr < 0.6 else 0
    score += pts; details.append(("Put/Call Ratio", pts, pcr))

    iv = options["avg_iv"]
    pts = 5 if not np.isnan(iv) and iv > 0.60 else 3 if not np.isnan(iv) and iv < 0.30 else 0
    score += pts; details.append(("Implied Volatility", pts, iv))

    gex = options["gamma_exposure_proxy"]
    pts = 4 if not np.isnan(gex) and gex > 0 else -4 if not np.isnan(gex) and gex < 0 else 0
    score += pts; details.append(("Gamma exposure proxy", pts, gex))

    pts = 6 if not np.isnan(iv) and iv > 0.60 and not np.isnan(change_5d) and change_5d < -3 else 0
    score += pts; details.append(("IV spike + price drop", pts, None))

    # Event risk: 10
    if days_to_earnings is not None and 0 <= days_to_earnings <= 7:
        pts = -6
    elif days_to_earnings is not None and -7 <= days_to_earnings < 0:
        pts = 6
    else:
        pts = 4
    score += pts; details.append(("Earnings timing", pts, days_to_earnings))

    # TSP/SR: 10
    pts = 5 if tsp_yes else 0
    score += pts; details.append(("TSP", pts, tsp_yes))

    pts = 5 if sr_yes else 0
    score += pts; details.append(("SR", pts, sr_yes))

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

    st.divider()
    st.caption("API key status")
    st.write("FMP:", "✅" if FMP_API_KEY else "❌")
    st.write("Barchart:", "✅" if BARCHART_API_KEY else "❌")
    st.write("Tradier:", "✅" if TRADIER_TOKEN else "❌")

if run and symbol:
    with st.spinner(f"Pulling live data for {symbol}..."):
        values, total_score, detail_df, prices = build_metrics(symbol, tsp_yes, sr_yes)

    col1, col2, col3 = st.columns(3)
    col1.metric("Ticker", symbol)
    col2.metric("Score / 110", f"{total_score:.0f}")
    col3.metric("Rating", rating(total_score))

    if np.isnan(safe_float(values["Current Price"])):
        st.error("No price data loaded. Make sure FMP_API_KEY is added in Streamlit Secrets, or try a common ticker like AAPL/META.")
        st.info('Streamlit → App settings → Secrets → add: FMP_API_KEY = "your_key"')
    else:
        st.success("Price and technical indicators loaded.")

    if "loaded" not in str(values.get("Options Data Status", "")).lower():
        st.warning(values.get("Options Data Status"))

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
