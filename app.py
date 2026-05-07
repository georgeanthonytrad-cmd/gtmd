import requests
import re
import os
import pandas as pd
import numpy as np
import streamlit as st
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit.components.v1 as components

try:
    import yfinance as yf
except Exception:
    yf = None


# -----------------------------
# PRO DASHBOARD ADD / AUTOSAVE HELPERS
# -----------------------------
WATCHLIST_FILE = "gtmd_pro_trader_watchlist_saved.csv"

PRO_WATCHLIST_COLUMNS = [
    "Ticker", "Company", "Entry Price", "Current Price", "Target Price", "Stop Loss",
    "Shares", "Status", "Priority", "Thesis", "Notes",
    "GTMD Score", "Rating", "Explosive Move Probability %", "Squeeze Score",
    "RVOL Alert", "RVOL Alert Reason", "Earnings Risk", "Last Checked"
]

def get_empty_pro_watchlist():
    return pd.DataFrame(columns=PRO_WATCHLIST_COLUMNS)

def normalize_pro_watchlist(df):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return get_empty_pro_watchlist()
    df = df.copy()
    for col in PRO_WATCHLIST_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    if "Ticker" in df.columns:
        df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
        df = df[df["Ticker"].notna() & (df["Ticker"] != "") & (df["Ticker"] != "NAN")]
        df = df.drop_duplicates(subset=["Ticker"], keep="last")
    return df[PRO_WATCHLIST_COLUMNS]

def save_pro_watchlist(df):
    try:
        normalize_pro_watchlist(df).to_csv(WATCHLIST_FILE, index=False)
        return True
    except Exception:
        return False

def load_pro_watchlist():
    try:
        if Path(WATCHLIST_FILE).exists():
            return normalize_pro_watchlist(pd.read_csv(WATCHLIST_FILE))
    except Exception:
        pass
    return get_empty_pro_watchlist()

def ensure_pro_watchlist_loaded():
    if "pro_watchlist_df" not in st.session_state:
        st.session_state["pro_watchlist_df"] = load_pro_watchlist()
    else:
        st.session_state["pro_watchlist_df"] = normalize_pro_watchlist(st.session_state["pro_watchlist_df"])

def add_tickers_to_pro_dashboard(tickers, source_df=None):
    ensure_pro_watchlist_loaded()
    if tickers is None:
        return 0

    clean_tickers = []
    for t in list(tickers):
        t = str(t).upper().strip()
        if t and t != "NAN" and t not in clean_tickers:
            clean_tickers.append(t)

    if not clean_tickers:
        return 0

    current = normalize_pro_watchlist(st.session_state.get("pro_watchlist_df"))
    existing = set(current["Ticker"].astype(str).str.upper().str.strip()) if not current.empty else set()
    rows = []

    source_lookup = {}
    if isinstance(source_df, pd.DataFrame) and not source_df.empty and "Ticker" in source_df.columns:
        temp = source_df.copy()
        temp["Ticker"] = temp["Ticker"].astype(str).str.upper().str.strip()
        source_lookup = {r["Ticker"]: r for _, r in temp.iterrows()}

    for ticker in clean_tickers:
        if ticker in existing:
            continue

        src = source_lookup.get(ticker)
        def src_get(col, default=""):
            try:
                if src is not None and col in src.index:
                    val = src[col]
                    if pd.notna(val):
                        return val
            except Exception:
                pass
            return default

        rows.append({
            "Ticker": ticker,
            "Company": src_get("Company", ""),
            "Entry Price": "",
            "Current Price": src_get("Current Price", ""),
            "Target Price": "",
            "Stop Loss": "",
            "Shares": "",
            "Status": "Watching",
            "Priority": "Medium",
            "Thesis": "",
            "Notes": "",
            "GTMD Score": src_get("Score", src_get("GTMD Score", "")),
            "Rating": src_get("Rating", ""),
            "Explosive Move Probability %": src_get("Explosive Move Probability %", ""),
            "Squeeze Score": src_get("Squeeze Score", ""),
            "RVOL Alert": src_get("RVOL Alert", ""),
            "RVOL Alert Reason": src_get("RVOL Alert Reason", ""),
            "Earnings Risk": src_get("Earnings Risk", ""),
            "Last Checked": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        })

    if rows:
        current = pd.concat([current, pd.DataFrame(rows)], ignore_index=True)
        current = normalize_pro_watchlist(current)
        st.session_state["pro_watchlist_df"] = current
        save_pro_watchlist(current)

    return len(rows)

def render_add_to_pro_dashboard_controls(df, key_prefix="scan"):
    """Compatibility wrapper: use the real Pro Dashboard add controls."""
    if "render_add_to_watchlist_controls" in globals():
        render_add_to_watchlist_controls(df, key_prefix=key_prefix)
        return

    # Fallback only if the newer function is unavailable.
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or "Ticker" not in df.columns:
        return

    st.markdown("#### Add to Pro Trader Dashboard")
    tickers_available = (
        df["Ticker"].dropna().astype(str).str.upper().str.strip().replace("", pd.NA).dropna().drop_duplicates().tolist()
    )
    selected = st.multiselect(
        "Select ticker(s) to add",
        options=tickers_available,
        key=f"{key_prefix}_add_selected_tickers"
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Add selected to Pro Dashboard", key=f"{key_prefix}_add_selected_btn"):
            selected_rows = df[df["Ticker"].astype(str).str.upper().isin(selected)].copy()
            added, updated = add_scanner_rows_to_pro_watchlist(selected_rows)
            st.success(f"Added {added} new ticker(s), updated {updated} existing ticker(s).")
    with col2:
        top_n = min(10, len(tickers_available))
        if st.button(f"Add top {top_n} to Pro Dashboard", key=f"{key_prefix}_add_top10_btn"):
            top_rows = df.head(top_n).copy()
            added, updated = add_scanner_rows_to_pro_watchlist(top_rows)
            st.success(f"Added {added} new ticker(s), updated {updated} existing ticker(s).")


# -----------------------------
# SETUP
# -----------------------------
st.set_page_config(page_title="GTMD Scanner", layout="wide")

# -----------------------------
# CUSTOM GTMD DARK CARD THEME
# -----------------------------
st.markdown(
    """
    <style>
    :root {
        --gtmd-bg-0: #070a10;
        --gtmd-bg-1: #0d111a;
        --gtmd-card: rgba(20, 26, 40, 0.82);
        --gtmd-border: rgba(255, 255, 255, 0.12);
        --gtmd-text: #f8fafc;
        --gtmd-muted: #aeb8c8;
        --gtmd-purple: #8b5cf6;
        --gtmd-purple-2: #6d28d9;
    }

    html, body, [class*="css"] {
        font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 65% 8%, rgba(139, 92, 246, 0.22) 0%, rgba(139, 92, 246, 0.02) 31%, transparent 50%),
            radial-gradient(circle at 10% 0%, rgba(96, 165, 250, 0.12) 0%, transparent 35%),
            linear-gradient(135deg, #111827 0%, #0a0e16 46%, #05070b 100%);
        color: var(--gtmd-text);
    }

    .block-container {
        padding-top: 1.35rem;
        padding-bottom: 3rem;
        max-width: 1380px;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(18, 23, 34, 0.96) 0%, rgba(7, 10, 16, 0.99) 100%);
        border-right: 1px solid var(--gtmd-border);
        box-shadow: 12px 0 38px rgba(0,0,0,0.24);
    }
    [data-testid="stSidebar"] * { color: var(--gtmd-text); }

    h1, h2, h3, h4, h5, h6 {
        color: var(--gtmd-text) !important;
        letter-spacing: -0.035em;
    }

    [data-testid="stMetric"],
    div[data-testid="stExpander"],
    [data-testid="stAlert"],
    div[data-testid="stDataFrame"] {
        background: linear-gradient(180deg, rgba(255,255,255,0.074), rgba(255,255,255,0.035));
        border: 1px solid var(--gtmd-border);
        border-radius: 22px;
        box-shadow: 0 18px 45px rgba(0,0,0,0.26);
    }

    [data-testid="stMetric"] {
        padding: 20px 18px;
        min-height: 112px;
        backdrop-filter: blur(10px);
    }
    [data-testid="stMetricLabel"] { color: #cbd5e1 !important; font-weight: 700; }
    [data-testid="stMetricValue"] { color: #ffffff !important; font-weight: 900; }

    div[data-testid="stDataFrame"] {
        overflow: hidden;
        background: rgba(13, 17, 26, 0.75);
    }

    .stButton > button,
    .stDownloadButton > button,
    button[kind="primary"],
    button[kind="secondary"] {
        width: 100%;
        border: 0 !important;
        border-radius: 16px !important;
        background: linear-gradient(135deg, #7c3aed 0%, #a855f7 52%, #6d28d9 100%) !important;
        color: white !important;
        font-weight: 850 !important;
        letter-spacing: 0.01em;
        padding: 0.72rem 1rem !important;
        box-shadow: 0 0 26px rgba(139,92,246,0.43), inset 0 1px 0 rgba(255,255,255,0.22) !important;
        transition: all 0.18s ease-in-out !important;
    }
    .stButton > button:hover,
    .stDownloadButton > button:hover {
        transform: translateY(-2px) scale(1.01);
        box-shadow: 0 0 36px rgba(168,85,247,0.68), 0 12px 28px rgba(0,0,0,0.30) !important;
        color: white !important;
    }

    .stTextInput input,
    .stNumberInput input,
    .stTextArea textarea,
    input,
    textarea {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        caret-color: #a855f7 !important;
        background: rgba(15, 23, 42, 0.96) !important;
        border: 1px solid rgba(139,92,246,0.55) !important;
        border-radius: 14px !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.05), 0 0 0 rgba(139,92,246,0) !important;
    }
    .stTextInput input:focus,
    .stNumberInput input:focus,
    .stTextArea textarea:focus {
        border-color: #a855f7 !important;
        box-shadow: 0 0 0 3px rgba(168,85,247,0.20), 0 0 18px rgba(139,92,246,0.26) !important;
    }
    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder {
        color: #94a3b8 !important;
        -webkit-text-fill-color: #94a3b8 !important;
    }

    [data-testid="stRadio"] > div,
    [data-testid="stCheckbox"],
    [data-testid="stSlider"] {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 10px 12px;
        margin-bottom: 8px;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background: transparent;}

    .gtmd-hero {
        padding: 34px 34px;
        margin: 0 0 24px 0;
        border-radius: 30px;
        background:
            linear-gradient(135deg, rgba(139,92,246,0.24), rgba(15,23,42,0.90) 52%, rgba(8,11,17,0.94)),
            radial-gradient(circle at 80% 20%, rgba(168,85,247,0.25), transparent 38%);
        border: 1px solid rgba(255,255,255,0.14);
        box-shadow: 0 24px 70px rgba(0,0,0,0.34), 0 0 46px rgba(139,92,246,0.16);
        position: relative;
        overflow: hidden;
    }
    .gtmd-title-wrap { display: flex; align-items: center; gap: 16px; position: relative; z-index: 1; }
    .gtmd-logo {
        width: 64px;
        height: 64px;
        border-radius: 20px;
        background: linear-gradient(135deg, #a78bfa, #7c3aed 48%, #5b21b6);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 0 36px rgba(139,92,246,0.72), inset 0 1px 0 rgba(255,255,255,0.34);
        font-size: 31px;
        color: white;
        font-weight: 950;
    }
    .gtmd-title { font-size: 46px; font-weight: 950; margin: 0; color: #ffffff; line-height: 1.02; }
    .gtmd-subtitle { color: #cbd5e1; font-size: 17px; margin-top: 10px; max-width: 780px; }
    .gtmd-pill-row { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; }
    .gtmd-pill {
        display: inline-block;
        padding: 8px 12px;
        border-radius: 999px;
        background: rgba(34,197,94,0.12);
        color: #86efac;
        border: 1px solid rgba(34,197,94,0.35);
        font-weight: 800;
        font-size: 13px;
    }
    .gtmd-pill-purple {
        background: rgba(139,92,246,0.14);
        color: #ddd6fe;
        border: 1px solid rgba(139,92,246,0.45);
    }
    .gtmd-section-card {
        padding: 22px 24px;
        margin: 18px 0 14px 0;
        border-radius: 24px;
        background: linear-gradient(180deg, rgba(255,255,255,0.074), rgba(255,255,255,0.035));
        border: 1px solid var(--gtmd-border);
        box-shadow: 0 18px 45px rgba(0,0,0,0.25);
    }
    .gtmd-card-title {
        font-size: 22px;
        font-weight: 900;
        color: white;
        margin-bottom: 4px;
    }
    .gtmd-card-subtitle {
        color: var(--gtmd-muted);
        font-size: 14px;
    }
    
    .gtmd-tv-card {
        border-radius: 22px;
        padding: 10px;
        background: linear-gradient(145deg, rgba(18, 24, 38, 0.96), rgba(7, 10, 18, 0.98));
        border: 1px solid rgba(139, 92, 246, 0.35);
        box-shadow: 0 0 35px rgba(124, 58, 237, 0.18);
        overflow: hidden;
    }
</style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# LOGIN
# -----------------------------
def login_screen():
    st.markdown(
        """
        <div style="text-align:center; padding: 2rem 0 1rem 0;">
            <h1>GTMD Scanner</h1>
            <p style="font-size:18px; color: gray;">Private access only</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    login_box = st.container(border=True)

    with login_box:
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_clicked = st.button("Sign in", use_container_width=True)

        correct_username = st.secrets.get("APP_USERNAME", "admin")
        correct_password = st.secrets.get("APP_PASSWORD", "")

        if login_clicked:
            if username == correct_username and password == correct_password and correct_password:
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.rerun()
            else:
                st.error("Incorrect username or password.")

    st.caption("Set APP_USERNAME and APP_PASSWORD in Streamlit Secrets.")


if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login_screen()
    st.stop()


with st.sidebar:
    st.success(f"Logged in as {st.session_state.get('username', 'user')}")
    if st.button("Log out"):
        st.session_state["authenticated"] = False
        st.session_state.pop("username", None)
        st.rerun()


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

def first_valid(*values):
    for value in values:
        v = safe_float(value)
        if is_num(v):
            return v
    return np.nan


def parse_finviz_number(x):
    """Parse Finviz values like 31.2, 2.4%, 1.5B, 850M into numbers."""
    try:
        if x is None:
            return np.nan
        x = str(x).replace(",", "").replace("%", "").strip()
        if x in ["", "-", "N/A", "nan", "None"]:
            return np.nan
        multiplier = 1
        if x[-1:].upper() == "T":
            multiplier = 1_000_000_000_000
            x = x[:-1]
        elif x[-1:].upper() == "B":
            multiplier = 1_000_000_000
            x = x[:-1]
        elif x[-1:].upper() == "M":
            multiplier = 1_000_000
            x = x[:-1]
        elif x[-1:].upper() == "K":
            multiplier = 1_000
            x = x[:-1]
        return float(x) * multiplier
    except Exception:
        return np.nan




def extract_metric_from_html(html, labels):
    """Very small HTML fallback parser for Yahoo/Finviz-style pages."""
    try:
        text = str(html)
        for label in labels:
            # Match either JSON-style key/raw values or table cells near the label.
            json_patterns = [
                rf'"{re.escape(label)}"\s*:\s*{{\s*"raw"\s*:\s*(-?\d+(?:\.\d+)?)',
                rf'"{re.escape(label)}"\s*:\s*(-?\d+(?:\.\d+)?)',
            ]
            for pat in json_patterns:
                m = re.search(pat, text, flags=re.IGNORECASE)
                if m:
                    return safe_float(m.group(1))

            # Table-style: label followed within a short span by a cell value.
            pat = rf'>{re.escape(label)}<.*?<td[^>]*>(.*?)</td>'
            m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
            if m:
                value = re.sub(r'<.*?>', '', m.group(1)).strip()
                return parse_finviz_number(value)
    except Exception:
        pass
    return np.nan

def raw_value(obj):
    """Extract Yahoo quoteSummary values that may look like {'raw': 123, 'fmt': '123'}."""
    try:
        if isinstance(obj, dict):
            if "raw" in obj:
                return obj.get("raw")
            if "fmt" in obj:
                return obj.get("fmt")
        return obj
    except Exception:
        return np.nan


def nested_raw(data, *path):
    """Safely extract a nested Yahoo quoteSummary value."""
    try:
        cur = data
        for key in path:
            cur = cur.get(key, {})
        return raw_value(cur)
    except Exception:
        return np.nan


def display_value(x):
    """Make Streamlit show missing values clearly instead of blank cells."""
    if isinstance(x, str):
        return x
    try:
        if np.isnan(float(x)):
            return "N/A"
    except Exception:
        pass
    return x



def is_num(x):
    try:
        return not np.isnan(float(x))
    except Exception:
        return False


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


def rating(score):
    if score >= 85:
        return "STRONG SETUP"
    if score >= 70:
        return "GOOD SETUP"
    if score >= 55:
        return "WATCHLIST / WAIT"
    return "AVOID / WEAK SETUP"


def clamp_score(x):
    """Clamp a numeric score into a clean 0-100 range."""
    try:
        if not is_num(x):
            return np.nan
        return int(round(max(0, min(100, float(x)))))
    except Exception:
        return np.nan


def compute_gtmd_score_layers(values, detail_df=None):
    """
    Option A score layers: explanatory only.
    These DO NOT change the main GTMD Score, Squeeze Score, Explosive Move Probability,
    RVOL Alert, or Playbook Signal. They only explain why a high-score stock may still
    be Watchlist / Wait.
    """
    try:
        detail_points = {}
        if isinstance(detail_df, pd.DataFrame) and not detail_df.empty:
            for _, row in detail_df.iterrows():
                factor = str(row.get("Factor", ""))
                detail_points[factor] = safe_float(row.get("Points"), 0)

        # Slow-changing company/valuation quality. Based on the same point blocks that
        # already feed the existing GTMD score, but normalized separately to 0-100.
        fundamental_raw = sum(detail_points.get(k, 0) for k in [
            "EPS growth",
            "Quarterly revenue growth",
            "Analyst target upside",
            "Cash runway",
            "Forward P/E < TTM P/E",
            "PEG",
            "P/E ratio reasonableness",
        ])
        fundamental_score = clamp_score((fundamental_raw / 60) * 100)

        # Chart/trend quality. Based on the existing technical point block only.
        technical_raw = sum(detail_points.get(k, 0) for k in [
            "5-day price change",
            "1-month price change",
            "RSI",
            "Distance from 50D MA",
            "Distance from 200D MA",
        ])
        technical_score = clamp_score((technical_raw / 32) * 100)

        # Entry timing quality. Separate from the main Score and separate from squeeze/explosive.
        # This answers: is this a clean entry right now, or should it sit on watchlist?
        timing_raw = 0
        rsi = safe_float(values.get("RSI"))
        change_5d = safe_float(values.get("5D Price Change %"))
        dist_50 = safe_float(values.get("Distance from 50D MA %"))
        dist_200 = safe_float(values.get("Distance from 200D MA %"))
        breakout_proximity = safe_float(values.get("Breakout Proximity %"))
        rvol = safe_float(values.get("Relative Volume"))
        days_earnings = safe_float(values.get("Days Until Earnings"))

        # Near support / 50D is cleaner than chasing far above it.
        if is_num(dist_50):
            if -5 <= dist_50 <= 3:
                timing_raw += 25
            elif -10 <= dist_50 < -5 or 3 < dist_50 <= 8:
                timing_raw += 15
            elif 8 < dist_50 <= 15:
                timing_raw += 6

        # Near the 20D high/breakout area is useful, but overextension gets less credit.
        if is_num(breakout_proximity):
            if -3 <= breakout_proximity <= 1:
                timing_raw += 25
            elif -7 <= breakout_proximity < -3:
                timing_raw += 15
            elif 1 < breakout_proximity <= 5:
                timing_raw += 8

        # Healthy RSI gets credit; very hot RSI is a wait/chase warning.
        if is_num(rsi):
            if 45 <= rsi <= 68:
                timing_raw += 20
            elif 35 <= rsi < 45 or 68 < rsi <= 75:
                timing_raw += 10
            elif rsi > 75:
                timing_raw -= 10

        # Controlled 5D move is cleaner than a vertical move.
        if is_num(change_5d):
            if -4 <= change_5d <= 8:
                timing_raw += 15
            elif 8 < change_5d <= 15:
                timing_raw += 5
            elif change_5d > 15:
                timing_raw -= 10

        # Volume confirmation is helpful but not required.
        if is_num(rvol):
            if 1.2 <= rvol <= 3.5:
                timing_raw += 10
            elif rvol > 3.5:
                timing_raw += 5

        # Major trend still matters for timing discipline.
        if is_num(dist_200):
            if dist_200 > 0:
                timing_raw += 5
            else:
                timing_raw -= 10

        # Upcoming earnings very close reduces entry timing quality.
        if is_num(days_earnings) and 0 <= days_earnings <= 7:
            timing_raw -= 10

        timing_score = clamp_score(timing_raw)

        return fundamental_score, technical_score, timing_score
    except Exception:
        return np.nan, np.nan, np.nan


def generate_trade_alert(values, entry_price=np.nan):
    score = safe_float(values.get("Score"))
    current_price = safe_float(values.get("Current Price"))
    rsi = safe_float(values.get("RSI"))
    change_5d = safe_float(values.get("5D Price Change %"))
    dist_50 = safe_float(values.get("Distance from 50D MA %"))
    dist_200 = safe_float(values.get("Distance from 200D MA %"))
    squeeze_score = safe_float(values.get("Squeeze Score"))
    squeeze_setup = str(values.get("Squeeze Setup", "NO"))
    entry_price = safe_float(entry_price)

    gain_loss_pct = np.nan
    if is_num(entry_price) and entry_price > 0 and is_num(current_price):
        gain_loss_pct = ((current_price / entry_price) - 1) * 100

    sell_reasons = []
    trim_reasons = []
    buy_reasons = []

    if is_num(score) and score < 55:
        sell_reasons.append("Score dropped below 55")
    if is_num(dist_200) and dist_200 < 0:
        sell_reasons.append("Price is below the 200D moving average")
    if is_num(gain_loss_pct) and gain_loss_pct <= -7:
        sell_reasons.append("Position is down 7% or more from entry")
    if is_num(dist_50) and dist_50 < -10:
        sell_reasons.append("Price is more than 10% below the 50D moving average")

    if is_num(gain_loss_pct) and gain_loss_pct >= 10:
        trim_reasons.append("Position is up 10% or more from entry")
    if is_num(rsi) and rsi >= 75:
        trim_reasons.append("RSI is 75 or higher")
    if is_num(change_5d) and change_5d > 8:
        trim_reasons.append("5-day price change is above +8%")

    if is_num(score) and score >= 85:
        buy_reasons.append("Score is 85 or higher")
    elif is_num(score) and score >= 70:
        buy_reasons.append("Score is 70 or higher")

    if squeeze_setup == "YES" and is_num(squeeze_score):
        buy_reasons.append(f"Squeeze setup detected, squeeze score {squeeze_score:.0f}/10")

    if sell_reasons:
        return "SELL", "Exit / avoid new entry", sell_reasons, gain_loss_pct
    if trim_reasons:
        if squeeze_setup == "YES":
            trim_reasons.append("Squeeze setup may be getting extended")
        return "TRIM", "Take partial profit or tighten stop", trim_reasons, gain_loss_pct
    if buy_reasons:
        if squeeze_setup == "YES" and is_num(score) and score >= 70:
            return "BUY (SQUEEZE)", "High score plus squeeze setup", buy_reasons, gain_loss_pct
        return "BUY", "Setup is acceptable for entry/watchlist", buy_reasons, gain_loss_pct
    return "WATCH", "No clean buy or exit signal", ["Setup is not strong enough for BUY, but no major SELL trigger"], gain_loss_pct


# -----------------------------
# YAHOO DATA
# -----------------------------
YAHOO_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept": "application/json,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

FINVIZ_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
@st.cache_data(ttl=900)
def get_yahoo_chart_history(symbol, retries=2, backoff_seconds=0.5):
    """
    Daily Yahoo chart history for full GTMD scoring.
    Hardened with retry/backoff + query1/query2 fallback because Streamlit Cloud
    can get intermittent Yahoo throttling during large scans.
    """
    params = {
        "range": "1y",
        "interval": "1d",
        "includePrePost": "false",
        "events": "div,splits",
    }
    errors = []

    for host in ["query1.finance.yahoo.com", "query2.finance.yahoo.com"]:
        url = f"https://{host}/v8/finance/chart/{symbol}"

        for attempt in range(retries + 1):
            try:
                response = requests.get(url, params=params, headers=YAHOO_HEADERS, timeout=20)

                if response.status_code in [429, 500, 502, 503, 504]:
                    errors.append(f"{host} HTTP {response.status_code} attempt {attempt + 1}/{retries + 1}")
                    if attempt < retries:
                        time.sleep(backoff_seconds * (attempt + 1))
                        continue
                    break

                if response.status_code != 200:
                    errors.append(f"{host} HTTP {response.status_code}: {response.text[:120]}")
                    break

                data = response.json()
                result = data.get("chart", {}).get("result", [])

                if not result:
                    error = data.get("chart", {}).get("error")
                    errors.append(f"{host} no result attempt {attempt + 1}/{retries + 1}: {error}")
                    if attempt < retries:
                        time.sleep(backoff_seconds * (attempt + 1))
                        continue
                    break

                result = result[0]
                timestamps = result.get("timestamp", [])
                quote = result.get("indicators", {}).get("quote", [{}])[0]
                close = quote.get("close", [])
                open_ = quote.get("open", [])
                high = quote.get("high", [])
                low = quote.get("low", [])
                volume = quote.get("volume", [])
                meta = result.get("meta", {}) or {}

                if not timestamps or not close:
                    errors.append(f"{host} missing timestamps/close attempt {attempt + 1}/{retries + 1}")
                    if attempt < retries:
                        time.sleep(backoff_seconds * (attempt + 1))
                        continue
                    break

                df = pd.DataFrame({
                    "date": pd.to_datetime(timestamps, unit="s"),
                    "open": open_ if len(open_) == len(timestamps) else [np.nan] * len(timestamps),
                    "high": high if len(high) == len(timestamps) else [np.nan] * len(timestamps),
                    "low": low if len(low) == len(timestamps) else [np.nan] * len(timestamps),
                    "close": close,
                    "volume": volume if len(volume) == len(timestamps) else [np.nan] * len(timestamps),
                })

                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.dropna(subset=["close"]).sort_values("date")

                if df.empty:
                    errors.append(f"{host} empty close dataframe attempt {attempt + 1}/{retries + 1}")
                    if attempt < retries:
                        time.sleep(backoff_seconds * (attempt + 1))
                        continue
                    break

                return meta, df, ""

            except Exception as e:
                errors.append(f"{host} failed attempt {attempt + 1}/{retries + 1}: {e}")
                if attempt < retries:
                    time.sleep(backoff_seconds * (attempt + 1))
                    continue
                break

    return {}, pd.DataFrame(), "Yahoo chart failed after retry/fallback: " + " | ".join(errors[-8:])
@st.cache_data(ttl=300)
def get_yahoo_intraday_candles(symbol):
    """
    1-hour OHLC candles for the visual chart.
    Yahoo usually supports 1h candles up to roughly 60 days.
    """
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "range": "60d",
        "interval": "1h",
        "includePrePost": "false",
        "events": "div,splits",
    }

    try:
        response = requests.get(url, params=params, headers=YAHOO_HEADERS, timeout=20)
        if response.status_code != 200:
            return pd.DataFrame(), f"Yahoo 1H candles HTTP {response.status_code}"

        data = response.json()
        result = data.get("chart", {}).get("result", [])
        if not result:
            error = data.get("chart", {}).get("error")
            return pd.DataFrame(), f"Yahoo 1H candles no result: {error}"

        result = result[0]
        timestamps = result.get("timestamp", [])
        quote = result.get("indicators", {}).get("quote", [{}])[0]

        if not timestamps:
            return pd.DataFrame(), "Yahoo 1H candles missing timestamps"

        df = pd.DataFrame({
            "date": pd.to_datetime(timestamps, unit="s"),
            "open": quote.get("open", []),
            "high": quote.get("high", []),
            "low": quote.get("low", []),
            "close": quote.get("close", []),
            "volume": quote.get("volume", []),
        })

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["open", "high", "low", "close"]).sort_values("date")
        return df, ""

    except Exception as e:
        return pd.DataFrame(), f"Yahoo 1H candles failed: {e}"


def tradingview_exchange_hint(symbol):
    """
    TradingView can usually resolve a plain ticker, but common exchange
    hints make the widget load faster for popular U.S. names.
    """
    nasdaq = {
        "AAPL", "MSFT", "META", "NVDA", "TSLA", "AMD", "AMZN", "GOOGL", "GOOG",
        "PLTR", "NFLX", "AVGO", "COST", "ADBE", "INTC", "CSCO", "PEP", "QCOM",
        "TXN", "AMAT", "MU", "MRVL", "SMCI", "LULU", "SBUX", "MELI", "SHOP"
    }
    nyse = {
        "BAC", "SCHW", "PSX", "JPM", "WMT", "DIS", "NKE", "ORCL", "CRM", "V",
        "MA", "HD", "MCD", "KO", "XOM", "CVX", "BA", "UNH", "GS", "MS", "SNOW",
        "NET", "DOCN", "GLW"
    }
    clean_symbol = str(symbol).upper().strip()
    if clean_symbol in nasdaq:
        return f"NASDAQ:{clean_symbol}"
    if clean_symbol in nyse:
        return f"NYSE:{clean_symbol}"
    return clean_symbol


def render_tradingview_chart(symbol):
    tv_symbol = tradingview_exchange_hint(symbol)
    container_id = f"tradingview_{str(symbol).upper().replace('-', '_').replace('.', '_')}"

    components.html(
        f"""
        <div class="gtmd-tv-card">
          <div class="tradingview-widget-container" style="height:680px;width:100%;">
            <div id="{container_id}" style="height:100%;width:100%;"></div>
            <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
            <script type="text/javascript">
            new TradingView.widget({{
              "autosize": true,
              "symbol": "{tv_symbol}",
              "interval": "60",
              "timezone": "America/Los_Angeles",
              "theme": "dark",
              "style": "1",
              "locale": "en",
              "toolbar_bg": "#0b1020",
              "enable_publishing": false,
              "allow_symbol_change": true,
              "hide_side_toolbar": false,
              "hide_top_toolbar": false,
              "save_image": false,
              "withdateranges": true,
              "range": "5D",
              "studies": [
                "Volume@tv-basicstudies",
                "RSI@tv-basicstudies"
              ],
              "container_id": "{container_id}"
            }});
            </script>
          </div>
        </div>
        """,
        height=720,
    )

@st.cache_data(ttl=3600)
def get_yfinance_info(symbol):
    if yf is None:
        return {}

    try:
        ticker = yf.Ticker(symbol)
        return ticker.get_info() or {}
    except Exception:
        return {}


@st.cache_data(ttl=3600)
def get_yahoo_quote_summary(symbol):
    """
    Direct Yahoo fallback for fundamentals.
    Tries both query1 and query2 because one sometimes fails on Streamlit Cloud.
    """
    modules = ",".join([
        "defaultKeyStatistics",
        "financialData",
        "summaryDetail",
        "price",
        "earningsTrend",
    ])
    params = {"modules": modules}
    errors = []

    for host in ["query1.finance.yahoo.com", "query2.finance.yahoo.com"]:
        url = f"https://{host}/v10/finance/quoteSummary/{symbol}"
        try:
            response = requests.get(url, params=params, headers=YAHOO_HEADERS, timeout=20)
            if response.status_code != 200:
                errors.append(f"{host} HTTP {response.status_code}: {response.text[:120]}")
                continue

            data = response.json()
            result = data.get("quoteSummary", {}).get("result", [])
            if result:
                return result[0] or {}, ""

            errors.append(f"{host} no result: {data.get('quoteSummary', {}).get('error')}")
        except Exception as e:
            errors.append(f"{host} failed: {e}")

    return {}, " | ".join(errors)


@st.cache_data(ttl=3600)
def get_yahoo_quote(symbol):
    """
    Extra Yahoo fallback for quote-level fundamentals.
    Tries query1 and query2 /v7/finance/quote.
    """
    params = {"symbols": symbol}
    errors = []

    for host in ["query1.finance.yahoo.com", "query2.finance.yahoo.com"]:
        url = f"https://{host}/v7/finance/quote"
        try:
            response = requests.get(url, params=params, headers=YAHOO_HEADERS, timeout=20)
            if response.status_code != 200:
                errors.append(f"{host} HTTP {response.status_code}: {response.text[:120]}")
                continue

            data = response.json()
            result = data.get("quoteResponse", {}).get("result", [])
            if result:
                return result[0] or {}, ""

            errors.append(f"{host} quote no result")
        except Exception as e:
            errors.append(f"{host} quote failed: {e}")

    return {}, " | ".join(errors)


@st.cache_data(ttl=3600)
def get_finviz_snapshot(symbol):
    """
    Finviz fallback for fields Yahoo frequently blanks out:
    P/E, Forward P/E, PEG, Target Price, Short Float.
    Uses both pandas table parsing and a regex fallback.
    """
    url = f"https://finviz.com/quote.ashx?t={symbol}&p=d"
    try:
        response = requests.get(url, headers=FINVIZ_HEADERS, timeout=20)
        if response.status_code != 200:
            return {}, f"Finviz HTTP {response.status_code}"

        html = response.text
        data = {}

        # First try pandas, which works when Finviz returns normal tables.
        try:
            tables = pd.read_html(html)
            flat = []
            for table in tables:
                for row in table.values.tolist():
                    flat.extend(row)

            for i in range(0, len(flat) - 1, 2):
                key = str(flat[i]).strip()
                value = flat[i + 1]
                if key and key not in data:
                    data[key] = value
        except Exception:
            data = {}

        # Regex backup if pandas cannot read the snapshot table.
        trailing_pe = parse_finviz_number(data.get("P/E"))
        if not is_num(trailing_pe):
            trailing_pe = extract_metric_from_html(html, ["P/E"])

        forward_pe = parse_finviz_number(data.get("Forward P/E"))
        if not is_num(forward_pe):
            forward_pe = extract_metric_from_html(html, ["Forward P/E"])

        peg = parse_finviz_number(data.get("PEG"))
        if not is_num(peg):
            peg = extract_metric_from_html(html, ["PEG"])

        target = parse_finviz_number(data.get("Target Price"))
        if not is_num(target):
            target = extract_metric_from_html(html, ["Target Price"])

        short_float = parse_finviz_number(data.get("Short Float"))
        if not is_num(short_float):
            short_float = extract_metric_from_html(html, ["Short Float"])

        eps_growth = parse_finviz_number(data.get("EPS past 5Y"))
        sales_growth = parse_finviz_number(data.get("Sales past 5Y"))

        return {
            "trailingPE": trailing_pe,
            "forwardPE": forward_pe,
            "pegRatio": peg,
            "targetMeanPrice": target,
            "shortPercentOfFloatPercent": short_float,
            "epsGrowthPast5YPercent": eps_growth,
            "salesGrowthPast5YPercent": sales_growth,
        }, ""
    except Exception as e:
        return {}, f"Finviz failed: {e}"


@st.cache_data(ttl=3600)
def get_yahoo_page_fallback(symbol):
    """
    Last-resort Yahoo web-page fallback. This is for cases where chart prices load
    but quoteSummary/quote endpoints return blanks on Streamlit Cloud.
    """
    pages = [
        f"https://finance.yahoo.com/quote/{symbol}/key-statistics?p={symbol}",
        f"https://finance.yahoo.com/quote/{symbol}/analysis?p={symbol}",
        f"https://finance.yahoo.com/quote/{symbol}?p={symbol}",
    ]
    errors = []
    combined = ""

    for url in pages:
        try:
            r = requests.get(url, headers=YAHOO_HEADERS, timeout=20)
            if r.status_code == 200 and r.text:
                combined += "\n" + r.text
            else:
                errors.append(f"{url.split('/quote/')[-1]} HTTP {r.status_code}")
        except Exception as e:
            errors.append(str(e))

    if not combined:
        return {}, " | ".join(errors)

    return {
        "trailingPE": extract_metric_from_html(combined, ["trailingPE", "Trailing P/E", "PE Ratio (TTM)"]),
        "forwardPE": extract_metric_from_html(combined, ["forwardPE", "Forward P/E", "Forward PE"]),
        "pegRatio": extract_metric_from_html(combined, ["pegRatio", "PEG Ratio", "PEG Ratio (5yr expected)"]),
        "targetMeanPrice": extract_metric_from_html(combined, ["targetMeanPrice", "1y Target Est", "Average Target Price"]),
        "shortPercentOfFloat": extract_metric_from_html(combined, ["shortPercentOfFloat"]),
        "shortPercentOfFloatPercent": extract_metric_from_html(combined, ["Short % of Float", "Short Float"]),
    }, " | ".join(errors)


@st.cache_data(ttl=3600)
def get_cashflow_data(symbol):
    if yf is None:
        return pd.DataFrame(), pd.DataFrame()

    try:
        ticker = yf.Ticker(symbol)
        annual_cf = ticker.cashflow
        quarterly_cf = ticker.quarterly_cashflow

        if annual_cf is None:
            annual_cf = pd.DataFrame()
        if quarterly_cf is None:
            quarterly_cf = pd.DataFrame()

        return annual_cf, quarterly_cf
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


def extract_operating_cash_flow_from_df(cashflow):
    if cashflow is None or cashflow.empty:
        return np.nan

    possible_rows = [
        "Operating Cash Flow",
        "Total Cash From Operating Activities",
        "Cash Flow From Continuing Operating Activities",
    ]

    for row in possible_rows:
        if row in cashflow.index:
            vals = pd.to_numeric(cashflow.loc[row], errors="coerce").dropna()
            if len(vals) > 0:
                return safe_float(vals.iloc[0])

    return np.nan


def get_operating_cash_flow(symbol):
    annual_cf, quarterly_cf = get_cashflow_data(symbol)

    # Prefer annual cash flow for runway stability, fallback to latest quarter annualized.
    annual_ocf = extract_operating_cash_flow_from_df(annual_cf)
    if is_num(annual_ocf):
        return annual_ocf

    quarterly_ocf = extract_operating_cash_flow_from_df(quarterly_cf)
    if is_num(quarterly_ocf):
        return quarterly_ocf * 4

    return np.nan



@st.cache_data(ttl=3600)
def get_income_balance_data(symbol):
    if yf is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    try:
        ticker = yf.Ticker(symbol)
        income = ticker.income_stmt
        quarterly_income = ticker.quarterly_income_stmt
        balance = ticker.balance_sheet
        if income is None:
            income = pd.DataFrame()
        if quarterly_income is None:
            quarterly_income = pd.DataFrame()
        if balance is None:
            balance = pd.DataFrame()
        return income, quarterly_income, balance
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def extract_statement_value(statement, possible_rows, index_position=0):
    if statement is None or statement.empty:
        return np.nan
    for row in possible_rows:
        if row in statement.index:
            vals = pd.to_numeric(statement.loc[row], errors="coerce").dropna()
            if len(vals) > index_position:
                return safe_float(vals.iloc[index_position])
    return np.nan


def compute_statement_fallbacks(symbol, current_price):
    income, quarterly_income, balance = get_income_balance_data(symbol)

    total_cash = extract_statement_value(balance, [
        "Cash And Cash Equivalents",
        "Cash Cash Equivalents And Short Term Investments",
        "Cash Financial",
    ])

    trailing_eps = extract_statement_value(income, ["Diluted EPS", "Basic EPS"])
    ttm_pe_from_eps = current_price / trailing_eps if is_num(current_price) and is_num(trailing_eps) and trailing_eps > 0 else np.nan

    eps_growth = np.nan
    latest_eps = extract_statement_value(income, ["Diluted EPS", "Basic EPS"], 0)
    prior_eps = extract_statement_value(income, ["Diluted EPS", "Basic EPS"], 1)
    if is_num(latest_eps) and is_num(prior_eps) and abs(prior_eps) > 0:
        eps_growth = ((latest_eps / prior_eps) - 1) * 100

    revenue_growth = np.nan
    latest_rev_q = extract_statement_value(quarterly_income, ["Total Revenue"], 0)
    year_ago_rev_q = extract_statement_value(quarterly_income, ["Total Revenue"], 4)
    if is_num(latest_rev_q) and is_num(year_ago_rev_q) and year_ago_rev_q > 0:
        revenue_growth = ((latest_rev_q / year_ago_rev_q) - 1) * 100
    else:
        latest_rev = extract_statement_value(income, ["Total Revenue"], 0)
        prior_rev = extract_statement_value(income, ["Total Revenue"], 1)
        if is_num(latest_rev) and is_num(prior_rev) and prior_rev > 0:
            revenue_growth = ((latest_rev / prior_rev) - 1) * 100

    return {
        "totalCash": total_cash,
        "ttmPEFromEPS": ttm_pe_from_eps,
        "epsGrowthPercent": eps_growth,
        "revenueGrowthPercent": revenue_growth,
    }


@st.cache_data(ttl=21600)
def get_next_earnings_date(symbol):
    """Best-effort earnings date lookup. Returns (date, days_until, status)."""
    if yf is None:
        return None, np.nan, "Earnings unavailable: yfinance not installed"
    try:
        ticker = yf.Ticker(symbol)
        cal = None
        try:
            cal = ticker.calendar
        except Exception:
            cal = None
        candidate = None
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            for key in ["Earnings Date", "EarningsDate", "Earnings"]:
                if key in cal.index:
                    raw = cal.loc[key].dropna().iloc[0] if hasattr(cal.loc[key], "dropna") else cal.loc[key]
                    candidate = raw[0] if isinstance(raw, (list, tuple, np.ndarray)) and len(raw) else raw
                    break
        elif isinstance(cal, dict):
            for key in ["Earnings Date", "EarningsDate", "earningsDate"]:
                raw = cal.get(key)
                if raw is not None:
                    candidate = raw[0] if isinstance(raw, (list, tuple, np.ndarray)) and len(raw) else raw
                    break
        if candidate is None:
            try:
                info = ticker.get_info() or {}
                ts = info.get("earningsTimestamp") or info.get("earningsTimestampStart")
                if ts:
                    candidate = pd.to_datetime(ts, unit="s")
            except Exception:
                pass
        if candidate is None:
            return None, np.nan, "No upcoming earnings date found"
        dt = pd.to_datetime(candidate, errors="coerce")
        if pd.isna(dt):
            return None, np.nan, "Earnings date could not be parsed"
        today = pd.Timestamp.today().normalize()
        days = int((dt.normalize() - today).days)
        return dt.date().isoformat(), days, "Loaded"
    except Exception as e:
        return None, np.nan, f"Earnings lookup failed: {e}"

def earnings_risk_label(days_until):
    if not is_num(days_until):
        return "Unknown"
    days_until = int(days_until)
    if days_until < 0:
        return "Recently reported"
    if days_until == 0:
        return "Earnings today 🚨"
    if days_until <= 3:
        return "Earnings very soon ⚠️"
    if days_until <= 7:
        return "Earnings this week ⚠️"
    if days_until <= 14:
        return "Earnings within 2 weeks"
    return "No near-term earnings risk"

def compute_smart_squeeze_score(values):
    """Smarter 0-10 squeeze model: fuel + compression + breakout readiness - trap risk."""
    raw = 0.0
    reasons = []
    cautions = []

    short_float = safe_float(values.get("Short % of Float"))
    change_5d = safe_float(values.get("5D Price Change %"))
    rsi = safe_float(values.get("RSI"))
    dist_50 = safe_float(values.get("Distance from 50D MA %"))
    dist_200 = safe_float(values.get("Distance from 200D MA %"))
    rvol = safe_float(values.get("Relative Volume"))
    volatility_20d = safe_float(values.get("20D Volatility %"))
    atr_pct = safe_float(values.get("ATR %"))
    range_tightness = safe_float(values.get("20D Range Tightness %"))
    breakout_proximity = safe_float(values.get("Breakout Proximity %"))
    trend_aligned = bool(values.get("Trend Alignment"))

    if is_num(short_float):
        if short_float >= 25:
            raw += 2.0; reasons.append("very high short-interest fuel")
        elif short_float >= 15:
            raw += 1.7; reasons.append("high short-interest fuel")
        elif short_float >= 8:
            raw += 1.1; reasons.append("moderate short-interest fuel")
        elif short_float >= 4:
            raw += 0.5; reasons.append("some short-interest fuel")
        else:
            cautions.append("low short-interest fuel")

    if is_num(range_tightness):
        if 4 <= range_tightness <= 12:
            raw += 2.0; reasons.append("tight 20D compression")
        elif 12 < range_tightness <= 20:
            raw += 1.2; reasons.append("moderate compression")
        elif range_tightness > 32:
            raw -= 1.2; cautions.append("range is too wide for a clean squeeze")

    if is_num(breakout_proximity):
        if -3 <= breakout_proximity <= 0.5:
            raw += 1.5; reasons.append("near 20D breakout level")
        elif -7 <= breakout_proximity < -3:
            raw += 0.8; reasons.append("within striking distance of breakout")
        elif breakout_proximity > 4:
            raw -= 0.8; cautions.append("already extended above recent breakout area")

    if is_num(change_5d):
        if 0 <= change_5d <= 8:
            raw += 1.1; reasons.append("controlled positive 5D momentum")
        elif 8 < change_5d <= 15:
            raw += 0.4; cautions.append("5D move is getting extended")
        elif change_5d > 15:
            raw -= 1.1; cautions.append("5D move may already be chasing")
        elif -4 <= change_5d < 0:
            raw += 0.3; reasons.append("minor pullback while setup remains intact")
        else:
            raw -= 0.6; cautions.append("recent momentum is weak")

    if is_num(rsi):
        if 45 <= rsi <= 68:
            raw += 1.1; reasons.append("RSI in squeeze launch zone")
        elif 38 <= rsi < 45:
            raw += 0.4; reasons.append("RSI starting to recover")
        elif rsi > 75:
            raw -= 1.0; cautions.append("RSI is too hot")
        elif rsi < 35:
            raw -= 0.7; cautions.append("RSI is still weak")

    if trend_aligned:
        raw += 0.8; reasons.append("trend alignment supports squeeze")
    elif is_num(dist_200) and dist_200 < 0:
        raw -= 0.9; cautions.append("below 200D trend filter")

    if is_num(dist_50):
        if -5 <= dist_50 <= 6:
            raw += 0.8; reasons.append("near 50D launch/support zone")
        elif dist_50 > 18:
            raw -= 0.8; cautions.append("far above 50D; late-entry risk")
        elif dist_50 < -12:
            raw -= 0.6; cautions.append("well below 50D")

    if is_num(rvol):
        if 1.2 <= rvol <= 3.5:
            raw += 0.7; reasons.append(f"volume confirmation ({rvol:.1f}x RVOL)")
        elif rvol > 4.5:
            raw += 0.2; cautions.append("very high RVOL; news spike risk")
        elif rvol < 0.7:
            raw -= 0.6; cautions.append("weak volume confirmation")

    if is_num(volatility_20d):
        if 25 <= volatility_20d <= 85:
            raw += 0.4; reasons.append("enough volatility to move")
        elif volatility_20d > 120:
            raw -= 1.0; cautions.append("extreme volatility trap risk")
        elif volatility_20d < 15:
            raw -= 0.4; cautions.append("low-volatility stock")

    if is_num(atr_pct):
        if atr_pct > 12:
            raw -= 0.6; cautions.append("ATR is very high")
        elif 1.5 <= atr_pct <= 8:
            raw += 0.3; reasons.append("healthy ATR range")

    score = int(max(0, min(10, round(raw))))
    setup = "YES" if score >= 7 else "WATCH" if score >= 5 else "NO"
    label = "A+ Squeeze" if score >= 9 else "Strong Squeeze" if score >= 7 else "Developing Squeeze" if score >= 5 else "No Clean Squeeze"
    reason_text = "; ".join(reasons[:6]) if reasons else "No strong squeeze drivers detected"
    caution_text = "; ".join(cautions[:5]) if cautions else "None"
    return score, setup, label, reason_text, caution_text

def compute_explosive_move_probability(values):
    """
    Smarter 0-100 move-potential model.
    It does not predict direction. It estimates whether a stock has ingredients
    for a larger-than-normal move: fuel, momentum, volume, compression, trend, and catalyst.
    """
    score = 0
    reasons = []
    cautions = []

    short_float = safe_float(values.get("Short % of Float"))
    change_5d = safe_float(values.get("5D Price Change %"))
    change_1m = safe_float(values.get("1M Price Change %"))
    rsi = safe_float(values.get("RSI"))
    dist_50 = safe_float(values.get("Distance from 50D MA %"))
    dist_200 = safe_float(values.get("Distance from 200D MA %"))
    squeeze_score = safe_float(values.get("Squeeze Score"))
    volatility_20d = safe_float(values.get("20D Volatility %"))
    rvol = safe_float(values.get("Relative Volume"))
    atr_pct = safe_float(values.get("ATR %"))
    range_tightness = safe_float(values.get("20D Range Tightness %"))
    breakout_proximity = safe_float(values.get("Breakout Proximity %"))
    days_earnings = safe_float(values.get("Days Until Earnings"))
    trend_aligned = bool(values.get("Trend Alignment"))

    if is_num(short_float):
        if short_float >= 25:
            score += 18; reasons.append("very high short-interest fuel")
        elif short_float >= 15:
            score += 14; reasons.append("high short-interest fuel")
        elif short_float >= 8:
            score += 8; reasons.append("moderate short-interest fuel")
        elif short_float >= 4:
            score += 4; reasons.append("some short-interest fuel")

    if is_num(rvol):
        if rvol >= 2.5:
            score += 16; reasons.append(f"major relative-volume surge ({rvol:.1f}x)")
        elif rvol >= 1.5:
            score += 11; reasons.append(f"above-average relative volume ({rvol:.1f}x)")
        elif rvol >= 1.1:
            score += 5; reasons.append(f"slightly elevated volume ({rvol:.1f}x)")
        elif rvol < 0.6:
            score -= 6; cautions.append("weak volume participation")

    if is_num(change_5d):
        if 2 <= change_5d <= 10:
            score += 14; reasons.append("constructive 5D momentum")
        elif 10 < change_5d <= 18:
            score += 7; cautions.append("large 5D move; chase risk rising")
        elif change_5d > 18:
            score -= 6; cautions.append("5D move may already be overextended")
        elif -3 <= change_5d < 2:
            score += 5; reasons.append("tight recent digestion")

    if is_num(change_1m):
        if 4 <= change_1m <= 25:
            score += 8; reasons.append("positive 1M trend")
        elif change_1m > 35:
            score -= 4; cautions.append("1M move is very extended")

    if is_num(rsi):
        if 45 <= rsi <= 65:
            score += 12; reasons.append("RSI in momentum sweet spot")
        elif 65 < rsi <= 72:
            score += 6; cautions.append("RSI strong but getting hot")
        elif rsi > 75:
            score -= 10; cautions.append("RSI overextended")
        elif 35 <= rsi < 45:
            score += 4; reasons.append("RSI recovering from lower zone")

    if is_num(range_tightness):
        if 4 <= range_tightness <= 12:
            score += 14; reasons.append("tight 20D range compression")
        elif 12 < range_tightness <= 22:
            score += 7; reasons.append("moderate range compression")
        elif range_tightness > 35:
            score -= 5; cautions.append("wide range; less clean compression")

    if is_num(breakout_proximity):
        if -3 <= breakout_proximity <= 1:
            score += 10; reasons.append("near 20D breakout area")
        elif -7 <= breakout_proximity < -3:
            score += 5; reasons.append("within striking distance of 20D high")

    if is_num(dist_50):
        if -4 <= dist_50 <= 6:
            score += 10; reasons.append("near 50D launch zone")
        elif 6 < dist_50 <= 14:
            score += 4; reasons.append("above 50D trend support")
        elif dist_50 > 18:
            score -= 8; cautions.append("far above 50D; extension risk")
        elif dist_50 < -12:
            score -= 6; cautions.append("well below 50D")

    if is_num(volatility_20d):
        if 25 <= volatility_20d <= 70:
            score += 9; reasons.append(f"active 20D volatility ({volatility_20d:.0f}%)")
        elif 70 < volatility_20d <= 110:
            score += 3; cautions.append("very high volatility")
        elif volatility_20d > 110:
            score -= 8; cautions.append("extreme volatility / small-cap risk")
        elif volatility_20d < 15:
            score -= 4; cautions.append("low volatility")

    if is_num(atr_pct):
        if 2 <= atr_pct <= 7:
            score += 5; reasons.append(f"healthy ATR movement ({atr_pct:.1f}%)")
        elif atr_pct > 10:
            score -= 4; cautions.append("ATR is very high")

    if trend_aligned:
        score += 8; reasons.append("50D/200D trend alignment")
    elif is_num(dist_200) and dist_200 < 0:
        score -= 8; cautions.append("below 200D trend filter")

    if is_num(squeeze_score):
        score += min(10, squeeze_score)
        if squeeze_score >= 8:
            reasons.append("squeeze setup confirmed")

    if is_num(days_earnings):
        if 0 <= days_earnings <= 3:
            score += 10; cautions.append("earnings catalyst very close")
        elif 4 <= days_earnings <= 10:
            score += 6; reasons.append("near-term earnings catalyst")
        elif -2 <= days_earnings < 0:
            score += 4; reasons.append("recent earnings reaction window")

    prob = int(max(0, min(100, round(score))))
    label = "Very High" if prob >= 85 else "High" if prob >= 70 else "Medium" if prob >= 50 else "Low"
    reason_text = "; ".join((reasons + cautions)[:7]) if (reasons or cautions) else "No strong explosive-move drivers detected"
    values["Explosive Move Cautions"] = "; ".join(cautions[:5]) if cautions else "None"
    return prob, label, reason_text

def compute_playbook_signal(values):
    score = safe_float(values.get("Score"))
    rsi = safe_float(values.get("RSI"))
    change_5d = safe_float(values.get("5D Price Change %"))
    change_1m = safe_float(values.get("1M Price Change %"))
    dist_50 = safe_float(values.get("Distance from 50D MA %"))
    dist_200 = safe_float(values.get("Distance from 200D MA %"))
    squeeze_setup = str(values.get("Squeeze Setup", "NO"))
    explosive = safe_float(values.get("Explosive Move Probability %"))
    if squeeze_setup == "YES" and is_num(score) and score >= 65:
        return "Squeeze Breakout", "High" if is_num(explosive) and explosive >= 70 else "Medium"
    if is_num(score) and score >= 70 and is_num(dist_50) and -5 <= dist_50 <= 3 and is_num(dist_200) and dist_200 > 0:
        return "Pullback to 50D", "High"
    if is_num(score) and score >= 70 and is_num(change_1m) and change_1m > 0 and is_num(rsi) and 45 <= rsi <= 68:
        return "Trend Continuation", "High" if score >= 80 else "Medium"
    if is_num(rsi) and rsi < 35 and is_num(change_5d) and change_5d < 0:
        return "Oversold Bounce Watch", "Medium"
    if is_num(score) and score < 55:
        return "Avoid / Weak", "Low"
    return "Watchlist / Wait", "Medium" if is_num(score) and score >= 55 else "Low"

def build_ai_score_explanation(values):
    parts = []
    risks = []
    score = safe_float(values.get("Score"))
    revenue = safe_float(values.get("Quarterly Revenue Growth %"))
    eps = safe_float(values.get("EPS Growth %"))
    rsi = safe_float(values.get("RSI"))
    dist_200 = safe_float(values.get("Distance from 200D MA %"))
    dist_50 = safe_float(values.get("Distance from 50D MA %"))
    peg = safe_float(values.get("PEG"))
    analyst = safe_float(values.get("Analyst Target Upside %"))
    short_float = safe_float(values.get("Short % of Float"))
    if is_num(score):
        parts.append(f"GTMD score is {score:.0f}/100, rated {values.get('Rating', 'N/A')}.")
    if is_num(revenue) and revenue > 10:
        parts.append(f"Revenue growth is strong at {revenue:.1f}%.")
    elif is_num(revenue) and revenue <= 0:
        risks.append("revenue growth is not positive")
    if is_num(eps) and eps > 15:
        parts.append(f"EPS growth is strong at {eps:.1f}%.")
    if is_num(dist_200):
        if dist_200 > 0:
            parts.append("Price is above the 200D moving average, which supports the trend.")
        else:
            risks.append("price is below the 200D moving average")
    if is_num(dist_50) and -5 <= dist_50 <= 5:
        parts.append("Price is near the 50D moving average, which can be a cleaner entry zone.")
    if is_num(rsi):
        if 45 <= rsi <= 68:
            parts.append(f"RSI is in a healthy momentum range at {rsi:.1f}.")
        elif rsi > 75:
            risks.append(f"RSI is elevated at {rsi:.1f}")
    if is_num(peg) and peg < 1.2:
        parts.append("PEG is under 1.2, which supports valuation versus growth.")
    if is_num(analyst) and analyst > 10:
        parts.append(f"Analyst target upside is positive at {analyst:.1f}%.")
    if is_num(short_float) and short_float > 20:
        risks.append("short interest is high, which can increase volatility")
    explanation = " ".join(parts[:5]) if parts else "Data is limited, so the explanation is mostly based on available technicals."
    if risks:
        explanation += " Main risk: " + "; ".join(risks[:3]) + "."
    return explanation


def compute_rvol_spike_alert(values):
    """Classify RVOL spike behavior for early breakout timing.
    Uses RVOL + breakout proximity + RSI + recent move + 50D trend to avoid chasing extended moves.
    """
    rvol = safe_float(values.get("Relative Volume"))
    breakout_proximity = safe_float(values.get("Breakout Proximity %"))
    rsi = safe_float(values.get("RSI"))
    change_5d = safe_float(values.get("5D Price Change %"))
    dist_50 = safe_float(values.get("Distance from 50D MA %"))

    if not is_num(rvol):
        return "Unknown", "RVOL unavailable"

    near_breakout = is_num(breakout_proximity) and -3 <= breakout_proximity <= 1
    within_striking = is_num(breakout_proximity) and -7 <= breakout_proximity < -3
    healthy_rsi = is_num(rsi) and 45 <= rsi <= 72
    extended_rsi = is_num(rsi) and rsi > 75
    not_chasing_5d = (not is_num(change_5d)) or change_5d <= 12
    extended_5d = is_num(change_5d) and change_5d > 15
    above_or_near_50d = (not is_num(dist_50)) or dist_50 >= -4

    if rvol >= 2.5 and (extended_rsi or extended_5d):
        return "⚠️ Strong Spike / Chase Risk", f"RVOL {rvol:.1f}x, but RSI/5D move looks extended"
    if rvol >= 2.5 and near_breakout and above_or_near_50d:
        return "🔥 Strong RVOL Breakout", f"RVOL {rvol:.1f}x while price is near the 20D breakout area"
    if rvol >= 1.5 and near_breakout and healthy_rsi and not_chasing_5d and above_or_near_50d:
        return "🔥 Early Breakout Trigger", f"RVOL {rvol:.1f}x + near breakout + healthy RSI"
    if rvol >= 1.5 and (near_breakout or within_striking):
        return "👀 Breakout Watch", f"RVOL {rvol:.1f}x and price is close to the 20D high"
    if rvol >= 1.2:
        return "👀 Warming Up", f"RVOL {rvol:.1f}x is above normal"
    if rvol < 0.8:
        return "Quiet", f"RVOL {rvol:.1f}x shows below-normal activity"
    return "None", f"RVOL {rvol:.1f}x is normal"

def enrich_gtmd_overlays(values):
    earnings_date, days_until, earnings_status = get_next_earnings_date(values.get("Ticker", ""))
    values["Next Earnings Date"] = earnings_date or "N/A"
    values["Days Until Earnings"] = days_until
    values["Earnings Risk"] = earnings_risk_label(days_until)
    values["Earnings Data Status"] = earnings_status
    explosive_prob, explosive_label, explosive_reasons = compute_explosive_move_probability(values)
    values["Explosive Move Probability %"] = explosive_prob
    values["Explosive Move Label"] = explosive_label
    values["Explosive Move Reasons"] = explosive_reasons
    playbook, confidence = compute_playbook_signal(values)
    values["Playbook Signal"] = playbook
    values["Playbook Confidence"] = confidence
    values["AI Score Explanation"] = build_ai_score_explanation(values)
    rvol_alert, rvol_reason = compute_rvol_spike_alert(values)
    values["RVOL Alert"] = rvol_alert
    values["RVOL Alert Reason"] = rvol_reason
    return values

def save_watchlist_snapshot(df, name="Default"):
    if "watchlist_snapshots" not in st.session_state:
        st.session_state["watchlist_snapshots"] = {}
    if df is None or df.empty or "Ticker" not in df.columns:
        return False
    keep_cols = [c for c in ["Ticker", "Score", "Rating", "Playbook Signal", "Explosive Move Probability %", "Earnings Risk", "Alert"] if c in df.columns]
    snap = df[keep_cols].copy()
    st.session_state["watchlist_snapshots"][name] = snap
    return True

def render_watchlist_tracker(df, key_prefix="watchlist"):
    st.markdown("""<div class="gtmd-section-card"><div class="gtmd-card-title">Watchlist Memory + Tracking</div><div class="gtmd-card-subtitle">Save the current results during this session and compare future scans against the saved snapshot.</div></div>""", unsafe_allow_html=True)
    col_a, col_b = st.columns([1, 2])
    with col_a:
        if st.button("Save current scan snapshot", key=f"{key_prefix}_save_snapshot"):
            if save_watchlist_snapshot(df, key_prefix):
                st.success("Snapshot saved for this session.")
            else:
                st.warning("No rows available to save.")
    snap = st.session_state.get("watchlist_snapshots", {}).get(key_prefix)
    if snap is not None and df is not None and not df.empty:
        current_cols = [c for c in ["Ticker", "Score", "Rating", "Playbook Signal", "Explosive Move Probability %", "Earnings Risk", "Alert"] if c in df.columns]
        current = df[current_cols].copy()
        merged = current.merge(snap[["Ticker", "Score"]].rename(columns={"Score": "Previous Score"}), on="Ticker", how="left")
        merged["Score Change"] = merged.apply(lambda r: r["Score"] - r["Previous Score"] if is_num(r.get("Score")) and is_num(r.get("Previous Score")) else np.nan, axis=1)
        with col_b:
            st.caption("Current score compared with the saved session snapshot.")
        st.dataframe(merged.sort_values("Score Change", ascending=False, na_position="last"), use_container_width=True)
    else:
        with col_b:
            st.caption("No saved snapshot yet. Run a scan, save it, then run again later to see score changes.")



# -----------------------------
# PERSISTENT SCANNER RESULT HELPERS
# -----------------------------
def save_scan_results_to_session(df, key):
    """Store scanner/live results so add-to-dashboard controls survive Streamlit reruns."""
    try:
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            st.session_state[key] = df.copy()
            st.session_state[f"{key}_saved_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            return True
    except Exception:
        pass
    return False

def get_scan_results_from_session(key):
    """Return saved scanner/live results."""
    try:
        df = st.session_state.get(key)
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            return df.copy()
    except Exception:
        pass
    return pd.DataFrame()


def order_dashboard_like_columns(df):
    """Put dashboard/calculated tables in the requested Pro Dashboard order when columns exist."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    requested = [
        "Ticker", "Status", "Shares", "Entry Price", "Current Price", "Target Price", "Stop Loss",
        "GTMD Score", "Fundamental Score", "Technical Score", "Timing Score", "Score Breakdown",
        "Explosive Move %", "Squeeze Score", "RVOL Alert", "RVOL Alert Reason",
        "Relative Volume", "Next Earnings Date", "Earnings Risk", "Last Checked", "Notes",
        "Setup Type", "Thesis"
    ]
    rename_map = {
        "Score": "GTMD Score",
        "Explosive Move Probability %": "Explosive Move %",
        "Days Until Earnings": "Days Until Earnings",
    }
    out = df.copy()
    for old, new in rename_map.items():
        if old in out.columns and new not in out.columns:
            out[new] = out[old]
    ordered = [c for c in requested if c in out.columns]
    remaining = [c for c in out.columns if c not in ordered]
    return out[ordered + remaining]


def render_saved_results_with_add_controls(df, title, key_prefix, storage_key):
    """Render persistent results table and add-to-Pro-Dashboard controls."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return

    st.markdown(f"### {title}")
    saved_at = st.session_state.get(f"{storage_key}_saved_at", "")
    if saved_at:
        st.caption(f"Last scan saved: {saved_at}")

    display_df = order_dashboard_like_columns(df)
    display_df = format_scanner_df(display_df).copy() if "format_scanner_df" in globals() else display_df.copy()
    for col in display_df.columns:
        display_df[col] = display_df[col].apply(display_value)
    st.dataframe(display_df, use_container_width=True)

    # Use the newer callback-based controls that save into the real Pro Dashboard dataframe.
    if "render_add_to_watchlist_controls" in globals():
        render_add_to_watchlist_controls(df, key_prefix=key_prefix)
    else:
        render_add_to_pro_dashboard_controls(df, key_prefix=key_prefix)


# -----------------------------
# SCORING MODEL — 100 POINTS
# Pure metric model: no manual confirmations
# -----------------------------
def build_metrics(symbol):
    debug = []

    yahoo_meta, prices, yahoo_error = get_yahoo_chart_history(symbol)

    if not prices.empty:
        debug.append("Yahoo direct chart history loaded")
    else:
        debug.append(f"Yahoo direct chart failed: {yahoo_error}")

    yf_info = get_yfinance_info(symbol)
    yahoo_summary, yahoo_summary_error = get_yahoo_quote_summary(symbol)
    yahoo_quote, yahoo_quote_error = get_yahoo_quote(symbol)
    finviz_snapshot, finviz_error = get_finviz_snapshot(symbol)
    yahoo_page_snapshot, yahoo_page_error = get_yahoo_page_fallback(symbol)

    if yf_info:
        debug.append("yfinance company/fundamental info loaded")
    else:
        debug.append("yfinance company/fundamental info not loaded")

    if yahoo_summary:
        debug.append("Yahoo quoteSummary fallback loaded")
    else:
        debug.append(f"Yahoo quoteSummary fallback not loaded: {yahoo_summary_error}")

    if yahoo_quote:
        debug.append("Yahoo quote fallback loaded")
    else:
        debug.append(f"Yahoo quote fallback not loaded: {yahoo_quote_error}")

    if finviz_snapshot and any(is_num(v) for v in finviz_snapshot.values()):
        debug.append("Finviz fundamentals fallback loaded")
    else:
        debug.append(f"Finviz fundamentals fallback not loaded: {finviz_error}")

    if yahoo_page_snapshot and any(is_num(v) for v in yahoo_page_snapshot.values()):
        debug.append("Yahoo web-page fallback loaded")
    else:
        debug.append(f"Yahoo web-page fallback not loaded: {yahoo_page_error}")

    current_price = safe_float(yahoo_meta.get("regularMarketPrice"))

    if not is_num(current_price):
        current_price = safe_float(yf_info.get("currentPrice"), safe_float(yf_info.get("regularMarketPrice")))

    if not is_num(current_price) and not prices.empty:
        current_price = safe_float(prices["close"].iloc[-1])

    statement_fallbacks = compute_statement_fallbacks(symbol, current_price)
    if statement_fallbacks and any(is_num(v) for v in statement_fallbacks.values()):
        debug.append("Financial statement fallback loaded")
    else:
        debug.append("Financial statement fallback not loaded")

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
        dist_50 = (close / last["sma50"] - 1) * 100 if is_num(safe_float(last["sma50"])) else np.nan
        dist_200 = (close / last["sma200"] - 1) * 100 if is_num(safe_float(last["sma200"])) else np.nan

        returns = prices["close"].pct_change()
        volatility_20d = safe_float(returns.tail(20).std() * np.sqrt(252) * 100) if len(prices) >= 21 else np.nan
        if "volume" in prices.columns and prices["volume"].notna().sum() >= 20:
            prices["avg_volume_20d"] = prices["volume"].rolling(20).mean()
            avg_volume_20d = safe_float(prices["avg_volume_20d"].iloc[-1])
            latest_volume = safe_float(prices["volume"].iloc[-1])
            relative_volume = latest_volume / avg_volume_20d if is_num(latest_volume) and is_num(avg_volume_20d) and avg_volume_20d > 0 else np.nan
        else:
            latest_volume = avg_volume_20d = relative_volume = np.nan
        if all(c in prices.columns for c in ["high", "low", "close"]):
            prev_close = prices["close"].shift(1)
            tr = pd.concat([(prices["high"] - prices["low"]).abs(), (prices["high"] - prev_close).abs(), (prices["low"] - prev_close).abs()], axis=1).max(axis=1)
            atr_14 = safe_float(tr.rolling(14).mean().iloc[-1]) if len(tr) >= 14 else np.nan
            atr_pct = (atr_14 / close) * 100 if is_num(atr_14) and is_num(close) and close > 0 else np.nan
            high_20 = safe_float(prices["high"].tail(20).max()) if prices["high"].notna().sum() >= 20 else np.nan
            low_20 = safe_float(prices["low"].tail(20).min()) if prices["low"].notna().sum() >= 20 else np.nan
            range_tightness_20d = ((high_20 - low_20) / close) * 100 if is_num(high_20) and is_num(low_20) and is_num(close) and close > 0 else np.nan
            breakout_proximity = ((close / high_20) - 1) * 100 if is_num(high_20) and high_20 > 0 and is_num(close) else np.nan
        else:
            atr_pct = range_tightness_20d = breakout_proximity = np.nan
        sma50 = safe_float(last["sma50"])
        sma200 = safe_float(last["sma200"])
        trend_alignment = bool(is_num(close) and is_num(sma50) and is_num(sma200) and close > sma50 > sma200)
    else:
        change_5d = change_1m = rsi = dist_50 = dist_200 = np.nan
        volatility_20d = latest_volume = avg_volume_20d = relative_volume = np.nan
        atr_pct = range_tightness_20d = breakout_proximity = np.nan
        trend_alignment = False

    # Fundamentals: try yfinance first, then direct Yahoo quoteSummary fallback.
    ttm_pe = first_valid(
        yf_info.get("trailingPE"),
        yahoo_quote.get("trailingPE"),
        nested_raw(yahoo_summary, "summaryDetail", "trailingPE"),
        nested_raw(yahoo_summary, "defaultKeyStatistics", "trailingPE"),
        finviz_snapshot.get("trailingPE"),
        yahoo_page_snapshot.get("trailingPE"),
        statement_fallbacks.get("ttmPEFromEPS"),
    )

    forward_pe = first_valid(
        yf_info.get("forwardPE"),
        yahoo_quote.get("forwardPE"),
        nested_raw(yahoo_summary, "defaultKeyStatistics", "forwardPE"),
        nested_raw(yahoo_summary, "summaryDetail", "forwardPE"),
        finviz_snapshot.get("forwardPE"),
        yahoo_page_snapshot.get("forwardPE"),
    )

    peg = first_valid(
        yf_info.get("pegRatio"),
        yahoo_quote.get("pegRatio"),
        nested_raw(yahoo_summary, "defaultKeyStatistics", "pegRatio"),
        finviz_snapshot.get("pegRatio"),
        yahoo_page_snapshot.get("pegRatio"),
    )

    eps_growth_raw = first_valid(
        yf_info.get("earningsQuarterlyGrowth"),
        nested_raw(yahoo_summary, "financialData", "earningsGrowth"),
    )
    eps_growth = eps_growth_raw * 100 if is_num(eps_growth_raw) else first_valid(
        statement_fallbacks.get("epsGrowthPercent"),
        finviz_snapshot.get("epsGrowthPast5YPercent"),
    )

    # Yahoo/yfinance revenueGrowth is most recent quarter YoY growth.
    revenue_growth_raw = first_valid(
        yf_info.get("revenueGrowth"),
        nested_raw(yahoo_summary, "financialData", "revenueGrowth"),
    )
    revenue_growth = revenue_growth_raw * 100 if is_num(revenue_growth_raw) else first_valid(
        statement_fallbacks.get("revenueGrowthPercent"),
        finviz_snapshot.get("salesGrowthPast5YPercent"),
    )

    avg_target = first_valid(
        yf_info.get("targetMeanPrice"),
        yahoo_quote.get("targetMeanPrice"),
        nested_raw(yahoo_summary, "financialData", "targetMeanPrice"),
        finviz_snapshot.get("targetMeanPrice"),
        yahoo_page_snapshot.get("targetMeanPrice"),
    )

    analyst_upside = (
        ((avg_target / current_price) - 1) * 100
        if is_num(current_price) and current_price > 0 and is_num(avg_target) and avg_target > 0
        else np.nan
    )

    forward_pe_less_than_ttm = (
        bool(forward_pe < ttm_pe)
        if is_num(forward_pe) and is_num(ttm_pe)
        else False
    )

    peg_under_1_2 = bool(peg < 1.2) if is_num(peg) else False

    short_float_raw = first_valid(
        yf_info.get("shortPercentOfFloat"),
        yahoo_quote.get("shortPercentOfFloat"),
        nested_raw(yahoo_summary, "defaultKeyStatistics", "shortPercentOfFloat"),
        yahoo_page_snapshot.get("shortPercentOfFloat"),
    )
    short_float = short_float_raw * 100 if is_num(short_float_raw) else first_valid(
        finviz_snapshot.get("shortPercentOfFloatPercent"),
        yahoo_page_snapshot.get("shortPercentOfFloatPercent")
    )

    total_cash = first_valid(
        yf_info.get("totalCash"),
        yahoo_quote.get("totalCash"),
        nested_raw(yahoo_summary, "financialData", "totalCash"),
        statement_fallbacks.get("totalCash"),
    )
    operating_cf = get_operating_cash_flow(symbol)

    cash_runway_months = np.nan
    if is_num(total_cash) and total_cash > 0 and is_num(operating_cf) and operating_cf < 0:
        monthly_burn = abs(operating_cf) / 12
        cash_runway_months = total_cash / monthly_burn if monthly_burn > 0 else np.nan
    elif is_num(total_cash) and total_cash > 0 and is_num(operating_cf) and operating_cf >= 0:
        # Positive operating cash flow means no burn-rate runway issue.
        cash_runway_months = 999

    # Smarter squeeze score is separate from the main 100-point GTMD score.
    # It detects short-interest fuel + compression + breakout readiness,
    # then penalizes trap conditions like overextension, weak volume, and extreme volatility.
    squeeze_seed_values = {
        "Short % of Float": short_float,
        "5D Price Change %": change_5d,
        "RSI": rsi,
        "Distance from 50D MA %": dist_50,
        "Distance from 200D MA %": dist_200,
        "20D Volatility %": volatility_20d,
        "Relative Volume": relative_volume,
        "ATR %": atr_pct,
        "20D Range Tightness %": range_tightness_20d,
        "Breakout Proximity %": breakout_proximity,
        "Trend Alignment": trend_alignment,
    }
    squeeze_score, squeeze_setup, squeeze_label, squeeze_reasons, squeeze_cautions = compute_smart_squeeze_score(squeeze_seed_values)

    values = {
        "Ticker": symbol,
        "Current Price": current_price,
        "TTM P/E": ttm_pe,
        "Forward P/E": forward_pe,
        "Forward P/E < TTM P/E": forward_pe_less_than_ttm,
        "PEG": peg,
        "PEG < 1.2": peg_under_1_2,
        "EPS Growth %": eps_growth,
        "Quarterly Revenue Growth %": revenue_growth,
        "Revenue Growth %": revenue_growth,
        "5D Price Change %": change_5d,
        "1M Price Change %": change_1m,
        "RSI": rsi,
        "Distance from 50D MA %": dist_50,
        "Distance from 200D MA %": dist_200,
        "20D Volatility %": volatility_20d,
        "Latest Volume": latest_volume,
        "Average Volume 20D": avg_volume_20d,
        "Relative Volume": relative_volume,
        "ATR %": atr_pct,
        "20D Range Tightness %": range_tightness_20d,
        "Breakout Proximity %": breakout_proximity,
        "Trend Alignment": trend_alignment,
        "Analyst Target Upside %": analyst_upside,
        "Short % of Float": short_float,
        "Total Cash": total_cash,
        "Operating Cash Flow": operating_cf,
        "Cash Runway Months": cash_runway_months,
        "Squeeze Score": squeeze_score,
        "Squeeze Setup": squeeze_setup,
        "Squeeze Label": squeeze_label,
        "Squeeze Reasons": squeeze_reasons,
        "Squeeze Cautions": squeeze_cautions,
        "Fundamental Data Status": "Loaded" if any(is_num(x) for x in [ttm_pe, forward_pe, peg, eps_growth, revenue_growth, avg_target, short_float, total_cash]) else "Limited / unavailable from Yahoo",
        "Data Source": "Yahoo direct chart + yfinance + Yahoo quoteSummary fallback",
    }

    score = 0
    details = []

    # FUNDAMENTALS — 40 POINTS
    pts = 10 if is_num(eps_growth) and eps_growth > 15 else 5 if is_num(eps_growth) and eps_growth > 0 else 0
    score += pts
    details.append(("EPS growth", pts, eps_growth, "Max 10"))

    pts = (
        15 if is_num(revenue_growth) and revenue_growth > 20 else
        10 if is_num(revenue_growth) and revenue_growth > 10 else
        5 if is_num(revenue_growth) and revenue_growth > 0 else
        0
    )
    score += pts
    details.append(("Quarterly revenue growth", pts, revenue_growth, "Max 15"))

    pts = 7 if is_num(analyst_upside) and analyst_upside > 10 else 3 if is_num(analyst_upside) and analyst_upside > 0 else 0
    score += pts
    details.append(("Analyst target upside", pts, analyst_upside, "Max 7"))

    pts = (
        8 if is_num(cash_runway_months) and cash_runway_months > 24 else
        5 if is_num(cash_runway_months) and cash_runway_months > 12 else
        2 if is_num(cash_runway_months) and cash_runway_months > 6 else
        0
    )
    score += pts
    runway_display = "Positive OCF / no burn issue" if cash_runway_months == 999 else cash_runway_months
    details.append(("Cash runway", pts, runway_display, "Max 8"))

    # VALUATION — 20 POINTS
    pts = 6 if forward_pe_less_than_ttm else 0
    score += pts
    details.append(("Forward P/E < TTM P/E", pts, forward_pe_less_than_ttm, "Max 6"))

    pts = 8 if peg_under_1_2 else 4 if is_num(peg) and peg < 1.8 else 0
    score += pts
    details.append(("PEG", pts, peg, "Max 8"))

    pts = 6 if is_num(ttm_pe) and 0 < ttm_pe < 35 else 3 if is_num(ttm_pe) and 35 <= ttm_pe < 60 else 0
    score += pts
    details.append(("P/E ratio reasonableness", pts, ttm_pe, "Max 6"))

    # MOMENTUM / TECHNICALS — 32 POINTS
    pts = score_range(change_5d, [
        (lambda x: -8 <= x <= -3, 6),
        (lambda x: -3 < x <= 3, 4),
        (lambda x: 3 < x <= 8, 2),
        (lambda x: x > 8, -2),
    ])
    score += pts
    details.append(("5-day price change", pts, change_5d, "Max 6"))

    pts = 6 if is_num(change_1m) and change_1m > 0 else 3 if is_num(change_1m) and -5 <= change_1m <= 0 else 0
    score += pts
    details.append(("1-month price change", pts, change_1m, "Max 6"))

    pts = score_range(rsi, [
        (lambda x: 30 <= x <= 45, 8),
        (lambda x: 45 < x <= 60, 6),
        (lambda x: 60 < x <= 70, 3),
        (lambda x: x < 30, 3),
        (lambda x: x > 70, -4),
    ])
    score += pts
    details.append(("RSI", pts, rsi, "Max 8"))

    pts = 6 if is_num(dist_50) and -5 <= dist_50 <= 2 else 3 if is_num(dist_50) and -10 <= dist_50 < -5 else 0
    score += pts
    details.append(("Distance from 50D MA", pts, dist_50, "Max 6"))

    pts = 6 if is_num(dist_200) and dist_200 > 0 else -6 if is_num(dist_200) and dist_200 < 0 else 0
    score += pts
    details.append(("Distance from 200D MA", pts, dist_200, "Max 6"))

    # SHORT INTEREST RISK FILTER — 8 POINTS
    # Low short interest earns a small safety premium.
    # High short interest does NOT receive a score penalty here.
    # Instead, high short interest is shown as a separate warning flag.
    # Squeeze potential remains separate in Squeeze Score / Squeeze Setup.
    pts = (
        8 if is_num(short_float) and short_float <= 5 else
        5 if is_num(short_float) and short_float <= 10 else
        2 if is_num(short_float) and short_float <= 20 else
        0 if is_num(short_float) and short_float > 20 else
        0
    )
    score += pts
    details.append(("Short interest risk filter", pts, short_float, "Max 8 / No penalty"))

    raw_score = score
    score = max(0, min(100, score))

    detail_df = pd.DataFrame(details, columns=["Factor", "Points", "Value", "Max Points"])
    values["Raw Score Before Cap"] = raw_score
    values["Score"] = score
    values["Rating"] = rating(score)

    # Option A: explanatory score layers only. These do not alter the existing Score,
    # Squeeze Score, Explosive Move Probability, RVOL Alert, or Playbook Signal.
    fundamental_score, technical_score, timing_score = compute_gtmd_score_layers(values, detail_df)
    values["Fundamental Score"] = fundamental_score
    values["Technical Score"] = technical_score
    values["Timing Score"] = timing_score
    values["Score Breakdown"] = (
        f"Fundamental {display_value(fundamental_score)}/100 | "
        f"Technical {display_value(technical_score)}/100 | "
        f"Timing {display_value(timing_score)}/100"
    )

    if is_num(short_float) and short_float > 20:
        values["Short Interest Warning"] = "HIGH RISK: short float > 20%"
    elif is_num(short_float) and short_float > 10:
        values["Short Interest Warning"] = "Moderate risk: short float > 10%"
    else:
        values["Short Interest Warning"] = "No major short-interest risk flag"

    alert, action, reasons, _ = generate_trade_alert(values)
    values["Alert"] = alert
    values["Suggested Action"] = action
    values["Alert Reasons"] = "; ".join(reasons)

    values = enrich_gtmd_overlays(values)

    return values, score, detail_df, prices, debug


def format_scanner_df(df):
    columns = [
        "Ticker", "Score", "Rating", "Fundamental Score", "Technical Score", "Timing Score", "Score Breakdown",
        "Playbook Signal", "Playbook Confidence",
        "Explosive Move Probability %", "Explosive Move Label", "RVOL Alert", "RVOL Alert Reason", "Relative Volume", "20D Volatility %", "ATR %", "20D Range Tightness %", "Breakout Proximity %", "Earnings Risk", "Days Until Earnings",
        "Alert", "Suggested Action", "Squeeze Score", "Squeeze Setup", "Squeeze Label",
        "Current Price", "RSI", "5D Price Change %", "1M Price Change %",
        "Distance from 50D MA %", "Distance from 200D MA %",
        "TTM P/E", "Forward P/E", "Forward P/E < TTM P/E", "PEG", "PEG < 1.2",
        "EPS Growth %", "Quarterly Revenue Growth %", "Analyst Target Upside %",
        "Short % of Float", "Short Interest Warning", "Cash Runway Months",
    ]
    available = [c for c in columns if c in df.columns]
    return df[available]



# -----------------------------
# LIVE UNIVERSE + FAST PRE-RANK ENGINE
# -----------------------------
DEFAULT_FALLBACK_UNIVERSE = """AAPL,MSFT,NVDA,GOOGL,GOOG,AMZN,META,TSLA,AVGO,LLY,JPM,V,UNH,XOM,MA,COST,NFLX,PG,JNJ,HD,WMT,ABBV,BAC,KO,CRM,ORCL,CVX,AMD,PEP,MRK,ADBE,LIN,CSCO,TMO,ACN,MCD,QCOM,IBM,ABT,GE,DIS,PM,TXN,INTU,VZ,CAT,AMAT,UBER,GS,AXP,MS,ISRG,NOW,NEE,PFE,RTX,SPGI,UNP,LOW,CMCSA,AMGN,SHOP,HON,BLK,SYK,BA,ETN,PANW,COP,TJX,DE,ADP,VRTX,MDT,ADI,LMT,GILD,MMC,PLD,CB,UPS,ANET,REGN,AMT,C,SBUX,KLAC,MO,SO,FI,INTC,ICE,NKE,DUK,MDLZ,EQIX,SHW,WM,ELV,APH,CDNS,CEG,PH,TT,MMM,AJG,CI,ORLY,WELL,TDG,MCO,CTAS,ABNB,COF,MSI,CRWD,ECL,ITW,USB,PNC,APO,CL,CMG,EMR,GD,FTNT,APD,NOC,BDX,CSX,TGT,FDX,SLB,MCK,ROP,HLT,ADSK,WDAY,ZTS,OKE,AON,TFC,TRV,SPOT,NSC,BK,AZO,GM,PCAR,CARR,PSX,ALL,DLR,FCX,MPC,AEP,KMI,URI,ROST,DHI,CPRT,WMB,AMP,JCI,DFS,PSA,O,CMI,FAST,EW,SRE,CCI,AME,AXON,FANG,HES,KR,LULU,MSCI,IDXX,GWW,PRU,VLO,OTIS,PEG,CTVA,PAYX,YUM,VST,GLW,EA,XEL,KDP,CBRE,NUE,RMD,ED,ETR,KVUE,FICO,EXC,D,LYV,WTW,ACGL,IR,IT,ODFL,HIG,TRGP,PCG,EFX,HPQ,IQV,MLM,MPWR,GEV,MTB,EIX,DD,ROK,AVB,VICI,XYL,BKR,WAB,TSCO,FIS,HUM,HWM,EBAY,UAL,IRM,ANSS,KEYS,CAH,BRO,LEN,STZ,GRMN,STT,MTD,DOW,PPG,BR,CDW,DOV,NVR,AWK,TROW,HAL,FITB,FE,WEC,VLTO,ADM,TYL,GPN,SBAC,PHM,DGX,HBAN,NTAP,CBOE,VMC,CHD,DECK,CPAY,CCL,STE,PPL,EQR,ES,ATO,EXR,BIIB,LYB,FTV,RJF,TDY,CTRA,WBD,WSM,ULTA,ON,RF,DRI,TSN,CINF,DTE,WRB,PODD,WY,STX,LDOS,NDAQ,LUV,TER,EL,NI,NRG,PTC,OMC,EVRG,MOH,HPE,CLX,CFG,ESS,ZBH,MAA,LH,PKG,ARE,BBY,COO,BALL,HOLX,FDS,JBHT,APTV,BAX,TXT,IEX,GEN,EXPE,SWKS,ALGN,CF,MKTX,DPZ,AKAM,INCY,VRSN,POOL,NDSN,JKHY,FFIV,SNA,PNR,UDR,BG,EPAM,CHRW,CPT,REG,UHS,AIZ,GNRC,IP,KMX,FOXA,FOX,RL,NWSA,NWS,PARA,WYNN,MTCH,TECH,QRVO,DAY,LKQ,MGM,HAS,HSIC,ALLE,AOS,TPR,GL,JNPR,MOS,IVZ,APA,MHK,CPB,FMC,CRL,BEN,PNW,HRL,DVA,ERIE,SW,SMCI,PLTR,SNOW,NET,DDOG,ZS,OKTA,TEAM,TTD,ROKU,PINS,SNAP,COIN,HOOD,SOFI,RIVN,LCID,NIO,XPEV,LI,ARM,TSM,ASML,MRVL,MU,ON,MPWR,DELL,HPE,NTNX,ESTC,HUBS,APP,AFRM,UPST,SE,JD,BABA,BIDU,PDD,TCOM,NTES,SONY,TM,F,GM,RACE,MBLY,LAZR,QS,CHPT,BLNK,PLUG,BE,FCEL,FSLR,ENPH,SEDG,RUN,NOVA,NEE,DUK,SO,D,AEP,EXC,XEL,SRE,PEG,ED,PCG,NRG,CVX,XOM,SLB,HAL,BKR,OXY,MPC,PSX,VLO,EOG,APA,HES,COP,OKE,WMB,KMI,SU,CNQ,BP,SHEL,TTE,FCX,NEM,GOLD,AEM,AA,ALB,LTHM,VALE,CLF,X,STLD,NUE,SCCO,TECK,CCJ,UEC,UUUU,DNN,IONQ,RGTI,QBTS,ASTS,RKLB,JOBY,ACHR,KTOS,AVAV,TXT,HON,MMM,EMR,ROK,PH,ITW,ETN,IR,XYL,TT,CMI,PCAR,DOV,FAST,GWW,URI,WSO,POOL,SWK,WHR,GNRC,TSCO,AZO,ORLY,AN,SAH,LAD,PAG,KMX,CPRT,ACGL,TRV,ALL,PGR,CB,MMC,AON,AJG,BRO,WTW,HIG,CINF,WRB,RLI,AXS,RE,GL,PRU,UNM,AMP,TROW,IVZ,BEN,NDAQ,ICE,CBOE,MCO,SPGI,MSCI,FICO,EXPD,JBHT,CHRW,LSTR,ODFL,SAIA,ARCB,KNX,XPO,TFII,CP,ZBRA,GRMN,JKHY,BR,FLT,FIS,GPN,PAYX,ADP,CTSH,INFY,ACN,IBM,HPQ,DELL,STX,WDC,LOGI,SONO,HPE,NTAP,SMAR,ESTC,DBX,BOX,PD,ZI,HUBS,APP,AFRM,UPST,SOFI,LC,OPEN,ZG,Z,EXPE,TRIP,BKNG,CCL,RCL,NCLH,HLT,MAR,H,WH,CHH,DRI,TXRH,CAKE,PLAY,DKNG,PENN,MGM,CZR,WYNN,LVS,TWLO,BAND"""

def normalize_symbol_for_yahoo(symbol):
    return str(symbol).strip().upper().replace(".", "-")

@st.cache_data(ttl=86400, show_spinner=False)
def get_live_stock_universe(max_symbols=5000):
    urls = [
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
    ]
    tickers, notes = [], []
    for url in urls:
        try:
            r = requests.get(url, timeout=20, headers=YAHOO_HEADERS)
            if r.status_code != 200 or "Symbol" not in r.text:
                notes.append(f"{url.split('/')[-1]} unavailable")
                continue
            lines = [line for line in r.text.splitlines() if "|" in line and not line.startswith("File Creation")]
            header = lines[0].split("|")
            for line in lines[1:]:
                row = dict(zip(header, line.split("|")))
                sym = row.get("Symbol") or row.get("ACT Symbol")
                if not sym or row.get("Test Issue", "N") == "Y" or row.get("ETF", "N") == "Y":
                    continue
                if row.get("Financial Status", "") in ["D", "E", "Q", "G"]:
                    continue
                sym = normalize_symbol_for_yahoo(sym)
                if sym and "$" not in sym and " " not in sym:
                    tickers.append(sym)
            notes.append(f"{url.split('/')[-1]} loaded")
        except Exception as e:
            notes.append(f"{url.split('/')[-1]} failed: {e}")
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        tickers = [t.strip().upper() for t in DEFAULT_FALLBACK_UNIVERSE.split(",") if t.strip()]
        notes.append("Fallback liquid universe used")
    return tickers[:max_symbols], "; ".join(notes)

@st.cache_data(ttl=3600, show_spinner=False)
def fast_prerank_metrics(symbol, retries=1, backoff_seconds=0.4):
    """Fast Yahoo-only pre-rank pull with light retry/backoff for Streamlit Cloud throttling."""
    symbol = normalize_symbol_for_yahoo(symbol)
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"range": "3mo", "interval": "1d", "includePrePost": "false"}

    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, headers=YAHOO_HEADERS, timeout=12)

            if r.status_code in [429, 500, 502, 503, 504]:
                if attempt < retries:
                    time.sleep(backoff_seconds * (attempt + 1))
                    continue
                return None

            if r.status_code != 200:
                return None

            result = r.json().get("chart", {}).get("result", [])
            if not result:
                if attempt < retries:
                    time.sleep(backoff_seconds * (attempt + 1))
                    continue
                return None

            quote = result[0].get("indicators", {}).get("quote", [{}])[0]
            close = pd.Series(quote.get("close", []), dtype="float64").dropna()
            volume = pd.Series(quote.get("volume", []), dtype="float64").dropna()
            if len(close) < 25 or len(volume) < 20:
                return None

            price = safe_float(close.iloc[-1])
            latest_volume = safe_float(volume.iloc[-1])
            avg_vol_20 = safe_float(volume.tail(20).mean())
            relative_volume = latest_volume / avg_vol_20 if is_num(latest_volume) and is_num(avg_vol_20) and avg_vol_20 > 0 else np.nan
            change_5d = safe_float(((close.iloc[-1] / close.iloc[-6]) - 1) * 100) if len(close) > 6 else np.nan
            volatility_20d = safe_float(close.pct_change().dropna().tail(20).std() * np.sqrt(252) * 100)
            dollar_volume = price * avg_vol_20 if is_num(price) and is_num(avg_vol_20) else np.nan

            # Pre-rank is intentionally lightweight: liquidity + movement + reasonable volatility.
            # Extreme volatility gets capped/penalized so wild small-caps do not dominate purely from big swings.
            pre_rank_score = 0
            if is_num(dollar_volume):
                pre_rank_score += min(60, np.log10(max(dollar_volume, 1)) * 6)
            if is_num(volatility_20d):
                if volatility_20d <= 75:
                    pre_rank_score += min(22, volatility_20d / 4)
                elif volatility_20d <= 100:
                    pre_rank_score += 18
                else:
                    pre_rank_score += max(5, 18 - ((volatility_20d - 100) / 10))
            if is_num(change_5d):
                pre_rank_score += min(18, abs(change_5d) * 1.2)

            return {"Ticker": symbol, "Price": price, "Latest Volume": latest_volume, "Avg Volume 20D": avg_vol_20, "Relative Volume": relative_volume, "Dollar Volume": dollar_volume, "20D Volatility %": volatility_20d, "5D Change %": change_5d, "Pre-Rank Score": pre_rank_score}
        except Exception:
            if attempt < retries:
                time.sleep(backoff_seconds * (attempt + 1))
                continue
            return None
    return None

def build_preranked_universe(tickers, min_price, min_avg_volume, min_volatility, min_rvol=0.0, max_workers=20):
    rows, errors = [], 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fast_prerank_metrics, t): t for t in tickers}
        for fut in as_completed(futures):
            try:
                row = fut.result()
                if row is None:
                    errors += 1
                    continue
                if is_num(row["Price"]) and row["Price"] < min_price:
                    continue
                if is_num(row["Avg Volume 20D"]) and row["Avg Volume 20D"] < min_avg_volume:
                    continue
                if is_num(row["20D Volatility %"]) and row["20D Volatility %"] < min_volatility:
                    continue
                if min_rvol > 0 and is_num(row.get("Relative Volume")) and row["Relative Volume"] < min_rvol:
                    continue
                rows.append(row)
            except Exception:
                errors += 1
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Pre-Rank Score", ascending=False).reset_index(drop=True)
    return df, errors

def scan_tickers_parallel(tickers, max_workers=8):
    results, errors = [], []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(build_metrics, t): t for t in tickers}
        for fut in as_completed(futures):
            ticker = futures[fut]
            try:
                values, score, _, _, debug = fut.result()
                if not is_num(safe_float(values.get("Current Price"))):
                    errors.append(f"{ticker}: no price data")
                else:
                    results.append(values)
            except Exception as e:
                errors.append(f"{ticker}: {e}")
    return results, errors



# -----------------------------
# QUICK TRADE MODE (+4% TARGET)
# -----------------------------
QUICK_TRADE_DEFAULT_RANGES = {
    "Relative Volume": (1.5, 3.5),
    "Timing Score": (70.0, 85.0),
    "Explosive Move Probability %": (65.0, 85.0),
    "Technical Score": (70.0, 88.0),
    "5D Price Change %": (0.0, 8.0),
    "Breakout Proximity %": (-3.0, 1.0),
    "Squeeze Score": (5.0, 8.0),
    "Score": (68.0, 87.0),
    "Fundamental Score": (55.0, 80.0),
}

QUICK_TRADE_METRICS = [
    ("Priority 1", "RVOL", "Relative Volume", "1.5–3.5x", "Confirms active participation and momentum", 3),
    ("Priority 1", "Timing Score", "Timing Score", "70–85", "Best short-term entry quality", 3),
    ("Priority 1", "Explosive Probability", "Explosive Move Probability %", "65–85%", "Measures likelihood of fast move", 3),
    ("Priority 2", "Technical Score", "Technical Score", "70–88", "Confirms healthy chart/trend", 2),
    ("Priority 2", "5D Price Change", "5D Price Change %", "0% to +8%", "Momentum without chasing", 2),
    ("Priority 2", "Breakout Proximity", "Breakout Proximity %", "-3% to +1%", "Near breakout zone before extension", 2),
    ("Priority 2", "Squeeze Score", "Squeeze Score", "5–8 / 10", "Strong compression/breakout setup", 2),
    ("Priority 3", "GTMD Score", "Score", "68–87", "Overall quality safety filter", 1),
    ("Priority 3", "Fundamental Score", "Fundamental Score", "55–80", "Avoid weak/unstable companies", 1),
]

def render_quick_trade_criteria_table(ranges):
    """Render the preset quick-trade criteria table."""
    rows = []
    for priority, metric, col, default_range, role, weight in QUICK_TRADE_METRICS:
        low, high = ranges.get(col, QUICK_TRADE_DEFAULT_RANGES[col])
        suffix = "x" if col == "Relative Volume" else "%" if col in ["Explosive Move Probability %", "5D Price Change %", "Breakout Proximity %"] else ""
        if col == "Squeeze Score":
            range_text = f"{low:g}–{high:g} / 10"
        elif col in ["5D Price Change %", "Breakout Proximity %"]:
            high_text = f"+{high:g}%" if high >= 0 else f"{high:g}%"
            range_text = f"{low:g}% to {high_text}"
        elif suffix:
            range_text = f"{low:g}–{high:g}{suffix}"
        else:
            range_text = f"{low:g}–{high:g}"
        rows.append({"Priority": priority, "Metric": metric, "Ideal Range": range_text, "Weight": weight, "Role": role})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

def quick_trade_pass(value, low, high):
    v = safe_float(value)
    if not is_num(v):
        return False
    return low <= v <= high

def add_quick_trade_columns(df, ranges):
    """Add quick-trade fit columns without changing any original scanner scores."""
    if df is None or df.empty:
        return df
    out = df.copy()
    total_possible = sum(item[5] for item in QUICK_TRADE_METRICS)
    labels = []
    fit_scores = []
    p1_counts = []
    p2_counts = []
    p3_counts = []
    missing_notes = []

    for _, row in out.iterrows():
        points = 0
        p1 = p2 = p3 = 0
        misses = []
        for priority, metric, col, _default_range, _role, weight in QUICK_TRADE_METRICS:
            low, high = ranges.get(col, QUICK_TRADE_DEFAULT_RANGES[col])
            passed = quick_trade_pass(row.get(col), low, high)
            if passed:
                points += weight
                if priority == "Priority 1":
                    p1 += 1
                elif priority == "Priority 2":
                    p2 += 1
                else:
                    p3 += 1
            else:
                misses.append(metric)

        fit_pct = round((points / total_possible) * 100, 1) if total_possible else np.nan
        if p1 == 3 and fit_pct >= 85:
            label = "A+ Quick Trade"
        elif p1 >= 2 and fit_pct >= 70:
            label = "Good Quick Trade"
        elif p1 >= 1 and fit_pct >= 55:
            label = "Watch / Needs Confirmation"
        else:
            label = "Avoid / Not Quick Trade"

        labels.append(label)
        fit_scores.append(fit_pct)
        p1_counts.append(f"{p1}/3")
        p2_counts.append(f"{p2}/4")
        p3_counts.append(f"{p3}/2")
        missing_notes.append(", ".join(misses[:5]) if misses else "All criteria met")

    out["Quick Trade Label"] = labels
    out["Quick Trade Fit %"] = fit_scores
    out["Priority 1 Matches"] = p1_counts
    out["Priority 2 Matches"] = p2_counts
    out["Priority 3 Matches"] = p3_counts
    out["Quick Trade Missing"] = missing_notes
    return out

# -----------------------------
# UI
# -----------------------------
st.markdown(
    """
    <div class="gtmd-hero">
        <div class="gtmd-title-wrap">
            <div class="gtmd-logo">✓</div>
            <div>
                <div class="gtmd-title">GTMD Scanner</div>
                <div class="gtmd-subtitle">Premium stock scoring, squeeze detection, and watchlist scanner with a clean card-based trading dashboard.</div>
                <div class="gtmd-pill-row">
                    <div class="gtmd-pill">System Protected</div>
                    <div class="gtmd-pill gtmd-pill-purple">GTMD Data Scanner Active</div>
                    <div class="gtmd-pill gtmd-pill-purple">Metric-Based Model</div>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("Educational stock scoring tool only. Not financial advice.")

mode = st.sidebar.radio(
    "Mode",
    [
        "Single Stock Analyzer",
        "Stock Scanner",
        "Quick Trade Mode (+4% Target)",
        "Live Universe Scanner",
        "Potential Stocks Spreadsheet",
        "Position Exit Calculator",
        "Notes",
    ],
)

st.sidebar.divider()
st.sidebar.caption("GTMD Data Source")
st.sidebar.write("Market chart data: ✅ Built-in")
st.sidebar.write("1H candlestick chart: ✅ TradingView widget")
st.sidebar.write("Live universe: ✅ NasdaqTrader daily stock list")
st.sidebar.write("yfinance info backup:", "✅ Available" if yf is not None else "❌ Missing yfinance")
st.sidebar.caption("Pure metric model plus playbook, earnings, explosive probability, and session watchlist tracking.")


# -----------------------------
# PRO WATCHLIST SYNC HELPERS
# -----------------------------
PRO_DASHBOARD_COLUMNS = [
    "Ticker",
    "Status",
    "Shares",
    "Entry Price",
    "Current Price",
    "Target Price",
    "Stop Loss",
    "GTMD Score",
    "Explosive Move %",
    "Squeeze Score",
    "RVOL Alert",
    "RVOL Alert Reason",
    "Relative Volume",
    "Next Earnings Date",
    "Earnings Risk",
    "Last Checked",
    "Notes",
    "Setup Type",
    "Thesis",
    "Priority",
]

PRO_NUMERIC_COLUMNS = [
    "Entry Price", "Current Price", "Target Price", "Stop Loss", "Shares",
    "GTMD Score", "Explosive Move %", "Squeeze Score", "Relative Volume"
]

# Persistent local save file for the Pro Trader Dashboard watchlist.
# On Streamlit Cloud this persists during normal app use, but you should still download
# CSV backups periodically because cloud file systems can reset on redeploy/restart.
PRO_WATCHLIST_FILE = "gtmd_pro_trader_watchlist_saved.csv"

def empty_pro_dashboard_df():
    """Return a clean, empty Pro Trader Dashboard dataframe with the correct columns."""
    return pd.DataFrame(columns=PRO_DASHBOARD_COLUMNS)

def clean_pro_dashboard_df(df):
    """Normalize dashboard columns and dtypes after loading, editing, or importing."""
    if df is None or df.empty:
        df = empty_pro_dashboard_df()
    else:
        df = df.copy()
    for col in PRO_DASHBOARD_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan if col in PRO_NUMERIC_COLUMNS else ""
    df = df[PRO_DASHBOARD_COLUMNS].copy()
    for col in PRO_NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Ticker" in df.columns:
        df["Ticker"] = df["Ticker"].fillna("").astype(str).str.upper().str.strip()
        df = df[df["Ticker"].astype(str).str.strip() != ""].copy()
        df = df.drop_duplicates(subset=["Ticker"], keep="last").reset_index(drop=True)
    return df

def load_pro_watchlist_from_disk():
    """Load saved watchlist from local CSV, or start empty if no saved file exists."""
    try:
        if os.path.exists(PRO_WATCHLIST_FILE):
            return clean_pro_dashboard_df(pd.read_csv(PRO_WATCHLIST_FILE))
    except Exception as e:
        st.warning(f"Could not load saved Pro Dashboard watchlist: {e}")
    return empty_pro_dashboard_df()

def save_pro_watchlist_to_disk(df):
    """Auto-save Pro Dashboard watchlist to local CSV."""
    try:
        clean_df = clean_pro_dashboard_df(df)
        clean_df.to_csv(PRO_WATCHLIST_FILE, index=False)
        return True
    except Exception as e:
        st.warning(f"Could not auto-save Pro Dashboard watchlist: {e}")
        return False

def ensure_pro_dashboard_df():
    """Create/fix the Pro Trader Dashboard dataframe in session_state, loading saved data if available."""
    if "pro_trader_dashboard_df" not in st.session_state:
        st.session_state["pro_trader_dashboard_df"] = load_pro_watchlist_from_disk()
    df = clean_pro_dashboard_df(st.session_state["pro_trader_dashboard_df"])
    st.session_state["pro_trader_dashboard_df"] = df.copy()
    return st.session_state["pro_trader_dashboard_df"]

def infer_setup_type_from_scanner_row(row):
    playbook = str(row.get("Playbook Signal", "")).strip()
    squeeze = str(row.get("Squeeze Setup", "")).strip().upper()
    if squeeze == "YES":
        return "Squeeze"
    if "Pullback" in playbook:
        return "Pullback"
    if "Trend" in playbook:
        return "Trend Continuation"
    if "Breakout" in playbook:
        return "Breakout"
    if "Oversold" in playbook:
        return "Oversold Bounce"
    return "Other"

def add_scanner_rows_to_pro_watchlist(rows_df, default_status="Watching", default_priority="Medium"):
    """Add/update scanner rows in the Pro Trader Dashboard watchlist."""
    ensure_pro_dashboard_df()
    if rows_df is None or rows_df.empty or "Ticker" not in rows_df.columns:
        return 0, 0

    dash = st.session_state["pro_trader_dashboard_df"].copy()
    existing = {str(t).upper().strip(): i for i, t in dash["Ticker"].fillna("").items() if str(t).strip()}
    added = 0
    updated = 0
    now_label = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

    for _, row in rows_df.iterrows():
        ticker = normalize_symbol_for_yahoo(str(row.get("Ticker", "")).upper().strip())
        if not ticker:
            continue
        new_data = {
            "Ticker": ticker,
            "Setup Type": infer_setup_type_from_scanner_row(row),
            "Thesis": f"Added from scanner. Playbook: {row.get('Playbook Signal', 'N/A')}. Rating: {row.get('Rating', 'N/A')}.",
            "Status": default_status,
            "Priority": default_priority,
            "Entry Price": np.nan,
            "Current Price": row.get("Current Price", np.nan),
            "Target Price": np.nan,
            "Stop Loss": np.nan,
            "Shares": 0,
            "GTMD Score": row.get("Score", np.nan),
            "Explosive Move %": row.get("Explosive Move Probability %", np.nan),
            "Squeeze Score": row.get("Squeeze Score", np.nan),
            "RVOL Alert": row.get("RVOL Alert", ""),
            "RVOL Alert Reason": row.get("RVOL Alert Reason", ""),
            "Relative Volume": row.get("Relative Volume", np.nan),
            "Earnings Risk": row.get("Earnings Risk", ""),
            "Next Earnings Date": row.get("Next Earnings Date", ""),
            "Last Checked": now_label,
            "Notes": row.get("AI Score Explanation", ""),
        }
        if ticker in existing:
            idx = existing[ticker]
            preserve_cols = ["Entry Price", "Target Price", "Stop Loss", "Shares", "Status", "Priority", "Thesis", "Notes"]
            for col, val in new_data.items():
                if col not in preserve_cols:
                    dash.at[idx, col] = val
            updated += 1
        else:
            dash = pd.concat([dash, pd.DataFrame([new_data])], ignore_index=True)
            existing[ticker] = len(dash) - 1
            added += 1

    for col in PRO_DASHBOARD_COLUMNS:
        if col not in dash.columns:
            dash[col] = np.nan if col in PRO_NUMERIC_COLUMNS else ""
    st.session_state["pro_trader_dashboard_df"] = clean_pro_dashboard_df(dash)
    save_pro_watchlist_to_disk(st.session_state["pro_trader_dashboard_df"])
    st.session_state["pro_dashboard_needs_refresh"] = True
    return added, updated



def render_top_rvol_alerts(df, title="Top RVOL Alerts"):
    """Show the most actionable RVOL spike alerts from any result dataframe."""
    if df is None or df.empty or "RVOL Alert" not in df.columns:
        return
    actionable = df[df["RVOL Alert"].astype(str).str.contains("🔥|👀", regex=True, na=False)].copy()
    if actionable.empty:
        return
    if "Relative Volume" in actionable.columns:
        sort_cols = ["Relative Volume"] + (["Score"] if "Score" in actionable.columns else [])
        actionable = actionable.sort_values(sort_cols, ascending=False, na_position="last")
    st.markdown(f"""<div class="gtmd-section-card"><div class="gtmd-card-title">{title}</div><div class="gtmd-card-subtitle">Early breakout / volume-spike candidates from the current results.</div></div>""", unsafe_allow_html=True)
    cols = [c for c in ["Ticker", "RVOL Alert", "RVOL Alert Reason", "Relative Volume", "Breakout Proximity %", "RSI", "5D Price Change %", "Score", "GTMD Score", "Explosive Move Probability %", "Explosive Move %"] if c in actionable.columns]
    display = actionable[cols].head(15).copy()
    for col in display.columns:
        display[col] = display[col].apply(display_value)
    st.dataframe(display, use_container_width=True)

def pro_dashboard_add_callback(rows_df, status_choice="Watching", priority_choice="Medium", action_label="selected"):
    """Streamlit-safe callback: stores scanner rows into Pro Dashboard during button callback before rerun."""
    added, updated = add_scanner_rows_to_pro_watchlist(rows_df, status_choice, priority_choice)
    st.session_state["pro_dashboard_last_add_message"] = f"Added {added} new ticker(s), updated {updated} existing ticker(s) from {action_label}."

def render_add_to_watchlist_controls(df, key_prefix="scanner"):
    """Reusable UI to push scanner/live results into the Pro Trader Dashboard."""
    if df is None or df.empty or "Ticker" not in df.columns:
        return
    st.markdown("""<div class="gtmd-section-card"><div class="gtmd-card-title">Send Scanner Picks to Pro Dashboard</div><div class="gtmd-card-subtitle">Select tickers from these results and save them into your Pro Trader Dashboard watchlist.</div></div>""", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2, 1, 1])
    tickers_available = df["Ticker"].dropna().astype(str).str.upper().drop_duplicates().tolist()
    default_selected = tickers_available[:5]
    selected = c1.multiselect("Choose tickers to add/update", tickers_available, default=default_selected, key=f"{key_prefix}_add_watchlist_select")
    status_choice = c2.selectbox("Status", ["Watching", "Ready", "In Position", "Avoid"], index=0, key=f"{key_prefix}_add_status")
    priority_choice = c3.selectbox("Priority", ["High", "Medium", "Low"], index=1, key=f"{key_prefix}_add_priority")
    add_col, top_col = st.columns(2)
    selected_rows = df[df["Ticker"].astype(str).str.upper().isin(selected)].copy()
    top10_rows = df.head(10).copy()

    add_col.button(
        "Add selected to Pro Dashboard",
        key=f"{key_prefix}_add_selected",
        on_click=pro_dashboard_add_callback,
        args=(selected_rows, status_choice, priority_choice, "selected scanner rows"),
    )
    top_col.button(
        "Add top 10 to Pro Dashboard",
        key=f"{key_prefix}_add_top10",
        on_click=pro_dashboard_add_callback,
        args=(top10_rows, status_choice, priority_choice, "top 10 scanner rows"),
    )

    message = st.session_state.pop("pro_dashboard_last_add_message", None)
    if message:
        st.success(message + " Open Potential Stocks Spreadsheet to view them.")



if mode == "Position Exit Calculator":
    with st.sidebar:
        st.header("Exit Calculator")
        calc_symbol = st.text_input("Ticker", value="META", key="exit_calc_ticker").upper().strip()
        calc_entry = st.number_input("Entry price", min_value=0.0, value=100.0, step=0.01, key="exit_calc_entry")
        calc_shares = st.number_input("Shares / contracts", min_value=0.0, value=100.0, step=1.0, key="exit_calc_shares")

        st.subheader("Adjustable Exit Rules")
        profit_pct = st.slider("Profit exit %", 1.0, 50.0, 5.0, step=0.5, key="exit_calc_profit_pct")
        partial_exit_pct = st.slider("Partial exit % at profit target", 0.0, 100.0, 50.0, step=5.0, key="exit_calc_partial_exit_pct")
        double_down_pct = st.slider("Double-down trigger drop %", 1.0, 50.0, 5.0, step=0.5, key="exit_calc_double_pct")
        hard_exit_pct = st.slider("Complete exit / stop-loss drop %", 1.0, 80.0, 10.0, step=0.5, key="exit_calc_stop_pct")
        double_down_shares = st.number_input("Extra shares if double-down", min_value=0.0, value=calc_shares, step=1.0, key="exit_calc_extra_shares")
        run_exit_calc = st.button("Calculate Exit Plan", key="exit_calc_run")

    st.markdown("""
    <div class="gtmd-section-card">
        <div class="gtmd-card-title">Position Exit Calculator</div>
        <div class="gtmd-card-subtitle">
            Enter your ticker and entry price, then adjust profit target, double-down trigger, and full-exit level.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if run_exit_calc and calc_symbol and calc_entry > 0:
        current_price = np.nan
        data_status = "Manual plan only"
        try:
            meta, hist, chart_error = get_yahoo_chart_history(normalize_symbol_for_yahoo(calc_symbol))
            if hist is not None and not hist.empty and "close" in hist.columns:
                current_price = safe_float(hist["close"].dropna().iloc[-1])
                data_status = "Current price loaded from Yahoo chart history"
            elif chart_error:
                data_status = chart_error
        except Exception as e:
            data_status = f"Current price unavailable: {e}"

        profit_exit_price = calc_entry * (1 + profit_pct / 100)
        double_down_price = calc_entry * (1 - double_down_pct / 100)
        hard_exit_price = calc_entry * (1 - hard_exit_pct / 100)

        original_cost = calc_entry * calc_shares
        double_down_cost = double_down_price * double_down_shares
        total_shares_after_dd = calc_shares + double_down_shares
        avg_cost_after_dd = ((original_cost + double_down_cost) / total_shares_after_dd) if total_shares_after_dd > 0 else np.nan
        profit_target_after_dd = avg_cost_after_dd * (1 + profit_pct / 100) if is_num(avg_cost_after_dd) else np.nan

        current_pl_pct = ((current_price / calc_entry) - 1) * 100 if is_num(current_price) and calc_entry > 0 else np.nan
        current_pl_dollars = (current_price - calc_entry) * calc_shares if is_num(current_price) else np.nan
        risk_dollars = (calc_entry - hard_exit_price) * calc_shares
        reward_dollars = (profit_exit_price - calc_entry) * calc_shares
        rr = reward_dollars / risk_dollars if risk_dollars > 0 else np.nan

        partial_exit_shares = calc_shares * (partial_exit_pct / 100)
        remaining_shares = max(calc_shares - partial_exit_shares, 0)
        locked_profit = (profit_exit_price - calc_entry) * partial_exit_shares
        remaining_position_value_at_target = remaining_shares * profit_exit_price
        remaining_original_cost = remaining_shares * calc_entry

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Ticker", calc_symbol)
        m2.metric("Entry", f"${calc_entry:,.2f}")
        m3.metric("Current", f"${current_price:,.2f}" if is_num(current_price) else "N/A")
        m4.metric("Current P/L", f"{current_pl_pct:,.2f}%" if is_num(current_pl_pct) else "N/A")
        m5.metric("Risk / Reward", f"{rr:.2f}R" if is_num(rr) else "N/A")
        st.caption(data_status)

        st.markdown("""
        <div class="gtmd-section-card">
            <div class="gtmd-card-title">Main Exit Levels</div>
            <div class="gtmd-card-subtitle">Auto-calculated from your entry price and slider percentages.</div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric(f"Profit Exit +{profit_pct:.1f}%", f"${profit_exit_price:,.2f}", f"+${reward_dollars:,.0f}" if calc_shares > 0 else None)
        c2.metric(f"Double Down -{double_down_pct:.1f}%", f"${double_down_price:,.2f}")
        c3.metric(f"Complete Exit -{hard_exit_pct:.1f}%", f"${hard_exit_price:,.2f}", f"-${risk_dollars:,.0f}" if calc_shares > 0 else None)

        plan_rows = [
            {"Action": "Take profit / exit", "Trigger %": profit_pct, "Trigger Price": profit_exit_price, "Estimated $ P/L": reward_dollars},
            {"Action": "Double down / add", "Trigger %": -double_down_pct, "Trigger Price": double_down_price, "Estimated $ P/L": (double_down_price - calc_entry) * calc_shares},
            {"Action": "Complete exit / stop", "Trigger %": -hard_exit_pct, "Trigger Price": hard_exit_price, "Estimated $ P/L": -risk_dollars},
        ]
        plan_df = pd.DataFrame(plan_rows)
        st.dataframe(
            plan_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Trigger %": st.column_config.NumberColumn("Trigger %", format="%.1f%%"),
                "Trigger Price": st.column_config.NumberColumn("Trigger Price", format="$%.2f"),
                "Estimated $ P/L": st.column_config.NumberColumn("Estimated $ P/L", format="$%.2f"),
            },
        )

        if partial_exit_pct > 0:
            st.markdown("""
            <div class="gtmd-section-card">
                <div class="gtmd-card-title">Partial Exit Plan</div>
                <div class="gtmd-card-subtitle">Shows what happens if you sell part of the position at the profit target and hold the rest.</div>
            </div>
            """, unsafe_allow_html=True)

            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Shares Sold at Target", f"{partial_exit_shares:,.0f}")
            p2.metric("Locked-In Profit", f"${locked_profit:,.2f}")
            p3.metric("Shares Still Held", f"{remaining_shares:,.0f}")
            p4.metric("Remaining Value at Target", f"${remaining_position_value_at_target:,.2f}")
            st.caption(f"Remaining original cost basis: ${remaining_original_cost:,.2f}. This ignores commissions, taxes, and slippage.")

        st.markdown("""
        <div class="gtmd-section-card">
            <div class="gtmd-card-title">If You Double Down</div>
            <div class="gtmd-card-subtitle">Shows the new average cost and new target if you add shares at the double-down level.</div>
        </div>
        """, unsafe_allow_html=True)

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("New Total Shares", f"{total_shares_after_dd:,.0f}")
        d2.metric("New Average Cost", f"${avg_cost_after_dd:,.2f}" if is_num(avg_cost_after_dd) else "N/A")
        d3.metric("Extra Add Cost", f"${double_down_cost:,.2f}")
        d4.metric(f"New +{profit_pct:.1f}% Target", f"${profit_target_after_dd:,.2f}" if is_num(profit_target_after_dd) else "N/A")

        st.caption("Educational calculator only. It does not decide whether averaging down is appropriate; it just calculates levels from your rules. Calculations ignore commissions, taxes, and slippage.")

    else:
        st.info("Enter ticker, entry price, and click Calculate Exit Plan.")


elif mode == "Single Stock Analyzer":
    with st.sidebar:
        st.header("Input")
        symbol = st.text_input("Ticker", value="META").upper().strip()

        st.subheader("Position / Exit Inputs")
        entry_price = st.number_input(
            "Entry price (optional, use 0 if not in position)",
            min_value=0.0,
            value=0.0,
            step=0.01,
        )

        run = st.button("Analyze")

    if run and symbol:
        with st.spinner(f"Pulling GTMD market data for {symbol}..."):
            values, total_score, detail_df, prices, debug = build_metrics(symbol)

        alert, action, reasons, gain_loss_pct = generate_trade_alert(values, entry_price)

        # Build a one-row dataframe so the Single Stock Analyzer can show the same
        # +4% Quick Trade score used in Quick Trade Mode. This does not alter the
        # main GTMD Score, Rating, Squeeze Score, Explosive Probability, or Playbook.
        single_row = pd.DataFrame([{**values, "Score": total_score, "Rating": rating(total_score), "Alert": alert}])
        single_row = add_quick_trade_columns(single_row, QUICK_TRADE_DEFAULT_RANGES)
        qt = single_row.iloc[0].to_dict() if not single_row.empty else {}

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Ticker", symbol)
        col2.metric("Score / 100", f"{total_score:.0f}")
        col3.metric("Rating", rating(total_score))
        col4.metric("Alert", alert)
        col5.metric("Squeeze", f"{values.get('Squeeze Score', 0)}/10")

        st.markdown("""<div class="gtmd-section-card"><div class="gtmd-card-title">+4% Quick Trade Score</div><div class="gtmd-card-subtitle">Separate short-term trade-fit score using RVOL, timing, explosive probability, technical confirmation, 5D move, breakout proximity, squeeze, GTMD safety floor, and fundamental safety filter.</div></div>""", unsafe_allow_html=True)
        q1, q2, q3, q4, q5 = st.columns(5)
        q1.metric("Quick Trade Fit", f"{safe_float(qt.get('Quick Trade Fit %'), 0):.1f}%")
        q2.metric("Quick Trade Label", qt.get("Quick Trade Label", "N/A"))
        q3.metric("Priority 1", qt.get("Priority 1 Matches", "N/A"))
        q4.metric("Priority 2", qt.get("Priority 2 Matches", "N/A"))
        q5.metric("Priority 3", qt.get("Priority 3 Matches", "N/A"))
        st.caption("Missing / outside ideal quick-trade criteria: " + str(qt.get("Quick Trade Missing", "N/A")))

        st.markdown("""<div class="gtmd-section-card"><div class="gtmd-card-title">Pro Signal Layers</div><div class="gtmd-card-subtitle">Playbook setup, explosive-move probability, earnings timing, and GTMD explanation.</div></div>""", unsafe_allow_html=True)
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Playbook", values.get("Playbook Signal", "N/A"))
        p2.metric("Confidence", values.get("Playbook Confidence", "N/A"))
        p3.metric("Explosive Move", f"{values.get('Explosive Move Probability %', 0)}%")
        p4.metric("RVOL Alert", values.get("RVOL Alert", "None"))
        st.caption(values.get("RVOL Alert Reason", ""))

        st.markdown("#### Score Layer Breakdown")
        b1, b2, b3 = st.columns(3)
        b1.metric("Fundamental Score", display_value(values.get("Fundamental Score")))
        b2.metric("Technical Score", display_value(values.get("Technical Score")))
        b3.metric("Timing Score", display_value(values.get("Timing Score")))
        st.caption(values.get("Score Breakdown", ""))

        st.info(values.get("AI Score Explanation", "No explanation available."))
        with st.expander("Explosive move reasons"):
            st.write(values.get("Explosive Move Reasons", "N/A"))
            cautions = values.get("Explosive Move Cautions", "None")
            if cautions and cautions != "None":
                st.warning(f"Cautions: {cautions}")
        with st.expander("Earnings details"):
            st.write(f"Next earnings date: {values.get('Next Earnings Date', 'N/A')}")
            st.write(f"Days until earnings: {display_value(values.get('Days Until Earnings'))}")
            st.write(values.get("Earnings Data Status", "N/A"))

        st.markdown("""<div class="gtmd-section-card"><div class="gtmd-card-title">Auto Trade Alert</div><div class="gtmd-card-subtitle">Signal summary based on score, momentum, risk, and squeeze setup.</div></div>""", unsafe_allow_html=True)
        if "BUY" in alert:
            st.success(f"{alert} — {action}")
        elif alert == "TRIM":
            st.warning(f"TRIM — {action}")
        elif alert == "SELL":
            st.error(f"SELL — {action}")
        else:
            st.info(f"WATCH — {action}")

        if not np.isnan(gain_loss_pct):
            st.metric("Gain/Loss From Entry", f"{gain_loss_pct:.2f}%")

        with st.expander("Why this alert fired"):
            for reason in reasons:
                st.write("-", reason)

        squeeze_status = values.get("Squeeze Setup")
        squeeze_label = values.get("Squeeze Label", "N/A")
        if squeeze_status == "YES":
            st.warning(f"🔥 {squeeze_label}: smarter squeeze model detected fuel + compression + breakout readiness.")
        elif squeeze_status == "WATCH":
            st.info(f"👀 {squeeze_label}: squeeze is developing, but not fully confirmed yet.")
        else:
            st.info("No clean squeeze setup detected.")

        with st.expander("Smart squeeze details"):
            st.write("**Reasons:**", values.get("Squeeze Reasons", "N/A"))
            st.write("**Cautions:**", values.get("Squeeze Cautions", "None"))

        if not is_num(safe_float(values["Current Price"])):
            st.error("No price data loaded. Try AAPL, META, MSFT, or SCHW. If still blank, Streamlit may be blocking Yahoo calls.")
            with st.expander("Debug data source status"):
                for item in debug:
                    st.write("-", item)
        else:
            st.success("Yahoo price and technical indicators loaded.")

        st.markdown("""<div class="gtmd-section-card"><div class="gtmd-card-title">Key Metrics</div><div class="gtmd-card-subtitle">Full metric readout for the selected ticker.</div></div>""", unsafe_allow_html=True)
        metric_df = pd.DataFrame([values]).T.reset_index()
        metric_df.columns = ["Metric", "Value"]
        metric_df["Value"] = metric_df["Value"].apply(display_value)
        st.dataframe(metric_df, use_container_width=True)

        st.button(
            "Add this ticker to Pro Dashboard",
            key=f"single_add_{symbol}",
            on_click=pro_dashboard_add_callback,
            args=(single_row, "Watching", "Medium", "single stock analyzer"),
        )
        single_msg = st.session_state.pop("pro_dashboard_last_add_message", None)
        if single_msg:
            st.success(single_msg + " Open Potential Stocks Spreadsheet to view it.")

        with st.expander("Debug data source status"):
            for item in debug:
                st.write("-", item)
            st.caption("If Yahoo is blocked by Streamlit Cloud, Finviz and financial statements are used as backups where possible.")

        st.markdown("""<div class="gtmd-section-card"><div class="gtmd-card-title">Score Breakdown</div><div class="gtmd-card-subtitle">Point-by-point explanation of the GTMD score.</div></div>""", unsafe_allow_html=True)
        detail_display_df = detail_df.copy()
        detail_display_df["Value"] = detail_display_df["Value"].apply(display_value)
        st.dataframe(detail_display_df, use_container_width=True)

        st.markdown("""<div class="gtmd-section-card"><div class="gtmd-card-title">TradingView 1H Candlestick Chart</div><div class="gtmd-card-subtitle">Live TradingView-style dark candlestick chart. No Plotly required.</div></div>""", unsafe_allow_html=True)
        render_tradingview_chart(symbol)

        st.markdown("""<div class="gtmd-section-card"><div class="gtmd-card-title">Decision Logic</div><div class="gtmd-card-subtitle">Final interpretation of the scanner score.</div></div>""", unsafe_allow_html=True)
        if total_score >= 85:
            st.success("Strong setup. Fundamentals, valuation, technicals, and positioning are aligned.")
        elif total_score >= 70:
            st.info("Good setup. Consider waiting for clean entry confirmation if price is extended.")
        elif total_score >= 55:
            st.warning("Watchlist only. Some factors are positive, but the full setup is not strong yet.")
        else:
            st.error("Weak setup. Avoid unless there is a separate catalyst or reversal confirmation.")
    else:
        st.info("Enter a ticker and click Analyze.")


elif mode == "Stock Scanner":
    with st.sidebar:
        st.header("Scanner Input")
        tickers_text = st.text_area("Tickers separated by commas", value="AAPL, META, MSFT, NVDA, SCHW, PLTR, TSLA, AMD, AMZN, GOOGL", height=150)
        min_score = st.slider("Minimum score to display", 0, 100, 55)
        min_avg_volume_manual = st.number_input(
            "Minimum 20D average volume",
            min_value=0,
            value=0,
            step=100000,
            help="Set to 0 to disable. Tickers with missing/unavailable 20D average volume are kept in the scan."
        )
        exclude_overbought = st.checkbox("Exclude RSI > 75", value=True)
        require_above_200 = st.checkbox("Require price above 200D MA", value=False)
        squeeze_only = st.checkbox("Show squeeze setups only", value=False)
        exclude_earnings_7d = st.checkbox("Exclude earnings within 7 days", value=False)
        scanner_workers = st.slider("Scanner speed / parallel workers", 1, 16, 8)
        st.caption("Scanner uses automatic score only. Manual confirmations were removed.")
        run_scanner = st.button("Run Scanner")

    if run_scanner:
        raw_tickers = [t.strip().upper() for t in tickers_text.replace("\n", ",").split(",")]
        tickers = sorted(list(dict.fromkeys([normalize_symbol_for_yahoo(t) for t in raw_tickers if t])))
        status = st.empty()
        status.write(f"Scanning {len(tickers)} tickers with {scanner_workers} parallel workers...")
        results, errors = scan_tickers_parallel(tickers, max_workers=scanner_workers)
        status.empty()

        if results:
            df = pd.DataFrame(results)
            df = df[df["Score"] >= min_score]

            # Manual scanner liquidity filter.
            # Important: missing/unavailable volume stays included instead of being filtered out.
            if min_avg_volume_manual > 0:
                volume_col = None
                for candidate_col in ["Average Volume 20D", "Avg Volume 20D", "20D Average Volume", "Average Volume"]:
                    if candidate_col in df.columns:
                        volume_col = candidate_col
                        break

                if volume_col:
                    volume_numeric = pd.to_numeric(df[volume_col], errors="coerce")
                    df = df[(volume_numeric.isna()) | (volume_numeric >= min_avg_volume_manual)]
                else:
                    st.warning("20D average volume data was not available, so the volume filter was skipped.")

            if exclude_overbought and "RSI" in df.columns:
                df = df[(df["RSI"].isna()) | (df["RSI"] <= 75)]
            if require_above_200 and "Distance from 200D MA %" in df.columns:
                df = df[(df["Distance from 200D MA %"].isna()) | (df["Distance from 200D MA %"] > 0)]
            if squeeze_only and "Squeeze Setup" in df.columns:
                df = df[df["Squeeze Setup"] == "YES"]
            if exclude_earnings_7d and "Days Until Earnings" in df.columns:
                df = df[(df["Days Until Earnings"].isna()) | (df["Days Until Earnings"] > 7) | (df["Days Until Earnings"] < 0)]

            if not df.empty:
                df = df.sort_values(["Score", "Explosive Move Probability %", "Squeeze Score"], ascending=False)
                save_scan_results_to_session(df, "scanner_results")
                st.success("Scanner results saved. You can now add selected tickers or the top 10 to Pro Dashboard.")
            else:
                st.warning("No stocks passed your filters.")
                st.session_state.pop("scanner_results", None)
        else:
            st.error("No scanner results loaded.")

        if errors:
            st.session_state["scanner_errors"] = errors
        else:
            st.session_state.pop("scanner_errors", None)

    saved_scanner_df = get_scan_results_from_session("scanner_results")
    if not saved_scanner_df.empty:
        st.markdown("""<div class="gtmd-section-card"><div class="gtmd-card-title">Scanner Results</div><div class="gtmd-card-subtitle">Persistent results. Add buttons stay visible after Streamlit reruns.</div></div>""", unsafe_allow_html=True)
        render_saved_results_with_add_controls(
            saved_scanner_df,
            "Saved Stock Scanner Results",
            "stock_scanner_saved",
            "scanner_results"
        )
        render_watchlist_tracker(saved_scanner_df, key_prefix="manual_scanner")

        st.markdown("""<div class="gtmd-section-card"><div class="gtmd-card-title">Top Setups</div><div class="gtmd-card-subtitle">Highest-ranked names from the current saved scan.</div></div>""", unsafe_allow_html=True)
        top = saved_scanner_df.head(5)
        if not top.empty:
            for _, row in top.iterrows():
                squeeze = f", Squeeze: {row.get('Squeeze Score', 0)}/10" if "Squeeze Score" in row else ""
                layer_text = f" | F/T/Timing: {display_value(row.get('Fundamental Score'))}/{display_value(row.get('Technical Score'))}/{display_value(row.get('Timing Score'))}"
                st.write(f"**{row['Ticker']}** — Score: **{row['Score']:.0f}**{layer_text}, Playbook: {row.get('Playbook Signal', 'N/A')}, Explosive: {row.get('Explosive Move Probability %', 'N/A')}%, Rating: {row['Rating']}, Alert: {row.get('Alert', 'N/A')}, RVOL: {row.get('RVOL Alert', 'N/A')}{squeeze}")
    elif not run_scanner:
        st.info("Enter a watchlist and click Run Scanner.")

    scanner_errors = st.session_state.get("scanner_errors", [])
    if scanner_errors:
        with st.expander("Errors / skipped tickers"):
            for err in scanner_errors[:250]:
                st.write("-", err)




elif mode == "Quick Trade Mode (+4% Target)":
    with st.sidebar:
        st.header("Quick Trade +4% Setup")
        qt_tickers_text = st.text_area(
            "Tickers separated by commas",
            value="AAPL, META, MSFT, NVDA, SCHW, PLTR, TSLA, AMD, AMZN, GOOGL",
            height=150,
            key="quick_trade_tickers"
        )
        qt_workers = st.slider("Scanner speed / parallel workers", 1, 16, 8, key="quick_trade_workers")
        use_saved_scan = st.checkbox("Use saved Stock Scanner results if available", value=True, key="quick_trade_use_saved")
        adjust_qt = st.checkbox("Adjust quick-trade ranges", value=False, key="quick_trade_adjust_ranges")

        qt_ranges = dict(QUICK_TRADE_DEFAULT_RANGES)
        if adjust_qt:
            st.subheader("Priority 1")
            qt_ranges["Relative Volume"] = st.slider("RVOL range", 0.0, 8.0, QUICK_TRADE_DEFAULT_RANGES["Relative Volume"], 0.1, key="qt_rvol")
            qt_ranges["Timing Score"] = st.slider("Timing Score range", 0.0, 100.0, QUICK_TRADE_DEFAULT_RANGES["Timing Score"], 1.0, key="qt_timing")
            qt_ranges["Explosive Move Probability %"] = st.slider("Explosive Probability range", 0.0, 100.0, QUICK_TRADE_DEFAULT_RANGES["Explosive Move Probability %"], 1.0, key="qt_explosive")

            st.subheader("Priority 2")
            qt_ranges["Technical Score"] = st.slider("Technical Score range", 0.0, 100.0, QUICK_TRADE_DEFAULT_RANGES["Technical Score"], 1.0, key="qt_technical")
            qt_ranges["5D Price Change %"] = st.slider("5D Price Change % range", -20.0, 30.0, QUICK_TRADE_DEFAULT_RANGES["5D Price Change %"], 0.5, key="qt_5d_change")
            qt_ranges["Breakout Proximity %"] = st.slider("Breakout Proximity % range", -20.0, 20.0, QUICK_TRADE_DEFAULT_RANGES["Breakout Proximity %"], 0.5, key="qt_breakout_proximity")
            qt_ranges["Squeeze Score"] = st.slider("Squeeze Score range", 0.0, 10.0, QUICK_TRADE_DEFAULT_RANGES["Squeeze Score"], 0.5, key="qt_squeeze")

            st.subheader("Priority 3")
            qt_ranges["Score"] = st.slider("GTMD Score range", 0.0, 100.0, QUICK_TRADE_DEFAULT_RANGES["Score"], 1.0, key="qt_gtmd")
            qt_ranges["Fundamental Score"] = st.slider("Fundamental Score range", 0.0, 100.0, QUICK_TRADE_DEFAULT_RANGES["Fundamental Score"], 1.0, key="qt_fundamental")

        run_quick_trade = st.button("Run Quick Trade Scan", key="quick_trade_run")

    st.markdown("""
    <div class="gtmd-section-card">
        <div class="gtmd-card-title">Quick Trade Mode: +4% Target</div>
        <div class="gtmd-card-subtitle">
            Preset scanner mode for short swing trades. It does not change GTMD Score, Squeeze Score, Explosive Probability, or Playbook logic — it adds a separate quick-trade fit label. RSI and 50D distance are intentionally excluded here to reduce double-counting because GTMD already includes them.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Quick Trade Criteria")
    render_quick_trade_criteria_table(qt_ranges)

    source_df = pd.DataFrame()
    if use_saved_scan:
        source_df = get_scan_results_from_session("scanner_results")
        if not source_df.empty:
            st.caption("Using saved Stock Scanner results. Click Run Quick Trade Scan to refresh from ticker input instead.")

    if run_quick_trade:
        raw_tickers = [t.strip().upper() for t in qt_tickers_text.replace("\n", ",").split(",")]
        tickers = sorted(list(dict.fromkeys([normalize_symbol_for_yahoo(t) for t in raw_tickers if t])))
        status = st.empty()
        status.write(f"Scanning {len(tickers)} tickers for Quick Trade Mode with {qt_workers} parallel workers...")
        results, errors = scan_tickers_parallel(tickers, max_workers=qt_workers)
        status.empty()
        if results:
            source_df = pd.DataFrame(results)
            st.session_state["quick_trade_errors"] = errors
        else:
            st.error("No quick-trade scanner results loaded.")
            source_df = pd.DataFrame()
            st.session_state["quick_trade_errors"] = errors

    if source_df is not None and not source_df.empty:
        qt_df = add_quick_trade_columns(source_df, qt_ranges)
        sort_cols = [c for c in ["Quick Trade Fit %", "Explosive Move Probability %", "Relative Volume", "Timing Score"] if c in qt_df.columns]
        if sort_cols:
            qt_df = qt_df.sort_values(sort_cols, ascending=False, na_position="last")
        save_scan_results_to_session(qt_df, "quick_trade_results")

    saved_qt_df = get_scan_results_from_session("quick_trade_results")
    if not saved_qt_df.empty:
        st.markdown("""<div class="gtmd-section-card"><div class="gtmd-card-title">Quick Trade Results</div><div class="gtmd-card-subtitle">Ranked by quick-trade fit for the +4% target setup.</div></div>""", unsafe_allow_html=True)
        preferred_cols = [
            "Ticker", "Quick Trade Label", "Quick Trade Fit %", "Priority 1 Matches", "Priority 2 Matches", "Priority 3 Matches",
            "Relative Volume", "Timing Score", "Explosive Move Probability %", "Technical Score", "5D Price Change %",
            "Breakout Proximity %", "Squeeze Score", "Score", "Fundamental Score", "Playbook Signal", "Rating", "Quick Trade Missing"
        ]
        show_cols = [c for c in preferred_cols if c in saved_qt_df.columns]
        display_qt = saved_qt_df[show_cols].copy()
        for col in display_qt.columns:
            display_qt[col] = display_qt[col].apply(display_value)
        st.dataframe(display_qt, use_container_width=True)

        render_add_to_watchlist_controls(saved_qt_df, key_prefix="quick_trade_mode")

        top_qt = saved_qt_df.head(5)
        if not top_qt.empty:
            st.markdown("#### Top Quick Trade Setups")
            for _, row in top_qt.iterrows():
                st.write(
                    f"**{row.get('Ticker', 'N/A')}** — {row.get('Quick Trade Label', 'N/A')} "
                    f"({row.get('Quick Trade Fit %', 'N/A')}% fit), "
                    f"P1: {row.get('Priority 1 Matches', 'N/A')}, "
                    f"RVOL: {display_value(row.get('Relative Volume'))}, "
                    f"Timing: {display_value(row.get('Timing Score'))}, "
                    f"Explosive: {display_value(row.get('Explosive Move Probability %'))}%"
                )

    qt_errors = st.session_state.get("quick_trade_errors", [])
    if qt_errors:
        with st.expander("Quick Trade errors / skipped tickers"):
            for err in qt_errors[:250]:
                st.write("-", err)


elif mode == "Potential Stocks Spreadsheet":
    ensure_pro_dashboard_df()
    st.markdown("""<div class="gtmd-section-card"><div class="gtmd-card-title">Pro Trader Dashboard</div><div class="gtmd-card-subtitle">Scanner picks now flow here automatically. Your watchlist auto-saves and auto-loads from a local CSV file.</div></div>""", unsafe_allow_html=True)
    st.caption(f"Auto-save file: {PRO_WATCHLIST_FILE}")

    dashboard_columns = PRO_DASHBOARD_COLUMNS

    numeric_dashboard_cols = ["Shares", "Entry Price", "Current Price", "Target Price", "Stop Loss", "GTMD Score", "Explosive Move %", "Squeeze Score", "Relative Volume"]


    with st.sidebar:
        st.header("Dashboard Controls")
        uploaded_watchlist = st.file_uploader("Import CSV watchlist", type=["csv"], key="pro_dashboard_import")
        refresh_scores = st.button("Refresh GTMD data for watchlist")
        clear_dashboard = st.button("Reset dashboard")

    if clear_dashboard:
        st.session_state["pro_trader_dashboard_df"] = empty_pro_dashboard_df()
        try:
            if os.path.exists(PRO_WATCHLIST_FILE):
                os.remove(PRO_WATCHLIST_FILE)
        except Exception as e:
            st.warning(f"Could not delete saved watchlist file: {e}")
        st.rerun()

    if uploaded_watchlist is not None:
        try:
            imported = pd.read_csv(uploaded_watchlist)
            for col in dashboard_columns:
                if col not in imported.columns:
                    imported[col] = np.nan if col in numeric_dashboard_cols else ""
            st.session_state["pro_trader_dashboard_df"] = clean_pro_dashboard_df(imported[dashboard_columns])
            save_pro_watchlist_to_disk(st.session_state["pro_trader_dashboard_df"])
            st.success("CSV imported and saved into dashboard.")
        except Exception as e:
            st.error(f"Could not import CSV: {e}")

    df_dash = st.session_state["pro_trader_dashboard_df"].copy()
    for col in dashboard_columns:
        if col not in df_dash.columns:
            df_dash[col] = np.nan if col in numeric_dashboard_cols else ""
    df_dash = clean_pro_dashboard_df(df_dash[dashboard_columns])

    auto_refresh_after_add = bool(st.session_state.pop("pro_dashboard_needs_refresh", False))
    if auto_refresh_after_add:
        st.info("New scanner picks were added, so the Pro Dashboard is recalculating them now.")

    if refresh_scores or auto_refresh_after_add:
        tickers = sorted(list(dict.fromkeys([normalize_symbol_for_yahoo(str(t).upper().strip()) for t in df_dash["Ticker"].dropna().tolist() if str(t).strip()])))
        if not tickers:
            st.warning("Add at least one ticker before refreshing GTMD data.")
        else:
            status = st.empty()
            status.write(f"Refreshing GTMD data for {len(tickers)} watchlist tickers...")
            results, errors = scan_tickers_parallel(tickers, max_workers=min(8, max(1, len(tickers))))
            status.empty()
            if results:
                scored = pd.DataFrame(results).set_index("Ticker")
                now_label = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
                for i, row in df_dash.iterrows():
                    ticker = normalize_symbol_for_yahoo(str(row.get("Ticker", "")).upper().strip())
                    if ticker in scored.index:
                        s = scored.loc[ticker]
                        df_dash.at[i, "Ticker"] = ticker
                        df_dash.at[i, "Current Price"] = s.get("Current Price", np.nan)
                        df_dash.at[i, "GTMD Score"] = s.get("Score", np.nan)
                        df_dash.at[i, "Explosive Move %"] = s.get("Explosive Move Probability %", np.nan)
                        df_dash.at[i, "Squeeze Score"] = s.get("Squeeze Score", np.nan)
                        df_dash.at[i, "Earnings Risk"] = s.get("Earnings Risk", "")
                        df_dash.at[i, "Next Earnings Date"] = s.get("Next Earnings Date", "")
                        df_dash.at[i, "Last Checked"] = now_label
                st.session_state["pro_trader_dashboard_df"] = clean_pro_dashboard_df(df_dash)
                save_pro_watchlist_to_disk(st.session_state["pro_trader_dashboard_df"])
                st.success("Dashboard refreshed and auto-saved with latest GTMD scanner data.")
            if errors:
                with st.expander("Refresh errors / skipped tickers"):
                    for err in errors[:250]:
                        st.write("-", err)

    st.info("Edit as many cells as you want, then click Save Dashboard Changes. This prevents Streamlit from rerunning after every single cell edit.")

    with st.form("pro_trader_dashboard_edit_form", clear_on_submit=False):
        edited = st.data_editor(
            df_dash,
            num_rows="dynamic",
            use_container_width=True,
            key="pro_trader_dashboard_editor",
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker", help="Stock ticker symbol"),
                "Setup Type": st.column_config.SelectboxColumn("Setup Type", options=["Squeeze", "Breakout", "Pullback", "Trend Continuation", "Earnings", "Oversold Bounce", "Long-Term", "Other"]),
                "Status": st.column_config.SelectboxColumn("Status", options=["Watching", "Ready", "In Position", "Trimmed", "Closed", "Avoid"]),
                "Priority": st.column_config.SelectboxColumn("Priority", options=["High", "Medium", "Low"]),
                "Entry Price": st.column_config.NumberColumn("Entry Price", format="$%.2f"),
                "Current Price": st.column_config.NumberColumn("Current Price", format="$%.2f"),
                "Target Price": st.column_config.NumberColumn("Target Price", format="$%.2f"),
                "Stop Loss": st.column_config.NumberColumn("Stop Loss", format="$%.2f"),
                "Shares": st.column_config.NumberColumn("Shares", step=1),
                "GTMD Score": st.column_config.NumberColumn("GTMD Score", format="%.0f"),
                "Explosive Move %": st.column_config.NumberColumn("Explosive Move %", format="%.0f%%"),
                "Squeeze Score": st.column_config.NumberColumn("Squeeze Score", format="%.0f/10"),
            }
        )
        save_dashboard_changes = st.form_submit_button("Save Dashboard Changes", use_container_width=True)

    if save_dashboard_changes:
        st.session_state["pro_trader_dashboard_df"] = clean_pro_dashboard_df(edited)
        save_pro_watchlist_to_disk(st.session_state["pro_trader_dashboard_df"])
        st.success("Dashboard changes saved.")
    else:
        edited = st.session_state["pro_trader_dashboard_df"].copy()

    calc = edited.copy()
    for col in numeric_dashboard_cols:
        calc[col] = pd.to_numeric(calc[col], errors="coerce")

    calc["P/L %"] = np.where((calc["Entry Price"] > 0) & (calc["Current Price"] > 0), ((calc["Current Price"] / calc["Entry Price"]) - 1) * 100, np.nan)
    calc["Position Value"] = np.where((calc["Shares"] > 0) & (calc["Current Price"] > 0), calc["Shares"] * calc["Current Price"], np.nan)
    calc["Risk/Reward"] = np.where((calc["Entry Price"] > 0) & (calc["Target Price"] > calc["Entry Price"]) & (calc["Stop Loss"] < calc["Entry Price"]), (calc["Target Price"] - calc["Entry Price"]) / (calc["Entry Price"] - calc["Stop Loss"]), np.nan)
    calc["Upside to Target %"] = np.where((calc["Current Price"] > 0) & (calc["Target Price"] > 0), ((calc["Target Price"] / calc["Current Price"]) - 1) * 100, np.nan)
    calc["Downside to Stop %"] = np.where((calc["Current Price"] > 0) & (calc["Stop Loss"] > 0), ((calc["Stop Loss"] / calc["Current Price"]) - 1) * 100, np.nan)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Tracked Stocks", len(calc.dropna(subset=["Ticker"])))
    m2.metric("Ready / In Position", int(calc["Status"].isin(["Ready", "In Position"]).sum()) if "Status" in calc.columns else 0)
    m3.metric("Avg GTMD Score", f"{calc['GTMD Score'].dropna().mean():.0f}" if calc["GTMD Score"].notna().any() else "N/A")
    m4.metric("Avg Explosive", f"{calc['Explosive Move %'].dropna().mean():.0f}%" if calc["Explosive Move %"].notna().any() else "N/A")
    m5.metric("Total Position Value", f"${calc['Position Value'].dropna().sum():,.0f}" if calc["Position Value"].notna().any() else "$0")

    st.markdown("""<div class="gtmd-section-card"><div class="gtmd-card-title">Calculated Pro Dashboard</div><div class="gtmd-card-subtitle">Auto-calculated P/L, risk/reward, upside, downside, and scanner data.</div></div>""", unsafe_allow_html=True)

    status_options = ["All"] + sorted([x for x in calc["Status"].dropna().unique().tolist() if str(x).strip()])
    f1, f2, f3 = st.columns([1, 1, 2])
    selected_status = f1.selectbox("Filter Status", status_options)
    min_dash_score = f2.slider("Minimum GTMD Score", 0, 100, 0, key="dash_min_score")
    search_text = f3.text_input("Search ticker/thesis/notes", "", key="dash_search")

    view = calc.copy()
    if selected_status != "All":
        view = view[view["Status"] == selected_status]
    if min_dash_score > 0:
        view = view[(view["GTMD Score"].isna()) | (view["GTMD Score"] >= min_dash_score)]
    if search_text.strip():
        stext = search_text.lower().strip()
        mask = view.apply(lambda row: stext in " ".join([str(v).lower() for v in row.values]), axis=1)
        view = view[mask]

    preferred_cols = ["Ticker", "Status", "Priority", "Setup Type", "Current Price", "Entry Price", "P/L %", "Target Price", "Stop Loss", "Risk/Reward", "Upside to Target %", "Downside to Stop %", "GTMD Score", "Explosive Move %", "Squeeze Score", "RVOL Alert", "RVOL Alert Reason", "Relative Volume", "Earnings Risk", "Next Earnings Date", "Thesis", "Notes"]
    view = view[[c for c in preferred_cols if c in view.columns]]
    if "GTMD Score" in view.columns:
        view = view.sort_values(["GTMD Score", "Explosive Move %"], ascending=False, na_position="last")
    st.dataframe(view.map(display_value) if hasattr(view, "map") else view.applymap(display_value), use_container_width=True)

elif mode == "Notes":
    NOTES_FILE = "gtmd_notes_saved.txt"

    st.markdown("""
    <div class="gtmd-section-card">
        <div class="gtmd-card-title">Notes</div>
        <div class="gtmd-card-subtitle">
            Clean paper for trade ideas, reminders, thesis notes, and scanner observations. Auto-saves locally.
        </div>
    </div>
    """, unsafe_allow_html=True)

    try:
        if "gtmd_notes_text" not in st.session_state:
            if os.path.exists(NOTES_FILE):
                with open(NOTES_FILE, "r", encoding="utf-8") as f:
                    st.session_state["gtmd_notes_text"] = f.read()
            else:
                st.session_state["gtmd_notes_text"] = ""
    except Exception:
        st.session_state["gtmd_notes_text"] = ""

    notes_text = st.text_area(
        "Write your notes here",
        value=st.session_state.get("gtmd_notes_text", ""),
        height=650,
        placeholder="Example: NVDA watching for RVOL spike near breakout... PLTR thesis... reminders..."
    )

    if notes_text != st.session_state.get("gtmd_notes_text", ""):
        st.session_state["gtmd_notes_text"] = notes_text
        try:
            with open(NOTES_FILE, "w", encoding="utf-8") as f:
                f.write(notes_text)
            st.success("Notes auto-saved.")
        except Exception as e:
            st.warning(f"Could not auto-save notes: {e}")

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download notes",
            data=notes_text.encode("utf-8"),
            file_name="gtmd_notes.txt",
            mime="text/plain"
        )
    with c2:
        if st.button("Clear notes"):
            st.session_state["gtmd_notes_text"] = ""
            try:
                with open(NOTES_FILE, "w", encoding="utf-8") as f:
                    f.write("")
            except Exception:
                pass
            st.rerun()



elif mode == "Live Universe Scanner":
    with st.sidebar:
        st.header("Live Universe")
        universe_limit = st.slider("Universe size to fetch", 250, 5000, 1500, step=250)
        min_price_live = st.number_input("Minimum price", min_value=0.0, value=50.0, step=1.0)
        min_volume_live = st.number_input("Minimum 20D average volume", min_value=0, value=2000000, step=100000)
        min_volatility_live = st.slider("Minimum 20D volatility %", 0, 150, 20)
        min_rvol_live = st.slider("Minimum relative volume (RVOL)", 0.0, 5.0, 1.0, step=0.1)
        pre_candidates = st.slider("Pre-ranked candidates to fully score", 10, 300, 75, step=5)
        if pre_candidates > 100:
            st.warning("Heavy scan: scoring more than 100 candidates can be slower and may trigger Yahoo/yfinance throttling on Streamlit Cloud.")
        else:
            st.caption("Tip: 75–100 full-score candidates is usually the best balance of speed and quality.")
        min_score_live = st.slider("Minimum final GTMD score", 0, 100, 55)
        live_workers = st.slider("Speed / parallel workers", 4, 24, 12)
        exclude_overbought_live = st.checkbox("Exclude RSI > 75", value=True, key="live_exclude_rsi")
        require_above_200_live = st.checkbox("Require price above 200D MA", value=False, key="live_above_200")
        squeeze_only_live = st.checkbox("Show squeeze setups only", value=False, key="live_squeeze")
        exclude_earnings_7d_live = st.checkbox("Exclude earnings within 7 days", value=False, key="live_exclude_earnings")
        run_live = st.button("Build Live Universe")

    if run_live:
        st.markdown("""<div class="gtmd-section-card"><div class="gtmd-card-title">Live Universe Engine</div><div class="gtmd-card-subtitle">Auto-fetches a broad daily U.S. stock universe, filters by liquidity/volatility/RVOL, pre-ranks strongest candidates, then fully scores the best names.</div></div>""", unsafe_allow_html=True)
        status = st.empty()
        progress = st.progress(0)
        status.write("Fetching daily stock universe...")
        universe, source_note = get_live_stock_universe(universe_limit)
        progress.progress(0.15)
        st.caption(f"Universe source: {source_note}")
        st.metric("Fetched Symbols", len(universe))

        status.write("Pre-ranking by liquidity, volatility, RVOL, and recent movement...")
        prerank_df, prerank_errors = build_preranked_universe(
            universe,
            min_price_live,
            min_volume_live,
            min_volatility_live,
            min_rvol_live,
            max_workers=live_workers
        )
        progress.progress(0.55)
        prerank_error_rate = (prerank_errors / max(len(universe), 1)) * 100

        if prerank_error_rate > 20:
            st.warning(f"High pre-rank data failure rate: {prerank_error_rate:.1f}% of the universe failed to load. Try lowering universe size, workers, or candidates.")

        if prerank_df.empty:
            status.empty()
            st.error("No symbols passed the live universe filters. Try lowering minimum price, volume, volatility, or RVOL.")
            st.session_state.pop("live_results", None)
        else:
            st.markdown("""<div class="gtmd-section-card"><div class="gtmd-card-title">Pre-Ranked Candidates</div><div class="gtmd-card-subtitle">Fast screen before the full GTMD fundamental + technical score.</div></div>""", unsafe_allow_html=True)
            pre_display = prerank_df.head(pre_candidates).copy()
            for col in pre_display.columns:
                pre_display[col] = pre_display[col].apply(display_value)
            st.dataframe(pre_display, use_container_width=True)

            selected = prerank_df.head(pre_candidates)["Ticker"].tolist()
            status.write(f"Fully scoring top {len(selected)} pre-ranked stocks...")
            scored_rows, score_errors = scan_tickers_parallel(selected, max_workers=min(live_workers, 16))
            progress.progress(1.0)
            status.empty()

            if scored_rows:
                df = pd.DataFrame(scored_rows)
                df = df[df["Score"] >= min_score_live]
                if exclude_overbought_live and "RSI" in df.columns:
                    df = df[(df["RSI"].isna()) | (df["RSI"] <= 75)]
                if require_above_200_live and "Distance from 200D MA %" in df.columns:
                    df = df[(df["Distance from 200D MA %"].isna()) | (df["Distance from 200D MA %"] > 0)]
                if squeeze_only_live and "Squeeze Setup" in df.columns:
                    df = df[df["Squeeze Setup"] == "YES"]
                if exclude_earnings_7d_live and "Days Until Earnings" in df.columns:
                    df = df[(df["Days Until Earnings"].isna()) | (df["Days Until Earnings"] > 7) | (df["Days Until Earnings"] < 0)]

                if not df.empty:
                    df = df.sort_values(["Score", "Explosive Move Probability %", "Squeeze Score"], ascending=False)
                    save_scan_results_to_session(df, "live_results")
                    st.success("Live Universe results saved. You can now add selected tickers or the top 10 to Pro Dashboard.")
                else:
                    st.warning("No stocks passed the final live filters.")
                    st.session_state.pop("live_results", None)

                st.session_state["live_scan_summary"] = {
                    "Pre-Ranked Passed": len(prerank_df),
                    "Fully Scored": len(scored_rows),
                    "Final Matches": len(df),
                    "Pre-Rank Failures": prerank_errors,
                    "Full-Score Failures": len(score_errors),
                }

                if score_errors:
                    st.session_state["live_score_errors"] = score_errors
                else:
                    st.session_state.pop("live_score_errors", None)
            else:
                st.error("Live universe pre-rank worked, but no full GTMD scores loaded.")
                st.session_state.pop("live_results", None)

    saved_live_df = get_scan_results_from_session("live_results")
    if not saved_live_df.empty:
        summary = st.session_state.get("live_scan_summary", {})
        if summary:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Pre-Ranked Passed", summary.get("Pre-Ranked Passed", "N/A"))
            c2.metric("Fully Scored", summary.get("Fully Scored", "N/A"))
            c3.metric("Final Matches", summary.get("Final Matches", "N/A"))
            c4.metric("Pre-Rank Failures", summary.get("Pre-Rank Failures", "N/A"))
            c5.metric("Full-Score Failures", summary.get("Full-Score Failures", "N/A"))

        st.markdown("""<div class="gtmd-section-card"><div class="gtmd-card-title">Live Universe Results</div><div class="gtmd-card-subtitle">Persistent results. Add buttons stay visible after Streamlit reruns.</div></div>""", unsafe_allow_html=True)
        render_saved_results_with_add_controls(
            saved_live_df,
            "Saved Live Universe Scanner Results",
            "live_universe_saved",
            "live_results"
        )
        render_watchlist_tracker(saved_live_df, key_prefix="live_universe")

        st.markdown("""<div class="gtmd-section-card"><div class="gtmd-card-title">Top Live Setups</div><div class="gtmd-card-subtitle">Highest-ranked names from today’s dynamic universe.</div></div>""", unsafe_allow_html=True)
        top = saved_live_df.head(10)
        if not top.empty:
            for _, row in top.iterrows():
                squeeze = f", Squeeze: {row.get('Squeeze Score', 0)}/10" if "Squeeze Score" in row else ""
                st.write(f"**{row['Ticker']}** — Score: **{row['Score']:.0f}**, Playbook: {row.get('Playbook Signal', 'N/A')}, Explosive: {row.get('Explosive Move Probability %', 'N/A')}%, Rating: {row['Rating']}, Alert: {row.get('Alert', 'N/A')}, RVOL: {row.get('RVOL Alert', 'N/A')}{squeeze}")

        live_score_errors = st.session_state.get("live_score_errors", [])
        if live_score_errors:
            with st.expander("Full-score errors / skipped tickers"):
                for err in live_score_errors[:250]:
                    st.write("-", err)
    elif not run_live:
        st.info("Click Build Live Universe to auto-fetch stocks, pre-rank them, and fully score the strongest candidates.")
