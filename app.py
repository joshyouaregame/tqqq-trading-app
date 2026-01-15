"""
DISCLAIMER:
This app is for EDUCATIONAL PURPOSES ONLY. It does NOT provide financial advice.
Trading leveraged ETFs like TQQQ is risky and can result in significant losses.
This tool SUPPORTS decisions; it does not automate trading.

MOBILE-FIRST STREAMLIT APP – TIER 1 (REAL TRADING)

Tier 1 Features:
- Email-only daily alerts (end-of-day)
- ATR-based stop-loss, take-profit, and TRAILING STOP
- Trend filter (QQQ 200 SMA)
- Volatility control (ATR percentile)
- Position sizing based on account risk
- Explicit NO-TRADE (CASH) alerts

Requirements:
pip install yfinance pandas numpy matplotlib streamlit

Run:
streamlit run app.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import smtplib
from email.mime.text import MIMEText

st.set_page_config(page_title="TQQQ Tier-1 Trading App", layout="wide")

st.title("📱 TQQQ Tier-1 Trading System")
st.warning("You are responsible for all trades. Leveraged ETFs are high risk.")

# =========================
# EMAIL + PUSH REMINDER ALERTS
# =========================

# PUSH NOTIFICATION (PUSHOVER)
# Sends a weekday reminder at 6:15am AEST to check email

import datetime
import pytz
import requests

PUSHOVER_USER_KEY = st.secrets.get("PUSHOVER_USER_KEY", "")
PUSHOVER_APP_TOKEN = st.secrets.get("PUSHOVER_APP_TOKEN", "")


def send_push(message):
    if not PUSHOVER_USER_KEY or not PUSHOVER_APP_TOKEN:
        return
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": PUSHOVER_APP_TOKEN,
            "user": PUSHOVER_USER_KEY,
            "message": message,
            "priority": 0,
        },
        timeout=10,
    )

# Time check (AEST)
aest = pytz.timezone("Australia/Sydney")
now = datetime.datetime.now(aest)

is_weekday = now.weekday() < 5
is_615 = now.hour == 6 and now.minute == 15

if is_weekday and is_615:
    send_push("📈 TQQQ reminder: Read today’s trading email before market open")

# =========================
# EMAIL ALERT SETUP (ONLY)
# =========================
st.sidebar.header("📧 Email Alerts")
email_to = st.sidebar.text_input("Recipient Email")
email_from = st.sidebar.text_input("Sender Gmail")
email_password = st.sidebar.text_input("Gmail App Password", type="password")

# =========================
# ACCOUNT & RISK SETTINGS
# =========================
st.sidebar.header("💰 Account Risk")
account_size = st.sidebar.number_input("Account Size ($)", value=50000)
risk_per_trade = st.sidebar.slider("Risk Per Trade (%)", 0.25, 2.0, 1.0) / 100

# =========================
# STRATEGY PARAMETERS
# =========================
st.sidebar.header("📊 Strategy Parameters")
short_ma = st.sidebar.slider("Short MA", 5, 50, 20)
long_ma = st.sidebar.slider("Long MA", 50, 200, 100)

rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)
rsi_buy = st.sidebar.slider("RSI Buy", 10, 40, 30)
rsi_sell = st.sidebar.slider("RSI Sell", 60, 90, 70)

atr_period = st.sidebar.slider("ATR Period", 5, 30, 14)
stop_atr = st.sidebar.slider("Initial Stop (ATR)", 0.5, 5.0, 2.0)
target_atr = st.sidebar.slider("Target (ATR)", 1.0, 10.0, 4.0)
trail_atr = st.sidebar.slider("Trailing Stop (ATR)", 0.5, 5.0, 2.0)

volatility_filter = st.sidebar.slider("Max ATR Percentile", 50, 100, 80)
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))

# =========================
# DATA
# =========================
@st.cache_data
def load_data():
    tqqq = yf.download("TQQQ", start=start_date)
    qqq = yf.download("QQQ", start=start_date)
    return tqqq.dropna(), qqq.dropna()

price, qqq = load_data()

# =========================
# INDICATORS
# =========================
price["SMA_S"] = price["Close"].rolling(short_ma).mean()
price["SMA_L"] = price["Close"].rolling(long_ma).mean()

# RSI
delta = price["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(rsi_period).mean()
avg_loss = loss.rolling(rsi_period).mean()
rs = avg_gain / avg_loss
price["RSI"] = 100 - (100 / (1 + rs))

# ATR
hl = price["High"] - price["Low"]
hc = (price["High"] - price["Close"].shift()).abs()
lc = (price["Low"] - price["Close"].shift()).abs()
tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
price["ATR"] = tr.rolling(atr_period).mean()
price["ATR_pct"] = price["ATR"].rank(pct=True) * 100

# Trend filter
qqq["SMA_200"] = qqq["Close"].rolling(200).mean()
price["Trend_OK"] = qqq["Close"] > qqq["SMA_200"]

# =========================
# SIGNAL ENGINE
# =========================
price["Signal"] = "CASH"
price["Stop"] = np.nan
price["Target"] = np.nan
price["Trail"] = np.nan

in_pos = False
entry = stop = target = trail = 0

for i in range(1, len(price)):
    row = price.iloc[i]

    if not in_pos:
        if (
            row["SMA_S"] > row["SMA_L"]
            and row["RSI"] < rsi_buy
            and row["Trend_OK"]
            and row["ATR_pct"] < volatility_filter
        ):
            entry = row["Close"]
            stop = entry - stop_atr * row["ATR"]
            target = entry + target_atr * row["ATR"]
            trail = stop
            price.iloc[i, price.columns.get_loc("Signal")] = "BUY"
            price.iloc[i, price.columns.get_loc("Stop")] = stop
            price.iloc[i, price.columns.get_loc("Target")] = target
            price.iloc[i, price.columns.get_loc("Trail")] = trail
            in_pos = True
        else:
            price.iloc[i, price.columns.get_loc("Signal")] = "NO TRADE"
    else:
        # Update trailing stop
        new_trail = row["Close"] - trail_atr * row["ATR"]
        trail = max(trail, new_trail)

        if row["Low"] <= trail:
            price.iloc[i, price.columns.get_loc("Signal")] = "TRAIL STOP"
            in_pos = False
        elif row["High"] >= target:
            price.iloc[i, price.columns.get_loc("Signal")] = "TAKE PROFIT"
            in_pos = False
        elif row["SMA_S"] < row["SMA_L"] or row["RSI"] > rsi_sell:
            price.iloc[i, price.columns.get_loc("Signal")] = "SELL"
            in_pos = False

# =========================
# POSITION SIZING
# =========================
latest = price.iloc[-1]
risk_dollars = account_size * risk_per_trade
if latest["ATR"] > 0:
    shares = int(risk_dollars / (stop_atr * latest["ATR"]))
else:
    shares = 0

# =========================
# EMAIL ALERT
# =========================
def send_email(subject, message):
    msg = MIMEText(message)
    msg["Subject"] = subject
    msg["From"] = email_from
    msg["To"] = email_to
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(email_from, email_password)
        server.send_message(msg)

alert_message = f"""
TQQQ DAILY SIGNAL (Tier-1)
-------------------------
Signal: {latest['Signal']}
Price: ${latest['Close']:.2f}
Shares (risk-based): {shares}
ATR: {latest['ATR']:.2f}
ATR Percentile: {latest['ATR_pct']:.0f}%
Trend Filter: {'ON' if latest['Trend_OK'] else 'OFF'}

This alert is generated after market close.
Not financial advice.
"""

if st.button("📧 Send Test Email"):
    send_email("TQQQ Daily Trading Signal", alert_message)
    st.success("Email sent")

# =========================
# DISPLAY
# =========================
st.subheader("📌 Current Status")
st.metric("Signal", latest["Signal"])
st.metric("Price", f"${latest['Close']:.2f}")
st.metric("Suggested Shares", shares)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(price.index, price["Close"], label="Price")
ax.plot(price.index, price["SMA_S"], label="Short MA")
ax.plot(price.index, price["SMA_L"], label="Long MA")
ax.legend()
st.pyplot(fig)

st.caption("Tier-1 discipline system for leveraged ETF trading. Past performance ≠ future results.")
