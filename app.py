import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import logging
from pathlib import Path

logging.getLogger("yfinance").setLevel(logging.CRITICAL)

st.set_page_config(page_title="Portfolio", layout="wide")

ACCENT = "#178C6B"
TEXT = "#1A1A1A"
MUTED = "#6B7280"
HAIRLINE = "#E4E9E7"


def load_css(path):
    with open(path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css(Path(__file__).parent / "style.css")

# Green-anchored palette for chart lines — market line is dark/dashed
# so it reads as a benchmark rather than another stock.
CHART_COLORWAY = ["#178C6B", "#3EB489", "#0E5E45", "#79C2A0", "#2D8F6F", "#5FA88C", "#0B4A38"]

_section_counter = {"i": 0}


def section_header(text):
    idx = _section_counter["i"]
    _section_counter["i"] += 1
    delay = idx * 0.08
    st.markdown(
        f'<div class="section-label reveal" style="animation-delay:{delay}s">{text}</div>'
        f'<hr class="hairline reveal" style="animation-delay:{delay + 0.03}s">',
        unsafe_allow_html=True,
    )


def style_fig(fig):
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color=TEXT, family="Inter, sans-serif"),
        colorway=CHART_COLORWAY,
        xaxis=dict(gridcolor=HAIRLINE, zerolinecolor=HAIRLINE),
        yaxis=dict(gridcolor=HAIRLINE, zerolinecolor=HAIRLINE),
        margin=dict(t=30, l=10, r=10, b=10),
        transition=dict(duration=500, easing="cubic-in-out"),
    )
    return fig


# --- Header ---
st.markdown('<div class="page-title reveal">Portfolio Management Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-subtitle reveal" style="animation-delay:0.05s">'
    'CAPM and Sharpe ratio for a self-picked stock portfolio</div>',
    unsafe_allow_html=True,
)
st.markdown('<hr class="hairline">', unsafe_allow_html=True)

# --- Universe of stocks to pick from ---
# Extend/replace this list with whatever tickers you care about.
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "JNJ",
    "WMT", "PG", "UNH", "HD", "MA", "DIS", "BAC", "XOM", "PFE", "KO",
    "NFLX", "ADBE", "CRM", "CSCO", "INTC", "VZ", "T", "NKE", "MRK", "ABT",
    "PEP", "CVX", "TMO", "ABBV", "AVGO", "COST", "ORCL", "ACN", "MCD", "DHR",
]

MARKET_TICKER = "^GSPC"  # S&P 500 as the market proxy

st.sidebar.markdown('<div class="section-label">Stock selector</div>', unsafe_allow_html=True)
selected = st.sidebar.multiselect(
    "Choose stocks for your portfolio", TICKERS, default=TICKERS[:5]
)
risk_free_annual = st.sidebar.number_input(
    "Risk-free rate (annual, %)", value=4.0, step=0.1
) / 100
period = st.sidebar.selectbox("History period", ["1y", "2y", "5y"], index=1)

if len(selected) == 0:
    st.warning("Select at least one stock in the sidebar.")
    st.stop()

weights = np.array([1 / len(selected)] * len(selected))


@st.cache_data
def get_prices(tickers, period):
    data = yf.download(tickers, period=period, auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
    return data


all_tickers = selected + [MARKET_TICKER]
prices = get_prices(all_tickers, period)

# Flag any ticker that came back with no usable data BEFORE dropping rows,
# since one all-NaN column would otherwise wipe every row via dropna().
missing = prices.columns[prices.isna().all()].tolist()
if missing:
    st.error(
        f"No price data returned for: {', '.join(missing)}. "
        "This is usually Yahoo Finance rate-limiting or a bad ticker symbol — "
        "try again in a minute, or remove that ticker."
    )
    st.stop()

prices = prices.dropna()
if prices.empty:
    st.error("No overlapping trading days left across the selected tickers after cleaning.")
    st.stop()

returns = prices.pct_change().dropna()

# --- Equal-weighted portfolio ---
portfolio_returns = returns[selected].dot(weights)
market_returns = returns[MARKET_TICKER]
var_market = np.var(market_returns)
market_annual_return = market_returns.mean() * 252

# --- CAPM per stock: beta is the OLS slope of stock returns on market returns ---
capm_rows = []
for t in selected:
    cov = np.cov(returns[t], market_returns)[0, 1]
    beta = cov / var_market
    expected_return = risk_free_annual + beta * (market_annual_return - risk_free_annual)
    capm_rows.append({"Ticker": t, "Beta": beta, "CAPM Expected Return": expected_return})

capm_df = pd.DataFrame(capm_rows).set_index("Ticker")

# --- Portfolio-level beta and Sharpe ratio ---
cov_portfolio = np.cov(portfolio_returns, market_returns)[0, 1]
portfolio_beta = cov_portfolio / var_market
portfolio_annual_return = portfolio_returns.mean() * 252
portfolio_annual_vol = portfolio_returns.std() * np.sqrt(252)
sharpe_ratio = (portfolio_annual_return - risk_free_annual) / portfolio_annual_vol

# --- Metrics ---
section_header("At a glance")
market_annual_vol = market_returns.std() * np.sqrt(252)
market_sharpe = (market_annual_return - risk_free_annual) / market_annual_vol

col1, col2, col3, col4 = st.columns(4)
col1.metric("Portfolio Annual Return", f"{portfolio_annual_return:.2%}")
col2.metric("Portfolio Beta", f"{portfolio_beta:.2f}")
col3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
col4.metric(
    "Market Sharpe", f"{market_sharpe:.2f}",
    delta=f"{sharpe_ratio - market_sharpe:+.2f} vs portfolio",
    delta_color="off",
)

st.caption(
    f"Market (S&P 500) annual return: {market_annual_return:.2%} | "
    f"Risk-free rate: {risk_free_annual:.2%}"
)

# --- CAPM table ---
section_header("CAPM: Beta and Expected Return per Stock")
st.dataframe(
    capm_df.style.format({"Beta": "{:.2f}", "CAPM Expected Return": "{:.2%}"}),
    width="stretch",
)

# --- Cumulative returns chart ---
section_header("Cumulative Returns")
cum_returns = (1 + returns[selected + [MARKET_TICKER]]).cumprod() - 1
fig1 = go.Figure()
for col in cum_returns.columns:
    is_market = col == MARKET_TICKER
    fig1.add_trace(go.Scatter(
        x=cum_returns.index, y=cum_returns[col], name=col,
        line=dict(dash="dash", color="#2D3142", width=2) if is_market else dict(width=2),
    ))
fig1.update_layout(yaxis_tickformat=".0%", hovermode="x unified")
st.plotly_chart(style_fig(fig1), width="stretch")

# --- Beta vs expected return (security market line) ---
section_header("Security Market Line: Beta vs CAPM Expected Return")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=capm_df["Beta"], y=capm_df["CAPM Expected Return"],
    mode="markers+text", text=capm_df.index, textposition="top center",
    marker=dict(size=10, color=ACCENT),
))
beta_range = np.linspace(0, capm_df["Beta"].max() * 1.2, 50)
sml = risk_free_annual + beta_range * (market_annual_return - risk_free_annual)
fig2.add_trace(go.Scatter(
    x=beta_range, y=sml, mode="lines", name="SML",
    line=dict(dash="dash", color="#2D3142"),
))
fig2.update_layout(xaxis_title="Beta", yaxis_title="Expected Return", yaxis_tickformat=".0%")
st.plotly_chart(style_fig(fig2), width="stretch")