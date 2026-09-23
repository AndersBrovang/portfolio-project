import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import logging
from pathlib import Path

logging.getLogger("yfinance").setLevel(logging.CRITICAL)

st.set_page_config(page_title="Portfolio Dashboard", layout="wide")

# --- Palette -----------------------------------------------------------------
ACCENT = "#178C6B"
POSITIVE = "#178C6B"
NEGATIVE = "#C2544F"
TEXT = "#1A1A1A"
MUTED = "#6B7280"
HAIRLINE = "#E4E9E7"
BENCHMARK = "#2D3142"

# Market line is dark and dashed so it reads as a benchmark, not another stock.
CHART_COLORWAY = ["#178C6B", "#3EB489", "#0E5E45", "#79C2A0", "#2D8F6F", "#5FA88C", "#0B4A38"]

TRADING_DAYS = 252
MARKET_TICKER = "^GSPC"  # S&P 500 as the market proxy


# --- Finance helpers ---------------------------------------------------------
def annualise_return(daily_returns):
    """Scale a mean daily return to an annual figure. Returns grow linearly with time."""
    return daily_returns.mean() * TRADING_DAYS


def annualise_vol(daily_returns):
    """Scale a daily standard deviation to an annual one.

    Volatility grows with the square root of time because variance is additive
    over independent periods and standard deviation is the square root of variance.
    """
    return daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS)


def beta_and_r2(asset_returns, market_returns):
    """Return the OLS slope of asset on market, plus the R-squared of that regression.

    Beta is Cov(asset, market) / Var(market). Both terms are taken from the same
    covariance matrix so the degrees of freedom always agree.

    With a single predictor, R-squared is the squared correlation between the two
    series. It says how much of the stock's movement the market actually explains,
    which is the missing piece of context next to beta: a beta of 1.4 built on an
    R-squared of 0.1 is a far weaker claim than the same beta at 0.8.
    """
    cov_matrix = np.cov(asset_returns, market_returns, ddof=1)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    r_squared = np.corrcoef(asset_returns, market_returns)[0, 1] ** 2
    return beta, r_squared


def capm_expected_return(beta, risk_free, market_return):
    """CAPM fair return: the risk-free rate plus beta times the equity risk premium."""
    return risk_free + beta * (market_return - risk_free)


def jensens_alpha(realised_return, beta, risk_free, market_return):
    """Realised return minus what CAPM predicted for that level of market risk.

    Positive alpha means the asset returned more than its risk exposure justified.
    """
    return realised_return - capm_expected_return(beta, risk_free, market_return)


def sortino_ratio(daily_returns, risk_free_annual):
    """Excess return divided by downside deviation only.

    The Sharpe ratio penalises upward and downward volatility equally, which
    punishes a portfolio for rising quickly. Sortino measures the spread of
    returns that fell short of the risk-free rate and ignores the rest.
    """
    daily_rf = risk_free_annual / TRADING_DAYS
    shortfall = np.minimum(daily_returns - daily_rf, 0)
    downside_dev = np.sqrt((shortfall ** 2).mean()) * np.sqrt(TRADING_DAYS)
    if downside_dev == 0:
        return np.nan
    return (annualise_return(daily_returns) - risk_free_annual) / downside_dev


def sharpe_ratio(daily_returns, risk_free_annual):
    """Excess return divided by total volatility."""
    vol = annualise_vol(daily_returns)
    if vol == 0:
        return np.nan
    return (annualise_return(daily_returns) - risk_free_annual) / vol


# --- Page furniture ----------------------------------------------------------
def load_css(path):
    with open(path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css(Path(__file__).parent / "style.css")

_section_counter = {"i": 0}


def section_header(text, caption=None):
    idx = _section_counter["i"]
    _section_counter["i"] += 1
    delay = idx * 0.06
    html = (
        f'<div class="section-block reveal" style="animation-delay:{delay}s">'
        f'<div class="section-label">{text}</div>'
    )
    if caption:
        html += f'<div class="section-caption">{caption}</div>'
    html += '<hr class="hairline"></div>'
    st.markdown(html, unsafe_allow_html=True)


def metric_card(label, value, sub=None, tone="neutral"):
    """A metric tile. `tone` drives the accent colour: positive, negative or neutral."""
    sub_html = f'<div class="metric-sub {tone}">{sub}</div>' if sub else ""
    return (
        f'<div class="metric-card {tone}">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'{sub_html}'
        f"</div>"
    )


def style_fig(fig, height=420):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family="Inter, sans-serif", size=13),
        colorway=CHART_COLORWAY,
        xaxis=dict(gridcolor=HAIRLINE, zerolinecolor=HAIRLINE, linecolor=HAIRLINE,
                   tickfont=dict(color=MUTED, size=11)),
        yaxis=dict(gridcolor=HAIRLINE, zerolinecolor=HAIRLINE, linecolor=HAIRLINE,
                   tickfont=dict(color=MUTED, size=11)),
        margin=dict(t=20, l=10, r=10, b=10),
        height=height,
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=HAIRLINE,
                        font=dict(color=TEXT, family="Inter, sans-serif", size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=11, color=MUTED), bgcolor="rgba(0,0,0,0)"),
        transition=dict(duration=400, easing="cubic-in-out"),
    )
    return fig


# --- Header ------------------------------------------------------------------
st.markdown(
    '<div class="hero reveal">'
    '<div class="hero-eyebrow">CAPM · Sharpe · Sortino · Jensen\'s Alpha</div>'
    '<h1 class="hero-title">Portfolio Management Dashboard</h1>'
    '<p class="hero-subtitle">Risk and return analysis for a self-selected stock '
    'portfolio, benchmarked against the S&amp;P 500.</p>'
    "</div>",
    unsafe_allow_html=True,
)

# --- Controls ----------------------------------------------------------------
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "JNJ",
    "WMT", "PG", "UNH", "HD", "MA", "DIS", "BAC", "XOM", "PFE", "KO",
    "NFLX", "ADBE", "CRM", "CSCO", "INTC", "VZ", "T", "NKE", "MRK", "ABT",
    "PEP", "CVX", "TMO", "ABBV", "AVGO", "COST", "ORCL", "ACN", "MCD", "DHR",
]

st.sidebar.markdown('<div class="sidebar-title">Configuration</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-group-label">Holdings</div>', unsafe_allow_html=True)
selected = st.sidebar.multiselect(
    "Choose stocks for your portfolio", TICKERS, default=TICKERS[:5],
    label_visibility="collapsed",
)
st.sidebar.caption("Equal-weighted: each holding receives 1/n of the portfolio.")

st.sidebar.markdown('<div class="sidebar-group-label">Parameters</div>', unsafe_allow_html=True)
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


prices = get_prices(selected + [MARKET_TICKER], period)

missing = prices.columns[prices.isna().all()].tolist()
if missing:
    st.error(
        f"No price data returned for: {', '.join(missing)}. "
        "This is usually Yahoo Finance rate-limiting or a bad ticker symbol. "
        "Try again in a minute, or remove that ticker."
    )
    st.stop()

prices = prices.dropna()
if prices.empty:
    st.error("No overlapping trading days left across the selected tickers after cleaning.")
    st.stop()

returns = prices.pct_change().dropna()

# --- Portfolio and market ----------------------------------------------------
portfolio_returns = returns[selected].dot(weights)
market_returns = returns[MARKET_TICKER]
market_annual_return = annualise_return(market_returns)

portfolio_beta, portfolio_r2 = beta_and_r2(portfolio_returns, market_returns)
portfolio_annual_return = annualise_return(portfolio_returns)
portfolio_annual_vol = annualise_vol(portfolio_returns)
portfolio_sharpe = sharpe_ratio(portfolio_returns, risk_free_annual)
portfolio_sortino = sortino_ratio(portfolio_returns, risk_free_annual)
portfolio_alpha = jensens_alpha(
    portfolio_annual_return, portfolio_beta, risk_free_annual, market_annual_return
)

market_sharpe = sharpe_ratio(market_returns, risk_free_annual)
market_sortino = sortino_ratio(market_returns, risk_free_annual)

# --- Per-stock CAPM ----------------------------------------------------------
rows = []
for t in selected:
    beta, r2 = beta_and_r2(returns[t], market_returns)
    realised = annualise_return(returns[t])
    expected = capm_expected_return(beta, risk_free_annual, market_annual_return)
    rows.append({
        "Ticker": t,
        "Beta": beta,
        "R2": r2,
        "Expected (CAPM)": expected,
        "Realised": realised,
        "Alpha": realised - expected,
    })

capm_df = pd.DataFrame(rows).set_index("Ticker")

# --- At a glance -------------------------------------------------------------
section_header(
    "At a glance",
    f"{len(selected)} holdings · {len(returns):,} trading days · "
    f"{prices.index[0]:%b %Y} to {prices.index[-1]:%b %Y}",
)

alpha_tone = "positive" if portfolio_alpha >= 0 else "negative"
cards = [
    metric_card("Annual Return", f"{portfolio_annual_return:.2%}",
                f"Market {market_annual_return:.2%}"),
    metric_card("Annual Volatility", f"{portfolio_annual_vol:.2%}",
                f"Risk-free {risk_free_annual:.2%}"),
    metric_card("Portfolio Beta", f"{portfolio_beta:.2f}",
                f"R² {portfolio_r2:.2f} explained by market"),
    metric_card("Jensen's Alpha", f"{portfolio_alpha:+.2%}",
                "above CAPM prediction" if portfolio_alpha >= 0 else "below CAPM prediction",
                tone=alpha_tone),
    metric_card("Sharpe Ratio", f"{portfolio_sharpe:.2f}",
                f"Market {market_sharpe:.2f}",
                tone="positive" if portfolio_sharpe >= market_sharpe else "negative"),
    metric_card("Sortino Ratio",
                "n/a" if np.isnan(portfolio_sortino) else f"{portfolio_sortino:.2f}",
                f"Market {market_sortino:.2f}" if not np.isnan(market_sortino) else "no downside days",
                # A NaN comparison is always False, so guard it rather than showing red
                tone="neutral" if np.isnan(portfolio_sortino) or np.isnan(market_sortino)
                else ("positive" if portfolio_sortino >= market_sortino else "negative")),
]
st.markdown(f'<div class="metric-grid reveal">{"".join(cards)}</div>', unsafe_allow_html=True)

# --- CAPM table --------------------------------------------------------------
section_header(
    "Per-stock breakdown",
    "R² is the share of each stock's daily movement explained by the market. "
    "Alpha is the realised return minus the CAPM prediction.",
)

styled = (
    capm_df.style
    .format({
        "Beta": "{:.2f}",
        "R2": "{:.2f}",
        "Expected (CAPM)": "{:.2%}",
        "Realised": "{:.2%}",
        "Alpha": "{:+.2%}",
    })
    .map(lambda v: f"color: {POSITIVE}; font-weight: 600;" if v >= 0
         else f"color: {NEGATIVE}; font-weight: 600;", subset=["Alpha"])
    .background_gradient(subset=["R2"], cmap="Greens", vmin=0, vmax=1)
)
st.dataframe(styled, width="stretch")

# --- Cumulative returns ------------------------------------------------------
section_header("Cumulative returns", "Growth of one unit invested at the start of the window.")

cum_returns = (1 + returns[selected + [MARKET_TICKER]]).cumprod() - 1
fig1 = go.Figure()
for col in cum_returns.columns:
    is_market = col == MARKET_TICKER
    fig1.add_trace(go.Scatter(
        x=cum_returns.index, y=cum_returns[col],
        name="S&P 500" if is_market else col,
        line=dict(dash="dash", color=BENCHMARK, width=2) if is_market else dict(width=1.8),
        hovertemplate="%{y:.1%}<extra>%{fullData.name}</extra>",
    ))
fig1.update_layout(yaxis_tickformat=".0%", hovermode="x unified")
st.plotly_chart(style_fig(fig1), width="stretch")

# --- Security market line ----------------------------------------------------
section_header(
    "Security Market Line",
    "The dashed line is the return CAPM considers fair for a given beta. "
    "Each vertical drop is that stock's alpha.",
)

fig2 = go.Figure()

beta_range = np.linspace(0, max(capm_df["Beta"].max() * 1.15, 1.2), 50)
fig2.add_trace(go.Scatter(
    x=beta_range,
    y=capm_expected_return(beta_range, risk_free_annual, market_annual_return),
    mode="lines", name="Security Market Line",
    line=dict(dash="dash", color=BENCHMARK, width=1.5),
    hoverinfo="skip",
))

# Vertical segments from the CAPM prediction to what each stock actually returned.
for t, row in capm_df.iterrows():
    fig2.add_trace(go.Scatter(
        x=[row["Beta"], row["Beta"]],
        y=[row["Expected (CAPM)"], row["Realised"]],
        mode="lines", showlegend=False, hoverinfo="skip",
        line=dict(color=POSITIVE if row["Alpha"] >= 0 else NEGATIVE, width=1.5),
        opacity=0.45,
    ))

for tone, label, mask in [
    ("positive", "Positive alpha", capm_df["Alpha"] >= 0),
    ("negative", "Negative alpha", capm_df["Alpha"] < 0),
]:
    subset = capm_df[mask]
    if subset.empty:
        continue
    fig2.add_trace(go.Scatter(
        x=subset["Beta"], y=subset["Realised"],
        mode="markers+text", name=label,
        text=subset.index,
        # Push labels away from the SML so positive and negative points do not collide
        textposition="top center" if tone == "positive" else "bottom center",
        textfont=dict(size=11, color=TEXT),
        marker=dict(size=11, color=POSITIVE if tone == "positive" else NEGATIVE,
                    line=dict(width=2, color="#FFFFFF")),
        customdata=np.stack([subset["Alpha"], subset["Expected (CAPM)"], subset["R2"]], axis=-1),
        hovertemplate=(
            "<b>%{text}</b><br>Beta %{x:.2f}<br>Realised %{y:.1%}"
            "<br>CAPM expected %{customdata[1]:.1%}<br>Alpha %{customdata[0]:+.1%}"
            "<br>R² %{customdata[2]:.2f}<extra></extra>"
        ),
    ))

fig2.update_layout(
    xaxis_title="Beta", yaxis_title="Annualised return", yaxis_tickformat=".0%",
)
st.plotly_chart(style_fig(fig2, height=460), width="stretch")

# --- Alpha ranking -----------------------------------------------------------
section_header("Alpha by holding", "Return earned above or below the CAPM prediction.")

alpha_sorted = capm_df["Alpha"].sort_values()
fig3 = go.Figure(go.Bar(
    x=alpha_sorted.values, y=alpha_sorted.index, orientation="h",
    marker=dict(color=[POSITIVE if v >= 0 else NEGATIVE for v in alpha_sorted.values]),
    hovertemplate="<b>%{y}</b><br>Alpha %{x:+.2%}<extra></extra>",
))
fig3.update_layout(
    xaxis_tickformat=".0%", xaxis_title="Jensen's alpha", bargap=0.45,
)
st.plotly_chart(style_fig(fig3, height=max(240, 46 * len(alpha_sorted))), width="stretch")

st.markdown(
    '<div class="footnote">Educational project. Historical data is used to estimate '
    'forward-looking quantities, so none of these figures are predictions. '
    'Not investment advice.</div>',
    unsafe_allow_html=True,
)
