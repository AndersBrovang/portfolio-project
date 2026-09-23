# Portfolio Management Dashboard

An interactive Streamlit dashboard that builds an equal-weighted stock portfolio and evaluates it
with two of the foundational tools of modern portfolio theory: **CAPM** (Capital Asset Pricing
Model) and the **Sharpe ratio**. Prices are pulled live from Yahoo Finance and benchmarked against
the S&P 500.

---

## Why I built this

Quantitative finance is one of those fields where it's easy to read about beta, alpha and
risk-adjusted returns and feel like you understand them, right up until you have to turn them into
code. The moment you have to decide which variance goes in the denominator, how to annualise a
daily standard deviation, or what to do when a ticker returns no data, the gaps in your
understanding become very obvious very quickly.

So the goal here is deliberately not return-maximisation. There is no alpha-seeking, no signal
mining, no backtested trading strategy. The goal is:

- **Understand the mechanics.** Implement CAPM and the Sharpe ratio from raw price data rather than
  calling a library function that hands me the answer.
- **Understand the assumptions.** Every formula below rests on assumptions that are wrong in
  interesting ways. Writing them down forces me to know where the model breaks.
- **Understand the engineering.** Real financial data is messy — missing days, rate-limited APIs,
  tickers that silently return nothing. Handling that is as much a part of the work as the maths.

This is a learning project, built in the open, and I expect to keep rewriting parts of it as my
understanding improves.

---

## What it does

| Feature | Description |
|---|---|
| Stock selector | Pick any subset of 40 large-cap US tickers from the sidebar |
| Adjustable inputs | Set the annual risk-free rate and the lookback window (1y / 2y / 5y) |
| Per-stock CAPM | Beta and CAPM expected return for every holding |
| Portfolio metrics | Annualised return, portfolio beta, Sharpe ratio |
| Benchmark comparison | Portfolio Sharpe vs. S&P 500 Sharpe over the same window |
| Cumulative returns chart | Growth of each holding vs. the market (dashed benchmark line) |
| Security Market Line | Each stock plotted in beta / expected-return space against the theoretical SML |

The portfolio is **equal-weighted**: with *n* selected stocks, each gets weight `1/n`.

---

## The maths

All calculations start from daily **adjusted** closing prices (`auto_adjust=True`), so dividends and
splits are already baked into the price series.

### 1. Daily simple returns

$$r_t = \frac{P_t - P_{t-1}}{P_{t-1}} = \frac{P_t}{P_{t-1}} - 1$$

Simple (arithmetic) returns are used rather than log returns because simple returns aggregate
correctly *across assets* — a portfolio's return is the weighted sum of its holdings' returns, which
is exactly what the next step needs. (Log returns aggregate correctly across *time* instead, which
is the trade-off.)

```python
returns = prices.pct_change().dropna()
```

### 2. Portfolio return

For weights $w_i$ and asset returns $r_{i,t}$:

$$r_{p,t} = \sum_{i=1}^{n} w_i \, r_{i,t}, \qquad w_i = \frac{1}{n}$$

```python
portfolio_returns = returns[selected].dot(weights)
```

### 3. Annualisation

Daily figures are scaled to annual using 252 trading days per year. Returns scale **linearly**,
volatility scales with the **square root of time** — because variance is additive over independent
periods, and standard deviation is the square root of variance:

$$\mu_{\text{annual}} = \bar{r}_{\text{daily}} \times 252$$

$$\sigma_{\text{annual}} = \sigma_{\text{daily}} \times \sqrt{252}$$

```python
portfolio_annual_return = portfolio_returns.mean() * 252
portfolio_annual_vol    = portfolio_returns.std() * np.sqrt(252)
```

### 4. Beta — sensitivity to the market

Beta measures how much an asset moves when the market moves. It is the covariance of the asset with
the market, normalised by the market's own variance:

$$\beta_i = \frac{\operatorname{Cov}(r_i, r_m)}{\operatorname{Var}(r_m)}$$

This is exactly the slope coefficient of an ordinary least squares regression of the asset's returns
on the market's returns. Interpretation:

- $\beta = 1$ — moves with the market
- $\beta > 1$ — amplifies market moves (more systematic risk)
- $\beta < 1$ — dampens them (defensive)
- $\beta < 0$ — moves against the market (rare)

```python
cov  = np.cov(returns[t], market_returns)[0, 1]
beta = cov / var_market
```

Portfolio beta is computed the same way, from the portfolio's return series. Because beta is linear,
this is equivalent to the weighted average of the individual betas:

$$\beta_p = \sum_{i=1}^{n} w_i \beta_i$$

### 5. CAPM — expected return

The Capital Asset Pricing Model says an asset's expected return is the risk-free rate plus a premium
proportional to the systematic risk it carries:

$$E(R_i) = R_f + \beta_i \left( E(R_m) - R_f \right)$$

where:

| Symbol | Meaning | Source in this app |
|---|---|---|
| $R_f$ | Risk-free rate | User input (sidebar, default 4%) |
| $\beta_i$ | Asset beta | Computed above |
| $E(R_m)$ | Expected market return | Realised annualised S&P 500 return over the window |
| $E(R_m) - R_f$ | Equity risk premium | The reward for bearing market risk |

The core claim of CAPM is that only **systematic** (undiversifiable) risk is compensated.
Idiosyncratic, company-specific risk is not rewarded, because a diversified investor can eliminate
it for free.

```python
expected_return = risk_free_annual + beta * (market_annual_return - risk_free_annual)
```

### 6. The Security Market Line

Plotting $E(R_i)$ against $\beta_i$ gives a straight line with intercept $R_f$ and slope equal to
the equity risk premium. That line is the SML, and it is CAPM's prediction of fair pricing.

In this dashboard every stock sits *exactly* on the line — which is a deliberate teaching artefact,
not a bug. Since expected return is *derived* from beta using the same equation that defines the
line, it cannot fall anywhere else. To find mispricing you have to compare CAPM's prediction against
what the stock **actually** returned. That gap is Jensen's alpha:

$$\alpha_i = R_{i,\text{realised}} - \left[ R_f + \beta_i (R_m - R_f) \right]$$

Positive alpha = outperformed what its risk level justified. (See the roadmap — this is the next
thing I'm adding.)

### 7. Sharpe ratio — return per unit of risk

$$S = \frac{R_p - R_f}{\sigma_p}$$

Excess return over the risk-free rate, divided by total volatility. Where beta only prices
*systematic* risk, the Sharpe ratio penalises **total** risk — so it answers a different question:
"how much return am I getting for every unit of bumpiness I have to tolerate?"

Rough reading: below 1 is unremarkable, above 1 is decent, above 2 is strong — but the number is
only meaningful when compared against something, which is why the dashboard also shows the S&P 500's
Sharpe ratio over the identical window.

```python
sharpe_ratio = (portfolio_annual_return - risk_free_annual) / portfolio_annual_vol
```

---

## Assumptions and limitations

Being explicit about what this model *cannot* tell you is the whole point of the exercise.

1. **The past is not the future.** Historical beta and realised returns are used as estimates of
   expected values. Beta in particular is unstable and drifts over time.
2. **Realised ≠ expected market return.** Using the trailing S&P 500 return as $E(R_m)$ is a rough
   proxy. Over a bad 1-year window it can even be negative, which makes the CAPM expected returns
   behave counter-intuitively (high beta then implies *lower* expected return).
3. **CAPM is a single-factor model.** It explains far less of the cross-section of returns than
   multi-factor models (Fama–French size/value, momentum, quality).
4. **Normality is assumed.** The Sharpe ratio treats volatility as a complete description of risk.
   Real returns have fat tails and negative skew, so it understates crash risk. It also punishes
   upside volatility identically to downside volatility — the Sortino ratio fixes that.
5. **Equal weighting is arbitrary.** It's a neutral starting point, not an optimised allocation.
6. **Frictionless world.** No transaction costs, no taxes, no bid-ask spread, no slippage.
7. **Survivorship bias.** The ticker universe is today's large caps, which by construction excludes
   everything that failed.
8. **Constant risk-free rate.** Treated as a single user-supplied number rather than a time series.

---

## Handling real-world data

A few things that only became obvious once the app was running against live data:

- **All-NaN columns must be caught before `dropna()`.** A single ticker that returns no data becomes
  an entirely empty column; calling `.dropna()` on the frame would then delete *every row* and leave
  an empty dataset with no obvious cause. The app checks for fully-empty columns first and reports
  which ticker failed.
- **Yahoo Finance rate-limits.** Failures are surfaced as a readable message rather than a stack
  trace, since "try again in a minute" is usually the correct fix.
- **Trading calendars have to be aligned.** Rows are kept only where every selected ticker *and* the
  benchmark have a price, so all correlations are computed over an identical set of dates.

---

## Roadmap

- [ ] **Jensen's alpha** — realised return vs. CAPM prediction, so the SML chart shows real
      dispersion around the line
- [ ] **R² of the beta regression** — how much of each stock's movement the market actually explains
- [ ] **Custom portfolio weights** — replace equal weighting with user-defined allocations
- [ ] **Efficient frontier** — Monte Carlo simulation of random weight vectors, plus the maximum
      Sharpe and minimum variance portfolios
- [ ] **Correlation heatmap** — make the diversification benefit visible
- [ ] **Downside risk metrics** — Sortino ratio, maximum drawdown, historical VaR
- [ ] **Rolling beta** — show how unstable beta is over time
- [ ] **Unit tests** — verify the maths against hand-computed fixtures
- [ ] **Deploy to Streamlit Community Cloud**