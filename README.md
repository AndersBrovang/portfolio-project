# Portfolio Management Dashboard

An interactive Streamlit dashboard that builds an equal-weighted stock portfolio and measures its
risk and return. It applies two standard tools from portfolio theory: the Capital Asset Pricing
Model (CAPM) and the Sharpe ratio. Price data is downloaded from Yahoo Finance, and the portfolio is
compared against the S&P 500 as a benchmark.

---

## Why I built this

I kept coming across terms like beta, alpha and risk-adjusted return in reading and lectures, and I
could repeat the definitions without being able to do much with them. Writing the calculations in
code exposed that gap fairly quickly. Working out which variance belongs in the denominator of beta,
why volatility is scaled by the square root of time, and what the program should do when a ticker
returns no data were all problems I had to solve before the dashboard would run at all.

The aim is not to maximise returns. There is no trading strategy here and no attempt to identify
undervalued stocks. What I wanted from the project was:

- To implement CAPM and the Sharpe ratio starting from raw price data, rather than calling a
  function that returns the answer directly.
- To write down the assumptions behind each formula, so that I know where the results stop being
  reliable.
- To work with real market data, which arrives with missing days, rate limits and occasional empty
  responses.

I expect to revise parts of this as I learn more. The limitations section reflects what I currently
understand about where the model falls short.

---

## What it does

| Feature | Description |
|---|---|
| Stock selector | Pick any subset of 40 large-cap US tickers from the sidebar |
| Adjustable inputs | Set the annual risk-free rate and the lookback window (1y / 2y / 5y) |
| Per-stock CAPM | Beta and CAPM expected return for every holding |
| Portfolio metrics | Annualised return, portfolio beta, Sharpe ratio |
| Benchmark comparison | Portfolio Sharpe against S&P 500 Sharpe over the same window |
| Cumulative returns chart | Growth of each holding against the market (dashed benchmark line) |
| Security Market Line | Each stock plotted by beta and expected return against the theoretical SML |

The portfolio is **equal-weighted**, so with *n* selected stocks each one is given a weight of `1/n`.

---

## The maths

All calculations start from daily **adjusted** closing prices (`auto_adjust=True`), which means
dividends and stock splits are already accounted for in the price series.

### 1. Daily simple returns

$$r_t = \frac{P_t - P_{t-1}}{P_{t-1}} = \frac{P_t}{P_{t-1}} - 1$$

I used simple returns rather than log returns. Simple returns can be added up across assets, so a
portfolio's return is just the weighted sum of the returns of the stocks in it, which is what the
next step requires. Log returns have the opposite property: they add up correctly over time but not
across assets.

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

Daily figures are converted to annual ones using 252 trading days per year. Returns are multiplied
by 252, but volatility is multiplied by the square root of 252. The reason is that variance adds up
over time while standard deviation does not, and standard deviation is the square root of variance.

$$\mu_{\text{annual}} = \bar{r}_{\text{daily}} \times 252$$

$$\sigma_{\text{annual}} = \sigma_{\text{daily}} \times \sqrt{252}$$

```python
portfolio_annual_return = portfolio_returns.mean() * 252
portfolio_annual_vol    = portfolio_returns.std() * np.sqrt(252)
```

### 4. Beta

Beta measures how strongly a stock tends to move when the market moves. It is the covariance between
the stock and the market, divided by the variance of the market:

$$\beta_i = \frac{\text{Cov}(r_i, r_m)}{\text{Var}(r_m)}$$

This is the same value as the slope of a linear regression of the stock's returns on the market's
returns, which I found helpful, because it turns beta from a finance formula into a statistics one.

- $\beta = 1$: the stock tends to move in line with the market.
- $\beta > 1$: the stock tends to move more than the market in both directions.
- $\beta < 1$: the stock tends to move less than the market.
- $\beta < 0$: the stock tends to move in the opposite direction, which is uncommon.

```python
cov  = np.cov(returns[t], market_returns)[0, 1]
beta = cov / var_market
```

Portfolio beta is calculated the same way, using the portfolio's own return series. Because the
formula is linear, this gives the same result as taking the weighted average of the individual
betas:

$$\beta_p = \sum_{i=1}^{n} w_i \beta_i$$

### 5. CAPM expected return

CAPM states that the expected return on an asset is the risk-free rate plus a premium that is
proportional to how much market risk the asset carries:

$$E(R_i) = R_f + \beta_i \left( E(R_m) - R_f \right)$$

| Symbol | Meaning | Where it comes from in this app |
|---|---|---|
| $R_f$ | Risk-free rate | User input in the sidebar, default 4% |
| $\beta_i$ | Asset beta | Calculated above |
| $E(R_m)$ | Expected market return | Annualised S&P 500 return over the selected window |
| $E(R_m) - R_f$ | Equity risk premium | The extra return expected for holding market risk |

The main idea behind CAPM is that investors are only compensated for market risk, meaning the risk
that affects everything at once and cannot be diversified away. Risk that is specific to one company
is not rewarded, because an investor holding a diversified portfolio can remove it at no cost.

```python
expected_return = risk_free_annual + beta * (market_annual_return - risk_free_annual)
```

### 6. The Security Market Line

Plotting expected return against beta produces a straight line. It starts at the risk-free rate and
its slope is the equity risk premium. This line is the Security Market Line, and it represents what
CAPM considers a fair return for a given level of market risk.

In this dashboard every stock sits exactly on the line. That is expected rather than a bug: the
expected return is calculated from beta using the same equation that draws the line, so no point can
fall anywhere else. To show whether a stock is mispriced, the CAPM prediction has to be compared
with the return the stock actually delivered. That difference is Jensen's alpha:

$$\alpha_i = R_{i,\text{realised}} - \left[ R_f + \beta_i (R_m - R_f) \right]$$

A positive alpha means the stock returned more than its level of risk would justify. Adding this is
the next item on the roadmap.

### 7. Sharpe ratio

$$S = \frac{R_p - R_f}{\sigma_p}$$

The Sharpe ratio takes the return earned above the risk-free rate and divides it by the portfolio's
volatility, giving a measure of return per unit of risk. Beta only accounts for market risk, while
the Sharpe ratio uses total volatility, so it also reflects risk that is specific to the individual
stocks in the portfolio.

As a rough guide, a value below 1 is weak, above 1 is reasonable and above 2 is strong. The number
means little on its own, so the dashboard also calculates the Sharpe ratio of the S&P 500 over the
same period as a point of comparison.

```python
sharpe_ratio = (portfolio_annual_return - risk_free_annual) / portfolio_annual_vol
```

---

## Assumptions and limitations

The results rest on a number of assumptions, and most of them are simplifications:

1. **Past data is used to estimate future values.** Beta and average returns are calculated from
   historical prices. Beta in particular is not stable and changes over time.
2. **Realised market return is not the same as expected market return.** Using the trailing S&P 500
   return as $E(R_m)$ is a rough substitute. In a poor year it can be negative, which makes the CAPM
   results behave strangely, since a high beta would then imply a lower expected return.
3. **CAPM only uses one factor.** Models that add further factors, such as the Fama-French size and
   value factors or momentum, explain differences in returns between stocks considerably better.
4. **Volatility is treated as a full description of risk.** Real returns have more extreme outcomes
   than a normal distribution would predict, so the Sharpe ratio understates the risk of a large
   loss. It also treats upward and downward volatility as equally bad, which the Sortino ratio
   corrects for.
5. **Costs are ignored.** There are no transaction costs, taxes, spreads or slippage in the model.
6. **Survivorship bias.** The list of tickers consists of companies that are large today, so it
   leaves out those that performed badly or failed.
7. **The risk-free rate is constant.** It is entered as a single number rather than a series that
   changes over time.

---

## Roadmap

- [ ] **Jensen's alpha**: compare realised returns with the CAPM prediction so the SML chart shows
      actual dispersion around the line
- [ ] **R-squared of the beta regression**: how much of each stock's movement the market explains
- [ ] **Custom portfolio weights** to replace the equal weighting
- [ ] **Efficient frontier** using a Monte Carlo simulation of random weights, marking the maximum
      Sharpe and minimum variance portfolios
- [ ] **Correlation heatmap** to make the effect of diversification visible
- [ ] **Downside risk measures**: Sortino ratio, maximum drawdown and historical VaR
- [ ] **Rolling beta** to show how much beta changes over time
- [ ] **Unit tests** checking the calculations against values worked out by hand
- [ ] **Deploy to Streamlit Community Cloud**