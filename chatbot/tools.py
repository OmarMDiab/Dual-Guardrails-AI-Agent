"""
chatbot/tools.py — LangChain @tool definitions.

Tools are bound to the LLM via .bind_tools().
The agent decides at runtime whether to call any tool.

Tools:
  search_financial_web      — Tavily web search (news, events, general data)
  get_stock_data            — yfinance real-time stock fundamentals
  get_crypto_price          — CoinGecko live crypto prices (no key needed)
  calculate_compound_growth — compound interest / savings projection
  calculate_loan_payment    — monthly mortgage / loan payment
"""

import os
import requests
from langchain_core.tools import tool
from tavily import TavilyClient


# ── Tool 1 : Tavily web search ────────────────────────────────────────────────

@tool
def search_financial_web(query: str) -> str:
    """Search the web for real-time financial information.

    Use for: recent news, earnings announcements, market events, analyst
    opinions, or any financial topic that requires up-to-date web data.
    Do NOT use for calculations or data available from other tools.
    """
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        return "Web search unavailable — TAVILY_API_KEY not configured."

    client = TavilyClient(api_key=api_key)
    try:
        response = client.search(
            query,
            search_depth="advanced",
            chunks_per_source=3,
            max_results=5,
            topic="finance",
        )
        results = response.get("results", [])
        if not results:
            return "No results found."
        parts = [
            f"[{r.get('title', 'Source')}]({r.get('url', '')}) · score={r.get('score', 0):.2f}\n{r.get('content', '').strip()}"
            for r in results
        ]
        print(f"[Tavily] {len(parts)} result(s) for {query!r}")
        return "\n\n".join(parts)
    except Exception as exc:
        print(f"[Tavily] error={exc}")
        return f"Search failed: {exc}"


# ── Tool 2 : yfinance stock data ──────────────────────────────────────────────

@tool
def get_stock_data(ticker: str) -> str:
    """Get real-time stock fundamentals for any publicly traded company.

    Returns: current price, market cap, P/E ratio, EPS, 52-week range,
    dividend yield, revenue, analyst target price, and recommendation.
    Use with ticker symbols: AAPL, MSFT, GOOGL, SPY, QQQ, BRK-B, etc.
    """
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker.strip().upper())
        info  = stock.info

        price   = info.get("currentPrice") or info.get("regularMarketPrice", "N/A")
        mkt_cap = info.get("marketCap")
        pe      = info.get("trailingPE", "N/A")
        eps     = info.get("trailingEps", "N/A")
        wk_low  = info.get("fiftyTwoWeekLow", "N/A")
        wk_high = info.get("fiftyTwoWeekHigh", "N/A")
        div_yld = info.get("dividendYield") or 0
        revenue = info.get("totalRevenue")
        target  = info.get("targetMeanPrice", "N/A")
        rec     = info.get("recommendationKey", "N/A").upper()
        name    = info.get("longName", ticker)

        lines = [
            f"**{name} ({ticker.upper()})**",
            f"Price            : ${price}",
            f"Market Cap       : ${mkt_cap:,}" if mkt_cap else "Market Cap       : N/A",
            f"P/E Ratio (TTM)  : {pe}",
            f"EPS (TTM)        : ${eps}",
            f"52-Week Range    : ${wk_low} – ${wk_high}",
            f"Dividend Yield   : {div_yld * 100:.2f}%",
            f"Revenue (TTM)    : ${revenue:,}" if revenue else "Revenue (TTM)    : N/A",
            f"Analyst Target   : ${target}",
            f"Recommendation   : {rec}",
        ]
        print(f"[yfinance] retrieved data for {ticker.upper()}")
        return "\n".join(lines)
    except Exception as exc:
        print(f"[yfinance] error={exc}")
        return f"Could not retrieve stock data for '{ticker}': {exc}"


# ── Tool 3 : CoinGecko crypto prices (no API key required) ───────────────────

@tool
def get_crypto_price(coin_id: str) -> str:
    """Get the current price and market data for a cryptocurrency from CoinGecko.

    Use CoinGecko coin IDs (lowercase):
      bitcoin, ethereum, solana, cardano, ripple, dogecoin,
      chainlink, polkadot, avalanche-2, uniswap, litecoin, etc.
    No API key required.
    """
    cid = coin_id.strip().lower()
    url = (
        f"https://api.coingecko.com/api/v3/simple/price"
        f"?ids={cid}&vs_currencies=usd"
        f"&include_market_cap=true&include_24hr_change=true&include_24hr_vol=true"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json().get(cid)
        if not data:
            return f"Coin '{cid}' not found on CoinGecko. Check the coin ID."
        price   = data.get("usd", "N/A")
        mkt_cap = data.get("usd_market_cap")
        chg_24h = data.get("usd_24h_change", 0)
        vol_24h = data.get("usd_24h_vol")
        arrow   = "▲" if chg_24h >= 0 else "▼"
        lines = [
            f"**{cid.title()}**",
            f"Price (USD)  : ${price:,.4f}" if isinstance(price, float) else f"Price (USD)  : ${price}",
            f"24h Change   : {arrow} {chg_24h:+.2f}%",
            f"Market Cap   : ${mkt_cap:,.0f}" if mkt_cap else "Market Cap   : N/A",
            f"24h Volume   : ${vol_24h:,.0f}" if vol_24h else "24h Volume   : N/A",
        ]
        print(f"[CoinGecko] {cid} = ${price}")
        return "\n".join(lines)
    except Exception as exc:
        print(f"[CoinGecko] error={exc}")
        return f"CoinGecko request failed for '{cid}': {exc}"


# ── Tool 4 : Compound growth calculator ──────────────────────────────────────

@tool
def calculate_compound_growth(
    principal: float,
    annual_rate_percent: float,
    years: int,
    monthly_contribution: float = 0.0,
) -> str:
    """Calculate investment or savings growth using compound interest.

    Use for: retirement projections, college savings, emergency fund goals,
    or any 'how much will X grow to in Y years at Z% return' question.

    Args:
        principal            : initial investment in USD
        annual_rate_percent  : expected annual return, e.g. 7 for 7%
        years                : investment horizon in years
        monthly_contribution : optional additional monthly contribution
    """
    r  = annual_rate_percent / 100 / 12
    n  = years * 12
    fv_p = principal * (1 + r) ** n
    fv_c = (monthly_contribution * (((1 + r) ** n - 1) / r)) if r > 0 else monthly_contribution * n
    fv   = fv_p + fv_c
    invested  = principal + monthly_contribution * n
    growth    = fv - invested

    return (
        f"**Compound Growth Projection**\n"
        f"Initial investment    : ${principal:,.2f}\n"
        f"Monthly contribution  : ${monthly_contribution:,.2f}\n"
        f"Annual return         : {annual_rate_percent}%\n"
        f"Time horizon          : {years} years\n"
        f"─────────────────────────────\n"
        f"Total invested        : ${invested:,.2f}\n"
        f"Total growth          : ${growth:,.2f}\n"
        f"**Final value         : ${fv:,.2f}**"
    )


# ── Tool 5 : Loan / mortgage payment calculator ───────────────────────────────

@tool
def calculate_loan_payment(
    principal: float,
    annual_rate_percent: float,
    years: int,
) -> str:
    """Calculate monthly payment and total cost for a loan or mortgage.

    Use for: mortgage affordability, car loan analysis, student debt,
    or any 'what will my monthly payment be' question.

    Args:
        principal           : loan amount in USD
        annual_rate_percent : annual interest rate, e.g. 6.5 for 6.5%
        years               : loan term in years
    """
    r = annual_rate_percent / 100 / 12
    n = years * 12
    monthly = (principal * r * (1 + r) ** n / ((1 + r) ** n - 1)) if r > 0 else principal / n
    total_paid     = monthly * n
    total_interest = total_paid - principal

    return (
        f"**Loan Payment Calculator**\n"
        f"Loan amount           : ${principal:,.2f}\n"
        f"Annual interest rate  : {annual_rate_percent}%\n"
        f"Term                  : {years} years ({n} payments)\n"
        f"─────────────────────────────\n"
        f"**Monthly payment     : ${monthly:,.2f}**\n"
        f"Total paid            : ${total_paid:,.2f}\n"
        f"Total interest        : ${total_interest:,.2f}"
    )


# ── Tool registry ─────────────────────────────────────────────────────────────

FINBOT_TOOLS = [
    search_financial_web,
    get_stock_data,
    get_crypto_price,
    calculate_compound_growth,
    calculate_loan_payment,
]
