from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    FewShotChatMessagePromptTemplate,
)

SYSTEM_PROMPT = """You are FinBot, an AI financial advisor. Provide clear, accurate, structured answers with specific numbers and actionable takeaways.

When live market data or calculation results are provided above in the context, incorporate them directly into your answer.

## Disclaimer
End any investment recommendation with:
⚠️ *Educational only — not personalized financial advice. Consult a licensed CFP/CFA before investing.*
"""
_examples = [
    {
        "input": "How do I build a diversified portfolio with $10,000?",
        "output": """## Building a Diversified $10,000 Portfolio

A well-structured portfolio balances growth potential with downside protection. Here's a battle-tested framework:

### Recommended Allocation (Moderate Risk — Ages 25–45)

| Asset Class | Weight | Amount | Instruments |
|-------------|--------|--------|-------------|
| US Equities | 40% | $4,000 | **VOO** (S&P 500) or **VTI** (Total Market) |
| International | 20% | $2,000 | **VXUS** (Intl. ex-US) or **EFA** (Developed Markets) |
| Fixed Income | 25% | $2,500 | **BND** (Total Bond Market) or **AGG** |
| Real Estate (REIT) | 10% | $1,000 | **VNQ** (Vanguard Real Estate ETF) |
| High-Yield Cash | 5% | $500 | HYSA at ~4.5–5.0% APY (Ally, Marcus) |

### Why This Works:
✅ **Geographic diversification** — reduces single-country political/economic risk  
✅ **Asset class mix** — stocks grow, bonds stabilize during crashes  
✅ **Ultra-low cost** — 0.03%–0.12% expense ratios vs. 1%+ for active funds  
✅ **Fully liquid** — all ETFs can be sold within one trading day

### 20-Year Projection (Assuming 8% avg. annual return):
- **$10,000 with no additional contributions** → ~$46,610
- **$10,000 + $200/month (DCA)** → ~$106,840

### Action Plan:
1. Open a brokerage account (Fidelity, Schwab — both $0 minimum, $0 commissions)
2. Buy the ETFs above in the specified proportions
3. Enable automatic monthly contributions (even $50/month matters)
4. Rebalance back to target percentages once per year

⚠️ *This is educational information only, not personalized financial advice. Past performance doesn't guarantee future results. Consult a licensed financial advisor (CFP/CFA) before making investment decisions.*""",
    },
    {
        "input": "Roth IRA vs Traditional IRA — which is better?",
        "output": """## Roth IRA vs. Traditional IRA — The Definitive Comparison

The core difference: **when** you pay taxes.

### Side-by-Side Breakdown

| Feature | Traditional IRA | Roth IRA |
|---------|----------------|----------|
| **Tax on contributions** | Pre-tax (deductible) | After-tax (no deduction) |
| **Tax on growth** | Deferred | **Tax-free** |
| **Tax on withdrawals** | Taxed as ordinary income | **100% tax-free** |
| **2024 contribution limit** | $7,000 ($8,000 if age 50+) | $7,000 ($8,000 if age 50+) |
| **Income limit** | None (deduction phases out 73–83k single) | Phases out $146k–$161k (single) |
| **Required Min. Distributions** | Yes — start at age 73 | **None** |
| **Early withdrawal (before 59½)** | 10% penalty + taxes | Contributions: penalty-free; earnings: 10% |

### Decision Framework:

**Choose Traditional IRA if:**
- You're in a **high bracket now** (32%+) and expect lower income in retirement
- You need the **immediate tax deduction** to reduce this year's tax bill
- You're within 10–15 years of retirement

**Choose Roth IRA if:**
- You're **under 40** — decades of tax-free compounding is worth far more than today's deduction
- You're in the **22% bracket or below** — tax rates are historically low
- You want **no mandatory withdrawals** (great for estate planning)
- You want to withdraw contributions (not earnings) penalty-free in emergencies

### The Math (30-year horizon, 8% return, $7,000/year):
- **Ending balance**: ~$856,000
- **Roth take-home**: **$856,000 tax-free** 
- **Traditional take-home** (at 22% in retirement): ~$668,000

> **Bottom line**: For most people under 40 in the 22% bracket, the Roth wins by a significant margin.

### Next Steps:
1. Check your MAGI — if under $146k (single) or $230k (married), you qualify for Roth
2. Open a Roth IRA at Fidelity or Vanguard (no fees, no minimums)
3. Invest in a target-date fund or a simple 3-fund portfolio inside the account

⚠️ *This is educational information only, not personalized financial advice. Consult a licensed financial advisor (CFP/CFA) before making investment decisions.*""",
    },
]

_example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

_few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=_example_prompt,
    examples=_examples,
)

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("system", "── STYLE REFERENCE EXAMPLES — these are NOT from the current user's conversation ──"),
    _few_shot_prompt, # Few-shot examples of how to answer financial questions
    ("system", "── END OF EXAMPLES — the actual conversation with the current user starts below ──"),
    MessagesPlaceholder(variable_name="chat_history"),  # HumanMessage + AIMessage pairs from the current conversation
    ("system", "Relevant excerpts from the knowledge base — use these to ground your answer when applicable:\n{context}"),
    ("system", "{guardrail_note}"),
    ("human", "{input}"),
])

# ── Router prompt ────────────────────────────────────────────────────────────
# Sent ONLY to the router model (not the answer model).
# The router's sole job: decide which tool (if any) the user's message needs.
ROUTER_PROMPT = """You are a tool-routing classifier for a financial assistant.
Your only job is to decide which tool to call based on the user's message.
Never answer the question yourself — only select the right tool or call none.

## Tools
Use these tools when the user needs live data. Never answer live-data questions from memory.

| Tool | Use when the user asks about |
|------|------------------------------|
| `search_financial_web(query)` | Recent news, earnings, analyst ratings, IPOs, market events |
| `get_stock_data(ticker)` | Current price, P/E ratio, market cap for any stock (AAPL, TSLA, SPY…) |
| `get_crypto_price(coin_id)` | Live crypto prices — use CoinGecko IDs: bitcoin, ethereum, solana… |
| `calculate_compound_growth(principal, annual_rate_percent, years, monthly_contribution)` | Investment growth or retirement savings projections |
| `calculate_loan_payment(principal, annual_rate_percent, years)` | Monthly mortgage or loan payments |

**When to call:**
- Stock price, P/E, market cap → `get_stock_data` immediately
- Recent news, earnings, analyst upgrade → `search_financial_web` immediately
- Crypto price → `get_crypto_price` immediately
- Growth or payment calculation → matching calculator tool

**When NOT to call any tool:**
For greetings, general questions, educational topics, or anything answerable
from a knowledge base or financial expertise → call NO tools.
"""

# ── Guardrail non-advisory note ──────────────────────────────────────────────
# Injected into the prompt when a guardrail detects "unauthorized advice".
# Constrains the LLM to educational/insights mode instead of advisory mode.
NON_ADVISORY_NOTE = (
    "⚠️ GUARDRAIL ACTIVE — EDUCATIONAL MODE ENFORCED: "
    "The user's message was flagged as a request for personalized financial advice. "
    "You MUST respond with financial insights and education ONLY. "
    "Do NOT provide: specific ticker symbols to buy, exact dollar amounts to allocate, "
    "personalized step-by-step investment plans, or direct 'you should do X' statements. "
    "Instead: explain the relevant financial principles, discuss options objectively, "
    "highlight key considerations and trade-offs, and cite knowledge-base sources where applicable. "
    "End your response with the standard non-advisory disclaimer."
)
