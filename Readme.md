# FinBot: Guarded AI Financial Advisor

A full-stack AI financial chatbot built on NVIDIA's API with dual safety guardrails, RAG over institutional research documents, live market tools, and a streaming chat interface.

---

## Features

| Layer               | What it does                                                                                                                              |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **Dual guardrails** | Every message is checked in parallel by `nemotron-3.5-content-safety` and `llama-3.1-nemotron-safety-guard-8b-v3` before reaching the LLM |
| **Advisory mode**   | Financial advice requests are detected and automatically downgraded to educational-only responses                                         |
| **RAG**             | ChromaDB vector store over institutional PDF reports; cosine-similarity retrieval with a 0.3 score threshold                              |
| **Tool use**        | 5 live tools — web search, stock data, crypto prices, compound-growth calculator, loan-payment calculator                                 |
| **Model choice**    | Switch between three NVIDIA models via one `.env` variable                                                                                |
| **Streaming UI**    | Real-time SSE streaming with chain-of-thought reasoning display, tool-call badges, and source attribution                                 |

---

## Architecture

```
Browser  ──SSE──►  Flask (app.py)
                      │
            ┌─────────▼──────────┐
            │  chatbot/chain.py  │
            │                    │
            │  ① Guardrails      │  ← parallel: nemotron-safety + nemotron-guard
            │  ② RAG retrieval   │  ← ChromaDB + NVIDIA embeddings
            │  ③ Tool router     │  ← LLM with tools bound (decides: tool or direct)
            │  ④ Tool execution  │  ← yfinance / CoinGecko / Tavily / calculators
            │  ⑤ Answer stream   │  ← LLM without tools (streams final response)
            └────────────────────┘
```

---

## Models

Set `FINBOT_MODEL` in `.env` to choose:

| Value                 | Model                               | Speed    | Notes                                           |
| --------------------- | ----------------------------------- | -------- | ----------------------------------------------- |
| `llama` **(default)** | `meta/llama-3.3-70b-instruct`       | ~5–15 s  | No thinking overhead, best tool use reliability |
| `super`               | `nvidia/nemotron-3-super-120b-a12b` | ~30–90 s | 12B active params, chain-of-thought reasoning   |
| `ultra`               | `nvidia/nemotron-3-ultra-550b-a55b` | ~1–5 min | 55B active params, highest quality              |

---

## Knowledge Base

PDFs in `knowledge_base/` are pre-chunked and embedded into ChromaDB via `ingest.py`:

```
knowledge_base/
├── BlackRock/          BII Mid-Year Outlook 2026
├── Federal Reserve/    Beige Book — June 2026
└── Vanguard/           Retirement Income · Global Equity · Inflation Hedging · Asset Location
```

---

## Tools

| Tool                        | Data source               | Trigger                                |
| --------------------------- | ------------------------- | -------------------------------------- |
| `search_financial_web`      | Tavily Search API         | Recent news, earnings, analyst ratings |
| `get_stock_data`            | yfinance                  | Live stock price, P/E, market cap      |
| `get_crypto_price`          | CoinGecko (no key needed) | Live crypto prices                     |
| `calculate_compound_growth` | Local calculation         | Investment / retirement projections    |
| `calculate_loan_payment`    | Local calculation         | Monthly mortgage or loan payments      |

---

## Setup

### 1. Clone and install

```bash
git clone <repo>
cd Code
pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env.example` to `.env` and fill in your keys:

```env
NVIDIA_API_KEY=nvapi-...
TAVILY_API_KEY=tvly-...
FINBOT_MODEL=llama          # llama | super | ultra
```

Get your NVIDIA API key free at [build.nvidia.com](https://build.nvidia.com).  
Get your Tavily key at [tavily.com](https://tavily.com).

### 3. Ingest the knowledge base

Run once (or whenever you add new PDFs to `knowledge_base/`):

```bash
python ingest.py
```

This embeds all PDFs using `nvidia/llama-nemotron-embed-1b-v2` and writes the vector store to `chroma_db/`.

### 4. Run

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000).

---

## Project Structure

```
Code/
├── app.py                  Flask application + SSE streaming endpoint
├── ingest.py               One-time PDF ingestion script
├── requirements.txt
├── .env                    API keys and model selection (not committed)
├── .env.example            Template for .env
│
├── chatbot/
│   ├── chain.py            Main pipeline: guardrails → RAG → routing → tools → answer
│   ├── guardrails.py       Parallel dual-guardrail checks
│   ├── prompts.py          System prompt, few-shot examples, PROMPT_TEMPLATE
│   ├── rag.py              ChromaDB singleton retriever
│   └── tools.py            LangChain @tool definitions
│
├── knowledge_base/         Source PDFs (organised by institution)
├── chroma_db/              Generated vector store (after running ingest.py)
│
├── templates/
│   └── index.html          Single-page chat UI
└── static/
    ├── css/style.css
    └── js/chat.js          SSE streaming, markdown rendering, tool-call UI
```

---

## Guardrails

Two safety models run **in parallel** on every user message before anything reaches the main LLM:

| Guard          | Model                                          | Catches                                                   |
| -------------- | ---------------------------------------------- | --------------------------------------------------------- |
| Content Safety | `nvidia/nemotron-3.5-content-safety`           | Violence, hate speech, illegal activity, prompt injection |
| Safety Guard   | `nvidia/llama-3.1-nemotron-safety-guard-8b-v3` | 23-category multilingual safety classification            |

**Advisory mode**: if either guard flags "unauthorized advice", the request passes through but the LLM is constrained to educational/informational responses only (no personalised recommendations, no specific tickers to buy).

---

## Adding Documents

1. Place PDF files in the appropriate subfolder under `knowledge_base/` (create a new subfolder if needed)
2. Re-run `python ingest.py`

The ingest script handles chunking (1,000 chars / 200 overlap), embedding, and upsert into ChromaDB automatically.

---

## Testing Guardrails

`trial_prompts.txt` contains a full QA suite covering:

- **Section A** — Content safety (jailbreaks, injection, role-play attacks)
- **Section B** — Harmful content (fraud, manipulation, money laundering)
- **Section C** — Chained attacks
- **Section D** — Advisory mode enforcement

---

## License

For educational and non-commercial use. Model licenses apply per the respective NVIDIA and Meta model cards.
"# FinBot" 
