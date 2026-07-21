import os
import json
import requests as http_requests
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from dotenv import load_dotenv
from chatbot.chain import FinancialChatbot
from chatbot.guardrails import GuardrailError

load_dotenv()

app = Flask(__name__)

try:
    chatbot = FinancialChatbot()
    _init_error = None
except Exception as exc:
    chatbot = None
    _init_error = str(exc)


_MODEL_LABELS = {
    "llama": "Llama 3.3 70B Instruct",
    "ultra": "Nemotron Ultra 550B",
    "super": "Nemotron Super 120B",
}


@app.route("/")
def index():
    _key = os.environ.get("FINBOT_MODEL", "ultra").strip().lower()
    model_label = _MODEL_LABELS.get(_key, "Nemotron Ultra 550B")
    return render_template("index.html", init_error=_init_error, model_label=model_label)


@app.route("/health")
def health():
    return jsonify({"status": "ok" if chatbot else "error", "error": _init_error})


@app.route("/api/market")
def market_data():
    """Return live market data for the sidebar pulse and bottom ticker tape."""
    import yfinance as yf

    # ── Stocks & indices via yfinance ────────────────────────────────
    STOCK_SYMBOLS = {
        "S&P 500": "^GSPC",
        "NASDAQ":  "^IXIC",
        "DOW":     "^DJI",
        "Gold":    "GC=F",
        "NVDA":    "NVDA",
        "AAPL":    "AAPL",
        "TSLA":    "TSLA",
        "MSFT":    "MSFT",
        "META":    "META",
        "AMZN":    "AMZN",
        "GOOGL":   "GOOGL",
        "SPY":     "SPY",
        "QQQ":     "QQQ",
    }

    def fmt_price(p):
        if p is None: return "N/A"
        return f"{p:,.2f}" if p < 10000 else f"{p:,.0f}"

    stock_data = {}
    try:
        all_syms = list(STOCK_SYMBOLS.values())
        tickers  = yf.Tickers(" ".join(all_syms))
        for label, sym in STOCK_SYMBOLS.items():
            try:
                info  = tickers.tickers[sym].fast_info
                price = info.last_price
                prev  = info.previous_close
                chg   = ((price - prev) / prev * 100) if prev else 0
                stock_data[label] = {
                    "price": fmt_price(price),
                    "chg":   f"{chg:+.2f}%",
                    "dir":   "up" if chg >= 0 else "down",
                }
            except Exception:
                stock_data[label] = {"price": "N/A", "chg": "N/A", "dir": ""}
    except Exception:
        pass

    # ── Crypto via CoinGecko ─────────────────────────────────────────
    crypto_data = {}
    try:
        resp = http_requests.get(
            "https://api.coingecko.com/api/v3/simple/price"
            "?ids=bitcoin,ethereum&vs_currencies=usd&include_24hr_change=true",
            timeout=8,
        )
        cg = resp.json()
        for cid, label in [("bitcoin", "BTC/USD"), ("ethereum", "ETH")]:
            d     = cg.get(cid, {})
            price = d.get("usd", 0)
            chg   = d.get("usd_24h_change", 0)
            crypto_data[label] = {
                "price": f"${price:,.0f}",
                "chg":   f"{chg:+.2f}%",
                "dir":   "up" if chg >= 0 else "down",
            }
    except Exception:
        pass

    # ── Build response ───────────────────────────────────────────────
    pulse = []
    for sym in ["S&P 500", "NASDAQ", "DOW", "BTC/USD", "Gold"]:
        d = {**stock_data, **crypto_data}.get(sym, {"price": "N/A", "chg": "N/A", "dir": ""})
        pulse.append({"sym": sym, **d})

    tape = []
    for sym in ["NVDA", "AAPL", "TSLA", "BTC/USD", "ETH", "SPY", "QQQ", "MSFT", "META", "AMZN", "GOOGL", "Gold"]:
        d = {**stock_data, **crypto_data}.get(sym, {"price": "N/A", "chg": "N/A", "dir": ""})
        tape.append({"sym": sym, **d})

    return jsonify({"pulse": pulse, "tape": tape})


@app.route("/chat", methods=["POST"])
def chat():
    if chatbot is None:
        return jsonify({"error": _init_error, "blocked": True}), 503

    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()
    chat_history = data.get("history", [])

    if not user_message:
        return jsonify({"error": "Message cannot be empty"}), 400

    def generate():
        try:
            for chunk in chatbot.stream(user_message, chat_history):
                yield f"data: {json.dumps(chunk)}\n\n"
        except GuardrailError as exc:
            yield f"data: {json.dumps({'error': exc.message, 'blocked': True})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)
