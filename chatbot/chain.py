import os
import re
import time
import warnings
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from chatbot.prompts import PROMPT_TEMPLATE, NON_ADVISORY_NOTE
from chatbot.guardrails import guardrail_chain, GuardrailError
from chatbot.rag import retrieve
from chatbot.tools import FINBOT_TOOLS

# reasoning_budget and chat_template_kwargs are not declared Pydantic fields in
# ChatNVIDIA — the library auto-routes them into model_kwargs and warns about it.
# This is correct behaviour; suppress the noise.
warnings.filterwarnings(
    "ignore",
    message=r"WARNING! (reasoning_budget|chat_template_kwargs) is not default parameter",
    category=UserWarning,
)


# Nemotron models inject citation markers like 【{"id":0,"cursor":0,"loc":0}】
# when referencing tool/context content.  Strip them before sending to the UI.
_CITATION_RE = re.compile(r'\u3010[^\u3011]*\u3011')

def _clean(text: str) -> str:
    return _CITATION_RE.sub('', text)


# ── Model configurations ─────────────────────────────────────────────────────
# Set FINBOT_MODEL=llama / super / ultra in your .env.  Defaults to 'llama'.
_MODEL_CONFIGS = {
    "llama": {
        "model":       "meta/llama-3.3-70b-instruct",
        "timeout":     120,   # 60 s was too short when answer follows a tool call
        "thinking":    False,   # no reasoning overhead — fastest, best tool use
        "temperature": 0.2,
        "top_p":       0.7,
    },
    "super": {
        "model":       "nvidia/nemotron-3-super-120b-a12b",
        "timeout":     180,
        "thinking":    False,  # no reasoning overhead — fastest, best tool use
        "temperature": 0.5,
        "top_p":       0.95,
    },
    "ultra": {
        "model":       "nvidia/nemotron-3-ultra-550b-a55b",
        "timeout":     600,
        "thinking":    True,
        "temperature": 0.5,
        "top_p":       0.95,
    },
}


# ── Pipeline step functions ───────────────────────────────────────────────────

def _check_guardrails(data: dict) -> dict:
    """Run both safety guards in parallel. Raises GuardrailError if blocked.
    If 'unauthorized advice' is detected, injects NON_ADVISORY_NOTE into the
    data so the LLM responds in educational/insights mode only.
    """
    guards = guardrail_chain.invoke(data["input"])
    if not guards["nemotron_safety"].get("safe", True):
        raise GuardrailError("nemotron_safety", "Your message was flagged by our content safety guardrail.")
    if not guards["nemotron_guard"].get("safe", True):
        raise GuardrailError("nemotron_guard",  "Your message was flagged by our content safety guardrail.")

    # Detect "unauthorized advice" flag from either guard
    advice_flagged = (
        "unauthorized advice" in guards["nemotron_safety"].get("label", "").lower()
        or "unauthorized advice" in str(guards["nemotron_guard"].get("label", "")).lower()
    )
    guardrail_note = NON_ADVISORY_NOTE if advice_flagged else ""
    if advice_flagged:
        print("[Guardrail] Advisory flag detected — enforcing educational mode.")
    return {**data, "guardrail_note": guardrail_note}


def _format_history(data: dict) -> dict:
    """Convert raw chat_history dicts to LangChain HumanMessage / AIMessage types."""
    history = []
    for msg in data.get("chat_history", []):
        role = msg.get("role", "")
        if role == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif role == "assistant":
            history.append(AIMessage(content=msg["content"]))
    return {**data, "chat_history": history}


def _add_rag_context(data: dict) -> dict:
    """Retrieve knowledge-base chunks. Stores context (for LLM) and sources (for UI)."""
    if "context" not in data:
        context, sources = retrieve(data["input"])
        return {**data, "context": context, "_sources": sources}
    return data


class FinancialChatbot:
    def __init__(self):
        api_key = os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            raise ValueError("NVIDIA_API_KEY environment variable is not set.")

        _model_key = os.environ.get("FINBOT_MODEL", "llama").strip().lower()
        _cfg       = _MODEL_CONFIGS.get(_model_key, _MODEL_CONFIGS["llama"])
        _thinking  = _cfg["thinking"]
        print(f"[FinBot] model={_cfg['model']}  thinking={_thinking}  (FINBOT_MODEL={_model_key!r})")

        _base_cfg = dict(
            model=_cfg["model"],
            api_key=api_key,
            temperature=_cfg["temperature"],
            top_p=_cfg["top_p"],
            max_tokens=16384,
            timeout=_cfg["timeout"],
        )

        if _thinking:
            # Reasoning models: one instance per thinking-depth level
            _base_cfg["chat_template_kwargs"] = {"enable_thinking": True}
            self._models = {
                "fast":     ChatNVIDIA(**_base_cfg, reasoning_budget=1024),
                "balanced": ChatNVIDIA(**_base_cfg, reasoning_budget=4096),
                "deep":     ChatNVIDIA(**_base_cfg, reasoning_budget=16384),
            }
            self._router_model = ChatNVIDIA(**_base_cfg, reasoning_budget=512)
        else:
            # Non-thinking model (e.g. Llama): single instance for all levels
            _m = ChatNVIDIA(**_base_cfg)
            self._models = {"fast": _m, "balanced": _m, "deep": _m}
            self._router_model = _m

        self._thinking = _thinking
        self._tool_map = {t.name: t for t in FINBOT_TOOLS}
        self._agent_models = {
            level: model.bind_tools(FINBOT_TOOLS)
            for level, model in self._models.items()
        }
        self._router_agent = self._router_model.bind_tools(FINBOT_TOOLS)

    def stream(self, user_message: str, chat_history: list):
        """
        Full agent flow with live status events:
          ① Safety checks  (guardrails, parallel)
          ② History format + RAG retrieval
          ③ Tool routing   — fast dedicated router model
          ④ Tool execution (if requested) + answer generation
        GuardrailError propagates to app.py if blocked.
        """
        # ─ Step 1: guardrails ────────────────────────────────────────────────
        yield {"status": "🛡️ Checking safety…"}
        data = _check_guardrails({"input": user_message, "chat_history": chat_history})
        data = _format_history(data)

        # ─ Step 2: RAG retrieval ─────────────────────────────────────────────
        yield {"status": "📚 Searching knowledge base…"}
        data = _add_rag_context(data)

        context        = data.get("context", "")
        guardrail_note = data.get("guardrail_note", "")
        history        = data.get("chat_history", [])
        sources        = data.get("_sources", [])
        yield {"sources": sources}

        # ─ Step 3: assemble prompt ───────────────────────────────────────────
        messages = PROMPT_TEMPLATE.invoke({
            "input":          user_message,
            "chat_history":   history,
            "context":        context,
            "guardrail_note": guardrail_note,
        }).to_messages()

        # ─ Step 4: tool routing (dedicated fast router) ──────────────────────
        _model = self._models["balanced"]  # answer model (4 096-token budget for reasoning models)

        # Inject a routing guardrail so the router only calls tools when the
        # query genuinely requires one.  Without this, the router defaults to
        # search_financial_web for greetings, general questions, etc.
        _routing_hint = (
            "Call a tool ONLY when the user explicitly needs one:\n"
            "- Live stock price, P/E, or market cap → get_stock_data\n"
            "- Live crypto price → get_crypto_price\n"
            "- Recent news, earnings, or analyst upgrade/downgrade → search_financial_web\n"
            "- Compound growth or loan payment calculation → calculator tools\n"
            "For greetings, general questions, educational topics, or anything "
            "answerable from the knowledge base above → call NO tools."
        )
        if context:
            _routing_hint += (
                "\nThe knowledge base already has relevant information for this "
                "query — do NOT call search_financial_web."
            )
        messages.append(SystemMessage(content=_routing_hint))

        yield {"status": "Routing…"}
        t0 = time.time()
        ai_msg = None
        for _chunk in self._router_agent.stream(messages):
            ai_msg = _chunk if ai_msg is None else ai_msg + _chunk
        t_route = time.time() - t0
        tool_calls = getattr(ai_msg, "tool_calls", [])
        print(f"[Router] {t_route:.1f}s │ tool_calls={len(tool_calls)} │ "
              f"names={[tc['name'] for tc in tool_calls]}")

        # ─ Step 5: tool execution (if router decided to call tools) ──────────
        if tool_calls:
            tool_results = []
            for tc in tool_calls:
                name = tc["name"]
                args = tc["args"]
                yield {"tool_call": name}
                print(f"[Tool ] calling {name}  args={args}")
                t0 = time.time()
                tool_fn = self._tool_map.get(name)
                result  = tool_fn.invoke(args) if tool_fn else "Tool not found."
                print(f"[Tool ] {name} done  {time.time()-t0:.1f}s │ result_len={len(result)}")
                tool_results.append(f"[{name} results]\n{result}")
            yield {"tool_done": True}
            # Inject tool results as a system message.
            # Do NOT append ai_msg — Nemotron does not handle ToolMessage format.
            messages.append(SystemMessage(content="\n\n".join(tool_results)))
            yield {"status": "📝 Composing answer…"}
        else:
            yield {"status": "Thinking…"}

        # ─ Step 6: answer generation (always uses user’s chosen budget) ─────
        # The system prompt tells the model it has tools, but _model has none
        # bound.  A neutral generation trigger prevents the 0.5 s empty-response
        # bug WITHOUT mentioning tools — any "no tools" phrasing gets echoed into
        # chat history and poisons the router on the very next turn.
        if not tool_calls:
            messages.append(SystemMessage(
                content=(
                    "Write your complete, expert financial response to the user's "
                    "question now. Draw on the knowledge base excerpts above and "
                    "your financial expertise where relevant."
                )
            ))

        t0 = time.time()
        total_chunks = reasoning_chunks = content_chunks = 0
        try:
            for chunk in _model.stream(messages):
                total_chunks += 1
                out = {}
                rc = (chunk.additional_kwargs or {}).get("reasoning_content", "")
                if rc:
                    out["reasoning"] = rc
                    reasoning_chunks += 1
                if chunk.content:
                    cleaned = _clean(chunk.content)
                    if cleaned:
                        out["content"] = cleaned
                        content_chunks += 1
                if out:
                    yield out
        except Exception as exc:
            print(f"[Answer] ERROR {type(exc).__name__}: {exc}")
            yield {"content": f"\n\n⚠️ *Generation error — {exc}*"}
        print(
            f"[Answer] {time.time()-t0:.1f}s │ "
            f"total={total_chunks} reasoning={reasoning_chunks} content={content_chunks}"
        )
        if total_chunks > 0 and content_chunks == 0:
            print("[Answer] WARNING: model returned chunks but zero content — "
                  "only reasoning was produced")
