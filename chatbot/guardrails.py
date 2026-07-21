import os
import json
import warnings
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# Suppress "type is unknown" warnings for newer NVIDIA models not yet in the
# langchain_nvidia_ai_endpoints registry — inference works fine regardless.
warnings.filterwarnings(
    "ignore",
    message=".*type is unknown and inference may fail.*",
    category=UserWarning,
)


# Categories that are acceptable on FinBot (educational platform with disclaimers).
# Both models flag financial advice as "unauthorized advice" — this is expected
# behaviour and must not block legitimate investment questions on this platform.
_ALLOWED_CATEGORIES = {"unauthorized advice"}


class GuardrailError(Exception):
    """Raised by the pipeline when a message is blocked by a guardrail."""
    def __init__(self, guardrail: str, message: str):
        self.guardrail = guardrail
        self.message   = message
        super().__init__(message)


# ── Guard 1 : Nemotron 3.5 Content Safety ──────────────────────────────────────
def _nemotron_safety_call(user_input: str) -> dict:
    """Nemotron 3.5 Content Safety — multimodal/multilingual input safety check."""
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        return {"safe": True, "error": "API key not configured"}

    client = ChatNVIDIA(
        model="nvidia/nemotron-3.5-content-safety",
        api_key=api_key,
        temperature=0.2,
        top_p=0.7,
        max_tokens=512,
    )
    try:
        response = client.invoke(
            [{"role": "user", "content": user_input}],
            chat_template_kwargs={"request_categories": "/categories", "enable_thinking": True},
        )
        label = response.content.strip().lower()
        # Actual output format: "user safety: safe" or
        # "user safety: unsafe\nsafety categories: <cat1, cat2>"
        if "user safety: unsafe" not in label:
            safe = True
        else:
            categories = set()
            for line in label.split("\n"):
                if "safety categories:" in line:
                    cats = line.split("safety categories:")[-1].strip()
                    categories = {c.strip() for c in cats.split(",")}
            safe = categories.issubset(_ALLOWED_CATEGORIES)
        print(f"[Nemotron-Safety] safe={safe}  label={label!r}")
        return {"safe": safe, "label": label}
    except Exception as exc:
        print(f"[Nemotron-Safety] error={exc} — failing open")
        return {"safe": True, "error": str(exc)}


# ── Guard 2 : Llama 3.1 Nemotron Safety Guard 8B ─────────────────────────────
def _nemotron_guard_call(user_input: str) -> dict:
    """Llama 3.1 Nemotron Safety Guard — 23-category multilingual safety classifier."""
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        return {"safe": True, "error": "API key not configured"}

    client = ChatNVIDIA(
        model="nvidia/llama-3.1-nemotron-safety-guard-8b-v3",
        api_key=api_key,
    )
    try:
        response = client.invoke([{"role": "user", "content": user_input}])
        content  = response.content.strip()
        print(f"[Nemotron-Guard]  raw={content!r}")
        # Normalise keys to lowercase to handle title-case responses
        # e.g. {"User Safety": "unsafe", "Safety Categories": "..."}
        try:
            raw_data    = json.loads(content)
            data        = {k.lower(): v for k, v in raw_data.items()}
            user_safety = data.get("user safety", "safe").lower()
            categories  = {c.strip().lower() for c in data.get("safety categories", "").split(",") if c.strip()}
        except (json.JSONDecodeError, AttributeError):
            user_safety = "unsafe" if "unsafe" in content.lower() else "safe"
            categories  = set()

        if user_safety != "unsafe":
            safe = True
        else:
            safe = categories.issubset(_ALLOWED_CATEGORIES)

        print(f"[Nemotron-Guard]  safe={safe}  categories={categories}")
        return {"safe": safe, "label": content}
    except Exception as exc:
        print(f"[Nemotron-Guard]  error={exc} — failing open")
        return {"safe": True, "error": str(exc)}


# ── LCEL guardrail chain ────────────────────────────────────────────────────────
# RunnableParallel runs both guards concurrently on the same input string.
# Result: {"nemotron_safety": {...}, "nemotron_guard": {...}}
guardrail_chain = RunnableParallel(
    nemotron_safety=RunnableLambda(_nemotron_safety_call),
    nemotron_guard=RunnableLambda(_nemotron_guard_call),
)
