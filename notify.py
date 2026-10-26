import os
from typing import Optional

import requests

_NTFY_URL = os.getenv("NTFY_URL", "https://ntfy.sh").rstrip("/")
_NTFY_TOPIC = os.getenv("NTFY_TOPIC", "")
_NTFY_TOKEN = os.getenv("NTFY_TOKEN", "")
_DISPATCH_URL = os.getenv("DISPATCH_URL", "")
# Push notifications are off unless NTFY_ENABLED=1 (and a topic is set); off also skips the per-run summary model call.
ENABLED = os.getenv("NTFY_ENABLED", "0") == "1" and bool(_NTFY_TOPIC)
# Flag degraded runs (⚠ title, reasons first, high priority) and notify on failed runs. 0 = off.
ALERT_PROBLEMS = os.getenv("NTFY_ON_PROBLEMS", "1") != "0"

_SUMMARY_PROMPT = (
    "Summarize the following research result in 2-3 plain sentences suitable for a mobile push notification. "
    "Be specific and informative — mention the most important findings. No markdown, no bullet points."
)


def summarize(client, result: str, session_id=None) -> str:
    import time
    import metrics
    from providers import FAST_MODEL, MODEL
    from pipeline import NUM_CTX
    model = FAST_MODEL or MODEL
    messages = [{"role": "user", "content": f"{_SUMMARY_PROMPT}\n\n{result[:4000]}"}]
    options = {"num_ctx": NUM_CTX, "num_predict": 120, "temperature": 0}
    rec = metrics.get(session_id)
    t0 = time.monotonic()
    try:
        resp = client.chat(
            model=model,
            messages=[
                {"role": "user", "content": f"{_SUMMARY_PROMPT}\n\n{result[:4000]}"},
            ],
            tools=[],
            temperature=0,
            max_tokens=120,
            num_ctx=NUM_CTX,  # same as the pipeline: a different num_ctx makes Ollama reload the model
        )
        text = (resp.message.content or "").strip()
        rec.llm("notify", sub=None, start=t0, model=model, options=options, messages=messages, output=text, usage=resp.usage)
        return text if text else result[:300]
    except Exception as e:
        rec.llm("notify", sub=None, start=t0, model=model, options=options, messages=messages, error=e)
        return result[:300]


def send(title: str, body: str, tags: list[str] = (), priority: str = "default") -> tuple[Optional[int], Optional[str]]:
    """Returns (HTTP status, error); (None, "disabled") unless NTFY_ENABLED=1 and a topic is set."""
    if not ENABLED:
        return None, "disabled"
    headers = {
        "Title": title,
        "Tags": ",".join(tags),
        "Priority": priority,
    }
    if _NTFY_TOKEN:
        headers["Authorization"] = f"Bearer {_NTFY_TOKEN}"
    if _DISPATCH_URL:
        headers["Click"] = _DISPATCH_URL
    try:
        r = requests.post(f"{_NTFY_URL}/{_NTFY_TOPIC}", data=body.encode(), headers=headers, timeout=5)
        return r.status_code, None if r.ok else r.text[:200]
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
