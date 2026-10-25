import os

import requests

_NTFY_URL = os.getenv("NTFY_URL", "https://ntfy.sh").rstrip("/")
_NTFY_TOPIC = os.getenv("NTFY_TOPIC", "")
_NTFY_TOKEN = os.getenv("NTFY_TOKEN", "")
_DISPATCH_URL = os.getenv("DISPATCH_URL", "")

_SUMMARY_PROMPT = (
    "Summarize the following research result in 2-3 plain sentences suitable for a mobile push notification. "
    "Be specific and informative — mention the most important findings. No markdown, no bullet points."
)


def summarize(client, result: str) -> str:
    from providers import FAST_MODEL, MODEL
    from pipeline import NUM_CTX
    model = FAST_MODEL or MODEL
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
        return text if text else result[:300]
    except Exception:
        return result[:300]


def send(title: str, body: str, tags: list[str] = (), priority: str = "default") -> None:
    if not _NTFY_TOPIC:
        return
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
        requests.post(f"{_NTFY_URL}/{_NTFY_TOPIC}", data=body.encode(), headers=headers, timeout=5)
    except Exception:
        pass
