import os

import requests

_NTFY_URL = os.getenv("NTFY_URL", "https://ntfy.sh").rstrip("/")
_NTFY_TOPIC = os.getenv("NTFY_TOPIC", "")
_NTFY_TOKEN = os.getenv("NTFY_TOKEN", "")


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
    try:
        requests.post(f"{_NTFY_URL}/{_NTFY_TOPIC}", data=body.encode(), headers=headers, timeout=5)
    except Exception:
        pass
