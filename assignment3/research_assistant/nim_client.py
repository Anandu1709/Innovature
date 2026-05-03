import os
import time
from typing import Any, List, Optional

import httpx


def require_api_key() -> str:
    key = os.getenv("NVIDIA_API_KEY") or os.getenv("NIM_API_KEY")
    if not key:
        raise RuntimeError(
            "Set NVIDIA_API_KEY (or NIM_API_KEY) in your environment or .env file."
        )
    return key


# Retry on these statuses; everything else (auth, bad request, etc.) fails fast.
_RETRY_STATUSES = {408, 425, 429, 500, 502, 503, 504}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def chat_completion(
    messages: List[dict],
    *,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    model: Optional[str] = None,
) -> str:
    """
    Call NVIDIA NIM OpenAI-compatible chat completions.

    Tunable via env vars (all optional):
      NIM_TIMEOUT      request timeout in seconds, default 45
      NIM_MAX_RETRIES  number of retries on 5xx/429/timeouts, default 2
      NIM_BACKOFF      base backoff seconds between retries, default 1.5
    """
    base = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
    model = model or os.getenv("NIM_CHAT_MODEL", "meta/llama-3.1-8b-instruct")
    url = f"{base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {require_api_key()}",
        "Content-Type": "application/json",
    }
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    timeout = _env_float("NIM_TIMEOUT", 45.0)
    max_retries = _env_int("NIM_MAX_RETRIES", 2)
    base_backoff = _env_float("NIM_BACKOFF", 1.5)

    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                r = client.post(url, headers=headers, json=body)
            if r.status_code in _RETRY_STATUSES:
                last_exc = httpx.HTTPStatusError(
                    f"{r.status_code} from NIM",
                    request=r.request,
                    response=r,
                )
                if attempt < max_retries:
                    time.sleep(base_backoff * (2 ** attempt))
                    continue
                r.raise_for_status()
            r.raise_for_status()
            data = r.json()
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError(f"Empty choices from NIM: {data}")
            msg = choices[0].get("message") or {}
            content = msg.get("content")
            if content is None:
                raise RuntimeError(f"Missing message.content: {data}")
            return content.strip()
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_exc = exc
            if attempt < max_retries:
                time.sleep(base_backoff * (2 ** attempt))
                continue
            raise

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("NIM chat_completion failed without an exception (unreachable)")
