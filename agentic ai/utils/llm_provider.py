"""
Centralized LLM Provider — Single source of truth for all Gemini API access.

Production-grade infrastructure layer providing:
  1. Single configuration point (model name via env var)
  2. Token bucket rate limiter (enforces RPM ceiling)
  3. Singleton LLM instances (reuses gRPC connections)
  4. Response cache (skips duplicate API calls)
  5. Request observability (counters + logging)

Usage:
  from utils.llm_provider import invoke_with_rate_limit, get_stats

  # Simple call — rate limited, cached, observable
  response = invoke_with_rate_limit(
      messages=[
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "What is SPI?"},
      ],
      temperature=0.3,
  )

  # Check stats
  print(get_stats())

Environment Variables:
  GOOGLE_API_KEY       — Required. Your Gemini API key.
  GEMINI_MODEL         — Optional. Model name. Default: "gemini-flash-latest"
  GEMINI_RPM_LIMIT     — Optional. Requests per minute. Default: 14
  LLM_CACHE_ENABLED    — Optional. "true"/"false". Default: "true"
"""

import os
import time
import hashlib
import logging
import threading
from pathlib import Path
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

# --- Config ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Single configuration point — change model here or via environment variable
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
RPM_LIMIT = int(os.getenv("GEMINI_RPM_LIMIT", "14"))
CACHE_ENABLED = os.getenv("LLM_CACHE_ENABLED", "true").lower() == "true"

log.info(
    f"[LLM] Provider initialized — model={MODEL_NAME}, "
    f"rpm_limit={RPM_LIMIT}, cache={'ON' if CACHE_ENABLED else 'OFF'}"
)


# =============================================================================
#  Token Bucket Rate Limiter
# =============================================================================

class TokenBucketRateLimiter:
    """
    Thread-safe token bucket rate limiter.

    Smoothly distributes requests across the minute window.
    If all tokens are consumed, the caller sleeps until a token
    is refilled, guaranteeing we never exceed RPM.
    """

    def __init__(self, rpm: int):
        self.rpm = rpm
        self.tokens = float(rpm)
        self.last_refill = time.monotonic()
        self.lock = threading.Lock()
        self._total_requests = 0
        self._total_waits = 0

    def acquire(self):
        """Block until a token is available, then consume it."""
        while True:
            with self.lock:
                self._refill()
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    self._total_requests += 1
                    return

                # Calculate how long until the next token arrives
                wait_time = (1.0 - self.tokens) * (60.0 / self.rpm)
                self._total_waits += 1

            log.info(
                f"[LLM] Rate limit reached — waiting {wait_time:.1f}s "
                f"(requests so far: {self._total_requests})"
            )
            time.sleep(wait_time)

    def _refill(self):
        """Refill tokens based on elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - self.last_refill

        # Add tokens proportional to elapsed time
        new_tokens = elapsed * (self.rpm / 60.0)
        if new_tokens > 0:
            self.tokens = min(float(self.rpm), self.tokens + new_tokens)
            self.last_refill = now

    @property
    def stats(self) -> dict:
        """Return current rate limiter statistics."""
        return {
            "total_requests": self._total_requests,
            "total_waits": self._total_waits,
            "tokens_available": round(self.tokens, 1),
        }


# =============================================================================
#  Response Cache
# =============================================================================

class ResponseCache:
    """
    Thread-safe in-memory LRU cache for LLM responses.

    Cache key = hash(messages + temperature).
    Particularly effective for the context router, which often
    classifies similar queries identically.
    """

    def __init__(self, enabled: bool = True, max_size: int = 100):
        self.enabled = enabled
        self._cache: dict[str, str] = {}
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

    @staticmethod
    def _make_key(messages: list, temperature: float, model_name: str = "") -> str:
        """Create a deterministic cache key from messages, temperature, and model."""
        raw = f"{model_name}:{temperature}:{str(messages)}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, messages: list, temperature: float, model_name: str = "", override_key: str = None) -> str | None:
        """Retrieve a cached response, or None on miss."""
        if not self.enabled:
            return None

        key = override_key or self._make_key(messages, temperature, model_name)
        with self._lock:
            if key in self._cache:
                self._hits += 1
                log.info(f"[LLM] Cache HIT (key={key[:8]}… total hits: {self._hits})")
                return self._cache[key]
            self._misses += 1
            log.info(f"[LLM] Cache MISS (key={key[:8]}… total misses: {self._misses})")
            return None

    def put(self, messages: list, temperature: float, response: str, model_name: str = "", override_key: str = None):
        """Store a response in the cache."""
        if not self.enabled:
            return

        key = override_key or self._make_key(messages, temperature, model_name)
        with self._lock:
            # Simple LRU: evict oldest when full
            if len(self._cache) >= self._max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            self._cache[key] = response

    @property
    def stats(self) -> dict:
        """Return current cache statistics."""
        total = self._hits + self._misses
        return {
            "cache_hits": self._hits,
            "cache_misses": self._misses,
            "cache_size": len(self._cache),
            "hit_rate": f"{self._hits / max(1, total) * 100:.1f}%",
        }


# =============================================================================
#  Singleton LLM Instances
# =============================================================================

_rate_limiter = TokenBucketRateLimiter(rpm=RPM_LIMIT)
_response_cache = ResponseCache(enabled=CACHE_ENABLED)
_llm_instances: dict[tuple, ChatGoogleGenerativeAI] = {}
_llm_lock = threading.Lock()


def get_llm(temperature: float = 0.3, model_name: str = None) -> ChatGoogleGenerativeAI:
    """
    Get a singleton LLM instance for the given temperature and model.

    Reuses gRPC connections by caching instances keyed on
    (model_name, temperature). Thread-safe.

    Args:
        temperature: Model temperature.
        model_name:  Optional model override. Defaults to global MODEL_NAME.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        raise ValueError(
            "GOOGLE_API_KEY not set. Update your .env file with a valid Gemini API key."
        )

    effective_model = model_name or MODEL_NAME
    cache_key = (effective_model, temperature)

    with _llm_lock:
        if cache_key not in _llm_instances:
            log.info(
                f"[LLM] Creating singleton instance: "
                f"model={effective_model}, temperature={temperature}"
            )
            _llm_instances[cache_key] = ChatGoogleGenerativeAI(
                model=effective_model,
                google_api_key=api_key,
                temperature=temperature,
            )

    return _llm_instances[cache_key]


# =============================================================================
#  Public API — Rate-Limited Invocation
# =============================================================================

def invoke_with_rate_limit(
    messages: list[dict],
    temperature: float = 0.3,
    use_cache: bool = True,
    model_name: str = None,
    cache_key: str = None,
) -> str:
    """
    Invoke the LLM with automatic rate limiting and optional caching.

    This is the ONLY function agents should call to interact with Gemini.

    Args:
        messages:    List of message dicts [{"role": "...", "content": "..."}].
        temperature: Model temperature (0.0 = deterministic, 0.3 = creative).
        use_cache:   Whether to check/populate the response cache.
                     Set False for verification calls where freshness matters.
        model_name:  Optional Gemini model override (e.g. "gemini-2.5-flash-lite").
                     Defaults to the global MODEL_NAME from .env.
        cache_key:   Optional custom cache key string. When provided, the cache
                     is keyed on this value (+ model + temperature) instead of
                     the full messages list. Useful for agents where prompts
                     vary (e.g. session history) but the core query is the same.

    Returns:
        The response content string (stripped).
    """
    effective_model = model_name or MODEL_NAME

    # Build override cache key if a custom cache_key was provided
    _override = None
    if cache_key is not None:
        raw = f"{effective_model}:{temperature}:{cache_key}"
        _override = hashlib.md5(raw.encode()).hexdigest()

    # 1. Check cache
    if use_cache:
        cached = _response_cache.get(messages, temperature, effective_model, override_key=_override)
        if cached is not None:
            return cached

    # 2. Acquire rate limit token (blocks if necessary)
    _rate_limiter.acquire()

    # 3. Invoke LLM
    llm = get_llm(temperature=temperature, model_name=effective_model)
    response = llm.invoke(messages)
    content = response.content.strip()

    # 4. Cache response
    if use_cache:
        _response_cache.put(messages, temperature, content, effective_model, override_key=_override)

    return content


# =============================================================================
#  Observability
# =============================================================================

def get_stats() -> dict:
    """
    Get combined stats from rate limiter and response cache.

    Useful for debugging, cost monitoring, and performance tuning.
    """
    return {
        "model": MODEL_NAME,
        "rpm_limit": RPM_LIMIT,
        **_rate_limiter.stats,
        **_response_cache.stats,
    }


def reset_stats():
    """
    Reset all counters and cache.

    Call this between test runs to get clean measurements.
    """
    global _rate_limiter, _response_cache
    _rate_limiter = TokenBucketRateLimiter(rpm=RPM_LIMIT)
    _response_cache = ResponseCache(enabled=CACHE_ENABLED)
    log.info("[LLM] Stats and cache reset")


# ─── Standalone Test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  LLM PROVIDER — Standalone Test")
    print(f"  Model:     {MODEL_NAME}")
    print(f"  RPM Limit: {RPM_LIMIT}")
    print(f"  Cache:     {'ON' if CACHE_ENABLED else 'OFF'}")
    print("=" * 60)

    # Test 1: Basic invocation
    print("\n  Test 1: Basic invocation")
    response = invoke_with_rate_limit(
        messages=[
            {"role": "system", "content": "Reply in exactly one word."},
            {"role": "user", "content": "What is 2+2?"},
        ],
        temperature=0.0,
    )
    print(f"  Response: {response}")

    # Test 2: Cache hit
    print("\n  Test 2: Cache hit (same messages)")
    response2 = invoke_with_rate_limit(
        messages=[
            {"role": "system", "content": "Reply in exactly one word."},
            {"role": "user", "content": "What is 2+2?"},
        ],
        temperature=0.0,
    )
    print(f"  Response: {response2}")

    # Test 3: Stats
    print(f"\n  Test 3: Provider stats")
    stats = get_stats()
    for k, v in stats.items():
        print(f"    {k}: {v}")

    print("\n" + "=" * 60 + "\n")
