"""
Response Cache — In-memory TTL cache for query responses.

Uses cachetools.TTLCache for automatic eviction and expiry.
Wraps it with normalization, stats tracking, and logging.

Cache key: normalized query string (lowercased, collapsed whitespace, no punctuation).
Cache value: (ChatResponse dict, cached_at timestamp).

Intentionally ephemeral — all entries lost on server restart.
"""

import re
import time
import logging
from copy import deepcopy
from typing import Optional

from cachetools import TTLCache

from api.models import ChatResponse

# --- Config ------------------------------------------------------------------
log = logging.getLogger(__name__)

CACHE_MAX_SIZE = 500        # Max entries before LRU eviction
CACHE_TTL_SECONDS = 3600    # 1 hour default
MIN_QUERY_WORDS = 3         # Skip cache for very short/vague queries

# --- Cache store -------------------------------------------------------------
_cache: TTLCache = TTLCache(maxsize=CACHE_MAX_SIZE, ttl=CACHE_TTL_SECONDS)

# --- Stats -------------------------------------------------------------------
_hits = 0
_misses = 0


# =============================================================================
#  Query Normalization
# =============================================================================

def normalize_query(query: str) -> str:
    """
    Normalize a query for cache key consistency.

    Steps:
      1. Lowercase
      2. Strip leading/trailing whitespace
      3. Collapse multiple spaces → single space
      4. Remove punctuation (keep word chars + spaces only)

    'Arduino PWM!' → 'arduino pwm'
    '  What   is  I2C?? ' → 'what is i2c'
    """
    query = query.lower().strip()
    query = re.sub(r"\s+", " ", query)
    query = re.sub(r"[^\w\s]", "", query)
    return query


def _is_cacheable_query(query: str) -> bool:
    """Check if a query is worth caching (not too short/vague)."""
    return len(query.split()) >= MIN_QUERY_WORDS


# =============================================================================
#  Cache Operations
# =============================================================================

def get(query: str) -> Optional[ChatResponse]:
    """
    Look up a cached response for the given query.

    Returns:
        A deep copy of the cached ChatResponse (with cache_hit=True,
        cached_at timestamp), or None on miss / short query.
    """
    global _hits, _misses

    normalized = normalize_query(query)

    if not _is_cacheable_query(normalized):
        log.info(f"[CACHE SKIP] Query too short ({len(normalized.split())} words): '{query}'")
        return None

    entry = _cache.get(normalized)

    if entry is not None:
        _hits += 1
        response_dict, cached_at = entry
        # Deep copy to avoid shared-state mutation across sessions
        response = ChatResponse(**deepcopy(response_dict))
        response.cache_hit = True
        response.cached_at = cached_at
        log.info(f"[CACHE HIT] '{query}' → normalized: '{normalized}'")
        return response

    _misses += 1
    log.info(f"[CACHE MISS] '{query}' → normalized: '{normalized}'")
    return None


def put(query: str, response: ChatResponse) -> None:
    """
    Store a response in cache keyed by the normalized query.

    Skips storage for short queries (< MIN_QUERY_WORDS words).
    """
    normalized = normalize_query(query)

    if not _is_cacheable_query(normalized):
        return

    # Store as dict + timestamp (avoid storing mutable Pydantic model directly)
    response_dict = response.model_dump()
    # Strip transient fields before caching
    response_dict.pop("cache_hit", None)
    response_dict.pop("cached_at", None)
    cached_at = time.time()

    _cache[normalized] = (response_dict, cached_at)
    log.info(
        f"[CACHE STORE] '{query}' → normalized: '{normalized}' | "
        f"cache_size={len(_cache)}/{CACHE_MAX_SIZE}"
    )


def clear() -> int:
    """Clear all cache entries. Returns count of entries removed."""
    count = len(_cache)
    _cache.clear()
    global _hits, _misses
    _hits = 0
    _misses = 0
    log.info(f"[CACHE CLEAR] Removed {count} entries, reset stats")
    return count


def stats() -> dict:
    """Return cache statistics for health/debug endpoints."""
    total = _hits + _misses
    return {
        "size": len(_cache),
        "max_size": CACHE_MAX_SIZE,
        "ttl_seconds": CACHE_TTL_SECONDS,
        "hits": _hits,
        "misses": _misses,
        "hit_rate": round(_hits / total, 3) if total > 0 else 0.0,
    }
