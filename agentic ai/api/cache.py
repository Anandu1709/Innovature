"""
Response Cache — In-memory TTL cache for query responses.

Uses cachetools.TTLCache for automatic eviction and expiry.
Wraps it with normalization, stats tracking, and logging.

Cache key: normalized query string + KB version suffix.
Cache value: (ChatResponse dict, cached_at timestamp).

KB Versioning:
  A monotonic counter (_kb_version) is incremented after every
  knowledge base mutation (PDF upload, URL ingestion, markdown paste,
  document deletion). Cache keys include the version suffix, so all
  prior entries become unreachable after a version bump. Stale orphans
  are naturally evicted by TTL/LRU.

Follow-up Detection:
  is_standalone_query() distinguishes self-contained questions
  (cacheable) from context-dependent follow-ups (not cacheable)
  using pronoun indicators and regex patterns.

Intentionally ephemeral — all entries lost on server restart.
"""

import re
import time
import logging
import threading
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
#  KB Version Tracking
# =============================================================================

_kb_version: int = 0
_kb_version_lock = threading.Lock()


def bump_kb_version() -> int:
    """
    Increment KB version after any knowledge base mutation.

    Call after: PDF upload, URL ingestion, markdown ingestion, document deletion.
    All prior cached responses become stale (unreachable by new keys).
    """
    global _kb_version
    with _kb_version_lock:
        _kb_version += 1
        log.info(
            f"[CACHE] KB version bumped to {_kb_version} — "
            f"all prior cached responses are now stale"
        )
        return _kb_version


def get_kb_version() -> int:
    """Return current KB version."""
    return _kb_version


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


def _make_cache_key(query: str) -> str:
    """Build a versioned, normalized cache key."""
    normalized = normalize_query(query)
    return f"{normalized}|v{_kb_version}"


def _is_cacheable_query(query: str) -> bool:
    """Check if a query is worth caching (not too short/vague)."""
    return len(query.split()) >= MIN_QUERY_WORDS


# =============================================================================
#  Follow-up Detection
# =============================================================================

# Pronouns and references that signal context-dependency.
# Intentionally conservative — excludes "which", "how", "what" because
# "Which GPIO pin supports PWM?" and "How does UART work?" are cacheable.
FOLLOWUP_PRONOUNS = {
    "it", "its", "that", "this", "them", "those",
    "previous", "earlier",
}

# Short queries / patterns that are almost always follow-ups
FOLLOWUP_PATTERNS = [
    r"^why\??$",                  # bare "why?"
    r"^how\??$",                  # bare "how?"
    r"^what about\b",             # "what about that?"
    r"^compare it\b",             # "compare it with..."
    r"^compare them\b",           # "compare them..."
    r"^and\s",                    # "and what about..."
    r"^but\s",                    # "but why..."
    r"^(yes|no|ok|okay|sure)\b",  # conversational acks
]


def is_standalone_query(query: str) -> bool:
    """
    Detect whether a query is self-contained (cacheable) vs context-dependent.

    Rules:
      1. Must have >= MIN_QUERY_WORDS words.
      2. Must NOT match a follow-up regex pattern.
      3. Must NOT have > 40% of words be follow-up pronouns.

    Examples:
      "What is UART?"                   → True  (standalone)
      "Which GPIO pin supports PWM?"    → True  (standalone)
      "How does UART work?"             → True  (standalone)
      "Why?"                            → False (too short + pattern)
      "What about that?"                → False (pattern match)
      "Compare it with the previous"    → False (pattern + pronouns)
    """
    normalized = normalize_query(query)
    words = normalized.split()

    # Too short to be meaningful
    if len(words) < MIN_QUERY_WORDS:
        return False

    # Check regex patterns
    for pattern in FOLLOWUP_PATTERNS:
        if re.match(pattern, normalized):
            return False

    # First word is a pronoun → almost always context-dependent
    # Catches: "it does not work", "that board is broken", "this is wrong"
    if words[0] in FOLLOWUP_PRONOUNS:
        return False

    # High ratio of follow-up pronouns = context-dependent
    followup_count = sum(1 for w in words if w in FOLLOWUP_PRONOUNS)
    if followup_count / len(words) > 0.4:
        return False

    return True


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

    key = _make_cache_key(query)
    entry = _cache.get(key)

    if entry is not None:
        _hits += 1
        response_dict, cached_at = entry
        # Deep copy to avoid shared-state mutation across sessions
        response = ChatResponse(**deepcopy(response_dict))
        response.cache_hit = True
        response.cached_at = cached_at
        log.info(f"[CACHE HIT] '{query}' → key: '{key}'")
        return response

    _misses += 1
    log.info(f"[CACHE MISS] '{query}' → key: '{key}'")
    return None


def put(query: str, response: ChatResponse) -> None:
    """
    Store a response in cache keyed by the versioned normalized query.

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

    key = _make_cache_key(query)
    _cache[key] = (response_dict, cached_at)
    log.info(
        f"[CACHE STORE] '{query}' → key: '{key}' | "
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
        "kb_version": _kb_version,
    }
