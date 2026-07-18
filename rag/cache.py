"""
Redis caching layer for the RAG system.

Provides:
- Query result caching (avoid re-running identical queries)
- Embedding caching (avoid re-embedding identical text)
- Session state for chat history
"""

import hashlib
import json
import os

import redis

# Default configuration
DEFAULT_REDIS_CONFIG = {
    "host": "localhost",
    "port": 6379,
    "db": 0,
    "decode_responses": True,
}

# Cache TTL defaults (in seconds)
QUERY_CACHE_TTL = 3600  # 1 hour for query results
EMBEDDING_CACHE_TTL = 86400  # 24 hours for embeddings
CHAT_HISTORY_TTL = 7200  # 2 hours for chat sessions

# Key prefixes
PREFIX_QUERY = "olmocr:query:"
PREFIX_EMBEDDING = "olmocr:emb:"
PREFIX_CHAT = "olmocr:chat:"
PREFIX_STATS = "olmocr:stats:"


def get_redis_config():
    """Get Redis configuration from environment or defaults."""
    return {
        "host": os.environ.get("OLMOCR_REDIS_HOST", DEFAULT_REDIS_CONFIG["host"]),
        "port": int(os.environ.get("OLMOCR_REDIS_PORT", DEFAULT_REDIS_CONFIG["port"])),
        "db": int(os.environ.get("OLMOCR_REDIS_DB", DEFAULT_REDIS_CONFIG["db"])),
        "decode_responses": True,
    }


_client = None


def get_client(config=None):
    """Get or create a Redis client (singleton)."""
    global _client
    if _client is None:
        if config is None:
            config = get_redis_config()
        _client = redis.Redis(**config)
    return _client


def reset_client():
    """Reset the singleton client (for testing)."""
    global _client
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass
    _client = None


def is_healthy():
    """Check if Redis is reachable."""
    try:
        client = get_client()
        return client.ping()
    except Exception:
        return False


def _make_hash(text):
    """Create a stable hash key from text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


# ── Query caching ──────────────────────────────────────────────


def cache_query_result(query, result, ttl=QUERY_CACHE_TTL):
    """Cache a query result.

    Args:
        query: The user's query string.
        result: The result dict to cache (must be JSON-serializable).
        ttl: Time-to-live in seconds.
    """
    client = get_client()
    key = PREFIX_QUERY + _make_hash(query)
    client.set(key, json.dumps(result), ex=ttl)


def get_cached_query(query):
    """Retrieve a cached query result.

    Args:
        query: The user's query string.

    Returns:
        Cached result dict, or None if not found.
    """
    client = get_client()
    key = PREFIX_QUERY + _make_hash(query)
    result = client.get(key)
    if result:
        return json.loads(result)
    return None


def invalidate_embedding_cache(model_name: str = None):
    """Clear cached embeddings, optionally only for a specific model.

    Embeddings are model-specific, so switching the embedding model leaves
    stale vectors cached under the old model name. Calling this on a model
    switch (or with ``None`` to clear everything) prevents serving vectors
    produced by a different dimensionality than the active collection.
    """
    client = get_client()
    if model_name:
        # Embedding keys are shaped PREFIX_EMBEDDING + hash(f"{model_name}:{text}")
        # so we cannot prefix-scan by model directly; instead rewrite current
        # keys by re-hashing. Simplest correct behaviour: clear all embeddings.
        keys = client.keys(PREFIX_EMBEDDING + "*")
        if keys:
            client.delete(*keys)
    else:
        keys = client.keys(PREFIX_EMBEDDING + "*")
        if keys:
            client.delete(*keys)


def invalidate_query_cache():
    """Clear all cached query results."""
    client = get_client()
    keys = client.keys(PREFIX_QUERY + "*")
    if keys:
        client.delete(*keys)


def invalidate_all_caches():
    """Clear query, embedding, and chat caches (e.g. on embedding-model switch)."""
    client = get_client()
    for prefix in (PREFIX_QUERY, PREFIX_EMBEDDING, PREFIX_CHAT):
        keys = client.keys(prefix + "*")
        if keys:
            client.delete(*keys)


# ── Embedding caching ─────────────────────────────────────────


def cache_embedding(text, embedding, model_name, ttl=EMBEDDING_CACHE_TTL):
    """Cache a text embedding.

    Args:
        text: The source text.
        embedding: The embedding vector as a list of floats.
        model_name: The embedding model used.
        ttl: Time-to-live in seconds.
    """
    client = get_client()
    cache_key = _make_hash(f"{model_name}:{text}")
    key = PREFIX_EMBEDDING + cache_key
    # Store as compact JSON — embeddings can be large
    client.set(key, json.dumps(embedding), ex=ttl)


def get_cached_embedding(text, model_name):
    """Retrieve a cached embedding.

    Args:
        text: The source text.
        model_name: The embedding model used.

    Returns:
        Embedding vector as a list of floats, or None if not cached.
    """
    client = get_client()
    cache_key = _make_hash(f"{model_name}:{text}")
    key = PREFIX_EMBEDDING + cache_key
    result = client.get(key)
    if result:
        return json.loads(result)
    return None


# ── Chat session history ──────────────────────────────────────


def save_chat_history(session_id, messages, ttl=CHAT_HISTORY_TTL):
    """Save chat history for a session.

    Args:
        session_id: Unique session identifier.
        messages: List of message dicts (role, content).
        ttl: Time-to-live in seconds.
    """
    client = get_client()
    key = PREFIX_CHAT + session_id
    client.set(key, json.dumps(messages), ex=ttl)


def get_chat_history(session_id):
    """Retrieve chat history for a session.

    Args:
        session_id: Unique session identifier.

    Returns:
        List of message dicts, or empty list if not found.
    """
    client = get_client()
    key = PREFIX_CHAT + session_id
    result = client.get(key)
    if result:
        return json.loads(result)
    return []


def clear_chat_history(session_id):
    """Clear chat history for a session."""
    client = get_client()
    key = PREFIX_CHAT + session_id
    client.delete(key)


# ── Statistics tracking ───────────────────────────────────────


def increment_stat(stat_name, amount=1):
    """Increment a statistics counter.

    Args:
        stat_name: Name of the statistic (e.g., 'queries', 'chunks_indexed').
        amount: Amount to increment by.
    """
    client = get_client()
    key = PREFIX_STATS + stat_name
    client.incr(key, amount)


def get_stat(stat_name):
    """Get a statistics counter value.

    Args:
        stat_name: Name of the statistic.

    Returns:
        Integer count, or 0 if not found.
    """
    client = get_client()
    key = PREFIX_STATS + stat_name
    result = client.get(key)
    return int(result) if result else 0


def get_all_stats():
    """Get all statistics counters.

    Returns:
        Dict of stat_name -> count.
    """
    client = get_client()
    keys = client.keys(PREFIX_STATS + "*")
    stats = {}
    for key in keys:
        stat_name = key.replace(PREFIX_STATS, "")
        stats[stat_name] = int(client.get(key) or 0)
    return stats


def get_cache_info():
    """Get cache usage information.

    Returns:
        Dict with counts of cached queries, embeddings, and chat sessions.
    """
    client = get_client()
    return {
        "cached_queries": len(client.keys(PREFIX_QUERY + "*")),
        "cached_embeddings": len(client.keys(PREFIX_EMBEDDING + "*")),
        "active_chat_sessions": len(client.keys(PREFIX_CHAT + "*")),
        "memory_used": client.info("memory").get("used_memory_human", "unknown"),
    }
