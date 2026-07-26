"""
Embedding layer — sentence-transformer embeddings with Qdrant vector storage.

Handles:
- Loading and managing the embedding model
- Encoding text chunks into dense vectors
- Upserting vectors into Qdrant with payload metadata
- Collection management (create, delete, info)
"""

import hashlib
import logging
import os
import re
import sys
import threading
from collections.abc import Generator
from datetime import date

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from settings_manager import load_settings

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_QDRANT_CONFIG = {
    "host": "localhost",
    "port": 6333,
    "grpc_port": 6334,
}

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"


def get_collection_name(model_name=None) -> str:
    """Get the Qdrant collection name based on the embedding model name.

    Avoids dimension collisions when switching models.
    """
    if model_name is None:
        try:
            model_name = load_settings().get("embedding_model", DEFAULT_EMBEDDING_MODEL)
        except Exception:
            model_name = DEFAULT_EMBEDDING_MODEL

    # Create a safe name (Qdrant collection names must match ^[a-zA-Z0-9_-]+$)
    safe_suffix = re.sub(r"[^a-zA-Z0-9_-]", "_", model_name)
    # limit length to avoid too long Qdrant names
    if len(safe_suffix) > 40:
        safe_suffix = safe_suffix[-40:]
    return f"olmocr_documents_{safe_suffix.lower().strip('_')}"


# Singleton model holder
_embedding_model = None
_embedding_model_name = None
_embedding_model_lock = threading.Lock()
_reranker_model = None
_reranker_model_name = None
_reranker_model_device = None
_reranker_model_lock = threading.Lock()


def get_qdrant_config():
    """Get Qdrant configuration from environment or defaults."""
    return {
        "host": os.environ.get("OLMOCR_QDRANT_HOST", DEFAULT_QDRANT_CONFIG["host"]),
        "port": int(os.environ.get("OLMOCR_QDRANT_PORT", DEFAULT_QDRANT_CONFIG["port"])),
    }


def get_qdrant_client(config=None):
    """Create a Qdrant client."""
    if config is None:
        config = get_qdrant_config()
    return QdrantClient(host=config["host"], port=config["port"])


def is_healthy():
    """Check if Qdrant is reachable."""
    try:
        client = get_qdrant_client()
        client.get_collections()
        return True
    except Exception:
        return False


def load_embedding_model(model_name=None, device=None):
    """Load the sentence-transformer embedding model.

    Uses lazy loading with singleton pattern. Auto-selects CUDA if available
    unless explicitly set to CPU.

    Args:
        model_name: HuggingFace model name. Defaults to BAAI/bge-large-en-v1.5.
        device: Device to load model on ('cuda', 'cpu', 'auto', or None).

    Returns:
        The loaded SentenceTransformer model.
    """
    global _embedding_model, _embedding_model_name

    if model_name is None:
        try:
            model_name = load_settings().get("embedding_model", DEFAULT_EMBEDDING_MODEL)
        except Exception:
            model_name = os.environ.get("OLMOCR_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

    if device is None:
        try:
            device = load_settings().get("embedding_device")
        except Exception:
            device = None
        if not device:
            device = os.environ.get("OLMOCR_EMBEDDING_DEVICE")

    if not device or device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    if _embedding_model is not None and _embedding_model_name == model_name:
        return _embedding_model

    with _embedding_model_lock:
        if _embedding_model is not None and _embedding_model_name == model_name:
            return _embedding_model

        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model '{model_name}' on {device}...")
        try:
            _embedding_model = SentenceTransformer(model_name, device=device)
        except Exception as e:
            if device != "cpu":
                logger.warning(
                    f"Failed to load embedding model '{model_name}' on {device}: {e}. "
                    f"Falling back to CPU."
                )
                if "torch" in sys.modules:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                _embedding_model = SentenceTransformer(model_name, device="cpu")
            else:
                raise
        _embedding_model_name = model_name
        logger.info(
            f"Embedding model loaded. Dimension: {_embedding_model.get_embedding_dimension()}"
        )
        return _embedding_model


def get_embedding_dimension(model_name=None):
    """Get the embedding dimension for the current model."""
    model = load_embedding_model(model_name)
    return model.get_embedding_dimension()


def encode_texts(texts: list[str], model_name=None, batch_size=64) -> list[list[float]]:
    """Encode a list of texts into dense vectors with bulk caching and acceleration.

    Args:
        texts: List of text strings to encode.
        model_name: Optional model name override.
        batch_size: Batch size for encoding.

    Returns:
        List of embedding vectors (list of floats).
    """
    if not texts:
        return []

    if model_name is None:
        try:
            model_name = load_settings().get("embedding_model", DEFAULT_EMBEDDING_MODEL)
        except Exception:
            model_name = os.environ.get("OLMOCR_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

    import rag.cache as cache

    redis_healthy = False
    try:
        redis_healthy = cache.is_healthy()
    except Exception:
        pass

    embeddings = [None] * len(texts)
    uncached_indices = []
    uncached_texts = []

    if redis_healthy:
        try:
            cached_vals = cache.get_cached_embeddings_bulk(texts, model_name)
            for idx, (text, cached_val) in enumerate(zip(texts, cached_vals, strict=False)):
                if cached_val is not None:
                    embeddings[idx] = cached_val
                else:
                    uncached_indices.append(idx)
                    uncached_texts.append(text)
        except Exception:
            uncached_indices = list(range(len(texts)))
            uncached_texts = texts
    else:
        uncached_indices = list(range(len(texts)))
        uncached_texts = texts

    if uncached_texts:
        model = load_embedding_model(model_name)
        new_embeddings = model.encode(
            uncached_texts,
            batch_size=batch_size,
            show_progress_bar=len(uncached_texts) > 50,
            normalize_embeddings=True,
        )
        new_embeddings_list = new_embeddings.tolist()

        for idx, new_idx in enumerate(uncached_indices):
            emb = new_embeddings_list[idx]
            embeddings[new_idx] = emb

        if redis_healthy:
            try:
                cache.cache_embeddings_bulk(uncached_texts, new_embeddings_list, model_name)
            except Exception:
                pass

    return embeddings


def encode_query(query: str, model_name=None) -> list[float]:
    """Encode a single query into a dense vector.

    Args:
        query: Query text string.
        model_name: Optional model name override.

    Returns:
        Embedding vector (list of floats).
    """
    if model_name is None:
        try:
            model_name = load_settings().get("embedding_model", DEFAULT_EMBEDDING_MODEL)
        except Exception:
            model_name = os.environ.get("OLMOCR_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

    import rag.cache as cache

    redis_healthy = False
    try:
        redis_healthy = cache.is_healthy()
    except Exception:
        pass

    if redis_healthy:
        try:
            cached_val = cache.get_cached_embedding(query, model_name)
            if cached_val is not None:
                return cached_val
        except Exception:
            pass

    model = load_embedding_model(model_name)
    embedding = model.encode(
        query,
        normalize_embeddings=True,
    )
    emb_list = embedding.tolist()

    if redis_healthy:
        try:
            cache.cache_embedding(query, emb_list, model_name)
        except Exception:
            pass

    return emb_list


def init_collection(dimension=None, model_name=None):
    """Create the Qdrant collection if it doesn't exist.

    If the collection already exists but its vector dimension differs from the
    current embedding model, a ``ValueError`` is raised rather than silently
    allowing a dimension-mismatch upsert (which Qdrant would reject anyway, but
    only after a confusing failure mid-index). The mismatch most commonly
    happens after an embedding-model switch; deleting the collection (via
    ``delete_collection``) and re-indexing resolves it.

    Args:
        dimension: Vector dimension. Auto-detected from model if not provided.
        model_name: Embedding model name (for dimension detection).
    """
    client = get_qdrant_client()
    collection_name = get_collection_name(model_name)

    # Check if collection exists
    collections = client.get_collections().collections
    existing = [c.name for c in collections]

    if dimension is None:
        dimension = get_embedding_dimension(model_name)

    if collection_name in existing:
        try:
            info = client.get_collection(collection_name)
            existing_dim = info.config.params.vectors.size
        except Exception:
            existing_dim = None

        # Only enforce a dimension mismatch when we can actually read a real
        # integer size. If the collection metadata is unavailable (older client
        # versions, mock environments, or permission errors) we cannot safely
        # assert a mismatch, so we leave the existing collection untouched.
        if isinstance(existing_dim, int) and existing_dim != dimension:
            raise ValueError(
                f"Qdrant collection '{collection_name}' exists with dimension "
                f"{existing_dim}, but embedding model requires {dimension}. "
                f"This usually means the embedding model was changed. Delete the "
                f"collection (e.g. via the cleanup tools) and re-index."
            )
        logger.info(f"Qdrant collection '{collection_name}' already exists.")
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=dimension,
            distance=Distance.COSINE,
        ),
    )
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="date_int",
            field_schema=PayloadSchemaType.INTEGER,
        )
    except Exception as e:
        logger.warning(f"Warning: could not create Qdrant payload index for date_int: {e}")
    logger.info(f"Created Qdrant collection '{collection_name}' with dimension {dimension}.")


def delete_collection(model_name=None):
    """Delete the Qdrant collection."""
    client = get_qdrant_client()
    collection_name = get_collection_name(model_name)
    try:
        client.delete_collection(collection_name=collection_name)
        logger.info(f"Deleted Qdrant collection '{collection_name}'.")
    except Exception as e:
        logger.error(f"Error deleting collection: {e}")


def get_collection_info(model_name=None):
    """Get information about the Qdrant collection.

    Returns:
        Dict with point count, vector dimension, etc.
    """
    client = get_qdrant_client()
    collection_name = get_collection_name(model_name)
    try:
        info = client.get_collection(collection_name=collection_name)
        return {
            "points_count": info.points_count,
            "vectors_count": getattr(info, "vectors_count", info.points_count),
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": info.status.value if info.status else "unknown",
        }
    except Exception:
        return {
            "points_count": 0,
            "vectors_count": 0,
            "indexed_vectors_count": 0,
            "status": "not_found",
        }


def _deterministic_point_id(chunk_id: str) -> str:
    """Derive a stable Qdrant point ID from a chunk ID.

    Using a deterministic ID (rather than a random UUID) makes re-indexing
    idempotent: upserting the same chunk overwrites the existing point instead
    of appending a duplicate vector. Qdrant accepts UUID-form strings as point
    IDs, so we hash the chunk_id into a UUID-shaped value.
    """
    digest = hashlib.sha256(chunk_id.encode("utf-8")).hexdigest()[:32]
    return f"{digest[0:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


def prepare_chunk_point_ids(chunks: list[dict], model_name=None) -> str:
    """Populate deterministic point IDs before any external mutation occurs."""
    if model_name is None:
        try:
            model_name = load_settings().get("embedding_model", DEFAULT_EMBEDDING_MODEL)
        except Exception:
            model_name = DEFAULT_EMBEDDING_MODEL
    for chunk in chunks:
        chunk["qdrant_point_id"] = _deterministic_point_id(chunk["chunk_id"])
        chunk["embedding_model"] = model_name
    return model_name


def snapshot_points(point_ids, model_name=None) -> dict[str, PointStruct]:
    """Read point payloads and vectors so an operation can restore overwrites."""
    ids = list(dict.fromkeys(str(point_id) for point_id in point_ids if point_id is not None))
    if not ids:
        return {}
    client = get_qdrant_client()
    collection_name = get_collection_name(model_name)
    snapshots: dict[str, PointStruct] = {}
    for start in range(0, len(ids), 256):
        records = client.retrieve(
            collection_name=collection_name,
            ids=ids[start : start + 256],
            with_payload=True,
            with_vectors=True,
        )
        for record in records:
            snapshots[str(record.id)] = PointStruct(
                id=record.id,
                vector=record.vector,
                payload=record.payload or {},
            )
    return snapshots


def delete_points(point_ids, model_name=None):
    """Delete an exact point-ID set; never broadens the selector to a run."""
    ids = list(dict.fromkeys(str(point_id) for point_id in point_ids if point_id is not None))
    if not ids:
        return
    client = get_qdrant_client()
    client.delete(
        collection_name=get_collection_name(model_name),
        points_selector=PointIdsList(points=ids),
        wait=True,
    )


def get_qdrant_point_ids_for_run(run_id, model_name=None) -> set[str]:
    """Return every point ID for a run in the selected collection."""
    client = get_qdrant_client()
    collection_name = get_collection_name(model_name)
    point_ids = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="run_id",
                        match=MatchValue(value=run_id),
                    )
                ]
            ),
            limit=256,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        point_ids.update(str(point.id) for point in points)
        if offset is None:
            break
    return point_ids


def rollback_point_mutations(touched_point_ids, snapshots, model_name=None):
    """Restore the exact Qdrant state captured before a failed operation.

    Only IDs absent from the snapshot were created by this operation, so only
    those IDs are deleted. Existing IDs are restored from their saved vectors
    and payloads instead of being deleted.
    """
    touched = {str(point_id) for point_id in touched_point_ids if point_id is not None}
    existing = set(snapshots)
    delete_points(touched - existing, model_name=model_name)
    if snapshots:
        get_qdrant_client().upsert(
            collection_name=get_collection_name(model_name),
            points=list(snapshots.values()),
            wait=True,
        )


def upsert_chunks_generator(
    chunks: list[dict],
    model_name=None,
    batch_size=32,
    pre_delete_run_ids: list[str] | None = None,
    full_reindex=False,
) -> Generator[dict, None, None]:
    """Embed and upsert chunks into Qdrant, yielding progress status dicts.

    Point IDs are derived deterministically from each chunk's ``chunk_id`` so
    that re-indexing upserts in place rather than creating duplicate vectors.
    Run-wide pre-deletion is available only to an explicitly authorised full
    reindex; retries and incremental uploads must use exact point-ID rollback.

    Args:
        chunks: List of chunk dicts from the chunker.
        model_name: Embedding model name.
        batch_size: Batch size for embedding.
        pre_delete_run_ids: Optional run IDs to clear. This is rejected unless
            ``full_reindex`` is explicitly true.
        full_reindex: Authorise destructive full-run deletion. Incremental
            uploads must leave this false.

    Yields:
        Dict progress update: {"stage": "embedding"|"indexing", "current": int, "total": int}
    """
    if not chunks:
        return

    client = get_qdrant_client()

    # Ensure collection exists
    init_collection(model_name=model_name)

    model_name = prepare_chunk_point_ids(chunks, model_name=model_name)

    collection_name = get_collection_name(model_name)

    # Legacy broad deletion is gated behind an explicit full-reindex signal.
    # CorpusIndexingService uses its safer snapshot/exact-ID replacement path.
    if pre_delete_run_ids:
        if not full_reindex:
            raise ValueError("pre_delete_run_ids is reserved for an explicit full-reindex workflow")
        for run_id in pre_delete_run_ids:
            delete_run_vectors(run_id, model_name=model_name)

    # Process in batches to support streaming progress
    total = len(chunks)

    # 1. First encode all texts.
    texts = [c["text"] for c in chunks]
    embeddings = []

    for i in range(0, total, batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_embs = encode_texts(batch_texts, model_name=model_name, batch_size=batch_size)
        embeddings.extend(batch_embs)
        yield {"stage": "embedding", "current": min(i + batch_size, total), "total": total}

    # 2. Build Qdrant points (deterministic IDs for idempotent upsert)
    points = []
    for chunk, embedding in zip(chunks, embeddings, strict=False):
        point_id = chunk["qdrant_point_id"]

        date_extracted = chunk.get("date_extracted")
        date_int = None
        if date_extracted:
            try:
                if isinstance(date_extracted, date):
                    date_extracted = date_extracted.isoformat()
                else:
                    date_extracted = date.fromisoformat(str(date_extracted)).isoformat()
                date_int = int(date_extracted.replace("-", ""))
            except (TypeError, ValueError):
                date_extracted = None

        payload = {
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "run_id": chunk["run_id"],
            "chunk_index": chunk["chunk_index"],
            "page_number": chunk.get("page_number"),
            "source_char_start": chunk.get("source_char_start"),
            "source_char_end": chunk.get("source_char_end"),
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
            "provenance_type": chunk.get("provenance_type"),
            "original_filename": chunk.get("original_filename"),
            "document_type": chunk.get("document_type", "unknown"),
            "author": chunk.get("author"),
            "date_extracted": date_extracted,
            "date_raw": chunk.get("date_raw"),
            "date_int": date_int,
            "section_type": chunk.get("section_type", "general"),
            "patient_name": chunk.get("patient_name"),
            "text_preview": chunk["text"][:200],  # Preview for debugging
        }

        points.append(
            PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload,
            )
        )

    # 3. Upsert in batches with exponential backoff retries
    import time

    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        max_retries = 3
        for attempt in range(max_retries):
            try:
                client.upsert(
                    collection_name=collection_name,
                    points=batch,
                )
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(0.5 * (2**attempt))
        yield {"stage": "indexing", "current": min(i + batch_size, total), "total": total}

    logger.info(f"Upserted {len(points)} vectors into Qdrant collection '{collection_name}'.")


def upsert_chunks(chunks: list[dict], model_name=None, batch_size=32) -> list[dict]:
    """Embed and upsert chunks into Qdrant.

    This is the main indexing function. It:
    1. Extracts text from each chunk
    2. Encodes all texts in batches
    3. Upserts vectors with metadata payload into Qdrant
    4. Returns the chunks with qdrant_point_id populated

    Args:
        chunks: List of chunk dicts from the chunker.
        model_name: Embedding model name.
        batch_size: Batch size for embedding.

    Returns:
        Updated chunks list with qdrant_point_id set.
    """
    for _ in upsert_chunks_generator(chunks, model_name=model_name, batch_size=batch_size):
        pass
    return chunks


def delete_run_vectors(run_id: str, model_name=None):
    """Delete all vectors for a given run from Qdrant.

    Args:
        run_id: The OCR run identifier.
    """
    client = get_qdrant_client()
    collection_name = get_collection_name(model_name)
    try:
        client.delete(
            collection_name=collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="run_id",
                        match=MatchValue(value=run_id),
                    )
                ]
            ),
        )
        logger.info(f"Deleted vectors for run {run_id} from Qdrant.")
    except Exception as e:
        logger.error(f"Error deleting run vectors: {e}")
        raise


def load_reranker_model(model_name=None, device=None):
    """Load the sentence-transformer cross-encoder reranker model.

    Uses lazy loading with singleton pattern.
    """
    global _reranker_model, _reranker_model_name, _reranker_model_device

    if model_name is None:
        try:
            model_name = load_settings().get("reranker_model", "BAAI/bge-reranker-large")
        except Exception:
            model_name = "BAAI/bge-reranker-large"

    if device is None:
        try:
            device = load_settings().get("reranker_device", "cuda")
        except Exception:
            device = "cuda"

    device = str(device).lower()
    if (
        _reranker_model is not None
        and _reranker_model_name == model_name
        and _reranker_model_device == device
    ):
        return _reranker_model

    with _reranker_model_lock:
        if (
            _reranker_model is not None
            and _reranker_model_name == model_name
            and _reranker_model_device == device
        ):
            return _reranker_model

        from sentence_transformers import CrossEncoder

        logger.info(f"Loading reranker model '{model_name}' on {device}...")
        _reranker_model = CrossEncoder(model_name, device=device)
        _reranker_model_name = model_name
        _reranker_model_device = device
        logger.info("Reranker model loaded successfully.")
        return _reranker_model
