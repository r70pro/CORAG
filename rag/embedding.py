"""
Embedding layer — sentence-transformer embeddings with Qdrant vector storage.

Handles:
- Loading and managing the embedding model
- Encoding text chunks into dense vectors
- Upserting vectors into Qdrant with payload metadata
- Collection management (create, delete, info)
"""

import os
import re
import uuid
import hashlib
from typing import List, Dict, Optional

from settings_manager import load_settings

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
    CollectionInfo,
)

# Default configuration
DEFAULT_QDRANT_CONFIG = {
    "host": "localhost",
    "port": 6333,
    "grpc_port": 6334,
}

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

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
    safe_suffix = re.sub(r'[^a-zA-Z0-9_-]', '_', model_name)
    # limit length to avoid too long Qdrant names
    if len(safe_suffix) > 40:
        safe_suffix = safe_suffix[-40:]
    return f"olmocr_documents_{safe_suffix.lower().strip('_')}"

# Singleton model holder
_embedding_model = None
_embedding_model_name = None


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


def load_embedding_model(model_name=None):
    """Load the sentence-transformer embedding model.

    Uses lazy loading with singleton pattern.

    Args:
        model_name: HuggingFace model name. Defaults to all-MiniLM-L6-v2.

    Returns:
        The loaded SentenceTransformer model.
    """
    global _embedding_model, _embedding_model_name

    if model_name is None:
        try:
            model_name = load_settings().get("embedding_model", DEFAULT_EMBEDDING_MODEL)
        except Exception:
            model_name = os.environ.get(
                "OLMOCR_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
            )

    if _embedding_model is not None and _embedding_model_name == model_name:
        return _embedding_model

    from sentence_transformers import SentenceTransformer

    device = os.environ.get("OLMOCR_EMBEDDING_DEVICE", "cpu")
    print(f"Loading embedding model '{model_name}' on {device}...")
    _embedding_model = SentenceTransformer(model_name, device=device)
    _embedding_model_name = model_name
    print(f"Embedding model loaded. Dimension: {_embedding_model.get_embedding_dimension()}")
    return _embedding_model


def get_embedding_dimension(model_name=None):
    """Get the embedding dimension for the current model."""
    model = load_embedding_model(model_name)
    return model.get_embedding_dimension()


def encode_texts(texts: List[str], model_name=None, batch_size=32) -> List[List[float]]:
    """Encode a list of texts into dense vectors.

    Args:
        texts: List of text strings to encode.
        model_name: Optional model name override.
        batch_size: Batch size for encoding.

    Returns:
        List of embedding vectors (list of floats).
    """
    if not texts:
        return []

    model = load_embedding_model(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 50,
        normalize_embeddings=True,  # For cosine similarity
    )
    return embeddings.tolist()


def encode_query(query: str, model_name=None) -> List[float]:
    """Encode a single query into a dense vector.

    Args:
        query: Query text string.
        model_name: Optional model name override.

    Returns:
        Embedding vector (list of floats).
    """
    model = load_embedding_model(model_name)
    embedding = model.encode(
        query,
        normalize_embeddings=True,
    )
    return embedding.tolist()


def init_collection(dimension=None, model_name=None):
    """Create the Qdrant collection if it doesn't exist.

    Args:
        dimension: Vector dimension. Auto-detected from model if not provided.
        model_name: Embedding model name (for dimension detection).
    """
    client = get_qdrant_client()
    collection_name = get_collection_name(model_name)

    # Check if collection exists
    collections = client.get_collections().collections
    existing = [c.name for c in collections]

    if collection_name in existing:
        print(f"Qdrant collection '{collection_name}' already exists.")
        return

    if dimension is None:
        dimension = get_embedding_dimension(model_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=dimension,
            distance=Distance.COSINE,
        ),
    )
    print(f"Created Qdrant collection '{collection_name}' with dimension {dimension}.")


def delete_collection(model_name=None):
    """Delete the Qdrant collection."""
    client = get_qdrant_client()
    collection_name = get_collection_name(model_name)
    try:
        client.delete_collection(collection_name=collection_name)
        print(f"Deleted Qdrant collection '{collection_name}'.")
    except Exception as e:
        print(f"Error deleting collection: {e}")


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
            "vectors_count": info.vectors_count,
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


def upsert_chunks(chunks: List[Dict], model_name=None, batch_size=32) -> List[Dict]:
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
    if not chunks:
        return []

    client = get_qdrant_client()

    # Ensure collection exists
    init_collection(model_name=model_name)

    # Extract texts and encode
    texts = [c["text"] for c in chunks]
    embeddings = encode_texts(texts, model_name=model_name, batch_size=batch_size)

    if model_name is None:
        try:
            model_name = load_settings().get("embedding_model", DEFAULT_EMBEDDING_MODEL)
        except Exception:
            model_name = DEFAULT_EMBEDDING_MODEL

    collection_name = get_collection_name(model_name)

    # Build Qdrant points
    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        point_id = str(uuid.uuid4())
        chunk["qdrant_point_id"] = point_id
        chunk["embedding_model"] = model_name

        payload = {
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "run_id": chunk["run_id"],
            "chunk_index": chunk["chunk_index"],
            "page_number": chunk.get("page_number"),
            "document_type": chunk.get("document_type", "unknown"),
            "author": chunk.get("author"),
            "date_extracted": chunk.get("date_extracted"),
            "section_type": chunk.get("section_type", "general"),
            "patient_name": chunk.get("patient_name"),
            "text_preview": chunk["text"][:200],  # Preview for debugging
        }

        points.append(PointStruct(
            id=point_id,
            vector=embedding,
            payload=payload,
        ))

    # Upsert in batches
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(
            collection_name=collection_name,
            points=batch,
        )

    print(f"Upserted {len(points)} vectors into Qdrant collection '{collection_name}'.")
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
        print(f"Deleted vectors for run {run_id} from Qdrant.")
    except Exception as e:
        print(f"Error deleting run vectors: {e}")
