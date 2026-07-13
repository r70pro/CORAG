"""
Retriever — query embedding, Qdrant similarity search, metadata filtering,
and Maximal Marginal Relevance (MMR) re-ranking.

This module connects user queries to the most relevant document chunks.
"""

from typing import List, Dict, Optional

from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    DatetimeRange,
)

from rag.embedding import (
    encode_query,
    get_qdrant_client,
    get_collection_name,
    init_collection,
)
from rag import db as rag_db


def search_similar(
    query: str,
    top_k: int = 8,
    doc_type_filter: Optional[str] = None,
    author_filter: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    run_id_filter: Optional[str] = None,
    doc_id_filter: Optional[str] = None,
    score_threshold: float = 0.25,
) -> List[Dict]:
    """Search for chunks similar to the query with optional metadata filters.

    Args:
        query: The user's natural language query.
        top_k: Number of results to return.
        doc_type_filter: Filter by document type (e.g., 'specialist_letter').
        author_filter: Filter by author name.
        date_from: Filter by date (ISO format, inclusive lower bound).
        date_to: Filter by date (ISO format, inclusive upper bound).
        run_id_filter: Filter by specific OCR run.
        doc_id_filter: Filter by specific document.
        score_threshold: Minimum similarity score to include.

    Returns:
        List of result dicts with chunk data, metadata, and similarity score.
    """
    # Encode the query
    query_vector = encode_query(query)

    # Build Qdrant filter conditions
    must_conditions = []

    if doc_type_filter:
        must_conditions.append(
            FieldCondition(key="document_type", match=MatchValue(value=doc_type_filter))
        )
    if author_filter:
        must_conditions.append(
            FieldCondition(key="author", match=MatchValue(value=author_filter))
        )
    if run_id_filter:
        must_conditions.append(
            FieldCondition(key="run_id", match=MatchValue(value=run_id_filter))
        )
    if doc_id_filter:
        must_conditions.append(
            FieldCondition(key="doc_id", match=MatchValue(value=doc_id_filter))
        )

    # Date range filter
    if date_from or date_to:
        range_kwargs = {}
        if date_from:
            range_kwargs["gte"] = date_from
        if date_to:
            range_kwargs["lte"] = date_to
        must_conditions.append(
            FieldCondition(key="date_extracted", match=None, range=DatetimeRange(**range_kwargs))
        )

    query_filter = Filter(must=must_conditions) if must_conditions else None

    # Search Qdrant
    client = get_qdrant_client()
    try:
        results = client.search(
            collection_name=get_collection_name(),
            query_vector=query_vector,
            query_filter=query_filter,
            limit=top_k * 2,  # Fetch extra for MMR re-ranking
            score_threshold=score_threshold,
            with_payload=True,
        )
    except Exception as e:
        error_msg = str(e)
        if "doesn't exist" in error_msg or "Not found" in error_msg:
            try:
                init_collection()
                results = client.search(
                    collection_name=get_collection_name(),
                    query_vector=query_vector,
                    query_filter=query_filter,
                    limit=top_k * 2,
                    score_threshold=score_threshold,
                    with_payload=True,
                )
            except Exception as e2:
                print(f"Error searching after collection initialization: {e2}")
                return []
        else:
            print(f"Error searching Qdrant: {e}")
            return []

    if not results:
        return []

    # Collect Qdrant point IDs for database enrichment
    qdrant_ids = [str(r.id) for r in results]
    db_chunks = {}
    try:
        db_results = rag_db.get_chunks_by_qdrant_ids(qdrant_ids)
        for row in db_results:
            db_chunks[row["qdrant_point_id"]] = row
    except Exception as e:
        print(f"Warning: could not enrich from DB: {e}")

    # Build result list
    enriched_results = []
    for result in results:
        point_id = str(result.id)
        payload = result.payload or {}

        # Merge Qdrant payload with PostgreSQL metadata
        db_data = db_chunks.get(point_id, {})

        enriched_results.append({
            "qdrant_point_id": point_id,
            "score": result.score,
            "chunk_id": payload.get("chunk_id") or db_data.get("chunk_id"),
            "doc_id": payload.get("doc_id") or db_data.get("doc_id"),
            "run_id": payload.get("run_id") or db_data.get("run_id"),
            "text": db_data.get("text", payload.get("text_preview", "")),
            "page_number": payload.get("page_number") or db_data.get("page_number"),
            "document_type": payload.get("document_type") or db_data.get("document_type"),
            "author": payload.get("author") or db_data.get("author"),
            "date_extracted": payload.get("date_extracted") or db_data.get("date_extracted"),
            "section_type": payload.get("section_type") or db_data.get("section_type"),
            "patient_name": payload.get("patient_name") or db_data.get("patient_name"),
            "original_filename": db_data.get("original_filename", ""),
        })

    # Apply MMR re-ranking for diversity
    if len(enriched_results) > top_k:
        enriched_results = _mmr_rerank(enriched_results, query_vector, top_k)
    else:
        enriched_results = enriched_results[:top_k]

    return enriched_results


def _mmr_rerank(
    results: List[Dict],
    query_vector: List[float],
    top_k: int,
    lambda_param: float = 0.7,
) -> List[Dict]:
    """Maximal Marginal Relevance re-ranking.

    Balances relevance (similarity to query) with diversity
    (dissimilarity to already-selected results).

    Args:
        results: List of result dicts with 'score' field.
        query_vector: The query embedding vector.
        top_k: Number of results to return.
        lambda_param: Balance between relevance (1.0) and diversity (0.0).

    Returns:
        Re-ranked list of results.
    """

    if len(results) <= top_k:
        return results

    # Use scores as proxy for relevance (already computed by Qdrant)
    selected = []
    candidates = list(range(len(results)))

    # Always pick the most relevant first
    best_idx = max(candidates, key=lambda i: results[i]["score"])
    selected.append(best_idx)
    candidates.remove(best_idx)

    while len(selected) < top_k and candidates:
        best_score = -float("inf")
        best_candidate = None

        for cand_idx in candidates:
            # Relevance component
            relevance = results[cand_idx]["score"]

            # Diversity component — max similarity to any selected result
            max_sim_to_selected = max(
                _text_similarity(results[cand_idx]["text"], results[sel_idx]["text"])
                for sel_idx in selected
            )

            # MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_candidate = cand_idx

        if best_candidate is not None:
            selected.append(best_candidate)
            candidates.remove(best_candidate)
        else:
            break

    return [results[i] for i in selected]


def _text_similarity(text1: str, text2: str) -> float:
    """Simple Jaccard similarity between two texts (for MMR diversity check).

    This avoids re-embedding just for diversity calculation.
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / len(union)


def format_context_for_llm(results: List[Dict]) -> str:
    """Format retrieved chunks into a context string for the LLM prompt.

    Each chunk is labelled with its source metadata for grounded responses.

    Args:
        results: List of search result dicts from search_similar().

    Returns:
        Formatted context string for prompt injection.
    """
    if not results:
        return "No relevant document excerpts found."

    context_parts = []
    for i, result in enumerate(results, 1):
        header_parts = [f"[Source {i}]"]

        if result.get("original_filename"):
            header_parts.append(f"File: {result['original_filename']}")
        if result.get("page_number"):
            header_parts.append(f"Page: {result['page_number']}")
        if result.get("author"):
            header_parts.append(f"Author: {result['author']}")
        if result.get("date_extracted"):
            header_parts.append(f"Date: {result['date_extracted']}")
        if result.get("document_type") and result["document_type"] != "unknown":
            doc_type_label = result["document_type"].replace("_", " ").title()
            header_parts.append(f"Type: {doc_type_label}")

        header = " | ".join(header_parts)
        text = result.get("text", "").strip()

        context_parts.append(f"{header}\n{text}")

    return "\n\n---\n\n".join(context_parts)


def get_available_filters() -> Dict:
    """Get available filter values from the indexed corpus.

    Returns:
        Dict with lists of unique authors, document types, run IDs, etc.
    """
    try:
        stats = rag_db.get_corpus_stats()
        return {
            "indexed_runs": stats.get("indexed_runs", 0),
            "indexed_documents": stats.get("indexed_documents", 0),
            "total_chunks": stats.get("total_chunks", 0),
            "unique_authors": stats.get("unique_authors", 0),
            "date_range": {
                "earliest": str(stats.get("earliest_date", "")) if stats.get("earliest_date") else None,
                "latest": str(stats.get("latest_date", "")) if stats.get("latest_date") else None,
            },
        }
    except Exception:
        return {
            "indexed_runs": 0,
            "indexed_documents": 0,
            "total_chunks": 0,
        }
