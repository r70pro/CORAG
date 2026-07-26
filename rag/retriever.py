"""
Retriever — query embedding, Qdrant similarity search, metadata filtering,
and Maximal Marginal Relevance (MMR) re-ranking.

This module connects user queries to the most relevant document chunks.
"""

import datetime
import logging
from typing import Any

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    Range,
)

from rag import db as rag_db
from rag.embedding import (
    encode_query,
    get_collection_name,
    get_qdrant_client,
    init_collection,
)

logger = logging.getLogger(__name__)


def search_similar(
    query: str,
    top_k: int = 8,
    doc_type_filter: str | None = None,
    author_filter: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    run_id_filter: str | None = None,
    doc_id_filter: str | None = None,
    score_threshold: float = 0.25,
    use_reranker: bool | None = None,
    reranker_model: str | None = None,
    reranker_device: str | None = None,
    progress_callback: Any | None = None,
) -> list[dict]:
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
        use_reranker: Enable or disable Cross-Encoder reranking.
        reranker_model: HuggingFace model name for Cross-Encoder.
        reranker_device: Device to run the Cross-Encoder on ('cuda' or 'cpu').

    Returns:
        List of result dicts with chunk data, metadata, and similarity score.
    """
    from settings_manager import load_settings

    settings = load_settings()

    if use_reranker is None:
        use_reranker = settings.get("use_reranker", True)
    if reranker_model is None:
        reranker_model = settings.get("reranker_model", "BAAI/bge-reranker-large")
    if reranker_device is None:
        reranker_device = settings.get("reranker_device", "cuda")

    # Fetch extra candidates if we are reranking to give the reranker a larger pool
    search_limit = top_k * 3 if use_reranker else top_k * 2
    if search_limit < 20 and use_reranker:
        search_limit = 20

    if progress_callback:
        progress_callback(0.1, "Encoding query and searching vector store...")

    # Encode the query
    query_vector = encode_query(query)

    # Build Qdrant filter conditions
    must_conditions = []

    if doc_type_filter:
        must_conditions.append(
            FieldCondition(key="document_type", match=MatchValue(value=doc_type_filter))
        )
    if author_filter:
        must_conditions.append(FieldCondition(key="author", match=MatchValue(value=author_filter)))
    if run_id_filter:
        resolved_run_id = run_id_filter
        if "/" in run_id_filter or "\\" in run_id_filter or run_id_filter.startswith("run_"):
            try:
                import os
                import re

                from rag.db import get_indexed_runs

                runs = get_indexed_runs()
                clean_filter = re.sub(r"\s*\(\d+\s+files?\)", "", run_id_filter).strip()
                for r in runs:
                    r_id = r.get("run_id", "")
                    r_dir = r.get("run_dir", "")
                    r_base = os.path.basename(r_dir)
                    if (
                        r_id in (run_id_filter, clean_filter)
                        or r_dir in (run_id_filter, clean_filter)
                        or r_base in (run_id_filter, clean_filter)
                    ):
                        resolved_run_id = r_id
                        break
            except Exception:
                pass
        must_conditions.append(
            FieldCondition(key="run_id", match=MatchValue(value=resolved_run_id))
        )
    if doc_id_filter:
        must_conditions.append(FieldCondition(key="doc_id", match=MatchValue(value=doc_id_filter)))
    date_from_norm = _normalize_iso_date(date_from) if date_from else None
    date_to_norm = _normalize_iso_date(date_to) if date_to else None

    # Date range filtering natively in Qdrant using the indexed `date_int` field
    if date_from_norm:
        try:
            from_int = int(date_from_norm.replace("-", ""))
            must_conditions.append(FieldCondition(key="date_int", range=Range(gte=float(from_int))))
        except Exception as e:
            logger.warning(f"Warning: could not parse date_from to int: {e}")
    if date_to_norm:
        try:
            to_int = int(date_to_norm.replace("-", ""))
            must_conditions.append(FieldCondition(key="date_int", range=Range(lte=float(to_int))))
        except Exception as e:
            logger.warning(f"Warning: could not parse date_to to int: {e}")

    query_filter = Filter(must=must_conditions) if must_conditions else None

    # Search Qdrant
    client = get_qdrant_client()
    try:
        results = client.search(
            collection_name=get_collection_name(),
            query_vector=query_vector,
            query_filter=query_filter,
            limit=search_limit,
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
                    limit=search_limit,
                    score_threshold=score_threshold,
                    with_payload=True,
                )
            except Exception as e2:
                logger.error(f"Error searching after collection initialization: {e2}")
                return []
        else:
            logger.error(f"Error searching Qdrant: {e}")
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
        logger.warning(f"Warning: could not enrich from DB: {e}")

    # Build result list
    enriched_results = []
    for result in results:
        point_id = str(result.id)
        payload = result.payload or {}

        # Merge Qdrant payload with PostgreSQL metadata
        db_data = db_chunks.get(point_id, {})

        def first_present(key, default=None, _payload=payload, _db_data=db_data):
            value = _payload.get(key)
            return value if value is not None else _db_data.get(key, default)

        enriched_results.append(
            {
                "qdrant_point_id": point_id,
                "score": result.score,
                "chunk_id": payload.get("chunk_id") or db_data.get("chunk_id"),
                "doc_id": payload.get("doc_id") or db_data.get("doc_id"),
                "run_id": payload.get("run_id") or db_data.get("run_id"),
                "text": db_data.get("text", payload.get("text_preview", "")),
                "page_number": first_present("page_number"),
                "source_char_start": first_present("source_char_start"),
                "source_char_end": first_present("source_char_end"),
                "page_start": first_present("page_start"),
                "page_end": first_present("page_end"),
                "provenance_type": first_present("provenance_type"),
                "document_type": payload.get("document_type") or db_data.get("document_type"),
                "author": payload.get("author") or db_data.get("author"),
                "date_extracted": payload.get("date_extracted") or db_data.get("date_extracted"),
                "date_raw": payload.get("date_raw") or db_data.get("date_raw"),
                "section_type": payload.get("section_type") or db_data.get("section_type"),
                "patient_name": payload.get("patient_name") or db_data.get("patient_name"),
                "original_filename": first_present("original_filename", ""),
            }
        )

    # Apply date-range filter in Python (Qdrant payload dates are plain
    # `YYYY-MM-DD` strings, not RFC3339 datetimes). Chunks with no parsed date
    # are excluded when a date bound is active.
    if date_from_norm or date_to_norm:

        def _in_range(res):
            d = _normalize_iso_date(str(res.get("date_extracted") or ""))
            if not d:
                return False
            if date_from_norm and d < date_from_norm:
                return False
            if date_to_norm and d > date_to_norm:
                return False
            return True

        enriched_results = [r for r in enriched_results if _in_range(r)]

    # Apply Reranker if enabled
    if use_reranker and enriched_results:
        if progress_callback:
            progress_callback(
                0.4, f"Reranking {len(enriched_results)} chunks using {reranker_model}..."
            )
        try:
            import math

            from rag.embedding import load_reranker_model

            reranker = load_reranker_model(reranker_model, reranker_device)

            # Prepare query-chunk pairs
            pairs = [[query, res["text"]] for res in enriched_results]
            scores = reranker.predict(pairs)

            # Map logits to probability range [0, 1] using sigmoid
            for res, score in zip(enriched_results, scores, strict=False):
                res["score"] = float(1 / (1 + math.exp(-score)))

            # Sort by new scores descending
            enriched_results.sort(key=lambda x: x["score"], reverse=True)
            logger.info(
                f"Reranking completed successfully for {len(enriched_results)} chunks using {reranker_model}."
            )
            if progress_callback:
                progress_callback(0.8, "Rerank completed. Applying diversity filters...")
        except Exception as e:
            logger.error(f"Error during reranking: {e}. Falling back to default retrieval.")
            if progress_callback:
                progress_callback(
                    0.8, f"Reranking failed: {e}. Falling back to default retrieval..."
                )

    # Apply MMR re-ranking for diversity
    if len(enriched_results) > top_k:
        enriched_results = _mmr_rerank(enriched_results, query_vector, top_k)
    else:
        enriched_results = enriched_results[:top_k]

    if progress_callback:
        progress_callback(1.0, "Retrieval & reranking complete.")

    return enriched_results


def _mmr_rerank(
    results: list[dict],
    query_vector: list[float],
    top_k: int,
    lambda_param: float = 0.7,
) -> list[dict]:
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

    # Pre-calculate token sets and their lengths to avoid millions of allocations inside the loop
    token_sets = [set((res["text"] or "").lower().split()) for res in results]
    set_lengths = [len(s) for s in token_sets]

    selected = []
    candidates = list(range(len(results)))

    # Always pick the most relevant first
    best_idx = max(candidates, key=lambda i: results[i]["score"])
    selected.append(best_idx)
    candidates.remove(best_idx)

    # Initialize max_sim_to_selected with similarity to the first selected item
    max_sim_to_selected = [0.0] * len(results)
    first_set = token_sets[best_idx]
    first_len = set_lengths[best_idx]
    if first_len > 0:
        for cand_idx in candidates:
            cand_set = token_sets[cand_idx]
            cand_len = set_lengths[cand_idx]
            if cand_len > 0:
                intersection_len = len(cand_set & first_set)
                union_len = cand_len + first_len - intersection_len
                max_sim_to_selected[cand_idx] = (
                    intersection_len / union_len if union_len > 0 else 0.0
                )

    while len(selected) < top_k and candidates:
        best_score = -float("inf")
        best_candidate = None

        for cand_idx in candidates:
            # Relevance component
            relevance = results[cand_idx]["score"]

            # Diversity component — max similarity to any selected result is pre-calculated/cached
            sim_val = max_sim_to_selected[cand_idx]

            # MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * sim_val

            if mmr_score > best_score:
                best_score = mmr_score
                best_candidate = cand_idx

        if best_candidate is not None:
            selected.append(best_candidate)
            candidates.remove(best_candidate)

            # Update max_sim_to_selected for remaining candidates with the newly selected candidate
            new_set = token_sets[best_candidate]
            new_len = set_lengths[best_candidate]
            if new_len > 0:
                for cand_idx in candidates:
                    cand_set = token_sets[cand_idx]
                    cand_len = set_lengths[cand_idx]
                    if cand_len > 0:
                        intersection_len = len(cand_set & new_set)
                        union_len = cand_len + new_len - intersection_len
                        sim = intersection_len / union_len if union_len > 0 else 0.0
                        if sim > max_sim_to_selected[cand_idx]:
                            max_sim_to_selected[cand_idx] = sim
        else:
            break

    return [results[i] for i in selected]


def _normalize_iso_date(value) -> str | None:
    """Normalise a user-supplied date to ``YYYY-MM-DD`` for comparison.

    Accepts ISO strings (``YYYY-MM-DD``, ``YYYY-MM``, ``YYYY``) as well as
    Unix epoch values (int/float seconds) such as those produced by some
    callers/tests. Returns ``None`` if it cannot be parsed, so callers can skip
    an invalid bound instead of crashing the query.
    """
    if value is None or value == "":
        return None
    # Epoch seconds (int/float) -> date in UTC.
    if isinstance(value, int | float):
        try:
            return datetime.datetime.utcfromtimestamp(float(value)).strftime("%Y-%m-%d")
        except (ValueError, OverflowError, OSError):
            return None
    text = str(value).strip()
    if not text:
        return None
    parts = text.split("-")
    try:
        if len(parts) == 3:
            y, m, d = (int(p) for p in parts)
            return datetime.date(y, m, d).isoformat()
        if len(parts) == 2:
            y, m = (int(p) for p in parts)
            return datetime.date(y, m, 1).isoformat()
        if len(parts) == 1:
            return datetime.date(int(parts[0]), 1, 1).isoformat()
    except (ValueError, TypeError, OverflowError):
        return None
    return None


def format_context_for_llm(results: list[dict]) -> str:
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
        provenance_type = result.get("provenance_type")
        page_start = result.get("page_start")
        page_end = result.get("page_end")
        if provenance_type == "external_markdown":
            header_parts.append("PDF provenance: none (external Markdown)")
        elif page_start is not None and page_end is not None:
            if page_start == page_end:
                header_parts.append(f"Page: {page_start}")
            else:
                header_parts.append(f"Pages: {page_start}-{page_end}")
        elif page_start is not None:
            header_parts.append(
                f"Page: {page_start} (start page only; end page not present in source metadata)"
            )
        else:
            header_parts.append("PDF page provenance: not present in source metadata")
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


def get_available_filters() -> dict:
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
                "earliest": str(stats.get("earliest_date", ""))
                if stats.get("earliest_date")
                else None,
                "latest": str(stats.get("latest_date", "")) if stats.get("latest_date") else None,
            },
        }
    except Exception:
        return {
            "indexed_runs": 0,
            "indexed_documents": 0,
            "total_chunks": 0,
        }
