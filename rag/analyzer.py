"""
LLM Analyzer — prompt assembly and vLLM OpenAI-compatible API integration.

Provides:
- System prompt templates for medicolegal analysis modes
- Context-aware prompt assembly from retrieved chunks
- Streaming chat completions via vLLM's OpenAI-compatible API
- Pre-built analysis templates (timeline, summary, inconsistencies)
"""

import json
import os
from collections.abc import Generator
from typing import Any

import httpx

from rag.retriever import format_context_for_llm, search_similar

# ── Loaded-model resolution cache ────────────────────────────
# The /models probe in analyze() is per-query expensive. Cache the *list of
# loaded models* per server URL for a short window so consecutive queries in a
# session don't each pay the HTTP round-trip. Equivalence is resolved at call
# time from the cached list, so different requested model names against the
# same server keep returning correct (possibly equivalent) resolutions.
_MODEL_CACHE_TTL = 30.0
_model_cache: dict[str, tuple] = {}  # url -> (timestamp, [loaded_model_ids])

# Known-equivalent model IDs: request -> served name.
_MODEL_EQUIVALENTS = {
    "microsoft/Phi-4-reasoning-plus": "nvidia/Phi-4-reasoning-plus-NVFP4",
}

# RAG analysis retrieval constants
STRUCTURED_MODE_MIN_TOP_K = 50
STRUCTURED_MODE_SCORE_THRESHOLD = 0.05


def _get_loaded_models(server_url: str) -> list[str]:
    """Fetch (and cache) the list of model IDs currently loaded in vLLM."""
    import time

    # Under test runs, skip the cache so each call re-probes (tests mock the
    # HTTP layer per-call and rely on fresh results).
    testing = os.environ.get("TESTING") is not None
    now = time.monotonic()
    cached = _model_cache.get(server_url)
    if not testing and cached is not None and now - cached[0] < _MODEL_CACHE_TTL:
        return cached[1]

    url = server_url.rstrip("/") + "/models"
    response = httpx.get(url, timeout=2.0)
    if response.status_code != 200:
        loaded: list[str] = []
    else:
        loaded = [m["id"] for m in response.json().get("data", [])]
    _model_cache[server_url] = (now, loaded)
    return loaded


def _resolve_loaded_model(server_url: str, model_name: str):
    """Resolve the model actually loaded in vLLM.

    Returns a ``(resolved_model, fell_back)`` tuple. The resolved model is the
    requested name when present, a known-equivalent loaded name when applicable,
    or the first loaded model as a last resort. ``fell_back`` is True only when
    neither an exact nor equivalent match was found (i.e. a genuine fallback).
    """
    loaded_models = _get_loaded_models(server_url)
    if not loaded_models:
        return model_name, False
    if model_name in loaded_models:
        return model_name, False

    equivalent = _MODEL_EQUIVALENTS.get(model_name)
    if equivalent in loaded_models:
        return equivalent, False

    # Reverse lookup: a loaded model that is equivalent to the requested one.
    for loaded in loaded_models:
        if _MODEL_EQUIVALENTS.get(loaded) == model_name:
            return loaded, False

    return loaded_models[0], True


def _is_equivalent(m1: str, m2: str) -> bool:
    return _MODEL_EQUIVALENTS.get(m1) == m2 or _MODEL_EQUIVALENTS.get(m2) == m1


def _map_equivalent(model_name: str, loaded_models: list) -> str:
    """Return ``model_name`` if present in ``loaded_models``, else the first
    known-equivalent entry, else ``model_name`` unchanged."""
    if model_name in loaded_models:
        return model_name
    for lm in loaded_models:
        if _MODEL_EQUIVALENTS.get(model_name) == lm or _MODEL_EQUIVALENTS.get(lm) == model_name:
            return lm
    return model_name


# ── System prompt templates ───────────────────────────────────

SYSTEM_PROMPTS = {
    "free_qa": """You are a medicolegal document analyst with expertise in personal injury, workers' compensation, and clinical documentation. You have been provided with excerpts from clinical records, specialist reports, and correspondence.

INSTRUCTIONS:
- Answer based ONLY on the provided document excerpts — do not hallucinate or assume facts not present in the sources
- Never use raw system source tags (like [Source 26] or [Source 52]) in final outputs
- Always cite the exact page number range of the original PDF document where the information is located
- Include robust verification details for every factual claim so that users can instantly verify the source when scrolling through the original file, including:
  * The exact document type and title (e.g., Operation Record, Specialist Correspondence)
  * The exact authoring physician or clinic (e.g., Dr. Gavin Weekes, Capital Radiology)
  * Identifying report details (e.g., Ref No: 2024AL0008570-1, Accession Number: 77.50382801)
- If multiple sources discuss the same event, synthesise the information and note any differences
- Use ISO date format (YYYY-MM-DD) when referencing dates
- If the answer cannot be determined from the provided excerpts, say so explicitly and suggest what additional documents might help
- Use clear, professional language appropriate for medicolegal analysis""",
    "timeline": """You are a medicolegal chronology specialist. Your task is to extract every dated event from the provided document excerpts and present them in strict chronological order.

INSTRUCTIONS:
- Extract EVERY event with a date (consultations, injuries, surgeries, referrals, reports, diagnoses, medication changes)
- Present as a markdown table with columns: Date | Event | Provider/Author | Source (PDF Page & Verifying Details)
- Use ISO date format (YYYY-MM-DD) for all dates
- If a date is ambiguous (e.g., "early 2018"), note the ambiguity but place it approximately
- For the "Source" column:
  * Never use raw system source tags (like [Source 26] or [Source 52]) in final outputs
  * Always cite the exact page number range of the original PDF document where the information is located
  * Include robust verification details for each entry so that users can instantly verify the source when scrolling through the original file, including:
    - The exact document type and title (e.g., Operation Record, Specialist Correspondence)
    - The exact authoring physician or clinic (e.g., Dr. Gavin Weekes, Capital Radiology)
    - Identifying report details (e.g., Ref No: 2024AL0008570-1, Accession Number: 77.50382801)
- Flag any inconsistencies in dates between different sources
- Order strictly by date, oldest first""",
    "injury_summary": """You are a medicolegal injury analyst. Your task is to produce a structured summary of the patient's injury, treatment, and outcomes from the provided document excerpts.

INSTRUCTIONS:
Generate a structured report with these sections:
1. **Patient Details** — Name, DOB, claim/reference numbers
2. **Mechanism of Injury** — How the injury occurred, date, circumstances
3. **Injuries Sustained** — List of all injuries/diagnoses with dates of diagnosis
4. **Treatment History** — All treatments, surgeries, therapies in chronological order
5. **Current Status** — Most recent assessment findings
6. **Medications** — All current and historical medications mentioned
7. **Providers Involved** — All treating practitioners with their roles
8. **Outstanding Issues** — Unresolved symptoms, pending treatments, or recommendations

For every factual claim or timeline entry in this summary:
- Never use raw system source tags (like [Source 26] or [Source 52]) in final outputs
- Always cite the exact page number range of the original PDF document where the information is located
- Include robust verification details so that users can instantly verify the source when scrolling through the original file, including:
  * The exact document type and title (e.g., Operation Record, Specialist Correspondence)
  * The exact authoring physician or clinic (e.g., Dr. Gavin Weekes, Capital Radiology)
  * Identifying report details (e.g., Ref No: 2024AL0008570-1, Accession Number: 77.50382801)
Flag any contradictions between providers.""",
    "inconsistency_finder": """You are a medicolegal document auditor specialising in identifying inconsistencies, contradictions, and discrepancies across clinical records.

INSTRUCTIONS:
- Compare accounts of the same events across different sources
- Identify discrepancies in: dates, injury descriptions, examination findings, treatment recommendations, patient-reported symptoms
- For each inconsistency, cite both sources with:
  * The exact page number range of the original PDF document where the information is located
  * Robust verification details (e.g., exact document type and title, authoring physician/clinic, Ref/Accession numbers)
  * Never use raw system source tags (like [Source 26] or [Source 52])
- Rate severity: MINOR (date formatting differences), MODERATE (differing clinical findings), MAJOR (contradictory diagnoses or recommendations)
- Present findings in a structured table: Issue | Source A Says | Source B Says | Severity
- Also note any gaps — events referenced but not documented""",
    "medication_tracker": """You are a clinical pharmacology analyst. Your task is to extract and track all medication references from the provided document excerpts.

INSTRUCTIONS:
- Extract every medication mentioned (name, dose, frequency, route, indication)
- Note the date and source where each medication is mentioned
- Track changes: new prescriptions, dose changes, cessations
- Present as a markdown table: Medication | Dose/Frequency | Date Started | Date Stopped | Prescriber | Source (PDF Page & Verifying Details)
- For the "Source" column:
  * Never use raw system source tags (like [Source 26] or [Source 52]) in final outputs
  * Always cite the exact page number range of the original PDF document where the information is located
  * Include robust verification details for each entry so that users can instantly verify the source when scrolling through the original file, including:
    - The exact document type and title (e.g., Operation Record, Specialist Correspondence)
    - The exact authoring physician or clinic (e.g., Dr. Gavin Weekes, Capital Radiology)
    - Identifying report details (e.g., Ref No: 2024AL0008570-1, Accession Number: 77.50382801)
- Flag any potential interactions or contraindications
- Note any allergies mentioned in the records""",
}


ANALYSIS_MODE_MAP = {
    "💬 Free Q&A": "free_qa",
    "free_qa": "free_qa",
    "📅 Timeline Generator": "timeline",
    "📋 Timeline": "timeline",
    "timeline": "timeline",
    "timeline_generator": "timeline",
    "🏥 Injury Summary": "injury_summary",
    "⚕️ Medical Summary": "injury_summary",
    "injury_summary": "injury_summary",
    "🔍 Inconsistency Finder": "inconsistency_finder",
    "⚖️ Injury Audit": "inconsistency_finder",
    "inconsistency_finder": "inconsistency_finder",
    "💊 Medication Tracker": "medication_tracker",
    "medication_tracker": "medication_tracker",
}


def get_analysis_modes():
    """Get available analysis modes and their descriptions."""
    return {
        "free_qa": "💬 Free Q&A — Ask anything about the documents",
        "timeline": "📅 Timeline Generator — Extract chronological events",
        "injury_summary": "🏥 Injury Summary — Structured injury/treatment report",
        "inconsistency_finder": "🔍 Inconsistency Finder — Cross-reference discrepancies",
        "medication_tracker": "💊 Medication Tracker — Track all medication references",
    }


def build_prompt(
    query: str,
    context: str,
    mode: str = "free_qa",
    chat_history: list[dict] | None = None,
) -> list[dict]:
    """Build the full message list for the LLM API call.

    Args:
        query: The user's question or instruction.
        context: Formatted context string from retrieved chunks.
        mode: Analysis mode key (maps to system prompt template).
        chat_history: Optional list of previous messages for multi-turn conversation.

    Returns:
        List of message dicts for OpenAI Chat Completions API.
    """
    system_prompt = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["free_qa"])

    messages = [{"role": "system", "content": system_prompt}]

    # Add chat history for multi-turn conversations
    if chat_history:
        for msg in chat_history[-6:]:  # Keep last 6 messages to manage context window
            messages.append(msg)

    # Build the user message with context
    user_message = f"""DOCUMENT EXCERPTS:

{context}

---

USER QUESTION:
{query}"""

    messages.append({"role": "user", "content": user_message})

    return messages


def query_llm_streaming(
    messages: list[dict],
    server_url: str,
    model_name: str,
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> Generator[str, None, None]:
    """Send messages to vLLM and stream the response.

    Args:
        messages: List of message dicts (system, user, assistant).
        server_url: vLLM OpenAI-compatible API URL (e.g., http://localhost:8000/v1).
        model_name: Model name as served by vLLM.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.

    Yields:
        Response text chunks as they arrive.
    """
    url = server_url.rstrip("/") + "/chat/completions"

    is_reasoning_model = "reasoning" in model_name.lower() or "r1" in model_name.lower()
    actual_temp = 0.7 if (is_reasoning_model and temperature == 0.1) else temperature

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": actual_temp,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if is_reasoning_model:
        payload["repetition_penalty"] = 1.05

    try:
        with httpx.stream(
            "POST",
            url,
            json=payload,
            timeout=httpx.Timeout(connect=10.0, read=600.0, write=10.0, pool=10.0),
        ) as response:
            if response.status_code != 200:
                yield f"\n\n⚠️ **Error**: LLM server returned HTTP {response.status_code}. "
                yield "Please ensure the analysis model is loaded in vLLM."
                return

            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue

                data_str = line[6:]  # Remove "data: " prefix
                if data_str.strip() == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                except json.JSONDecodeError:
                    continue

    except httpx.ConnectError:
        yield "\n\n⚠️ **Error**: Cannot connect to LLM server at "
        yield f"`{server_url}`. Please ensure vLLM is running with the analysis model loaded."
    except httpx.ReadTimeout:
        yield "\n\n⚠️ **Error**: LLM response timed out. The query may be too complex."
    except Exception as e:
        yield f"\n\n⚠️ **Error**: {str(e)}"


def query_llm(
    messages: list[dict],
    server_url: str,
    model_name: str,
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> str:
    """Send messages to vLLM and return the full response (non-streaming).

    Args:
        messages: List of message dicts.
        server_url: vLLM OpenAI-compatible API URL.
        model_name: Model name as served by vLLM.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.

    Returns:
        Full response text.
    """
    url = server_url.rstrip("/") + "/chat/completions"

    is_reasoning_model = "reasoning" in model_name.lower() or "r1" in model_name.lower()
    actual_temp = 0.7 if (is_reasoning_model and temperature == 0.1) else temperature

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": actual_temp,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if is_reasoning_model:
        payload["repetition_penalty"] = 1.05

    try:
        response = httpx.post(
            url,
            json=payload,
            timeout=httpx.Timeout(connect=10.0, read=600.0, write=10.0, pool=10.0),
        )

        if response.status_code != 200:
            return f"⚠️ Error: LLM server returned HTTP {response.status_code}."

        data = response.json()
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "No response generated.")
        return "No response generated."

    except httpx.ConnectError:
        return f"⚠️ Error: Cannot connect to LLM server at {server_url}."
    except Exception as e:
        return f"⚠️ Error: {str(e)}"


def replace_source_tags_in_string(text: str, results: list[dict]) -> str:
    """Replace abstract [Source X] references with detailed citations."""
    import re

    pattern = re.compile(
        r"\[[Ss]ources?\s*:?\s*(\d+(?:\s*(?:,|\b)\s*\d+)*)\]|\b[Ss]ources?\s+(\d+(?:\s*(?:,|\b)\s*\d+)*)\b",
        re.IGNORECASE,
    )

    def get_citation_for_idx(idx: int) -> str | None:
        if 1 <= idx <= len(results):
            result = results[idx - 1]
            parts = []

            # 1. Author
            author = result.get("author") or ""
            if author:
                parts.append(author)

            # 2. Document type
            doc_type = result.get("document_type") or ""
            if doc_type and doc_type != "unknown":
                doc_type = doc_type.replace("_", " ").title()
                parts.append(doc_type)

            # 3. Date
            date = result.get("date_extracted") or ""
            if date:
                parts.append(date)

            # 4. Page number
            page = result.get("page_number")
            if page:
                parts.append(f"p. {page}")

            # 5. Identifying report details if present in chunk text
            chunk_text = result.get("text", "")
            ref_match = re.search(
                r"\b(?:Ref(?:\s*No)?\.?\s*:\s*|Accession(?:\s*Number)?\.?\s*:\s*)([A-Z0-9_\-]+(?:\.[A-Z0-9_\-]+)*)",
                chunk_text,
                re.IGNORECASE,
            )
            if ref_match:
                ref_val = ref_match.group(0).strip()
                ref_val = ref_val.rstrip(",.;:")
                parts.append(ref_val)

            # Check if original filename is present, and append if details are minimal
            filename = result.get("original_filename") or ""
            if filename and len(parts) < 2 and filename not in parts:
                parts.append(filename)

            if parts:
                return ", ".join(parts)
            else:
                return f"Source {idx}"
        return None

    def replacer(match):
        bracketed_str = match.group(1)
        word_str = match.group(2)
        source_str = bracketed_str or word_str
        if not source_str:
            return match.group(0)

        indices = [int(num) for num in re.findall(r"\d+", source_str)]
        citations = []
        for idx in indices:
            cit = get_citation_for_idx(idx)
            if cit:
                citations.append(cit)
            else:
                citations.append(f"Source {idx}")
        if citations:
            return "(" + "; ".join(citations) + ")"
        return match.group(0)

    return pattern.sub(replacer, text)


def replace_source_tags_streaming(generator, results: list[dict]) -> Generator[str, None, None]:
    """Wraps an LLM streaming generator and replaces source tags on the fly."""
    import re

    buffer = ""
    for chunk in generator:
        if not chunk:
            continue
        buffer += chunk

        while True:
            start_idx = buffer.find("[")
            if start_idx == -1:
                # No bracket. Check if there's a potential unbracketed Source prefix at the end of the buffer.
                lower_buf = buffer.lower()
                last_source_idx = -1
                for prefix in ["source", "sources"]:
                    idx = lower_buf.rfind(prefix)
                    if idx > last_source_idx:
                        last_source_idx = idx

                if last_source_idx != -1:
                    suffix = buffer[last_source_idx:]
                    if re.match(r"^[Ss]ources?\s*\d*$", suffix):
                        if last_source_idx > 0:
                            yield replace_source_tags_in_string(buffer[:last_source_idx], results)
                            buffer = buffer[last_source_idx:]
                        break

                yield replace_source_tags_in_string(buffer, results)
                buffer = ""
                break

            if start_idx > 0:
                yield replace_source_tags_in_string(buffer[:start_idx], results)
                buffer = buffer[start_idx:]

            close_idx = buffer.find("]")
            if close_idx == -1:
                if len(buffer) > 150:
                    yield replace_source_tags_in_string(buffer, results)
                    buffer = ""
                break
            else:
                tag = buffer[: close_idx + 1]
                replaced_tag = replace_source_tags_in_string(tag, results)
                yield replaced_tag
                buffer = buffer[close_idx + 1 :]

    if buffer:
        yield replace_source_tags_in_string(buffer, results)


def analyze(
    query: str,
    mode: str = "free_qa",
    server_url: str = "http://localhost:8000/v1",
    model_name: str = "nvidia/Phi-4-reasoning-plus-NVFP4",
    top_k: int = 8,
    chat_history: list[dict] | None = None,
    run_id_filter: str | None = None,
    doc_type_filter: str | None = None,
    author_filter: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    stream: bool = True,
    progress_callback: Any | None = None,
    **search_kwargs,
) -> Generator[str, None, None]:
    """Full RAG analysis pipeline: retrieve → prompt → generate.

    This is the main entry point for the analysis engine.

    Args:
        query: User's question or analysis request.
        mode: Analysis mode (free_qa, timeline, injury_summary, etc.).
        server_url: vLLM API URL.
        model_name: Model name served by vLLM.
        top_k: Number of chunks to retrieve.
        chat_history: Previous conversation messages.
        run_id_filter: Optional run/case ID filter for case isolation.
        doc_type_filter: Optional document type filter.
        author_filter: Optional author filter.
        date_from: Optional date range start.
        date_to: Optional date range end.
        stream: Whether to stream the response.
        progress_callback: Callback to report retrieval/rerank progress.
        **search_kwargs: Additional kwargs for search_similar().

    Yields:
        Response text chunks (if streaming) or full response.
    """
    # For structured/analytical modes, automatically increase top_k to at least 50
    # to guarantee comprehensive document coverage. A looser score threshold is
    # used so the expanded candidate pool is actually retrieved (independent of
    # whether a case filter is active).
    if mode in ["timeline", "injury_summary", "inconsistency_finder", "medication_tracker"]:
        top_k = max(top_k, STRUCTURED_MODE_MIN_TOP_K)
        if "score_threshold" not in search_kwargs:
            search_kwargs["score_threshold"] = STRUCTURED_MODE_SCORE_THRESHOLD

    # Step 1: Retrieve relevant chunks
    results = search_similar(
        query=query,
        top_k=top_k,
        run_id_filter=run_id_filter,
        doc_type_filter=doc_type_filter,
        author_filter=author_filter,
        date_from=date_from,
        date_to=date_to,
        progress_callback=progress_callback,
        **search_kwargs,
    )

    if not results:
        yield "No relevant document excerpts found in the indexed corpus. "
        yield "Please ensure documents have been indexed using the 'Build Index' button."
        return

    # Truncate context to fit the *analysis* model's max context window.
    # NOTE: we must use the analysis model's limit, not the OCR container's
    # (docker_max_model_len), which can differ on a shared vLLM server.
    from settings_manager import MODEL_MAX_CONTENT_LENGTHS, load_settings

    settings = load_settings()
    # 1) Best: a known limit for the exact analysis model name.
    # 2) Fallback: the configured OCR container limit.
    # 3) Last resort: a conservative default.
    max_model_len = (
        settings.get("docker_max_model_len")
        or MODEL_MAX_CONTENT_LENGTHS.get(model_name)
        or MODEL_MAX_CONTENT_LENGTHS.get(settings.get("analysis_model_name"))
        or 131072
    )
    max_prompt_tokens = max(int(max_model_len) - 5120, 2048)

    # Estimate base prompt and overall tokens
    def estimate_tokens(msgs: list[dict]) -> int:
        try:
            import tiktoken

            encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            encoding = None

        if encoding is None:
            total_chars = 0
            for m in msgs:
                total_chars += len(m.get("role", ""))
                content = m.get("content", "")
                if isinstance(content, str):
                    total_chars += len(content)
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            total_chars += len(item["text"])
                        elif isinstance(item, str):
                            total_chars += len(item)
            return total_chars // 4

        num_tokens = 0
        for message in msgs:
            num_tokens += 4
            for key, value in message.items():
                if isinstance(value, str):
                    num_tokens += len(encoding.encode(value))
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and "text" in item:
                            num_tokens += len(encoding.encode(item["text"]))
                        elif isinstance(item, str):
                            num_tokens += len(encoding.encode(item))
                if key == "name":
                    num_tokens += -1
        num_tokens += 2
        return num_tokens

    # Build prompt and check length
    context = format_context_for_llm(results)
    messages = build_prompt(query, context, mode, chat_history)
    estimated_total = estimate_tokens(messages)

    warning_msg = None
    if estimated_total > max_prompt_tokens:
        # Drop the least-relevant chunks (results are pre-sorted by score,
        # most relevant first) until the prompt fits, keeping at least the
        # single most relevant chunk. Linear scan from the full set down to 1
        # preserves as many chunks as the window allows rather than collapsing
        # to one chunk on the first non-fit.
        truncated_results = list(results)
        while len(truncated_results) > 1:
            trial = truncated_results[:-1]
            messages_trial = build_prompt(query, format_context_for_llm(trial), mode, chat_history)
            if estimate_tokens(messages_trial) <= max_prompt_tokens:
                truncated_results = trial
                break
            truncated_results = trial
        results = truncated_results

        warning_msg = (
            f"⚠️ **Note**: The retrieved context was too large for the model's context window "
            f"({estimated_total} estimated tokens vs limit of {max_prompt_tokens}). "
            f"It has been truncated to the top {len(results)} most relevant chunks.\n\n"
        )

    # Step 2: Format context (using final resolved results)
    context = format_context_for_llm(results)

    # Step 3: Build prompt (using final resolved context)
    messages = build_prompt(query, context, mode, chat_history)

    # Step 4: Query LLM
    resolved_model = model_name
    if os.environ.get("TESTING") != "true":
        try:
            resolved_model, fell_back = _resolve_loaded_model(server_url, model_name)
            if fell_back:
                yield (
                    f"⚠️ **Note**: Model `{model_name}` is not loaded in vLLM. "
                    f"Falling back to `{resolved_model}`.\n\n"
                )
        except Exception:
            pass

    if stream:
        if warning_msg:
            yield warning_msg
        raw_stream = query_llm_streaming(messages, server_url, resolved_model)
        yield from replace_source_tags_streaming(raw_stream, results)
    else:
        response_text = query_llm(messages, server_url, resolved_model)
        processed_text = replace_source_tags_in_string(response_text, results)
        if warning_msg:
            yield warning_msg + processed_text
        else:
            yield processed_text
