"""
LLM Analyzer — prompt assembly and vLLM OpenAI-compatible API integration.

Provides:
- System prompt templates for medicolegal analysis modes
- Context-aware prompt assembly from retrieved chunks
- Streaming chat completions via vLLM's OpenAI-compatible API
- Pre-built analysis templates (timeline, summary, inconsistencies)
"""

import json
import logging
import os
from collections.abc import Generator
from functools import lru_cache
from typing import Any

import httpx

from rag.analysis_policy import get_analysis_policy
from rag.retriever import format_context_for_llm, search_comprehensive, search_similar
from rag.upstream import CircuitOpenError, request_with_retry

logger = logging.getLogger(__name__)

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
CONSERVATIVE_ANALYSIS_CONTEXT_LENGTH = 32768
# KIRAG's generic token estimator can under-count a served model's native
# tokenizer, and vLLM then adds that model's chat-template special tokens.
# Keep a conservative fixed margin so a prompt estimated near the apparent
# boundary is not rejected by the server after native tokenization.
NATIVE_CHAT_TEMPLATE_TOKEN_RESERVE = 64
GENERIC_TOKEN_ESTIMATE_RESERVE = 1024
# Backward-compatible public name used by validation tests and callers.
CHAT_TEMPLATE_TOKEN_RESERVE = GENERIC_TOKEN_ESTIMATE_RESERVE


class ContextWindowError(ValueError):
    """Raised when a requested generation cannot fit in the analysis context window."""


@lru_cache(maxsize=8)
def _get_local_analysis_tokenizer(model_name: str):
    """Load a cached model tokenizer without initiating a network download."""
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception:
        return None


def _analysis_context_length(
    resolved_model: str,
    configured_analysis_model: object,
    model_lengths: dict[str, int],
) -> int:
    """Resolve context length without consulting OCR/container settings."""
    exact_length = model_lengths.get(resolved_model)
    if exact_length is not None:
        return int(exact_length)

    if isinstance(configured_analysis_model, str):
        configured_length = model_lengths.get(configured_analysis_model)
        if configured_length is not None:
            return int(configured_length)

    return CONSERVATIVE_ANALYSIS_CONTEXT_LENGTH


def _validate_output_token_request(max_tokens: int, max_model_len: int) -> None:
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise ContextWindowError("max_tokens must be a positive integer")
    if max_tokens >= max_model_len:
        raise ContextWindowError(
            f"Requested output tokens ({max_tokens}) must be smaller than the "
            f"analysis model context window ({max_model_len})"
        )


def _validate_managed_context_invariant(
    server_url: str,
    served_context: int,
    full_model_context: int,
) -> None:
    """Fail closed when managed OCR and analysis context states disagree."""
    if os.environ.get("TESTING") == "true" or ":8002" not in server_url:
        return
    try:
        from docker_manager import get_docker_status

        ocr_status = get_docker_status("olmocr")
    except Exception as exc:
        raise ContextWindowError("Unable to verify OCR state for analysis context allocation") from exc
    if ocr_status == "running":
        if served_context != 32768:
            raise ContextWindowError(
                f"OCR is active but analysis is serving {served_context:,} tokens; expected 32,768"
            )
    elif ocr_status in {"exited", "stopped", "created"}:
        if served_context != full_model_context:
            raise ContextWindowError(
                f"OCR is inactive but analysis is serving {served_context:,} tokens; "
                f"expected the full {full_model_context:,}-token model allocation"
            )
    else:
        raise ContextWindowError(
            f"OCR state '{ocr_status}' is not stable enough to start analysis"
        )


def invalidate_model_cache():
    """Clear the model resolution cache.

    Call this after a vLLM container is recreated so that subsequent
    queries re-probe the server instead of using stale cached model IDs.
    """
    _model_cache.clear()


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
    response = request_with_retry(lambda: httpx.get(url, timeout=2.0))
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

    fallback = loaded_models[0]
    logger.warning(
        "Model '%s' not loaded in vLLM at %s. Falling back to '%s'.",
        model_name,
        server_url,
        fallback,
    )
    return fallback, True


def _get_served_model_context_length(server_url: str, model_name: str) -> int | None:
    """Return the live vLLM context limit when the models endpoint supplies it."""
    try:
        response = request_with_retry(
            lambda: httpx.get(server_url.rstrip("/") + "/models", timeout=2.0)
        )
        if response.status_code != 200:
            return None
        for model in response.json().get("data", []):
            if model.get("id") != model_name:
                continue
            value = model.get("max_model_len")
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
    except Exception:
        pass
    return None


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
    "general_knowledge": """You are a general-purpose assistant operating in General Knowledge mode.

Answer the user's question using your general knowledge and reasoning. No case documents or retrieved excerpts are available in this mode.

INSTRUCTIONS:
- Do not claim that your answer is based on the user's indexed documents
- Do not generate document citations, PDF page references, source tags, or case-specific facts
- If the user asks about a particular case or document, explain that General Knowledge mode cannot inspect it and ask them to use an appropriate RAG mode
- Clearly distinguish established information from uncertainty or opinion
- For medical, legal, or other high-stakes questions, provide general educational information rather than personalised professional advice
- Use clear examples where they improve understanding""",
    "free_qa": """You are a medicolegal document analyst with expertise in personal injury, workers' compensation, and clinical documentation. You have been provided with excerpts from clinical records, specialist reports, and correspondence.

INSTRUCTIONS:
- Answer based ONLY on the provided document excerpts — do not hallucinate or assume facts not present in the sources
- Use the supplied [Source N] tag for each factual claim; the application replaces it with verified metadata before display
- Cite an exact PDF page range only when both page endpoints are supplied
- Include robust verification details for every factual claim so that users can instantly verify the source when scrolling through the original file, including:
  * The source-supported document type and original filename
  * The source-supported authoring physician or explicitly labeled clinic
  * Identifying report details only when present in the excerpt
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
  * Use the supplied [Source N] tag; the application replaces it with verified metadata before display
  * Cite an exact PDF page range only when both page endpoints are supplied
  * Include robust verification details for each entry so that users can instantly verify the source when scrolling through the original file, including:
    - The source-supported document type and original filename
    - The source-supported authoring physician or explicitly labeled clinic
    - Identifying report details only when present in the excerpt
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
- Use the supplied [Source N] tag; the application replaces it with verified metadata before display
- Cite an exact PDF page range only when both page endpoints are supplied
- Include robust verification details so that users can instantly verify the source when scrolling through the original file, including:
  * The source-supported document type and original filename
  * The source-supported authoring physician or explicitly labeled clinic
  * Identifying report details only when present in the excerpt
Flag any contradictions between providers.""",
    "inconsistency_finder": """You are a medicolegal document auditor specialising in identifying inconsistencies, contradictions, and discrepancies across clinical records.

INSTRUCTIONS:
- Compare accounts of the same events across different sources
- Identify discrepancies in: dates, injury descriptions, examination findings, treatment recommendations, patient-reported symptoms
- For each inconsistency, cite both sources with:
  * The exact original-PDF page range only when both endpoints are supplied
  * Source-supported document type, filename, author/clinic, and reference details
  * The supplied [Source N] tag, which the application replaces before display
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
  * Use the supplied [Source N] tag; the application replaces it with verified metadata before display
  * Cite an exact PDF page range only when both page endpoints are supplied
  * Include robust verification details for each entry so that users can instantly verify the source when scrolling through the original file, including:
    - The source-supported document type and original filename
    - The source-supported authoring physician or explicitly labeled clinic
    - Identifying report details only when present in the excerpt
- Flag any potential interactions or contraindications
- Note any allergies mentioned in the records""",
    "causation": """You are a senior medicolegal analyst assessing causation. Analyse temporal sequence, mechanism, objective findings, pre-existing conditions, alternative and intervening causes, and all treating or expert opinions. Separate documented fact, quoted clinical opinion, and your evidence-grounded inference. Address supporting and contrary evidence, missing evidence, and uncertainty. Do not express a conclusion more strongly than the records permit.""",
    "prognosis": """You are a senior medicolegal analyst assessing prognosis. Analyse longitudinal symptoms, objective findings, response to treatment, functional trajectory, prognostic opinions, barriers to recovery, and uncertainty. Distinguish documented facts, clinician opinions, and evidence-grounded inference; address both favourable and adverse evidence.""",
    "work_capacity": """You are a senior medicolegal analyst assessing work capacity. Analyse pre-injury duties, certified restrictions, functional evidence, attempted returns, employer accommodations, treating and independent opinions, and changes over time. Distinguish fact, clinical opinion, and inference and identify conflicts and missing vocational evidence.""",
    "treatment_planning": """You are a senior medicolegal analyst reviewing treatment planning. Analyse documented treatment, response, outstanding recommendations, contraindications, competing recommendations, and evidentiary gaps. Describe record-supported considerations rather than prescribing care. Distinguish facts, clinician recommendations, and evidence-grounded inference.""",
}

PROVENANCE_INSTRUCTIONS = """

NON-NEGOTIABLE PROVENANCE RULES:
- Treat the provenance fields in each excerpt header as the complete source of citation metadata.
- Never infer or invent a PDF page, page range, provider, author, clinic, filename, document type, claim/reference number, or accession number.
- If a field is unavailable, omit it or write "Not present in source".
- If an excerpt says it is external Markdown, state that it has no original-PDF page provenance; do not assign it a page.
- Retain the source's original date expression. Only present a normalized ISO date when it is a valid calendar date.
"""


ANALYSIS_MODE_MAP = {
    "🌐 General Knowledge": "general_knowledge",
    "general_knowledge": "general_knowledge",
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
    "🧬 Causation Analysis": "causation",
    "causation": "causation",
    "📈 Prognosis Analysis": "prognosis",
    "prognosis": "prognosis",
    "🧑‍💼 Work Capacity": "work_capacity",
    "work_capacity": "work_capacity",
    "🩺 Treatment Planning": "treatment_planning",
    "treatment_planning": "treatment_planning",
}


def get_analysis_modes():
    """Get available analysis modes and their descriptions."""
    return {
        "general_knowledge": "🌐 General Knowledge — Chat without document retrieval",
        "free_qa": "💬 Free Q&A — Ask anything about the documents",
        "timeline": "📅 Timeline Generator — Extract chronological events",
        "injury_summary": "🏥 Injury Summary — Structured injury/treatment report",
        "inconsistency_finder": "🔍 Inconsistency Finder — Cross-reference discrepancies",
        "medication_tracker": "💊 Medication Tracker — Track all medication references",
        "causation": "🧬 Causation Analysis — Assess competing causal evidence",
        "prognosis": "📈 Prognosis Analysis — Assess likely clinical and functional course",
        "work_capacity": "🧑‍💼 Work Capacity — Assess evidence of capacity and restrictions",
        "treatment_planning": "🩺 Treatment Planning — Review documented treatment needs",
    }


OUTPUT_LIMIT_WARNING = (
    "\n\n> ⚠️ **Incomplete response:** generation reached the configured output-token "
    "limit. Increase Maximum Output Tokens and regenerate before relying on or exporting "
    "this analysis."
)


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
    if mode != "general_knowledge":
        system_prompt += PROVENANCE_INSTRUCTIONS

    messages = [{"role": "system", "content": system_prompt}]

    # Add chat history for multi-turn conversations
    if chat_history:
        for msg in chat_history[-6:]:  # Keep last 6 messages to manage context window
            messages.append(msg)

    if mode == "general_knowledge":
        messages.append({"role": "user", "content": query})
        return messages

    # Build the user message with retrieved document context.
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
    max_tokens: int | None = None,
    enable_thinking: bool = False,
    reasoning_callback: Any | None = None,
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

    normalized_model_name = model_name.lower()
    is_reasoning_model = "reasoning" in normalized_model_name or "r1" in normalized_model_name
    is_qwen3_model = "qwen3" in normalized_model_name
    actual_temp = 0.7 if (is_reasoning_model and temperature == 0.1) else temperature

    if max_tokens is None:
        # analyze() supplies the exact live remaining capacity. Direct callers
        # use the conservative model allocation without an extra network probe.
        max_tokens = CONSERVATIVE_ANALYSIS_CONTEXT_LENGTH - 1
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": actual_temp,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if is_reasoning_model:
        payload["repetition_penalty"] = 1.05
    if is_qwen3_model:
        payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

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

            hit_output_limit = False
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
                        choice = choices[0]
                        if choice.get("finish_reason") == "length":
                            hit_output_limit = True
                        delta = choice.get("delta", {})
                        reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                        if reasoning:
                            if enable_thinking:
                                if reasoning_callback:
                                    reasoning_callback(reasoning)
                            else:
                                # vLLM's Qwen parser can classify a non-thinking
                                # final answer as reasoning. In extraction modes
                                # it is answer content, never an audit trace.
                                yield reasoning
                        content = delta.get("content", "")
                        if content:
                            yield content
                except json.JSONDecodeError:
                    continue
            if hit_output_limit:
                yield OUTPUT_LIMIT_WARNING

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
    max_tokens: int | None = None,
    enable_thinking: bool = False,
    reasoning_callback: Any | None = None,
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

    normalized_model_name = model_name.lower()
    is_reasoning_model = "reasoning" in normalized_model_name or "r1" in normalized_model_name
    is_qwen3_model = "qwen3" in normalized_model_name
    actual_temp = 0.7 if (is_reasoning_model and temperature == 0.1) else temperature

    if max_tokens is None:
        max_tokens = CONSERVATIVE_ANALYSIS_CONTEXT_LENGTH - 1
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": actual_temp,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if is_reasoning_model:
        payload["repetition_penalty"] = 1.05
    if is_qwen3_model:
        payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

    try:
        response = request_with_retry(
            lambda: httpx.post(
                url,
                json=payload,
                timeout=httpx.Timeout(connect=10.0, read=600.0, write=10.0, pool=10.0),
            )
        )

        if response.status_code != 200:
            return f"⚠️ Error: LLM server returned HTTP {response.status_code}."

        data = response.json()
        choices = data.get("choices", [])
        if choices:
            choice = choices[0]
            message = choice.get("message", {})
            reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
            if reasoning and enable_thinking and reasoning_callback:
                reasoning_callback(reasoning)
            content = message.get("content")
            if not content and reasoning and not enable_thinking:
                content = reasoning
            if not content:
                content = "No response generated."
            if choice.get("finish_reason") == "length":
                return content + OUTPUT_LIMIT_WARNING
            return content
        return "No response generated."

    except httpx.ConnectError:
        return f"⚠️ Error: Cannot connect to LLM server at {server_url}."
    except CircuitOpenError:
        return "⚠️ Error: LLM service is temporarily unavailable; retry after the recovery window."
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

            filename = result.get("original_filename")
            if filename:
                parts.append(str(filename))

            # Only cite author and document type when source extraction supplied them.
            author = result.get("author") or ""
            if author:
                parts.append(author)

            doc_type = result.get("document_type") or ""
            if doc_type and doc_type != "unknown":
                doc_type = doc_type.replace("_", " ").title()
                parts.append(doc_type)

            date = result.get("date_extracted") or ""
            if date:
                parts.append(date)

            provenance_type = result.get("provenance_type")
            page_start = result.get("page_start")
            page_end = result.get("page_end")
            if provenance_type == "external_markdown":
                parts.append("external Markdown; no original-PDF page provenance")
            elif page_start is not None and page_end is not None:
                if page_start == page_end:
                    parts.append(f"p. {page_start}")
                else:
                    parts.append(f"pp. {page_start}-{page_end}")
            elif page_start is not None:
                parts.append(
                    f"p. {page_start} (start page only; end page not present in source metadata)"
                )
            else:
                parts.append("original-PDF page provenance not present")

            chunk_text = result.get("text", "")
            ref_match = re.search(
                r"\b(?:(?:Ref|Claim)(?:erence)?(?:\s*(?:No|Number))?\.?\s*:\s*|"
                r"Accession(?:\s*(?:No|Number))?\.?\s*:\s*)"
                r"([A-Z0-9_\-]+(?:\.[A-Z0-9_\-]+)*)",
                chunk_text,
                re.IGNORECASE,
            )
            if ref_match:
                ref_val = ref_match.group(0).strip()
                ref_val = ref_val.rstrip(",.;:")
                parts.append(ref_val)

            if parts:
                return ", ".join(parts)
            else:
                return f"Source {idx}; provenance metadata not present"
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
    max_tokens: int | None = None,
    progress_callback: Any | None = None,
    reasoning_callback: Any | None = None,
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
        max_tokens: Maximum number of response tokens to request from the analysis model.
        progress_callback: Callback to report retrieval/rerank progress.
        **search_kwargs: Additional kwargs for search_similar().

    Yields:
        Response text chunks (if streaming) or full response.
    """
    policy = get_analysis_policy(mode)
    mode = policy.mode
    results: list[dict] = []
    if policy.uses_retrieval:
        top_k = max(top_k, policy.min_top_k)
        if "score_threshold" not in search_kwargs:
            search_kwargs["score_threshold"] = policy.score_threshold

        # Step 1: Retrieve relevant chunks. General Knowledge bypasses this block.
        search_function = search_comprehensive if policy.comprehensive_retrieval else search_similar
        comprehensive_kwargs = {"search_function": search_similar} if policy.comprehensive_retrieval else {}
        results = search_function(
            query=query,
            top_k=top_k,
            run_id_filter=run_id_filter,
            doc_type_filter=doc_type_filter,
            author_filter=author_filter,
            date_from=date_from,
            date_to=date_to,
            progress_callback=progress_callback,
            **comprehensive_kwargs,
            **search_kwargs,
        )

        if not results:
            yield "No relevant document excerpts found in the indexed corpus. "
            yield "Please ensure documents have been indexed using the 'Build Index' button."
            return

        if progress_callback:
            progress_callback(0.82, f"Preparing {len(results)} retrieved excerpts for analysis…")
    elif progress_callback:
        progress_callback(0.82, "Preparing general-knowledge conversation…")

    # Resolve the model used for analysis before calculating its prompt budget.
    resolved_model = model_name
    model_fallback_warning = None
    if os.environ.get("TESTING") != "true":
        try:
            resolved_model, fell_back = _resolve_loaded_model(server_url, model_name)
            if fell_back:
                model_fallback_warning = (
                    f"⚠️ **Note**: Model `{model_name}` is not loaded in vLLM. "
                    f"Falling back to `{resolved_model}`.\n\n"
                )
        except Exception:
            pass

    # Truncate context to fit the analysis model. OCR container configuration
    # is deliberately excluded because it may describe a different model.
    from settings_manager import MODEL_MAX_CONTENT_LENGTHS, load_settings

    settings = load_settings()
    max_model_len = _analysis_context_length(
        resolved_model,
        settings.get("analysis_model_name"),
        MODEL_MAX_CONTENT_LENGTHS,
    )
    if os.environ.get("TESTING") != "true":
        served_context_length = _get_served_model_context_length(server_url, resolved_model)
        if served_context_length is not None:
            max_model_len = served_context_length
            full_context = int(MODEL_MAX_CONTENT_LENGTHS.get(resolved_model, max_model_len))
            _validate_managed_context_invariant(server_url, max_model_len, full_context)
    if max_tokens is not None:
        _validate_output_token_request(max_tokens, max_model_len)
    analysis_tokenizer = _get_local_analysis_tokenizer(resolved_model)
    token_reserve = (
        NATIVE_CHAT_TEMPLATE_TOKEN_RESERVE
        if analysis_tokenizer is not None
        else GENERIC_TOKEN_ESTIMATE_RESERVE
    )
    # Preserve generation room while allowing evidence to use the full live
    # model allocation. The final request receives every token left after the
    # actual prompt; there is no fixed application-wide completion ceiling.
    minimum_generation_tokens = min(8192 if policy.enable_thinking else 4096, max_model_len // 3)
    max_prompt_tokens = max_model_len - minimum_generation_tokens - token_reserve
    if max_tokens is not None:
        max_prompt_tokens = min(max_prompt_tokens, max_model_len - max_tokens - token_reserve)
    if max_prompt_tokens < 1:
        raise ContextWindowError(
            f"Requested output tokens ({max_tokens}) leave no room for the analysis "
            "prompt after chat-template overhead"
        )

    # Estimate base prompt and overall tokens
    def estimate_tokens(msgs: list[dict]) -> int:
        if analysis_tokenizer is not None:
            try:
                encoded = analysis_tokenizer.apply_chat_template(
                    msgs,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=policy.enable_thinking,
                )
                input_ids = encoded.get("input_ids", encoded)
                return len(input_ids)
            except Exception:
                pass

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
    context = format_context_for_llm(results) if policy.uses_retrieval else ""
    messages = build_prompt(query, context, mode, chat_history)
    estimated_total = estimate_tokens(messages)

    warning_msg = None
    if estimated_total > max_prompt_tokens and policy.uses_retrieval:
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
    elif estimated_total > max_prompt_tokens:
        raise ContextWindowError(
            "The general-knowledge prompt and recent chat history exceed the model context window"
        )

    # Step 2: Format context (using final resolved results)
    context = format_context_for_llm(results) if policy.uses_retrieval else ""

    # Step 3: Build prompt (using final resolved context)
    messages = build_prompt(query, context, mode, chat_history)
    final_prompt_tokens = estimate_tokens(messages)
    remaining_context = max_model_len - final_prompt_tokens - token_reserve
    requested_generation_tokens = remaining_context if max_tokens is None else min(max_tokens, remaining_context)
    if requested_generation_tokens < 1:
        raise ContextWindowError(
            "The analysis prompt leaves no context available for model generation"
        )

    # Step 4: Query LLM
    if progress_callback:
        generation_message = (
            "Generating the source-grounded answer…"
            if policy.uses_retrieval
            else "Generating a general-knowledge answer without document retrieval…"
        )
        progress_callback(0.9, generation_message)
    if model_fallback_warning:
        yield model_fallback_warning

    if stream:
        if warning_msg:
            yield warning_msg
        raw_stream = query_llm_streaming(
            messages, server_url, resolved_model, max_tokens=requested_generation_tokens,
            enable_thinking=policy.enable_thinking, reasoning_callback=reasoning_callback,
        )
        if policy.uses_retrieval:
            yield from replace_source_tags_streaming(raw_stream, results)
        else:
            yield from raw_stream
    else:
        response_text = query_llm(
            messages, server_url, resolved_model, max_tokens=requested_generation_tokens,
            enable_thinking=policy.enable_thinking, reasoning_callback=reasoning_callback,
        )
        processed_text = (
            replace_source_tags_in_string(response_text, results)
            if policy.uses_retrieval
            else response_text
        )
        if warning_msg:
            yield warning_msg + processed_text
        else:
            yield processed_text
