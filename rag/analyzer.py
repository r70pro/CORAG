"""
LLM Analyzer — prompt assembly and vLLM OpenAI-compatible API integration.

Provides:
- System prompt templates for medicolegal analysis modes
- Context-aware prompt assembly from retrieved chunks
- Streaming chat completions via vLLM's OpenAI-compatible API
- Pre-built analysis templates (timeline, summary, inconsistencies)
"""

import os
import json
import httpx
from typing import List, Dict, Optional, Generator

from rag.retriever import search_similar, format_context_for_llm


# ── System prompt templates ───────────────────────────────────

SYSTEM_PROMPTS = {
    "free_qa": """You are a medicolegal document analyst with expertise in personal injury, workers' compensation, and clinical documentation. You have been provided with excerpts from clinical records, specialist reports, and correspondence.

INSTRUCTIONS:
- Answer based ONLY on the provided document excerpts — do not hallucinate or assume facts not present in the sources
- Cite the specific source number [Source N], file name, page number, and date for every factual claim
- If multiple sources discuss the same event, synthesise the information and note any differences
- Use ISO date format (YYYY-MM-DD) when referencing dates
- If the answer cannot be determined from the provided excerpts, say so explicitly and suggest what additional documents might help
- Use clear, professional language appropriate for medicolegal analysis""",

    "timeline": """You are a medicolegal chronology specialist. Your task is to extract every dated event from the provided document excerpts and present them in strict chronological order.

INSTRUCTIONS:
- Extract EVERY event with a date (consultations, injuries, surgeries, referrals, reports, diagnoses, medication changes)
- Present as a markdown table with columns: Date | Event | Provider/Author | Source
- Use ISO date format (YYYY-MM-DD) for all dates
- If a date is ambiguous (e.g., "early 2018"), note the ambiguity but place it approximately
- Include the source reference [Source N] for each entry
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

Cite [Source N] for every claim. Flag any contradictions between providers.""",

    "inconsistency_finder": """You are a medicolegal document auditor specialising in identifying inconsistencies, contradictions, and discrepancies across clinical records.

INSTRUCTIONS:
- Compare accounts of the same events across different sources
- Identify discrepancies in: dates, injury descriptions, examination findings, treatment recommendations, patient-reported symptoms
- For each inconsistency, cite both sources with their respective claims
- Rate severity: MINOR (date formatting differences), MODERATE (differing clinical findings), MAJOR (contradictory diagnoses or recommendations)
- Present findings in a structured table: Issue | Source A Says | Source B Says | Severity
- Also note any gaps — events referenced but not documented""",

    "medication_tracker": """You are a clinical pharmacology analyst. Your task is to extract and track all medication references from the provided document excerpts.

INSTRUCTIONS:
- Extract every medication mentioned (name, dose, frequency, route, indication)
- Note the date and source where each medication is mentioned
- Track changes: new prescriptions, dose changes, cessations
- Present as a markdown table: Medication | Dose/Frequency | Date Started | Date Stopped | Prescriber | Source
- Flag any potential interactions or contraindications
- Note any allergies mentioned in the records""",
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
    chat_history: Optional[List[Dict]] = None,
) -> List[Dict]:
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
    messages: List[Dict],
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
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0),
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
    messages: List[Dict],
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
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0),
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


def analyze(
    query: str,
    mode: str = "free_qa",
    server_url: str = "http://localhost:8000/v1",
    model_name: str = "nvidia/Phi-4-reasoning-plus-NVFP4",
    top_k: int = 8,
    chat_history: Optional[List[Dict]] = None,
    doc_type_filter: Optional[str] = None,
    author_filter: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    stream: bool = True,
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
        doc_type_filter: Optional document type filter.
        author_filter: Optional author filter.
        date_from: Optional date range start.
        date_to: Optional date range end.
        stream: Whether to stream the response.
        **search_kwargs: Additional kwargs for search_similar().

    Yields:
        Response text chunks (if streaming) or full response.
    """
    # Step 1: Retrieve relevant chunks
    results = search_similar(
        query=query,
        top_k=top_k,
        doc_type_filter=doc_type_filter,
        author_filter=author_filter,
        date_from=date_from,
        date_to=date_to,
        **search_kwargs,
    )

    if not results:
        yield "No relevant document excerpts found in the indexed corpus. "
        yield "Please ensure documents have been indexed using the 'Build Index' button."
        return

    # Step 2: Format context
    context = format_context_for_llm(results)

    # Step 3: Build prompt
    messages = build_prompt(query, context, mode, chat_history)

    # Step 4: Query LLM
    resolved_model = model_name
    if os.environ.get("TESTING") != "true":
        try:
            url = server_url.rstrip("/") + "/models"
            response = httpx.get(url, timeout=2.0)
            if response.status_code == 200:
                data = response.json()
                loaded_models = [m["id"] for m in data.get("data", [])]
                if loaded_models:
                    # Normalize/map known equivalent models to avoid unnecessary fallbacks
                    equivalents = {
                        "microsoft/Phi-4-reasoning-plus": "nvidia/Phi-4-reasoning-plus-NVFP4",
                    }
                    def is_equivalent(m1: str, m2: str) -> bool:
                        if m1 == m2:
                            return True
                        if equivalents.get(m1) == m2:
                            return True
                        if equivalents.get(m2) == m1:
                            return True
                        return False

                    if model_name in loaded_models:
                        resolved_model = model_name
                    else:
                        equivalent_model = None
                        for lm in loaded_models:
                            if is_equivalent(model_name, lm):
                                equivalent_model = lm
                                break
                        if equivalent_model:
                            resolved_model = equivalent_model
                        else:
                            resolved_model = loaded_models[0]
                            yield f"⚠️ **Note**: Model `{model_name}` is not loaded in vLLM. Falling back to `{resolved_model}`.\n\n"
        except Exception:
            pass

    if stream:
        yield from query_llm_streaming(messages, server_url, resolved_model)
    else:
        yield query_llm(messages, server_url, resolved_model)

