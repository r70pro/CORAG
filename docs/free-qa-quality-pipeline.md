# Free Q&A quality pipeline

Free Q&A uses deterministic planning and validation so normal requests do not
incur a second LLM call or lose streaming time-to-first-token performance.

## Request path

1. Existing hybrid/multi-facet retrieval gathers the evidence.
2. `rag.free_qa_quality.classify_free_qa` identifies factual, summary,
   comparison, chronology, broad-scope and multipart requests.
3. A compact evidence ledger labels direct excerpts versus index/reference
   evidence and flags obvious future-reference/document-date conflicts.
4. Task-specific instructions require complete coverage, calibrated
   uncertainty, primary-source preference and compact chronology formatting.
5. The completion budget adapts from 1,536 tokens for simple questions to
   4,096 tokens for broad or evidence-heavy questions, always bounded by the
   user setting and live model context.
6. Existing streaming source-tag resolution and output-limit detection remain
   in place. Non-streaming calls also log deterministic quality findings,
   including unsupported corpus-wide characterisations, speculative claims that
   dates are metadata/OCR/template errors, and fully normalised dates absent
   from the retrieved evidence.

Dates are evidence, not values for the model to repair. The drafting contract
requires dates to be preserved as written unless internal source text proves a
correction. Retrieval results also cannot establish that the entire indexed
corpus is clinically insubstantial; corpus-wide descriptions must be supported
by the deterministic evidence ledger.

## Latency properties

Classification, ledger construction and validation are local linear scans.
They make no network or model calls. Streaming still begins with the first
model token. The existing CPU reranker remains disabled for Free Q&A.

Broader questions can intentionally generate more answer tokens so they finish
instead of truncating. Their prompt tells the model to use compact tables or
entries, limiting unnecessary decoding.

## Operational metrics

Application logs include a `free_qa_plan` record with task, scope, evidence
counts and selected budget. Non-streaming responses include a
`free_qa_quality` record for truncation, invalid citation count and the unsafe
combination of a completeness claim with truncation.

Production dashboards should track time to first token, median and p95 total
latency, output-limit rate, selected output budget, invalid citations and user
feedback by task type. Suggested gates are under 5% time-to-first-token
regression, under 10% median completion regression, and under 0.5% truncation.

## Test coverage

`tests/test_free_qa_quality.py` covers task classification, adaptive budgets,
chronology rules, evidence-status/date-conflict detection, substantive-content
signals, fabricated date detection and unsupported corpus characterisations.
Analyzer integration tests verify that Free Q&A remains non-thinking and uses
the selected completion budget.

## Future conditional review

A second model review is deliberately not enabled for Free Q&A. If production
metrics justify it, add it only for flagged high-risk responses (date conflict,
invalid source identifier, high-stakes request or repeated truncation) and
measure p95 latency separately. Expert and Judge modes already provide the
explicit high-assurance two-pass path.
