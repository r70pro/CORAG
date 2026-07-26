# Phase 0 — Baseline and regression harness

Captured: 2026-07-26 (Australia/Melbourne)

## Outcome

Phase 0 establishes a red regression baseline without changing production
behaviour. The normal Python regression run reports 1 passing control and 11
strict expected failures. Forced-red mode turns those markers off and produces
11 ordinary failures. The frontend has one additional malicious-HTML contract
that passes as an expected failure normally and fails in forced-red mode.

The strict expected-failure markers are intentional temporary controls. Once a
defect is fixed, its marker must be removed in the same change; strict mode
causes an unexpected pass to fail the suite rather than silently hiding it.

## Critical disclosure chain

`PHASE0-SEC-004` proves that API authentication fails open when
`KIRAG_API_KEY` is unset. An unauthenticated caller can then reach the document
information endpoint, where `PHASE0-SEC-001` and `PHASE0-SEC-002` prove that an
absolute filename, including one supplied URL-encoded, escapes the selected run
and reads arbitrary UTF-8 files. `PHASE0-SEC-003` proves a second escape through
a symlink placed under a run's markdown directory.

The existing literal `..` rejection is retained as a passing control. It does
not mitigate absolute paths or symlink resolution.

## Regression inventory

| Contract | Risk | Expected secure/correct behaviour | Current evidence |
|---|---|---|---|
| `PHASE0-SEC-001` | Critical | Reject absolute document filenames | Does not raise; outside file is read |
| `PHASE0-SEC-002` | Critical | Reject a URL-decoded path that resolves outside the run | Does not raise after URL decoding |
| `PHASE0-SEC-003` | Critical | Reject symlinks escaping the run root | Symlink target is followed |
| `PHASE0-SEC-004` | Critical chain enabler | Reject unauthenticated API access even when configuration is missing | Verification returns successfully |
| literal `..` control | Security control | Reject parent-directory components | Passes with HTTP 400 |
| `PHASE0-SEC-005` | High | Enforce `KIRAG_MAX_UPLOAD_BYTES` before persistence | Oversized body is buffered and written |
| `PHASE0-SEC-005B` | High | Apply the same limit to Markdown ingestion | Oversized Markdown is buffered before indexing |
| `PHASE0-SEC-006` | High | Escape untrusted case metadata before HTML interpolation | Active `<img onerror>` is emitted |
| frontend malicious HTML | High | Render document HTML only after sanitisation | `<img onerror>` is inserted into the DOM |
| `PHASE0-IDX-001` | High correctness/data loss | Add only new vectors and retain every existing case vector | Existing run is passed to `pre_delete_run_ids`; old vectors are cleared |
| `PHASE0-PROV-001` | High correctness | Preserve leading-whitespace offsets and page mapping | Expected offset/page `50/2`, recorded `0/1` |
| `PHASE0-PROV-002` | High correctness | Keep chunk text as exact source slices with unique ranges | Paragraph normalisation creates duplicate and incorrect ranges |
| `PHASE0-PROV-003` | Release/citation quality | Record both page boundaries for a cross-page chunk | Only the starting `page_number` exists |

The provenance inputs are durable JSON fixtures covering leading whitespace,
paragraph normalisation, and a chunk crossing two PDF pages.

## Test evidence

Normal Python baseline:

```text
.venv/bin/python -m pytest tests/regression -q -rxX
1 passed, 11 xfailed
```

Forced-red Python baseline:

```text
.venv/bin/python -m pytest tests/regression --runxfail -q
11 failed, 1 passed
```

Normal frontend baseline:

```text
npm test -- --runInBand src/components/__tests__/PdfInspector.test.tsx
2 passed
```

Forced-red frontend baseline:

```text
PHASE0_RUN_FAILING=1 npm test -- --runInBand \
  src/components/__tests__/PdfInspector.test.tsx
1 failed, 1 passed
```

The forced-red frontend failure identifies the injected element as:

```text
<img onerror="window.phase0Xss=1" src="x" />
```

Static validation:

```text
.venv/bin/ruff check tests/regression \
  scripts/capture_reconciliation_baseline.py \
  scripts/create_phase0_object_backups.py
npm run typecheck
```

Both pass.

Repository-wide validation:

```text
.venv/bin/python -m pytest -q
1 failed, 511 passed, 11 xfailed

npm test -- --runInBand
10 suites passed, 25 tests passed
```

The single Python failure predates this harness:
`tests/test_api.py::TestAPI::test_consolidated_case_summary` mocks
`get_indexed_runs`, while the endpoint now calls `get_runs_with_stats`. The
mock therefore misses the database call and the assertion receives an error
response without `stats`. It is recorded as a release-quality test/API drift
issue, not attributed to the Phase 0 additions.

## Reconciliation baseline

The read-only capture at `2026-07-26T03:42:41.520071+00:00` records:

| Store | Baseline |
|---|---:|
| PostgreSQL runs | 0 |
| PostgreSQL chunks | 0 |
| PostgreSQL vector references | 0 |
| Qdrant collections | 1 |
| Qdrant vectors | 0 |
| Per-run PostgreSQL/Qdrant deltas | None; no registered runs |

The collection is `olmocr_documents_baai_bge-large-en-v1_5`. MinIO is not
empty despite the empty relational registry: it contains 9 objects in 2
buckets, totalling 7,638,528 bytes. That is preserved as baseline state and
should be treated as possible orphaned object-store data during a later
reconciliation phase.

The capture script emits per-run PostgreSQL document/chunk/vector-reference
counts, per-collection Qdrant vector counts, deltas, and Qdrant-only run IDs.
It can be rerun after each indexing operation:

```text
.venv/bin/python scripts/capture_reconciliation_baseline.py \
  --output reports/phase0/reconciliation-after.json
```

## Backup evidence

Backup root:

```text
backups/phase0_20260726T134340+1000
```

The directory is excluded from Git and contains:

| Artifact | Verification |
|---|---|
| `workspace.tar.zst` (3,103,482,210 bytes) | `zstd -t` passed; archive listing includes PostgreSQL, Redis, Qdrant, and MinIO volume trees |
| `postgresql.dump` | PostgreSQL custom dump; `pg_restore --list` passed with 23 TOC entries |
| Qdrant full snapshot (67,655,680 bytes) | Downloaded from Qdrant and validated as a readable POSIX tar archive |
| MinIO logical export | 9/9 objects copied; bucket, key, ETag, and size recorded |
| `config/.env` | Copied with mode `0600` |
| `config/settings.json` | Copied with mode `0600` |
| reconciliation JSON | Copied into the backup set |
| `SHA256SUMS` | Every backed-up file passes `sha256sum -c` |

PostgreSQL, Qdrant, MinIO, and Redis were stopped before Phase 0, started only
for consistent logical capture, and restored to the stopped state after the
backup. Final container states are all `Exited`.

## Release-quality observations retained for later phases

- The API's missing-key behaviour is an insecure configuration default, not
  merely a route-local validation bug.
- Upload handling buffers entire bodies before persistence and has no
  documented or enforced size policy.
- HTML safety differs across the Python dashboard and React preview, requiring
  a shared trust/sanitisation policy.
- Chunk provenance currently cannot support exact source highlighting or
  unambiguous citations for cross-page chunks.
- PostgreSQL/Qdrant are empty while MinIO retains objects, so future indexing
  and cleanup work must not assume all stores have matching lifecycle state.
- Existing-case indexing and its rollback both operate at run scope, making a
  single-document addition capable of removing unrelated vectors.

## Phase 0 artifacts

- `tests/regression/test_phase0_security_regressions.py`
- `tests/regression/test_phase0_indexing_regressions.py`
- `tests/regression/test_phase0_provenance_regressions.py`
- `tests/regression/fixtures/provenance_cases.json`
- `frontend/src/components/__tests__/PdfInspector.test.tsx`
- `scripts/capture_reconciliation_baseline.py`
- `scripts/create_phase0_object_backups.py`
- `reports/phase0/reconciliation-baseline.json`
- `backups/phase0_20260726T134340+1000`

No application behaviour was changed in Phase 0.
