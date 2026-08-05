# Chronology reliability model

All chronology profiles enumerate the complete selected case and use the same
source-grounding rules. Profile selection changes extraction depth and cost; it
does not relax date or provenance validation.

## Validation guarantees

- Normalized dates must be derived independently from a verbatim date expression
  in the cited source unit.
- Australian clinical durations such as `1-2/52`, `6/52`, and `2/12` are not
  calendar dates and are rejected if a model attempts to normalize them as one.
- A source quotation must be contiguous and present in exactly one cited unit.
- Providers must occur in source text or match source author metadata. Metadata
  authors used without event-text support are labeled as documenting providers.
- Storage chunks containing multiple explicit visit headers are conservatively
  split into encounter source units before extraction.
- Fast and Thorough extraction is paginated. Pages must advance monotonically
  through source quotation positions and explicitly report completion.
- Failed, truncated, repeated, or ungrounded pages are not checkpointed.

## Profiles

- **Ultra-Fast** creates clause-local dated evidence candidates, applies bounded
  semantic filtering, and is best for rapid review. It can leave implicit or
  relative dates unresolved.
- **Fast** produces a compact structured chronology using small concurrent
  batches and bounded continuation pages.
- **Thorough** uses the same bounded pagination and grounding rules while
  extracting separate symptoms, diagnoses, investigations, findings, and plans.

## Checkpoints and migration

Chronology schema version 2 and Fast checkpoint version 4 invalidate checkpoints
created before deterministic date grounding was introduced. Checkpoint paths are
derived from the profile, schema/validator version, source-unit IDs, source text,
and page number. Cached pages are parsed and validated against current source text
before their events are accepted.

Do not manually rename old checkpoint files into the new namespace. They may
contain model interpretations such as treating `1-2/52` as 1 February 1952.

## Completeness

The rendered audit reports source units successfully processed rather than
assuming every discovered unit completed. Any failed batch, unreadable document,
or incomplete unit places an `INCOMPLETE CHRONOLOGY` warning at the top of the
result. Such output may still contain useful validated events but must not be
treated as a complete case chronology.
