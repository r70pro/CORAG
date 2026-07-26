BEGIN;

ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS source_char_start INTEGER,
    ADD COLUMN IF NOT EXISTS source_char_end INTEGER,
    ADD COLUMN IF NOT EXISTS page_start INTEGER,
    ADD COLUMN IF NOT EXISTS page_end INTEGER,
    ADD COLUMN IF NOT EXISTS provenance_type TEXT;

-- Existing char/page columns were calculated by the legacy reconstructing
-- splitter and cannot safely be promoted to exact source provenance. They
-- intentionally remain NULL until each document is re-indexed.

CREATE INDEX IF NOT EXISTS idx_chunks_page_range
    ON chunks(page_start, page_end);

COMMIT;
