"""
PostgreSQL database layer for the RAG document registry.

Manages:
- Document registry (source PDFs, metadata, indexing status)
- Chunk metadata (text, page, author, date, section type)
- Run tracking (which OCR runs have been indexed)
"""

import os
import json
import psycopg2
import psycopg2.extras
from contextlib import contextmanager

# Default connection parameters
DEFAULT_DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "olmocr_rag",
    "user": "olmocr",
    "password": "olmocr_rag_2026",
}


def get_db_config():
    """Get database configuration from environment or defaults."""
    return {
        "host": os.environ.get("OLMOCR_PG_HOST", DEFAULT_DB_CONFIG["host"]),
        "port": int(os.environ.get("OLMOCR_PG_PORT", DEFAULT_DB_CONFIG["port"])),
        "dbname": os.environ.get("OLMOCR_PG_DB", DEFAULT_DB_CONFIG["dbname"]),
        "user": os.environ.get("OLMOCR_PG_USER", DEFAULT_DB_CONFIG["user"]),
        "password": os.environ.get("OLMOCR_PG_PASS", DEFAULT_DB_CONFIG["password"]),
    }


@contextmanager
def get_connection(config=None):
    """Context manager for database connections."""
    if config is None:
        config = get_db_config()
    conn = psycopg2.connect(**config)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_schema():
    """Create the database schema if it doesn't exist."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ocr_runs (
                    run_id          TEXT PRIMARY KEY,
                    run_dir         TEXT NOT NULL,
                    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    indexed_at      TIMESTAMP WITH TIME ZONE,
                    total_documents INTEGER DEFAULT 0,
                    total_chunks    INTEGER DEFAULT 0,
                    status          TEXT DEFAULT 'pending'
                );

                CREATE TABLE IF NOT EXISTS documents (
                    doc_id          TEXT PRIMARY KEY,
                    run_id          TEXT NOT NULL REFERENCES ocr_runs(run_id) ON DELETE CASCADE,
                    original_filename TEXT NOT NULL,
                    pdf_total_pages INTEGER DEFAULT 0,
                    markdown_path   TEXT,
                    minio_pdf_key   TEXT,
                    minio_md_key    TEXT,
                    olmocr_version  TEXT,
                    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    indexed_at      TIMESTAMP WITH TIME ZONE,
                    metadata_json   JSONB DEFAULT '{}'::jsonb
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id        TEXT PRIMARY KEY,
                    doc_id          TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                    run_id          TEXT NOT NULL REFERENCES ocr_runs(run_id) ON DELETE CASCADE,
                    chunk_index     INTEGER NOT NULL,
                    text            TEXT NOT NULL,
                    char_start      INTEGER NOT NULL,
                    char_end        INTEGER NOT NULL,
                    page_number     INTEGER,
                    document_type   TEXT,
                    author          TEXT,
                    date_extracted  DATE,
                    date_raw        TEXT,
                    section_type    TEXT,
                    patient_name    TEXT,
                    token_count     INTEGER DEFAULT 0,
                    embedding_model TEXT,
                    qdrant_point_id TEXT,
                    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_run_id ON chunks(run_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_page ON chunks(page_number);
                CREATE INDEX IF NOT EXISTS idx_chunks_date ON chunks(date_extracted);
                CREATE INDEX IF NOT EXISTS idx_chunks_author ON chunks(author);
                CREATE INDEX IF NOT EXISTS idx_chunks_doctype ON chunks(document_type);
                CREATE INDEX IF NOT EXISTS idx_documents_run_id ON documents(run_id);
            """)


def is_healthy():
    """Check if PostgreSQL is reachable and schema exists."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                return True
    except Exception:
        return False


# ── Run operations ──────────────────────────────────────────────

def register_run(run_id, run_dir, total_documents=0):
    """Register a new OCR run in the database."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO ocr_runs (run_id, run_dir, total_documents, status)
                VALUES (%s, %s, %s, 'pending')
                ON CONFLICT (run_id) DO UPDATE SET
                    run_dir = EXCLUDED.run_dir,
                    total_documents = EXCLUDED.total_documents
            """, (run_id, run_dir, total_documents))


def mark_run_indexed(run_id, total_chunks):
    """Mark a run as fully indexed."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE ocr_runs
                SET indexed_at = NOW(), total_chunks = %s, status = 'indexed'
                WHERE run_id = %s
            """, (total_chunks, run_id))


def get_indexed_runs():
    """Get all indexed runs."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT run_id, run_dir, created_at, indexed_at,
                       total_documents, total_chunks, status
                FROM ocr_runs
                WHERE status = 'indexed'
                ORDER BY created_at DESC
            """)
            return cur.fetchall()


def get_all_runs():
    """Get all runs (indexed and pending)."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT run_id, run_dir, created_at, indexed_at,
                       total_documents, total_chunks, status
                FROM ocr_runs
                ORDER BY created_at DESC
            """)
            return cur.fetchall()


def is_run_indexed(run_id):
    """Check if a run has already been indexed."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT status FROM ocr_runs WHERE run_id = %s",
                (run_id,)
            )
            row = cur.fetchone()
            return row is not None and row[0] == 'indexed'


# ── Document operations ────────────────────────────────────────

def register_document(doc_id, run_id, original_filename, pdf_total_pages=0,
                      markdown_path=None, minio_pdf_key=None, minio_md_key=None,
                      olmocr_version=None, metadata=None):
    """Register a document in the registry."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO documents
                    (doc_id, run_id, original_filename, pdf_total_pages,
                     markdown_path, minio_pdf_key, minio_md_key,
                     olmocr_version, metadata_json)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (doc_id) DO UPDATE SET
                    markdown_path = EXCLUDED.markdown_path,
                    minio_pdf_key = EXCLUDED.minio_pdf_key,
                    minio_md_key = EXCLUDED.minio_md_key,
                    metadata_json = EXCLUDED.metadata_json
            """, (doc_id, run_id, original_filename, pdf_total_pages,
                  markdown_path, minio_pdf_key, minio_md_key,
                  olmocr_version,
                  json.dumps(metadata or {})))


def mark_document_indexed(doc_id):
    """Mark a document as indexed."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE documents SET indexed_at = NOW() WHERE doc_id = %s",
                (doc_id,)
            )


def get_documents_for_run(run_id):
    """Get all documents for a given run."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT doc_id, original_filename, pdf_total_pages,
                       markdown_path, minio_pdf_key, minio_md_key,
                       olmocr_version, indexed_at, metadata_json
                FROM documents
                WHERE run_id = %s
                ORDER BY original_filename
            """, (run_id,))
            return cur.fetchall()


def get_all_documents():
    """Get all documents across all runs."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT d.doc_id, d.run_id, d.original_filename,
                       d.pdf_total_pages, d.markdown_path,
                       d.indexed_at, d.metadata_json,
                       r.run_dir, r.created_at as run_created_at
                FROM documents d
                JOIN ocr_runs r ON d.run_id = r.run_id
                ORDER BY r.created_at DESC, d.original_filename
            """)
            return cur.fetchall()


# ── Chunk operations ───────────────────────────────────────────

def insert_chunks(chunks_list):
    """Bulk insert chunks into the database.

    Args:
        chunks_list: List of dicts with chunk data.
    """
    if not chunks_list:
        return

    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO chunks
                    (chunk_id, doc_id, run_id, chunk_index, text,
                     char_start, char_end, page_number, document_type,
                     author, date_extracted, date_raw, section_type,
                     patient_name, token_count, embedding_model, qdrant_point_id)
                VALUES %s
                ON CONFLICT (chunk_id) DO NOTHING
                """,
                [
                    (
                        c["chunk_id"], c["doc_id"], c["run_id"], c["chunk_index"],
                        c["text"], c["char_start"], c["char_end"],
                        c.get("page_number"), c.get("document_type"),
                        c.get("author"), c.get("date_extracted"), c.get("date_raw"),
                        c.get("section_type"), c.get("patient_name"),
                        c.get("token_count", 0), c.get("embedding_model"),
                        c.get("qdrant_point_id")
                    )
                    for c in chunks_list
                ],
                page_size=100
            )


def get_chunks_for_document(doc_id):
    """Get all chunks for a document, ordered by position."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT chunk_id, chunk_index, text, char_start, char_end,
                       page_number, document_type, author, date_extracted,
                       date_raw, section_type, patient_name, token_count,
                       qdrant_point_id
                FROM chunks
                WHERE doc_id = %s
                ORDER BY chunk_index
            """, (doc_id,))
            return cur.fetchall()


def get_chunk_by_qdrant_id(qdrant_point_id):
    """Retrieve chunk metadata by its Qdrant point ID."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT c.chunk_id, c.doc_id, c.text, c.page_number,
                       c.document_type, c.author, c.date_extracted,
                       c.section_type, c.patient_name,
                       d.original_filename, d.run_id
                FROM chunks c
                JOIN documents d ON c.doc_id = d.doc_id
                WHERE c.qdrant_point_id = %s
            """, (qdrant_point_id,))
            return cur.fetchone()


def get_chunks_by_qdrant_ids(qdrant_point_ids):
    """Retrieve chunk metadata for multiple Qdrant point IDs."""
    if not qdrant_point_ids:
        return []
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT c.chunk_id, c.doc_id, c.text, c.page_number,
                       c.document_type, c.author, c.date_extracted,
                       c.date_raw, c.section_type, c.patient_name,
                       d.original_filename, d.run_id, c.qdrant_point_id
                FROM chunks c
                JOIN documents d ON c.doc_id = d.doc_id
                WHERE c.qdrant_point_id = ANY(%s)
            """, (qdrant_point_ids,))
            return cur.fetchall()


def get_corpus_stats():
    """Get aggregate statistics about the indexed corpus."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    (SELECT COUNT(*) FROM ocr_runs WHERE status = 'indexed') as indexed_runs,
                    (SELECT COUNT(*) FROM documents WHERE indexed_at IS NOT NULL) as indexed_documents,
                    (SELECT COUNT(*) FROM chunks) as total_chunks,
                    (SELECT COUNT(DISTINCT author) FROM chunks WHERE author IS NOT NULL) as unique_authors,
                    (SELECT MIN(date_extracted) FROM chunks WHERE date_extracted IS NOT NULL) as earliest_date,
                    (SELECT MAX(date_extracted) FROM chunks WHERE date_extracted IS NOT NULL) as latest_date
            """)
            return cur.fetchone()


def delete_run_data(run_id):
    """Delete all data for a run (chunks, documents, and run record)."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Cascading delete handles chunks and documents
            cur.execute("DELETE FROM ocr_runs WHERE run_id = %s", (run_id,))
