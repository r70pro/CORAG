import datetime
import hashlib
import os
import re
import shutil

from settings_manager import WORKSPACE_DIR, load_settings


class CorpusIndexingService:
    last_created_run_id = None

    @staticmethod
    def index_run(run_dir, force=False):
        """Index a single OCR run into the RAG system.

        Chunks all markdown files, embeds them, and stores in Qdrant + PostgreSQL.

        Args:
            run_dir: Path to the OCR run directory.
            force: If True, bypass the early is_run_indexed check for manual re-indexing.

        Yields:
            Status update strings.
        """
        if not run_dir or not os.path.exists(run_dir):
            yield "⚠️ Invalid run directory."
            return

        run_name = os.path.basename(run_dir)

        # Extract a stable run_id from the directory name
        run_id = hashlib.sha256(run_dir.encode()).hexdigest()[:16]

        yield f"🔄 Starting indexing for **{run_name}**...\n"

        # Check if already indexed (unless force=True)
        if not force:
            try:
                from rag.db import is_run_indexed

                if is_run_indexed(run_id):
                    yield f"ℹ️ Run **{run_name}** is already indexed. Skipping.\n"
                    yield "✅ Done."
                    return
            except Exception as e:
                yield f"⚠️ Could not check index status: {e}\n"

        # Step 1: Chunk documents
        yield "📄 Chunking documents...\n"
        try:
            from rag.chunker import chunk_documents_from_run

            settings = load_settings()
            chunk_results = chunk_documents_from_run(
                run_dir=run_dir,
                run_id=run_id,
                max_chunk_size=settings.get("chunk_size", 800),
                chunk_overlap=settings.get("chunk_overlap", 100),
            )
        except Exception as e:
            yield f"❌ Chunking failed: {e}\n"
            return

        if not chunk_results:
            yield "⚠️ No markdown files found in this run.\n"
            return

        total_chunks = sum(len(info["chunks"]) for info in chunk_results.values())
        total_docs = len(chunk_results)
        yield f"  Found **{total_docs}** document(s), **{total_chunks}** chunk(s).\n"

        # Step 2: Register run in PostgreSQL
        yield "💾 Registering run in database...\n"
        try:
            from rag.db import (
                insert_chunks,
                mark_document_indexed,
                mark_run_indexed,
                register_document,
                register_run,
            )

            register_run(run_id, run_dir, total_documents=total_docs)

            for doc_id, info in chunk_results.items():
                md_file = info["md_file"]
                # Extract original filename (strip numeric prefix)
                orig_match = re.match(r"^\d+_(.*)", md_file)
                orig_name = orig_match.group(1) if orig_match else md_file

                pdf_pages = 0
                # Count pages from page ranges if available
                if info.get("page_ranges"):
                    pdf_pages = len(info["page_ranges"])

                register_document(
                    doc_id=doc_id,
                    run_id=run_id,
                    original_filename=orig_name,
                    pdf_total_pages=pdf_pages,
                    markdown_path=info["md_path"],
                )
        except Exception as e:
            yield f"❌ Database registration failed: {e}\n"
            return

        # Step 3: Upload to MinIO
        yield "☁️ Uploading to object storage...\n"
        try:
            from rag.storage import upload_markdown, upload_pdf

            for doc_id, info in chunk_results.items():
                # Upload markdown
                if os.path.exists(info["md_path"]):
                    upload_markdown(run_id, doc_id, info["md_path"])

                # Upload corresponding PDF if exists
                pdf_filename = info["md_file"].replace(".md", ".pdf")
                pdf_path = os.path.join(run_dir, "inputs", pdf_filename)
                if os.path.exists(pdf_path):
                    upload_pdf(run_id, doc_id, pdf_path)

        except Exception as e:
            yield f"⚠️ Storage upload warning: {e}\n"
            # Non-fatal — indexing can continue without MinIO

        # Step 4: Embed and upsert into Qdrant.
        #
        # Ordering for crash-safety: the PostgreSQL chunk rows are the source of
        # truth and are written BEFORE the Qdrant vectors. If the process dies
        # between the two, the run is NOT marked indexed, so a retry will
        # re-upsert (idempotently — point IDs are derived from chunk_id) and then
        # persist the chunk rows. Conversely we never end up with vectors that
        # have no DB row. We also pre-delete any existing points for this run so
        # a retry after a partial Qdrant write cannot leave stale duplicates.
        try:
            from rag.embedding import upsert_chunks_generator

            all_chunks = []
            for info in chunk_results.values():
                all_chunks.extend(info["chunks"])

            total_chunks_count = len(all_chunks)
            yield f"🧠 Embedding and indexing {total_chunks_count} chunks...\n"

            for progress_info in upsert_chunks_generator(
                all_chunks, batch_size=32, pre_delete_run_ids=[run_id]
            ):
                stage = progress_info["stage"]
                current = progress_info["current"]
                total = progress_info["total"]
                pct = int((current / total) * 100) if total > 0 else 0
                if stage == "embedding":
                    yield f"[PROGRESS:embedding:{current}/{total}] 🧠 Embedding chunks ({pct}%)...\n"
                else:
                    yield f"[PROGRESS:indexing:{current}/{total}] ⚡ Indexing chunks in vector store ({pct}%)...\n"

            # Persist chunk metadata in PostgreSQL (idempotent via chunk_id PK).
            insert_chunks(all_chunks)

            # Mark documents and run as indexed
            for doc_id in chunk_results:
                mark_document_indexed(doc_id)
            mark_run_indexed(run_id, total_chunks=total_chunks_count)

        except Exception as e:
            yield f"❌ Embedding/indexing failed: {e}\n"
            yield "⏪ Rolling back any vectors written for this run...\n"
            try:
                from rag.embedding import delete_run_vectors

                delete_run_vectors(run_id)
            except Exception as rollback_err:
                yield f"⚠️ Rollback warning: {rollback_err}\n"
            return

        # Step 5: Invalidate query cache
        try:
            from rag.cache import invalidate_query_cache

            invalidate_query_cache()
        except Exception:
            pass  # Non-fatal

        yield f"\n✅ Successfully indexed **{run_name}**: {total_docs} document(s), {total_chunks} chunk(s).\n"

    @staticmethod
    def index_all_runs(get_available_runs_fn=None, force=False):
        """Index all available OCR runs into the RAG corpus.

        Yields:
            Status update strings.
        """
        if get_available_runs_fn is None:
            from settings_manager import get_available_runs as get_runs

            get_available_runs_fn = get_runs
        runs = get_available_runs_fn()
        if not runs:
            yield "⚠️ No completed OCR runs found in workspace.\n"
            return

        yield f"🔄 Indexing **{len(runs)}** run(s) into the corpus...\n\n"

        for display_name, run_dir in runs:
            yield f"--- Processing: {display_name} ---\n"
            yield from CorpusIndexingService.index_run(run_dir, force=force)
            yield "\n"

        yield "\n✅ All runs processed."

    @staticmethod
    def add_markdown_to_case(files, case_option, new_case_name):
        """Upload and index markdown files to a new or existing case.

        Yields:
            Status update strings.
        """
        if not files:
            yield "⚠️ No files uploaded.\n"
            return

        from rag.cache import invalidate_query_cache
        from rag.chunker import chunk_document
        from rag.db import (
            get_connection,
            get_runs_with_stats,
            insert_chunks,
            mark_document_indexed,
            mark_run_indexed,
            register_document,
            register_run,
        )
        from rag.storage import upload_markdown

        if case_option == "new":
            if not new_case_name or not new_case_name.strip():
                yield "❌ Error: New case name is required.\n"
                return

            # Create a new run/case directory
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            clean_name = re.sub(r"[^a-zA-Z0-9_-]", "_", new_case_name.strip())
            run_name = f"run_{clean_name}_{timestamp}"
            run_dir = os.path.join(WORKSPACE_DIR, run_name)
            run_id = hashlib.sha256(run_dir.encode()).hexdigest()[:16]
            CorpusIndexingService.last_created_run_id = run_id

            yield f"📁 Creating new case: **{new_case_name}**...\n"
        else:
            # Find existing run details
            run_id = case_option
            CorpusIndexingService.last_created_run_id = run_id
            run_dir = None
            try:
                runs = get_runs_with_stats()
                for r in runs:
                    if r.get("run_id") == run_id:
                        run_dir = r.get("run_dir")
                        break
            except Exception as e:
                yield f"❌ Failed to fetch case information: {e}\n"
                return

            if not run_dir:
                # Fallback if get_runs_with_stats fails or doesn't have it
                # Query ocr_runs directly
                try:
                    with get_connection() as conn:
                        with conn.cursor() as cur:
                            cur.execute("SELECT run_dir FROM ocr_runs WHERE run_id = %s", (run_id,))
                            row = cur.fetchone()
                            if row:
                                run_dir = row[0]
                except Exception as e:
                    yield f"❌ Failed to retrieve run directory: {e}\n"
                    return

            if not run_dir:
                yield "❌ Error: Could not locate existing case directory.\n"
                return

            run_name = os.path.basename(run_dir)
            yield f"📁 Adding to existing case: **{run_name}**...\n"

        # Set up directory paths
        markdown_inputs_dir = os.path.join(run_dir, "markdown", "inputs")
        try:
            os.makedirs(markdown_inputs_dir, exist_ok=True)
        except Exception as e:
            yield f"❌ Failed to create directories: {e}\n"
            return

        # Step 1: Copy uploaded files to the case directory
        copied_files = []
        for file_info in files:
            file_path = file_info.name if hasattr(file_info, "name") else str(file_info)
            if not os.path.exists(file_path):
                continue
            filename = os.path.basename(file_path)
            dest_path = os.path.join(markdown_inputs_dir, filename)
            try:
                shutil.copy(file_path, dest_path)
                copied_files.append((filename, dest_path))
                yield f"📄 Copied **{filename}** to case storage.\n"
            except Exception as e:
                yield f"⚠️ Warning: Could not copy {filename}: {e}\n"

        if not copied_files:
            yield "❌ Error: No files were successfully copied.\n"
            return

        # Step 2: Register/update run in PostgreSQL
        yield "💾 Registering case metadata in database...\n"
        try:
            # Determine total documents currently in DB for this run_id
            current_docs_count = 0
            if case_option != "new":
                with get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT COUNT(*) FROM documents WHERE run_id = %s", (run_id,))
                        current_docs_count = cur.fetchone()[0]

            new_total_docs = current_docs_count + len(copied_files)
            register_run(run_id, run_dir, total_documents=new_total_docs)
        except Exception as e:
            yield f"❌ Database run registration failed: {e}\n"
            return

        # Step 3: Process each file (chunk, register document, embed, upsert)
        settings = load_settings()
        max_chunk_size = settings.get("chunk_size", 800)
        chunk_overlap = settings.get("chunk_overlap", 100)

        all_new_chunks = []

        for filename, md_path in copied_files:
            yield f"⚙️ Processing **{filename}**...\n"

            # Read contents
            try:
                with open(md_path, encoding="utf-8") as f:
                    markdown_text = f.read()
            except Exception as e:
                yield f"⚠️ Error reading {filename}: {e}. Skipping.\n"
                continue

            # Generate doc_id
            doc_id = hashlib.sha256(f"{run_id}:{filename}".encode()).hexdigest()[:24]

            # Register document
            try:
                register_document(
                    doc_id=doc_id,
                    run_id=run_id,
                    original_filename=filename,
                    pdf_total_pages=0,
                    markdown_path=md_path,
                )
            except Exception as e:
                yield f"⚠️ Database document registration failed for {filename}: {e}. Skipping.\n"
                continue

            # Upload to MinIO
            try:
                upload_markdown(run_id, doc_id, md_path)
                yield f"☁️ Uploaded **{filename}** to object storage.\n"
            except Exception as e:
                yield f"⚠️ Storage upload warning for {filename}: {e}\n"

            # Chunk document
            try:
                chunks = chunk_document(
                    markdown_text=markdown_text,
                    doc_id=doc_id,
                    run_id=run_id,
                    page_ranges=[],
                    max_chunk_size=max_chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                all_new_chunks.extend(chunks)
                yield f"🧩 Created **{len(chunks)}** chunk(s) for {filename}.\n"
            except Exception as e:
                yield f"⚠️ Chunking failed for {filename}: {e}. Skipping.\n"
                continue

        if not all_new_chunks:
            yield "❌ Error: No chunks generated from the uploaded files.\n"
            return

        # Step 4: Embed and upsert into Qdrant & Postgres.
        # Postgres chunk rows are written AFTER the vectors (the db rows are the
        # source of truth and idempotent on chunk_id), and we pre-delete any
        # existing points for the run so a retry cannot create duplicate vectors.
        try:
            from rag.embedding import delete_run_vectors, upsert_chunks_generator

            total_chunks_count = len(all_new_chunks)
            yield f"🧠 Embedding and indexing {total_chunks_count} chunks...\n"

            for progress_info in upsert_chunks_generator(
                all_new_chunks, batch_size=32, pre_delete_run_ids=[run_id]
            ):
                stage = progress_info["stage"]
                current = progress_info["current"]
                total = progress_info["total"]
                pct = int((current / total) * 100) if total > 0 else 0
                if stage == "embedding":
                    yield f"[PROGRESS:embedding:{current}/{total}] 🧠 Embedding chunks ({pct}%)...\n"
                else:
                    yield f"[PROGRESS:indexing:{current}/{total}] ⚡ Indexing chunks in vector store ({pct}%)...\n"

            insert_chunks(all_new_chunks)

            # Mark all documents as indexed
            for filename, _ in copied_files:
                doc_id = hashlib.sha256(f"{run_id}:{filename}".encode()).hexdigest()[:24]
                mark_document_indexed(doc_id)

            # Get total chunks for the run to mark run indexed
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM chunks WHERE run_id = %s", (run_id,))
                    total_chunks_in_run = cur.fetchone()[0]

            mark_run_indexed(run_id, total_chunks=total_chunks_in_run)

        except Exception as e:
            yield f"❌ Embedding/indexing failed: {e}\n"
            yield "⏪ Rolling back any vectors written for this case...\n"
            try:
                from rag.embedding import delete_run_vectors

                delete_run_vectors(run_id)
            except Exception as rollback_err:
                yield f"⚠️ Rollback warning: {rollback_err}\n"
            return

        # Step 5: Invalidate query cache
        try:
            invalidate_query_cache()
        except Exception:
            pass

        yield f"\n✅ Successfully uploaded and indexed **{len(copied_files)}** markdown file(s) into case **{run_name}**!\n"
