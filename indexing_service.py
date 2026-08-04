import datetime
import hashlib
import os
import re
import shutil
import tempfile
from pathlib import Path

from path_security import (
    PathSecurityError,
    require_approved_file,
    resolve_file_under,
    resolve_run_under,
    resolve_under,
    validate_filename,
)
from settings_manager import WORKSPACE_DIR, load_settings


class CorpusIndexingService:
    last_created_run_id = None

    @staticmethod
    def index_run(run_dir, force=False, full_reindex=False):
        """Index a single OCR run into the RAG system.

        Chunks all markdown files, embeds them, and stores in Qdrant + PostgreSQL.

        Args:
            run_dir: Path to the OCR run directory.
            force: If True, bypass the early is_run_indexed check.
            full_reindex: Explicitly replace the complete run point set. This
                is the only workflow allowed to remove points absent from the
                newly generated run.

        Yields:
            Status update strings.
        """
        if not run_dir:
            yield "⚠️ Invalid run directory."
            return

        candidate_run = Path(run_dir)
        try:
            run_name = candidate_run.name
            safe_run_dir = resolve_run_under(WORKSPACE_DIR, run_name)
        except PathSecurityError:
            yield "⚠️ Invalid run directory."
            return
        if candidate_run.resolve() != safe_run_dir or not safe_run_dir.is_dir():
            yield "⚠️ Invalid run directory."
            return
        run_dir = str(safe_run_dir)

        # Extract a stable run_id from the directory name
        run_id = hashlib.sha256(run_dir.encode()).hexdigest()[:16]

        yield f"🔄 Starting indexing for **{run_name}**...\n"

        # Check if already indexed (unless force=True)
        if not force and not full_reindex:
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
        if total_chunks == 0:
            yield "⚠️ No chunks were generated; the run was not marked indexed.\n"
            return

        # Step 2: Stage a complete point set for the affected documents. The
        # PostgreSQL writes remain in one transaction from pending through
        # indexed. Qdrant mutations are journalled by exact point ID so failure
        # can restore overwritten vectors and remove only newly created points.
        yield "💾 Registering run in database...\n"
        rollback_snapshots = {}
        touched_point_ids = set()
        qdrant_mutation_started = False
        phase = "database registration"
        try:
            from rag.db import (
                delete_documents_not_in_run,
                get_point_ids_for_documents,
                get_point_ids_for_run,
                get_run_totals,
                indexing_transaction,
                mark_document_indexed,
                mark_run_indexed,
                mark_run_pending,
                register_document,
                register_run,
                replace_document_chunks,
            )
            from rag.embedding import (
                delete_points,
                get_qdrant_point_ids_for_run,
                init_collection,
                prepare_chunk_point_ids,
                rollback_point_mutations,
                snapshot_points,
                upsert_chunks_generator,
            )

            all_chunks = []
            for info in chunk_results.values():
                all_chunks.extend(info["chunks"])

            model_name = prepare_chunk_point_ids(all_chunks)
            new_point_ids = {chunk["qdrant_point_id"] for chunk in all_chunks}
            doc_ids = set(chunk_results)
            total_chunks_count = len(all_chunks)
            init_collection(model_name=model_name)

            with indexing_transaction(run_id) as connection:
                register_run(run_id, run_dir, total_documents=total_docs, connection=connection)
                mark_run_pending(run_id, connection=connection)

                if full_reindex:
                    old_point_ids = get_point_ids_for_run(
                        run_id, connection=connection
                    ) | get_qdrant_point_ids_for_run(run_id, model_name=model_name)
                else:
                    old_point_ids = get_point_ids_for_documents(doc_ids, connection=connection)
                touched_point_ids = old_point_ids | new_point_ids
                rollback_snapshots = snapshot_points(touched_point_ids, model_name=model_name)

                for doc_id, info in chunk_results.items():
                    register_document(
                        doc_id=doc_id,
                        run_id=run_id,
                        original_filename=info.get("original_filename") or info["md_file"],
                        pdf_total_pages=len(info.get("page_ranges") or []),
                        markdown_path=info["md_path"],
                        connection=connection,
                    )

                replace_document_chunks(doc_ids, all_chunks, connection=connection)
                if full_reindex:
                    delete_documents_not_in_run(run_id, doc_ids, connection=connection)

                phase = "embedding/indexing"
                yield f"🧠 Embedding and indexing {total_chunks_count} chunks...\n"
                qdrant_mutation_started = True
                for progress_info in upsert_chunks_generator(
                    all_chunks,
                    model_name=model_name,
                    batch_size=max(1, int(settings.get("embedding_batch_size", 64))),
                ):
                    stage = progress_info["stage"]
                    current = progress_info["current"]
                    total = progress_info["total"]
                    pct = int((current / total) * 100) if total > 0 else 0
                    if stage == "embedding":
                        yield f"[PROGRESS:embedding:{current}/{total}] 🧠 Embedding chunks ({pct}%)...\n"
                    else:
                        yield f"[PROGRESS:indexing:{current}/{total}] ⚡ Indexing chunks in vector store ({pct}%)...\n"

                stale_point_ids = old_point_ids - new_point_ids
                delete_points(stale_point_ids, model_name=model_name)

                phase = "database finalisation"
                for doc_id in doc_ids:
                    mark_document_indexed(doc_id, connection=connection)
                authoritative_docs, authoritative_chunks = get_run_totals(
                    run_id, connection=connection
                )
                mark_run_indexed(
                    run_id,
                    total_chunks=authoritative_chunks,
                    total_documents=authoritative_docs,
                    connection=connection,
                )

        except Exception as e:
            label = (
                "Database registration"
                if phase == "database registration"
                else "Embedding/indexing"
            )
            yield f"❌ {label} failed: {e}\n"
            if qdrant_mutation_started:
                yield "⏪ Restoring the pre-operation vector point set...\n"
                try:
                    rollback_point_mutations(
                        touched_point_ids,
                        rollback_snapshots,
                        model_name=model_name,
                    )
                except Exception as rollback_err:
                    yield f"⚠️ Rollback warning: {rollback_err}\n"
            return

        # Step 3: Upload source objects only after the searchable index commits.
        yield "☁️ Uploading to object storage...\n"
        try:
            from rag.storage import upload_markdown, upload_pdf

            for doc_id, info in chunk_results.items():
                try:
                    md_path = require_approved_file(info["md_path"], {safe_run_dir}, {".md"})
                except PathSecurityError:
                    continue
                if md_path.is_file():
                    upload_markdown(run_id, doc_id, str(md_path))

                pdf_filename = info["md_file"].replace(".md", ".pdf")
                try:
                    pdf_path = resolve_file_under(
                        resolve_under(safe_run_dir, "inputs"), pdf_filename, {".pdf"}
                    )
                except PathSecurityError:
                    continue
                if pdf_path.is_file():
                    upload_pdf(run_id, doc_id, str(pdf_path))
        except Exception as e:
            yield f"⚠️ Storage upload warning: {e}\n"

        # Step 4: Invalidate query cache
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
            get_point_ids_for_documents,
            get_run_totals,
            get_runs_with_stats,
            indexing_transaction,
            mark_document_indexed,
            mark_run_indexed,
            mark_run_pending,
            register_document,
            register_run,
            replace_document_chunks,
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
            try:
                run_dir = str(resolve_run_under(WORKSPACE_DIR, run_name))
            except PathSecurityError:
                yield "❌ Error: Invalid case name.\n"
                return
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

            candidate_run = Path(run_dir)
            try:
                run_name = candidate_run.name
                safe_run_dir = resolve_run_under(WORKSPACE_DIR, run_name)
            except PathSecurityError:
                yield "❌ Error: Could not locate existing case directory.\n"
                return
            if candidate_run.resolve() != safe_run_dir:
                yield "❌ Error: Could not locate existing case directory.\n"
                return
            run_dir = str(safe_run_dir)
            yield f"📁 Adding to existing case: **{run_name}**...\n"

        # Set up directory paths
        markdown_inputs_dir = resolve_under(run_dir, "markdown", "inputs")
        try:
            markdown_inputs_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            yield f"❌ Failed to create directories: {e}\n"
            return

        # Step 1: Copy uploads into an operation-specific staging directory.
        # Destination files are not replaced until vectors and database rows
        # have both been prepared successfully.
        try:
            staging_dir = Path(tempfile.mkdtemp(prefix=".indexing-", dir=str(markdown_inputs_dir)))
        except Exception as e:
            yield f"❌ Failed to create staging directory: {e}\n"
            return

        copied_files = []
        seen_filenames = set()
        for file_info in files:
            file_path = file_info.name if hasattr(file_info, "name") else str(file_info)
            if not os.path.exists(file_path):
                continue
            filename = Path(file_path).name
            original_name_value = getattr(file_info, "original_filename", None)
            original_filename = (
                original_name_value if isinstance(original_name_value, str) else filename
            )
            try:
                validate_filename(filename, {".md"})
                if filename in seen_filenames:
                    yield f"⚠️ Warning: Duplicate upload name {filename}; skipping duplicate.\n"
                    continue
                seen_filenames.add(filename)
                staged_path = resolve_file_under(staging_dir, filename, {".md"})
                destination_path = resolve_file_under(markdown_inputs_dir, filename, {".md"})
                shutil.copy(file_path, staged_path)
                copied_files.append(
                    (filename, str(original_filename), staged_path, destination_path)
                )
                yield f"📄 Staged **{original_filename}** for case storage.\n"
            except Exception as e:
                yield f"⚠️ Warning: Could not copy {filename}: {e}\n"

        if not copied_files:
            shutil.rmtree(staging_dir, ignore_errors=True)
            yield "❌ Error: No files were successfully copied.\n"
            return

        # Step 2: Parse and chunk every staged file before mutating either data
        # store. A filename maps deterministically to one document replacement.
        settings = load_settings()
        max_chunk_size = settings.get("chunk_size", 800)
        chunk_overlap = settings.get("chunk_overlap", 100)

        all_new_chunks = []
        processed_documents = []

        for filename, original_filename, staged_path, destination_path in copied_files:
            yield f"⚙️ Processing **{original_filename}**...\n"

            try:
                with open(staged_path, encoding="utf-8") as f:
                    markdown_text = f.read()
            except Exception as e:
                yield f"⚠️ Error reading {original_filename}: {e}. Skipping.\n"
                continue

            doc_id = hashlib.sha256(f"{run_id}:{filename}".encode()).hexdigest()[:24]

            try:
                chunks = chunk_document(
                    markdown_text=markdown_text,
                    doc_id=doc_id,
                    run_id=run_id,
                    page_ranges=[],
                    max_chunk_size=max_chunk_size,
                    chunk_overlap=chunk_overlap,
                    original_filename=original_filename,
                    provenance_type="external_markdown",
                )
                if not chunks:
                    yield f"⚠️ No chunks created for {original_filename}. Skipping.\n"
                    continue
                all_new_chunks.extend(chunks)
                processed_documents.append(
                    {
                        "filename": filename,
                        "original_filename": original_filename,
                        "staged_path": staged_path,
                        "destination_path": destination_path,
                        "doc_id": doc_id,
                    }
                )
                yield f"🧩 Created **{len(chunks)}** chunk(s) for {original_filename}.\n"
            except Exception as e:
                yield f"⚠️ Chunking failed for {original_filename}: {e}. Skipping.\n"
                continue

        if not all_new_chunks:
            shutil.rmtree(staging_dir, ignore_errors=True)
            yield "❌ Error: No chunks generated from the uploaded files.\n"
            return

        # Step 3: Replace the affected document point sets. PostgreSQL stays in
        # one pending -> indexed transaction. Qdrant rollback is an exact-ID
        # journal: new IDs are deleted and overwritten IDs are restored.
        yield "💾 Registering case metadata in database...\n"
        rollback_snapshots = {}
        touched_point_ids = set()
        qdrant_mutation_started = False
        file_journal = []
        phase = "database registration"
        try:
            from rag.embedding import (
                delete_points,
                init_collection,
                prepare_chunk_point_ids,
                rollback_point_mutations,
                snapshot_points,
                upsert_chunks_generator,
            )

            model_name = prepare_chunk_point_ids(all_new_chunks)
            init_collection(model_name=model_name)
            new_point_ids = {chunk["qdrant_point_id"] for chunk in all_new_chunks}
            doc_ids = {document["doc_id"] for document in processed_documents}
            total_chunks_count = len(all_new_chunks)

            with indexing_transaction(run_id) as connection:
                register_run(
                    run_id,
                    run_dir,
                    total_documents=len(processed_documents),
                    connection=connection,
                )
                mark_run_pending(run_id, connection=connection)

                old_point_ids = get_point_ids_for_documents(doc_ids, connection=connection)
                touched_point_ids = old_point_ids | new_point_ids
                rollback_snapshots = snapshot_points(touched_point_ids, model_name=model_name)

                for document in processed_documents:
                    register_document(
                        doc_id=document["doc_id"],
                        run_id=run_id,
                        original_filename=document["original_filename"],
                        pdf_total_pages=0,
                        markdown_path=str(document["destination_path"]),
                        connection=connection,
                    )

                replace_document_chunks(doc_ids, all_new_chunks, connection=connection)

                phase = "embedding/indexing"
                yield f"🧠 Embedding and indexing {total_chunks_count} chunks...\n"
                qdrant_mutation_started = True
                for progress_info in upsert_chunks_generator(
                    all_new_chunks, model_name=model_name, batch_size=32
                ):
                    stage = progress_info["stage"]
                    current = progress_info["current"]
                    total = progress_info["total"]
                    pct = int((current / total) * 100) if total > 0 else 0
                    if stage == "embedding":
                        yield f"[PROGRESS:embedding:{current}/{total}] 🧠 Embedding chunks ({pct}%)...\n"
                    else:
                        yield f"[PROGRESS:indexing:{current}/{total}] ⚡ Indexing chunks in vector store ({pct}%)...\n"

                delete_points(old_point_ids - new_point_ids, model_name=model_name)

                # Final source-file replacement is journalled too, so a commit
                # failure restores any previous markdown bytes.
                phase = "file/database finalisation"
                backup_dir = staging_dir / "backups"
                backup_dir.mkdir(exist_ok=True)
                for document in processed_documents:
                    destination = document["destination_path"]
                    backup = None
                    if destination.exists():
                        backup = backup_dir / document["filename"]
                        shutil.copy(destination, backup)
                    os.replace(document["staged_path"], destination)
                    file_journal.append((destination, backup))

                for doc_id in doc_ids:
                    mark_document_indexed(doc_id, connection=connection)
                authoritative_docs, authoritative_chunks = get_run_totals(
                    run_id, connection=connection
                )
                mark_run_indexed(
                    run_id,
                    total_chunks=authoritative_chunks,
                    total_documents=authoritative_docs,
                    connection=connection,
                )

        except Exception as e:
            label = (
                "Database run registration"
                if phase == "database registration"
                else "Embedding/indexing"
            )
            yield f"❌ {label} failed: {e}\n"
            for destination, backup in reversed(file_journal):
                try:
                    if backup is not None and backup.exists():
                        os.replace(backup, destination)
                    elif destination.exists():
                        destination.unlink()
                except Exception as file_rollback_err:
                    yield f"⚠️ File rollback warning: {file_rollback_err}\n"
            if qdrant_mutation_started:
                yield "⏪ Restoring only the vector points touched by this upload...\n"
                try:
                    rollback_point_mutations(
                        touched_point_ids,
                        rollback_snapshots,
                        model_name=model_name,
                    )
                except Exception as rollback_err:
                    yield f"⚠️ Rollback warning: {rollback_err}\n"
            shutil.rmtree(staging_dir, ignore_errors=True)
            return

        shutil.rmtree(staging_dir, ignore_errors=True)

        # Step 4: Object storage is non-authoritative and is updated only after
        # the database/vector operation has committed successfully.
        for document in processed_documents:
            try:
                upload_markdown(
                    run_id,
                    document["doc_id"],
                    str(document["destination_path"]),
                )
                yield f"☁️ Uploaded **{document['original_filename']}** to object storage.\n"
            except Exception as e:
                yield (f"⚠️ Storage upload warning for {document['original_filename']}: {e}\n")

        # Step 5: Invalidate query cache
        try:
            invalidate_query_cache()
        except Exception:
            pass

        yield f"\n✅ Successfully uploaded and indexed **{len(processed_documents)}** markdown file(s) into case **{run_name}**!\n"
