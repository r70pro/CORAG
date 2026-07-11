"""
MinIO blob storage layer for PDFs and Markdown files.

Provides:
- Upload original PDFs and extracted markdown to MinIO
- Retrieve files by key
- Bucket management with auto-creation
"""

import os
import io
from minio import Minio

# Default configuration
DEFAULT_MINIO_CONFIG = {
    "endpoint": "localhost:9000",
    "access_key": "olmocr_minio",
    "secret_key": "olmocr_minio_2026",
    "secure": False,
}

# Bucket names
BUCKET_PDFS = "olmocr-pdfs"
BUCKET_MARKDOWN = "olmocr-markdown"


def get_minio_config():
    """Get MinIO configuration from environment or defaults."""
    return {
        "endpoint": os.environ.get("OLMOCR_MINIO_ENDPOINT", DEFAULT_MINIO_CONFIG["endpoint"]),
        "access_key": os.environ.get("OLMOCR_MINIO_ACCESS_KEY", DEFAULT_MINIO_CONFIG["access_key"]),
        "secret_key": os.environ.get("OLMOCR_MINIO_SECRET_KEY", DEFAULT_MINIO_CONFIG["secret_key"]),
        "secure": os.environ.get("OLMOCR_MINIO_SECURE", "false").lower() == "true",
    }


def get_client(config=None):
    """Create a MinIO client."""
    if config is None:
        config = get_minio_config()
    return Minio(
        config["endpoint"],
        access_key=config["access_key"],
        secret_key=config["secret_key"],
        secure=config["secure"],
    )


def init_buckets():
    """Create required buckets if they don't exist."""
    client = get_client()
    for bucket in [BUCKET_PDFS, BUCKET_MARKDOWN]:
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)


def is_healthy():
    """Check if MinIO is reachable."""
    try:
        client = get_client()
        client.list_buckets()
        return True
    except Exception:
        return False


def upload_pdf(run_id, doc_id, file_path):
    """Upload a PDF file to MinIO.

    Args:
        run_id: The OCR run identifier.
        doc_id: The document identifier.
        file_path: Local path to the PDF file.

    Returns:
        The MinIO object key.
    """
    client = get_client()
    init_buckets()

    filename = os.path.basename(file_path)
    object_key = f"{run_id}/{doc_id}/{filename}"

    client.fput_object(
        BUCKET_PDFS,
        object_key,
        file_path,
        content_type="application/pdf",
    )
    return object_key


def upload_markdown(run_id, doc_id, file_path):
    """Upload a markdown file to MinIO.

    Args:
        run_id: The OCR run identifier.
        doc_id: The document identifier.
        file_path: Local path to the markdown file.

    Returns:
        The MinIO object key.
    """
    client = get_client()
    init_buckets()

    filename = os.path.basename(file_path)
    object_key = f"{run_id}/{doc_id}/{filename}"

    client.fput_object(
        BUCKET_MARKDOWN,
        object_key,
        file_path,
        content_type="text/markdown",
    )
    return object_key


def upload_markdown_text(run_id, doc_id, filename, text_content):
    """Upload markdown content directly from a string.

    Args:
        run_id: The OCR run identifier.
        doc_id: The document identifier.
        filename: The filename to use in MinIO.
        text_content: The markdown text content.

    Returns:
        The MinIO object key.
    """
    client = get_client()
    init_buckets()

    object_key = f"{run_id}/{doc_id}/{filename}"
    data = text_content.encode("utf-8")

    client.put_object(
        BUCKET_MARKDOWN,
        object_key,
        io.BytesIO(data),
        length=len(data),
        content_type="text/markdown",
    )
    return object_key


def download_file(bucket, object_key, dest_path):
    """Download a file from MinIO to a local path.

    Args:
        bucket: The bucket name (BUCKET_PDFS or BUCKET_MARKDOWN).
        object_key: The object key in the bucket.
        dest_path: Local destination path.
    """
    client = get_client()
    client.fget_object(bucket, object_key, dest_path)


def get_file_content(bucket, object_key):
    """Get file content as bytes.

    Args:
        bucket: The bucket name.
        object_key: The object key.

    Returns:
        File content as bytes.
    """
    client = get_client()
    response = client.get_object(bucket, object_key)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def get_markdown_text(object_key):
    """Get markdown content as a string.

    Args:
        object_key: The object key in the markdown bucket.

    Returns:
        Markdown text content.
    """
    content = get_file_content(BUCKET_MARKDOWN, object_key)
    return content.decode("utf-8")


def list_objects(bucket, prefix=""):
    """List objects in a bucket with optional prefix.

    Args:
        bucket: The bucket name.
        prefix: Optional prefix filter.

    Returns:
        List of object keys.
    """
    client = get_client()
    objects = client.list_objects(bucket, prefix=prefix, recursive=True)
    return [obj.object_name for obj in objects]


def delete_run_objects(run_id):
    """Delete all objects for a given run from both buckets.

    Args:
        run_id: The OCR run identifier.
    """
    client = get_client()
    for bucket in [BUCKET_PDFS, BUCKET_MARKDOWN]:
        if client.bucket_exists(bucket):
            objects = client.list_objects(bucket, prefix=f"{run_id}/", recursive=True)
            for obj in objects:
                client.remove_object(bucket, obj.object_name)


def get_storage_stats():
    """Get storage usage statistics.

    Returns:
        Dict with total objects and estimated size per bucket.
    """
    client = get_client()
    stats = {}

    for bucket in [BUCKET_PDFS, BUCKET_MARKDOWN]:
        if client.bucket_exists(bucket):
            total_objects = 0
            total_size = 0
            for obj in client.list_objects(bucket, recursive=True):
                total_objects += 1
                total_size += obj.size or 0
            stats[bucket] = {
                "objects": total_objects,
                "size_bytes": total_size,
            }
        else:
            stats[bucket] = {"objects": 0, "size_bytes": 0}

    return stats
