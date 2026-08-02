"""
Centralised secret/configuration resolution for backing services.

All credentials fall back to environment variables (or a .env file loaded by
settings_manager) and never require hardcoded plaintext in source code.
Defaults are provided only as a last resort for local single-developer setups.
"""

import os

# Credentials are resolved from the environment first. These are the same
# variable names consumed by docker-compose.rag.yml, so the app and the
# containers stay in sync without duplicating plaintext.
UNSAFE_DEFAULT_DB_PASSWORD = "change_me_in_production"
UNSAFE_DEFAULT_MINIO_ACCESS_KEY = "minio_access_5c6d3284f18b"
UNSAFE_DEFAULT_MINIO_SECRET_KEY = "change_me_minio_secret"

DEFAULT_DB_PASSWORD = UNSAFE_DEFAULT_DB_PASSWORD
DEFAULT_MINIO_ACCESS_KEY = UNSAFE_DEFAULT_MINIO_ACCESS_KEY
DEFAULT_MINIO_SECRET_KEY = UNSAFE_DEFAULT_MINIO_SECRET_KEY


def _ensure_dotenv_loaded():
    if "OLMOCR_PG_PASS" not in os.environ or "OLMOCR_MINIO_SECRET_KEY" not in os.environ:
        dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        if os.path.exists(dotenv_path):
            try:
                with open(dotenv_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip("'\"")
                        if k and v and k not in os.environ:
                            os.environ[k] = v
            except Exception:
                pass


_ensure_dotenv_loaded()


def get_db_password() -> str:
    return os.environ.get("OLMOCR_PG_PASS", DEFAULT_DB_PASSWORD)


def get_minio_access_key() -> str:
    return os.environ.get("OLMOCR_MINIO_ACCESS_KEY", DEFAULT_MINIO_ACCESS_KEY)


def get_minio_secret_key() -> str:
    return os.environ.get("OLMOCR_MINIO_SECRET_KEY", DEFAULT_MINIO_SECRET_KEY)


def credentials_are_default() -> bool:
    """Return True if any backing-service credential is still the unsafe default."""
    return (
        get_db_password() == UNSAFE_DEFAULT_DB_PASSWORD
        or get_minio_access_key() == UNSAFE_DEFAULT_MINIO_ACCESS_KEY
        or get_minio_secret_key() == UNSAFE_DEFAULT_MINIO_SECRET_KEY
    )
