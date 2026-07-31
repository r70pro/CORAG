"""
RAG Infrastructure Manager — Docker Compose lifecycle for PostgreSQL, Redis, MinIO, Qdrant.

Extends the existing docker_manager.py pattern to manage the RAG services.
"""

import subprocess
import time
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

from audit_log import audit_event
from settings_manager import WORKSPACE_DIR


def _installed_compose_file() -> Path:
    """Return the bundled Compose file in a checkout or an installed wheel."""
    checkout_file = Path(__file__).resolve().with_name("docker-compose.rag.yml")
    if checkout_file.is_file():
        return checkout_file

    try:
        package = distribution("kirag")
    except PackageNotFoundError as exc:
        raise FileNotFoundError("KIRAG's bundled docker-compose.rag.yml was not found") from exc

    for entry in package.files or ():
        if entry.as_posix().endswith("share/kirag/docker-compose.rag.yml"):
            installed_file = Path(package.locate_file(entry)).resolve()
            if installed_file.is_file():
                return installed_file
    raise FileNotFoundError("KIRAG's bundled docker-compose.rag.yml was not found")


COMPOSE_FILE = str(_installed_compose_file())
_checkout_root = Path(__file__).resolve().parent
PROJECT_DIR = str(
    _checkout_root
    if (_checkout_root / "docker-compose.rag.yml").is_file()
    else Path(WORKSPACE_DIR).resolve().parent
)


def _run_compose(args: list, timeout: int = 60) -> tuple[bool, str]:
    """Run a docker compose command.

    Args:
        args: Command arguments after 'docker compose'.
        timeout: Command timeout in seconds.

    Returns:
        (success, message) tuple.
    """
    cmd = [
        "docker",
        "compose",
        "--project-directory",
        PROJECT_DIR,
        "-f",
        COMPOSE_FILE,
        *args,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=PROJECT_DIR,
        )
        if result.returncode == 0:
            return True, result.stdout.strip() or "OK"
        else:
            return False, result.stderr.strip() or f"Exit code {result.returncode}"
    except subprocess.TimeoutExpired:
        return False, f"Command timed out after {timeout}s"
    except Exception as e:
        return False, str(e)


def start_rag_infrastructure() -> tuple[bool, str]:
    """Start all RAG infrastructure services (PostgreSQL, Redis, MinIO, Qdrant).

    Returns:
        (success, message) tuple.
    """
    audit_event("infrastructure_create", "attempt", stack="rag")
    success, msg = _run_compose(["up", "-d", "--wait"], timeout=120)
    if success:
        audit_event("infrastructure_create", "success", stack="rag")
        return True, "RAG infrastructure started successfully."
    audit_event("infrastructure_create", "failure", stack="rag", error=msg)
    return False, f"Failed to start RAG infrastructure: {msg}"


def stop_rag_infrastructure() -> tuple[bool, str]:
    """Stop all RAG infrastructure services.

    Returns:
        (success, message) tuple.
    """
    audit_event("infrastructure_shutdown", "attempt", stack="rag")
    success, msg = _run_compose(["stop"], timeout=30)
    if success:
        audit_event("infrastructure_shutdown", "success", stack="rag")
        return True, "RAG infrastructure stopped."
    audit_event("infrastructure_shutdown", "failure", stack="rag", error=msg)
    return False, f"Failed to stop: {msg}"


def destroy_rag_infrastructure(remove_volumes: bool = False) -> tuple[bool, str]:
    """Stop and remove all RAG infrastructure containers.

    Args:
        remove_volumes: If True, also remove persistent volumes.

    Returns:
        (success, message) tuple.
    """
    args = ["down"]
    if remove_volumes:
        args.append("-v")
    audit_event(
        "infrastructure_delete",
        "attempt",
        stack="rag",
        remove_volumes=remove_volumes,
    )
    success, msg = _run_compose(args, timeout=30)
    if success:
        audit_event(
            "infrastructure_delete",
            "success",
            stack="rag",
            remove_volumes=remove_volumes,
        )
        return True, "RAG infrastructure destroyed."
    audit_event(
        "infrastructure_delete",
        "failure",
        stack="rag",
        remove_volumes=remove_volumes,
        error=msg,
    )
    return False, f"Failed to destroy: {msg}"


def get_rag_service_status() -> dict[str, str]:
    """Get the status of each RAG service.

    Returns:
        Dict mapping service name to status string.
    """
    services = {
        "postgres": "unknown",
        "redis": "unknown",
        "minio": "unknown",
        "qdrant": "unknown",
    }

    try:
        result = subprocess.run(
            [
                "docker",
                "compose",
                "--project-directory",
                PROJECT_DIR,
                "-f",
                COMPOSE_FILE,
                "ps",
                "--format",
                "json",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=PROJECT_DIR,
        )
        if result.returncode == 0 and result.stdout.strip():
            import json

            # docker compose ps --format json outputs one JSON object per line
            for line in result.stdout.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    container = json.loads(line)
                    name = container.get("Service", container.get("Name", ""))
                    state = container.get("State", container.get("Status", "unknown"))
                    health = container.get("Health", "")

                    for service_key in services:
                        if service_key in name.lower():
                            if "running" in str(state).lower():
                                if health and "unhealthy" in str(health).lower():
                                    services[service_key] = "unhealthy"
                                elif health and "healthy" in str(health).lower():
                                    services[service_key] = "healthy"
                                else:
                                    services[service_key] = "running"
                            elif "exited" in str(state).lower():
                                services[service_key] = "stopped"
                            else:
                                services[service_key] = str(state).lower()
                            break
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass

    return services


def get_rag_status_html() -> str:
    """Generate HTML status badges for all RAG services.

    Returns:
        HTML string with status badges.
    """
    statuses = get_rag_service_status()

    badge_map = {
        "healthy": ("badge-success", "✓"),
        "running": ("badge-running", "↻"),
        "unhealthy": ("badge-failed", "✗"),
        "stopped": ("badge-stopped", "⏹"),
        "unknown": ("badge-idle", "?"),
    }

    labels = {
        "postgres": "PostgreSQL",
        "redis": "Redis",
        "minio": "MinIO",
        "qdrant": "Qdrant",
    }

    badges = []
    for service, status in statuses.items():
        css_class, icon = badge_map.get(status, ("badge-idle", "?"))
        label = labels.get(service, service)
        badges.append(f"<span class='{css_class}' style='margin:2px 4px;'>{icon} {label}</span>")

    return "<div style='display:flex; flex-wrap:wrap; gap:4px;'>" + "".join(badges) + "</div>"


def is_rag_infrastructure_ready() -> bool:
    """Check if all RAG infrastructure services are healthy.

    Returns:
        True if all services report healthy/running.
    """
    statuses = get_rag_service_status()
    return all(s in ("healthy", "running") for s in statuses.values())


def init_rag_database():
    """Initialize the PostgreSQL schema for RAG.

    Should be called after infrastructure is started.

    Returns:
        (success, message) tuple.
    """
    try:
        from rag.db import init_schema, is_healthy

        # Wait for PostgreSQL to be ready
        for _ in range(10):
            if is_healthy():
                break
            time.sleep(1)
        else:
            return False, "PostgreSQL not ready after 10 seconds."

        init_schema()
        return True, "Database schema initialized."
    except Exception as e:
        return False, f"Failed to initialize database: {e}"


def init_rag_storage():
    """Initialize MinIO buckets for RAG.

    Should be called after infrastructure is started.

    Returns:
        (success, message) tuple.
    """
    try:
        from rag.storage import init_buckets, is_healthy

        for _ in range(10):
            if is_healthy():
                break
            time.sleep(1)
        else:
            return False, "MinIO not ready after 10 seconds."

        init_buckets()
        return True, "Storage buckets initialized."
    except Exception as e:
        return False, f"Failed to initialize storage: {e}"


def init_rag_vector_store():
    """Initialize Qdrant collection for RAG.

    Should be called after infrastructure is started.

    Returns:
        (success, message) tuple.
    """
    try:
        from rag.embedding import init_collection, is_healthy

        for _ in range(10):
            if is_healthy():
                break
            time.sleep(1)
        else:
            return False, "Qdrant not ready after 10 seconds."

        init_collection()
        return True, "Vector collection initialized."
    except Exception as e:
        return False, f"Failed to initialize vector store: {e}"


def initialize_rag_services() -> tuple[bool, str]:
    """Idempotently initialize schemas, buckets, and the vector collection."""
    messages = []

    success, msg = init_rag_database()
    messages.append(f"Database: {msg}")

    success2, msg2 = init_rag_storage()
    messages.append(f"Storage: {msg2}")

    success3, msg3 = init_rag_vector_store()
    messages.append(f"Vector Store: {msg3}")
    return success and success2 and success3, "\n".join(messages)


def start_and_init_rag() -> tuple[bool, str]:
    """Start RAG infrastructure and initialize all services.

    This is the main entry point for bringing up the full RAG stack.

    Returns:
        (success, message) tuple with detailed status.
    """
    messages = []

    # Start Docker services
    success, msg = start_rag_infrastructure()
    messages.append(f"Infrastructure: {msg}")
    if not success:
        return False, "\n".join(messages)

    # Wait a moment for services to stabilize
    time.sleep(2)

    initialized, init_message = initialize_rag_services()
    messages.append(init_message)
    return initialized, "\n".join(messages)
