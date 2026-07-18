"""
RAG Infrastructure Manager — Docker Compose lifecycle for PostgreSQL, Redis, MinIO, Qdrant.

Extends the existing docker_manager.py pattern to manage the RAG services.
"""

import os
import subprocess
import time

# Path to the docker-compose file
COMPOSE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docker-compose.rag.yml")
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def _run_compose(args: list, timeout: int = 60) -> tuple[bool, str]:
    """Run a docker compose command.

    Args:
        args: Command arguments after 'docker compose'.
        timeout: Command timeout in seconds.

    Returns:
        (success, message) tuple.
    """
    cmd = ["docker", "compose", "-f", COMPOSE_FILE] + args
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
    success, msg = _run_compose(["up", "-d", "--wait"], timeout=120)
    if success:
        return True, "RAG infrastructure started successfully."
    return False, f"Failed to start RAG infrastructure: {msg}"


def stop_rag_infrastructure() -> tuple[bool, str]:
    """Stop all RAG infrastructure services.

    Returns:
        (success, message) tuple.
    """
    success, msg = _run_compose(["stop"], timeout=30)
    if success:
        return True, "RAG infrastructure stopped."
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
    success, msg = _run_compose(args, timeout=30)
    if success:
        return True, "RAG infrastructure destroyed."
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
            ["docker", "compose", "-f", COMPOSE_FILE, "ps", "--format", "json"],
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

    # Initialize database
    success, msg = init_rag_database()
    messages.append(f"Database: {msg}")

    # Initialize storage
    success2, msg2 = init_rag_storage()
    messages.append(f"Storage: {msg2}")

    # Initialize vector store
    success3, msg3 = init_rag_vector_store()
    messages.append(f"Vector Store: {msg3}")

    all_ok = success and success2 and success3
    return all_ok, "\n".join(messages)
