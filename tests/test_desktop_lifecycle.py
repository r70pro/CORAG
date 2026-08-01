"""Regression tests for desktop-style start and shutdown semantics."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_shutdown_disables_and_stops_the_complete_system_stack():
    unit = (ROOT / "deploy/systemd/kirag-shutdown.service.in").read_text()

    assert "systemctl disable --now" in unit
    assert "kirag-frontend.service" in unit
    assert "kirag-api.service" in unit
    assert "kirag-infrastructure.service" in unit
    assert "--user --machine=@KIRAG_USER@.host disable --now" in unit


def test_desktop_launcher_starts_user_stack_without_enabling_boot_start():
    launcher = (ROOT / "scripts/launch-kirag.sh").read_text()

    assert 'systemctl --user start --no-block "$SERVICE" "$INFRA_SERVICE"' in launcher
    assert "systemctl enable" not in launcher
    assert 'curl --silent --fail --max-time 2 "$API_LIVE_URL"' in launcher
    assert 'open_startup_page' in launcher
    readiness_loop = launcher.split("for _attempt", 1)[1]
    assert "open_app" in readiness_loop
    assert "STARTUP_PAGE_OPENED == 0" not in readiness_loop


def test_startup_page_redirects_when_the_frontend_is_ready():
    page = (ROOT / "deploy/desktop/startup.html").read_text()

    assert "KIRAG is starting" in page
    assert "http://127.0.0.1:3000" in page
    assert "window.location.replace(appUrl)" in page


def test_frontend_cannot_outlive_the_api_service():
    unit = (ROOT / "deploy/systemd-user/kirag-frontend.service.in").read_text()

    assert "BindsTo=kirag-api.service" in unit


def test_api_does_not_wait_for_slow_inference_startup():
    unit = (ROOT / "deploy/systemd-user/kirag-api.service.in").read_text()
    assert "Wants=kirag-infrastructure.service" in unit
    assert "Requires=kirag-infrastructure.service" not in unit
    assert "After=kirag-infrastructure.service" not in unit


def test_installers_do_not_enable_application_autostart():
    system_installer = (ROOT / "scripts/install-systemd-services.sh").read_text()
    user_installer = (ROOT / "scripts/install-user-services.sh").read_text()
    desktop_installer = (ROOT / "scripts/install-desktop-launcher.sh").read_text()

    assert "systemctl disable kirag-infrastructure kirag-api kirag-frontend" in system_installer
    assert "systemctl enable kirag-infrastructure" not in system_installer
    assert "systemctl --user disable --now kirag-frontend.service kirag-api.service kirag-infrastructure.service" in user_installer
    assert "systemctl --user enable kirag-api.service" not in user_installer
    assert '"$ROOT_DIR/scripts/install-user-services.sh"' in desktop_installer


def test_user_shutdown_stops_containers_without_root():
    unit = (ROOT / "deploy/systemd-user/kirag-shutdown.service.in").read_text()
    infra = (ROOT / "deploy/systemd-user/kirag-infrastructure.service.in").read_text()

    assert "systemctl --user disable --now" in unit
    assert "sudo" not in unit
    assert "ExecStop=/usr/bin/docker compose" in infra
    assert "sudo" not in infra
