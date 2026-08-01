#!/usr/bin/env bash
set -u

# Desktop-friendly launcher for the supervised KIRAG installation.
APP_URL="${KIRAG_APP_URL:-http://127.0.0.1:3000}"
API_LIVE_URL="${KIRAG_API_LIVE_URL:-http://127.0.0.1:8001/livez}"
SERVICE="${KIRAG_FRONTEND_SERVICE:-kirag-frontend.service}"
INFRA_SERVICE="${KIRAG_INFRA_SERVICE:-kirag-infrastructure.service}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STARTUP_PAGE="${KIRAG_STARTUP_PAGE:-${ROOT_DIR}/deploy/desktop/startup.html}"
LOG_DIR="${XDG_STATE_HOME:-${HOME}/.local/state}/kirag"
LOG_FILE="${LOG_DIR}/launcher.log"

mkdir -p "$LOG_DIR"
exec >>"$LOG_FILE" 2>&1

notify() {
    if command -v notify-send >/dev/null 2>&1; then
        notify-send "KIRAG" "$1"
    fi
}

open_app() {
    if command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$APP_URL" >/dev/null 2>&1
    else
        notify "KIRAG is ready at $APP_URL"
    fi
}

open_startup_page() {
    if command -v xdg-open >/dev/null 2>&1 && [[ -f "$STARTUP_PAGE" ]]; then
        xdg-open "$STARTUP_PAGE" >/dev/null 2>&1
    fi
}

if curl --silent --fail --max-time 2 "$APP_URL" >/dev/null 2>&1 &&
        curl --silent --fail --max-time 2 "$API_LIVE_URL" >/dev/null 2>&1; then
    open_app
    exit 0
fi

notify "Starting KIRAG…"
open_startup_page

# Always submit a start transaction when either readiness check failed. This
# repairs a partially running stack too: starting an already-active frontend
# still starts any inactive Required dependencies.
if ! systemctl --user start --no-block "$SERVICE" "$INFRA_SERVICE"; then
    notify "KIRAG could not be started. See $LOG_FILE"
    exit 1
fi

# Cold model/container startup can take several minutes. Open the UI as soon
# as the frontend answers, without tying readiness to a fixed sleep.
for _attempt in $(seq 1 900); do
    if curl --silent --fail --max-time 2 "$APP_URL" >/dev/null 2>&1 &&
            curl --silent --fail --max-time 2 "$API_LIVE_URL" >/dev/null 2>&1; then
        # Do not rely solely on a file:// startup page to detect HTTP services.
        # Browser cross-origin/local-network policies can leave that page stuck
        # even though the launcher's own readiness probes have succeeded.
        open_app
        exit 0
    fi
    sleep 1
done

notify "KIRAG did not become ready. See $LOG_FILE"
exit 1
