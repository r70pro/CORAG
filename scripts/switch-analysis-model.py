#!/usr/bin/env python3
"""CLI wrapper for the same guarded analysis switch used by React."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis_profiles import (  # noqa: E402
    ANALYSIS_PROFILES,
    TERMINAL_STATES,
    get_operation,
    start_switch,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=sorted(ANALYSIS_PROFILES))
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args()
    operation = start_switch(args.model)
    operation_id = operation["id"]
    last_state = ""
    while True:
        current = get_operation(operation_id)
        if not current:
            raise RuntimeError(f"Switch operation disappeared: {operation_id}")
        marker = f"{current['state']}:{current.get('message', '')}:{current.get('progress', 0)}"
        if marker != last_state:
            print(
                json.dumps(
                    {
                        "id": operation_id,
                        "state": current["state"],
                        "progress": current.get("progress", 0),
                        "message": current.get("message", ""),
                    }
                )
            )
            last_state = marker
        if current["state"] in TERMINAL_STATES:
            return 0 if current["state"] == "completed" else 1
        time.sleep(max(0.2, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
