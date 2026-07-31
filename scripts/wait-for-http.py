#!/usr/bin/env python3
"""Wait for an HTTP endpoint using bounded exponential backoff."""

from __future__ import annotations

import argparse
import time
import urllib.error
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    deadline = time.monotonic() + args.timeout
    delay = 0.5
    last_error = "not attempted"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(args.url, timeout=5) as response:
                if 200 <= response.status < 300:
                    return 0
                last_error = f"HTTP {response.status}"
        except (OSError, urllib.error.URLError) as exc:
            last_error = str(exc)
        time.sleep(min(delay, max(0, deadline - time.monotonic())))
        delay = min(delay * 1.7, 10)
    print(f"Timed out waiting for {args.url}: {last_error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
