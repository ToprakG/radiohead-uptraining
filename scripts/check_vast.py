"""Verify the Vast.ai API key with a read-only account request."""

import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def api_key():
    env_file = Path(__file__).resolve().parents[1] / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "VAST_API_KEY":
                return value.strip().strip("'\"")
    return os.environ.get("VAST_API_KEY", "").strip()


def main():
    key = api_key()
    if not key:
        print("Set VAST_API_KEY in the repository's .env.", file=sys.stderr)
        return 2

    request = Request(
        "https://console.vast.ai/api/v0/users/current/",
        headers={"Authorization": f"Bearer {key}"},
    )
    try:
        with urlopen(request, timeout=15):
            print("Vast.ai authentication verified.")
    except HTTPError as error:
        try:
            details = json.loads(error.read(4096))
        except ValueError:
            details = {}
        if isinstance(details, dict) and details.get("msg") == "Invalid user key":
            print("Vast.ai rejected VAST_API_KEY as invalid.", file=sys.stderr)
        else:
            print(f"Vast.ai account check returned HTTP {error.code}.", file=sys.stderr)
        return 1
    except (URLError, TimeoutError, ValueError):
        print("Unable to complete the Vast.ai account check.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
