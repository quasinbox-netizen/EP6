"""A thin HTTP layer: retry with backoff, a fixed User-Agent, clear errors.

Every data source goes through this module so that retrying and rate limits
live in one place and tests can replace them.

Important: error messages ALWAYS pass through `redact`. The API key travels in
the query string, so a raw URL in an exception body would end up in logs, in
the console and in any traceback you paste into a bug report.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Any

USER_AGENT = "btc-cycle-lab/0.1 (local research)"
DEFAULT_TIMEOUT = 30

# Parameter names whose values must never reach a message.
SECRET_PARAMS = ("api_key", "apikey", "token", "access_key", "secret")
_SECRET_PATTERN = re.compile(
    r"(?i)\b(" + "|".join(SECRET_PARAMS) + r")=([^&\s\"']+)"
)


def redact(text: str) -> str:
    """Replace secret values in a string with ***."""
    return _SECRET_PATTERN.sub(r"\1=***", text)


class FetchError(RuntimeError):
    """Failure while fetching data from an external source."""

    def __init__(self, message: str):
        super().__init__(redact(message))


def get_json(
    url: str,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = 3,
    backoff: float = 1.5,
    sleep: float = 0.0,
) -> Any:
    """Fetch JSON. Retries on network errors and 5xx/429, not on other 4xx."""
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
            if sleep:
                time.sleep(sleep)
            return json.loads(payload)
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in (429, 500, 502, 503, 504):
                # The response body usually says WHAT is wrong with the request.
                try:
                    detail = exc.read().decode("utf-8", "replace")[:300]
                except Exception:
                    detail = ""
                raise FetchError(f"{url} -> HTTP {exc.code} {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(backoff ** attempt)
    raise FetchError(f"{url} -> failed after {retries} attempts: {last_error!r}")


# Binance answers HTTP 451 to requests from the United States, which is where
# GitHub's runners are. data-api.binance.vision is Binance's own market-data
# mirror: the same public endpoints and the same numbers, without the region
# block. The main host stays first so a machine that can reach it behaves
# exactly as before.
BINANCE_HOSTS = ("https://api.binance.com", "https://data-api.binance.vision")


def get_binance(path: str, **kwargs) -> Any:
    """GET a public Binance endpoint, falling back to the mirror if blocked."""
    errors = []
    for host in BINANCE_HOSTS:
        try:
            return get_json(host + path, **kwargs)
        except FetchError as error:
            errors.append(str(error))
    raise FetchError(" | ".join(errors))
