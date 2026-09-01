"""Cienka warstwa HTTP: retry z backoffem, staly User-Agent, jasne bledy.

Kazde zrodlo danych przechodzi przez ten modul, zeby ponawianie i limity
zapytan byly w jednym miejscu, a testy mogly je podmienic.

Wazne: komunikaty bledow ZAWSZE przechodza przez `redact`. Klucz API siedzi
w query stringu, wiec surowy URL w tresci wyjatku wyladowalby w logach,
w konsoli i w tracebacku wyslanym komukolwiek do pomocy.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Any

USER_AGENT = "btc-cycle-lab/0.1 (local research; contact: local)"
DEFAULT_TIMEOUT = 30

# Nazwy parametrow, ktorych wartosci nigdy nie moga trafic do komunikatu.
SECRET_PARAMS = ("api_key", "apikey", "token", "access_key", "secret")
_SECRET_PATTERN = re.compile(
    r"(?i)\b(" + "|".join(SECRET_PARAMS) + r")=([^&\s\"']+)"
)


def redact(text: str) -> str:
    """Zamienia wartosci sekretow w tekscie na ***."""
    return _SECRET_PATTERN.sub(r"\1=***", text)


class FetchError(RuntimeError):
    """Blad pobrania danych ze zrodla zewnetrznego."""

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
    """Pobiera JSON. Ponawia przy bledach sieci i 5xx/429, nie przy 4xx."""
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
                # Tresc odpowiedzi zwykle mowi, CO jest nie tak z zapytaniem.
                try:
                    detail = exc.read().decode("utf-8", "replace")[:300]
                except Exception:
                    detail = ""
                raise FetchError(f"{url} -> HTTP {exc.code} {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(backoff ** attempt)
    raise FetchError(f"{url} -> nieudane po {retries} probach: {last_error!r}")
