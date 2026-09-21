"""One cloud tick: the daily refresh when it is due, then the agent.

GitHub runs this every fifteen minutes (.github/workflows/lab.yml) in place of
the two Windows scheduled tasks, so the site keeps itself current with the PC
switched off. It reuses those tasks' scripts unchanged - the cloud decides
only *when*, never *what*.

When the daily refresh is due: after 08:00 local time, once per day. A cloud
schedule is a request, not a promise - GitHub delays or drops scheduled runs
when it is busy - so "the 08:00 run" would sometimes never happen. Checking on
every tick whether today's refresh has run yet makes a late tick pick it up.
A failed refresh is retried on the next ticks, at most three times a day, so a
broken step cannot turn into ninety-six failures.

Prints both logs' new lines, because the Actions page is where anybody will
look when something is wrong.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import agent_tick  # noqa: E402
import daily_record  # noqa: E402

MARKER = ROOT / "data" / "processed" / "daily_last_run.txt"
DAILY_HOUR = 8
MAX_ATTEMPTS = 3


def _marker() -> tuple[str, int, bool]:
    """(day, attempts that day, succeeded)."""
    if not MARKER.exists():
        return "", 0, False
    parts = MARKER.read_text(encoding="utf-8").split()
    if len(parts) != 3:
        return "", 0, False
    return parts[0], int(parts[1]), parts[2] == "ok"


def daily_due(now: datetime, forced: bool) -> bool:
    if forced:
        return True
    day, attempts, ok = _marker()
    today = f"{now:%Y-%m-%d}"
    if now.hour < DAILY_HOUR:
        return False
    if day != today:
        return True
    return not ok and attempts < MAX_ATTEMPTS


def _tail(path: Path, start: int) -> None:
    if path.exists():
        text = path.read_text(encoding="utf-8", errors="replace")
        print(text[start:] if start <= len(text) else text[-4000:])


def _size(path: Path) -> int:
    return len(path.read_text(encoding="utf-8", errors="replace")) if path.exists() else 0


def main() -> int:
    now = datetime.now()
    forced = os.environ.get("LAB_FORCE_DAILY", "").lower() == "true"
    code = 0
    ran_daily = False

    if daily_due(now, forced):
        ran_daily = True
        day, attempts, _ = _marker()
        attempts = attempts + 1 if day == f"{now:%Y-%m-%d}" else 1
        start = _size(daily_record.LOG)
        print(f"--- daily refresh (attempt {attempts} today) ---", flush=True)
        result = daily_record.main()
        _tail(daily_record.LOG, start)
        MARKER.write_text(f"{now:%Y-%m-%d} {attempts} {'ok' if result == 0 else 'failed'}",
                          encoding="utf-8")
        code = code or result

    start = _size(agent_tick.LOG)
    print("--- agent tick ---", flush=True)
    result = agent_tick.main()
    _tail(agent_tick.LOG, start)
    code = code or result

    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"daily={'true' if ran_daily else 'false'}\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
