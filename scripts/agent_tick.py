"""One agent tick, for the scheduler: look, write it down, publish if it matters.

Separate from the morning job because it runs on a different clock - every
fifteen minutes against once a day - and because it must stay cheap. It writes
one small JSON file and pushes only when the agent's state changed or the feed
has gone stale; a quarter-hourly push would be a hundred commits a day.

Runs windowless, so everything goes to a log rather than a console nobody
sees. A tick that cannot reach the network is a normal event, not a failure:
the agent simply has nothing to record that minute.

By hand, the same way the scheduler does it:

    .venv\\Scripts\\python.exe scripts\\agent_tick.py
"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "data" / "processed" / "agent.log"
LOG_LIMIT_BYTES = 500_000


def interpreter() -> str:
    """A console interpreter even when the scheduler started pythonw."""
    current = Path(sys.executable)
    if current.name.lower() == "pythonw.exe":
        console = current.with_name("python.exe")
        if console.exists():
            return str(console)
    return sys.executable


def trim(path: Path) -> None:
    if not path.exists() or path.stat().st_size <= LOG_LIMIT_BYTES:
        return
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    path.write_text("\n".join(lines[len(lines) // 2:]) + "\n", encoding="utf-8")


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    trim(LOG)
    command = [interpreter(), str(ROOT / "src" / "cli.py"), "agent"]

    try:
        finished = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=600,
        )
    except Exception as error:  # a scheduled tick must never raise
        with LOG.open("a", encoding="utf-8") as handle:
            handle.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}  could not start: {error}\n")
        return 1

    with LOG.open("a", encoding="utf-8") as handle:
        for line in finished.stdout.strip().splitlines():
            handle.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {line}\n")
        if finished.stderr.strip():
            handle.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}  [stderr] "
                         f"{finished.stderr.strip().splitlines()[-1]}\n")
    return finished.returncode


if __name__ == "__main__":
    raise SystemExit(main())
