"""The daily job: refresh the prices, then write down what the app claims.

Meant for a scheduler, so it has to behave like something nobody is watching:

* it ingests first, because `ledger --record` records from the last close in
  the database. Recording without refreshing would re-record the same origin
  every day - harmless, since the ledger replaces rows for an origin it
  already holds, but it would never advance;
* a failed ingest does not stop the recording. A day with no network is still
  a day the app can be held to what it already believed;
* everything goes to a log file rather than a console nobody sees, and the log
  is trimmed so an unattended job cannot fill a disk over a year.

Run it by hand the same way the scheduler does:

    .venv\\Scripts\\python.exe scripts\\daily_record.py
"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "data" / "processed" / "daily.log"
LOG_LIMIT_BYTES = 1_000_000

STEPS = (
    ("ingest", ["ingest", "--what", "prices"]),
    ("ledger", ["ledger", "--record"]),
)


def interpreter() -> str:
    """A console interpreter, even when the scheduler started pythonw.

    The task runs windowless so it does not flash a console every morning,
    but the CLI it calls is a console program and writes to stdout; handing
    the children pythonw leaves that output nowhere.
    """
    current = Path(sys.executable)
    if current.name.lower() == "pythonw.exe":
        console = current.with_name("python.exe")
        if console.exists():
            return str(console)
    return sys.executable


def log(handle, message: str) -> None:
    handle.write(message.rstrip() + "\n")
    handle.flush()


def trim(path: Path) -> None:
    if not path.exists() or path.stat().st_size <= LOG_LIMIT_BYTES:
        return
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    path.write_text("\n".join(lines[len(lines) // 2:]) + "\n", encoding="utf-8")


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    trim(LOG)
    failures = []

    with LOG.open("a", encoding="utf-8") as handle:
        log(handle, "")
        log(handle, f"=== {datetime.now():%Y-%m-%d %H:%M:%S} ===")
        for name, arguments in STEPS:
            command = [interpreter(), str(ROOT / "src" / "cli.py"), *arguments]
            log(handle, f"--- {name}: {' '.join(arguments)}")
            try:
                finished = subprocess.run(
                    command, cwd=ROOT, capture_output=True, text=True,
                    encoding="utf-8", errors="replace", timeout=1800,
                )
            except Exception as error:  # a scheduler job must not raise
                log(handle, f"{name} could not start: {error}")
                failures.append(name)
                continue

            log(handle, finished.stdout.strip())
            if finished.stderr.strip():
                log(handle, f"[stderr] {finished.stderr.strip()}")
            if finished.returncode != 0:
                log(handle, f"{name} exited with {finished.returncode}")
                failures.append(name)

        # Only the recording decides the exit code. A missing network is a
        # normal day, not a failed job.
        if "ledger" in failures:
            log(handle, "RESULT: nothing was recorded today")
            return 1
        log(handle, "RESULT: recorded" + (" (ingest failed)" if failures else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
