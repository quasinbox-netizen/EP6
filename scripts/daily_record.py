"""The daily job: refresh the data, record the claims, republish the site.

Meant for a scheduler, so it has to behave like something nobody is watching:

* it ingests first, because everything downstream records from the last close
  in the database. Recording without refreshing would re-record the same
  origin every day - harmless, since the ledger replaces rows for an origin it
  already holds, but it would never advance;
* a failed step does not stop the ones after it. A day with no network is
  still a day the app can be held to what it already believed;
* the publish step commits and pushes, because the site is served from the
  repository and a build nobody sends is a build nobody sees. It commits ONLY
  the built pages, so an unrelated edit sitting in the working tree cannot be
  swept into an unattended commit;
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from publish.deploy import commit_and_push  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "data" / "processed" / "daily.log"
LOG_LIMIT_BYTES = 1_000_000
PUBLISHED = "docs"

STEPS = (
    ("ingest", ["ingest", "--what", "prices"]),
    ("ledger", ["ledger", "--record"]),
    ("paper", ["paper"]),
    ("publish", ["publish", "--out", PUBLISHED]),
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


def run(handle, name: str, command: list, timeout: int = 1800) -> bool:
    log(handle, f"--- {name}: {' '.join(command[1:])}")
    try:
        finished = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
    except Exception as error:  # a scheduler job must not raise
        log(handle, f"{name} could not start: {error}")
        return False
    log(handle, finished.stdout.strip())
    if finished.stderr.strip():
        log(handle, f"[stderr] {finished.stderr.strip()}")
    if finished.returncode != 0:
        log(handle, f"{name} exited with {finished.returncode}")
        return False
    return True



def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    trim(LOG)
    failures = []
    today = f"{datetime.now():%Y-%m-%d}"

    with LOG.open("a", encoding="utf-8") as handle:
        log(handle, "")
        log(handle, f"=== {datetime.now():%Y-%m-%d %H:%M:%S} ===")
        for name, arguments in STEPS:
            command = [interpreter(), str(ROOT / "src" / "cli.py"), *arguments]
            if not run(handle, name, command):
                failures.append(name)

        if "publish" in failures:
            log(handle, "RESULT: the site was not rebuilt, so nothing was pushed")
        else:
            ok, lines = commit_and_push(
                ROOT, [PUBLISHED], f"Refresh the published site for {today}"
            )
            for line in lines:
                log(handle, f"[git] {line}")
            if not ok:
                failures.append("git")

        # Only the recording and the publishing decide the exit code. A missing
        # network is a normal day, not a failed job.
        blocking = [name for name in failures if name in {"ledger", "publish", "git"}]
        if blocking:
            log(handle, f"RESULT: failed at {', '.join(blocking)}")
            return 1
        log(handle, "RESULT: recorded and published"
                    + (f" (soft failures: {', '.join(failures)})" if failures else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
