"""Sending the built files to where the web server reads them.

Two jobs push from this machine without anyone watching - the morning
refresh and the agent's tick - and both push to a public repository. The
rules that make that safe live here rather than in each caller:

* only the paths handed in are staged. A job that ran `git add -A` would
  eventually push somebody's half-finished edit to a public repo, and the
  first anyone would know of it is when it appeared on the site;
* nothing is committed when nothing changed, so the history does not fill
  with empty mornings;
* every git call returns its output to the caller for the log instead of
  raising. A scheduler job that dies on a network hiccup is a job that
  silently stops publishing.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

TIMEOUT = 600


def _git(root: Path, arguments: list) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=TIMEOUT,
    )


def has_changes(root: Path, paths: list) -> bool | None:
    """True if any of `paths` differs from HEAD. None if git could not say."""
    finished = _git(root, ["status", "--porcelain", "--", *paths])
    if finished.returncode != 0:
        return None
    return bool(finished.stdout.strip())


def commit_and_push(root: Path, paths: list, message: str) -> tuple[bool, list]:
    """Stage exactly `paths`, commit them, push. Returns (ok, log lines)."""
    root = Path(root)
    lines = []

    changed = has_changes(root, paths)
    if changed is None:
        return False, ["git status failed"]
    if not changed:
        return True, ["nothing changed; nothing pushed"]

    for name, arguments in (
        ("add", ["add", "--", *paths]),
        ("commit", ["commit", "-m", message]),
        ("push", ["push", "origin", "HEAD"]),
    ):
        finished = _git(root, arguments)
        if finished.returncode != 0:
            lines.append(f"{name} failed: {(finished.stderr or finished.stdout).strip()}")
            return False, lines
        lines.append(f"{name}: {(finished.stdout or 'ok').strip().splitlines()[0] if finished.stdout.strip() else 'ok'}")
    return True, lines
