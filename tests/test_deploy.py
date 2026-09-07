"""Pushing from a job nobody is watching.

Two schedulers push from this machine to a public repository, and both do it
while the user is asleep or elsewhere. What must never happen is a commit
containing something nobody meant to publish, so the one rule these tests
protect is that only the paths handed in are ever staged.
"""
from __future__ import annotations

import subprocess

import pytest

from publish.deploy import commit_and_push, has_changes


def git(root, *arguments):
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True, text=True, check=True,
    )


@pytest.fixture
def repo(tmp_path):
    """A repository with one commit and no remote."""
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "published").mkdir()
    (tmp_path / "published" / "page.html").write_text("first", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("not for the world", encoding="utf-8")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "start")
    return tmp_path


def test_nothing_changed_means_nothing_committed(repo):
    ok, lines = commit_and_push(repo, ["published"], "no-op")

    assert ok
    assert "nothing changed" in lines[0]


def test_only_the_named_paths_are_staged(repo):
    """The rule that keeps an unattended push from publishing a stray edit."""
    (repo / "published" / "page.html").write_text("second", encoding="utf-8")
    (repo / "secret.txt").write_text("still not for the world, and edited", encoding="utf-8")

    # No remote here, so the push fails - after the commit, which is what we check.
    commit_and_push(repo, ["published"], "publish the page")

    committed = git(repo, "show", "--name-only", "--format=", "HEAD").stdout.split()
    assert committed == ["published/page.html"]

    still_dirty = git(repo, "status", "--porcelain").stdout
    assert "secret.txt" in still_dirty


def test_a_failed_push_is_reported_rather_than_raised(repo):
    (repo / "published" / "page.html").write_text("third", encoding="utf-8")

    ok, lines = commit_and_push(repo, ["published"], "publish again")

    assert not ok
    assert any("push failed" in line for line in lines)


def test_has_changes_sees_only_what_it_was_asked_about(repo):
    (repo / "secret.txt").write_text("edited", encoding="utf-8")

    assert has_changes(repo, ["published"]) is False
    assert has_changes(repo, ["secret.txt"]) is True
