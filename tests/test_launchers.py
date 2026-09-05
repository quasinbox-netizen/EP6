"""The launchers, which are the first thing a new user touches and the last
thing anyone tests.

Every failure guarded here actually happened, and each one looked to the user
like "it does not start" with nothing on screen to say why.
"""
from __future__ import annotations

import pathlib
import re

import pytest

from config import load_config

ROOT = load_config().root
CMD_FILES = sorted(ROOT.glob("*.cmd"))


def test_there_are_cmd_launchers():
    assert CMD_FILES, "no .cmd launchers found - has the layout changed?"


@pytest.mark.parametrize("path", CMD_FILES, ids=lambda p: p.name)
def test_cmd_files_use_crlf(path: pathlib.Path):
    """cmd.exe needs CRLF, and the failure it produces is baffling.

    A running .cmd is re-read by byte offset. With LF-only endings those offsets
    land mid-line, and `call :label` fails with "The system cannot find the
    batch label specified" - but only for labels far enough into the file, so
    the same script appears to work until someone adds a subroutine near the
    end. .gitattributes asks for CRLF on checkout; this catches a file written
    afterwards by an editor or a script that did not honour it.
    """
    data = path.read_bytes()
    lone_lf = data.count(b"\n") - data.count(b"\r\n")
    assert lone_lf == 0, f"{path.name} has {lone_lf} LF-only line endings"


def test_posix_launcher_is_not_named_btc():
    """An extensionless `btc` beside `btc.cmd` breaks Windows for beginners.

    Explorer hides known extensions by default, so the two appear in the folder
    as identical entries called "btc" - and the one Windows cannot run sorts
    first. Double-clicking it does nothing, which is exactly the bug report
    this test exists to prevent.
    """
    assert not (ROOT / "btc").exists(), (
        "an extensionless 'btc' is back; it is indistinguishable from btc.cmd "
        "in Explorer. Name the POSIX launcher btc.sh."
    )
    assert (ROOT / "btc.sh").exists()


def test_posix_launcher_uses_lf_and_has_a_shebang():
    data = (ROOT / "btc.sh").read_bytes()
    assert data.startswith(b"#!"), "the shebang must be the first bytes"
    assert b"\r\n" not in data, "CRLF breaks the shebang on Linux and macOS"


def test_cmd_launcher_holds_the_window_open_when_double_clicked():
    """A console window closes the instant the script returns.

    Started from Explorer that makes a usage message, a missing interpreter or
    a failed install invisible - the window flashes and is gone, and the user
    reports that nothing happens. The launcher pauses on those paths, and only
    when Explorer started it (%cmdcmdline% carries /c; an interactive prompt
    does not).
    """
    text = (ROOT / "btc.cmd").read_text(encoding="utf-8", errors="replace")
    assert "cmdcmdline" in text, "no way to tell a double-click from a typed command"
    assert re.search(r"^\s*pause\s*$", text, re.M | re.I), "nothing holds the window open"

    labels = set(re.findall(r"^\s*:([A-Za-z_][\w]*)", text, re.M))
    called = set(re.findall(r"call\s+:([A-Za-z_][\w]*)", text))
    missing = called - labels
    assert not missing, f"btc.cmd calls labels it does not define: {sorted(missing)}"


def test_no_arguments_does_not_fall_through_to_the_usage_dump():
    """Double-clicking passes no arguments, so that path must be deliberate.

    Left to argparse it prints a wall of usage text and exits non-zero, which
    is not an answer to "what do I do now".
    """
    text = (ROOT / "btc.cmd").read_text(encoding="utf-8", errors="replace")
    assert re.search(r'if\s+"%~1"==""', text), "no branch for the no-argument case"
    assert "dashboard" in text, "the menu should point somewhere useful"
