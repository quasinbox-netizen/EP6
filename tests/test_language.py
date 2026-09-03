"""The repository is published in English; this keeps it that way.

The project was written in Polish and translated. Translation is a one-off act,
but staying translated is not: `{"kategoria": ...}` survived the sweep and rode
into a public repo, into a CSV column header, and onto the dashboard, because
nothing was watching. This test watches.

Two detectors, because Polish leaks in two forms:

1. Diacritics, which are unambiguous. The character class here is written out
   character by character on purpose. An earlier version used the range
   [A-z] style shortcut `[Ą-ż]`, which spans the Latin block and matches plain
   ASCII letters - with re.IGNORECASE it flagged the letter "i" in every file.
   A guard that cries wolf gets deleted, so this one enumerates.

2. ASCII-only Polish words, which diacritics cannot catch - "kategoria" is the
   case that got through. The list is deliberately conservative: every entry is
   a word that cannot appear in English prose or in an identifier. Ambiguous
   ones ("dla" reads as an acronym, "ale" is a beer, "data" and "do" are plain
   English) are left out. Missing a word is a bug report; a false positive is
   a test nobody trusts.
"""
from __future__ import annotations

import re

import pytest

from config import load_config

DIACRITICS = re.compile(r"[ąćęłńóśźż"
                        r"ĄĆĘŁŃÓŚŹŻ]")

POLISH_WORDS = re.compile(
    r"\b(?:kategoria|kategorie|kategorii|wynik|wyniki|wyniku|liczba|liczby"
    r"|zwrot|zwroty|okno|okna|wykres|wykresy|blad|bledy|plik|pliki|pliku"
    r"|sprawdz|oblicz|zapisz|uruchom|przyklad|przyklady|tylko|ktory|ktore"
    r"|jesli|jest|oraz|sredni|srednia|cena|ceny|dzien|miesiac|rok|lata"
    r"|zaden|kazdy|wiecej|mniej|niz|bardzo|teraz|potem|zawsze|nigdy|halvingi|halvingu|halvingow|swieca|swiece|okresu)\b",
    re.IGNORECASE,
)

SCANNED_SUFFIXES = {".py", ".md", ".toml", ".yml", ".yaml", ".cfg", ".txt", ".csv"}
SKIPPED_DIRECTORIES = {".git", ".venv", "__pycache__", ".pytest_tmp", ".pytest-tmp",
                       "node_modules", ".ruff_cache", "htmlcov"}

# This file necessarily contains the words it hunts for.
SELF = "test_language.py"

# Non-English text is sometimes the point. test_phase1_ingest.py feeds Polish
# and German into the ingest path on purpose, to prove descriptions survive the
# round trip through SQLite and a cp1252 Windows console. Marking such a line
# is an explicit, greppable claim that the text is a fixture rather than a leak
# - which is the whole difference between the two, and not something a regex
# can decide. Blanket-excluding the file would have hidden a real leak in it.
ALLOW_MARKER = "non-english-ok"


def scanned_files():
    root = load_config().root
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in SCANNED_SUFFIXES:
            continue
        if any(part in SKIPPED_DIRECTORIES for part in path.parts):
            continue
        if path.name == SELF:
            continue
        yield path


@pytest.mark.parametrize("detector,label", [
    (DIACRITICS, "Polish diacritics"),
    (POLISH_WORDS, "a Polish word"),
])
def test_repository_contains_no_polish(detector, label):
    offenders = []
    for path in scanned_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if ALLOW_MARKER in line:
                continue
            match = detector.search(line)
            if match:
                offenders.append(f"{path.name}:{number}: {match.group(0)!r} in {line.strip()[:70]}")

    assert not offenders, f"{label} found in {len(offenders)} place(s):\n" + "\n".join(offenders[:25])


def test_the_detectors_actually_detect():
    """A guard that cannot fail is decoration.

    Both patterns are asserted against a known-Polish string and a known-English
    one. The English case is the important half: it is what would have failed
    under the old [Ą-ż] range, and it is why this test exists at all.
    """
    assert DIACRITICS.search("zwrot ze świecy")
    assert POLISH_WORDS.search('{"kategoria": category}')

    english = "The first significant window is identified in the price index."
    assert not DIACRITICS.search(english)
    assert not POLISH_WORDS.search(english)


def test_the_allowlist_marker_is_not_a_blanket_exemption():
    """The escape hatch must cost a line, not a file.

    A per-file exclusion list rots: someone excludes a file for one fixture and
    every later leak in it goes unseen. Marking is per line, so an exemption
    stays scoped to the exact text it was granted for.
    """
    marked = [
        path.name
        for path in scanned_files()
        if ALLOW_MARKER in path.read_text(encoding="utf-8", errors="ignore")
    ]
    assert marked, "no line uses the marker - has the mechanism been bypassed?"
