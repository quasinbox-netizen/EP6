#!/usr/bin/env python3
"""Cross-platform launcher and installer. Works on Windows, macOS and Linux.

    python run.py ingest --what all      # download data
    python run.py all                    # full analysis
    python run.py dashboard              # browser dashboard
    python run.py test                   # test suite
    python run.py doctor                 # environment diagnostics

On first run this creates a virtual environment next to this file and installs
the dependencies from requirements.txt. Nothing is installed system-wide and
nothing is written outside this directory.

This file uses only the standard library, so it runs before any dependency
exists. Do not import pandas or anything else from here.

Why a Python launcher instead of shell scripts: the venv layout differs between
platforms (Scripts/python.exe on Windows, bin/python elsewhere), and shell
scripts would need one copy per platform plus an execute bit that does not
survive a zip download. One Python file has neither problem.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"
MIN_PYTHON = (3, 11)

# Imported to decide whether the environment is usable. Deliberately the heavy
# ones - if these import, everything lighter does too.
PROBE_IMPORTS = ("pandas", "numpy", "scipy", "statsmodels", "streamlit", "yaml")

# Windows caps paths at 260 characters unless long paths are enabled, and
# installing Streamlit unpacks deeply nested example files. A long repo path
# fails during install with a confusing "No such file or directory".
WINDOWS_PATH_BUDGET = 120


def venv_python(root: Path = VENV) -> Path:
    """Interpreter inside the virtual environment, per platform layout."""
    if os.name == "nt":
        return root / "Scripts" / "python.exe"
    return root / "bin" / "python"


def is_usable(python: Path) -> bool:
    """True when the environment exists and its dependencies import.

    A virtual environment stores absolute paths internally, so moving the
    folder breaks it. Rather than guessing, we try the imports: that catches
    a moved folder, a partial install and a half-upgraded environment alike.
    """
    if not python.exists():
        return False
    probe = "import " + ", ".join(PROBE_IMPORTS)
    result = subprocess.run(
        [str(python), "-c", probe],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def warn_about_long_paths() -> None:
    if os.name != "nt" or len(str(ROOT)) <= WINDOWS_PATH_BUDGET:
        return
    print(
        f"\n[warning] The path to this folder is long:\n  {ROOT}\n"
        "  Windows limits paths to 260 characters and installing dependencies\n"
        "  creates deeply nested files. If setup fails, move this folder closer\n"
        "  to the drive root, for example C:\\projects\\btc.\n"
    )


def create_environment() -> Path:
    """Create the virtual environment and install dependencies."""
    if sys.version_info < MIN_PYTHON:
        sys.exit(
            f"[error] Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer is required, "
            f"found {sys.version.split()[0]}.\n"
            "        Install it from https://www.python.org/downloads/ and run this again."
        )

    print("\n[setup] Preparing the environment. This takes a few minutes, once.\n")
    warn_about_long_paths()

    if VENV.exists():
        print("[setup] The existing environment is broken (moved folder?). Rebuilding.")
        shutil.rmtree(VENV, ignore_errors=True)

    venv.EnvBuilder(with_pip=True, clear=True).create(VENV)
    python = venv_python()

    subprocess.run([str(python), "-m", "pip", "install", "--quiet", "--upgrade", "pip"],
                   check=False)
    result = subprocess.run(
        [str(python), "-m", "pip", "install", "--quiet", "-r", str(REQUIREMENTS)],
        check=False,
    )
    if result.returncode != 0:
        print(
            "\n[error] Installing dependencies failed.\n"
            "        On Windows a long folder path is the usual cause - see the warning above.\n"
            "        On Linux you may be missing build tools: apt install python3-dev build-essential"
        )
        sys.exit(result.returncode)

    seed_env_file()
    print("[setup] Done.\n")
    return python


def seed_env_file() -> None:
    """Create .env from the template so the FRED key has an obvious home."""
    target, template = ROOT / ".env", ROOT / ".env.example"
    if target.exists() or not template.exists():
        return
    shutil.copyfile(template, target)
    print("[setup] Created .env from the template. Paste a FRED key there for M2 data.")


def ensure_environment() -> Path:
    python = venv_python()
    if is_usable(python):
        return python
    return create_environment()


def run(python: Path, arguments: list[str]) -> int:
    return subprocess.run([str(python), *arguments], cwd=str(ROOT), check=False).returncode


def doctor(python: Path) -> int:
    """Print what the environment looks like - first thing to ask for in a bug report."""
    import platform

    print(f"platform          : {platform.platform()}")
    print(f"launcher python   : {sys.version.split()[0]} ({sys.executable})")
    print(f"project root      : {ROOT}")
    print(f"path length       : {len(str(ROOT))} characters"
          + (" (long, see warning)" if len(str(ROOT)) > WINDOWS_PATH_BUDGET else ""))
    print(f"virtual env       : {VENV} ({'present' if VENV.exists() else 'missing'})")
    print(f"env interpreter   : {venv_python()}")
    print(f"dependencies ok   : {is_usable(venv_python())}")
    print(f".env present      : {(ROOT / '.env').exists()}")
    print(f"database present  : {(ROOT / 'data' / 'processed' / 'lab.sqlite').exists()}")
    # Flush before handing stdout to a child process, or its output lands first.
    sys.stdout.flush()
    return run(python, ["-c", "import pandas, numpy, scipy, statsmodels, streamlit;"
                              "print('pandas', pandas.__version__);"
                              "print('numpy', numpy.__version__);"
                              "print('scipy', scipy.__version__);"
                              "print('statsmodels', statsmodels.__version__);"
                              "print('streamlit', streamlit.__version__)"])


def main(argv: list[str]) -> int:
    if argv and argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0

    python = ensure_environment()

    if not argv:
        return run(python, [str(ROOT / "src" / "cli.py"), "--help"])

    command, rest = argv[0], argv[1:]

    if command == "dashboard":
        port = rest[0] if rest else "8511"
        print(f"Dashboard starting on http://localhost:{port}")
        print("Press Ctrl+C in this window to stop.\n")
        return run(python, [
            "-m", "streamlit", "run", str(ROOT / "dashboard" / "app.py"),
            "--server.port", port, "--browser.gatherUsageStats", "false",
        ])

    if command == "test":
        if rest and rest[0] == "offline":
            return run(python, ["-m", "pytest", "-q", "-m", "not network", *rest[1:]])
        return run(python, ["-m", "pytest", "-q", *rest])

    if command == "doctor":
        return doctor(python)

    return run(python, [str(ROOT / "src" / "cli.py"), command, *rest])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
