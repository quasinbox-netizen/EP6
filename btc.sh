#!/usr/bin/env sh
# ============================================================================
#  btc-cycle-lab - macOS and Linux convenience wrapper around run.py
#
#      ./btc ingest --what all
#      ./btc all
#      ./btc dashboard
#      ./btc test offline
#      ./btc doctor
#
#  All the real work (creating the virtual environment, installing
#  dependencies, dispatching) lives in run.py so that Windows, macOS and
#  Linux share one implementation. This file only finds a Python to start it.
#
#  If the execute bit did not survive the download, either run
#  `chmod +x btc` once, or skip this wrapper and use `python3 run.py` directly.
# ============================================================================
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

find_python() {
    for candidate in python3.13 python3.12 python3.11 python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then
            if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; then
                printf '%s' "$candidate"
                return 0
            fi
        fi
    done
    return 1
}

if ! PYTHON=$(find_python); then
    cat >&2 <<'EOF'

[error] No Python 3.11 or newer found.

        macOS  : brew install python@3.13
                 (or download from https://www.python.org/downloads/macos/)
        Ubuntu : sudo apt install python3 python3-venv python3-pip
        Fedora : sudo dnf install python3 python3-virtualenv

        Note for Debian/Ubuntu: python3-venv is a separate package and this
        tool needs it to create the environment.
EOF
    exit 1
fi

exec "$PYTHON" "$ROOT/run.py" "$@"
