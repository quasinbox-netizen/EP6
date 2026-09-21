#!/usr/bin/env bash
# The lab's working state, carried between cloud runs - encrypted.
#
# The jobs that keep the site current need files git does not track: the
# 11 MB database, the agent's journal, the ledger of recorded claims, the
# paper portfolio. The repository is public (GitHub Pages needs it to be), and
# the database holds the Yahoo Finance control series, whose terms forbid
# redistribution. So the state never travels in the clear: it is packed,
# encrypted with STATE_KEY, and only the ciphertext goes into the Actions
# cache or onto the `lab-state` branch.
#
#   pack    data/processed (untracked files only) -> .state/state.enc
#   unpack  .state/state.enc, or the lab-state branch if there is none -> data/processed
#   seed    .state/state.enc -> force-push as the single commit of lab-state
#
# Run from the repository root. STATE_KEY comes from the environment, or from
# .env when run by hand on the machine that holds the original data.
set -euo pipefail

if [ -z "${STATE_KEY:-}" ] && [ -f .env ]; then
  STATE_KEY="$(grep -E '^STATE_KEY=' .env | head -n1 | cut -d= -f2- || true)"
  export STATE_KEY
fi
if [ -z "${STATE_KEY:-}" ]; then
  echo "STATE_KEY is not set" >&2
  exit 1
fi

mkdir -p .state
ENC=.state/state.enc
CIPHER=(-aes-256-cbc -pbkdf2 -iter 200000 -md sha256)

case "${1:-}" in
  pack)
    # Only what git does not already carry. A tracked file restored from an
    # older archive would silently undo a commit.
    git ls-files --others --ignored --exclude-standard -z data/processed \
      | tar --null -czf .state/state.tar.gz -T -
    openssl enc "${CIPHER[@]}" -salt -pass env:STATE_KEY \
      -in .state/state.tar.gz -out "$ENC"
    rm -f .state/state.tar.gz
    echo "packed $(du -h "$ENC" | cut -f1)"
    ;;
  unpack)
    if [ ! -s "$ENC" ]; then
      echo "no cached state; taking the seed from the lab-state branch"
      git fetch --quiet origin lab-state
      git show FETCH_HEAD:state.enc > "$ENC"
    fi
    openssl enc -d "${CIPHER[@]}" -pass env:STATE_KEY \
      -in "$ENC" -out .state/state.tar.gz
    tar -xzf .state/state.tar.gz
    rm -f .state/state.tar.gz
    if [ ! -s data/processed/lab.sqlite ]; then
      echo "the unpacked state has no database" >&2
      exit 1
    fi
    echo "unpacked: database $(du -h data/processed/lab.sqlite | cut -f1)"
    ;;
  seed)
    # One commit, no parents, force-pushed: the branch is a locker, not a
    # history, and yesterday's ciphertext is of no use to anyone.
    blob="$(git hash-object -w "$ENC")"
    tree="$(printf '100644 blob %s\tstate.enc\n' "$blob" | git mktree)"
    commit="$(git -c user.name="${GIT_AUTHOR_NAME:-lab}" -c user.email="${GIT_AUTHOR_EMAIL:-lab@localhost}" \
      commit-tree "$tree" -m "Encrypted lab state $(date -u +%Y-%m-%d)")"
    git push --quiet --force origin "$commit:refs/heads/lab-state"
    echo "seeded lab-state at ${commit:0:7}"
    ;;
  *)
    echo "usage: $0 pack|unpack|seed" >&2
    exit 2
    ;;
esac
