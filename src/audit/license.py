"""Offline licence keys.

The product runs entirely on the customer's machine and never calls home, so
licensing has to work with no server and no network. That rules out the usual
shape (call an API, get a yes) and it rules out anything symmetric: an HMAC
scheme would need the signing secret inside the copy handed to the customer,
where it forges keys for everybody.

So the key IS the licence. The vendor signs a small payload with an Ed25519
private key that never leaves the vendor's machine; the shipped code carries
only the matching public key and verifies. Forging a key requires breaking
Ed25519, and the worst an attacker with the source can do is delete the check
- which is true of every offline licence ever written and is a support problem,
not a security one.

WHAT IS AND IS NOT ENFORCED
---------------------------
Enforced: signature, expiry, product name, and that the payload was not
edited. Not enforced: how many machines run it, or revocation. Machine
binding needs a stable hardware id, which is hostile to customers who replace
laptops, and revocation needs the network this design deliberately does not
use. A perpetual-fallback expiry model (see `expires`) fits a sold tool better
than either.

FAILURE IS NOT A CRASH
----------------------
Every failure path returns an unlicensed `Licence`, never an exception and
never a traceback. A paying customer whose clock is wrong, whose file is
corrupt or whose Python lacks `cryptography` gets a plain sentence and an
evaluation-mode run, because a tool that refuses to start is worth less than
a tool that says what is wrong.
"""
from __future__ import annotations

import base64
import binascii
import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

PRODUCT = "strategy-reality-check"
KEY_ENV = "SRC_LICENCE_KEY"
KEY_FILENAME = "licence.key"

# The vendor's Ed25519 public key, base64 (raw 32 bytes), baked into a release.
# Empty here on purpose and permanently: this repository is public, and a build
# published with a key in it is a build anyone can read the key out of. It is
# filled in by the release process, or left empty and supplied at runtime by
# VENDOR_KEY_FILENAME below.
PUBLIC_KEY_B64 = ""

# The runtime alternative to editing the line above. `audit.keytool install`
# writes this file; `vendor_key` reads it. Two reasons it exists:
#
#   * editing a source file before every release is a step someone eventually
#     forgets, and the failure is silent - the build ships, and every paying
#     customer lands in evaluation mode;
#   * a vendor key in version control is a vendor key in every fork.
#
# It is gitignored. A build that carries neither runs in evaluation mode, which
# is the right default for a repository anyone can clone.
VENDOR_KEY_FILENAME = "vendor.pub"
VENDOR_KEY_ENV = "SRC_VENDOR_KEY"

# What evaluation mode changes: the number of permutation draws, and nothing
# else.
#
# It used to trim the record to its most recent 400 rows as well, and that was
# wrong in a way that took a real run to see. Truncation does not lower the
# PRECISION of an answer, it asks a DIFFERENT QUESTION - of whatever window the
# last 400 rows happen to be. On this project's own demo file the trimmed
# window showed +135% at Sharpe 1.70 and the full record showed +50% at Sharpe
# 0.46 with a 65% drawdown; the verdicts differed too, 4 failures against 6.
# A prospect and a customer would have been shown different findings about the
# same strategy, and the prospect's could have been the flattering one.
#
# Draws are the honest lever: they set how finely a p-value can be read, they
# cannot change which side of a threshold the truth falls on, and a report that
# ran 200 of them says so on its face.
EVAL_PERMUTATIONS = 200


@dataclass(frozen=True)
class Licence:
    valid: bool
    reason: str
    licensee: str = ""
    product: str = PRODUCT
    issued: str = ""
    expires: str = ""
    seats: int = 0

    @property
    def mode(self) -> str:
        return "licensed" if self.valid else "evaluation"

    def banner(self) -> str:
        if self.valid:
            tail = f", licence runs to {self.expires}" if self.expires else ""
            return f"Licensed to {self.licensee}{tail}."
        return (
            f"EVALUATION MODE - {self.reason} "
            f"The whole record is audited; permutation draws are capped at "
            f"{EVAL_PERMUTATIONS}, so p-values are coarser, and the report "
            "carries an evaluation watermark."
        )


def _unlicensed(reason: str) -> Licence:
    return Licence(valid=False, reason=reason)


def _decode(token: str) -> tuple[dict, bytes, bytes]:
    """Split ``<payload-b64>.<signature-b64>`` into payload, its bytes and sig."""
    token = "".join(token.split())
    if token.count(".") != 1:
        raise ValueError("a licence key has the form <payload>.<signature>")
    payload_b64, signature_b64 = token.split(".")
    payload_bytes = base64.urlsafe_b64decode(_pad(payload_b64))
    signature = base64.urlsafe_b64decode(_pad(signature_b64))
    payload = json.loads(payload_bytes.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("licence payload is not an object")
    return payload, payload_bytes, signature


def _pad(value: str) -> str:
    return value + "=" * (-len(value) % 4)


def vendor_key(*, root: Path | None = None) -> str:
    """The public key this build verifies against: baked in, env, or file.

    Order matters and is the reverse of `read_key`'s. A key compiled into the
    release wins, because that is the deliberate act of shipping; the file and
    the environment are the development and self-hosting paths, and neither
    should be able to silently replace a released build's key - an attacker who
    can drop a `vendor.pub` next to a *signed* build would otherwise mint their
    own licences for it.
    """
    if PUBLIC_KEY_B64:
        return PUBLIC_KEY_B64
    from_env = os.environ.get(VENDOR_KEY_ENV, "").strip()
    if from_env:
        return from_env
    for candidate in _vendor_key_paths(root):
        try:
            if candidate.is_file() and candidate.stat().st_size < 8192:
                text = candidate.read_text(encoding="utf-8").strip()
                if text:
                    return text
        except OSError:
            continue
    return ""


def _vendor_key_paths(root: Path | None) -> list[Path]:
    paths = []
    if root is not None:
        paths.append(Path(root) / VENDOR_KEY_FILENAME)
    paths.append(Path(__file__).resolve().parents[2] / VENDOR_KEY_FILENAME)
    paths.append(Path.cwd() / VENDOR_KEY_FILENAME)
    # Deduplicate while keeping order: the same path reached two ways should
    # not be read twice, and `dict.fromkeys` is the cheapest stable way to say so.
    return list(dict.fromkeys(paths))


def read_key(explicit: str | None = None, *, root: Path | None = None) -> str:
    """Find the licence key: argument, then environment, then licence.key.

    The file is looked for beside the project and in the user's config
    directory, so a customer can drop it next to the app without knowing what
    a home directory is.
    """
    if explicit:
        return explicit.strip()
    from_env = os.environ.get(KEY_ENV, "").strip()
    if from_env:
        return from_env
    candidates = []
    if root is not None:
        candidates.append(Path(root) / KEY_FILENAME)
    candidates.append(Path.cwd() / KEY_FILENAME)
    home = Path.home()
    candidates.append(home / ".config" / PRODUCT / KEY_FILENAME)
    candidates.append(home / f".{PRODUCT}.key")
    for candidate in candidates:
        try:
            if candidate.is_file() and candidate.stat().st_size < 8192:
                return candidate.read_text(encoding="utf-8").strip()
        except OSError:
            continue
    return ""


def verify(
    token: str | None = None,
    *,
    public_key_b64: str | None = None,
    root: Path | None = None,
    today: date | None = None,
) -> Licence:
    """Check a licence key. Never raises; an invalid key is a result, not an error."""
    key_b64 = vendor_key(root=root) if public_key_b64 is None else public_key_b64
    if not key_b64:
        return _unlicensed(
            "this build carries no vendor key, so nothing can be licensed."
        )

    token = token if token is not None else read_key(root=root)
    if not token:
        return _unlicensed(
            f"no licence key found (set {KEY_ENV} or put {KEY_FILENAME} beside the app)."
        )

    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError:
        return _unlicensed(
            "the `cryptography` package is not installed, so the licence cannot be "
            "checked. Run the installer again or `pip install cryptography`."
        )

    try:
        payload, payload_bytes, signature = _decode(token)
    except (ValueError, binascii.Error, json.JSONDecodeError, UnicodeDecodeError):
        return _unlicensed("the licence key is malformed.")

    try:
        public_key = Ed25519PublicKey.from_public_bytes(base64.b64decode(_pad(key_b64)))
    except (ValueError, binascii.Error):
        return _unlicensed("this build's vendor key is unreadable.")

    try:
        public_key.verify(signature, payload_bytes)
    except InvalidSignature:
        return _unlicensed("the licence key's signature does not match.")
    except Exception:  # pragma: no cover - defensive: never crash on a bad key
        return _unlicensed("the licence key could not be verified.")

    if str(payload.get("product", "")) != PRODUCT:
        return _unlicensed(
            f"this key is for {payload.get('product', 'another product')!r}, not {PRODUCT!r}."
        )

    expires = str(payload.get("expires", ""))
    if expires:
        try:
            deadline = date.fromisoformat(expires)
        except ValueError:
            return _unlicensed("the licence key has an unreadable expiry date.")
        now = today or datetime.now(timezone.utc).date()
        if now > deadline:
            return _unlicensed(f"the licence expired on {expires}.")

    return Licence(
        valid=True,
        reason="",
        licensee=str(payload.get("licensee", "unnamed")),
        product=PRODUCT,
        issued=str(payload.get("issued", "")),
        expires=expires,
        seats=int(payload.get("seats", 1) or 1),
    )
