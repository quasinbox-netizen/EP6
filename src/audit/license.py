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

# The vendor's Ed25519 public key, base64 (raw 32 bytes). Replace this with
# the value printed by `python -m audit.keytool generate` before shipping.
# Empty means unsigned builds: everything runs in evaluation mode, which is
# the correct default for a repository that is public.
PUBLIC_KEY_B64 = ""

# What evaluation mode allows. Generous on purpose: the checks that cost
# nothing to run are the ones that earn the sale, and a crippled demo of a
# statistics tool cannot demonstrate statistics.
EVAL_MAX_ROWS = 400
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
            f"Reports are limited to {EVAL_MAX_ROWS} rows and "
            f"{EVAL_PERMUTATIONS} permutations, and carry an evaluation watermark."
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
    key_b64 = PUBLIC_KEY_B64 if public_key_b64 is None else public_key_b64
    if not key_b64:
        return _unlicensed("this build carries no vendor key, so nothing can be licensed.")

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
