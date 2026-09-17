"""Vendor-side key management. Not shipped to customers.

    python -m audit.keytool generate
    python -m audit.keytool issue --licensee "ACME sp. z o.o." --months 12
    python -m audit.keytool check  --key "<token>"

`generate` prints a private key and a public key. The public key goes into
`license.PUBLIC_KEY_B64` in the copy you ship. The private key goes somewhere
that is not this repository and is not a cloud drive: losing it means every
future licence has to be re-issued under a new public key, and leaking it
means anyone can mint licences for your product forever.

The private key is read from SRC_SIGNING_KEY or a file path, never from the
command line, because shell history is a file too.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

SIGNING_KEY_ENV = "SRC_SIGNING_KEY"

if __package__ in (None, ""):  # allow `python src/audit/keytool.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from audit.license import PRODUCT, verify  # type: ignore
else:
    from .license import PRODUCT, verify


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _require_cryptography():
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    except ImportError as exc:
        raise SystemExit(
            "the `cryptography` package is required to issue licences: "
            "pip install cryptography"
        ) from exc
    return serialization, Ed25519PrivateKey


def cmd_generate(args) -> int:
    serialization, Ed25519PrivateKey = _require_cryptography()
    private = Ed25519PrivateKey.generate()
    private_raw = private.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_raw = private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    print("PRIVATE KEY (keep offline, never commit):")
    print(f"  {_b64(private_raw)}")
    print()
    print("PUBLIC KEY - paste into src/audit/license.py as PUBLIC_KEY_B64:")
    print(f'  PUBLIC_KEY_B64 = "{_b64(public_raw)}"')
    if args.out:
        target = Path(args.out)
        if target.exists():
            raise SystemExit(f"refusing to overwrite an existing key at {target}")
        target.write_text(_b64(private_raw), encoding="utf-8")
        with contextlib.suppress(OSError):
            target.chmod(0o600)
        print(f"\nprivate key written to {target} (mode 600)")
    return 0


def _load_private():
    serialization, Ed25519PrivateKey = _require_cryptography()
    material = os.environ.get(SIGNING_KEY_ENV, "").strip()
    if not material:
        path = os.environ.get("SRC_SIGNING_KEY_FILE", "").strip()
        if path and Path(path).is_file():
            material = Path(path).read_text(encoding="utf-8").strip()
    if not material:
        raise SystemExit(
            f"no signing key. Set {SIGNING_KEY_ENV} to the base64 private key, or "
            "SRC_SIGNING_KEY_FILE to a file holding it."
        )
    try:
        raw = base64.b64decode(material)
    except Exception as exc:
        raise SystemExit("the signing key is not valid base64.") from exc
    if len(raw) != 32:
        raise SystemExit(f"an Ed25519 private key is 32 bytes; got {len(raw)}.")
    return Ed25519PrivateKey.from_private_bytes(raw)


def cmd_issue(args) -> int:
    private = _load_private()
    issued = date.today()
    if args.expires:
        try:
            expires = date.fromisoformat(args.expires)
        except ValueError as exc:
            raise SystemExit("--expires needs an ISO date, e.g. 2027-01-31") from exc
    else:
        expires = issued + timedelta(days=int(round(args.months * 30.44)))
    if expires <= issued:
        raise SystemExit("the expiry date is not in the future.")

    payload = {
        "product": PRODUCT,
        "licensee": args.licensee,
        "issued": issued.isoformat(),
        "expires": expires.isoformat(),
        "seats": int(args.seats),
        # A nonce makes two licences for the same customer on the same day
        # distinct, so a leaked key can be traced to the order that produced it.
        "ref": args.ref or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
    }
    payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    signature = private.sign(payload_bytes)
    token = f"{_b64url(payload_bytes)}.{_b64url(signature)}"

    print(f"licence for {payload['licensee']} ({payload['seats']} seat(s)), "
          f"valid to {payload['expires']}, ref {payload['ref']}:")
    print()
    print(token)
    if args.out:
        Path(args.out).write_text(token + "\n", encoding="utf-8")
        print(f"\nwritten to {args.out}")
    return 0


def cmd_check(args) -> int:
    from . import license as lic  # noqa: PLC0415 - re-read so a patched key is seen

    public = args.public_key or lic.PUBLIC_KEY_B64
    result = verify(args.key, public_key_b64=public)
    print(result.banner())
    if result.valid:
        print(f"  licensee {result.licensee} | issued {result.issued} | "
              f"expires {result.expires} | seats {result.seats}")
    return 0 if result.valid else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audit.keytool", description="vendor-side licence key management"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="create a new vendor keypair")
    generate.add_argument("--out", default=None, help="write the private key here")
    generate.set_defaults(func=cmd_generate)

    issue = sub.add_parser("issue", help="sign a licence for a customer")
    issue.add_argument("--licensee", required=True, help="customer name as it appears in the app")
    issue.add_argument("--months", type=float, default=12.0)
    issue.add_argument("--expires", default=None, help="explicit ISO expiry, overrides --months")
    issue.add_argument("--seats", type=int, default=1)
    issue.add_argument("--ref", default=None, help="order reference, for tracing a leak")
    issue.add_argument("--out", default=None, help="write the key to this file")
    issue.set_defaults(func=cmd_issue)

    check = sub.add_parser("check", help="verify a key as the customer's copy would")
    check.add_argument("--key", required=True)
    check.add_argument("--public-key", default=None)
    check.set_defaults(func=cmd_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
