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
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

SIGNING_KEY_ENV = "SRC_SIGNING_KEY"

# This module has to import the same way whether it is run as
# `python -m audit.keytool` (a package), as `python src/audit/keytool.py` (a
# script with no package at all) or through `run.py keytool`. The shim below
# binds the licence module ONCE, as `lic`, and everything downstream uses that
# name - a relative `from . import license` inside a function looks harmless
# and raises "attempted relative import with no known parent package" the first
# time someone runs the file directly, which is the first thing a vendor does.
if __package__ in (None, ""):  # `python src/audit/keytool.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from audit import license as lic  # type: ignore
else:
    from . import license as lic

_pad = lic._pad
PRODUCT = lic.PRODUCT
verify = lic.verify
vendor_key = lic.vendor_key


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
    public_b64 = _b64(public_raw)

    if args.out:
        target = Path(args.out).expanduser()
        if target.exists():
            raise SystemExit(
                f"refusing to overwrite an existing key at {target}.\n"
                "If that file is the signing key for a product you have already "
                "sold licences for, overwriting it invalidates every one of them."
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        # Create with the right mode from the start. Writing first and
        # chmod-ing after leaves the key world-readable for the moment in
        # between, which on a shared machine is a moment too long.
        _write_private(target, _b64(private_raw))
        print(f"private key written to {target}")
        print(_permissions_note(target))
        # Deliberately NOT printed. A production signing key on stdout is a
        # production signing key in scrollback, in the screen recording of the
        # call where it was set up, and in whatever ships terminal logs.
        print("The private key itself was not displayed. It is only in that file.")
    else:
        print("PRIVATE KEY - this is now in your terminal scrollback:")
        print(f"  {_b64(private_raw)}")
        print()
        print("Pass --out <path> instead to write it straight to a file and keep")
        print("it off the screen. Treat a key printed here as compromised if the")
        print("terminal was shared, recorded, or is logged.")

    print()
    print("PUBLIC KEY (safe to disclose):")
    print(f"  {public_b64}")
    print()
    print("Install it in this checkout with:")
    print(f'  python run.py keytool install --public-key "{public_b64}"')
    return 0


def _write_private(target: Path, material: str) -> None:
    """Write a secret with owner-only permissions where the OS supports them."""
    try:
        handle = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except (AttributeError, OSError):
        target.write_text(material, encoding="utf-8")
        return
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        stream.write(material)


def _permissions_note(target: Path) -> str:
    """Say what the file's permissions actually are, never what they ought to be.

    The earlier version printed "(mode 600)" unconditionally. On Windows the
    chmod is close to a no-op, so it announced a protection it had not applied
    - which is worse than saying nothing, because the reader stops checking.
    """
    if os.name == "nt":
        return (
            "NOTE: Windows does not apply POSIX permissions. Restrict the file "
            "to your account with:\n"
            f'  icacls "{target}" /inheritance:r /grant:r "%USERNAME%:F"'
        )
    try:
        mode = target.stat().st_mode & 0o777
    except OSError:
        return "Could not read the file's permissions back; check them yourself."
    if mode == 0o600:
        return "Permissions: 600 (owner read/write only)."
    return (
        f"WARNING: permissions are {mode:03o}, not 600. Fix with:\n"
        f"  chmod 600 {target}"
    )


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
    public = args.public_key or vendor_key(root=_repo_root())
    result = verify(args.key, public_key_b64=public)
    print(result.banner())
    if result.valid:
        print(f"  licensee {result.licensee} | issued {result.issued} | "
              f"expires {result.expires} | seats {result.seats}")
    return 0 if result.valid else 1


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def cmd_install(args) -> int:
    """Write the vendor public key so this checkout verifies licences.

    Deliberately refuses to overwrite without --force. Replacing a vendor key
    invalidates every licence already issued under the old one, which is a
    thing to do on purpose and never as a side effect of re-running a setup
    command.
    """
    key = args.public_key.strip()
    try:
        raw = base64.b64decode(_pad(key))
    except Exception as exc:
        raise SystemExit("the public key is not valid base64.") from exc
    if len(raw) != 32:
        raise SystemExit(f"an Ed25519 public key is 32 bytes; got {len(raw)}.")

    target = Path(args.out) if args.out else _repo_root() / lic.VENDOR_KEY_FILENAME
    if target.exists() and not args.force:
        current = target.read_text(encoding="utf-8").strip()
        if current == key:
            print(f"{target} already holds this key; nothing to do.")
            return 0
        raise SystemExit(
            f"{target} already holds a different vendor key.\n"
            "Replacing it invalidates every licence issued under the old one. "
            "Pass --force if that is what you mean."
        )
    target.write_text(key + "\n", encoding="utf-8")
    print(f"vendor key installed at {target}")
    print("This checkout now verifies licences signed by the matching private key.")
    return 0


def cmd_status(args) -> int:
    """Say which key this build would verify against, and where it came from."""
    root = _repo_root()
    print(f"product           : {PRODUCT}")
    if lic.PUBLIC_KEY_B64:
        source = "compiled into this build (src/audit/license.py)"
    elif os.environ.get(lic.VENDOR_KEY_ENV, "").strip():
        source = f"environment ({lic.VENDOR_KEY_ENV})"
    else:
        found = [p for p in lic._vendor_key_paths(root) if p.is_file()]
        source = str(found[0]) if found else "nowhere - evaluation mode"
    key = vendor_key(root=root)
    print(f"vendor key        : {key or '(none)'}")
    print(f"      from        : {source}")
    print(f"licence key       : {'found' if lic.read_key(root=root) else 'not found'}")
    print(f"mode              : {verify(root=root).mode}")
    print(f"evaluation limit  : {lic.EVAL_PERMUTATIONS} permutation draws "
          "(the whole record is audited either way)")
    return 0


def cmd_selftest(args) -> int:
    """Prove the whole chain without persisting a key anywhere.

    Generates a throwaway pair in memory, signs a licence, verifies it, then
    checks that an edited payload and an expired date are both rejected. It
    touches no file, so it is safe to run on a machine that must never hold a
    signing key - which is exactly the machine where someone doubts the setup.
    """
    serialization, Ed25519PrivateKey = _require_cryptography()
    private = Ed25519PrivateKey.generate()
    public = _b64(private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ))

    def sign(**overrides) -> str:
        payload = {
            "product": PRODUCT, "licensee": "selftest",
            "issued": date.today().isoformat(),
            "expires": (date.today() + timedelta(days=30)).isoformat(),
            "seats": 1, "ref": "SELFTEST",
        }
        payload.update(overrides)
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return f"{_b64url(raw)}.{_b64url(private.sign(raw))}"

    checks = []
    good = sign()
    checks.append(("a signed licence verifies", verify(good, public_key_b64=public).valid))
    checks.append((
        "an expired licence is refused",
        not verify(sign(expires=(date.today() - timedelta(days=1)).isoformat()),
                   public_key_b64=public).valid,
    ))
    checks.append((
        "another product's licence is refused",
        not verify(sign(product="something-else"), public_key_b64=public).valid,
    ))
    payload_b64, signature_b64 = good.split(".")
    tampered = json.loads(base64.urlsafe_b64decode(_pad(payload_b64)))
    tampered["expires"] = "2099-01-01"
    forged = _b64url(json.dumps(tampered, sort_keys=True, separators=(",", ":")).encode())
    checks.append((
        "an edited payload breaks the signature",
        not verify(f"{forged}.{signature_b64}", public_key_b64=public).valid,
    ))
    checks.append(("garbage is refused without raising",
                   not verify("not-a-key", public_key_b64=public).valid))

    for label, ok in checks:
        print(f"  [{'+' if ok else 'x'}] {label}")
    failed = [label for label, ok in checks if not ok]
    if failed:
        print(f"\nSELFTEST FAILED: {len(failed)} check(s). Do not ship this build.")
        return 1
    print("\nselftest passed; no key was written to disk.")
    return 0


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

    install = sub.add_parser("install", help="install the vendor PUBLIC key in this checkout")
    install.add_argument("--public-key", required=True)
    install.add_argument("--out", default=None, help="write somewhere other than the repo root")
    install.add_argument("--force", action="store_true",
                         help="replace an existing key, invalidating licences issued under it")
    install.set_defaults(func=cmd_install)

    status = sub.add_parser("status", help="which key this build verifies against, and from where")
    status.set_defaults(func=cmd_status)

    selftest = sub.add_parser(
        "selftest", help="prove signing and verification work, writing nothing to disk"
    )
    selftest.set_defaults(func=cmd_selftest)

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
