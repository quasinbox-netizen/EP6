# Security Policy

## Reporting a vulnerability

Do not open a public issue for a security problem. Use GitHub's private
vulnerability reporting on this repository ("Security" tab -> "Report a
vulnerability"), which reaches the maintainer without disclosing the issue.

Please include what you did, what happened, and what you expected. A patch is
welcome but not required.

There is no bug bounty for this project.

## Scope

This is a local research tool. It has no server, no authentication and no
multi-user surface, so the interesting attack surface is narrow:

**In scope**

* Leaking the user's API key — into logs, error messages, committed files or
  outbound requests other than the provider it belongs to.
* Code execution triggered by data fetched from a provider, or by a crafted
  `events.csv` / `pmi.csv`.
* Path traversal or writes outside the project folder.
* Dependency vulnerabilities that are actually reachable from this code.

**Out of scope**

* Anything requiring an attacker who already controls the machine.
* Denial of service against third-party APIs, which is the provider's concern.
* Exposing the dashboard to a network yourself. Streamlit binds to localhost by
  default; if you change that, securing it is yours.
* Statistical or methodological disagreements. Those are issues, not
  vulnerabilities, and are welcome as issues.

## What the project already does

* `.env` is excluded from version control, so a `git push` cannot publish a
  key.
* Error messages pass through `ingest.http.redact`, which replaces the value of
  any `api_key`, `apikey`, `token`, `access_key` or `secret` URL parameter with
  `***`. This exists because the FRED key travels in the query string and a raw
  URL in a traceback would leak it. It has tests.
* All outbound requests are HTTPS.
* The tool sends the FRED key only to `api.stlouisfed.org`.
* No dependency is installed system-wide; everything lives in `.venv` inside
  the project folder.

## Supported versions

The `main` branch. This project has no release branches and no backports.
