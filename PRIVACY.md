# Privacy Policy

Last updated: 2026-09-01

This is a short policy because the software does very little that touches
privacy. It is honest rather than long: it describes what actually happens, not
what a template says usually happens.

## Summary

**This tool collects nothing about you.** There is no account, no telemetry, no
analytics, no cookies, no advertising, and no server operated by us. It runs
entirely on your own computer and stores its data in its own folder.

The one privacy-relevant fact worth knowing: when you download data, your
computer contacts third-party APIs directly, and those providers see your IP
address.

## What runs where

The software is a local command-line tool plus a local dashboard. The dashboard
is served by Streamlit on `localhost` and is reachable only from your own
machine unless you deliberately expose the port. We do not host anything and we
receive no data from your installation — not even a version check.

## What is stored on your machine

Inside the project folder only:

| Location | Contents |
|---|---|
| `data/processed/lab.sqlite` | prices, macro series, events you downloaded |
| `data/processed/*.csv` | analysis results |
| `.env` | your own API key, if you added one |
| `.venv/` | the Python environment |

Nothing is written outside the project folder. To remove everything, delete the
folder.

## Third parties your computer contacts

Running `ingest` makes direct HTTPS requests to the providers below. We are not
a party to those requests — they happen between your machine and the provider,
and each provider will see your IP address and can log the request under its
own privacy policy:

| Provider | What is requested | Their policy |
|---|---|---|
| Binance | daily BTC/USDT candles | https://www.binance.com/en/privacy |
| Bitstamp | daily BTC/USD candles | https://www.bitstamp.net/privacy-policy/ |
| Coinbase | daily BTC/USD candles | https://www.coinbase.com/legal/privacy |
| Yahoo Finance | index and commodity quotes | https://legal.yahoo.com/us/en/yahoo/privacy/index.html |
| FRED (St. Louis Fed) | macroeconomic series | https://fred.stlouisfed.org/legal/ |
| PyPI | Python packages, at install time | https://www.python.org/privacy/ |

No request carries any identifier we generate. Requests to FRED include the API
key you supplied, because FRED requires it.

The dashboard is started with `--browser.gatherUsageStats false`, which
disables Streamlit's own usage telemetry.

## Your API key

If you paste a FRED key into `.env`, it stays in that file on your disk. It is
sent only to FRED, only over HTTPS, and only when fetching macro data. `.env`
is excluded from version control so that a `git push` cannot publish it.

Error messages are passed through a redaction filter that replaces the value of
any `api_key`, `token` or `secret` URL parameter with `***`, so a stack trace
you paste into a bug report does not leak your key. If you ever suspect a key
was exposed, revoke it at your provider and paste a new one.

## Children

The software is not directed at children and collects no data from anyone.

## If you host the dashboard publicly

This policy covers the software as distributed, which runs locally. If you
deploy the dashboard to a public address, **you** become the operator: your
hosting provider will log visitor IP addresses and requests, and you will need
your own privacy policy covering that. Check also that the data provider terms
in `DATA_SOURCES.md` permit what you intend to serve — at least one of them
restricts redistribution.

## Changes

Material changes to this policy will be recorded in the project's git history,
which is the authoritative record of when it changed and to what.

## Contact

Open an issue in the project repository.
