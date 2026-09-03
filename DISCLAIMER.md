# Disclaimer

**This software is a research tool. It is not investment advice.**

Read this before you use any number this tool produces to decide what to do
with money.

## Not financial advice

Nothing produced by this software is a recommendation to buy, sell or hold any
asset. The authors are not investment advisers, brokers, or analysts, and are
not registered with any financial regulator. No output of this tool takes your
circumstances, risk tolerance, tax position or jurisdiction into account,
because it knows nothing about them.

If you want advice, talk to a licensed adviser in your country.

## What the tool actually does

It measures whether historical patterns in the Bitcoin price are
distinguishable from random noise. That is a different question from "what
will happen next", and the tool makes no attempt to answer the second one.

Its own headline finding, on data from 2011 to 2026, is **negative**: the
apparent effect of Bitcoin halvings does not survive correction for multiple
testing, does not replicate out of sample, and cannot be distinguished from
what the NASDAQ did over the same windows. A tool whose main result is "we
found no evidence" is not a tool for picking trades.

## Backtests are not forecasts

The backtesting engine deliberately includes transaction costs, slippage and
execution lag, because omitting them makes every strategy look better than it
is. Even so, a backtest remains a description of the past under simplifying
assumptions. It does not model:

* taxes, in any jurisdiction,
* funding or borrowing costs,
* liquidity limits — it assumes your order does not move the price,
* exchange outages, withdrawal freezes, or counterparty failure,
* your own behaviour during an 85% drawdown, which the historical record
  contains twice.

The slippage figure in the configuration is an **assumption, not a
measurement**. Change it and the ranking of strategies changes.

## Data may be wrong

Prices come from public exchange APIs and may contain gaps, outliers or
divergences between venues; the tool reports these rather than hiding them,
but reporting is not fixing. Macroeconomic series get revised after
publication. The list of historical events in `data/raw/events.csv` is
maintained by hand, was assembled after the fact, and is therefore subject to
selection bias — a point the tool states in its own output.

Nobody warrants that any of this data is accurate, complete or current.

## Past performance

Past performance does not indicate future results. Bitcoin has repeatedly lost
more than 70% of its value from a peak. You can lose everything you put into
it.

## No warranty

This software is provided "as is", without warranty of any kind. See `LICENSE`
for the full terms. The authors are not liable for any loss arising from use of
this software or reliance on its output.

## Your responsibility

You are responsible for complying with the laws and regulations that apply to
you, including those governing financial instruments, market data licensing and
taxation. You are also responsible for respecting the terms of service of the
data providers this tool queries — see `DATA_SOURCES.md`.
