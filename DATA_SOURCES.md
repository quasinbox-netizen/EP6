# Data Sources, Attribution and Restrictions

This tool does not ship market data. It fetches it, at your request, from the
providers below. Those are **your** agreements with those providers, not ours.
Read this section before publishing anything derived from the data.

## Price data

| Source | Instrument | Coverage | Key | Endpoint |
|---|---|---|---|---|
| Binance | BTC/USDT daily | from 2017-08-17 | no | `api.binance.com/api/v3/klines` |
| Bitstamp | BTC/USD daily | from 2011-08-18 | no | `www.bitstamp.net/api/v2/ohlc` |
| Coinbase | BTC/USD daily | from 2015-07-20 | no | `api.exchange.coinbase.com/products` |
| Yahoo Finance | indices, gold | from 2009 | no | `query1.finance.yahoo.com/v8/finance/chart` |

Binance quotes BTC against USDT rather than USD. Over the common history the
median difference against Bitstamp's USD price is 0.06%, but during USDT
de-pegging episodes it is larger. The stitching report prints this instead of
hiding it.

## Macroeconomic data

Series from the **Federal Reserve Bank of St. Louis (FRED)** require a free API
key, which you supply yourself in `.env`. FRED asks that you cite the source
when you publish anything based on its data:

> Data source: Federal Reserve Bank of St. Louis, FRED
> (https://fred.stlouisfed.org/), retrieved via the FRED® API.

FRED® is a registered trademark of the Federal Reserve Bank of St. Louis. This
project is not affiliated with, endorsed by, or reviewed by the Federal Reserve.
Series used: `M2SL`, `INDPRO`, `DFF`, `UNRATE`.

Terms: https://fred.stlouisfed.org/legal/

## Restrictions you should know about

**Yahoo Finance prohibits redistribution.** Its terms permit personal,
non-commercial use and do not allow you to republish or resell the data. This
matters in two places:

1. Do not commit raw Yahoo series to a public repository. This project does
   not: `data/processed/lab.sqlite` is excluded from version control, and the
   files that *are* committed contain aggregated statistics (event-study
   averages, confidence intervals, backtest metrics), not the underlying
   quotes.
2. If you host the dashboard publicly, you are serving that data onward. Check
   the terms first, or replace the control-group source with one whose licence
   permits it.

**Exchange APIs are rate-limited and have their own terms.** The tool paces its
requests and identifies itself with a `User-Agent`, but heavy or automated use
is your responsibility. Binance and other exchanges restrict access from some
jurisdictions.

**ISM Manufacturing PMI is not included.** Its licence does not allow
redistribution through FRED, so the tool cannot fetch it. If you have your own
licensed copy, drop it into `data/raw/manual/pmi.csv` as
`observation_date,value[,available_from]` and the tool will pick it up. Do not
commit licensed data to a public repository.

## Point-in-time handling

Every row carries two dates: the observation date and `available_from`, the day
the value was publicly known. FRED data is fetched as **first releases**
(`output_type=4`) rather than the current revised values, because the revised
figure for March 2020 M2 was not knowable in March 2020. Measured on the real
vintage history of `M2SL`, the median publication lag is 43 days.

This is a methodological choice, not a legal one, but it affects what the
numbers mean: results computed from revised data are not comparable to results
computed from first releases.

## Software dependencies

Installed from PyPI at setup time: pandas, numpy, scipy, statsmodels, pyyaml,
requests, python-dotenv, streamlit, plotly, pytest. Each carries its own
licence — permissive (BSD/MIT/Apache) in every case at the versions pinned in
`requirements.txt`. Run `pip licenses` in the virtual environment if you need
the exact list for a compliance review.

## No affiliation

This project is not affiliated with, sponsored by or endorsed by Binance,
Bitstamp, Coinbase, Yahoo, the Federal Reserve, ISM, or any exchange or index
provider. All trademarks belong to their owners.
