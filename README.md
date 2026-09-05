# btc-cycle-lab

[![tests](https://github.com/quasinbox-netizen/EP6/actions/workflows/tests.yml/badge.svg)](https://github.com/quasinbox-netizen/EP6/actions/workflows/tests.yml)

A local research lab for testing whether the Bitcoin halving cycle and macro
events explain anything in the price of BTC.

The project is built around a single premise: **finding a pattern is easy,
showing it is not chance is hard.** So every result leaves here with a
confidence interval, a count of observations and a correction for the number of
hypotheses tested — and every backtest is compared against buy-and-hold.

**Its own headline finding is negative.** On data from 2011 to 2026 the
apparent halving effect does not survive correction for multiple testing, does
not replicate out of sample, cannot be distinguished from what the NASDAQ did
over the same windows, and survives none of 160 ways of specifying the
question. What remains once a pre-event baseline is subtracted is drift. See
[Results](#results).

> **This is not investment advice.** Read [DISCLAIMER.md](DISCLAIMER.md) before
> using any number from this tool to decide what to do with money.

---

## Quick start

You need Python 3.11 or newer. Everything else happens automatically — the
launcher creates a virtual environment and installs dependencies on first run.
Nothing is installed system-wide.

**Windows**

```bat
.\btc.cmd ingest --what all
```

**macOS and Linux**

```bash
./btc.sh ingest --what all
```

**Any platform, no wrapper:**

```bash
python run.py ingest --what all
```

The wrappers are one-line shims around `run.py`; use whichever is convenient.

**Without a terminal at all (Windows):** double-click **`dashboard.cmd`** in the
folder. It sets everything up on first run and opens the dashboard in a browser.
Double-clicking `btc.cmd` shows the list of commands instead, since a launcher
with no command has nothing to launch.

### Platform support

| platform | Python | verified |
|---|---|---|
| Windows 10/11 | 3.11, 3.13 | every push |
| Linux (Ubuntu) | 3.11, 3.13 | every push |
| macOS | 3.11, 3.13 | every push |

All six combinations run the full offline suite through `run.py`, so each one
exercises the real installation path — interpreter discovery, virtual
environment creation, dependency install — not just the tests. The two POSIX
jobs additionally assert that `./btc.sh` still has its execute bit and runs.

Six jobs on every push. Actions minutes are free on public repositories, so
there is no reason to hold macOS back to a schedule - and one workflow covering
everything cannot drift out of step with a second one testing the same thing
differently.
The examples below use `run.py` because it is identical everywhere.

Then run any of these:

| command | what it does |
|---|---|
| `run.py ingest --what all` | download prices, macro, control group and events |
| `run.py quality` | data quality report and the seam between exchanges |
| `run.py features` | build and save the feature frame |
| `run.py study --post 365` | event study around halvings and event categories |
| `run.py control` | control group: NASDAQ, S&P 500, gold |
| `run.py macro` | liquidity axis: real M2 vs the dollar proxy |
| `run.py validate` | hypothesis scan with correction + out-of-sample replication |
| `run.py walkforward` | walk-forward validation across 13 disjoint windows |
| `run.py speccurve` | specification curve: 160 ways of asking the same question (~10 min) |
| `run.py forecast` | directional forecast, scored against three baselines |
| `run.py range --days 10` | how far the price may move, as a calibrated interval |
| `run.py backtest` | strategies vs buy-and-hold |
| `run.py all` | everything in sequence |
| `run.py dashboard` | browser dashboard on port 8511 |
| `run.py test` | the test suite (`test offline` skips network tests) |
| `run.py doctor` | environment diagnostics — start here in a bug report |

### Notes per platform

**Windows.** On PowerShell the `.\` prefix is required; in `cmd.exe` it is
required when `NoDefaultCurrentDirectoryInExePath` is set, so the examples
always include it. The wrapper is `.cmd` rather than `.ps1` on purpose: the
default PowerShell execution policy blocks unsigned `.ps1` files.

**Keep the folder close to the drive root**, e.g. `C:\projects\btc`. Windows
limits paths to 260 characters and installing Streamlit unpacks deeply nested
example files; with a repo path longer than about 120 characters the install
fails with `No such file or directory`. The launcher warns about this before
it starts.

**Debian and Ubuntu.** `python3-venv` is a separate package and is required:

```bash
sudo apt install python3 python3-venv python3-pip
```

**macOS.** The system Python is usually old enough to matter; `brew install
python@3.13` or the installer from python.org both work.

**If `./btc.sh` is not executable** (the bit does not survive a zip download),
either `chmod +x btc.sh` once or just use `python3 run.py`.

The POSIX wrapper is `btc.sh` rather than plain `btc` for the sake of Windows:
Explorer hides known extensions by default, so an extensionless `btc` and
`btc.cmd` appear in the folder as two identical entries called "btc", with the
unrunnable one sorting first. Anyone double-clicking picks the wrong file and
sees nothing happen.

### Portability

The whole folder can be copied to a USB stick or another machine. A virtual
environment stores absolute paths internally, so it breaks when moved — the
launcher detects this (by trying to import the dependencies) and rebuilds it
from scratch. The database, configuration and results live relative to the
project folder, so they travel with it.

What moving does **not** carry: the `.env` file with your FRED key, which is
outside version control. The launcher recreates it from the template, but the
key has to be pasted again. Without it everything works except the M2 series.

---

## Where the data comes from

| Source | Coverage | API key | Role |
|---|---|---|---|
| Binance (BTC/USDT) | from 2017-08-17 | no | deepest market — source of truth from 2017 |
| Bitstamp (BTC/USD) | from 2011-08-18 | no | the only one covering the 2012 and 2016 halvings |
| Coinbase (BTC/USD) | from 2015-07-20 | no | cross-validation |
| Yahoo Finance | from 2009 | no | DXY, S&P 500, yields, gold |
| Yahoo Finance (control) | from 2011 | no | NASDAQ (^IXIC), S&P 500, gold |
| FRED | from the 1960s | **yes** (free) | M2, industrial production, Fed funds, unemployment |

Binance has the most data in terms of depth and volume, but its history starts
in 2017 — the exchange did not exist before that. So the series is **stitched**:
Bitstamp up to 2017-08, Binance afterwards. The seam is explicit and checked on
the overlap (median divergence on shared days: **0.06%**, maximum 9% in
December 2017, at the peak of the mania, when exchanges genuinely diverged).

M2 and PMI have no free key-less API. M2 comes from FRED once you paste a free
key into `.env`; ISM PMI has a licence that forbids redistribution, so drop
your own file into `data/raw/manual/pmi.csv` and the tool will pick it up.
Without a FRED key the macro phase falls back to market proxies (DXY, the rate
curve) — and `run.py macro` reports which source the liquidity axis is built
from, plus, when both are available, how much they agree.

See [DATA_SOURCES.md](DATA_SOURCES.md) for attribution requirements and known
redistribution restrictions. **Yahoo Finance prohibits redistributing its
data** — this matters if you host the dashboard publicly.

### First releases, not revisions

FRED data is fetched as **first releases** (`output_type=4`), so
`available_from` is the date the number actually entered circulation. Measured
on the real M2SL vintages: the **median publication lag is 43 days**, maximum
58 — two weeks more than the intuitive "a month after the period ends". All 559
observations received a date from the vintage archive, with zero fallbacks.

Three edge cases, each with a test in `tests/test_fred_vintages.py`:

* the `1776-07-04` sentinel — a series with no vintage archive,
* a lag longer than ~5 months — an observation older than the ALFRED archive,
  or the result of a methodology revision,
* **too many vintages** — FRED serves first releases up to 2000 vintage dates
  and DFF has 5113. The series exceeding that limit are the *daily* ones, i.e.
  exactly those published the next day and essentially never revised, so we
  fetch them without the archive using a fixed one-day lag.

Error messages pass through `ingest.http.redact` — the API key travels in the
query string, so a raw URL inside an exception would leak it into logs.

---

## The time contract — the most important thing in this repo

Every row in the database carries **two** dates:

* `date` — the day the value refers to,
* `available_from` — the day it was publicly known.

For prices the difference is one day (the bar for day D closes at midnight).
For M2 it is about six weeks. The entire analysis filters on `available_from`,
never on `date`.

`features/checks.py` verifies this point-in-time: for a chosen day t the
features are built twice — once with the whole history, once with only the data
published up to t — and must come out identical. The test deliberately includes
a **planted leak** (normalising by the median of the whole sample) to prove the
detector works.

---

## Forecasting: what the tool will and will not do

It does **not** forecast a price level. At 4% daily volatility the confidence
interval around a 30-day price is wider than the forecast, so the number would
carry no information.

It does estimate the **probability that the 30-day forward return is positive**,
and then spends most of its effort deciding whether that estimate is worth
anything. The model is a ridge-penalised logistic regression on the same
point-in-time features as everything else; the penalty is chosen on an inner
chronological split of each training window, never on the test window.

Three baselines have to be beaten before the model means anything:

| baseline | the objection it answers |
|---|---|
| `always_up` | "the asset just goes up" — uses the training-window base rate |
| `coin_flip` | a constant 0.5, the reference point for the Brier score |
| `momentum` | "last month continues" — the cheapest real predictor |

`always_up` is the one that matters. Bitcoin rose in **59%** of the 30-day
windows in this sample, so a model reporting 58% accuracy has demonstrated
nothing — and that is how most "BTC prediction models" are presented, against
an implied 50% nobody actually competes with.

Two further details the report insists on:

* **Overlapping labels.** A 30-day return on consecutive days shares 29 days of
  the future. 4644 daily predictions are not 4644 observations, so the headline
  metrics are computed on 155 non-overlapping rows. Both numbers are printed.
* **Discrimination is not calibration.** AUC says whether the ranking works;
  the Brier score also punishes being confidently wrong. A model can rank well
  and still lose on Brier — which is exactly what happens when the base rate
  swings between windows, as it does here (0.27 to 0.72 across test windows in
  the synthetic power test).

## Control group: the placebo test

A halving is a **Bitcoin-only** event. The NASDAQ over the same window has no
way of knowing about it — so if it reacts the same way, we are measuring the
common move of risk assets rather than a reward halving.

The test is **paired across events** (difference in differences): for each
halving we compute Bitcoin's CAR and the control's CAR over the same window and
infer from the distribution of their difference. Pairing matters — the 2020
halving fell in the middle of the pandemic rebound, which lifted both assets;
comparing two separate means would lose that.

**The calendar is not a detail.** The NASDAQ trades ~252 days a year, Bitcoin
365. Without alignment, "365 days after the halving" means 365 *rows* for the
NASDAQ, i.e. about 511 calendar days — and an event falling on a weekend does
not exist in the index at all. On test data that costs **3 of 4 events**
(`test_without_calendar_alignment_most_events_are_lost`). So the control series
is mapped onto the full calendar with `ffill`, a strictly backward-looking
operation.

---

## Methodological decisions (and why)

**The unit of observation is the event, not the day.** With four halvings the
uncertainty comes from having had four halvings, not from having had 1460 days.
Inference therefore uses the t distribution with n−1 degrees of freedom across
events. The percentile bootstrap is reported alongside but **does not decide**:
at n = 3–5 it produced ~30% false discoveries instead of 5% (measured in
`test_false_positive_rate_stays_near_nominal`).

**Abnormal returns are measured against a pre-event window** (−250..−31 days),
not against the full-sample mean — the full-sample mean already contains
whatever the event is supposed to explain.

**The "window vs rest of sample" test uses circular shifts of the mask.** A
plain t-test on daily returns assumes independence, which prices do not have,
and systematically overstates significance. Rotation preserves the
autocorrelation of both series.

**An embargo between training and test.** The target `fwd_return_90d` on day t
contains prices from t+90, so without a gap the last training days see the test
set.

**Walk-forward instead of a single split.** Splitting by cycle gives *one* test
set — "it did not replicate" is a single observation there. Walk-forward gives
13 disjoint yearly windows (2013-11 → 2026-09) and lets us count how often the
sign of an effect survives the move from training to test. Under the null that
is a coin flip, so the number of agreeing folds is binomial.

The step between folds **must** be at least the length of the test window,
otherwise the windows overlap and the t-test across folds stops being valid.
The pipeline checks disjointness on the actual indices, not on the parameters,
and reports no t-test when they overlap. The first version of the config had
this bug (a 182-day step with a 365-day window) and overstated significance.

**Costs are charged on turnover, not per trade.** A signal from day t takes
effect with a lag — zero lag means trading at a price that has not settled yet.
`test_execution_lag_blocks_same_day_knowledge` shows the difference: the same
"signal" returns +1000× with no lag and nothing with one.

---

## Results

On 5496 days, 2011-08-18 → 2026-09-03. The full snapshot lives in
[data/processed/RESULTS.md](data/processed/RESULTS.md).

**Event study, halvings.** CAR after 365 days = **+125.2%**, confidence
interval **[−190.5%, +441.0%]**, p = 0.296, n = 4. The effect may be enormous
or negative — with four observations there is no way to tell. That is not a
failure of the method; it is all the information there is.

**Control group** (CAR at 365 days, paired across events):

| | CAR BTC | CAR control | difference | 95% CI | p |
|---|---|---|---|---|---|
| vs NASDAQ | +125.2% | +17.0% | +108.2% | [−169.4%, +385.8%] | 0.303 |
| vs S&P 500 | +125.2% | +13.7% | +111.5% | [−167.7%, +390.7%] | 0.293 |
| vs gold | +125.2% | −17.3% | +142.5% | [−200.5%, +485.4%] | 0.278 |

None of these differences is distinguishable from zero. The placebo works in
the other direction: run on the NASDAQ, the method does *not* find a halving
effect there (+17.0%, p = 0.307) — so the absence of a result for BTC is not
the method being powerless.

**Validation.** 26 hypotheses (halving windows × event categories × macro
phases): **0** significant raw, **0** after Benjamini-Hochberg correction.
Chance alone would have given ~1.3. One hypothesis passes the out-of-sample
replication rule, `event_credit_event_7d`, and it is reported with the two
facts that sink it: it was not significant in the training window (p = 0.42),
so there was nothing to replicate, and its corrected q = 0.75.

**Placebo group.** The `protocol_upgrade` category holds the five consensus
changes enforced on Bitcoin mainnet inside the price window — complete by
construction and pre-announced, so nothing could have been learned on those
dates. In walk-forward it produces the **lowest raw p-value in the table**
(0.032), and in the forecast model the third-largest coefficient. Uncorrected,
this pipeline would announce that Bitcoin reacts to software upgrades whose
dates were public months in advance. Read it as calibration: a raw p between
0.03 and 0.25 is routinely produced here by a variable known to be empty.

**Specification curve.** 160 combinations of price series, horizon, return
type, abnormal-or-raw and estimation window. Median CAR **+5.2%**, **2 of 160**
significant uncorrected. Shifting the same four dates to a random place in the
history and rebuilding the whole curve, 200 times, produces **15.1** significant
specifications on average — *seven times more than the real halvings*. Curve
level p = 0.706. The curve also names what the apparent effect is: removing the
pre-event baseline grows it eightfold, moving that baseline closer to the event
nearly erases it, and it scales with the horizon without limit. Those are the
signatures of drift.

**What is predictable: the range, not the direction.** Everything above says
direction is not forecastable here. None of it says the *size* of the next move
is not, and it is: volatility clusters. `run.py range --days 10` fits
GARCH(1,1) with Student-t innovations, simulates the horizon forward and quotes
an interval — and refuses to quote one until the interval has passed a coverage
backtest, testing on non-overlapping windows whether a 90% interval really
contained the outcome 90% of the time. The interval is centred on today's
price, never on the historical trend: fitting that trend would encode the one
thing this project showed it cannot predict.

**Walk-forward.** 13 disjoint windows: **0** significant after correction, on
both the sign test and the out-of-sample effect. The strongest case is
`phase_expanding_rising` with **5 of 5** matching signs — the best possible
outcome — and still p = 0.0625, because at five windows significance is
unreachable.

**Liquidity axis: real M2 vs the dollar proxy.** The two agree on **42.3%** of
5130 compared days — *below* chance level (50% for a binary axis), because in
this sample they are opposed. The phase "liquidity rising, rates rising"
returns **−20.6%** a year computed from M2 and **+105.9%** computed from the
proxy. Same label, opposite conclusion. Do not treat the proxy as an
approximation of M2.

**Backtest** (10 bps fees + 15 bps slippage):

| strategy | return | CAGR | Sharpe | maxDD | exposure |
|---|---|---|---|---|---|
| buy and hold | +715,635% | 80.3% | 1.14 | −84.9% | 100% |
| trend 50/200 | +885,999% | 82.9% | **1.27** | −78.3% | 61% |
| halving +365d | +254,351% | 68.4% | **1.42** | −71.0% | 27% |
| halving +180d | +2,555% | 24.3% | 0.87 | −70.3% | 13% |
| macro (M2 up, rates down) | +257% | 8.8% | 0.46 | −57.4% | 10% |

The only strategy with a better Sharpe than buy-and-hold at meaningful
exposure — `halving +365d`, 1.42 at 27% time in market — **fails every
statistical check above and is indistinguishable from the NASDAQ**. Four cycles
are four observations, and "be in the market for a year after the halving"
largely overlaps with "be in the market during a bull run".

---

## Layout

```
run.py           cross-platform launcher and installer
btc.cmd / btc.sh thin wrappers around it
src/
├── ingest/      fetching (4 exchanges, FRED, Yahoo, manual CSV) + quality checks
├── features/    halving distance, macro phase, event flags + look-ahead detector
├── analysis/    event study, HAC correlations, control group (placebo)
├── backtest/    engine with costs and slippage + strategies
├── validation/  splits, walk-forward, Bonferroni/BH, synthetic data
├── pipeline.py  assembles everything - used by both the CLI and the dashboard
└── cli.py
dashboard/       Streamlit - presentation only, no logic (a test enforces this)
tests/           176 offline tests + network tests, one file per phase
```

The dashboard must not compute statistics. If a number appears on screen it
comes from `pipeline.py`, so the terminal and the browser can never disagree.

---

## Limitations worth remembering

* **The event registry was assembled after the fact** and is biased by that —
  we remember the events the market reacted to. The `source` column in
  `data/raw/events.csv` is deliberately empty: fill it in and verify the dates
  before drawing conclusions. Add the boring events too, the ones that looked
  alarming and came to nothing. This is the weakest part of the project and the
  most valuable thing to contribute.
* **Four halvings is the ceiling on statistical power.**
  `test_five_events_cannot_detect_a_drift_buried_in_btc_scale_noise` exists so
  that nobody "improves" the method into finding effects the data cannot
  support.
* **Binance quotes USDT, not USD.** Outside de-pegging episodes the difference
  is a fraction of a percent, but the stitching reports it rather than
  silently accepting it.
* **The backtest models no funding, taxes or liquidity limits** on large
  orders. The 15 bps slippage is an assumption, not a measurement.

## What next

1. Fill in `source` in the event registry and add events the market ignored.
2. Add a second control asset class outside equities.
3. Package for PyPI so `pip install` works.

## Documents

| File | Contents |
|---|---|
| [DISCLAIMER.md](DISCLAIMER.md) | not investment advice — read this first |
| [LICENSE](LICENSE) | MIT |
| [TERMS.md](TERMS.md) | terms of use |
| [PRIVACY.md](PRIVACY.md) | privacy policy (the tool collects nothing) |
| [DATA_SOURCES.md](DATA_SOURCES.md) | attribution and redistribution restrictions |
| [CONTRIBUTING.md](CONTRIBUTING.md) | how to contribute, and the house rule |
| [SECURITY.md](SECURITY.md) | reporting vulnerabilities |
| [CHANGELOG.md](CHANGELOG.md) | what changed, and which changes moved the numbers |
