# Changelog

Notable changes to btc-cycle-lab. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

One rule specific to this project: **a change that alters what a number means
is a breaking change**, even when no interface moves. Adding a hypothesis to
the scan changes every corrected p-value in it; changing an estimation window
changes every abnormal return. Those belong under **Changed** with the effect
on published results spelled out, never under **Fixed** as a detail.

## [1.0.0] — 2026-09-05

First release. The project answers its question, and the answer is negative:
**on data from 2011 to 2026 there is no detectable halving effect in the price
of BTC.** Four independent checks agree, and each closes a different way of
disputing it.

| check | the objection it answers | result |
| --- | --- | --- |
| control group | "everything rose, so what?" | difference vs NASDAQ p = 0.30 |
| placebo category | "your method finds effects everywhere" | it does — raw p = 0.03 where nothing happened |
| walk-forward | "it worked in the past" | 0 of 26 stable across 13 disjoint windows |
| specification curve | "you picked the wrong window" | 160 specifications, none of it survives |

The binding constraint is four halvings. No method fixes that, and none of the
above is expected to change without a fifth.

### Analysis

- Event study with inference across **events**, not days: t distribution with
  n−1 degrees of freedom, so n=4 gives a critical value of 3.18 rather than
  1.96. A percentile bootstrap is reported alongside but never used to decide
  — at n<10 it produced ~30% false discoveries in testing.
- Hypothesis scan over 26 combinations of window, event category and macro
  phase, with Benjamini-Hochberg correction: **0 significant, raw or
  corrected.** Replication out of sample is split by halving cycle, and one
  hypothesis passes that rule — `event_credit_event_7d`. It is reported with
  the two facts that sink it: its training-window effect was not significant
  (p = 0.42), so there was nothing to replicate, and its corrected q = 0.75.
  The rule was deliberately not tightened after seeing which hypothesis it
  would exclude; a `significant_in_train` column surfaces the fact instead.
- Walk-forward validation across 13 disjoint windows, with an embargo covering
  the target horizon.
- Control groups (NASDAQ, S&P 500, gold) compared **paired by event**, with a
  placebo run on the control itself to show the comparison can detect nothing
  when there is nothing.
- `protocol_upgrade` placebo category: the five consensus changes enforced on
  mainnet inside the price window. Complete by construction and pre-announced,
  so any effect found there measures the method. It produces the lowest raw
  p-value in walk-forward and the third-largest forecast coefficient.
- Specification curve: 160 combinations of price series, horizon, return type,
  abnormal-or-raw and estimation window, with inference by circular permutation
  of the whole curve rather than by counting significant specifications.
- Directional forecast (30-day horizon) with ridge-penalised logistic
  regression, scored against three baselines and thinned to non-overlapping
  labels. Verdict: **no edge**.
- Backtests against buy-and-hold with fees and slippage.

### Data

- Prices stitched across Bitstamp, Coinbase and Binance with the seam reported
  and checked on overlaps (median divergence 0.06%).
- Macro from FRED as **first releases** (`output_type=4`), so `available_from`
  is when a number entered circulation. Measured median publication lag for
  M2SL: 43 days.
- Point-in-time contract throughout: every row carries both the observation
  date and the publication date, and all analysis filters on the latter.
- Hand-maintained event registry, every date checked against a named source.

### Platforms

- Windows, macOS and Linux on Python 3.11 and 3.13, all six combinations
  verified on every push through the real installation path.
- `run.py` does the work — virtual environment, dependencies, dispatch — with
  `btc.cmd` and `btc.sh` as thin wrappers.
- Streamlit dashboard on port 8511.

### Documentation

MIT licence, plus DISCLAIMER, PRIVACY, TERMS, DATA_SOURCES, SECURITY and
CONTRIBUTING. Yahoo Finance prohibits redistributing its data, which matters
for anyone hosting the dashboard publicly; only derived statistics are
committed here.

---

## Development history

Kept because several entries changed what the published numbers mean, and a
reader comparing an old figure to a current one deserves to know why.

### Methodological changes

- **Specification curve** (2026-09-04) — showed the apparent halving effect is
  drift: removing the pre-event baseline grows it eightfold, moving the
  baseline closer to the event nearly erases it, and it scales with the
  horizon without limit.
- **Placebo category, `cycle_extreme` removed** (2026-09-04) — the scan went
  from 23 hypotheses with 3 skipped to 26 with none skipped, so every corrected
  p-value from before that date is not comparable with one from after. A cycle
  extreme is identified from the price itself, so an event study around one
  recovers its own selection rule as a finding.
- **Event registry sourced** (2026-09-03) — two of twenty dates were wrong.
  `mtgox_halt` moved 2014-02-25 → 2014-02-24 and `cme_futures` 2017-12-17 →
  2017-12-18, and a missing Mt. Gox event was added. Credit-event results
  changed; halving results did not, because no halving date was wrong.
- **Inference switched from bootstrap to the t distribution** — the bootstrap
  was miscalibrated at these sample sizes.
- **Walk-forward replaced the single cycle split** (2026-09-02) — one test
  window is one observation.

### Bugs whose fixes changed results

- **Event registry stored by merge instead of replace** — the primary key is
  (name, date), so correcting a date left the old row behind and the event
  study averaged over both. The registry now replaces; prices keep their
  append-only upsert.
- **Events with no estimation baseline entered as flat zeros** — subtracting a
  NaN baseline made the whole row NaN, and `nancumsum` scored that as an event
  after which nothing happened. Affected permutation draws near the start of
  the history, which is the null distribution every specification-curve
  p-value is measured against.
- **Benjamini-Hochberg poisoned by NaN p-values** — the running minimum from
  the end contaminated the whole column.

### Packaging and platform

- **Launchers fixed** (2026-09-05) — `.cmd` files had LF line endings, and
  cmd.exe re-reads a running batch file by byte offset, so `call :label` failed
  for labels far enough into the file. The POSIX launcher was renamed `btc` →
  **`btc.sh`**: Explorer hides known extensions, so the two appeared as
  identical entries and the unrunnable one sorted first. Double-clicking now
  either works or says why.
- **English throughout and publication documents** (2026-09-03), with a test
  that keeps it that way after `{"kategoria": ...}` reached a public repo. <!-- non-english-ok: naming the bug -->
  That test caught this very line, which is the behaviour intended.
- **UTF-8 output on Windows** — a legacy console code page turned an em dash in
  an event description into a crash.

[1.0.0]: https://github.com/quasinbox-netizen/EP6/releases/tag/v1.0.0
