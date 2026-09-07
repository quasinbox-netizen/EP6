# Changelog

Notable changes to btc-cycle-lab. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

One rule specific to this project: **a change that alters what a number means
is a breaking change**, even when no interface moves. Adding a hypothesis to
the scan changes every corrected p-value in it; changing an estimation window
changes every abnormal return. Those belong under **Changed** with the effect
on published results spelled out, never under **Fixed** as a detail.

## [1.1.0] - 2026-09-07

### Changed

- **The published paper portfolio was restated: -14.5% since the last halving
  becomes -7.7%.** `run_paper` applies the one-day execution lag itself and was
  being handed a series the backtest engine had already lagged, so every entry
  and exit landed a day late and the account paid for a day of moves it was not
  in. Every trade date on the page moves back one day (the 2024-08-11 sell is
  now 2024-08-10, and so on). The rule did not change and its verdict did not
  change: it still loses to holding, which returned +21.6% over the same window.
  The correction is printed on the page itself, not only here.
- **The two constants that size the portfolio are now disclosed as fitted.**
  The volatility target of 60% is BTC's own median forecast volatility over
  this history (measured at 59.95%), not an outside convention; the 30%
  rebalance band was selected as the highest-Sharpe variant among those
  compared, on the same history the edge test then scores. The comparison is
  published in full, including the variant that scores higher than the one
  adopted. No number changed - what changed is that the page no longer presents
  either as though it came from outside the data.

- **`sizing` now edge-tests the row it sizes with.** It tested whichever
  variant had the best Sharpe, so the p-value on screen belonged to a strategy
  the portfolio does not run - `vol target, EWMA band 10%`, p = 0.40 - while
  the position came from `vol target, band 30%`, p = 0.079. Choosing the row on
  its Sharpe and then testing that same row on that same sample is the
  selection this project rejects elsewhere. The verdict is unchanged: neither
  is significant, and the sizing still earns no risk-adjusted edge. The
  best-scoring row is still printed by name.

- **The rebalance band is no longer defended on its Sharpe.** A sweep of all 61
  values from 0% to 60% shows the adopted 30% ranks **34th**, below the middle
  of its own grid; the best is 42% at 1.1886 against 1.1316. Neighbouring bands
  differ by 0.0149 on average against a standard error of 0.0202 for any one of
  these Sharpes, so the ranking is noise. From 52% the band is wider than the
  position ever moves: the rule stops trading and takes buy-and-hold's Sharpe
  by construction, which is why the highest scores cluster at the right-hand
  edge. **The band was not moved** - adopting 42% for winning this sweep is the
  mistake the sweep exposes. It is now defended on turnover, which is a cost
  and does not move, rather than on a score that does.

- **The volatility target is no longer defended on its Sharpe either, and for a
  different reason: that was never the right measurement.** The target is a dial
  on how much risk to carry - across 39 values from 10% to 200% it moves the
  worst drawdown from -18% to -83% - and a Sharpe ratio is built to be
  indifferent to how large a position is. On that statistic the adopted 60%
  ranks 14th of 39 (1.1316 against 1.2151 for the best, at 30%). The number that
  does describe what the target is doing had never been published at all: **at
  60% the position sits at the no-borrowing cap on 50.2% of days**, so on half
  of them the rule is not sizing anything, it is holding the asset. From a
  target of 160% it is pinned on 99% of days and the rule simply is
  buy-and-hold, its drawdown converging on buy-and-hold's own -83%. **The target
  was not moved**; what changed is that it is defended as a risk choice rather
  than a measured optimum, with the half-of-days figure attached.

- **Every sizing figure is restated: the cached volatility forecast had drifted
  from the prices it was built from.** The stitch now prefers Binance from
  2017-08-17, where Binance's history begins, in place of Bitstamp; the cache
  predated that, so 3,305 of 4,035 daily forecasts behind the published sizing
  were computed from a price series the database no longer holds. The band's
  rank in its own sweep goes from 34th of 61 to **last**, the target's from
  14th of 39 to 36th, and today's position from 0.82 to 0.79. The estimator is
  unchanged and was verified deterministic: identical prices give identical
  forecasts, and appending four days changes nothing before the last day, so
  the whole move is the input. Both figures are on the page's corrections log.
  The rank collapsing from mid-grid to bottom on a change of price source is
  the sharpest evidence yet for what the sweep already argued - that ranking
  these constants on this sample measures nothing.
- `backtest/rank_history.py`: a log of where each constant placed in its own
  sweep, one row per run, written every time the sizing is rebuilt and never
  pruned. The sweeps argue that ranking these constants measures nothing using
  statistics computed inside a single run; this makes the same argument by
  keeping the ranking and watching it move, which needs no standard error to
  believe. Rows are keyed on the day they were RECORDED rather than the last
  day of data used, because those come apart exactly when it matters - two runs
  over the same window can disagree once the window is restated underneath
  them, and both answers are worth keeping in the order they were believed.
  Charted on the agent page as each value's place in its own grid, so a
  61-value sweep and a 39-value one share an axis; below three distinct days
  the section says how many runs it holds and withholds the chart rather than
  drawing a trend through two points.
- The band's turnover figures on the agent page are read from the sweep instead
  of written into the prose. They said "roughly 8.9x to 1.1x" and the refreshed
  data already says 8.8x to 1.2x - harmless the day it was written, wrong within
  a week of the daily job being switched on.
- **The daily job now rebuilds the volatility forecast and the sizing tables.**
  `sizing_today.csv` holds the position the agent page publishes, and it moved
  only when the command was run by hand - so the site could advertise a daily
  refresh while quoting a size computed on an arbitrary earlier day, with
  nothing on the page to show the difference. The step runs `sizing --refresh`
  (a bare `sizing` re-reads the cached forecast and would have been a no-op
  dressed as a fix), sits between `ingest` and `paper` because both `paper` and
  `publish` read what it writes, and is non-blocking: a failed refit leaves
  yesterday's sizing in place and the day is still recorded. The four sizing
  tables are now committed with the pages that chart them.

### Added

- `backtest/sweep.py`: the band sweep, with the statistics that disqualify its
  own argmax - the adopted band's rank, the mean step between neighbours, Lo's
  iid standard error (the optimistic one, stated as such), and the band from
  which the rule stops trading. Saved as `sizing_sweep.csv` and charted on the
  agent page with the frozen region shaded rather than cropped. `target_sweep`
  does the same for the volatility target and additionally records
  `at_the_cap`, the share of days the position is pinned at the no-borrowing
  limit - the column that says what a risk dial is doing, which no score can.
  Saved as `sizing_target_sweep.csv` and charted with Sharpe and drawdown on
  the same axes, because the comparison between those two lines is the point.
- `publish/disclosure.py`: an append-only corrections log rendered on the page
  whose figures moved, and the provenance note for the sizing constants. A
  restatement absent from that log is a restatement the reader never sees, so
  it is pinned by tests rather than by review.
- `sizing_today.csv` records `target_annual_volatility`, so the published
  provenance names the target the run actually used and cannot drift from it.

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
