# Results snapshot — 2026-09-04

Generated with `python run.py all --post 365`, plus `python run.py speccurve`,
which `all` leaves out because its 200 permutations take about ten minutes.
To reproduce: `python run.py ingest --what all`, then `python run.py all`.

Sample: **5494 days**, 2011-08-18 → 2026-09-01. Stitched series — Bitstamp for
2191 days up to 2017-08-16, Binance for 3303 days afterwards. Median divergence
on the overlap 0.06%, maximum 9.0% (2017-12-23, a genuine divergence between
exchanges at the peak of the mania). No missing days, no duplicates.

---

## Halvings

CAR computed across events, confidence interval from the t distribution with
n−1 degrees of freedom.

| horizon | CAR | 95% CI | p |
|---|---|---|---|
| 30 days | −0.8% | [−25.7%, +24.0%] | 0.920 |
| 90 days | +12.1% | [−67.7%, +91.9%] | 0.663 |
| 180 days | +47.7% | [−117.7%, +213.2%] | 0.426 |
| 365 days | **+125.2%** | **[−190.5%, +441.0%]** | **0.296** |

n = 4. The effect may be enormous or negative — with four observations there is
no way to decide.

## Control group (placebo test)

Difference paired across events, 365-day horizon, calendar windows.

| | CAR BTC | CAR control | difference | 95% CI | p |
|---|---|---|---|---|---|
| vs NASDAQ | +125.2% | +17.0% | +108.2% | [−169.4%, +385.8%] | 0.303 |
| vs S&P 500 | +125.2% | +13.7% | +111.5% | [−167.7%, +390.7%] | 0.293 |
| vs gold | +125.2% | −17.3% | +142.5% | [−200.5%, +485.4%] | 0.278 |

No difference is distinguishable from zero. The placebo works in the other
direction: run on the NASDAQ, the method finds no halving effect there
(+17.0%, p = 0.307), so the absence of a result for BTC is not the method being
powerless.

## Event categories (p-values are RAW, before correction)

| category | n | CAR(365d) | raw p |
|---|---|---|---|
| regulation | 3 | −194.4% | 0.081 |
| market_structure | 3 | −301.3% | 0.197 |
| macro | 4 | +106.1% | 0.287 |
| halving | 4 | +125.2% | 0.296 |
| credit_event | 7 | −131.4% | 0.261 |

These numbers must not be read as results — they are the input to the
correction below.

## Validation

**26 hypotheses** (halving windows × event categories × macro phases):

* significant raw: **0** — chance alone would give ~1.3,
* after Benjamini-Hochberg correction: **0**,
* skipped for lack of data: **0** - the `cycle_extreme` category was removed
  (a cycle top is identified from the price itself, so an event study around
  one recovers its own selection rule) and `protocol_upgrade` replaced it,
* replicating out of sample (cycles 0–2 → 3–4): **1**, and it needs unpacking.

`event_credit_event_7d` — the week after a credit event — passes the
replication rule: same sign in both windows, significant in the test window
(p = 0.031), effect larger rather than smaller. It is still not evidence of
anything, for two reasons that the table now shows explicitly:

1. **It was never established in training.** The training-window effect had
   p = 0.42. Replication means an effect found in one sample reappearing in
   another; there was nothing to reappear. The `significant_in_train` column
   exists so this cannot hide.
2. **It does not survive multiple testing.** Across the full sample its raw
   p = 0.095 becomes **q = 0.75** after correction — nowhere near significant
   among 26 hypotheses.

The replication rule was deliberately left as it is rather than tightened to
require training significance. Changing a criterion after seeing which
hypothesis it would exclude is the same error this project exists to catch,
merely pointed in the flattering direction. The fact is surfaced as a column
instead, and the reader can weigh it.

## Walk-forward validation (13 disjoint windows, 2013-11 → 2026-09)

For each hypothesis we check how often the sign of the effect survives the move
from the training window to the test window. Under the null that is a coin flip.

| hypothesis | folds | agreeing | agreement | sign p | q (BH) | mean OOS effect | effect p | effect q |
|---|---|---|---|---|---|---|---|---|
| phase_expanding_rising | 5 | 5 | 100% | 0.063 | 1.000 | −0.0028 | 0.017 | 0.300 |
| phase_contracting_rising | 11 | 4 | 36% | 0.549 | 1.000 | +0.0005 | 0.638 | 0.932 |
| phase_contracting_falling | 10 | 6 | 60% | 0.754 | 1.000 | +0.0006 | 0.651 | 0.932 |
| halving_after_365d | 7 | 3 | 43% | 1.000 | 1.000 | +0.0097 | 0.344 | 0.932 |
| halving_after_180d | 4 | 2 | 50% | 1.000 | 1.000 | −0.0008 | 0.619 | 0.932 |

**Significant after correction: 0** — on neither the sign test nor the
out-of-sample effect. Both statistics go through the correction; correcting
only one and quoting the other raw would mean picking the convenient one after
seeing the results.

A power limit worth remembering: `phase_expanding_rising` has **5 out of 5**
matching signs, the best possible outcome — and still p = 0.0625. At five
windows significance cannot be reached even with perfect stability.

## Liquidity axis: real M2 vs the dollar proxy

Agreement on **42.3%** of 5130 compared days — *below* chance level (50% for a
binary axis), because in this sample the two are opposed.

| | proxy: contracting | proxy: expanding |
|---|---|---|
| **M2: contracting** | 1588 | 2193 |
| **M2: expanding** | 767 | 582 |

The phase "liquidity rising, rates rising" returns −20.6% a year computed from
M2 and +105.9% computed from the proxy. Same label, opposite conclusion.

## Backtest (10 bps fees + 15 bps slippage)

| strategy | return | CAGR | Sharpe | maxDD | exposure |
|---|---|---|---|---|---|
| buy and hold | +715,635% | 80.3% | 1.14 | −84.9% | 100% |
| trend 50/200 | +885,999% | 82.9% | **1.27** | −78.3% | 61% |
| halving +365d | +254,351% | 68.4% | **1.42** | −71.0% | 27% |
| halving +180d | +2,555% | 24.3% | 0.87 | −70.3% | 13% |
| halving +90d | +242% | 8.5% | 0.76 | −21.8% | 7% |
| halving +30d | +12% | 0.7% | 0.13 | −20.5% | 2% |
| macro (M2 up, rates down) | +257% | 8.8% | 0.46 | −57.4% | 10% |

---

## The specification curve

Everything above holds four analytical decisions fixed: the estimation window
(-250..-31), abnormal rather than raw returns, log rather than simple returns,
and the price series stitched across three exchanges. All 26 hypotheses inherit
all four. If one is wrong they are wrong together, and no other check here
would notice.

So the halving study was re-run under every combination of them - 160
specifications, varying also the horizon and which exchange history to trust.

**The curve.** Median CAR across specifications **+5.2%**; 75% of them positive;
**2 of 160** significant at 5% uncorrected.

**Inference.** The count of significant specifications is not a test. These are
re-analyses of the same four halvings, so they move together and the count is
nowhere near binomial. Inference comes from shifting the four dates as a block
to a random place in the price history and rebuilding the entire curve, 200
times. The shift is circular, and because the halvings span 4161 days of a
5496-day index, 81% of the draws wrap - keeping the gaps between
events circularly rather than on the calendar.

| | observed | under the null |
| --- | --- | --- |
| median \|CAR\| exceeded | — | in **70.6%** of draws |
| significant specifications | **2** | **15.1** on average, 43 at the 95th percentile |

**VERDICT: NOT ROBUST.** Random placement of the same four dates reproduces this
curve, and in fact produces *seven times more* significant specifications than
the real halvings do. The result does not depend on the specification, because
there is no result to depend on it.

### Which choices actually move the answer

This is the part worth keeping. Median CAR, grouped by one decision at a time:

| decision | value | median CAR |
| --- | --- | --- |
| horizon | 30 days | −0.9% |
| | 365 days | **+104.3%** |
| baseline | abnormal (subtract pre-event window) | +3.1% |
| | raw (subtract nothing) | **+24.5%** |
| estimation window | −90..−11 (ends near the event) | +0.3% |
| | −250..−31 (project default) | +10.1% |
| price series | Coinbase (3 of 4 halvings) | +0.5% |
| | Bitstamp (4 of 4) | +9.5% |
| return type | log | +5.3% |
| | simple | +4.7% |

Three things fall out of this table.

1. **The apparent halving effect is drift.** Subtracting a pre-event baseline
   removes seven eighths of it (+24.5% → +3.1%), and moving that baseline
   closer to the event removes almost all of what is left (+10.1% → +0.3%).
   What looks like a response to the halving is the market's own trend, already
   under way before the event and simply continued after it.
2. **It is a function of how long you wait, not of what happened.** The measured
   effect grows monotonically with the horizon, from −0.9% at 30 days to +104%
   at 365. An event response does not behave like that; holding a rising asset
   for longer does.
3. **Log versus simple returns is the choice that does not matter** - a
   difference of 0.6 points - while which exchange's history you trust changes
   the answer twentyfold. The debates worth having are not the ones about
   notation.

### The two significant specifications

Both are the same configuration:

    binance/30d/log/raw/none    n=2   CAR +11.9%   p=0.033
    binance/30d/simple/raw/none n=2   CAR +13.4%   p=0.036

Binance's history begins in August 2017, so it covers **two** of the four
halvings, and `raw` means no baseline was subtracted at all. The only
specifications that find anything are the ones using the least data and the
weakest correction. That is what a false positive looks like when you can see
where it came from.

## The placebo group

Every result below rests on one question: how often does this method report an
effect where there is none? The event registry now answers it with data rather
than assertion.

`protocol_upgrade` holds the five consensus rule changes enforced on Bitcoin
mainnet inside the price window - BIP66, BIP65, CSV, SegWit, Taproot. The
category is chosen to be uninformative in two independent ways:

* **Complete.** The protocol enumerates its members. Nobody selected these
  five, so they cannot have been selected for having moved the price.
* **Pre-announced.** Every activation height was locked in weeks ahead. Nobody
  learned anything on the day.

Nothing happened on those dates, in the only sense that matters to a market. So
whatever the analysis reports around them is a measurement of the analysis.

Here is what it reports.

| where | placebo result | rank among all hypotheses |
| --- | --- | --- |
| event study, 365d | CAR −20.4%, p = 0.76 | weakest of six categories |
| hypothesis scan, 30d | p = 0.209 | **5th of 26** |
| walk-forward, 30d | p = 0.032 | **strongest raw p-value in the table** |
| forecast model | coefficient −0.032 | **3rd largest of all features** |

The last two rows are the useful ones.

In walk-forward, the placebo produces the smallest raw p-value of any
hypothesis tested - smaller than every halving window, every regulatory event,
every macro phase. Run without correction, this pipeline would have announced
that Bitcoin reacts to software upgrades whose dates were public months in
advance. After Benjamini-Hochberg the same number is q = 0.33 and the finding
evaporates, which is the entire argument for correcting, produced here by the
data instead of by a simulation.

In the forecast model, the placebo feature receives the third-largest
coefficient of any predictor, ahead of `halving_after_365d`. A penalised model
fitted on 26 candidate signals will happily assign weight to a column that
cannot carry information. That is what overfitting looks like from the inside,
and it is why the verdict is read from out-of-sample scores and never from the
coefficient table.

**Calibration for reading everything else in this document:** a raw p-value
between 0.03 and 0.25 is routinely produced here by a variable known to be
empty. The halving's 90-day and 180-day windows (p = 0.64 and p = 0.41) are
*less* distinguishable from noise than the placebo is. Nothing in this project
should be believed on the strength of an uncorrected p-value.

## A note on the event registry

Every date in `data/raw/events.csv` was checked against a source before this
snapshot, and two were wrong:

* `mtgox_halt` was 2014-02-25; trading was actually suspended on **2014-02-24**,
* `cme_futures` was 2017-12-17; CME's own release announces an **18 December**
  launch (the book opened 17 Dec at 23:00 UTC, inside the 18 December daily bar
  this project uses).

One event was added: Mt. Gox halting **withdrawals** on 2014-02-07, which moved
the price more than the shutdown three weeks later and was simply missing.

That is a 10% error rate on the dates in a file whose numbers feed every event
study here — which is the argument for the `source` column existing at all.
Correcting them changed the credit-event results; the halving results did not
move, because no halving date was wrong.

## Directional forecast (30-day horizon, 13 folds)

Probability that the forward 30-day return is positive, scored on 155
non-overlapping rows pooled across the walk-forward folds.

| predictor | Brier | log loss | accuracy | AUC | base rate |
|---|---|---|---|---|---|
| model | 0.2805 | 1.2817 | 54.8% | 0.474 | 59.4% |
| always_up | 0.2471 | 0.6876 | 59.4% | — | 59.4% |
| coin_flip | 0.2500 | 0.6931 | 40.7% | 0.500 | 59.4% |
| momentum | 0.2476 | 0.6893 | 59.4% | 0.465 | 59.4% |

**Verdict: NO EDGE.** The model loses to every baseline on the Brier score
(−12.2% vs coin flip, −13.5% vs always-up, −13.3% vs momentum), and its AUC of
0.474 means its ranking is, if anything, marginally worse than random.

The penalty was selected per fold on an inner split of that fold's training
window, so this is not the result of a badly tuned model: with a fixed weak
penalty the Brier score was 0.39, far worse still.

Calibration in the middle buckets is reasonable (gap −0.03 and +0.06 where 85%
of the predictions fall), so the model is not nonsense — it simply has no
information about the future beyond the base rate.

The most recent prediction at the time of this snapshot was 54.6% against a
training base rate of 56.6%, i.e. 2 points *below* "the asset usually goes up".
Given the verdict above, that number is decoration.

## Conclusion

The only strategy with a better Sharpe than buy-and-hold at meaningful exposure
(`halving +365d`, 1.42 at 27% time in market) **fails statistical validation
and is indistinguishable from the NASDAQ**. Four cycles are four observations,
and "be in the market for a year after the halving" largely overlaps with "be
in the market during a bull run".

Four independent checks now say the same thing, and each closes a different
escape route:

| check | the objection it answers | result |
| --- | --- | --- |
| control group | "everything rose, so what?" | difference vs NASDAQ p = 0.30 |
| placebo category | "your method finds effects everywhere" | it does: raw p = 0.03 where nothing happened |
| walk-forward | "it worked in the past" | 0 of 26 stable out of sample |
| specification curve | "you picked the wrong window" | 160 windows, none of it survives |

The specification curve also says what the apparent effect actually is. Stop
subtracting a pre-event baseline and it grows eightfold; move the baseline
closer to the event and it nearly vanishes; extend the horizon and it grows
without limit. Those are the signatures of drift, not of a response to an
event. The halving is not moving the price in this data - the market's own
trend is, and the halving happens to sit inside it.

That is a result, not a failure. The project was built to tell a pattern from
noise, and it did.

---

## What is not here

`features.csv` (3.9 MB) and `lab.sqlite` (11 MB) are deliberately outside
version control — one command regenerates them, and versioning them would bloat
the history on every run.
