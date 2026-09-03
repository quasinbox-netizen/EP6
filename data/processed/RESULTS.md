# Results snapshot — 2026-09-01

Generated with `python run.py all --post 365`.
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
| market_structure | 3 | −304.7% | 0.197 |
| macro | 4 | +106.1% | 0.287 |
| halving | 4 | +125.2% | 0.296 |
| credit_event | 6 | −81.8% | 0.494 |

These numbers must not be read as results — they are the input to the
correction below.

## Validation

**23 hypotheses** (halving windows × event categories × macro phases):

* significant raw: **0** — chance alone would give ~1.2,
* after Benjamini-Hochberg correction: **0**,
* replicating out of sample (cycles 0–2 → 3–4): **0**,
* skipped for lack of data: 3 (the `cycle_extreme` category is still empty).

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

That is a result, not a failure. The project was built to tell a pattern from
noise, and it did.

---

## What is not here

`features.csv` (3.9 MB) and `lab.sqlite` (11 MB) are deliberately outside
version control — one command regenerates them, and versioning them would bloat
the history on every run.
