# Contributing

Contributions are welcome. This project has one unusual house rule, so read the
first section before writing code.

## The house rule: a result is not a number

Every statistic that leaves this codebase must carry the information needed to
judge it: a confidence interval, the number of observations, and — where
several hypotheses were tested — a correction for that. A pull request that
reports a mean without its uncertainty will be asked to add it, even if the
mean is correct.

Two consequences worth stating explicitly:

* **Do not "improve" a method so that it starts finding effects.** Several
  tests exist specifically to pin down the limits of what this sample can
  support, for instance that four halvings cannot reach significance and that
  five walk-forward folds cannot either at perfect sign agreement. If your
  change makes one of those tests fail, the change is wrong, not the test.
* **Negative results are results.** The project's headline finding is that
  there is no detectable halving effect. Contributions that strengthen a
  negative result are as valuable as ones that find something.

## Setup

```
python run.py test
```

That creates the virtual environment, installs dependencies and runs the suite.
Nothing else is needed. Python 3.11 or newer.

## Before opening a pull request

1. `python run.py test` passes. Tests that reach real APIs are marked
   `network`; `python run.py test offline` skips them, but run the full suite at
   least once before submitting.
2. New behaviour has a test. New *statistics* have two: one showing the method
   stays quiet on pure noise, and one showing it detects an effect that was
   deliberately injected. A method that never finds anything passes the first
   test alone.
3. Anything touching data timing keeps the point-in-time contract intact — see
   the "Time contract" section of the README. The look-ahead detector in
   `features/checks.py` must stay green.
4. Comments explain *why*, not *what*. The code says what it does.

## Style

* Follow the surrounding code. It is plain, typed where it helps, and avoids
  cleverness.
* English only, in code, comments, docstrings and output.
* No new dependency without a reason in the pull request description. The
  dependency list is deliberately short.

## Where things live

| Path | Contents |
|---|---|
| `src/ingest/` | fetching data, quality checks |
| `src/features/` | feature construction, look-ahead detector |
| `src/analysis/` | event study, correlations, control group |
| `src/backtest/` | engine and strategies |
| `src/validation/` | splits, multiple-testing correction, synthetic data |
| `src/pipeline.py` | assembles the above; both CLI and dashboard call it |
| `dashboard/` | presentation only, no analysis logic (a test enforces this) |

The dashboard must not compute statistics. If a number needs to appear on
screen, it comes from `pipeline.py`, so that the terminal and the browser can
never disagree.

## Adding historical events

`data/raw/events.csv` is maintained by hand and is the weakest part of the
project: it was assembled after the fact, so it over-represents events that
moved the market. The most useful contribution is filling in the `source`
column and adding events that looked alarming at the time and turned out to be
nothing. Include a source for every row you add.

## Licence

By contributing you agree that your contribution is licensed under the MIT
Licence, the same as the rest of the project.
