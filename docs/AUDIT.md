# Strategy Reality Check

Audit a trading track record that was produced somewhere else.

The lab was built to interrogate one hypothesis of its own, and its own
headline finding is negative. The machinery it grew doing that — a permutation
null that re-times a rule instead of reshuffling prices, a leave-one-out
fragility check, a correction for how many variants were tried, an
out-of-sample split with an embargo — answers a question almost nobody asks
about their own backtest: **how much of this could be luck?**

This command points that machinery at a CSV you bring in.

---

## Quick start

```bash
python run.py audit --example demo.csv
python run.py audit --file demo.csv --variants-tried 1
```

The first command writes a synthetic track record: a 20/80 moving-average
crossover on a random walk. It looks respectable — roughly +135% with a Sharpe
of 1.7 — and it contains no edge whatsoever, because there is nothing in a
random walk to have an edge on. The audit rejects it. That is the demo.

On your own file:

```bash
python run.py audit \
  --file my-strategy.csv \
  --variants-tried 40 \
  --cost-bps 25
```

Three files come back: the report in the terminal, a self-contained HTML
report next to the input, and the same content as JSON.

---

## What the file needs to contain

One column must be a date (`date`, `time`, `timestamp`, `datetime`). Then one
of three shapes, in order of how much can be tested:

| shape | columns | what you get |
|---|---|---|
| **positions** | `date`, `close`, `position` | everything, including the re-timing test |
| **equity** | `date`, `equity` | metrics, bootstrap, out-of-sample, concentration |
| **returns** | `date`, `return` | the same as equity |

`position` is a fraction of capital: `1.0` fully invested, `0` flat, `-1`
fully short, `0.5` half. Not a number of contracts and not a percentage.
Add an optional `benchmark` column and it is used instead of buy-and-hold.

Common aliases are matched case-insensitively (`Close Price`, `Weight`,
`Balance`, `NAV`, `daily_return`), and accents are folded away before
matching, so a Polish or German export whose headers carry diacritics is
read without renaming anything. Semicolon-separated files with comma
decimals — the usual European export — are read correctly.

### The one setting you must get right

`--position-convention` decides what a row's `position` means:

* `decided` **(default)** — the position was decided at that row's close and
  earns the **next** day's return.
* `held` — the position was already held **through** that row's return,
  because whoever produced the file already applied the lag.

Guess wrong towards `held` when the truth is `decided` and you have invented a
day of look-ahead, which flatters the result. The default is the other
direction on purpose.

---

## What it checks

| check | question | needs positions |
|---|---|---|
| The record as given | what does the file claim? | no |
| **Timing vs chance** | would the same rule at random times have done as well? | **yes** |
| **Fragility** | does it survive removing its best single trade? | **yes** |
| **Multiple testing** | how many variants were tried before this one? | no |
| Sharpe interval | how wide is the Sharpe really? | no |
| Out of sample | does the second half resemble the first? | no |
| Cost sensitivity | at what level of costs does the profit die? | yes |
| Versus holding | did it beat just buying the thing? | no |
| Concentration | how much comes from a handful of days? | no |
| Survivability | could a person have held it to the end? | no |

Three of these do the real work.

**Timing vs chance** keeps the price path exactly as it happened and moves the
*rule* instead, circularly shifting the position series a couple of thousand
times. Exposure, turnover, trading costs and the rule's own twitchiness are all
preserved; only the alignment with the market is destroyed. The obvious
alternative — shuffling the returns — is the wrong null and a flattering one,
because shuffling destroys volatility clustering and drift at once and almost
any rule beats it.

**Fragility** re-runs that test with each trade removed in turn. A permutation
p-value looks precise no matter how few trades produced it: the draws multiply
the *arrangements* of the evidence, not the evidence. A result that one
omission destroys is reported as fragile whatever its p-value says.

**Multiple testing** is the one that needs your honesty rather than your data.
`--variants-tried` is every parameter set, every lookback and every asset you
tried and discarded, not just the one you kept. Run 100 rules on pure noise and
about five come back significant at 0.05 — and it is those five that get
written up. Without the count, the check returns WEAK and the report can never
say SURVIVES, which is the intended behaviour: an undeclared search is not a
clean result.

---

## The verdict

- **SURVIVES** — every check that counts passed, nothing was skipped.
- **UNPROVEN** — nothing failed, but something was weak or could not be run.
  The usual outcome, and not an accusation: it means the record is consistent
  with a real edge *and* consistent with luck, and this file cannot separate
  them.
- **REJECTED** — at least one check failed.

There is no score out of ten. A score invites you to tune the strategy until
the number goes up, which is the exact behaviour — searching until something
passes — that the multiplicity check exists to punish.

---

## Options

| flag | meaning |
|---|---|
| `--file PATH` | the CSV to audit |
| `--example PATH` | write a synthetic track record and exit |
| `--variants-tried N` | how many rules/parameter sets were tried in total |
| `--cost-bps N` | one-way cost on turnover, positions form only |
| `--position-convention` | `decided` (default) or `held` |
| `--permutations N` | permutation draws (default 2000) |
| `--label TEXT` | name in the report header |
| `--out DIR` | where the HTML and JSON go |
| `--strict` | exit non-zero when the verdict is REJECTED |
| `--licence KEY` | licence key, overriding the file and the environment |

---

## Licensing

The tool runs entirely on your machine and never calls home, so licences are
offline: the key *is* the licence. It carries a payload (licensee, expiry,
seats) signed with an Ed25519 key that stays with the vendor, and the shipped
code carries only the public half.

Without a key the tool runs in evaluation mode: the most recent 400
observations, 200 permutation draws, and a watermark on the report. **Nothing
is falsified in evaluation mode** — the verdict logic is identical and a
weakened test is stated as weakened. A free tier that lies is worse than none.

Key placement, in order: `--licence`, then `SRC_LICENCE_KEY`, then a file named
`licence.key` beside the app or in `~/.config/strategy-reality-check/`.

Vendor side (see `src/audit/keytool.py`, not shipped to customers):

```bash
python -m audit.keytool generate --out ~/.keys/src-signing.key
# paste the printed public key into src/audit/license.py

export SRC_SIGNING_KEY_FILE=~/.keys/src-signing.key
python -m audit.keytool issue --licensee "ACME sp. z o.o." --months 12 --ref ORD-1042
```

Lose the private key and every future licence needs re-issuing under a new
public key. Leak it and anyone can mint licences for your product forever. It
does not belong in this repository, in a cloud drive, or in shell history —
which is why `issue` reads it from the environment or a file and never from an
argument.

---

## What this is not

It is an audit of a past track record. It measures how much of a result could
be chance, cost, or a handful of days.

It makes no forecast. It produces no signal. It does not tell anyone what to
buy, sell or hold, and no output of it is a personal recommendation. That
boundary is not decoration — see [SELLING.md](SELLING.md) before charging
anyone money for it.
