"""The agent: watches the price continuously, trades only where it was tested.

This is the part of the project that runs while nobody is looking, so the
first thing to be clear about is what it is allowed to do.

WATCHING IS CONTINUOUS. Every run fetches the live price, works out where it
stands against the rule's trigger, and writes a line saying what it sees and
why. That journal is the point: a record of what the agent believed at a
timestamp, written before the outcome existed.

TRADING IS NOT. The rule is a crossover of daily moving averages, and every
test in this project scored it on daily closes - the coverage walk, the
random-timing comparison, the cost model, all of it. An agent that acted on
an intraday tick would be running a DIFFERENT rule, one nobody has tested,
while wearing the credibility of the one that was. So the position changes
when a daily bar settles, and intraday observations are recorded rather than
acted on.

That is not timidity. Intraday execution is a perfectly good idea; it is just
a different strategy, and it needs intraday history, its own backtest and its
own calibration before it is worth a cent. Until then this agent says what it
sees and waits for the close.

The journal rolls: observations are trimmed to the last few thousand rows,
which is a few weeks of ticks. Nothing is lost by that, because the agent does
not trade - the positions and their trade log come from the settled daily run
in backtest/paper.py, which is rebuilt from the price history every morning and
published in full. The journal is the record of what was SEEN, not of what was
done.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ingest.http import FetchError, get_json

TICKER_URL = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"

JOURNAL_COLUMNS = [
    "timestamp", "price", "weight", "flip_level", "distance", "state", "note",
]
# A quarter-hourly agent writes about 35,000 lines a year, and the recent ones
# are what anybody reads. Nothing irreplaceable is lost when older rows go: the
# positions and trades are rebuilt from the price history every morning.
JOURNAL_LIMIT = 5_000

# How far from the trigger counts as "about to happen". Chosen to be wide
# enough that a reader gets warning and narrow enough to mean something.
ARMED_BAND = 0.02


@dataclass(frozen=True)
class Observation:
    """What the agent saw on one run."""

    timestamp: pd.Timestamp
    price: float
    weight: float
    flip_level: float | None
    state: str
    note: str

    @property
    def distance(self) -> float | None:
        """How far the price is from the level, as a fraction of the price."""
        if not self.flip_level or self.price <= 0:
            return None
        return (self.flip_level - self.price) / self.price

    def as_row(self) -> dict:
        return {
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "price": round(self.price, 2),
            "weight": round(self.weight, 4),
            "flip_level": None if self.flip_level is None else round(self.flip_level, 2),
            "distance": None if self.distance is None else round(self.distance, 5),
            "state": self.state,
            "note": self.note,
        }


def fetch_price(url: str = TICKER_URL) -> float:
    """The live quote. Raises FetchError, which the caller logs rather than dies on."""
    payload = get_json(url, retries=3)
    price = float(payload["price"])
    if price <= 0:
        raise FetchError(f"{url} -> nonsense price {price!r}")
    return price


def observe(
    price: float,
    *,
    weight: float,
    flip_level: float | None,
    flip_text: str = "",
    now: pd.Timestamp | None = None,
) -> Observation:
    """Turn a price into a sentence about where the rule stands.

    The states are deliberately about the RULE, not about the market:
    `holding` and `flat` are positions, `armed` means the price has crossed
    what the rule watches and the next settled close would act on it. Nothing
    here says the price is going anywhere.
    """
    stamp = pd.Timestamp(now or datetime.now(timezone.utc)).tz_localize(None)
    holding = weight > 0

    if not flip_level:
        state = "holding" if holding else "flat"
        return Observation(stamp, price, weight, None, state,
                           flip_text or "no price trigger for this rule")

    crossed = price <= flip_level if holding else price >= flip_level
    distance = abs(flip_level - price) / price
    if crossed:
        state = "armed"
        note = (f"price {price:,.0f} is past {flip_level:,.0f}; the next settled "
                f"close {'exits' if holding else 'enters'}")
    elif distance <= ARMED_BAND:
        state = "near"
        note = f"{distance:.2%} from {flip_level:,.0f}"
    else:
        state = "holding" if holding else "flat"
        note = f"{distance:.1%} from {flip_level:,.0f}"
    return Observation(stamp, price, weight, flip_level, state, note)


def load_journal(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=JOURNAL_COLUMNS)
    return pd.read_csv(path).reindex(columns=JOURNAL_COLUMNS)


def append_observation(path: str | Path, observation: Observation) -> pd.DataFrame:
    """Write one line, and keep the file from growing without limit."""
    path = Path(path)
    journal = load_journal(path)
    journal = pd.concat(
        [journal, pd.DataFrame([observation.as_row()])], ignore_index=True
    )
    if len(journal) > JOURNAL_LIMIT:
        journal = journal.iloc[-JOURNAL_LIMIT:].reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    journal.to_csv(path, index=False)
    return journal


def should_publish(
    journal: pd.DataFrame,
    *,
    last_published: pd.Timestamp | None,
    now: pd.Timestamp | None = None,
    heartbeat_hours: float = 3.0,
) -> tuple[bool, str]:
    """Whether this run is worth pushing to the site.

    A quarter-hourly push would be a hundred commits a day and a history
    nobody can read. So the site is refreshed when something changed that a
    reader would care about - the state crossed into or out of `armed` - and
    otherwise on a slow heartbeat so the page never looks abandoned.
    """
    if journal.empty:
        return False, "nothing recorded yet"

    states = journal["state"].tolist()
    if len(states) >= 2 and states[-1] != states[-2]:
        return True, f"state changed from {states[-2]} to {states[-1]}"

    stamp = pd.Timestamp(now or datetime.now(timezone.utc)).tz_localize(None)
    if last_published is None:
        return True, "nothing published yet"
    hours = (stamp - pd.Timestamp(last_published)).total_seconds() / 3600.0
    if hours >= heartbeat_hours:
        return True, f"{hours:.1f}h since the last publish"
    return False, f"{hours:.1f}h since the last publish, state unchanged"


def feed(journal: pd.DataFrame, *, rows: int = 40) -> dict:
    """What the site shows: the recent observations, newest first."""
    if journal.empty:
        return {"observations": [], "updated": None}
    recent = journal.tail(rows).iloc[::-1]
    return {
        "updated": str(journal.iloc[-1]["timestamp"]),
        "observations": recent.to_dict("records"),
    }
