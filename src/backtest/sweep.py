"""Every value of the two constants that size the position, not the few tried.

A parameter defended by a comparison against two neighbours is not defended. The
band was adopted for scoring the best Sharpe among {0%, 10%, 30%}, and the target
was never swept at all - and a grid that small cannot tell a real maximum from a
bump, because it has no shape to show. These sweeps put the shape on the page.

Neither of them comes back with a better value. They come back with the evidence
that ranking these constants on this sample is not a thing worth doing, and the
two fail in different ways, which is the reason they are both here.

THE BAND is a threshold: it decides when to trade. Its surface is noise. The
maximum is nowhere near the adopted value, one-percentage-point steps move the
Sharpe by more than a fifth of the whole grid's range, and the best-scoring
region sits directly beside the point where the band grows wider than the
position ever moves - so the rule stops trading and inherits buy-and-hold's
Sharpe by construction. Scoring highest by nearly switching yourself off is not
a finding.

THE TARGET is not a threshold at all. It is a dial on how much risk to carry,
and it moves the average position from a few percent to nearly everything. That
makes the Sharpe column close to useless for choosing one - Sharpe is built to
be indifferent to how large a position is - while drawdown, which nobody
published, runs from -18% to -83% across the same grid. Reading a risk dial off
the one statistic designed not to see risk is the mistake this sweep exists to
make visible. Both of ITS ends degenerate too: high targets pin the position at
the no-borrowing cap until the rule simply is buy-and-hold, and low ones shrink
the position under a band that does not shrink with it, until it barely trades.

So each sweep exists to disqualify its own argmax, and neither constant moved.
They are defended on the grounds that do not move under a second look - turnover
for the band, which is a cost, and the risk level for the target, which is a
choice - rather than on a score that does. Publishing only the winning row would
have hidden all of that behind one number.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.engine import BacktestConfig, run_backtest
from backtest.sizing import apply_rebalance_band, volatility_target_position

# One percentage point from nothing to sixty. The top of the range is past the
# point where the rule stops trading, deliberately: that plateau is part of the
# picture, and a grid that stopped before it would hide the reason the highest
# scores are where they are.
DEFAULT_BANDS: tuple[float, ...] = tuple(
    round(float(band), 4) for band in np.arange(0.0, 0.605, 0.01)
)

# A band is "frozen" when the weight never changes again after the position is
# first entered. Counting changes rather than thresholding turnover matters:
# turnover is a rate, so any fixed cut-off means something different on a
# three-year window than on an eleven-year one, and the claim being made here -
# that the rule has stopped trading - is exact and deserves an exact test.
FROZEN_CHANGES = 1

# Ten percent to two hundred, in five-point steps. The range runs well past any
# defensible target on purpose: both ends of it are degenerate, and a grid that
# stopped at the sensible values would show a curve without showing that the
# curve stops meaning anything on either side of them.
DEFAULT_TARGETS: tuple[float, ...] = tuple(
    round(float(target), 4) for target in np.arange(0.10, 2.005, 0.05)
)

# The share of days at the no-borrowing cap past which the rule has stopped
# sizing and is simply holding the asset.
PINNED_SHARE = 0.99


@dataclass(frozen=True)
class Sweep:
    """The sweep, and the statistics that say how much to trust its ranking."""

    table: pd.DataFrame          # indexed by the parameter that was varied
    adopted: float
    parameter: str = "band"

    @property
    def sharpe(self) -> pd.Series:
        return self.table["sharpe"]

    @property
    def best_band(self) -> float:
        return float(self.sharpe.idxmax())

    @property
    def rank(self) -> int:
        """Where the adopted band places, 1 being the best."""
        return int((self.sharpe > self.sharpe.loc[self.adopted]).sum()) + 1

    @property
    def step(self) -> float:
        """Mean absolute change in Sharpe between neighbouring bands.

        The honest measure of how flat this surface is. A grid whose ranking
        meant anything would move smoothly; this one jumps.
        """
        return float(self.sharpe.diff().abs().mean())

    @property
    def standard_error(self) -> float:
        """Standard error of a single Sharpe estimate, Lo (2002), iid case.

        `sqrt((1 + S^2/2) / n)`. It assumes returns are independent, which
        these are not, so it is the OPTIMISTIC figure - real uncertainty is
        wider. That direction matters: the argument here is that differences
        across the grid are small relative to the noise, and using the
        optimistic noise makes that argument harder to make, not easier.
        """
        adopted = float(self.sharpe.loc[self.adopted])
        days = int(self.table["days"].loc[self.adopted])
        return float(np.sqrt((1.0 + adopted ** 2 / 2.0) / days))

    @property
    def frozen_from(self) -> float | None:
        """The value from which the weight never changes again after entry.

        On a band sweep this fires at the wide end, where the band outgrows the
        position's own movement.

        On a TARGET sweep it usually does not fire, and the reason is worth
        knowing rather than papering over: the band is an absolute width, so a
        small target shrinks the position underneath a band that stays 0.30
        wide, and trading nearly stops without stopping. At a 10% target the
        position averages 0.08 and changes three times in eleven years. That is
        not frozen by this test's definition, and it is not a working rule
        either - which is why the sweep publishes the position column beside
        the score, instead of relying on one threshold to catch every way a
        parameter can leave the range where it does anything.
        """
        quiet = self.table.index[
            self.table["n_position_changes"] <= FROZEN_CHANGES]
        return float(quiet.min()) if len(quiet) else None

    @property
    def pinned_from(self) -> float | None:
        """The target from which the position sits at the cap nearly always.

        Past it the rule holds the asset outright on all but a handful of days,
        so it is buy-and-hold under another name - and its drawdown converges
        on buy-and-hold's, which is what gives that away.
        """
        if "at_the_cap" not in self.table.columns:
            return None
        pinned = self.table.index[self.table["at_the_cap"] >= PINNED_SHARE]
        return float(pinned.min()) if len(pinned) else None

    @property
    def at_the_cap(self) -> float | None:
        """Share of days the adopted value spends pinned to the cap."""
        if "at_the_cap" not in self.table.columns:
            return None
        return float(self.table["at_the_cap"].loc[self.adopted])

    def summary(self) -> str:
        adopted = float(self.sharpe.loc[self.adopted])
        best = float(self.sharpe.max())
        plural = f"{self.parameter}s"
        lines = [
            f"{len(self.table)} {plural} from {self.table.index.min():.0%} to "
            f"{self.table.index.max():.0%}: Sharpe {self.sharpe.min():.4f} to "
            f"{best:.4f}.",
            f"The adopted {self.adopted:.0%} scores {adopted:.4f} and ranks "
            f"{self.rank} of {len(self.table)}; the best is {self.best_band:.0%} "
            f"at {best:.4f}.",
            f"Neighbouring {plural} differ by {self.step:.4f} on average, against "
            f"a standard error of {self.standard_error:.4f} for any one of these "
            "Sharpes - so the ranking is mostly noise.",
        ]
        if self.at_the_cap is not None:
            lines.append(
                f"At the adopted {self.adopted:.0%} the position sits at the "
                f"no-borrowing cap on {self.at_the_cap:.1%} of days, so on those "
                "days the rule is not sizing anything - it is holding the asset."
            )
        if self.frozen_from is not None:
            lines.append(
                f"From {self.frozen_from:.0%} the weight never changes after "
                "entry: the rule stops trading and takes buy-and-hold's Sharpe "
                "by construction."
            )
        if self.pinned_from is not None:
            lines.append(
                f"From {self.pinned_from:.0%} the position is at the cap on "
                f"{PINNED_SHARE:.0%} of days or more: the rule IS buy-and-hold, "
                "and its drawdown converges on buy-and-hold's."
            )
        return "\n".join(lines)


def band_sweep(
    close: pd.Series,
    target: pd.Series,
    settings: BacktestConfig,
    *,
    adopted: float,
    bands: tuple[float, ...] = DEFAULT_BANDS,
) -> Sweep:
    """Run the sizing rule at every band and collect what each one earned.

    `target` is the unbanded position, so every row differs from every other in
    exactly one number and nothing else. The adopted band is included whether or
    not it lands on the grid - a sweep that could not price the value actually
    in use would be answering a different question.
    """
    grid = sorted({round(float(band), 4) for band in bands} | {round(float(adopted), 4)})
    rows = []
    for band in grid:
        run = run_backtest(close, apply_rebalance_band(target, band), settings,
                           name=f"band {band:.0%}")
        rows.append({"band": band, **run.metrics})
    return Sweep(table=pd.DataFrame(rows).set_index("band"),
                 adopted=round(float(adopted), 4))


def target_sweep(
    close: pd.Series,
    volatility: pd.Series,
    settings: BacktestConfig,
    *,
    adopted: float,
    band: float,
    targets: tuple[float, ...] = DEFAULT_TARGETS,
) -> Sweep:
    """Run the sizing rule at every volatility target, with the band held fixed.

    The target is not the same kind of parameter as the band, and the sweep has
    to be read differently because of it. The band is a threshold: it decides
    when to trade. The target is a dial on how much risk to carry, and it moves
    the average position from a few percent to nearly everything.

    That makes the Sharpe column close to useless for choosing one, which is the
    point of running this. Sharpe is built to be indifferent to how much of a
    position you hold; drawdown is not. So the table records `at_the_cap`
    alongside the metrics - the share of days the position is pinned at the
    no-borrowing limit - because that is the column which says what the target
    is actually doing, and it is the one nobody published.
    """
    grid = sorted({round(float(value), 4) for value in targets}
                  | {round(float(adopted), 4)})
    rows = []
    for value in grid:
        raw = volatility_target_position(
            volatility, target_annual_volatility=value
        ).reindex(close.index).ffill()
        run = run_backtest(close, apply_rebalance_band(raw, band), settings,
                           name=f"target {value:.0%}")
        rows.append({
            "target": value,
            # Pinned rather than merely large: at 1.0 the formula is asking for
            # more than it is allowed and the position stops responding to the
            # forecast at all.
            "at_the_cap": float((raw >= 0.999).mean()),
            **run.metrics,
        })
    return Sweep(table=pd.DataFrame(rows).set_index("target"),
                 adopted=round(float(adopted), 4), parameter="target")
