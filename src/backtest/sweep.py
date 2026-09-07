"""Every value of the rebalance band, not the three that happened to be tried.

A parameter defended by a comparison against two neighbours is not defended. The
band was adopted because it scored the best Sharpe among {0%, 10%, 30%} - and a
grid that small cannot tell a real maximum from a bump, because it has no shape
to show. This sweeps the whole range at one-percentage-point steps so the shape
is on the page and the reader can see what kind of maximum it is.

What comes back is not a better band. It is the evidence that ranking bands on
this sample is not a thing worth doing:

- the maximum is nowhere near the adopted value, and moves if you look again;
- one-percentage-point steps move the Sharpe by more than a third of the whole
  grid's range, which is what a noisy surface looks like;
- and the best-scoring region sits directly beside the point where the band
  grows wider than the position ever moves, so the rule stops trading and
  inherits buy-and-hold's Sharpe by construction. Scoring highest by nearly
  switching yourself off is not a finding.

So the sweep exists to disqualify its own argmax, and the band stays where it
is - defended on turnover, which is a cost fact and does not move, rather than
on a score that does. Publishing only the winning row would have hidden all of
that behind one number.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.engine import BacktestConfig, run_backtest
from backtest.sizing import apply_rebalance_band

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


@dataclass(frozen=True)
class Sweep:
    """The sweep, and the statistics that say how much to trust its ranking."""

    table: pd.DataFrame          # indexed by band
    adopted: float

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
        """The band at which the rule stops trading, if the grid reaches it."""
        quiet = self.table.index[
            self.table["n_position_changes"] <= FROZEN_CHANGES]
        return float(quiet.min()) if len(quiet) else None

    def summary(self) -> str:
        adopted = float(self.sharpe.loc[self.adopted])
        best = float(self.sharpe.max())
        lines = [
            f"{len(self.table)} bands from {self.table.index.min():.0%} to "
            f"{self.table.index.max():.0%}: Sharpe {self.sharpe.min():.4f} to "
            f"{best:.4f}.",
            f"The adopted {self.adopted:.0%} scores {adopted:.4f} and ranks "
            f"{self.rank} of {len(self.table)}; the best is {self.best_band:.0%} "
            f"at {best:.4f}.",
            f"Neighbouring bands differ by {self.step:.4f} on average, against a "
            f"standard error of {self.standard_error:.4f} for any one of these "
            "Sharpes - so the ranking is mostly noise.",
        ]
        if self.frozen_from is not None:
            lines.append(
                f"From {self.frozen_from:.0%} the band is wider than the position "
                "ever moves: the rule stops trading and takes buy-and-hold's "
                "Sharpe by construction. The best-scoring bands sit next to it."
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
