"""Reading a track record that someone else produced.

Everything in this file is defensive. The input is a CSV a stranger exported
from TradingView, a broker statement, a spreadsheet or their own Python, and
the failure mode that matters is not a crash - it is a file that parses into
something subtly wrong and produces a confident verdict about nothing.

So: every column is matched by alias, every value is range-checked, and
anything ambiguous raises `LoadError` with the fix in the message rather than
guessing. A refusal to audit is a correct outcome.

THE ONE CONVENTION THE USER MUST GET RIGHT
------------------------------------------
A `position` column is ambiguous in a way no amount of parsing can resolve:
does row t hold the position DECIDED at the close of day t (applied to day
t+1's return), or the position HELD THROUGH day t's return (already lagged by
whoever produced the file)?

Guessing wrong in one direction invents a day of look-ahead and flatters the
result. Guessing wrong in the other adds a day of lag and penalises it. The
default is `decided`, the penalising one, because a tool that exists to say
"this might be luck" must not have its thumb on the optimistic side of the
scale. `--position-convention held` is there for people who know their export.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Below this, every downstream test is either impossible (edge_test refuses
# under 50) or so wide as to be meaningless. Refusing is kinder than a report
# full of NaN.
MIN_ROWS = 60

# A file bigger than this is not a daily track record, it is tick data or a
# mistake, and reading it would exhaust memory on the user's laptop.
MAX_BYTES = 64 * 1024 * 1024
MAX_ROWS = 500_000

# Positions outside this are almost always a units error - percentages written
# as 100 instead of 1.0, or contract counts. We refuse rather than audit a
# strategy that appears to run at 100x leverage.
MAX_ABS_POSITION = 10.0

CONVENTIONS = ("decided", "held")

_ALIASES: dict[str, tuple[str, ...]] = {
    "date": ("date", "time", "timestamp", "datetime", "day", "dt", "data"),
    "close": ("close", "price", "px", "close_price", "adj_close", "last", "kurs"),
    "position": ("position", "pos", "weight", "exposure", "signal", "target",
                 "position_size", "pozycja"),
    "equity": ("equity", "balance", "nav", "cumulative", "account_value",
               "portfolio_value", "kapital"),
    "return": ("return", "returns", "ret", "pnl_pct", "daily_return",
               "pct_change", "stopa_zwrotu"),
    "benchmark": ("benchmark", "bench", "bh", "buy_and_hold", "spx", "index"),
}


class LoadError(ValueError):
    """The file cannot be audited, with the reason stated in plain language."""


@dataclass(frozen=True)
class AuditInput:
    """A track record normalised into the shapes the checks need.

    `net_returns` is always present - it is the series every metric is built
    from. `positions` and `asset_returns` are present only when the file
    carried enough to reconstruct them, and their absence disables the
    re-timing test rather than faking it.
    """

    net_returns: pd.Series
    positions: pd.Series | None = None
    asset_returns: pd.Series | None = None
    benchmark_returns: pd.Series | None = None
    source: str = "input.csv"
    form: str = "returns"          # "positions" | "equity" | "returns"
    convention: str = "decided"
    notes: list[str] = field(default_factory=list)

    @property
    def can_retime(self) -> bool:
        """Is the permutation test available?

        It needs both a position series and the underlying asset's returns:
        the null re-times the RULE against the SAME price path, and neither
        half can be reconstructed from an equity curve alone.
        """
        return self.positions is not None and self.asset_returns is not None

    @property
    def equity(self) -> pd.Series:
        """Compounded equity, starting at exactly 1.0 one period before day one.

        The leading point is not decoration. Without it the series starts at
        ``1 + r0``, so the first day's return divides out of every total and
        the first day's loss cannot appear in a drawdown - which understates
        the worst drawdown of any strategy that opened badly.
        """
        returns = self.net_returns.fillna(0.0)
        index = returns.index
        step = (index[1] - index[0]) if len(index) > 1 else pd.Timedelta(days=1)
        curve = (1.0 + returns).cumprod()
        start = pd.Series([1.0], index=pd.DatetimeIndex([index[0] - step]))
        return pd.concat([start, curve])

    @property
    def span(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        index = self.net_returns.index
        return index[0], index[-1]


# European platforms export accented headers: "kapitał", "Schluß".  # non-english-ok
# Folding the accents away before matching lets the alias table stay plain
# ASCII instead of carrying one spelling variant per language.
def _fold(name: str) -> str:
    """Lower-case, underscore-joined and stripped of diacritics.

    A column the user can read is a column this tool can read: renaming headers
    by hand before an audit is the kind of friction that ends with the file
    never being audited at all.
    """
    decomposed = unicodedata.normalize("NFKD", str(name))
    ascii_only = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return ascii_only.strip().lower().replace(" ", "_").replace("-", "_")


def _normalise_columns(frame: pd.DataFrame) -> dict[str, str]:
    """Map canonical names to the actual column names present, by alias."""
    lookup: dict[str, str] = {}
    seen: dict[str, str] = {}
    for column in frame.columns:
        seen.setdefault(_fold(column), column)
    for canonical, aliases in _ALIASES.items():
        for alias in aliases:
            if alias in seen:
                lookup[canonical] = seen[alias]
                break
    return lookup


def _sniff_separator(header: str) -> str:
    """Pick the delimiter by counting it in the HEADER line, not the data.

    pandas' own sniffer reads the whole sample, and on a European export -
    ``date;close;position`` over rows like ``2021-01-01;100,23;1,0`` - it finds
    more commas than semicolons and splits on the decimal separator. Every
    column then holds text, and the file is rejected as unreadable. Header
    names do not contain decimal commas, so counting there is unambiguous.
    """
    counts = {sep: header.count(sep) for sep in (";", "\t", ",", "|")}
    best = max(counts, key=lambda sep: counts[sep])
    return best if counts[best] > 0 else ","


def _read_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise LoadError(f"no such file: {path}")
    if path.stat().st_size == 0:
        raise LoadError(f"{path.name} is empty")
    if path.stat().st_size > MAX_BYTES:
        raise LoadError(
            f"{path.name} is {path.stat().st_size / 1e6:.0f} MB. This audits a daily "
            "track record; a file this large is usually tick data exported by mistake."
        )
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            header = path.read_text(encoding=encoding).splitlines()[0]
            break
        except (UnicodeDecodeError, IndexError):
            continue
    else:
        raise LoadError(f"could not read {path.name} as text")

    separator = _sniff_separator(header)
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            frame = pd.read_csv(path, sep=separator, engine="python",
                                encoding=encoding, nrows=MAX_ROWS + 1)
            break
        except UnicodeDecodeError:
            continue
        except Exception as exc:  # pandas raises several types for a bad file
            raise LoadError(f"could not parse {path.name} as CSV: {exc}") from exc
    else:
        raise LoadError(f"could not decode {path.name}")
    if len(frame) > MAX_ROWS:
        raise LoadError(f"{path.name} has more than {MAX_ROWS:,} rows")
    return frame


def _numeric(frame: pd.DataFrame, column: str, label: str) -> pd.Series:
    """Coerce a column to float, tolerating thousands separators and % signs."""
    raw = frame[column]
    # NOT `dtype == object`: pandas 3 reads text columns as the "str" dtype, and
    # an object check silently skips the cleanup on exactly the European exports
    # it exists for.
    if not pd.api.types.is_numeric_dtype(raw):
        raw = (
            raw.astype(str)
            .str.strip()
            .str.replace("%", "", regex=False)
            .str.replace(" ", "", regex=False)
            .str.replace(" ", "", regex=False)
            # 1.234,56 -> 1234.56, but only when both separators are present,
            # so a plain 1.5 is left alone.
            .str.replace(r"^(-?\d{1,3}(?:\.\d{3})+),(\d+)$", r"\1\2", regex=True)
            .str.replace(r"^(-?\d+),(\d+)$", r"\1.\2", regex=True)
            .str.replace(",", "", regex=False)
        )
    values = pd.to_numeric(raw, errors="coerce")
    if values.notna().sum() == 0:
        raise LoadError(f"column {column!r} ({label}) holds no numbers")
    return values.astype(float)


def _parse_dates(frame: pd.DataFrame, column: str) -> pd.DatetimeIndex:
    parsed = pd.to_datetime(frame[column], errors="coerce", utc=False, format="mixed")
    bad = int(parsed.isna().sum())
    if bad:
        first = frame.loc[parsed.isna(), column].astype(str).head(3).tolist()
        raise LoadError(
            f"{bad} row(s) have an unreadable date in column {column!r}, "
            f"for example {first}. Use ISO dates (2024-03-01)."
        )
    index = pd.DatetimeIndex(parsed)
    if index.tz is not None:
        index = index.tz_convert(None)
    return index.normalize()


def load_csv(
    path: str | Path,
    *,
    convention: str = "decided",
    cost_rate: float | None = None,
) -> AuditInput:
    """Read a track record and normalise it, or refuse with a reason.

    Three accepted shapes, in order of how much can be tested afterwards:

    1. ``date, close, position`` - the full audit. Returns are reconstructed
       from the price path, so the rule can be re-timed against it.
    2. ``date, equity`` - metrics, bootstrap and out-of-sample only.
    3. ``date, return`` - the same as (2); returns are taken as given.

    ``cost_rate`` applies a one-way cost to turnover in shape (1). It is
    ignored in (2) and (3), where costs are whatever the producer already
    subtracted - and a note says so, because an equity curve that silently
    excludes costs is the single most common way a track record lies.
    """
    if convention not in CONVENTIONS:
        raise LoadError(f"unknown position convention {convention!r}; use one of {CONVENTIONS}")

    path = Path(path)
    frame = _read_frame(path)
    columns = _normalise_columns(frame)
    notes: list[str] = []

    if "date" not in columns:
        raise LoadError(
            "no date column found. One column must be named date, time, timestamp "
            f"or datetime. Columns seen: {list(frame.columns)[:12]}"
        )

    frame = frame.copy()
    index = _parse_dates(frame, columns["date"])
    frame.index = index
    frame = frame.sort_index()

    duplicates = int(frame.index.duplicated().sum())
    if duplicates:
        frame = frame[~frame.index.duplicated(keep="last")]
        notes.append(
            f"{duplicates} duplicate date(s) dropped, keeping the last row of each."
        )

    if len(frame) < MIN_ROWS:
        raise LoadError(
            f"{len(frame)} usable rows. At least {MIN_ROWS} are needed before any of "
            "these tests says anything; below that the intervals are wider than the "
            "result."
        )

    benchmark_returns = None
    if "benchmark" in columns:
        benchmark_returns = _series_to_returns(
            _numeric(frame, columns["benchmark"], "benchmark")
        )

    has_position = "position" in columns and "close" in columns
    if has_position:
        return _from_positions(
            frame, columns, convention=convention, cost_rate=cost_rate or 0.0,
            benchmark_returns=benchmark_returns, source=path.name, notes=notes,
        )

    if "position" in columns and "close" not in columns:
        notes.append(
            "a position column was found but no price column, so the rule cannot be "
            "re-timed against the market it traded. Add a `close` column to unlock "
            "the strongest test in this report."
        )

    if "equity" in columns:
        return _from_equity(
            frame, columns, benchmark_returns=benchmark_returns,
            source=path.name, notes=notes, convention=convention,
        )

    if "return" in columns:
        return _from_returns(
            frame, columns, benchmark_returns=benchmark_returns,
            source=path.name, notes=notes, convention=convention,
        )

    raise LoadError(
        "nothing auditable found. Provide one of: `close` + `position`, or `equity`, "
        f"or `return`. Columns seen: {list(frame.columns)[:12]}"
    )


def _series_to_returns(values: pd.Series) -> pd.Series:
    """An equity-like or return-like column, coerced to simple returns.

    The heuristic is the only one in this file, and it is safe in the sense
    that it cannot silently mislabel: a column that is always positive and
    ends far from where it started is a level, not a return series.
    """
    clean = values.dropna()
    if clean.empty:
        return pd.Series(dtype=float, index=values.index)
    looks_like_level = bool((clean > 0).all() and clean.max() / max(clean.min(), 1e-12) > 1.5)
    if looks_like_level:
        return values.pct_change()
    return values


def _validate_returns(returns: pd.Series, label: str) -> pd.Series:
    out = returns.replace([np.inf, -np.inf], np.nan)
    finite = out.dropna()
    if len(finite) < MIN_ROWS - 1:
        raise LoadError(
            f"{label} has only {len(finite)} usable values after dropping blanks."
        )
    if (finite <= -1.0).any():
        n = int((finite <= -1.0).sum())
        raise LoadError(
            f"{label} contains {n} value(s) of -100% or worse. Either the account "
            "was wiped out, or the column is in percent and needs dividing by 100."
        )
    if finite.abs().max() > 5.0:
        raise LoadError(
            f"{label} reaches {finite.abs().max():.1f} (i.e. {finite.abs().max():.0%}) "
            "in a single period. If the column is in percent, divide it by 100."
        )
    return out


def _from_positions(
    frame, columns, *, convention, cost_rate, benchmark_returns, source, notes
) -> AuditInput:
    close = _numeric(frame, columns["close"], "price")
    position = _numeric(frame, columns["position"], "position")

    if (close.dropna() <= 0).any():
        raise LoadError("the price column contains zero or negative values.")
    gaps = int(close.isna().sum())
    if gaps:
        close = close.ffill()
        notes.append(f"{gaps} missing price(s) carried forward from the previous day.")
        if close.isna().any():
            raise LoadError("the price column starts with blanks; the first row needs a price.")

    if position.isna().any():
        blanks = int(position.isna().sum())
        position = position.fillna(0.0)
        notes.append(f"{blanks} blank position(s) read as flat (0).")
    if position.abs().max() > MAX_ABS_POSITION:
        raise LoadError(
            f"the position column reaches {position.abs().max():.1f}. Positions are "
            "expected as a fraction of capital (1.0 = fully invested). A value of 100 "
            "usually means the column is in percent."
        )

    # SIMPLE returns, not log. The report prints a Sharpe in the metric table and
    # another inside the permutation test, and if one is computed on log returns
    # and the other on simple ones they differ by about sigma^2/2 - which on a
    # 55%-volatility series is a quarter of a Sharpe point. Two different numbers
    # for the same quantity in one report is a defect, whatever the footnote says.
    asset_returns = close.pct_change().fillna(0.0)

    engine_positions = position
    if convention == "held":
        # The file already lagged the rule, so hand the engine the NEXT row's
        # holding: strategy_returns applies positions[:-1] to returns[1:], and
        # lagging twice would delay every entry by a day that never happened.
        engine_positions = position.shift(-1).fillna(0.0)
        notes.append(
            "positions read as already held through each row's return "
            "(--position-convention held)."
        )
    else:
        notes.append(
            "positions read as decided at each row's close and applied to the next "
            "day's return (the default, and the conservative reading)."
        )

    from backtest.edge import strategy_returns  # local: keeps import cost off startup

    realised = strategy_returns(
        engine_positions.to_numpy(), asset_returns.to_numpy(), cost_rate
    )
    net = pd.Series(realised, index=frame.index[1:], name="net_return")
    net = _validate_returns(net, "the reconstructed return series")

    if cost_rate:
        notes.append(
            f"costs applied by this tool: {cost_rate * 10_000:.0f} bps of turnover, "
            "one way."
        )
    else:
        notes.append(
            "no trading cost was applied (--cost-bps 0). Whatever this report shows "
            "is before fees, spread and slippage."
        )

    return AuditInput(
        net_returns=net,
        positions=engine_positions,
        asset_returns=asset_returns,
        benchmark_returns=_align(benchmark_returns, net),
        source=source,
        form="positions",
        convention=convention,
        notes=notes,
    )


def _from_equity(frame, columns, *, benchmark_returns, source, notes, convention) -> AuditInput:
    equity = _numeric(frame, columns["equity"], "equity")
    if (equity.dropna() <= 0).any():
        raise LoadError("the equity column contains zero or negative values.")
    equity = equity.ffill()
    net = _validate_returns(equity.pct_change().dropna(), "the equity curve")
    notes.append(
        "read as an equity curve. Costs are whatever the file already has in it - "
        "this tool cannot tell whether fees were subtracted, and a curve that "
        "excludes them is the most common way a track record flatters itself."
    )
    notes.append(
        "no position column, so the re-timing test is unavailable: the strongest "
        "check here needs to know WHEN the rule was in the market."
    )
    return AuditInput(
        net_returns=net, benchmark_returns=_align(benchmark_returns, net),
        source=source, form="equity", convention=convention, notes=notes,
    )


def _from_returns(frame, columns, *, benchmark_returns, source, notes, convention) -> AuditInput:
    values = _numeric(frame, columns["return"], "return")
    if values.dropna().abs().max() > 1.5:
        values = values / 100.0
        notes.append("the return column looked like percent and was divided by 100.")
    net = _validate_returns(values.dropna(), "the return column")
    notes.append(
        "read as a return series. Costs are whatever the file already has in it."
    )
    notes.append(
        "no position column, so the re-timing test is unavailable: the strongest "
        "check here needs to know WHEN the rule was in the market."
    )
    return AuditInput(
        net_returns=net, benchmark_returns=_align(benchmark_returns, net),
        source=source, form="returns", convention=convention, notes=notes,
    )


def _align(benchmark: pd.Series | None, net: pd.Series) -> pd.Series | None:
    if benchmark is None:
        return None
    aligned = benchmark.reindex(net.index)
    if aligned.notna().sum() < MIN_ROWS - 1:
        return None
    return aligned
