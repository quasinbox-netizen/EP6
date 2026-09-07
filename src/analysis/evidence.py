"""One number for "how much has this project actually found", and its parts.

A summary score is the easiest thing in a research tool to abuse, so this one
is built to be hard to inflate:

* every component is a ratio the rest of the project already computes, and
  each is reported next to the score rather than folded away;
* every component is bounded at zero from below. A result worse than chance -
  which is where several of them sit - contributes nothing rather than a
  negative number that another component could cancel out;
* the weights are here, in the open, and they are equal. Tuning weights until
  the total looks better is the same move as tuning a hypothesis until its
  p-value does;
* a component with no data yet scores zero and says "no data", never a
  flattering default. An empty track record is not a perfect one.

The score is a reading of evidence strength, not a market call. Nothing in it
says which way a price goes, and a high value would mean "look again", never
"act". On this project's real numbers it sits near zero, which is the finding
rather than a bug.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Equal weights, listed rather than hidden in a dict comprehension so that a
# change to them is visible in a diff.
WEIGHTS = {
    "survivors": 1.0,
    "robustness": 1.0,
    "event_strength": 1.0,
    "replication": 1.0,
    "track_record": 1.0,
}

# The band labels. Deliberately flippant at the bottom, because that is where
# the honest reading of this project lives and a solemn label there would
# invite the reader to look for something else.
BANDS = (
    (10.0, "NOISE", "The skull is winning. Nothing here survives its own tests."),
    (30.0, "STILL NOISE", "Better dressed, but the tests still say no."),
    (50.0, "SOMETHING MOVED", "Probably the analyst. Worth a second look, not a position."),
    (70.0, "INTERESTING", "Several checks agree for once. Read the components before believing it."),
    (100.1, "TELL SOMEONE", "If this is real, it is worth writing up properly."),
)


@dataclass(frozen=True)
class Component:
    name: str
    value: float  # 0..1
    detail: str
    has_data: bool = True


@dataclass
class EvidenceScore:
    components: list = field(default_factory=list)

    @property
    def score(self) -> float:
        """0-100. Components without data count as zero, and say so."""
        total = sum(WEIGHTS.get(c.name, 1.0) for c in self.components)
        if total <= 0:
            return 0.0
        earned = sum(WEIGHTS.get(c.name, 1.0) * max(0.0, min(1.0, c.value))
                     for c in self.components)
        return round(100.0 * earned / total, 1)

    @property
    def label(self) -> str:
        return self._band()[1]

    @property
    def blurb(self) -> str:
        return self._band()[2]

    def _band(self):
        for ceiling, label, blurb in BANDS:
            if self.score < ceiling:
                return ceiling, label, blurb
        return BANDS[-1]

    def as_dict(self) -> dict:
        return {
            "score": self.score,
            "label": self.label,
            "blurb": self.blurb,
            "components": [
                {"name": c.name, "value": round(c.value, 3),
                 "detail": c.detail, "hasData": c.has_data}
                for c in self.components
            ],
        }


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return max(0.0, min(1.0, numerator / denominator))


def build(
    *,
    survivors: int = 0,
    hypotheses: int = 0,
    specifications_significant: float = 0.0,
    specifications_null_mean: float = 0.0,
    halving_p_value: float = 1.0,
    replicated: int = 0,
    replication_attempts: int = 0,
    settled_claims: int = 0,
    kept_claims: int = 0,
) -> EvidenceScore:
    """Assemble the score from numbers the rest of the project produced.

    Every argument is a count or a ratio that already exists somewhere in
    data/processed - nothing here recomputes a result, so the meter cannot
    disagree with the tab it summarises.
    """
    components = [
        Component(
            "survivors",
            _ratio(survivors, hypotheses),
            f"{survivors} of {hypotheses} hypotheses survive the correction",
            has_data=hypotheses > 0,
        ),
        # Below the null is not a small positive: random dates beating the real
        # ones is evidence against, and it earns zero rather than a fraction.
        Component(
            "robustness",
            _ratio(specifications_significant - specifications_null_mean,
                   max(specifications_null_mean, 1.0)),
            f"{specifications_significant:.0f} specifications significant against "
            f"{specifications_null_mean:.1f} for random dates",
            has_data=specifications_null_mean > 0,
        ),
        # Linear from p=0.05 down to p=0: a p of 0.30 is not "a bit of
        # evidence", it is none by the threshold this project uses everywhere.
        Component(
            "event_strength",
            _ratio(0.05 - halving_p_value, 0.05),
            f"halving event study p = {halving_p_value:.2f}",
        ),
        Component(
            "replication",
            _ratio(replicated, replication_attempts),
            f"{replicated} of {replication_attempts} replicate out of sample",
            has_data=replication_attempts > 0,
        ),
        Component(
            "track_record",
            _ratio(kept_claims, settled_claims),
            (f"{kept_claims} of {settled_claims} recorded claims kept their promise"
             if settled_claims else "no claim has matured yet"),
            has_data=settled_claims > 0,
        ),
    ]
    return EvidenceScore(components=components)
