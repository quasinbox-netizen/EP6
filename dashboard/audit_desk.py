"""Strategy Reality Check - the drag-and-drop front end.

Standalone on purpose. The research dashboard next door is one long page about
this project's own hypothesis; this is a single-purpose tool for somebody
else's CSV, and coupling the two would mean a customer paging past four years
of halving analysis to reach the thing they paid for.

Presentation only, like every other file in this directory. Every number on
screen comes from `audit.report.run_audit`, so the browser and the terminal
cannot disagree.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from audit.license import verify as verify_licence  # noqa: E402
from audit.loader import LoadError, load_csv  # noqa: E402
from audit.render import to_html  # noqa: E402
from audit.report import DEFAULT_PERMUTATIONS, REJECTED, SURVIVES, run_audit  # noqa: E402

st.set_page_config(page_title="Strategy Reality Check", layout="centered")

_BADGE = {
    "PASS": ("#0ca30c", "+", "PASSED"),
    "FAIL": ("#d03b3b", "x", "FAILED"),
    "WARN": ("#ec835a", "!", "WEAK"),
    "N/A": ("#77766f", "-", "NOT RUN"),
}
_VERDICT_COLOUR = {SURVIVES: "#0ca30c", REJECTED: "#d03b3b"}


def main() -> None:
    st.title("Strategy Reality Check")
    st.caption(
        "Upload a track record. This reports how much of it could be chance, "
        "cost, or a handful of days. It makes no forecast and gives no advice."
    )

    licence = verify_licence(root=ROOT)
    (st.success if licence.valid else st.info)(licence.banner())

    with st.sidebar:
        st.header("How it was produced")
        variants = st.number_input(
            "Variants tried in total", min_value=0, value=0, step=1,
            help="Every parameter set, lookback and asset you tried and discarded - "
                 "not just the one you kept. 0 means you would rather not say, and "
                 "the report will say so.",
        )
        cost_bps = st.number_input(
            "One-way cost, bps", min_value=0.0, max_value=500.0, value=0.0, step=5.0,
            help="Applied to turnover. Only possible when the file has a position "
                 "column; an equity curve already contains whatever costs it contains.",
        )
        convention = st.radio(
            "A row's position earns...",
            options=("decided", "held"),
            format_func=lambda value: {
                "decided": "the NEXT day's return (default)",
                "held": "that same row's return (already lagged)",
            }[value],
            help="Choosing 'held' when the file means 'decided' invents a day of "
                 "look-ahead and flatters the result.",
        )
        permutations = st.select_slider(
            "Permutation draws", options=(500, 1000, 2000, 5000), value=DEFAULT_PERMUTATIONS,
        )

    uploaded = st.file_uploader(
        "Track record (CSV)",
        type=("csv", "txt"),
        help="A date column, plus either close+position, or equity, or return.",
    )

    with st.expander("What the file needs to contain"):
        st.markdown(
            "| shape | columns | what you get |\n"
            "|---|---|---|\n"
            "| **positions** | `date`, `close`, `position` | everything, "
            "including the re-timing test |\n"
            "| **equity** | `date`, `equity` | metrics, bootstrap, out-of-sample |\n"
            "| **returns** | `date`, `return` | the same as equity |\n\n"
            "`position` is a fraction of capital: `1.0` fully invested, `0` flat, "
            "`-1` fully short. Not contracts, not percent. An optional `benchmark` "
            "column replaces buy-and-hold. Semicolon files with comma decimals are "
            "read correctly."
        )

    if uploaded is None:
        st.stop()

    # Streamlit hands over a buffer; the loader takes a path so that the same
    # code serves the CLI and this page without a second parsing route.
    with tempfile.TemporaryDirectory() as workspace:
        path = Path(workspace) / (uploaded.name or "upload.csv")
        path.write_bytes(uploaded.getvalue())
        try:
            data = load_csv(path, convention=convention, cost_rate=cost_bps / 10_000.0)
        except LoadError as exc:
            st.error(f"**This file cannot be audited.**\n\n{exc}")
            st.stop()

        with st.spinner("Re-timing the rule against chance..."):
            report = run_audit(
                data,
                licence=licence,
                variants_tried=int(variants) or None,
                cost_bps=cost_bps,
                permutations=int(permutations),
                label=uploaded.name or "upload",
            )

        colour = _VERDICT_COLOUR.get(report.verdict, "#ec835a")
        st.markdown(
            f"## <span style='color:{colour}'>{report.verdict}</span>",
            unsafe_allow_html=True,
        )
        st.write(report.summary)

        counts = report.meta["counts"]
        columns = st.columns(4)
        for column, (caption, key) in zip(
            columns,
            (("passed", "passed"), ("failed", "failed"),
             ("weak", "weak"), ("not run", "not_run")),
        ):
            column.metric(caption, counts[key])

        st.divider()
        for check in report.checks:
            colour, glyph, word = _BADGE.get(check.verdict, _BADGE["N/A"])
            label = f"{check.title}" + (f"  -  {word}" if check.blocking else "")
            with st.expander(label, expanded=check.verdict in ("FAIL", "WARN")):
                st.markdown(
                    f"<span style='color:{colour};font-weight:700'>{glyph} {word}</span>"
                    if check.blocking else "&nbsp;",
                    unsafe_allow_html=True,
                )
                st.caption(check.question)
                st.write(check.headline)
                if check.detail:
                    st.caption(check.detail)
                if check.numbers:
                    st.table({
                        "value": {
                            key.replace("_", " "): str(value)
                            for key, value in check.numbers.items()
                        }
                    })

        if report.meta.get("notes"):
            with st.expander("How the file was read"):
                for note in report.meta["notes"]:
                    st.markdown(f"- {note}")

        stem = Path(uploaded.name or "report").stem
        st.divider()
        st.download_button(
            "Download the report (HTML)",
            data=to_html(report, data).encode("utf-8"),
            file_name=f"{stem}-reality-check.html",
            mime="text/html",
        )
        st.caption(
            "This is an audit of a past track record, not advice. It makes no "
            "forecast, and nothing here is a recommendation to buy, sell or hold "
            "any instrument."
        )


main()
