"""Dashboard Streamlit - wylacznie warstwa prezentacji.

Zasada: zero logiki badawczej w tym pliku. Kazda liczba pochodzi z modulu
w src/ i przechodzi przez te same funkcje, co CLI. Jesli chcesz zmienic
sposob liczenia czegokolwiek, zrob to w src/ - inaczej wykres i terminal
zaczna pokazywac rozne rzeczy, a wierzyc bedziesz temu ladniejszemu.

Uruchomienie:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from config import load_config  # noqa: E402
from features.halving import CONFIRMED_HALVINGS  # noqa: E402
from pipeline import (  # noqa: E402
    category_event_studies,
    control_comparison,
    halving_event_study,
    load_lab_data,
    out_of_sample_check,
    run_strategies,
    scan_hypotheses,
)
from validation.multiple_testing import summarize  # noqa: E402

st.set_page_config(page_title="BTC Cycle Lab", layout="wide")

COLORS = {
    "price": "#e8a33d",
    "car": "#4c8bf5",
    "band": "rgba(76, 139, 245, 0.18)",
    "halving": "rgba(232, 163, 61, 0.14)",
    "zero": "#8a8a8a",
}


@st.cache_data(show_spinner="Wczytuje dane z lokalnej bazy...")
def cached_data():
    return load_lab_data()


@st.cache_data(show_spinner="Licze event study...")
def cached_halving_study(post: int):
    return halving_event_study(cached_data(), post=post)


@st.cache_data(show_spinner="Licze event study dla kategorii...")
def cached_category_studies(post: int):
    return category_event_studies(cached_data(), post=post)


@st.cache_data(show_spinner="Porownuje z grupa kontrolna...")
def cached_control(post: int):
    return control_comparison(cached_data(), post=post)


@st.cache_data(show_spinner="Skanuje hipotezy...")
def cached_scan():
    return scan_hypotheses(cached_data())


@st.cache_data(show_spinner="Sprawdzam replikacje poza proba...")
def cached_out_of_sample():
    return out_of_sample_check(cached_data())


@st.cache_data(show_spinner="Uruchamiam backtesty...")
def cached_backtests():
    table, results = run_strategies(cached_data())
    curves = pd.DataFrame({r.name: r.equity for r in results})
    return table, curves


def price_chart(data, halving_window_days: int, show_events: bool) -> go.Figure:
    frame = data.features
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["close"],
            name="BTC/USD",
            line=dict(color=COLORS["price"], width=1.2),
        )
    )
    for halving in CONFIRMED_HALVINGS:
        end = halving + pd.Timedelta(days=halving_window_days)
        figure.add_vrect(x0=halving, x1=end, fillcolor=COLORS["halving"], line_width=0)
        figure.add_vline(x=halving, line=dict(color=COLORS["price"], width=1, dash="dot"))

    if show_events and not data.events.empty:
        non_halving = data.events[data.events["category"] != "halving"]
        prices_by_day = frame["close"]
        for _, event in non_halving.iterrows():
            day = pd.Timestamp(event["available_from"]).normalize()
            if day not in prices_by_day.index:
                continue
            figure.add_trace(
                go.Scatter(
                    x=[day],
                    y=[prices_by_day.loc[day]],
                    mode="markers",
                    marker=dict(size=8, symbol="diamond"),
                    name=event["name"],
                    hovertext=f"{event['category']}: {event['description']}",
                    showlegend=False,
                )
            )

    figure.update_layout(
        yaxis_type="log",
        height=460,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title="cena (skala log)",
        hovermode="x unified",
    )
    return figure


def car_chart(result, title: str) -> go.Figure:
    table = result.table.dropna(subset=["car"])
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=table.index, y=table["car_ci_high"], line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=table.index, y=table["car_ci_low"], fill="tonexty",
            fillcolor=COLORS["band"], line=dict(width=0),
            name="95% przedzial ufnosci",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=table.index, y=table["car"], name="sredni CAR",
            line=dict(color=COLORS["car"], width=2),
        )
    )
    figure.add_hline(y=0, line=dict(color=COLORS["zero"], dash="dash", width=1))
    figure.update_layout(
        title=title,
        height=420,
        xaxis_title="dni od zdarzenia",
        yaxis_title="skumulowany zwrot nadzwyczajny",
        margin=dict(l=10, r=10, t=50, b=10),
        yaxis_tickformat=".0%",
    )
    return figure


def equity_chart(curves: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    for column in curves.columns:
        figure.add_trace(
            go.Scatter(
                x=curves.index, y=curves[column], name=column,
                line=dict(width=2.5 if column == "kup i trzymaj" else 1.4),
            )
        )
    figure.update_layout(
        yaxis_type="log", height=440, yaxis_title="kapital (skala log)",
        margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
    )
    return figure


def main() -> None:
    config = load_config()
    st.title("BTC Cycle Lab")
    st.caption(
        "Narzedzie do badania, czy cykl halvingowy i zdarzenia makro cokolwiek "
        "wyjasniaja w cenie BTC. Kazdy wynik jest podany z przedzialem ufnosci "
        "i liczba obserwacji - bez tego nie jest wynikiem."
    )

    data = cached_data()
    if data.is_empty:
        st.error(
            "Baza jest pusta. Uruchom najpierw pobieranie danych:\n\n"
            "`python -m cli ingest --what all` (z katalogu repo, PYTHONPATH=src)"
        )
        st.stop()

    with st.sidebar:
        st.header("Ustawienia")
        halving_window_days = st.slider("Okno halvingowe na wykresie (dni)", 30, 730, 365, 5)
        study_post = st.slider("Horyzont event study (dni po zdarzeniu)", 30, 730, 365, 5)
        show_events = st.checkbox("Pokaz zdarzenia na wykresie", value=True)
        st.divider()
        st.caption(
            f"Dane: {data.features.index.min().date()} - {data.features.index.max().date()} "
            f"({len(data.features)} dni)\n\n"
            f"Zszycie: {', '.join(config['price']['stitch_priority'])}\n\n"
            f"Koszty: {config['backtest']['fee_bps']} bps + "
            f"{config['backtest']['slippage_bps']} bps poslizgu"
        )

    tab_price, tab_study, tab_control, tab_validation, tab_backtest = st.tabs(
        ["Cena i cykle", "Event study", "Grupa kontrolna", "Walidacja", "Backtest"]
    )

    with tab_price:
        st.plotly_chart(
            price_chart(data, halving_window_days, show_events), use_container_width=True
        )
        columns = st.columns(4)
        columns[0].metric("Dni w probie", f"{len(data.features):,}")
        columns[1].metric("Halvingi w probie", len(CONFIRMED_HALVINGS))
        columns[2].metric("Zdarzenia w rejestrze", len(data.events))
        columns[3].metric("Serie makro", data.macro["series"].nunique() if not data.macro.empty else 0)
        with st.expander("Rejestr zdarzen"):
            st.dataframe(
                data.events.loc[:, ["date", "category", "name", "description"]],
                use_container_width=True, hide_index=True,
            )

    with tab_study:
        study = cached_halving_study(study_post)
        st.plotly_chart(
            car_chart(study, f"Halvingi (n={study.n_events})"), use_container_width=True
        )
        st.info(study.summary())
        if study.n_events < 10:
            st.warning(
                f"Przedzial ufnosci jest szeroki, bo probka liczy {study.n_events} zdarzen. "
                "To nie jest wada metody - to jest cala dostepna informacja."
            )

        st.subheader("Kategorie zdarzen")
        studies = cached_category_studies(min(study_post, 180))
        if studies:
            rows = [
                {"kategoria": name, "n": result.n_events, **result.car_summary}
                for name, result in studies.items()
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            choice = st.selectbox("Wykres dla kategorii", sorted(studies))
            st.plotly_chart(
                car_chart(studies[choice], f"{choice} (n={studies[choice].n_events})"),
                use_container_width=True,
            )
        st.caption(
            "P-value w tej tabeli sa SUROWE. Do wnioskowania sluzy zakladka Walidacja."
        )

    with tab_control:
        st.caption(
            "Halving dotyczy wylacznie bitcoina, wiec to, co robily w tym samym "
            "oknie aktywa bez halvingu, jest testem placebo. Roznica liczona jest "
            "parami po zdarzeniach, zeby wspolne warunki makro sie skrocily."
        )
        report = cached_control(study_post)
        if "error" in report:
            st.warning(report["error"])
        else:
            for name, comparison in report["comparisons"].items():
                st.subheader(f"BTC vs {name}")
                if comparison.table.empty:
                    st.info("brak wspolnych zdarzen z pelnym oknem")
                    continue
                st.dataframe(comparison.table, use_container_width=True)
                verdict = comparison.verdict()
                if "NIE jest wspolny" in verdict:
                    st.success(verdict)
                else:
                    st.warning(verdict)
                placebo = report["placebos"][name]
                if placebo.car_summary:
                    st.caption(
                        f"Placebo - {name} potraktowany jak BTC: "
                        f"CAR({placebo.car_summary['offset']}d) = "
                        f"{placebo.car_summary['car']:+.1%}, "
                        f"p = {placebo.car_summary['p_value']:.3f}"
                    )
                with st.expander(f"CAR pojedynczych zdarzen ({name})"):
                    st.dataframe(comparison.per_event, use_container_width=True)

    with tab_validation:
        scan = cached_scan()
        if scan.empty:
            st.info("Brak hipotez do sprawdzenia.")
        else:
            st.subheader("Skan okien z korekta na wielokrotne testowanie")
            st.info(summarize(scan))
            st.dataframe(
                scan.loc[
                    :, ["hypothesis", "n_in", "mean_in", "mean_out", "difference",
                        "p_value", "p_adjusted", "significant_adjusted"]
                ],
                use_container_width=True, hide_index=True,
            )
        out_of_sample = cached_out_of_sample()
        if not out_of_sample.empty:
            st.subheader("Replikacja poza proba (podzial po cyklach)")
            st.dataframe(
                out_of_sample.loc[
                    :, ["hypothesis", "train_effect", "test_effect", "same_sign",
                        "significant_out_of_sample", "effect_retained", "replicated"]
                ],
                use_container_width=True, hide_index=True,
            )
            survivors = out_of_sample[out_of_sample["replicated"]]["hypothesis"].tolist()
            if survivors:
                st.success(f"Przetrwalo poza proba: {', '.join(survivors)}")
            else:
                st.warning(
                    "Zadna hipoteza nie powtorzyla sie poza proba treningowa. "
                    "To najczestszy i najbardziej pouczajacy wynik w tym projekcie."
                )

    with tab_backtest:
        table, curves = cached_backtests()
        if table.empty:
            st.info("Brak wynikow backtestu.")
        else:
            st.plotly_chart(equity_chart(curves), use_container_width=True)
            st.dataframe(
                table.loc[
                    :, ["total_return", "cagr", "sharpe", "sortino", "max_drawdown",
                        "calmar", "win_rate", "time_in_market", "turnover_annual", "total_cost"]
                ],
                use_container_width=True,
            )
            baseline = table.loc["kup i trzymaj"]
            better = [
                name for name in table.index
                if name != "kup i trzymaj" and table.loc[name, "sharpe"] > baseline["sharpe"]
            ]
            st.caption(
                "Porownanie zawsze wzgledem kup-i-trzymaj. Lepszy Sharpe: "
                + (", ".join(better) if better else "zadna strategia")
            )


if __name__ == "__main__":
    main()
