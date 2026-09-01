# Wyniki przebiegu — 2026-09-01

Snapshot wygenerowany komendą `.\btc.cmd all --post 365`.
Odtworzenie: `.\btc.cmd ingest --what all` a potem `.\btc.cmd all`.

Próba: **5494 dni**, 2011-08-18 → 2026-09-01. Szereg zszywany — Bitstamp 2191 dni
do 2017-08-16, dalej Binance 3303 dni. Mediana rozbieżności na zakładce 0,06%,
maksimum 9,0% (2017-12-23, realny rozjazd giełd w szczycie manii). Zero
brakujących dni, zero duplikatów.

---

## Halvingi

CAR liczony po zdarzeniach, przedział ufności z rozkładu t o n−1 stopniach swobody.

| horyzont | CAR | 95% CI | p |
|---|---|---|---|
| 30 dni | −0,8% | [−25,7%, +24,0%] | 0,920 |
| 90 dni | +12,1% | [−67,7%, +91,9%] | 0,663 |
| 180 dni | +47,7% | [−117,7%, +213,2%] | 0,426 |
| 365 dni | **+125,2%** | **[−190,5%, +441,0%]** | **0,296** |

n = 4. Efekt może być ogromny albo ujemny — przy czterech obserwacjach nie da się
tego rozstrzygnąć.

## Grupa kontrolna (test placebo)

Różnica sparowana po zdarzeniach, horyzont 365 dni, okna kalendarzowe.

| | CAR BTC | CAR kontroli | różnica | 95% CI | p |
|---|---|---|---|---|---|
| vs NASDAQ | +125,2% | +17,0% | +108,2% | [−169,4%, +385,8%] | 0,303 |
| vs S&P 500 | +125,2% | +13,7% | +111,5% | [−167,7%, +390,7%] | 0,293 |
| vs złoto | +125,2% | −17,3% | +142,5% | [−200,5%, +485,4%] | 0,278 |

Żadnej różnicy nie da się odróżnić od zera. Placebo działa poprawnie w drugą
stronę: metoda puszczona na NASDAQ nie znajduje tam efektu halvingu
(+17,0%, p = 0,307), więc brak wyniku dla BTC nie wynika z bezsilności metody.

## Kategorie zdarzeń (p-value SUROWE, przed korektą)

| kategoria | n | CAR(365d) | p surowe |
|---|---|---|---|
| regulation | 3 | −194,4% | 0,081 |
| market_structure | 3 | −304,7% | 0,197 |
| macro | 4 | +106,1% | 0,287 |
| halving | 4 | +125,2% | 0,296 |
| credit_event | 6 | −81,8% | 0,494 |

Tych liczb nie wolno czytać jako wyników — to wejście do korekty poniżej.

## Walidacja

**23 hipotezy** (okna halvingowe × kategorie zdarzeń × fazy makro):

* istotnych surowo: **0** — sam przypadek dałby ~1,2,
* po korekcie Benjamini-Hochberg: **0**,
* replikuje się poza próbą (cykle 0–2 → 3–4): **0**,
* pominięte, bo brak danych: 3 (kategoria `cycle_extreme` jest jeszcze pusta).

## Oś płynności: prawdziwe M2 vs proxy dolarowe

Zgodność **42,3%** na 5130 porównanych dni — poniżej poziomu przypadku (50% dla
osi binarnej), bo w tej próbie są przeciwstawne.

| | proxy: contracting | proxy: expanding |
|---|---|---|
| **M2: contracting** | 1588 | 2193 |
| **M2: expanding** | 767 | 582 |

Faza „płynność rośnie, stopy rosną" ma −20,6% rocznie licząc z M2 i +105,9%
licząc z proxy. Ta sama etykieta, przeciwny wniosek.

## Backtest (10 bps prowizji + 15 bps poślizgu)

| strategia | zwrot | CAGR | Sharpe | maxDD | ekspozycja |
|---|---|---|---|---|---|
| kup i trzymaj | +715 635% | 80,3% | 1,14 | −84,9% | 100% |
| trend 50/200 | +885 999% | 82,9% | **1,27** | −78,3% | 61% |
| halving +365d | +254 351% | 68,4% | **1,42** | −71,0% | 27% |
| halving +180d | +2 555% | 24,3% | 0,87 | −70,3% | 13% |
| halving +90d | +242% | 8,5% | 0,76 | −21,8% | 7% |
| halving +30d | +12% | 0,7% | 0,13 | −20,5% | 2% |
| makro (M2 ↑, stopy ↓) | +257% | 8,8% | 0,46 | −57,4% | 10% |

---

## Wniosek

Jedyna strategia z lepszym Sharpe'em niż kup-i-trzymaj przy sensownej ekspozycji
(`halving +365d`, 1,42 przy 27% czasu w rynku) **nie przechodzi walidacji
statystycznej i nie odróżnia się od NASDAQ-a**. Cztery cykle to cztery
obserwacje, a „bądź w rynku rok po halvingu" pokrywa się w dużej mierze
z „bądź w rynku w hossie".

To jest wynik, nie porażka. Projekt powstał po to, żeby odróżnić wzorzec od szumu,
i odróżnił.

---

## Czego tu nie ma

`features.csv` (3,9 MB) i `lab.sqlite` (11 MB) są celowo poza kontrolą wersji —
to dane odtwarzalne z jednej komendy, a ich wersjonowanie zaśmiecałoby historię
przy każdym przebiegu.
