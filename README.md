# btc-cycle-lab

Lokalne laboratorium do badania, czy cykl halvingowy i zdarzenia makro cokolwiek
wyjaśniają w cenie BTC.

Projekt jest zbudowany wokół jednej tezy: **łatwo znaleźć wzorzec, trudno pokazać,
że nie jest przypadkiem.** Dlatego każdy wynik wychodzi stąd z przedziałem ufności,
liczbą obserwacji i korektą na liczbę przetestowanych hipotez — a backtest zawsze
porównuje się z „kup i trzymaj".

---

## Szybki start

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
```

Pobranie danych (kilka minut, wszystkie źródła są darmowe i bez klucza):

```bash
PYTHONPATH=src .venv/Scripts/python -m cli ingest --what all
```

Pełny przebieg analizy:

```bash
PYTHONPATH=src .venv/Scripts/python -m cli all
```

Grupa kontrolna — czy „efekt halvingu" widać też tam, gdzie halvingu nie ma:

```bash
PYTHONPATH=src .venv/Scripts/python -m cli control --post 365
PYTHONPATH=src .venv/Scripts/python -m cli control --category credit_event --post 90
```

Oś płynności — czym jest liczona i czy wybór źródła zmienia wnioski:

```bash
PYTHONPATH=src .venv/Scripts/python -m cli macro
PYTHONPATH=src .venv/Scripts/python -m cli macro --check-key
```

Dashboard:

```bash
.venv/Scripts/python -m streamlit run dashboard/app.py
```

Testy:

```bash
.venv/Scripts/python -m pytest
```

Testy sieciowe (odpytują prawdziwe API) są oznaczone i domyślnie się uruchamiają;
`pytest -m "not network"` je pomija.

---

## Skąd biorą się dane

| Źródło | Zakres | Klucz API | Rola |
|---|---|---|---|
| Binance (BTC/USDT) | od 2017-08-17 | nie | najgłębszy rynek — źródło prawdy od 2017 |
| Bitstamp (BTC/USD) | od 2011-08-18 | nie | jedyne pokrywające halvingi 2012 i 2016 |
| Coinbase (BTC/USD) | od 2015-07-20 | nie | walidacja krzyżowa |
| Yahoo Finance | od 2009 | nie | DXY, S&P 500, rentowności, złoto |
| Yahoo Finance (kontrola) | od 2011 | nie | NASDAQ (^IXIC), S&P 500, złoto — grupa kontrolna |
| FRED | od lat 60. | **tak** (darmowy) | M2, produkcja przemysłowa, stopa Fed, bezrobocie |

Binance ma najwięcej danych w sensie głębokości i wolumenu, ale jego historia zaczyna
się w 2017 r. — sama giełda wcześniej nie istniała. Dlatego szereg jest **zszywany**:
Bitstamp do 2017-08, dalej Binance. Szew jest jawny i sprawdzany na zakładce
(mediana rozbieżności na wspólnych dniach: **0,06%**, maksimum 9% w grudniu 2017,
czyli w szczycie manii, gdy giełdy realnie się rozjeżdżały).

M2 i PMI nie mają darmowego API bez klucza. M2 zaciągniesz z FRED po wklejeniu
darmowego klucza do `.env`; ISM PMI ma licencję, która wyklucza redystrybucję —
wrzuć własny plik do `data/raw/manual/pmi.csv`, moduł go podchwyci.
Bez klucza FRED faza makro liczy się z proxy rynkowych (DXY, krzywa stóp) —
`cli macro` pokazuje, którym źródłem liczona jest oś płynności, a gdy dostępne
są oba, mierzy ich zgodność.

Dane z FRED pobierane są jako **pierwsze publikacje** (`output_type=4`), więc
`available_from` to prawdziwa data wejścia liczby do obiegu. Zmierzone na
prawdziwych wersjach M2SL: **mediana opóźnienia publikacji to 43 dni**, maksimum 58
— o dwa tygodnie więcej, niż podpowiada intuicja „miesiąc po końcu okresu".
Wszystkie 559 obserwacji dostało datę z archiwum wersji, zero fallbacku.

Trzy przypadki brzegowe, każdy z testem w `tests/test_fred_vintages.py`:

* sentinel `1776-07-04` — seria bez archiwum wersji,
* opóźnienie dłuższe niż ~5 miesięcy — obserwacja starsza od archiwum ALFRED
  albo skutek rewizji metodologii,
* **za dużo wersji** — FRED oddaje pierwsze publikacje do 2000 dat wersji, a DFF
  ma ich 5113. Przekraczają ten limit serie *dzienne*, czyli dokładnie te
  publikowane następnego dnia i praktycznie nierewidowane, więc pobieramy je
  bez archiwum wersji, ze stałym opóźnieniem 1 dnia.

Komunikaty błędów przechodzą przez `ingest.http.redact` — klucz API siedzi
w query stringu, więc surowy URL w treści wyjątku byłby wyciekiem do logów.

---

## Grupa kontrolna: test placebo

Halving jest zdarzeniem **wyłącznie bitcoinowym**. NASDAQ w tym samym oknie nie ma
prawa nic o nim wiedzieć — więc jeśli reaguje tak samo, to znaczy, że mierzymy
wspólny ruch rynków ryzyka, a nie połowienie nagrody.

Test jest **sparowany po zdarzeniach** (różnica w różnicach): dla każdego halvingu
liczymy CAR bitcoina i CAR kontroli w tym samym oknie, a wnioskujemy z rozkładu
ich różnicy. Parowanie ma znaczenie — halving 2020 wypadł w środku pandemicznego
odbicia, które podniosło oba aktywa; porównanie dwóch osobnych średnich zgubiłoby
ten fakt.

**Kalendarz to nie szczegół.** NASDAQ handluje się ~252 dni w roku, BTC 365.
Bez wyrównania „365 dni po halvingu" znaczy dla NASDAQ 365 *wierszy*, czyli
ok. 511 dni kalendarzowych — a zdarzenie wypadające w weekend w ogóle nie istnieje
w indeksie. Na danych testowych kosztuje to **3 z 4 zdarzeń**
(`test_without_calendar_alignment_most_events_are_lost`). Dlatego szereg kontrolny
przenosimy na pełny kalendarz przez `ffill` — operację wyłącznie wsteczną.

## Kontrakt czasowy — najważniejsza rzecz w tym repo

Każdy wiersz w bazie ma **dwie** daty:

* `date` — dzień, którego wartość dotyczy,
* `available_from` — dzień, w którym była publicznie znana.

Dla cen różnica to jeden dzień (bar dnia D domyka się o północy). Dla M2 to
około miesiąca. Cała analiza filtruje po `available_from`, nigdy po `date`.
Dane z FRED pobieramy jako **pierwsze publikacje** (`output_type=4`), a nie
zrewidowane — zrewidowane M2 za marzec 2020 poznaliśmy w 2021 r.

Sprawdza to `features/checks.py` metodą punkt-w-czasie: dla wybranego dnia t cechy
budowane są dwa razy — raz z całą historią, raz wyłącznie z danych opublikowanych
do t — i muszą wyjść identycznie. Test celowo zawiera też **podłożony wyciek**
(normalizacja medianą z całej próby), żeby udowodnić, że detektor działa.

---

## Struktura

```
src/
├── ingest/      pobieranie (4 giełdy, FRED, Yahoo, ręczne CSV) + kontrola jakości
├── features/    halving_distance, macro_phase, event_flags + detektor look-ahead
├── analysis/    event_study, korelacje z błędami HAC, grupa kontrolna (placebo)
├── backtest/    silnik z kosztami i poślizgiem + strategie
├── validation/  podziały po cyklach, walk-forward, Bonferroni/BH, dane syntetyczne
├── pipeline.py  spina wszystko — używane i przez CLI, i przez dashboard
└── cli.py
dashboard/       Streamlit — czysta prezentacja, zero logiki (pilnuje tego test)
tests/           137 testów: po jednym pliku na fazę + wersje FRED + grupa kontrolna
```

---

## Decyzje metodologiczne (i dlaczego takie)

**Jednostką obserwacji jest zdarzenie, nie dzień.** Przy czterech halvingach
niepewność bierze się z tego, że mieliśmy cztery halvingi — nie z tego, że
mieliśmy 1460 dni. Wnioskowanie idzie więc z rozkładu t o n−1 stopniach swobody
po zdarzeniach. Bootstrap percentylowy jest raportowany obok, ale **nie decyduje**:
przy n = 3–5 dawał ~30% fałszywych odkryć zamiast 5% (mierzone w
`test_false_positive_rate_stays_near_nominal`).

**Zwrot nadzwyczajny liczony względem okna sprzed zdarzenia** (−250..−31 dni),
nie względem średniej z całej próby — bo średnia z całej próby zawiera to,
co zdarzenie ma dopiero wyjaśnić.

**Test „okno vs reszta próby" przez cykliczne przesunięcia maski.** Zwykły test t
na dziennych zwrotach zakłada niezależność, której w cenach nie ma, i systematycznie
zawyża istotność. Permutacja przez rotację zachowuje autokorelację obu szeregów.

**Embargo między treningiem a testem.** Cel `fwd_return_90d` w dniu t zawiera ceny
z t+90, więc bez luki ostatnie dni treningu widzą zbiór testowy.

**Koszt naliczany od obrotu, nie od liczby transakcji.** Sygnał z dnia t wchodzi
z opóźnieniem — zerowe opóźnienie to handel po cenie, która dopiero się ustala.
Różnicę widać w `test_execution_lag_blocks_same_day_knowledge`: ten sam „sygnał"
daje +1000× bez opóźnienia i nic z opóźnieniem.

---

## Co wyszło na prawdziwych danych (2011-08 – 2026-09, 5494 dni)

**Event study, halvingi:** CAR po 365 dniach = **+125%**, przedział ufności
**[−190%, +441%]**, p = 0,30, n = 4.
Czyli: efekt może być ogromny albo ujemny — przy czterech obserwacjach nie da się
tego rozstrzygnąć. To nie jest porażka metody, to jest cała dostępna informacja.

**Skan 23 hipotez** (okna halvingowe × kategorie zdarzeń × fazy makro): 0 istotnych
surowo, 0 po korekcie BH. Sam przypadek dałby ~1,2 „odkrycia".

**Oś płynności: prawdziwe M2 vs proxy dolarowe.** Zanim doszedł klucz FRED, oś
płynności liczona była z odwróconego indeksu dolara. Obie wersje zgadzają się
w **42,3%** dni — czyli *poniżej* poziomu przypadku (dla osi binarnej losowo byłoby
50%), bo w tej próbie są wręcz przeciwstawne:

| | proxy: contracting | proxy: expanding |
|---|---|---|
| **M2: contracting** | 1588 | 2193 |
| **M2: expanding** | 767 | 582 |

Skutek jest jakościowy, nie ilościowy. Faza „płynność rośnie, stopy rosną" ma
**−20,6%** w skali roku licząc z M2 i **+105,9%** licząc z proxy. Ta sama etykieta,
przeciwny wniosek. Dlatego `cli macro` raportuje, którym źródłem liczona jest oś —
wniosek z proxy nie przenosi się na M2 i odwrotnie.

**Replikacja poza próbą** (cykle 0–2 → trening, 3–4 → test): **żadna** hipoteza się
nie powtórzyła. Efekty zmieniały znak albo kurczyły się do ułamka.

**Grupa kontrolna wokół halvingów** (CAR po 365 dniach, n = 4):

| | CAR BTC | CAR kontroli | różnica | 95% CI różnicy | p |
|---|---|---|---|---|---|
| vs NASDAQ | +125,2% | +17,0% | +108,2% | [−169,4%, +385,8%] | 0,303 |
| vs S&P 500 | +125,2% | +13,7% | +111,5% | [−167,7%, +390,7%] | 0,293 |
| vs złoto | +125,2% | −17,3% | +142,5% | [−200,5%, +485,4%] | 0,278 |

Bitcoin rósł po halvingach mocniej niż grupa kontrolna, ale przy czterech
zdarzeniach **nie da się tego odróżnić od zera**. Placebo działa poprawnie w drugą
stronę: metoda potraktowana na NASDAQ nie „znajduje" tam efektu halvingu
(CAR +17,0%, p = 0,307) — czyli brak wyniku dla BTC nie wynika z tego, że metoda
nigdy nic nie wykrywa.

W krótkim oknie (30 dni) kontrola wypada **lepiej** niż BTC: NASDAQ +5,3% vs
BTC −0,9%.

**Backtest** (10 bps prowizji + 15 bps poślizgu):

| strategia | zwrot | CAGR | Sharpe | maxDD | ekspozycja |
|---|---|---|---|---|---|
| kup i trzymaj | +715 635% | 80,3% | 1,14 | −84,9% | 100% |
| trend 50/200 | +885 999% | 82,9% | **1,27** | −78,3% | 61% |
| halving +365d | +254 351% | 68,4% | **1,42** | −71,0% | 27% |
| halving +180d | +2 555% | 24,3% | 0,87 | −70,3% | 13% |
| makro (płynność ↑, stopy ↓, M2) | +257% | 8,8% | 0,46 | −57,4% | 10% |

„Halving +365d" ma najlepszy Sharpe przy 27% czasu w rynku — i jednocześnie
nie przechodzi walidacji statystycznej. To jest dokładnie ta sytuacja, dla której
powstał ten projekt: **wynik wygląda dobrze i nie ma za nim dowodu.** Cztery cykle
to cztery obserwacje, a strategia „bądź w rynku rok po halvingu" pokrywa się
w dużej mierze z „bądź w rynku w hossie".

---

## Ograniczenia, o których trzeba pamiętać

* **Rejestr zdarzeń jest ułożony po fakcie** i przez to obciążony — pamiętamy te
  zdarzenia, po których rynek się poruszył. Kolumna `source` w `data/raw/events.csv`
  jest celowo pusta: uzupełnij i zweryfikuj daty, zanim cokolwiek z tego wywnioskujesz.
  Dodawaj też zdarzenia „nudne", które wtedy wyglądały groźnie, a skończyły się niczym.
* **Cztery halvingi to sufit mocy statystycznej.** Test
  `test_five_events_cannot_detect_a_drift_buried_in_btc_scale_noise` pilnuje,
  żeby nikt „nie poprawił" metody tak, by zaczęła znajdować efekty, na których
  wykrycie nie ma danych.
* **Binance kwotuje USDT, nie USD.** Poza epizodami utraty parytetu różnica jest
  ułamkiem procenta, ale zszycie ją raportuje zamiast milcząco akceptować.
* **Backtest nie zna finansowania, podatków ani ograniczeń płynności** przy dużych
  zleceniach. Poślizg 15 bps jest założeniem, nie pomiarem.

---

## Co dalej

1. Uzupełnić `source` w rejestrze zdarzeń i dodać zdarzenia bez reakcji rynku.
2. ~~Wkleić klucz FRED i powtórzyć fazę makro na prawdziwym M2 zamiast proxy.~~ zrobione
3. ~~Dołożyć drugą klasę aktywów jako grupę kontrolną.~~ zrobione — NASDAQ, S&P 500, złoto
4. Walk-forward na oknach rocznych zamiast pojedynczego podziału po cyklach.
