@echo off
rem ============================================================================
rem  Uruchamia testy. Bez argumentu: caly zestaw wraz z testami sieciowymi.
rem      test.cmd              - wszystko
rem      test.cmd offline      - pomija testy odpytujace prawdziwe API
rem      test.cmd tests\test_control_group.py  - pojedynczy plik
rem ============================================================================
setlocal
set "ROOT=%~dp0"
set "PY=%ROOT%.venv\Scripts\python.exe"

call "%ROOT%btc.cmd" --help >nul 2>&1
if not exist "%PY%" (
    echo [blad] Srodowisko nie jest gotowe - uruchom najpierw: btc.cmd
    exit /b 1
)

pushd "%ROOT%"
if /i "%~1"=="offline" (
    "%PY%" -m pytest -q -m "not network"
) else (
    "%PY%" -m pytest -q %*
)
set "CODE=%errorlevel%"
popd
exit /b %CODE%
