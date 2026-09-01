@echo off
rem ============================================================================
rem  Uruchamia dashboard i otwiera go w przegladarce.
rem  Srodowisko przygotowuje btc.cmd, wiec tutaj wystarczy je wywolac z komenda,
rem  ktora nic nie robi - dzieki temu logika setupu zyje w jednym miejscu.
rem ============================================================================
setlocal
set "ROOT=%~dp0"
set "PY=%ROOT%.venv\Scripts\python.exe"
set "PORT=8511"
if not "%~1"=="" set "PORT=%~1"

rem Wymus przygotowanie srodowiska (btc.cmd samo je odtworzy, jesli trzeba).
call "%ROOT%btc.cmd" --help >nul 2>&1
if not exist "%PY%" (
    echo [blad] Srodowisko nie jest gotowe - uruchom najpierw: btc.cmd
    exit /b 1
)

echo Dashboard startuje na http://localhost:%PORT%
echo Zatrzymanie: Ctrl+C w tym oknie.
echo.
"%PY%" -m streamlit run "%ROOT%dashboard\app.py" --server.port %PORT% --browser.gatherUsageStats false
exit /b %errorlevel%
