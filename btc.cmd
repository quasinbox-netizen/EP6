@echo off
rem ============================================================================
rem  btc-cycle-lab - launcher dla Windows.
rem
rem  Uzycie (z cmd.exe albo z PowerShella):
rem      .\btc.cmd ingest --what all
rem      .\btc.cmd study --post 365
rem      .\btc.cmd control
rem      .\btc.cmd validate
rem      .\btc.cmd backtest
rem      .\btc.cmd all
rem
rem  Skrypt sam sie podnosi: jesli brakuje srodowiska albo zostalo zepsute
rem  przeniesieniem katalogu (venv ma w srodku sciezki bezwzgledne), odtwarza je.
rem  Dzieki temu caly folder mozna skopiowac na inny komputer z Pythonem
rem  i po prostu uruchomic.
rem
rem  Uwaga dla edytujacych: zadnych zagniezdzonych blokow w nawiasach.
rem  cmd rozwija zmienne przy parsowaniu CALEGO bloku, wiec %errorlevel% i
rem  zmienne ustawiane w petli czytaja sie w srodku bloku jako wartosci
rem  sprzed jego wykonania. Stad wszedzie goto i podprocedury.
rem ============================================================================
setlocal
set "ROOT=%~dp0"
set "VENV=%ROOT%.venv"
set "PY=%VENV%\Scripts\python.exe"

if not exist "%PY%" goto setup

rem Venv istnieje - ale czy dziala? Po przeniesieniu katalogu nie zadziala.
"%PY%" -c "import pandas, streamlit, scipy, statsmodels, yaml" >nul 2>&1
if errorlevel 1 goto setup
goto run

:setup
echo.
echo [setup] Przygotowuje srodowisko - to potrwa kilka minut tylko za pierwszym razem.
echo.
if exist "%VENV%" echo [setup] Istniejace srodowisko jest niesprawne (np. katalog zostal przeniesiony) - odtwarzam.
if exist "%VENV%" rmdir /s /q "%VENV%"

call :check_path_length
call :find_python
if not defined PYCMD goto no_python

echo [setup] Uzywam: %PYCMD% %PYARG%
%PYCMD% %PYARG% -m venv "%VENV%"
if errorlevel 1 goto venv_failed

"%PY%" -m pip install --quiet --upgrade pip
"%PY%" -m pip install --quiet -r "%ROOT%requirements.txt"
if errorlevel 1 goto deps_failed

if exist "%ROOT%.env" goto env_ready
if not exist "%ROOT%.env.example" goto env_ready
copy /y "%ROOT%.env.example" "%ROOT%.env" >nul
echo [setup] Utworzylem .env z szablonu - wklej tam klucz FRED, jesli chcesz danych M2.

:env_ready
echo [setup] Gotowe.
echo.

:run
if "%~1"=="" goto usage
"%PY%" "%ROOT%src\cli.py" %*
exit /b %errorlevel%

:usage
"%PY%" "%ROOT%src\cli.py" --help
exit /b %errorlevel%

rem ---------------------------------------------------------------------------
rem Bledy
rem ---------------------------------------------------------------------------
:no_python
echo.
echo [blad] Nie znalazlem Pythona 3.11 lub nowszego.
echo        Zainstaluj go z https://www.python.org/downloads/windows/
echo        i zaznacz "Add python.exe to PATH".
exit /b 1

:venv_failed
echo [blad] Nie udalo sie utworzyc srodowiska wirtualnego.
exit /b 1

:deps_failed
echo.
echo [blad] Nie udalo sie zainstalowac zaleznosci z requirements.txt.
echo.
echo        Jesli w komunikacie wyzej widac "No such file or directory" i bardzo
echo        dluga sciezke, to limit 260 znakow Windows. Dwa wyjscia:
echo          1) przenies caly folder blizej korzenia dysku, np. C:\projekty\btc,
echo          2) albo wlacz dlugie sciezki w systemie (jednorazowo, jako admin):
echo             reg add HKLM\SYSTEM\CurrentControlSet\Control\FileSystem ^
echo                 /v LongPathsEnabled /t REG_DWORD /d 1 /f
exit /b 1

rem ---------------------------------------------------------------------------
rem Ostrzezenie o limicie 260 znakow. Streamlit rozpakowuje bardzo gleboko
rem zagniezdzone pliki przykladowe, wiec instalacja pada juz przy sciezce repo
rem dluzszej niz mniej wiecej 120 znakow.
rem ---------------------------------------------------------------------------
:check_path_length
set "PATHCHECK=%ROOT%"
if "%PATHCHECK:~120,1%"=="" goto :eof
echo.
echo [uwaga] Sciezka do repo jest dluga:
echo         %ROOT%
echo         Windows ma limit 260 znakow, a instalacja zaleznosci tworzy gleboko
echo         zagniezdzone pliki. Jesli setup padnie, przenies folder blizej
echo         korzenia dysku, np. do C:\projekty\btc.
echo.
goto :eof

rem ---------------------------------------------------------------------------
rem Szuka Pythona, od najlepiej przetestowanej wersji. 3.13 to wersja, na
rem ktorej projekt powstal. Interpreter trzymamy w DWOCH zmiennych, bo
rem "py -3.13" to dwa tokeny - wpisany w jedna zmienna i zacytowany rozjezdza
rem parsowanie cmd (objaw: "- was unexpected at this time").
rem ---------------------------------------------------------------------------
:find_python
set "PYCMD="
set "PYARG="
call :try_py 3.13
if defined PYCMD goto :eof
call :try_py 3.12
if defined PYCMD goto :eof
call :try_py 3.11
if defined PYCMD goto :eof
call :try_plain_python
goto :eof

:try_py
py -%1 -c "import sys" >nul 2>&1
if errorlevel 1 goto :eof
set "PYCMD=py"
set "PYARG=-%1"
goto :eof

:try_plain_python
python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if errorlevel 1 goto :eof
set "PYCMD=python"
set "PYARG="
goto :eof
