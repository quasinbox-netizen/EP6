@echo off
rem ===========================================================================
rem  btc-cycle-lab - Windows convenience wrapper around run.py
rem
rem      btc ingest --what all
rem      btc all
rem      btc dashboard
rem      btc test offline
rem      btc doctor
rem
rem  All the real work (creating the virtual environment, installing
rem  dependencies, dispatching) lives in run.py so that Windows, macOS and
rem  Linux share one implementation. This file only finds a Python to start it.
rem ===========================================================================
setlocal
set "ROOT=%~dp0"

rem ---------------------------------------------------------------------------
rem Double-clicking this file in Explorer runs it with no arguments. run.py then
rem prints its usage and exits, the console window closes with it, and from the
rem outside that is indistinguishable from "nothing happened" - which is exactly
rem what it looked like. So a run with no arguments shows a short menu instead,
rem and keeps the window open when it was started from Explorer rather than
rem typed into a shell. cmd.exe records how it was invoked in %cmdcmdline%; a
rem double-click always carries /c, an interactive prompt does not.
rem ---------------------------------------------------------------------------
if "%~1"=="" goto menu

call :find_python
if not defined PYCMD goto no_python

%PYCMD% %PYARG% "%ROOT%run.py" %*
set "CODE=%errorlevel%"
rem A failure deserves to be readable. Double-clicked, the window closes the
rem instant the command returns, so a missing dependency or a failed install
rem scrolls past and vanishes - the same way the usage text did. Hold the
rem window open on a non-zero exit, but only when Explorer started it; pausing
rem a command typed into a shell would be an annoyance, and pausing a
rem successful one would break piping output anywhere.
if not "%CODE%"=="0" call :pause_if_double_clicked
exit /b %CODE%

:menu
call :find_python
if not defined PYCMD goto no_python
echo.
echo   btc-cycle-lab
echo.
echo   This launcher needs a command. The usual ones:
echo.
echo     btc dashboard         open the dashboard in a browser (start here)
echo     btc ingest --what all download the data (do this first, once)
echo     btc all               run the whole analysis
echo     btc doctor            environment diagnostics
echo.
echo   Or double-click dashboard.cmd, which does the first one for you.
echo.
echo   The full list:
echo.
%PYCMD% %PYARG% "%ROOT%run.py" --help
echo.
call :pause_if_double_clicked
exit /b 0

rem ---------------------------------------------------------------------------
rem cmd.exe records its own invocation in %cmdcmdline%. Explorer always starts a
rem double-clicked .cmd with /c, so the window will close the moment the script
rem returns; an interactive prompt has no /c and the window stays anyway.
rem ---------------------------------------------------------------------------
:pause_if_double_clicked
echo %cmdcmdline% | find /i "/c" >nul
if errorlevel 1 goto :eof
echo.
pause
goto :eof

:no_python
echo.
echo [error] No Python 3.11 or newer found.
echo         Install it from https://www.python.org/downloads/windows/
echo         and tick "Add python.exe to PATH".
call :pause_if_double_clicked
exit /b 1

rem ---------------------------------------------------------------------------
rem Newest tested version first. The interpreter is kept in TWO variables
rem because "py -3.13" is two tokens - quoted as one it breaks the cmd parser
rem (symptom: "- was unexpected at this time").
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
