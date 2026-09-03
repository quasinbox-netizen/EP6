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

call :find_python
if not defined PYCMD goto no_python

%PYCMD% %PYARG% "%ROOT%run.py" %*
exit /b %errorlevel%

:no_python
echo.
echo [error] No Python 3.11 or newer found.
echo         Install it from https://www.python.org/downloads/windows/
echo         and tick "Add python.exe to PATH".
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
