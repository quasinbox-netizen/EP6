@echo off
rem Convenience wrapper: same as `btc.cmd dashboard`.
rem Kept because it is a shorter thing to type and to remember.
rem Optional argument: the port (default 8511).
setlocal
call "%~dp0btc.cmd" dashboard %*
exit /b %errorlevel%
