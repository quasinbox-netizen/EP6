@echo off
rem Convenience wrapper: same as `btc.cmd test`.
rem   test.cmd            - everything, including tests that reach real APIs
rem   test.cmd offline    - skips the network tests
rem   test.cmd tests\test_control_group.py  - a single file
setlocal
call "%~dp0btc.cmd" test %*
exit /b %errorlevel%
