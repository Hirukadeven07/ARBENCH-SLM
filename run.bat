@echo off
REM ARBench one-command launcher (Windows cmd). Usage:  run.bat [--profile quick] [...]
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
exit /b %ERRORLEVEL%
