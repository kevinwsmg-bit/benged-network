@echo off
title Benged Network - setup
cd /d "%~dp0"
echo.
echo  BENGED NETWORK - one-time setup
echo  ================================
echo.

rem The text reader needs Python 3.10, 3.11 or 3.12 (3.13 and 3.14 are too new).
rem Several Python versions can be installed side by side; we pick a supported one.
call :findpy
if defined PY goto havepy

echo  Python 3.12 is not installed (newer versions like 3.14 are too new for the reader).
echo  Installing Python 3.12 now. It sits next to any other Python you have.
echo.
winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
call :findpy
if defined PY goto havepy

echo.
echo  Could not install Python 3.12 automatically.
echo  1. Download Python 3.12 from https://www.python.org/downloads/release/python-3129/
echo     (scroll down, "Windows installer (64-bit)")
echo  2. On the first installer screen, tick "Add python.exe to PATH"
echo  3. Run SETUP.bat again.
echo.
pause
exit /b 1

:havepy
echo  Using Python:
%PY% --version
if exist .venv (
  echo  Removing the old .venv folder ...
  rmdir /s /q .venv
)
echo  Creating a private Python folder for the overlay (.venv) ...
%PY% -m venv .venv
if errorlevel 1 goto fail

echo  Installing the reader (about 150 MB, a few minutes) ...
.venv\Scripts\python -m pip install --upgrade pip >nul
.venv\Scripts\python -m pip install -r requirements.txt
if errorlevel 1 goto fail

echo.
echo  Done. Double-click START.bat whenever you stream.
echo.
pause
exit /b 0

:fail
echo.
echo  Setup failed. Screenshot this window and send it to Kevin.
pause
exit /b 1

:findpy
set "PY="
for %%v in (3.12 3.11 3.10) do (
  if not defined PY (
    py -%%v --version >nul 2>nul && set "PY=py -%%v"
  )
)
if defined PY exit /b 0
rem no py launcher entry: try the per-user install folder directly
for %%v in (312 311 310) do (
  if not defined PY (
    if exist "%LOCALAPPDATA%\Programs\Python\Python%%v\python.exe" set "PY="%LOCALAPPDATA%\Programs\Python\Python%%v\python.exe""
  )
)
exit /b 0
