@echo off
REM ============================================================
REM  One-click: push this repo to YOUR GitHub (private by default)
REM  Run from PowerShell as:   .\push_to_github.bat
REM  It will open a GitHub login prompt the first time.
REM ============================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "REPO_NAME=rag-covid-interviews"
set "VISIBILITY=--private"
REM  -> change to  set "VISIBILITY=--public"  if you want it public.

REM --- locate gh (GitHub CLI) ---
set "GH="
where gh >nul 2>nul && set "GH=gh"
if not defined GH (
  set "CAND=%LOCALAPPDATA%\Microsoft\WinGet\Packages\GitHub.cli_Microsoft.Winget.Source_8wekyb3d8bbwe\bin\gh.exe"
  if exist "!CAND!" set "GH=!CAND!"
)
if not defined GH (
  echo [ERROR] GitHub CLI not found. Close this window, open a NEW PowerShell, and try again.
  echo         If it still fails, install it with:  winget install GitHub.cli
  pause & exit /b 1
)
echo Using gh: !GH!

echo.
echo [1/4] Checking GitHub authentication...
"!GH!" auth status >nul 2>nul
if errorlevel 1 (
  echo     Not logged in - launching GitHub login. Follow the prompts ^(choose: GitHub.com,
  echo     HTTPS, authenticate Git = Yes, login with a web browser^).
  "!GH!" auth login
  if errorlevel 1 ( echo [ERROR] Login failed. & pause & exit /b 1 )
)

echo.
echo [2/4] Setting git author from your GitHub account...
for /f "usebackq delims=" %%i in (`"!GH!" api user --jq .login`) do set "GHUSER=%%i"
if defined GHUSER (
  git config user.name "!GHUSER!"
  git config user.email "!GHUSER!@users.noreply.github.com"
  git commit --amend --reset-author --no-edit >nul 2>nul
  echo     Author set to !GHUSER!
)

echo.
echo [3/4] Creating %VISIBILITY:--=% repo "%REPO_NAME%" and pushing...
"!GH!" repo create "%REPO_NAME%" %VISIBILITY% --source . --remote origin --push
if errorlevel 1 (
  echo [ERROR] Repo create/push failed. If the repo already exists, push with:
  echo         git push -u origin main
  pause & exit /b 1
)

echo.
echo [4/4] Success^! Your repo URL:
"!GH!" repo view "%REPO_NAME%" --json url --jq .url
echo.
pause
endlocal
