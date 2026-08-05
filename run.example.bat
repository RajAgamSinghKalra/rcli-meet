@echo off
setlocal

:: Copy this file to run.bat and fill in the paths for your machine, then
:: just double-click run.bat (or run it from PowerShell/cmd) each time.

:: Path to your @runanywhere/electron dist/ build.
set "RCLI_MEET_SDK_DIST=C:/path/to/runanywhere-electron/dist"

:: A RunAnywhere LLM catalog id (auto-downloads, e.g. qwen2.5-3b) or a local
:: GGUF path.
set "RCLI_MEET_LLM_PATH=qwen2.5-3b"

:: Only needed if `python`/`py` on PATH resolves to the Windows Store alias
:: stub instead of a real interpreter -- point this at your real python.exe.
set "RCLI_MEET_PYTHON=python"

cd /d "%~dp0"
:: quiet.js filters the native addon's log spam; use src\main.js to see it.
node src\quiet.js --minutes 20 %*

pause
