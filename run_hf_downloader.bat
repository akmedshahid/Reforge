@echo off
setlocal

cd /d %~dp0

echo ============================================
echo Advanced Hugging Face Full Repo Downloader
echo ============================================

where py >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    py -3 hf_model_downloader.py
    goto :eof
)

where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    python hf_model_downloader.py
    goto :eof
)

echo Python 3 is not installed or not on PATH.
pause
