@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else if exist "%USERPROFILE%\.venv\Scripts\python.exe" (
    set "PY=%USERPROFILE%\.venv\Scripts\python.exe"
) else (
    echo Virtual environment not found. Read README_FA.md.
    pause
    exit /b 1
)
"%PY%" -m streamlit run app.py
pause
