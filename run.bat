@echo off
setlocal
cd /d "%~dp0"

rem RecallOS - one-click launcher (Windows)

if not exist ".env" (
    echo [RecallOS] .env not found - copying template from .env.example
    copy /y ".env.example" ".env" >NUL
    echo [RecallOS] .env created. Please edit it and set your DEEPSEEK_API_KEY.
    echo.
)

if exist "venv\Scripts\python.exe" (
    set "PYTHON=venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

echo [RecallOS] Launching with %PYTHON% ...
"%PYTHON%" -m streamlit run app.py

endlocal