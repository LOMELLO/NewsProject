@echo off
rem Запуск NewsProject на Windows (из корня проекта)
cd /d "%~dp0"
python -m app.main
pause
