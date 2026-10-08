@echo off
rem Abre a pagina local onde voce anexa o audio (http://127.0.0.1:8790).
cd /d "%~dp0"
rem Configure JEV_KEY_FILE e DEEPSEEK_KEY_FILE no ambiente, se desejar usar APIs.
python -s app.py
pause
