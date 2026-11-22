@echo off
rem Prepara el entorno la primera vez y abre EcoVoz.
cd /d "%~dp0"
if exist .venv\Scripts\python.exe goto run
python -m venv .venv || goto error
where nvidia-smi >nul 2>nul
if not errorlevel 1 (
  .venv\Scripts\python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130 || goto error
)
.venv\Scripts\python -m pip install -r requirements.txt || goto error
.venv\Scripts\python -m pip install -e . --no-deps || goto error

:run
.venv\Scripts\python app.py --open %*
if errorlevel 1 goto error
exit /b 0

:error
echo No se pudo iniciar EcoVoz. Revise el mensaje anterior.
pause
exit /b 1
