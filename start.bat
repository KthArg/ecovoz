@echo off
rem Sets up the environment the first time and opens EcoVoz.
cd /d "%~dp0"
if exist .venv\ready goto run
python -m venv .venv || goto error
where nvidia-smi >nul 2>nul
if not errorlevel 1 (
  .venv\Scripts\python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130 || goto error
)
.venv\Scripts\python -m pip install -r requirements.txt || goto error
.venv\Scripts\python -m pip install -e . --no-deps || goto error
type nul > .venv\ready

:run
.venv\Scripts\python app.py --open %*
if errorlevel 1 goto error
exit /b 0

:error
echo EcoVoz could not start. Check the message above.
pause
exit /b 1
