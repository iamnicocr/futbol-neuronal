@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
 echo Creando entorno virtual...
 py -m venv .venv
 if errorlevel 1 (python -m venv .venv)
)
echo Instalando dependencias...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
 echo No se pudieron instalar dependencias. Revisa Internet y Python.
 pause
 exit /b 1
)
echo Iniciando Futbol Neuronal...
".venv\Scripts\python.exe" -m streamlit run app.py
pause
