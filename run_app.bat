@echo off
REM One-click launcher for the RAG Streamlit app (Windows).
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [setup] Creating virtual environment ...
  python -m venv .venv
  call .venv\Scripts\python.exe -m pip install --upgrade pip
  call .venv\Scripts\python.exe -m pip install -r requirements.txt
)
if not exist "storage\index\chunks.json" (
  echo [setup] Building knowledge base for the first time ...
  call .venv\Scripts\python.exe build_index.py
)
echo [run] Launching Streamlit at http://localhost:8501 ...
call .venv\Scripts\python.exe -m streamlit run app.py
