@echo off
setlocal
set "SWARM_ROOT=%~dp0"
set "VENV_DIR=%SWARM_ROOT%.venv"
if not exist "%VENV_DIR%\Scripts\python.exe" (
    python -m venv "%VENV_DIR%"
    if errorlevel 1 goto :error
)
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 goto :error
python -m pip install -U pip
if errorlevel 1 goto :error
python -m pip install -U reticulum rns "huggingface_hub[cli]"
if errorlevel 1 goto :error
set "RNS_CONFIG_DIR=%SWARM_ROOT%"
set "HF_HOME=%SWARM_ROOT%hf"
set "SWARM_MODELS=%SWARM_ROOT%models"
if not defined SWARM_LLM_REPO set "SWARM_LLM_REPO=Qwen/Qwen2.5-0.5B-Instruct"
if not defined SWARM_EMBED_REPO set "SWARM_EMBED_REPO=BAAI/bge-m3"
if not defined SWARM_RNS_LOGLEVEL set "SWARM_RNS_LOGLEVEL=4"
set "SWARM_GROUP_ID=public"
set "SWARM_NETWORK=1"
set "SWARM_NODE_ID=%COMPUTERNAME%"
set "SWARM_LLM_PATH=%SWARM_MODELS%\llm"
set "SWARM_EMBED_PATH=%SWARM_MODELS%\embeddings"
set "PYTHONPATH=%SWARM_ROOT%src;%PYTHONPATH%"
powershell -NoProfile -Command "(Get-Content '%RNS_CONFIG_DIR%config') -replace '^\s*group_id\s*=.*$','    group_id = %SWARM_GROUP_ID%' | Set-Content '%RNS_CONFIG_DIR%config'"
powershell -NoProfile -Command "(Get-Content '%RNS_CONFIG_DIR%config') -replace '^\s*loglevel\s*=.*$','  loglevel = %SWARM_RNS_LOGLEVEL%' | Set-Content '%RNS_CONFIG_DIR%config'"
if not exist "%SWARM_MODELS%" mkdir "%SWARM_MODELS%"
if not exist "%SWARM_LLM_PATH%" mkdir "%SWARM_LLM_PATH%"
if not exist "%SWARM_EMBED_PATH%" mkdir "%SWARM_EMBED_PATH%"
huggingface-cli download "%SWARM_LLM_REPO%" --local-dir "%SWARM_LLM_PATH%" --local-dir-use-symlinks False
if errorlevel 1 goto :error
huggingface-cli download "%SWARM_EMBED_REPO%" --local-dir "%SWARM_EMBED_PATH%" --local-dir-use-symlinks False
if errorlevel 1 goto :error
start "" rnsd --config "%RNS_CONFIG_DIR%"
timeout /t 3 /nobreak >nul
python "%SWARM_ROOT%examples\minimal_demo.py"
pause
exit /b 0
:error
echo Error al iniciar. Revisa la salida anterior.
pause
