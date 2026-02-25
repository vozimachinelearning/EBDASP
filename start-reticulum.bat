@echo off
setlocal
set "SWARM_ROOT=%~dp0"
set "RNS_CONFIG_DIR=%SWARM_ROOT%"
set "HF_HOME=%SWARM_ROOT%hf"
set "SWARM_MODELS=%SWARM_ROOT%models"
if not defined SWARM_LLM_REPO set "SWARM_LLM_REPO=Qwen/Qwen2.5-0.5B-Instruct"
if not defined SWARM_EMBED_REPO set "SWARM_EMBED_REPO=BAAI/bge-m3"
set "SWARM_LLM_PATH=%SWARM_MODELS%\llm"
set "SWARM_EMBED_PATH=%SWARM_MODELS%\embeddings"
if not exist "%SWARM_MODELS%" mkdir "%SWARM_MODELS%"
if not exist "%SWARM_LLM_PATH%" mkdir "%SWARM_LLM_PATH%"
if not exist "%SWARM_EMBED_PATH%" mkdir "%SWARM_EMBED_PATH%"
where huggingface-cli >nul 2>&1
if errorlevel 1 python -m pip install -U "huggingface_hub[cli]"
huggingface-cli download "%SWARM_LLM_REPO%" --local-dir "%SWARM_LLM_PATH%" --local-dir-use-symlinks False
huggingface-cli download "%SWARM_EMBED_REPO%" --local-dir "%SWARM_EMBED_PATH%" --local-dir-use-symlinks False
rnsd --config "%RNS_CONFIG_DIR%"
