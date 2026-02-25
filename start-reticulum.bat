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
if not defined SWARM_LAN_PORT set "SWARM_LAN_PORT=4242"
if not defined SWARM_LAN_ENABLED set "SWARM_LAN_ENABLED=No"
if not defined SWARM_PUBLIC_SEEDS set "SWARM_PUBLIC_SEEDS=RNS Testnet BetweenTheBorders,RNS Transport US East,Beleth RNS Hub"
set "SWARM_GROUP_ID=public"
set "SWARM_NETWORK=1"
set "SWARM_NODE_ID=%COMPUTERNAME%"
set "SWARM_LLM_PATH=%SWARM_MODELS%\llm"
set "SWARM_EMBED_PATH=%SWARM_MODELS%\embeddings"
set "PYTHONPATH=%SWARM_ROOT%src;%PYTHONPATH%"
powershell -NoProfile -Command "(Get-Content '%RNS_CONFIG_DIR%config') -replace '^\s*group_id\s*=.*$','    group_id = %SWARM_GROUP_ID%' | Set-Content '%RNS_CONFIG_DIR%config'"
powershell -NoProfile -Command "(Get-Content '%RNS_CONFIG_DIR%config') -replace '^\s*loglevel\s*=.*$','  loglevel = %SWARM_RNS_LOGLEVEL%' | Set-Content '%RNS_CONFIG_DIR%config'"
powershell -NoProfile -Command "$path='%RNS_CONFIG_DIR%config';$content=Get-Content $path -Raw;$content=$content -replace '(?ms)^(\s*\[\[Default Local Interface\]\].*?\n\s*enabled\s*=\s*)(Yes|No)\s*$','$1%SWARM_LAN_ENABLED%';Set-Content $path $content"
powershell -NoProfile -Command "$path='%RNS_CONFIG_DIR%config';$content=Get-Content $path -Raw;$content=[regex]::Replace($content,'(?ms)(^\s*\[\[(?!Swarm LAN Peer ).*?\]\].*?\n\s*type\s*=\s*TCPClientInterface.*?\n\s*enabled\s*=\s*)(Yes|No)\s*$','${1}No');$seeds=@();if($env:SWARM_PUBLIC_SEEDS){$seeds=$env:SWARM_PUBLIC_SEEDS.Split(',')|ForEach-Object{$_.Trim()}|Where-Object{$_}};foreach($seed in $seeds){$escaped=[regex]::Escape($seed);$pattern='(?ms)(^\s*\[\['+$escaped+'\]\].*?\n\s*type\s*=\s*TCPClientInterface.*?\n\s*enabled\s*=\s*)(Yes|No)\s*$';$content=[regex]::Replace($content,$pattern,'${1}Yes')};Set-Content $path $content"
powershell -NoProfile -Command "$path='%RNS_CONFIG_DIR%config';$content=Get-Content $path -Raw;$content=$content -replace '(?ms)^\s*\[\[Swarm LAN Server\]\].*?(?=^\s*\[\[|\Z)','';$content=$content -replace '(?ms)^\s*\[\[Swarm LAN Peer.*?\]\].*?(?=^\s*\[\[|\Z)','';if($env:SWARM_LAN_ENABLED -match '^(Yes|True|1)$'){ $server=\"`r`n  [[Swarm LAN Server]]`r`n    type = TCPServerInterface`r`n    enabled = Yes`r`n    listen_port = %SWARM_LAN_PORT%`r`n\";$peers=\"\";if($env:SWARM_LAN_PEERS){$env:SWARM_LAN_PEERS.Split(',')|ForEach-Object{$h=$_.Trim();if($h){$peers+=\"`r`n  [[Swarm LAN Peer $h]]`r`n    type = TCPClientInterface`r`n    enabled = Yes`r`n    target_host = $h`r`n    target_port = %SWARM_LAN_PORT%`r`n\"}}};$content+=$server+$peers };Set-Content $path $content"
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
