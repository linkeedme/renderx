@echo off
chcp 65001 >nul
REM =============================================================================
REM RenderX - Instalacao Completa (Primeira Vez)
REM Instala Python, FFmpeg, dependencias e configura tudo automaticamente
REM =============================================================================

cd /d "%~dp0"

echo.
echo  +=====================================================================+
echo  ^|                                                                     ^|
echo  ^|     RENDERX v3.2 - INSTALACAO COMPLETA                             ^|
echo  ^|     Equipe Matrix                                                   ^|
echo  ^|                                                                     ^|
echo  +=====================================================================+
echo.
echo   Este script vai instalar tudo que voce precisa:
echo.
echo   [1] Python 3.11 (se nao estiver instalado)
echo   [2] FFmpeg (se nao estiver instalado)
echo   [3] Ambiente virtual Python
echo   [4] Todas as dependencias (opencv, customtkinter, etc)
echo   [5] Arquivos de configuracao
echo.
echo  +=====================================================================+
echo.
pause

REM =============================================================================
REM ETAPA 1: Verificar/Instalar Python
REM =============================================================================
echo.
echo   [ETAPA 1/5] Verificando Python...

python --version >nul 2>&1
if errorlevel 1 (
    echo   [!] Python nao encontrado. Instalando...
    echo.
    
    REM Verificar se winget esta disponivel
    winget --version >nul 2>&1
    if errorlevel 1 (
        echo   [ERRO] Winget nao encontrado!
        echo   Por favor, instale o Python manualmente:
        echo   https://www.python.org/downloads/
        echo.
        echo   Marque a opcao "Add Python to PATH" durante a instalacao!
        pause
        exit /b 1
    )
    
    echo   Instalando Python 3.11 via winget...
    winget install Python.Python.3.11 --silent --accept-package-agreements --accept-source-agreements
    
    if errorlevel 1 (
        echo   [ERRO] Falha ao instalar Python!
        echo   Instale manualmente: https://www.python.org/downloads/
        pause
        exit /b 1
    )
    
    echo   [OK] Python instalado!
    echo   [!] IMPORTANTE: Feche e abra o terminal novamente para continuar.
    pause
    exit /b 0
) else (
    for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYVER=%%i
    echo   [OK] Python encontrado: %PYVER%
)

REM =============================================================================
REM ETAPA 2: Verificar/Instalar FFmpeg
REM =============================================================================
echo.
echo   [ETAPA 2/5] Verificando FFmpeg...

ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo   [!] FFmpeg nao encontrado. Instalando...
    
    REM Verificar se winget esta disponivel
    winget --version >nul 2>&1
    if errorlevel 1 (
        echo   [!] Winget nao disponivel. Tentando chocolatey...
        
        choco --version >nul 2>&1
        if errorlevel 1 (
            echo   [!] Chocolatey nao encontrado. Instalando...
            powershell -NoProfile -ExecutionPolicy Bypass -Command "Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))"
        )
        
        echo   Instalando FFmpeg via Chocolatey...
        choco install ffmpeg -y
    ) else (
        echo   Instalando FFmpeg via winget...
        winget install Gyan.FFmpeg --silent --accept-package-agreements --accept-source-agreements
    )
    
    if errorlevel 1 (
        echo   [ERRO] Falha ao instalar FFmpeg automaticamente!
        echo.
        echo   Instale manualmente:
        echo   1. Baixe de: https://www.gyan.dev/ffmpeg/builds/
        echo   2. Extraia para C:\ffmpeg
        echo   3. Adicione C:\ffmpeg\bin ao PATH do sistema
        echo.
        pause
        exit /b 1
    )
    
    echo   [OK] FFmpeg instalado!
    echo   [!] Pode ser necessario reiniciar o terminal.
) else (
    echo   [OK] FFmpeg encontrado!
)

REM =============================================================================
REM ETAPA 3: Criar Ambiente Virtual
REM =============================================================================
echo.
echo   [ETAPA 3/5] Criando ambiente virtual Python...

if exist "venv\Scripts\python.exe" (
    echo   [OK] Ambiente virtual ja existe.
) else (
    python -m venv venv
    if errorlevel 1 (
        echo   [ERRO] Falha ao criar ambiente virtual!
        pause
        exit /b 1
    )
    echo   [OK] Ambiente virtual criado!
)

REM =============================================================================
REM ETAPA 4: Instalar Dependencias
REM =============================================================================
echo.
echo   [ETAPA 4/5] Instalando dependencias Python...
echo   (isso pode demorar alguns minutos)
echo.

venv\Scripts\python.exe -m pip install --upgrade pip --quiet
venv\Scripts\python.exe -m pip install -r requirements.txt

if errorlevel 1 (
    echo   [!] Algumas dependencias falharam. Tentando instalacao alternativa...
    venv\Scripts\python.exe -m pip install opencv-python numpy Pillow customtkinter assemblyai httpx --only-binary=:all:
)

echo   [OK] Dependencias instaladas!

REM =============================================================================
REM ETAPA 5: Configurar Arquivos
REM =============================================================================
echo.
echo   [ETAPA 5/5] Configurando arquivos...

REM Criar whisk_keys.json se nao existir
if not exist "whisk_keys.json" (
    if exist "whisk_keys.example.json" (
        copy "whisk_keys.example.json" "whisk_keys.json" >nul
        echo   [OK] Criado: whisk_keys.json (edite com seus tokens)
    )
)

REM Criar final_settings.json se nao existir
if not exist "final_settings.json" (
    if exist "final_settings.example.json" (
        copy "final_settings.example.json" "final_settings.json" >nul
        echo   [OK] Criado: final_settings.json (configure suas chaves de API)
    )
)

REM Criar pastas necessarias
if not exist "EFEITOS\VSLs" mkdir "EFEITOS\VSLs"
if not exist "EFEITOS\BACKLOG_VIDEOS" mkdir "EFEITOS\BACKLOG_VIDEOS"
if not exist "MATERIAL" mkdir "MATERIAL"
if not exist "SAIDA" mkdir "SAIDA"
echo   [OK] Pastas criadas!

REM =============================================================================
REM INSTALACAO CONCLUIDA
REM =============================================================================
echo.
echo  +=====================================================================+
echo  ^|                                                                     ^|
echo  ^|     INSTALACAO CONCLUIDA COM SUCESSO!                              ^|
echo  ^|                                                                     ^|
echo  +=====================================================================+
echo.
echo   IMPORTANTE - Configure suas chaves de API:
echo.
echo   1. Edite "whisk_keys.json" - adicione seus tokens do Whisk
echo   2. Edite "final_settings.json" - adicione suas chaves:
echo      - assemblyai_key (para legendas)
echo      - darkvi_api_key (para TTS)
echo.
echo   Coloque seus arquivos em:
echo   - MATERIAL\       = audios e textos de entrada
echo   - EFEITOS\VSLs\   = seus videos de VSL
echo.
echo  +=====================================================================+
echo.

set /p ABRIR="   Deseja abrir o RenderX agora? (S/N): "

if /i "%ABRIR%"=="S" (
    echo.
    echo   Iniciando RenderX...
    call INICIAR.bat
) else (
    echo.
    echo   Para iniciar depois, execute: INICIAR.bat
    echo.
    pause
)

