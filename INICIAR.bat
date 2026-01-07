@echo off
chcp 65001 >nul
REM Script de inicializacao para Windows
REM RenderX - Editor de Video

cd /d "%~dp0"

echo.
echo  +=====================================================================+
echo  ^|     ^>^> RENDERX - EDITOR DE VIDEO ^>^>                                ^|
echo  ^|                                                                     ^|
echo  ^|   RECURSOS:                                                         ^|
echo  ^|   * Processamento em Lote                                           ^|
echo  ^|   * Geracao de Audio TTS (DARKVI/TALKIFY)                           ^|
echo  ^|   * Sistema de Legendas (SRT/AssemblyAI)                            ^|
echo  ^|   * Efeitos Visuais (Zoom, Transicoes, Overlay)                     ^|
echo  ^|   * Mixagem de Audio                                                ^|
echo  ^|   * Processamento Paralelo                                          ^|
echo  +=====================================================================+
echo.

REM Verificar Python
python --version >nul 2>&1
if errorlevel 1 (
    echo   [ERRO] Python nao encontrado!
    echo   Instale o Python 3.10 ou 3.11 e adicione ao PATH.
    echo   Download: https://www.python.org/downloads/
    pause
    exit /b 1
)
echo   [OK] Python encontrado.

REM Verificar FFmpeg
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo   [ERRO] FFmpeg nao encontrado no PATH!
    echo   Baixe de: https://ffmpeg.org/download.html
    pause
    exit /b 1
)
echo   [OK] FFmpeg encontrado.

REM Criar/Ativar ambiente virtual
if exist "venv\Scripts\python.exe" (
    echo   [OK] Ambiente virtual encontrado.
) else (
    echo   [!] Criando ambiente virtual...
    python -m venv venv
    if errorlevel 1 (
        echo   [ERRO] Falha ao criar ambiente virtual!
        pause
        exit /b 1
    )
    echo   [OK] Ambiente virtual criado.
)

REM Instalar/Verificar dependencias usando o Python do venv
echo   [OK] Verificando dependencias...
venv\Scripts\python.exe -c "import cv2; import numpy; import customtkinter" 2>nul
if errorlevel 1 (
    echo   [!] Instalando dependencias... Aguarde...
    venv\Scripts\python.exe -m pip install --upgrade pip >nul 2>&1
    venv\Scripts\python.exe -m pip install opencv-python numpy Pillow customtkinter assemblyai httpx --only-binary=:all:
    if errorlevel 1 (
        echo   [!] Tentando instalacao alternativa...
        venv\Scripts\python.exe -m pip install opencv-python numpy Pillow customtkinter assemblyai httpx
    )
)

echo.
echo   Iniciando RenderX...
echo.

venv\Scripts\python.exe iniciar_render.py

if errorlevel 1 (
    echo.
    echo   [ERRO] Falha ao executar o aplicativo.
    echo.
    echo   Tente executar manualmente:
    echo   1. Abra o CMD nesta pasta
    echo   2. Execute: venv\Scripts\python.exe -m pip install opencv-python numpy customtkinter
    echo   3. Execute: venv\Scripts\python.exe iniciar_render.py
    echo.
    pause
)
