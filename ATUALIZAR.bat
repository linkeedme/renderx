@echo off
chcp 65001 >nul
REM =============================================================================
REM RenderX - Script de Atualizacao
REM Baixa a versao mais recente do GitHub
REM =============================================================================

cd /d "%~dp0"

echo.
echo  +=====================================================================+
echo  ^|     RENDERX - ATUALIZACAO                                          ^|
echo  +=====================================================================+
echo.

REM Verificar se Git esta instalado
git --version >nul 2>&1
if errorlevel 1 (
    echo   [ERRO] Git nao encontrado!
    echo   Instale o Git: https://git-scm.com/download/win
    pause
    exit /b 1
)
echo   [OK] Git encontrado.

REM Verificar se eh um repositorio Git
if not exist ".git" (
    echo   [ERRO] Esta pasta nao eh um repositorio Git!
    echo   Clone o repositorio primeiro:
    echo   git clone https://github.com/linkeedme/renderx.git
    pause
    exit /b 1
)

REM Salvar configuracoes locais (backup)
echo.
echo   Fazendo backup das configuracoes...
if exist "final_settings.json" (
    copy /Y "final_settings.json" "final_settings.backup.json" >nul
    echo   [OK] Backup: final_settings.json
)
if exist "whisk_keys.json" (
    copy /Y "whisk_keys.json" "whisk_keys.backup.json" >nul
    echo   [OK] Backup: whisk_keys.json
)

REM Atualizar do GitHub
echo.
echo   Baixando atualizacoes do GitHub...
git fetch origin main
if errorlevel 1 (
    echo   [ERRO] Falha ao conectar com GitHub!
    echo   Verifique sua conexao com a internet.
    pause
    exit /b 1
)

git reset --hard origin/main
if errorlevel 1 (
    echo   [ERRO] Falha ao atualizar!
    pause
    exit /b 1
)

echo   [OK] Codigo atualizado!

REM Restaurar configuracoes
echo.
echo   Restaurando configuracoes...
if exist "final_settings.backup.json" (
    copy /Y "final_settings.backup.json" "final_settings.json" >nul
    del "final_settings.backup.json" >nul
    echo   [OK] Restaurado: final_settings.json
)
if exist "whisk_keys.backup.json" (
    copy /Y "whisk_keys.backup.json" "whisk_keys.json" >nul
    del "whisk_keys.backup.json" >nul
    echo   [OK] Restaurado: whisk_keys.json
)

REM Atualizar dependencias se necessario
echo.
echo   Verificando dependencias...
if exist "venv\Scripts\python.exe" (
    venv\Scripts\python.exe -m pip install -r requirements.txt --quiet
    echo   [OK] Dependencias atualizadas!
) else (
    echo   [!] Ambiente virtual nao encontrado.
    echo   Execute INICIAR.bat para criar.
)

echo.
echo  +=====================================================================+
echo  ^|     ATUALIZACAO CONCLUIDA!                                         ^|
echo  +=====================================================================+
echo.
echo   Pressione qualquer tecla para iniciar o RenderX...
pause >nul

REM Iniciar o programa
call INICIAR.bat

