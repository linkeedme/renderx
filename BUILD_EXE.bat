@echo off
REM Script para gerar executável .exe do RenderX no Windows

echo ============================================================
echo   RenderX - Gerador de Executavel
echo ============================================================
echo.

REM Ativar ambiente virtual se existir
if exist venv\Scripts\activate.bat (
    echo Ativando ambiente virtual...
    call venv\Scripts\activate.bat
)

REM Executar script de build
python build_exe.py

echo.
echo Pressione qualquer tecla para sair...
pause >nul
