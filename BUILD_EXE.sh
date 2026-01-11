#!/bin/bash
# Script para gerar executável do RenderX no macOS/Linux

echo "============================================================"
echo "  RenderX - Gerador de Executável"
echo "============================================================"
echo ""

# Ativar ambiente virtual se existir
if [ -f "venv/bin/activate" ]; then
    echo "Ativando ambiente virtual..."
    source venv/bin/activate
fi

# Executar script de build
python3 build_exe.py

echo ""
echo "Pressione Enter para sair..."
read
