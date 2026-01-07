#!/bin/bash
# Script para iniciar a aplicação com o ambiente virtual ativado

cd "$(dirname "$0")"

# Ativar ambiente virtual
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Ambiente virtual ativado"
else
    echo "AVISO: Ambiente virtual não encontrado!"
    echo "Criando ambiente virtual..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Instalando dependências..."
    pip install httpx customtkinter pillow opencv-python requests numpy opencv-python-headless
fi

# Verificar se httpx está instalado
python3 -c "import httpx" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Instalando httpx..."
    pip install httpx
fi

# Executar aplicação usando o Python do venv
echo "Iniciando aplicação..."
exec python3 iniciar_render.py
