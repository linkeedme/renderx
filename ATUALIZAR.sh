#!/bin/bash
# =============================================================================
# RenderX - Script de Atualização
# Baixa a versão mais recente do GitHub
# =============================================================================

cd "$(dirname "$0")"

echo ""
echo "  +====================================================================="
echo "  |     RENDERX - ATUALIZAÇÃO                                          |"
echo "  +====================================================================="
echo ""

# Verificar se Git está instalado
if ! command -v git &> /dev/null; then
    echo "  [ERRO] Git não encontrado!"
    echo "  Instale o Git: brew install git (Mac) ou apt install git (Linux)"
    exit 1
fi
echo "  [OK] Git encontrado."

# Verificar se é um repositório Git
if [ ! -d ".git" ]; then
    echo "  [ERRO] Esta pasta não é um repositório Git!"
    echo "  Clone o repositório primeiro:"
    echo "  git clone https://github.com/linkeedme/renderx.git"
    exit 1
fi

# Salvar configurações locais (backup)
echo ""
echo "  Fazendo backup das configurações..."
if [ -f "final_settings.json" ]; then
    cp "final_settings.json" "final_settings.backup.json"
    echo "  [OK] Backup: final_settings.json"
fi
if [ -f "whisk_keys.json" ]; then
    cp "whisk_keys.json" "whisk_keys.backup.json"
    echo "  [OK] Backup: whisk_keys.json"
fi

# Atualizar do GitHub
echo ""
echo "  Baixando atualizações do GitHub..."
git fetch origin main
if [ $? -ne 0 ]; then
    echo "  [ERRO] Falha ao conectar com GitHub!"
    echo "  Verifique sua conexão com a internet."
    exit 1
fi

git reset --hard origin/main
if [ $? -ne 0 ]; then
    echo "  [ERRO] Falha ao atualizar!"
    exit 1
fi

echo "  [OK] Código atualizado!"

# Restaurar configurações
echo ""
echo "  Restaurando configurações..."
if [ -f "final_settings.backup.json" ]; then
    cp "final_settings.backup.json" "final_settings.json"
    rm "final_settings.backup.json"
    echo "  [OK] Restaurado: final_settings.json"
fi
if [ -f "whisk_keys.backup.json" ]; then
    cp "whisk_keys.backup.json" "whisk_keys.json"
    rm "whisk_keys.backup.json"
    echo "  [OK] Restaurado: whisk_keys.json"
fi

# Atualizar dependências se necessário
echo ""
echo "  Verificando dependências..."
if [ -f "venv/bin/python" ]; then
    source venv/bin/activate
    pip install -r requirements.txt --quiet
    echo "  [OK] Dependências atualizadas!"
else
    echo "  [!] Ambiente virtual não encontrado."
    echo "  Execute ./INICIAR.sh para criar."
fi

echo ""
echo "  +====================================================================="
echo "  |     ATUALIZAÇÃO CONCLUÍDA!                                         |"
echo "  +====================================================================="
echo ""
echo "  Para iniciar o RenderX, execute: ./INICIAR.sh"

