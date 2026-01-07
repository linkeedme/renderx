#!/bin/bash
# =============================================================================
# RenderX - Instalação Completa (Primeira Vez)
# Instala Python, FFmpeg, dependências e configura tudo automaticamente
# =============================================================================

cd "$(dirname "$0")"

echo ""
echo "  +====================================================================="
echo "  |                                                                     |"
echo "  |     RENDERX v3.2 - INSTALAÇÃO COMPLETA                             |"
echo "  |     Equipe Matrix                                                   |"
echo "  |                                                                     |"
echo "  +====================================================================="
echo ""
echo "  Este script vai instalar tudo que você precisa:"
echo ""
echo "  [1] Python 3.11 (se não estiver instalado)"
echo "  [2] FFmpeg (se não estiver instalado)"
echo "  [3] Ambiente virtual Python"
echo "  [4] Todas as dependências (opencv, customtkinter, etc)"
echo "  [5] Arquivos de configuração"
echo ""
echo "  +====================================================================="
echo ""
read -p "  Pressione ENTER para continuar..."

# Detectar sistema operacional
OS="$(uname -s)"
echo ""
echo "  Sistema detectado: $OS"

# =============================================================================
# ETAPA 1: Verificar/Instalar Python
# =============================================================================
echo ""
echo "  [ETAPA 1/5] Verificando Python..."

if ! command -v python3 &> /dev/null; then
    echo "  [!] Python não encontrado. Instalando..."
    
    if [ "$OS" = "Darwin" ]; then
        # macOS
        if ! command -v brew &> /dev/null; then
            echo "  [!] Homebrew não encontrado. Instalando..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        brew install python@3.11
    else
        # Linux
        if command -v apt &> /dev/null; then
            sudo apt update
            sudo apt install -y python3.11 python3.11-venv python3-pip
        elif command -v dnf &> /dev/null; then
            sudo dnf install -y python3.11
        elif command -v pacman &> /dev/null; then
            sudo pacman -S python
        else
            echo "  [ERRO] Gerenciador de pacotes não reconhecido!"
            echo "  Instale Python 3.11 manualmente."
            exit 1
        fi
    fi
    
    echo "  [OK] Python instalado!"
else
    PYVER=$(python3 --version 2>&1)
    echo "  [OK] Python encontrado: $PYVER"
fi

# =============================================================================
# ETAPA 2: Verificar/Instalar FFmpeg
# =============================================================================
echo ""
echo "  [ETAPA 2/5] Verificando FFmpeg..."

if ! command -v ffmpeg &> /dev/null; then
    echo "  [!] FFmpeg não encontrado. Instalando..."
    
    if [ "$OS" = "Darwin" ]; then
        # macOS
        brew install ffmpeg
    else
        # Linux
        if command -v apt &> /dev/null; then
            sudo apt install -y ffmpeg
        elif command -v dnf &> /dev/null; then
            sudo dnf install -y ffmpeg
        elif command -v pacman &> /dev/null; then
            sudo pacman -S ffmpeg
        else
            echo "  [ERRO] Instale FFmpeg manualmente!"
            exit 1
        fi
    fi
    
    echo "  [OK] FFmpeg instalado!"
else
    echo "  [OK] FFmpeg encontrado!"
fi

# =============================================================================
# ETAPA 3: Criar Ambiente Virtual
# =============================================================================
echo ""
echo "  [ETAPA 3/5] Criando ambiente virtual Python..."

if [ -f "venv/bin/python" ]; then
    echo "  [OK] Ambiente virtual já existe."
else
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "  [ERRO] Falha ao criar ambiente virtual!"
        exit 1
    fi
    echo "  [OK] Ambiente virtual criado!"
fi

# =============================================================================
# ETAPA 4: Instalar Dependências
# =============================================================================
echo ""
echo "  [ETAPA 4/5] Instalando dependências Python..."
echo "  (isso pode demorar alguns minutos)"
echo ""

source venv/bin/activate
pip install --upgrade pip --quiet
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "  [!] Algumas dependências falharam. Tentando instalação alternativa..."
    pip install opencv-python numpy Pillow customtkinter assemblyai httpx
fi

echo "  [OK] Dependências instaladas!"

# =============================================================================
# ETAPA 5: Configurar Arquivos
# =============================================================================
echo ""
echo "  [ETAPA 5/5] Configurando arquivos..."

# Criar whisk_keys.json se não existir
if [ ! -f "whisk_keys.json" ] && [ -f "whisk_keys.example.json" ]; then
    cp "whisk_keys.example.json" "whisk_keys.json"
    echo "  [OK] Criado: whisk_keys.json (edite com seus tokens)"
fi

# Criar final_settings.json se não existir
if [ ! -f "final_settings.json" ] && [ -f "final_settings.example.json" ]; then
    cp "final_settings.example.json" "final_settings.json"
    echo "  [OK] Criado: final_settings.json (configure suas chaves de API)"
fi

# Criar pastas necessárias
mkdir -p "EFEITOS/VSLs"
mkdir -p "EFEITOS/BACKLOG_VIDEOS"
mkdir -p "MATERIAL"
mkdir -p "SAIDA"
echo "  [OK] Pastas criadas!"

# Tornar scripts executáveis
chmod +x INICIAR.sh ATUALIZAR.sh 2>/dev/null

# =============================================================================
# INSTALAÇÃO CONCLUÍDA
# =============================================================================
echo ""
echo "  +====================================================================="
echo "  |                                                                     |"
echo "  |     INSTALAÇÃO CONCLUÍDA COM SUCESSO!                              |"
echo "  |                                                                     |"
echo "  +====================================================================="
echo ""
echo "  IMPORTANTE - Configure suas chaves de API:"
echo ""
echo "  1. Edite \"whisk_keys.json\" - adicione seus tokens do Whisk"
echo "  2. Edite \"final_settings.json\" - adicione suas chaves:"
echo "     - assemblyai_key (para legendas)"
echo "     - darkvi_api_key (para TTS)"
echo ""
echo "  Coloque seus arquivos em:"
echo "  - MATERIAL/       = áudios e textos de entrada"
echo "  - EFEITOS/VSLs/   = seus vídeos de VSL"
echo ""
echo "  +====================================================================="
echo ""

read -p "  Deseja abrir o RenderX agora? (s/n): " ABRIR

if [ "$ABRIR" = "s" ] || [ "$ABRIR" = "S" ]; then
    echo ""
    echo "  Iniciando RenderX..."
    ./INICIAR.sh
else
    echo ""
    echo "  Para iniciar depois, execute: ./INICIAR.sh"
    echo ""
fi

