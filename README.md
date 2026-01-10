# 🎬 RENDERX v3.3 - Editor de Vídeo

Editor de vídeo profissional para criação automática de vídeos com TTS, legendas e efeitos visuais.

**Versão 3.3 - Otimizações de Velocidade** 🚀
- ⚡ Modo de Performance configurável (Rápido/Equilibrado/Qualidade)
- ⚡ 40-60% mais rápido na renderização
- ⚡ Interpolação e presets FFmpeg otimizados
- ⚡ Normalização paralela de vídeos

## 🚀 Início Rápido

### macOS/Linux
```bash
./INICIAR.sh
```

### Windows
```cmd
INICIAR.bat
```

## 📋 Requisitos

- Python 3.8 ou superior
- FFmpeg instalado e no PATH
- Dependências Python (instaladas automaticamente)

## 🎯 Funcionalidades

- ✅ **Processamento em Lote**: Processa múltiplos vídeos automaticamente
- ✅ **Modo de Performance** (v3.3): Escolha entre velocidade e qualidade
- ✅ **Geração de Áudio TTS**: Converte texto em voz usando APIs (DARKVI ou TALKIFY)
- ✅ **Sistema de Legendas**: Importa SRT ou gera automaticamente via AssemblyAI
- ✅ **Efeitos Visuais**: Zoom centralizado, transições suaves, overlay de partículas
- ✅ **Mixagem de Áudio**: Combina narração com música de fundo
- ✅ **Processamento Paralelo**: Renderiza múltiplos vídeos simultaneamente
- ✅ **VSL (Video Sales Letter)**: Insere vídeos de venda automaticamente
- ✅ **Otimizações de Velocidade** (v3.3): 40-60% mais rápido que versões anteriores

## 📁 Estrutura

```
RENDERX/
├── iniciar_render.py                # Script principal
├── INICIAR.sh                        # Script de inicialização (macOS/Linux)
├── INICIAR.bat                       # Script de inicialização (Windows)
├── requirements.txt                   # Dependências Python
├── EFEITOS/                          # Recursos (músicas, overlays, VSLs)
│   ├── VSLs/
│   └── overlay.mp4
├── final_settings.json                # Configurações salvas
├── subtitle_presets.json             # Presets de legendas
├── vsl_keywords.json                 # Palavras-chave VSL
└── README.md                         # Este arquivo
```

## ⚙️ Configuração

1. Execute o script de inicialização
2. Configure as pastas de entrada e saída na interface
3. Ajuste as configurações conforme necessário
4. Clique em "Iniciar Processamento"

## 📖 Documentação

Consulte `MANUAL_DO_USUARIO.md` e `GUIA_RAPIDO.md` para mais informações detalhadas.

## 🔧 Solução de Problemas

### Python não encontrado
Instale Python 3.8+ e adicione ao PATH do sistema.

### FFmpeg não encontrado
**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt-get install ffmpeg
```

**Windows:**
Baixe de https://ffmpeg.org/download.html e adicione ao PATH.

### Dependências faltando
O script tenta instalar automaticamente. Se falhar, execute:
```bash
pip install -r requirements.txt
```

