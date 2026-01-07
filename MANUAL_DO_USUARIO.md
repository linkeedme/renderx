# 📖 MANUAL DO USUÁRIO - EDITOR DE VÍDEO ESPIRITUALIDADE

## 🎯 ÍNDICE

1. [Introdução](#introdução)
2. [Requisitos do Sistema](#requisitos-do-sistema)
3. [Instalação e Configuração Inicial](#instalação-e-configuração-inicial)
4. [Interface do Usuário](#interface-do-usuário)
5. [Configurações Básicas](#configurações-básicas)
6. [Processo de Renderização](#processo-de-renderização)
7. [Recursos Avançados](#recursos-avançados)
8. [Solução de Problemas](#solução-de-problemas)
9. [Dicas e Boas Práticas](#dicas-e-boas-práticas)

---

## 📌 INTRODUÇÃO

### O que é esta ferramenta?

O **Editor de Vídeo Espiritualidade** é uma aplicação profissional para criação automática de vídeos a partir de:
- **Áudios de narração** (MP3, WAV, M4A, etc.)
- **Textos** que podem ser convertidos em áudio via TTS (Text-to-Speech)
- **Imagens** que são combinadas com efeitos de zoom e transições
- **Legendas** automáticas ou importadas
- **Música de fundo** e **overlays** (efeitos visuais)

### Principais Funcionalidades

✅ **Processamento em Lote**: Processa múltiplos vídeos automaticamente  
✅ **Geração de Áudio TTS**: Converte texto em voz usando APIs (DARKVI ou TALKIFY)  
✅ **Sistema de Legendas**: Importa SRT ou gera automaticamente via AssemblyAI  
✅ **Efeitos Visuais**: Zoom centralizado, transições suaves, overlay de partículas  
✅ **Mixagem de Áudio**: Combina narração com música de fundo  
✅ **Processamento Paralelo**: Renderiza múltiplos vídeos simultaneamente  
✅ **VSL (Video Sales Letter)**: Insere vídeos de venda automaticamente  

---

## 💻 REQUISITOS DO SISTEMA

### Software Necessário

- **Python 3.8 ou superior**
- **FFmpeg** (para processamento de áudio/vídeo)
- **Sistema Operacional**: macOS, Linux ou Windows

### Dependências Python

A ferramenta instala automaticamente:
- `opencv-python` (processamento de vídeo)
- `numpy` (cálculos matemáticos)
- `customtkinter` (interface gráfica)
- `assemblyai` (transcrição de áudio)
- `httpx` (requisições HTTP para APIs TTS)

### Hardware Recomendado

- **CPU**: Processador multi-core (Ryzen 7 ou equivalente)
- **RAM**: 8GB mínimo, 16GB+ recomendado
- **GPU**: Opcional, mas acelera o processamento
- **Espaço em Disco**: Depende do tamanho dos vídeos (reserve pelo menos 10GB)

---

## 🚀 INSTALAÇÃO E CONFIGURAÇÃO INICIAL

### Passo 1: Verificar Python

Abra o terminal e verifique se o Python está instalado:

```bash
python3 --version
```

Se não estiver instalado, baixe em: https://www.python.org/downloads/

### Passo 2: Instalar FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

**Windows:**
Baixe de: https://ffmpeg.org/download.html e adicione ao PATH

### Passo 3: Executar a Ferramenta

**macOS/Linux:**
```bash
cd "/Users/davi/Desktop/FERRAMENTAS/EDITOR DE VIDEO"
chmod +x INICIAR.sh
./INICIAR.sh
```

**Windows:**
```cmd
cd "C:\caminho\para\EDITOR DE VIDEO"
INICIAR_FINAL.bat
```

O script irá:
1. Verificar se Python e FFmpeg estão instalados
2. Criar/ativar ambiente virtual (`venv`)
3. Instalar dependências automaticamente
4. Iniciar a interface gráfica

---

## 🖥️ INTERFACE DO USUÁRIO

### Estrutura da Interface

A interface é dividida em **abas/seções**:

#### 1. **Aba "Configurações Gerais"**
- Pastas de entrada e saída
- Configurações de resolução e zoom
- Duração de imagens e transições

#### 2. **Aba "Áudio e Efeitos"**
- Música de fundo
- Overlay (efeitos visuais)
- Volume e opacidade

#### 3. **Aba "Legendas"**
- Ativar/desativar legendas
- Método de geração (SRT ou AssemblyAI)
- Personalização (fonte, cores, posição)

#### 4. **Aba "TTS (Text-to-Speech)"**
- Configurar APIs (DARKVI ou TALKIFY)
- Selecionar voz
- Gerar áudio a partir de texto

#### 5. **Aba "VSL"**
- Configurar inserção de vídeos de venda
- Palavras-chave para detecção

### Tema Visual

A interface usa tema **escuro** com cores **pretas e laranjas** para melhor visualização.

---

## ⚙️ CONFIGURAÇÕES BÁSICAS

### 1. Configurar Pastas

#### Pasta de Entrada (Materiais)
Esta pasta deve conter:
- **Arquivos de áudio** (`.mp3`, `.wav`, `.m4a`, etc.)
- **Arquivos de texto** (`.txt`) - serão convertidos em áudio se TTS estiver ativo
- **Subpastas são suportadas** - a ferramenta varre recursivamente

**Exemplo de estrutura:**
```
MATERIAIS/
├── video1.mp3
├── video2.mp3
├── serie_a/
│   ├── ep1.mp3
│   └── ep2.mp3
└── textos/
    └── script.txt
```

#### Pasta de Saída
Vídeos renderizados serão salvos aqui, **replicando a estrutura de pastas** da entrada.

#### Pasta de Imagens
Banco de imagens usado para criar os vídeos. Formatos suportados:
- `.png`
- `.jpg` / `.jpeg`
- `.webp`

**Importante**: Imagens usadas são movidas para subpasta `UTILIZADAS/` para evitar repetição.

### 2. Configurações de Vídeo

#### Resolução
- **720p** (1280x720) - Padrão, mais rápido
- **1080p** (1920x1080) - Maior qualidade, mais lento

#### Zoom
- **Modo**: `zoom_in` (zoom para dentro) ou `zoom_out` (zoom para fora)
- **Escala**: 1.0 = sem zoom, 1.15 = zoom de 15% (padrão)

#### Duração
- **Duração da Imagem**: Tempo que cada imagem aparece (padrão: 8 segundos)
- **Duração da Transição**: Tempo de fade entre imagens (padrão: 1 segundo)
- **Imagens por Vídeo**: Quantas imagens usar por vídeo (padrão: 50)

### 3. Processamento Paralelo

- **Vídeos Paralelos**: Quantos vídeos renderizar ao mesmo tempo (1-4)
  - Mais vídeos = mais rápido, mas consome mais recursos
- **Threads por Vídeo**: Threads de processamento por vídeo (padrão: 6)

**Recomendação**: 
- CPU com 8+ cores: 2-3 vídeos paralelos
- CPU com 4 cores: 1-2 vídeos paralelos

---

## 🎬 PROCESSO DE RENDERIZAÇÃO

### Fluxo Completo

#### 1. Preparação dos Materiais

**Opção A: Usar Áudios Existentes**
- Coloque arquivos de áudio na pasta de materiais
- A ferramenta detecta automaticamente

**Opção B: Gerar Áudio via TTS**
- Coloque arquivos `.txt` na pasta de materiais
- Configure TTS (veja seção "TTS" abaixo)
- A ferramenta gera áudio automaticamente

#### 2. Configurar Legendas (Opcional)

**Método 1: Importar SRT**
- Crie arquivos `.srt` com mesmo nome do áudio
- Exemplo: `video1.mp3` → `video1.srt`

**Método 2: AssemblyAI (Automático)**
- Configure chave da API AssemblyAI
- A ferramenta transcreve o áudio automaticamente
- Gera legendas sincronizadas

#### 3. Iniciar Renderização

1. Clique em **"Iniciar Processamento em Lote"**
2. A ferramenta:
   - Escaneia a pasta de materiais
   - Para cada arquivo encontrado:
     - Gera áudio (se for texto e TTS ativo)
     - Seleciona imagens exclusivas
     - Cria vídeo com zoom e transições
     - Adiciona música de fundo (se configurada)
     - Adiciona overlay (se configurado)
     - Adiciona legendas (se ativado)
     - Adiciona VSL (se detectado)
     - Salva na pasta de saída

#### 4. Monitoramento

Durante o processamento, você verá:
- Progresso de cada vídeo
- Tempo estimado
- Logs de erros (se houver)

---

## 🎨 RECURSOS AVANÇADOS

### 1. Sistema de Legendas

#### Personalização Completa

**Fonte:**
- Nome da fonte (ex: "Arial", "Helvetica")
- Tamanho (em pixels)

**Cores:**
- **Cor Principal**: Cor do texto
- **Cor da Borda**: Cor do contorno
- **Cor da Sombra**: Cor da sombra
- **Cor do Karaokê**: Cor para efeito karaokê (destaque palavra por palavra)

**Efeitos:**
- **Tamanho da Borda**: Espessura do contorno (0-10)
- **Tamanho da Sombra**: Intensidade da sombra (0-10)
- **Karaokê**: Ativa destaque palavra por palavra

**Posicionamento:**
- 9 pontos de posição (1-9):
  - 1: Canto superior esquerdo
  - 2: Superior central
  - 3: Canto superior direito
  - 4: Centro esquerdo
  - **5: Centro (padrão)**
  - 6: Centro direito
  - 7: Canto inferior esquerdo
  - 8: Inferior central
  - 9: Canto inferior direito

#### Presets de Legendas

Você pode salvar e carregar configurações de legendas:
- **Salvar Preset**: Salva configuração atual com um nome
- **Carregar Preset**: Aplica configuração salva

Arquivo: `subtitle_presets.json`

### 2. Text-to-Speech (TTS)

#### Configurar API DARKVI

1. Obtenha token de API em: https://darkvi.com
2. Na aba "TTS", selecione **"DARKVI"** como provider
3. Cole o token no campo **"API Key"**
4. Clique em **"Listar Vozes"** para ver vozes disponíveis
5. Selecione uma voz da lista
6. Ative **"TTS Habilitado"**

**Limites DARKVI:**
- Máximo de 80.000 caracteres por requisição
- Processamento assíncrono (pode levar alguns minutos)

#### Configurar API TALKIFY

1. Obtenha token de API em: https://talkifydev.com
2. Selecione **"TALKIFY"** como provider
3. Cole o token
4. Informe o **Voice ID** (consulte documentação da API)
5. Ative TTS

#### Gerar Áudio

Quando TTS está ativo:
- Arquivos `.txt` na pasta de materiais são automaticamente convertidos em áudio
- Áudios gerados são salvos em: `AUDIOS_GERADOS/`
- O vídeo usa o áudio gerado

**Formato do texto:**
- Use arquivos `.txt` simples
- Sem limite de linhas (mas respeite limite de caracteres da API)

### 3. VSL (Video Sales Letter)

O sistema detecta automaticamente quando inserir um VSL baseado em palavras-chave no texto/áudio.

#### Configuração

1. Ative **"Usar VSL"**
2. Configure pasta de VSLs: `EFEITOS/VSLs/`
3. Configure arquivo de palavras-chave: `vsl_keywords.json`

#### Arquivo de Palavras-chave

Formato JSON com palavras-chave por idioma:

```json
{
  "portugues": ["prosperidade", "comprar", "vsl", "oferta"],
  "ingles": ["product", "offer", "buy", "vsl"],
  "espanhol": ["oferta", "comprar", "vsl"]
}
```

#### Funcionamento

1. Sistema analisa texto/áudio
2. Se detectar palavras-chave, busca VSL correspondente ao idioma
3. Insere VSL no vídeo (geralmente no início ou fim)

**Estrutura de pastas VSL:**
```
EFEITOS/VSLs/
├── VSL_portugues.mp4
├── VSL_ingles.mp4
└── VSL_espanhol.mp4
```

### 4. Música de Fundo

#### Adicionar Música

1. Na aba "Áudio e Efeitos"
2. Clique em **"Selecionar Música"**
3. Escolha arquivo de áudio (MP3, WAV, etc.)
4. Ajuste **Volume** (0.0 a 1.0)
   - 0.2 = 20% (padrão, música baixa)
   - 0.5 = 50% (música média)
   - 1.0 = 100% (música alta)

**Dica**: Use volume baixo (0.1-0.3) para não competir com a narração.

#### Mixagem Automática

A ferramenta:
- Combina narração + música
- Ajusta duração da música para corresponder ao vídeo
- Faz fade in/out suave

### 5. Overlay (Efeitos Visuais)

#### Adicionar Overlay

1. Na aba "Áudio e Efeitos"
2. Clique em **"Selecionar Overlay"**
3. Escolha vídeo de overlay (ex: partículas, poeira)
4. Ajuste **Opacidade** (0.0 a 1.0)
   - 0.3 = 30% (padrão, sutil)
   - 0.5 = 50% (moderado)
   - 1.0 = 100% (intenso)

**Tipos de Overlay:**
- Partículas
- Poeira
- Efeitos de luz
- Texturas

O overlay é aplicado com **blend mode "screen"** para efeito natural.

### 6. Smart Crop 16:9

A ferramenta automaticamente:
- Detecta proporção da imagem
- Faz crop inteligente para 16:9
- Centraliza conteúdo importante
- **Sem bordas pretas**

### 7. Zoom Centralizado

O zoom usa **matriz de rotação** para:
- Zoom suave a partir do centro
- Sem distorção
- Transições fluidas

---

## 🔧 SOLUÇÃO DE PROBLEMAS

### Erro: "Python não encontrado"

**Solução:**
- Instale Python 3.8+ de https://www.python.org
- Adicione ao PATH do sistema
- Reinicie o terminal

### Erro: "FFmpeg não encontrado"

**Solução:**
- Instale FFmpeg (veja seção "Instalação")
- Verifique se está no PATH: `ffmpeg -version`

### Erro: "Token DARKVI inválido"

**Solução:**
- Verifique se o token está correto
- Confirme que o token não expirou
- Teste no site da DARKVI

### Erro: "Imagens insuficientes"

**Solução:**
- Adicione mais imagens na pasta de imagens
- Reduza "Imagens por Vídeo" nas configurações
- Verifique se há imagens na pasta `UTILIZADAS/` (mova de volta se necessário)

### Vídeo sem áudio

**Solução:**
- Verifique se o arquivo de áudio existe
- Confirme formato suportado (MP3, WAV, M4A, etc.)
- Se usar TTS, verifique se o áudio foi gerado em `AUDIOS_GERADOS/`

### Legendas não aparecem

**Solução:**
- Verifique se "Usar Legendas" está ativado
- Confirme que arquivo SRT existe (se método SRT)
- Verifique chave AssemblyAI (se método AssemblyAI)
- Ajuste cor das legendas (pode estar igual ao fundo)

### Processamento muito lento

**Solução:**
- Reduza "Vídeos Paralelos" para 1
- Reduza "Threads por Vídeo"
- Use resolução 720p em vez de 1080p
- Reduza "Imagens por Vídeo"
- Feche outros programas pesados

### Vídeo com bordas pretas

**Solução:**
- Use imagens em proporção 16:9
- O Smart Crop deve resolver, mas imagens muito diferentes podem ter bordas mínimas

---

## 💡 DICAS E BOAS PRÁTICAS

### Organização de Arquivos

```
PROJETO/
├── MATERIAIS/          # Áudios e textos
│   ├── serie_a/
│   └── serie_b/
├── IMAGENS/            # Banco de imagens
│   └── UTILIZADAS/     # Imagens já usadas (automático)
├── SAIDA/              # Vídeos renderizados
└── EFEITOS/
    ├── musica.mp3
    ├── overlay.mp4
    └── VSLs/
        └── VSL_portugues.mp4
```

### Qualidade das Imagens

- **Resolução mínima**: 1920x1080 para vídeos 1080p
- **Formato**: JPG (menor tamanho) ou PNG (melhor qualidade)
- **Proporção**: Preferir 16:9 para evitar crop

### Qualidade do Áudio

- **Formato**: MP3 (128-192 kbps) ou WAV (melhor qualidade)
- **Duração**: Sem limite, mas vídeos muito longos demoram mais
- **Volume**: Normalize o áudio antes (evite clipping)

### Performance

**Para processar muitos vídeos:**
1. Use modo lote (processa tudo de uma vez)
2. Configure 2-3 vídeos paralelos (se CPU potente)
3. Use resolução 720p para testes, 1080p para final
4. Processe durante a noite (videos longos)

**Para processar rápido:**
1. Reduza "Imagens por Vídeo" (30-40 em vez de 50)
2. Reduza "Duração da Imagem" (6s em vez de 8s)
3. Desative overlay (economiza processamento)
4. Use menos threads (4 em vez de 6)

### Legendas

**Para melhor legibilidade:**
- Use fonte grande (48-60px)
- Contraste alto (branco com borda preta)
- Posição inferior central (8) ou inferior esquerdo (7)
- Ative sombra para destacar

**Para efeito profissional:**
- Use karaokê para destaque palavra por palavra
- Ajuste timing manualmente no SRT (se importar)
- Use fonte sem serifa (Arial, Helvetica)

### TTS

**Para melhor qualidade:**
- Use DARKVI (melhor qualidade de voz)
- Escolha voz adequada ao conteúdo
- Revise texto antes (sem erros de digitação)
- Quebre textos muito longos em múltiplos arquivos

**Para economizar créditos:**
- Revise textos antes de gerar
- Use TALKIFY se tiver créditos limitados
- Gere áudios em lote (mais eficiente)

### VSL

**Para inserção automática:**
- Use palavras-chave claras no texto
- Mantenha VSLs organizados por idioma
- Teste detecção com palavras-chave simples primeiro

---

## 📝 RESUMO RÁPIDO

### Checklist Antes de Renderizar

- [ ] Pastas configuradas (entrada, saída, imagens)
- [ ] Imagens suficientes na pasta
- [ ] Áudios ou textos na pasta de materiais
- [ ] TTS configurado (se usar textos)
- [ ] Legendas configuradas (se usar)
- [ ] Música e overlay (opcional)
- [ ] Configurações de vídeo ajustadas
- [ ] Processamento paralelo configurado

### Comandos Úteis

**Verificar Python:**
```bash
python3 --version
```

**Verificar FFmpeg:**
```bash
ffmpeg -version
```

**Ativar ambiente virtual:**
```bash
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate    # Windows
```

**Instalar dependências manualmente:**
```bash
pip install opencv-python numpy customtkinter assemblyai httpx
```

---

## 📞 SUPORTE

### Arquivos de Configuração

- `final_settings.json` - Configurações principais
- `subtitle_presets.json` - Presets de legendas
- `vsl_keywords.json` - Palavras-chave VSL

### Logs

A ferramenta exibe logs na interface durante o processamento. Em caso de erro:
1. Anote a mensagem de erro
2. Verifique os logs na interface
3. Consulte a seção "Solução de Problemas"

### Documentação das APIs

- **DARKVI**: Consulte `darkvi-api-doc.txt`
- **TALKIFY**: Consulte `talkify-api-doc.txt`

---

## 🎉 CONCLUSÃO

Esta ferramenta foi desenvolvida para automatizar a criação de vídeos profissionais com:
- **Eficiência**: Processamento em lote e paralelo
- **Qualidade**: Efeitos visuais e mixagem de áudio
- **Flexibilidade**: Múltiplas opções de personalização
- **Automação**: TTS, legendas automáticas, VSL inteligente

**Boa renderização! 🚀**

---

*Versão do Manual: 1.0*  
*Última atualização: 2025*




