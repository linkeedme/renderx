# ⚡ GUIA RÁPIDO - EDITOR DE VÍDEO

## 🚀 INÍCIO RÁPIDO

### 1. Iniciar a Ferramenta

**macOS/Linux:**
```bash
./INICIAR.sh
```

**Windows:**
```cmd
INICIAR_FINAL.bat
```

### 2. Configuração Mínima

1. **Pasta de Materiais**: Onde estão seus áudios/textos
2. **Pasta de Saída**: Onde salvar os vídeos
3. **Pasta de Imagens**: Banco de imagens para os vídeos
4. Clique em **"Iniciar Processamento"**

---

## 📋 CHECKLIST RÁPIDO

### Antes de Renderizar

- [ ] Pastas configuradas
- [ ] Imagens na pasta (mínimo 50)
- [ ] Áudios/textos na pasta de materiais
- [ ] Configurações básicas ajustadas

### Configurações Recomendadas (Iniciante)

- **Resolução**: 720p
- **Vídeos Paralelos**: 1
- **Imagens por Vídeo**: 50
- **Duração da Imagem**: 8 segundos
- **Zoom**: 1.15 (zoom_in)

---

## 🎯 FUNCIONALIDADES PRINCIPAIS

### TTS (Text-to-Speech)

**DARKVI:**
1. Obter token em https://darkvi.com
2. Selecionar "DARKVI" como provider
3. Colar token
4. Listar e selecionar voz
5. Ativar "TTS Habilitado"

**TALKIFY:**
1. Obter token em https://talkifydev.com
2. Selecionar "TALKIFY"
3. Colar token e Voice ID
4. Ativar TTS

### Legendas

**Método SRT:**
- Criar arquivo `.srt` com mesmo nome do áudio
- Exemplo: `video.mp3` → `video.srt`

**Método AssemblyAI:**
- Configurar chave da API
- Sistema transcreve automaticamente

**Personalização:**
- Fonte, tamanho, cores
- Posição (1-9)
- Efeitos (borda, sombra, karaokê)

### VSL (Video Sales Letter)

1. Ativar "Usar VSL"
2. Configurar pasta: `EFEITOS/VSLs/`
3. Adicionar palavras-chave em `vsl_keywords.json`
4. Sistema detecta e insere automaticamente

---

## ⚙️ CONFIGURAÇÕES COMUNS

### Resolução
- **720p**: Mais rápido, boa qualidade
- **1080p**: Melhor qualidade, mais lento

### Zoom
- **zoom_in**: Zoom para dentro (padrão)
- **zoom_out**: Zoom para fora
- **Escala**: 1.0 (sem zoom) a 1.5 (zoom 50%)

### Processamento
- **Vídeos Paralelos**: 1-4 (mais = mais rápido, mais recursos)
- **Threads por Vídeo**: 4-8 (padrão: 6)

### Áudio
- **Volume da Música**: 0.1-0.3 (baixo) para não competir com narração
- **Opacidade Overlay**: 0.2-0.4 (sutil)

---

## 🎨 POSIÇÕES DE LEGENDAS

```
1  2  3    ← Superior
4  5  6    ← Centro
7  8  9    ← Inferior
```

**Recomendado**: 8 (inferior central) ou 7 (inferior esquerdo)

---

## 📁 ESTRUTURA DE PASTAS

```
PROJETO/
├── MATERIAIS/          # Áudios e textos aqui
├── IMAGENS/            # Banco de imagens
│   └── UTILIZADAS/     # Auto (imagens usadas)
├── SAIDA/              # Vídeos renderizados
└── EFEITOS/
    ├── musica.mp3
    ├── overlay.mp4
    └── VSLs/
```

---

## 🔧 SOLUÇÃO RÁPIDA DE PROBLEMAS

| Problema | Solução |
|----------|---------|
| Python não encontrado | Instalar Python 3.8+ |
| FFmpeg não encontrado | `brew install ffmpeg` (macOS) |
| Token inválido | Verificar token na API |
| Imagens insuficientes | Adicionar mais imagens ou reduzir "Imagens por Vídeo" |
| Vídeo sem áudio | Verificar formato do arquivo |
| Legendas não aparecem | Ativar "Usar Legendas" e verificar cores |
| Processamento lento | Reduzir vídeos paralelos e resolução |

---

## 💡 DICAS RÁPIDAS

### Performance
- Use 720p para testes
- 1 vídeo paralelo = mais estável
- Reduza imagens por vídeo se lento

### Qualidade
- Imagens 1920x1080 para 1080p
- Áudio MP3 128-192 kbps
- Música volume baixo (0.2)

### Legendas
- Fonte grande (48-60px)
- Contraste alto (branco/preto)
- Posição inferior (7 ou 8)

### TTS
- Revise texto antes
- DARKVI = melhor qualidade
- Quebre textos longos

---

## 📞 ARQUIVOS IMPORTANTES

- `final_settings.json` - Configurações salvas
- `subtitle_presets.json` - Presets de legendas
- `vsl_keywords.json` - Palavras-chave VSL
- `MANUAL_DO_USUARIO.md` - Manual completo

---

## 🎬 FLUXO DE TRABALHO

1. **Preparar materiais**
   - Áudios/textos na pasta de materiais
   - Imagens na pasta de imagens

2. **Configurar**
   - Pastas
   - Resolução e zoom
   - TTS (se usar textos)
   - Legendas (se usar)

3. **Renderizar**
   - Clicar em "Iniciar Processamento"
   - Aguardar conclusão

4. **Verificar**
   - Vídeos na pasta de saída
   - Verificar qualidade e sincronização

---

**Para mais detalhes, consulte o MANUAL_DO_USUARIO.md**




