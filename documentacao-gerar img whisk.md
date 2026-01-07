# 📘 Documentação Técnica - Sistema de Geração de Imagens

**Versão:** 2.0.0  
**Data:** Dezembro 2024  
**Propósito:** Documentação técnica completa do funcionamento atual do sistema para análise de implementação em servidor VPS com DOKPLOY

---

## 📑 Índice

1. [Visão Geral da Arquitetura](#visão-geral-da-arquitetura)
2. [Componentes Principais](#componentes-principais)
3. [Fluxo de Funcionamento](#fluxo-de-funcionamento)
4. [API e Integração Externa](#api-e-integração-externa)
5. [Gerenciamento de Tokens](#gerenciamento-de-tokens)
6. [Processamento de Imagens](#processamento-de-imagens)
7. [Estrutura de Dados](#estrutura-de-dados)
8. [Dependências e Requisitos](#dependências-e-requisitos)
9. [Pontos Críticos para Servidor](#pontos-críticos-para-servidor)
10. [Considerações para DOKPLOY](#considerações-para-dokploy)

---

## 🏗️ Visão Geral da Arquitetura

### Arquitetura Atual

O sistema é uma **aplicação desktop** desenvolvida em Python com interface gráfica (CustomTkinter) que funciona da seguinte forma:

```
┌─────────────────────────────────────────────────────────────┐
│                    INTERFACE GRÁFICA                         │
│              (interface_moderna.py)                         │
│  - CustomTkinter (GUI)                                      │
│  - Gerenciamento de tokens                                  │
│  - Controle de geração                                      │
│  - Upload de arquivos TXT                                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              GERADOR DE IMAGENS                              │
│      (gerador_imagens_automatico.py)                        │
│  - Classe: GeradorImagensAutomatico                         │
│  - Processamento de prompts                                │
│  - Comunicação com API                                      │
│  - Salvamento de imagens                                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              EXTRATOR DE TOKENS                             │
│          (token_extractor.py)                                │
│  - Extração de cookies do navegador                         │
│  - Busca de tokens do Google Whisk                          │
│  - Validação de tokens                                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              API EXTERNA                                     │
│    Google Imagen 3.5 (aisandbox-pa.googleapis.com)         │
│  - Endpoint: /v1/whisk:generateImage                         │
│  - Autenticação: Bearer Token                               │
└─────────────────────────────────────────────────────────────┘
```

### Características da Arquitetura

- **Tipo:** Aplicação Desktop (GUI)
- **Linguagem:** Python 3.7+
- **Interface:** CustomTkinter (baseado em Tkinter)
- **Processamento:** Síncrono com threading para UI
- **Armazenamento:** Sistema de arquivos local
- **Autenticação:** Tokens Bearer (JWT-like)

---

## 🧩 Componentes Principais

### 1. `interface_moderna.py` - Interface Gráfica

**Responsabilidades:**
- Interface gráfica do usuário (GUI)
- Gerenciamento visual de tokens
- Upload e processamento de arquivos TXT
- Controle de geração (iniciar/parar)
- Exibição de progresso e logs
- Seleção de pasta de saída

**Características Técnicas:**
- Framework: CustomTkinter 5.2.0+
- Threading: Usa `threading` para não bloquear UI durante geração
- Tema: Dark mode com paleta laranja (#FF6600)
- Estado: Mantém estado da aplicação (tokens, prompts, configurações)

**Métodos Principais:**
```python
- criar_interface()          # Cria toda a UI
- inicializar_gerador()      # Inicializa o gerador com tokens
- buscar_token_automatico()   # Extrai token do navegador
- adicionar_token_manual()   # Adiciona token manualmente
- processar_geracao()        # Inicia processamento em thread
- atualizar_progresso()      # Atualiza UI com progresso
```

### 2. `gerador_imagens_automatico.py` - Motor de Geração

**Classe Principal:** `GeradorImagensAutomatico`

**Responsabilidades:**
- Comunicação com API do Google Imagen
- Gerenciamento de múltiplos tokens (rotação automática)
- Processamento de prompts (manual ou arquivo TXT)
- Geração e salvamento de imagens
- Tratamento de erros e fallback de tokens
- Organização de arquivos (pastas por prompt ou única pasta)

**Características Técnicas:**
- Biblioteca HTTP: `requests` 2.31.0+
- Processamento de imagens: `Pillow` 10.0.0+
- Formato de saída: PNG
- Numeração: Sequencial global (1.png, 2.png, ...)
- Timeout: 90 segundos por requisição

**Métodos Principais:**
```python
- __init__()                    # Inicialização com tokens e configurações
- gerar_imagem()                # Faz requisição à API e retorna bytes
- processar_prompt_multiplo()   # Processa um prompt N vezes
- salvar_imagem()               # Salva imagem em disco
- _obter_proximo_token_valido() # Rotaciona tokens em caso de erro
- _marcar_token_como_invalido() # Marca token temporariamente inválido
```

**Fluxo de Geração:**
1. Recebe prompt e quantidade
2. Cria pasta (se `separar_por_pasta=True`)
3. Para cada iteração:
   - Chama `gerar_imagem(prompt)`
   - Se sucesso: salva com `salvar_imagem()`
   - Se erro: tenta próximo token automaticamente
4. Atualiza contadores e estatísticas

### 3. `token_extractor.py` - Extração de Tokens

**Responsabilidades:**
- Extração de cookies do navegador (Chrome, Firefox, Safari, Edge)
- Busca de token `__Secure-next-auth.session-token` do Google Whisk
- Extração de email associado ao token
- Validação de tokens

**Características Técnicas:**
- Biblioteca: `browser-cookie3` 0.19.0+
- Domínio: `labs.google`
- Cookie específico: `__Secure-next-auth.session-token`
- API de sessão: `https://labs.google/fx/api/auth/session`

**Métodos Principais:**
```python
- buscar_token_whisk()              # Busca token do navegador
- extrair_cookie_do_navegador()     # Extrai cookie específico
- buscar_email_da_api()             # Busca email da API de sessão
- validar_token()                   # Valida formato do token
- listar_navegadores_disponiveis() # Lista navegadores disponíveis
```

**⚠️ IMPORTANTE:** Esta funcionalidade **NÃO funcionará em servidor** sem navegador instalado e acesso aos cookies do usuário.

---

## 🔄 Fluxo de Funcionamento

### Fluxo Completo de Geração

```
1. USUÁRIO INICIA APLICAÇÃO
   │
   ├─> Carrega tokens de tokens_bearer.json (se existir)
   │
   └─> Inicializa interface gráfica

2. USUÁRIO CONFIGURA TOKENS
   │
   ├─> Opção A: Busca automática do navegador
   │   └─> token_extractor.py extrai cookie
   │       └─> Salva em tokens_bearer.json
   │
   └─> Opção B: Adiciona token manualmente
       └─> Salva em tokens_bearer.json

3. USUÁRIO PREPARA PROMPTS
   │
   ├─> Opção A: Entrada manual
   │   └─> Digita prompt + quantidade
   │
   └─> Opção B: Upload arquivo TXT
       └─> parsear_prompts_multilinha() processa arquivo
           └─> Prompts separados por linhas em branco

4. USUÁRIO INICIA GERAÇÃO
   │
   ├─> Interface cria thread separada
   │
   └─> Thread chama processar_prompt_multiplo_gui()

5. PARA CADA PROMPT:
   │
   ├─> Cria pasta (se separar_por_pasta=True)
   │
   └─> PARA CADA ITERAÇÃO (quantidade):
       │
       ├─> gerar_imagem(prompt)
       │   │
       │   ├─> Seleciona token (rotação automática)
       │   │
       │   ├─> Faz POST para API
       │   │   POST https://aisandbox-pa.googleapis.com/v1/whisk:generateImage
       │   │   Headers: Authorization: Bearer {token}
       │   │   Body: JSON com prompt, seed, configurações
       │   │
       │   ├─> Se sucesso (200):
       │   │   └─> Decodifica base64 da resposta
       │   │       └─> Retorna bytes da imagem
       │   │
       │   └─> Se erro (401, 403, 429):
       │       └─> Marca token como inválido
       │           └─> Tenta próximo token automaticamente
       │
       ├─> Se imagem gerada:
       │   │
       │   ├─> salvar_imagem(image_bytes, nome_arquivo)
       │   │   │
       │   │   ├─> Converte bytes para PIL Image
       │   │   │
       │   │   └─> Salva como PNG na pasta destino
       │   │
       │   └─> Atualiza contador global
       │
       └─> Atualiza progresso na UI (via callback)

6. FINALIZAÇÃO
   │
   ├─> Mostra estatísticas finais
   │
   └─> Imagens salvas em IMAGENS/ (ou pasta personalizada)
```

### Fluxo de Rotação de Tokens

```
Token Atual → Requisição → Resposta
                    │
        ┌───────────┴───────────┐
        │                       │
    Sucesso (200)          Erro (401/403/429)
        │                       │
        │                   Marca como inválido
        │                       │
        │                   Próximo token
        │                       │
        │                   Nova requisição
        │                       │
        └───────────┬───────────┘
                    │
            Continua processamento
```

---

## 🌐 API e Integração Externa

### Endpoint da API

**URL Base:** `https://aisandbox-pa.googleapis.com`

**Endpoint:** `/v1/whisk:generateImage`

**Método:** `POST`

**Autenticação:** Bearer Token (JWT-like)

### Estrutura da Requisição

**Headers:**
```json
{
  "Authorization": "Bearer {token}",
  "Content-Type": "application/json; charset=UTF-8"
}
```

**Body (JSON):**
```json
{
  "clientContext": {
    "workflowId": "c4dd24a1-c7e8-4057-9c25-1d2635673bd1",
    "tool": "BACKBONE",
    "sessionId": ";1757860178254"
  },
  "imageModelSettings": {
    "imageModel": "IMAGEN_3_5",
    "aspectRatio": "IMAGE_ASPECT_RATIO_LANDSCAPE"
  },
  "mediaCategory": "MEDIA_CATEGORY_BOARD",
  "prompt": "{prompt_do_usuario}",
  "seed": {numero_aleatorio_1_a_1000000}
}
```

**Parâmetros Importantes:**
- `imageModel`: "IMAGEN_3_5" (fixo)
- `aspectRatio`: "IMAGE_ASPECT_RATIO_LANDSCAPE" (pode ser alterado)
- `seed`: Número aleatório para variação
- `prompt`: Texto do usuário

### Estrutura da Resposta

**Sucesso (200):**
```json
{
  "imagePanels": [
    {
      "generatedImages": [
        {
          "encodedImage": "{base64_string_da_imagem}"
        }
      ]
    }
  ]
}
```

**Erros Comuns:**
- `401`: Token inválido ou expirado
- `403`: Token sem permissão ou créditos esgotados
- `429`: Rate limit excedido
- `500`: Erro interno do servidor

### Tratamento de Erros

O sistema implementa:
1. **Retry automático** com próximo token em caso de erro
2. **Marcação temporária** de tokens inválidos
3. **Reset automático** se todos os tokens falharem
4. **Timeout** de 90 segundos por requisição
5. **Logging detalhado** de todos os erros

---

## 🔐 Gerenciamento de Tokens

### Armazenamento de Tokens

**Arquivo:** `tokens_bearer.json`

**Formato:**
```json
{
  "tokens": [
    {
      "token": "eyJhbGciOiJSUzI1NiIs...",
      "email": "usuario@example.com"
    },
    {
      "token": "eyJhbGciOiJSUzI1NiIs...",
      "email": null
    }
  ],
  "total": 2,
  "ultima_atualizacao": "2024-12-15T10:30:00"
}
```

### Sistema de Rotação

**Características:**
- Suporta múltiplos tokens simultaneamente
- Rotação automática em caso de erro
- Estatísticas por token (sucessos, erros)
- Marcação temporária de tokens inválidos
- Reset automático após tentar todos

**Lógica de Seleção:**
1. Filtra tokens válidos (não marcados como inválidos)
2. Se todos inválidos, reseta e tenta todos novamente
3. Rotaciona para próximo token após cada requisição
4. Em caso de erro, marca token atual como inválido e tenta próximo

### Validação de Tokens

**Formato Esperado:**
- String não vazia
- Mínimo 50 caracteres
- Geralmente começa com "eyJ" (JWT)

**Validação Atual:**
- Verifica se não está vazio
- Verifica tamanho mínimo (50 caracteres)
- Não valida assinatura JWT (apenas formato básico)

---

## 🖼️ Processamento de Imagens

### Geração de Imagens

**Processo:**
1. Recebe prompt do usuário
2. Faz requisição à API
3. Recebe imagem em base64
4. Decodifica base64 para bytes
5. Converte bytes para PIL Image
6. Salva como PNG

**Bibliotecas Utilizadas:**
- `base64`: Decodificação
- `PIL (Pillow)`: Manipulação de imagens
- `io.BytesIO`: Buffer de memória

### Organização de Arquivos

**Modo 1: Separar por Pasta (padrão)**
```
IMAGENS/
├── nome_do_prompt_1/
│   ├── 1.png
│   ├── 2.png
│   └── 3.png
└── nome_do_prompt_2/
    ├── 4.png
    └── 5.png
```

**Modo 2: Mesma Pasta**
```
IMAGENS/
├── 1.png
├── 2.png
├── 3.png
├── 4.png
└── 5.png
```

**Limpeza de Nomes:**
- Remove caracteres especiais
- Substitui espaços por underscores
- Limita a 50 caracteres
- Converte para lowercase

### Numeração Sequencial

- Contador global que incrementa a cada imagem
- Não reinicia entre prompts
- Formato: `{contador}.png`
- Inicia em 1

---

## 💾 Estrutura de Dados

### Prompts

**Formato de Entrada (TXT):**
```
Prompt linha 1
Prompt linha 2
Prompt linha 3

Segundo prompt
com múltiplas linhas

Terceiro prompt
```

**Processamento:**
- Prompts separados por uma ou mais linhas em branco
- Cada prompt pode ter múltiplas linhas
- Espaços preservados (exceto no início/fim)

**Estrutura Interna:**
```python
prompts = [
    {
        "prompt": "Texto do prompt completo",
        "quantidade": 3
    },
    ...
]
```

### Configurações

**Variáveis de Instância:**
```python
self.tokens = [{"token": str, "email": str|None}, ...]
self.pasta_png = Path("IMAGENS")
self.api_url = "https://aisandbox-pa.googleapis.com/v1/whisk:generateImage"
self.imagens_geradas = 0
self.erros_ocorridos = 0
self.contador_global = 1
self.separar_por_pasta = True
```

### Estatísticas

**Estrutura:**
```python
{
    "imagens_geradas": 0,
    "erros_ocorridos": 0,
    "contador_global": 1,
    "pasta_destino": "/caminho/para/IMAGENS",
    "remover_fundo_ativo": False
}
```

**Estatísticas por Token:**
```python
{
    "token_string": {
        "sucessos": 0,
        "erros": 0,
        "ultimo_erro": None
    }
}
```

---

## 📦 Dependências e Requisitos

### Dependências Python

**Core:**
- `requests>=2.31.0` - Requisições HTTP
- `Pillow>=10.0.0` - Manipulação de imagens
- `customtkinter>=5.2.0` - Interface gráfica (apenas desktop)

**Opcionais:**
- `rembg>=2.0.50` - Remoção de fundo (desabilitado atualmente)
- `browser-cookie3>=0.19.0` - Extração de cookies (apenas desktop)
- `selenium>=4.15.0` - Automação de navegador (não usado atualmente)

**Biblioteca Padrão:**
- `os`, `sys`, `json`, `base64`, `random`, `datetime`
- `typing`, `pathlib`, `re`, `logging`, `io`, `threading`

### Requisitos de Sistema

**Desktop (Atual):**
- Python 3.7+
- Interface gráfica (X11/Wayland no Linux, GUI no macOS/Windows)
- Navegador instalado (para extração de tokens)
- Acesso ao sistema de arquivos
- Conexão com internet

**Memória:**
- Mínimo: 2 GB RAM
- Recomendado: 4 GB+ RAM

**Disco:**
- Aplicação: ~100 MB
- Dependências: ~200 MB
- Imagens geradas: Variável (depende do uso)

**Rede:**
- Conexão estável com internet
- Acesso a `aisandbox-pa.googleapis.com`
- Acesso a `labs.google` (para extração de tokens)

---

## ⚠️ Pontos Críticos para Servidor

### 1. Interface Gráfica (GUI)

**Problema:**
- `customtkinter` requer interface gráfica (X11/Wayland)
- Servidores geralmente não têm display

**Solução Necessária:**
- Remover completamente a GUI
- Criar API REST ou CLI
- Usar framework web (Flask/FastAPI) para interface

### 2. Extração de Tokens do Navegador

**Problema:**
- `browser-cookie3` requer acesso aos cookies do navegador do usuário
- Servidor não tem acesso aos cookies do cliente
- Navegador não está disponível no servidor

**Solução Necessária:**
- Interface web para usuário inserir token manualmente
- Ou API para receber token do cliente
- Armazenar tokens no banco de dados (não em arquivo JSON)

### 3. Sistema de Arquivos Local

**Problema:**
- Atualmente salva em `IMAGENS/` no sistema de arquivos local
- Em servidor, precisa de storage persistente
- Múltiplos usuários podem gerar conflitos

**Solução Necessária:**
- Usar storage remoto (S3, Google Cloud Storage, etc.)
- Ou banco de dados para metadados
- Sistema de namespaces por usuário/sessão

### 4. Processamento Síncrono

**Problema:**
- Processamento atual é síncrono (bloqueia thread)
- Para múltiplos usuários, precisa ser assíncrono

**Solução Necessária:**
- Usar filas (Celery, RQ, etc.)
- Processamento em background
- WebSockets ou polling para atualizar progresso

### 5. Threading para UI

**Problema:**
- Threading atual é apenas para não bloquear UI
- Em servidor, precisa de workers/processos separados

**Solução Necessária:**
- Workers assíncronos
- Sistema de filas
- Gerenciamento de processos

### 6. Armazenamento de Estado

**Problema:**
- Estado atual é em memória (variáveis de instância)
- Em servidor stateless, estado precisa ser persistido

**Solução Necessária:**
- Banco de dados para estado
- Redis para cache/sessões
- Armazenar progresso e resultados

### 7. Segurança

**Problema:**
- Tokens armazenados em arquivo JSON (não seguro)
- Sem autenticação de usuários
- Sem validação de entrada

**Solução Necessária:**
- Criptografar tokens no banco de dados
- Autenticação de usuários (JWT, OAuth, etc.)
- Validação e sanitização de inputs
- Rate limiting por usuário

---

## 🚀 Considerações para DOKPLOY

### DOKPLOY - Visão Geral

DOKPLOY é uma plataforma de deploy similar ao Heroku, que permite:
- Deploy de aplicações via Git
- Build automático
- Gerenciamento de containers
- Variáveis de ambiente
- Logs centralizados

### Adaptações Necessárias

#### 1. Remover GUI

**Ação:**
- Remover `interface_moderna.py` completamente
- Criar API REST com Flask ou FastAPI
- Criar interface web (HTML/JS) ou manter apenas API

**Exemplo de Estrutura:**
```
app/
├── api/
│   ├── routes.py          # Endpoints REST
│   └── models.py          # Modelos de dados
├── core/
│   ├── gerador.py         # Lógica de geração (adaptada)
│   └── storage.py         # Gerenciamento de storage
├── web/
│   └── static/            # Frontend (opcional)
└── app.py                # Aplicação principal
```

#### 2. API REST

**Endpoints Sugeridos:**
```
POST   /api/v1/generate        # Gerar imagem
GET    /api/v1/jobs/{id}       # Status do job
GET    /api/v1/images/{id}     # Download da imagem
POST   /api/v1/tokens           # Adicionar token
GET    /api/v1/tokens           # Listar tokens
DELETE /api/v1/tokens/{id}      # Remover token
POST   /api/v1/upload           # Upload arquivo TXT
```

#### 3. Processamento Assíncrono

**Opções:**
- **Celery + Redis**: Sistema de filas robusto
- **RQ (Redis Queue)**: Mais simples que Celery
- **Background Tasks (FastAPI)**: Para casos simples

**Fluxo:**
```
Cliente → API → Fila → Worker → Storage → Notificação
```

#### 4. Storage

**Opções:**
- **S3/Google Cloud Storage**: Storage de objetos
- **Volume persistente DOKPLOY**: Se disponível
- **Banco de dados**: Para metadados (PostgreSQL)

**Estrutura:**
```
storage/
├── {user_id}/
│   ├── {job_id}/
│   │   ├── 1.png
│   │   ├── 2.png
│   │   └── metadata.json
```

#### 5. Banco de Dados

**Tabelas Sugeridas:**
```sql
users (id, email, created_at)
tokens (id, user_id, token_encrypted, email, created_at)
jobs (id, user_id, status, prompt, quantidade, created_at)
images (id, job_id, filename, path, created_at)
```

#### 6. Variáveis de Ambiente

**Configurações:**
```env
DATABASE_URL=postgresql://...
REDIS_URL=redis://...
STORAGE_TYPE=s3|local|gcs
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
S3_BUCKET=...
API_TIMEOUT=90
MAX_TOKENS_PER_USER=10
```

#### 7. Dockerfile

**Estrutura:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000"]
```

#### 8. Estrutura de Projeto para DOKPLOY

```
projeto/
├── app.py                 # Aplicação principal (Flask/FastAPI)
├── requirements.txt       # Dependências (sem customtkinter)
├── Dockerfile            # Container Docker
├── .env.example          # Exemplo de variáveis
├── api/
│   ├── __init__.py
│   ├── routes.py         # Endpoints
│   └── models.py         # Modelos
├── core/
│   ├── __init__.py
│   ├── gerador.py        # Adaptado de gerador_imagens_automatico.py
│   ├── storage.py        # Gerenciamento de storage
│   └── tokens.py         # Gerenciamento de tokens
├── workers/
│   ├── __init__.py
│   └── image_worker.py   # Worker para processar imagens
└── static/               # Frontend (opcional)
    ├── index.html
    └── app.js
```

### Checklist de Migração

- [ ] Remover dependências de GUI (customtkinter)
- [ ] Criar API REST (Flask/FastAPI)
- [ ] Implementar processamento assíncrono (Celery/RQ)
- [ ] Configurar storage (S3/local)
- [ ] Configurar banco de dados (PostgreSQL)
- [ ] Implementar autenticação de usuários
- [ ] Criptografar tokens no banco
- [ ] Implementar rate limiting
- [ ] Criar Dockerfile
- [ ] Configurar variáveis de ambiente
- [ ] Implementar logs estruturados
- [ ] Testes de carga
- [ ] Monitoramento e alertas

---

## 📊 Resumo Técnico

### Arquitetura Atual
- **Tipo:** Desktop Application (GUI)
- **Interface:** CustomTkinter
- **Processamento:** Síncrono com threading para UI
- **Storage:** Sistema de arquivos local
- **Tokens:** Arquivo JSON local

### Arquitetura Necessária para Servidor
- **Tipo:** Web Application (API + Frontend)
- **Interface:** API REST + Web UI
- **Processamento:** Assíncrono com filas
- **Storage:** Cloud Storage ou volume persistente
- **Tokens:** Banco de dados criptografado

### Principais Mudanças
1. **GUI → API REST**
2. **Síncrono → Assíncrono**
3. **Arquivo local → Banco de dados**
4. **Sistema de arquivos → Cloud Storage**
5. **Sem autenticação → Com autenticação**
6. **Single user → Multi-user**

---

## 🔍 Pontos de Atenção

### Performance
- API do Google pode ter rate limits
- Processamento de imagens pode ser lento
- Múltiplos usuários simultâneos precisam de workers suficientes

### Custos
- Storage de imagens pode crescer rapidamente
- Requisições à API podem ter custos
- Workers assíncronos consomem recursos

### Segurança
- Tokens são sensíveis (criptografar)
- Validar todos os inputs
- Rate limiting por usuário
- Logs não devem expor tokens

### Escalabilidade
- Workers podem escalar horizontalmente
- Storage precisa ser escalável
- Banco de dados precisa de índices adequados

---

**Documentação criada em:** Dezembro 2024  
**Versão do sistema documentado:** 2.0.0  
**Próximos passos:** Análise de viabilidade e planejamento de migração
