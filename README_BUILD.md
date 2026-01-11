# Como Gerar Executável .exe do RenderX

Este guia explica como gerar um executável .exe do RenderX usando PyInstaller.

## Pré-requisitos

1. **Python 3.10 ou 3.11** instalado
2. **Todas as dependências** instaladas (veja `requirements.txt`)
3. **PyInstaller** será instalado automaticamente se não estiver presente

## Método Rápido

### Windows
```batch
BUILD_EXE.bat
```

### macOS/Linux
```bash
./BUILD_EXE.sh
```

## Método Manual

### 1. Instalar PyInstaller
```bash
pip install pyinstaller
```

### 2. Executar o script de build
```bash
python build_exe.py
```

## O que será gerado

O script criará:
- **`dist/RenderX.exe`** (Windows) ou **`dist/RenderX`** (macOS/Linux) - O executável principal
- **Arquivos de configuração** copiados para a pasta `dist/`

## Estrutura após o build

```
dist/
  ├── RenderX.exe (ou RenderX no macOS/Linux)
  ├── keys_assembly.json
  ├── whisk_keys.example.json
  ├── subtitle_presets.json
  ├── image_prompts.json
  ├── vsl_keywords.json
  ├── opencv_settings.json
  └── final_settings.example.json
```

## Importante

⚠️ **FFmpeg**: O executável NÃO inclui FFmpeg. É necessário que o FFmpeg esteja:
- Instalado no sistema
- Adicionado ao PATH do sistema
- Ou colocado na mesma pasta do executável

⚠️ **Recursos**: A pasta `EFEITOS/` será incluída no executável, mas você pode precisar copiá-la manualmente para o mesmo diretório do executável se usar recursos externos.

⚠️ **Primeira Execução**: Na primeira execução, o executável pode demorar um pouco mais para iniciar enquanto extrai os arquivos temporários.

## Tamanho do Executável

O executável gerado terá aproximadamente:
- **150-300 MB** (incluindo todas as bibliotecas: OpenCV, NumPy, CustomTkinter, etc.)

## Distribuição

Para distribuir o RenderX:
1. Copie a pasta `dist/` inteira
2. Ou copie apenas o executável e os arquivos de configuração necessários
3. Certifique-se de que o FFmpeg está disponível no sistema destino

## Solução de Problemas

### Erro: "ModuleNotFoundError"
- Verifique se todas as dependências estão instaladas
- Execute `pip install -r requirements.txt`

### Executável não inicia
- Verifique se o FFmpeg está instalado e no PATH
- Tente executar com console (`--console` em vez de `--windowed`) para ver erros

### Tamanho muito grande
- O executável inclui todas as bibliotecas (OpenCV, NumPy, etc.)
- Considere usar `--onedir` em vez de `--onefile` para distribuir como pasta

### Recursos não encontrados
- Certifique-se de que a pasta `EFEITOS/` está no mesmo diretório do executável
- Ou copie os recursos necessários para o diretório do executável
