# -*- coding: utf-8 -*-
"""
Prompt Manager - Gerenciador de Prompts Externos com Cache
==========================================================
Carrega prompts de arquivos externos (MD/TXT) e gerencia cache
para evitar regeneração desnecessária de imagens.

Suporta seleção de prompts predefinidos do arquivo image_prompts.json
para diferentes canais/estilos.
"""

import os
import json
import hashlib
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

# Caminho do arquivo de prompts
SCRIPT_DIR = Path(__file__).parent.absolute()
IMAGE_PROMPTS_FILE = SCRIPT_DIR / "image_prompts.json"


def load_image_prompts(file_path: str = None) -> Dict:
    """
    Carrega prompts do arquivo image_prompts.json.
    
    Args:
        file_path: Caminho do arquivo (opcional)
        
    Returns:
        Dicionário com prompts
    """
    path = Path(file_path) if file_path else IMAGE_PROMPTS_FILE
    
    if not path.exists():
        return {"prompts": [], "default_prompt": ""}
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Erro ao carregar image_prompts.json: {e}")
        return {"prompts": [], "default_prompt": ""}


def save_image_prompts(data: Dict, file_path: str = None):
    """
    Salva prompts no arquivo image_prompts.json.
    
    Args:
        data: Dicionário com prompts
        file_path: Caminho do arquivo (opcional)
    """
    path = Path(file_path) if file_path else IMAGE_PROMPTS_FILE
    
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[ERROR] Erro ao salvar image_prompts.json: {e}")


def get_prompt_list() -> List[Tuple[str, str]]:
    """
    Retorna lista de prompts para exibição.
    
    Returns:
        Lista de tuplas (id, nome)
    """
    data = load_image_prompts()
    prompts = []
    
    for p in data.get("prompts", []):
        prompt_id = p.get("id", "")
        name = p.get("name", prompt_id)
        if prompt_id:
            prompts.append((prompt_id, name))
    
    return prompts


def get_prompt_names() -> List[str]:
    """
    Retorna lista de nomes dos prompts para combobox.
    
    Returns:
        Lista de nomes
    """
    prompts = get_prompt_list()
    return [name for _, name in prompts]


def get_prompt_by_id(prompt_id: str) -> Optional[Dict]:
    """
    Retorna prompt pelo ID.
    
    Args:
        prompt_id: ID do prompt
        
    Returns:
        Dicionário do prompt ou None
    """
    data = load_image_prompts()
    
    for p in data.get("prompts", []):
        if p.get("id") == prompt_id:
            return p
    
    return None


def get_prompt_by_name(name: str) -> Optional[Dict]:
    """
    Retorna prompt pelo nome.
    
    Args:
        name: Nome do prompt
        
    Returns:
        Dicionário do prompt ou None
    """
    data = load_image_prompts()
    
    for p in data.get("prompts", []):
        if p.get("name") == name:
            return p
    
    return None


def get_default_prompt_id() -> str:
    """
    Retorna ID do prompt padrão.
    
    Returns:
        ID do prompt padrão
    """
    data = load_image_prompts()
    return data.get("default_prompt", "")


def set_default_prompt(prompt_id: str):
    """
    Define o prompt padrão.
    
    Args:
        prompt_id: ID do prompt
    """
    data = load_image_prompts()
    data["default_prompt"] = prompt_id
    save_image_prompts(data)


def build_prompt_from_template(prompt_id: str, srt_text: str = "", use_srt_text: bool = False) -> str:
    """
    Constrói prompt final a partir de um template.
    
    Args:
        prompt_id: ID do prompt
        srt_text: Texto do SRT (IGNORADO por padrão para evitar texto nas imagens)
        use_srt_text: Se True, inclui o texto do SRT no prompt (NÃO RECOMENDADO)
        
    Returns:
        Prompt formatado
    """
    prompt_data = get_prompt_by_id(prompt_id)
    
    if not prompt_data:
        # Fallback para prompt espiritual padrão SEM texto do SRT
        return "Cinematic photorealistic scene, 35mm film texture, teal and amber color grade, volumetric lighting, mystical atmosphere, abstract geometric symbols, no text, no letters, no words, no writing, clean image"
    
    prefix = prompt_data.get("prefix", "")
    suffix = prompt_data.get("suffix", "")
    template = prompt_data.get("template", "")
    quality_boost = prompt_data.get("quality_boost", "")
    
    # Construir prompt: template + prefix + suffix + quality_boost
    # NÃO incluir texto do SRT para evitar que apareça nas imagens
    parts = []
    
    if template:
        parts.append(template.strip())
    
    if prefix:
        parts.append(prefix.strip())
    
    # REMOVIDO: Não incluir texto do SRT - isso faz o modelo gerar texto na imagem!
    # Se realmente precisar do contexto do SRT, use use_srt_text=True
    if use_srt_text and srt_text:
        # Extrair apenas palavras-chave temáticas (máximo 5 palavras)
        import re
        words = re.findall(r'\b[a-zA-ZÀ-ÿ]{4,}\b', srt_text)
        # Filtrar palavras comuns que não agregam valor visual
        stop_words = {'that', 'this', 'with', 'from', 'have', 'been', 'were', 'will', 'your', 'they', 
                      'como', 'para', 'você', 'isso', 'esta', 'esse', 'uma', 'que', 'por', 'mais',
                      'sont', 'vous', 'cette', 'dans', 'pour', 'avec', 'les', 'des', 'une'}
        keywords = [w for w in words[:20] if w.lower() not in stop_words][:3]
        if keywords:
            parts.append(f"theme: {', '.join(keywords)}")
    
    if suffix:
        parts.append(suffix.strip())
    
    if quality_boost:
        parts.append(quality_boost.strip())
    
    # Adicionar instrução explícita para evitar texto
    parts.append("absolutely no text, no letters, no words, no writing in the image")
    
    return ", ".join(parts)


def get_negative_prompt(prompt_id: str) -> str:
    """
    Obtém o negative prompt para um template específico.
    
    Args:
        prompt_id: ID do prompt
        
    Returns:
        Negative prompt string
    """
    data = load_image_prompts()
    prompt_data = get_prompt_by_id(prompt_id)
    
    # Negative prompt específico do template
    specific_negative = ""
    if prompt_data:
        specific_negative = prompt_data.get("negative_prompt", "")
    
    # Negative prompt global
    global_negative = data.get("global_negative_prompt", "")
    
    # Combinar ambos
    negatives = []
    if specific_negative:
        negatives.append(specific_negative.strip())
    if global_negative:
        negatives.append(global_negative.strip())
    
    return ", ".join(negatives) if negatives else "text, letters, words, writing, watermark, blurry, low quality"


def get_full_prompt_config(prompt_id: str, srt_text: str) -> Dict:
    """
    Retorna configuração completa do prompt incluindo negative prompt.
    
    Args:
        prompt_id: ID do prompt
        srt_text: Texto do SRT
        
    Returns:
        Dict com 'prompt' e 'negative_prompt'
    """
    return {
        "prompt": build_prompt_from_template(prompt_id, srt_text),
        "negative_prompt": get_negative_prompt(prompt_id)
    }


def get_style_modifiers(prompt_id: str) -> List[str]:
    """
    Retorna os modificadores de estilo para um prompt.
    
    Args:
        prompt_id: ID do prompt
        
    Returns:
        Lista de modificadores de estilo
    """
    prompt_data = get_prompt_by_id(prompt_id)
    if prompt_data:
        return prompt_data.get("style_modifiers", [])
    return []


def add_prompt(prompt_id: str, name: str, description: str = "", 
               template: str = "", prefix: str = "", suffix: str = ""):
    """
    Adiciona um novo prompt ao arquivo.
    
    Args:
        prompt_id: ID único do prompt
        name: Nome para exibição
        description: Descrição
        template: Template base
        prefix: Prefixo adicionado antes do texto
        suffix: Sufixo adicionado após o texto
    """
    data = load_image_prompts()
    
    # Verificar se já existe
    for p in data.get("prompts", []):
        if p.get("id") == prompt_id:
            return  # Já existe
    
    if "prompts" not in data:
        data["prompts"] = []
    
    data["prompts"].append({
        "id": prompt_id,
        "name": name,
        "description": description,
        "template": template,
        "prefix": prefix,
        "suffix": suffix
    })
    
    save_image_prompts(data)


def remove_prompt(prompt_id: str) -> bool:
    """
    Remove um prompt do arquivo.
    
    Args:
        prompt_id: ID do prompt
        
    Returns:
        True se removido
    """
    data = load_image_prompts()
    prompts = data.get("prompts", [])
    
    for i, p in enumerate(prompts):
        if p.get("id") == prompt_id:
            del prompts[i]
            data["prompts"] = prompts
            save_image_prompts(data)
            return True
    
    return False


@dataclass
class PromptConfig:
    """Configuração do prompt."""
    prompt_file: str = ""           # Caminho do arquivo de prompt
    base_prompt: str = ""           # Prompt base carregado
    style: str = "spiritual"        # Estilo: spiritual, cinematic, etc.
    seed: Optional[int] = None      # Seed para reprodutibilidade
    aspect_ratio: str = "16:9"      # Proporção da imagem
    model: str = "imagen_3_5"       # Modelo de geração
    consistency: bool = True        # Manter consistência de personagens
    extra_params: Dict = None       # Parâmetros extras


class PromptManager:
    """Gerenciador de prompts externos com cache."""

    CACHE_FILE = "prompt_cache.json"
    CONFIG_FILE = "prompt_config.json"

    def __init__(self, workspace_dir: str = ".", log_callback=None):
        """
        Inicializa o gerenciador de prompts.

        Args:
            workspace_dir: Diretório de trabalho
            log_callback: Função opcional para logging
        """
        self.workspace_dir = workspace_dir
        self.log = log_callback or (lambda msg, level="INFO": print(f"[{level}] {msg}"))
        self.config = PromptConfig()
        self.cache = {}
        self._load_config()
        self._load_cache()

    def _get_cache_path(self) -> str:
        """Retorna caminho do arquivo de cache."""
        return os.path.join(self.workspace_dir, self.CACHE_FILE)

    def _get_config_path(self) -> str:
        """Retorna caminho do arquivo de config."""
        return os.path.join(self.workspace_dir, self.CONFIG_FILE)

    def _load_config(self):
        """Carrega configuração do arquivo."""
        config_path = self._get_config_path()
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.config = PromptConfig(**data)
                self.log(f"Config de prompt carregada: {config_path}", "OK")
            except Exception as e:
                self.log(f"Erro ao carregar config: {e}", "WARN")

    def save_config(self):
        """Salva configuração no arquivo."""
        config_path = self._get_config_path()
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                data = asdict(self.config)
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.log(f"Config de prompt salva: {config_path}", "OK")
        except Exception as e:
            self.log(f"Erro ao salvar config: {e}", "ERROR")

    def _load_cache(self):
        """Carrega cache do arquivo."""
        cache_path = self._get_cache_path()
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                self.log(f"Cache de prompts carregado: {len(self.cache)} entradas", "OK")
            except Exception as e:
                self.log(f"Erro ao carregar cache: {e}", "WARN")
                self.cache = {}

    def _save_cache(self):
        """Salva cache no arquivo."""
        cache_path = self._get_cache_path()
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.log(f"Erro ao salvar cache: {e}", "ERROR")

    def load_prompt_file(self, file_path: str) -> str:
        """
        Carrega prompt de arquivo externo (MD ou TXT).

        Args:
            file_path: Caminho do arquivo

        Returns:
            Conteúdo do prompt
        """
        if not os.path.exists(file_path):
            self.log(f"Arquivo de prompt não encontrado: {file_path}", "ERROR")
            return ""

        encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']
        content = None

        for enc in encodings:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    content = f.read().strip()
                break
            except UnicodeDecodeError:
                continue

        if content is None:
            self.log(f"Não foi possível ler o arquivo: {file_path}", "ERROR")
            return ""

        # Processar Markdown: remover headers e links
        if file_path.endswith('.md'):
            content = self._process_markdown(content)

        self.config.prompt_file = file_path
        self.config.base_prompt = content
        self.save_config()

        self.log(f"Prompt carregado: {len(content)} caracteres", "OK")
        return content

    def _process_markdown(self, content: str) -> str:
        """
        Processa conteúdo Markdown, extraindo texto limpo.

        Args:
            content: Conteúdo Markdown

        Returns:
            Texto limpo
        """
        import re
        
        # Remover headers (#, ##, etc.)
        content = re.sub(r'^#+\s+', '', content, flags=re.MULTILINE)
        
        # Remover links [text](url)
        content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)
        
        # Remover bold/italic
        content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content)
        content = re.sub(r'\*([^*]+)\*', r'\1', content)
        content = re.sub(r'_([^_]+)_', r'\1', content)
        
        # Remover linhas em branco extras
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        return content.strip()

    def compute_hash(self, text: str, params: Dict = None) -> str:
        """
        Computa hash SHA256 do texto + parâmetros.

        Args:
            text: Texto do prompt
            params: Parâmetros adicionais

        Returns:
            Hash SHA256
        """
        data = text
        if params:
            data += json.dumps(params, sort_keys=True)
        
        return hashlib.sha256(data.encode('utf-8')).hexdigest()[:16]

    def is_cached(self, prompt_text: str, image_index: int) -> Optional[str]:
        """
        Verifica se uma imagem já está em cache.

        Args:
            prompt_text: Texto do prompt
            image_index: Índice da imagem

        Returns:
            Caminho da imagem em cache ou None
        """
        cache_key = f"{self.compute_hash(prompt_text)}_{image_index:03d}"
        
        if cache_key in self.cache:
            cached_path = self.cache[cache_key].get("image_path")
            if cached_path and os.path.exists(cached_path):
                self.log(f"Cache hit: imagem {image_index}", "OK")
                return cached_path
        
        return None

    def add_to_cache(self, prompt_text: str, image_index: int, image_path: str):
        """
        Adiciona imagem ao cache.

        Args:
            prompt_text: Texto do prompt
            image_index: Índice da imagem
            image_path: Caminho da imagem gerada
        """
        cache_key = f"{self.compute_hash(prompt_text)}_{image_index:03d}"
        
        self.cache[cache_key] = {
            "image_path": image_path,
            "prompt_hash": self.compute_hash(prompt_text),
            "index": image_index
        }
        
        self._save_cache()

    def build_image_prompt(self, srt_text: str, image_index: int) -> str:
        """
        Constrói prompt completo para geração de imagem.

        Args:
            srt_text: Texto do SRT (cues agrupados)
            image_index: Índice da imagem

        Returns:
            Prompt formatado
        """
        base = self.config.base_prompt or ""
        
        # Estilo baseado na configuração
        style_prompts = {
            "spiritual": "Spiritual atmosphere, divine light, ethereal, chosen ones theme",
            "cinematic": "Cinematic, dramatic lighting, movie quality, 4K",
            "mystical": "Mystical, magical, otherworldly, enchanting",
            "dark": "Dark aesthetic, moody, atmospheric, noir",
            "nature": "Natural beauty, serene, peaceful, organic"
        }
        
        style = style_prompts.get(self.config.style, style_prompts["spiritual"])
        
        # Construir prompt final
        prompt_parts = []
        
        if base:
            prompt_parts.append(base)
        
        prompt_parts.append(srt_text)
        prompt_parts.append(style)
        prompt_parts.append("high resolution, detailed, professional quality")
        
        if self.config.consistency:
            prompt_parts.append("consistent character design, same art style")
        
        return ", ".join(prompt_parts)

    def clear_cache(self):
        """Limpa todo o cache."""
        self.cache = {}
        self._save_cache()
        self.log("Cache de prompts limpo", "OK")

    def get_cache_stats(self) -> Dict:
        """
        Retorna estatísticas do cache.

        Returns:
            Dict com estatísticas
        """
        valid = 0
        invalid = 0
        
        for key, value in self.cache.items():
            path = value.get("image_path", "")
            if path and os.path.exists(path):
                valid += 1
            else:
                invalid += 1
        
        return {
            "total": len(self.cache),
            "valid": valid,
            "invalid": invalid
        }

