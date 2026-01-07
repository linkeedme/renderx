# -*- coding: utf-8 -*-
"""
Módulo de Gerenciamento de Pool de Tokens Whisk
================================================
Gerenciador thread-safe para múltiplas API keys do Whisk,
permitindo geração paralela de imagens com proteção contra rate limiting.

Arquivo de configuração: whisk_keys.json
"""

import threading
import time
import requests
import base64
import random
import json
import os
from typing import Optional, List, Dict, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from PIL import Image
from io import BytesIO

# Caminho padrão do arquivo de keys
SCRIPT_DIR = Path(__file__).parent.absolute()
WHISK_KEYS_FILE = SCRIPT_DIR / "whisk_keys.json"


def load_whisk_keys(file_path: str = None) -> Dict:
    """
    Carrega configuração de tokens do arquivo whisk_keys.json.
    
    Args:
        file_path: Caminho do arquivo (opcional, usa padrão se não fornecido)
        
    Returns:
        Dicionário com tokens e configurações
    """
    path = Path(file_path) if file_path else WHISK_KEYS_FILE
    
    if not path.exists():
        return {"tokens": [], "settings": {}}
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data
    except Exception as e:
        print(f"[WARN] Erro ao carregar whisk_keys.json: {e}")
        return {"tokens": [], "settings": {}}


def save_whisk_keys(data: Dict, file_path: str = None):
    """
    Salva configuração de tokens no arquivo whisk_keys.json.
    
    Args:
        data: Dicionário com tokens e configurações
        file_path: Caminho do arquivo (opcional)
    """
    path = Path(file_path) if file_path else WHISK_KEYS_FILE
    
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[ERROR] Erro ao salvar whisk_keys.json: {e}")


def get_enabled_tokens(file_path: str = None) -> List[str]:
    """
    Retorna lista de tokens habilitados do arquivo.
    
    Args:
        file_path: Caminho do arquivo (opcional)
        
    Returns:
        Lista de tokens (strings)
    """
    data = load_whisk_keys(file_path)
    tokens = []
    
    for entry in data.get("tokens", []):
        if isinstance(entry, dict):
            if entry.get("enabled", True):
                key = entry.get("key", "")
                if key:
                    tokens.append(key)
        elif isinstance(entry, str) and entry:
            tokens.append(entry)
    
    return tokens


def get_whisk_settings(file_path: str = None) -> Dict:
    """
    Retorna configurações do arquivo whisk_keys.json.
    
    Args:
        file_path: Caminho do arquivo (opcional)
        
    Returns:
        Dicionário de configurações
    """
    data = load_whisk_keys(file_path)
    return data.get("settings", {})


def add_token_to_file(key: str, email: str = None, enabled: bool = True, file_path: str = None):
    """
    Adiciona um novo token ao arquivo.
    
    Args:
        key: Token Bearer
        email: Email associado (opcional)
        enabled: Se o token está habilitado
        file_path: Caminho do arquivo (opcional)
    """
    data = load_whisk_keys(file_path)
    
    # Verificar se já existe
    for entry in data.get("tokens", []):
        if isinstance(entry, dict) and entry.get("key") == key:
            return  # Já existe
        elif entry == key:
            return  # Já existe
    
    # Adicionar novo token
    if "tokens" not in data:
        data["tokens"] = []
    
    data["tokens"].append({
        "key": key,
        "email": email,
        "enabled": enabled
    })
    
    save_whisk_keys(data, file_path)


def update_or_add_token_by_email(key: str, email: str = None, enabled: bool = True, file_path: str = None) -> str:
    """
    Atualiza token existente se email já existe, senão adiciona novo.
    
    Esta função é útil para captura automática de tokens, onde o mesmo
    usuário pode atualizar seu token expirado.
    
    Args:
        key: Token Bearer
        email: Email associado (usado para identificar token existente)
        enabled: Se o token está habilitado
        file_path: Caminho do arquivo (opcional)
        
    Returns:
        "atualizado" se atualizou token existente, "adicionado" se criou novo
    """
    data = load_whisk_keys(file_path)
    
    if "tokens" not in data:
        data["tokens"] = []
    
    # Se temos email, procurar por token existente com mesmo email
    if email:
        for i, entry in enumerate(data["tokens"]):
            if isinstance(entry, dict) and entry.get("email") == email:
                # Atualizar token existente
                data["tokens"][i]["key"] = key
                data["tokens"][i]["enabled"] = enabled
                save_whisk_keys(data, file_path)
                return "atualizado"
    
    # Verificar se token já existe (mesmo que email seja diferente)
    for entry in data["tokens"]:
        if isinstance(entry, dict) and entry.get("key") == key:
            return "já existe"
        elif entry == key:
            return "já existe"
    
    # Adicionar novo token
    data["tokens"].append({
        "key": key,
        "email": email,
        "enabled": enabled
    })
    
    save_whisk_keys(data, file_path)
    return "adicionado"


def remove_token_from_file(index: int, file_path: str = None) -> bool:
    """
    Remove um token do arquivo pelo índice.
    
    Args:
        index: Índice do token
        file_path: Caminho do arquivo (opcional)
        
    Returns:
        True se removido com sucesso
    """
    data = load_whisk_keys(file_path)
    tokens = data.get("tokens", [])
    
    if 0 <= index < len(tokens):
        del tokens[index]
        data["tokens"] = tokens
        save_whisk_keys(data, file_path)
        return True
    
    return False


def toggle_token_in_file(index: int, file_path: str = None) -> bool:
    """
    Alterna o status enabled de um token.
    
    Args:
        index: Índice do token
        file_path: Caminho do arquivo (opcional)
        
    Returns:
        Novo status do token
    """
    data = load_whisk_keys(file_path)
    tokens = data.get("tokens", [])
    
    if 0 <= index < len(tokens):
        if isinstance(tokens[index], dict):
            tokens[index]["enabled"] = not tokens[index].get("enabled", True)
            save_whisk_keys(data, file_path)
            return tokens[index]["enabled"]
    
    return False


def get_tokens_summary(file_path: str = None) -> str:
    """
    Retorna resumo dos tokens para exibição.
    
    Args:
        file_path: Caminho do arquivo (opcional)
        
    Returns:
        String com resumo (ex: "3 tokens (2 ativos)")
    """
    data = load_whisk_keys(file_path)
    tokens = data.get("tokens", [])
    
    total = len(tokens)
    enabled = 0
    
    for entry in tokens:
        if isinstance(entry, dict):
            if entry.get("enabled", True):
                enabled += 1
        else:
            enabled += 1
    
    if total == 0:
        return "Nenhum token configurado"
    elif total == enabled:
        return f"{total} token{'s' if total > 1 else ''} configurado{'s' if total > 1 else ''}"
    else:
        return f"{total} token{'s' if total > 1 else ''} ({enabled} ativo{'s' if enabled > 1 else ''})"


class TokenStatus(Enum):
    """Status possíveis de um token."""
    AVAILABLE = "available"
    IN_USE = "in_use"
    COOLDOWN = "cooldown"
    INVALID = "invalid"


@dataclass
class TokenStats:
    """Estatísticas de uso de um token."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rate_limited_count: int = 0
    last_request_time: Optional[datetime] = None
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None
    cooldown_until: Optional[datetime] = None
    
    def to_dict(self) -> dict:
        """Converte para dicionário serializável."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "rate_limited_count": self.rate_limited_count,
            "last_request_time": self.last_request_time.isoformat() if self.last_request_time else None,
            "last_error": self.last_error,
            "last_error_time": self.last_error_time.isoformat() if self.last_error_time else None,
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
            "success_rate": (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0
        }


@dataclass
class TokenEntry:
    """Entrada de token no pool."""
    token: str
    index: int
    email: Optional[str] = None
    status: TokenStatus = TokenStatus.AVAILABLE
    stats: TokenStats = field(default_factory=TokenStats)
    lock: threading.Lock = field(default_factory=threading.Lock)
    
    def __hash__(self):
        return hash(self.token)
    
    def get_masked_token(self) -> str:
        """Retorna token mascarado para exibição segura."""
        if len(self.token) > 20:
            return f"{self.token[:10]}...{self.token[-10:]}"
        return self.token[:5] + "..." if len(self.token) > 5 else self.token


class WhiskTokenPoolManager:
    """
    Gerenciador centralizado de pool de tokens Whisk.
    
    Características:
    - Thread-safe: múltiplos workers podem usar simultaneamente
    - Rate limiting: delay configurável entre requisições por token
    - Cooldown automático: tokens com erro entram em cooldown
    - Estatísticas: rastreia uso e erros por token
    - Checkout/checkin: garante uso exclusivo de token por worker
    """
    
    # Configurações padrão
    DEFAULT_COOLDOWN_SECONDS = 60  # 1 minuto de cooldown após erro
    DEFAULT_REQUEST_DELAY = 1.0   # 1 segundo entre requisições por token
    DEFAULT_TIMEOUT = 90.0        # Timeout de requisição
    MAX_RETRIES = 3               # Máximo de tentativas por requisição
    
    API_URL = "https://aisandbox-pa.googleapis.com/v1/whisk:generateImage"
    
    def __init__(
        self,
        tokens: List[str] = None,
        cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS,
        request_delay: float = DEFAULT_REQUEST_DELAY,
        timeout: float = DEFAULT_TIMEOUT,
        log_callback: Callable[[str, str], None] = None
    ):
        """
        Inicializa o gerenciador de pool de tokens.
        
        Args:
            tokens: Lista de tokens Bearer
            cooldown_seconds: Segundos de cooldown após erro
            request_delay: Delay mínimo entre requisições por token
            timeout: Timeout para requisições HTTP
            log_callback: Função de callback para logs (message, level)
        """
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._tokens: Dict[int, TokenEntry] = {}
        self._next_index = 0
        
        self.cooldown_seconds = cooldown_seconds
        self.request_delay = request_delay
        self.timeout = timeout
        self.log = log_callback or (lambda msg, level="INFO": print(f"[{level}] {msg}"))
        
        # Adicionar tokens iniciais
        if tokens:
            for token in tokens:
                if token and token.strip():
                    self.add_token(token.strip())
    
    def add_token(self, token: str, email: Optional[str] = None) -> int:
        """
        Adiciona um novo token ao pool.
        
        Args:
            token: Token Bearer
            email: Email associado (opcional)
            
        Returns:
            Índice do token adicionado
        """
        with self._lock:
            # Verificar se token já existe
            for entry in self._tokens.values():
                if entry.token == token:
                    self.log(f"Token já existe no índice {entry.index}", "WARN")
                    return entry.index
            
            index = self._next_index
            self._next_index += 1
            
            self._tokens[index] = TokenEntry(
                token=token,
                index=index,
                email=email,
                status=TokenStatus.AVAILABLE
            )
            
            self.log(f"Token adicionado no índice {index}", "INFO")
            self._condition.notify_all()
            return index
    
    def remove_token(self, index: int) -> bool:
        """
        Remove um token do pool.
        
        Args:
            index: Índice do token a remover
            
        Returns:
            True se removido, False se não encontrado
        """
        with self._lock:
            if index in self._tokens:
                del self._tokens[index]
                self.log(f"Token removido do índice {index}", "INFO")
                return True
            return False
    
    def get_token_count(self) -> int:
        """Retorna número de tokens no pool."""
        with self._lock:
            return len(self._tokens)
    
    def get_available_count(self) -> int:
        """Retorna número de tokens disponíveis."""
        with self._lock:
            self._update_cooldowns()
            return sum(1 for t in self._tokens.values() 
                      if t.status == TokenStatus.AVAILABLE)
    
    def _update_cooldowns(self):
        """Atualiza status de tokens em cooldown (deve ser chamado com lock)."""
        now = datetime.now()
        for entry in self._tokens.values():
            if entry.status == TokenStatus.COOLDOWN:
                if entry.stats.cooldown_until and now >= entry.stats.cooldown_until:
                    entry.status = TokenStatus.AVAILABLE
                    entry.stats.cooldown_until = None
                    self.log(f"Token {entry.index} saiu do cooldown", "INFO")
    
    def checkout_token(self, timeout: float = 30.0) -> Optional[TokenEntry]:
        """
        Faz checkout de um token disponível para uso exclusivo.
        
        Args:
            timeout: Tempo máximo de espera por token disponível
            
        Returns:
            TokenEntry se disponível, None se timeout
        """
        deadline = time.time() + timeout
        
        with self._condition:
            while True:
                self._update_cooldowns()
                
                # Buscar token disponível
                for entry in self._tokens.values():
                    if entry.status == TokenStatus.AVAILABLE:
                        entry.status = TokenStatus.IN_USE
                        self.log(f"Token {entry.index} em uso", "DEBUG")
                        return entry
                
                # Calcular tempo restante
                remaining = deadline - time.time()
                if remaining <= 0:
                    self.log("Timeout aguardando token disponível", "WARN")
                    return None
                
                # Aguardar notificação ou timeout
                self._condition.wait(timeout=min(remaining, 1.0))
    
    def checkin_token(self, entry: TokenEntry, success: bool = True, error_code: Optional[int] = None):
        """
        Faz checkin de um token após uso.
        
        Args:
            entry: TokenEntry a devolver
            success: Se a requisição foi bem-sucedida
            error_code: Código de erro HTTP (se houver)
        """
        with self._condition:
            if entry.index not in self._tokens:
                return
            
            entry.stats.total_requests += 1
            entry.stats.last_request_time = datetime.now()
            
            if success:
                entry.stats.successful_requests += 1
                entry.status = TokenStatus.AVAILABLE
            else:
                entry.stats.failed_requests += 1
                
                if error_code in (401, 403):
                    # Token inválido ou sem permissão
                    entry.status = TokenStatus.INVALID
                    entry.stats.last_error = f"HTTP {error_code}: Token inválido ou sem créditos"
                    entry.stats.last_error_time = datetime.now()
                    self.log(f"Token {entry.index} marcado como inválido (HTTP {error_code})", "ERROR")
                    
                elif error_code == 429:
                    # Rate limit - entrar em cooldown
                    entry.status = TokenStatus.COOLDOWN
                    entry.stats.rate_limited_count += 1
                    entry.stats.cooldown_until = datetime.now() + timedelta(seconds=self.cooldown_seconds)
                    entry.stats.last_error = "Rate limit excedido"
                    entry.stats.last_error_time = datetime.now()
                    self.log(f"Token {entry.index} em cooldown por {self.cooldown_seconds}s (rate limit)", "WARN")
                    
                else:
                    # Outros erros - cooldown curto
                    entry.status = TokenStatus.COOLDOWN
                    entry.stats.cooldown_until = datetime.now() + timedelta(seconds=self.cooldown_seconds // 2)
                    entry.stats.last_error = f"HTTP {error_code}" if error_code else "Erro desconhecido"
                    entry.stats.last_error_time = datetime.now()
                    self.log(f"Token {entry.index} em cooldown curto", "WARN")
            
            self._condition.notify_all()
    
    def get_all_tokens_status(self) -> List[dict]:
        """
        Retorna status de todos os tokens.
        
        Returns:
            Lista de dicionários com status de cada token
        """
        with self._lock:
            self._update_cooldowns()
            result = []
            for entry in self._tokens.values():
                result.append({
                    "index": entry.index,
                    "masked_token": entry.get_masked_token(),
                    "email": entry.email,
                    "status": entry.status.value,
                    "stats": entry.stats.to_dict()
                })
            return result
    
    def get_aggregate_stats(self) -> dict:
        """
        Retorna estatísticas agregadas do pool.
        
        Returns:
            Dicionário com estatísticas agregadas
        """
        with self._lock:
            self._update_cooldowns()
            
            total_tokens = len(self._tokens)
            available = sum(1 for t in self._tokens.values() if t.status == TokenStatus.AVAILABLE)
            in_use = sum(1 for t in self._tokens.values() if t.status == TokenStatus.IN_USE)
            cooldown = sum(1 for t in self._tokens.values() if t.status == TokenStatus.COOLDOWN)
            invalid = sum(1 for t in self._tokens.values() if t.status == TokenStatus.INVALID)
            
            total_requests = sum(t.stats.total_requests for t in self._tokens.values())
            successful = sum(t.stats.successful_requests for t in self._tokens.values())
            failed = sum(t.stats.failed_requests for t in self._tokens.values())
            rate_limited = sum(t.stats.rate_limited_count for t in self._tokens.values())
            
            return {
                "total_tokens": total_tokens,
                "available": available,
                "in_use": in_use,
                "cooldown": cooldown,
                "invalid": invalid,
                "total_requests": total_requests,
                "successful_requests": successful,
                "failed_requests": failed,
                "rate_limited_count": rate_limited,
                "success_rate": (successful / total_requests * 100) if total_requests > 0 else 0
            }
    
    def reset_token(self, index: int) -> bool:
        """
        Reseta status e estatísticas de um token.
        
        Args:
            index: Índice do token
            
        Returns:
            True se resetado, False se não encontrado
        """
        with self._lock:
            if index in self._tokens:
                entry = self._tokens[index]
                entry.status = TokenStatus.AVAILABLE
                entry.stats = TokenStats()
                self.log(f"Token {index} resetado", "INFO")
                self._condition.notify_all()
                return True
            return False
    
    def reset_all_tokens(self):
        """Reseta status de todos os tokens (exceto os inválidos com 401/403)."""
        with self._lock:
            for entry in self._tokens.values():
                if entry.status in (TokenStatus.COOLDOWN, TokenStatus.IN_USE):
                    entry.status = TokenStatus.AVAILABLE
                    entry.stats.cooldown_until = None
            self.log("Todos os tokens resetados", "INFO")
            self._condition.notify_all()
    
    def generate_image(
        self,
        prompt: str,
        aspect_ratio: str = "IMAGE_ASPECT_RATIO_LANDSCAPE",
        checkout_timeout: float = 30.0
    ) -> Tuple[Optional[bytes], Optional[int]]:
        """
        Gera uma imagem usando um token do pool.
        
        Args:
            prompt: Texto do prompt
            aspect_ratio: Aspect ratio da imagem
            checkout_timeout: Timeout para checkout de token
            
        Returns:
            Tupla (image_bytes, error_code) - image_bytes é None em caso de erro
        """
        entry = self.checkout_token(checkout_timeout)
        if not entry:
            self.log("Nenhum token disponível para geração", "ERROR")
            return None, None
        
        try:
            # Respeitar delay entre requisições
            if entry.stats.last_request_time:
                elapsed = (datetime.now() - entry.stats.last_request_time).total_seconds()
                if elapsed < self.request_delay:
                    time.sleep(self.request_delay - elapsed)
            
            # Payload da requisição
            payload = {
                "clientContext": {
                    "workflowId": "c4dd24a1-c7e8-4057-9c25-1d2635673bd1",
                    "tool": "BACKBONE",
                    "sessionId": f";{int(time.time() * 1000)}"
                },
                "imageModelSettings": {
                    "imageModel": "IMAGEN_3_5",
                    "aspectRatio": aspect_ratio
                },
                "mediaCategory": "MEDIA_CATEGORY_BOARD",
                "prompt": prompt,
                "seed": random.randint(1, 1000000)
            }
            
            headers = {
                "Authorization": f"Bearer {entry.token}",
                "Content-Type": "application/json; charset=UTF-8"
            }
            
            # Fazer requisição
            self.log(f"Gerando imagem com token {entry.index}...", "INFO")
            response = requests.post(
                self.API_URL,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                # Sucesso
                data = response.json()
                image_panels = data.get("imagePanels", [])
                if image_panels:
                    generated_images = image_panels[0].get("generatedImages", [])
                    if generated_images:
                        encoded_image = generated_images[0].get("encodedImage")
                        if encoded_image:
                            image_bytes = base64.b64decode(encoded_image)
                            self.checkin_token(entry, success=True)
                            self.log(f"Imagem gerada com sucesso (token {entry.index})", "OK")
                            return image_bytes, None
                
                # Resposta sem imagem
                self.checkin_token(entry, success=False)
                self.log("Resposta sem imagem válida", "ERROR")
                return None, None
            else:
                # Erro HTTP
                self.checkin_token(entry, success=False, error_code=response.status_code)
                return None, response.status_code
                
        except requests.exceptions.Timeout:
            self.log(f"Timeout na requisição (token {entry.index})", "ERROR")
            self.checkin_token(entry, success=False)
            return None, None
            
        except Exception as e:
            self.log(f"Erro ao gerar imagem: {str(e)}", "ERROR")
            self.checkin_token(entry, success=False)
            return None, None
    
    def save_image(self, image_bytes: bytes, output_path: str) -> bool:
        """
        Salva imagem em arquivo.
        
        Args:
            image_bytes: Bytes da imagem
            output_path: Caminho de saída
            
        Returns:
            True se sucesso
        """
        try:
            import os
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            
            image = Image.open(BytesIO(image_bytes))
            image.save(output_path, "PNG")
            
            file_size = os.path.getsize(output_path)
            self.log(f"Imagem salva: {output_path} ({file_size / 1024:.1f} KB)", "OK")
            return True
            
        except Exception as e:
            self.log(f"Erro ao salvar imagem: {str(e)}", "ERROR")
            return False


# Instância global do pool manager (singleton)
_global_pool_manager: Optional[WhiskTokenPoolManager] = None
_global_pool_lock = threading.Lock()


def get_pool_manager(
    tokens: List[str] = None,
    cooldown_seconds: int = None,
    request_delay: float = None,
    log_callback: Callable[[str, str], None] = None,
    reinitialize: bool = False,
    load_from_file: bool = True
) -> WhiskTokenPoolManager:
    """
    Obtém instância global do pool manager.
    
    Se nenhum token for fornecido e load_from_file=True, carrega do whisk_keys.json.
    
    Args:
        tokens: Lista de tokens (opcional, carrega do arquivo se não fornecido)
        cooldown_seconds: Segundos de cooldown (opcional, usa do arquivo ou padrão)
        request_delay: Delay entre requisições (opcional, usa do arquivo ou padrão)
        log_callback: Callback de log
        reinitialize: Se True, recria a instância
        load_from_file: Se True, carrega tokens do whisk_keys.json quando tokens=None
        
    Returns:
        Instância do WhiskTokenPoolManager
    """
    global _global_pool_manager
    
    with _global_pool_lock:
        if _global_pool_manager is None or reinitialize:
            # Carregar do arquivo se não foram fornecidos tokens
            if tokens is None and load_from_file:
                tokens = get_enabled_tokens()
                settings = get_whisk_settings()
                
                # Usar configurações do arquivo se não foram fornecidas
                if cooldown_seconds is None:
                    cooldown_seconds = settings.get("cooldown_seconds", WhiskTokenPoolManager.DEFAULT_COOLDOWN_SECONDS)
                if request_delay is None:
                    request_delay = settings.get("request_delay", WhiskTokenPoolManager.DEFAULT_REQUEST_DELAY)
            
            # Valores padrão se ainda None
            if cooldown_seconds is None:
                cooldown_seconds = WhiskTokenPoolManager.DEFAULT_COOLDOWN_SECONDS
            if request_delay is None:
                request_delay = WhiskTokenPoolManager.DEFAULT_REQUEST_DELAY
            
            _global_pool_manager = WhiskTokenPoolManager(
                tokens=tokens,
                cooldown_seconds=cooldown_seconds,
                request_delay=request_delay,
                log_callback=log_callback
            )
        elif tokens:
            # Adicionar novos tokens se fornecidos
            for token in tokens:
                if token and token.strip():
                    _global_pool_manager.add_token(token.strip())
        
        return _global_pool_manager

