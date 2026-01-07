# -*- coding: utf-8 -*-
"""
Módulo de Geração de Imagens via API Whisk (Gemini)
====================================================
Gera imagens usando a API do Google Imagen 3.5 (Whisk).
Suporta geração paralela com múltiplos tokens via WhiskTokenPoolManager.
"""

import requests
import base64
import random
import time
import os
from typing import Optional, List, Dict, Callable, Tuple
from pathlib import Path
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import threading


class WhiskImageGenerator:
    """Gerador de imagens via API Whisk (Google Imagen 3.5)."""

    def __init__(self, tokens: List[str], log_callback=None):
        """
        Inicializa o gerador de imagens.

        Args:
            tokens: Lista de tokens Bearer para autenticação
            log_callback: Função opcional para logging (message, level)
        """
        self.tokens = tokens if isinstance(tokens, list) else [tokens] if tokens else []
        self.log = log_callback or (lambda msg, level="INFO": print(f"[{level}] {msg}"))
        self.current_token_index = 0
        self.invalid_tokens = set()
        self.api_url = "https://aisandbox-pa.googleapis.com/v1/whisk:generateImage"
        self.timeout = 90.0
        self.max_retries = 3

    def _get_next_valid_token(self) -> Optional[str]:
        """
        Obtém próximo token válido (rotação automática).

        Returns:
            Token válido ou None se não houver
        """
        if not self.tokens:
            return None

        # Se todos os tokens foram marcados como inválidos, resetar
        if len(self.invalid_tokens) >= len(self.tokens):
            self.log("Todos os tokens foram marcados como inválidos. Resetando...", "WARN")
            self.invalid_tokens.clear()

        # Tentar encontrar um token válido
        attempts = 0
        while attempts < len(self.tokens):
            token = self.tokens[self.current_token_index]
            self.current_token_index = (self.current_token_index + 1) % len(self.tokens)

            if token not in self.invalid_tokens:
                return token

            attempts += 1

        return None

    def _mark_token_as_invalid(self, token: str):
        """
        Marca um token como inválido temporariamente.

        Args:
            token: Token a marcar como inválido
        """
        self.invalid_tokens.add(token)
        self.log(f"Token marcado como inválido (total inválidos: {len(self.invalid_tokens)})", "WARN")

    # Negative prompt padrão para evitar texto nas imagens
    DEFAULT_NEGATIVE_PROMPT = "text, letters, words, writing, captions, watermark, signature, logo, labels, numbers, typography, font, subtitle, title, banner, UI elements, blurry, low quality, distorted, deformed"

    def create_spiritual_prompt(self, text: str, include_no_text: bool = True) -> str:
        """
        Cria prompt espiritual a partir do texto do SRT.

        Args:
            text: Texto combinado dos blocos SRT
            include_no_text: Se True, adiciona instrução para evitar texto na imagem

        Returns:
            Prompt formatado com estilo espiritual
        """
        # Limpar texto (remover quebras de linha excessivas)
        text_clean = " ".join(text.split())
        
        # Criar prompt espiritual com instrução para evitar texto
        base_prompt = f"Espiritual, Chosen Ones, {text_clean}, estilo místico, atmosfera sagrada, iluminação divina, energia transcendental, high quality, photorealistic, 4K"
        
        if include_no_text:
            # Adicionar instrução explícita para evitar texto
            base_prompt += ", no text in image, no letters, no words, no writing, clean image without any text"
        
        return base_prompt

    def generate_image_from_prompt(self, prompt: str, negative_prompt: str = None) -> Optional[bytes]:
        """
        Gera imagem a partir de um prompt via API Whisk.

        Args:
            prompt: Texto do prompt para geração
            negative_prompt: Prompt negativo (coisas a evitar na imagem)

        Returns:
            Bytes da imagem gerada (PNG) ou None em caso de erro
        """
        token = self._get_next_valid_token()
        if not token:
            self.log("Nenhum token válido disponível", "ERROR")
            return None

        # Usar negative prompt padrão se não fornecido
        if negative_prompt is None:
            negative_prompt = self.DEFAULT_NEGATIVE_PROMPT
        
        # Adicionar instrução de "no text" ao prompt principal para reforçar
        enhanced_prompt = prompt
        if "no text" not in prompt.lower() and "no letters" not in prompt.lower():
            enhanced_prompt = f"{prompt}, no text, no letters, no words, no writing in the image"

        # Payload conforme documentação
        payload = {
            "clientContext": {
                "workflowId": "c4dd24a1-c7e8-4057-9c25-1d2635673bd1",
                "tool": "BACKBONE",
                "sessionId": f";{int(time.time() * 1000)}"
            },
            "imageModelSettings": {
                "imageModel": "IMAGEN_3_5",
                "aspectRatio": "IMAGE_ASPECT_RATIO_LANDSCAPE"
            },
            "mediaCategory": "MEDIA_CATEGORY_BOARD",
            "prompt": enhanced_prompt,
            "seed": random.randint(1, 1000000)
        }
        
        # Adicionar negative prompt se a API suportar
        if negative_prompt:
            payload["negativePrompt"] = negative_prompt

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=UTF-8"
        }

        for attempt in range(self.max_retries):
            try:
                self.log(f"Gerando imagem (tentativa {attempt + 1}/{self.max_retries})...", "INFO")
                
                response = requests.post(
                    self.api_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    # Sucesso - decodificar imagem
                    data = response.json()
                    
                    # Extrair imagem base64 da resposta
                    image_panels = data.get("imagePanels", [])
                    if not image_panels:
                        self.log("Resposta sem imagePanels", "ERROR")
                        return None

                    generated_images = image_panels[0].get("generatedImages", [])
                    if not generated_images:
                        self.log("Resposta sem generatedImages", "ERROR")
                        return None

                    encoded_image = generated_images[0].get("encodedImage")
                    if not encoded_image:
                        self.log("Resposta sem encodedImage", "ERROR")
                        return None

                    # Decodificar base64
                    image_bytes = base64.b64decode(encoded_image)
                    self.log("Imagem gerada com sucesso!", "OK")
                    return image_bytes

                elif response.status_code == 401:
                    # Token inválido
                    self._mark_token_as_invalid(token)
                    token = self._get_next_valid_token()
                    if not token:
                        self.log("Nenhum token válido disponível após erro 401", "ERROR")
                        return None
                    headers["Authorization"] = f"Bearer {token}"
                    continue

                elif response.status_code == 403:
                    # Sem permissão ou créditos esgotados
                    self._mark_token_as_invalid(token)
                    self.log("Erro 403: Token sem permissão ou créditos esgotados", "ERROR")
                    token = self._get_next_valid_token()
                    if not token:
                        return None
                    headers["Authorization"] = f"Bearer {token}"
                    continue

                elif response.status_code == 429:
                    # Rate limit
                    self.log("Rate limit excedido. Aguardando...", "WARN")
                    time.sleep(5)  # Aguardar 5 segundos
                    continue

                else:
                    self.log(f"Erro HTTP {response.status_code}: {response.text[:200]}", "ERROR")
                    if attempt < self.max_retries - 1:
                        time.sleep(2)
                        continue
                    return None

            except requests.exceptions.Timeout:
                self.log(f"Timeout na requisição (tentativa {attempt + 1})", "WARN")
                if attempt < self.max_retries - 1:
                    time.sleep(2)
                    continue
                return None

            except Exception as e:
                self.log(f"Erro ao gerar imagem: {str(e)}", "ERROR")
                if attempt < self.max_retries - 1:
                    time.sleep(2)
                    continue
                return None

        return None

    def save_image(self, image_bytes: bytes, output_path: str) -> bool:
        """
        Salva imagem em arquivo.

        Args:
            image_bytes: Bytes da imagem
            output_path: Caminho onde salvar

        Returns:
            True se sucesso, False caso contrário
        """
        try:
            # Criar diretório se não existir
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

            # Converter bytes para PIL Image e salvar
            image = Image.open(BytesIO(image_bytes))
            image.save(output_path, "PNG")
            
            file_size = os.path.getsize(output_path)
            self.log(f"Imagem salva: {output_path} ({file_size / 1024:.1f} KB)", "OK")
            return True

        except Exception as e:
            self.log(f"Erro ao salvar imagem: {str(e)}", "ERROR")
            return False

    def generate_and_save_image(self, prompt: str, output_path: str) -> bool:
        """
        Gera imagem e salva em arquivo.

        Args:
            prompt: Texto do prompt
            output_path: Caminho onde salvar a imagem

        Returns:
            True se sucesso, False caso contrário
        """
        image_bytes = self.generate_image_from_prompt(prompt)
        if image_bytes:
            return self.save_image(image_bytes, output_path)
        return False

    def validate_config(self) -> tuple[bool, str]:
        """
        Valida configuração (tokens disponíveis).

        Returns:
            Tupla (is_valid, error_message)
        """
        if not self.tokens:
            return False, "Nenhum token configurado"
        
        if len(self.tokens) == 0:
            return False, "Lista de tokens vazia"

        return True, ""


@dataclass
class ImageGenerationTask:
    """Tarefa de geração de imagem para processamento paralelo."""
    index: int
    prompt: str
    output_path: str
    success: bool = False
    error: Optional[str] = None


@dataclass
class ParallelGenerationResult:
    """Resultado da geração paralela de imagens."""
    total: int
    successful: int
    failed: int
    tasks: List[ImageGenerationTask]
    
    @property
    def success_rate(self) -> float:
        """Taxa de sucesso em porcentagem."""
        return (self.successful / self.total * 100) if self.total > 0 else 0


class ParallelWhiskGenerator:
    """
    Gerador paralelo de imagens usando WhiskTokenPoolManager.
    
    Permite gerar múltiplas imagens simultaneamente usando
    diferentes tokens do pool, maximizando throughput e
    evitando rate limiting.
    """
    
    def __init__(
        self,
        tokens: List[str],
        max_workers: int = None,
        cooldown_seconds: int = 60,
        request_delay: float = 1.0,
        log_callback: Callable[[str, str], None] = None
    ):
        """
        Inicializa o gerador paralelo.
        
        Args:
            tokens: Lista de tokens Bearer
            max_workers: Número máximo de workers paralelos (padrão: número de tokens)
            cooldown_seconds: Segundos de cooldown após erro de rate limit
            request_delay: Delay mínimo entre requisições por token
            log_callback: Função de callback para logs (message, level)
        """
        from whisk_pool_manager import get_pool_manager
        
        self.log = log_callback or (lambda msg, level="INFO": print(f"[{level}] {msg}"))
        
        # Obter pool manager global com os tokens
        self.pool_manager = get_pool_manager(
            tokens=tokens,
            cooldown_seconds=cooldown_seconds,
            request_delay=request_delay,
            log_callback=self.log,
            reinitialize=True
        )
        
        # Número de workers = mínimo entre tokens disponíveis e max_workers
        token_count = self.pool_manager.get_token_count()
        self.max_workers = min(max_workers or token_count, token_count) if token_count > 0 else 1
        
        self._progress_lock = threading.Lock()
        self._completed_count = 0
        self._total_count = 0
    
    # Negative prompt padrão para evitar texto nas imagens
    DEFAULT_NEGATIVE_PROMPT = "text, letters, words, writing, captions, watermark, signature, logo, labels, numbers, typography, font, subtitle, title, banner, UI elements, blurry, low quality, distorted, deformed"

    def create_spiritual_prompt(self, text: str, include_no_text: bool = True) -> str:
        """
        Cria prompt espiritual a partir do texto.
        
        Args:
            text: Texto base
            include_no_text: Se True, adiciona instrução para evitar texto na imagem
            
        Returns:
            Prompt formatado com estilo espiritual
        """
        text_clean = " ".join(text.split())
        base_prompt = f"Espiritual, Chosen Ones, {text_clean}, estilo místico, atmosfera sagrada, iluminação divina, energia transcendental, high quality, photorealistic"
        
        if include_no_text:
            base_prompt += ", no text in image, no letters, no words, no writing, clean image without any text"
        
        return base_prompt
    
    def _generate_single_image(
        self,
        task: ImageGenerationTask,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> ImageGenerationTask:
        """
        Gera uma única imagem (executado em worker thread).
        
        Args:
            task: Tarefa de geração
            progress_callback: Callback de progresso (current, total, message)
            
        Returns:
            Tarefa atualizada com resultado
        """
        try:
            # Gerar imagem usando pool manager
            image_bytes, error_code = self.pool_manager.generate_image(
                prompt=task.prompt,
                checkout_timeout=60.0
            )
            
            if image_bytes:
                # Salvar imagem
                success = self.pool_manager.save_image(image_bytes, task.output_path)
                task.success = success
                if not success:
                    task.error = "Falha ao salvar imagem"
            else:
                task.success = False
                task.error = f"Falha na geração (código: {error_code})" if error_code else "Falha na geração"
            
        except Exception as e:
            task.success = False
            task.error = str(e)
        
        # Atualizar progresso
        with self._progress_lock:
            self._completed_count += 1
            if progress_callback:
                try:
                    progress_callback(
                        self._completed_count,
                        self._total_count,
                        f"Imagem {task.index + 1}: {'OK' if task.success else task.error}"
                    )
                except Exception:
                    pass
        
        return task
    
    def generate_images_parallel(
        self,
        prompts: List[str],
        output_paths: List[str],
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> ParallelGenerationResult:
        """
        Gera múltiplas imagens em paralelo.
        
        Args:
            prompts: Lista de prompts para geração
            output_paths: Lista de caminhos de saída correspondentes
            progress_callback: Callback de progresso (current, total, message)
            
        Returns:
            ParallelGenerationResult com estatísticas e resultados
        """
        if len(prompts) != len(output_paths):
            raise ValueError("Número de prompts deve ser igual ao número de output_paths")
        
        # Criar tarefas
        tasks = [
            ImageGenerationTask(index=i, prompt=prompt, output_path=path)
            for i, (prompt, path) in enumerate(zip(prompts, output_paths))
        ]
        
        if not tasks:
            return ParallelGenerationResult(total=0, successful=0, failed=0, tasks=[])
        
        # Reset contadores
        with self._progress_lock:
            self._completed_count = 0
            self._total_count = len(tasks)
        
        self.log(f"Iniciando geração paralela de {len(tasks)} imagens com {self.max_workers} workers", "INFO")
        
        if progress_callback:
            progress_callback(0, len(tasks), "Iniciando geração paralela...")
        
        # Executar em paralelo
        completed_tasks = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._generate_single_image, task, progress_callback): task
                for task in tasks
            }
            
            for future in as_completed(futures):
                try:
                    completed_task = future.result()
                    completed_tasks.append(completed_task)
                except Exception as e:
                    task = futures[future]
                    task.success = False
                    task.error = str(e)
                    completed_tasks.append(task)
        
        # Ordenar por índice original
        completed_tasks.sort(key=lambda t: t.index)
        
        # Calcular estatísticas
        successful = sum(1 for t in completed_tasks if t.success)
        failed = len(completed_tasks) - successful
        
        result = ParallelGenerationResult(
            total=len(completed_tasks),
            successful=successful,
            failed=failed,
            tasks=completed_tasks
        )
        
        self.log(f"Geração concluída: {successful}/{len(tasks)} imagens ({result.success_rate:.1f}% sucesso)", "OK")
        
        return result
    
    def generate_images_from_texts(
        self,
        texts: List[str],
        output_dir: str,
        prefix: str = "generated_image",
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> ParallelGenerationResult:
        """
        Gera imagens a partir de textos, criando prompts espirituais automaticamente.
        
        Args:
            texts: Lista de textos para criar prompts
            output_dir: Diretório de saída
            prefix: Prefixo para nomes de arquivos
            progress_callback: Callback de progresso
            
        Returns:
            ParallelGenerationResult
        """
        # Criar diretório de saída
        os.makedirs(output_dir, exist_ok=True)
        
        # Criar prompts e paths
        prompts = [self.create_spiritual_prompt(text) for text in texts]
        output_paths = [
            os.path.join(output_dir, f"{prefix}_{i:04d}.png")
            for i in range(len(texts))
        ]
        
        return self.generate_images_parallel(prompts, output_paths, progress_callback)
    
    def get_pool_stats(self) -> dict:
        """Retorna estatísticas do pool de tokens."""
        return self.pool_manager.get_aggregate_stats()
    
    def get_tokens_status(self) -> List[dict]:
        """Retorna status de todos os tokens."""
        return self.pool_manager.get_all_tokens_status()
