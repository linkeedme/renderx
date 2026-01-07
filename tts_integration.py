# -*- coding: utf-8 -*-
"""
Módulo de Integração TTS - DARKVI e TALKIFY APIs
================================================
Gerencia geração de áudio via APIs de Text-to-Speech
"""

import httpx
import time
import os
from typing import Optional, List, Dict, Tuple


class TTSGenerator:
    """Gerador de áudio via APIs TTS (DARKVI e TALKIFY)."""

    def __init__(self, provider: str, api_key: str, log_callback=None):
        """
        Inicializa o gerador TTS.

        Args:
            provider: "darkvi" ou "talkify"
            api_key: Token de autenticação da API
            log_callback: Função opcional para logging (message, level)
        """
        self.provider = provider.lower()
        self.api_key = api_key
        self.log = log_callback or (lambda msg, level="INFO": print(f"[{level}] {msg}"))
        
        # Timeouts e retries
        self.timeout = 60.0
        self.max_retries = 3
        self.poll_interval = 2.0  # Segundos entre verificações de status
        self.max_poll_time = 300.0  # Máximo 5 minutos de polling

    def list_voices(self) -> List[Dict[str, str]]:
        """
        Lista vozes disponíveis da API.

        Returns:
            Lista de dicionários com 'id' e 'name' de cada voz
        """
        if self.provider == "darkvi":
            return self._list_darkvi_voices()
        elif self.provider == "talkify":
            return self._list_talkify_voices()
        else:
            return []

    def _list_darkvi_voices(self) -> List[Dict[str, str]]:
        """Lista vozes da API DARKVI."""
        try:
            url = "https://darkvi.com/api/tts/voices"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            self.log(f"Fazendo requisição para: {url}", "DEBUG")
            self.log(f"API Key (primeiros 10 chars): {self.api_key[:10]}...", "DEBUG")
            
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(url, headers=headers)
                
                self.log(f"Status code: {response.status_code}", "DEBUG")
                
                response.raise_for_status()
                
                # Verificar content-type
                content_type = response.headers.get("content-type", "")
                self.log(f"Content-Type: {content_type}", "DEBUG")
                
                voices = response.json()
                self.log(f"Resposta JSON: {voices}", "DEBUG")
                
                if isinstance(voices, list):
                    self.log(f"Lista de vozes recebida: {len(voices)} itens", "DEBUG")
                    # Garantir que id e name sejam strings não vazias
                    # DARKVI retorna 'idApi' em vez de 'id'
                    result = []
                    for i, v in enumerate(voices):
                        # Tentar 'idApi' primeiro (formato DARKVI), depois 'id' (formato padrão)
                        voice_id = str(v.get("idApi", v.get("id", ""))).strip()
                        voice_name = str(v.get("name", "")).strip()
                        self.log(f"Voz {i+1}: id='{voice_id}', name='{voice_name}'", "DEBUG")
                        if voice_id:  # Só adicionar se tiver ID
                            result.append({"id": voice_id, "name": voice_name if voice_name else voice_id})
                        else:
                            self.log(f"Voz {i+1} sem ID válido, pulando", "WARNING")
                    self.log(f"Vozes válidas processadas: {len(result)}", "DEBUG")
                    return result
                else:
                    self.log(f"Resposta não é uma lista: {type(voices)}", "ERROR")
                    return []
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                self.log("Token DARKVI inválido ou expirado", "ERROR")
            elif e.response.status_code == 404:
                self.log("Nenhuma voz encontrada na DARKVI", "WARNING")
            else:
                self.log(f"Erro ao listar vozes DARKVI: {e.response.status_code}", "ERROR")
            return []
        except Exception as e:
            self.log(f"Erro ao listar vozes DARKVI: {str(e)}", "ERROR")
            return []

    def _list_talkify_voices(self) -> List[Dict[str, str]]:
        """Lista vozes da API TALKIFY."""
        # TALKIFY não tem endpoint de listagem de vozes na documentação
        # Retorna lista vazia - vozes devem ser configuradas manualmente
        self.log("TALKIFY não possui endpoint de listagem de vozes. Use o voiceId diretamente.", "INFO")
        return []

    def generate_audio(self, text: str, voice_id: str, output_path: str, title: Optional[str] = None) -> bool:
        """
        Gera áudio a partir de texto via API.

        Args:
            text: Texto a ser convertido
            voice_id: ID da voz a ser usada
            output_path: Caminho onde salvar o arquivo MP3
            title: Título opcional (DARKVI)

        Returns:
            True se sucesso, False caso contrário
        """
        if self.provider == "darkvi":
            return self._generate_darkvi(text, voice_id, output_path, title)
        elif self.provider == "talkify":
            return self._generate_talkify(text, voice_id, output_path)
        else:
            self.log(f"Provider desconhecido: {self.provider}", "ERROR")
            return False

    def _generate_darkvi(self, text: str, voice_id: str, output_path: str, title: Optional[str] = None) -> bool:
        """Gera áudio via DARKVI API."""
        try:
            # Validar limite de caracteres
            if len(text) > 80000:
                self.log(f"Texto muito longo ({len(text)} chars). Limite DARKVI: 80000", "ERROR")
                return False

            # 1. Criar job de geração
            url = "https://darkvi.com/api/tts"
            # Garantir que o token não tenha espaços extras
            clean_token = self.api_key.strip()
            # Headers exatamente como no exemplo que funciona
            # NÃO adicionar Content-Type - httpx adiciona automaticamente quando usa json=
            headers = {
                "Authorization": f"Bearer {clean_token}"
            }
            self.log(f"Header Authorization: Bearer {clean_token[:10]}...{clean_token[-10:] if len(clean_token) > 20 else ''}", "DEBUG")
            
            # Payload conforme documentação
            payload = {
                "text": text,
                "voice": voice_id  # Usando o idApi (UUID) retornado pela API
            }
            if title:
                payload["title"] = title

            self.log(f"Enviando texto para DARKVI (voice: {voice_id}, text length: {len(text)})...", "INFO")
            self.log(f"URL: {url}", "DEBUG")
            self.log(f"Headers (sem token completo): Authorization: Bearer {clean_token[:15]}...", "DEBUG")
            self.log(f"Payload: {payload}", "DEBUG")
            
            # Enviar exatamente como no exemplo do Playground
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    url,
                    json=payload,  # httpx automaticamente adiciona Content-Type: application/json quando usa json=
                    headers=headers
                )
                
                self.log(f"Status code: {response.status_code}", "DEBUG")
                
                # Se erro, logar resposta completa
                if response.status_code != 201:
                    try:
                        error_data = response.json()
                        self.log(f"Resposta de erro: {error_data}", "ERROR")
                    except:
                        self.log(f"Resposta de erro (texto): {response.text[:500]}", "ERROR")
                
                response.raise_for_status()
                
                data = response.json()
                self.log(f"Resposta DARKVI: {data}", "DEBUG")
                
                if not data.get("ok"):
                    error_msg = data.get("message", "Erro desconhecido")
                    self.log(f"Erro DARKVI: {error_msg}", "ERROR")
                    return False

                audio_id = data.get("data", {}).get("created", {}).get("id")
                if not audio_id:
                    self.log("Resposta DARKVI sem ID de áudio", "ERROR")
                    return False

                self.log(f"Job criado: {audio_id}. Aguardando processamento...", "INFO")

                # 2. Polling do status
                if not self._poll_darkvi_status(audio_id):
                    return False

                # 3. Download do áudio
                return self._download_darkvi_audio(audio_id, output_path)

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                self.log("Token DARKVI inválido ou expirado", "ERROR")
            else:
                self.log(f"Erro HTTP DARKVI: {e.response.status_code}", "ERROR")
            return False
        except Exception as e:
            self.log(f"Erro ao gerar áudio DARKVI: {str(e)}", "ERROR")
            return False

    def _poll_darkvi_status(self, audio_id: str) -> bool:
        """Faz polling do status do áudio DARKVI até estar pronto."""
        url = f"https://darkvi.com/api/tts/{audio_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        start_time = time.time()
        attempts = 0

        with httpx.Client(timeout=self.timeout) as client:
            while time.time() - start_time < self.max_poll_time:
                try:
                    response = client.get(url, headers=headers)
                    response.raise_for_status()
                    
                    data = response.json()
                    status = data.get("status", "").upper()
                    
                    if status == "DONE":
                        self.log("Áudio DARKVI processado com sucesso!", "OK")
                        return True
                    elif status == "PENDING" or status == "PROCESSING":
                        attempts += 1
                        if attempts % 5 == 0:  # Log a cada 5 tentativas
                            self.log(f"Aguardando processamento... ({int(time.time() - start_time)}s)", "INFO")
                        time.sleep(self.poll_interval)
                    else:
                        self.log(f"Status desconhecido DARKVI: {status}", "ERROR")
                        return False

                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 404:
                        self.log("Áudio DARKVI não encontrado", "ERROR")
                        return False
                    time.sleep(self.poll_interval)
                except Exception as e:
                    self.log(f"Erro ao verificar status DARKVI: {str(e)}", "WARNING")
                    time.sleep(self.poll_interval)

        self.log("Timeout ao aguardar processamento DARKVI", "ERROR")
        return False

    def _download_darkvi_audio(self, audio_id: str, output_path: str) -> bool:
        """Baixa áudio gerado da DARKVI."""
        try:
            url = f"https://darkvi.com/api/tts/audios/{audio_id}"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            # Criar diretório se não existir
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(url, headers=headers)
                response.raise_for_status()
                
                # Verificar content-type
                content_type = response.headers.get("content-type", "")
                if "audio" not in content_type.lower():
                    self.log(f"Resposta DARKVI não é áudio: {content_type}", "ERROR")
                    return False

                # Salvar arquivo
                with open(output_path, "wb") as f:
                    f.write(response.content)

                file_size = os.path.getsize(output_path)
                self.log(f"Áudio DARKVI salvo: {output_path} ({file_size / 1024:.1f} KB)", "OK")
                return True

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                self.log("Áudio DARKVI não encontrado para download", "ERROR")
            else:
                self.log(f"Erro HTTP ao baixar DARKVI: {e.response.status_code}", "ERROR")
            return False
        except Exception as e:
            self.log(f"Erro ao baixar áudio DARKVI: {str(e)}", "ERROR")
            return False

    def _generate_talkify(self, text: str, voice_id: str, output_path: str) -> bool:
        """Gera áudio via TALKIFY API."""
        try:
            url = "https://api.talkifydev.com/tts/jobs"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "audioName": "audio",
                "audio": {
                    "text": text,
                    "voiceId": voice_id,
                    "rate": 1
                }
            }

            self.log(f"Enviando texto para TALKIFY (voice: {voice_id})...", "INFO")

            with httpx.Client(timeout=self.timeout) as client:
                # 1. Criar job
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                
                # TALKIFY pode retornar o job ID de várias formas
                # Verificar diferentes formatos de resposta
                try:
                    # Tentar JSON primeiro
                    try:
                        data = response.json()
                        job_id = data.get("id") or data.get("jobId") or data.get("job_id") or data.get("job")
                    except:
                        data = None
                        job_id = None
                    
                    # Se não tiver no JSON, tentar header Location
                    if not job_id:
                        location = response.headers.get("Location", "")
                        if location:
                            job_id = location.split("/")[-1].strip()
                    
                    # Se ainda não tiver, verificar se a resposta é o ID diretamente (texto)
                    if not job_id:
                        text_response = response.text.strip()
                        # Verificar se parece um UUID (36 caracteres com hífens)
                        if len(text_response) >= 30 and "-" in text_response:
                            job_id = text_response
                    
                    if not job_id:
                        # Log da resposta para debug
                        self.log(f"Resposta TALKIFY: {response.text[:200]}", "DEBUG")
                        raise ValueError("Job ID não encontrado na resposta TALKIFY")
                    
                    self.log(f"Job TALKIFY criado: {job_id}. Baixando áudio...", "INFO")
                    
                    # 2. Download direto (TALKIFY gera imediatamente ou em pouco tempo)
                    # Tentar download com retry
                    return self._download_talkify_audio(job_id, output_path)

                except (ValueError, KeyError) as e:
                    self.log(f"Formato de resposta TALKIFY inesperado: {str(e)}", "ERROR")
                    if hasattr(response, 'text'):
                        self.log(f"Resposta completa: {response.text[:500]}", "DEBUG")
                    return False

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                self.log("Token TALKIFY inválido ou expirado", "ERROR")
            else:
                self.log(f"Erro HTTP TALKIFY: {e.response.status_code}", "ERROR")
                try:
                    error_data = e.response.json()
                    self.log(f"Detalhes: {error_data}", "ERROR")
                except:
                    pass
            return False
        except Exception as e:
            self.log(f"Erro ao gerar áudio TALKIFY: {str(e)}", "ERROR")
            return False

    def _download_talkify_audio(self, job_id: str, output_path: str, max_retries: int = 5) -> bool:
        """Baixa áudio gerado da TALKIFY com retry."""
        url = f"https://api.talkifydev.com/tts/jobs/{job_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        # Criar diretório se não existir
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        for attempt in range(max_retries):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.get(url, headers=headers)
                    
                    # 202 Accepted significa que ainda está processando
                    if response.status_code == 202:
                        if attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 2  # Backoff exponencial
                            self.log(f"TALKIFY ainda processando. Aguardando {wait_time}s...", "INFO")
                            time.sleep(wait_time)
                            continue
                        else:
                            self.log("Timeout ao aguardar processamento TALKIFY", "ERROR")
                            return False
                    
                    response.raise_for_status()
                    
                    # Verificar content-type
                    content_type = response.headers.get("content-type", "")
                    if "audio" not in content_type.lower() and "application/json" in content_type.lower():
                        # Pode ser que retorne JSON com status
                        try:
                            data = response.json()
                            if data.get("status") == "processing":
                                if attempt < max_retries - 1:
                                    time.sleep(2)
                                    continue
                        except:
                            pass
                    
                    # Salvar arquivo
                    with open(output_path, "wb") as f:
                        f.write(response.content)

                    file_size = os.path.getsize(output_path)
                    if file_size > 0:
                        self.log(f"Áudio TALKIFY salvo: {output_path} ({file_size / 1024:.1f} KB)", "OK")
                        return True
                    else:
                        self.log("Arquivo TALKIFY vazio", "ERROR")
                        return False

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    self.log("Job TALKIFY não encontrado", "ERROR")
                    return False
                elif e.response.status_code == 202:
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                else:
                    self.log(f"Erro HTTP ao baixar TALKIFY: {e.response.status_code}", "ERROR")
                    return False
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                self.log(f"Erro ao baixar áudio TALKIFY: {str(e)}", "ERROR")
                return False

        return False

    def validate_config(self) -> Tuple[bool, str]:
        """
        Valida configuração da API.

        Returns:
            Tupla (is_valid, error_message)
        """
        if not self.api_key or not self.api_key.strip():
            return False, "API key não configurada"

        if self.provider not in ["darkvi", "talkify"]:
            return False, f"Provider inválido: {self.provider}"

        # Testar conexão listando vozes
        try:
            voices = self.list_voices()
            if self.provider == "darkvi" and len(voices) == 0:
                return False, "Não foi possível listar vozes DARKVI. Verifique o token."
            return True, ""
        except Exception as e:
            return False, f"Erro ao validar API: {str(e)}"

