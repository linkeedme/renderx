# -*- coding: utf-8 -*-
"""
Backlog Video Manager - Gerenciador de Vídeos do Backlog
=========================================================
Gerencia seleção, movimentação e concatenação de vídeos do backlog para cobrir
duração do áudio. Vídeos usados são movidos para pasta "USADOS" para evitar repetição.
"""

import os
import json
import random
import subprocess
import sys
import shutil
import time
from typing import List, Optional, Dict, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


class BacklogVideoManager:
    """Gerenciador de vídeos do backlog."""

    VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.mkv', '.webm')
    CACHE_FILE = "backlog_cache.json"
    USED_FOLDER_NAME = "USADOS"

    def __init__(self, log_callback=None):
        """
        Inicializa o gerenciador de backlog.

        Args:
            log_callback: Função opcional para logging (message, level)
        """
        self.log = log_callback or (lambda msg, level="INFO": print(f"[{level}] {msg}"))
        self.cache_file = self.CACHE_FILE
        self.duration_cache = {}
        self._load_cache()

    def _load_cache(self):
        """Carrega cache de durações de vídeos."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.duration_cache = json.load(f)
                self.log(f"Cache de durações carregado: {len(self.duration_cache)} vídeos", "DEBUG")
            except Exception as e:
                self.log(f"Erro ao carregar cache: {e}", "WARN")
                self.duration_cache = {}

    def _save_cache(self):
        """Salva cache de durações de vídeos."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.duration_cache, f, indent=2)
        except Exception as e:
            self.log(f"Erro ao salvar cache: {e}", "WARN")

    def get_video_duration(self, video_path: str, use_cache: bool = True) -> Optional[float]:
        """
        Obtém a duração de um vídeo em segundos.

        Args:
            video_path: Caminho do vídeo
            use_cache: Se deve usar cache

        Returns:
            Duração em segundos ou None se erro
        """
        # Verificar cache primeiro
        if use_cache and video_path in self.duration_cache:
            cached_duration = self.duration_cache[video_path]
            # Verificar se arquivo ainda existe e tem mesma data de modificação
            if os.path.exists(video_path):
                return cached_duration

        # Obter duração via ffprobe
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
            duration = float(result.stdout.strip())
            
            # Salvar no cache
            if use_cache:
                self.duration_cache[video_path] = duration
                self._save_cache()
            
            return duration
        except Exception as e:
            self.log(f"Erro ao obter duração de {os.path.basename(video_path)}: {e}", "WARN")
            return None

    def scan_backlog_videos(self, backlog_folder: str, used_folder_name: str = None) -> List[str]:
        """
        Escaneia pasta de backlog e retorna lista de vídeos disponíveis.
        Ignora completamente a pasta "USADOS".

        Args:
            backlog_folder: Pasta do backlog
            used_folder_name: Nome da pasta de usados (padrão: "USADOS")

        Returns:
            Lista de caminhos de vídeos disponíveis
        """
        used_folder_name = used_folder_name or self.USED_FOLDER_NAME
        
        if not os.path.exists(backlog_folder):
            self.log(f"Pasta do backlog não encontrada: {backlog_folder}", "WARN")
            return []

        videos = []
        used_folder_path = os.path.join(backlog_folder, used_folder_name)

        try:
            for item in os.listdir(backlog_folder):
                item_path = os.path.join(backlog_folder, item)
                
                # Ignorar diretórios (especialmente a pasta USADOS)
                if os.path.isdir(item_path):
                    continue
                
                # Ignorar arquivos que não são vídeos
                if not item.lower().endswith(self.VIDEO_EXTENSIONS):
                    continue
                
                videos.append(item_path)
            
            videos = sorted(videos)  # Ordenar por nome
            self.log(f"Encontrados {len(videos)} vídeos disponíveis no backlog", "INFO")
            return videos
            
        except Exception as e:
            self.log(f"Erro ao escanear backlog: {e}", "ERROR")
            return []

    def calculate_videos_needed(self, audio_duration: float, backlog_folder: str, 
                                used_folder_name: str = None, 
                                avg_video_duration: float = None) -> Tuple[int, float]:
        """
        Calcula quantidade aproximada de vídeos necessários baseado na duração do áudio.

        Args:
            audio_duration: Duração do áudio em segundos
            backlog_folder: Pasta do backlog
            used_folder_name: Nome da pasta de usados
            avg_video_duration: Duração média estimada dos vídeos (se None, calcula)

        Returns:
            Tupla (quantidade_estimada, duração_média_estimada)
        """
        used_folder_name = used_folder_name or self.USED_FOLDER_NAME
        
        # Se não foi fornecida duração média, tentar calcular
        if avg_video_duration is None:
            videos = self.scan_backlog_videos(backlog_folder, used_folder_name)
            if not videos:
                # Fallback: assumir 8 segundos por vídeo
                avg_video_duration = 8.0
                self.log("Nenhum vídeo encontrado, usando duração média padrão de 8s", "WARN")
            else:
                # Calcular média das durações (usando cache quando possível)
                durations = []
                sample_size = min(10, len(videos))  # Amostrar até 10 vídeos
                sample_videos = random.sample(videos, sample_size)
                
                for video_path in sample_videos:
                    duration = self.get_video_duration(video_path)
                    if duration:
                        durations.append(duration)
                
                if durations:
                    avg_video_duration = sum(durations) / len(durations)
                else:
                    avg_video_duration = 8.0  # Fallback
                
                self.log(f"Duração média estimada dos vídeos: {avg_video_duration:.2f}s", "INFO")

        # Calcular quantidade necessária (com margem de segurança)
        videos_needed = int(audio_duration / avg_video_duration) + 2  # +2 para garantir cobertura
        
        return videos_needed, avg_video_duration

    def select_backlog_videos(self, backlog_folder: str, target_duration: float,
                             used_folder_name: str = None) -> List[str]:
        """
        Seleciona vídeos do backlog suficientes para cobrir a duração alvo.
        Ignora vídeos já na pasta USADOS.

        Args:
            backlog_folder: Pasta do backlog
            target_duration: Duração alvo em segundos
            used_folder_name: Nome da pasta de usados

        Returns:
            Lista de caminhos de vídeos selecionados
        """
        used_folder_name = used_folder_name or self.USED_FOLDER_NAME
        
        available_videos = self.scan_backlog_videos(backlog_folder, used_folder_name)
        
        if not available_videos:
            self.log("Nenhum vídeo disponível no backlog", "ERROR")
            return []

        # Embaralhar para variedade
        shuffled = available_videos.copy()
        random.shuffle(shuffled)

        selected_videos = []
        total_duration = 0.0

        for video_path in shuffled:
            duration = self.get_video_duration(video_path)
            if duration is None:
                continue  # Pular vídeos com erro
            
            selected_videos.append(video_path)
            total_duration += duration
            
            # Se atingiu ou ultrapassou a duração alvo, parar
            if total_duration >= target_duration:
                break
        
        if not selected_videos:
            self.log("Não foi possível selecionar vídeos suficientes", "ERROR")
            return []

        self.log(f"Selecionados {len(selected_videos)} vídeos (~{total_duration:.1f}s) para cobrir {target_duration:.1f}s", "INFO")
        return selected_videos

    def move_used_videos_to_used_folder(self, video_paths: List[str], backlog_folder: str,
                                       used_folder_name: str = None) -> List[str]:
        """
        Move vídeos selecionados para pasta "USADOS" para evitar repetição.

        Args:
            video_paths: Lista de caminhos de vídeos a mover
            backlog_folder: Pasta do backlog
            used_folder_name: Nome da pasta de usados (padrão: "USADOS")

        Returns:
            Lista de novos caminhos (após movimentação)
        """
        used_folder_name = used_folder_name or self.USED_FOLDER_NAME
        used_folder_path = os.path.join(backlog_folder, used_folder_name)
        
        # Criar pasta USADOS se não existir
        try:
            os.makedirs(used_folder_path, exist_ok=True)
        except Exception as e:
            self.log(f"Erro ao criar pasta USADOS: {e}", "ERROR")
            return video_paths  # Retornar caminhos originais se falhar

        new_paths = []
        moved_count = 0

        for video_path in video_paths:
            if not os.path.exists(video_path):
                self.log(f"Vídeo não encontrado: {os.path.basename(video_path)}", "WARN")
                continue

            base_name = os.path.basename(video_path)
            dest_path = os.path.join(used_folder_path, base_name)

            # Se arquivo já existe em USADOS, adicionar timestamp
            if os.path.exists(dest_path):
                name, ext = os.path.splitext(base_name)
                timestamp = int(time.time())
                dest_path = os.path.join(used_folder_path, f"{name}_{timestamp}{ext}")

            try:
                shutil.move(video_path, dest_path)
                new_paths.append(dest_path)
                moved_count += 1
                self.log(f"Movido para USADOS: {base_name}", "DEBUG")
                
                # Atualizar cache se o caminho mudou
                if video_path in self.duration_cache:
                    duration = self.duration_cache.pop(video_path)
                    self.duration_cache[dest_path] = duration
                
            except Exception as e:
                self.log(f"Erro ao mover {base_name} para USADOS: {e}", "ERROR")
                new_paths.append(video_path)  # Manter caminho original se falhar

        if moved_count > 0:
            self._save_cache()  # Salvar cache atualizado
            self.log(f"{moved_count} vídeo(s) movido(s) para pasta USADOS", "OK")

        return new_paths

    def normalize_video(self, video_path: str, output_path: str, resolution: Tuple[int, int],
                       use_gpu: bool = False) -> Optional[str]:
        """
        Normaliza um vídeo: ajusta resolução e codec para consistência.

        Args:
            video_path: Caminho do vídeo de entrada
            output_path: Caminho do vídeo de saída
            resolution: Tupla (width, height)
            use_gpu: Se deve usar GPU encoding

        Returns:
            Caminho do vídeo normalizado ou None se erro
        """
        width, height = resolution
        
        # Filtrar vídeo: smart crop 16:9, ajustar resolução e forçar FPS consistente
        # Forçar 30fps para evitar problemas de sincronização e congelamento
        video_filter = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"fps=30"
        )

        # Encoder args com FPS fixo (otimizado para velocidade)
        if use_gpu:
            encoder_args = ["-c:v", "h264_nvenc", "-preset", "p1", "-b:v", "3M", "-r", "30"]
        else:
            encoder_args = ["-c:v", "libx264", "-preset", "superfast", "-b:v", "3M", "-r", "30", "-vsync", "cfr"]

        # Verificar se tem áudio
        has_audio = True
        try:
            cmd_check = [
                "ffprobe", "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=codec_type",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ]
            result = subprocess.run(
                cmd_check,
                capture_output=True,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
            has_audio = result.returncode == 0 and "audio" in result.stdout.lower()
        except:
            has_audio = False

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", video_filter,
            *encoder_args,
            "-pix_fmt", "yuv420p",
        ]

        if has_audio:
            cmd.extend(["-c:a", "aac", "-b:a", "128k"])
        else:
            cmd.append("-an")

        cmd.append(output_path)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )

            if result.returncode == 0:
                return output_path
            else:
                self.log(f"Erro ao normalizar vídeo: {result.stderr[-200:] if result.stderr else ''}", "ERROR")
                return None
        except Exception as e:
            self.log(f"Erro ao normalizar vídeo: {e}", "ERROR")
            return None

    def concatenate_videos(self, video_paths: List[str], output_path: str, 
                          resolution: Tuple[int, int], use_gpu: bool = False,
                          normalize: bool = True, temp_dir: str = None) -> Optional[str]:
        """
        Concatena vídeos sem transições usando concat demuxer.

        Args:
            video_paths: Lista de caminhos de vídeos
            output_path: Caminho do vídeo de saída
            resolution: Tupla (width, height)
            use_gpu: Se deve usar GPU encoding
            normalize: Se deve normalizar vídeos antes (recomendado)
            temp_dir: Diretório temporário (se None, cria um)

        Returns:
            Caminho do vídeo concatenado ou None se erro
        """
        if not video_paths:
            self.log("Nenhum vídeo fornecido para concatenar", "ERROR")
            return None

        import tempfile

        if temp_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="backlog_concat_")
            cleanup_temp = True
        else:
            cleanup_temp = False

        try:
            normalized_videos = []

            if normalize:
                # Normalizar todos os vídeos em paralelo (otimizado para velocidade)
                self.log(f"Normalizando {len(video_paths)} vídeo(s) em paralelo...", "INFO")
                
                # Função auxiliar para normalizar um vídeo
                def normalize_single(args):
                    i, video_path = args
                    normalized_path = os.path.join(temp_dir, f"normalized_{i:04d}.mp4")
                    normalized = self.normalize_video(video_path, normalized_path, resolution, use_gpu)
                    if normalized:
                        return (i, normalized)
                    else:
                        self.log(f"Falha ao normalizar vídeo {i+1}, pulando...", "WARN")
                        return None
                
                # Processar em paralelo (usar até 4 threads para não sobrecarregar I/O)
                max_workers = min(4, len(video_paths))
                normalized_results = {}
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(normalize_single, (i, video_path)): i 
                        for i, video_path in enumerate(video_paths)
                    }
                    
                    for future in as_completed(futures):
                        result = future.result()
                        if result:
                            idx, normalized_path = result
                            normalized_results[idx] = normalized_path
                
                # Ordenar por índice para manter ordem original
                normalized_videos = [normalized_results[i] for i in sorted(normalized_results.keys())]
                
                if len(normalized_videos) < len(video_paths):
                    self.log(f"Aviso: {len(normalized_videos)}/{len(video_paths)} vídeos normalizados com sucesso", "WARN")
            else:
                normalized_videos = video_paths

            if not normalized_videos:
                self.log("Nenhum vídeo foi normalizado com sucesso", "ERROR")
                return None

            # Criar arquivo de lista para concat demuxer
            concat_list_path = os.path.join(temp_dir, "concat_list.txt")
            with open(concat_list_path, 'w', encoding='utf-8') as f:
                for video_path in normalized_videos:
                    # Escapar caminhos para Windows e FFmpeg
                    safe_path = video_path.replace('\\', '/').replace("'", "'\\''")
                    f.write(f"file '{safe_path}'\n")

            # Encoder args com FPS fixo para garantir consistência (otimizado para velocidade)
            if use_gpu:
                encoder_args = ["-c:v", "h264_nvenc", "-preset", "p1", "-b:v", "5M", "-r", "30"]
            else:
                encoder_args = ["-c:v", "libx264", "-preset", "superfast", "-b:v", "5M", "-r", "30"]

            # Concatenar vídeos com FPS fixo e sincronização
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", concat_list_path,
                *encoder_args,
                "-c:a", "aac", "-b:a", "192k",
                "-pix_fmt", "yuv420p",
                "-vsync", "cfr",  # Constant frame rate para evitar congelamentos
                output_path
            ]

            self.log(f"Concatenando {len(normalized_videos)} vídeo(s)...", "INFO")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )

            if result.returncode == 0:
                self.log(f"Vídeos concatenados com sucesso: {output_path}", "OK")
                return output_path
            else:
                self.log(f"Erro ao concatenar vídeos: {result.stderr[-300:] if result.stderr else ''}", "ERROR")
                return None

        except Exception as e:
            self.log(f"Erro ao concatenar vídeos: {e}", "ERROR")
            import traceback
            self.log(traceback.format_exc(), "ERROR")
            return None
        finally:
            # Limpar temporários
            if cleanup_temp and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass

