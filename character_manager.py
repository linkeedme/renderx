# -*- coding: utf-8 -*-
"""
Character Manager - Gerenciador de Personagem Principal
========================================================
Aplica personagem principal (PNG ou MP4) sobre vídeo base em posições configuráveis:
centro, esquerdo ou direito.
"""

import os
import subprocess
import sys
from typing import Optional, Tuple
from pathlib import Path


class CharacterManager:
    """Gerenciador de personagem principal."""

    VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.mkv', '.webm')
    IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')

    POSITION_MAP = {
        "center": "center",
        "centro": "center",
        "left": "left",
        "esquerdo": "left",
        "right": "right",
        "direito": "right"
    }

    def __init__(self, log_callback=None):
        """
        Inicializa o gerenciador de personagem.

        Args:
            log_callback: Função opcional para logging (message, level)
        """
        self.log = log_callback or (lambda msg, level="INFO": print(f"[{level}] {msg}"))

    def _normalize_position(self, position: str) -> str:
        """
        Normaliza nome da posição.

        Args:
            position: Nome da posição (centro/center, esquerdo/left, direito/right)

        Returns:
            Posição normalizada: "center", "left" ou "right"
        """
        position_lower = position.lower().strip()
        return self.POSITION_MAP.get(position_lower, "right")  # Default: direita

    def _get_overlay_position(self, position: str, video_width: int, video_height: int,
                              char_width: int, char_height: int, margin: int = 20) -> Tuple[int, int]:
        """
        Calcula posição (x, y) do overlay baseado na posição configurada.

        Args:
            position: Posição ("center", "left", "right")
            video_width: Largura do vídeo base
            video_height: Altura do vídeo base
            char_width: Largura do personagem
            char_height: Altura do personagem
            margin: Margem em pixels (para esquerdo e direito)

        Returns:
            Tupla (x, y) da posição do overlay
        """
        position = self._normalize_position(position)

        # Calcular posição Y (verticalmente centralizado)
        y = (video_height - char_height) // 2

        if position == "center":
            # Centralizado horizontalmente
            x = (video_width - char_width) // 2
        elif position == "left":
            # Canto esquerdo com margem
            x = margin
        else:  # right
            # Canto direito com margem
            x = video_width - char_width - margin

        return (x, y)

    def _get_character_dimensions(self, character_path: str, is_video: bool) -> Optional[Tuple[int, int]]:
        """
        Obtém dimensões do personagem (PNG ou vídeo).

        Args:
            character_path: Caminho do personagem
            is_video: Se é vídeo (True) ou imagem (False)

        Returns:
            Tupla (width, height) ou None se erro
        """
        if is_video:
            # Para vídeo, usar ffprobe
            cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "default=noprint_wrappers=1:nokey=1",
                character_path
            ]
        else:
            # Para imagem, usar ffprobe também (mais confiável)
            cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "default=noprint_wrappers=1:nokey=1",
                character_path
            ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                width = int(lines[0])
                height = int(lines[1])
                return (width, height)
        except Exception as e:
            self.log(f"Erro ao obter dimensões do personagem: {e}", "WARN")

        return None

    def _detect_character_type(self, character_path: str) -> Optional[str]:
        """
        Detecta se o personagem é vídeo ou imagem.

        Args:
            character_path: Caminho do personagem

        Returns:
            "video", "image" ou None se inválido
        """
        if not os.path.exists(character_path):
            return None

        ext = os.path.splitext(character_path)[1].lower()
        
        if ext in self.VIDEO_EXTENSIONS:
            return "video"
        elif ext in self.IMAGE_EXTENSIONS:
            return "image"
        else:
            return None

    def apply_character(self, video_path: str, character_path: str, position: str,
                       output_path: str, video_resolution: Tuple[int, int] = None) -> Optional[str]:
        """
        Aplica personagem principal sobre vídeo base.

        Args:
            video_path: Caminho do vídeo base
            character_path: Caminho do personagem (PNG ou MP4)
            position: Posição ("center", "left", "right" ou variações em português)
            output_path: Caminho do vídeo de saída
            video_resolution: Resolução do vídeo base (width, height). Se None, detecta automaticamente.

        Returns:
            Caminho do vídeo com personagem ou None se erro
        """
        if not os.path.exists(video_path):
            self.log(f"Vídeo base não encontrado: {video_path}", "ERROR")
            return None

        if not os.path.exists(character_path):
            self.log(f"Personagem não encontrado: {character_path}", "ERROR")
            return None

        # Detectar tipo do personagem
        char_type = self._detect_character_type(character_path)
        if char_type is None:
            self.log(f"Formato inválido do personagem: {character_path}", "ERROR")
            return None

        is_video = (char_type == "video")
        self.log(f"Aplicando personagem ({char_type}) na posição '{position}'...", "INFO")

        # Obter resolução do vídeo base se não fornecida
        if video_resolution is None:
            video_resolution = self._get_video_resolution(video_path)
            if video_resolution is None:
                self.log("Não foi possível obter resolução do vídeo base", "ERROR")
                return None

        video_width, video_height = video_resolution

        # Obter dimensões do personagem
        char_dimensions = self._get_character_dimensions(character_path, is_video)
        if char_dimensions is None:
            self.log("Não foi possível obter dimensões do personagem", "ERROR")
            return None

        char_width, char_height = char_dimensions

        # Calcular posição do overlay
        x, y = self._get_overlay_position(position, video_width, video_height,
                                         char_width, char_height)

        self.log(f"Posição do personagem: ({x}, {y})", "DEBUG")

        # Construir filter_complex do FFmpeg
        if is_video:
            # Personagem é vídeo: usar loop se necessário
            filter_complex = (
                f"[0:v]scale={video_width}:{video_height}:force_original_aspect_ratio=decrease,"
                f"pad={video_width}:{video_height}:(ow-iw)/2:(oh-ih)/2:color=black[vbase];"
                f"[1:v]format=rgba[char];"
                f"[vbase][char]overlay={x}:{y}:shortest=1:format=auto[vout]"
            )
            
            # Para vídeo, usar stream_loop para garantir que cubra toda a duração
            input_args = [
                "-stream_loop", "-1",  # Loop infinito do personagem
                "-i", character_path
            ]
        else:
            # Personagem é imagem PNG
            filter_complex = (
                f"[0:v]scale={video_width}:{video_height}:force_original_aspect_ratio=decrease,"
                f"pad={video_width}:{video_height}:(ow-iw)/2:(oh-ih)/2:color=black[vbase];"
                f"[1:v]format=rgba,scale={char_width}:{char_height}[char];"
                f"[vbase][char]overlay={x}:{y}:format=auto[vout]"
            )
            
            input_args = ["-i", character_path]

        # Construir comando FFmpeg
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            *input_args,
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-map", "0:a?",  # Mapear áudio do vídeo base se existir
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "copy",  # Copiar áudio sem re-encoding
            "-pix_fmt", "yuv420p",
            "-shortest",  # Para vídeo, garantir que termine quando o vídeo base terminar
            output_path
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )

            if result.returncode == 0:
                self.log(f"Personagem aplicado com sucesso: {output_path}", "OK")
                return output_path
            else:
                self.log(f"Erro ao aplicar personagem: {result.stderr[-300:] if result.stderr else ''}", "ERROR")
                return None
        except Exception as e:
            self.log(f"Erro ao aplicar personagem: {e}", "ERROR")
            import traceback
            self.log(traceback.format_exc(), "ERROR")
            return None

    def _get_video_resolution(self, video_path: str) -> Optional[Tuple[int, int]]:
        """
        Obtém resolução do vídeo.

        Args:
            video_path: Caminho do vídeo

        Returns:
            Tupla (width, height) ou None se erro
        """
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                width = int(lines[0])
                height = int(lines[1])
                return (width, height)
        except Exception as e:
            self.log(f"Erro ao obter resolução do vídeo: {e}", "WARN")
        return None

    def get_character_info(self, character_path: str) -> Optional[Dict]:
        """
        Obtém informações do personagem.

        Args:
            character_path: Caminho do personagem

        Returns:
            Dict com informações ou None se erro
        """
        if not os.path.exists(character_path):
            return None

        char_type = self._detect_character_type(character_path)
        if char_type is None:
            return None

        dimensions = self._get_character_dimensions(character_path, char_type == "video")
        
        return {
            "path": character_path,
            "type": char_type,
            "exists": True,
            "dimensions": dimensions
        }

