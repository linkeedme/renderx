# -*- coding: utf-8 -*-
"""
Overlay Manager - Gerenciador de Overlays Fornecidos pelo Usuário
=================================================================
Gerencia overlays (PNG ou vídeo) colocados pelo usuário em uma pasta.
Aplica shuffle global (1 overlay por vídeo) sem repetição consecutiva.

Formatos Suportados:
- Vídeo: .mp4, .mov, .webm (loop automático)
- Imagem: .png (com alpha, overlay estático)
"""

import os
import json
import random
import subprocess
from typing import List, Optional, Dict
from collections import deque


class OverlayManager:
    """Gerenciador de overlays do usuário."""

    # Extensões suportadas
    VIDEO_EXTENSIONS = ('.mp4', '.mov', '.webm', '.avi', '.mkv')
    IMAGE_EXTENSIONS = ('.png',)
    
    HISTORY_FILE = "overlay_history.json"

    def __init__(self, overlay_folder: str = "", history_file: str = None,
                 max_history: int = 10, log_callback=None):
        """
        Inicializa o gerenciador de overlays.

        Args:
            overlay_folder: Pasta com overlays do usuário
            history_file: Arquivo de histórico (para evitar repetição)
            max_history: Máximo de overlays no histórico
            log_callback: Função opcional para logging
        """
        self.overlay_folder = overlay_folder
        self.history_file = history_file or self.HISTORY_FILE
        self.max_history = max_history
        self.log = log_callback or (lambda msg, level="INFO": print(f"[{level}] {msg}"))
        self.used_overlays = deque(maxlen=self.max_history)
        self._load_history()

    def _load_history(self):
        """Carrega histórico de overlays usados."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.used_overlays.extend(data.get("used_overlays", []))
                self.log(f"Histórico de overlays carregado: {len(self.used_overlays)} itens", "DEBUG")
            except Exception as e:
                self.log(f"Erro ao carregar histórico de overlays: {e}", "WARN")

    def _save_history(self):
        """Salva histórico de overlays usados."""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump({"used_overlays": list(self.used_overlays)}, f, indent=2)
        except Exception as e:
            self.log(f"Erro ao salvar histórico de overlays: {e}", "ERROR")

    def set_folder(self, folder: str):
        """
        Define a pasta de overlays.

        Args:
            folder: Caminho da pasta
        """
        self.overlay_folder = folder
        self.log(f"Pasta de overlays definida: {folder}", "OK")

    def scan_overlays(self) -> List[Dict]:
        """
        Varre a pasta e lista todos os overlays disponíveis.

        Returns:
            Lista de dicts com 'path', 'name', 'type' (video/image)
        """
        overlays = []
        
        if not self.overlay_folder or not os.path.exists(self.overlay_folder):
            self.log(f"Pasta de overlays não encontrada: {self.overlay_folder}", "WARN")
            return overlays

        for filename in os.listdir(self.overlay_folder):
            filepath = os.path.join(self.overlay_folder, filename)
            
            if not os.path.isfile(filepath):
                continue

            ext = os.path.splitext(filename)[1].lower()
            
            if ext in self.VIDEO_EXTENSIONS:
                overlays.append({
                    "path": filepath,
                    "name": filename,
                    "type": "video"
                })
            elif ext in self.IMAGE_EXTENSIONS:
                overlays.append({
                    "path": filepath,
                    "name": filename,
                    "type": "image"
                })

        self.log(f"Overlays encontrados: {len(overlays)}", "OK")
        return overlays

    def get_next_overlay(self) -> Optional[Dict]:
        """
        Seleciona próximo overlay sem repetir os últimos usados.

        Returns:
            Dict com info do overlay ou None se não houver
        """
        available = self.scan_overlays()
        
        if not available:
            self.log("Nenhum overlay disponível", "WARN")
            return None

        # Filtrar overlays que não estão no histórico recente
        candidates = [o for o in available if o["path"] not in self.used_overlays]

        if not candidates:
            self.log("Todos os overlays foram usados. Resetando histórico.", "INFO")
            self.used_overlays.clear()
            candidates = available

        if candidates:
            selected = random.choice(candidates)
            self.log(f"Overlay selecionado: {selected['name']} ({selected['type']})", "OK")
            return selected

        return None

    def mark_overlay_used(self, overlay_path: str):
        """
        Marca overlay como usado.

        Args:
            overlay_path: Caminho do overlay
        """
        if overlay_path not in self.used_overlays:
            self.used_overlays.append(overlay_path)
            self._save_history()
            self.log(f"Overlay marcado como usado: {os.path.basename(overlay_path)}", "DEBUG")

    def apply_overlay(self, video_path: str, overlay_info: Dict, 
                     output_path: str, opacity: float = 0.3) -> Optional[str]:
        """
        Aplica overlay ao vídeo usando FFmpeg.

        Args:
            video_path: Caminho do vídeo base
            overlay_info: Dict com info do overlay (path, type)
            output_path: Caminho do vídeo de saída
            opacity: Opacidade do overlay (0.0-1.0)

        Returns:
            Caminho do vídeo com overlay ou None em caso de erro
        """
        overlay_path = overlay_info["path"]
        overlay_type = overlay_info["type"]

        try:
            if overlay_type == "image":
                # Overlay de imagem PNG (estático)
                cmd = [
                    "ffmpeg", "-y",
                    "-i", video_path,
                    "-i", overlay_path,
                    "-filter_complex",
                    f"[1:v]format=rgba,colorchannelmixer=aa={opacity}[ov];"
                    f"[0:v][ov]overlay=0:0:format=auto",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-c:a", "copy",
                    output_path
                ]
            else:
                # Overlay de vídeo (com loop)
                cmd = [
                    "ffmpeg", "-y",
                    "-i", video_path,
                    "-stream_loop", "-1",  # Loop infinito
                    "-i", overlay_path,
                    "-filter_complex",
                    f"[1:v]format=rgba,colorchannelmixer=aa={opacity}[ov];"
                    f"[0:v][ov]overlay=0:0:shortest=1:format=auto",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-c:a", "copy",
                    output_path
                ]

            self.log(f"Aplicando overlay: {overlay_info['name']}", "INFO")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                self.log(f"Overlay aplicado com sucesso", "OK")
                return output_path
            else:
                self.log(f"Erro FFmpeg: {result.stderr[:200]}", "ERROR")
                return None

        except Exception as e:
            self.log(f"Erro ao aplicar overlay: {e}", "ERROR")
            return None

    def apply_overlay_blend_screen(self, video_path: str, overlay_info: Dict,
                                   output_path: str) -> Optional[str]:
        """
        Aplica overlay com blend mode 'screen' (ideal para luz/partículas).

        Args:
            video_path: Caminho do vídeo base
            overlay_info: Dict com info do overlay
            output_path: Caminho do vídeo de saída

        Returns:
            Caminho do vídeo com overlay ou None
        """
        overlay_path = overlay_info["path"]
        overlay_type = overlay_info["type"]

        try:
            if overlay_type == "image":
                cmd = [
                    "ffmpeg", "-y",
                    "-i", video_path,
                    "-i", overlay_path,
                    "-filter_complex",
                    "[0:v][1:v]blend=all_mode=screen:all_opacity=0.5",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-c:a", "copy",
                    output_path
                ]
            else:
                cmd = [
                    "ffmpeg", "-y",
                    "-i", video_path,
                    "-stream_loop", "-1",
                    "-i", overlay_path,
                    "-filter_complex",
                    "[1:v]scale=iw:ih[ov];"
                    "[0:v][ov]blend=all_mode=screen:all_opacity=0.5:shortest=1",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-c:a", "copy",
                    output_path
                ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                return output_path
            else:
                self.log(f"Erro FFmpeg blend: {result.stderr[:200]}", "ERROR")
                return None

        except Exception as e:
            self.log(f"Erro ao aplicar overlay blend: {e}", "ERROR")
            return None

    def get_overlay_stats(self) -> Dict:
        """
        Retorna estatísticas dos overlays.

        Returns:
            Dict com estatísticas
        """
        overlays = self.scan_overlays()
        videos = sum(1 for o in overlays if o["type"] == "video")
        images = sum(1 for o in overlays if o["type"] == "image")
        
        return {
            "total": len(overlays),
            "videos": videos,
            "images": images,
            "used_recently": len(self.used_overlays),
            "folder": self.overlay_folder
        }

    def clear_history(self):
        """Limpa histórico de overlays usados."""
        self.used_overlays.clear()
        self._save_history()
        self.log("Histórico de overlays limpo", "OK")



