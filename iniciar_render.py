#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RenderX v3.2 - Equipe Matrix
=====================================
Versao: 3.2 - Detecção de Hardware + Setup Rápido + Pipeline SRT Avançado

RECURSOS:
- MODO LOTE: Processa multiplos audios de uma pasta
- PARALELISMO: 2-4 videos renderizados simultaneamente
- IMAGENS EXCLUSIVAS: Cada video usa imagens unicas (move para UTILIZADAS)
- GERAÇÃO DE IMAGENS VIA SRT: Agrupa blocos SRT e gera imagens via API Whisk
- ANIMAÇÕES VARIADAS: 6 tipos de animações Ken Burns (Zoom/Pan)
- BACKLOG DE ÁUDIOS: Sistema inteligente de rotação de áudios de fundo
- Zoom com ancora no CENTRO (getRotationMatrix2D)
- Pipeline Paralelo (clips + fades simultaneos)
- Modo Imagens Fixas com Loop
- Smart Crop 16:9 (sem bordas pretas)
- Overlay (poeira/particulas) com blend screen
- Mixagem de audio (narracao + musica de fundo)
- Interface CustomTkinter moderna e responsiva

ESTRUTURA DO LOTE:
- Pasta de Audios: Varrida recursivamente (subpastas suportadas)
- Pasta de Saida: Replica estrutura de pastas
- Banco de Imagens: Unico para todos, imagens usadas vao para UTILIZADAS

HARDWARE ALVO:
- CPU: Ryzen 7 9800X3D (16 threads)
- GPU: RTX 5070 Ti (16GB VRAM, NVENC)
- RAM: 64GB
"""

import cv2
import numpy as np
import os
import sys
import json
import threading
import tempfile
import shutil
import time
import queue
import subprocess
import math
import re
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from tkinter import font as tkfont, colorchooser
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

# Verificar se httpx está disponível (necessário para TTS)
try:
    import httpx
except ImportError:
    script_dir = Path(__file__).parent.absolute()
    venv_python = script_dir / "venv" / "bin" / "python3"
    
    # Tentar executar com o venv se disponível
    if venv_python.exists() and venv_python.is_file():
        print("=" * 60)
        print("Módulo 'httpx' não encontrado!")
        print("=" * 60)
        print("\nTentando executar com ambiente virtual...")
        print(f"\nExecute no terminal:")
        print(f"  cd {script_dir}")
        print(f"  source venv/bin/activate")
        print(f"  python iniciar_render.py")
        print(f"\nOu use o script:")
        print(f"  ./iniciar.sh")
        print("=" * 60)
        
        # Tentar executar automaticamente com o venv
        try:
            import subprocess
            result = subprocess.run(
                [str(venv_python), __file__] + sys.argv[1:],
                cwd=str(script_dir)
            )
            sys.exit(result.returncode)
        except Exception as e:
            print(f"\nErro ao executar com venv: {e}")
            print("\nPor favor, execute manualmente com o venv ativado.")
            sys.exit(1)
    else:
        print("=" * 60)
        print("ERRO: Módulo 'httpx' não encontrado!")
        print("=" * 60)
        print("\nInstale com:")
        print("  pip install httpx")
        print("\nOu crie um ambiente virtual:")
        print(f"  cd {script_dir}")
        print("  python3 -m venv venv")
        print("  source venv/bin/activate")
        print("  pip install httpx")
        print("=" * 60)
        sys.exit(1)

# Novos módulos
from srt_generator import SRTGenerator, SRTBlock
from image_generator import WhiskImageGenerator, ParallelWhiskGenerator
from ken_burns_engine import KenBurnsEngine
from audio_backlog_manager import AudioBacklogManager
from motion_shuffler import MotionShuffler, ImageEffectAssigner
from prompt_manager import (
    PromptManager,
    get_prompt_list,
    get_prompt_names,
    get_prompt_by_id,
    get_prompt_by_name,
    get_default_prompt_id,
    set_default_prompt,
    build_prompt_from_template,
    get_negative_prompt,
    get_full_prompt_config,
    IMAGE_PROMPTS_FILE
)
from overlay_manager import OverlayManager
from vsl_manager import (
    list_available_vsls,
    get_vsl_names,
    get_vsl_path,
    get_vsl_summary,
    get_vsls_with_display_names
)
from whisk_pool_manager import (
    get_pool_manager, 
    get_enabled_tokens, 
    get_tokens_summary,
    load_whisk_keys,
    save_whisk_keys,
    add_token_to_file,
    remove_token_from_file,
    update_or_add_token_by_email,
    WHISK_KEYS_FILE
)
# Sistema de licenças removido (legado em _LEGADO_LICENCAS/)
# from license_manager import (
#     verify_license,
#     activate_license,
#     generate_hwid,
#     get_hwid_short,
#     get_license_info
# )

# CustomTkinter para UI moderna
try:
    import customtkinter as ctk
    from tkinter import filedialog, messagebox
except ImportError:
    print("CustomTkinter nao encontrado. Instalando...")
    subprocess.run([sys.executable, "-m", "pip", "install", "customtkinter"], check=True)
    import customtkinter as ctk
    from tkinter import filedialog, messagebox

# =============================================================================
# TEMA E CORES
# =============================================================================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

CORES = {
    # Backgrounds - profundidade estilo GitHub Dark / VS Code
    "bg_dark": "#0D1117",         # Fundo principal
    "bg_section": "#161B22",      # Secoes/cards
    "bg_card": "#21262D",         # Cards elevados
    "bg_input": "#0D1117",        # Campos de entrada
    "bg_hover": "#30363D",        # Hover states
    
    # Accent - Azul premium
    "accent": "#58A6FF",          # Azul principal
    "accent_hover": "#79C0FF",    # Hover
    "accent_dark": "#388BFD",     # Variante escura
    
    # Semanticas
    "success": "#3FB950",         # Verde sucesso
    "warning": "#D29922",         # Amarelo aviso
    "error": "#F85149",           # Vermelho erro
    "info": "#58A6FF",            # Azul info
    
    # Texto
    "text": "#E6EDF3",            # Texto principal
    "text_dim": "#8B949E",        # Texto secundario
    "text_muted": "#484F58",      # Texto desabilitado
    
    # Bordas
    "border": "#30363D",
    "border_focus": "#58A6FF",
}


# =============================================================================
# CONFIGURACOES PADRAO
# =============================================================================
# Obter diretório do script para caminhos relativos
SCRIPT_DIR = Path(__file__).parent.absolute()
CONFIG_FILE = str(SCRIPT_DIR / "final_settings.json")
SUBTITLE_PRESETS_FILE = str(SCRIPT_DIR / "subtitle_presets.json")

DEFAULT_CONFIG = {
    # Modo Lote - Pastas principais
    "batch_input_folder": "",      # Pasta dos materiais (txt e audios, subpastas serao varridas)
    "batch_output_folder": "",     # Pasta de saida (replica estrutura)
    "batch_images_folder": "",     # Banco de imagens unico
    # Configuracoes de video
    "resolution": "720p",
    "zoom_mode": "zoom_in",
    "zoom_scale": 1.15,
    "image_duration": 8.0,
    "transition_duration": 1.0,
    "images_per_video": 50,        # Imagens por video (modo lote)
    "fps": 24,
    "video_bitrate": "4M",         # Bitrate do video (2M, 4M, 6M, 8M ou "auto")
    "video_crf": 23,               # CRF para modo CPU (qualidade constante)
    # Paralelismo
    "parallel_videos": 2,          # Videos simultaneos (1-4)
    "threads_per_video": 6,        # Threads por video
    # Opcoes
    "music_path": "",
    "music_volume": 0.2,
    "overlay_path": "",
    "overlay_opacity": 0.3,
    # Legendas
    "use_subtitles": False,
    "subtitle_method": "srt",      # "srt" ou "assemblyai"
    "assemblyai_key": "",
    "sub_font_name": "Arial",
    "sub_font_size": 48,
    "sub_color_primary": "#FFFFFF",
    "sub_color_outline": "#000000",
    "sub_color_shadow": "#80000000",
    "sub_color_karaoke": "#FFFF00",
    "sub_outline_size": 2,
    "sub_shadow_size": 2,
    "sub_use_karaoke": True,
    "sub_position": "2",           # ASS alignment (1-9)
    # TTS API
    "tts_provider": "none",        # "darkvi", "talkify", "none"
    "darkvi_api_key": "",
    "talkify_api_key": "",
    "tts_voice_id": "",
    "tts_voice_name": "",
    "tts_enabled": False,
    "generated_audio_folder": "AUDIOS_GERADOS",  # Subpasta para áudios gerados
    # VSL (Video Sales Letter)
    "use_vsl": True,  # Padrão: sempre ativo
    "vsl_folder": str(SCRIPT_DIR / "EFEITOS" / "VSLs"),
    "vsl_keywords_file": str(SCRIPT_DIR / "vsl_keywords.json"),
    "vsl_language": "portugues",  # Idioma das palavras-chave para busca
    "selected_vsl": "",  # Nome do arquivo da VSL selecionada
    "vsl_insertion_mode": "keyword",  # "keyword" (buscar palavra-chave), "fixed" (posição fixa) ou "range" (aleatório em range)
    "vsl_fixed_position": 60.0,  # Posição fixa em segundos (se vsl_insertion_mode="fixed")
    "vsl_range_start_min": 1.0,  # Minuto inicial do range (se vsl_insertion_mode="range")
    "vsl_range_end_min": 3.0,  # Minuto final do range (se vsl_insertion_mode="range")
    # Backlog Videos (Vídeos Introdutórios)
    "use_backlog_videos": False,
    "backlog_folder": str(SCRIPT_DIR / "EFEITOS" / "BACKLOG_VIDEOS"),
    "backlog_video_count": 6,
    "backlog_audio_volume": 0.25,
    "backlog_transition_duration": 0.5,  # Crossfade entre vídeos
    "backlog_fade_out_duration": 1.0,     # Fade out no último
    # Geração de Imagens via SRT
    "use_srt_based_images": False,
    "image_source": "generate",           # "generate" ou "backlog"
    "images_backlog_folder": "",          # Pasta de imagens backlog (se image_source="backlog")
    "whisk_api_tokens": [],               # Lista de tokens Whisk para geração
    "whisk_parallel_workers": 0,          # Workers paralelos (0 = número de tokens)
    "whisk_token_cooldown": 60,           # Cooldown após rate limit (segundos)
    "whisk_request_delay": 1.0,           # Delay entre requisições por token
    "selected_prompt_id": "spiritual_chosen",  # ID do prompt de imagem selecionado
    "use_varied_animations": True,        # Usar animações variadas (Ken Burns)
    "pan_amount": 0.2,                    # Quantidade de pan (0.0-1.0)
    # Pipeline SRT v3.1
    "subtitle_mode": "full",              # "full" (burn-in) ou "none" (exportar separado)
    "swap_every_n_cues": 3,               # Trocar imagem a cada N cues do SRT
    "prompt_file": "",                    # Arquivo externo de prompt (MD/TXT)
    "overlay_folder": "",                 # Pasta de overlays do usuário
    "use_random_overlays": False,         # Usar overlays aleatórios
    # Backlog de Áudios
    "audio_backlog_folder": "",          # Pasta de áudios backlog
    "use_audio_backlog": False,          # Usar backlog de áudios
    "audio_backlog_history_file": "audio_backlog_history.json",
    # Modo 1 Imagem (Pêndulo)
    "video_mode": "traditional",         # "traditional", "srt", "single_image"
    "pendulum_amplitude": 1.6,           # Amplitude da oscilação em graus
    "pendulum_crop_ratio": 1.0,          # Ratio de crop (0.5-1.0)
    "pendulum_zoom": 2.0,                # Zoom da imagem (1.0-3.0)
    "pendulum_cell_duration": 10.0,      # Duração da célula base em segundos
    "chroma_color": "00b140",            # Cor do chroma key (hex sem #)
    "chroma_similarity": 0.2,            # Similaridade do chroma key (0.01-0.5)
    "chroma_blend": 0.1                  # Blend do chroma key (0.0-0.5)
}

RESOLUTIONS = {
    "720p": (1280, 720),
    "1080p": (1920, 1080)
}

# Extensoes de audio suportadas
AUDIO_EXTENSIONS = ('.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac')
# Extensoes de texto suportadas
TEXT_EXTENSIONS = ('.txt',)


# =============================================================================
# SISTEMA DE RESERVA DE IMAGENS (EXCLUSIVIDADE)
# =============================================================================
class ImageReservationSystem:
    """Sistema para garantir imagens exclusivas entre videos."""

    def __init__(self, images_folder):
        self.images_folder = images_folder
        self.lock = threading.Lock()
        self.reserved = set()      # Imagens reservadas por workers ativos
        self.used = set()          # Imagens ja usadas (serao movidas depois)
        self.utilized_folder = os.path.join(images_folder, "UTILIZADAS")

    def scan_available_images(self):
        """Retorna lista de imagens disponiveis (exclui UTILIZADAS e reservadas)."""
        images = []
        if not os.path.exists(self.images_folder):
            return images

        for f in os.listdir(self.images_folder):
            full_path = os.path.join(self.images_folder, f)
            # Ignorar diretorios (incluindo UTILIZADAS)
            if os.path.isdir(full_path):
                continue
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                if full_path not in self.reserved and full_path not in self.used:
                    images.append(full_path)

        return sorted(images)

    def get_available_count(self):
        """Retorna quantidade de imagens disponiveis."""
        with self.lock:
            return len(self.scan_available_images())

    def reserve_images(self, count, shuffle=True):
        """Reserva N imagens exclusivas para um video."""
        import random
        with self.lock:
            available = self.scan_available_images()
            if len(available) < count:
                raise Exception(f"Imagens insuficientes: {len(available)} disponiveis, {count} necessarias")

            if shuffle:
                random.shuffle(available)

            selected = available[:count]
            self.reserved.update(selected)
            return selected

    def release_and_mark_used(self, images):
        """Libera reserva e marca como usadas (para mover depois)."""
        with self.lock:
            for img in images:
                self.reserved.discard(img)
                self.used.add(img)

    def move_used_images(self):
        """Move todas as imagens usadas para pasta UTILIZADAS."""
        with self.lock:
            os.makedirs(self.utilized_folder, exist_ok=True)
            moved_count = 0

            for img_path in list(self.used):
                if os.path.exists(img_path):
                    base_name = os.path.basename(img_path)
                    dest = os.path.join(self.utilized_folder, base_name)

                    # Evitar sobrescrita
                    if os.path.exists(dest):
                        name, ext = os.path.splitext(base_name)
                        dest = os.path.join(self.utilized_folder, f"{name}_{int(time.time())}{ext}")

                    try:
                        shutil.move(img_path, dest)
                        moved_count += 1
                    except Exception as e:
                        print(f"Erro ao mover {img_path}: {e}")

            self.used.clear()
            return moved_count


# =============================================================================
# ESTRUTURA DE JOB PARA LOTE
# =============================================================================
class BatchJob:
    """Representa um video a ser processado no lote."""

    def __init__(self, audio_path, source_folder, output_path, txt_path=None):
        self.audio_path = audio_path
        self.source_folder = source_folder
        self.output_path = output_path
        self.txt_path = txt_path  # Caminho do arquivo de texto associado (se existir)
        self.status = "pending"  # pending, processing, done, error
        self.used_images = []    # Lista de imagens usadas
        self.progress = 0.0      # Progresso individual (0-100)
        self.error_msg = ""      # Mensagem de erro se houver
        self.start_time = None
        self.end_time = None

    @property
    def name(self):
        return os.path.splitext(os.path.basename(self.audio_path))[0]

    @property
    def duration_str(self):
        if self.start_time and self.end_time:
            delta = self.end_time - self.start_time
            return f"{int(delta)}s"
        return "-"


# =============================================================================
# ESTRUTURAS DE DADOS
# =============================================================================
@dataclass
class ImageBlock:
    """Representa um bloco de imagem com animação e timing."""
    image_path: str
    start_time: float  # segundos
    end_time: float    # segundos
    duration: float    # end_time - start_time
    animation_type: str  # "zoom_in", "zoom_out", "pan_left_right", etc.


# =============================================================================
# MOTOR DE RENDERIZACAO - VERSAO FINAL
# =============================================================================
class FinalSlideshowEngine:
    """Motor de renderizacao com zoom centralizado e pipeline paralelo."""

    def __init__(self, log_queue, progress_callback=None):
        self.log_queue = log_queue
        self.progress_callback = progress_callback
        self.cancelled = False
        self.completed_clips = {}
        self.total_clips = 0
        self.lock = threading.Lock()
        self.start_time = None

        # Pipeline paralelo (clips + xfade simultaneos)
        self.ready_clips = {}              # clip_index -> clip_path
        self.xfade_queue = queue.Queue()   # pares (idx, clip_a, clip_b) para xfade
        self.xfaded_clips = {}             # idx -> path do clip xfaded
        self.next_xfade_index = 0          # proximo par a ser enfileirado
        self.clips_generation_done = False # flag para sinalizar fim da geracao

        # Compatibilidade com pipeline antigo (para render_all_clips_async)
        self.stitching_queue = queue.Queue()
        self.stitched_batches = set()
        self.chunks = []
        self.stitcher_done = False

        # Novos componentes
        self.srt_generator = SRTGenerator(self.log)
        self.ken_burns_engine = KenBurnsEngine(self.log)
        self.motion_shuffler = MotionShuffler(self.log)
        self.prompt_manager = PromptManager(str(SCRIPT_DIR), self.log)
        self.overlay_manager = OverlayManager(log_callback=self.log)

    def get_encoder_args(self, config, use_gpu=None):
        """Retorna os argumentos do encoder baseado na configuração de bitrate.
        
        Args:
            config: Dicionário de configuração com 'video_bitrate'
            use_gpu: Se None, usa self.check_gpu_available()
        
        Returns:
            Lista de argumentos para o encoder FFmpeg
        """
        if use_gpu is None:
            use_gpu = self.check_gpu_available()
        
        bitrate = config.get("video_bitrate", "4M")
        crf = config.get("video_crf", 23)
        
        if use_gpu:
            if bitrate == "auto":
                # Modo CRF para GPU (usa -cq em vez de -b:v)
                return ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", str(crf)]
            else:
                return ["-c:v", "h264_nvenc", "-preset", "p4", "-b:v", bitrate]
        else:
            if bitrate == "auto":
                # Modo CRF para CPU
                return ["-c:v", "libx264", "-preset", "fast", "-crf", str(crf)]
            else:
                # Modo bitrate fixo para CPU (usa -b:v com maxrate/bufsize)
                return ["-c:v", "libx264", "-preset", "fast", "-b:v", bitrate, "-maxrate", bitrate, "-bufsize", f"{int(bitrate[:-1])*2}M"]

    def log(self, message, level="INFO"):
        """Envia mensagem para a fila de log (thread-safe)."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = {
            "INFO": "",
            "OK": "[OK]",
            "WARN": "[!]",
            "ERROR": "[X]",
            "ENGINE": ">>",
            "STITCH": "[S]",
            "SPEED": "[*]"
        }.get(level, "")
        self.log_queue.put(f"[{timestamp}] {prefix} {message}")

    def update_progress(self, stage, current, total, extra=""):
        """Atualiza o progresso (thread-safe via callback)."""
        if self.progress_callback:
            self.progress_callback(stage, current, total, extra)

    def get_audio_duration(self, audio_path):
        """Obtem a duracao do audio em segundos."""
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            audio_path
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
            return float(result.stdout.strip())
        except Exception as e:
            self.log(f"Erro ao obter duracao do audio: {e}", "ERROR")
            return None

    def get_video_duration(self, video_path):
        """Obtem a duracao do video em segundos."""
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
            return float(result.stdout.strip())
        except:
            return None

    def get_video_resolution(self, video_path):
        """Obtém a resolução do vídeo (width, height)."""
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
            lines = result.stdout.strip().split('\n')
            width = int(lines[0])
            height = int(lines[1])
            return (width, height)
        except Exception as e:
            self.log(f"Erro ao obter resolução do vídeo: {e}", "ERROR")
            return None

    def get_image_files(self, folder):
        """Lista todas as imagens validas na pasta."""
        extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        images = []
        for f in Path(folder).iterdir():
            if f.is_file() and f.suffix.lower() in extensions:
                images.append(str(f))
        return sorted(images)

    def check_gpu_available(self):
        """Verifica se NVENC esta disponivel."""
        try:
            cmd = ["ffmpeg", "-hide_banner", "-encoders"]
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
            return "h264_nvenc" in result.stdout
        except:
            return False

    def calculate_images_needed(self, audio_duration, image_duration, transition_duration):
        """Calcula quantas imagens sao necessarias considerando overlap."""
        if image_duration <= transition_duration:
            return 0
        duracao_liquida = image_duration - transition_duration
        num_images = math.ceil((audio_duration - transition_duration) / duracao_liquida)
        return max(1, num_images)

    # =========================================================================
    # SMART CROP 16:9 - SEM BORDAS PRETAS
    # =========================================================================
    def smart_crop_16x9(self, img, target_w, target_h):
        """
        Crop centralizado para 16:9, SEM bordas pretas.

        1. Calcula a proporcao atual vs alvo
        2. Corta o excesso (laterais ou topo/base)
        3. Redimensiona para resolucao alvo
        """
        h, w = img.shape[:2]
        target_ratio = target_w / target_h  # 16:9 = 1.777...
        current_ratio = w / h

        if current_ratio > target_ratio:
            # Imagem mais larga que 16:9 - cortar laterais
            new_w = int(h * target_ratio)
            x_offset = (w - new_w) // 2
            img_cropped = img[:, x_offset:x_offset + new_w]
        else:
            # Imagem mais alta que 16:9 - cortar topo/base
            new_h = int(w / target_ratio)
            y_offset = (h - new_h) // 2
            img_cropped = img[y_offset:y_offset + new_h, :]

        # Redimensionar para resolucao alvo com LANCZOS4
        return cv2.resize(img_cropped, (target_w, target_h),
                          interpolation=cv2.INTER_LANCZOS4)

    # =========================================================================
    # ZOOM CENTRALIZADO - METODO OFICIAL OpenCV
    # =========================================================================
    def generate_zoom_clip(self, image_path, output_path, config, clip_index):
        """
        Gera clipe com zoom CENTRALIZADO usando getRotationMatrix2D.

        METODO OFICIAL OpenCV:
        cv2.getRotationMatrix2D(center, angle=0, scale) gera automaticamente
        a matriz que ancora a escala no centro especificado:

        M = [[scale,    0,    (1-scale)*cx],
             [   0,  scale,   (1-scale)*cy]]

        Quando scale > 1.0: imagem AMPLIA ao redor do centro
        Quando scale < 1.0: imagem ENCOLHE ao redor do centro
        """
        thread_name = threading.current_thread().name
        filename = Path(image_path).name

        try:
            if self.cancelled:
                return None

            # Parametros
            width, height = RESOLUTIONS[config["resolution"]]
            fps = config["fps"]
            duration = config["image_duration"]
            zoom_max = config["zoom_scale"]
            zoom_mode = config["zoom_mode"]
            total_frames = int(duration * fps)

            # 1. Carregar imagem
            img = cv2.imread(image_path)
            if img is None:
                self.log(f"[{thread_name}] Erro ao carregar: {filename}", "ERROR")
                return None

            # 2. Smart Crop 16:9 (SEM bordas pretas!)
            img_base = self.smart_crop_16x9(img, width, height)

            # 3. CENTRO da imagem (ANCORA DO ZOOM!)
            center = (width / 2.0, height / 2.0)

            # 4. Criar VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            if not out.isOpened():
                self.log(f"[{thread_name}] Erro ao criar VideoWriter", "ERROR")
                return None

            # 5. Gerar frames com ZOOM CENTRALIZADO
            for frame_num in range(total_frames):
                if self.cancelled:
                    out.release()
                    return None

                # Progresso normalizado (0.0 -> 1.0)
                t = frame_num / (total_frames - 1) if total_frames > 1 else 0

                # Calcular escala atual
                if zoom_mode == "zoom_in":
                    # Zoom In: 1.0 -> zoom_max (imagem AMPLIA, bordas cortadas)
                    # Parece que estamos "aproximando" do centro
                    scale = 1.0 + (zoom_max - 1.0) * t
                else:
                    # Zoom Out: zoom_max -> 1.0 (imagem volta ao normal)
                    # Parece que estamos "afastando" do centro
                    scale = zoom_max - (zoom_max - 1.0) * t

                # METODO OFICIAL OpenCV para escala centrada
                # getRotationMatrix2D(center, angle, scale)
                # angle=0 significa apenas escala, sem rotacao
                M = cv2.getRotationMatrix2D(center, 0, scale)

                # Aplicar transformacao com interpolacao de alta qualidade
                frame = cv2.warpAffine(
                    img_base, M, (width, height),
                    flags=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_REPLICATE
                )

                out.write(frame)

            out.release()
            return output_path

        except Exception as e:
            self.log(f"[{thread_name}] Excecao em {filename}: {str(e)}", "ERROR")
            return None

    # =========================================================================
    # CALLBACK DE CLIPE COMPLETO (PARA PIPELINE ASSINCRONO)
    # =========================================================================
    def on_clip_complete(self, clip_index, clip_path, temp_dir, config):
        """Callback quando um clipe fica pronto - verifica se pode costurar."""
        with self.lock:
            self.completed_clips[clip_index] = clip_path

            # Verificar se temos sequencia completa para stitching
            batch_size = 10
            batch_start = (clip_index // batch_size) * batch_size
            batch_end = min(batch_start + batch_size, self.total_clips)

            # Verificar se todos os clipes do batch estao prontos
            batch_complete = all(
                i in self.completed_clips
                for i in range(batch_start, batch_end)
            )

            if batch_complete and batch_start not in self.stitched_batches:
                self.stitched_batches.add(batch_start)
                # Coletar paths do batch em ordem
                batch_paths = [self.completed_clips[i] for i in range(batch_start, batch_end)]
                self.stitching_queue.put((batch_start, batch_end, batch_paths))

    # =========================================================================
    # WORKER DE STITCHING (CONSUMER)
    # =========================================================================
    def stitcher_worker(self, config, temp_dir):
        """Thread que consome batches e faz stitching em paralelo."""
        use_gpu = self.check_gpu_available()
        transition_duration = config["transition_duration"]

        while not self.stitcher_done or not self.stitching_queue.empty():
            try:
                batch_start, batch_end, batch_paths = self.stitching_queue.get(timeout=1)

                batch_num = batch_start // 10 + 1
                self.log(f">> Costurando Lote {batch_num} (clipes {batch_start+1}-{batch_end})...", "STITCH")

                # Costurar batch com xfade
                chunk_path = self.stitch_batch(batch_paths, batch_start, config, temp_dir, use_gpu)

                if chunk_path:
                    with self.lock:
                        self.chunks.append((batch_start, chunk_path))
                    self.log(f">> Lote {batch_num} pronto!", "STITCH")
                else:
                    self.log(f">> Lote {batch_num} FALHOU!", "ERROR")

            except queue.Empty:
                continue
            except Exception as e:
                self.log(f"Erro no stitcher: {e}", "ERROR")

    def stitch_batch(self, batch_paths, batch_start, config, temp_dir, use_gpu):
        """Costura um batch de clipes com xfade."""
        if len(batch_paths) == 1:
            return batch_paths[0]

        transition_duration = config["transition_duration"]
        current_video = batch_paths[0]

        for i in range(1, len(batch_paths)):
            if self.cancelled:
                return None

            next_clip = batch_paths[i]
            output_clip = os.path.join(temp_dir, f"batch_{batch_start:03d}_merge_{i:03d}.mp4")

            current_duration = self.get_video_duration(current_video)
            if current_duration is None:
                current_video = next_clip
                continue

            offset = max(0, current_duration - transition_duration)

            # Encoder args
            encoder_args = self.get_encoder_args(config, use_gpu)

            xfade_cmd = [
                "ffmpeg", "-y",
                "-i", current_video,
                "-i", next_clip,
                "-filter_complex",
                f"[0:v][1:v]xfade=transition=fade:duration={transition_duration}:offset={offset}[vout]",
                "-map", "[vout]",
                *encoder_args,
                "-pix_fmt", "yuv420p",
                output_clip
            ]

            result = subprocess.run(
                xfade_cmd,
                capture_output=True,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )

            if result.returncode != 0:
                current_video = next_clip
                continue

            current_video = output_clip

        return current_video

    # =========================================================================
    # PIPELINE PARALELO - CLIPS + XFADE SIMULTANEOS
    # =========================================================================
    def try_queue_xfade(self, clip_index, clip_path):
        """
        Verifica se pode enfileirar xfade assim que clips consecutivos ficam prontos.
        NAO espera batch de 10 - faz xfade assim que possivel.
        """
        with self.lock:
            self.ready_clips[clip_index] = clip_path

            # Tentar enfileirar todos os pares possiveis
            while self.next_xfade_index < self.total_clips - 1:
                idx = self.next_xfade_index
                # Precisa de clip idx E idx+1
                if idx in self.ready_clips and (idx + 1) in self.ready_clips:
                    pair = (idx, self.ready_clips[idx], self.ready_clips[idx + 1])
                    self.xfade_queue.put(pair)
                    self.next_xfade_index += 1
                else:
                    break  # Aguardar proximo clip

    def xfade_worker(self, config, temp_dir, use_gpu):
        """Worker que processa xfade de pares de clips continuamente."""
        transition = config["transition_duration"]

        while not self.cancelled:
            try:
                # Pega par da fila (timeout para verificar cancelamento)
                pair = self.xfade_queue.get(timeout=0.5)
                if pair is None:  # Sentinel para parar
                    break

                idx, clip_a, clip_b = pair
                output = os.path.join(temp_dir, f"xfade_{idx:04d}.mp4")

                result = self.do_xfade(clip_a, clip_b, output, transition, use_gpu)

                if result:
                    with self.lock:
                        self.xfaded_clips[idx] = output
                    self.log(f"Xfade {idx+1}->{idx+2} OK", "STITCH")

            except queue.Empty:
                # Verificar se geracao terminou e fila vazia
                if self.clips_generation_done and self.xfade_queue.empty():
                    break
                continue
            except Exception as e:
                self.log(f"Erro no xfade_worker: {e}", "ERROR")

    def do_xfade(self, clip_a, clip_b, output, transition_duration, use_gpu):
        """Faz xfade entre dois clips."""
        duration_a = self.get_video_duration(clip_a)
        if duration_a is None:
            return None

        offset = max(0, duration_a - transition_duration)

        encoder_args = self.get_encoder_args({}, use_gpu)  # Config padrão para xfade

        cmd = [
            "ffmpeg", "-y",
            "-i", clip_a,
            "-i", clip_b,
            "-filter_complex",
            f"[0:v][1:v]xfade=transition=fade:duration={transition_duration}:offset={offset}[vout]",
            "-map", "[vout]",
            *encoder_args,
            "-pix_fmt", "yuv420p",
            output
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )

        return output if result.returncode == 0 else None

    def merge_xfaded_clips(self, temp_dir):
        """Concatena clips xfaded em ordem para formar o video final."""
        if len(self.xfaded_clips) == 0:
            # Se so tem 1 clip, nao houve xfade
            if len(self.ready_clips) == 1:
                return list(self.ready_clips.values())[0]
            return None

        # Ordenar clips xfaded por indice
        sorted_indices = sorted(self.xfaded_clips.keys())
        sorted_clips = [self.xfaded_clips[i] for i in sorted_indices]

        if len(sorted_clips) == 1:
            return sorted_clips[0]

        # Concatenar todos os clips xfaded
        # Cada xfade_N contem clip_N fundido com clip_N+1
        # Precisamos encadear: xfade_0 + xfade_1 + xfade_2 ...
        # Mas isso nao funciona assim - cada xfade ja contem a transicao

        # Na verdade, o resultado de xfade(clip_0, clip_1) ja e o clip fundido
        # Precisamos aplicar xfade sequencialmente

        # Para simplificar: o ultimo xfade_N contem o video completo ate clip_N+1
        # Vamos encadear diferente - usar o metodo de concat simples
        return sorted_clips[-1] if sorted_clips else None

    def render_all_clips_parallel(self, images, config, temp_dir):
        """
        Pipeline PARALELO: geracao de clips + xfade simultaneos.
        Nao espera batches de 10 - faz xfade assim que pares ficam prontos.
        """
        self.total_clips = len(images)
        self.ready_clips = {}
        self.xfaded_clips = {}
        self.next_xfade_index = 0
        self.clips_generation_done = False
        self.start_time = time.time()

        num_threads = config["threads"]
        # Dividir threads: ~70% geracao, ~30% xfade
        gen_threads = max(1, int(num_threads * 0.7))
        xfade_threads = max(1, int(num_threads * 0.3))

        use_gpu = self.check_gpu_available()

        self.log("=" * 50, "INFO")
        self.log(f"PIPELINE PARALELO - {gen_threads} gen + {xfade_threads} xfade", "ENGINE")
        self.log(f"Total de clipes: {self.total_clips}", "INFO")
        self.log(f"Zoom: {config['zoom_mode']} -> {config['zoom_scale']}x", "INFO")
        self.log("=" * 50, "INFO")

        # Iniciar workers de xfade (consumers)
        xfade_workers = []
        for _ in range(xfade_threads):
            t = threading.Thread(
                target=self.xfade_worker,
                args=(config, temp_dir, use_gpu),
                daemon=True
            )
            t.start()
            xfade_workers.append(t)

        # Gerar clips (producers)
        completed_count = 0
        with ThreadPoolExecutor(max_workers=gen_threads, thread_name_prefix="Gen") as executor:
            futures = {}

            for i, img_path in enumerate(images):
                if self.cancelled:
                    break

                clip_path = os.path.join(temp_dir, f"clip_{i:04d}.mp4")
                future = executor.submit(
                    self.generate_zoom_clip,
                    img_path, clip_path, config, i
                )
                futures[future] = i

            for future in as_completed(futures):
                if self.cancelled:
                    self.clips_generation_done = True
                    for _ in range(xfade_threads):
                        self.xfade_queue.put(None)
                    return None

                try:
                    idx = futures[future]
                    result = future.result()

                    if result:
                        completed_count += 1

                        elapsed = time.time() - self.start_time
                        speed = completed_count / elapsed if elapsed > 0 else 0
                        remaining = (self.total_clips - completed_count) / speed if speed > 0 else 0

                        self.log(f"Clipe {completed_count}/{self.total_clips} ({speed:.2f}/s)", "OK")
                        self.update_progress(
                            "clips",
                            completed_count,
                            self.total_clips,
                            f"{speed:.2f}/s | ETA: {int(remaining)}s"
                        )

                        # Enfileirar xfade se par disponivel
                        self.try_queue_xfade(idx, result)
                    else:
                        self.log(f"Clipe {idx+1} FALHOU", "ERROR")

                except Exception as e:
                    self.log(f"Erro: {e}", "ERROR")

        # Sinalizar fim da geracao
        self.clips_generation_done = True

        # Enviar sentinels para parar workers
        for _ in range(xfade_threads):
            self.xfade_queue.put(None)

        # Aguardar workers finalizarem
        for t in xfade_workers:
            t.join(timeout=300)

        if self.cancelled:
            return None

        self.log("-" * 50, "INFO")
        self.log(f"Geracao concluida: {completed_count} clips", "OK")
        self.log(f"Xfades realizados: {len(self.xfaded_clips)}", "OK")

        # Retornar lista de chunks para compatibilidade com o resto do pipeline
        # Se temos xfades, retornar o video resultante
        if len(self.xfaded_clips) > 0:
            # Fazer merge sequencial dos xfades
            return self.merge_xfaded_sequential(temp_dir, use_gpu, config)
        elif len(self.ready_clips) == 1:
            # So 1 clip, retornar diretamente
            return [list(self.ready_clips.values())[0]]
        else:
            return None

    def merge_xfaded_sequential(self, temp_dir, use_gpu, config):
        """Faz merge sequencial dos clips xfaded."""
        # Os xfaded_clips precisam ser fundidos sequencialmente
        # xfade_0 = clip_0 + clip_1 com fade
        # xfade_1 = clip_1 + clip_2 com fade
        # Precisamos: xfade_0 fundido com clip_2, clip_3, etc.

        # Abordagem mais simples: fazer xfade encadeado
        if len(self.xfaded_clips) == 0:
            return None

        sorted_indices = sorted(self.xfaded_clips.keys())

        # O primeiro xfade ja contem clip_0 + clip_1
        current = self.xfaded_clips[sorted_indices[0]]

        # Para cada xfade subsequente, precisamos adicionar o clip extra
        # xfade_1 tem clip_1 + clip_2, mas clip_1 ja esta em xfade_0
        # Entao precisamos adicionar clip_2 a current, depois clip_3, etc.

        # Simplificacao: como cada xfade_N = clip_N + clip_N+1
        # e xfade_N+1 = clip_N+1 + clip_N+2
        # ha sobreposicao. Vamos usar o ready_clips diretamente.

        # Metodo correto: fazer xfade encadeado
        transition = config["transition_duration"]

        for i in range(1, self.total_clips):
            if i not in self.ready_clips:
                continue

            next_clip = self.ready_clips[i]
            output = os.path.join(temp_dir, f"chain_{i:04d}.mp4")

            result = self.do_xfade(current, next_clip, output, transition, use_gpu)
            if result:
                current = result
            else:
                self.log(f"Erro no merge encadeado {i}", "ERROR")

        return [current] if current else None

    # =========================================================================
    # RENDER PARALELO COM PIPELINE ASSINCRONO (LEGADO)
    # =========================================================================
    def render_all_clips_async(self, images, config, temp_dir):
        """Gera clipes em paralelo com stitching assincrono."""
        self.total_clips = len(images)
        self.completed_clips = {}
        self.stitched_batches = set()
        self.chunks = []
        self.stitcher_done = False
        self.start_time = time.time()

        num_threads = config["threads"]

        self.log("=" * 50, "INFO")
        self.log(f"RENDER FARM ASSINCRONO - {num_threads} threads", "ENGINE")
        self.log(f"Total de clipes: {self.total_clips}", "INFO")
        self.log(f"Zoom: {config['zoom_mode']} -> {config['zoom_scale']}x (CENTRALIZADO)", "INFO")
        self.log("=" * 50, "INFO")

        # Iniciar thread de stitching (consumer)
        stitcher_thread = threading.Thread(
            target=self.stitcher_worker,
            args=(config, temp_dir),
            daemon=True
        )
        stitcher_thread.start()

        # Gerar clipes (producer)
        completed_count = 0
        with ThreadPoolExecutor(max_workers=num_threads, thread_name_prefix="Worker") as executor:
            futures = {}

            for i, img_path in enumerate(images):
                if self.cancelled:
                    break

                clip_path = os.path.join(temp_dir, f"clip_{i:04d}.mp4")
                future = executor.submit(
                    self.generate_zoom_clip,
                    img_path, clip_path, config, i
                )
                futures[future] = i

            for future in as_completed(futures):
                if self.cancelled:
                    self.stitcher_done = True
                    executor.shutdown(wait=False, cancel_futures=True)
                    return None

                try:
                    idx = futures[future]
                    result = future.result()

                    if result:
                        completed_count += 1

                        elapsed = time.time() - self.start_time
                        speed = completed_count / elapsed if elapsed > 0 else 0
                        remaining = (self.total_clips - completed_count) / speed if speed > 0 else 0

                        self.log(f"Clipe {completed_count}/{self.total_clips} OK ({speed:.2f}/s)", "OK")
                        self.update_progress(
                            "clips",
                            completed_count,
                            self.total_clips,
                            f"{speed:.2f} clips/s | ETA: {int(remaining)}s"
                        )

                        # Callback para verificar stitching
                        self.on_clip_complete(idx, result, temp_dir, config)
                    else:
                        self.log(f"Clipe {idx+1} FALHOU", "ERROR")

                except Exception as e:
                    self.log(f"Erro ao coletar resultado: {e}", "ERROR")

        # Sinalizar fim da geracao
        self.stitcher_done = True
        stitcher_thread.join(timeout=300)  # Aguardar stitching finalizar

        if self.cancelled:
            return None

        # Ordenar chunks
        self.chunks.sort(key=lambda x: x[0])
        chunk_paths = [path for idx, path in self.chunks]

        self.log("-" * 50, "INFO")
        self.log(f"RENDER FARM concluido!", "OK")
        self.log(f"  Clipes: {completed_count}/{self.total_clips}", "INFO")
        self.log(f"  Chunks: {len(chunk_paths)}", "INFO")
        self.log("-" * 50, "INFO")

        return chunk_paths

    # =========================================================================
    # CONCATENAR CHUNKS FINAIS
    # =========================================================================
    def concat_chunks(self, chunk_paths, output_path, temp_dir):
        """Concatena os chunks finais."""
        if len(chunk_paths) == 0:
            return None
        if len(chunk_paths) == 1:
            shutil.copy(chunk_paths[0], output_path)
            return output_path

        self.log(f"Concatenando {len(chunk_paths)} chunks...", "INFO")

        list_file = os.path.join(temp_dir, "chunks_list.txt")
        with open(list_file, "w", encoding="utf-8") as f:
            for chunk_path in chunk_paths:
                abs_path = os.path.abspath(chunk_path)
                safe_path = abs_path.replace("\\", "/").replace("'", "'\\''")
                f.write(f"file '{safe_path}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_file,
            "-c", "copy",
            output_path
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )

        if result.returncode != 0:
            self.log(f"Erro no concat: {result.stderr[-300:] if result.stderr else ''}", "ERROR")
            return None

        self.log("Concatenacao OK!", "OK")
        return output_path

    # =========================================================================
    # OVERLAY (POEIRA/PARTICULAS)
    # =========================================================================
    def apply_overlay(self, video_path, overlay_path, output_path, opacity=0.3):
        """Aplica overlay com blend screen (remove preto) em toda a duração do vídeo."""
        self.log(f"Aplicando overlay (opacidade: {opacity:.0%})...", "INFO")

        use_gpu = self.check_gpu_available()
        encoder_args = self.get_encoder_args({}, use_gpu)  # Config padrão para overlay

        # Obter duração do vídeo principal
        video_duration = self.get_video_duration(video_path)
        if video_duration is None:
            self.log("Não foi possível obter duração do vídeo", "WARN")
            video_duration = 0  # Usar shortest como fallback

        # Obter resolução do vídeo principal para ajustar overlay
        video_resolution = self.get_video_resolution(video_path)
        if video_resolution:
            video_width, video_height = video_resolution
        else:
            # Fallback: assumir 720p
            video_width, video_height = 1280, 720

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-stream_loop", "-1",
            "-i", overlay_path,
            "-filter_complex",
            # Processar overlay: ajustar resolução e aplicar opacidade
            f"[1:v]scale={video_width}:{video_height}:force_original_aspect_ratio=decrease,"
            f"pad={video_width}:{video_height}:(ow-iw)/2:(oh-ih)/2:color=black@0,"
            f"format=rgba,colorchannelmixer=aa={opacity}[ov];"
            # Aplicar overlay acima do vídeo (overlay normal preserva cores do vídeo base)
            f"[0:v][ov]overlay=0:0:eof_action=pass[out]",
            "-map", "[out]",
            "-map", "0:a?",  # Mapear áudio do vídeo principal se existir
            *encoder_args,
            "-c:a", "copy",  # Copiar áudio sem re-encoding
            "-pix_fmt", "yuv420p",
        ]
        
        # Se temos duração conhecida, limitar pela duração do vídeo principal
        if video_duration and video_duration > 0:
            cmd.extend(["-t", str(video_duration)])
        
        cmd.append(output_path)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )

        if result.returncode != 0:
            self.log(f"Erro no overlay: {result.stderr[-200:] if result.stderr else ''}", "ERROR")
            return None

        self.log("Overlay aplicado em toda duração!", "OK")
        return output_path

    # =========================================================================
    # MIXAGEM DE AUDIO
    # =========================================================================
    def mix_audio(self, video_path, narration_path, music_path, output_path, music_volume=0.2, config=None):
        """
        Mixa narracao com musica de fundo.
        
        Se houver VSL inserida no vídeo, preserva o áudio da VSL durante seu intervalo
        e aplica narração + música apenas nas partes fora da VSL.
        """
        self.log(f"Mixando audio (musica: {music_volume:.0%})...", "INFO")

        # Verificar se deve usar backlog de áudios (apenas para música de fundo)
        # Nota: Os áudios de narração/voz (narration_path) não são afetados pelo backlog
        original_music_path = music_path  # Guardar caminho original para comparação
        if config and config.get("use_audio_backlog", False):
            audio_backlog_folder = config.get("audio_backlog_folder", "")
            if audio_backlog_folder and os.path.exists(audio_backlog_folder):
                try:
                    audio_manager = AudioBacklogManager(
                        history_file=config.get("audio_backlog_history_file", "audio_backlog_history.json"),
                        log_callback=self.log
                    )
                    selected_audio = audio_manager.get_next_audio(audio_backlog_folder)
                    if selected_audio and os.path.exists(selected_audio):
                        music_path = selected_audio
                        self.log(f"Usando áudio de fundo do backlog: {os.path.basename(selected_audio)}", "INFO")
                        # Registrar áudio usado após mixagem bem-sucedida
                        # (será feito no final se sucesso)
                    else:
                        self.log("Nenhum áudio disponível no backlog, usando música padrão configurada", "WARN")
                except Exception as e:
                    self.log(f"Erro ao usar backlog de áudios de fundo: {str(e)}", "WARN")
            else:
                self.log("Pasta de backlog de áudios não configurada ou não encontrada, usando música padrão", "WARN")

        use_gpu = self.check_gpu_available()
        encoder_args = self.get_encoder_args(config if config else {}, use_gpu)

        # Verificar se há VSL inserida - se sim, precisamos preservar o áudio da VSL
        vsl_start = config.get("vsl_inserted_start") if config else None
        vsl_duration = config.get("vsl_inserted_duration") if config else None
        
        if vsl_start is not None and vsl_duration is not None:
            # ============================================================
            # MODO COM VSL: Preservar áudio da VSL, aplicar narração+música fora
            # ============================================================
            self.log(f"VSL detectada: {vsl_start:.2f}s - {vsl_start + vsl_duration:.2f}s", "INFO")
            self.log("Preservando áudio da VSL durante mixagem...", "INFO")
            
            vsl_end = vsl_start + vsl_duration
            
            # Estratégia:
            # 1. Criar áudio mixado (narração + música) para a duração total
            # 2. Usar asplit para dividir em duas cópias (FFmpeg não permite usar mesmo stream 2x)
            # 3. Cortar parte1 (0 até vsl_start) e parte2 (de vsl_start em diante)
            # 4. Extrair áudio da VSL do vídeo (já está embutido)
            # 5. Concatenar: parte1_mixado + audio_vsl + parte2_mixado
            
            # Filter complex com asplit para poder usar o stream mixado duas vezes
            filter_complex = (
                # Preparar música de fundo com loop e volume
                f"[2:a]volume={music_volume},aloop=loop=-1:size=2e+09[bg];"
                # Mixar narração com música de fundo
                f"[1:a][bg]amix=inputs=2:duration=first[mixed];"
                # IMPORTANTE: Usar asplit para criar duas cópias do áudio mixado
                f"[mixed]asplit=2[mixed1][mixed2];"
                # Parte 1 do áudio mixado (0 até vsl_start)
                f"[mixed1]atrim=0:{vsl_start},asetpts=PTS-STARTPTS[mix_part1];"
                # Parte 2 do áudio mixado (de vsl_start em diante - continua após a VSL)
                f"[mixed2]atrim={vsl_start},asetpts=PTS-STARTPTS[mix_part2];"
                # Extrair áudio da VSL do vídeo (de vsl_start até vsl_end no vídeo final)
                f"[0:a]atrim={vsl_start}:{vsl_end},asetpts=PTS-STARTPTS[vsl_audio];"
                # Concatenar: parte1 + áudio_vsl + parte2
                f"[mix_part1][vsl_audio][mix_part2]concat=n=3:v=0:a=1[aout]"
            )
            
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,      # Input 0: vídeo com VSL (tem áudio da VSL embutido)
                "-i", narration_path,  # Input 1: narração
                "-stream_loop", "-1",
                "-i", music_path,      # Input 2: música de fundo
                "-filter_complex", filter_complex,
                "-map", "0:v",
                "-map", "[aout]",
                *encoder_args,
                "-c:a", "aac", "-b:a", "192k",
                "-pix_fmt", "yuv420p",
                "-shortest",
                output_path
            ]
        else:
            # ============================================================
            # MODO SEM VSL: Mixagem normal
            # ============================================================
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", narration_path,
                "-stream_loop", "-1",
                "-i", music_path,
                "-filter_complex",
                f"[2:a]volume={music_volume}[bg];"
                f"[1:a][bg]amix=inputs=2:duration=first[aout]",
                "-map", "0:v",
                "-map", "[aout]",
                *encoder_args,
                "-c:a", "aac", "-b:a", "192k",
                "-pix_fmt", "yuv420p",
                "-shortest",
                output_path
            ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )

        if result.returncode != 0:
            self.log(f"Erro na mixagem: {result.stderr[-300:] if result.stderr else ''}", "ERROR")
            return None

        # Registrar áudio de fundo usado no backlog (se estava usando backlog)
        # Nota: Isso só afeta música de fundo, não áudios de narração
        if config and config.get("use_audio_backlog", False) and music_path != original_music_path:
            try:
                audio_manager = AudioBacklogManager(
                    history_file=config.get("audio_backlog_history_file", "audio_backlog_history.json"),
                    log_callback=self.log
                )
                audio_manager.mark_audio_used(music_path)
                self.log(f"Áudio de fundo registrado no backlog: {os.path.basename(music_path)}", "OK")
            except Exception as e:
                self.log(f"Erro ao registrar áudio de fundo usado: {str(e)}", "WARN")

        self.log("Audio mixado!", "OK")
        return output_path

    # =========================================================================
    # FINALIZACAO SIMPLES (SO AUDIO)
    # =========================================================================
    def finalize_simple(self, video_path, audio_path, output_path, config=None):
        """
        Adiciona apenas o audio principal (narração).
        
        Se houver VSL inserida, preserva o áudio da VSL e aplica narração apenas fora.
        """
        self.log("Finalizando video (adicionando audio)...", "INFO")

        use_gpu = self.check_gpu_available()
        if use_gpu:
            encoder_args = ["-c:v", "h264_nvenc", "-preset", "p4", "-b:v", "8M"]
        else:
            encoder_args = ["-c:v", "libx264", "-preset", "fast", "-crf", "23"]

        # Verificar se há VSL inserida
        vsl_start = config.get("vsl_inserted_start") if config else None
        vsl_duration = config.get("vsl_inserted_duration") if config else None
        
        if vsl_start is not None and vsl_duration is not None:
            # ============================================================
            # MODO COM VSL: Preservar áudio da VSL, aplicar narração fora
            # ============================================================
            self.log(f"VSL detectada: {vsl_start:.2f}s - {vsl_start + vsl_duration:.2f}s", "INFO")
            self.log("Preservando áudio da VSL...", "INFO")
            
            vsl_end = vsl_start + vsl_duration
            
            filter_complex = (
                # IMPORTANTE: Usar asplit para criar duas cópias da narração
                f"[1:a]asplit=2[narr1][narr2];"
                # Parte 1 da narração (0 até vsl_start)
                f"[narr1]atrim=0:{vsl_start},asetpts=PTS-STARTPTS[narr_part1];"
                # Parte 2 da narração (de vsl_start em diante)
                f"[narr2]atrim={vsl_start},asetpts=PTS-STARTPTS[narr_part2];"
                # Extrair áudio da VSL do vídeo
                f"[0:a]atrim={vsl_start}:{vsl_end},asetpts=PTS-STARTPTS[vsl_audio];"
                # Concatenar: narração_parte1 + áudio_vsl + narração_parte2
                f"[narr_part1][vsl_audio][narr_part2]concat=n=3:v=0:a=1[aout]"
            )
            
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,   # Input 0: vídeo com VSL
                "-i", audio_path,   # Input 1: narração
                "-filter_complex", filter_complex,
                "-map", "0:v",
                "-map", "[aout]",
                *encoder_args,
                "-c:a", "aac", "-b:a", "192k",
                "-pix_fmt", "yuv420p",
                "-shortest",
                output_path
            ]
        else:
            # ============================================================
            # MODO SEM VSL: Adicionar narração normalmente
            # ============================================================
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", audio_path,
                "-map", "0:v",
                "-map", "1:a",
                *encoder_args,
                "-c:a", "aac", "-b:a", "192k",
                "-pix_fmt", "yuv420p",
                "-shortest",
                output_path
            ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )

        if result.returncode != 0:
            self.log(f"Erro ao finalizar: {result.stderr[-300:] if result.stderr else ''}", "ERROR")
            return False

        self.log("Finalizacao OK!", "OK")
        return True

    # =========================================================================
    # MODO 1 IMAGEM - PÊNDULO COM LOOP SEAMLESS
    # =========================================================================
    def generate_pendulum_cell(self, image_path: str, config: dict, temp_dir: str) -> Optional[str]:
        """
        Gera célula base com efeito pêndulo usando OpenCV.
        
        O efeito pêndulo cria uma oscilação suave da imagem que faz um ciclo
        completo (ida e volta) durante a duração da célula, permitindo loop
        seamless sem cortes visíveis.
        
        Args:
            image_path: Caminho da imagem
            config: Configuração do vídeo
            temp_dir: Diretório temporário
            
        Returns:
            Caminho do vídeo da célula ou None em caso de erro
        """
        try:
            self.log("Gerando célula pêndulo...", "INFO")
            
            # Carregar imagem
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                self.log(f"Erro ao carregar imagem: {image_path}", "ERROR")
                return None
            
            has_alpha = img.shape[2] == 4 if len(img.shape) > 2 else False
            
            # Parâmetros do pêndulo
            amplitude_graus = config.get("pendulum_amplitude", 1.6)
            crop_ratio = config.get("pendulum_crop_ratio", 1.0)
            zoom = config.get("pendulum_zoom", 2.0)
            cell_duration = config.get("pendulum_cell_duration", 10.0)
            
            # Parâmetros de vídeo
            width, height = RESOLUTIONS[config.get("resolution", "720p")]
            fps = config.get("fps", 24)
            total_frames = int(cell_duration * fps)
            
            original_w = img.shape[1]
            original_h = img.shape[0]
            
            self.log(f"Pêndulo: amplitude={amplitude_graus}°, zoom={zoom}x, duração={cell_duration}s", "INFO")
            
            # Aplicar crop se necessário
            if crop_ratio < 1.0:
                crop_w = int(original_w * crop_ratio)
                crop_h = int(original_h * crop_ratio)
                crop_x = (original_w - crop_w) // 2
                crop_y = (original_h - crop_h) // 2
                img_processed = img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
            else:
                img_processed = img.copy()
                crop_w = original_w
                crop_h = original_h
            
            # Aplicar zoom
            zoom_w = int(crop_w * zoom)
            zoom_h = int(crop_h * zoom)
            img_zoomed = cv2.resize(img_processed, (zoom_w, zoom_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Calcular escala necessária para cobrir rotação
            max_angle_rad = abs(amplitude_graus) * math.pi / 180.0
            rotation_factor = 1.0 / math.cos(max_angle_rad) if max_angle_rad > 0 else 1.0
            
            canvas_diagonal = math.sqrt(width**2 + height**2)
            img_diagonal = math.sqrt(zoom_w**2 + zoom_h**2)
            scale_needed = (canvas_diagonal * rotation_factor) / img_diagonal if img_diagonal > 0 else 1.0
            
            final_w = int(zoom_w * scale_needed)
            final_h = int(zoom_h * scale_needed)
            
            # Garantir tamanho mínimo
            if final_w < width:
                final_w = int(width * 1.1)
            if final_h < height:
                final_h = int(height * 1.1)
            
            img_resized = cv2.resize(img_zoomed, (final_w, final_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Criar vídeo
            temp_video_path = os.path.join(temp_dir, "pendulum_cell.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                self.log("Erro ao criar video writer!", "ERROR")
                return None
            
            center_img_x = final_w // 2
            center_img_y = final_h // 2
            center_canvas_x = width // 2
            center_canvas_y = height // 2
            
            half_cycle = cell_duration / 2.0
            
            for frame_num in range(total_frames):
                if self.cancelled:
                    out.release()
                    return None
                
                t = (frame_num / fps) % cell_duration
                
                # Movimento de pêndulo: -amplitude -> +amplitude -> -amplitude
                if t <= half_cycle:
                    progress = t / half_cycle
                    angle_degrees = -amplitude_graus + (2 * amplitude_graus * progress)
                else:
                    progress = (t - half_cycle) / half_cycle
                    angle_degrees = amplitude_graus - (2 * amplitude_graus * progress)
                
                # Rotacionar imagem
                rotation_matrix = cv2.getRotationMatrix2D((center_img_x, center_img_y), angle_degrees, 1.0)
                
                if has_alpha:
                    img_rotated = cv2.warpAffine(img_resized, rotation_matrix, (final_w, final_h),
                                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                                 borderValue=(0, 0, 0, 0))
                else:
                    img_rotated = cv2.warpAffine(img_resized, rotation_matrix, (final_w, final_h),
                                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                                 borderValue=(0, 0, 0))
                
                # Criar frame de saída
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                
                # Posicionar imagem no centro do canvas
                x = center_canvas_x - final_w // 2
                y = center_canvas_y - final_h // 2
                
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(width, x + final_w)
                y2 = min(height, y + final_h)
                
                img_x1 = max(0, -x)
                img_y1 = max(0, -y)
                img_x2 = img_x1 + (x2 - x1)
                img_y2 = img_y1 + (y2 - y1)
                
                if x2 > x1 and y2 > y1:
                    if has_alpha:
                        img_alpha = img_rotated[img_y1:img_y2, img_x1:img_x2, 3:4] / 255.0
                        frame[y1:y2, x1:x2] = (
                            frame[y1:y2, x1:x2] * (1 - img_alpha) +
                            img_rotated[img_y1:img_y2, img_x1:img_x2, :3] * img_alpha
                        ).astype(np.uint8)
                    else:
                        frame[y1:y2, x1:x2] = img_rotated[img_y1:img_y2, img_x1:img_x2]
                
                out.write(frame)
                
                # Atualizar progresso
                if frame_num % (fps * 2) == 0:  # A cada 2 segundos
                    progress_pct = (frame_num / total_frames) * 100
                    self.update_progress("pendulum", frame_num, total_frames, f"{progress_pct:.0f}%")
            
            out.release()
            
            self.log(f"Célula pêndulo gerada: {cell_duration}s", "OK")
            return temp_video_path
            
        except Exception as e:
            self.log(f"Erro ao gerar célula pêndulo: {str(e)}", "ERROR")
            return None

    def apply_overlay_to_cell(self, pendulum_video: str, config: dict, temp_dir: str) -> Optional[str]:
        """
        Aplica overlay com chroma key na célula base do pêndulo.
        
        Args:
            pendulum_video: Caminho do vídeo da célula pêndulo
            config: Configuração do vídeo
            temp_dir: Diretório temporário
            
        Returns:
            Caminho do vídeo com overlay ou o vídeo original se não houver overlay
        """
        try:
            overlay_path = config.get("overlay_path", "")
            
            if not overlay_path or not os.path.exists(overlay_path):
                self.log("Sem overlay configurado, usando célula pêndulo pura", "INFO")
                return pendulum_video
            
            self.log("Aplicando overlay com chroma key...", "INFO")
            
            # Parâmetros do chroma key
            chroma_color = config.get("chroma_color", "00b140")
            similarity = config.get("chroma_similarity", 0.2)
            blend = config.get("chroma_blend", 0.1)
            
            width, height = RESOLUTIONS[config.get("resolution", "720p")]
            cell_duration = config.get("pendulum_cell_duration", 10.0)
            
            output_path = os.path.join(temp_dir, "cell_with_overlay.mp4")
            
            # Construir comando FFmpeg
            use_gpu = self.check_gpu_available()
            
            cmd = ["ffmpeg", "-y"]
            
            if use_gpu:
                cmd.extend(["-hwaccel", "auto"])
            
            # Inputs
            cmd.extend(["-i", pendulum_video])
            cmd.extend(["-stream_loop", "-1", "-i", overlay_path])
            
            # Filtros
            filters = []
            filters.append(f"[0:v]scale={width}:{height},setpts=PTS-STARTPTS[base]")
            filters.append(f"[1:v]scale={width}:{height}[overlay_sc]")
            filters.append(f"[overlay_sc]chromakey=color=0x{chroma_color}:similarity={similarity}:blend={blend}[overlay_key]")
            filters.append("[base][overlay_key]overlay=(W-w)/2:(H-h)/2[vout]")
            
            cmd.extend(["-filter_complex", ";".join(filters)])
            cmd.extend(["-map", "[vout]"])
            
            # Encoder
            if use_gpu:
                cmd.extend([
                    "-c:v", "h264_nvenc",
                    "-preset", "p4",
                    "-b:v", "8M"
                ])
            else:
                cmd.extend(["-c:v", "libx264", "-preset", "fast", "-crf", "23"])
            
            cmd.extend([
                "-t", str(cell_duration),
                "-pix_fmt", "yuv420p",
                "-an",
                output_path
            ])
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
            
            if result.returncode != 0:
                self.log(f"Erro ao aplicar overlay: {result.stderr[-300:] if result.stderr else ''}", "ERROR")
                return pendulum_video  # Fallback: retornar vídeo sem overlay
            
            self.log("Overlay aplicado com sucesso!", "OK")
            return output_path
            
        except Exception as e:
            self.log(f"Erro ao aplicar overlay: {str(e)}", "ERROR")
            return pendulum_video

    def loop_cell_to_duration(self, cell_video: str, audio_duration: float, config: dict, temp_dir: str) -> Optional[str]:
        """
        Faz loop da célula base até cobrir a duração do áudio.
        
        Args:
            cell_video: Caminho do vídeo da célula
            audio_duration: Duração do áudio em segundos
            config: Configuração do vídeo
            temp_dir: Diretório temporário
            
        Returns:
            Caminho do vídeo loopado ou None em caso de erro
        """
        try:
            cell_duration = self.get_video_duration(cell_video)
            if cell_duration is None or cell_duration <= 0:
                self.log("Erro ao obter duração da célula!", "ERROR")
                return None
            
            loops_needed = math.ceil(audio_duration / cell_duration)
            self.log(f"Loop: {loops_needed}x célula de {cell_duration:.1f}s para cobrir {audio_duration:.1f}s", "INFO")
            
            if loops_needed <= 1:
                # Não precisa fazer loop, apenas cortar no tamanho do áudio
                return cell_video
            
            # Criar lista de concat para loop
            loop_list = os.path.join(temp_dir, "loop_list.txt")
            with open(loop_list, "w", encoding="utf-8") as f:
                for _ in range(loops_needed):
                    safe_path = cell_video.replace("\\", "/").replace("'", "'\\''")
                    f.write(f"file '{safe_path}'\n")
            
            looped_video = os.path.join(temp_dir, "looped_video.mp4")
            
            # FFmpeg concat com corte no tamanho do áudio
            use_gpu = self.check_gpu_available()
            if use_gpu:
                encoder_args = ["-c:v", "h264_nvenc", "-preset", "p4", "-b:v", "8M"]
            else:
                encoder_args = ["-c:v", "libx264", "-preset", "fast", "-crf", "23"]
            
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", loop_list,
                *encoder_args,
                "-pix_fmt", "yuv420p",
                "-t", str(audio_duration),
                looped_video
            ]
            
            self.log(f"Fazendo loop do vídeo ({loops_needed}x)...", "INFO")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
            
            if result.returncode != 0:
                self.log(f"Erro no loop: {result.stderr[-300:] if result.stderr else ''}", "ERROR")
                return None
            
            self.log("Loop concluído!", "OK")
            return looped_video
            
        except Exception as e:
            self.log(f"Erro ao fazer loop: {str(e)}", "ERROR")
            return None

    def render_single_image_loop(self, config: dict, temp_dir: str) -> Optional[str]:
        """
        Pipeline completo do Modo 1 Imagem com loop seamless.
        
        1. Seleciona 1 imagem aleatória do banco de imagens
        2. Gera célula base com efeito pêndulo
        3. Aplica overlay com chroma key
        4. Faz loop até cobrir a duração do áudio
        
        Args:
            config: Configuração do vídeo
            temp_dir: Diretório temporário
            
        Returns:
            Caminho do vídeo final ou None em caso de erro
        """
        try:
            self.log("+================================================+", "INFO")
            self.log("|  MODO 1 IMAGEM - PÊNDULO COM LOOP SEAMLESS    |", "ENGINE")
            self.log("+================================================+", "INFO")
            
            # Obter duração do áudio
            audio_path = config.get("audio_path", "")
            audio_duration = self.get_audio_duration(audio_path)
            if audio_duration is None:
                self.log("Erro ao obter duração do áudio!", "ERROR")
                return None
            
            self.log(f"Duração do áudio: {audio_duration:.1f}s", "INFO")
            
            # ETAPA 1: Selecionar imagem aleatória do banco
            self.log("--- ETAPA 1: Selecionando imagem aleatória ---", "INFO")
            
            images_folder = config.get("batch_images_folder", "")
            if not images_folder:
                images_folder = config.get("images_backlog_folder", "")
            
            if not images_folder or not os.path.exists(images_folder):
                self.log(f"Pasta de imagens não encontrada: {images_folder}", "ERROR")
                return None
            
            image_files = self.get_image_files(images_folder)
            if not image_files:
                self.log("Nenhuma imagem encontrada no banco!", "ERROR")
                return None
            
            import random
            selected_image = random.choice(image_files)
            self.log(f"Imagem selecionada: {os.path.basename(selected_image)}", "OK")
            
            # ETAPA 2: Gerar célula pêndulo
            self.log("--- ETAPA 2: Gerando célula pêndulo ---", "INFO")
            
            pendulum_cell = self.generate_pendulum_cell(selected_image, config, temp_dir)
            if pendulum_cell is None:
                self.log("Falha ao gerar célula pêndulo!", "ERROR")
                return None
            
            # ETAPA 3: Aplicar overlay
            self.log("--- ETAPA 3: Aplicando overlay ---", "INFO")
            
            cell_with_overlay = self.apply_overlay_to_cell(pendulum_cell, config, temp_dir)
            if cell_with_overlay is None:
                self.log("Falha ao aplicar overlay!", "ERROR")
                return None
            
            # ETAPA 4: Fazer loop até cobrir duração do áudio
            self.log("--- ETAPA 4: Loop seamless ---", "INFO")
            
            looped_video = self.loop_cell_to_duration(cell_with_overlay, audio_duration, config, temp_dir)
            if looped_video is None:
                self.log("Falha ao fazer loop!", "ERROR")
                return None
            
            self.log("Modo 1 Imagem concluído com sucesso!", "OK")
            return looped_video
            
        except Exception as e:
            self.log(f"Erro no pipeline Modo 1 Imagem: {str(e)}", "ERROR")
            return None

    # =========================================================================
    # MODO LOOP - IMAGENS FIXAS
    # =========================================================================
    def render_with_loop(self, images, config, temp_dir, fixed_count, audio_duration):
        """
        Renderiza com numero fixo de imagens e faz loop ate cobrir o audio.

        Muito mais rapido para videos longos porque:
        1. Gera apenas X imagens (ex: 40)
        2. Faz loop do video ate cobrir a duracao do audio
        """
        # Pegar apenas as primeiras X imagens
        selected_images = images[:min(fixed_count, len(images))]
        self.log(f"Modo Loop: usando {len(selected_images)} imagens", "INFO")

        # Calcular duracao do clipe base
        image_duration = config["image_duration"]
        transition_duration = config["transition_duration"]
        base_duration = len(selected_images) * image_duration - (len(selected_images) - 1) * transition_duration

        self.log(f"Duracao do clipe base: {base_duration:.1f}s", "INFO")

        # Gerar clipes com o pipeline existente
        chunk_paths = self.render_all_clips_async(selected_images, config, temp_dir)
        if chunk_paths is None or len(chunk_paths) == 0:
            self.log("Falha na geracao dos clipes!", "ERROR")
            return None

        # Concatenar chunks para formar o clipe base
        base_clip = os.path.join(temp_dir, "base_clip.mp4")
        result = self.concat_chunks(chunk_paths, base_clip, temp_dir)
        if result is None:
            self.log("Falha na concatenacao do clipe base!", "ERROR")
            return None

        # Obter duracao real do clipe base
        real_base_duration = self.get_video_duration(base_clip)
        if real_base_duration is None or real_base_duration <= 0:
            self.log("Erro ao obter duracao do clipe base!", "ERROR")
            return None

        self.log(f"Duracao real do clipe base: {real_base_duration:.1f}s", "INFO")

        # Calcular quantas vezes precisa repetir
        loops_needed = math.ceil(audio_duration / real_base_duration)
        self.log(f"Loops necessarios: {loops_needed}", "INFO")

        if loops_needed <= 1:
            # Nao precisa fazer loop, apenas cortar no tamanho do audio
            return base_clip

        # Criar lista de concat para loop
        loop_list = os.path.join(temp_dir, "loop_list.txt")
        with open(loop_list, "w", encoding="utf-8") as f:
            for _ in range(loops_needed):
                safe_path = base_clip.replace("\\", "/").replace("'", "'\\''")
                f.write(f"file '{safe_path}'\n")

        looped_clip = os.path.join(temp_dir, "looped.mp4")

        # FFmpeg concat com corte no tamanho do audio
        use_gpu = self.check_gpu_available()
        if use_gpu:
            encoder_args = ["-c:v", "h264_nvenc", "-preset", "p4", "-b:v", "8M"]
        else:
            encoder_args = ["-c:v", "libx264", "-preset", "fast", "-crf", "23"]

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", loop_list,
            *encoder_args,
            "-pix_fmt", "yuv420p",
            "-t", str(audio_duration),  # Cortar no tamanho do audio
            looped_clip
        ]

        self.log(f"Fazendo loop do video ({loops_needed}x)...", "INFO")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )

        if result.returncode != 0:
            self.log(f"Erro no loop: {result.stderr[-300:] if result.stderr else ''}", "ERROR")
            return None

        self.log("Loop concluido!", "OK")
        return looped_clip

    # =========================================================================
    # PROCESSAMENTO SRT E GERAÇÃO DE IMAGENS
    # =========================================================================
    def process_srt_to_image_blocks(self, config, temp_dir) -> List[ImageBlock]:
        """
        Processa SRT e gera blocos de imagem com animações.

        Args:
            config: Configuração do vídeo
            temp_dir: Diretório temporário

        Returns:
            Lista de ImageBlock
        """
        image_blocks = []
        
        # Resetar motion shuffler para este vídeo e configurar efeitos habilitados
        self.motion_shuffler.reset()
        enabled_effects = config.get("enabled_effects", None)
        if enabled_effects:
            self.motion_shuffler.set_enabled_effects(enabled_effects)
            self.log(f"Efeitos habilitados: {len(enabled_effects)} ({', '.join(enabled_effects)})", "DEBUG")
        
        # Carregar prompt externo se configurado
        prompt_file = config.get("prompt_file", "")
        if prompt_file and os.path.exists(prompt_file):
            self.prompt_manager.load_prompt_file(prompt_file)
            self.log(f"Prompt externo carregado: {prompt_file}", "OK")
        
        # Configurar overlay se habilitado
        overlay_folder = config.get("overlay_folder", "")
        if config.get("use_random_overlays", False) and overlay_folder:
            self.overlay_manager.set_folder(overlay_folder)
        
        # 1. Obter/gerar SRT
        # Usar srt_source se disponível, senão usar subtitle_method, senão "assemblyai"
        srt_source = config.get("srt_source")
        if not srt_source:
            srt_source = config.get("subtitle_method", "assemblyai")
        # Mapear "srt" para "file"
        if srt_source == "srt":
            srt_source = "file"
        
        srt_blocks = None

        if srt_source == "file":
            srt_path = config.get("srt_path", "")
            if srt_path and os.path.exists(srt_path):
                srt_blocks = self.srt_generator.parse_srt_file(srt_path)
            else:
                self.log("Arquivo SRT não encontrado!", "ERROR")
                return []

        elif srt_source == "assemblyai":
            audio_path = config.get("audio_path", "")
            api_key = config.get("assemblyai_key", "")
            if audio_path and api_key:
                try:
                    srt_blocks = self.srt_generator.generate_srt_from_assemblyai(audio_path, api_key)
                except Exception as e:
                    self.log(f"Erro ao gerar SRT via AssemblyAI: {str(e)}", "ERROR")
                    return []
            else:
                self.log("Audio path ou AssemblyAI key não configurados", "ERROR")
                return []

        elif srt_source == "darkvi":
            # TODO: Implementar quando DarkVie tiver endpoint SRT
            self.log("DarkVie SRT ainda não implementado. Use AssemblyAI ou arquivo.", "WARN")
            return []

        if not srt_blocks or len(srt_blocks) == 0:
            self.log("Nenhum bloco SRT encontrado", "ERROR")
            return []

        # 2. Agrupar blocos (N por imagem, configurável via swap_every_n_cues)
        group_size = config.get("swap_every_n_cues", 3)
        groups = self.srt_generator.group_blocks(srt_blocks, group_size=group_size)
        self.log(f"Agrupados {len(srt_blocks)} blocos em {len(groups)} grupos (a cada {group_size} cues)", "INFO")

        # 3. Processar cada grupo
        image_source = config.get("image_source", "generate")
        use_varied_animations = config.get("use_varied_animations", True)
        
        # GERAÇÃO PARALELA: Se source é "generate", usar ParallelWhiskGenerator
        if image_source == "generate":
            image_paths = self._generate_images_parallel(groups, temp_dir, config)
        else:
            image_paths = None

        for i, group in enumerate(groups):
            if self.cancelled:
                return []

            # Calcular duração do grupo
            start_time, end_time = self.srt_generator.get_group_duration(group)
            duration = end_time - start_time

            # Selecionar animação (sem repetição consecutiva via motion_shuffler)
            if use_varied_animations:
                animation_type = self.motion_shuffler.get_next_effect()
                letter = self.motion_shuffler.get_effect_letter(animation_type)
                self.log(f"Grupo {i+1}: Efeito {letter} ({animation_type})", "DEBUG")
            else:
                # Usar zoom_in como padrão
                animation_type = "zoom_in"

            # Obter imagem (usar a gerada em paralelo ou selecionar do backlog)
            image_path = None
            if image_source == "generate" and image_paths:
                # Usar imagem já gerada em paralelo
                image_path = image_paths.get(i)
            elif image_source == "backlog":
                image_path = self.select_image_from_backlog(config)

            if not image_path or not os.path.exists(image_path):
                self.log(f"Erro ao obter imagem para grupo {i+1}", "ERROR")
                continue

            # Criar ImageBlock
            image_block = ImageBlock(
                image_path=image_path,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                animation_type=animation_type
            )
            image_blocks.append(image_block)

        return image_blocks
    
    def _generate_images_parallel(self, groups: List[List[SRTBlock]], temp_dir: str, 
                                  config: dict) -> dict:
        """
        Gera imagens para todos os grupos em paralelo usando múltiplos tokens.
        
        Tokens são carregados automaticamente do arquivo whisk_keys.json.
        
        Args:
            groups: Lista de grupos de blocos SRT
            temp_dir: Diretório temporário
            config: Configuração do vídeo
            
        Returns:
            Dicionário {índice: caminho_da_imagem}
        """
        # Carregar tokens do arquivo whisk_keys.json
        tokens = get_enabled_tokens()
        if not tokens:
            self.log("Nenhum token Whisk configurado em whisk_keys.json", "ERROR")
            self.log("Edite o arquivo whisk_keys.json para adicionar tokens", "WARN")
            return {}
        
        self.log(f"Carregados {len(tokens)} tokens do arquivo whisk_keys.json", "OK")
        
        # Carregar configurações do arquivo
        from whisk_pool_manager import get_whisk_settings
        settings = get_whisk_settings()
        
        # Configurações de paralelismo (prioridade: config da UI > arquivo > padrão)
        max_workers = config.get("whisk_parallel_workers", settings.get("parallel_workers", 0))
        if max_workers <= 0:
            max_workers = len(tokens)  # Usar número de tokens como padrão
        
        cooldown = settings.get("cooldown_seconds", 60)
        delay = settings.get("request_delay", 1.0)
        
        self.log(f"Iniciando geração paralela de {len(groups)} imagens com {min(max_workers, len(tokens))} workers", "INFO")
        
        # Criar gerador paralelo
        parallel_generator = ParallelWhiskGenerator(
            tokens=tokens,
            max_workers=max_workers,
            cooldown_seconds=cooldown,
            request_delay=delay,
            log_callback=self.log
        )
        
        # Preparar prompts e output paths
        prompts = []
        output_paths = []
        
        # Obter ID do prompt selecionado
        prompt_id = config.get("selected_prompt_id", get_default_prompt_id())
        prompt_data = get_prompt_by_id(prompt_id)
        prompt_name = prompt_data.get("name", "Padrão") if prompt_data else "Padrão"
        self.log(f"Usando estilo de prompt: {prompt_name}", "INFO")
        
        for i, group in enumerate(groups):
            combined_text = self.srt_generator.get_group_text(group)
            # Usar prompt do arquivo image_prompts.json
            prompt = build_prompt_from_template(prompt_id, combined_text)
            output_path = os.path.join(temp_dir, f"generated_image_{i:04d}.png")
            
            prompts.append(prompt)
            output_paths.append(output_path)
        
        # Callback de progresso
        def progress_callback(current, total, message):
            self.log(f"Geração paralela: {current}/{total} - {message}", "INFO")
        
        # Gerar em paralelo
        result = parallel_generator.generate_images_parallel(
            prompts=prompts,
            output_paths=output_paths,
            progress_callback=progress_callback
        )
        
        # Construir dicionário de resultados
        image_paths = {}
        for task in result.tasks:
            if task.success and os.path.exists(task.output_path):
                image_paths[task.index] = task.output_path
        
        self.log(f"Geração paralela concluída: {result.successful}/{result.total} imagens ({result.success_rate:.1f}%)", 
                "OK" if result.successful > 0 else "WARN")
        
        # Log estatísticas do pool
        stats = parallel_generator.get_pool_stats()
        self.log(f"Pool stats: {stats['available']}/{stats['total_tokens']} tokens disponíveis, "
                f"{stats['rate_limited_count']} rate limits", "DEBUG")
        
        return image_paths

    def generate_image_from_srt_group(self, group: List[SRTBlock], temp_dir: str, 
                                      index: int, config: dict) -> Optional[str]:
        """
        Gera imagem a partir de um grupo de blocos SRT.

        Args:
            group: Lista de blocos SRT
            temp_dir: Diretório temporário
            index: Índice do grupo
            config: Configuração do vídeo

        Returns:
            Caminho da imagem gerada ou None
        """
        try:
            # Combinar texto dos blocos
            combined_text = self.srt_generator.get_group_text(group)

            # Criar prompt espiritual
            whisk_generator = WhiskImageGenerator(
                tokens=config.get("whisk_api_tokens", []),
                log_callback=self.log
            )
            prompt = whisk_generator.create_spiritual_prompt(combined_text)

            # Gerar imagem
            output_path = os.path.join(temp_dir, f"generated_image_{index:04d}.png")
            success = whisk_generator.generate_and_save_image(prompt, output_path)

            if success:
                return output_path
            else:
                return None

        except Exception as e:
            self.log(f"Erro ao gerar imagem: {str(e)}", "ERROR")
            return None

    def select_image_from_backlog(self, config: dict) -> Optional[str]:
        """
        Seleciona imagem do backlog usando ImageReservationSystem.

        Args:
            config: Configuração do vídeo

        Returns:
            Caminho da imagem selecionada ou None
        """
        try:
            images_folder = config.get("images_backlog_folder", "")
            if not images_folder:
                self.log("Pasta de imagens backlog não configurada", "ERROR")
                return None

            # Usar ImageReservationSystem existente
            # Nota: Isso requer que o sistema já tenha sido inicializado
            # Por enquanto, vamos fazer uma seleção simples
            import random
            image_files = self.get_image_files(images_folder)
            
            if not image_files:
                self.log("Nenhuma imagem encontrada no backlog", "ERROR")
                return None

            selected = random.choice(image_files)
            return selected

        except Exception as e:
            self.log(f"Erro ao selecionar imagem do backlog: {str(e)}", "ERROR")
            return None

    def render_image_with_animation(self, image_block: ImageBlock, output_path: str, 
                                   config: dict) -> Optional[str]:
        """
        Renderiza imagem com animação Ken Burns.

        Args:
            image_block: Bloco de imagem com animação
            output_path: Caminho do vídeo de saída
            config: Configuração do vídeo

        Returns:
            Caminho do vídeo gerado ou None
        """
        try:
            width, height = RESOLUTIONS[config["resolution"]]
            fps = config.get("fps", 24)
            zoom_scale = config.get("zoom_scale", 1.15)
            pan_amount = config.get("pan_amount", 0.2)

            result = self.ken_burns_engine.render_animation(
                image_path=image_block.image_path,
                output_path=output_path,
                animation_type=image_block.animation_type,
                width=width,
                height=height,
                duration=image_block.duration,
                fps=fps,
                zoom_scale=zoom_scale,
                pan_amount=pan_amount
            )

            return result

        except Exception as e:
            self.log(f"Erro ao renderizar animação: {str(e)}", "ERROR")
            return None

    def render_srt_based_video(self, image_blocks: List[ImageBlock], config: dict, 
                              temp_dir: str) -> Optional[str]:
        """
        Renderiza vídeo baseado em blocos de imagem SRT com hard cut.

        Args:
            image_blocks: Lista de blocos de imagem
            config: Configuração do vídeo
            temp_dir: Diretório temporário

        Returns:
            Caminho do vídeo renderizado ou None
        """
        try:
            # Renderizar cada bloco
            clip_paths = []
            for i, block in enumerate(image_blocks):
                if self.cancelled:
                    return None

                clip_path = os.path.join(temp_dir, f"srt_clip_{i:04d}.mp4")
                result = self.render_image_with_animation(block, clip_path, config)

                if result:
                    clip_paths.append(result)
                else:
                    self.log(f"Erro ao renderizar clip {i+1}", "ERROR")

            if not clip_paths:
                self.log("Nenhum clip renderizado", "ERROR")
                return None

            # Concatenar com hard cut (sem transição)
            output_path = os.path.join(temp_dir, "srt_video.mp4")
            result = self.concat_clips_hard_cut(clip_paths, output_path)

            return result

        except Exception as e:
            self.log(f"Erro ao renderizar vídeo SRT: {str(e)}", "ERROR")
            return None

    def concat_clips_hard_cut(self, clip_paths: List[str], output_path: str) -> Optional[str]:
        """
        Concatena clips com hard cut (sem transição).

        Args:
            clip_paths: Lista de caminhos dos clips
            output_path: Caminho do vídeo de saída

        Returns:
            Caminho do vídeo concatenado ou None
        """
        try:
            # Criar arquivo de lista para FFmpeg concat demuxer
            list_file = output_path.replace(".mp4", "_list.txt")
            with open(list_file, 'w', encoding='utf-8') as f:
                for clip_path in clip_paths:
                    # Escapar caminhos para Windows
                    escaped_path = clip_path.replace('\\', '/')
                    f.write(f"file '{escaped_path}'\n")

            # Usar FFmpeg concat demuxer (hard cut)
            use_gpu = self.check_gpu_available()
            if use_gpu:
                encoder_args = ["-c:v", "h264_nvenc", "-preset", "p4", "-b:v", "8M"]
            else:
                encoder_args = ["-c:v", "libx264", "-preset", "fast", "-crf", "23"]

            cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", list_file,
                *encoder_args,
                "-c:a", "copy",
                "-pix_fmt", "yuv420p",
                output_path
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )

            # Limpar arquivo de lista
            try:
                os.remove(list_file)
            except:
                pass

            if result.returncode != 0:
                self.log(f"Erro FFmpeg ao concatenar: {result.stderr[-300:] if result.stderr else ''}", "ERROR")
                return None

            self.log("Clips concatenados com hard cut!", "OK")
            return output_path

        except Exception as e:
            self.log(f"Erro ao concatenar clips: {str(e)}", "ERROR")
            return None

    def _export_events_to_srt(self, events: list, output_path: str):
        """
        Exporta eventos de legenda para arquivo SRT.

        Args:
            events: Lista de eventos de legenda (dicts com start_ms, end_ms, text)
            output_path: Caminho do arquivo SRT de saída
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, event in enumerate(events, 1):
                    # Converter ms para formato SRT (HH:MM:SS,mmm)
                    start_ms = event.get('start_ms', 0)
                    end_ms = event.get('end_ms', 0)
                    text = event.get('text', '')
                    
                    start_time = self._ms_to_srt_time(start_ms)
                    end_time = self._ms_to_srt_time(end_ms)
                    
                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{text}\n")
                    f.write("\n")
            
            self.log(f"SRT exportado: {len(events)} legendas", "OK")
        except Exception as e:
            self.log(f"Erro ao exportar SRT: {e}", "ERROR")

    def _ms_to_srt_time(self, ms: int) -> str:
        """Converte milissegundos para formato SRT (HH:MM:SS,mmm)."""
        hours = ms // 3600000
        ms %= 3600000
        minutes = ms // 60000
        ms %= 60000
        seconds = ms // 1000
        milliseconds = ms % 1000
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def _srt_time_to_ms(self, time_str: str) -> int:
        """Converte tempo SRT (HH:MM:SS,mmm) para milissegundos."""
        try:
            # Formato: HH:MM:SS,mmm ou HH:MM:SS.mmm
            time_str = time_str.replace(',', '.')
            parts = time_str.split(':')
            hours = int(parts[0])
            minutes = int(parts[1])
            sec_parts = parts[2].split('.')
            seconds = int(sec_parts[0])
            milliseconds = int(sec_parts[1]) if len(sec_parts) > 1 else 0
            return (hours * 3600 + minutes * 60 + seconds) * 1000 + milliseconds
        except Exception:
            return 0

    def _find_keyword_in_srt(self, srt_path: str, keywords: dict, language: str) -> tuple:
        """
        Busca palavras-chave em um arquivo SRT.
        
        Args:
            srt_path: Caminho do arquivo SRT
            keywords: Dicionário de palavras-chave por idioma
            language: Idioma para buscar
            
        Returns:
            Tupla (timestamp_ms, keyword_found) ou (None, None)
        """
        try:
            # Obter palavras-chave do idioma
            lang_keywords = keywords.get(language, [])
            if not lang_keywords:
                self.log(f"Nenhuma palavra-chave para idioma '{language}'", "WARN")
                return None, None
            
            # Ler arquivo SRT
            encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']
            content = None
            
            for enc in encodings:
                try:
                    with open(srt_path, 'r', encoding=enc) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                self.log(f"Não foi possível ler arquivo SRT: {srt_path}", "ERROR")
                return None, None
            
            # Parsear SRT
            import re
            # Padrão para blocos SRT: número, timestamp, texto
            pattern = r'(\d+)\s*\n(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*\n(.*?)(?=\n\n|\n\d+\s*\n|$)'
            matches = re.findall(pattern, content, re.DOTALL)
            
            # Buscar palavras-chave
            for index, start_time, end_time, text in matches:
                text_lower = text.lower().strip()
                for keyword in lang_keywords:
                    if keyword.lower() in text_lower:
                        timestamp_ms = self._srt_time_to_ms(start_time)
                        self.log(f"Palavra-chave '{keyword}' encontrada no SRT em {start_time}", "INFO")
                        return timestamp_ms, keyword
            
            return None, None
            
        except Exception as e:
            self.log(f"Erro ao buscar palavras-chave no SRT: {e}", "ERROR")
            return None, None

    # =========================================================================
    # PIPELINE COMPLETO
    # =========================================================================
    def render_full_video(self, config):
        """Pipeline completo de renderizacao."""
        try:
            # Criar pasta temporaria
            temp_dir = tempfile.mkdtemp(prefix="final_slideshow_")
            self.log(f"Temp: {temp_dir}", "INFO")

            # Obter duracao do audio
            audio_duration = self.get_audio_duration(config["audio_path"])
            if audio_duration is None:
                return None

            minutes = int(audio_duration // 60)
            seconds = int(audio_duration % 60)
            self.log(f"Audio: {minutes}m {seconds}s ({audio_duration:.1f}s)", "INFO")

            # Verificar modo de vídeo
            video_mode = config.get("video_mode", "traditional")
            use_srt_based = config.get("use_srt_based_images", False) or video_mode == "srt"
            use_single_image = video_mode == "single_image"
            
            # ETAPA 0: Backlog Videos (se ativo) - Funciona em todos os modos (exceto single_image)
            backlog_intro = None
            if config.get("use_backlog_videos", False):
                self.log("+================================================+", "INFO")
                self.log("|  ETAPA 0: BACKLOG VIDEOS (INTRO)              |", "ENGINE")
                self.log("+================================================+", "INFO")
                
                backlog_folder = config.get("backlog_folder", str(SCRIPT_DIR / "EFEITOS" / "BACKLOG_VIDEOS"))
                if backlog_folder:
                    if not os.path.isabs(backlog_folder):
                        backlog_folder = str(SCRIPT_DIR / backlog_folder)
                else:
                    backlog_folder = str(SCRIPT_DIR / "EFEITOS" / "BACKLOG_VIDEOS")
                
                backlog_intro_path = os.path.join(temp_dir, "backlog_intro.mp4")
                backlog_engine = BacklogVideoEngine(self.log_queue)
                
                use_gpu = self.check_gpu_available()
                backlog_intro = backlog_engine.create_backlog_intro(
                    backlog_folder=backlog_folder,
                    output_path=backlog_intro_path,
                    resolution=config["resolution"],
                    target_duration=60.0,  # Sempre 1 minuto (60 segundos)
                    audio_volume=config.get("backlog_audio_volume", 0.25),
                    transition_duration=config.get("backlog_transition_duration", 0.5),
                    fade_out_duration=config.get("backlog_fade_out_duration", 1.0),
                    use_gpu=use_gpu
                )
                
                if backlog_intro is None:
                    self.log("Falha ao criar intro do backlog. Continuando sem intro...", "WARN")
                else:
                    self.log(f"Intro do backlog criada: {backlog_intro}", "OK")
            
            if use_single_image:
                # MODO 1 IMAGEM: Pêndulo com loop seamless
                video_stitched = self.render_single_image_loop(config, temp_dir)
                if video_stitched is None:
                    self.log("Falha no Modo 1 Imagem!", "ERROR")
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    return None

                current_video = video_stitched
                
            elif use_srt_based:
                # MODO SRT: Processar SRT e gerar/selecionar imagens
                self.log("+================================================+", "INFO")
                self.log("|  MODO: IMAGENS BASEADAS EM SRT                |", "ENGINE")
                self.log("+================================================+", "INFO")
                
                image_blocks = self.process_srt_to_image_blocks(config, temp_dir)
                if not image_blocks:
                    self.log("Erro ao processar SRT para imagens", "ERROR")
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    return None

                # Renderizar vídeo baseado em SRT
                video_stitched = self.render_srt_based_video(image_blocks, config, temp_dir)
                if video_stitched is None:
                    self.log("Falha na renderização baseada em SRT!", "ERROR")
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    return None

                current_video = video_stitched
                
            else:
                # LÓGICA ANTIGA: Seleção baseada em duração fixa
                # Listar imagens (suporta lista pre-selecionada ou pasta)
                if "images_list" in config and config["images_list"]:
                    all_images = config["images_list"]
                    self.log(f"Imagens pre-selecionadas: {len(all_images)}", "INFO")
                else:
                    all_images = self.get_image_files(config["images_folder"])
                    self.log(f"Imagens disponiveis: {len(all_images)}", "INFO")

                if len(all_images) == 0:
                    self.log("ERRO: Nenhuma imagem encontrada!", "ERROR")
                    return None

                # Calcular quantas imagens precisamos
                num_needed = self.calculate_images_needed(
                    audio_duration,
                    config["image_duration"],
                    config["transition_duration"]
                )
                self.log(f"Imagens necessarias: {num_needed}", "INFO")

                # Selecionar imagens
                if len(all_images) >= num_needed:
                    import random
                    selected_images = random.sample(all_images, num_needed)
                else:
                    selected_images = (all_images * ((num_needed // len(all_images)) + 1))[:num_needed]

                # Verificar modo de imagens
                use_fixed = config.get("use_fixed_images", False)
                fixed_count = config.get("fixed_images_count", 40)

                self.log("-" * 50, "INFO")
                self.log(f"CONFIGURACAO:", "INFO")
                self.log(f"  Resolucao: {config['resolution']}", "INFO")
                self.log(f"  Duracao/img: {config['image_duration']}s | Transicao: {config['transition_duration']}s", "INFO")
                self.log(f"  Zoom: {config['zoom_mode']} -> {config['zoom_scale']}x (CENTRALIZADO)", "INFO")
                self.log(f"  Threads: {config['threads']}", "INFO")
                self.log(f"  Modo: {'LOOP (' + str(fixed_count) + ' imagens)' if use_fixed else 'COMPLETO'}", "INFO")
                self.log("-" * 50, "INFO")

                # ETAPA 1: Geracao de video (modo normal ou loop)
                if use_fixed:
                    # MODO LOOP: Gera clipe com X imagens e faz loop
                    self.log("+================================================+", "INFO")
                    self.log("|  ETAPA 1: MODO LOOP (IMAGENS FIXAS)            |", "ENGINE")
                    self.log("+================================================+", "INFO")

                    video_stitched = self.render_with_loop(
                        all_images, config, temp_dir, fixed_count, audio_duration
                    )
                    if video_stitched is None:
                        self.log("Falha no modo loop!", "ERROR")
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        return None
                else:
                    # MODO COMPLETO: Gera todas as imagens necessarias
                    self.log("+================================================+", "INFO")
                    self.log("|  ETAPA 1: GERACAO + STITCHING ASSINCRONO       |", "ENGINE")
                    self.log("+================================================+", "INFO")

                    chunk_paths = self.render_all_clips_async(selected_images, config, temp_dir)
                    if chunk_paths is None or len(chunk_paths) == 0:
                        self.log("Falha na geracao/stitching!", "ERROR")
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        return None

                    # ETAPA 2: Concatenar chunks
                    self.log("+================================================+", "INFO")
                    self.log("|  ETAPA 2: CONCATENACAO DE CHUNKS               |", "ENGINE")
                    self.log("+================================================+", "INFO")

                    video_stitched = os.path.join(temp_dir, "stitched.mp4")
                    result = self.concat_chunks(chunk_paths, video_stitched, temp_dir)
                    if result is None:
                        self.log("Falha na concatenacao!", "ERROR")
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        return None

                # Definir current_video para modo antigo
                current_video = video_stitched

            # ETAPA 2.5: Concatenar backlog intro + vídeo principal (se backlog existe)
            # Funciona em ambos os modos (tradicional e SRT-based)
            if backlog_intro and os.path.exists(backlog_intro):
                self.log("+================================================+", "INFO")
                self.log("|  ETAPA 2.5: CONCATENAR INTRO + VIDEO PRINCIPAL|", "ENGINE")
                self.log("+================================================+", "INFO")
                
                # Aplicar overlay no backlog intro primeiro (se overlay estiver configurado)
                backlog_with_overlay = backlog_intro
                if config.get("overlay_path") and os.path.exists(config["overlay_path"]):
                    self.log("Aplicando overlay no intro do backlog...", "INFO")
                    backlog_overlay_path = os.path.join(temp_dir, "backlog_intro_overlay.mp4")
                    result_overlay = self.apply_overlay(
                        backlog_intro,
                        config["overlay_path"],
                        backlog_overlay_path,
                        config.get("overlay_opacity", 0.3)
                    )
                    if result_overlay:
                        backlog_with_overlay = backlog_overlay_path
                        self.log("Overlay aplicado no intro do backlog!", "OK")
                
                # Concatenar intro com vídeo principal usando crossfade
                # Usar current_video que funciona em ambos os modos (SRT e tradicional)
                main_video = current_video if use_srt_based else video_stitched
                video_with_intro = os.path.join(temp_dir, "with_intro.mp4")
                use_gpu = self.check_gpu_available()
                
                # Criar arquivo de lista para concat
                concat_list_path = os.path.join(temp_dir, "intro_concat_list.txt")
                with open(concat_list_path, 'w', encoding='utf-8') as f:
                    # Escapar caminhos para Windows
                    safe_intro = backlog_with_overlay.replace('\\', '/').replace("'", "'\\''")
                    safe_video = main_video.replace('\\', '/').replace("'", "'\\''")
                    f.write(f"file '{safe_intro}'\n")
                    f.write(f"file '{safe_video}'\n")
                
                # Usar filter_complex para crossfade entre intro e vídeo principal
                intro_duration = self.get_video_duration(backlog_with_overlay)
                if intro_duration is None:
                    intro_duration = 48.0  # Fallback: 6 vídeos * 8s
                
                transition_dur = config.get("backlog_transition_duration", 0.5)
                xfade_offset = max(0, intro_duration - transition_dur)
                
                if use_gpu:
                    encoder_args = ["-c:v", "h264_nvenc", "-preset", "p4", "-b:v", "8M"]
                else:
                    encoder_args = ["-c:v", "libx264", "-preset", "fast", "-crf", "23"]
                
                # Tentar usar xfade para transição suave
                filter_complex = (
                    f"[0:v]setpts=PTS-STARTPTS[v0];"
                    f"[1:v]setpts=PTS-STARTPTS[v1];"
                    f"[v0][v1]xfade=transition=fade:duration={transition_dur}:offset={xfade_offset}[vout];"
                    f"[0:a][1:a]acrossfade=d={transition_dur}[aout]"
                )
                
                cmd = [
                    "ffmpeg", "-y",
                    "-i", backlog_with_overlay,
                    "-i", main_video,
                    "-filter_complex", filter_complex,
                    "-map", "[vout]",
                    "-map", "[aout]",
                    *encoder_args,
                    "-c:a", "aac", "-b:a", "192k",
                    "-pix_fmt", "yuv420p",
                    video_with_intro
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                )
                
                if result.returncode != 0:
                    # Fallback: concat simples sem crossfade
                    self.log("Erro no crossfade intro, tentando concatenação simples...", "WARN")
                    cmd_simple = [
                        "ffmpeg", "-y",
                        "-f", "concat", "-safe", "0",
                        "-i", concat_list_path,
                        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                        "-c:a", "aac", "-b:a", "192k",
                        "-pix_fmt", "yuv420p",
                        video_with_intro
                    ]
                    result = subprocess.run(
                        cmd_simple,
                        capture_output=True,
                        text=True,
                        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                    )
                    
                    if result.returncode != 0:
                        self.log(f"Erro ao concatenar intro: {result.stderr[-300:] if result.stderr else ''}", "WARN")
                        # Continuar sem intro se falhar - manter o vídeo original
                        pass
                    else:
                        current_video = video_with_intro
                        if not use_srt_based:
                            video_stitched = video_with_intro
                        self.log("Intro concatenada com sucesso!", "OK")
                else:
                    current_video = video_with_intro
                    if not use_srt_based:
                        video_stitched = video_with_intro
                    self.log("Intro concatenada com crossfade!", "OK")

            # ETAPA 3: Overlay (opcional) - Aplicar apenas se não foi aplicado no backlog
            # Garantir que current_video está definido corretamente
            if not use_srt_based:
                # No modo tradicional, current_video pode não ter sido atualizado se não houve backlog
                if not backlog_intro or not os.path.exists(backlog_intro):
                    current_video = video_stitched
            backlog_has_overlay = backlog_intro and os.path.exists(backlog_intro) and config.get("overlay_path") and os.path.exists(config["overlay_path"])
            
            if config.get("overlay_path") and os.path.exists(config["overlay_path"]) and not backlog_has_overlay:
                self.log("+================================================+", "INFO")
                self.log("|  ETAPA 3: OVERLAY                              |", "ENGINE")
                self.log("+================================================+", "INFO")

                video_overlay = os.path.join(temp_dir, "with_overlay.mp4")
                result = self.apply_overlay(
                    current_video,
                    config["overlay_path"],
                    video_overlay,
                    config.get("overlay_opacity", 0.3)
                )
                if result:
                    current_video = video_overlay
            elif backlog_has_overlay:
                self.log("Overlay já aplicado no intro do backlog, aplicando no vídeo principal também...", "INFO")
                video_overlay = os.path.join(temp_dir, "with_overlay.mp4")
                result = self.apply_overlay(
                    current_video,
                    config["overlay_path"],
                    video_overlay,
                    config.get("overlay_opacity", 0.3)
                )
                if result:
                    current_video = video_overlay

            # ETAPA 3.5: PREPARAÇÃO VSL - Detectar palavra-chave e preparar dados
            # A inserção real do VSL acontece DEPOIS das legendas (para ficar por cima)
            vsl_start_time = None
            vsl_end_time = None
            vsl_start_sec = None
            vsl_duration = None
            vsl_path = None
            pre_generated_events = None  # Para reutilizar eventos se já foram gerados
            
            # Debug: verificar configuração VSL
            use_vsl_config = config.get("use_vsl", False)
            vsl_insertion_mode = config.get("vsl_insertion_mode", "keyword")
            self.log(f"Config VSL: use_vsl={use_vsl_config}, modo={vsl_insertion_mode}", "INFO")
            
            if use_vsl_config:
                self.log("+================================================+", "INFO")
                self.log("|  ETAPA 3.5: PREPARAÇÃO VSL                     |", "ENGINE")
                self.log("+================================================+", "INFO")
                
                try:
                    # Usar VSL selecionada diretamente
                    vsl_folder = config.get("vsl_folder", str(SCRIPT_DIR / "EFEITOS" / "VSLs"))
                    if not os.path.isabs(vsl_folder):
                        vsl_folder = str(SCRIPT_DIR / vsl_folder)
                    
                    # Obter VSL selecionada na interface
                    selected_vsl = config.get("selected_vsl", "")
                    
                    if selected_vsl and selected_vsl != "Nenhuma VSL encontrada":
                        # Usar VSL selecionada
                        vsl_path = os.path.join(vsl_folder, selected_vsl)
                        self.log(f"Usando VSL selecionada: {selected_vsl}", "INFO")
                    else:
                        # Fallback: usar primeira VSL disponível
                        if os.path.exists(vsl_folder):
                            vsl_files = [f for f in os.listdir(vsl_folder) 
                                       if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm'))]
                            if vsl_files:
                                vsl_path = os.path.join(vsl_folder, vsl_files[0])
                                self.log(f"Nenhuma VSL selecionada, usando: {vsl_files[0]}", "INFO")
                            else:
                                vsl_path = None
                                self.log(f"Nenhuma VSL disponível na pasta: {vsl_folder}", "WARN")
                        else:
                            vsl_path = None
                            self.log(f"Pasta de VSLs não existe: {vsl_folder}", "WARN")
                    
                    if not vsl_path or not os.path.exists(vsl_path):
                        self.log(f"VSL não encontrada: {vsl_path}", "WARN")
                        vsl_path = None
                    else:
                        # Obter duração do VSL usando VSLEngine
                        vsl_engine_temp = VSLEngine(self.log_queue)
                        vsl_duration = vsl_engine_temp.get_video_duration(vsl_path)
                        
                        if not vsl_duration:
                            self.log("Não foi possível obter duração do VSL", "WARN")
                            vsl_path = None
                        else:
                            # MODO DE POSIÇÃO FIXA - Não requer legendas nem AssemblyAI
                            if vsl_insertion_mode == "fixed":
                                vsl_fixed_position = config.get("vsl_fixed_position", 60.0)
                                vsl_start_sec = float(vsl_fixed_position)
                                
                                # Verificar se a posição é válida (não ultrapassa duração do áudio)
                                if vsl_start_sec >= audio_duration:
                                    self.log(f"Posição fixa ({vsl_start_sec}s) maior que duração do áudio ({audio_duration:.1f}s). Ajustando para 50% do vídeo.", "WARN")
                                    vsl_start_sec = audio_duration * 0.5
                                
                                self.log(f"VSL em posição fixa: {vsl_start_sec:.2f}s", "OK")
                                self.log(f"VSL será inserido: início={vsl_start_sec:.2f}s, fim={vsl_start_sec + vsl_duration:.2f}s", "INFO")
                            
                            # MODO DE RANGE ALEATÓRIO - Posição aleatória dentro de um intervalo
                            elif vsl_insertion_mode == "range":
                                import random
                                
                                # Obter range em minutos e converter para segundos
                                range_start_min = float(config.get("vsl_range_start_min", 1.0))
                                range_end_min = float(config.get("vsl_range_end_min", 3.0))
                                
                                # Garantir que start < end
                                if range_start_min > range_end_min:
                                    range_start_min, range_end_min = range_end_min, range_start_min
                                
                                range_start_sec = range_start_min * 60.0
                                range_end_sec = range_end_min * 60.0
                                
                                self.log(f"Range VSL configurado: {range_start_min:.1f}min - {range_end_min:.1f}min ({range_start_sec:.0f}s - {range_end_sec:.0f}s)", "INFO")
                                
                                # Validar que o range está dentro da duração do vídeo
                                max_start_position = audio_duration - vsl_duration
                                
                                if max_start_position <= 0:
                                    self.log(f"VSL ({vsl_duration:.1f}s) é maior que o vídeo ({audio_duration:.1f}s). Pulando VSL.", "WARN")
                                    vsl_path = None
                                elif range_start_sec >= audio_duration:
                                    # Range inteiro fora do vídeo - usar fallback
                                    self.log(f"Range ({range_start_sec:.0f}s) começa após fim do vídeo ({audio_duration:.1f}s). Usando 50% do vídeo.", "WARN")
                                    vsl_start_sec = audio_duration * 0.5
                                    if vsl_start_sec + vsl_duration > audio_duration:
                                        vsl_start_sec = max(0, audio_duration - vsl_duration)
                                else:
                                    # Ajustar range_end para não ultrapassar a posição máxima válida
                                    effective_range_end = min(range_end_sec, max_start_position)
                                    effective_range_start = min(range_start_sec, effective_range_end)
                                    
                                    # Gerar posição aleatória dentro do range efetivo
                                    vsl_start_sec = random.uniform(effective_range_start, effective_range_end)
                                    
                                    self.log(f"Posição aleatória gerada: {vsl_start_sec:.2f}s (range efetivo: {effective_range_start:.0f}s - {effective_range_end:.0f}s)", "OK")
                                
                                if vsl_path and vsl_start_sec is not None:
                                    self.log(f"VSL será inserido: início={vsl_start_sec:.2f}s, fim={vsl_start_sec + vsl_duration:.2f}s", "INFO")
                            
                            # MODO DE PALAVRA-CHAVE - Requer legendas/transcrição
                            elif vsl_insertion_mode == "keyword":
                                # Carregar palavras-chave
                                keywords_file = config.get("vsl_keywords_file", str(SCRIPT_DIR / "vsl_keywords.json"))
                                if not os.path.isabs(keywords_file):
                                    keywords_file = str(SCRIPT_DIR / keywords_file)
                                self.log(f"Carregando palavras-chave de: {keywords_file}", "INFO")
                                keywords = load_vsl_keywords(keywords_file)
                                
                                if not keywords:
                                    self.log(f"Nenhuma palavra-chave VSL encontrada em {keywords_file}. Pulando inserção de VSL.", "WARN")
                                    vsl_path = None
                                else:
                                    # Se legendas estão ativas com AssemblyAI, usar transcrição
                                    if config.get("use_subtitles") and config.get("subtitle_method") == "assemblyai":
                                        api_key = config.get("assemblyai_key", "")
                                        if api_key:
                                            subtitle_engine = SubtitleEngine(self.log_queue)
                                            use_karaoke = config.get("sub_options", {}).get("use_karaoke", True)
                                            
                                            # Obter dados completos da transcrição
                                            events, transcript, words = subtitle_engine.generate_with_assemblyai(
                                                config["audio_path"],
                                                api_key,
                                                use_karaoke,
                                                return_full_data=True
                                            )
                                            
                                            # Armazenar eventos para reutilizar na etapa de legendas
                                            pre_generated_events = events
                                            
                                            # Buscar palavra-chave no idioma selecionado
                                            vsl_language = config.get("vsl_language", "portugues")
                                            self.log(f"Buscando palavras-chave no idioma: {vsl_language}", "INFO")
                                            keyword_timestamp_ms, keyword_found = find_keyword_timestamp(
                                                words, keywords, vsl_language, log_callback=self.log
                                            )
                                            
                                            if keyword_timestamp_ms is not None:
                                                vsl_start_sec = keyword_timestamp_ms / 1000.0
                                                vsl_end_sec = vsl_start_sec + vsl_duration
                                                
                                                # Calcular quando as legendas devem voltar
                                                fade_duration = min(0.5, vsl_duration / 4)
                                                vsl_end_for_subtitles_sec = vsl_start_sec + vsl_duration - 0.1
                                                
                                                # Converter para formato ASS
                                                vsl_start_time = subtitle_engine.ms_to_ass_time(int(keyword_timestamp_ms))
                                                vsl_end_time = subtitle_engine.ms_to_ass_time(int(vsl_end_for_subtitles_sec * 1000))
                                                
                                                self.log(f"Palavra-chave '{keyword_found}' encontrada em {vsl_start_sec:.2f}s", "OK")
                                                self.log(f"VSL será inserido: início={vsl_start_sec:.2f}s, fim={vsl_end_sec:.2f}s", "INFO")
                                                self.log(f"Legendas serão ocultadas de {vsl_start_sec:.2f}s até {vsl_end_for_subtitles_sec:.2f}s", "INFO")
                                            else:
                                                self.log(f"Nenhuma palavra-chave encontrada no texto para idioma '{vsl_language}'", "WARN")
                                                vsl_path = None
                                        else:
                                            self.log("Chave API AssemblyAI não configurada. Use modo 'Posição fixa' ou configure AssemblyAI.", "WARN")
                                            vsl_path = None
                                    
                                    # Se método é SRT, tentar buscar no arquivo SRT
                                    elif config.get("use_subtitles") and config.get("subtitle_method") == "srt":
                                        srt_path = config.get("srt_path", "")
                                        if srt_path and os.path.exists(srt_path):
                                            self.log(f"Buscando palavras-chave no arquivo SRT: {srt_path}", "INFO")
                                            # Ler SRT e buscar palavras-chave
                                            vsl_language = config.get("vsl_language", "portugues")
                                            keyword_timestamp_ms, keyword_found = self._find_keyword_in_srt(
                                                srt_path, keywords, vsl_language
                                            )
                                            
                                            if keyword_timestamp_ms is not None:
                                                vsl_start_sec = keyword_timestamp_ms / 1000.0
                                                self.log(f"Palavra-chave '{keyword_found}' encontrada em {vsl_start_sec:.2f}s", "OK")
                                                self.log(f"VSL será inserido: início={vsl_start_sec:.2f}s, fim={vsl_start_sec + vsl_duration:.2f}s", "INFO")
                                            else:
                                                self.log(f"Nenhuma palavra-chave encontrada no SRT para idioma '{vsl_language}'", "WARN")
                                                vsl_path = None
                                        else:
                                            self.log("Arquivo SRT não encontrado. Use modo 'Posição fixa' ou forneça um arquivo SRT.", "WARN")
                                            vsl_path = None
                                    else:
                                        # Sem legendas - sugerir modo fixo
                                        self.log("Modo palavra-chave requer legendas ativas. Use modo 'Posição fixa' ou ative legendas.", "WARN")
                                        vsl_path = None
                
                except Exception as e:
                    self.log(f"Erro ao processar VSL: {e}", "ERROR")
                    import traceback
                    self.log(traceback.format_exc(), "ERROR")
                    vsl_path = None

            # ETAPA 4: Legendas (se ativas)
            # Verificar modo de legenda: "full" = burn-in, "none" = exportar SRT separado
            subtitle_mode = config.get("subtitle_mode", "full")
            
            if config.get("use_subtitles"):
                self.log("+================================================+", "INFO")
                self.log("|  ETAPA 4: LEGENDAS                             |", "ENGINE")
                self.log(f"|  Modo: {subtitle_mode.upper()}                              |", "ENGINE")
                self.log("+================================================+", "INFO")

                use_gpu = self.check_gpu_available()
                subtitle_engine = SubtitleEngine(self.log_queue)

                try:
                    # Reutilizar eventos se já foram gerados na etapa VSL
                    if pre_generated_events is not None:
                        events = pre_generated_events
                        self.log("Reutilizando eventos de legenda gerados na etapa VSL", "INFO")
                    else:
                        # Obter eventos conforme metodo
                        if config["subtitle_method"] == "srt":
                            srt_path = config.get("srt_path", "")
                            if srt_path and os.path.exists(srt_path):
                                events = subtitle_engine.parse_srt(srt_path)
                            else:
                                self.log("Arquivo SRT nao encontrado!", "WARN")
                                events = []
                        else:
                            api_key = config.get("assemblyai_key", "")
                            if api_key:
                                use_karaoke = config.get("sub_options", {}).get("use_karaoke", True)
                                events = subtitle_engine.generate_with_assemblyai(
                                    config["audio_path"],
                                    api_key,
                                    use_karaoke
                                )
                            else:
                                self.log("Chave API AssemblyAI nao configurada!", "WARN")
                                events = []

                    if events:
                        if subtitle_mode == "full":
                            # MODO FULL: Queimar legendas no vídeo (burn-in)
                            ass_path = os.path.join(temp_dir, "subtitles.ass")
                            subtitle_engine.create_ass_file(
                                events,
                                config.get("sub_options", {}),
                                ass_path,
                                config["resolution"],
                                vsl_start_time=vsl_start_time,
                                vsl_end_time=vsl_end_time
                            )

                            video_with_subs = os.path.join(temp_dir, "with_subtitles.mp4")
                            subtitle_engine.burn_subtitles(current_video, ass_path, video_with_subs, use_gpu)
                            current_video = video_with_subs
                            self.log("Legendas queimadas no vídeo (burn-in)", "OK")
                        else:
                            # MODO NONE: Exportar SRT separadamente (soft subtitles)
                            # Determinar caminho de saída do SRT
                            if "output_path" in config and config["output_path"]:
                                srt_output = config["output_path"].replace(".mp4", ".srt")
                            else:
                                srt_output = os.path.join(temp_dir, "subtitles.srt")
                            
                            # Exportar eventos para SRT
                            self._export_events_to_srt(events, srt_output)
                            self.log(f"SRT exportado: {srt_output}", "OK")
                            self.log("Vídeo SEM legendas queimadas (soft subtitles)", "OK")
                    else:
                        self.log("Nenhum evento de legenda gerado", "WARN")

                except Exception as e:
                    self.log(f"Erro ao processar legendas: {e}", "ERROR")

            # VSL será inserida na ETAPA 6 (última etapa, após vídeo 100% finalizado)
            # Guardamos os dados aqui para usar depois
            vsl_data_for_final = {
                "use_vsl": use_vsl_config,
                "vsl_path": vsl_path,
                "vsl_start_sec": vsl_start_sec,
                "vsl_duration": vsl_duration
            } if use_vsl_config and vsl_path and vsl_start_sec is not None else None

            # ETAPA 4.5: Overlay (se configurado)
            if config.get("use_random_overlays", False) and config.get("overlay_folder"):
                self.log("+================================================+", "INFO")
                self.log("|  ETAPA 4.5: OVERLAY                            |", "ENGINE")
                self.log("+================================================+", "INFO")
                
                try:
                    self.overlay_manager.set_folder(config["overlay_folder"])
                    overlay_info = self.overlay_manager.get_next_overlay()
                    
                    if overlay_info:
                        video_with_overlay = os.path.join(temp_dir, "with_overlay.mp4")
                        result = self.overlay_manager.apply_overlay(
                            current_video, overlay_info, video_with_overlay,
                            opacity=config.get("overlay_opacity", 0.3)
                        )
                        
                        if result:
                            current_video = video_with_overlay
                            self.overlay_manager.mark_overlay_used(overlay_info["path"])
                            self.log(f"Overlay aplicado: {overlay_info['name']}", "OK")
                        else:
                            self.log("Falha ao aplicar overlay, continuando sem...", "WARN")
                    else:
                        self.log("Nenhum overlay disponível", "WARN")
                        
                except Exception as e:
                    self.log(f"Erro ao aplicar overlay: {e}", "ERROR")

            # ETAPA 5: Audio
            self.log("+================================================+", "INFO")
            self.log("|  ETAPA 5: FINALIZACAO (AUDIO)                  |", "ENGINE")
            self.log("+================================================+", "INFO")

            # Suporta output_path direto (modo lote) ou output_folder (modo individual)
            if "output_path" in config and config["output_path"]:
                output_path = config["output_path"]
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"Slideshow_FINAL_{timestamp}.mp4"
                output_path = os.path.join(config["output_folder"], output_filename)

            # Com ou sem musica de fundo
            music_path = config.get("music_path", "")
            music_volume = config.get("music_volume", 0.2)
            
            # Debug: mostrar configuração de música
            self.log(f"Música configurada: '{music_path}'", "DEBUG")
            self.log(f"Volume música: {music_volume:.0%}", "DEBUG")
            
            if music_path and os.path.exists(music_path):
                self.log(f"Mixando com música de fundo: {os.path.basename(music_path)}", "INFO")
                video_final = os.path.join(temp_dir, "with_audio.mp4")
                result = self.mix_audio(
                    current_video,
                    config["audio_path"],
                    music_path,
                    video_final,
                    music_volume,
                    config
                )
                if result:
                    shutil.copy(video_final, output_path)
                else:
                    # Fallback para audio simples
                    self.log("Falha na mixagem, usando apenas narração", "WARN")
                    self.finalize_simple(current_video, config["audio_path"], output_path, config)
            else:
                if music_path:
                    self.log(f"Arquivo de música não encontrado: {music_path}", "WARN")
                else:
                    self.log("Nenhuma música de fundo configurada", "INFO")
                self.finalize_simple(current_video, config["audio_path"], output_path, config)

            # ETAPA 6: INSERÇÃO VSL (ÚLTIMA ETAPA - após vídeo 100% finalizado)
            # A VSL é inserida no vídeo FINAL, cortando e concatenando
            # Isso garante que o áudio da VSL seja preservado corretamente
            if vsl_data_for_final and vsl_data_for_final.get("use_vsl"):
                self.log("+================================================+", "INFO")
                self.log("|  ETAPA 6: INSERÇÃO VSL (ÚLTIMA ETAPA)          |", "ENGINE")
                self.log("+================================================+", "INFO")
                
                vsl_path = vsl_data_for_final["vsl_path"]
                vsl_start_sec = vsl_data_for_final["vsl_start_sec"]
                vsl_duration = vsl_data_for_final["vsl_duration"]
                
                self.log(f"VSL: {os.path.basename(vsl_path)}", "INFO")
                self.log(f"Ponto de inserção: {vsl_start_sec:.2f}s", "INFO")
                self.log(f"Duração VSL: {vsl_duration:.2f}s", "INFO")
                
                try:
                    use_gpu = self.check_gpu_available()
                    vsl_engine = VSLEngine(self.log_queue)
                    
                    # Criar arquivo temporário para o vídeo com VSL
                    video_with_vsl = os.path.join(temp_dir, "final_with_vsl.mp4")
                    
                    # Inserir VSL no vídeo finalizado
                    result = vsl_engine.insert_vsl(
                        output_path,  # Vídeo já finalizado (com áudio, legendas, etc)
                        vsl_path,
                        vsl_start_sec,
                        video_with_vsl,
                        use_gpu
                    )
                    
                    if result and os.path.exists(video_with_vsl):
                        # Substituir o vídeo final pelo vídeo com VSL
                        shutil.copy(video_with_vsl, output_path)
                        self.log("VSL inserido com sucesso no vídeo final!", "OK")
                        self.log(f"Vídeo PAUSA em {vsl_start_sec:.2f}s -> VSL -> CONTINUA", "OK")
                    else:
                        self.log("Falha ao inserir VSL. Vídeo final mantido sem VSL.", "WARN")
                        
                except Exception as e:
                    self.log(f"Erro ao inserir VSL: {e}", "ERROR")
                    import traceback
                    self.log(traceback.format_exc(), "ERROR")
                    self.log("Vídeo final mantido sem VSL.", "WARN")

            # Limpar temp
            shutil.rmtree(temp_dir, ignore_errors=True)
            self.log("Temp removido", "INFO")

            return output_path

        except Exception as e:
            self.log(f"Erro fatal: {str(e)}", "ERROR")
            import traceback
            self.log(traceback.format_exc(), "ERROR")
            return None


# =============================================================================
# FUNCOES AUXILIARES VSL
# =============================================================================
def detect_language(audio_path=None, text_path=None, transcript=None, fallback="portugues"):
    """
    Detecta o idioma do conteúdo por múltiplos métodos.
    
    Args:
        audio_path: Caminho do arquivo de áudio
        text_path: Caminho do arquivo de texto
        transcript: Objeto transcript do AssemblyAI (com language_code)
        fallback: Idioma padrão se não detectar
    
    Returns:
        String com nome do idioma (ex: "portugues", "ingles")
    """
    # Método 1: Nome do arquivo de texto
    if text_path:
        filename_lower = os.path.basename(text_path).lower()
        # Padrões comuns: "Roteiro 1 português.txt", "roteiro_ingles.txt"
        language_patterns = {
            "portugues": ["português", "portugues", "pt", "pt-br", "pt_br"],
            "ingles": ["inglês", "ingles", "english", "en", "en-us"],
            "espanhol": ["espanhol", "español", "es", "es-es"],
            "frances": ["francês", "frances", "français", "fr", "fr-fr"],
            "alemao": ["alemão", "alemao", "deutsch", "de", "de-de"]
        }
        
        for lang, patterns in language_patterns.items():
            for pattern in patterns:
                if pattern in filename_lower:
                    return lang
    
    # Método 2: Nome do arquivo de áudio
    if audio_path:
        filename_lower = os.path.basename(audio_path).lower()
        language_patterns = {
            "portugues": ["português", "portugues", "pt", "pt-br"],
            "ingles": ["inglês", "ingles", "english", "en"],
            "espanhol": ["espanhol", "español", "es"],
            "frances": ["francês", "frances", "français", "fr"],
            "alemao": ["alemão", "alemao", "deutsch", "de"]
        }
        
        for lang, patterns in language_patterns.items():
            for pattern in patterns:
                if pattern in filename_lower:
                    return lang
    
    # Método 3: Transcrição AssemblyAI
    if transcript and hasattr(transcript, 'language_code'):
        lang_code = transcript.language_code
        lang_map = {
            "pt": "portugues",
            "en": "ingles",
            "es": "espanhol",
            "fr": "frances",
            "de": "alemao"
        }
        if lang_code in lang_map:
            return lang_map[lang_code]
    
    # Fallback
    return fallback


def load_vsl_keywords(keywords_file="vsl_keywords.json"):
    """Carrega palavras-chave de VSL do arquivo JSON."""
    try:
        # Se o caminho não for absoluto, tentar relativo ao diretório de trabalho atual
        if not os.path.isabs(keywords_file):
            # Tentar no diretório atual primeiro
            if not os.path.exists(keywords_file):
                # Tentar no diretório do script se __file__ estiver disponível
                try:
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    alt_path = os.path.join(script_dir, keywords_file)
                    if os.path.exists(alt_path):
                        keywords_file = alt_path
                except:
                    pass
        
        if not os.path.exists(keywords_file):
            return {}
        
        with open(keywords_file, 'r', encoding='utf-8') as f:
            keywords = json.load(f)
            return keywords
    except Exception as e:
        print(f"Erro ao carregar palavras-chave VSL: {e}")
        import traceback
        print(traceback.format_exc())
        return {}


def find_keyword_timestamp(transcript_words, keywords, language=None, log_callback=None):
    """
    Busca primeira ocorrência de palavra-chave no texto transcrito.
    
    Args:
        transcript_words: Lista de objetos Word do AssemblyAI
        keywords: Dicionário com palavras-chave por idioma
        language: Idioma específico ou None para buscar em todos os idiomas
        log_callback: Função opcional para logging (message, level)
    
    Returns:
        Tupla (timestamp_ms, palavra_encontrada) ou (None, None) se não encontrar
    """
    if not transcript_words:
        if log_callback:
            log_callback("Nenhuma palavra no transcript para buscar", "WARN")
        return None, None
    
    if not keywords:
        if log_callback:
            log_callback("Nenhuma palavra-chave carregada", "WARN")
        return None, None
    
    # Se idioma especificado, buscar apenas nele
    # Se None, buscar em todos os idiomas
    if language is not None:
        if language not in keywords:
            if log_callback:
                log_callback(f"Idioma '{language}' não encontrado nas palavras-chave. Idiomas disponíveis: {list(keywords.keys())}", "WARN")
            return None, None
        languages_to_search = [language]
    else:
        languages_to_search = list(keywords.keys())
    
    # Coletar todas as palavras-chave de todos os idiomas a buscar
    all_keywords = []
    for lang in languages_to_search:
        all_keywords.extend([kw.lower() for kw in keywords[lang]])
    
    if log_callback:
        log_callback(f"Buscando palavras-chave: {all_keywords}", "INFO")
    
    # Buscar primeira ocorrência
    for word in transcript_words:
        word_text = word.text.lower().strip()
        # Verificar se a palavra contém alguma palavra-chave
        for keyword in all_keywords:
            if keyword in word_text or word_text == keyword:
                if log_callback:
                    log_callback(f"Palavra-chave '{keyword}' encontrada na palavra '{word.text}' em {word.start}ms", "OK")
                return word.start, keyword
    
    if log_callback:
        # Log das primeiras palavras para debug
        first_words = [w.text for w in transcript_words[:10]]
        log_callback(f"Nenhuma palavra-chave encontrada. Primeiras palavras: {first_words}", "WARN")
    
    return None, None


# =============================================================================
# MOTOR DE LEGENDAS
# =============================================================================
class SubtitleEngine:
    """Motor para geracao e aplicacao de legendas."""

    def __init__(self, log_queue):
        self.log_queue = log_queue

    def log(self, message, level="INFO"):
        """Envia mensagem para a fila de log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = {"INFO": "", "OK": "[OK]", "WARN": "[!]", "ERROR": "[X]"}.get(level, "")
        self.log_queue.put(f"[{timestamp}] {prefix} {message}")

    def parse_srt(self, srt_path):
        """Le arquivo SRT com multiplos encodings."""
        encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']
        srt_content = None

        for enc in encodings:
            try:
                with open(srt_path, 'r', encoding=enc) as f:
                    srt_content = f.read()
                self.log(f"SRT lido com encoding: {enc}", "OK")
                break
            except UnicodeDecodeError:
                continue

        if not srt_content:
            raise Exception(f"Nao foi possivel ler o arquivo SRT: {srt_path}")

        events = []
        for block in srt_content.strip().split('\n\n'):
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                time_line = lines[1]
                text = ' '.join(lines[2:])
                text = re.sub(r'<[^>]+>', '', text).replace('\n', '\\N')

                match = re.match(
                    r'(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})',
                    time_line
                )
                if match:
                    h1, m1, s1, ms1, h2, m2, s2, ms2 = match.groups()
                    start = f"{h1}:{m1}:{s1}.{ms1[:2]}"
                    end = f"{h2}:{m2}:{s2}.{ms2[:2]}"
                    events.append((start, end, text))

        self.log(f"SRT parseado: {len(events)} eventos", "OK")
        return events

    def generate_with_assemblyai(self, audio_path, api_key, use_karaoke=True, return_full_data=False):
        """
        Transcreve audio com AssemblyAI e retorna eventos com timing.
        
        Args:
            audio_path: Caminho do arquivo de áudio
            api_key: Chave API do AssemblyAI
            use_karaoke: Se deve usar efeito karaoke
            return_full_data: Se True, retorna também transcript e palavras completas
        
        Returns:
            Se return_full_data=False: Lista de eventos (start, end, text)
            Se return_full_data=True: Tupla (events, transcript, words)
        """
        try:
            import assemblyai as aai
            aai.settings.api_key = api_key

            self.log("Iniciando transcricao AssemblyAI...", "INFO")

            config = aai.TranscriptionConfig(language_detection=True)
            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe(audio_path, config=config)

            if transcript.status == aai.TranscriptStatus.error:
                raise Exception(f"Erro AssemblyAI: {transcript.error}")

            if not transcript.words:
                raise Exception("Nenhuma palavra detectada no audio")

            self.log(f"Transcricao concluida: {len(transcript.words)} palavras", "OK")

            # Agrupar palavras em segmentos (4 palavras por linha)
            events = []
            segment_words = []
            max_words = 4

            for i, word in enumerate(transcript.words):
                segment_words.append(word)

                if len(segment_words) >= max_words or i == len(transcript.words) - 1:
                    start_ms = segment_words[0].start
                    end_ms = segment_words[-1].end

                    start = self.ms_to_ass_time(start_ms)
                    end = self.ms_to_ass_time(end_ms)

                    if use_karaoke:
                        text_parts = []
                        for w in segment_words:
                            duration_k = max(1, (w.end - w.start) // 10)
                            text_parts.append(f"{{\\k{duration_k}}}{w.text}")
                        text = " ".join(text_parts)
                    else:
                        text = " ".join(w.text for w in segment_words)

                    events.append((start, end, text))
                    segment_words = []

            self.log(f"Eventos gerados: {len(events)}", "OK")
            
            if return_full_data:
                return events, transcript, transcript.words
            return events

        except ImportError:
            raise Exception("Biblioteca 'assemblyai' nao instalada. Execute: pip install assemblyai")

    def ms_to_ass_time(self, ms):
        """Converte milissegundos para formato ASS (H:MM:SS.CC)."""
        h = ms // 3600000
        m = (ms % 3600000) // 60000
        s = (ms % 60000) // 1000
        cs = (ms % 1000) // 10
        return f"{h}:{m:02}:{s:02}.{cs:02}"

    def hex_to_ass_color(self, hex_color):
        """Converte cor hex RGB para formato ASS BGR."""
        if not hex_color or hex_color == "None":
            return "&H00FFFFFF"
        hex_val = hex_color.lstrip('#')
        if len(hex_val) == 6:
            r, g, b = hex_val[0:2], hex_val[2:4], hex_val[4:6]
            return f"&H00{b}{g}{r}".upper()
        elif len(hex_val) == 8:
            a, r, g, b = hex_val[0:2], hex_val[2:4], hex_val[4:6], hex_val[6:8]
            return f"&H{a}{b}{g}{r}".upper()
        return "&H00FFFFFF"

    def create_ass_file(self, events, options, output_path, resolution="720p", vsl_start_time=None, vsl_end_time=None):
        """
        Cria arquivo ASS com eventos e estilo.
        
        Args:
            events: Lista de tuplas (start, end, text)
            options: Dicionário com opções de estilo
            output_path: Caminho de saída
            resolution: Resolução do vídeo
            vsl_start_time: Timestamp de início do VSL (formato ASS) ou None
            vsl_end_time: Timestamp de fim do VSL (formato ASS) ou None
        """
        width, height = {"720p": (1280, 720), "1080p": (1920, 1080)}[resolution]

        primary = self.hex_to_ass_color(options.get('color_primary', '#FFFFFF'))
        outline = self.hex_to_ass_color(options.get('color_outline', '#000000'))
        shadow = self.hex_to_ass_color(options.get('color_shadow', '#80000000'))
        karaoke = self.hex_to_ass_color(options.get('color_karaoke', '#FFFF00'))

        font_name = options.get('font_name', 'Arial')
        font_size = options.get('font_size', 48)
        outline_size = options.get('outline_size', 2)
        shadow_size = options.get('shadow_size', 2)
        position = options.get('position', '2')

        header = f"""[Script Info]
Title: Legendas Editor Espiritualidade
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},{primary},{karaoke},{outline},{shadow},-1,0,0,0,100,100,0,0,1,{outline_size},{shadow_size},{position},10,10,30,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

        dialogues = []
        vsl_period_active = vsl_start_time is not None and vsl_end_time is not None
        
        for start, end, text in events:
            # Se há período de VSL, pular legendas que se sobrepõem
            if vsl_period_active:
                # Converter timestamps ASS para comparação (formato H:MM:SS.CC)
                def ass_time_to_seconds(ass_time):
                    parts = ass_time.split(':')
                    h = int(parts[0])
                    m = int(parts[1])
                    s_parts = parts[2].split('.')
                    s = int(s_parts[0])
                    cs = int(s_parts[1]) if len(s_parts) > 1 else 0
                    return h * 3600 + m * 60 + s + cs / 100.0
                
                event_start_sec = ass_time_to_seconds(start)
                event_end_sec = ass_time_to_seconds(end)
                vsl_start_sec = ass_time_to_seconds(vsl_start_time)
                vsl_end_sec = ass_time_to_seconds(vsl_end_time)
                
                # Se o evento se sobrepõe ao período do VSL, não adicionar
                if not (event_end_sec <= vsl_start_sec or event_start_sec >= vsl_end_sec):
                    continue  # Pular este evento (dentro do período VSL)
            
            dialogues.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")

        ass_content = header + "\n".join(dialogues)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(ass_content)

        skipped = len(events) - len(dialogues)
        if skipped > 0:
            self.log(f"Arquivo ASS criado: {len(dialogues)} legendas ({skipped} omitidas durante VSL)", "OK")
        else:
            self.log(f"Arquivo ASS criado: {len(events)} legendas", "OK")
        return output_path

    def burn_subtitles(self, video_path, ass_path, output_path, use_gpu=False):
        """Aplica legendas ASS no video com FFmpeg."""
        self.log("Aplicando legendas no video...", "INFO")

        # Escape do path para Windows
        safe_ass = ass_path.replace("\\", "/").replace(":", "\\:")

        if use_gpu:
            encoder = ["-c:v", "h264_nvenc", "-preset", "p4", "-b:v", "8M"]
        else:
            encoder = ["-c:v", "libx264", "-preset", "fast", "-crf", "23"]

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"ass=filename='{safe_ass}'",
            *encoder,
            "-c:a", "copy",
            "-pix_fmt", "yuv420p",
            output_path
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )

        if result.returncode != 0:
            self.log(f"Erro FFmpeg: {result.stderr[-300:] if result.stderr else ''}", "ERROR")
            raise Exception(f"Erro ao aplicar legendas")

        self.log("Legendas aplicadas com sucesso!", "OK")
        return output_path


# =============================================================================
# MOTOR DE VSL (VIDEO SALES LETTER)
# =============================================================================
class VSLEngine:
    """Motor para inserção de VSL no vídeo."""

    def __init__(self, log_queue):
        self.log_queue = log_queue

    def log(self, message, level="INFO"):
        """Envia mensagem para a fila de log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = {"INFO": "", "OK": "[OK]", "WARN": "[!]", "ERROR": "[X]"}.get(level, "")
        self.log_queue.put(f"[{timestamp}] {prefix} {message}")

    def get_video_duration(self, video_path):
        """Obtém a duração do vídeo em segundos."""
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
            return float(result.stdout.strip())
        except Exception as e:
            self.log(f"Erro ao obter duração do VSL: {e}", "ERROR")
            return None

    def get_video_resolution(self, video_path):
        """Obtém a resolução do vídeo (width, height)."""
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
            lines = result.stdout.strip().split('\n')
            width = int(lines[0])
            height = int(lines[1])
            return (width, height)
        except Exception as e:
            self.log(f"Erro ao obter resolução do vídeo: {e}", "ERROR")
            return None

    def insert_vsl(self, video_path, vsl_path, start_time_sec, output_path, use_gpu=False):
        """
        Insere VSL no vídeo - O VÍDEO PAUSA, TOCA A VSL COM SEU ÁUDIO, E DEPOIS CONTINUA.
        
        MÉTODO: Cortar e Concatenar com filter_complex
        1. Corta o vídeo do início até start_time_sec (Parte 1)
        2. Prepara a VSL com seu próprio áudio (ou silêncio se não tiver)
        3. Corta o vídeo de start_time_sec até o fim (Parte 2)
        4. Concatena usando filter_complex para garantir sincronização A/V
        
        Timeline:
        [0s -------- 65s] [VSL 55s com áudio] [65s -------- fim]
             Parte 1           VSL inserida        Parte 2
             (PAUSA)          (TOCA VSL)      (CONTINUA de onde parou)
        
        Args:
            video_path: Caminho do vídeo base (com narração)
            vsl_path: Caminho do arquivo VSL (com seu próprio áudio)
            start_time_sec: Timestamp de início em segundos (onde pausar o vídeo)
            output_path: Caminho de saída
            use_gpu: Se deve usar GPU para encoding
        
        Returns:
            Caminho do vídeo com VSL inserido ou None em caso de erro
        """
        try:
            import tempfile
            import shutil
            
            # Obter duração do VSL
            vsl_duration = self.get_video_duration(vsl_path)
            if vsl_duration is None:
                self.log("Nao foi possivel obter duracao do VSL", "ERROR")
                return None
            
            self.log(f"VSL duracao: {vsl_duration:.2f}s", "DEBUG")

            # Obter duração do vídeo base
            base_duration = self.get_video_duration(video_path)
            if base_duration is None:
                self.log("Nao foi possivel obter duracao do video base", "ERROR")
                return None
            
            self.log(f"Video base duracao: {base_duration:.2f}s", "DEBUG")

            # Validar que o ponto de inserção está dentro do vídeo
            if start_time_sec >= base_duration:
                self.log(f"Ponto de insercao ({start_time_sec}s) apos fim do video ({base_duration}s)", "ERROR")
                return None

            self.log(f"Inserindo VSL em {start_time_sec:.2f}s (duracao VSL: {vsl_duration:.2f}s)...", "INFO")
            self.log(f"Video vai PAUSAR em {start_time_sec:.2f}s, tocar VSL, e CONTINUAR de {start_time_sec:.2f}s", "INFO")

            # Obter resolução do vídeo base
            base_resolution = self.get_video_resolution(video_path)
            if base_resolution is None:
                self.log("Nao foi possivel obter resolucao do video base", "ERROR")
                return None

            base_width, base_height = base_resolution
            self.log(f"Resolucao do video base: {base_width}x{base_height}", "INFO")

            # Configurar encoder
            if use_gpu:
                encoder_args = ["-c:v", "h264_nvenc", "-preset", "p4", "-b:v", "8M"]
            else:
                encoder_args = ["-c:v", "libx264", "-preset", "fast", "-crf", "23"]

            # Criar diretório temporário
            temp_dir = tempfile.mkdtemp(prefix="vsl_insert_")
            
            try:
                # Verificar se a VSL tem áudio
                cmd_check_vsl_audio = [
                    "ffprobe", "-v", "error",
                    "-select_streams", "a:0",
                    "-show_entries", "stream=codec_type",
                    "-of", "csv=p=0",
                    vsl_path
                ]
                audio_check = subprocess.run(cmd_check_vsl_audio, capture_output=True, text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
                vsl_has_audio = "audio" in audio_check.stdout.lower()
                
                if vsl_has_audio:
                    self.log("VSL tem audio - sera preservado", "INFO")
                else:
                    self.log("VSL nao tem audio - sera adicionado silencio", "WARN")

                # Verificar se o vídeo base tem áudio
                cmd_check_base_audio = [
                    "ffprobe", "-v", "error",
                    "-select_streams", "a:0",
                    "-show_entries", "stream=codec_type",
                    "-of", "csv=p=0",
                    video_path
                ]
                base_audio_check = subprocess.run(cmd_check_base_audio, capture_output=True, text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
                base_has_audio = "audio" in base_audio_check.stdout.lower()
                
                if base_has_audio:
                    self.log("Video base tem audio", "INFO")
                else:
                    self.log("Video base NAO tem audio - sera adicionado silencio", "WARN")

                # ============================================================
                # MÉTODO ÚNICO: Usar filter_complex para concatenar tudo
                # Isso garante sincronização perfeita de áudio e vídeo
                # ============================================================
                
                has_part1 = start_time_sec > 0.5
                has_part2 = start_time_sec < (base_duration - 0.5)
                
                # Construir comando FFmpeg com filter_complex
                cmd = ["ffmpeg", "-y"]
                
                # Input 0: Vídeo base
                cmd.extend(["-i", video_path])
                
                # Input 1: VSL
                cmd.extend(["-i", vsl_path])
                
                # Inputs de silêncio separados para cada parte que precisar
                # (FFmpeg não permite reutilizar o mesmo stream múltiplas vezes)
                silence_input_idx = 2
                silence_inputs_needed = 0
                if not base_has_audio:
                    if has_part1:
                        silence_inputs_needed += 1
                    if has_part2:
                        silence_inputs_needed += 1
                if not vsl_has_audio:
                    silence_inputs_needed += 1
                
                # Adicionar inputs de silêncio necessários
                for _ in range(max(1, silence_inputs_needed)):
                    cmd.extend(["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"])
                
                # Construir filter_complex
                filter_parts = []
                video_concat_inputs = []
                audio_concat_inputs = []
                current_silence_idx = 2  # Começa no input 2
                
                # Parte 1: Início do vídeo (0 até start_time_sec)
                if has_part1:
                    filter_parts.append(f"[0:v]trim=0:{start_time_sec},setpts=PTS-STARTPTS,scale={base_width}:{base_height},fps=24,setsar=1[v0]")
                    if base_has_audio:
                        filter_parts.append(f"[0:a]atrim=0:{start_time_sec},asetpts=PTS-STARTPTS,aresample=48000[a0]")
                    else:
                        # Vídeo base não tem áudio - usar silêncio
                        filter_parts.append(f"[{current_silence_idx}:a]atrim=0:{start_time_sec},asetpts=PTS-STARTPTS[a0]")
                        current_silence_idx += 1
                    video_concat_inputs.append("[v0]")
                    audio_concat_inputs.append("[a0]")
                
                # VSL: Redimensionar e preparar (trim para garantir duração exata)
                filter_parts.append(f"[1:v]trim=0:{vsl_duration},setpts=PTS-STARTPTS,scale={base_width}:{base_height}:force_original_aspect_ratio=decrease,pad={base_width}:{base_height}:(ow-iw)/2:(oh-ih)/2:color=black,fps=24,setsar=1[v1]")
                video_concat_inputs.append("[v1]")
                
                if vsl_has_audio:
                    filter_parts.append(f"[1:a]aresample=48000,asetpts=PTS-STARTPTS[a1]")
                    audio_concat_inputs.append("[a1]")
                else:
                    # Usar silêncio, cortado para duração da VSL
                    filter_parts.append(f"[{current_silence_idx}:a]atrim=0:{vsl_duration},asetpts=PTS-STARTPTS[a1]")
                    current_silence_idx += 1
                    audio_concat_inputs.append("[a1]")
                
                # Parte 2: Continuação do vídeo (de start_time_sec até o fim)
                if has_part2:
                    filter_parts.append(f"[0:v]trim={start_time_sec}:{base_duration},setpts=PTS-STARTPTS,scale={base_width}:{base_height},fps=24,setsar=1[v2]")
                    if base_has_audio:
                        filter_parts.append(f"[0:a]atrim={start_time_sec},asetpts=PTS-STARTPTS,aresample=48000[a2]")
                    else:
                        # Vídeo base não tem áudio - usar silêncio
                        remaining_duration = base_duration - start_time_sec
                        filter_parts.append(f"[{current_silence_idx}:a]atrim=0:{remaining_duration},asetpts=PTS-STARTPTS[a2]")
                        current_silence_idx += 1
                    video_concat_inputs.append("[v2]")
                    audio_concat_inputs.append("[a2]")
                
                # Concatenar vídeo
                n_parts = len(video_concat_inputs)
                filter_parts.append(f"{''.join(video_concat_inputs)}concat=n={n_parts}:v=1:a=0[vout]")
                
                # Concatenar áudio
                filter_parts.append(f"{''.join(audio_concat_inputs)}concat=n={n_parts}:v=0:a=1[aout]")
                
                filter_complex = ";".join(filter_parts)
                
                self.log(f"Concatenando {n_parts} partes com filter_complex...", "INFO")
                
                cmd.extend([
                    "-filter_complex", filter_complex,
                    "-map", "[vout]",
                    "-map", "[aout]",
                    *encoder_args,
                    "-c:a", "aac", "-b:a", "192k",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    output_path
                ])
                
                result = subprocess.run(cmd, capture_output=True, text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
                
                if result.returncode != 0:
                    self.log(f"Erro no filter_complex: {result.stderr[-500:] if result.stderr else ''}", "ERROR")
                    
                    # Fallback: Método antigo com arquivos separados
                    self.log("Tentando método alternativo (arquivos separados)...", "WARN")
                    return self._insert_vsl_fallback(video_path, vsl_path, start_time_sec, output_path, 
                                                     use_gpu, temp_dir, base_width, base_height, 
                                                     vsl_duration, base_duration, vsl_has_audio, 
                                                     base_has_audio, encoder_args)
                
                # Verificar resultado final
                if not os.path.exists(output_path) or os.path.getsize(output_path) < 10000:
                    self.log("Video final nao foi criado corretamente", "ERROR")
                    return None
                
                final_duration = self.get_video_duration(output_path)
                expected_duration = (start_time_sec if has_part1 else 0) + vsl_duration + (base_duration - start_time_sec if has_part2 else 0)
                
                self.log(f"VSL inserido com sucesso!", "OK")
                self.log(f"Duracao final: {final_duration:.2f}s (esperado: ~{expected_duration:.2f}s)", "OK")
                self.log(f"Video PAUSA em {start_time_sec:.2f}s -> VSL ({vsl_duration:.2f}s) -> CONTINUA de {start_time_sec:.2f}s", "OK")
                
                return output_path
                
            finally:
                # Limpar arquivos temporários
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except:
                    pass

        except Exception as e:
            self.log(f"Erro ao inserir VSL: {str(e)}", "ERROR")
            import traceback
            self.log(traceback.format_exc(), "ERROR")
            return None

    def _insert_vsl_fallback(self, video_path, vsl_path, start_time_sec, output_path, 
                             use_gpu, temp_dir, base_width, base_height, 
                             vsl_duration, base_duration, vsl_has_audio, 
                             base_has_audio, encoder_args):
        """
        Método fallback para inserção de VSL usando arquivos separados.
        Usado quando o filter_complex falha.
        
        Agora suporta vídeos base sem áudio (modo 1 imagem).
        """
        try:
            has_part1 = start_time_sec > 0.5
            has_part2 = start_time_sec < (base_duration - 0.5)
            
            # ============================================================
            # PARTE 1: Extrair início do vídeo (0 até start_time_sec)
            # ============================================================
            part1_path = os.path.join(temp_dir, "part1.mp4")
            
            if has_part1:
                self.log(f"Extraindo Parte 1: 0s ate {start_time_sec:.2f}s", "DEBUG")
                
                if base_has_audio:
                    # Vídeo base tem áudio - copiar normalmente
                    cmd_part1 = [
                        "ffmpeg", "-y",
                        "-i", video_path,
                        "-t", str(start_time_sec),
                        "-vf", f"scale={base_width}:{base_height},fps=24",
                        *encoder_args,
                        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
                        "-pix_fmt", "yuv420p",
                        part1_path
                    ]
                else:
                    # Vídeo base NÃO tem áudio - adicionar silêncio
                    cmd_part1 = [
                        "ffmpeg", "-y",
                        "-i", video_path,
                        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
                        "-t", str(start_time_sec),
                        "-vf", f"scale={base_width}:{base_height},fps=24",
                        "-map", "0:v",
                        "-map", "1:a",
                        *encoder_args,
                        "-c:a", "aac", "-b:a", "192k",
                        "-pix_fmt", "yuv420p",
                        "-shortest",
                        part1_path
                    ]
                    
                result = subprocess.run(cmd_part1, capture_output=True, text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
                if result.returncode != 0:
                    self.log(f"Erro ao extrair parte 1: {result.stderr[-300:] if result.stderr else ''}", "ERROR")
                    return None
                self.log(f"Parte 1 OK", "OK")
            
            # ============================================================
            # PARTE 2: Preparar VSL
            # ============================================================
            vsl_prepared_path = os.path.join(temp_dir, "vsl_prepared.mp4")
            
            if vsl_has_audio:
                cmd_vsl = [
                    "ffmpeg", "-y",
                    "-i", vsl_path,
                    "-vf", f"scale={base_width}:{base_height}:force_original_aspect_ratio=decrease,pad={base_width}:{base_height}:(ow-iw)/2:(oh-ih)/2:color=black,fps=24",
                    *encoder_args,
                    "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
                    "-pix_fmt", "yuv420p",
                    "-shortest",
                    vsl_prepared_path
                ]
            else:
                cmd_vsl = [
                    "ffmpeg", "-y",
                    "-i", vsl_path,
                    "-f", "lavfi", "-i", f"anullsrc=channel_layout=stereo:sample_rate=48000",
                    "-vf", f"scale={base_width}:{base_height}:force_original_aspect_ratio=decrease,pad={base_width}:{base_height}:(ow-iw)/2:(oh-ih)/2:color=black,fps=24",
                    "-map", "0:v",
                    "-map", "1:a",
                    *encoder_args,
                    "-c:a", "aac", "-b:a", "192k",
                    "-pix_fmt", "yuv420p",
                    "-t", str(vsl_duration),
                    vsl_prepared_path
                ]
            
            result = subprocess.run(cmd_vsl, capture_output=True, text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
            if result.returncode != 0:
                self.log(f"Erro ao preparar VSL: {result.stderr[-300:] if result.stderr else ''}", "ERROR")
                return None
            self.log(f"VSL preparada OK", "OK")
            
            # ============================================================
            # PARTE 3: Extrair continuação do vídeo
            # ============================================================
            part2_path = os.path.join(temp_dir, "part2.mp4")
            
            if has_part2:
                self.log(f"Extraindo Parte 2: {start_time_sec:.2f}s ate {base_duration:.2f}s", "DEBUG")
                
                if base_has_audio:
                    # Vídeo base tem áudio - copiar normalmente
                    cmd_part2 = [
                        "ffmpeg", "-y",
                        "-ss", str(start_time_sec),
                        "-i", video_path,
                        "-vf", f"scale={base_width}:{base_height},fps=24",
                        *encoder_args,
                        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
                        "-pix_fmt", "yuv420p",
                        part2_path
                    ]
                else:
                    # Vídeo base NÃO tem áudio - adicionar silêncio
                    remaining_duration = base_duration - start_time_sec
                    cmd_part2 = [
                        "ffmpeg", "-y",
                        "-ss", str(start_time_sec),
                        "-i", video_path,
                        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
                        "-t", str(remaining_duration),
                        "-vf", f"scale={base_width}:{base_height},fps=24",
                        "-map", "0:v",
                        "-map", "1:a",
                        *encoder_args,
                        "-c:a", "aac", "-b:a", "192k",
                        "-pix_fmt", "yuv420p",
                        "-shortest",
                        part2_path
                    ]
                    
                result = subprocess.run(cmd_part2, capture_output=True, text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
                if result.returncode != 0:
                    self.log(f"Erro ao extrair parte 2: {result.stderr[-300:] if result.stderr else ''}", "ERROR")
                    return None
                self.log(f"Parte 2 OK", "OK")
            
            # ============================================================
            # CONCATENAR usando concat demuxer (mais robusto para arquivos preparados)
            # ============================================================
            parts = []
            if has_part1 and os.path.exists(part1_path):
                parts.append(part1_path)
            parts.append(vsl_prepared_path)
            if has_part2 and os.path.exists(part2_path):
                parts.append(part2_path)
            
            self.log(f"Concatenando {len(parts)} partes com concat demuxer...", "INFO")
            
            # Criar arquivo de lista para concat demuxer
            concat_list_path = os.path.join(temp_dir, "concat_list.txt")
            with open(concat_list_path, "w", encoding="utf-8") as f:
                for p in parts:
                    # Escapar caracteres especiais no caminho
                    escaped_path = p.replace("\\", "/").replace("'", "'\\''")
                    f.write(f"file '{escaped_path}'\n")
            
            # Usar concat demuxer - mais confiável quando os arquivos já estão preparados
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_list_path,
                "-c", "copy",  # Copiar streams sem re-encoding (mais rápido)
                "-movflags", "+faststart",
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
            
            if result.returncode != 0:
                self.log(f"Erro no concat demuxer: {result.stderr[-400:] if result.stderr else ''}", "WARN")
                
                # Fallback: Re-encode com filter_complex
                self.log("Tentando re-encode com filter_complex...", "INFO")
                cmd = ["ffmpeg", "-y"]
                for p in parts:
                    cmd.extend(["-i", p])
                
                n = len(parts)
                filter_v = "".join([f"[{i}:v]" for i in range(n)]) + f"concat=n={n}:v=1:a=0[vout]"
                filter_a = "".join([f"[{i}:a]" for i in range(n)]) + f"concat=n={n}:v=0:a=1[aout]"
                
                cmd.extend([
                    "-filter_complex", f"{filter_v};{filter_a}",
                    "-map", "[vout]",
                    "-map", "[aout]",
                    *encoder_args,
                    "-c:a", "aac", "-b:a", "192k",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    output_path
                ])
                
                result = subprocess.run(cmd, capture_output=True, text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
                
                if result.returncode != 0:
                    self.log(f"Erro ao concatenar: {result.stderr[-400:] if result.stderr else ''}", "ERROR")
                    return None
            
            if not os.path.exists(output_path) or os.path.getsize(output_path) < 10000:
                self.log("Video final nao foi criado corretamente", "ERROR")
                return None
            
            final_duration = self.get_video_duration(output_path)
            self.log(f"VSL inserido com sucesso (fallback)! Duracao: {final_duration:.2f}s", "OK")
            
            return output_path
            
        except Exception as e:
            self.log(f"Erro no fallback: {str(e)}", "ERROR")
            return None


# =============================================================================
# MOTOR DE BACKLOG VIDEOS (VIDEOS INTRODUTORIOS)
# =============================================================================
class BacklogVideoEngine:
    """Motor para criação de sequência introdutória com vídeos do backlog."""

    def __init__(self, log_queue):
        self.log_queue = log_queue
        self.shuffle_lock = threading.Lock()  # Lock para thread-safety no embaralhamento

    def log(self, message, level="INFO"):
        """Envia mensagem para a fila de log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = {"INFO": "", "OK": "[OK]", "WARN": "[!]", "ERROR": "[X]"}.get(level, "")
        self.log_queue.put(f"[{timestamp}] {prefix} {message}")

    def get_backlog_videos(self, backlog_folder):
        """
        Lista todos os vídeos disponíveis na pasta do backlog.
        
        Args:
            backlog_folder: Caminho da pasta do backlog
            
        Returns:
            Lista de caminhos de vídeos ou lista vazia se erro
        """
        if not os.path.exists(backlog_folder):
            self.log(f"Pasta do backlog não encontrada: {backlog_folder}. Criando pasta...", "WARN")
            try:
                os.makedirs(backlog_folder, exist_ok=True)
                self.log(f"Pasta criada: {backlog_folder}", "INFO")
            except Exception as e:
                self.log(f"Erro ao criar pasta do backlog: {str(e)}", "ERROR")
                return []
        
        video_extensions = ('.mp4', '.mov', '.avi', '.mkv', '.webm')
        videos = []
        
        try:
            for f in os.listdir(backlog_folder):
                full_path = os.path.join(backlog_folder, f)
                if os.path.isfile(full_path) and f.lower().endswith(video_extensions):
                    videos.append(full_path)
            
            videos = sorted(videos)  # Ordenar por nome
            self.log(f"Encontrados {len(videos)} vídeos no backlog", "INFO")
            return videos
        except Exception as e:
            self.log(f"Erro ao listar vídeos do backlog: {str(e)}", "ERROR")
            return []

    def shuffle_and_rename(self, video_paths, backlog_folder):
        """
        Embaralha lista de vídeos e renomeia com prefixo numérico.
        Thread-safe.
        
        Args:
            video_paths: Lista de caminhos de vídeos
            backlog_folder: Pasta do backlog
            
        Returns:
            Lista de caminhos renomeados (mesma ordem embaralhada)
        """
        if not video_paths:
            return []
        
        with self.shuffle_lock:
            try:
                import random
                # Criar cópia embaralhada
                shuffled = video_paths.copy()
                random.shuffle(shuffled)
                
                # Renomear arquivos com prefixo numérico
                renamed_paths = []
                for i, video_path in enumerate(shuffled):
                    # Obter nome original do arquivo
                    original_name = os.path.basename(video_path)
                    # Remover prefixo numérico existente se houver
                    if original_name[0:4].isdigit() and original_name[4] == '_':
                        original_name = original_name[5:]  # Remove "001_" por exemplo
                    
                    # Criar novo nome com prefixo de 3 dígitos
                    prefix = f"{i+1:03d}"  # 001, 002, ..., 100
                    new_name = f"{prefix}_{original_name}"
                    new_path = os.path.join(backlog_folder, new_name)
                    
                    # Renomear arquivo
                    if video_path != new_path:
                        try:
                            # Verificar se arquivo está sendo usado
                            if os.path.exists(new_path):
                                # Se já existe com esse nome, usar timestamp
                                base, ext = os.path.splitext(new_name)
                                new_name = f"{base}_{int(time.time())}{ext}"
                                new_path = os.path.join(backlog_folder, new_name)
                            
                            os.rename(video_path, new_path)
                            renamed_paths.append(new_path)
                            self.log(f"Renomeado: {os.path.basename(video_path)} -> {os.path.basename(new_path)}", "DEBUG")
                        except Exception as e:
                            self.log(f"Erro ao renomear {video_path}: {str(e)}", "WARN")
                            renamed_paths.append(video_path)  # Manter original se falhar
                    else:
                        renamed_paths.append(video_path)
                
                self.log(f"Embaralhamento concluído: {len(renamed_paths)} vídeos renomeados", "OK")
                return renamed_paths
                
            except Exception as e:
                self.log(f"Erro no embaralhamento: {str(e)}", "ERROR")
                return video_paths  # Retornar original se falhar

    def get_video_duration(self, video_path):
        """Obtém a duração do vídeo em segundos."""
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
            return float(result.stdout.strip())
        except Exception as e:
            self.log(f"Erro ao obter duração do vídeo {os.path.basename(video_path)}: {e}", "WARN")
            return None

    def get_video_resolution(self, video_path):
        """Obtém a resolução do vídeo (width, height)."""
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
            lines = result.stdout.strip().split('\n')
            width = int(lines[0])
            height = int(lines[1])
            return (width, height)
        except Exception as e:
            self.log(f"Erro ao obter resolução do vídeo: {e}", "WARN")
            return None

    def create_backlog_intro(self, backlog_folder, output_path, resolution, target_duration=60.0,
                             audio_volume=0.25, transition_duration=0.5, fade_out_duration=1.0, 
                             use_gpu=False):
        """
        Cria sequência introdutória de vídeos do backlog com duração total de 1 minuto.
        
        Args:
            backlog_folder: Pasta com vídeos do backlog
            output_path: Caminho de saída do vídeo intro
            resolution: Resolução ("720p" ou "1080p")
            target_duration: Duração total desejada em segundos (padrão: 60.0s = 1 minuto)
            audio_volume: Volume do áudio (0.0-1.0, padrão: 0.25)
            transition_duration: Duração do crossfade entre vídeos (padrão: 0.5s)
            fade_out_duration: Duração do fade out no último vídeo (padrão: 1.0s)
            use_gpu: Se deve usar GPU para encoding
            
        Returns:
            Caminho do vídeo intro ou None em caso de erro
        """
        try:
            # Obter lista de vídeos
            all_videos = self.get_backlog_videos(backlog_folder)
            
            if len(all_videos) == 0:
                self.log("Nenhum vídeo encontrado no backlog", "ERROR")
                return None
            
            # Selecionar vídeos até atingir a duração alvo (1 minuto)
            selected_videos = []
            total_duration = 0.0
            
            for video_path in all_videos:
                duration = self.get_video_duration(video_path)
                if duration is None:
                    duration = 8.0  # Fallback: assumir 8 segundos
                
                # Adicionar vídeo à seleção
                selected_videos.append(video_path)
                total_duration += duration
                
                # Se já atingiu ou ultrapassou o tempo alvo, parar
                if total_duration >= target_duration:
                    break
            
            if len(selected_videos) == 0:
                self.log("Não foi possível selecionar vídeos suficientes", "ERROR")
                return None
            
            # Ajustar último vídeo se necessário para atingir exatamente 60s
            if total_duration > target_duration and len(selected_videos) > 0:
                # Cortar último vídeo para atingir exatamente 60s
                last_video = selected_videos[-1]
                last_duration = self.get_video_duration(last_video)
                if last_duration:
                    excess = total_duration - target_duration
                    if excess > 0 and last_duration > excess:
                        # Vamos cortar o último vídeo durante o processamento
                        self.log(f"Ajustando último vídeo: removendo {excess:.1f}s para atingir exatamente {target_duration}s", "INFO")
            
            self.log(f"Selecionados {len(selected_videos)} vídeos para ~{min(total_duration, target_duration):.1f}s de intro", "INFO")
            
            self.log(f"Criando intro com {len(selected_videos)} vídeos do backlog...", "INFO")
            
            # Obter resolução alvo
            target_width, target_height = RESOLUTIONS[resolution]
            
            # Criar pasta temporária para processar vídeos
            temp_dir = tempfile.mkdtemp(prefix="backlog_intro_")
            
            # Calcular durações de todos os vídeos
            video_durations = []
            for video_path in selected_videos:
                dur = self.get_video_duration(video_path)
                if dur is None:
                    dur = 8.0
                video_durations.append(dur)
            
            # Calcular duração acumulada até antes do último vídeo
            cumulative_before_last = sum(video_durations[:-1]) if len(video_durations) > 1 else 0.0
            
            # Processar cada vídeo: ajustar resolução, reduzir áudio, aplicar fade out no último
            processed_videos = []
            for i, video_path in enumerate(selected_videos):
                is_last = (i == len(selected_videos) - 1)
                processed_path = os.path.join(temp_dir, f"processed_{i:03d}.mp4")
                
                # Obter duração do vídeo
                original_duration = video_durations[i]
                duration = original_duration
                
                # Se é o último vídeo e ultrapassou o tempo alvo, cortar
                if is_last and cumulative_before_last + duration > target_duration:
                    max_duration = target_duration - cumulative_before_last
                    if max_duration > 0:
                        duration = max_duration
                        self.log(f"Cortando último vídeo para {duration:.1f}s para atingir exatamente {target_duration}s", "INFO")
                    else:
                        self.log("Último vídeo não cabe no tempo alvo, pulando", "WARN")
                        continue
                
                # Construir filtros FFmpeg
                # 1. Ajustar resolução (smart crop 16:9)
                video_filter = (
                    f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,"
                    f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:color=black"
                )
                
                # 2. Fade out no último vídeo
                if is_last and fade_out_duration > 0:
                    fade_start = max(0, duration - fade_out_duration)
                    video_filter += f",fade=t=out:st={fade_start}:d={fade_out_duration}:alpha=1"
                
                # Configurar encoder
                if use_gpu:
                    encoder_args = ["-c:v", "h264_nvenc", "-preset", "p4", "-b:v", "8M"]
                else:
                    encoder_args = ["-c:v", "libx264", "-preset", "fast", "-crf", "23"]
                
                # Verificar se vídeo tem áudio
                cmd_check_audio = [
                    "ffprobe", "-v", "error",
                    "-select_streams", "a:0",
                    "-show_entries", "stream=codec_type",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    video_path
                ]
                has_audio = False
                try:
                    result_check = subprocess.run(
                        cmd_check_audio,
                        capture_output=True,
                        text=True,
                        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                    )
                    has_audio = result_check.returncode == 0 and "audio" in result_check.stdout.lower()
                except:
                    pass
                
                # Construir comando FFmpeg
                cmd = [
                    "ffmpeg", "-y",
                    "-i", video_path,
                    "-vf", video_filter,
                    *encoder_args,
                    "-pix_fmt", "yuv420p",
                ]
                
                # Adicionar filtro de áudio apenas se houver áudio
                if has_audio:
                    cmd.extend(["-af", f"volume={audio_volume}"])
                    cmd.extend(["-c:a", "aac", "-b:a", "128k"])
                else:
                    cmd.append("-an")  # Sem áudio
                
                # Se precisar cortar o vídeo para ajustar duração (último vídeo)
                if is_last and duration < original_duration:
                    cmd.extend(["-t", str(duration)])
                else:
                    cmd.append("-shortest")
                
                cmd.append(processed_path)
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                )
                
                if result.returncode != 0:
                    self.log(f"Erro ao processar vídeo {i+1}: {result.stderr[-200:] if result.stderr else ''}", "WARN")
                    continue
                
                processed_videos.append(processed_path)
                self.log(f"Vídeo {i+1}/{len(selected_videos)} processado", "INFO")
            
            if len(processed_videos) == 0:
                self.log("Nenhum vídeo foi processado com sucesso", "ERROR")
                shutil.rmtree(temp_dir, ignore_errors=True)
                return None
            
            # Concatenar vídeos com crossfade
            if len(processed_videos) == 1:
                # Apenas um vídeo, apenas copiar
                shutil.copy(processed_videos[0], output_path)
            else:
                # Múltiplos vídeos: concatenar com crossfade usando xfade
                self.log(f"Concatenando {len(processed_videos)} vídeos com crossfade...", "INFO")
                
                # Carregar todos os vídeos
                input_args = []
                for video_path in processed_videos:
                    input_args.extend(["-i", video_path])
                
                # Calcular durações para xfade
                durations = []
                for video_path in processed_videos:
                    dur = self.get_video_duration(video_path)
                    if dur is None:
                        dur = 8.0  # Fallback
                    durations.append(dur)
                
                # Construir filter_complex com xfade
                filter_parts = []
                
                # Preparar cada vídeo com setpts
                for i in range(len(processed_videos)):
                    filter_parts.append(f"[{i}:v]setpts=PTS-STARTPTS[v{i}]")
                
                # Aplicar crossfade entre vídeos consecutivos
                # xfade offset é o tempo absoluto onde a transição começa
                current_output = "v0"
                cumulative_time = 0
                
                for i in range(1, len(processed_videos)):
                    # Offset: tempo acumulado até o fim do vídeo anterior menos duração da transição
                    xfade_offset = cumulative_time + durations[i-1] - transition_duration
                    xfade_offset = max(0, xfade_offset)  # Não pode ser negativo
                    
                    output_label = f"vout{i}" if i < len(processed_videos) - 1 else "vfinal"
                    filter_parts.append(f"[{current_output}][v{i}]xfade=transition=fade:duration={transition_duration}:offset={xfade_offset}[{output_label}]")
                    
                    current_output = output_label
                    cumulative_time += durations[i-1]
                
                # Mixar áudios com crossfade suave
                audio_parts = []
                for i in range(len(processed_videos)):
                    audio_parts.append(f"[{i}:a]")
                audio_filter = f"{''.join(audio_parts)}amix=inputs={len(processed_videos)}:duration=longest:dropout_transition=2[aout]"
                
                # Combinar filtros
                video_filter = ";".join(filter_parts)
                full_filter = f"{video_filter};{audio_filter}"
                
                cmd = [
                    "ffmpeg", "-y",
                    *input_args,
                    "-filter_complex", full_filter,
                    "-map", "[vfinal]",
                    "-map", "[aout]",
                    *encoder_args,
                    "-c:a", "aac", "-b:a", "192k",
                    "-pix_fmt", "yuv420p",
                    output_path
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                )
                
                if result.returncode != 0:
                    # Fallback: usar concat simples sem crossfade
                    self.log("Erro no crossfade, tentando concatenação simples...", "WARN")
                    concat_list_path = os.path.join(temp_dir, "concat_list.txt")
                    with open(concat_list_path, 'w', encoding='utf-8') as f:
                        for p in processed_videos:
                            # Escapar caminhos para Windows
                            safe_path = p.replace('\\', '/').replace("'", "'\\''")
                            f.write(f"file '{safe_path}'\n")
                    
                    cmd_simple = [
                        "ffmpeg", "-y",
                        "-f", "concat", "-safe", "0",
                        "-i", concat_list_path,
                        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                        "-c:a", "aac", "-b:a", "192k",
                        "-pix_fmt", "yuv420p",
                        output_path
                    ]
                    result = subprocess.run(
                        cmd_simple,
                        capture_output=True,
                        text=True,
                        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                    )
                    
                    if result.returncode != 0:
                        self.log(f"Erro ao concatenar vídeos: {result.stderr[-300:] if result.stderr else ''}", "ERROR")
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        return None
            
            # Limpar temp
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            # Embaralhar após uso
            self.log("Embaralhando backlog para próxima vez...", "INFO")
            self.shuffle_and_rename(all_videos, backlog_folder)
            
            self.log(f"Intro de backlog criada com sucesso: {output_path}", "OK")
            return output_path
            
        except Exception as e:
            self.log(f"Erro ao criar intro do backlog: {str(e)}", "ERROR")
            import traceback
            self.log(traceback.format_exc(), "ERROR")
            return None


# =============================================================================
# INTERFACE GRAFICA (CUSTOMTKINTER) - MODO LOTE
# =============================================================================
class FinalSlideshowApp(ctk.CTk):
    """Interface moderna para processamento em lote."""

    def __init__(self):
        super().__init__()

        self.title("RenderX v3.2 - Equipe Matrix")
        self.geometry("900x1000")
        self.minsize(800, 900)
        self.configure(fg_color=CORES["bg_dark"])

        # Grid responsivo (3 linhas: header, conteudo, footer)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)  # Conteudo expande
        self.grid_rowconfigure(2, weight=0)  # Footer fixo

        # Variaveis de controle
        self.config = self.load_config()
        self.log_queue = queue.Queue()
        self.cancel_requested = False

        # Sistema de lote
        self.batch_jobs = []
        self.batch_queue = queue.Queue()
        self.image_system = None
        self.workers = []
        self.jobs_lock = threading.Lock()

        # Variaveis CTk - Pastas do Lote
        self.batch_input_var = ctk.StringVar(value=self.config.get("batch_input_folder", ""))
        self.batch_output_var = ctk.StringVar(value=self.config.get("batch_output_folder", ""))
        self.batch_images_var = ctk.StringVar(value=self.config.get("batch_images_folder", ""))

        # Configuracoes de video
        self.resolution_var = ctk.StringVar(value=self.config.get("resolution", "720p"))
        self.zoom_mode_var = ctk.StringVar(value=self.config.get("zoom_mode", "zoom_in"))
        self.zoom_scale_var = ctk.DoubleVar(value=self.config.get("zoom_scale", 1.15))
        self.duration_var = ctk.DoubleVar(value=self.config.get("image_duration", 8.0))
        self.transition_var = ctk.DoubleVar(value=self.config.get("transition_duration", 1.0))
        self.images_per_video_var = ctk.IntVar(value=self.config.get("images_per_video", 50))
        self.video_bitrate_var = ctk.StringVar(value=self.config.get("video_bitrate", "4M"))

        # Paralelismo
        self.parallel_videos_var = ctk.IntVar(value=self.config.get("parallel_videos", 2))
        self.threads_per_video_var = ctk.IntVar(value=self.config.get("threads_per_video", 6))

        # Opcoes
        self.music_var = ctk.StringVar(value=self.config.get("music_path", ""))
        self.overlay_var = ctk.StringVar(value=self.config.get("overlay_path", ""))
        self.music_volume_var = ctk.DoubleVar(value=self.config.get("music_volume", 0.2))
        self.overlay_opacity_var = ctk.DoubleVar(value=self.config.get("overlay_opacity", 0.3))

        # Legendas
        self.use_subtitles_var = ctk.BooleanVar(value=self.config.get("use_subtitles", False))
        self.subtitle_method_var = ctk.StringVar(value=self.config.get("subtitle_method", "srt"))
        self.srt_path_var = ctk.StringVar(value=self.config.get("srt_path", ""))
        self.assemblyai_key_var = ctk.StringVar(value=self.config.get("assemblyai_key", ""))
        self.sub_font_var = ctk.StringVar(value=self.config.get("sub_font_name", "Arial"))
        self.sub_font_size_var = ctk.IntVar(value=self.config.get("sub_font_size", 48))
        self.sub_color_primary_var = ctk.StringVar(value=self.config.get("sub_color_primary", "#FFFFFF"))
        self.sub_color_outline_var = ctk.StringVar(value=self.config.get("sub_color_outline", "#000000"))
        self.sub_color_shadow_var = ctk.StringVar(value=self.config.get("sub_color_shadow", "#80000000"))
        self.sub_color_karaoke_var = ctk.StringVar(value=self.config.get("sub_color_karaoke", "#FFFF00"))
        self.sub_outline_size_var = ctk.IntVar(value=self.config.get("sub_outline_size", 2))
        self.sub_shadow_size_var = ctk.IntVar(value=self.config.get("sub_shadow_size", 2))
        self.sub_use_karaoke_var = ctk.BooleanVar(value=self.config.get("sub_use_karaoke", True))
        self.sub_position_var = ctk.StringVar(value=self.config.get("sub_position", "2"))

        # VSL (Video Sales Letter)
        self.use_vsl_var = ctk.BooleanVar(value=self.config.get("use_vsl", True))  # Padrão: True
        # Fallback para nome antigo vsl_fallback_language se vsl_language não existir
        vsl_lang_default = self.config.get("vsl_language", self.config.get("vsl_fallback_language", "portugues"))
        self.vsl_language_var = ctk.StringVar(value=vsl_lang_default)
        # Modo de inserção VSL: "keyword", "fixed" ou "range"
        self.vsl_insertion_mode_var = ctk.StringVar(value=self.config.get("vsl_insertion_mode", "keyword"))
        self.vsl_fixed_position_var = ctk.DoubleVar(value=self.config.get("vsl_fixed_position", 60.0))
        # Range aleatório para VSL (em minutos)
        self.vsl_range_start_var = ctk.StringVar(value=str(self.config.get("vsl_range_start_min", 1.0)))
        self.vsl_range_end_var = ctk.StringVar(value=str(self.config.get("vsl_range_end_min", 3.0)))
        # Backlog Videos
        self.use_backlog_videos_var = ctk.BooleanVar(value=self.config.get("use_backlog_videos", False))
        backlog_folder_default = self.config.get("backlog_folder", str(SCRIPT_DIR / "EFEITOS" / "BACKLOG_VIDEOS"))
        self.backlog_folder_var = ctk.StringVar(value=backlog_folder_default)
        self.backlog_status_var = ctk.StringVar(value="Verificando...")
        
        # Modo de Vídeo: "traditional", "srt", "single_image"
        # Converter valor antigo (bool) para novo formato (string)
        old_srt_value = self.config.get("use_srt_based_images", False)
        video_mode_default = self.config.get("video_mode", "traditional")
        if video_mode_default == "traditional" and old_srt_value:
            video_mode_default = "srt"
        self.video_mode_var = ctk.StringVar(value=video_mode_default)
        self.image_source_var = ctk.StringVar(value=self.config.get("image_source", "generate"))
        self.images_backlog_folder_var = ctk.StringVar(value=self.config.get("images_backlog_folder", ""))
        # Tokens Whisk agora são carregados do arquivo whisk_keys.json
        self.use_varied_animations_var = ctk.BooleanVar(value=self.config.get("use_varied_animations", True))
        self.pan_amount_var = ctk.DoubleVar(value=self.config.get("pan_amount", 0.2))
        
        # Seleção de efeitos de animação (15 efeitos disponíveis)
        # Mapeamento: A-O para os 15 efeitos Ken Burns
        self.EFFECT_NAMES = {
            "A": "Pan Esq→Dir",
            "B": "Pan Dir→Esq", 
            "C": "Pan Cima→Baixo",
            "D": "Pan Baixo→Cima",
            "E": "Pan Esq→Dir + Zoom In",
            "F": "Pan Dir→Esq + Zoom In",
            "G": "Pan Cima→Baixo + Zoom In",
            "H": "Pan Baixo→Cima + Zoom In",
            "I": "Pan Esq→Dir + Zoom Out",
            "J": "Pan Dir→Esq + Zoom Out",
            "K": "Pan Cima→Baixo + Zoom Out",
            "L": "Pan Baixo→Cima + Zoom Out",
            "M": "Rotação Leve",
            "N": "Zoom In (puro)",
            "O": "Zoom Out (puro)"
        }
        # Carregar efeitos habilitados do config (padrão: todos habilitados)
        default_effects = list(self.EFFECT_NAMES.keys())
        enabled_effects = self.config.get("enabled_effects", default_effects)
        self.effect_vars = {}
        for letter in self.EFFECT_NAMES.keys():
            self.effect_vars[letter] = ctk.BooleanVar(value=letter in enabled_effects)
        
        # Pipeline SRT v3.1
        self.subtitle_mode_var = ctk.StringVar(value=self.config.get("subtitle_mode", "full"))
        self.swap_every_n_cues_var = ctk.IntVar(value=self.config.get("swap_every_n_cues", 3))
        self.prompt_file_var = ctk.StringVar(value=self.config.get("prompt_file", ""))
        self.overlay_folder_var = ctk.StringVar(value=self.config.get("overlay_folder", ""))
        self.use_random_overlays_var = ctk.BooleanVar(value=self.config.get("use_random_overlays", False))
        self.whisk_workers_var = ctk.IntVar(value=self.config.get("whisk_parallel_workers", 0))
        
        # Backlog de Áudios
        self.audio_backlog_folder_var = ctk.StringVar(value=self.config.get("audio_backlog_folder", ""))
        self.use_audio_backlog_var = ctk.BooleanVar(value=self.config.get("use_audio_backlog", False))

        # Modo 1 Imagem (Pêndulo)
        self.pendulum_amplitude_var = ctk.DoubleVar(value=self.config.get("pendulum_amplitude", 1.6))
        self.pendulum_crop_ratio_var = ctk.DoubleVar(value=self.config.get("pendulum_crop_ratio", 1.0))
        self.pendulum_zoom_var = ctk.DoubleVar(value=self.config.get("pendulum_zoom", 2.0))
        self.pendulum_cell_duration_var = ctk.DoubleVar(value=self.config.get("pendulum_cell_duration", 10.0))
        # Chroma Key para overlay
        self.chroma_color_var = ctk.StringVar(value=self.config.get("chroma_color", "00b140"))
        self.chroma_similarity_var = ctk.DoubleVar(value=self.config.get("chroma_similarity", 0.2))
        self.chroma_blend_var = ctk.DoubleVar(value=self.config.get("chroma_blend", 0.1))

        # TTS API
        self.tts_provider_var = ctk.StringVar(value=self.config.get("tts_provider", "none"))
        self.darkvi_api_key_var = ctk.StringVar(value=self.config.get("darkvi_api_key", ""))
        self.talkify_api_key_var = ctk.StringVar(value=self.config.get("talkify_api_key", ""))
        self.tts_voice_id_var = ctk.StringVar(value=self.config.get("tts_voice_id", ""))
        self.tts_voice_name_var = ctk.StringVar(value=self.config.get("tts_voice_name", ""))
        self.tts_voice_combo_var = ctk.StringVar(value="")  # Variável separada para o combobox
        self.tts_enabled_var = ctk.BooleanVar(value=self.config.get("tts_enabled", False))
        self.generated_audio_folder_var = ctk.StringVar(value=self.config.get("generated_audio_folder", "AUDIOS_GERADOS"))
        self.tts_voices_list = []  # Lista de vozes disponíveis

        # Contadores do scan
        self.audio_count = 0
        self.images_count = 0
        self.txt_count = 0
        self.audio_to_generate_count = 0

        self.setup_ui()
        self.after(100, self.process_log_queue)

        # Protocolo de fechamento
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def _on_mousewheel(self, event):
        """Handler para scroll do mouse/trackpad."""
        if not hasattr(self, 'main_scroll'):
            return
        
        try:
            # Acessar o canvas interno do CTkScrollableFrame
            canvas = self.main_scroll._parent_canvas
            if canvas:
                # macOS usa delta diferente
                if sys.platform == "darwin":
                    # macOS: delta pode ser positivo ou negativo
                    delta = -event.delta
                else:
                    # Windows/Linux: delta é múltiplo de 120
                    delta = -1 if event.delta > 0 else 1
                
                # Scroll no canvas
                canvas.yview_scroll(int(delta), "units")
        except Exception:
            # Fallback: tentar método alternativo
            try:
                if sys.platform == "darwin":
                    delta = -event.delta
                else:
                    delta = -1 if event.delta > 0 else 1
                self.main_scroll._scrollbar.set(self.main_scroll._scrollbar.get()[0] + delta * 0.1)
            except:
                pass

    def on_closing(self):
        """Trata fechamento da janela."""
        if self.workers and not self.cancel_requested:
            if messagebox.askokcancel("Sair", "Processamento em andamento. Deseja sair?"):
                self.cancel_requested = True
                self.after(1000, self.destroy)
        else:
            self.destroy()

    def load_config(self):
        """Carrega configuracoes do arquivo."""
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    return {**DEFAULT_CONFIG, **json.load(f)}
        except:
            pass
        return DEFAULT_CONFIG.copy()

    def save_config(self):
        """Salva configuracoes no arquivo."""
        config = {
            # Pastas do lote
            "batch_input_folder": self.batch_input_var.get(),
            "batch_output_folder": self.batch_output_var.get(),
            "batch_images_folder": self.batch_images_var.get(),
            # Configuracoes de video
            "resolution": self.resolution_var.get(),
            "zoom_mode": self.zoom_mode_var.get(),
            "zoom_scale": self.zoom_scale_var.get(),
            "image_duration": self.duration_var.get(),
            "transition_duration": self.transition_var.get(),
            "images_per_video": int(self.images_per_video_var.get()),
            "video_bitrate": self.video_bitrate_var.get(),
            # Paralelismo
            "parallel_videos": int(self.parallel_videos_var.get()),
            "threads_per_video": int(self.threads_per_video_var.get()),
            # Opcoes
            "music_path": self.music_var.get(),
            "music_volume": self.music_volume_var.get(),
            "overlay_path": self.overlay_var.get(),
            "overlay_opacity": self.overlay_opacity_var.get(),
            # Legendas
            "use_subtitles": self.use_subtitles_var.get(),
            "subtitle_method": self.subtitle_method_var.get(),
            "assemblyai_key": self.assemblyai_key_var.get(),
            "sub_font_name": self.sub_font_var.get(),
            "sub_font_size": int(self.sub_font_size_var.get()),
            "sub_color_primary": self.sub_color_primary_var.get(),
            "sub_color_outline": self.sub_color_outline_var.get(),
            "sub_color_shadow": self.sub_color_shadow_var.get(),
            "sub_color_karaoke": self.sub_color_karaoke_var.get(),
            "sub_outline_size": int(self.sub_outline_size_var.get()),
            "sub_shadow_size": int(self.sub_shadow_size_var.get()),
            "sub_use_karaoke": self.sub_use_karaoke_var.get(),
            "sub_position": self.sub_position_var.get(),
            # TTS API
            "tts_provider": self.tts_provider_var.get(),
            "darkvi_api_key": self.darkvi_api_key_var.get(),
            "talkify_api_key": self.talkify_api_key_var.get(),
            "tts_voice_id": self.tts_voice_id_var.get(),
            "tts_voice_name": self.tts_voice_name_var.get(),
            "tts_enabled": self.tts_enabled_var.get(),
            "generated_audio_folder": self.generated_audio_folder_var.get(),
            # VSL
            "use_vsl": self.use_vsl_var.get(),
            "vsl_folder": self.config.get("vsl_folder", str(SCRIPT_DIR / "EFEITOS" / "VSLs")),
            "vsl_keywords_file": self.config.get("vsl_keywords_file", str(SCRIPT_DIR / "vsl_keywords.json")),
            "vsl_language": self.vsl_language_var.get() if hasattr(self, 'vsl_language_var') else "portugues",
            "selected_vsl": self.selected_vsl_var.get() if hasattr(self, 'selected_vsl_var') else "",
            "vsl_insertion_mode": self.vsl_insertion_mode_var.get() if hasattr(self, 'vsl_insertion_mode_var') else "keyword",
            "vsl_fixed_position": self.vsl_fixed_position_var.get() if hasattr(self, 'vsl_fixed_position_var') else 60.0,
            "vsl_range_start_min": float(self.vsl_range_start_var.get()) if hasattr(self, 'vsl_range_start_var') else 1.0,
            "vsl_range_end_min": float(self.vsl_range_end_var.get()) if hasattr(self, 'vsl_range_end_var') else 3.0,
            # Backlog Videos
            "use_backlog_videos": self.use_backlog_videos_var.get(),
            "backlog_folder": self.backlog_folder_var.get(),
            "backlog_video_count": self.config.get("backlog_video_count", 6),
            "backlog_audio_volume": self.config.get("backlog_audio_volume", 0.25),
            "backlog_transition_duration": self.config.get("backlog_transition_duration", 0.5),
            "backlog_fade_out_duration": self.config.get("backlog_fade_out_duration", 1.0),
            # Modo de Vídeo
            "video_mode": self.video_mode_var.get(),
            "use_srt_based_images": self.video_mode_var.get() == "srt",  # Compatibilidade
            "image_source": self.image_source_var.get(),
            "images_backlog_folder": self.images_backlog_folder_var.get(),
            "whisk_api_tokens": get_enabled_tokens(),  # Carregado do whisk_keys.json
            "whisk_parallel_workers": self.whisk_workers_var.get(),  # Workers paralelos (0 = auto)
            "selected_prompt_id": self.get_selected_prompt_id(),  # Prompt de imagem selecionado
            "use_varied_animations": self.use_varied_animations_var.get(),
            "enabled_effects": self.get_enabled_effects(),  # Efeitos de animação selecionados
            "pan_amount": self.pan_amount_var.get(),
            # Pipeline SRT v3.1
            "subtitle_mode": self.subtitle_mode_var.get(),
            "swap_every_n_cues": self.swap_every_n_cues_var.get(),
            "prompt_file": self.prompt_file_var.get(),
            "overlay_folder": self.overlay_folder_var.get(),
            "use_random_overlays": self.use_random_overlays_var.get(),
            # Backlog de Áudios
            "audio_backlog_folder": self.audio_backlog_folder_var.get(),
            "use_audio_backlog": self.use_audio_backlog_var.get(),
            # Modo 1 Imagem (Pêndulo)
            "pendulum_amplitude": self.pendulum_amplitude_var.get(),
            "pendulum_crop_ratio": self.pendulum_crop_ratio_var.get(),
            "pendulum_zoom": self.pendulum_zoom_var.get(),
            "pendulum_cell_duration": self.pendulum_cell_duration_var.get(),
            "chroma_color": self.chroma_color_var.get(),
            "chroma_similarity": self.chroma_similarity_var.get(),
            "chroma_blend": self.chroma_blend_var.get()
        }
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
        except:
            pass

    def setup_ui(self):
        """Configura a interface para modo lote."""
        # ===== HEADER =====
        self.create_header()

        # ===== SCROLLABLE MAIN =====
        self.main_scroll = ctk.CTkScrollableFrame(
            self, fg_color="transparent"
        )
        self.main_scroll.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 10))
        self.main_scroll.grid_columnconfigure(0, weight=1)
        
        # Habilitar scroll com mouse wheel
        # CTkScrollableFrame já tem scroll nativo, mas vamos garantir que funcione
        def setup_scroll_binding():
            try:
                # Bind no canvas interno do scrollable frame
                canvas = self.main_scroll._parent_canvas
                if canvas:
                    canvas.bind("<MouseWheel>", self._on_mousewheel)
                    canvas.bind("<Button-4>", self._on_mousewheel)  # Linux
                    canvas.bind("<Button-5>", self._on_mousewheel)  # Linux
                    # macOS trackpad
                    canvas.bind("<Shift-MouseWheel>", self._on_mousewheel)
            except:
                pass
            # Bind também no frame principal como fallback
            self.main_scroll.bind("<MouseWheel>", self._on_mousewheel)
            self.main_scroll.bind("<Button-4>", self._on_mousewheel)
            self.main_scroll.bind("<Button-5>", self._on_mousewheel)
        
        # Configurar scroll após um pequeno delay para garantir que o canvas esteja criado
        self.after(200, setup_scroll_binding)

        # ===== SECOES - FLUXO OTIMIZADO v3.1 =====
        
        # 1. ENTRADA - Onde estão os materiais
        self.create_folders_section()
        
        # 2. MODO DE VÍDEO - Tradicional ou Pipeline SRT
        self.create_video_mode_section()
        
        # 3. ÁUDIO - Música de fundo e TTS
        self.create_audio_section()
        
        # 4. LEGENDAS - Configurações de texto
        self.create_subtitles_section()
        
        # 5. EFEITOS VISUAIS - Overlay, VSL, Vídeos Intro
        self.create_effects_section()
        
        # 6. PERFORMANCE - Paralelismo
        self.create_parallelism_section()
        
        # 7. AÇÕES
        self.create_buttons_section()
        self.create_progress_section()
        self.create_log_section()
        
        # 7. Sobre/Versão
        self.create_about_section()
        
        # 8. Footer com Status
        self.create_footer()

    def create_header(self):
        """Cria o cabecalho premium."""
        header = ctk.CTkFrame(self, fg_color=CORES["bg_card"], corner_radius=0, height=90)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)
        header.grid_columnconfigure(1, weight=1)

        # Logo/Icone
        logo_frame = ctk.CTkFrame(header, fg_color="transparent", width=80)
        logo_frame.grid(row=0, column=0, rowspan=2, padx=(20, 10), pady=15)
        
        logo_label = ctk.CTkLabel(
            logo_frame,
            text="🎬",
            font=ctk.CTkFont(size=42),
            text_color=CORES["accent"]
        )
        logo_label.pack()

        # Titulo principal
        title_frame = ctk.CTkFrame(header, fg_color="transparent")
        title_frame.grid(row=0, column=1, sticky="sw", pady=(20, 0))
        
        title = ctk.CTkLabel(
            title_frame,
            text="RenderX",
            font=ctk.CTkFont(family="Segoe UI", size=28, weight="bold"),
            text_color=CORES["text"]
        )
        title.pack(side="left")
        
        # Badge de versao
        version_badge = ctk.CTkLabel(
            title_frame,
            text=" v3.2 ",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=CORES["bg_dark"],
            fg_color=CORES["accent"],
            corner_radius=4
        )
        version_badge.pack(side="left", padx=(10, 0), pady=(5, 0))

        # Subtitulo
        subtitle = ctk.CTkLabel(
            header,
            text="Video Slideshow Engine  •  Equipe Matrix",
            font=ctk.CTkFont(size=12),
            text_color=CORES["text_dim"]
        )
        subtitle.grid(row=1, column=1, sticky="nw", pady=(2, 0))

        # Separador inferior
        separator = ctk.CTkFrame(header, fg_color=CORES["border"], height=1)
        separator.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(0, 0))

    def create_folders_section(self):
        """Secao de pastas do lote."""
        section = ctk.CTkFrame(self.main_scroll, fg_color=CORES["bg_section"], corner_radius=12, border_width=1, border_color=CORES["border"])
        section.pack(fill="x", pady=8, padx=10)
        section.grid_columnconfigure(1, weight=1)

        # Header com título e botão de criar pastas
        header_frame = ctk.CTkFrame(section, fg_color="transparent")
        header_frame.grid(row=0, column=0, columnspan=3, sticky="ew", padx=15, pady=(15, 8))
        header_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header_frame, text="📁  PASTAS DO LOTE",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=CORES["text"]
        ).grid(row=0, column=0, sticky="w")

        # Botão para criar pastas do lote automaticamente
        ctk.CTkButton(
            header_frame, text="📂 Criar Pastas do Lote", width=160,
            fg_color=CORES["warning"], hover_color="#9B59B6",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self.create_batch_folders
        ).grid(row=0, column=1, sticky="e")

        # Pasta dos Materiais (Entrada)
        ctk.CTkLabel(section, text="Pasta dos Materiais:", text_color=CORES["text"]).grid(
            row=1, column=0, sticky="w", padx=(15, 10), pady=5)
        ctk.CTkEntry(
            section, textvariable=self.batch_input_var, width=350,
            fg_color=CORES["bg_input"], border_color=CORES["accent"]
        ).grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        ctk.CTkButton(
            section, text="...", width=40,
            fg_color=CORES["accent"], hover_color=CORES["accent_hover"],
            command=lambda: self.select_batch_folder("input")
        ).grid(row=1, column=2, padx=(5, 15), pady=5)

        # Pasta de Saida
        ctk.CTkLabel(section, text="Pasta de Saida:", text_color=CORES["text"]).grid(
            row=2, column=0, sticky="w", padx=(15, 10), pady=5)
        ctk.CTkEntry(
            section, textvariable=self.batch_output_var, width=350,
            fg_color=CORES["bg_input"], border_color=CORES["success"]
        ).grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        ctk.CTkButton(
            section, text="...", width=40,
            fg_color=CORES["success"], hover_color="#3db892",
            command=lambda: self.select_batch_folder("output")
        ).grid(row=2, column=2, padx=(5, 15), pady=5)

        # Banco de Imagens
        ctk.CTkLabel(section, text="Banco de Imagens:", text_color=CORES["text"]).grid(
            row=3, column=0, sticky="w", padx=(15, 10), pady=5)
        ctk.CTkEntry(
            section, textvariable=self.batch_images_var, width=350,
            fg_color=CORES["bg_input"], border_color=CORES["info"]
        ).grid(row=3, column=1, sticky="ew", padx=5, pady=5)
        ctk.CTkButton(
            section, text="...", width=40,
            fg_color=CORES["info"], hover_color="#CC7A00",
            command=lambda: self.select_batch_folder("images")
        ).grid(row=3, column=2, padx=(5, 15), pady=5)

        # Status e Botao Escanear
        status_frame = ctk.CTkFrame(section, fg_color="transparent")
        status_frame.grid(row=4, column=0, columnspan=3, sticky="ew", padx=15, pady=(5, 15))

        self.scan_status_label = ctk.CTkLabel(
            status_frame,
            text="Clique em 'Escanear' para verificar pastas",
            text_color=CORES["text_dim"],
            font=ctk.CTkFont(size=11)
        )
        self.scan_status_label.pack(side="left")

        ctk.CTkButton(
            status_frame, text="Escanear Pastas", width=120,
            fg_color=CORES["accent"], hover_color=CORES["accent_hover"],
            command=self.scan_folders
        ).pack(side="right")

    def on_voice_combo_changed(self, choice):
        """Callback quando o usuário seleciona uma voz no combobox."""
        if not choice or choice == "Nenhuma voz carregada":
            return
        
        # Extrair ID da seleção (formato: "Nome (id)")
        if "(" in choice and ")" in choice:
            try:
                # Extrair tudo entre parênteses
                start = choice.rfind("(")
                end = choice.rfind(")")
                if start != -1 and end != -1 and end > start:
                    voice_id = choice[start+1:end].strip()
                    # Encontrar voz correspondente na lista
                    for v in self.tts_voices_list:
                        if v['id'] == voice_id:
                            self.tts_voice_id_var.set(v['id'])
                            self.tts_voice_name_var.set(v['name'])
                            self.log(f"Voz selecionada: {v['name']} ({v['id']})", "INFO")
                            return
            except Exception as e:
                self.log(f"Erro ao processar seleção de voz: {str(e)}", "ERROR")
        
        # Se não encontrou, tentar atualizar do valor atual do combobox
        self.update_voice_id_from_combo()
    
    def update_voice_id_from_combo(self):
        """Atualiza voice_id_var a partir do valor atual do combobox."""
        current_value = self.tts_voice_combo_var.get()
        if not current_value or current_value == "Nenhuma voz carregada":
            return
        
        self.log(f"Tentando extrair ID de: '{current_value}'", "DEBUG")
        
        # Extrair ID da seleção (formato: "Nome (id)")
        if "(" in current_value and ")" in current_value:
            try:
                start = current_value.rfind("(")
                end = current_value.rfind(")")
                if start != -1 and end != -1 and end > start:
                    voice_id = current_value[start+1:end].strip()
                    self.log(f"ID extraído: '{voice_id}'", "DEBUG")
                    
                    if not voice_id:
                        self.log("ID vazio após extração!", "ERROR")
                        return
                    
                    # Encontrar voz correspondente na lista
                    for v in self.tts_voices_list:
                        v_id = v.get('id', '').strip()
                        if v_id == voice_id:
                            self.tts_voice_id_var.set(v_id)
                            self.tts_voice_name_var.set(v.get('name', '').strip())
                            self.log(f"Voice ID atualizado: {v_id}", "DEBUG")
                            return
                    
                    # Se não encontrou, tentar buscar pelo nome
                    voice_name = current_value[:start].strip()
                    for v in self.tts_voices_list:
                        if v.get('name', '').strip() == voice_name:
                            v_id = v.get('id', '').strip()
                            if v_id:
                                self.tts_voice_id_var.set(v_id)
                                self.tts_voice_name_var.set(v.get('name', '').strip())
                                self.log(f"Voice ID encontrado pelo nome: {v_id}", "DEBUG")
                                return
            except Exception as e:
                self.log(f"Erro ao extrair ID: {str(e)}", "ERROR")

    # Seção TTS antiga removida - agora em create_audio_section

    def load_tts_voices(self):
        """Carrega vozes disponíveis da API selecionada."""
        provider = self.tts_provider_var.get()
        if provider == "none":
            messagebox.showwarning("Aviso", "Selecione um provider primeiro!")
            return

        api_key = self.darkvi_api_key_var.get() if provider == "darkvi" else self.talkify_api_key_var.get()
        if not api_key or not api_key.strip():
            messagebox.showerror("Erro", f"Configure a API key do {provider.upper()} primeiro!")
            return

        self.log(f"Carregando vozes do {provider.upper()}...", "INFO")
        
        try:
            # Tentar importar httpx primeiro para verificar se está disponível
            try:
                import httpx
            except ImportError:
                messagebox.showerror(
                    "Erro", 
                    "Módulo 'httpx' não encontrado!\n\n"
                    "Execute no terminal:\n"
                    "cd /Users/davi/Desktop/FERRAMENTAS/RENDERX\n"
                    "source venv/bin/activate\n"
                    "pip install httpx\n\n"
                    "Ou execute a aplicação com:\n"
                    "source venv/bin/activate && python iniciar_render.py"
                )
                return
            
            from tts_integration import TTSGenerator
            # Criar callback de log compatível
            def log_callback(msg, level="INFO"):
                self.log(msg, level)
            generator = TTSGenerator(provider, api_key, log_callback)
            voices = generator.list_voices()

            if not voices:
                messagebox.showwarning("Aviso", f"Nenhuma voz encontrada no {provider.upper()}!")
                return

            self.tts_voices_list = voices
            # Log das vozes recebidas para debug
            self.log(f"Vozes recebidas da API: {voices[:2] if len(voices) > 0 else 'nenhuma'}", "DEBUG")
            
            # Formatar vozes para o combobox, garantindo que ID não esteja vazio
            voice_values = []
            for v in voices:
                voice_id = v.get('id', '').strip() if v.get('id') else ''
                voice_name = v.get('name', '').strip() if v.get('name') else ''
                
                # Log para debug
                self.log(f"Processando voz: id='{voice_id}', name='{voice_name}'", "DEBUG")
                
                if not voice_id:
                    self.log(f"Aviso: Voz '{voice_name}' sem ID, pulando...", "WARNING")
                    continue
                if not voice_name:
                    voice_name = voice_id
                voice_values.append(f"{voice_name} ({voice_id})")
            
            if not voice_values:
                messagebox.showerror("Erro", "Nenhuma voz válida encontrada (sem ID)!")
                return
            
            # Atualizar combobox
            self.tts_voice_combo.configure(values=voice_values)
            
            # Log para debug
            self.log(f"Vozes formatadas para combobox: {voice_values[:3]}...", "DEBUG")
            
            # Selecionar primeira voz se não houver seleção ou se a voz atual não está na lista
            current_voice_id = self.tts_voice_id_var.get()
            if not current_voice_id or current_voice_id not in [v['id'] for v in voices]:
                if voices:
                    first_voice = voices[0]
                    self.tts_voice_id_var.set(first_voice['id'])
                    self.tts_voice_name_var.set(first_voice['name'])
                    first_voice_display = f"{first_voice['name']} ({first_voice['id']})"
                    self.tts_voice_combo_var.set(first_voice_display)
                    # Forçar atualização do combobox
                    self.tts_voice_combo.set(first_voice_display)
            else:
                # Manter a voz atual se ela estiver na lista
                for v in voices:
                    if v['id'] == current_voice_id:
                        current_display = f"{v['name']} ({v['id']})"
                        self.tts_voice_combo_var.set(current_display)
                        self.tts_voice_combo.set(current_display)
                        break
            
            self.log(f"{len(voices)} vozes carregadas do {provider.upper()}", "OK")
            messagebox.showinfo("Sucesso", f"{len(voices)} vozes carregadas!")

        except Exception as e:
            self.log(f"Erro ao carregar vozes: {str(e)}", "ERROR")
            messagebox.showerror("Erro", f"Erro ao carregar vozes:\n{str(e)}")

    def create_video_mode_section(self):
        """Seção unificada de modo de vídeo - Tradicional ou Pipeline SRT."""
        section = ctk.CTkFrame(self.main_scroll, fg_color=CORES["bg_section"], corner_radius=12, border_width=1, border_color=CORES["border"])
        section.pack(fill="x", pady=8, padx=10)

        ctk.CTkLabel(
            section, text="🎬  MODO DE VÍDEO",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=CORES["text"]
        ).pack(anchor="w", padx=15, pady=(15, 8))

        # ===== SELEÇÃO DE MODO =====
        mode_frame = ctk.CTkFrame(section, fg_color="transparent")
        mode_frame.pack(fill="x", padx=15, pady=5)

        ctk.CTkRadioButton(
            mode_frame, text="Modo Tradicional (múltiplas imagens do banco)",
            variable=self.video_mode_var, value="traditional",
            fg_color=CORES["accent"], hover_color=CORES["accent_hover"],
            command=self.toggle_video_mode
        ).pack(anchor="w")
        
        ctk.CTkRadioButton(
            mode_frame, text="Pipeline SRT (imagens seguem narrativa do áudio)",
            variable=self.video_mode_var, value="srt",
            fg_color=CORES["accent"], hover_color=CORES["accent_hover"],
            command=self.toggle_video_mode
        ).pack(anchor="w", pady=(5, 0))
        
        ctk.CTkRadioButton(
            mode_frame, text="Modo 1 Imagem (loop seamless com pêndulo)",
            variable=self.video_mode_var, value="single_image",
            fg_color=CORES["accent"], hover_color=CORES["accent_hover"],
            command=self.toggle_video_mode
        ).pack(anchor="w", pady=(5, 0))

        # ===== CONFIGURAÇÕES COMUNS =====
        common_frame = ctk.CTkFrame(section, fg_color="transparent")
        common_frame.pack(fill="x", padx=15, pady=10)
        
        # Resolução
        res_frame = ctk.CTkFrame(common_frame, fg_color="transparent")
        res_frame.pack(fill="x")
        ctk.CTkLabel(res_frame, text="Resolução:", text_color=CORES["text"]).pack(side="left")
        ctk.CTkRadioButton(
            res_frame, text="720p", variable=self.resolution_var, value="720p",
            fg_color=CORES["accent"], hover_color=CORES["accent_hover"]
        ).pack(side="left", padx=(10, 5))
        ctk.CTkRadioButton(
            res_frame, text="1080p", variable=self.resolution_var, value="1080p",
            fg_color=CORES["accent"], hover_color=CORES["accent_hover"]
        ).pack(side="left")

        # Bitrate do vídeo
        bitrate_frame = ctk.CTkFrame(common_frame, fg_color="transparent")
        bitrate_frame.pack(fill="x", pady=(10, 0))
        ctk.CTkLabel(bitrate_frame, text="Bitrate:", text_color=CORES["text"]).pack(side="left")
        ctk.CTkComboBox(
            bitrate_frame, 
            variable=self.video_bitrate_var,
            values=["2M", "4M", "6M", "8M", "auto"],
            width=100,
            fg_color=CORES["bg_input"],
            border_color=CORES["accent"],
            button_color=CORES["accent"],
            button_hover_color=CORES["accent_hover"],
            dropdown_fg_color=CORES["bg_input"]
        ).pack(side="left", padx=(10, 5))
        ctk.CTkLabel(
            bitrate_frame, 
            text="(2M=leve, 4M=equilibrado, 8M=alta qualidade, auto=CRF)",
            text_color=CORES["text_dim"], 
            font=ctk.CTkFont(size=10)
        ).pack(side="left", padx=(5, 0))

        # Animações variadas (aplica a AMBOS os modos)
        anim_frame = ctk.CTkFrame(common_frame, fg_color="transparent")
        anim_frame.pack(fill="x", pady=(10, 0))
        ctk.CTkCheckBox(
            anim_frame, text="Animações variadas (sem repetir em sequência)",
            variable=self.use_varied_animations_var,
            fg_color=CORES["accent"], hover_color=CORES["accent_hover"],
            text_color=CORES["text"],
            command=self.toggle_zoom_options
        ).pack(side="left")
        
        # Botão para abrir seleção de efeitos
        ctk.CTkButton(
            anim_frame, text="Selecionar Efeitos", width=120,
            fg_color=CORES["accent"], hover_color=CORES["accent_hover"],
            command=self.open_effects_selector
        ).pack(side="left", padx=(15, 0))
        
        # Label mostrando quantos efeitos estão selecionados
        self.effects_count_label = ctk.CTkLabel(
            anim_frame, text=f"({sum(v.get() for v in self.effect_vars.values())}/15)",
            text_color=CORES["text_dim"], font=ctk.CTkFont(size=10)
        )
        self.effects_count_label.pack(side="left", padx=(5, 0))
        
        ctk.CTkLabel(
            common_frame,
            text="Selecione quais efeitos entrarão na randomização (mínimo 2)",
            text_color=CORES["text_dim"], font=ctk.CTkFont(size=10)
        ).pack(anchor="w", pady=(2, 0))

        # ===== PAINEL MODO TRADICIONAL =====
        self.traditional_mode_frame = ctk.CTkFrame(section, fg_color=CORES["bg_dark"], corner_radius=8)
        
        ctk.CTkLabel(
            self.traditional_mode_frame, text="Opções do Modo Tradicional",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=CORES["text_dim"]
        ).pack(anchor="w", padx=10, pady=(10, 5))

        # Efeito zoom fixo (só aparece se NÃO usar animações variadas)
        self.zoom_options_frame = ctk.CTkFrame(self.traditional_mode_frame, fg_color="transparent")
        zoom_row = ctk.CTkFrame(self.zoom_options_frame, fg_color="transparent")
        zoom_row.pack(fill="x")
        ctk.CTkLabel(zoom_row, text="Efeito:", text_color=CORES["text"]).pack(side="left")
        ctk.CTkRadioButton(
            zoom_row, text="Zoom In", variable=self.zoom_mode_var, value="zoom_in",
            fg_color=CORES["accent"], hover_color=CORES["accent_hover"]
        ).pack(side="left", padx=(10, 5))
        ctk.CTkRadioButton(
            zoom_row, text="Zoom Out", variable=self.zoom_mode_var, value="zoom_out",
            fg_color=CORES["accent"], hover_color=CORES["accent_hover"]
        ).pack(side="left")
        self.create_slider(self.zoom_options_frame, "Escala Zoom:", self.zoom_scale_var, 1.05, 1.50, "x")

        # Sliders tradicionais
        self.create_slider(self.traditional_mode_frame, "Duração/Imagem:", self.duration_var, 3, 30, "s")
        self.create_slider(self.traditional_mode_frame, "Transição:", self.transition_var, 0.5, 5, "s")
        self.create_slider(self.traditional_mode_frame, "Imagens/Vídeo:", self.images_per_video_var, 10, 100, "", is_int=True)
        
        # Mostrar/ocultar opções de zoom conforme animações variadas
        self.toggle_zoom_options()

        # ===== PAINEL PIPELINE SRT =====
        self.srt_mode_frame = ctk.CTkFrame(section, fg_color=CORES["bg_dark"], corner_radius=8)
        
        ctk.CTkLabel(
            self.srt_mode_frame, text="Opções do Pipeline SRT",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=CORES["text_dim"]
        ).pack(anchor="w", padx=10, pady=(10, 5))

        # Frases por imagem (slider)
        cues_frame = ctk.CTkFrame(self.srt_mode_frame, fg_color="transparent")
        cues_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(cues_frame, text="Cues por imagem:", text_color=CORES["text"]).pack(side="left")
        self.cues_value_label = ctk.CTkLabel(cues_frame, text=str(self.swap_every_n_cues_var.get()), 
                                             text_color=CORES["accent"], width=30)
        self.cues_value_label.pack(side="right", padx=5)
        ctk.CTkSlider(
            cues_frame, from_=1, to=100, number_of_steps=99,
            variable=self.swap_every_n_cues_var, width=150,
            progress_color=CORES["accent"], button_color=CORES["accent"],
            command=lambda v: self.cues_value_label.configure(text=str(int(v)))
        ).pack(side="right", padx=10)

        # Fonte das imagens
        source_frame = ctk.CTkFrame(self.srt_mode_frame, fg_color="transparent")
        source_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(source_frame, text="Imagens:", text_color=CORES["text"]).pack(side="left")
        ctk.CTkRadioButton(
            source_frame, text="Gerar (API Whisk)", variable=self.image_source_var,
            value="generate", fg_color=CORES["accent"], hover_color=CORES["accent_hover"],
            command=self.toggle_srt_source
        ).pack(side="left", padx=10)
        ctk.CTkRadioButton(
            source_frame, text="Backlog", variable=self.image_source_var,
            value="backlog", fg_color=CORES["accent"], hover_color=CORES["accent_hover"],
            command=self.toggle_srt_source
        ).pack(side="left")

        # Campo: Pasta de imagens backlog
        self.srt_backlog_frame = ctk.CTkFrame(self.srt_mode_frame, fg_color="transparent")
        self.srt_backlog_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(self.srt_backlog_frame, text="Pasta Backlog:", text_color=CORES["text"]).grid(
            row=0, column=0, sticky="w", padx=(10, 5), pady=5)
        ctk.CTkEntry(self.srt_backlog_frame, textvariable=self.images_backlog_folder_var, 
                    fg_color=CORES["bg_input"], border_color=CORES["accent"]).grid(
            row=0, column=1, sticky="ew", padx=5, pady=5)
        ctk.CTkButton(self.srt_backlog_frame, text="...", width=40, fg_color=CORES["accent"],
                     command=self.select_images_backlog_folder).grid(row=0, column=2, padx=5, pady=5)

        # Campo: Tokens Whisk (carregado do arquivo whisk_keys.json)
        self.whisk_frame = ctk.CTkFrame(self.srt_mode_frame, fg_color="transparent")
        self.whisk_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(self.whisk_frame, text="Tokens Whisk:", text_color=CORES["text"]).grid(
            row=0, column=0, sticky="w", padx=(10, 5), pady=5)
        
        # Label que mostra resumo dos tokens
        self.whisk_tokens_summary_var = ctk.StringVar(value=get_tokens_summary())
        self.whisk_tokens_label = ctk.CTkLabel(
            self.whisk_frame, 
            textvariable=self.whisk_tokens_summary_var,
            text_color=CORES["accent"],
            font=ctk.CTkFont(weight="bold")
        )
        self.whisk_tokens_label.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        # Botão para editar arquivo
        ctk.CTkButton(
            self.whisk_frame, 
            text="Editar Keys", 
            width=90, 
            fg_color=CORES["accent"],
            hover_color=CORES["accent_hover"],
            command=self.open_whisk_keys_file
        ).grid(row=0, column=2, padx=5, pady=5)
        
        # Botão para atualizar
        ctk.CTkButton(
            self.whisk_frame, 
            text="↻", 
            width=30, 
            fg_color=CORES["bg_card"],
            hover_color=CORES["bg_hover"],
            command=self.refresh_whisk_tokens
        ).grid(row=0, column=3, padx=(0, 5), pady=5)
        
        # Botão para capturar novo token automaticamente
        self.capture_token_btn = ctk.CTkButton(
            self.whisk_frame, 
            text="+ Novo Token", 
            width=100, 
            fg_color=CORES["success"],
            hover_color="#2d8a4e",
            command=self.capture_whisk_token_auto
        )
        self.capture_token_btn.grid(row=0, column=4, padx=5, pady=5)

        # Campo: Workers Paralelos para geração de imagens
        workers_frame = ctk.CTkFrame(self.srt_mode_frame, fg_color="transparent")
        workers_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(workers_frame, text="Workers Paralelos:", text_color=CORES["text"]).pack(side="left")
        
        # Label mostrando valor atual (0 = automático)
        def get_workers_label(val):
            v = int(val)
            return "Auto" if v == 0 else str(v)
        
        self.whisk_workers_value_label = ctk.CTkLabel(
            workers_frame, 
            text=get_workers_label(self.whisk_workers_var.get()), 
            text_color=CORES["accent"], 
            width=40
        )
        self.whisk_workers_value_label.pack(side="right", padx=5)
        
        ctk.CTkSlider(
            workers_frame, from_=0, to=20, number_of_steps=20,
            variable=self.whisk_workers_var, width=150,
            progress_color=CORES["accent"], button_color=CORES["accent"],
            command=lambda v: self.whisk_workers_value_label.configure(text=get_workers_label(v))
        ).pack(side="right", padx=10)
        
        ctk.CTkLabel(
            workers_frame, 
            text="(0 = nº de tokens)", 
            text_color=CORES["text_dim"],
            font=ctk.CTkFont(size=10)
        ).pack(side="right", padx=(0, 5))

        # Campo: Seletor de Prompt de Imagem (carregado de image_prompts.json)
        self.prompt_selector_frame = ctk.CTkFrame(self.srt_mode_frame, fg_color="transparent")
        self.prompt_selector_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(self.prompt_selector_frame, text="Estilo do Prompt:", text_color=CORES["text"]).grid(
            row=0, column=0, sticky="w", padx=(10, 5), pady=5)
        
        # Obter lista de prompts e prompt padrão
        prompt_names = get_prompt_names()
        default_prompt_id = get_default_prompt_id()
        default_prompt = get_prompt_by_id(default_prompt_id)
        default_name = default_prompt.get("name", prompt_names[0] if prompt_names else "")
        
        self.selected_prompt_var = ctk.StringVar(value=default_name)
        self.prompt_combobox = ctk.CTkComboBox(
            self.prompt_selector_frame,
            variable=self.selected_prompt_var,
            values=prompt_names if prompt_names else ["Nenhum prompt configurado"],
            fg_color=CORES["bg_input"],
            border_color=CORES["accent"],
            button_color=CORES["accent"],
            button_hover_color=CORES["accent_hover"],
            dropdown_fg_color=CORES["bg_card"],
            dropdown_hover_color=CORES["bg_hover"],
            width=250,
            command=self.on_prompt_selected
        )
        self.prompt_combobox.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        # Botão para editar prompts
        ctk.CTkButton(
            self.prompt_selector_frame, 
            text="Editar Prompts", 
            width=100, 
            fg_color=CORES["accent"],
            hover_color=CORES["accent_hover"],
            command=self.open_image_prompts_file
        ).grid(row=0, column=2, padx=5, pady=5)
        
        # Botão para atualizar lista
        ctk.CTkButton(
            self.prompt_selector_frame, 
            text="↻", 
            width=30, 
            fg_color=CORES["bg_card"],
            hover_color=CORES["bg_hover"],
            command=self.refresh_prompt_list
        ).grid(row=0, column=3, padx=(0, 5), pady=5)
        
        # Label de descrição do prompt selecionado
        self.prompt_description_var = ctk.StringVar(value="")
        self.update_prompt_description()
        ctk.CTkLabel(
            self.prompt_selector_frame, 
            textvariable=self.prompt_description_var,
            text_color=CORES["text_dim"],
            font=ctk.CTkFont(size=11, slant="italic")
        ).grid(row=1, column=0, columnspan=4, sticky="w", padx=(10, 5), pady=(0, 5))

        # Arquivo de Prompt (só para geração)
        self.prompt_frame_srt = ctk.CTkFrame(self.srt_mode_frame, fg_color="transparent")
        self.prompt_frame_srt.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(self.prompt_frame_srt, text="Prompt Externo:", text_color=CORES["text"]).grid(
            row=0, column=0, sticky="w", padx=(10, 5), pady=5)
        ctk.CTkEntry(self.prompt_frame_srt, textvariable=self.prompt_file_var,
                    fg_color=CORES["bg_input"], border_color=CORES["accent"],
                    placeholder_text="image_prompt.txt").grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        ctk.CTkButton(self.prompt_frame_srt, text="...", width=40, fg_color=CORES["accent"],
                     command=self.select_prompt_file).grid(row=0, column=2, padx=5, pady=5)

        # Info
        ctk.CTkLabel(
            self.srt_mode_frame,
            text="Troca de imagem sincronizada com o texto da legenda",
            text_color=CORES["text_dim"], font=ctk.CTkFont(size=10)
        ).pack(anchor="w", padx=10, pady=(5, 10))

        # ===== PAINEL MODO 1 IMAGEM (PÊNDULO) =====
        self.single_image_mode_frame = ctk.CTkFrame(section, fg_color=CORES["bg_dark"], corner_radius=8)
        
        ctk.CTkLabel(
            self.single_image_mode_frame, text="Opções do Modo 1 Imagem",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=CORES["text_dim"]
        ).pack(anchor="w", padx=10, pady=(10, 5))
        
        # Descrição
        ctk.CTkLabel(
            self.single_image_mode_frame,
            text="Usa 1 imagem aleatória do Banco de Imagens com efeito pêndulo (oscilação suave)",
            text_color=CORES["text_dim"], font=ctk.CTkFont(size=10)
        ).pack(anchor="w", padx=10, pady=(0, 10))
        
        # Sliders de configuração do pêndulo
        pendulum_frame = ctk.CTkFrame(self.single_image_mode_frame, fg_color="transparent")
        pendulum_frame.pack(fill="x", padx=10, pady=5)
        
        # Amplitude
        amp_frame = ctk.CTkFrame(pendulum_frame, fg_color="transparent")
        amp_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(amp_frame, text="Amplitude:", text_color=CORES["text"], width=100).pack(side="left")
        self.amp_value_label = ctk.CTkLabel(amp_frame, text=f"{self.pendulum_amplitude_var.get():.1f}°", 
                                           text_color=CORES["accent"], width=50)
        self.amp_value_label.pack(side="right", padx=5)
        ctk.CTkSlider(
            amp_frame, from_=0.5, to=5.0, number_of_steps=45,
            variable=self.pendulum_amplitude_var, width=200,
            progress_color=CORES["accent"], button_color=CORES["accent"],
            command=lambda v: self.amp_value_label.configure(text=f"{float(v):.1f}°")
        ).pack(side="right", padx=10)
        
        # Zoom
        zoom_frame = ctk.CTkFrame(pendulum_frame, fg_color="transparent")
        zoom_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(zoom_frame, text="Zoom:", text_color=CORES["text"], width=100).pack(side="left")
        self.pend_zoom_value_label = ctk.CTkLabel(zoom_frame, text=f"{self.pendulum_zoom_var.get():.1f}x", 
                                                  text_color=CORES["accent"], width=50)
        self.pend_zoom_value_label.pack(side="right", padx=5)
        ctk.CTkSlider(
            zoom_frame, from_=1.0, to=3.0, number_of_steps=20,
            variable=self.pendulum_zoom_var, width=200,
            progress_color=CORES["accent"], button_color=CORES["accent"],
            command=lambda v: self.pend_zoom_value_label.configure(text=f"{float(v):.1f}x")
        ).pack(side="right", padx=10)
        
        # Crop Ratio
        crop_frame = ctk.CTkFrame(pendulum_frame, fg_color="transparent")
        crop_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(crop_frame, text="Crop:", text_color=CORES["text"], width=100).pack(side="left")
        self.crop_value_label = ctk.CTkLabel(crop_frame, text=f"{self.pendulum_crop_ratio_var.get():.2f}", 
                                            text_color=CORES["accent"], width=50)
        self.crop_value_label.pack(side="right", padx=5)
        ctk.CTkSlider(
            crop_frame, from_=0.5, to=1.0, number_of_steps=50,
            variable=self.pendulum_crop_ratio_var, width=200,
            progress_color=CORES["accent"], button_color=CORES["accent"],
            command=lambda v: self.crop_value_label.configure(text=f"{float(v):.2f}")
        ).pack(side="right", padx=10)
        
        # Duração da célula
        cell_frame = ctk.CTkFrame(pendulum_frame, fg_color="transparent")
        cell_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(cell_frame, text="Célula (s):", text_color=CORES["text"], width=100).pack(side="left")
        self.cell_value_label = ctk.CTkLabel(cell_frame, text=f"{self.pendulum_cell_duration_var.get():.0f}s", 
                                            text_color=CORES["accent"], width=50)
        self.cell_value_label.pack(side="right", padx=5)
        ctk.CTkSlider(
            cell_frame, from_=5.0, to=20.0, number_of_steps=15,
            variable=self.pendulum_cell_duration_var, width=200,
            progress_color=CORES["accent"], button_color=CORES["accent"],
            command=lambda v: self.cell_value_label.configure(text=f"{float(v):.0f}s")
        ).pack(side="right", padx=10)
        
        # Separador
        ctk.CTkLabel(
            self.single_image_mode_frame, text="Chroma Key (Overlay)",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=CORES["text_dim"]
        ).pack(anchor="w", padx=10, pady=(10, 5))
        
        chroma_frame = ctk.CTkFrame(self.single_image_mode_frame, fg_color="transparent")
        chroma_frame.pack(fill="x", padx=10, pady=5)
        
        # Cor do Chroma
        color_frame = ctk.CTkFrame(chroma_frame, fg_color="transparent")
        color_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(color_frame, text="Cor (hex):", text_color=CORES["text"], width=100).pack(side="left")
        ctk.CTkEntry(
            color_frame, textvariable=self.chroma_color_var, width=100,
            fg_color=CORES["bg_input"], border_color=CORES["accent"],
            placeholder_text="00b140"
        ).pack(side="left", padx=5)
        
        # Similarity
        sim_frame = ctk.CTkFrame(chroma_frame, fg_color="transparent")
        sim_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(sim_frame, text="Similarity:", text_color=CORES["text"], width=100).pack(side="left")
        self.sim_value_label = ctk.CTkLabel(sim_frame, text=f"{self.chroma_similarity_var.get():.2f}", 
                                           text_color=CORES["accent"], width=50)
        self.sim_value_label.pack(side="right", padx=5)
        ctk.CTkSlider(
            sim_frame, from_=0.01, to=0.5, number_of_steps=49,
            variable=self.chroma_similarity_var, width=200,
            progress_color=CORES["accent"], button_color=CORES["accent"],
            command=lambda v: self.sim_value_label.configure(text=f"{float(v):.2f}")
        ).pack(side="right", padx=10)
        
        # Blend
        blend_frame = ctk.CTkFrame(chroma_frame, fg_color="transparent")
        blend_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(blend_frame, text="Blend:", text_color=CORES["text"], width=100).pack(side="left")
        self.blend_value_label = ctk.CTkLabel(blend_frame, text=f"{self.chroma_blend_var.get():.2f}", 
                                             text_color=CORES["accent"], width=50)
        self.blend_value_label.pack(side="right", padx=5)
        ctk.CTkSlider(
            blend_frame, from_=0.0, to=0.5, number_of_steps=50,
            variable=self.chroma_blend_var, width=200,
            progress_color=CORES["accent"], button_color=CORES["accent"],
            command=lambda v: self.blend_value_label.configure(text=f"{float(v):.2f}")
        ).pack(side="right", padx=10)
        
        # Info final
        ctk.CTkLabel(
            self.single_image_mode_frame,
            text="A célula base é loopada sem cortes visíveis até cobrir a duração do áudio",
            text_color=CORES["text_dim"], font=ctk.CTkFont(size=10)
        ).pack(anchor="w", padx=10, pady=(5, 10))

        # Mostrar painel correto
        self.toggle_video_mode()

    def toggle_video_mode(self):
        """Alterna entre modo tradicional, Pipeline SRT e Modo 1 Imagem."""
        mode = self.video_mode_var.get()
        
        # Esconder todos os painéis
        self.traditional_mode_frame.pack_forget()
        self.srt_mode_frame.pack_forget()
        if hasattr(self, 'single_image_mode_frame'):
            self.single_image_mode_frame.pack_forget()
        
        if mode == "srt":
            self.srt_mode_frame.pack(fill="x", padx=15, pady=(5, 10))
            self.toggle_srt_source()
        elif mode == "single_image":
            if hasattr(self, 'single_image_mode_frame'):
                self.single_image_mode_frame.pack(fill="x", padx=15, pady=(5, 10))
        else:  # traditional
            self.traditional_mode_frame.pack(fill="x", padx=15, pady=(5, 10))
            self.toggle_zoom_options()

    def toggle_zoom_options(self):
        """Mostra/oculta opções de zoom fixo baseado em animações variadas."""
        try:
            if self.use_varied_animations_var.get():
                self.zoom_options_frame.pack_forget()
            else:
                # Mostrar após o título do modo tradicional
                self.zoom_options_frame.pack(fill="x", padx=10, pady=5, after=self.traditional_mode_frame.winfo_children()[0])
        except:
            pass
    
    def open_effects_selector(self):
        """Abre janela para selecionar quais efeitos usar na randomização."""
        # Criar janela de seleção
        selector_window = ctk.CTkToplevel(self)
        selector_window.title("Selecionar Efeitos de Animação")
        selector_window.geometry("500x520")
        selector_window.resizable(False, False)
        selector_window.transient(self)
        selector_window.grab_set()
        
        # Centralizar janela
        selector_window.update_idletasks()
        x = (selector_window.winfo_screenwidth() - 500) // 2
        y = (selector_window.winfo_screenheight() - 520) // 2
        selector_window.geometry(f"+{x}+{y}")
        
        # Configurar cores
        selector_window.configure(fg_color=CORES["bg_dark"])
        
        # Título
        ctk.CTkLabel(
            selector_window, 
            text="Selecione os Efeitos para Randomização",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=CORES["accent"]
        ).pack(pady=(15, 5))
        
        ctk.CTkLabel(
            selector_window,
            text="Mínimo 2 efeitos. Mesmo com 2, haverá randomização.",
            text_color=CORES["text_dim"],
            font=ctk.CTkFont(size=11)
        ).pack(pady=(0, 15))
        
        # Frame para checkboxes (2 colunas)
        effects_frame = ctk.CTkFrame(selector_window, fg_color=CORES["bg_card"])
        effects_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Criar checkboxes para cada efeito
        row = 0
        col = 0
        for letter, name in self.EFFECT_NAMES.items():
            cb = ctk.CTkCheckBox(
                effects_frame,
                text=f"{letter}: {name}",
                variable=self.effect_vars[letter],
                fg_color=CORES["accent"],
                hover_color=CORES["accent_hover"],
                text_color=CORES["text"],
                font=ctk.CTkFont(size=12),
                command=lambda: self.update_effects_count()
            )
            cb.grid(row=row, column=col, sticky="w", padx=15, pady=8)
            col += 1
            if col > 1:
                col = 0
                row += 1
        
        # Botões de ação rápida
        buttons_frame = ctk.CTkFrame(selector_window, fg_color="transparent")
        buttons_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkButton(
            buttons_frame, text="Selecionar Todos", width=120,
            fg_color=CORES["accent"], hover_color=CORES["accent_hover"],
            command=lambda: self.set_all_effects(True)
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            buttons_frame, text="Desmarcar Todos", width=120,
            fg_color=CORES["bg_input"], hover_color=CORES["accent_hover"],
            command=lambda: self.set_all_effects(False)
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            buttons_frame, text="Apenas Pan", width=100,
            fg_color=CORES["bg_input"], hover_color=CORES["accent_hover"],
            command=lambda: self.set_effects_preset("pan")
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            buttons_frame, text="Apenas Zoom", width=100,
            fg_color=CORES["bg_input"], hover_color=CORES["accent_hover"],
            command=lambda: self.set_effects_preset("zoom")
        ).pack(side="left", padx=5)
        
        # Botão fechar
        ctk.CTkButton(
            selector_window, text="Confirmar", width=150,
            fg_color=CORES["accent"], hover_color=CORES["accent_hover"],
            command=selector_window.destroy
        ).pack(pady=15)
    
    def update_effects_count(self):
        """Atualiza o contador de efeitos selecionados."""
        count = sum(v.get() for v in self.effect_vars.values())
        self.effects_count_label.configure(text=f"({count}/15)")
        
        # Garantir mínimo de 2 efeitos
        if count < 2:
            self.effects_count_label.configure(text_color=CORES["error"] if hasattr(CORES, "error") else "#FF4444")
        else:
            self.effects_count_label.configure(text_color=CORES["text_dim"])
    
    def set_all_effects(self, value: bool):
        """Define todos os efeitos como habilitados ou desabilitados."""
        for var in self.effect_vars.values():
            var.set(value)
        self.update_effects_count()
    
    def set_effects_preset(self, preset: str):
        """Aplica um preset de efeitos."""
        # Primeiro desmarcar todos
        for var in self.effect_vars.values():
            var.set(False)
        
        if preset == "pan":
            # Apenas efeitos de pan (A, B, C, D)
            for letter in ["A", "B", "C", "D"]:
                self.effect_vars[letter].set(True)
        elif preset == "zoom":
            # Efeitos com zoom (E-L + N, O puros)
            for letter in ["E", "F", "G", "H", "I", "J", "K", "L", "N", "O"]:
                self.effect_vars[letter].set(True)
        
        self.update_effects_count()
    
    def get_enabled_effects(self) -> list:
        """Retorna lista de letras dos efeitos habilitados."""
        enabled = [letter for letter, var in self.effect_vars.items() if var.get()]
        # Garantir mínimo de 2 efeitos (pegar os 2 primeiros se necessário)
        if len(enabled) < 2:
            all_letters = list(self.EFFECT_NAMES.keys())
            while len(enabled) < 2:
                for letter in all_letters:
                    if letter not in enabled:
                        enabled.append(letter)
                        break
        return enabled

    def toggle_srt_source(self):
        """Alterna campos de fonte de imagem SRT."""
        if self.image_source_var.get() == "backlog":
            self.whisk_frame.pack_forget()
            self.prompt_selector_frame.pack_forget()
            self.srt_backlog_frame.pack(fill="x", padx=0, pady=5)
        else:
            self.srt_backlog_frame.pack_forget()
            self.whisk_frame.pack(fill="x", padx=0, pady=5)
            self.prompt_selector_frame.pack(fill="x", padx=0, pady=5)

    def get_system_hardware_info(self):
        """Detecta informações reais do hardware do sistema."""
        hardware_info = {
            "cpu": "CPU Desconhecida",
            "cpu_threads": 0,
            "gpu": "N/A",
            "gpu_vram": "",
            "ram": "N/A"
        }
        
        # Detectar CPU
        try:
            import platform
            system = platform.system()
            
            if system == "Darwin":  # macOS
                # Obter nome do processador
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0 and result.stdout.strip():
                    hardware_info["cpu"] = result.stdout.strip()
                
                # Obter número de threads
                result = subprocess.run(
                    ["sysctl", "-n", "hw.logicalcpu"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    hardware_info["cpu_threads"] = int(result.stdout.strip())
                    
            elif system == "Windows":
                import os
                hardware_info["cpu"] = os.environ.get("PROCESSOR_IDENTIFIER", "CPU Desconhecida")
                hardware_info["cpu_threads"] = os.cpu_count() or 0
                
            else:  # Linux
                try:
                    with open("/proc/cpuinfo", "r") as f:
                        for line in f:
                            if "model name" in line:
                                hardware_info["cpu"] = line.split(":")[1].strip()
                                break
                    hardware_info["cpu_threads"] = os.cpu_count() or 0
                except:
                    hardware_info["cpu_threads"] = os.cpu_count() or 0
        except Exception as e:
            hardware_info["cpu_threads"] = os.cpu_count() or 0
        
        # Detectar GPU NVIDIA
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split(",")
                hardware_info["gpu"] = parts[0].strip()
                if len(parts) > 1:
                    vram_mb = int(parts[1].strip())
                    vram_gb = vram_mb / 1024
                    hardware_info["gpu_vram"] = f"{vram_gb:.0f}GB VRAM"
        except:
            pass
        
        # Detectar RAM
        try:
            import psutil
            mem = psutil.virtual_memory()
            total_gb = mem.total / (1024**3)
            hardware_info["ram"] = f"{total_gb:.0f}GB RAM"
        except:
            try:
                import platform
                if platform.system() == "Darwin":
                    result = subprocess.run(
                        ["sysctl", "-n", "hw.memsize"],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0:
                        mem_bytes = int(result.stdout.strip())
                        hardware_info["ram"] = f"{mem_bytes / (1024**3):.0f}GB RAM"
            except:
                pass
        
        return hardware_info

    def format_hardware_string(self):
        """Formata a string de hardware para exibição."""
        hw = self.get_system_hardware_info()
        
        # Simplificar nome da CPU (remover partes desnecessárias)
        cpu_name = hw["cpu"]
        # Remover "with Radeon Graphics" e similares
        cpu_name = cpu_name.replace(" with Radeon Graphics", "").replace(" with Radeon Vega Graphics", "")
        # Limitar tamanho
        if len(cpu_name) > 35:
            cpu_name = cpu_name[:32] + "..."
        
        cpu_part = f"{cpu_name} ({hw['cpu_threads']} threads)"
        
        gpu_part = hw["gpu"]
        if hw["gpu_vram"]:
            gpu_part = f"{hw['gpu']} ({hw['gpu_vram']})"
        
        return f"Hardware: {cpu_part} | {gpu_part} | {hw['ram']}"

    def get_recommended_settings(self):
        """Calcula configurações recomendadas baseadas no hardware detectado."""
        hw = self.get_system_hardware_info()
        threads = hw["cpu_threads"]
        
        # Extrair RAM em GB
        ram_gb = 8  # default
        try:
            ram_str = hw["ram"].replace("GB RAM", "").strip()
            ram_gb = int(ram_str)
        except:
            pass
        
        # Verificar se tem GPU NVIDIA
        has_nvidia = hw["gpu"] != "N/A" and "N/A" not in hw["gpu"]
        
        # Lógica de recomendação baseada no hardware
        # Regra: deixar ~25% dos threads livres para o sistema
        available_threads = max(2, int(threads * 0.75))
        
        # Configurações baseadas em RAM e threads
        if ram_gb >= 32 and threads >= 12:
            # Hardware potente
            if has_nvidia:
                rec_videos = min(4, available_threads // 4)
                rec_threads = min(8, available_threads // rec_videos)
            else:
                rec_videos = min(3, available_threads // 4)
                rec_threads = min(6, available_threads // rec_videos)
        elif ram_gb >= 16 and threads >= 8:
            # Hardware médio-alto
            rec_videos = min(3, available_threads // 3)
            rec_threads = min(6, available_threads // rec_videos)
        elif ram_gb >= 8 and threads >= 4:
            # Hardware médio
            rec_videos = min(2, available_threads // 2)
            rec_threads = min(4, available_threads // rec_videos)
        else:
            # Hardware básico
            rec_videos = 1
            rec_threads = min(4, max(2, available_threads))
        
        # Garantir valores mínimos
        rec_videos = max(1, min(4, rec_videos))
        rec_threads = max(2, min(8, rec_threads))
        
        return {
            "videos": rec_videos,
            "threads": rec_threads,
            "total_threads": rec_videos * rec_threads,
            "available_threads": available_threads,
            "has_nvidia": has_nvidia
        }

    def apply_recommended_settings(self):
        """Aplica as configurações recomendadas aos sliders."""
        rec = self.get_recommended_settings()
        self.parallel_videos_var.set(rec["videos"])
        self.threads_per_video_var.set(rec["threads"])
        self.log(f"✅ Configuração recomendada aplicada: {rec['videos']} vídeos x {rec['threads']} threads", "INFO")

    def create_parallelism_section(self):
        """Secao de controles de paralelismo."""
        section = ctk.CTkFrame(self.main_scroll, fg_color=CORES["bg_section"], corner_radius=12, border_width=1, border_color=CORES["border"])
        section.pack(fill="x", pady=8, padx=10)

        ctk.CTkLabel(
            section, text="⚡  PERFORMANCE",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=CORES["text"]
        ).pack(anchor="w", padx=15, pady=(15, 8))

        # Info de hardware - detectado dinamicamente
        hardware_text = self.format_hardware_string()
        info = ctk.CTkLabel(
            section,
            text=hardware_text,
            text_color=CORES["text_dim"],
            font=ctk.CTkFont(size=10)
        )
        info.pack(anchor="w", padx=15, pady=(0, 5))

        # Frame para recomendação de configuração
        rec_frame = ctk.CTkFrame(section, fg_color=CORES["bg_input"], corner_radius=8)
        rec_frame.pack(fill="x", padx=15, pady=(5, 10))
        
        rec = self.get_recommended_settings()
        gpu_info = "✓ GPU NVIDIA detectada" if rec["has_nvidia"] else "⚠ Sem GPU NVIDIA"
        
        rec_text = ctk.CTkLabel(
            rec_frame,
            text=f"💡 Configuração recomendada: {rec['videos']} vídeos × {rec['threads']} threads = {rec['total_threads']} threads ({gpu_info})",
            text_color=CORES["success"],
            font=ctk.CTkFont(size=11)
        )
        rec_text.pack(side="left", padx=10, pady=8)
        
        ctk.CTkButton(
            rec_frame, text="Aplicar", width=70,
            fg_color=CORES["success"], hover_color="#3db892",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self.apply_recommended_settings
        ).pack(side="right", padx=10, pady=5)

        self.create_slider(section, "Videos Simultaneos:", self.parallel_videos_var, 1, 4, "", is_int=True)
        self.create_slider(section, "Threads/Video:", self.threads_per_video_var, 2, 8, "", is_int=True)

        # Calculo de recursos
        calc_frame = ctk.CTkFrame(section, fg_color="transparent")
        calc_frame.pack(fill="x", padx=15, pady=(0, 15))

        self.resources_label = ctk.CTkLabel(
            calc_frame,
            text="Uso estimado: 2 videos x 6 threads = 12 threads",
            text_color=CORES["info"],
            font=ctk.CTkFont(size=11)
        )
        self.resources_label.pack(side="left")

        # Atualizar calculo quando sliders mudam
        self.parallel_videos_var.trace_add("write", self.update_resources_label)
        self.threads_per_video_var.trace_add("write", self.update_resources_label)

    def update_resources_label(self, *args):
        """Atualiza label de recursos estimados."""
        videos = self.parallel_videos_var.get()
        threads = self.threads_per_video_var.get()
        total = videos * threads
        self.resources_label.configure(
            text=f"Uso estimado: {videos} videos x {threads} threads = {total} threads"
        )

    def create_audio_section(self):
        """Seção unificada de áudio - Música de fundo, TTS e Backlog."""
        section = ctk.CTkFrame(self.main_scroll, fg_color=CORES["bg_section"], corner_radius=12, border_width=1, border_color=CORES["border"])
        section.pack(fill="x", pady=8, padx=10)

        ctk.CTkLabel(
            section, text="🔊  ÁUDIO",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=CORES["text"]
        ).pack(anchor="w", padx=15, pady=(15, 8))

        # ===== MÚSICA DE FUNDO =====
        music_frame = ctk.CTkFrame(section, fg_color="transparent")
        music_frame.pack(fill="x", padx=15, pady=5)
        music_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(music_frame, text="Música de Fundo:", text_color=CORES["text"]).grid(
            row=0, column=0, sticky="w", padx=(0, 10), pady=5)
        ctk.CTkEntry(music_frame, textvariable=self.music_var,
                    fg_color=CORES["bg_input"], border_color=CORES["warning"]).grid(
            row=0, column=1, sticky="ew", padx=5, pady=5)
        ctk.CTkButton(music_frame, text="...", width=40, fg_color=CORES["warning"],
                     hover_color="#FFA000", text_color=CORES["bg_dark"],
                     command=lambda: self.select_file("music")).grid(
            row=0, column=2, padx=5, pady=5)

        self.create_slider(section, "Volume Música:", self.music_volume_var, 0, 1, "%", is_percent=True)

        # ===== BACKLOG DE ÁUDIOS =====
        backlog_check = ctk.CTkCheckBox(
            section, text="Usar Backlog de Áudios (rotação automática)",
            variable=self.use_audio_backlog_var,
            fg_color=CORES["accent"], hover_color=CORES["accent_hover"],
            text_color=CORES["text"],
            command=self.toggle_audio_backlog
        )
        backlog_check.pack(anchor="w", padx=15, pady=5)

        # Frame de backlog
        self.audio_backlog_frame = ctk.CTkFrame(section, fg_color="transparent")
        self.audio_backlog_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(self.audio_backlog_frame, text="Pasta de Áudios:", text_color=CORES["text"]).grid(
            row=0, column=0, sticky="w", padx=(15, 5), pady=5)
        ctk.CTkEntry(self.audio_backlog_frame, textvariable=self.audio_backlog_folder_var,
                    fg_color=CORES["bg_input"], border_color=CORES["accent"]).grid(
            row=0, column=1, sticky="ew", padx=5, pady=5)
        ctk.CTkButton(self.audio_backlog_frame, text="...", width=40, fg_color=CORES["accent"],
                     command=self.select_audio_backlog_folder).grid(row=0, column=2, padx=5, pady=5)

        ctk.CTkLabel(
            self.audio_backlog_frame,
            text="(O sistema alterna entre os áudios da pasta, evitando repetição)",
            text_color=CORES["text_dim"], font=ctk.CTkFont(size=10)
        ).grid(row=1, column=0, columnspan=3, sticky="w", padx=15, pady=(0, 5))

        if self.use_audio_backlog_var.get():
            self.audio_backlog_frame.pack(fill="x")

        # ===== TTS - GERAÇÃO DE VOZ =====
        tts_header = ctk.CTkFrame(section, fg_color="transparent")
        tts_header.pack(fill="x", padx=15, pady=(10, 5))
        
        # Sincronizar tts_expanded_var com o valor salvo de tts_enabled_var
        self.tts_expanded_var = ctk.BooleanVar(value=self.tts_enabled_var.get())
        ctk.CTkCheckBox(
            tts_header, text="Gerar Áudio via TTS (Text-to-Speech)",
            variable=self.tts_expanded_var,
            fg_color=CORES["accent"], hover_color=CORES["accent_hover"],
            text_color=CORES["text"],
            command=self.toggle_tts_panel
        ).pack(side="left")

        # Frame TTS (oculto por padrão)
        self.tts_panel_frame = ctk.CTkFrame(section, fg_color=CORES["bg_dark"], corner_radius=8)
        
        # Provider
        prov_frame = ctk.CTkFrame(self.tts_panel_frame, fg_color="transparent")
        prov_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(prov_frame, text="Provedor:", text_color=CORES["text"]).pack(side="left")
        ctk.CTkRadioButton(prov_frame, text="DarkVie", variable=self.tts_provider_var,
                          value="darkvi", fg_color=CORES["accent"],
                          command=self.toggle_tts_provider).pack(side="left", padx=10)
        ctk.CTkRadioButton(prov_frame, text="Talkify", variable=self.tts_provider_var,
                          value="talkify", fg_color=CORES["accent"],
                          command=self.toggle_tts_provider).pack(side="left")

        # API Keys
        self.tts_keys_frame = ctk.CTkFrame(self.tts_panel_frame, fg_color="transparent")
        self.tts_keys_frame.pack(fill="x", padx=10, pady=5)
        self.tts_keys_frame.grid_columnconfigure(1, weight=1)
        
        self.darkvi_key_label = ctk.CTkLabel(self.tts_keys_frame, text="DarkVie API Key:", text_color=CORES["text"])
        self.darkvi_key_entry = ctk.CTkEntry(self.tts_keys_frame, textvariable=self.darkvi_api_key_var,
                                            fg_color=CORES["bg_input"], border_color=CORES["accent"], show="*")
        
        self.talkify_key_label = ctk.CTkLabel(self.tts_keys_frame, text="Talkify API Key:", text_color=CORES["text"])
        self.talkify_key_entry = ctk.CTkEntry(self.tts_keys_frame, textvariable=self.talkify_api_key_var,
                                             fg_color=CORES["bg_input"], border_color=CORES["accent"], show="*")

        # Vozes
        voice_frame = ctk.CTkFrame(self.tts_panel_frame, fg_color="transparent")
        voice_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkButton(voice_frame, text="Carregar Vozes", width=120,
                     fg_color=CORES["accent"], hover_color=CORES["accent_hover"],
                     command=self.load_tts_voices).pack(side="left")
        
        self.tts_voice_combo = ctk.CTkComboBox(voice_frame, variable=self.tts_voice_name_var,
                                              values=["Selecione..."], width=250,
                                              fg_color=CORES["bg_input"], border_color=CORES["accent"])
        self.tts_voice_combo.pack(side="left", padx=10)

        # Info
        ctk.CTkLabel(
            self.tts_panel_frame,
            text="O áudio gerado substitui a narração do vídeo",
            text_color=CORES["text_dim"], font=ctk.CTkFont(size=10)
        ).pack(anchor="w", padx=10, pady=(0, 10))
        
        # Se TTS estava habilitado nas configurações salvas, expandir o painel
        if self.tts_enabled_var.get():
            self.tts_panel_frame.pack(fill="x", padx=15, pady=(0, 10))
            self.toggle_tts_provider()

    def toggle_audio_backlog(self):
        """Mostra/oculta painel de backlog de áudios."""
        if self.use_audio_backlog_var.get():
            self.audio_backlog_frame.pack(fill="x")
        else:
            self.audio_backlog_frame.pack_forget()

    def toggle_tts_panel(self):
        """Mostra/oculta painel TTS."""
        # Sincronizar tts_enabled_var com o checkbox
        self.tts_enabled_var.set(self.tts_expanded_var.get())
        
        if self.tts_expanded_var.get():
            self.tts_panel_frame.pack(fill="x", padx=15, pady=(0, 10))
            self.toggle_tts_provider()
        else:
            self.tts_panel_frame.pack_forget()

    def toggle_tts_provider(self):
        """Alterna campos de provedor TTS."""
        # Limpar campos
        for widget in self.tts_keys_frame.winfo_children():
            widget.grid_forget()
        
        if self.tts_provider_var.get() == "darkvi":
            self.darkvi_key_label.grid(row=0, column=0, sticky="w", padx=(0, 5), pady=5)
            self.darkvi_key_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        else:
            self.talkify_key_label.grid(row=0, column=0, sticky="w", padx=(0, 5), pady=5)
            self.talkify_key_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

    def create_subtitles_section(self):
        """Seção de legendas simplificada."""
        section = ctk.CTkFrame(self.main_scroll, fg_color=CORES["bg_section"], corner_radius=12, border_width=1, border_color=CORES["border"])
        section.pack(fill="x", pady=8, padx=10)

        ctk.CTkLabel(
            section, text="💬  LEGENDAS",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=CORES["text"]
        ).pack(anchor="w", padx=15, pady=(15, 8))

        # Checkbox principal
        self.subs_check = ctk.CTkCheckBox(
            section, text="Usar legendas no vídeo",
            variable=self.use_subtitles_var,
            fg_color=CORES["accent"],
            hover_color=CORES["accent_hover"],
            text_color=CORES["text"],
            command=self.toggle_subtitles_panel
        )
        self.subs_check.pack(anchor="w", padx=15, pady=5)

        # Frame de opções
        self.subtitles_options_frame = ctk.CTkFrame(section, fg_color="transparent")

        # Modo de legenda (Full ou Sem)
        mode_frame = ctk.CTkFrame(self.subtitles_options_frame, fg_color="transparent")
        mode_frame.pack(fill="x", padx=15, pady=5)
        ctk.CTkLabel(mode_frame, text="Modo:", text_color=CORES["text"]).pack(side="left")
        ctk.CTkRadioButton(
            mode_frame, text="Queimar no vídeo (burn-in)", variable=self.subtitle_mode_var,
            value="full", fg_color=CORES["accent"], hover_color=CORES["accent_hover"]
        ).pack(side="left", padx=10)
        ctk.CTkRadioButton(
            mode_frame, text="Exportar SRT separado", variable=self.subtitle_mode_var,
            value="none", fg_color=CORES["accent"], hover_color=CORES["accent_hover"]
        ).pack(side="left")

        # Fonte das legendas
        method_frame = ctk.CTkFrame(self.subtitles_options_frame, fg_color="transparent")
        method_frame.pack(fill="x", padx=15, pady=5)
        ctk.CTkLabel(method_frame, text="Fonte:", text_color=CORES["text"]).pack(side="left")
        ctk.CTkRadioButton(
            method_frame, text="Importar SRT", variable=self.subtitle_method_var,
            value="srt", fg_color=CORES["accent"], hover_color=CORES["accent_hover"],
            command=self.toggle_subtitle_method
        ).pack(side="left", padx=10)
        ctk.CTkRadioButton(
            method_frame, text="AssemblyAI (transcrição)", variable=self.subtitle_method_var,
            value="assemblyai", fg_color=CORES["accent"], hover_color=CORES["accent_hover"],
            command=self.toggle_subtitle_method
        ).pack(side="left")

        # Campo SRT
        self.srt_frame = ctk.CTkFrame(self.subtitles_options_frame, fg_color="transparent")
        self.srt_frame.pack(fill="x", padx=15, pady=5)
        ctk.CTkLabel(self.srt_frame, text="Arquivo SRT:", text_color=CORES["text"]).pack(side="left")
        ctk.CTkEntry(
            self.srt_frame, textvariable=self.srt_path_var, width=300,
            fg_color=CORES["bg_input"], border_color=CORES["accent"]
        ).pack(side="left", padx=5)
        ctk.CTkButton(
            self.srt_frame, text="...", width=40,
            fg_color=CORES["accent"], hover_color=CORES["accent_hover"],
            command=self.select_srt_file
        ).pack(side="left")

        # Campo API Key
        self.api_frame = ctk.CTkFrame(self.subtitles_options_frame, fg_color="transparent")
        ctk.CTkLabel(self.api_frame, text="Chave API:", text_color=CORES["text"]).pack(side="left")
        ctk.CTkEntry(
            self.api_frame, textvariable=self.assemblyai_key_var, width=350, show="*",
            fg_color=CORES["bg_input"], border_color=CORES["accent"]
        ).pack(side="left", padx=5)

        # Personalizacao
        self.create_subtitle_customization(self.subtitles_options_frame)

        # Presets
        preset_frame = ctk.CTkFrame(self.subtitles_options_frame, fg_color="transparent")
        preset_frame.pack(fill="x", padx=15, pady=(5, 15))
        ctk.CTkButton(
            preset_frame, text="Salvar Preset",
            fg_color=CORES["success"], hover_color="#3db892",
            command=self.save_subtitle_preset
        ).pack(side="left", padx=5)
        ctk.CTkButton(
            preset_frame, text="Carregar Preset",
            fg_color=CORES["info"], hover_color="#CC7A00",
            command=self.load_subtitle_preset
        ).pack(side="left", padx=5)

        # Mostrar/ocultar conforme checkbox
        if self.use_subtitles_var.get():
            self.subtitles_options_frame.pack(fill="x")
            self.toggle_subtitle_method()

    def create_effects_section(self):
        """Seção unificada de efeitos visuais - Overlay, VSL, Vídeos Intro."""
        section = ctk.CTkFrame(self.main_scroll, fg_color=CORES["bg_section"], corner_radius=12, border_width=1, border_color=CORES["border"])
        section.pack(fill="x", pady=8, padx=10)

        ctk.CTkLabel(
            section, text="✨  EFEITOS VISUAIS",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=CORES["text"]
        ).pack(anchor="w", padx=15, pady=(15, 8))

        # ===== OVERLAY =====
        overlay_check = ctk.CTkCheckBox(
            section, text="Usar Overlay (partículas, grão, efeitos)",
            variable=self.use_random_overlays_var,
            fg_color=CORES["accent"], hover_color=CORES["accent_hover"],
            text_color=CORES["text"],
            command=self.toggle_overlay_options
        )
        overlay_check.pack(anchor="w", padx=15, pady=5)

        # Frame de overlay
        self.overlay_options_frame = ctk.CTkFrame(section, fg_color="transparent")
        self.overlay_options_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(self.overlay_options_frame, text="Pasta de Overlays:", text_color=CORES["text"]).grid(
            row=0, column=0, sticky="w", padx=(15, 5), pady=5)
        ctk.CTkEntry(self.overlay_options_frame, textvariable=self.overlay_folder_var,
                    fg_color=CORES["bg_input"], border_color=CORES["accent"],
                    placeholder_text="PNG ou vídeos MP4/MOV").grid(
            row=0, column=1, sticky="ew", padx=5, pady=5)
        ctk.CTkButton(self.overlay_options_frame, text="...", width=40, fg_color=CORES["accent"],
                     command=self.select_overlay_folder).grid(row=0, column=2, padx=5, pady=5)

        # Slider de opacidade em frame separado para evitar conflito pack/grid
        opacity_frame = ctk.CTkFrame(self.overlay_options_frame, fg_color="transparent")
        opacity_frame.grid(row=1, column=0, columnspan=3, sticky="ew", padx=0, pady=5)
        self.create_slider(opacity_frame, "Opacidade:", self.overlay_opacity_var, 0, 1, "%", is_percent=True)

        ctk.CTkLabel(
            self.overlay_options_frame,
            text="(1 overlay por vídeo, rotação automática entre os disponíveis)",
            text_color=CORES["text_dim"], font=ctk.CTkFont(size=10)
        ).grid(row=2, column=0, columnspan=3, sticky="w", padx=15, pady=(0, 5))

        if self.use_random_overlays_var.get():
            self.overlay_options_frame.pack(fill="x")

        # ===== VSL =====
        ctk.CTkCheckBox(
            section, text="Usar VSL (Video Sales Letter)",
            variable=self.use_vsl_var,
            fg_color=CORES["accent"], hover_color=CORES["accent_hover"],
            text_color=CORES["text"],
            command=self.toggle_vsl_selector
        ).pack(anchor="w", padx=15, pady=5)

        ctk.CTkLabel(
            section,
            text="(Insere vídeo promocional quando detecta palavra-chave na narração)",
            text_color=CORES["text_dim"], font=ctk.CTkFont(size=10)
        ).pack(anchor="w", padx=30)

        # Frame seletor de VSL
        self.vsl_selector_frame = ctk.CTkFrame(section, fg_color="transparent")
        self.vsl_selector_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(self.vsl_selector_frame, text="VSL para usar:", text_color=CORES["text"]).grid(
            row=0, column=0, sticky="w", padx=(30, 5), pady=5)
        
        # Obter lista de VSLs disponíveis
        vsl_names = get_vsl_names()
        default_vsl = self.config.get("selected_vsl", vsl_names[0] if vsl_names else "")
        
        self.selected_vsl_var = ctk.StringVar(value=default_vsl)
        self.vsl_combobox = ctk.CTkComboBox(
            self.vsl_selector_frame,
            variable=self.selected_vsl_var,
            values=vsl_names if vsl_names else ["Nenhuma VSL encontrada"],
            fg_color=CORES["bg_input"],
            border_color=CORES["accent"],
            button_color=CORES["accent"],
            button_hover_color=CORES["accent_hover"],
            dropdown_fg_color=CORES["bg_card"],
            dropdown_hover_color=CORES["bg_hover"],
            width=250,
            command=self.on_vsl_selected
        )
        self.vsl_combobox.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        # Botão para abrir pasta de VSLs
        ctk.CTkButton(
            self.vsl_selector_frame, 
            text="Abrir Pasta", 
            width=90, 
            fg_color=CORES["accent"],
            hover_color=CORES["accent_hover"],
            command=self.open_vsl_folder
        ).grid(row=0, column=2, padx=5, pady=5)
        
        # Botão para atualizar lista
        ctk.CTkButton(
            self.vsl_selector_frame, 
            text="↻", 
            width=30, 
            fg_color=CORES["bg_card"],
            hover_color=CORES["bg_hover"],
            command=self.refresh_vsl_list
        ).grid(row=0, column=3, padx=(0, 5), pady=5)
        
        # Label de status
        self.vsl_status_var = ctk.StringVar(value=get_vsl_summary())
        ctk.CTkLabel(
            self.vsl_selector_frame, 
            textvariable=self.vsl_status_var,
            text_color=CORES["text_dim"],
            font=ctk.CTkFont(size=11)
        ).grid(row=1, column=0, columnspan=4, sticky="w", padx=(30, 5), pady=(0, 5))
        
        # Modo de inserção VSL
        ctk.CTkLabel(self.vsl_selector_frame, text="Modo de inserção:", text_color=CORES["text"]).grid(
            row=2, column=0, sticky="w", padx=(30, 5), pady=5)
        
        vsl_mode_frame = ctk.CTkFrame(self.vsl_selector_frame, fg_color="transparent")
        vsl_mode_frame.grid(row=2, column=1, columnspan=3, sticky="w", padx=5, pady=5)
        
        ctk.CTkRadioButton(
            vsl_mode_frame, text="Palavra-chave (requer legendas)",
            variable=self.vsl_insertion_mode_var, value="keyword",
            fg_color=CORES["accent"], hover_color=CORES["accent_hover"],
            text_color=CORES["text"],
            command=self.toggle_vsl_mode_options
        ).pack(side="left", padx=(0, 15))
        
        ctk.CTkRadioButton(
            vsl_mode_frame, text="Posição fixa",
            variable=self.vsl_insertion_mode_var, value="fixed",
            fg_color=CORES["accent"], hover_color=CORES["accent_hover"],
            text_color=CORES["text"],
            command=self.toggle_vsl_mode_options
        ).pack(side="left", padx=(0, 15))
        
        ctk.CTkRadioButton(
            vsl_mode_frame, text="Range aleatório",
            variable=self.vsl_insertion_mode_var, value="range",
            fg_color=CORES["accent"], hover_color=CORES["accent_hover"],
            text_color=CORES["text"],
            command=self.toggle_vsl_mode_options
        ).pack(side="left")
        
        # Frame para opções de palavra-chave
        self.vsl_keyword_options_frame = ctk.CTkFrame(self.vsl_selector_frame, fg_color="transparent")
        self.vsl_keyword_options_frame.grid(row=3, column=0, columnspan=4, sticky="ew", padx=(30, 5), pady=5)
        
        # Seletor de idioma para palavras-chave
        ctk.CTkLabel(self.vsl_keyword_options_frame, text="Idioma das palavras-chave:", text_color=CORES["text"]).grid(
            row=0, column=0, sticky="w", padx=(0, 5), pady=5)
        
        # Carregar idiomas disponíveis do arquivo de keywords
        vsl_languages = self.get_vsl_languages()
        
        self.vsl_language_combobox = ctk.CTkComboBox(
            self.vsl_keyword_options_frame,
            variable=self.vsl_language_var,
            values=vsl_languages if vsl_languages else ["portugues"],
            fg_color=CORES["bg_input"],
            border_color=CORES["accent"],
            button_color=CORES["accent"],
            button_hover_color=CORES["accent_hover"],
            dropdown_fg_color=CORES["bg_card"],
            dropdown_hover_color=CORES["bg_hover"],
            width=200
        )
        self.vsl_language_combobox.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        ctk.CTkLabel(
            self.vsl_keyword_options_frame, 
            text="(Busca palavras-chave apenas no idioma selecionado - requer AssemblyAI ou SRT)",
            text_color=CORES["text_dim"],
            font=ctk.CTkFont(size=10)
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 5))
        
        # Frame para opções de posição fixa
        self.vsl_fixed_options_frame = ctk.CTkFrame(self.vsl_selector_frame, fg_color="transparent")
        self.vsl_fixed_options_frame.grid(row=4, column=0, columnspan=4, sticky="ew", padx=(30, 5), pady=5)
        
        ctk.CTkLabel(self.vsl_fixed_options_frame, text="Inserir VSL em:", text_color=CORES["text"]).grid(
            row=0, column=0, sticky="w", padx=(0, 5), pady=5)
        
        self.vsl_fixed_position_entry = ctk.CTkEntry(
            self.vsl_fixed_options_frame,
            textvariable=self.vsl_fixed_position_var,
            fg_color=CORES["bg_input"],
            border_color=CORES["accent"],
            width=80
        )
        self.vsl_fixed_position_entry.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        ctk.CTkLabel(self.vsl_fixed_options_frame, text="segundos", text_color=CORES["text"]).grid(
            row=0, column=2, sticky="w", padx=(0, 10), pady=5)
        
        ctk.CTkLabel(
            self.vsl_fixed_options_frame, 
            text="(A VSL será inserida nesta posição do vídeo, sem necessidade de legendas)",
            text_color=CORES["text_dim"],
            font=ctk.CTkFont(size=10)
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 5))
        
        # Frame para opções de range aleatório
        self.vsl_range_options_frame = ctk.CTkFrame(self.vsl_selector_frame, fg_color="transparent")
        self.vsl_range_options_frame.grid(row=5, column=0, columnspan=4, sticky="ew", padx=(30, 5), pady=5)
        
        ctk.CTkLabel(self.vsl_range_options_frame, text="Inserir VSL entre:", text_color=CORES["text"]).grid(
            row=0, column=0, sticky="w", padx=(0, 5), pady=5)
        
        self.vsl_range_start_entry = ctk.CTkEntry(
            self.vsl_range_options_frame,
            textvariable=self.vsl_range_start_var,
            fg_color=CORES["bg_input"],
            border_color=CORES["accent"],
            width=60
        )
        self.vsl_range_start_entry.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        ctk.CTkLabel(self.vsl_range_options_frame, text="min  e", text_color=CORES["text"]).grid(
            row=0, column=2, sticky="w", padx=(0, 5), pady=5)
        
        self.vsl_range_end_entry = ctk.CTkEntry(
            self.vsl_range_options_frame,
            textvariable=self.vsl_range_end_var,
            fg_color=CORES["bg_input"],
            border_color=CORES["accent"],
            width=60
        )
        self.vsl_range_end_entry.grid(row=0, column=3, sticky="w", padx=5, pady=5)
        
        ctk.CTkLabel(self.vsl_range_options_frame, text="min", text_color=CORES["text"]).grid(
            row=0, column=4, sticky="w", padx=(0, 10), pady=5)
        
        ctk.CTkLabel(
            self.vsl_range_options_frame, 
            text="(Cada vídeo terá a VSL em uma posição aleatória dentro deste intervalo)",
            text_color=CORES["text_dim"],
            font=ctk.CTkFont(size=10)
        ).grid(row=1, column=0, columnspan=5, sticky="w", pady=(0, 5))
        
        # Inicializar visibilidade das opções
        self.toggle_vsl_mode_options()
        
        # Mostrar se VSL está ativo
        if self.use_vsl_var.get():
            self.vsl_selector_frame.pack(fill="x", pady=(0, 5))

        # ===== VÍDEOS INTRODUTÓRIOS =====
        intro_check = ctk.CTkCheckBox(
            section, text="Usar Vídeos Introdutórios (backlog de intros)",
            variable=self.use_backlog_videos_var,
            fg_color=CORES["accent"], hover_color=CORES["accent_hover"],
            text_color=CORES["text"],
            command=self.toggle_intro_videos
        )
        intro_check.pack(anchor="w", padx=15, pady=5)

        # Frame de intros
        self.intro_videos_frame = ctk.CTkFrame(section, fg_color="transparent")
        self.intro_videos_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(self.intro_videos_frame, text="Pasta de Intros:", text_color=CORES["text"]).grid(
            row=0, column=0, sticky="w", padx=(15, 5), pady=5)
        ctk.CTkEntry(self.intro_videos_frame, textvariable=self.backlog_folder_var,
                    fg_color=CORES["bg_input"], border_color=CORES["accent"]).grid(
            row=0, column=1, sticky="ew", padx=5, pady=5)
        ctk.CTkButton(self.intro_videos_frame, text="...", width=40, fg_color=CORES["accent"],
                     command=self.select_backlog_folder).grid(row=0, column=2, padx=5, pady=5)

        # Status
        status_row = ctk.CTkFrame(self.intro_videos_frame, fg_color="transparent")
        status_row.grid(row=1, column=0, columnspan=3, sticky="w", padx=15, pady=5)
        ctk.CTkLabel(status_row, text="Status:", text_color=CORES["text_dim"]).pack(side="left")
        self.backlog_status_label = ctk.CTkLabel(status_row, textvariable=self.backlog_status_var,
                                                 text_color=CORES["text"])
        self.backlog_status_label.pack(side="left", padx=5)

        if self.use_backlog_videos_var.get():
            self.intro_videos_frame.pack(fill="x")

        # Espaço final
        ctk.CTkFrame(section, fg_color="transparent", height=10).pack()

    def toggle_overlay_options(self):
        """Mostra/oculta opções de overlay."""
        if self.use_random_overlays_var.get():
            self.overlay_options_frame.pack(fill="x")
        else:
            self.overlay_options_frame.pack_forget()

    def toggle_intro_videos(self):
        """Mostra/oculta opções de vídeos introdutórios."""
        if self.use_backlog_videos_var.get():
            self.intro_videos_frame.pack(fill="x")
            self.update_backlog_status()
        else:
            self.intro_videos_frame.pack_forget()
        
        # Atualizar status quando pasta mudar
        self.backlog_folder_var.trace_add("write", lambda *args: self.update_backlog_status())
        
        # Atualizar status inicial
        self.update_backlog_status()

    def toggle_vsl_selector(self):
        """Mostra/oculta seletor de VSL."""
        if self.use_vsl_var.get():
            self.vsl_selector_frame.pack(fill="x", pady=(0, 5))
            self.refresh_vsl_list()
        else:
            self.vsl_selector_frame.pack_forget()
    
    def toggle_vsl_mode_options(self):
        """Mostra/oculta opções de modo VSL (keyword, fixed ou range)."""
        mode = self.vsl_insertion_mode_var.get()
        if mode == "keyword":
            self.vsl_keyword_options_frame.grid()
            self.vsl_fixed_options_frame.grid_remove()
            self.vsl_range_options_frame.grid_remove()
        elif mode == "fixed":
            self.vsl_keyword_options_frame.grid_remove()
            self.vsl_fixed_options_frame.grid()
            self.vsl_range_options_frame.grid_remove()
        else:  # range
            self.vsl_keyword_options_frame.grid_remove()
            self.vsl_fixed_options_frame.grid_remove()
            self.vsl_range_options_frame.grid()
    
    def on_vsl_selected(self, choice):
        """Callback quando VSL é selecionada."""
        self.log(f"VSL selecionada: {choice}", "INFO")
    
    def open_vsl_folder(self):
        """Abre a pasta de VSLs no explorador de arquivos."""
        vsl_folder = self.config.get("vsl_folder", str(SCRIPT_DIR / "EFEITOS" / "VSLs"))
        
        if not os.path.exists(vsl_folder):
            os.makedirs(vsl_folder, exist_ok=True)
        
        if sys.platform == "win32":
            os.startfile(vsl_folder)
        elif sys.platform == "darwin":
            subprocess.run(["open", vsl_folder])
        else:
            subprocess.run(["xdg-open", vsl_folder])
        
        self.log(f"Pasta de VSLs aberta: {vsl_folder}", "INFO")
    
    def refresh_vsl_list(self):
        """Atualiza lista de VSLs disponíveis."""
        vsl_names = get_vsl_names()
        
        if vsl_names:
            self.vsl_combobox.configure(values=vsl_names)
            # Se a VSL atual não existe mais, selecionar a primeira
            if self.selected_vsl_var.get() not in vsl_names:
                self.selected_vsl_var.set(vsl_names[0])
        else:
            self.vsl_combobox.configure(values=["Nenhuma VSL encontrada"])
            self.selected_vsl_var.set("Nenhuma VSL encontrada")
        
        self.vsl_status_var.set(get_vsl_summary())
        
        # Atualizar também a lista de idiomas
        if hasattr(self, 'vsl_language_combobox'):
            languages = self.get_vsl_languages()
            if languages:
                self.vsl_language_combobox.configure(values=languages)
                if self.vsl_language_var.get() not in languages:
                    self.vsl_language_var.set(languages[0])
        
        self.log("Lista de VSLs atualizada", "INFO")
    
    def get_vsl_languages(self):
        """Retorna lista de idiomas disponíveis no arquivo de keywords."""
        keywords_file = self.config.get("vsl_keywords_file", str(SCRIPT_DIR / "vsl_keywords.json"))
        if not os.path.isabs(keywords_file):
            keywords_file = str(SCRIPT_DIR / keywords_file)
        
        try:
            if os.path.exists(keywords_file):
                with open(keywords_file, 'r', encoding='utf-8') as f:
                    keywords = json.load(f)
                    return list(keywords.keys())
        except Exception as e:
            self.log(f"Erro ao carregar idiomas de VSL: {e}", "WARN")
        
        return ["portugues"]  # Fallback

    # Métodos de toggle e seleção removidos - agora em seções unificadas

    def select_images_backlog_folder(self):
        """Abre dialog para selecionar pasta de imagens backlog."""
        folder = filedialog.askdirectory(
            title="Selecionar Pasta de Imagens Backlog",
            initialdir=self.images_backlog_folder_var.get() if self.images_backlog_folder_var.get() else str(SCRIPT_DIR)
        )
        if folder:
            self.images_backlog_folder_var.set(folder)

    def open_whisk_keys_file(self):
        """Abre o arquivo whisk_keys.json para edição."""
        import subprocess
        import platform
        
        file_path = str(WHISK_KEYS_FILE)
        
        try:
            if platform.system() == "Darwin":  # macOS
                subprocess.run(["open", file_path])
            elif platform.system() == "Windows":
                os.startfile(file_path)
            else:  # Linux
                subprocess.run(["xdg-open", file_path])
            
            self.log(f"Abrindo arquivo: {file_path}", "INFO")
        except Exception as e:
            self.log(f"Erro ao abrir arquivo: {e}", "ERROR")
            messagebox.showerror("Erro", f"Não foi possível abrir o arquivo:\n{file_path}")

    def refresh_whisk_tokens(self):
        """Atualiza o resumo de tokens do arquivo."""
        summary = get_tokens_summary()
        self.whisk_tokens_summary_var.set(summary)
        
        # Também recarrega o pool manager
        tokens = get_enabled_tokens()
        self.log(f"Tokens Whisk recarregados: {len(tokens)} encontrados", "OK")

    def capture_whisk_token_auto(self):
        """
        Captura automaticamente o token Bearer do Whisk.
        Abre um navegador, aguarda login e intercepta o token.
        """
        import threading
        
        # Desabilitar botão imediatamente (na thread principal)
        self.capture_token_btn.configure(state="disabled", text="Capturando...")
        self.log("Iniciando captura automática de token Whisk...", "INFO")
        self.log("Um navegador será aberto. Faça login no Google se necessário.", "INFO")
        
        def capture_thread():
            try:
                # Importar módulo de captura
                try:
                    from whisk_token_capture import capture_whisk_token
                except ImportError as e:
                    self.after(0, lambda: self.log(f"Módulo de captura não encontrado: {e}", "ERROR"))
                    self.after(0, lambda: self.log("Verifique se whisk_token_capture.py existe e playwright está instalado.", "ERROR"))
                    self.after(0, lambda: messagebox.showerror(
                        "Erro", 
                        f"Módulo de captura não encontrado.\n\nInstale o playwright:\npip install playwright\nplaywright install chromium"
                    ))
                    return
                
                # Função de log thread-safe
                def safe_log(msg, level="INFO"):
                    self.after(0, lambda: self.log(msg, level))
                
                # Executar captura
                result = capture_whisk_token(log_callback=safe_log, timeout_seconds=180)
                
                if result.success and result.token:
                    # Salvar token
                    action = update_or_add_token_by_email(
                        key=result.token,
                        email=result.email
                    )
                    
                    if result.email:
                        self.after(0, lambda: self.log(f"Token {action} para {result.email}", "OK"))
                    else:
                        self.after(0, lambda: self.log(f"Token {action} (email não detectado)", "OK"))
                    
                    # Atualizar UI
                    self.after(0, self.refresh_whisk_tokens)
                    self.after(0, lambda: messagebox.showinfo(
                        "Sucesso", 
                        f"Token capturado com sucesso!\n\nEmail: {result.email or 'Não detectado'}\nAção: {action}"
                    ))
                else:
                    error_msg = result.error or "Erro desconhecido"
                    self.after(0, lambda: self.log(f"Falha na captura: {error_msg}", "ERROR"))
                    self.after(0, lambda: messagebox.showerror(
                        "Erro na Captura", 
                        f"Não foi possível capturar o token.\n\n{error_msg}"
                    ))
                    
            except Exception as e:
                self.after(0, lambda: self.log(f"Erro durante captura: {str(e)}", "ERROR"))
                self.after(0, lambda: messagebox.showerror(
                    "Erro", 
                    f"Erro durante captura de token:\n{str(e)}"
                ))
            finally:
                # Reabilitar botão
                self.after(0, lambda: self.capture_token_btn.configure(
                    state="normal", 
                    text="+ Novo Token"
                ))
        
        # Executar em thread separada para não bloquear UI
        thread = threading.Thread(target=capture_thread, daemon=True)
        thread.start()

    def open_image_prompts_file(self):
        """Abre o arquivo image_prompts.json para edição."""
        import subprocess
        import platform
        
        file_path = str(IMAGE_PROMPTS_FILE)
        
        try:
            if platform.system() == "Darwin":  # macOS
                subprocess.run(["open", file_path])
            elif platform.system() == "Windows":
                os.startfile(file_path)
            else:  # Linux
                subprocess.run(["xdg-open", file_path])
            
            self.log(f"Abrindo arquivo: {file_path}", "INFO")
        except Exception as e:
            self.log(f"Erro ao abrir arquivo: {e}", "ERROR")
            messagebox.showerror("Erro", f"Não foi possível abrir o arquivo:\n{file_path}")

    def refresh_prompt_list(self):
        """Atualiza a lista de prompts do combobox."""
        prompt_names = get_prompt_names()
        
        if prompt_names:
            self.prompt_combobox.configure(values=prompt_names)
            
            # Manter seleção atual se ainda existir
            current = self.selected_prompt_var.get()
            if current not in prompt_names:
                self.selected_prompt_var.set(prompt_names[0])
                self.update_prompt_description()
            
            self.log(f"Lista de prompts atualizada: {len(prompt_names)} prompts", "OK")
        else:
            self.prompt_combobox.configure(values=["Nenhum prompt configurado"])
            self.selected_prompt_var.set("Nenhum prompt configurado")
            self.log("Nenhum prompt encontrado em image_prompts.json", "WARN")

    def on_prompt_selected(self, choice):
        """Callback quando um prompt é selecionado."""
        prompt_data = get_prompt_by_name(choice)
        if prompt_data:
            prompt_id = prompt_data.get("id", "")
            set_default_prompt(prompt_id)
            self.update_prompt_description()
            self.log(f"Prompt selecionado: {choice}", "OK")

    def update_prompt_description(self):
        """Atualiza a descrição do prompt selecionado."""
        prompt_name = self.selected_prompt_var.get()
        prompt_data = get_prompt_by_name(prompt_name)
        
        if prompt_data:
            description = prompt_data.get("description", "")
            self.prompt_description_var.set(f"→ {description}" if description else "")
        else:
            self.prompt_description_var.set("")

    def get_selected_prompt_id(self) -> str:
        """Retorna o ID do prompt selecionado."""
        prompt_name = self.selected_prompt_var.get()
        prompt_data = get_prompt_by_name(prompt_name)
        
        if prompt_data:
            return prompt_data.get("id", "spiritual_chosen")
        return "spiritual_chosen"

    def select_prompt_file(self):
        """Abre dialog para selecionar arquivo de prompt."""
        file_path = filedialog.askopenfilename(
            title="Selecionar Arquivo de Prompt",
            filetypes=[("Arquivos de texto", "*.txt *.md"), ("Todos", "*.*")],
            initialdir=str(SCRIPT_DIR)
        )
        if file_path:
            self.prompt_file_var.set(file_path)
            self.save_config()

    def select_overlay_folder(self):
        """Abre dialog para selecionar pasta de overlays."""
        folder = filedialog.askdirectory(
            title="Selecionar Pasta de Overlays",
            initialdir=self.overlay_folder_var.get() if self.overlay_folder_var.get() else str(SCRIPT_DIR)
        )
        if folder:
            self.overlay_folder_var.set(folder)
            self.save_config()

    def toggle_overlay_folder(self):
        """Mostra/oculta campo de pasta de overlays."""
        if self.use_random_overlays_var.get():
            self.overlay_folder_frame.pack(fill="x", padx=15, pady=5)
        else:
            self.overlay_folder_frame.pack_forget()

    # Seção de backlog de áudios removida - agora em create_audio_section

    def select_audio_backlog_folder(self):
        """Abre dialog para selecionar pasta de áudios backlog."""
        folder = filedialog.askdirectory(
            title="Selecionar Pasta de Áudios Backlog",
            initialdir=self.audio_backlog_folder_var.get() if self.audio_backlog_folder_var.get() else str(SCRIPT_DIR)
        )
        if folder:
            self.audio_backlog_folder_var.set(folder)

    def select_backlog_folder(self):
        """Abre dialog para selecionar pasta do backlog."""
        folder = filedialog.askdirectory(
            title="Selecionar Pasta do Backlog de Vídeos",
            initialdir=self.backlog_folder_var.get() if self.backlog_folder_var.get() else str(SCRIPT_DIR)
        )
        if folder:
            self.backlog_folder_var.set(folder)
            self.update_backlog_status()

    def update_backlog_status(self):
        """Atualiza status dos videos do backlog."""
        if not self.use_backlog_videos_var.get():
            self.backlog_status_var.set("Desabilitado")
            self.backlog_status_label.configure(text_color=CORES["text_dim"])
            return
        
        backlog_folder = self.backlog_folder_var.get()
        if not backlog_folder:
            backlog_folder = self.config.get("backlog_folder", str(SCRIPT_DIR / "EFEITOS" / "BACKLOG_VIDEOS"))
        
        if not os.path.isabs(backlog_folder):
            backlog_folder = str(SCRIPT_DIR / backlog_folder)
        
        if not os.path.exists(backlog_folder):
            self.backlog_status_var.set("Pasta não encontrada")
            self.backlog_status_label.configure(text_color=CORES["error"])
            return
        
        # Contar videos
        video_extensions = ('.mp4', '.mov', '.avi', '.mkv', '.webm')
        video_count = 0
        try:
            for f in os.listdir(backlog_folder):
                full_path = os.path.join(backlog_folder, f)
                if os.path.isfile(full_path) and f.lower().endswith(video_extensions):
                    video_count += 1
        except:
            pass
        
        required = self.config.get("backlog_video_count", 6)
        if video_count >= required:
            self.backlog_status_var.set(f"{video_count} videos disponíveis (OK)")
            self.backlog_status_label.configure(text_color=CORES["success"])
        else:
            self.backlog_status_var.set(f"{video_count} videos (mínimo: {required})")
            self.backlog_status_label.configure(text_color=CORES["warning"])

    def create_subtitle_customization(self, parent):
        """Cria campos de personalizacao de legendas."""
        custom_frame = ctk.CTkFrame(parent, fg_color=CORES["bg_input"], corner_radius=0)
        custom_frame.pack(fill="x", padx=15, pady=10)

        # Titulo
        ctk.CTkLabel(
            custom_frame, text="Personalizacao",
            font=ctk.CTkFont(weight="bold"),
            text_color=CORES["accent"]
        ).pack(anchor="w", padx=10, pady=5)

        # Linha 1: Fonte + Tamanho
        row1 = ctk.CTkFrame(custom_frame, fg_color="transparent")
        row1.pack(fill="x", padx=10, pady=3)
        ctk.CTkLabel(row1, text="Fonte:", text_color=CORES["text"]).pack(side="left")
        self.font_combo = ctk.CTkComboBox(
            row1, variable=self.sub_font_var, values=self.get_system_fonts(), width=150,
            fg_color=CORES["bg_dark"], border_color=CORES["accent"]
        )
        self.font_combo.pack(side="left", padx=5)
        ctk.CTkLabel(row1, text="Tamanho:", text_color=CORES["text"]).pack(side="left", padx=(20, 5))
        ctk.CTkEntry(
            row1, textvariable=self.sub_font_size_var, width=60,
            fg_color=CORES["bg_dark"], border_color=CORES["accent"]
        ).pack(side="left")

        # Linha 2: Cores
        row2 = ctk.CTkFrame(custom_frame, fg_color="transparent")
        row2.pack(fill="x", padx=10, pady=3)
        self.create_color_picker(row2, "Principal:", self.sub_color_primary_var)
        self.create_color_picker(row2, "Contorno:", self.sub_color_outline_var)
        self.create_color_picker(row2, "Sombra:", self.sub_color_shadow_var)
        self.create_color_picker(row2, "Karaoke:", self.sub_color_karaoke_var)

        # Linha 3: Tamanhos + Karaoke
        row3 = ctk.CTkFrame(custom_frame, fg_color="transparent")
        row3.pack(fill="x", padx=10, pady=3)
        ctk.CTkLabel(row3, text="Contorno:", text_color=CORES["text"]).pack(side="left")
        ctk.CTkEntry(
            row3, textvariable=self.sub_outline_size_var, width=40,
            fg_color=CORES["bg_dark"], border_color=CORES["accent"]
        ).pack(side="left", padx=5)
        ctk.CTkLabel(row3, text="Sombra:", text_color=CORES["text"]).pack(side="left", padx=(20, 5))
        ctk.CTkEntry(
            row3, textvariable=self.sub_shadow_size_var, width=40,
            fg_color=CORES["bg_dark"], border_color=CORES["accent"]
        ).pack(side="left")
        ctk.CTkCheckBox(
            row3, text="Efeito Karaoke", variable=self.sub_use_karaoke_var,
            fg_color=CORES["accent"], hover_color=CORES["accent_hover"],
            text_color=CORES["text"]
        ).pack(side="left", padx=20)

        # Linha 4: Grid de Posicao (3x3)
        self.create_position_grid(custom_frame)

    def create_color_picker(self, parent, label, variable):
        """Cria um color picker inline."""
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(side="left", padx=5)

        ctk.CTkLabel(frame, text=label, text_color=CORES["text"]).pack(side="left")

        current_color = variable.get() or "#FFFFFF"
        display_color = current_color if len(current_color) <= 7 else "#" + current_color[3:9]

        btn = ctk.CTkButton(
            frame, text="", width=25, height=25,
            fg_color=display_color, hover_color=display_color,
            border_width=1, border_color=CORES["text"],
            command=lambda: self.pick_color(variable, btn)
        )
        btn.pack(side="left", padx=3)

    def pick_color(self, variable, button):
        """Abre o color chooser e atualiza a variavel."""
        current = variable.get() or "#FFFFFF"
        if len(current) > 7:
            current = "#" + current[3:9]
        color = colorchooser.askcolor(color=current, title="Escolha uma cor")
        if color[1]:
            variable.set(color[1])
            button.configure(fg_color=color[1], hover_color=color[1])

    def create_position_grid(self, parent):
        """Grid 3x3 para posicao da legenda (numpad style)."""
        pos_frame = ctk.CTkFrame(parent, fg_color="transparent")
        pos_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(pos_frame, text="Posicao:", text_color=CORES["text"]).pack(side="left")

        grid = ctk.CTkFrame(pos_frame, fg_color="transparent")
        grid.pack(side="left", padx=10)

        # ASS alignments: 7=TL 8=TC 9=TR / 4=ML 5=MC 6=MR / 1=BL 2=BC 3=BR
        positions = [
            ("7", "8", "9"),  # Superior
            ("4", "5", "6"),  # Meio
            ("1", "2", "3")   # Inferior
        ]

        for row_idx, row in enumerate(positions):
            for col_idx, align_val in enumerate(row):
                btn = ctk.CTkRadioButton(
                    grid, text="", variable=self.sub_position_var, value=align_val,
                    width=20, radiobutton_width=15, radiobutton_height=15,
                    fg_color=CORES["accent"], hover_color=CORES["accent_hover"]
                )
                btn.grid(row=row_idx, column=col_idx, padx=3, pady=3)

        ctk.CTkLabel(
            pos_frame, text="(Padrao: Inferior Centro)",
            text_color=CORES["text_dim"], font=ctk.CTkFont(size=10)
        ).pack(side="left", padx=10)

    def get_system_fonts(self):
        """Retorna lista de fontes do sistema."""
        try:
            fonts = sorted(list(tkfont.families()))
            # Filtrar fontes comuns e uteis
            common = ["Arial", "Verdana", "Tahoma", "Georgia", "Times New Roman", "Calibri", "Segoe UI"]
            result = [f for f in common if f in fonts]
            result.extend([f for f in fonts if f not in result])
            return result[:50]  # Limitar a 50 fontes
        except:
            return ["Arial", "Verdana", "Tahoma", "Georgia", "Times New Roman"]

    def toggle_subtitles_panel(self):
        """Mostra/oculta painel de legendas."""
        if self.use_subtitles_var.get():
            self.subtitles_options_frame.pack(fill="x")
            self.toggle_subtitle_method()
        else:
            self.subtitles_options_frame.pack_forget()

    def toggle_subtitle_method(self):
        """Mostra campo SRT ou API conforme metodo selecionado."""
        if self.subtitle_method_var.get() == "srt":
            self.srt_frame.pack(fill="x", padx=15, pady=5)
            self.api_frame.pack_forget()
        else:
            self.srt_frame.pack_forget()
            self.api_frame.pack(fill="x", padx=15, pady=5)

    def select_srt_file(self):
        """Seleciona arquivo SRT."""
        path = filedialog.askopenfilename(
            title="Selecionar Arquivo SRT",
            filetypes=[("SRT", "*.srt"), ("Todos", "*.*")]
        )
        if path:
            self.srt_path_var.set(path)

    def save_subtitle_preset(self):
        """Salva configuracoes de legenda em preset."""
        # Dialog simples para nome do preset
        dialog = ctk.CTkInputDialog(text="Nome do preset:", title="Salvar Preset")
        name = dialog.get_input()
        if not name:
            return

        preset = {
            "font_name": self.sub_font_var.get(),
            "font_size": self.sub_font_size_var.get(),
            "color_primary": self.sub_color_primary_var.get(),
            "color_outline": self.sub_color_outline_var.get(),
            "color_shadow": self.sub_color_shadow_var.get(),
            "color_karaoke": self.sub_color_karaoke_var.get(),
            "outline_size": self.sub_outline_size_var.get(),
            "shadow_size": self.sub_shadow_size_var.get(),
            "use_karaoke": self.sub_use_karaoke_var.get(),
            "position": self.sub_position_var.get()
        }

        # Carregar presets existentes
        presets = {}
        if os.path.exists(SUBTITLE_PRESETS_FILE):
            try:
                with open(SUBTITLE_PRESETS_FILE, 'r', encoding='utf-8') as f:
                    presets = json.load(f)
            except:
                pass

        presets[name] = preset

        with open(SUBTITLE_PRESETS_FILE, 'w', encoding='utf-8') as f:
            json.dump(presets, f, indent=2)

        self.log(f"Preset '{name}' salvo!")
        messagebox.showinfo("Sucesso", f"Preset '{name}' salvo com sucesso!")

    def load_subtitle_preset(self):
        """Carrega preset de legendas."""
        if not os.path.exists(SUBTITLE_PRESETS_FILE):
            messagebox.showinfo("Aviso", "Nenhum preset salvo ainda.")
            return

        try:
            with open(SUBTITLE_PRESETS_FILE, 'r', encoding='utf-8') as f:
                presets = json.load(f)
        except:
            messagebox.showerror("Erro", "Erro ao ler presets.")
            return

        if not presets:
            messagebox.showinfo("Aviso", "Nenhum preset salvo ainda.")
            return

        # Criar dialog de selecao
        names = list(presets.keys())

        # Dialog simples com combobox
        dialog = ctk.CTkToplevel(self)
        dialog.title("Carregar Preset")
        dialog.geometry("300x150")
        dialog.transient(self)
        dialog.grab_set()

        ctk.CTkLabel(dialog, text="Selecione o preset:", text_color=CORES["text"]).pack(pady=10)
        combo = ctk.CTkComboBox(dialog, values=names, width=200)
        combo.pack(pady=5)
        combo.set(names[0])

        def apply():
            selected = combo.get()
            if selected in presets:
                preset = presets[selected]
                self.sub_font_var.set(preset.get("font_name", "Arial"))
                self.sub_font_size_var.set(preset.get("font_size", 48))
                self.sub_color_primary_var.set(preset.get("color_primary", "#FFFFFF"))
                self.sub_color_outline_var.set(preset.get("color_outline", "#000000"))
                self.sub_color_shadow_var.set(preset.get("color_shadow", "#80000000"))
                self.sub_color_karaoke_var.set(preset.get("color_karaoke", "#FFFF00"))
                self.sub_outline_size_var.set(preset.get("outline_size", 2))
                self.sub_shadow_size_var.set(preset.get("shadow_size", 2))
                self.sub_use_karaoke_var.set(preset.get("use_karaoke", True))
                self.sub_position_var.set(preset.get("position", "2"))
                self.log(f"Preset '{selected}' carregado!")
            dialog.destroy()

        ctk.CTkButton(
            dialog, text="Aplicar",
            fg_color=CORES["accent"], hover_color=CORES["accent_hover"],
            command=apply
        ).pack(pady=15)

    def create_buttons_section(self):
        """Secao de botoes premium."""
        btn_frame = ctk.CTkFrame(self.main_scroll, fg_color=CORES["bg_card"], corner_radius=12, border_width=1, border_color=CORES["border"])
        btn_frame.pack(fill="x", pady=12, padx=10)

        # Container interno para centralizar
        inner_frame = ctk.CTkFrame(btn_frame, fg_color="transparent")
        inner_frame.pack(pady=20)

        self.render_btn = ctk.CTkButton(
            inner_frame,
            text="▶  INICIAR LOTE",
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color=CORES["success"],
            hover_color="#2EA043",
            text_color="#FFFFFF",
            height=50,
            width=200,
            corner_radius=10,
            command=self.start_batch_render
        )
        self.render_btn.pack(side="left", padx=15)

        self.cancel_btn = ctk.CTkButton(
            inner_frame,
            text="⏹  CANCELAR",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=CORES["bg_hover"],
            hover_color=CORES["error"],
            text_color=CORES["text_dim"],
            height=50,
            width=150,
            corner_radius=10,
            state="disabled",
            command=self.cancel_batch
        )
        self.cancel_btn.pack(side="left", padx=15)

    def create_progress_section(self):
        """Secao de progresso premium com barras individual e total."""
        progress_frame = ctk.CTkFrame(self.main_scroll, fg_color=CORES["bg_section"], corner_radius=12, border_width=1, border_color=CORES["border"])
        progress_frame.pack(fill="x", pady=8, padx=10)

        # Titulo da secao
        ctk.CTkLabel(
            progress_frame,
            text="📊  PROGRESSO",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=CORES["text"]
        ).pack(anchor="w", padx=15, pady=(15, 10))

        # Status da fila em card destacado
        status_card = ctk.CTkFrame(progress_frame, fg_color=CORES["bg_card"], corner_radius=8)
        status_card.pack(fill="x", padx=15, pady=(0, 10))
        
        self.queue_label = ctk.CTkLabel(
            status_card,
            text="📦 Fila: 0 vídeos  •  ✅ Concluídos: 0/0",
            text_color=CORES["text"],
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.queue_label.pack(anchor="w", padx=15, pady=12)

        # Card do video atual
        current_card = ctk.CTkFrame(progress_frame, fg_color=CORES["bg_card"], corner_radius=8)
        current_card.pack(fill="x", padx=15, pady=(0, 10))

        ctk.CTkLabel(
            current_card,
            text="Vídeo atual:",
            text_color=CORES["text_dim"],
            font=ctk.CTkFont(size=11)
        ).pack(anchor="w", padx=15, pady=(12, 2))

        self.current_video_label = ctk.CTkLabel(
            current_card,
            text="Aguardando...",
            text_color=CORES["accent"],
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.current_video_label.pack(anchor="w", padx=15, pady=(0, 8))

        self.progress_bar_current = ctk.CTkProgressBar(
            current_card,
            height=12,
            progress_color=CORES["accent"],
            fg_color=CORES["bg_dark"],
            corner_radius=6
        )
        self.progress_bar_current.pack(fill="x", padx=15, pady=(0, 12))
        self.progress_bar_current.set(0)

        # Card do progresso total
        total_card = ctk.CTkFrame(progress_frame, fg_color=CORES["bg_card"], corner_radius=8)
        total_card.pack(fill="x", padx=15, pady=(0, 15))

        ctk.CTkLabel(
            total_card,
            text="Progresso total:",
            text_color=CORES["text_dim"],
            font=ctk.CTkFont(size=11)
        ).pack(anchor="w", padx=15, pady=(12, 5))

        self.progress_bar_total = ctk.CTkProgressBar(
            total_card,
            height=18,
            progress_color=CORES["success"],
            fg_color=CORES["bg_dark"],
            corner_radius=9
        )
        self.progress_bar_total.pack(fill="x", padx=15, pady=(0, 5))
        self.progress_bar_total.set(0)

        self.progress_label_total = ctk.CTkLabel(
            total_card,
            text="⏳ Aguardando início...",
            text_color=CORES["text_dim"],
            font=ctk.CTkFont(size=11)
        )
        self.progress_label_total.pack(pady=(0, 12))

    def create_log_section(self):
        """Secao de log premium com syntax highlighting."""
        log_frame = ctk.CTkFrame(self.main_scroll, fg_color=CORES["bg_section"], corner_radius=12, border_width=1, border_color=CORES["border"])
        log_frame.pack(fill="both", expand=True, pady=8, padx=10)

        # Header do log
        log_header = ctk.CTkFrame(log_frame, fg_color="transparent")
        log_header.pack(fill="x", padx=15, pady=(15, 8))

        ctk.CTkLabel(
            log_header,
            text="📋  LOG EM TEMPO REAL",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=CORES["text"]
        ).pack(side="left")

        # Botao para limpar log
        clear_btn = ctk.CTkButton(
            log_header,
            text="🗑️ Limpar",
            width=80,
            height=28,
            font=ctk.CTkFont(size=11),
            fg_color=CORES["bg_hover"],
            hover_color=CORES["error"],
            text_color=CORES["text_dim"],
            corner_radius=6,
            command=lambda: self.log_text.delete("1.0", "end")
        )
        clear_btn.pack(side="right")

        # Botao para copiar log
        copy_btn = ctk.CTkButton(
            log_header,
            text="📋 Copiar Log",
            width=100,
            height=28,
            font=ctk.CTkFont(size=11),
            fg_color=CORES["accent"],
            hover_color=CORES["accent_hover"],
            text_color=CORES["text"],
            corner_radius=6,
            command=self.copy_log_to_clipboard
        )
        copy_btn.pack(side="right", padx=(0, 8))

        # Terminal-like log area
        self.log_text = ctk.CTkTextbox(
            log_frame,
            height=220,
            fg_color=CORES["bg_dark"],
            text_color=CORES["text"],
            font=ctk.CTkFont(family="Consolas", size=11),
            corner_radius=8,
            border_width=1,
            border_color=CORES["border"]
        )
        self.log_text.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        
        # Configurar tags para syntax highlighting
        self.log_text.tag_config("OK", foreground=CORES["success"])
        self.log_text.tag_config("ERROR", foreground=CORES["error"])
        self.log_text.tag_config("WARN", foreground=CORES["warning"])
        self.log_text.tag_config("INFO", foreground=CORES["accent"])

    def create_about_section(self):
        """Seção de informações sobre versão e funcionalidades."""
        section = ctk.CTkFrame(self.main_scroll, fg_color=CORES["bg_section"], corner_radius=12, border_width=1, border_color=CORES["border"])
        section.pack(fill="x", pady=8, padx=10)

        # Título com checkbox para expandir/recolher
        title_frame = ctk.CTkFrame(section, fg_color="transparent")
        title_frame.pack(fill="x", padx=15, pady=(15, 8))

        self.about_expanded_var = ctk.BooleanVar(value=False)
        
        expand_btn = ctk.CTkCheckBox(
            title_frame,
            text="ℹ️  SOBRE / VERSÃO",
            variable=self.about_expanded_var,
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=CORES["text"],
            fg_color=CORES["accent"],
            hover_color=CORES["accent_hover"],
            command=self.toggle_about_panel
        )
        expand_btn.pack(side="left")

        # Frame de conteúdo (oculto por padrão)
        self.about_content_frame = ctk.CTkFrame(section, fg_color="transparent")

        # Informações da versão
        version_info = """
═══════════════════════════════════════════════════════════════
                    RENDERX v3.2
        Desenvolvido por Equipe Matrix
═══════════════════════════════════════════════════════════════

VERSÃO ATUAL: 3.2
Data de Lançamento: Janeiro 2025

═══════════════════════════════════════════════════════════════
                    FUNCIONALIDADES
═══════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────┐
│ VERSÃO 3.2 - DETECÇÃO DE HARDWARE E SETUP RÁPIDO            │
└─────────────────────────────────────────────────────────────┘

✓ DETECÇÃO AUTOMÁTICA DE HARDWARE (v3.2)
  • CPU: Nome, modelo e número de threads
  • GPU: Detecção de NVIDIA via nvidia-smi
  • RAM: Memória total do sistema
  • Exibição em tempo real na interface

✓ CONFIGURAÇÃO RECOMENDADA (v3.2)
  • Cálculo inteligente baseado no hardware
  • Reserva 25% dos threads para o sistema
  • Considera RAM e presença de GPU NVIDIA
  • Botão "Aplicar" para configurar automaticamente

✓ CRIAR PASTAS DO LOTE (v3.2)
  • Botão no topo da seção de pastas
  • Cria automaticamente: MATERIAL, SAIDA, IMAGENS ORACULO
  • Define caminhos automaticamente após criação
  • Setup rápido para novos projetos

┌─────────────────────────────────────────────────────────────┐
│ VERSÃO 3.1 - PIPELINE SRT AVANÇADO                          │
└─────────────────────────────────────────────────────────────┘

✓ MODOS DE LEGENDA (v3.1)
  • Full: Legendas queimadas no vídeo (burn-in)
  • Sem: Exporta SRT separado (soft subtitles)
  • Escolha via interface

✓ SLIDER DE FRASES POR IMAGEM (v3.1)
  • Configurável de 1 a 10 cues por imagem
  • Troca de imagem baseada em blocos SRT
  • Sincronização precisa com narrativa

✓ 14 ANIMAÇÕES KEN BURNS (v3.1)
  • Pan básico (4 direções): A, B, C, D
  • Pan + Zoom In (4 direções): E, F, G, H
  • Pan + Zoom Out (4 direções): I, J, K, L
  • Rotação leve (2-5 graus): M
  • Punch-in (zoom rápido): N
  • Shuffle inteligente SEM repetição consecutiva
  • Hard Cut entre imagens

✓ PROMPTS EXTERNOS (v3.1)
  • Arquivo externo (.md ou .txt)
  • Cache SHA256 para evitar regeneração
  • Configuração persistente

✓ SISTEMA DE OVERLAYS (v3.1)
  • Pasta com overlays do usuário
  • Suporta PNG (com alpha) e vídeos (MP4/MOV/WEBM)
  • 1 overlay por vídeo (shuffle global)
  • Histórico para evitar repetição

┌─────────────────────────────────────────────────────────────┐
│ VERSÃO 3.0 - FUNCIONALIDADES                                │
└─────────────────────────────────────────────────────────────┘

✓ GERAÇÃO DE IMAGENS VIA SRT (v3.0)
  • Agrupamento automático de blocos SRT
  • Geração de imagens via API Whisk (Google Imagen 3.5)
  • Seleção de imagens do backlog baseada em SRT
  • Prompts automáticos com estilo "Espiritual/Chosen Ones"
  • Sincronização precisa com timestamps do SRT

✓ BACKLOG DE ÁUDIOS DE FUNDO (v3.0)
  • Sistema inteligente de rotação de áudios
  • Evita repetição dos últimos 10 áudios usados
  • Histórico persistente em JSON
  • Apenas para música/efeitos de fundo
  • Áudios de narração não são afetados

┌─────────────────────────────────────────────────────────────┐
│ VERSÃO 2.0 - FUNCIONALIDADES BASE                           │
└─────────────────────────────────────────────────────────────┘

✓ MODO LOTE (v2.0)
  • Processamento de múltiplos áudios de uma pasta
  • Varredura recursiva (subpastas suportadas)
  • Replicação automática da estrutura de pastas na saída

✓ PARALELISMO (v2.0)
  • 1-4 vídeos renderizados simultaneamente
  • 2-8 threads por vídeo
  • Otimizado para hardware de alto desempenho

✓ IMAGENS EXCLUSIVAS (v2.0)
  • Sistema de reserva de imagens
  • Imagens usadas movidas para pasta UTILIZADAS
  • Garantia de imagens únicas por vídeo

✓ ZOOM CENTRALIZADO (v2.0)
  • Zoom com âncora no centro (getRotationMatrix2D)
  • Smart Crop 16:9 (sem bordas pretas)
  • Modos: Zoom In / Zoom Out
  • Escala configurável

✓ PIPELINE PARALELO (v2.0)
  • Geração de clips e fades simultâneos
  • Stitching assíncrono em batches
  • Otimização de performance

✓ MODO LOOP (v2.0)
  • Modo de imagens fixas com loop
  • Configurável (padrão: 40 imagens)
  • Loop automático até duração do áudio

✓ OVERLAY (v2.0)
  • Efeito de poeira/partículas
  • Blend screen
  • Opacidade configurável

✓ MIXAGEM DE ÁUDIO (v2.0)
  • Narração + música de fundo
  • Volume configurável para música
  • Loop automático da música

┌─────────────────────────────────────────────────────────────┐
│ LEGENDAS E SUBTÍTULOS (v2.0)                                 │
└─────────────────────────────────────────────────────────────┘

✓ Geração via AssemblyAI
✓ Importação de arquivos SRT
✓ Personalização completa:
  - Fonte, tamanho, cores
  - Contorno, sombra
  - Efeito karaoke
  - Posicionamento (9 posições)
✓ Presets salvos/carregados

┌─────────────────────────────────────────────────────────────┐
│ VSL - VIDEO SALES LETTER (v2.0)                            │
└─────────────────────────────────────────────────────────────┘

✓ Inserção automática de VSLs
✓ Detecção por palavras-chave
✓ Suporte a múltiplos idiomas
✓ Fallback configurável

┌─────────────────────────────────────────────────────────────┐
│ BACKLOG VIDEOS - INTRO (v2.0)                               │
└─────────────────────────────────────────────────────────────┘

✓ Vídeos introdutórios do backlog
✓ Duração fixa de 60 segundos
✓ Crossfade entre vídeos
✓ Fade out no último vídeo
✓ Overlay aplicável

┌─────────────────────────────────────────────────────────────┐
│ TTS - TEXT TO SPEECH (v2.0)                                 │
└─────────────────────────────────────────────────────────────┘

✓ Integração com DarkVie API
✓ Integração com Talkify API
✓ Listagem de vozes disponíveis
✓ Geração de áudio a partir de texto
✓ Pasta configurável para áudios gerados

═══════════════════════════════════════════════════════════════
                    CONFIGURAÇÕES
═══════════════════════════════════════════════════════════════

• Resolução: 720p / 1080p
• FPS: 24
• Formatos de saída: MP4 (H.264)
• Codec de vídeo: NVENC (GPU) / libx264 (CPU)
• Codec de áudio: AAC

═══════════════════════════════════════════════════════════════
                    HARDWARE RECOMENDADO
═══════════════════════════════════════════════════════════════

• CPU: Ryzen 7 9800X3D (16 threads)
• GPU: RTX 5070 Ti (16GB VRAM, NVENC)
• RAM: 64GB
• Armazenamento: SSD recomendado

═══════════════════════════════════════════════════════════════
                    DEPENDÊNCIAS
═══════════════════════════════════════════════════════════════

• Python 3.14+
• OpenCV (cv2)
• NumPy
• CustomTkinter
• Pillow (PIL)
• httpx
• requests
• FFmpeg (sistema)

═══════════════════════════════════════════════════════════════
                    CHANGELOG
═══════════════════════════════════════════════════════════════

v3.2 (Janeiro 2025)
  + Detecção automática de hardware real (CPU, GPU, RAM)
  + Configuração recomendada baseada no hardware detectado
  + Botão "Aplicar" para usar configuração ideal automaticamente
  + Botão "Criar Pastas do Lote" para setup rápido
  + Criação automática das pastas MATERIAL, SAIDA e IMAGENS ORACULO
  + Caminhos configurados automaticamente após criar pastas
  + Correções de versão na interface

v3.1 (Dezembro 2024)
  + Modos de legenda: Full (burn-in) ou Sem (SRT separado)
  + Slider de frases por imagem (1-10 cues)
  + 14 animações Ken Burns (pan, zoom, rotação, punch-in)
  + Suporte a prompts externos (.md/.txt)
  + Sistema de overlays (PNG com alpha, vídeos MP4/MOV/WEBM)
  + Histórico de overlays para evitar repetição

v3.0 (Dezembro 2024)
  + Geração de imagens baseada em SRT
  + Integração com API Whisk (Google Imagen 3.5)
  + Animações variadas Ken Burns (6 tipos)
  + Backlog de áudios de fundo
  + Hard cut entre imagens
  + Sincronização precisa com timestamps SRT

v2.0 (Versão Base)
  + Modo lote
  + Paralelismo
  + Imagens exclusivas
  + Zoom centralizado
  + Pipeline paralelo
  + Legendas e VSL
  + TTS integration

═══════════════════════════════════════════════════════════════
        """

        # Textbox com scroll para informações
        about_textbox = ctk.CTkTextbox(
            self.about_content_frame,
            height=400,
            fg_color=CORES["bg_dark"],
            text_color=CORES["text"],
            font=ctk.CTkFont(family="Consolas", size=10),
            wrap="word"
        )
        about_textbox.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        about_textbox.insert("1.0", version_info)
        about_textbox.configure(state="disabled")  # Somente leitura

        # Mostrar/ocultar conforme checkbox
        if self.about_expanded_var.get():
            self.about_content_frame.pack(fill="both", expand=True)

    def toggle_about_panel(self):
        """Mostra/oculta painel de informações sobre versão."""
        if self.about_expanded_var.get():
            self.about_content_frame.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        else:
            self.about_content_frame.pack_forget()

    def create_footer(self):
        """Cria footer premium com status do sistema."""
        footer = ctk.CTkFrame(self, fg_color=CORES["bg_card"], corner_radius=0, height=36)
        footer.grid(row=2, column=0, sticky="ew")
        footer.grid_propagate(False)
        footer.grid_columnconfigure(1, weight=1)

        # Status GPU
        gpu_status = self.check_gpu_status()
        gpu_color = CORES["success"] if gpu_status["available"] else CORES["text_dim"]
        gpu_text = f"🖥️ GPU: {gpu_status['name']}" if gpu_status["available"] else "🖥️ GPU: CPU Mode"
        
        gpu_label = ctk.CTkLabel(
            footer,
            text=gpu_text,
            font=ctk.CTkFont(size=11),
            text_color=gpu_color
        )
        gpu_label.grid(row=0, column=0, padx=(15, 20), pady=8)

        # Separador
        ctk.CTkLabel(
            footer,
            text="•",
            text_color=CORES["text_muted"]
        ).grid(row=0, column=1, padx=5, sticky="w")

        # Memoria
        memory_info = self.get_memory_info()
        memory_label = ctk.CTkLabel(
            footer,
            text=f"💾 RAM: {memory_info}",
            font=ctk.CTkFont(size=11),
            text_color=CORES["text_dim"]
        )
        memory_label.grid(row=0, column=2, padx=5, pady=8, sticky="w")

        # Versao (direita)
        version_label = ctk.CTkLabel(
            footer,
            text="RenderX v3.2  •  Matrix Team",
            font=ctk.CTkFont(size=10),
            text_color=CORES["text_muted"]
        )
        version_label.grid(row=0, column=3, padx=(0, 15), pady=8, sticky="e")

    def check_gpu_status(self):
        """Verifica status da GPU NVIDIA."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                return {"available": True, "name": result.stdout.strip().split('\n')[0][:20]}
        except:
            pass
        return {"available": False, "name": "N/A"}

    def get_memory_info(self):
        """Obtem informacao de memoria do sistema."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            used_gb = mem.used / (1024**3)
            total_gb = mem.total / (1024**3)
            return f"{used_gb:.1f}/{total_gb:.1f} GB"
        except:
            return "N/A"

    def create_slider(self, parent, label, variable, min_val, max_val, suffix, is_int=False, is_percent=False):
        """Cria slider estilizado."""
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", padx=15, pady=3)

        ctk.CTkLabel(frame, text=label, text_color=CORES["text"], width=140).pack(side="left")

        slider = ctk.CTkSlider(
            frame,
            from_=min_val,
            to=max_val,
            variable=variable,
            width=200,
            progress_color=CORES["accent"],
            button_color=CORES["accent"],
            button_hover_color=CORES["accent_hover"]
        )
        slider.pack(side="left", fill="x", expand=True, padx=10)

        # Valor inicial formatado
        val = variable.get()
        if is_percent:
            text = f"{val:.0%}"
        elif is_int:
            text = f"{int(val)}"
        else:
            text = f"{val:.2f}{suffix}"

        value_label = ctk.CTkLabel(
            frame, text=text,
            text_color=CORES["accent"],
            width=60
        )
        value_label.pack(side="left")

        # Callback para atualizar label
        def update_label(val):
            if isinstance(val, str):
                val = float(val)
            if is_percent:
                value_label.configure(text=f"{val:.0%}")
            elif is_int:
                value_label.configure(text=f"{int(val)}")
            else:
                value_label.configure(text=f"{val:.2f}{suffix}")

        slider.configure(command=update_label)

    def select_file(self, file_type):
        """Abre dialogo para selecionar arquivo."""
        if file_type == "music":
            path = filedialog.askopenfilename(
                title="Selecionar Musica de Fundo",
                filetypes=[("Audio", "*.mp3 *.wav *.m4a *.aac"), ("Todos", "*.*")]
            )
            if path:
                self.music_var.set(path)
        elif file_type == "overlay":
            path = filedialog.askopenfilename(
                title="Selecionar Overlay",
                filetypes=[("Video", "*.mp4 *.mov *.webm"), ("Todos", "*.*")]
            )
            if path:
                self.overlay_var.set(path)

    def create_batch_folders(self):
        """Cria as pastas do lote dentro da pasta da ferramenta e seta os caminhos."""
        try:
            # Pastas a serem criadas dentro da pasta da ferramenta
            folders = {
                "MATERIAL": self.batch_input_var,
                "SAIDA": self.batch_output_var,
                "IMAGENS ORACULO": self.batch_images_var
            }
            
            created = []
            for folder_name, var in folders.items():
                folder_path = SCRIPT_DIR / folder_name
                folder_path.mkdir(parents=True, exist_ok=True)
                var.set(str(folder_path))
                created.append(folder_name)
            
            self.log(f"✅ Pastas criadas com sucesso: {', '.join(created)}", "INFO")
            messagebox.showinfo(
                "Pastas Criadas",
                f"As seguintes pastas foram criadas em:\n{SCRIPT_DIR}\n\n"
                f"• MATERIAL (Pasta dos Materiais)\n"
                f"• SAIDA (Pasta de Saída)\n"
                f"• IMAGENS ORACULO (Banco de Imagens)\n\n"
                f"Os caminhos já foram configurados automaticamente."
            )
            
        except Exception as e:
            self.log(f"❌ Erro ao criar pastas: {str(e)}", "ERROR")
            messagebox.showerror("Erro", f"Não foi possível criar as pastas:\n{str(e)}")

    def select_batch_folder(self, folder_type):
        """Abre dialogo para selecionar pasta do lote."""
        titles = {
            "input": "Selecionar Pasta dos Materiais",
            "output": "Selecionar Pasta de Saida",
            "images": "Selecionar Banco de Imagens"
        }
        path = filedialog.askdirectory(title=titles.get(folder_type, "Selecionar Pasta"))
        if path:
            if folder_type == "input":
                self.batch_input_var.set(path)
            elif folder_type == "output":
                self.batch_output_var.set(path)
            elif folder_type == "images":
                self.batch_images_var.set(path)

    def log(self, message, level="INFO"):
        """Adiciona mensagem ao log (thread-safe via queue)."""
        # Formatar mensagem com level se fornecido
        if level and level != "INFO":
            formatted = f"[{level}] {message}"
        else:
            formatted = message
        self.log_queue.put(formatted)

    def process_log_queue(self):
        """Processa mensagens da fila de log com syntax highlighting."""
        while True:
            try:
                message = self.log_queue.get_nowait()
                
                # Determinar tag baseado no conteudo
                tag = None
                if "[OK]" in message or "✓" in message or "sucesso" in message.lower():
                    tag = "OK"
                elif "[X]" in message or "[ERROR]" in message or "erro" in message.lower():
                    tag = "ERROR"
                elif "[!]" in message or "[WARN]" in message or "aviso" in message.lower():
                    tag = "WARN"
                elif "[INFO]" in message or ">>" in message:
                    tag = "INFO"
                
                # Inserir com tag se aplicavel
                if tag:
                    self.log_text.insert("end", message + "\n", tag)
                else:
                    self.log_text.insert("end", message + "\n")
                    
                self.log_text.see("end")
            except queue.Empty:
                break
        self.after(100, self.process_log_queue)

    def copy_log_to_clipboard(self):
        """Copia todo o conteúdo do log para a área de transferência."""
        log_content = self.log_text.get("1.0", "end-1c")
        if log_content.strip():
            self.clipboard_clear()
            self.clipboard_append(log_content)
            # Feedback visual temporário
            self.log("📋 Log copiado para a área de transferência!", "INFO")
        else:
            messagebox.showinfo("Log Vazio", "Não há conteúdo no log para copiar.")

    def scan_folders(self):
        """Escaneia pastas e atualiza contadores."""
        input_folder = self.batch_input_var.get()
        images_folder = self.batch_images_var.get()

        # Validar pastas
        errors = []
        if not input_folder or not os.path.isdir(input_folder):
            errors.append("Pasta dos materiais invalida")
        if not images_folder or not os.path.isdir(images_folder):
            errors.append("Banco de imagens invalido")

        if errors:
            self.scan_status_label.configure(
                text=" | ".join(errors),
                text_color=CORES["error"]
            )
            return

        # Contar arquivos (recursivo)
        self.audio_count = 0
        self.txt_count = 0
        self.audio_to_generate_count = 0
        
        # Agrupar por nome base para identificar o que precisa ser gerado
        file_groups = {}
        for root, dirs, files in os.walk(input_folder):
            for f in files:
                base_name = os.path.splitext(f)[0]
                ext = os.path.splitext(f)[1].lower()
                key = (base_name, os.path.relpath(root, input_folder))
                
                if key not in file_groups:
                    file_groups[key] = {"txt": False, "audio": False}
                
                if ext in TEXT_EXTENSIONS:
                    self.txt_count += 1
                    file_groups[key]["txt"] = True
                elif ext in AUDIO_EXTENSIONS:
                    self.audio_count += 1
                    file_groups[key]["audio"] = True
        
        # Contar quantos txt precisam gerar áudio (sempre contar, mesmo se TTS desabilitado)
        for group in file_groups.values():
            if group["txt"] and not group["audio"]:
                self.audio_to_generate_count += 1

        # Contar imagens (apenas pasta principal, excluindo UTILIZADAS)
        self.images_count = 0
        utilized_folder = os.path.join(images_folder, "UTILIZADAS")
        for f in os.listdir(images_folder):
            full_path = os.path.join(images_folder, f)
            if os.path.isdir(full_path):
                continue
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                self.images_count += 1

        # Calcular total de videos (audios existentes + audios a gerar)
        total_videos = self.audio_count + self.audio_to_generate_count
        imgs_needed = total_videos * self.images_per_video_var.get()
        status_color = CORES["success"] if self.images_count >= imgs_needed else CORES["warning"]

        status_text = f"Audios: {self.audio_count} | TXTs: {self.txt_count}"
        if self.audio_to_generate_count > 0:
            if self.tts_enabled_var.get():
                status_text += f" | A gerar: {self.audio_to_generate_count}"
            else:
                status_text += f" | TXTs sem audio: {self.audio_to_generate_count} (habilite TTS)"
        status_text += f" | Imagens: {self.images_count} | Necessarias: ~{imgs_needed}"

        self.scan_status_label.configure(
            text=status_text,
            text_color=status_color
        )
        self.log(f"Scan: {self.audio_count} audios, {self.txt_count} txts, {self.audio_to_generate_count} a gerar, {self.images_count} imagens")

    def validate_batch_inputs(self):
        """Valida entradas antes de iniciar lote."""
        if not self.batch_input_var.get() or not os.path.isdir(self.batch_input_var.get()):
            messagebox.showerror("Erro", "Selecione uma pasta dos materiais valida!")
            return False
        if not self.batch_output_var.get():
            messagebox.showerror("Erro", "Selecione uma pasta de saida!")
            return False
        if not self.batch_images_var.get() or not os.path.isdir(self.batch_images_var.get()):
            messagebox.showerror("Erro", "Selecione um banco de imagens valido!")
            return False

        # Criar pasta de saida se nao existir
        os.makedirs(self.batch_output_var.get(), exist_ok=True)

        return True

    def scan_batch_folder(self):
        """Varre pasta de entrada buscando audios e txts, faz correspondencia por nome base."""
        input_root = self.batch_input_var.get()
        output_root = self.batch_output_var.get()
        audio_folder_name = self.generated_audio_folder_var.get()
        jobs = []

        # Dicionario para agrupar arquivos por nome base
        # key: (nome_base, rel_path) -> {txt_path, audio_path}
        file_groups = {}

        # Escanear todos os arquivos (incluindo subpastas de áudios gerados)
        for root, dirs, files in os.walk(input_root):
            rel_path = os.path.relpath(root, input_root)
            if rel_path == ".":
                rel_path_key = ""
            else:
                rel_path_key = rel_path

            for f in files:
                base_name = os.path.splitext(f)[0]
                ext = os.path.splitext(f)[1].lower()
                full_path = os.path.join(root, f)
                
                # Determinar rel_path base (sem subpasta de áudios gerados)
                # Se está em subpasta de áudios gerados, normalizar para o rel_path do txt
                is_in_generated_folder = False
                if audio_folder_name and audio_folder_name:
                    # Verificar se está dentro de uma subpasta com o nome configurado
                    path_parts = rel_path_key.split(os.path.sep)
                    if audio_folder_name in path_parts:
                        is_in_generated_folder = True
                        # Remover a parte da subpasta de áudios gerados
                        base_rel_path_parts = []
                        skip_next = False
                        for part in path_parts:
                            if skip_next:
                                skip_next = False
                                continue
                            if part == audio_folder_name:
                                continue
                            base_rel_path_parts.append(part)
                        base_rel_path = os.path.sep.join(base_rel_path_parts) if base_rel_path_parts else ""
                    else:
                        base_rel_path = rel_path_key
                else:
                    base_rel_path = rel_path_key
                
                key = (base_name, base_rel_path)
                
                if key not in file_groups:
                    file_groups[key] = {"txt_path": None, "audio_path": None, "rel_path": base_rel_path}
                
                if ext in TEXT_EXTENSIONS:
                    file_groups[key]["txt_path"] = full_path
                elif ext in AUDIO_EXTENSIONS:
                    # Prioridade: áudio na raiz > áudio em subpasta de gerados
                    current_audio = file_groups[key]["audio_path"]
                    if not current_audio:
                        file_groups[key]["audio_path"] = full_path
                    elif is_in_generated_folder:
                        # Novo está em subpasta, manter o anterior se existir
                        pass
                    else:
                        # Novo está na raiz ou outra subpasta, substituir se anterior estava em subpasta gerada
                        if current_audio and audio_folder_name:
                            current_parts = current_audio.split(os.path.sep)
                            if audio_folder_name in current_parts:
                                # Substituir áudio gerado por áudio na raiz
                                file_groups[key]["audio_path"] = full_path

        # Processar grupos e criar jobs
        for (base_name, rel_path_key), group in file_groups.items():
            txt_path = group["txt_path"]
            audio_path = group["audio_path"]
            rel_path = group["rel_path"]

            # Se tem áudio, usar áudio (ignorar txt se existir)
            if audio_path:
                # Calcular caminho de saida mantendo estrutura
                if rel_path == "":
                    output_subdir = output_root
                else:
                    output_subdir = os.path.join(output_root, rel_path)

                output_path = os.path.join(output_subdir, f"{base_name}_final.mp4")

                # Pular se ja existe
                if os.path.exists(output_path):
                    self.log(f"[SKIP] Ja existe: {os.path.basename(output_path)}")
                    continue

                job = BatchJob(audio_path, os.path.dirname(audio_path), output_path, txt_path=txt_path)
                jobs.append(job)
            
            # Se tem apenas txt (sem áudio), será gerado via API depois
            elif txt_path:
                # Este será processado em generate_missing_audios()
                # Não criar job ainda, apenas marcar para geração
                pass

        return jobs

    def build_config_for_job(self, job, images):
        """Constroi configuracao para um job especifico."""
        music_path = self.music_var.get()
        if music_path:
            self.log(f"Config música: {os.path.basename(music_path)} (vol: {self.music_volume_var.get():.0%})", "DEBUG")
        else:
            self.log("Config música: Nenhuma música configurada", "DEBUG")
        
        return {
            "audio_path": job.audio_path,
            "images_list": images,  # Lista de imagens pre-selecionadas
            "output_path": job.output_path,
            "music_path": music_path,
            "overlay_path": self.overlay_var.get(),
            "resolution": self.resolution_var.get(),
            "zoom_mode": self.zoom_mode_var.get(),
            "zoom_scale": self.zoom_scale_var.get(),
            "image_duration": self.duration_var.get(),
            "transition_duration": self.transition_var.get(),
            "threads": int(self.threads_per_video_var.get()),
            "fps": 24,
            "video_bitrate": self.video_bitrate_var.get(),
            "music_volume": self.music_volume_var.get(),
            "overlay_opacity": self.overlay_opacity_var.get(),
            "use_fixed_images": True,  # Sempre usa imagens fixas no lote
            "fixed_images_count": len(images),
            # Legendas
            "use_subtitles": self.use_subtitles_var.get(),
            "subtitle_method": self.subtitle_method_var.get(),
            "srt_source": self.subtitle_method_var.get(),  # Usar subtitle_method como srt_source
            "srt_path": self.srt_path_var.get(),
            "assemblyai_key": self.assemblyai_key_var.get(),
            # Modo de Vídeo
            "video_mode": self.video_mode_var.get(),
            "use_srt_based_images": self.video_mode_var.get() == "srt",  # Compatibilidade
            "image_source": self.image_source_var.get(),
            "images_backlog_folder": self.images_backlog_folder_var.get(),
            "whisk_api_tokens": get_enabled_tokens(),  # Carregado do whisk_keys.json
            "whisk_parallel_workers": self.whisk_workers_var.get(),  # Workers paralelos (0 = auto)
            "selected_prompt_id": self.get_selected_prompt_id(),  # Prompt de imagem selecionado
            "use_varied_animations": self.use_varied_animations_var.get(),
            "enabled_effects": self.get_enabled_effects(),  # Efeitos de animação selecionados
            "pan_amount": self.pan_amount_var.get(),
            # Pipeline SRT v3.1
            "subtitle_mode": self.subtitle_mode_var.get(),
            "swap_every_n_cues": self.swap_every_n_cues_var.get(),
            "prompt_file": self.prompt_file_var.get(),
            "overlay_folder": self.overlay_folder_var.get(),
            "use_random_overlays": self.use_random_overlays_var.get(),
            # Backlog de Áudios
            "audio_backlog_folder": self.audio_backlog_folder_var.get(),
            "use_audio_backlog": self.use_audio_backlog_var.get(),
            "audio_backlog_history_file": self.config.get("audio_backlog_history_file", "audio_backlog_history.json"),
            "sub_options": {
                "font_name": self.sub_font_var.get(),
                "font_size": int(self.sub_font_size_var.get()),
                "color_primary": self.sub_color_primary_var.get(),
                "color_outline": self.sub_color_outline_var.get(),
                "color_shadow": self.sub_color_shadow_var.get(),
                "color_karaoke": self.sub_color_karaoke_var.get(),
                "outline_size": int(self.sub_outline_size_var.get()),
                "shadow_size": int(self.sub_shadow_size_var.get()),
                "use_karaoke": self.sub_use_karaoke_var.get(),
                "position": self.sub_position_var.get()
            },
            # VSL (Video Sales Letter)
            "use_vsl": self.use_vsl_var.get(),
            "vsl_folder": self.config.get("vsl_folder", str(SCRIPT_DIR / "EFEITOS" / "VSLs")),
            "vsl_keywords_file": self.config.get("vsl_keywords_file", str(SCRIPT_DIR / "vsl_keywords.json")),
            "vsl_language": self.vsl_language_var.get() if hasattr(self, 'vsl_language_var') else "portugues",
            "selected_vsl": self.selected_vsl_var.get() if hasattr(self, 'selected_vsl_var') else "",
            "vsl_insertion_mode": self.vsl_insertion_mode_var.get() if hasattr(self, 'vsl_insertion_mode_var') else "keyword",
            "vsl_fixed_position": self.vsl_fixed_position_var.get() if hasattr(self, 'vsl_fixed_position_var') else 60.0,
            "vsl_range_start_min": float(self.vsl_range_start_var.get()) if hasattr(self, 'vsl_range_start_var') else 1.0,
            "vsl_range_end_min": float(self.vsl_range_end_var.get()) if hasattr(self, 'vsl_range_end_var') else 3.0,
            # Backlog Videos
            "use_backlog_videos": self.use_backlog_videos_var.get(),
            "backlog_folder": self.backlog_folder_var.get() if self.backlog_folder_var.get() else str(SCRIPT_DIR / "EFEITOS" / "BACKLOG_VIDEOS"),
            "backlog_video_count": self.config.get("backlog_video_count", 6),
            "backlog_audio_volume": self.config.get("backlog_audio_volume", 0.25),
            "backlog_transition_duration": self.config.get("backlog_transition_duration", 0.5),
            "backlog_fade_out_duration": self.config.get("backlog_fade_out_duration", 1.0),
            "text_path": getattr(job, 'txt_path', None),
            # Modo 1 Imagem (Pêndulo)
            "pendulum_amplitude": self.pendulum_amplitude_var.get(),
            "pendulum_crop_ratio": self.pendulum_crop_ratio_var.get(),
            "pendulum_zoom": self.pendulum_zoom_var.get(),
            "pendulum_cell_duration": self.pendulum_cell_duration_var.get(),
            "chroma_color": self.chroma_color_var.get(),
            "chroma_similarity": self.chroma_similarity_var.get(),
            "chroma_blend": self.chroma_blend_var.get(),
            # Banco de imagens para modo single_image
            "batch_images_folder": self.batch_images_var.get()
        }

    def update_job_progress(self, job, progress):
        """Atualiza progresso de um job especifico."""
        job.progress = progress
        # UI atualizada via after no main thread
        self.after(0, lambda: self.progress_bar_current.set(progress / 100))

    def update_batch_ui(self):
        """Atualiza UI com status do lote."""
        with self.jobs_lock:
            total = len(self.batch_jobs)
            done = sum(1 for j in self.batch_jobs if j.status == "done")
            processing = [j for j in self.batch_jobs if j.status == "processing"]

        self.queue_label.configure(text=f"Fila: {total} videos | Concluidos: {done}/{total}")

        if processing:
            current = processing[0]
            self.current_video_label.configure(text=f"{current.name} ({current.progress:.0f}%)")
        else:
            self.current_video_label.configure(text="-")

        if total > 0:
            self.progress_bar_total.set(done / total)
            self.progress_label_total.configure(text=f"{done}/{total} videos concluidos")

    def batch_video_worker(self, worker_id):
        """Worker que processa videos da fila."""
        while not self.cancel_requested:
            try:
                job = self.batch_queue.get(timeout=2)
            except queue.Empty:
                # Verificar se ainda ha jobs pendentes
                with self.jobs_lock:
                    pending = any(j.status == "pending" for j in self.batch_jobs)
                if not pending:
                    break
                continue

            if job is None:
                break

            try:
                job.status = "processing"
                job.start_time = time.time()
                self.log(f"[Worker {worker_id}] Iniciando: {job.name}")
                self.after(0, self.update_batch_ui)

                # Criar pasta de saida
                os.makedirs(os.path.dirname(job.output_path), exist_ok=True)

                # Reservar imagens exclusivas
                images_needed = self.images_per_video_var.get()
                try:
                    images = self.image_system.reserve_images(images_needed)
                    job.used_images = images
                except Exception as e:
                    job.status = "error"
                    job.error_msg = str(e)
                    self.log(f"[Worker {worker_id}] Erro ao reservar imagens: {e}")
                    continue

                # Configuracao do video
                config = self.build_config_for_job(job, images)

                # Criar engine para este job
                def progress_cb(stage, current, total, extra=""):
                    if total > 0:
                        self.update_job_progress(job, (current / total) * 100)

                engine = FinalSlideshowEngine(self.log_queue, progress_cb)

                # Renderizar
                result = engine.render_full_video(config)

                if result and os.path.exists(result):
                    job.status = "done"
                    job.end_time = time.time()
                    # Marcar imagens como usadas
                    self.image_system.release_and_mark_used(job.used_images)
                    self.log(f"[Worker {worker_id}] Concluido: {job.name} ({job.duration_str})")
                else:
                    job.status = "error"
                    job.error_msg = "Render falhou"
                    # Liberar imagens reservadas (nao mover)
                    with self.image_system.lock:
                        self.image_system.reserved -= set(job.used_images)
                    self.log(f"[Worker {worker_id}] Falhou: {job.name}")

            except Exception as e:
                job.status = "error"
                job.error_msg = str(e)
                self.log(f"[Worker {worker_id}] Erro: {e}")

            finally:
                self.batch_queue.task_done()
                self.after(0, self.update_batch_ui)

        self.log(f"[Worker {worker_id}] Encerrado")

    def generate_missing_audios(self):
        """Gera áudios faltantes a partir de arquivos .txt via API."""
        if not self.tts_enabled_var.get():
            return True

        provider = self.tts_provider_var.get()
        if provider == "none":
            self.log("TTS desabilitado ou provider não selecionado", "WARNING")
            return True

        api_key_raw = self.darkvi_api_key_var.get() if provider == "darkvi" else self.talkify_api_key_var.get()
        # Remover espaços e quebras de linha do token
        api_key = api_key_raw.strip() if api_key_raw else ""
        
        # Atualizar voice_id do combobox antes de verificar
        self.update_voice_id_from_combo()
        voice_id = self.tts_voice_id_var.get()

        if not api_key or not api_key.strip():
            self.log("API key não configurada!", "ERROR")
            messagebox.showerror("Erro", f"Configure a API key do {provider.upper()}!")
            return False

        if not voice_id or not voice_id.strip():
            self.log("Voz não selecionada!", "ERROR")
            self.log(f"Valor do combobox: {self.tts_voice_combo_var.get()}", "DEBUG")
            self.log(f"Valor do voice_id_var: {self.tts_voice_id_var.get()}", "DEBUG")
            messagebox.showerror("Erro", "Selecione uma voz primeiro!\n\nClique em 'Carregar Vozes' e selecione uma voz no dropdown.")
            return False

        # Importar TTSGenerator
        try:
            from tts_integration import TTSGenerator
        except ImportError:
            self.log("Erro ao importar tts_integration", "ERROR")
            return False

        # Validar configuração
        # Criar callback de log compatível
        def log_callback(msg, level="INFO"):
            self.log(msg, level)
        
        # Log do token (primeiros e últimos caracteres para debug)
        if api_key:
            token_preview = f"{api_key[:10]}...{api_key[-10:]}" if len(api_key) > 20 else api_key[:10] + "..."
            self.log(f"Token formatado: {token_preview} (tamanho: {len(api_key)} chars)", "DEBUG")
        
        generator = TTSGenerator(provider, api_key, log_callback)
        is_valid, error_msg = generator.validate_config()
        if not is_valid:
            self.log(f"Configuração TTS inválida: {error_msg}", "ERROR")
            messagebox.showerror("Erro", f"Configuração TTS inválida:\n{error_msg}")
            return False

        input_root = self.batch_input_var.get()
        audio_folder_name = self.generated_audio_folder_var.get()
        
        # Encontrar todos os .txt sem áudio correspondente
        txt_files_to_process = []
        file_groups = {}

        for root, dirs, files in os.walk(input_root):
            rel_path = os.path.relpath(root, input_root)
            if rel_path == ".":
                rel_path_key = ""
            else:
                rel_path_key = rel_path
            
            # Ignorar subpastas de áudios gerados para os txts
            # (não queremos pegar txts de dentro da pasta de áudios gerados)
            is_in_generated_folder = False
            if audio_folder_name:
                path_parts = rel_path_key.split(os.path.sep) if rel_path_key else []
                if audio_folder_name in path_parts:
                    is_in_generated_folder = True

            for f in files:
                base_name = os.path.splitext(f)[0]
                ext = os.path.splitext(f)[1].lower()
                full_path = os.path.join(root, f)
                
                # Para txts, usar rel_path sem a pasta de áudios gerados
                if is_in_generated_folder:
                    # Remover pasta de áudios gerados do path
                    path_parts = rel_path_key.split(os.path.sep)
                    base_rel_parts = [p for p in path_parts if p != audio_folder_name]
                    base_rel_path = os.path.sep.join(base_rel_parts) if base_rel_parts else ""
                else:
                    base_rel_path = rel_path_key
                
                key = (base_name, base_rel_path)
                
                if key not in file_groups:
                    file_groups[key] = {"txt_path": None, "audio_path": None, "rel_path": base_rel_path}
                
                if ext in TEXT_EXTENSIONS and not is_in_generated_folder:
                    file_groups[key]["txt_path"] = full_path
                elif ext in AUDIO_EXTENSIONS:
                    # Áudio encontrado - pode estar na raiz ou na subpasta de gerados
                    file_groups[key]["audio_path"] = full_path

        # Identificar txts sem áudio (verificar também na subpasta de áudios gerados)
        for (base_name, rel_path_key), group in file_groups.items():
            txt_path = group["txt_path"]
            audio_path = group["audio_path"]
            
            if txt_path and not audio_path:
                # Verificar se existe áudio na subpasta de áudios gerados
                if audio_folder_name:
                    txt_dir = os.path.dirname(txt_path)
                    possible_audio_dir = os.path.join(txt_dir, audio_folder_name)
                    possible_audio_path = os.path.join(possible_audio_dir, f"{base_name}.mp3")
                    
                    if os.path.exists(possible_audio_path):
                        self.log(f"Áudio já existe em subpasta: {base_name}.mp3", "DEBUG")
                        continue
                
                txt_files_to_process.append({
                    "txt_path": txt_path,
                    "base_name": base_name,
                    "rel_path": group["rel_path"]
                })

        if not txt_files_to_process:
            self.log("Nenhum áudio precisa ser gerado", "INFO")
            return True

        self.log(f"Gerando {len(txt_files_to_process)} áudio(s) via {provider.upper()}...", "INFO")

        # Gerar áudios
        success_count = 0
        error_count = 0

        for i, item in enumerate(txt_files_to_process, 1):
            txt_path = item["txt_path"]
            base_name = item["base_name"]
            rel_path = item["rel_path"]

            # Determinar onde salvar o áudio
            if rel_path == "":
                audio_dir = input_root
            else:
                audio_dir = os.path.join(input_root, rel_path)

            # Criar subpasta para áudios gerados se configurado
            if audio_folder_name:
                audio_dir = os.path.join(audio_dir, audio_folder_name)
                os.makedirs(audio_dir, exist_ok=True)

            audio_path = os.path.join(audio_dir, f"{base_name}.mp3")

            # Verificar se já existe (pode ter sido gerado anteriormente)
            if os.path.exists(audio_path):
                self.log(f"[{i}/{len(txt_files_to_process)}] Áudio já existe: {base_name}.mp3", "INFO")
                success_count += 1
                continue

            # Ler texto do arquivo
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                
                if not text:
                    self.log(f"[{i}/{len(txt_files_to_process)}] Arquivo vazio: {base_name}.txt", "WARNING")
                    error_count += 1
                    continue

                self.log(f"[{i}/{len(txt_files_to_process)}] Gerando áudio: {base_name}...", "INFO")

                # Gerar áudio
                title = f"{base_name} (gerado automaticamente)"
                success = generator.generate_audio(text, voice_id, audio_path, title)

                if success and os.path.exists(audio_path):
                    self.log(f"[{i}/{len(txt_files_to_process)}] Áudio gerado: {base_name}.mp3", "OK")
                    success_count += 1
                else:
                    self.log(f"[{i}/{len(txt_files_to_process)}] Falha ao gerar: {base_name}.txt", "ERROR")
                    error_count += 1

            except Exception as e:
                self.log(f"[{i}/{len(txt_files_to_process)}] Erro ao processar {base_name}.txt: {str(e)}", "ERROR")
                error_count += 1

        # Resumo
        self.log(f"Geração concluída: {success_count} sucesso, {error_count} erros", "INFO")

        if error_count > 0:
            messagebox.showwarning(
                "Aviso",
                f"Geração de áudios concluída com erros:\n"
                f"Sucesso: {success_count}\n"
                f"Erros: {error_count}"
            )

        return success_count > 0

    def _generate_missing_audios_threaded(self):
        """
        Versão thread-safe da geração de áudios.
        Esta função roda em uma thread separada para não bloquear a UI.
        """
        if not self.tts_enabled_var.get():
            return True

        provider = self.tts_provider_var.get()
        if provider == "none":
            self.log("TTS desabilitado ou provider não selecionado", "WARNING")
            return True

        api_key_raw = self.darkvi_api_key_var.get() if provider == "darkvi" else self.talkify_api_key_var.get()
        api_key = api_key_raw.strip() if api_key_raw else ""
        
        # Pegar voice_id já atualizado (foi atualizado na thread principal antes de chamar esta função)
        voice_id = self.tts_voice_id_var.get()

        if not api_key or not api_key.strip():
            self.log("API key não configurada!", "ERROR")
            self.after(0, lambda: messagebox.showerror("Erro", f"Configure a API key do {provider.upper()}!"))
            return False

        if not voice_id or not voice_id.strip():
            self.log("Voz não selecionada!", "ERROR")
            self.after(0, lambda: messagebox.showerror("Erro", "Selecione uma voz primeiro!\n\nClique em 'Carregar Vozes' e selecione uma voz no dropdown."))
            return False

        # Importar TTSGenerator
        try:
            from tts_integration import TTSGenerator
        except ImportError:
            self.log("Erro ao importar tts_integration", "ERROR")
            return False

        # Criar callback de log thread-safe (já usa queue internamente)
        def log_callback(msg, level="INFO"):
            self.log(msg, level)
        
        # Log do token (primeiros e últimos caracteres para debug)
        if api_key:
            token_preview = f"{api_key[:10]}...{api_key[-10:]}" if len(api_key) > 20 else api_key[:10] + "..."
            self.log(f"Token formatado: {token_preview} (tamanho: {len(api_key)} chars)", "DEBUG")
        
        generator = TTSGenerator(provider, api_key, log_callback)
        is_valid, error_msg = generator.validate_config()
        if not is_valid:
            self.log(f"Configuração TTS inválida: {error_msg}", "ERROR")
            self.after(0, lambda: messagebox.showerror("Erro", f"Configuração TTS inválida:\n{error_msg}"))
            return False

        input_root = self.batch_input_var.get()
        audio_folder_name = self.generated_audio_folder_var.get()
        
        # Encontrar todos os .txt sem áudio correspondente
        txt_files_to_process = []
        file_groups = {}

        for root, dirs, files in os.walk(input_root):
            rel_path = os.path.relpath(root, input_root)
            if rel_path == ".":
                rel_path_key = ""
            else:
                rel_path_key = rel_path
            
            is_in_generated_folder = False
            if audio_folder_name:
                path_parts = rel_path_key.split(os.path.sep) if rel_path_key else []
                if audio_folder_name in path_parts:
                    is_in_generated_folder = True

            for f in files:
                base_name = os.path.splitext(f)[0]
                ext = os.path.splitext(f)[1].lower()
                full_path = os.path.join(root, f)
                
                if is_in_generated_folder:
                    path_parts = rel_path_key.split(os.path.sep)
                    base_rel_parts = [p for p in path_parts if p != audio_folder_name]
                    base_rel_path = os.path.sep.join(base_rel_parts) if base_rel_parts else ""
                else:
                    base_rel_path = rel_path_key
                
                key = (base_name, base_rel_path)
                
                if key not in file_groups:
                    file_groups[key] = {"txt_path": None, "audio_path": None, "rel_path": base_rel_path}
                
                if ext in TEXT_EXTENSIONS and not is_in_generated_folder:
                    file_groups[key]["txt_path"] = full_path
                elif ext in AUDIO_EXTENSIONS:
                    file_groups[key]["audio_path"] = full_path

        # Identificar txts sem áudio
        for (base_name, rel_path_key), group in file_groups.items():
            txt_path = group["txt_path"]
            audio_path = group["audio_path"]
            
            if txt_path and not audio_path:
                if audio_folder_name:
                    txt_dir = os.path.dirname(txt_path)
                    possible_audio_dir = os.path.join(txt_dir, audio_folder_name)
                    possible_audio_path = os.path.join(possible_audio_dir, f"{base_name}.mp3")
                    
                    if os.path.exists(possible_audio_path):
                        self.log(f"Áudio já existe em subpasta: {base_name}.mp3", "DEBUG")
                        continue
                
                txt_files_to_process.append({
                    "txt_path": txt_path,
                    "base_name": base_name,
                    "rel_path": group["rel_path"]
                })

        if not txt_files_to_process:
            self.log("Nenhum áudio precisa ser gerado", "INFO")
            return True

        total_audios = len(txt_files_to_process)
        self.log(f"Gerando {total_audios} áudio(s) via {provider.upper()}...", "INFO")
        
        # Atualizar label de progresso na UI (thread-safe via after)
        self.after(0, lambda: self.progress_label_total.configure(text=f"Gerando áudios: 0/{total_audios}"))

        # Gerar áudios
        success_count = 0
        error_count = 0

        for i, item in enumerate(txt_files_to_process, 1):
            # Verificar cancelamento
            if self.cancel_requested:
                self.log("Geração de áudios cancelada pelo usuário", "WARNING")
                break
                
            txt_path = item["txt_path"]
            base_name = item["base_name"]
            rel_path = item["rel_path"]

            # Determinar onde salvar o áudio
            if rel_path == "":
                audio_dir = input_root
            else:
                audio_dir = os.path.join(input_root, rel_path)

            # Criar subpasta para áudios gerados se configurado
            if audio_folder_name:
                audio_dir = os.path.join(audio_dir, audio_folder_name)
                os.makedirs(audio_dir, exist_ok=True)

            audio_path = os.path.join(audio_dir, f"{base_name}.mp3")

            # Verificar se já existe
            if os.path.exists(audio_path):
                self.log(f"[{i}/{total_audios}] Áudio já existe: {base_name}.mp3", "INFO")
                success_count += 1
                # Atualizar progresso na UI
                progress_i = i
                self.after(0, lambda p=progress_i: self.progress_label_total.configure(text=f"Gerando áudios: {p}/{total_audios}"))
                self.after(0, lambda p=progress_i, t=total_audios: self.progress_bar_total.set(p / t))
                continue

            # Ler texto do arquivo
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                
                if not text:
                    self.log(f"[{i}/{total_audios}] Arquivo vazio: {base_name}.txt", "WARNING")
                    error_count += 1
                    continue

                self.log(f"[{i}/{total_audios}] Gerando áudio: {base_name}...", "INFO")
                
                # Atualizar UI com nome do áudio atual
                current_name = base_name
                self.after(0, lambda n=current_name: self.current_video_label.configure(text=f"Gerando: {n}..."))

                # Gerar áudio
                title = f"{base_name} (gerado automaticamente)"
                success = generator.generate_audio(text, voice_id, audio_path, title)

                if success and os.path.exists(audio_path):
                    self.log(f"[{i}/{total_audios}] Áudio gerado: {base_name}.mp3", "OK")
                    success_count += 1
                else:
                    self.log(f"[{i}/{total_audios}] Falha ao gerar: {base_name}.txt", "ERROR")
                    error_count += 1
                
                # Atualizar progresso na UI
                progress_i = i
                self.after(0, lambda p=progress_i: self.progress_label_total.configure(text=f"Gerando áudios: {p}/{total_audios}"))
                self.after(0, lambda p=progress_i, t=total_audios: self.progress_bar_total.set(p / t))

            except Exception as e:
                self.log(f"[{i}/{total_audios}] Erro ao processar {base_name}.txt: {str(e)}", "ERROR")
                error_count += 1

        # Resumo
        self.log(f"Geração concluída: {success_count} sucesso, {error_count} erros", "INFO")
        
        # Resetar barra de progresso para o render de vídeos
        self.after(0, lambda: self.progress_bar_total.set(0))
        self.after(0, lambda: self.progress_label_total.configure(text="Aguardando início..."))

        if error_count > 0:
            # Usar after para mostrar messagebox na thread principal
            self.after(0, lambda: messagebox.showwarning(
                "Aviso",
                f"Geração de áudios concluída com erros:\n"
                f"Sucesso: {success_count}\n"
                f"Erros: {error_count}"
            ))

        return success_count > 0 or error_count == 0

    def start_batch_render(self):
        """Inicia processamento em lote."""
        if not self.validate_batch_inputs():
            return

        self.save_config()
        self.log_text.delete("1.0", "end")

        # Desabilitar botões imediatamente para feedback visual
        self.render_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")

        # Se TTS está habilitado, gerar áudios em thread separada primeiro
        if self.tts_enabled_var.get():
            self.log("Verificando áudios a gerar...", "INFO")
            self.current_video_label.configure(text="Gerando áudios via API...")
            
            # Atualizar voice_id do combobox ANTES de iniciar a thread (na thread principal)
            self.update_voice_id_from_combo()
            
            # Executar geração de áudios em thread separada
            def audio_generation_thread():
                try:
                    success = self._generate_missing_audios_threaded()
                    # Voltar para a thread principal para continuar
                    self.after(0, lambda: self._continue_batch_render_after_audio(success))
                except Exception as e:
                    self.log(f"Erro na geração de áudios: {str(e)}", "ERROR")
                    self.after(0, lambda: self._continue_batch_render_after_audio(False))
            
            threading.Thread(target=audio_generation_thread, daemon=True).start()
        else:
            # Sem TTS, continuar diretamente
            self._continue_batch_render_after_audio(True)

    def _continue_batch_render_after_audio(self, audio_success):
        """Continua o processamento em lote após geração de áudios."""
        # Se houve erro na geração de áudios, perguntar se deseja continuar
        if not audio_success:
            if not messagebox.askyesno(
                "Aviso",
                "Houve erros na geração de áudios.\n"
                "Deseja continuar com os áudios disponíveis?"
            ):
                self.render_btn.configure(state="normal")
                self.cancel_btn.configure(state="disabled")
                self.current_video_label.configure(text="Aguardando...")
                return

        # Criar jobs (agora incluindo áudios gerados)
        self.batch_jobs = self.scan_batch_folder()

        if not self.batch_jobs:
            messagebox.showinfo("Aviso", "Nenhum audio novo encontrado para processar.")
            self.render_btn.configure(state="normal")
            self.cancel_btn.configure(state="disabled")
            self.current_video_label.configure(text="Aguardando...")
            return

        # Inicializar sistema de imagens
        self.image_system = ImageReservationSystem(self.batch_images_var.get())

        # Verificar se ha imagens suficientes (apenas se NÃO estiver no modo de geração de imagens)
        video_mode = self.video_mode_var.get()
        image_source = self.image_source_var.get()
        
        # Se está usando Pipeline SRT com geração de imagens, não precisa verificar banco de imagens
        if video_mode == "srt" and image_source == "generate":
            self.log("Modo Pipeline SRT ativo - imagens serão geradas via API Whisk", "INFO")
        elif video_mode == "single_image":
            self.log("Modo 1 Imagem ativo - será usada 1 imagem aleatória do banco", "INFO")
        else:
            total_needed = len(self.batch_jobs) * self.images_per_video_var.get()
            available = self.image_system.get_available_count()

            if available < total_needed:
                if not messagebox.askyesno(
                    "Aviso",
                    f"Imagens insuficientes!\n\n"
                    f"Disponiveis: {available}\n"
                    f"Necessarias: {total_needed}\n\n"
                    f"Deseja continuar mesmo assim?\n"
                    f"(Alguns videos podem falhar)"
                ):
                    self.render_btn.configure(state="normal")
                    self.cancel_btn.configure(state="disabled")
                    return

        # Resetar estado
        self.cancel_requested = False
        self.workers = []
        self.batch_queue = queue.Queue()

        # Adicionar jobs a fila
        for job in self.batch_jobs:
            self.batch_queue.put(job)

        # Atualizar UI
        self.progress_bar_current.set(0)
        self.progress_bar_total.set(0)
        self.update_batch_ui()

        # Log inicial
        num_workers = self.parallel_videos_var.get()
        self.log("+================================================+")
        self.log("|  RenderX v3.2 - Equipe Matrix         |")
        self.log(f"|  Videos: {len(self.batch_jobs)} | Workers: {num_workers}                    |")
        self.log("+================================================+")

        # Iniciar workers
        for i in range(num_workers):
            worker = threading.Thread(
                target=self.batch_video_worker,
                args=(i + 1,),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)

        # Thread de monitoramento
        def monitor():
            for w in self.workers:
                w.join()

            # Mover imagens usadas apos todos terminarem
            if not self.cancel_requested:
                moved = self.image_system.move_used_images()
                self.log(f"Movidas {moved} imagens para UTILIZADAS")

            # Estatisticas finais
            done = sum(1 for j in self.batch_jobs if j.status == "done")
            errors = sum(1 for j in self.batch_jobs if j.status == "error")

            self.log("+================================================+")
            self.log("|       LOTE FINALIZADO!                         |")
            self.log("+================================================+")
            self.log(f"Concluidos: {done} | Erros: {errors}")

            self.after(0, lambda: self.render_btn.configure(state="normal"))
            self.after(0, lambda: self.cancel_btn.configure(state="disabled"))

            if done > 0:
                self.after(0, lambda: messagebox.showinfo(
                    "Lote Concluido",
                    f"Videos gerados: {done}\nErros: {errors}"
                ))

        threading.Thread(target=monitor, daemon=True).start()

    def cancel_batch(self):
        """Cancela o processamento em lote."""
        self.cancel_requested = True
        self.log("[!] Cancelamento solicitado... Aguardando workers...")

# =============================================================================
# MAIN (Sistema de licenças removido - legado em _LEGADO_LICENCAS/)
# =============================================================================
if __name__ == "__main__":
    app = FinalSlideshowApp()
    app.mainloop()
