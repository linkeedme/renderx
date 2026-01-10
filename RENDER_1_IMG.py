"""
RENDER V7
=========
Desenvolvido por: Perrenoud
Interface modernizada com CustomTkinter
Funcionalidade 100% preservada do V6
"""

# ============================================================================
# IMPORTS
# ============================================================================

import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import math
import os
import sys
import subprocess
import tempfile
import shutil
import threading
import queue
import time
import re
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict, Any

# Requests e urllib3
try:
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except ImportError as e:
    ctk.CTk().withdraw()
    messagebox.showerror(
        "Dependencia Faltando",
        f"Modulo nao encontrado: {e.name}\n\nExecute: pip install requests urllib3"
    )
    sys.exit(1)


# ============================================================================
# TEMA E CORES
# ============================================================================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

# Paleta Matrix
CORES = {
    "bg_dark": "#0D0D0D",
    "bg_section": "#1A1A1A",
    "bg_input": "#242424",
    "accent": "#00E676",
    "accent_hover": "#00C853",
    "accent_dark": "#00A94F",
    "text": "#E0E0E0",
    "text_dim": "#808080",
    "error": "#FF5252",
    "warning": "#FFB300",
    "info": "#29B6F6",
    "purple": "#9C27B0"
}


# ============================================================================
# CONSTANTES
# ============================================================================

API_BASE = "https://darkvi.com/api/tts"
API_VOICES = "https://darkvi.com/api/tts/voices"
MAX_CONSECUTIVE_FAILURES = 2
COOLDOWN_SECONDS = 60
MAX_TEXT_LENGTH = 80000


# ============================================================================
# DATACLASS
# ============================================================================

@dataclass
class FactoryJob:
    """Representa um trabalho no pipeline Factory"""
    txt_path: str
    image_path: str
    source_folder: str
    audio_path: Optional[str] = None
    output_path: Optional[str] = None
    status: str = "pending"
    error_message: Optional[str] = None


# ============================================================================
# WIDGET: GAVETA COLLAPSIBLE
# ============================================================================

class CollapsibleSection(ctk.CTkFrame):
    """Seção expansível/retrátil (gaveta)"""

    def __init__(self, parent, title, expanded=False, **kwargs):
        super().__init__(parent, fg_color=CORES["bg_section"], corner_radius=8, **kwargs)

        self.expanded = expanded
        self.title = title

        # Header clicável
        self.header = ctk.CTkFrame(self, fg_color="transparent", cursor="hand2")
        self.header.pack(fill="x", padx=5, pady=5)

        self.toggle_btn = ctk.CTkLabel(
            self.header,
            text="▶" if not expanded else "▼",
            font=ctk.CTkFont(size=12),
            text_color=CORES["accent"],
            width=20
        )
        self.toggle_btn.pack(side="left", padx=(5, 10))

        self.title_label = ctk.CTkLabel(
            self.header,
            text=title,
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=CORES["accent"]
        )
        self.title_label.pack(side="left")

        # Container do conteúdo
        self.content = ctk.CTkFrame(self, fg_color="transparent")
        if expanded:
            self.content.pack(fill="x", padx=10, pady=(0, 10))

        # Bind clique
        self.header.bind("<Button-1>", self.toggle)
        self.toggle_btn.bind("<Button-1>", self.toggle)
        self.title_label.bind("<Button-1>", self.toggle)

    def toggle(self, event=None):
        self.expanded = not self.expanded
        if self.expanded:
            self.toggle_btn.configure(text="▼")
            self.content.pack(fill="x", padx=10, pady=(0, 10))
        else:
            self.toggle_btn.configure(text="▶")
            self.content.pack_forget()

    def get_content(self):
        return self.content


# ============================================================================
# CLASSE PRINCIPAL
# ============================================================================

class RenderV7(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("RENDER V7")
        self.geometry("1100x800")
        self.minsize(960, 600)
        self.configure(fg_color=CORES["bg_dark"])

        # Caminhos base
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.settings_file = os.path.join(self.base_dir, "v7_settings.json")

        # ===== VARIAVEIS COMPARTILHADAS =====
        self.pasta_raiz = None
        self.pasta_saida = None
        self.overlay_path = None
        self.vinheta_path = None
        self.musica_path = None

        # ===== CONTROLE DE PROCESSAMENTO =====
        self.processing = False
        self.cancel_requested = False
        self.temp_files = []

        # Factory Pipeline
        self.audio_queue = queue.Queue()
        self.video_queue = queue.Queue()
        self.audio_producing = False
        self.consecutive_failures = 0

        # Contadores
        self.total_jobs = 0
        self.completed_audio = 0
        self.completed_video = 0
        self.success_count = 0
        self.error_count = 0

        # Legacy Mode
        self.completed_videos_legacy = 0
        self.total_videos_legacy = 0

        # ===== LOCKS THREAD-SAFE =====
        self.log_lock = threading.Lock()
        self.progress_lock = threading.Lock()
        self.api_key_lock = threading.Lock()
        self.temp_files_lock = threading.Lock()
        self.failure_lock = threading.Lock()

        # ===== GPU =====
        self.gpu_index = None

        # ===== SISTEMA DE API KEYS =====
        self.api_keys_list = []
        self.current_key_index = 0
        self.voices_map = {}

        # ===== VARIAVEIS DE CONTROLE =====
        self.amplitude_var = ctk.DoubleVar(value=1.6)
        self.crop_ratio_var = ctk.DoubleVar(value=1.0)
        self.zoom_var = ctk.DoubleVar(value=2.0)
        self.chroma_color_var = ctk.StringVar(value="00b140")
        self.chroma_similarity_var = ctk.DoubleVar(value=0.2)
        self.chroma_blend_var = ctk.DoubleVar(value=0.1)
        self.musica_volume_var = ctk.DoubleVar(value=0.3)
        self.voice_name_var = ctk.StringVar(value="")
        self.polling_interval_var = ctk.DoubleVar(value=2.0)
        self.max_retries_var = ctk.IntVar(value=3)
        self.audio_threads_var = ctk.IntVar(value=2)
        self.video_threads_var = ctk.IntVar(value=1)
        self.usar_gpu_var = ctk.BooleanVar(value=True)

        # ===== INICIALIZACAO =====
        self.detect_rtx_gpu()
        self.load_settings()
        self.create_interface()
        self.center_window()

        # Protocolo de fechamento
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    # ========================================================================
    # UTILITARIOS
    # ========================================================================

    def center_window(self):
        self.update_idletasks()
        w = self.winfo_width()
        h = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (w // 2)
        y = (self.winfo_screenheight() // 2) - (h // 2)
        self.geometry(f'{w}x{h}+{x}+{y}')

    def on_closing(self):
        if self.processing:
            if messagebox.askokcancel("Sair", "Processo em andamento. Sair?"):
                self.cancel_requested = True
                self.after(1000, self.destroy)
        else:
            self.destroy()

    # ========================================================================
    # DETECCAO DE GPU
    # ========================================================================

    def detect_rtx_gpu(self):
        """Detecta GPU RTX (ignora iGPU)"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name', '--format=csv,noheader'],
                capture_output=True, text=True, check=True, timeout=5
            )
            for line in result.stdout.strip().split('\n'):
                parts = line.split(',', 1)
                if len(parts) < 2:
                    continue
                index = parts[0].strip()
                name = parts[1].strip()
                if 'RTX' in name.upper() or 'GEFORCE' in name.upper():
                    self.gpu_index = index
                    print(f"GPU: {name} (index {index})")
                    return
            self.gpu_index = '0'
        except Exception:
            self.gpu_index = '0'

    def check_nvenc_support(self):
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True, text=True, timeout=5
            )
            return "h264_nvenc" in result.stdout
        except:
            return False

    # ========================================================================
    # PERSISTENCIA
    # ========================================================================

    def load_settings(self):
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)

                self.pasta_raiz = settings.get('pasta_raiz')
                self.pasta_saida = settings.get('pasta_saida')
                self.overlay_path = settings.get('overlay_path')
                self.vinheta_path = settings.get('vinheta_path')
                self.musica_path = settings.get('musica_path')

                audio_cfg = settings.get('audio_config', {})
                self.voice_name_var.set(audio_cfg.get('voice_name', ''))
                self.audio_threads_var.set(audio_cfg.get('audio_threads', 2))
                self.polling_interval_var.set(audio_cfg.get('polling_interval', 2.0))
                self.max_retries_var.set(audio_cfg.get('max_retries', 3))

                video_cfg = settings.get('video_config', {})
                self.amplitude_var.set(video_cfg.get('amplitude', 1.6))
                self.crop_ratio_var.set(video_cfg.get('crop_ratio', 1.0))
                self.zoom_var.set(video_cfg.get('zoom', 2.0))
                self.chroma_color_var.set(video_cfg.get('chroma_color', '00b140'))
                self.chroma_similarity_var.set(video_cfg.get('chroma_similarity', 0.2))
                self.chroma_blend_var.set(video_cfg.get('chroma_blend', 0.1))
                self.musica_volume_var.set(video_cfg.get('musica_volume', 0.3))
                self.usar_gpu_var.set(video_cfg.get('usar_gpu', True))
                self.video_threads_var.set(video_cfg.get('video_threads', 1))

                keys_data = settings.get('api_keys_list', [])
                self.api_keys_list = []
                for key_info in keys_data:
                    self.api_keys_list.append({
                        "name": key_info.get("name", "Sem Nome"),
                        "key": key_info.get("key", ""),
                        "disabled_until": None
                    })
        except Exception as e:
            print(f"Erro ao carregar configuracoes: {e}")

    def save_settings(self):
        try:
            keys_to_save = [{"name": k["name"], "key": k["key"]} for k in self.api_keys_list]

            settings = {
                'pasta_raiz': self.pasta_raiz,
                'pasta_saida': self.pasta_saida,
                'overlay_path': self.overlay_path,
                'vinheta_path': self.vinheta_path,
                'musica_path': self.musica_path,
                'audio_config': {
                    'voice_name': self.voice_name_var.get(),
                    'audio_threads': self.audio_threads_var.get(),
                    'polling_interval': self.polling_interval_var.get(),
                    'max_retries': self.max_retries_var.get()
                },
                'video_config': {
                    'amplitude': self.amplitude_var.get(),
                    'crop_ratio': self.crop_ratio_var.get(),
                    'zoom': self.zoom_var.get(),
                    'chroma_color': self.chroma_color_var.get(),
                    'chroma_similarity': self.chroma_similarity_var.get(),
                    'chroma_blend': self.chroma_blend_var.get(),
                    'musica_volume': self.musica_volume_var.get(),
                    'usar_gpu': self.usar_gpu_var.get(),
                    'video_threads': self.video_threads_var.get()
                },
                'api_keys_list': keys_to_save
            }

            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Erro ao salvar: {e}")

    # ========================================================================
    # INTERFACE PRINCIPAL
    # ========================================================================

    def create_interface(self):
        # Grid responsivo
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # ===== HEADER =====
        self.create_header()

        # ===== TABVIEW =====
        self.tabview = ctk.CTkTabview(self, fg_color=CORES["bg_dark"])
        self.tabview.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 10))

        self.tab_gerar = self.tabview.add("Gerar Videos")
        self.tab_importar = self.tabview.add("Importar Audio")

        # Conteúdo das abas
        self.create_gerar_tab()
        self.create_importar_tab()

        # ===== FOOTER =====
        self.create_footer()

        # Atualizar labels
        self.update_path_labels()
        self.update_keys_status()

    def create_header(self):
        header = ctk.CTkFrame(self, fg_color=CORES["bg_section"], corner_radius=0, height=70)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)
        header.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            header,
            text="RENDER V7",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color=CORES["accent"]
        )
        title.grid(row=0, column=0, pady=(15, 2))

        subtitle = ctk.CTkLabel(
            header,
            text="Desenvolvido por Perrenoud",
            font=ctk.CTkFont(size=11),
            text_color=CORES["text_dim"]
        )
        subtitle.grid(row=1, column=0, pady=(0, 10))

    # ========================================================================
    # ABA: GERAR VIDEOS
    # ========================================================================

    def create_gerar_tab(self):
        # Frame scrollable
        self.tab_gerar.grid_columnconfigure(0, weight=1)
        self.tab_gerar.grid_rowconfigure(0, weight=1)

        scroll = ctk.CTkScrollableFrame(
            self.tab_gerar,
            fg_color="transparent"
        )
        scroll.grid(row=0, column=0, sticky="nsew")
        scroll.grid_columnconfigure(0, weight=1)

        row = 0

        # ===== SECAO: PASTAS =====
        pastas_frame = ctk.CTkFrame(scroll, fg_color=CORES["bg_section"], corner_radius=8)
        pastas_frame.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
        pastas_frame.grid_columnconfigure(1, weight=1)
        row += 1

        ctk.CTkLabel(
            pastas_frame, text="Pastas",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=CORES["accent"]
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=15, pady=(10, 5))

        # Pasta Entrada
        ctk.CTkLabel(pastas_frame, text="Entrada:", text_color=CORES["text"]).grid(
            row=1, column=0, sticky="w", padx=(15, 10), pady=5)

        self.entrada_label = ctk.CTkLabel(
            pastas_frame, text="Nenhuma", text_color=CORES["text_dim"],
            font=ctk.CTkFont(size=11)
        )
        self.entrada_label.grid(row=1, column=1, sticky="w", padx=5)

        ctk.CTkButton(
            pastas_frame, text="Selecionar", width=100,
            fg_color=CORES["accent"], hover_color=CORES["accent_hover"],
            text_color=CORES["bg_dark"],
            command=self.select_pasta_entrada
        ).grid(row=1, column=2, padx=(10, 15), pady=5)

        # Pasta Saída
        ctk.CTkLabel(pastas_frame, text="Saída:", text_color=CORES["text"]).grid(
            row=2, column=0, sticky="w", padx=(15, 10), pady=5)

        self.saida_label = ctk.CTkLabel(
            pastas_frame, text="Nenhuma", text_color=CORES["text_dim"],
            font=ctk.CTkFont(size=11)
        )
        self.saida_label.grid(row=2, column=1, sticky="w", padx=5)

        ctk.CTkButton(
            pastas_frame, text="Selecionar", width=100,
            fg_color=CORES["info"], hover_color="#1E88E5",
            text_color=CORES["bg_dark"],
            command=self.select_pasta_saida
        ).grid(row=2, column=2, padx=(10, 15), pady=5)

        # Botão Escanear
        scan_frame = ctk.CTkFrame(pastas_frame, fg_color="transparent")
        scan_frame.grid(row=3, column=0, columnspan=3, pady=10, padx=15, sticky="ew")

        ctk.CTkButton(
            scan_frame, text="ESCANEAR PASTA", width=150,
            fg_color=CORES["warning"], hover_color="#FFA000",
            text_color=CORES["bg_dark"], font=ctk.CTkFont(weight="bold"),
            command=self.scan_factory_folder
        ).pack(side="left")

        self.scan_status = ctk.CTkLabel(
            scan_frame, text="Selecione uma pasta para escanear",
            text_color=CORES["text_dim"], font=ctk.CTkFont(size=11)
        )
        self.scan_status.pack(side="left", padx=15)

        # ===== SECAO: API KEYS =====
        keys_frame = ctk.CTkFrame(scroll, fg_color=CORES["bg_section"], corner_radius=8)
        keys_frame.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
        keys_frame.grid_columnconfigure(0, weight=1)
        row += 1

        ctk.CTkLabel(
            keys_frame, text="API Keys",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=CORES["accent"]
        ).grid(row=0, column=0, sticky="w", padx=15, pady=(10, 5))

        self.keys_status_label = ctk.CTkLabel(
            keys_frame, text="Nenhuma chave configurada",
            text_color=CORES["warning"], font=ctk.CTkFont(size=12)
        )
        self.keys_status_label.grid(row=1, column=0, sticky="w", padx=15, pady=5)

        # Lista de chaves (frame)
        self.keys_list_frame = ctk.CTkFrame(keys_frame, fg_color=CORES["bg_input"], corner_radius=5)
        self.keys_list_frame.grid(row=2, column=0, sticky="ew", padx=15, pady=5)
        self.keys_list_frame.grid_columnconfigure(0, weight=1)

        # Botões de chaves
        keys_btn_frame = ctk.CTkFrame(keys_frame, fg_color="transparent")
        keys_btn_frame.grid(row=3, column=0, sticky="w", padx=15, pady=(5, 10))

        ctk.CTkButton(
            keys_btn_frame, text="+ Adicionar", width=100,
            fg_color=CORES["accent"], hover_color=CORES["accent_hover"],
            text_color=CORES["bg_dark"],
            command=self.add_key_dialog
        ).pack(side="left", padx=(0, 5))

        ctk.CTkButton(
            keys_btn_frame, text="- Remover", width=100,
            fg_color=CORES["error"], hover_color="#D32F2F",
            command=self.remove_selected_key
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            keys_btn_frame, text="Testar", width=80,
            fg_color=CORES["info"], hover_color="#1E88E5",
            text_color=CORES["bg_dark"],
            command=self.test_selected_key
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            keys_btn_frame, text="Reativar", width=80,
            fg_color=CORES["purple"], hover_color="#7B1FA2",
            command=self.reactivate_all_keys
        ).pack(side="left", padx=5)

        self.refresh_keys_list()

        # ===== SECAO: VOZ =====
        voz_frame = ctk.CTkFrame(scroll, fg_color=CORES["bg_section"], corner_radius=8)
        voz_frame.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
        voz_frame.grid_columnconfigure(1, weight=1)
        row += 1

        ctk.CTkLabel(
            voz_frame, text="Voz",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=CORES["accent"]
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=15, pady=(10, 5))

        ctk.CTkLabel(voz_frame, text="Selecionar:", text_color=CORES["text"]).grid(
            row=1, column=0, sticky="w", padx=(15, 10), pady=8)

        self.voice_combo = ctk.CTkComboBox(
            voz_frame, width=250, values=[], variable=self.voice_name_var,
            fg_color=CORES["bg_input"], border_color=CORES["accent"],
            button_color=CORES["accent"], button_hover_color=CORES["accent_hover"],
            dropdown_fg_color=CORES["bg_input"]
        )
        self.voice_combo.grid(row=1, column=1, sticky="w", padx=5, pady=8)

        ctk.CTkButton(
            voz_frame, text="Listar Vozes", width=110,
            fg_color=CORES["purple"], hover_color="#7B1FA2",
            command=self.fetch_voices
        ).grid(row=1, column=2, padx=(10, 15), pady=8)

        # ===== GAVETA: VIDEO SETTINGS =====
        self.video_drawer = CollapsibleSection(scroll, "Video Settings", expanded=False)
        self.video_drawer.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
        row += 1

        vc = self.video_drawer.get_content()
        vc.grid_columnconfigure(1, weight=1)

        # Overlay
        ctk.CTkLabel(vc, text="Overlay:", text_color=CORES["text"]).grid(
            row=0, column=0, sticky="w", padx=5, pady=5)
        self.overlay_label = ctk.CTkLabel(vc, text="Nenhum", text_color=CORES["text_dim"])
        self.overlay_label.grid(row=0, column=1, sticky="w", padx=5)

        overlay_btns = ctk.CTkFrame(vc, fg_color="transparent")
        overlay_btns.grid(row=0, column=2, sticky="e", padx=5)
        ctk.CTkButton(overlay_btns, text="Sel.", width=50, fg_color=CORES["purple"],
                      hover_color="#7B1FA2", command=self.select_overlay).pack(side="left", padx=2)
        ctk.CTkButton(overlay_btns, text="X", width=30, fg_color=CORES["error"],
                      hover_color="#D32F2F", command=self.remove_overlay).pack(side="left", padx=2)

        # Chroma
        ctk.CTkLabel(vc, text="Chroma Hex:", text_color=CORES["text"]).grid(
            row=1, column=0, sticky="w", padx=5, pady=5)
        chroma_entry = ctk.CTkEntry(vc, textvariable=self.chroma_color_var, width=100,
                                    fg_color=CORES["bg_input"], border_color=CORES["accent"])
        chroma_entry.grid(row=1, column=1, sticky="w", padx=5)

        # Similarity & Blend
        sliders_row = ctk.CTkFrame(vc, fg_color="transparent")
        sliders_row.grid(row=2, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

        ctk.CTkLabel(sliders_row, text="Similarity:", text_color=CORES["text"]).pack(side="left")
        self.sim_slider = ctk.CTkSlider(sliders_row, from_=0.01, to=0.5, variable=self.chroma_similarity_var,
                                        width=100, progress_color=CORES["accent"])
        self.sim_slider.pack(side="left", padx=5)
        self.sim_value = ctk.CTkLabel(sliders_row, text=f"{self.chroma_similarity_var.get():.2f}",
                                      text_color=CORES["accent"], width=40)
        self.sim_value.pack(side="left")

        ctk.CTkLabel(sliders_row, text="Blend:", text_color=CORES["text"]).pack(side="left", padx=(20, 0))
        self.blend_slider = ctk.CTkSlider(sliders_row, from_=0.0, to=0.5, variable=self.chroma_blend_var,
                                          width=100, progress_color=CORES["accent"])
        self.blend_slider.pack(side="left", padx=5)
        self.blend_value = ctk.CTkLabel(sliders_row, text=f"{self.chroma_blend_var.get():.2f}",
                                        text_color=CORES["accent"], width=40)
        self.blend_value.pack(side="left")

        self.sim_slider.configure(command=lambda v: self.sim_value.configure(text=f"{float(v):.2f}"))
        self.blend_slider.configure(command=lambda v: self.blend_value.configure(text=f"{float(v):.2f}"))

        # Vinheta
        ctk.CTkLabel(vc, text="Vinheta:", text_color=CORES["text"]).grid(
            row=3, column=0, sticky="w", padx=5, pady=5)
        self.vinheta_label = ctk.CTkLabel(vc, text="Nenhuma", text_color=CORES["text_dim"])
        self.vinheta_label.grid(row=3, column=1, sticky="w", padx=5)

        vinheta_btns = ctk.CTkFrame(vc, fg_color="transparent")
        vinheta_btns.grid(row=3, column=2, sticky="e", padx=5)
        ctk.CTkButton(vinheta_btns, text="Sel.", width=50, fg_color=CORES["warning"],
                      hover_color="#FFA000", text_color=CORES["bg_dark"],
                      command=self.select_vinheta).pack(side="left", padx=2)
        ctk.CTkButton(vinheta_btns, text="X", width=30, fg_color=CORES["error"],
                      hover_color="#D32F2F", command=self.remove_vinheta).pack(side="left", padx=2)

        # Música
        ctk.CTkLabel(vc, text="Música:", text_color=CORES["text"]).grid(
            row=4, column=0, sticky="w", padx=5, pady=5)
        self.musica_label = ctk.CTkLabel(vc, text="Nenhuma", text_color=CORES["text_dim"])
        self.musica_label.grid(row=4, column=1, sticky="w", padx=5)

        musica_btns = ctk.CTkFrame(vc, fg_color="transparent")
        musica_btns.grid(row=4, column=2, sticky="e", padx=5)
        ctk.CTkButton(musica_btns, text="Sel.", width=50, fg_color=CORES["info"],
                      hover_color="#1E88E5", text_color=CORES["bg_dark"],
                      command=self.select_musica).pack(side="left", padx=2)
        ctk.CTkButton(musica_btns, text="X", width=30, fg_color=CORES["error"],
                      hover_color="#D32F2F", command=self.remove_musica).pack(side="left", padx=2)

        # Volume música
        vol_row = ctk.CTkFrame(vc, fg_color="transparent")
        vol_row.grid(row=5, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(vol_row, text="Volume:", text_color=CORES["text"]).pack(side="left")
        self.vol_slider = ctk.CTkSlider(vol_row, from_=0.0, to=1.0, variable=self.musica_volume_var,
                                        width=150, progress_color=CORES["accent"])
        self.vol_slider.pack(side="left", padx=10)
        self.vol_value = ctk.CTkLabel(vol_row, text=f"{self.musica_volume_var.get():.2f}",
                                      text_color=CORES["accent"], width=40)
        self.vol_value.pack(side="left")
        self.vol_slider.configure(command=lambda v: self.vol_value.configure(text=f"{float(v):.2f}"))

        # Pêndulo
        pend_row = ctk.CTkFrame(vc, fg_color="transparent")
        pend_row.grid(row=6, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

        ctk.CTkLabel(pend_row, text="Amplitude:", text_color=CORES["text"]).pack(side="left")
        self.amp_slider = ctk.CTkSlider(pend_row, from_=0.1, to=10.0, variable=self.amplitude_var,
                                        width=100, progress_color=CORES["accent"])
        self.amp_slider.pack(side="left", padx=5)
        self.amp_value = ctk.CTkLabel(pend_row, text=f"{self.amplitude_var.get():.1f}°",
                                      text_color=CORES["accent"], width=40)
        self.amp_value.pack(side="left")

        ctk.CTkLabel(pend_row, text="Crop:", text_color=CORES["text"]).pack(side="left", padx=(20, 0))
        self.crop_slider = ctk.CTkSlider(pend_row, from_=0.5, to=1.0, variable=self.crop_ratio_var,
                                         width=80, progress_color=CORES["accent"])
        self.crop_slider.pack(side="left", padx=5)
        self.crop_value = ctk.CTkLabel(pend_row, text=f"{self.crop_ratio_var.get():.2f}",
                                       text_color=CORES["accent"], width=35)
        self.crop_value.pack(side="left")

        ctk.CTkLabel(pend_row, text="Zoom:", text_color=CORES["text"]).pack(side="left", padx=(15, 0))
        self.zoom_slider = ctk.CTkSlider(pend_row, from_=1.0, to=3.0, variable=self.zoom_var,
                                         width=80, progress_color=CORES["accent"])
        self.zoom_slider.pack(side="left", padx=5)
        self.zoom_value = ctk.CTkLabel(pend_row, text=f"{self.zoom_var.get():.1f}x",
                                       text_color=CORES["accent"], width=35)
        self.zoom_value.pack(side="left")

        self.amp_slider.configure(command=lambda v: self.amp_value.configure(text=f"{float(v):.1f}°"))
        self.crop_slider.configure(command=lambda v: self.crop_value.configure(text=f"{float(v):.2f}"))
        self.zoom_slider.configure(command=lambda v: self.zoom_value.configure(text=f"{float(v):.1f}x"))

        # ===== GAVETA: PERFORMANCE =====
        self.perf_drawer = CollapsibleSection(scroll, "Performance", expanded=False)
        self.perf_drawer.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
        row += 1

        pc = self.perf_drawer.get_content()

        threads_row = ctk.CTkFrame(pc, fg_color="transparent")
        threads_row.pack(fill="x", pady=5)

        ctk.CTkLabel(threads_row, text="Audio Threads:", text_color=CORES["text"]).pack(side="left", padx=5)
        self.audio_t_slider = ctk.CTkSlider(threads_row, from_=1, to=20, variable=self.audio_threads_var,
                                            width=120, progress_color=CORES["accent"], number_of_steps=19)
        self.audio_t_slider.pack(side="left", padx=5)
        self.audio_t_value = ctk.CTkLabel(threads_row, text=str(self.audio_threads_var.get()),
                                          text_color=CORES["accent"], width=30)
        self.audio_t_value.pack(side="left")

        ctk.CTkLabel(threads_row, text="Video Threads:", text_color=CORES["text"]).pack(side="left", padx=(30, 5))
        self.video_t_slider = ctk.CTkSlider(threads_row, from_=1, to=5, variable=self.video_threads_var,
                                            width=80, progress_color=CORES["accent"], number_of_steps=4)
        self.video_t_slider.pack(side="left", padx=5)
        self.video_t_value = ctk.CTkLabel(threads_row, text=str(self.video_threads_var.get()),
                                          text_color=CORES["accent"], width=25)
        self.video_t_value.pack(side="left")

        self.audio_t_slider.configure(command=lambda v: self.audio_t_value.configure(text=str(int(float(v)))))
        self.video_t_slider.configure(command=lambda v: self.video_t_value.configure(text=str(int(float(v)))))

        gpu_row = ctk.CTkFrame(pc, fg_color="transparent")
        gpu_row.pack(fill="x", pady=5)

        self.gpu_check = ctk.CTkCheckBox(
            gpu_row, text="Usar GPU (NVENC + hwaccel)",
            variable=self.usar_gpu_var,
            fg_color=CORES["accent"], hover_color=CORES["accent_hover"],
            text_color=CORES["text"]
        )
        self.gpu_check.pack(side="left", padx=5)

        # ===== SECAO: LOGS =====
        logs_frame = ctk.CTkFrame(scroll, fg_color=CORES["bg_section"], corner_radius=8)
        logs_frame.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
        logs_frame.grid_columnconfigure(0, weight=1)
        row += 1

        ctk.CTkLabel(
            logs_frame, text="Logs",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=CORES["accent"]
        ).grid(row=0, column=0, sticky="w", padx=15, pady=(10, 5))

        # Log Resumido
        ctk.CTkLabel(logs_frame, text="Status:", text_color=CORES["text"],
                     font=ctk.CTkFont(size=11)).grid(row=1, column=0, sticky="w", padx=15, pady=(5, 2))

        self.log_resumido = ctk.CTkTextbox(
            logs_frame, height=100, fg_color=CORES["bg_dark"],
            text_color=CORES["accent"], font=ctk.CTkFont(family="Consolas", size=10)
        )
        self.log_resumido.grid(row=2, column=0, sticky="ew", padx=15, pady=5)

        # Log Detalhado
        ctk.CTkLabel(logs_frame, text="Detalhes:", text_color=CORES["text"],
                     font=ctk.CTkFont(size=11)).grid(row=3, column=0, sticky="w", padx=15, pady=(5, 2))

        self.log_detalhado = ctk.CTkTextbox(
            logs_frame, height=120, fg_color=CORES["bg_dark"],
            text_color=CORES["info"], font=ctk.CTkFont(family="Consolas", size=10)
        )
        self.log_detalhado.grid(row=4, column=0, sticky="ew", padx=15, pady=(5, 15))

    # ========================================================================
    # ABA: IMPORTAR AUDIO
    # ========================================================================

    def create_importar_tab(self):
        self.tab_importar.grid_columnconfigure(0, weight=1)
        self.tab_importar.grid_rowconfigure(0, weight=1)

        scroll = ctk.CTkScrollableFrame(self.tab_importar, fg_color="transparent")
        scroll.grid(row=0, column=0, sticky="nsew")
        scroll.grid_columnconfigure(0, weight=1)

        row = 0

        # Info
        info_frame = ctk.CTkFrame(scroll, fg_color=CORES["bg_section"], corner_radius=8)
        info_frame.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
        row += 1

        ctk.CTkLabel(
            info_frame,
            text="Use este modo quando você JÁ TEM os arquivos MP3 prontos.\n"
                 "Pareamento por NOME: audio.mp3 + audio.png",
            text_color=CORES["text_dim"], font=ctk.CTkFont(size=11)
        ).pack(padx=15, pady=15)

        # Pastas
        pastas_frame = ctk.CTkFrame(scroll, fg_color=CORES["bg_section"], corner_radius=8)
        pastas_frame.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
        pastas_frame.grid_columnconfigure(1, weight=1)
        row += 1

        ctk.CTkLabel(
            pastas_frame, text="Pastas",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=CORES["accent"]
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=15, pady=(10, 5))

        ctk.CTkLabel(pastas_frame, text="Entrada:", text_color=CORES["text"]).grid(
            row=1, column=0, sticky="w", padx=(15, 10), pady=5)

        self.legacy_entrada_label = ctk.CTkLabel(
            pastas_frame, text="Nenhuma", text_color=CORES["text_dim"]
        )
        self.legacy_entrada_label.grid(row=1, column=1, sticky="w", padx=5)

        ctk.CTkButton(
            pastas_frame, text="Selecionar", width=100,
            fg_color=CORES["accent"], hover_color=CORES["accent_hover"],
            text_color=CORES["bg_dark"],
            command=self.select_legacy_entrada
        ).grid(row=1, column=2, padx=(10, 15), pady=5)

        ctk.CTkLabel(pastas_frame, text="Saída:", text_color=CORES["text"]).grid(
            row=2, column=0, sticky="w", padx=(15, 10), pady=(5, 15))

        self.legacy_saida_label = ctk.CTkLabel(
            pastas_frame, text="Nenhuma", text_color=CORES["text_dim"]
        )
        self.legacy_saida_label.grid(row=2, column=1, sticky="w", padx=5, pady=(5, 15))

        ctk.CTkButton(
            pastas_frame, text="Selecionar", width=100,
            fg_color=CORES["info"], hover_color="#1E88E5",
            text_color=CORES["bg_dark"],
            command=self.select_legacy_saida
        ).grid(row=2, column=2, padx=(10, 15), pady=(5, 15))

        # Log
        log_frame = ctk.CTkFrame(scroll, fg_color=CORES["bg_section"], corner_radius=8)
        log_frame.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
        log_frame.grid_columnconfigure(0, weight=1)
        row += 1

        ctk.CTkLabel(
            log_frame, text="Log",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=CORES["accent"]
        ).grid(row=0, column=0, sticky="w", padx=15, pady=(10, 5))

        self.legacy_log = ctk.CTkTextbox(
            log_frame, height=300, fg_color=CORES["bg_dark"],
            text_color=CORES["accent"], font=ctk.CTkFont(family="Consolas", size=10)
        )
        self.legacy_log.grid(row=1, column=0, sticky="ew", padx=15, pady=(5, 15))

    # ========================================================================
    # FOOTER
    # ========================================================================

    def create_footer(self):
        footer = ctk.CTkFrame(self, fg_color=CORES["bg_section"], corner_radius=0, height=100)
        footer.grid(row=2, column=0, sticky="ew")
        footer.grid_propagate(False)
        footer.grid_columnconfigure(0, weight=1)

        # Barras de progresso
        bars = ctk.CTkFrame(footer, fg_color="transparent")
        bars.grid(row=0, column=0, pady=10)

        # Audio
        ctk.CTkLabel(bars, text="Audio:", text_color=CORES["text"]).pack(side="left", padx=5)
        self.progress_audio = ctk.CTkProgressBar(bars, width=200, progress_color=CORES["accent"])
        self.progress_audio.pack(side="left", padx=5)
        self.progress_audio.set(0)
        self.progress_audio_label = ctk.CTkLabel(bars, text="0%", text_color=CORES["text"], width=80)
        self.progress_audio_label.pack(side="left", padx=5)

        # Video
        ctk.CTkLabel(bars, text="Video:", text_color=CORES["text"]).pack(side="left", padx=(30, 5))
        self.progress_video = ctk.CTkProgressBar(bars, width=200, progress_color=CORES["info"])
        self.progress_video.pack(side="left", padx=5)
        self.progress_video.set(0)
        self.progress_video_label = ctk.CTkLabel(bars, text="0%", text_color=CORES["text"], width=80)
        self.progress_video_label.pack(side="left", padx=5)

        # Botões
        btns = ctk.CTkFrame(footer, fg_color="transparent")
        btns.grid(row=1, column=0, pady=5)

        self.start_btn = ctk.CTkButton(
            btns, text="INICIAR PROCESSAMENTO", width=220, height=40,
            fg_color=CORES["accent"], hover_color=CORES["accent_hover"],
            text_color=CORES["bg_dark"], font=ctk.CTkFont(size=14, weight="bold"),
            command=self.start_processing
        )
        self.start_btn.pack(side="left", padx=10)

        self.cancel_btn = ctk.CTkButton(
            btns, text="CANCELAR", width=120, height=40,
            fg_color=CORES["error"], hover_color="#D32F2F",
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self.cancel_processing, state="disabled"
        )
        self.cancel_btn.pack(side="left", padx=10)

    # ========================================================================
    # HANDLERS DE SELECAO
    # ========================================================================

    def select_pasta_entrada(self):
        folder = filedialog.askdirectory(title="Selecione a Pasta de Entrada")
        if folder:
            self.pasta_raiz = folder
            self.entrada_label.configure(text=os.path.basename(folder), text_color=CORES["accent"])
            self.legacy_entrada_label.configure(text=os.path.basename(folder), text_color=CORES["accent"])
            self.log_resumido_msg(f"Pasta entrada: {folder}")
            self.save_settings()
            self.scan_factory_folder()

    def select_pasta_saida(self):
        folder = filedialog.askdirectory(title="Selecione a Pasta de Saída")
        if folder:
            self.pasta_saida = folder
            self.saida_label.configure(text=os.path.basename(folder), text_color=CORES["info"])
            self.legacy_saida_label.configure(text=os.path.basename(folder), text_color=CORES["info"])
            self.save_settings()

    def select_legacy_entrada(self):
        folder = filedialog.askdirectory(title="Selecione a Pasta de Entrada")
        if folder:
            self.pasta_raiz = folder
            self.legacy_entrada_label.configure(text=os.path.basename(folder), text_color=CORES["accent"])
            self.entrada_label.configure(text=os.path.basename(folder), text_color=CORES["accent"])
            self.save_settings()

    def select_legacy_saida(self):
        folder = filedialog.askdirectory(title="Selecione a Pasta de Saída")
        if folder:
            self.pasta_saida = folder
            self.legacy_saida_label.configure(text=os.path.basename(folder), text_color=CORES["info"])
            self.saida_label.configure(text=os.path.basename(folder), text_color=CORES["info"])
            self.save_settings()

    def select_overlay(self):
        file = filedialog.askopenfilename(title="Selecione o Overlay",
                                          filetypes=[("Videos", "*.mp4 *.mov *.avi")])
        if file:
            self.overlay_path = file
            self.overlay_label.configure(text=os.path.basename(file), text_color=CORES["purple"])
            self.save_settings()

    def remove_overlay(self):
        self.overlay_path = None
        self.overlay_label.configure(text="Nenhum", text_color=CORES["text_dim"])
        self.save_settings()

    def select_vinheta(self):
        file = filedialog.askopenfilename(title="Selecione a Vinheta PNG",
                                          filetypes=[("PNG", "*.png")])
        if file:
            self.vinheta_path = file
            self.vinheta_label.configure(text=os.path.basename(file), text_color=CORES["warning"])
            self.save_settings()

    def remove_vinheta(self):
        self.vinheta_path = None
        self.vinheta_label.configure(text="Nenhuma", text_color=CORES["text_dim"])
        self.save_settings()

    def select_musica(self):
        file = filedialog.askopenfilename(title="Selecione a Música",
                                          filetypes=[("Audio", "*.mp3 *.wav *.m4a")])
        if file:
            self.musica_path = file
            self.musica_label.configure(text=os.path.basename(file), text_color=CORES["info"])
            self.save_settings()

    def remove_musica(self):
        self.musica_path = None
        self.musica_label.configure(text="Nenhuma", text_color=CORES["text_dim"])
        self.save_settings()

    def update_path_labels(self):
        if self.pasta_raiz:
            self.entrada_label.configure(text=os.path.basename(self.pasta_raiz), text_color=CORES["accent"])
            self.legacy_entrada_label.configure(text=os.path.basename(self.pasta_raiz), text_color=CORES["accent"])
        if self.pasta_saida:
            self.saida_label.configure(text=os.path.basename(self.pasta_saida), text_color=CORES["info"])
            self.legacy_saida_label.configure(text=os.path.basename(self.pasta_saida), text_color=CORES["info"])
        if self.overlay_path:
            self.overlay_label.configure(text=os.path.basename(self.overlay_path), text_color=CORES["purple"])
        if self.vinheta_path:
            self.vinheta_label.configure(text=os.path.basename(self.vinheta_path), text_color=CORES["warning"])
        if self.musica_path:
            self.musica_label.configure(text=os.path.basename(self.musica_path), text_color=CORES["info"])

    # ========================================================================
    # LOGGING
    # ========================================================================

    def log_resumido_msg(self, message):
        def _log():
            timestamp = time.strftime("%H:%M:%S")
            self.log_resumido.insert("end", f"[{timestamp}] {message}\n")
            self.log_resumido.see("end")
        with self.log_lock:
            self.after(0, _log)

    def log_detalhado_msg(self, message, tag=None):
        def _log():
            timestamp = time.strftime("%H:%M:%S")
            prefix = f"[{timestamp}]"
            if tag:
                prefix += f" [{tag}]"
            self.log_detalhado.insert("end", f"{prefix} {message}\n")
            self.log_detalhado.see("end")
        with self.log_lock:
            self.after(0, _log)

    def log_legacy(self, message, thread_id=None):
        def _log():
            timestamp = time.strftime("%H:%M:%S")
            if thread_id:
                prefix = f"[{timestamp}] [T{thread_id}]"
            else:
                prefix = f"[{timestamp}]"
            self.legacy_log.insert("end", f"{prefix} {message}\n")
            self.legacy_log.see("end")
        with self.log_lock:
            self.after(0, _log)

    # ========================================================================
    # GERENCIAMENTO DE CHAVES API
    # ========================================================================

    def refresh_keys_list(self):
        # Limpa frame
        for widget in self.keys_list_frame.winfo_children():
            widget.destroy()

        if not self.api_keys_list:
            ctk.CTkLabel(
                self.keys_list_frame, text="Nenhuma chave cadastrada",
                text_color=CORES["text_dim"], font=ctk.CTkFont(size=11)
            ).pack(pady=10)
        else:
            for idx, key_info in enumerate(self.api_keys_list):
                row = ctk.CTkFrame(self.keys_list_frame, fg_color="transparent")
                row.pack(fill="x", padx=5, pady=2)

                name = key_info.get("name", "Sem Nome")
                key = key_info.get("key", "")
                disabled_until = key_info.get("disabled_until")

                # Radio/checkbox simulado
                self.selected_key_idx = ctk.IntVar(value=-1)
                rb = ctk.CTkRadioButton(
                    row, text="", variable=self.selected_key_idx, value=idx,
                    fg_color=CORES["accent"], hover_color=CORES["accent_hover"],
                    width=20
                )
                rb.pack(side="left", padx=(5, 10))

                if disabled_until and datetime.now() < disabled_until:
                    remaining = (disabled_until - datetime.now()).seconds
                    status = f"Bloq ({remaining}s)"
                    status_color = CORES["error"]
                else:
                    status = "Ativa"
                    status_color = CORES["accent"]
                    key_info["disabled_until"] = None

                ctk.CTkLabel(row, text=name, text_color=CORES["text"], width=120).pack(side="left")
                ctk.CTkLabel(row, text=status, text_color=status_color, width=80).pack(side="left", padx=10)

                key_masked = key[:6] + "..." + key[-6:] if len(key) > 12 else key[:4] + "..."
                ctk.CTkLabel(row, text=key_masked, text_color=CORES["text_dim"],
                             font=ctk.CTkFont(family="Consolas", size=10)).pack(side="left", padx=10)

        self.update_keys_status()

    def update_keys_status(self):
        total = len(self.api_keys_list)
        active = sum(1 for k in self.api_keys_list
                     if not k.get("disabled_until") or datetime.now() >= k.get("disabled_until"))

        if total == 0:
            self.keys_status_label.configure(text="Nenhuma chave configurada", text_color=CORES["warning"])
        else:
            color = CORES["accent"] if active > 0 else CORES["error"]
            self.keys_status_label.configure(
                text=f"{active}/{total} chave(s) ativa(s)",
                text_color=color
            )

    def add_key_dialog(self):
        dialog = ctk.CTkToplevel(self)
        dialog.title("Adicionar API Key")
        dialog.geometry("450x200")
        dialog.configure(fg_color=CORES["bg_section"])
        dialog.transient(self)
        dialog.grab_set()

        # Centraliza
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - 225
        y = (dialog.winfo_screenheight() // 2) - 100
        dialog.geometry(f'+{x}+{y}')

        ctk.CTkLabel(dialog, text="Nome da Conta:", text_color=CORES["text"]).pack(pady=(20, 5))
        name_entry = ctk.CTkEntry(dialog, width=350, fg_color=CORES["bg_input"], border_color=CORES["accent"])
        name_entry.pack()
        name_entry.insert(0, f"Conta {len(self.api_keys_list) + 1}")

        ctk.CTkLabel(dialog, text="API Key:", text_color=CORES["text"]).pack(pady=(10, 5))
        key_entry = ctk.CTkEntry(dialog, width=350, fg_color=CORES["bg_input"], border_color=CORES["accent"])
        key_entry.pack()

        def save_key():
            name = name_entry.get().strip()
            key = key_entry.get().strip()

            if not name or not key:
                messagebox.showerror("Erro", "Preencha todos os campos!")
                return

            for existing in self.api_keys_list:
                if existing["key"] == key:
                    messagebox.showerror("Erro", "Esta chave já foi adicionada!")
                    return

            self.api_keys_list.append({"name": name, "key": key, "disabled_until": None})
            self.save_settings()
            self.refresh_keys_list()
            dialog.destroy()
            messagebox.showinfo("Sucesso", f"Chave '{name}' adicionada!")

        btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_frame.pack(pady=15)

        ctk.CTkButton(btn_frame, text="Salvar", width=100, fg_color=CORES["accent"],
                      hover_color=CORES["accent_hover"], text_color=CORES["bg_dark"],
                      command=save_key).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="Cancelar", width=100, fg_color=CORES["text_dim"],
                      hover_color="#666666", command=dialog.destroy).pack(side="left", padx=5)

    def remove_selected_key(self):
        if hasattr(self, 'selected_key_idx'):
            idx = self.selected_key_idx.get()
            if idx >= 0 and idx < len(self.api_keys_list):
                name = self.api_keys_list[idx]["name"]
                if messagebox.askyesno("Confirmar", f"Remover '{name}'?"):
                    del self.api_keys_list[idx]
                    self.save_settings()
                    self.refresh_keys_list()
                return
        messagebox.showwarning("Aviso", "Selecione uma chave para remover!")

    def test_selected_key(self):
        if hasattr(self, 'selected_key_idx'):
            idx = self.selected_key_idx.get()
            if idx >= 0 and idx < len(self.api_keys_list):
                key_info = self.api_keys_list[idx]
                try:
                    headers = {"Authorization": f"Bearer {key_info['key']}"}
                    response = requests.get(API_VOICES, headers=headers, timeout=10, verify=False)

                    if response.status_code == 200:
                        data = response.json()
                        count = len(data) if isinstance(data, list) else 0
                        messagebox.showinfo("Sucesso", f"Chave válida!\n{count} vozes disponíveis.")
                    elif response.status_code == 401:
                        messagebox.showerror("Erro", "Chave inválida ou expirada!")
                    else:
                        messagebox.showwarning("Aviso", f"HTTP {response.status_code}")
                except Exception as e:
                    messagebox.showerror("Erro", str(e))
                return
        messagebox.showwarning("Aviso", "Selecione uma chave para testar!")

    def reactivate_all_keys(self):
        for key_info in self.api_keys_list:
            key_info["disabled_until"] = None
        self.refresh_keys_list()
        messagebox.showinfo("Sucesso", "Todas as chaves reativadas!")

    def get_next_api_key(self):
        with self.api_key_lock:
            if not self.api_keys_list:
                return None, None

            attempts = 0
            max_attempts = len(self.api_keys_list)

            while attempts < max_attempts:
                key_info = self.api_keys_list[self.current_key_index]
                self.current_key_index = (self.current_key_index + 1) % len(self.api_keys_list)

                disabled_until = key_info.get("disabled_until")

                if disabled_until is None or datetime.now() >= disabled_until:
                    key_info["disabled_until"] = None
                    return key_info["key"], key_info["name"]

                attempts += 1

            return None, None

    def disable_key_temporarily(self, key, duration_seconds=COOLDOWN_SECONDS):
        with self.api_key_lock:
            for key_info in self.api_keys_list:
                if key_info["key"] == key:
                    key_info["disabled_until"] = datetime.now() + timedelta(seconds=duration_seconds)
                    self.log_detalhado_msg(f"Chave '{key_info['name']}' bloqueada por {duration_seconds}s", "API")
                    self.after(0, self.refresh_keys_list)
                    break

    # ========================================================================
    # API DARKVI
    # ========================================================================

    def fetch_voices(self):
        api_key, key_name = self.get_next_api_key()
        if not api_key:
            messagebox.showerror("Erro", "Nenhuma chave API configurada!")
            return

        try:
            self.log_detalhado_msg(f"Buscando vozes com '{key_name}'...", "API")

            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.get(API_VOICES, headers=headers, timeout=15, verify=False)

            if response.status_code == 200:
                data = response.json()

                if isinstance(data, list):
                    self.voices_map = {}

                    for v in data:
                        vid = v.get('idApi', '') or v.get('id', '')
                        vname = v.get('name', '').strip()

                        if vid and vname:
                            self.voices_map[vname] = vid

                    if self.voices_map:
                        sorted_names = sorted(self.voices_map.keys())
                        self.voice_combo.configure(values=sorted_names)

                        saved_voice = self.voice_name_var.get()
                        if saved_voice in sorted_names:
                            self.voice_combo.set(saved_voice)
                        else:
                            self.voice_combo.set(sorted_names[0])
                            self.voice_name_var.set(sorted_names[0])

                        self.save_settings()
                        self.log_resumido_msg(f"Vozes carregadas: {len(sorted_names)}")
                        messagebox.showinfo("Sucesso", f"{len(sorted_names)} vozes carregadas!")

            elif response.status_code == 401:
                messagebox.showerror("Erro", "Token inválido ou expirado!")
            else:
                messagebox.showerror("Erro", f"HTTP {response.status_code}")

        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def get_voice_id(self):
        voice_name = self.voice_name_var.get()
        if voice_name and voice_name in self.voices_map:
            return self.voices_map[voice_name]
        return None

    def submit_tts_request(self, text, voice_id, api_key, title=""):
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {"text": text, "voice": voice_id}
        if title:
            payload["title"] = title

        response = requests.post(API_BASE, headers=headers, json=payload, timeout=30, verify=False)

        if response.status_code in (402, 429):
            self.disable_key_temporarily(api_key, COOLDOWN_SECONDS)
            raise Exception(f"RATE_LIMIT: HTTP {response.status_code}")

        if response.status_code == 201:
            data = response.json()
            if data.get('ok'):
                audio_id = data.get('data', {}).get('created', {}).get('id')
                if audio_id:
                    return audio_id
            raise Exception(f"Resposta inesperada: {data}")

        try:
            err = response.json()
            msg = err.get('message', str(err))
        except:
            msg = response.text[:200]

        raise Exception(f"HTTP {response.status_code}: {msg}")

    def poll_tts_status(self, audio_id, api_key, thread_id=None):
        polling_interval = self.polling_interval_var.get()
        max_attempts = 180

        url = f"{API_BASE}/{audio_id}"

        for attempt in range(max_attempts):
            if self.cancel_requested:
                raise Exception("Cancelado pelo usuário")

            try:
                headers = {"Authorization": f"Bearer {api_key}"}
                response = requests.get(url, headers=headers, timeout=30, verify=False)

                if response.status_code == 200:
                    data = response.json()

                    if isinstance(data, dict):
                        status = data.get('status', '').upper()

                        if not status and 'data' in data:
                            status = data.get('data', {}).get('status', '').upper()

                        if status == 'DONE':
                            return True
                        elif status in ('PENDING', 'PROCESSING', 'QUEUED'):
                            time.sleep(polling_interval)
                            continue
                        elif status in ('FAILED', 'ERROR'):
                            raise Exception(f"Processamento falhou: {data}")

                elif response.status_code == 404:
                    raise Exception(f"Audio não encontrado: {audio_id}")

                time.sleep(polling_interval)

            except requests.exceptions.RequestException:
                time.sleep(polling_interval)
                continue

        raise Exception("Timeout: audio não ficou pronto em 6 minutos")

    def download_audio(self, audio_id, api_key, output_path, filename="audio"):
        safe_filename = "".join(c for c in filename if c.isalnum() or c in ('-', '_', ' ')).strip()
        if not safe_filename:
            safe_filename = "audio"

        url = f"{API_BASE}/audios/{audio_id}?name={safe_filename}"
        headers = {"Authorization": f"Bearer {api_key}"}

        response = requests.get(url, headers=headers, timeout=120, verify=False)

        if response.status_code == 200:
            content_type = response.headers.get('Content-Type', '')
            if 'audio' in content_type or len(response.content) > 1000:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                return True
            else:
                raise Exception("Resposta não contém audio válido")

        elif response.status_code == 404:
            raise Exception("Audio não encontrado para download")
        else:
            try:
                err = response.json()
                msg = err.get('message', str(err))
            except:
                msg = response.text[:200]
            raise Exception(f"HTTP {response.status_code}: {msg}")

    # ========================================================================
    # SCAN DE PASTA
    # ========================================================================

    def scan_factory_folder(self):
        if not self.pasta_raiz:
            self.scan_status.configure(text="Nenhuma pasta selecionada", text_color=CORES["error"])
            return

        if not os.path.exists(self.pasta_raiz):
            self.scan_status.configure(text="Pasta não existe!", text_color=CORES["error"])
            return

        txt_files = []
        img_files = []

        for root, dirs, files in os.walk(self.pasta_raiz):
            for f in files:
                full_path = os.path.join(root, f)
                if f.lower().endswith('.txt'):
                    txt_files.append(full_path)
                elif f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    img_files.append(full_path)

        txt_files.sort()
        img_files.sort()

        txt_count = len(txt_files)
        img_count = len(img_files)

        if txt_count == img_count and txt_count > 0:
            self.scan_status.configure(
                text=f"{txt_count} TXTs + {img_count} Imagens = {txt_count} pares OK",
                text_color=CORES["accent"]
            )
        elif txt_count != img_count:
            self.scan_status.configure(
                text=f"ERRO: {txt_count} TXTs != {img_count} Imagens",
                text_color=CORES["error"]
            )
        else:
            self.scan_status.configure(text="Nenhum arquivo encontrado", text_color=CORES["warning"])

    # ========================================================================
    # PROCESSAMENTO
    # ========================================================================

    def start_processing(self):
        current_tab = self.tabview.get()

        if current_tab == "Gerar Videos":
            self.start_factory_pipeline()
        else:
            self.start_legacy_render()

    def cancel_processing(self):
        if messagebox.askyesno("Cancelar", "Deseja realmente cancelar?"):
            self.cancel_requested = True
            self.log_resumido_msg("CANCELAMENTO SOLICITADO...")
            self.log_legacy("CANCELAMENTO SOLICITADO...")

    # ========================================================================
    # LOGICA DE VIDEO (OPENCV + FFMPEG)
    # ========================================================================

    def get_audio_duration(self, audio_path):
        try:
            cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                   "-of", "default=noprint_wrappers=1:nokey=1", audio_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except:
            pass
        return None

    def generate_pendulo_base(self, image_path, thread_id=None):
        try:
            tag = f"VIDEO T{thread_id}" if thread_id else "VIDEO"
            self.log_detalhado_msg("Gerando pendulo 10s...", tag)

            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                self.log_detalhado_msg(f"Erro ao carregar imagem: {image_path}", tag)
                return None

            has_alpha = img.shape[2] == 4 if len(img.shape) > 2 else False

            amplitude_graus = self.amplitude_var.get()
            crop_ratio = self.crop_ratio_var.get()
            zoom = self.zoom_var.get()

            fps = 24
            width = 1280
            height = 720
            duration = 10.0
            cycle_duration = 10.0
            total_frames = int(duration * fps)

            original_w = img.shape[1]
            original_h = img.shape[0]

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

            zoom_w = int(crop_w * zoom)
            zoom_h = int(crop_h * zoom)
            img_zoomed = cv2.resize(img_processed, (zoom_w, zoom_h), interpolation=cv2.INTER_LINEAR)

            max_angle_rad = abs(amplitude_graus) * math.pi / 180.0
            rotation_factor = 1.0 / math.cos(max_angle_rad) if max_angle_rad > 0 else 1.0

            canvas_diagonal = math.sqrt(width**2 + height**2)
            img_diagonal = math.sqrt(zoom_w**2 + zoom_h**2)
            scale_needed = (canvas_diagonal * rotation_factor) / img_diagonal if img_diagonal > 0 else 1.0

            final_w = int(zoom_w * scale_needed)
            final_h = int(zoom_h * scale_needed)

            if final_w < width:
                final_w = int(width * 1.1)
            if final_h < height:
                final_h = int(height * 1.1)

            img_resized = cv2.resize(img_zoomed, (final_w, final_h), interpolation=cv2.INTER_LINEAR)

            temp_dir = tempfile.mkdtemp(prefix=f"v7_t{thread_id}_")
            temp_video_path = os.path.join(temp_dir, "pendulo_base_10s.mp4")

            with self.temp_files_lock:
                self.temp_files.append(temp_dir)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

            if not out.isOpened():
                self.log_detalhado_msg("Erro ao criar video writer!", tag)
                return None

            center_img_x = final_w // 2
            center_img_y = final_h // 2
            center_canvas_x = width // 2
            center_canvas_y = height // 2

            half_cycle = cycle_duration / 2.0

            for frame_num in range(total_frames):
                if self.cancel_requested:
                    out.release()
                    return None

                t = (frame_num / fps) % cycle_duration

                if t <= half_cycle:
                    progress = t / half_cycle
                    angle_degrees = -amplitude_graus + (2 * amplitude_graus * progress)
                else:
                    progress = (t - half_cycle) / half_cycle
                    angle_degrees = amplitude_graus - (2 * amplitude_graus * progress)

                rotation_matrix = cv2.getRotationMatrix2D((center_img_x, center_img_y), angle_degrees, 1.0)

                if has_alpha:
                    img_rotated = cv2.warpAffine(img_resized, rotation_matrix, (final_w, final_h),
                                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                                 borderValue=(0, 0, 0, 0))
                else:
                    img_rotated = cv2.warpAffine(img_resized, rotation_matrix, (final_w, final_h),
                                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                                 borderValue=(0, 0, 0))

                frame = np.zeros((height, width, 3), dtype=np.uint8)

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

            out.release()

            self.log_detalhado_msg("Pendulo 10s OK!", tag)
            return temp_video_path

        except Exception as e:
            self.log_detalhado_msg(f"Erro ao gerar pendulo: {str(e)}", tag if thread_id else "VIDEO")
            return None

    def create_cell_base(self, pendulo_10s, thread_id=None):
        try:
            tag = f"VIDEO T{thread_id}" if thread_id else "VIDEO"

            has_overlay = self.overlay_path and os.path.exists(self.overlay_path)
            has_vinheta = self.vinheta_path and os.path.exists(self.vinheta_path)

            if not has_overlay and not has_vinheta:
                self.log_detalhado_msg("Sem overlay/vinheta - pulando celula base", tag)
                return pendulo_10s

            self.log_detalhado_msg("Criando celula base (10s)...", tag)

            temp_dir = os.path.dirname(pendulo_10s)
            cell_base_path = os.path.join(temp_dir, "celula_base_completa.mp4")

            env = os.environ.copy()

            if self.usar_gpu_var.get() and self.gpu_index:
                env['CUDA_VISIBLE_DEVICES'] = self.gpu_index
                env['OPENCV_OPENCL_DEVICE'] = 'disabled'
                env['OPENCV_OPENCL_RUNTIME'] = ''

            cmd = ["ffmpeg", "-y", "-vsync", "0"]

            if self.usar_gpu_var.get() and self.check_nvenc_support():
                cmd.extend(["-hwaccel", "auto"])

            cmd.extend(["-i", pendulo_10s])

            input_idx = 1

            if has_overlay:
                cmd.extend(["-i", self.overlay_path, "-t", "10"])
                input_idx += 1

            if has_vinheta:
                cmd.extend(["-loop", "1", "-i", self.vinheta_path, "-t", "10"])
                input_idx += 1

            filters = []
            filters.append("[0:v]scale=1280:720,setpts=PTS-STARTPTS[base]")
            current_layer = "[base]"

            if has_overlay:
                chroma_color = self.chroma_color_var.get()
                similarity = self.chroma_similarity_var.get()
                blend = self.chroma_blend_var.get()

                filters.append("[1:v]scale=1280:720[overlay_sc]")
                filters.append(f"[overlay_sc]chromakey=color=0x{chroma_color}:similarity={similarity}:blend={blend}[overlay_key]")
                filters.append(f"{current_layer}[overlay_key]overlay=(W-w)/2:(H-h)/2[with_overlay]")
                current_layer = "[with_overlay]"

            if has_vinheta:
                vinheta_input_idx = 2 if has_overlay else 1
                filters.append(f"[{vinheta_input_idx}:v]scale=1280:720,format=rgba[vinheta_sc]")
                filters.append(f"{current_layer}[vinheta_sc]overlay=(W-w)/2:(H-h)/2[vout]")
            else:
                filters.append(f"{current_layer}null[vout]")

            cmd.extend(["-filter_complex", ";".join(filters)])
            cmd.extend(["-map", "[vout]"])

            if self.usar_gpu_var.get() and self.check_nvenc_support():
                cmd.extend([
                    "-c:v", "h264_nvenc",
                    "-gpu", "0",
                    "-preset", "p1",
                    "-tune", "hq",
                    "-rc", "vbr",
                    "-cq", "23",
                    "-b:v", "8M",
                    "-profile:v", "high"
                ])
            else:
                cmd.extend(["-c:v", "libx264", "-preset", "superfast", "-crf", "23"])

            cmd.extend([
                "-t", "10",
                "-pix_fmt", "yuv420p",
                "-an",
                cell_base_path
            ])

            process = subprocess.Popen(cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, env=env)
            process.wait()

            if process.returncode == 0:
                self.log_detalhado_msg("Celula base OK!", tag)
                return cell_base_path
            else:
                self.log_detalhado_msg(f"Erro celula base (cod: {process.returncode})", tag)
                return None

        except Exception as e:
            self.log_detalhado_msg(f"Erro celula base: {str(e)}", tag if thread_id else "VIDEO")
            return None

    def assemble_final_video(self, cell_base, audio_path, output_path, audio_duration, thread_id=None):
        try:
            tag = f"VIDEO T{thread_id}" if thread_id else "VIDEO"
            self.log_detalhado_msg("Montando video final...", tag)

            loops_needed = math.ceil(audio_duration / 10.0)

            env = os.environ.copy()

            if self.usar_gpu_var.get() and self.gpu_index:
                env['CUDA_VISIBLE_DEVICES'] = self.gpu_index
                env['OPENCV_OPENCL_DEVICE'] = 'disabled'
                env['OPENCV_OPENCL_RUNTIME'] = ''

            cmd = ["ffmpeg", "-y", "-vsync", "0"]

            if self.usar_gpu_var.get() and self.check_nvenc_support():
                cmd.extend(["-hwaccel", "auto"])

            cmd.extend(["-stream_loop", str(loops_needed - 1), "-i", cell_base])
            cmd.extend(["-i", audio_path])

            has_musica = self.musica_path and os.path.exists(self.musica_path)

            if has_musica:
                cmd.extend(["-stream_loop", "-1", "-i", self.musica_path])

            if not has_musica:
                cmd.extend(["-map", "0:v", "-map", "1:a"])
            else:
                filters = []
                filters.append("[0:v]null[vout]")

                vol = self.musica_volume_var.get()
                filters.append("[1:a]volume=1.0[nar]")
                filters.append(f"[2:a]volume={vol}[mus]")
                filters.append("[nar][mus]amix=inputs=2:duration=first[aout]")

                cmd.extend(["-filter_complex", ";".join(filters)])
                cmd.extend(["-map", "[vout]", "-map", "[aout]"])

            if self.usar_gpu_var.get() and self.check_nvenc_support():
                cmd.extend([
                    "-c:v", "h264_nvenc",
                    "-gpu", "0",
                    "-preset", "p1",
                    "-tune", "hq",
                    "-rc", "vbr",
                    "-cq", "23",
                    "-b:v", "8M",
                    "-profile:v", "high"
                ])
            else:
                cmd.extend(["-c:v", "libx264", "-preset", "superfast", "-crf", "23"])

            cmd.extend([
                "-c:a", "aac",
                "-b:a", "192k",
                "-t", str(audio_duration),
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                output_path
            ])

            process = subprocess.Popen(cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, env=env)
            process.wait()

            if process.returncode == 0:
                self.log_detalhado_msg(f"Salvo: {os.path.basename(output_path)}", tag)
                return True
            else:
                self.log_detalhado_msg(f"Erro FFmpeg (cod: {process.returncode})", tag)
                return False

        except Exception as e:
            self.log_detalhado_msg(f"Erro montagem: {str(e)}", tag if thread_id else "VIDEO")
            return False

    def cleanup_temp_files(self):
        self.log_detalhado_msg("Limpando temporarios...", "SYSTEM")
        with self.temp_files_lock:
            for path in self.temp_files:
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                    elif os.path.isdir(path):
                        shutil.rmtree(path)
                except:
                    pass
            self.temp_files.clear()

    # ========================================================================
    # FACTORY PIPELINE
    # ========================================================================

    def scan_for_pairs(self):
        txt_files = []
        img_files = []

        for root, dirs, files in os.walk(self.pasta_raiz):
            for f in files:
                full_path = os.path.join(root, f)
                if f.lower().endswith('.txt'):
                    txt_files.append((full_path, root))
                elif f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    img_files.append(full_path)

        txt_files.sort(key=lambda x: x[0])
        img_files.sort()

        if len(txt_files) != len(img_files):
            raise Exception(f"Quantidades diferentes: {len(txt_files)} TXTs vs {len(img_files)} Imagens")

        pairs = []
        for i, (txt_path, folder) in enumerate(txt_files):
            pairs.append((txt_path, img_files[i], folder))

        return pairs

    def start_factory_pipeline(self):
        if self.processing:
            messagebox.showwarning("Aviso", "Já existe um processo em andamento!")
            return

        if not self.pasta_raiz:
            messagebox.showerror("Erro", "Selecione uma pasta de entrada!")
            return

        if not self.pasta_saida:
            messagebox.showerror("Erro", "Selecione uma pasta de saída!")
            return

        active_keys = [k for k in self.api_keys_list
                       if not k.get("disabled_until") or datetime.now() >= k.get("disabled_until")]
        if not active_keys:
            messagebox.showerror("Erro", "Nenhuma chave API ativa!\nConfigure em API Keys.")
            return

        if not self.voice_name_var.get() or not self.voices_map:
            messagebox.showerror("Erro", "Clique em 'Listar Vozes' e selecione uma voz!")
            return

        self.processing = True
        self.cancel_requested = False
        self.consecutive_failures = 0
        self.completed_audio = 0
        self.completed_video = 0
        self.success_count = 0
        self.error_count = 0
        self.current_key_index = 0

        self.start_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.log_resumido.delete("1.0", "end")
        self.log_detalhado.delete("1.0", "end")

        self.save_settings()

        threading.Thread(target=self.factory_pipeline_process, daemon=True).start()

    def factory_pipeline_process(self):
        try:
            self.log_resumido_msg("=" * 50)
            self.log_resumido_msg("RENDER V7 - Gerar Videos")
            self.log_resumido_msg("=" * 50)

            self.log_detalhado_msg(f"Pasta entrada: {self.pasta_raiz}", "SYSTEM")
            self.log_detalhado_msg(f"Pasta saida: {self.pasta_saida}", "SYSTEM")
            self.log_detalhado_msg(f"GPU: {self.gpu_index}", "SYSTEM")
            self.log_detalhado_msg(f"Audio threads: {self.audio_threads_var.get()}", "SYSTEM")
            self.log_detalhado_msg(f"Video threads: {self.video_threads_var.get()}", "SYSTEM")

            try:
                pairs = self.scan_for_pairs()
            except Exception as e:
                self.log_resumido_msg(f"ERRO: {str(e)}")
                messagebox.showerror("Erro", str(e))
                return

            if not pairs:
                self.log_resumido_msg("Nenhum par encontrado!")
                messagebox.showinfo("Aviso", "Nenhum arquivo para processar!")
                return

            self.total_jobs = len(pairs)
            self.log_resumido_msg(f"Encontrados {self.total_jobs} pares para processar")

            self.audio_queue = queue.Queue()
            self.video_queue = queue.Queue()

            for txt_path, img_path, folder in pairs:
                base_name = os.path.splitext(os.path.basename(txt_path))[0]
                mp3_path = os.path.splitext(txt_path)[0] + '.mp3'

                rel_path = os.path.relpath(folder, self.pasta_raiz)
                output_subdir = os.path.join(self.pasta_saida, rel_path)
                mp4_path = os.path.join(output_subdir, f"{base_name}_final.mp4")

                if os.path.exists(mp3_path) and os.path.exists(mp4_path):
                    self.log_detalhado_msg(f"Pulando (completo): {base_name}", "SKIP")
                    self.completed_audio += 1
                    self.completed_video += 1
                    continue

                job = FactoryJob(
                    txt_path=txt_path,
                    image_path=img_path,
                    source_folder=folder
                )

                if os.path.exists(mp3_path):
                    job.audio_path = mp3_path
                    job.status = "audio_done"
                    self.video_queue.put(job)
                    self.completed_audio += 1
                    self.log_detalhado_msg(f"MP3 existente: {base_name}", "SKIP")
                else:
                    self.audio_queue.put(job)

            self.update_factory_progress()

            self.audio_producing = True

            audio_threads = []
            for i in range(self.audio_threads_var.get()):
                t = threading.Thread(target=self.audio_producer_worker, args=(i+1,), daemon=True)
                t.start()
                audio_threads.append(t)

            video_threads = []
            for i in range(self.video_threads_var.get()):
                t = threading.Thread(target=self.video_consumer_worker, args=(i+1,), daemon=True)
                t.start()
                video_threads.append(t)

            for t in audio_threads:
                t.join()

            self.audio_producing = False

            for _ in range(self.video_threads_var.get()):
                self.video_queue.put(None)

            for t in video_threads:
                t.join()

            self.log_resumido_msg("=" * 50)
            if self.cancel_requested:
                self.log_resumido_msg("CANCELADO")
            else:
                self.log_resumido_msg("CONCLUIDO!")
            self.log_resumido_msg(f"Sucessos: {self.success_count} | Erros: {self.error_count}")
            self.log_resumido_msg("=" * 50)

            if not self.cancel_requested:
                msg = f"Concluído!\n\nSucessos: {self.success_count}\nErros: {self.error_count}"
                self.after(0, lambda: messagebox.showinfo("Concluído", msg))

        except Exception as e:
            import traceback
            self.log_detalhado_msg(f"ERRO CRITICO: {traceback.format_exc()}", "ERROR")
            self.after(0, lambda: messagebox.showerror("Erro", str(e)))

        finally:
            self.cleanup_temp_files()
            self.processing = False
            self.cancel_requested = False
            self.after(0, lambda: self.start_btn.configure(state="normal"))
            self.after(0, lambda: self.cancel_btn.configure(state="disabled"))
            self.after(0, self.refresh_keys_list)

    def audio_producer_worker(self, thread_id):
        tag = f"AUDIO T{thread_id}"

        while not self.cancel_requested:
            try:
                job = self.audio_queue.get(timeout=1)
            except queue.Empty:
                if self.audio_queue.empty():
                    break
                continue

            if job is None:
                break

            filename = os.path.basename(job.txt_path)
            base_name = os.path.splitext(filename)[0]
            mp3_path = os.path.splitext(job.txt_path)[0] + '.mp3'

            self.log_detalhado_msg(f"Processando: {filename}", tag)

            try:
                try:
                    with open(job.txt_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                except UnicodeDecodeError:
                    with open(job.txt_path, 'r', encoding='latin-1') as f:
                        text = f.read().strip()

                if not text:
                    self.log_resumido_msg(f"PULADO (vazio): {base_name}")
                    job.status = "error"
                    job.error_message = "Arquivo vazio"
                    with self.progress_lock:
                        self.error_count += 1
                    continue

                if len(text) > MAX_TEXT_LENGTH:
                    self.log_resumido_msg(f"ERRO (muito longo): {base_name}")
                    job.status = "error"
                    job.error_message = f"Texto muito longo ({len(text)} chars)"
                    with self.progress_lock:
                        self.error_count += 1
                    self.check_consecutive_failure()
                    continue

                api_key, key_name = self.get_next_api_key()
                if not api_key:
                    self.log_detalhado_msg("Sem chaves ativas, aguardando...", tag)
                    time.sleep(10)
                    api_key, key_name = self.get_next_api_key()
                    if not api_key:
                        self.log_resumido_msg(f"ERRO (sem chaves): {base_name}")
                        job.status = "error"
                        self.check_consecutive_failure()
                        continue

                self.log_detalhado_msg(f"Enviando via '{key_name}'...", tag)
                voice_id = self.get_voice_id()
                audio_id = self.submit_tts_request(text, voice_id, api_key, base_name)
                self.log_detalhado_msg(f"UUID: {audio_id[:12]}...", tag)

                self.log_detalhado_msg("Aguardando processamento...", tag)
                self.poll_tts_status(audio_id, api_key, thread_id)

                self.log_detalhado_msg("Baixando...", tag)
                self.download_audio(audio_id, api_key, mp3_path, base_name)

                job.audio_path = mp3_path
                job.status = "audio_done"
                self.video_queue.put(job)

                self.log_resumido_msg(f"MP3 OK: {base_name} -> renderizando...")

                with self.progress_lock:
                    self.completed_audio += 1

                with self.failure_lock:
                    self.consecutive_failures = 0

                self.update_factory_progress()

            except Exception as e:
                error_msg = str(e)
                self.log_detalhado_msg(f"ERRO: {error_msg}", tag)
                self.log_resumido_msg(f"FALHOU: {base_name}")

                job.status = "error"
                job.error_message = error_msg

                with self.progress_lock:
                    self.error_count += 1

                if self.check_consecutive_failure():
                    break

                self.update_factory_progress()

    def video_consumer_worker(self, thread_id):
        tag = f"VIDEO T{thread_id}"

        while not self.cancel_requested:
            try:
                job = self.video_queue.get(timeout=2)
            except queue.Empty:
                if not self.audio_producing and self.video_queue.empty():
                    break
                continue

            if job is None:
                break

            if job.status != "audio_done":
                continue

            base_name = os.path.splitext(os.path.basename(job.txt_path))[0]

            self.log_detalhado_msg(f"Renderizando: {base_name}", tag)

            try:
                rel_path = os.path.relpath(job.source_folder, self.pasta_raiz)
                output_subdir = os.path.join(self.pasta_saida, rel_path)
                os.makedirs(output_subdir, exist_ok=True)
                output_path = os.path.join(output_subdir, f"{base_name}_final.mp4")

                if os.path.exists(output_path):
                    self.log_detalhado_msg(f"Video já existe: {base_name}", tag)
                    job.status = "video_done"
                    with self.progress_lock:
                        self.completed_video += 1
                        self.success_count += 1
                    self.update_factory_progress()
                    continue

                audio_duration = self.get_audio_duration(job.audio_path)
                if not audio_duration:
                    raise Exception("Erro ao obter duração do audio")

                self.log_detalhado_msg(f"Duração: {audio_duration:.2f}s", tag)

                pendulo_10s = self.generate_pendulo_base(job.image_path, thread_id)
                if not pendulo_10s:
                    raise Exception("Falha ao gerar pendulo")

                cell_base = self.create_cell_base(pendulo_10s, thread_id)
                if not cell_base:
                    raise Exception("Falha ao criar celula base")

                success = self.assemble_final_video(cell_base, job.audio_path, output_path, audio_duration, thread_id)

                if success:
                    job.status = "video_done"
                    job.output_path = output_path
                    self.log_resumido_msg(f"VIDEO OK: {base_name}_final.mp4")

                    with self.progress_lock:
                        self.completed_video += 1
                        self.success_count += 1
                else:
                    raise Exception("Falha na montagem final")

            except Exception as e:
                self.log_detalhado_msg(f"ERRO: {str(e)}", tag)
                self.log_resumido_msg(f"VIDEO FALHOU: {base_name}")

                job.status = "error"
                job.error_message = str(e)

                with self.progress_lock:
                    self.error_count += 1

            self.update_factory_progress()

    def check_consecutive_failure(self):
        with self.failure_lock:
            self.consecutive_failures += 1

            if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                self.cancel_requested = True
                self.log_resumido_msg("=" * 50)
                self.log_resumido_msg(f"PARADO: {MAX_CONSECUTIVE_FAILURES} falhas consecutivas!")
                self.log_resumido_msg("Verifique a API e suas chaves.")
                self.log_resumido_msg("=" * 50)

                self.after(0, lambda: messagebox.showerror(
                    "Pipeline Interrompido",
                    f"{MAX_CONSECUTIVE_FAILURES} falhas consecutivas!\n\n"
                    "Verifique:\n- API Darkvi\n- Chaves ativas\n- Conexão"
                ))
                return True

        return False

    def update_factory_progress(self):
        def _update():
            if self.total_jobs > 0:
                audio_pct = self.completed_audio / self.total_jobs
                video_pct = self.completed_video / self.total_jobs

                self.progress_audio.set(audio_pct)
                self.progress_video.set(video_pct)

                self.progress_audio_label.configure(text=f"{audio_pct*100:.1f}% ({self.completed_audio}/{self.total_jobs})")
                self.progress_video_label.configure(text=f"{video_pct*100:.1f}% ({self.completed_video}/{self.total_jobs})")

        with self.progress_lock:
            self.after(0, _update)

    # ========================================================================
    # LEGACY MODE (IMPORTAR AUDIO)
    # ========================================================================

    def start_legacy_render(self):
        if self.processing:
            messagebox.showwarning("Aviso", "Já existe um processo em andamento!")
            return

        if not self.pasta_raiz:
            messagebox.showerror("Erro", "Selecione uma pasta de entrada!")
            return

        if not self.pasta_saida:
            messagebox.showerror("Erro", "Selecione uma pasta de saída!")
            return

        self.processing = True
        self.cancel_requested = False
        self.completed_videos_legacy = 0
        self.total_videos_legacy = 0

        self.start_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.legacy_log.delete("1.0", "end")

        self.save_settings()

        threading.Thread(target=self.legacy_render_process, daemon=True).start()

    def legacy_render_process(self):
        try:
            self.log_legacy("=" * 60)
            self.log_legacy("RENDER V7 - Importar Audio")
            self.log_legacy("Desenvolvido por Perrenoud")
            self.log_legacy("=" * 60)
            self.log_legacy(f"GPU: {self.gpu_index}")
            self.log_legacy(f"Renders Simultâneos: {self.video_threads_var.get()}")
            self.log_legacy("=" * 60)

            self.log_legacy(f"\nEscaneando: {self.pasta_raiz}")
            jobs = list(self.scan_folders_legacy())

            if not jobs:
                self.log_legacy("\nNenhuma pasta válida encontrada!")
                messagebox.showwarning("Aviso", "Nenhuma pasta válida!")
                return

            self.log_legacy(f"\nEncontradas {len(jobs)} pasta(s) válida(s)")

            tasks = []
            for folder, pairs in jobs:
                for audio_file, image_file in pairs:
                    audio_path = os.path.join(folder, audio_file)
                    image_path = os.path.join(folder, image_file)
                    tasks.append((audio_path, image_path, folder))

            self.total_videos_legacy = len(tasks)
            self.log_legacy(f"Total de videos: {self.total_videos_legacy}\n")

            max_workers = self.video_threads_var.get()
            self.log_legacy(f"Iniciando pool com {max_workers} worker(s)...\n")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_task = {}
                for idx, task in enumerate(tasks, 1):
                    if self.cancel_requested:
                        break
                    future = executor.submit(self.legacy_render_single_video_wrapper, task, idx)
                    future_to_task[future] = (idx, task)

                for future in as_completed(future_to_task):
                    if self.cancel_requested:
                        executor.shutdown(wait=False, cancel_futures=True)
                        break

                    idx, task = future_to_task[future]
                    try:
                        success = future.result()
                        if success:
                            self.log_legacy(f"Video {idx} concluído!", thread_id=idx)
                        else:
                            self.log_legacy(f"Video {idx} FALHOU!", thread_id=idx)
                    except Exception as e:
                        self.log_legacy(f"Video {idx} ERRO: {str(e)}", thread_id=idx)

                    self.completed_videos_legacy += 1
                    self.update_legacy_progress()

            if self.cancel_requested:
                self.log_legacy("\nPROCESSAMENTO CANCELADO")
            else:
                self.log_legacy("\n" + "=" * 60)
                self.log_legacy("RENDERIZAÇÃO CONCLUÍDA!")
                self.log_legacy("=" * 60)
                messagebox.showinfo("Concluído", f"{self.completed_videos_legacy}/{self.total_videos_legacy} videos renderizados!")

        except Exception as e:
            import traceback
            self.log_legacy(f"\nERRO CRÍTICO:\n{traceback.format_exc()}")
            messagebox.showerror("Erro", str(e))

        finally:
            self.cleanup_temp_files()
            self.processing = False
            self.cancel_requested = False
            self.after(0, lambda: self.start_btn.configure(state="normal"))
            self.after(0, lambda: self.cancel_btn.configure(state="disabled"))

    def scan_folders_legacy(self):
        for root, dirs, files in os.walk(self.pasta_raiz):
            audios = sorted([f for f in files if f.lower().endswith(('.mp3', '.wav', '.m4a'))])
            images = sorted([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])

            if audios and images:
                if len(audios) == len(images):
                    yield (root, list(zip(audios, images)))
                else:
                    rel_path = os.path.relpath(root, self.pasta_raiz)
                    self.log_legacy(f"Pasta '{rel_path}': {len(audios)} audios, {len(images)} imagens - PULADA")

    def legacy_render_single_video_wrapper(self, task, thread_id):
        audio_path, image_path, source_folder = task
        audio_file = os.path.basename(audio_path)

        self.log_legacy(f"Iniciando: {audio_file}", thread_id=thread_id)

        try:
            return self.legacy_render_single_video(audio_path, image_path, source_folder, thread_id)
        except Exception as e:
            self.log_legacy(f"Erro: {str(e)}", thread_id=thread_id)
            return False

    def legacy_render_single_video(self, audio_path, image_path, source_folder, thread_id):
        try:
            audio_duration = self.get_audio_duration(audio_path)
            if not audio_duration:
                self.log_legacy("Erro ao obter duração do audio!", thread_id=thread_id)
                return False

            self.log_legacy(f"Duração: {audio_duration:.2f}s", thread_id=thread_id)

            pendulo_10s = self.generate_pendulo_base(image_path, thread_id)
            if not pendulo_10s:
                return False

            cell_base = self.create_cell_base(pendulo_10s, thread_id)
            if not cell_base:
                return False

            audio_name = os.path.splitext(os.path.basename(audio_path))[0]
            rel_path = os.path.relpath(source_folder, self.pasta_raiz)
            output_subdir = os.path.join(self.pasta_saida, rel_path)
            os.makedirs(output_subdir, exist_ok=True)
            output_path = os.path.join(output_subdir, f"{audio_name}_final.mp4")

            success = self.assemble_final_video(cell_base, audio_path, output_path, audio_duration, thread_id)

            return success

        except Exception as e:
            self.log_legacy(f"Erro: {str(e)}", thread_id=thread_id)
            return False

    def update_legacy_progress(self):
        def _update():
            if self.total_videos_legacy > 0:
                progress = self.completed_videos_legacy / self.total_videos_legacy
                self.progress_video.set(progress)
                self.progress_video_label.configure(text=f"{progress*100:.1f}% ({self.completed_videos_legacy}/{self.total_videos_legacy})")

        with self.progress_lock:
            self.after(0, _update)


# ============================================================================
# MAIN
# ============================================================================

def main():
    try:
        import cv2
        import numpy
    except ImportError as e:
        print(f"ERRO: Dependência não encontrada: {e}")
        print("\nExecute: pip install opencv-python numpy")
        input("Pressione Enter para sair...")
        sys.exit(1)

    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
    except:
        print("AVISO: FFmpeg não encontrado!")
        print("Baixe de: https://ffmpeg.org/download.html\n")

    app = RenderV7()
    app.mainloop()


if __name__ == "__main__":
    main()
