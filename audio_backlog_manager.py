# -*- coding: utf-8 -*-
"""
Sistema de Gerenciamento de Backlog de Áudios
=============================================
Gerencia histórico de áudios usados para evitar repetição entre vídeos.
"""

import os
import json
import random
from typing import List, Optional
from pathlib import Path
from datetime import datetime


class AudioBacklogManager:
    """Gerencia histórico de áudios usados para evitar repetição."""

    def __init__(self, history_file: str = "audio_backlog_history.json", max_history: int = 50):
        """
        Inicializa o gerenciador de backlog de áudios.

        Args:
            history_file: Caminho do arquivo JSON para armazenar histórico
            max_history: Número máximo de entradas no histórico (padrão: 50)
        """
        self.history_file = history_file
        self.max_history = max_history
        self.history = self.load_history()

    def load_history(self) -> List[str]:
        """
        Carrega histórico de áudios usados do arquivo JSON.

        Returns:
            Lista de caminhos de áudios usados (mais recentes primeiro)
        """
        if not os.path.exists(self.history_file):
            return []

        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Retornar lista de áudios (mais recentes primeiro)
                return data.get("used_audios", [])
        except (json.JSONDecodeError, IOError) as e:
            print(f"[WARN] Erro ao carregar histórico de áudios: {e}")
            return []

    def save_history(self):
        """Salva histórico de áudios usados no arquivo JSON."""
        try:
            data = {
                "used_audios": self.history,
                "last_updated": datetime.now().isoformat(),
                "total_entries": len(self.history)
            }
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"[WARN] Erro ao salvar histórico de áudios: {e}")

    def get_available_audios(self, audio_folder: str) -> List[str]:
        """
        Lista áudios disponíveis na pasta especificada.

        Args:
            audio_folder: Caminho da pasta com áudios

        Returns:
            Lista de caminhos completos dos arquivos de áudio
        """
        if not os.path.exists(audio_folder):
            return []

        audio_extensions = ('.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac')
        audios = []

        for file in os.listdir(audio_folder):
            file_path = os.path.join(audio_folder, file)
            if os.path.isfile(file_path) and file.lower().endswith(audio_extensions):
                audios.append(file_path)

        return sorted(audios)

    def get_recently_used(self, count: int = 10) -> List[str]:
        """
        Retorna lista dos últimos N áudios usados.

        Args:
            count: Número de áudios recentes a retornar

        Returns:
            Lista dos últimos N áudios usados
        """
        return self.history[:count]

    def mark_audio_used(self, audio_path: str):
        """
        Registra um áudio como usado.

        Args:
            audio_path: Caminho do áudio usado
        """
        # Normalizar caminho (absoluto)
        normalized_path = os.path.abspath(audio_path)

        # Remover se já existe (para mover para o topo)
        if normalized_path in self.history:
            self.history.remove(normalized_path)

        # Adicionar no início (mais recente primeiro)
        self.history.insert(0, normalized_path)

        # Limitar histórico ao máximo especificado
        if len(self.history) > self.max_history:
            self.history = self.history[:self.max_history]

        # Salvar histórico
        self.save_history()

    def get_next_audio(self, audio_folder: str, exclude_recent: int = 10) -> Optional[str]:
        """
        Seleciona próximo áudio evitando repetição dos últimos N usados.

        Args:
            audio_folder: Pasta com áudios disponíveis
            exclude_recent: Número de áudios recentes a evitar (padrão: 10)

        Returns:
            Caminho do próximo áudio ou None se não houver disponível
        """
        # Listar áudios disponíveis
        available_audios = self.get_available_audios(audio_folder)

        if not available_audios:
            return None

        # Normalizar caminhos para comparação
        available_audios = [os.path.abspath(a) for a in available_audios]

        # Obter áudios recentemente usados
        recently_used = set(self.get_recently_used(exclude_recent))

        # Filtrar áudios não usados recentemente
        unused_audios = [a for a in available_audios if a not in recently_used]

        # Se todos foram usados recentemente, resetar e usar todos
        if not unused_audios:
            print(f"[INFO] Todos os áudios foram usados recentemente. Resetando histórico.")
            unused_audios = available_audios
            # Limpar histórico (mas manter estrutura)
            self.history = []

        # Selecionar aleatoriamente
        selected = random.choice(unused_audios)
        return selected

    def reset_history(self):
        """Reseta o histórico de áudios usados."""
        self.history = []
        self.save_history()

    def get_history_count(self) -> int:
        """
        Retorna número de áudios no histórico.

        Returns:
            Número de entradas no histórico
        """
        return len(self.history)







