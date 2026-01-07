# -*- coding: utf-8 -*-
"""
Módulo de Geração e Processamento de Arquivos SRT
==================================================
Gerencia obtenção e processamento de arquivos de legendas SRT.
"""

import os
import re
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SRTBlock:
    """Representa um bloco de legenda SRT."""
    index: int
    start_time: str  # Formato HH:MM:SS,mmm
    end_time: str    # Formato HH:MM:SS,mmm
    text: str


class SRTGenerator:
    """Gerador e processador de arquivos SRT."""

    def __init__(self, log_callback=None):
        """
        Inicializa o gerador de SRT.

        Args:
            log_callback: Função opcional para logging (message, level)
        """
        self.log = log_callback or (lambda msg, level="INFO": print(f"[{level}] {msg}"))

    def parse_srt_file(self, srt_path: str) -> List[SRTBlock]:
        """
        Parse arquivo SRT em blocos estruturados.

        Args:
            srt_path: Caminho do arquivo SRT

        Returns:
            Lista de blocos SRT parseados
        """
        encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']
        srt_content = None

        # Tentar ler com diferentes encodings
        for enc in encodings:
            try:
                with open(srt_path, 'r', encoding=enc) as f:
                    srt_content = f.read()
                self.log(f"SRT lido com encoding: {enc}", "OK")
                break
            except UnicodeDecodeError:
                continue

        if not srt_content:
            raise Exception(f"Não foi possível ler o arquivo SRT: {srt_path}")

        blocks = []
        for block_text in srt_content.strip().split('\n\n'):
            lines = block_text.strip().split('\n')
            if len(lines) >= 3:
                try:
                    index = int(lines[0])
                    time_line = lines[1]
                    text = ' '.join(lines[2:])
                    text = re.sub(r'<[^>]+>', '', text).replace('\n', '\\N')

                    # Parse timestamp: HH:MM:SS,mmm --> HH:MM:SS,mmm
                    match = re.match(
                        r'(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})',
                        time_line
                    )
                    if match:
                        start_time = f"{match.group(1)}:{match.group(2)}:{match.group(3)},{match.group(4)}"
                        end_time = f"{match.group(5)}:{match.group(6)}:{match.group(7)},{match.group(8)}"
                        blocks.append(SRTBlock(index, start_time, end_time, text))
                except (ValueError, IndexError):
                    continue

        self.log(f"SRT parseado: {len(blocks)} blocos", "OK")
        return blocks

    def srt_time_to_seconds(self, srt_time: str) -> float:
        """
        Converte timestamp SRT (HH:MM:SS,mmm) para segundos.

        Args:
            srt_time: Timestamp no formato HH:MM:SS,mmm

        Returns:
            Tempo em segundos (float)
        """
        parts = srt_time.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds_parts = parts[2].split(',')
        seconds = int(seconds_parts[0])
        milliseconds = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0

        return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0

    def group_blocks(self, blocks: List[SRTBlock], group_size: int = 3) -> List[List[SRTBlock]]:
        """
        Agrupa blocos SRT em grupos de N blocos.

        Args:
            blocks: Lista de blocos SRT
            group_size: Tamanho do grupo (padrão: 3)

        Returns:
            Lista de grupos de blocos
        """
        groups = []
        for i in range(0, len(blocks), group_size):
            group = blocks[i:i + group_size]
            groups.append(group)
        return groups

    def get_group_text(self, group: List[SRTBlock]) -> str:
        """
        Combina texto de um grupo de blocos SRT.

        Args:
            group: Lista de blocos SRT

        Returns:
            Texto combinado
        """
        texts = [block.text for block in group]
        return " ".join(texts)

    def get_group_duration(self, group: List[SRTBlock]) -> Tuple[float, float]:
        """
        Calcula duração de um grupo de blocos SRT.

        Args:
            group: Lista de blocos SRT

        Returns:
            Tupla (start_time_seconds, end_time_seconds)
        """
        if not group:
            return (0.0, 0.0)

        # Start time do primeiro bloco
        start_time = self.srt_time_to_seconds(group[0].start_time)
        # End time do último bloco
        end_time = self.srt_time_to_seconds(group[-1].end_time)

        return (start_time, end_time)

    def generate_srt_from_assemblyai(self, audio_path: str, api_key: str) -> List[SRTBlock]:
        """
        Gera blocos SRT a partir de áudio usando AssemblyAI.

        Args:
            audio_path: Caminho do arquivo de áudio
            api_key: Chave API do AssemblyAI

        Returns:
            Lista de blocos SRT
        """
        try:
            import assemblyai as aai
            aai.settings.api_key = api_key

            self.log("Iniciando transcrição AssemblyAI...", "INFO")

            config = aai.TranscriptionConfig(language_detection=True)
            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe(audio_path, config=config)

            if transcript.status == aai.TranscriptStatus.error:
                raise Exception(f"Erro AssemblyAI: {transcript.error}")

            if not transcript.words:
                raise Exception("Nenhuma palavra detectada no áudio")

            self.log(f"Transcrição concluída: {len(transcript.words)} palavras", "OK")

            # Agrupar palavras em segmentos (4 palavras por linha)
            blocks = []
            segment_words = []
            max_words = 4

            for i, word in enumerate(transcript.words):
                segment_words.append(word)

                if len(segment_words) >= max_words or i == len(transcript.words) - 1:
                    start_ms = segment_words[0].start
                    end_ms = segment_words[-1].end

                    # Converter para formato SRT
                    start_time = self.ms_to_srt_time(start_ms)
                    end_time = self.ms_to_srt_time(end_ms)
                    text = " ".join(w.text for w in segment_words)

                    blocks.append(SRTBlock(len(blocks) + 1, start_time, end_time, text))
                    segment_words = []

            self.log(f"Blocos SRT gerados: {len(blocks)}", "OK")
            return blocks

        except ImportError:
            raise Exception("Biblioteca 'assemblyai' não instalada. Execute: pip install assemblyai")

    def ms_to_srt_time(self, ms: int) -> str:
        """
        Converte milissegundos para formato SRT (HH:MM:SS,mmm).

        Args:
            ms: Milissegundos

        Returns:
            Timestamp no formato HH:MM:SS,mmm
        """
        hours = ms // 3600000
        minutes = (ms % 3600000) // 60000
        seconds = (ms % 60000) // 1000
        milliseconds = ms % 1000

        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def fetch_srt_from_darkvi(self, audio_id: str, api_key: str) -> Optional[List[SRTBlock]]:
        """
        Busca SRT via API DarkVie (se disponível).

        Nota: Esta função é um placeholder. A API DarkVie pode não ter
        endpoint específico para SRT. Será necessário verificar documentação.

        Args:
            audio_id: ID do áudio na DarkVie
            api_key: Chave API da DarkVie

        Returns:
            Lista de blocos SRT ou None se não disponível
        """
        # TODO: Implementar quando documentação da DarkVie estiver disponível
        # Por enquanto, retorna None e usa AssemblyAI como fallback
        self.log("API DarkVie para SRT ainda não implementada. Use AssemblyAI.", "WARN")
        return None

    def save_srt_file(self, blocks: List[SRTBlock], output_path: str):
        """
        Salva blocos SRT em arquivo.

        Args:
            blocks: Lista de blocos SRT
            output_path: Caminho do arquivo de saída
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for block in blocks:
                f.write(f"{block.index}\n")
                f.write(f"{block.start_time} --> {block.end_time}\n")
                f.write(f"{block.text}\n")
                f.write("\n")

        self.log(f"Arquivo SRT salvo: {output_path}", "OK")







