# -*- coding: utf-8 -*-
"""
Motion Shuffler - Seletor Inteligente de Efeitos
=================================================
Gerencia a seleção aleatória de efeitos sem repetição em sequência.

Regras:
- Imagens ficam em ORDEM (seguem a história do SRT)
- Apenas EFEITOS são shuffled
- Efeito não pode repetir em sequência (verifica anterior)
"""

import random
from typing import List, Optional
from ken_burns_engine import KenBurnsEngine


class MotionShuffler:
    """Gerenciador de shuffle de efeitos sem repetição."""

    def __init__(self, log_callback=None):
        """
        Inicializa o shuffler.

        Args:
            log_callback: Função opcional para logging (message, level)
        """
        self.log = log_callback or (lambda msg, level="INFO": print(f"[{level}] {msg}"))
        self.all_effects = KenBurnsEngine.ANIMATION_TYPES.copy()
        self.effect_map = KenBurnsEngine.EFFECT_MAP.copy()
        self.last_effect = None
        self.used_effects_in_video = []

    def reset(self):
        """Reseta o estado para um novo vídeo."""
        self.last_effect = None
        self.used_effects_in_video = []

    def get_next_effect(self, exclude_last: bool = True) -> str:
        """
        Seleciona o próximo efeito sem repetir o anterior.

        Args:
            exclude_last: Se True, exclui o último efeito usado

        Returns:
            Nome do efeito selecionado
        """
        available = self.all_effects.copy()

        # Remover o último efeito usado (sem repetição em sequência)
        if exclude_last and self.last_effect and self.last_effect in available:
            available.remove(self.last_effect)

        # Se não houver efeitos disponíveis, usar todos
        if not available:
            available = self.all_effects.copy()

        # Selecionar aleatoriamente
        selected = random.choice(available)
        self.last_effect = selected
        self.used_effects_in_video.append(selected)

        return selected

    def get_effect_letter(self, effect: str) -> Optional[str]:
        """
        Retorna a letra correspondente ao efeito.

        Args:
            effect: Nome do efeito

        Returns:
            Letra (A-N) ou None
        """
        return KenBurnsEngine.get_letter_by_effect(effect)

    def generate_effects_for_images(self, num_images: int) -> List[str]:
        """
        Gera uma lista de efeitos para N imagens sem repetição consecutiva.

        Args:
            num_images: Número de imagens

        Returns:
            Lista de nomes de efeitos
        """
        self.reset()
        effects = []

        for i in range(num_images):
            effect = self.get_next_effect()
            effects.append(effect)
            self.log(f"Imagem {i+1:03d}: Efeito {self.get_effect_letter(effect)} ({effect})", "DEBUG")

        return effects

    def generate_effects_with_letters(self, num_images: int) -> List[dict]:
        """
        Gera lista de efeitos com metadados (letra, nome).

        Args:
            num_images: Número de imagens

        Returns:
            Lista de dicts com 'letter', 'effect', 'index'
        """
        self.reset()
        result = []

        for i in range(num_images):
            effect = self.get_next_effect()
            letter = self.get_effect_letter(effect)
            result.append({
                "index": i + 1,
                "letter": letter,
                "effect": effect
            })

        return result

    def get_effect_stats(self) -> dict:
        """
        Retorna estatísticas de uso dos efeitos no vídeo atual.

        Returns:
            Dict com contagem de cada efeito
        """
        stats = {}
        for effect in self.used_effects_in_video:
            stats[effect] = stats.get(effect, 0) + 1
        return stats

    def validate_no_consecutive_repeats(self, effects: List[str]) -> bool:
        """
        Valida se não há repetições consecutivas.

        Args:
            effects: Lista de efeitos

        Returns:
            True se válido (sem repetições consecutivas)
        """
        for i in range(1, len(effects)):
            if effects[i] == effects[i - 1]:
                return False
        return True


class ImageEffectAssigner:
    """Atribui efeitos a imagens mantendo ordem e shuffling de efeitos."""

    def __init__(self, log_callback=None):
        """
        Inicializa o atribuidor.

        Args:
            log_callback: Função opcional para logging
        """
        self.log = log_callback or (lambda msg, level="INFO": print(f"[{level}] {msg}"))
        self.shuffler = MotionShuffler(log_callback)

    def assign_effects_to_images(self, image_paths: List[str]) -> List[dict]:
        """
        Atribui efeitos a uma lista de imagens (em ordem).

        Args:
            image_paths: Lista de caminhos de imagens (já em ordem)

        Returns:
            Lista de dicts com 'image_path', 'effect', 'letter', 'order'
        """
        num_images = len(image_paths)
        effects = self.shuffler.generate_effects_for_images(num_images)

        result = []
        for i, (img_path, effect) in enumerate(zip(image_paths, effects)):
            letter = self.shuffler.get_effect_letter(effect)
            result.append({
                "order": i + 1,
                "image_path": img_path,
                "effect": effect,
                "letter": letter
            })
            self.log(f"{i+1:03d}.png {letter} -> {effect}", "OK")

        return result

    def get_assignment_summary(self, assignments: List[dict]) -> str:
        """
        Gera um resumo textual das atribuições.

        Formato: "1.png A  2.png C  3.png F  4.png B"

        Args:
            assignments: Lista de atribuições

        Returns:
            String formatada
        """
        parts = []
        for a in assignments:
            parts.append(f"{a['order']}.png {a['letter']}")
        return "  ".join(parts)



