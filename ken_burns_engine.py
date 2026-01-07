# -*- coding: utf-8 -*-
"""
Motor de Animações Ken Burns v2.0
==================================
Aplica efeitos de movimento variados (Zoom, Pan, Rotation, Punch-in) em imagens para vídeo.

14 Efeitos Disponíveis:
- A: pan_left_to_right
- B: pan_right_to_left
- C: pan_top_to_bottom
- D: pan_bottom_to_top
- E: pan_left_to_right_zoom_in
- F: pan_right_to_left_zoom_in
- G: pan_top_to_bottom_zoom_in
- H: pan_bottom_to_top_zoom_in
- I: pan_left_to_right_zoom_out
- J: pan_right_to_left_zoom_out
- K: pan_top_to_bottom_zoom_out
- L: pan_bottom_to_top_zoom_out
- M: rotation_light
- N: punch_in
"""

import cv2
import numpy as np
import random
from typing import List, Optional


class KenBurnsEngine:
    """Motor para aplicar animações Ken Burns em imagens."""

    # Mapeamento de letras para efeitos (A-N)
    EFFECT_MAP = {
        "A": "pan_left_to_right",
        "B": "pan_right_to_left",
        "C": "pan_top_to_bottom",
        "D": "pan_bottom_to_top",
        "E": "pan_left_to_right_zoom_in",
        "F": "pan_right_to_left_zoom_in",
        "G": "pan_top_to_bottom_zoom_in",
        "H": "pan_bottom_to_top_zoom_in",
        "I": "pan_left_to_right_zoom_out",
        "J": "pan_right_to_left_zoom_out",
        "K": "pan_top_to_bottom_zoom_out",
        "L": "pan_bottom_to_top_zoom_out",
        "M": "rotation_light",
        "N": "punch_in"
    }

    # Todos os tipos de animação disponíveis (14 efeitos)
    ANIMATION_TYPES = list(EFFECT_MAP.values())

    def __init__(self, log_callback=None):
        """
        Inicializa o motor de animações.

        Args:
            log_callback: Função opcional para logging (message, level)
        """
        self.log = log_callback or (lambda msg, level="INFO": print(f"[{level}] {msg}"))

    @staticmethod
    def get_effect_by_letter(letter: str) -> Optional[str]:
        """
        Retorna o efeito correspondente à letra (A-N).

        Args:
            letter: Letra do efeito (A-N)

        Returns:
            Nome do efeito ou None se inválido
        """
        return KenBurnsEngine.EFFECT_MAP.get(letter.upper())

    @staticmethod
    def get_letter_by_effect(effect: str) -> Optional[str]:
        """
        Retorna a letra correspondente ao efeito.

        Args:
            effect: Nome do efeito

        Returns:
            Letra (A-N) ou None se inválido
        """
        for letter, eff in KenBurnsEngine.EFFECT_MAP.items():
            if eff == effect:
                return letter
        return None

    @staticmethod
    def select_random_effect() -> str:
        """
        Seleciona um efeito aleatório da lista disponível.

        Returns:
            Nome do efeito selecionado
        """
        return random.choice(KenBurnsEngine.ANIMATION_TYPES)

    def smart_crop_16x9(self, img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        """
        Crop centralizado para 16:9, sem bordas pretas.

        Args:
            img: Imagem OpenCV (BGR)
            target_w: Largura alvo
            target_h: Altura alvo

        Returns:
            Imagem cortada e redimensionada
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

        # Redimensionar para resolução alvo com LANCZOS4
        return cv2.resize(img_cropped, (target_w, target_h),
                         interpolation=cv2.INTER_LANCZOS4)

    # =========================================================================
    # EFEITOS BÁSICOS: ZOOM
    # =========================================================================

    def apply_zoom_in(self, img: np.ndarray, width: int, height: int, 
                     duration: float, fps: int, zoom_scale: float = 1.15) -> List[np.ndarray]:
        """Aplica zoom de aproximação (zoom in)."""
        img_base = self.smart_crop_16x9(img, width, height)
        center = (width / 2.0, height / 2.0)
        # Aumentar zoom para durações longas
        effective_zoom = min(zoom_scale + (duration / 60.0) * 0.1, 1.35)
        total_frames = int(duration * fps)
        frames = []

        for frame_num in range(total_frames):
            t = frame_num / (total_frames - 1) if total_frames > 1 else 0
            eased_t = self._ease_in_out(t)
            scale = 1.0 + (effective_zoom - 1.0) * eased_t
            M = cv2.getRotationMatrix2D(center, 0, scale)
            frame = cv2.warpAffine(
                img_base, M, (width, height),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_REPLICATE
            )
            frames.append(frame)

        return frames

    def apply_zoom_out(self, img: np.ndarray, width: int, height: int,
                      duration: float, fps: int, zoom_scale: float = 1.15) -> List[np.ndarray]:
        """Aplica zoom de afastamento (zoom out)."""
        img_base = self.smart_crop_16x9(img, width, height)
        center = (width / 2.0, height / 2.0)
        effective_zoom = min(zoom_scale + (duration / 60.0) * 0.1, 1.35)
        total_frames = int(duration * fps)
        frames = []

        for frame_num in range(total_frames):
            t = frame_num / (total_frames - 1) if total_frames > 1 else 0
            eased_t = self._ease_in_out(t)
            scale = effective_zoom - (effective_zoom - 1.0) * eased_t
            M = cv2.getRotationMatrix2D(center, 0, scale)
            frame = cv2.warpAffine(
                img_base, M, (width, height),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_REPLICATE
            )
            frames.append(frame)

        return frames
    
    def _apply_zoom_out_old(self, img: np.ndarray, width: int, height: int,
                      duration: float, fps: int, zoom_scale: float = 1.15) -> List[np.ndarray]:
        """Aplica zoom de afastamento (zoom out)."""
        img_base = self.smart_crop_16x9(img, width, height)
        center = (width / 2.0, height / 2.0)
        total_frames = int(duration * fps)
        frames = []

        for frame_num in range(total_frames):
            t = frame_num / (total_frames - 1) if total_frames > 1 else 0
            scale = zoom_scale - (zoom_scale - 1.0) * t
            M = cv2.getRotationMatrix2D(center, 0, scale)
            frame = cv2.warpAffine(
                img_base, M, (width, height),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_REPLICATE
            )
            frames.append(frame)

        return frames

    # =========================================================================
    # EFEITOS BÁSICOS: PAN (SEM ZOOM)
    # =========================================================================

    def apply_pan_left_to_right(self, img: np.ndarray, width: int, height: int,
                                duration: float, fps: int, pan_amount: float = 0.2) -> List[np.ndarray]:
        """Aplica pan da esquerda para direita."""
        img_base = self.smart_crop_16x9(img, width, height)
        # Aumentar pan_amount para durações longas (mais movimento)
        effective_pan = min(pan_amount * max(1.0, duration / 8.0), 0.4)
        pan_pixels = int(width * effective_pan)
        img_large = cv2.resize(img_base, (width + pan_pixels * 2, height),
                              interpolation=cv2.INTER_LANCZOS4)
        total_frames = int(duration * fps)
        frames = []

        for frame_num in range(total_frames):
            t = frame_num / (total_frames - 1) if total_frames > 1 else 0
            # Usar easing para movimento mais natural
            eased_t = self._ease_in_out(t)
            x_offset = int(pan_pixels * (1 - eased_t))
            frame = img_large[:, x_offset:x_offset + width]
            frames.append(frame)

        return frames
    
    @staticmethod
    def _ease_in_out(t: float) -> float:
        """Função de easing suave (ease-in-out quadrático)."""
        if t < 0.5:
            return 2 * t * t
        return 1 - pow(-2 * t + 2, 2) / 2

    def apply_pan_right_to_left(self, img: np.ndarray, width: int, height: int,
                                duration: float, fps: int, pan_amount: float = 0.2) -> List[np.ndarray]:
        """Aplica pan da direita para esquerda."""
        img_base = self.smart_crop_16x9(img, width, height)
        effective_pan = min(pan_amount * max(1.0, duration / 8.0), 0.4)
        pan_pixels = int(width * effective_pan)
        img_large = cv2.resize(img_base, (width + pan_pixels * 2, height),
                              interpolation=cv2.INTER_LANCZOS4)
        total_frames = int(duration * fps)
        frames = []

        for frame_num in range(total_frames):
            t = frame_num / (total_frames - 1) if total_frames > 1 else 0
            eased_t = self._ease_in_out(t)
            x_offset = int(pan_pixels * eased_t)
            frame = img_large[:, x_offset:x_offset + width]
            frames.append(frame)

        return frames

    def apply_pan_top_to_bottom(self, img: np.ndarray, width: int, height: int,
                                duration: float, fps: int, pan_amount: float = 0.2) -> List[np.ndarray]:
        """Aplica pan de cima para baixo."""
        img_base = self.smart_crop_16x9(img, width, height)
        effective_pan = min(pan_amount * max(1.0, duration / 8.0), 0.4)
        pan_pixels = int(height * effective_pan)
        img_large = cv2.resize(img_base, (width, height + pan_pixels * 2),
                              interpolation=cv2.INTER_LANCZOS4)
        total_frames = int(duration * fps)
        frames = []

        for frame_num in range(total_frames):
            t = frame_num / (total_frames - 1) if total_frames > 1 else 0
            eased_t = self._ease_in_out(t)
            y_offset = int(pan_pixels * (1 - eased_t))
            frame = img_large[y_offset:y_offset + height, :]
            frames.append(frame)

        return frames

    def apply_pan_bottom_to_top(self, img: np.ndarray, width: int, height: int,
                                duration: float, fps: int, pan_amount: float = 0.2) -> List[np.ndarray]:
        """Aplica pan de baixo para cima."""
        img_base = self.smart_crop_16x9(img, width, height)
        effective_pan = min(pan_amount * max(1.0, duration / 8.0), 0.4)
        pan_pixels = int(height * effective_pan)
        img_large = cv2.resize(img_base, (width, height + pan_pixels * 2),
                              interpolation=cv2.INTER_LANCZOS4)
        total_frames = int(duration * fps)
        frames = []

        for frame_num in range(total_frames):
            t = frame_num / (total_frames - 1) if total_frames > 1 else 0
            eased_t = self._ease_in_out(t)
            y_offset = int(pan_pixels * eased_t)
            frame = img_large[y_offset:y_offset + height, :]
            frames.append(frame)

        return frames

    # =========================================================================
    # EFEITOS COMBINADOS: PAN + ZOOM IN
    # =========================================================================

    def apply_pan_left_to_right_zoom_in(self, img: np.ndarray, width: int, height: int,
                                        duration: float, fps: int, pan_amount: float = 0.15,
                                        zoom_scale: float = 1.10) -> List[np.ndarray]:
        """Aplica pan esquerda->direita com zoom in simultâneo."""
        img_base = self.smart_crop_16x9(img, width, height)
        center = (width / 2.0, height / 2.0)
        pan_pixels = int(width * pan_amount)
        total_frames = int(duration * fps)
        frames = []

        for frame_num in range(total_frames):
            t = frame_num / (total_frames - 1) if total_frames > 1 else 0
            
            # Zoom progressivo
            scale = 1.0 + (zoom_scale - 1.0) * t
            M = cv2.getRotationMatrix2D(center, 0, scale)
            zoomed = cv2.warpAffine(img_base, M, (width, height),
                                   flags=cv2.INTER_LANCZOS4,
                                   borderMode=cv2.BORDER_REPLICATE)
            
            # Pan horizontal
            tx = int(pan_pixels * (t - 0.5) * 2)  # -pan_pixels a +pan_pixels
            M_pan = np.float32([[1, 0, tx], [0, 1, 0]])
            frame = cv2.warpAffine(zoomed, M_pan, (width, height),
                                  borderMode=cv2.BORDER_REPLICATE)
            frames.append(frame)

        return frames

    def apply_pan_right_to_left_zoom_in(self, img: np.ndarray, width: int, height: int,
                                        duration: float, fps: int, pan_amount: float = 0.15,
                                        zoom_scale: float = 1.10) -> List[np.ndarray]:
        """Aplica pan direita->esquerda com zoom in simultâneo."""
        img_base = self.smart_crop_16x9(img, width, height)
        center = (width / 2.0, height / 2.0)
        pan_pixels = int(width * pan_amount)
        total_frames = int(duration * fps)
        frames = []

        for frame_num in range(total_frames):
            t = frame_num / (total_frames - 1) if total_frames > 1 else 0
            
            scale = 1.0 + (zoom_scale - 1.0) * t
            M = cv2.getRotationMatrix2D(center, 0, scale)
            zoomed = cv2.warpAffine(img_base, M, (width, height),
                                   flags=cv2.INTER_LANCZOS4,
                                   borderMode=cv2.BORDER_REPLICATE)
            
            tx = int(pan_pixels * (0.5 - t) * 2)  # +pan_pixels a -pan_pixels
            M_pan = np.float32([[1, 0, tx], [0, 1, 0]])
            frame = cv2.warpAffine(zoomed, M_pan, (width, height),
                                  borderMode=cv2.BORDER_REPLICATE)
            frames.append(frame)

        return frames

    def apply_pan_top_to_bottom_zoom_in(self, img: np.ndarray, width: int, height: int,
                                        duration: float, fps: int, pan_amount: float = 0.15,
                                        zoom_scale: float = 1.10) -> List[np.ndarray]:
        """Aplica pan cima->baixo com zoom in simultâneo."""
        img_base = self.smart_crop_16x9(img, width, height)
        center = (width / 2.0, height / 2.0)
        pan_pixels = int(height * pan_amount)
        total_frames = int(duration * fps)
        frames = []

        for frame_num in range(total_frames):
            t = frame_num / (total_frames - 1) if total_frames > 1 else 0
            
            scale = 1.0 + (zoom_scale - 1.0) * t
            M = cv2.getRotationMatrix2D(center, 0, scale)
            zoomed = cv2.warpAffine(img_base, M, (width, height),
                                   flags=cv2.INTER_LANCZOS4,
                                   borderMode=cv2.BORDER_REPLICATE)
            
            ty = int(pan_pixels * (t - 0.5) * 2)
            M_pan = np.float32([[1, 0, 0], [0, 1, ty]])
            frame = cv2.warpAffine(zoomed, M_pan, (width, height),
                                  borderMode=cv2.BORDER_REPLICATE)
            frames.append(frame)

        return frames

    def apply_pan_bottom_to_top_zoom_in(self, img: np.ndarray, width: int, height: int,
                                        duration: float, fps: int, pan_amount: float = 0.15,
                                        zoom_scale: float = 1.10) -> List[np.ndarray]:
        """Aplica pan baixo->cima com zoom in simultâneo."""
        img_base = self.smart_crop_16x9(img, width, height)
        center = (width / 2.0, height / 2.0)
        pan_pixels = int(height * pan_amount)
        total_frames = int(duration * fps)
        frames = []

        for frame_num in range(total_frames):
            t = frame_num / (total_frames - 1) if total_frames > 1 else 0
            
            scale = 1.0 + (zoom_scale - 1.0) * t
            M = cv2.getRotationMatrix2D(center, 0, scale)
            zoomed = cv2.warpAffine(img_base, M, (width, height),
                                   flags=cv2.INTER_LANCZOS4,
                                   borderMode=cv2.BORDER_REPLICATE)
            
            ty = int(pan_pixels * (0.5 - t) * 2)
            M_pan = np.float32([[1, 0, 0], [0, 1, ty]])
            frame = cv2.warpAffine(zoomed, M_pan, (width, height),
                                  borderMode=cv2.BORDER_REPLICATE)
            frames.append(frame)

        return frames

    # =========================================================================
    # EFEITOS COMBINADOS: PAN + ZOOM OUT
    # =========================================================================

    def apply_pan_left_to_right_zoom_out(self, img: np.ndarray, width: int, height: int,
                                         duration: float, fps: int, pan_amount: float = 0.15,
                                         zoom_scale: float = 1.10) -> List[np.ndarray]:
        """Aplica pan esquerda->direita com zoom out simultâneo."""
        img_base = self.smart_crop_16x9(img, width, height)
        center = (width / 2.0, height / 2.0)
        pan_pixels = int(width * pan_amount)
        total_frames = int(duration * fps)
        frames = []

        for frame_num in range(total_frames):
            t = frame_num / (total_frames - 1) if total_frames > 1 else 0
            
            scale = zoom_scale - (zoom_scale - 1.0) * t
            M = cv2.getRotationMatrix2D(center, 0, scale)
            zoomed = cv2.warpAffine(img_base, M, (width, height),
                                   flags=cv2.INTER_LANCZOS4,
                                   borderMode=cv2.BORDER_REPLICATE)
            
            tx = int(pan_pixels * (t - 0.5) * 2)
            M_pan = np.float32([[1, 0, tx], [0, 1, 0]])
            frame = cv2.warpAffine(zoomed, M_pan, (width, height),
                                  borderMode=cv2.BORDER_REPLICATE)
            frames.append(frame)

        return frames

    def apply_pan_right_to_left_zoom_out(self, img: np.ndarray, width: int, height: int,
                                         duration: float, fps: int, pan_amount: float = 0.15,
                                         zoom_scale: float = 1.10) -> List[np.ndarray]:
        """Aplica pan direita->esquerda com zoom out simultâneo."""
        img_base = self.smart_crop_16x9(img, width, height)
        center = (width / 2.0, height / 2.0)
        pan_pixels = int(width * pan_amount)
        total_frames = int(duration * fps)
        frames = []

        for frame_num in range(total_frames):
            t = frame_num / (total_frames - 1) if total_frames > 1 else 0
            
            scale = zoom_scale - (zoom_scale - 1.0) * t
            M = cv2.getRotationMatrix2D(center, 0, scale)
            zoomed = cv2.warpAffine(img_base, M, (width, height),
                                   flags=cv2.INTER_LANCZOS4,
                                   borderMode=cv2.BORDER_REPLICATE)
            
            tx = int(pan_pixels * (0.5 - t) * 2)
            M_pan = np.float32([[1, 0, tx], [0, 1, 0]])
            frame = cv2.warpAffine(zoomed, M_pan, (width, height),
                                  borderMode=cv2.BORDER_REPLICATE)
            frames.append(frame)

        return frames

    def apply_pan_top_to_bottom_zoom_out(self, img: np.ndarray, width: int, height: int,
                                         duration: float, fps: int, pan_amount: float = 0.15,
                                         zoom_scale: float = 1.10) -> List[np.ndarray]:
        """Aplica pan cima->baixo com zoom out simultâneo."""
        img_base = self.smart_crop_16x9(img, width, height)
        center = (width / 2.0, height / 2.0)
        pan_pixels = int(height * pan_amount)
        total_frames = int(duration * fps)
        frames = []

        for frame_num in range(total_frames):
            t = frame_num / (total_frames - 1) if total_frames > 1 else 0
            
            scale = zoom_scale - (zoom_scale - 1.0) * t
            M = cv2.getRotationMatrix2D(center, 0, scale)
            zoomed = cv2.warpAffine(img_base, M, (width, height),
                                   flags=cv2.INTER_LANCZOS4,
                                   borderMode=cv2.BORDER_REPLICATE)
            
            ty = int(pan_pixels * (t - 0.5) * 2)
            M_pan = np.float32([[1, 0, 0], [0, 1, ty]])
            frame = cv2.warpAffine(zoomed, M_pan, (width, height),
                                  borderMode=cv2.BORDER_REPLICATE)
            frames.append(frame)

        return frames

    def apply_pan_bottom_to_top_zoom_out(self, img: np.ndarray, width: int, height: int,
                                         duration: float, fps: int, pan_amount: float = 0.15,
                                         zoom_scale: float = 1.10) -> List[np.ndarray]:
        """Aplica pan baixo->cima com zoom out simultâneo."""
        img_base = self.smart_crop_16x9(img, width, height)
        center = (width / 2.0, height / 2.0)
        pan_pixels = int(height * pan_amount)
        total_frames = int(duration * fps)
        frames = []

        for frame_num in range(total_frames):
            t = frame_num / (total_frames - 1) if total_frames > 1 else 0
            
            scale = zoom_scale - (zoom_scale - 1.0) * t
            M = cv2.getRotationMatrix2D(center, 0, scale)
            zoomed = cv2.warpAffine(img_base, M, (width, height),
                                   flags=cv2.INTER_LANCZOS4,
                                   borderMode=cv2.BORDER_REPLICATE)
            
            ty = int(pan_pixels * (0.5 - t) * 2)
            M_pan = np.float32([[1, 0, 0], [0, 1, ty]])
            frame = cv2.warpAffine(zoomed, M_pan, (width, height),
                                  borderMode=cv2.BORDER_REPLICATE)
            frames.append(frame)

        return frames

    # =========================================================================
    # EFEITOS ESPECIAIS: ROTATION E PUNCH-IN
    # =========================================================================

    def apply_rotation_light(self, img: np.ndarray, width: int, height: int,
                             duration: float, fps: int, max_angle: float = 3.0) -> List[np.ndarray]:
        """
        Aplica rotação leve (2-5 graus) suave.
        
        Args:
            max_angle: Ângulo máximo de rotação em graus (padrão: 3.0)
        """
        img_base = self.smart_crop_16x9(img, width, height)
        center = (width / 2.0, height / 2.0)
        total_frames = int(duration * fps)
        frames = []

        for frame_num in range(total_frames):
            t = frame_num / (total_frames - 1) if total_frames > 1 else 0
            # Rotação suave: 0 -> max_angle -> 0 (ida e volta)
            # Usando seno para suavidade
            angle = max_angle * np.sin(t * np.pi)
            
            # Escala ligeiramente maior para evitar bordas pretas
            scale = 1.05
            M = cv2.getRotationMatrix2D(center, angle, scale)
            frame = cv2.warpAffine(
                img_base, M, (width, height),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_REPLICATE
            )
            frames.append(frame)

        return frames

    def apply_punch_in(self, img: np.ndarray, width: int, height: int,
                       duration: float, fps: int, punch_scale: float = 1.25) -> List[np.ndarray]:
        """
        Aplica punch-in (zoom rápido central com desaceleração).
        
        Efeito: Zoom rápido no início, desacelera no final.
        
        Args:
            punch_scale: Escala máxima do punch (padrão: 1.25)
        """
        img_base = self.smart_crop_16x9(img, width, height)
        center = (width / 2.0, height / 2.0)
        total_frames = int(duration * fps)
        frames = []

        for frame_num in range(total_frames):
            t = frame_num / (total_frames - 1) if total_frames > 1 else 0
            # Curva de ease-out (rápido no início, lento no final)
            # Usando função quadrática invertida
            eased_t = 1 - (1 - t) ** 2
            
            scale = 1.0 + (punch_scale - 1.0) * eased_t
            M = cv2.getRotationMatrix2D(center, 0, scale)
            frame = cv2.warpAffine(
                img_base, M, (width, height),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_REPLICATE
            )
            frames.append(frame)

        return frames

    # =========================================================================
    # RENDERIZAÇÃO PRINCIPAL
    # =========================================================================

    def render_animation(self, image_path: str, output_path: str, animation_type: str,
                        width: int, height: int, duration: float, fps: int = 24,
                        zoom_scale: float = 1.15, pan_amount: float = 0.2) -> Optional[str]:
        """
        Renderiza animação completa em arquivo de vídeo.

        Args:
            image_path: Caminho da imagem de entrada
            output_path: Caminho do vídeo de saída
            animation_type: Tipo de animação (ver ANIMATION_TYPES)
            width: Largura do vídeo
            height: Altura do vídeo
            duration: Duração em segundos
            fps: Frames por segundo
            zoom_scale: Escala do zoom
            pan_amount: Quantidade de pan

        Returns:
            Caminho do vídeo gerado ou None em caso de erro
        """
        try:
            # Carregar imagem
            img = cv2.imread(image_path)
            if img is None:
                self.log(f"Erro ao carregar imagem: {image_path}", "ERROR")
                return None

            # Mapear tipo de animação para método
            animation_methods = {
                # Pan básico
                "pan_left_to_right": lambda: self.apply_pan_left_to_right(img, width, height, duration, fps, pan_amount),
                "pan_right_to_left": lambda: self.apply_pan_right_to_left(img, width, height, duration, fps, pan_amount),
                "pan_top_to_bottom": lambda: self.apply_pan_top_to_bottom(img, width, height, duration, fps, pan_amount),
                "pan_bottom_to_top": lambda: self.apply_pan_bottom_to_top(img, width, height, duration, fps, pan_amount),
                # Zoom básico
                "zoom_in": lambda: self.apply_zoom_in(img, width, height, duration, fps, zoom_scale),
                "zoom_out": lambda: self.apply_zoom_out(img, width, height, duration, fps, zoom_scale),
                # Pan + Zoom In
                "pan_left_to_right_zoom_in": lambda: self.apply_pan_left_to_right_zoom_in(img, width, height, duration, fps),
                "pan_right_to_left_zoom_in": lambda: self.apply_pan_right_to_left_zoom_in(img, width, height, duration, fps),
                "pan_top_to_bottom_zoom_in": lambda: self.apply_pan_top_to_bottom_zoom_in(img, width, height, duration, fps),
                "pan_bottom_to_top_zoom_in": lambda: self.apply_pan_bottom_to_top_zoom_in(img, width, height, duration, fps),
                # Pan + Zoom Out
                "pan_left_to_right_zoom_out": lambda: self.apply_pan_left_to_right_zoom_out(img, width, height, duration, fps),
                "pan_right_to_left_zoom_out": lambda: self.apply_pan_right_to_left_zoom_out(img, width, height, duration, fps),
                "pan_top_to_bottom_zoom_out": lambda: self.apply_pan_top_to_bottom_zoom_out(img, width, height, duration, fps),
                "pan_bottom_to_top_zoom_out": lambda: self.apply_pan_bottom_to_top_zoom_out(img, width, height, duration, fps),
                # Especiais
                "rotation_light": lambda: self.apply_rotation_light(img, width, height, duration, fps),
                "punch_in": lambda: self.apply_punch_in(img, width, height, duration, fps),
            }

            if animation_type not in animation_methods:
                self.log(f"Tipo de animação desconhecido: {animation_type}", "ERROR")
                return None

            frames = animation_methods[animation_type]()

            # Criar VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            if not out.isOpened():
                self.log(f"Erro ao criar VideoWriter: {output_path}", "ERROR")
                return None

            # Escrever frames
            for frame in frames:
                out.write(frame)

            out.release()
            return output_path

        except Exception as e:
            self.log(f"Erro ao renderizar animação: {str(e)}", "ERROR")
            return None
