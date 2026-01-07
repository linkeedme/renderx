# -*- coding: utf-8 -*-
"""
VSL Manager - Gerenciador de VSLs (Video Sales Letters)
========================================================
Lista e gerencia VSLs disponíveis para seleção por canal.
"""

import os
from typing import List, Optional, Tuple
from pathlib import Path

# Extensões de vídeo suportadas
VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.mkv', '.webm')

# Pasta padrão de VSLs
SCRIPT_DIR = Path(__file__).parent.absolute()
DEFAULT_VSL_FOLDER = SCRIPT_DIR / "EFEITOS" / "VSLs"


def get_vsl_folder(custom_folder: str = None) -> Path:
    """
    Retorna o caminho da pasta de VSLs.
    
    Args:
        custom_folder: Pasta customizada (opcional)
        
    Returns:
        Path da pasta de VSLs
    """
    if custom_folder and os.path.isdir(custom_folder):
        return Path(custom_folder)
    return DEFAULT_VSL_FOLDER


def list_available_vsls(folder: str = None) -> List[Tuple[str, str]]:
    """
    Lista todas as VSLs disponíveis na pasta.
    
    Args:
        folder: Pasta de VSLs (opcional, usa padrão)
        
    Returns:
        Lista de tuplas (nome_arquivo, caminho_completo)
    """
    vsl_folder = get_vsl_folder(folder)
    vsls = []
    
    if not vsl_folder.exists():
        return vsls
    
    for file in sorted(vsl_folder.iterdir()):
        if file.is_file() and file.suffix.lower() in VIDEO_EXTENSIONS:
            vsls.append((file.name, str(file)))
    
    return vsls


def get_vsl_names(folder: str = None) -> List[str]:
    """
    Retorna lista de nomes de VSLs para combobox.
    
    Args:
        folder: Pasta de VSLs (opcional)
        
    Returns:
        Lista de nomes de arquivos
    """
    vsls = list_available_vsls(folder)
    return [name for name, _ in vsls]


def get_vsl_path(name: str, folder: str = None) -> Optional[str]:
    """
    Retorna o caminho completo de uma VSL pelo nome.
    
    Args:
        name: Nome do arquivo da VSL
        folder: Pasta de VSLs (opcional)
        
    Returns:
        Caminho completo ou None
    """
    vsls = list_available_vsls(folder)
    
    for vsl_name, vsl_path in vsls:
        if vsl_name == name:
            return vsl_path
    
    return None


def get_vsl_display_name(filename: str) -> str:
    """
    Converte nome de arquivo para nome amigável.
    
    Ex: VSL_portugues.mp4 -> "Português"
        VSL_ingles.mp4 -> "Inglês"
        minha_vsl_canal1.mp4 -> "minha_vsl_canal1"
    
    Args:
        filename: Nome do arquivo
        
    Returns:
        Nome amigável para exibição
    """
    # Remover extensão
    name = os.path.splitext(filename)[0]
    
    # Se começa com VSL_, extrair o resto
    if name.lower().startswith("vsl_"):
        name = name[4:]
    
    # Capitalizar primeira letra
    if name:
        name = name[0].upper() + name[1:]
    
    return name


def get_vsls_with_display_names(folder: str = None) -> List[Tuple[str, str, str]]:
    """
    Retorna lista de VSLs com nomes de exibição.
    
    Args:
        folder: Pasta de VSLs (opcional)
        
    Returns:
        Lista de tuplas (nome_arquivo, caminho, nome_exibição)
    """
    vsls = list_available_vsls(folder)
    result = []
    
    for filename, path in vsls:
        display_name = get_vsl_display_name(filename)
        result.append((filename, path, display_name))
    
    return result


def get_vsl_summary(folder: str = None) -> str:
    """
    Retorna resumo das VSLs para exibição.
    
    Args:
        folder: Pasta de VSLs (opcional)
        
    Returns:
        String com resumo
    """
    vsls = list_available_vsls(folder)
    count = len(vsls)
    
    if count == 0:
        return "Nenhuma VSL encontrada"
    elif count == 1:
        return f"1 VSL disponível"
    else:
        return f"{count} VSLs disponíveis"



