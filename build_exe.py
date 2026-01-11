#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para gerar executável .exe do RenderX usando PyInstaller
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def build_exe():
    """Gera executável .exe do RenderX."""
    
    # Verificar se PyInstaller está instalado
    try:
        import PyInstaller
        print("✓ PyInstaller encontrado")
    except ImportError:
        print("✗ PyInstaller não encontrado. Instalando...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)
        print("✓ PyInstaller instalado")
    
    # Diretório do script
    script_dir = Path(__file__).parent.absolute()
    
    # Caminho do script principal
    main_script = script_dir / "iniciar_render.py"
    
    if not main_script.exists():
        print(f"✗ Erro: {main_script} não encontrado!")
        return False
    
    print(f"📦 Gerando executável do RenderX...")
    print(f"📄 Script principal: {main_script}")
    
    # Limpar builds anteriores
    build_dir = script_dir / "build"
    dist_dir = script_dir / "dist"
    spec_file = script_dir / "renderx.spec"
    
    if build_dir.exists():
        print("🗑️  Removendo build anterior...")
        shutil.rmtree(build_dir)
    
    if dist_dir.exists():
        print("🗑️  Removendo dist anterior...")
        shutil.rmtree(dist_dir)
    
    # Separador de dados para PyInstaller (depende do sistema)
    data_sep = ";" if sys.platform == "win32" else ":"
    
    # Comando PyInstaller base
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name=RenderX",
        "--onefile",  # Arquivo único
        "--windowed",  # Sem console (GUI) - use --console para ver erros
        "--clean",  # Limpar cache antes de buildar
        "--noconfirm",  # Sobrescrever sem perguntar
        "--hidden-import=customtkinter",
        "--hidden-import=cv2",
        "--hidden-import=numpy",
        "--hidden-import=PIL",
        "--hidden-import=PIL._tkinter_finder",
        "--hidden-import=assemblyai",
        "--hidden-import=httpx",
        "--hidden-import=fastapi",
        "--hidden-import=uvicorn",
        "--hidden-import=playwright",
        "--hidden-import=queue",
        "--hidden-import=threading",
        "--hidden-import=concurrent.futures",
        "--hidden-import=dataclasses",
        "--hidden-import=pathlib",
        "--collect-all=customtkinter",
        "--collect-all=cv2",
        "--collect-all=PIL",
    ]
    
    # Adicionar pasta EFEITOS se existir
    efeitos_path = script_dir / "EFEITOS"
    if efeitos_path.exists():
        cmd.extend(["--add-data", f"EFEITOS{data_sep}EFEITOS"])
    
    # Adicionar script principal
    cmd.append(str(main_script))
    
    try:
        print("\n🔨 Executando PyInstaller...")
        print("⏳ Isso pode levar alguns minutos...\n")
        
        result = subprocess.run(cmd, cwd=script_dir, check=True)
        
        # Verificar se o executável foi gerado
        exe_name = "RenderX.exe" if sys.platform == "win32" else "RenderX"
        exe_path = dist_dir / exe_name
        
        if exe_path.exists():
            print(f"\n✅ Executável gerado com sucesso!")
            print(f"📁 Localização: {exe_path}")
            print(f"📊 Tamanho: {exe_path.stat().st_size / (1024*1024):.2f} MB")
            
            # Copiar arquivos necessários para dist
            print("\n📋 Copiando arquivos de configuração...")
            
            # Criar estrutura de pastas necessárias
            config_files = [
                "keys_assembly.json",
                "whisk_keys.example.json",
                "subtitle_presets.json",
                "image_prompts.json",
                "vsl_keywords.json",
                "opencv_settings.json",
                "final_settings.example.json"
            ]
            
            for config_file in config_files:
                src = script_dir / config_file
                if src.exists():
                    shutil.copy2(src, dist_dir / config_file)
                    print(f"  ✓ {config_file}")
            
            print("\n✅ Build concluído!")
            print(f"\n💡 O executável está em: {dist_dir}")
            print("💡 Você pode distribuir toda a pasta 'dist' ou apenas o executável.")
            
            return True
        else:
            print(f"\n✗ Erro: Executável não foi gerado!")
            print(f"   Esperado em: {exe_path}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Erro ao executar PyInstaller: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Erro inesperado: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("  RenderX - Gerador de Executável")
    print("=" * 60)
    print()
    
    success = build_exe()
    
    sys.exit(0 if success else 1)
