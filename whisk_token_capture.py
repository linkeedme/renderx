# -*- coding: utf-8 -*-
"""
Módulo de Captura Automática de Token Whisk
============================================
Usa Playwright para abrir o navegador, interceptar requisições de rede
e capturar automaticamente o token Bearer do Google Whisk.

Compatível com Windows, macOS e Linux.
"""

import asyncio
import platform
import sys
import os
from pathlib import Path
from typing import Optional, Tuple, Callable
from dataclasses import dataclass

# Caminho para armazenar dados do navegador (sessão persistente)
SCRIPT_DIR = Path(__file__).parent.absolute()
BROWSER_DATA_DIR = SCRIPT_DIR / ".playwright_data"

# URL do Whisk
WHISK_URL = "https://labs.google/fx/pt/tools/whisk/project"

# URL que será interceptada para capturar o token
TARGET_API_URL = "aisandbox-pa.googleapis.com/v1/whisk:getVideoCreditStatus"


@dataclass
class CaptureResult:
    """Resultado da captura de token."""
    success: bool
    token: Optional[str] = None
    email: Optional[str] = None
    error: Optional[str] = None


def check_playwright_installed() -> bool:
    """Verifica se o Playwright está instalado."""
    try:
        from playwright.sync_api import sync_playwright
        return True
    except ImportError:
        return False


def install_playwright_browsers(log_callback: Callable[[str, str], None] = None) -> bool:
    """
    Instala os navegadores do Playwright se necessário.
    
    Args:
        log_callback: Função de callback para logs
        
    Returns:
        True se instalação bem-sucedida
    """
    log = log_callback or (lambda msg, level: print(f"[{level}] {msg}"))
    
    try:
        import subprocess
        log("Instalando navegador Chromium do Playwright...", "INFO")
        
        # Usar o executável Python correto
        python_exe = sys.executable
        
        result = subprocess.run(
            [python_exe, "-m", "playwright", "install", "chromium"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutos timeout
        )
        
        if result.returncode == 0:
            log("Chromium instalado com sucesso!", "OK")
            return True
        else:
            log(f"Erro ao instalar Chromium: {result.stderr}", "ERROR")
            return False
            
    except subprocess.TimeoutExpired:
        log("Timeout ao instalar navegador", "ERROR")
        return False
    except Exception as e:
        log(f"Erro ao instalar navegador: {str(e)}", "ERROR")
        return False


async def _capture_token_async(
    log_callback: Callable[[str, str], None] = None,
    timeout_seconds: int = 180,
    headless: bool = False
) -> CaptureResult:
    """
    Versão assíncrona da captura de token.
    
    Args:
        log_callback: Função de callback para logs
        timeout_seconds: Timeout máximo em segundos
        headless: Se True, executa sem interface gráfica (não recomendado para login)
        
    Returns:
        CaptureResult com token e email ou erro
    """
    log = log_callback or (lambda msg, level: print(f"[{level}] {msg}"))
    
    captured_token: Optional[str] = None
    captured_email: Optional[str] = None
    
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        return CaptureResult(
            success=False,
            error="Playwright não está instalado. Execute: pip install playwright"
        )
    
    async with async_playwright() as p:
        try:
            # Criar diretório para dados do navegador
            BROWSER_DATA_DIR.mkdir(exist_ok=True)
            
            log("Iniciando navegador...", "INFO")
            
            # Usar contexto persistente para manter sessão do Google
            context = await p.chromium.launch_persistent_context(
                user_data_dir=str(BROWSER_DATA_DIR),
                headless=headless,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                ],
                viewport={"width": 1280, "height": 800},
                locale="pt-BR"
            )
            
            page = context.pages[0] if context.pages else await context.new_page()
            
            # Variável para armazenar o token capturado
            token_captured_event = asyncio.Event()
            
            async def handle_request(request):
                nonlocal captured_token
                
                # Verificar se é a requisição alvo
                if TARGET_API_URL in request.url:
                    auth_header = request.headers.get("authorization", "")
                    if auth_header.startswith("Bearer "):
                        captured_token = auth_header.replace("Bearer ", "")
                        log("Token Bearer capturado!", "OK")
                        token_captured_event.set()
            
            # Registrar interceptador de requisições
            page.on("request", handle_request)
            
            log(f"Navegando para {WHISK_URL}...", "INFO")
            log("Aguardando login do usuário (se necessário)...", "INFO")
            
            # Navegar para a página do Whisk
            await page.goto(WHISK_URL, wait_until="domcontentloaded", timeout=60000)
            
            # Aguardar até capturar o token ou timeout
            try:
                await asyncio.wait_for(
                    token_captured_event.wait(),
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                log("Timeout aguardando token. Usuário não fez login?", "WARN")
                await context.close()
                return CaptureResult(
                    success=False,
                    error="Timeout: não foi possível capturar o token. Certifique-se de fazer login."
                )
            
            # Token capturado, agora extrair email da página
            log("Extraindo email do usuário...", "INFO")
            
            # Aguardar um pouco para a página carregar completamente
            await asyncio.sleep(2)
            
            # Tentar extrair email de diferentes locais da página
            email_selectors = [
                # Avatar/perfil do Google
                '[data-email]',
                '[aria-label*="@"]',
                'img[alt*="@"]',
                # Elementos de texto que podem conter email
                'span:has-text("@gmail.com")',
                'span:has-text("@googlemail.com")',
                'div:has-text("@gmail.com")',
                # Menu de conta Google
                '[data-identifier]',
            ]
            
            for selector in email_selectors:
                try:
                    element = await page.query_selector(selector)
                    if element:
                        # Tentar diferentes atributos
                        email = await element.get_attribute("data-email")
                        if not email:
                            email = await element.get_attribute("data-identifier")
                        if not email:
                            email = await element.get_attribute("aria-label")
                        if not email:
                            email = await element.get_attribute("alt")
                        if not email:
                            text = await element.text_content()
                            if text and "@" in text:
                                # Extrair email do texto
                                import re
                                match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
                                if match:
                                    email = match.group(0)
                        
                        if email and "@" in email:
                            captured_email = email.strip()
                            log(f"Email encontrado: {captured_email}", "OK")
                            break
                except Exception:
                    continue
            
            # Se não encontrou email, tentar clicar no avatar para abrir menu
            if not captured_email:
                try:
                    # Procurar por avatar/botão de perfil
                    avatar_selectors = [
                        'img[src*="googleusercontent.com"]',
                        '[aria-label*="Conta"]',
                        '[aria-label*="Account"]',
                        'button:has(img[src*="googleusercontent"])',
                    ]
                    
                    for selector in avatar_selectors:
                        avatar = await page.query_selector(selector)
                        if avatar:
                            await avatar.click()
                            await asyncio.sleep(1)
                            
                            # Procurar email no menu aberto
                            menu_text = await page.content()
                            import re
                            match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', menu_text)
                            if match:
                                captured_email = match.group(0)
                                log(f"Email encontrado no menu: {captured_email}", "OK")
                                break
                except Exception as e:
                    log(f"Não foi possível extrair email do menu: {str(e)}", "WARN")
            
            # Fechar navegador
            log("Fechando navegador...", "INFO")
            await context.close()
            
            if captured_token:
                if not captured_email:
                    log("Email não encontrado automaticamente", "WARN")
                
                return CaptureResult(
                    success=True,
                    token=captured_token,
                    email=captured_email
                )
            else:
                return CaptureResult(
                    success=False,
                    error="Token não foi capturado"
                )
                
        except Exception as e:
            log(f"Erro durante captura: {str(e)}", "ERROR")
            return CaptureResult(
                success=False,
                error=str(e)
            )


def capture_whisk_token(
    log_callback: Callable[[str, str], None] = None,
    timeout_seconds: int = 180,
    headless: bool = False
) -> CaptureResult:
    """
    Captura automaticamente o token Bearer do Whisk.
    
    Abre um navegador, navega até a página do Whisk, aguarda o usuário
    fazer login (se necessário) e intercepta a requisição para capturar
    o token Bearer.
    
    Args:
        log_callback: Função de callback para logs (msg, level)
        timeout_seconds: Tempo máximo de espera em segundos (padrão: 180)
        headless: Se True, executa sem interface gráfica (não recomendado)
        
    Returns:
        CaptureResult com:
        - success: True se captura bem-sucedida
        - token: Token Bearer capturado (ou None)
        - email: Email do usuário (ou None se não encontrado)
        - error: Mensagem de erro (ou None se sucesso)
        
    Exemplo:
        result = capture_whisk_token()
        if result.success:
            print(f"Token: {result.token[:50]}...")
            print(f"Email: {result.email}")
        else:
            print(f"Erro: {result.error}")
    """
    log = log_callback or (lambda msg, level: print(f"[{level}] {msg}"))
    
    # Verificar se Playwright está instalado
    if not check_playwright_installed():
        log("Playwright não encontrado. Instalando...", "WARN")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "playwright"])
            log("Playwright instalado. Instalando navegadores...", "INFO")
            install_playwright_browsers(log)
        except Exception as e:
            return CaptureResult(
                success=False,
                error=f"Falha ao instalar Playwright: {str(e)}"
            )
    
    # Executar captura assíncrona
    try:
        # Em Windows, pode ser necessário usar um event loop específico
        if platform.system() == "Windows":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        # Criar novo event loop se necessário
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(
            _capture_token_async(log, timeout_seconds, headless)
        )
        
        return result
        
    except Exception as e:
        log(f"Erro ao executar captura: {str(e)}", "ERROR")
        return CaptureResult(
            success=False,
            error=str(e)
        )


def capture_and_save_token(
    log_callback: Callable[[str, str], None] = None,
    timeout_seconds: int = 180
) -> bool:
    """
    Captura token e salva automaticamente no arquivo whisk_keys.json.
    
    Se o email já existir, atualiza o token. Caso contrário, adiciona novo.
    
    Args:
        log_callback: Função de callback para logs
        timeout_seconds: Timeout em segundos
        
    Returns:
        True se captura e salvamento bem-sucedidos
    """
    log = log_callback or (lambda msg, level: print(f"[{level}] {msg}"))
    
    # Capturar token
    result = capture_whisk_token(log, timeout_seconds)
    
    if not result.success:
        log(f"Falha na captura: {result.error}", "ERROR")
        return False
    
    if not result.token:
        log("Token não capturado", "ERROR")
        return False
    
    # Importar função de salvamento
    try:
        from whisk_pool_manager import update_or_add_token_by_email
        
        # Salvar token
        action = update_or_add_token_by_email(
            key=result.token,
            email=result.email
        )
        
        if result.email:
            log(f"Token {action} para {result.email}", "OK")
        else:
            log(f"Token {action} (email desconhecido)", "OK")
        
        return True
        
    except ImportError:
        log("Erro: whisk_pool_manager não encontrado", "ERROR")
        return False
    except Exception as e:
        log(f"Erro ao salvar token: {str(e)}", "ERROR")
        return False


# Para testes diretos
if __name__ == "__main__":
    print("=== Teste de Captura de Token Whisk ===\n")
    
    def test_log(msg, level):
        print(f"[{level}] {msg}")
    
    result = capture_whisk_token(log_callback=test_log)
    
    print("\n=== Resultado ===")
    print(f"Sucesso: {result.success}")
    if result.success:
        print(f"Token: {result.token[:50]}..." if result.token else "Token: None")
        print(f"Email: {result.email}")
    else:
        print(f"Erro: {result.error}")

