# --- src/scraper_tab.py ---

import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
import threading
import queue # Para comunicação entre thread e GUI
import sys
import os
import pandas as pd
from datetime import date, timedelta, datetime # Adicionado datetime
from typing import Optional
import logging


# --- Configurar Path ---
# Assume que scraper_tab.py está DENTRO de src
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR) # Diretório pai (futebol_analytics)
if CURRENT_DIR not in sys.path: sys.path.insert(0, CURRENT_DIR)
if BASE_DIR not in sys.path: sys.path.insert(0, BASE_DIR)

# --- Importar Módulos do Projeto ---
try:
    # Importa direto, pois BASE_DIR e CURRENT_DIR (src) estão no path
    # Certifique-se que scraper_predictor.py existe e pode ser importado
    try:
        from scraper_predictor import scrape_upcoming_fixtures
    except ImportError:
        print("Erro: Falha ao importar 'scrape_upcoming_fixtures' de 'scraper_predictor'. Verifique o arquivo.")
        raise # Re-levanta o erro

    # Certifique-se que github_manager.py existe e pode ser importado
    try:
        from github_manager import GitHubManager
    except ImportError:
         print("Erro: Falha ao importar 'GitHubManager' de 'github_manager'. Verifique o arquivo.")
         raise # Re-levanta o erro

    # Certifique-se que config.py existe e pode ser importado
    try:
        from config import (
            SCRAPER_TARGET_DAY, GITHUB_REPO_NAME, CHROMEDRIVER_PATH
            # GITHUB_PREDICTIONS_PATH não é usado diretamente aqui para salvar o RAW
        )
    except ImportError:
         print("Erro: Falha ao importar variáveis de 'config'. Verifique o arquivo.")
         raise # Re-levanta o erro

    # Certifique-se que logger_config.py existe e pode ser importado
    try:
        from logger_config import setup_logger
    except ImportError:
         print("Erro: Falha ao importar 'setup_logger' de 'logger_config'. Verifique o arquivo.")
         raise # Re-levanta o erro

    from dotenv import load_dotenv # dotenv é opcional, mas bom ter

except ImportError as e:
    print(f"Erro CRÍTICO ao importar módulos em scraper_tab.py: {e}")
    # Tenta mostrar erro na GUI se possível (difícil neste ponto)
    try:
        root_err = tk.Tk(); root_err.withdraw()
        messagebox.showerror("Erro de Importação (Scraper Tab)", f"Não foi possível importar módulos essenciais:\n{e}\n\nVerifique o console/logs.")
        root_err.destroy()
    except Exception: pass
    sys.exit(1) # Importante sair se módulos cruciais faltam

logger = setup_logger("ScraperUploadTab")

class ScraperUploadTab:
    def __init__(self, parent_frame, main_root):
        self.parent = parent_frame
        self.main_tk_root = main_root # Necessário para o .after()
        self.gui_queue = queue.Queue()
        self.stop_processing_queue = False
        self.scraper_thread : Optional[threading.Thread] = None
        self.upload_thread : Optional[threading.Thread] = None
        self.scraped_data: Optional[pd.DataFrame] = None # Para armazenar os dados raspados
        self.github_manager: Optional[GitHubManager] = None

        self.create_widgets()
        # Garante que a pasta de logs existe ao iniciar
        log_dir = os.path.join(BASE_DIR, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.process_gui_queue() # Inicia o processador da fila

    def log_to_widget(self, message: str):
        """ Envia mensagem de log para a área de texto na GUI. """
        self.gui_queue.put(("log", message))

    def _update_log_widget(self, message):
        """ Atualiza o widget de log (executado pelo loop da GUI). """
        try:
            if hasattr(self, 'log_area') and self.log_area.winfo_exists():
                self.log_area.config(state='normal')
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") # Usa datetime importado
                self.log_area.insert(tk.END, f"[{timestamp}] {message}\n")
                self.log_area.see(tk.END)
                self.log_area.config(state='disabled')
        except tk.TclError: pass # Ignora erro se widget não existir mais
        except Exception as e: logger.error(f"Erro interno ao atualizar log widget: {e}")

    def set_status(self, message: str):
        """ Envia atualização de status para a GUI. """
        self.gui_queue.put(("status", message))

    def _update_status_label(self, message):
        """ Atualiza a label de status (executado pelo loop da GUI). """
        try:
            if hasattr(self, 'status_label') and self.status_label.winfo_exists():
                self.status_label.config(text=message)
        except tk.TclError: pass
        except Exception as e: logger.error(f"Erro interno ao atualizar status label: {e}")

    def set_button_state(self, button_name: str, state: str):
         """ Envia comando para mudar estado de botão para a GUI. """
         self.gui_queue.put(("button_state", (button_name, state)))

    def _update_button_state(self, button_info):
         """ Muda o estado do botão (executado pelo loop da GUI). """
         button_name, state = button_info
         try:
            button = getattr(self, button_name, None)
            if button and isinstance(button, ttk.Button) and button.winfo_exists():
                 button.config(state=state)
         except tk.TclError: pass
         except Exception as e: logger.error(f"Erro interno ao atualizar estado do botão '{button_name}': {e}")


    def create_widgets(self):
        """Cria os widgets para a aba de Scraper & Upload."""
        main_frame = ttk.Frame(self.parent, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Painel de Controle ---
        control_frame = ttk.LabelFrame(main_frame, text=" Ações ", padding="10")
        control_frame.pack(fill=tk.X, pady=5)

        # Botão Scraper
        self.run_scraper_button = ttk.Button(
            control_frame,
            text=f"Executar Coleta ({SCRAPER_TARGET_DAY.capitalize()})",
            command=self.start_scraping_thread,
            width=30
        )
        self.run_scraper_button.pack(side=tk.LEFT, padx=(0, 10), pady=5)

        # Botão Upload
        self.upload_button = ttk.Button(
            control_frame,
            text="Fazer Upload para GitHub",
            command=self.start_upload_thread,
            state=tk.DISABLED, # Começa desabilitado
            width=30
        )
        self.upload_button.pack(side=tk.LEFT, padx=(0, 10), pady=5)

        # Label de Status
        self.status_label = ttk.Label(control_frame, text="Pronto.", width=40, anchor="w")
        self.status_label.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.X, expand=True)


        # --- Painel de Logs ---
        log_frame = ttk.LabelFrame(main_frame, text=" Logs da Coleta / Upload ", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        self.log_area = ScrolledText(
            log_frame, height=20, state='disabled', wrap=tk.WORD, font=("Consolas", 9)
        )
        self.log_area.pack(fill=tk.BOTH, expand=True)

        self.log_to_widget("Aba de Coleta e Upload inicializada.")
        self.log_to_widget(f"Alvo da coleta: Jogos de '{SCRAPER_TARGET_DAY}'.")
        self.log_to_widget(f"Repositório GitHub Alvo: '{GITHUB_REPO_NAME}' (verifique .env ou ambiente).")


    def start_scraping_thread(self):
        """Inicia a tarefa de scraping em uma nova thread."""
        # Verifica se já existe uma thread rodando
        if self.scraper_thread and self.scraper_thread.is_alive():
            self.log_to_widget("Coleta já está em andamento.")
            return

        self.log_to_widget("Iniciando processo de coleta...")
        self.set_status(f"Coletando jogos de '{SCRAPER_TARGET_DAY}'...")
        self.set_button_state("run_scraper_button", tk.DISABLED)
        self.set_button_state("upload_button", tk.DISABLED)
        self.scraped_data = None # Limpa dados anteriores

        # Cria e inicia a thread
        self.scraper_thread = threading.Thread(target=self._run_scrape_task, daemon=True)
        self.scraper_thread.start()

    def _run_scrape_task(self):
        """Executa a lógica de scraping (roda na thread)."""
        try:
            logger.info("Thread de scraping iniciada.")
            self.log_to_widget("Thread de scraping iniciada...") # Log para GUI

            # --- Lógica do scraper ---
            df_fixtures = scrape_upcoming_fixtures(
                headless=True, # Executar sem janela para automação
                chromedriver_path=CHROMEDRIVER_PATH # Passa o caminho do config
            )
            # --- Fim da lógica do scraper ---

            if df_fixtures is None:
                self.log_to_widget("ERRO: Scraper falhou ou não retornou dados.")
                self.set_status("Erro na coleta.")
                logger.error("Scraping falhou, retornou None.")
            elif df_fixtures.empty:
                self.log_to_widget("AVISO: Scraper rodou, mas não encontrou jogos para as ligas alvo.")
                self.set_status("Nenhum jogo encontrado.")
                logger.warning("Scraping retornou DataFrame vazio.")
                self.scraped_data = df_fixtures # Guarda o DF vazio mesmo assim
            else:
                n_jogos = len(df_fixtures)
                self.log_to_widget(f"Sucesso: {n_jogos} jogos coletados.")
                self.set_status(f"{n_jogos} jogos coletados com sucesso.")
                logger.info(f"{n_jogos} jogos coletados.")
                # Mostra amostra no log da GUI
                with pd.option_context('display.max_rows', 5, 'display.max_columns', 10, 'display.width', 120):
                     log_sample = f"Amostra:\n{df_fixtures.head().to_string()}"
                     self.log_to_widget(log_sample)
                self.scraped_data = df_fixtures # Armazena os dados

                # Habilita o botão de upload se dados foram coletados
                self.set_button_state("upload_button", tk.NORMAL)

        except Exception as e:
            errmsg = f"Erro inesperado durante a coleta: {e}"
            self.log_to_widget(f"ERRO: {errmsg}")
            self.set_status("Erro crítico na coleta.")
            logger.error(errmsg, exc_info=True)
        finally:
            # Reabilita o botão de raspar, independentemente do resultado
            self.set_button_state("run_scraper_button", tk.NORMAL)
            logger.info("Thread de scraping finalizada.")

    def start_upload_thread(self):
        """Inicia a tarefa de upload em uma nova thread."""
        if self.scraped_data is None:
            messagebox.showwarning("Dados Ausentes", "Execute a coleta primeiro antes de fazer o upload.", parent=self.parent)
            return
        if self.scraped_data.empty:
             messagebox.showinfo("Dados Vazios", "Não há jogos coletados para fazer upload.", parent=self.parent)
             return
        # Verifica se já existe uma thread rodando
        if self.upload_thread and self.upload_thread.is_alive():
             self.log_to_widget("Upload já está em andamento.")
             return

        self.log_to_widget("Iniciando processo de upload para GitHub...")
        self.set_status("Fazendo upload...")
        self.set_button_state("run_scraper_button", tk.DISABLED)
        self.set_button_state("upload_button", tk.DISABLED)

        # Cria e inicia a thread
        self.upload_thread = threading.Thread(target=self._run_upload_task, daemon=True)
        self.upload_thread.start()

    def _run_upload_task(self):
        """Executa a lógica de upload para o GitHub (roda na thread)."""
        try:
            logger.info("Thread de upload iniciada.")
            self.log_to_widget("Thread de upload iniciada...")

            # 1. Carregar .env (necessário para GitHubManager dentro da thread)
            self.log_to_widget("Carregando config do GitHub...")
            logger.debug("Tentando carregar .env para GitHubManager na thread.")
            # Usa BASE_DIR (diretório do projeto futebol_analytics)
            dotenv_path = os.path.join(BASE_DIR, '.env')
            if os.path.exists(dotenv_path):
                load_dotenv(dotenv_path=dotenv_path, override=True)
                logger.debug(".env carregado na thread.")
            else:
                 logger.debug(".env não encontrado na thread, usando env vars.")

            # 2. Inicializar GitHub Manager
            try:
                # Verifica se já tem uma instância reutilizável
                if self.github_manager is None:
                    self.log_to_widget("Inicializando conexão com GitHub...")
                    self.github_manager = GitHubManager() # Pode levantar erro se token/repo faltar
                else:
                     self.log_to_widget("Reutilizando conexão com GitHub...")
            except Exception as e_gh_init:
                self.log_to_widget(f"ERRO: Falha ao conectar ao GitHub: {e_gh_init}")
                logger.error(f"Falha ao inicializar GitHub Manager na thread: {e_gh_init}", exc_info=True)
                self.set_status("Erro na conexão com GitHub.")
                # Reabilita botões no finally
                return

            # 3. Definir Nome/Caminho do Arquivo
            target_day = SCRAPER_TARGET_DAY
            if target_day == "today": file_date_str = date.today().strftime("%Y-%m-%d")
            elif target_day == "tomorrow": file_date_str = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
            else: file_date_str = target_day # Assume formato YYYY-MM-DD

            csv_filename = f"scraped_fixtures_{file_date_str}.csv"
            # Salva em um subdiretório para dados brutos
            github_path = f"data/raw_scraped/{csv_filename}"
            self.log_to_widget(f"Preparando para salvar em: {GITHUB_REPO_NAME}/{github_path}")

            # 4. Fazer Upload
            commit_message = f"Atualiza/Cria jogos raspados para {file_date_str}"
            success = self.github_manager.save_or_update_github(
                df=self.scraped_data,
                path=github_path,
                commit_message_prefix=commit_message # Passa como prefixo
            )

            if success:
                self.log_to_widget("Upload para GitHub concluído com sucesso!")
                self.set_status("Upload concluído.")
                logger.info(f"Upload para {github_path} bem-sucedido.")
            else:
                self.log_to_widget("Falha no Upload para GitHub. Verifique os logs.")
                self.set_status("Falha no upload.")
                logger.error(f"Falha no upload para {github_path}.")

        except Exception as e:
            errmsg = f"Erro inesperado durante o upload: {e}"
            self.log_to_widget(f"ERRO: {errmsg}")
            self.set_status("Erro crítico no upload.")
            logger.error(errmsg, exc_info=True)
        finally:
            # Reabilita botões após a tentativa de upload
            self.set_button_state("run_scraper_button", tk.NORMAL)
            # Só reabilita upload se ainda tiver dados (mesmo que upload falhou)
            if self.scraped_data is not None and not self.scraped_data.empty:
                self.set_button_state("upload_button", tk.NORMAL)
            else:
                 self.set_button_state("upload_button", tk.DISABLED) # Mantém desabilitado se não há dados
            logger.info("Thread de upload finalizada.")


    def process_gui_queue(self):
        """Processa mensagens da fila para atualizar a GUI."""
        if self.stop_processing_queue:
            logger.debug("ScraperUploadTab: Parando fila GUI.")
            return # Sai e não reagenda

        try:
            while True: # Processa todas as mensagens pendentes
                try:
                    message_type, payload = self.gui_queue.get_nowait()

                    # <<< Seu código de processamento de mensagens para esta aba >>>
                    if message_type == "log":
                        self._update_log_widget(payload)
                    elif message_type == "status":
                        self._update_status_label(payload)
                    elif message_type == "button_state":
                        self._update_button_state(payload)
                    # Adicione outros tipos específicos desta aba, se houver
                    else:
                        logger.warning(f"Tipo de mensagem desconhecido na fila GUI (Scraper): {message_type}")
                    # <<< Fim do processamento >>>

                except queue.Empty: # Fila vazia, sai do loop while
                    break
                except Exception as e:
                    logger.error(f"Erro ao processar fila da GUI (Scraper): {e}", exc_info=True)
                    break # Sai do loop em caso de erro inesperado no processamento

        finally:
            # <<< PASSO 2: REAGENDA SÓ SE NÃO FOR PARAR >>>
            if not self.stop_processing_queue:
                try:
                    # Verifica se a janela principal ainda existe
                    if hasattr(self.main_tk_root, 'winfo_exists') and self.main_tk_root.winfo_exists():
                        self.main_tk_root.after(100, self.process_gui_queue)
                except Exception as e_resched:
                    # Evita logar erro se já está parando
                    if not self.stop_processing_queue:
                        logger.error(f"Erro reagendar fila GUI (Scraper): {e_resched}")

# --- Bloco para teste independente (opcional) ---
if __name__ == "__main__":
    # Configuração de log para teste
    test_logger = setup_logger("ScraperTabTest", level=logging.DEBUG)

    root = tk.Tk()
    root.title("Teste Aba Scraper")
    root.geometry("800x600")

    # Frame principal para a aba
    tab_frame = ttk.Frame(root)
    tab_frame.pack(expand=True, fill="both")

    # Instancia a classe da aba
    # O segundo argumento é a raiz principal do Tkinter
    app_tab = ScraperUploadTab(tab_frame, root)

    root.mainloop()