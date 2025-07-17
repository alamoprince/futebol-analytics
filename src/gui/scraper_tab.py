import tkinter as tk
from tkinter import ttk, messagebox
import customtkinter as ctk
from tkinter.scrolledtext import ScrolledText
import threading
import queue 
import sys
import os
import pandas as pd
from datetime import date, timedelta, datetime 
from typing import Optional, Any
import logging

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR) 
if CURRENT_DIR not in sys.path: sys.path.insert(0, CURRENT_DIR)
if BASE_DIR not in sys.path: sys.path.insert(0, BASE_DIR)

try:
    from logger_config import setup_logger
except ImportError:
    import tkinter as tk
    from tkinter import messagebox
    root_err = tk.Tk(); root_err.withdraw()
    messagebox.showerror("Erro de Importação (Scraper Tab)", "Erro: Falha ao importar 'setup_logger' de 'logger_config'. Verifique o arquivo.")
    root_err.destroy()
    import sys
    sys.exit(1)

logger = setup_logger("ScraperTab")
try:
    
    try:
        from src.scrapper.scraper_data import scrape_upcoming_fixtures
    except ImportError:
        logger.error("Erro: Falha ao importar 'scrape_upcoming_fixtures' de 'scraper_predictor'. Verifique o arquivo.")
        raise 

    try:
        from src.scrapper.github_manager import GitHubManager
    except ImportError:
        logger.error("Erro: Falha ao importar 'GitHubManager' de 'github_manager'. Verifique o arquivo.")
        raise 

    try:
        from config import (
            SCRAPER_TARGET_DAY, GITHUB_REPO_NAME, CHROMEDRIVER_PATH
            
        )
    except ImportError:
        logger.error("Erro: Falha ao importar variáveis de 'config'. Verifique o arquivo.")
        raise 

    from dotenv import load_dotenv 

except ImportError as e:
    logger.error(f"Erro CRÍTICO ao importar módulos em scraper_tab.py: {e}")
    try:
        root_err = tk.Tk(); root_err.withdraw()
        messagebox.showerror("Erro de Importação (Scraper Tab)", f"Não foi possível importar módulos essenciais:\n{e}\n\nVerifique o console/logs.")
        root_err.destroy()
    except Exception: pass
    sys.exit(1) 

class ScraperUploadTab:
    def __init__(self, parent_frame, main_root):
        self.parent = parent_frame
        self.main_tk_root = main_root 
        self.gui_queue = queue.Queue()
        self.stop_processing_queue = False
        self.active_thread: Optional[threading.Thread] = None
        self.is_process_running = False
        self.upload_thread: Optional[threading.Thread] = None
        self.scraped_data: Optional[pd.DataFrame] = None
        self.github_manager: Optional[GitHubManager] = None
        self.progress_bar: Optional[ctk.CTkProgressBar] = None
        self.progress_label: Optional[ctk.CTkLabel] = None
        self.progress_max_value: int = 1

        self.create_widgets()
        log_dir = os.path.join(BASE_DIR, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.process_gui_queue() 

    def log_to_widget(self, message: str):

        self.gui_queue.put(("log", message))

    def _update_log_widget(self, message):

        try:
            if hasattr(self, 'log_area') and self.log_area.winfo_exists():
                self.log_area.config(state='normal')
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
                self.log_area.insert(tk.END, f"[{timestamp}] {message}\n")
                self.log_area.see(tk.END)
                self.log_area.config(state='disabled')
        except tk.TclError: 
            pass 
        except Exception as e: 
            logger.error(f"Erro interno ao atualizar log widget: {e}")

    def set_status(self, message: str):

        self.gui_queue.put(("status", message))

    def _update_status_label(self, message):

        try:
            if hasattr(self, 'status_label') and self.status_label.winfo_exists():
                self.status_label.config(text=message)
        except tk.TclError: 
            pass
        except Exception as e: 
            logger.error(f"Erro interno ao atualizar status label: {e}")

    def set_button_state(self, button_name: str, state: str):
        self.gui_queue.put(("button_state", (button_name, state)))

    def _update_button_state(self, button_info):

        button_name, state = button_info
        try:
            button = getattr(self, button_name, None)
            if button and isinstance(button, ctk.CTkButton) and button.winfo_exists():
                button.config(state=state)
        except tk.TclError: 
             pass
        except Exception as e: 
            logger.error(f"Erro interno ao atualizar estado do botão '{button_name}': {e}")

    def create_widgets(self):
        
        main_frame = ctk.CTkFrame(self.parent, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        main_frame.grid_rowconfigure(2, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        control_frame = ctk.CTkFrame(main_frame)
        control_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        self.run_scraper_today_button = ctk.CTkButton(
            control_frame,
            text="Coletar & Enviar (Hoje)",
            command=lambda: self.start_scraping_and_upload_thread('today')
        )
        self.run_scraper_today_button.pack(side="left", padx=10, pady=10)

        self.run_scraper_tomorrow_button = ctk.CTkButton(
            control_frame,
            text="Coletar & Enviar (Amanhã)",
            command=lambda: self.start_scraping_and_upload_thread('tomorrow')
        )
        self.run_scraper_tomorrow_button.pack(side="left", padx=10, pady=10)

        progress_frame = ctk.CTkFrame(main_frame)
        progress_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        progress_frame.grid_columnconfigure(1, weight=1)

        self.progress_label = ctk.CTkLabel(progress_frame, text="Pronto.", anchor="w")
        self.progress_label.grid(row=0, column=0, padx=10, pady=5)
        
        self.progress_bar = ctk.CTkProgressBar(progress_frame, orientation="horizontal")
        self.progress_bar.set(0)
        self.progress_bar.grid(row=0, column=1, sticky="ew", padx=10, pady=5)

        log_frame = ctk.CTkFrame(main_frame)
        log_frame.grid(row=2, column=0, sticky="nsew")
        log_frame.grid_rowconfigure(0, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)

        self.log_area = ScrolledText(
            log_frame, height=10, wrap="word", font=("Consolas", 9),
            relief="flat", borderwidth=0,
            bg="#2B2B2B", fg="white", insertbackground="white"
        )
        self.log_area.grid(row=0, column=0, sticky="nsew", padx=1, pady=1)
        self.log_area.config(state='disabled')

        self.log_to_widget("Aba de Coleta e Upload inicializada.")

    def start_scraping_and_upload_thread(self, target_day: str):
        if self.is_process_running:
            self.log_to_widget("AVISO: Um processo de coleta já está em andamento.")
            return

        self.is_process_running = True
        self.scraped_data = None
        self.log_to_widget(f"Iniciando processo de 'Coletar & Enviar' para: {target_day.upper()}")
        
        self.gui_queue.put(("set_buttons_state", "disabled"))

        self.active_thread = threading.Thread(
            target=self._run_scrape_and_upload_task,
            args=(target_day,),
            daemon=True
        )
        self.active_thread.start()

    def start_scraping_thread(self):

        if self.scraper_thread and self.scraper_thread.is_alive():
            self.log_to_widget("Coleta já está em andamento.")
            return

        self.log_to_widget("Iniciando processo de coleta...")
        self.set_status(f"Coletando jogos de '{SCRAPER_TARGET_DAY}'...")
        self.set_button_state("run_scraper_button", tk.DISABLED)
        self.set_button_state("upload_button", tk.DISABLED)
        self.scraped_data = None 

        self.scraper_thread = threading.Thread(target=self._run_scrape_task, daemon=True)
        self.scraper_thread.start()

    def _run_scrape_and_upload_task(self, target_day: str):

        try:
            self.log_to_widget(f"Etapa 1: Coletando jogos de '{target_day}'...")
            
            def scraper_progress_callback(status: str, payload: Any):
                if status == "start": self.gui_queue.put(("progress_start", payload))
                elif status == "update": self.gui_queue.put(("progress_update", payload))

            df_fixtures = scrape_upcoming_fixtures(
                headless=True,
                chromedriver_path=CHROMEDRIVER_PATH,
                target_day=target_day, 
                progress_callback=scraper_progress_callback
            )

            if df_fixtures is None or df_fixtures.empty:
                self.log_to_widget("AVISO: Coleta falhou ou não encontrou jogos. Processo interrompido.")
                logger.warning(f"Coleta para '{target_day}' não retornou dados. Upload cancelado.")
                return 

            n_jogos = len(df_fixtures)
            self.log_to_widget(f"Etapa 1 concluída: {n_jogos} jogos coletados.")
            self.scraped_data = df_fixtures

            self.log_to_widget("Etapa 2: Iniciando upload automático para o GitHub...")
            self._perform_upload(target_day)

        except Exception as e:
            errmsg = f"ERRO CRÍTICO no processo de 'Coletar & Enviar': {e}"
            self.log_to_widget(errmsg)
            logger.error(errmsg, exc_info=True)
        finally:
            self.gui_queue.put(("set_buttons_state", "normal"))
            self.gui_queue.put(("progress_end", None))
            self.is_process_running = False
            logger.info(f"Thread de 'Coletar & Enviar' para '{target_day}' finalizada.")

    def start_upload_thread(self):

        if self.scraped_data is None:
            messagebox.showwarning("Dados Ausentes", "Execute a coleta primeiro antes de fazer o upload.", parent=self.parent)
            return
        if self.scraped_data.empty:
            messagebox.showinfo("Dados Vazios", "Não há jogos coletados para fazer upload.", parent=self.parent)
            return
        if self.upload_thread and self.upload_thread.is_alive():
            self.log_to_widget("Upload já está em andamento.")
            return

        self.log_to_widget("Iniciando processo de upload para GitHub...")
        self.set_status("Fazendo upload...")
        self.set_button_state("run_scraper_button", tk.DISABLED)
        self.set_button_state("upload_button", tk.DISABLED)

        self.upload_thread = threading.Thread(target=self._run_upload_task, daemon=True)
        self.upload_thread.start()

    def _perform_upload(self, target_day: str):
        try:
            if self.github_manager is None:
                self.log_to_widget("... inicializando conexão com GitHub.")
                dotenv_path = os.path.join(BASE_DIR, '.env')
                if os.path.exists(dotenv_path): load_dotenv(dotenv_path=dotenv_path, override=True)
                self.github_manager = GitHubManager()

            if target_day == "today":
                file_date = date.today()
            else: 
                file_date = date.today() + timedelta(days=1)
            
            file_date_str = file_date.strftime("%Y-%m-%d")
            csv_filename = f"scraped_fixtures_{file_date_str}.csv"
            github_path = f"data/raw_scraped/{csv_filename}"
            self.log_to_widget(f"... preparando para salvar em: {GITHUB_REPO_NAME}/{github_path}")

            commit_message = f"Atualiza/Cria jogos raspados para {file_date_str}"
            
            success = self.github_manager.save_or_update_github(
                df=self.scraped_data,
                path=github_path,
                commit_message_prefix=commit_message
            )

            if success:
                self.log_to_widget("Etapa 2 concluída: Upload para GitHub bem-sucedido!")
                logger.info(f"Upload para {github_path} bem-sucedido.")
            else:
                self.log_to_widget("ERRO: Falha no upload para GitHub. Verifique os logs.")
                logger.error(f"Falha no upload para {github_path}.")

        except Exception as e:
            errmsg = f"Erro inesperado durante o upload: {e}"
            self.log_to_widget(f"ERRO: {errmsg}")
            logger.error(errmsg, exc_info=True)

    def process_gui_queue(self):

        if self.stop_processing_queue:
            logger.debug("ScraperUploadTab: Parando o processamento da fila da GUI.")
            return

        try:
            while True:
                try:
                    message_type, payload = self.gui_queue.get_nowait()

                    if message_type == "log":
                        self._update_log_widget(payload)

                    elif message_type == "progress_start":
                        if self.progress_bar and self.progress_label and self.progress_bar.winfo_exists():
                            self.progress_max_value, start_text = payload
                            if self.progress_max_value <= 0:
                                self.progress_max_value = 1
                            self.progress_bar.set(0)
                            self.progress_label.configure(text=start_text)

                    elif message_type == "progress_update":
                        if self.progress_bar and self.progress_label and self.progress_bar.winfo_exists():
                            current_value, text = payload
                            progress_fraction = float(current_value) / float(self.progress_max_value)
                            progress_fraction = max(0.0, min(1.0, progress_fraction))
                            self.progress_bar.set(progress_fraction)
                            self.progress_label.configure(text=text)

                    elif message_type == "progress_end":
                        if self.progress_bar and self.progress_label and self.progress_bar.winfo_exists():
                            self.progress_bar.set(0)
                            self.progress_label.configure(text="Pronto.")
                    
                    elif message_type == "set_buttons_state":
                        state = payload 
                        if hasattr(self, 'run_scraper_today_button') and self.run_scraper_today_button.winfo_exists():
                            self.run_scraper_today_button.configure(state=state)
                        if hasattr(self, 'run_scraper_tomorrow_button') and self.run_scraper_tomorrow_button.winfo_exists():
                            self.run_scraper_tomorrow_button.configure(state=state)

                except queue.Empty:
                    break
                except Exception as e:
                    logger.error(f"Erro ao processar uma mensagem da fila da GUI (Scraper): {e}", exc_info=True)
                    break 

        finally:
            if not self.stop_processing_queue:
                try:
                    if self.main_tk_root and self.main_tk_root.winfo_exists():
                        self.main_tk_root.after(100, self.process_gui_queue)
                except Exception as e_resched:
                    if not self.stop_processing_queue:
                        logger.error(f"Erro crítico ao reagendar a fila da GUI (Scraper): {e_resched}")

if __name__ == "__main__":
    test_logger = setup_logger("ScraperTabTest", level=logging.DEBUG)

    root = tk.Tk()
    root.title("Teste Aba Scraper")
    root.geometry("800x600")

    tab_frame = ctk.CTkFrame(root)
    tab_frame.pack(expand=True, fill="both")

    app_tab = ScraperUploadTab(tab_frame, root)

    root.mainloop()