# --- app_launcher.py ---
import tkinter as tk
from tkinter import ttk, messagebox
import sys
import os
import time # Necessário para sleep em on_closing
from logger_config import setup_logger
import datetime # Não sobrescreve time

logger = setup_logger("MainApp")

# --- Path Setup ---
APP_DIR = os.path.dirname(os.path.abspath(__file__)) # Este é o diretório 'src'
SRC_DIR = APP_DIR
BASE_DIR = os.path.dirname(SRC_DIR)
if BASE_DIR not in sys.path: sys.path.insert(0, BASE_DIR)
if SRC_DIR not in sys.path: sys.path.insert(0, SRC_DIR)
logger.info(f"APP_DIR (launcher): {APP_DIR}")
logger.info(f"SRC_DIR: {SRC_DIR}")
logger.info(f"BASE_DIR: {BASE_DIR}")
logger.info(f"sys.path includes SRC_DIR?: {SRC_DIR in sys.path}, BASE_DIR?: {BASE_DIR in sys.path}")
# --- End Path Setup ---

try:
    # --- Importa as QUATRO classes ---
    from src.scrapper.scraper_tab import ScraperUploadTab
    from src.gui.training_tab import FootballPredictorDashboard
    from src.gui.analyzer_tab import FeatureAnalyzerApp
    from src.gui.interpreter_tab import ModelInterpreterApp 

except ImportError as e:
    logger.error(f"Erro ao importar classes da GUI: {e}", exc_info=True)
    logger.error("Verifique os nomes dos arquivos/classes e se os arquivos estão em 'src'.")
    try:
        root_err = tk.Tk(); root_err.withdraw()
        messagebox.showerror("Erro de Importação", f"Não foi possível importar componentes da GUI:\n{e}\n\nVerifique o console/logs.")
        root_err.destroy()
    except tk.TclError: pass
    sys.exit(1)
except Exception as e_init:
    logger.critical(f"Erro fatal durante imports iniciais: {e_init}", exc_info=True)
    try: messagebox.showerror("Erro Fatal", f"Erro durante inicialização:\n{e_init}")
    except: pass
    sys.exit(1)

class MainApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("Futebol Predictor Pro (v4 Abas)") # Título Atualizado
        self.root.geometry("1200x900") # Aumentar tamanho talvez
        self.root.minsize(950, 750)

        self.notebook = ttk.Notebook(self.root)

        # --- Cria os Frames para as QUATRO abas ---
        self.tab1_frame = ttk.Frame(self.notebook) # Scraper
        self.tab2_frame = ttk.Frame(self.notebook) # Treino/Previsão
        self.tab3_frame = ttk.Frame(self.notebook) # Análise Features (Geral)
        self.tab4_frame = ttk.Frame(self.notebook) # <<< NOVA ABA: Interpretação Modelo

        # --- Adiciona os frames ao Notebook ---
        self.notebook.add(self.tab1_frame, text=' Coleta & Upload ')
        self.notebook.add(self.tab2_frame, text=' Treino & Previsão ')
        self.notebook.add(self.tab3_frame, text=' Análise de Features ')
        self.notebook.add(self.tab4_frame, text=' Interpretação Modelo ') 

        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)

        # --- Instancia Aba 1: Scraper ---
        try:
            self.scraper_tab = ScraperUploadTab(self.tab1_frame, main_root=self.root) # Precisa de main_root
            logger.info("ScraperUploadTab instanciada.")
        except Exception as e: logger.error(f"Erro instanciar ScraperUploadTab: {e}", exc_info=True); messagebox.showerror("Erro Aba 1", f"{e}"); ttk.Label(self.tab1_frame, text=f"Erro: {e}").pack()

        # --- Instancia Aba 2: Treino/Previsão ---
        try:
            self.predictor_dashboard = FootballPredictorDashboard(self.tab2_frame, main_root=self.root) # Precisa de main_root
            logger.info("FootballPredictorDashboard instanciada.")
        except Exception as e: logger.error(f"Erro instanciar FootballPredictorDashboard: {e}", exc_info=True); messagebox.showerror("Erro Aba 2", f"{e}"); ttk.Label(self.tab2_frame, text=f"Erro: {e}").pack()

        # --- Instancia Aba 3: Análise Features (SIMPLIFICADA) ---
        # Instancia ANTES da Aba 4 para passar a referência
        try:
            # NÃO precisa mais de main_root
            self.feature_analyzer = FeatureAnalyzerApp(self.tab3_frame)
            logger.info("FeatureAnalyzerApp instanciada.")
        except Exception as e: logger.error(f"Erro instanciar FeatureAnalyzerApp: {e}", exc_info=True); messagebox.showerror("Erro Aba 3", f"{e}"); ttk.Label(self.tab3_frame, text=f"Erro: {e}").pack()

        # --- Instancia Aba 4: Interpretação Modelo ---
        try:
            # Passa o frame, main_root E a referência da aba de análise para acessar os dados
            if hasattr(self, 'feature_analyzer'): # Garante que a aba 3 foi criada
                 self.model_interpreter = ModelInterpreterApp(self.tab4_frame, self.root, analyzer_app_ref=self.feature_analyzer)
                 logger.info("ModelInterpreterApp instanciada.")
            else:
                 logger.error("Não foi possível instanciar ModelInterpreterApp pois FeatureAnalyzerApp falhou.")
                 ttk.Label(self.tab4_frame, text="Erro: Dependência da Aba 3 falhou.").pack()
        except Exception as e:
            logger.error(f"Erro ao instanciar ModelInterpreterApp: {e}", exc_info=True)
            messagebox.showerror("Erro Aba 4", f"Falha ao carregar Aba Interpretação:\n{e}")
            ttk.Label(self.tab4_frame, text=f"Erro ao carregar:\n{e}", foreground="red").pack(pady=20)


        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        """Lida com o fechamento da janela principal, parando filas de todas as abas."""
        if messagebox.askokcancel("Sair", "Deseja realmente sair?"):
            logger.info("Fechando aplicação.")

            logger.debug("Tentando sinalizar parada das filas da GUI...")
            tabs_to_stop = {
                "FeatureAnalyzerApp": getattr(self, 'feature_analyzer', None), # Agora não tem mais fila
                "FootballPredictorDashboard": getattr(self, 'predictor_dashboard', None),
                "ScraperUploadTab": getattr(self, 'scraper_tab', None),
                "ModelInterpreterApp": getattr(self, 'model_interpreter', None) 
            }

            for tab_name, tab_instance in tabs_to_stop.items():
                # Verifica se a instância existe e se tem o método/flag esperado
                if tab_instance and hasattr(tab_instance, 'stop_processing_queue'):
                    try:
                        logger.debug(f"ANTES: Flag parada para {tab_name}: {getattr(tab_instance, 'stop_processing_queue', 'N/A')}")
                        tab_instance.stop_processing_queue = True
                        logger.debug(f"DEPOIS: Flag parada para {tab_name}: {getattr(tab_instance, 'stop_processing_queue', 'N/A')}") # Confirma mudança
                    except Exception as e_signal:
                        logger.warning(f"Erro ao sinalizar parada para {tab_name}: {e_signal}")

            self.root.update_idletasks()
            try:
                time.sleep(0.15) # Pequena pausa
            except AttributeError as e_sleep: logger.error(f"ERRO FATAL no fechamento (time.sleep): {e_sleep}. Verifique import 'time'.")
            except Exception as e_sleep_other: logger.warning(f"Erro durante time.sleep: {e_sleep_other}")

            # --- Tenta fechar WebDriver ---
            try:
                 if hasattr(self, 'scraper_tab') and hasattr(self.scraper_tab, 'driver_instance') and self.scraper_tab.driver_instance:
                      if hasattr(self.scraper_tab.driver_instance, 'quit') and callable(self.scraper_tab.driver_instance.quit): self.scraper_tab.driver_instance.quit(); logger.info("WebDriver fechado.")
                      else: logger.warning("Driver scraper sem método quit.")
            except Exception as e_quit: logger.warning(f"Aviso: Erro ao fechar driver scraper: {e_quit}")

            logger.info("Destruindo janela principal.")
            self.root.destroy()

# --- Execução Principal ---
if __name__ == "__main__":
    try: from ctypes import windll; windll.shcore.SetProcessDpiAwareness(1); logger.info("DPI Awareness set.")
    except Exception: logger.debug("Falha DPI Awareness.")
    main_root = tk.Tk()
    app = MainApplication(main_root)
    main_root.mainloop()