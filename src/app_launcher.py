# --- app_launcher.py ---
import tkinter as tk
from tkinter import ttk, messagebox
import sys
import os
from logger_config import setup_logger
import time 
import datetime

logger = setup_logger("MainApp")

# --- Path Setup (Verificar se está correto para sua estrutura) ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
# Se app_launcher.py estiver na raiz (futebol_analytics):
SRC_DIR = os.path.join(APP_DIR, 'src')
BASE_DIR = APP_DIR
# Se app_launcher.py estiver DENTRO de src:
# SRC_DIR = APP_DIR
# BASE_DIR = os.path.dirname(SRC_DIR)

if BASE_DIR not in sys.path: sys.path.insert(0, BASE_DIR)
if SRC_DIR not in sys.path: sys.path.insert(0, SRC_DIR)
logger.info(f"APP_DIR (launcher): {APP_DIR}")
logger.info(f"SRC_DIR: {SRC_DIR}")
logger.info(f"BASE_DIR: {BASE_DIR}")
logger.info(f"sys.path includes SRC_DIR?: {SRC_DIR in sys.path}, BASE_DIR?: {BASE_DIR in sys.path}")
# --- End Path Setup ---

try:
    # --- Importa as TRÊS classes ---
    from scraper_tab import ScraperUploadTab            # <<< NOVA ABA
    from main import FootballPredictorDashboard         # Aba de Treino/Previsão
    from feature_analyzer_tab import FeatureAnalyzerApp # Aba de Análise

except ImportError as e:
    logger.error(f"Erro ao importar classes da GUI: {e}", exc_info=True) # Adiciona traceback ao log
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
        self.root.title("Futebol Predictor Pro (v3 Abas)") # Título Atualizado
        self.root.geometry("1100x850") # Ajustar tamanho se necessário
        self.root.minsize(900, 700)

        # Cria o Notebook (abas)
        self.notebook = ttk.Notebook(self.root)

        # --- Cria os Frames para as TRÊS abas ---
        self.tab1_frame = ttk.Frame(self.notebook) # Scraper
        self.tab2_frame = ttk.Frame(self.notebook) # Treino/Previsão
        self.tab3_frame = ttk.Frame(self.notebook) # Análise Features

        # --- Adiciona os frames ao Notebook NA ORDEM DESEJADA ---
        self.notebook.add(self.tab1_frame, text=' Coleta & Upload ')
        self.notebook.add(self.tab2_frame, text=' Treino & Previsão ')
        self.notebook.add(self.tab3_frame, text=' Análise de Features ')

        # Empacota o notebook
        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)

        # --- Instancia a Classe de Scraper/Upload na Primeira Aba ---
        try:
            # Passa o frame da aba E a janela principal (para .after())
            self.scraper_tab = ScraperUploadTab(self.tab1_frame, main_root=self.root)
            logger.info("ScraperUploadTab instanciada com sucesso.")
        except Exception as e:
            logger.error(f"Erro ao instanciar ScraperUploadTab: {e}", exc_info=True)
            messagebox.showerror("Erro Aba 1", f"Falha ao carregar Aba Coleta:\n{e}")
            ttk.Label(self.tab1_frame, text=f"Erro ao carregar:\n{e}", foreground="red").pack(pady=20)

        # --- Instancia a Classe de Treino/Previsão na Segunda Aba ---
        try:
            # Passa o frame da aba E a janela principal
            self.predictor_dashboard = FootballPredictorDashboard(self.tab2_frame, main_root=self.root)
            logger.info("FootballPredictorDashboard instanciada com sucesso.")
        except Exception as e:
            logger.error(f"Erro ao instanciar FootballPredictorDashboard: {e}", exc_info=True)
            messagebox.showerror("Erro Aba 2", f"Falha ao carregar Aba Treino/Previsão:\n{e}")
            ttk.Label(self.tab2_frame, text=f"Erro ao carregar:\n{e}", foreground="red").pack(pady=20)

        # --- Instancia a Classe de Análise na Terceira Aba ---
        try:
            # Passa apenas o frame da aba como pai
            self.feature_analyzer = FeatureAnalyzerApp(self.tab3_frame, self.root)
            logger.info("FeatureAnalyzerApp instanciada com sucesso.")
        except Exception as e:
            logger.error(f"Erro ao instanciar FeatureAnalyzerApp: {e}", exc_info=True)
            messagebox.showerror("Erro Aba 3", f"Falha ao carregar Aba Análise:\n{e}")
            ttk.Label(self.tab3_frame, text=f"Erro ao carregar:\n{e}", foreground="red").pack(pady=20)

        # --- Tratamento de Fechamento ---
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        if messagebox.askokcancel("Sair", "Deseja realmente sair?"):
            logger.info("Fechando aplicação.")

            # --- CANCELAR O AFTER DA ABA DE ANÁLISE ---
            try:
                if hasattr(self, 'feature_analyzer') and hasattr(self.feature_analyzer, 'main_tk_root'):
                    if hasattr(self.feature_analyzer, 'stop_processing_queue'):
                        logger.debug("Sinalizando para FeatureAnalyzerApp parar a fila da GUI...")
                        self.feature_analyzer.stop_processing_queue = True
                    else:
                        logger.warning("FeatureAnalyzerApp não possui o flag 'stop_processing_queue'.")
                    # Damos um pequeno tempo para a última execução da fila talvez ocorrer
                    self.root.update_idletasks() # Processa eventos pendentes
                    time.sleep(0.15) # Espera um pouco mais que o intervalo do after (100ms)

            except Exception as e_cancel:
                logger.warning(f"Aviso: Erro ao tentar sinalizar parada da fila da GUI da Aba Análise: {e_cancel}")

            # Tentar fechar o driver do scraper se ele existir e estiver ativo
            try:
                 if hasattr(self, 'scraper_tab') and hasattr(self.scraper_tab, 'driver_instance') and self.scraper_tab.driver_instance:
                      logger.info("Tentando fechar WebDriver do scraper (se ativo)...")
                      # Adiciona uma checagem se o método quit existe antes de chamar
                      if hasattr(self.scraper_tab.driver_instance, 'quit') and callable(self.scraper_tab.driver_instance.quit):
                           self.scraper_tab.driver_instance.quit()
                      else:
                           logger.warning("Driver do scraper não possui método quit ou não é chamável.")
            except Exception as e_quit:
                 logger.warning(f"Aviso: Erro ao tentar fechar driver do scraper ao sair: {e_quit}")

            # Destrói a janela principal
            self.root.destroy()

# --- Execução Principal ---
if __name__ == "__main__":
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
        logger.info("DPI Awareness set (Windows).")
    except Exception:
        logger.debug("Falha ao configurar DPI Awareness (provavelmente não é Windows).")

    # Cria a janela principal do Tkinter
    main_root = tk.Tk()
    # Instancia a aplicação principal que gerencia as abas
    app = MainApplication(main_root)
    # Inicia o loop principal da GUI
    main_root.mainloop()