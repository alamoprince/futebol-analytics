import customtkinter as ctk
from tkinter import messagebox
import time

# Importações necessárias para a janela e suas dependências
from src.logger_config import setup_logger
from strategies.base_strategy import BettingStrategy

# Usa importações relativas para as outras abas na mesma pasta 'gui'
from .scraper_tab import ScraperUploadTab
from .training_tab import FootballPredictorDashboard
from .analyzer_tab import FeatureAnalyzerApp
from .interpreter_tab import ModelInterpreterApp

logger = setup_logger("MainWindow")

class MainApplication(ctk.CTk):
    """
    A janela principal da aplicação. Herda de ctk.CTk para ser a janela raiz.
    É inicializada com uma estratégia específica que dita o comportamento das abas.
    """
    
    # O construtor __init__ AGORA ACEITA o argumento 'strategy'
    def __init__(self, strategy: BettingStrategy):
        super().__init__()
        
        self.active_strategy = strategy
        
        self.title(f"Futebol Predictor Pro - Estratégia: {self.active_strategy.get_display_name()}")
        self.geometry("1300x850")
        self.minsize(1000, 750)

        self.tab_view = ctk.CTkTabview(self, width=1280, height=830)
        self.tab_view.pack(padx=20, pady=20, fill="both", expand=True)

        self.tab_view.add("Coleta & Upload")
        self.tab_view.add("Análise de Features")
        self.tab_view.add("Treino & Previsão")
        self.tab_view.add("Interpretação")
        
        self.tabs = {}

        logger.info("Instanciando abas da aplicação...")
        
        # Instancia as abas na ordem correta, passando as dependências necessárias
        self.tabs['scraper'] = self._create_tab(ScraperUploadTab, "Coleta & Upload")
        self.tabs['analyzer'] = self._create_tab(FeatureAnalyzerApp, "Análise de Features", strategy=self.active_strategy)
        
        analyzer_tab_ref = self.tabs.get('analyzer')
        
        self.tabs['training'] = self.tabs['training'] = self._create_tab(
            FootballPredictorDashboard, 
            "Treino & Previsão", 
            strategy=self.active_strategy,
            analyzer_app_ref=analyzer_tab_ref
        )
        self.tabs['interpreter'] = self._create_tab(ModelInterpreterApp, "Interpretação", strategy=self.active_strategy, analyzer_app_ref=analyzer_tab_ref)
        
        logger.info("Todas as abas instanciadas com sucesso.")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _create_tab(self, TabClass: type, tab_name: str, **kwargs):
        frame = self.tab_view.tab(tab_name)
        try:
            # Passa o frame, a janela principal (self), e quaisquer outros argumentos
            tab_instance = TabClass(frame, main_root=self, **kwargs)
            logger.debug(f"Aba '{tab_name}' instanciada.")
            return tab_instance
        except Exception as e:
            logger.error(f"Erro ao instanciar a aba '{tab_name}': {e}", exc_info=True)
            ctk.CTkLabel(frame, text=f"Erro ao carregar aba:\n{e}", text_color="red").pack(pady=20)
            return None

    def on_closing(self):
        if messagebox.askokcancel("Sair", "Deseja realmente sair?"):
            logger.info("Fechando aplicação.")
            
            for tab_instance in self.tabs.values():
                if tab_instance and hasattr(tab_instance, 'stop_processing_queue'):
                    tab_instance.stop_processing_queue = True
            
            self.after(150, self.destroy)