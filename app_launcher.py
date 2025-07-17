import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
import sys
import os
import time
from src.logger_config import setup_logger

logger = setup_logger("MainApp")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(APP_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

SRC_DIR = os.path.join(os.path.dirname(__file__), 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from src.gui.main_window import MainApplication 
    from src.gui.strategy_selector_window import StrategySelectorWindow
    from strategies.base_strategy import BettingStrategy

except ImportError as e:
    logger.error(f"Erro ao importar classes da GUI ou Estratégias: {e}", exc_info=True)
    try:
        root_err = tk.Tk()
        root_err.withdraw()
        messagebox.showerror("Erro de Importação", f"Não foi possível importar componentes essenciais:\n{e}\n\nVerifique os nomes e caminhos dos arquivos e as dependências.")
        root_err.destroy()
    except tk.TclError:
        pass
    sys.exit(1)
except Exception as e_init:
    logger.critical(f"Erro fatal durante os imports iniciais: {e_init}", exc_info=True)
    try:
        messagebox.showerror("Erro Fatal", f"Ocorreu um erro fatal durante a inicialização:\n{e_init}")
    except:
        pass
    sys.exit(1)

if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    
    selector_app = StrategySelectorWindow()
    selector_app.mainloop()

    selected_strategy_class = selector_app.get_selected_strategy_class()

    if selected_strategy_class:
        active_strategy = selected_strategy_class()
        
        app = MainApplication(strategy=active_strategy)
        app.mainloop()
    else:
        print("Nenhuma estratégia selecionada. Encerrando a aplicação.")