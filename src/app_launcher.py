# --- app_launcher.py ---
import tkinter as tk
from tkinter import ttk, messagebox
import sys
import os

# --- Path Setup (Corrigido: Assumindo app_launcher.py DENTRO de src) ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = APP_DIR
BASE_DIR = os.path.dirname(SRC_DIR)
if BASE_DIR not in sys.path: sys.path.insert(0, BASE_DIR)
if SRC_DIR not in sys.path: sys.path.insert(0, SRC_DIR)
print(f"APP_DIR (launcher): {APP_DIR}")
print(f"SRC_DIR: {SRC_DIR}")
print(f"BASE_DIR: {BASE_DIR}")
print(f"sys.path includes SRC_DIR?: {SRC_DIR in sys.path}, BASE_DIR?: {BASE_DIR in sys.path}")
# --- End Path Setup ---

try:
    # --- Importa as DUAS classes separadas ---
    from feature_analyzer_tab import FeatureAnalyzerApp
    from main import FootballPredictorDashboard # main.py contém FootballPredictorDashboard
except ImportError as e:
    print(f"Erro ao importar classes da GUI: {e}")
    print("Verifique os nomes dos arquivos/classes e se os arquivos estão em 'src'.")
    try: # Tenta mostrar erro na GUI se possível
        root_err = tk.Tk(); root_err.withdraw()
        messagebox.showerror("Erro de Importação", f"Não foi possível importar componentes da GUI:\n{e}\n\nVerifique o console.")
        root_err.destroy()
    except tk.TclError: pass
    sys.exit(1)
except Exception as e_init:
     print(f"Erro fatal durante imports iniciais: {e_init}")
     try: messagebox.showerror("Erro Fatal", f"Erro durante inicialização:\n{e_init}")
     except: pass
     sys.exit(1)

class MainApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("Futebol Predictor Pro (BackDraw_MultiSelect) - Abas") # Título mais específico
        self.root.geometry("1000x800")
        self.root.minsize(800, 600)

        # Cria o Notebook (abas) diretamente na janela principal
        self.notebook = ttk.Notebook(self.root)

        # Cria o Frame para a primeira aba
        self.tab1_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1_frame, text=' Análise de Features ')

        # Cria o Frame para a segunda aba
        self.tab2_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.tab2_frame, text=' Treino e Previsão ')

        # Empacota o notebook para preencher a janela
        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)

        # --- Instancia a Classe de Análise na Primeira Aba ---
        try:
            # Passa apenas o frame da aba como pai
            self.feature_analyzer = FeatureAnalyzerApp(self.tab1_frame)
            print("FeatureAnalyzerApp instanciada com sucesso.")
        except Exception as e:
            print(f"Erro ao instanciar FeatureAnalyzerApp: {e}")
            import traceback; traceback.print_exc()
            messagebox.showerror("Erro Aba 1", f"Falha ao carregar Aba Análise:\n{e}")
            ttk.Label(self.tab1_frame, text=f"Erro:\n{e}", foreground="red").pack(pady=20)

        # --- Instancia a Classe de Treino/Previsão na Segunda Aba ---
        try:
            # Passa o frame da aba E a janela principal (para o .after())
            self.predictor_dashboard = FootballPredictorDashboard(self.tab2_frame, main_root=self.root)
            print("FootballPredictorDashboard instanciada com sucesso.")
        except Exception as e:
            print(f"Erro ao instanciar FootballPredictorDashboard: {e}")
            import traceback; traceback.print_exc()
            messagebox.showerror("Erro Aba 2", f"Falha ao carregar Aba Treino/Previsão:\n{e}")
            ttk.Label(self.tab2_frame, text=f"Erro:\n{e}", foreground="red").pack(pady=20)

        # --- Tratamento de Fechamento ---
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        if messagebox.askokcancel("Sair", "Deseja realmente sair?"):
            # (Adicionar lógica para parar threads se necessário)
            print("Fechando aplicação.")
            self.root.destroy()

# --- Execução Principal ---
if __name__ == "__main__":
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
        print("DPI Awareness set (Windows).")
    except Exception: pass # Ignora se não for Windows ou der erro

    main_root = tk.Tk()
    app = MainApplication(main_root)
    main_root.mainloop()