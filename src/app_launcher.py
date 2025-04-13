# app_launcher.py
import tkinter as tk
from tkinter import ttk, messagebox
import sys
import os

APP_DIR = os.path.dirname(os.path.abspath(__file__)) 
SRC_DIR = APP_DIR                            
BASE_DIR = os.path.dirname(SRC_DIR)          

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR) 

print(f"APP_DIR (localização de app_launcher.py): {APP_DIR}")
print(f"SRC_DIR (deve ser o mesmo que APP_DIR): {SRC_DIR}")
print(f"BASE_DIR (Pai de SRC_DIR): {BASE_DIR}")
print(f"sys.path inclui SRC_DIR?: {SRC_DIR in sys.path}, BASE_DIR?: {BASE_DIR in sys.path}")


try:
    # Importa as classes necessárias para a GUI
    from feature_analyzer_tab import FeatureAnalyzerApp
    from main import FootballPredictorDashboard 
except ImportError as e:
    print(f"Erro ao importar classes da GUI: {e}")
    print("Por favor, certifique-se de que:")
    print("1. app_launcher.py está no local correto em relação ao diretório 'src'.")
    print("2. Os arquivos Python (feature_analyzer_tab.py, main.py, etc.) estão no diretório 'src'.")
    print("3. Os nomes dos arquivos/classes correspondem aos imports.")
    # Mostra o erro em uma janela popup, se o tkinter estiver disponível
    try:
       root_err = tk.Tk()
       root_err.withdraw()
       messagebox.showerror("Erro de Importação", f"Não foi possível importar os componentes da GUI:\n{e}\n\nVerifique os detalhes do caminho no console.")
       root_err.destroy()
    except tk.TclError:
       pass # Caso o tkinter também falhe
    sys.exit(1)
except Exception as e_init:
    print(f"Erro fatal durante os imports iniciais: {e_init}")
    messagebox.showerror("Erro Fatal", f"Erro durante a inicialização:\n{e_init}")
    sys.exit(1)


class MainApplication:
    def __init__(self, root):
       self.root = root
       self.root.title("Futebol Analytics Dashboard")
       # Ajusta o tamanho conforme necessário
       self.root.geometry("1000x800")
       self.root.minsize(800, 600)

       # --- Cria o Notebook (Abas) ---
       self.notebook = ttk.Notebook(self.root)

       # --- Cria Frames para cada Aba ---
       self.tab1_frame = ttk.Frame(self.notebook)
       self.tab2_frame = ttk.Frame(self.notebook)

       self.notebook.add(self.tab1_frame, text='Análise de Features')
       self.notebook.add(self.tab2_frame, text='Treino e Previsão')

       self.notebook.pack(expand=True, fill='both', padx=10, pady=10)

       # --- Instancia a lógica da GUI em cada Frame de Aba ---
       # Passa o frame específico da aba como o pai
       try:
          self.feature_analyzer = FeatureAnalyzerApp(self.tab1_frame)
          print("FeatureAnalyzerApp instanciado com sucesso.")
       except Exception as e:
          print(f"Erro ao instanciar FeatureAnalyzerApp: {e}")
          messagebox.showerror("Erro na Aba 1", f"Falha ao carregar a aba de Análise de Features:\n{e}")
          # Adiciona um rótulo na aba indicando o erro
          ttk.Label(self.tab1_frame, text=f"Erro ao carregar a aba:\n{e}", foreground="red").pack(pady=20)


       try:
          # Passa a referência da janela principal se necessário pela classe,
          # embora idealmente ela deva usar seu parent_frame.
          # Vamos assumir que FootballPredictorDashboard precisa da janela principal para chamadas `after`.
          self.predictor_dashboard = FootballPredictorDashboard(self.tab2_frame, main_root=self.root)
          print("FootballPredictorDashboard instanciado com sucesso.")
       except Exception as e:
          print(f"Erro ao instanciar FootballPredictorDashboard: {e}")
          import traceback
          traceback.print_exc() # Imprime o rastreamento completo para depuração
          messagebox.showerror("Erro na Aba 2", f"Falha ao carregar a aba de Treino/Previsão:\n{e}")
          ttk.Label(self.tab2_frame, text=f"Erro ao carregar a aba:\n{e}", foreground="red").pack(pady=20)


       # --- Lida com o Fechamento ---
       self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
       # Adiciona qualquer lógica de limpeza necessária aqui
       if messagebox.askokcancel("Sair", "Deseja realmente sair da aplicação?"):
          # Potencialmente para threads se o dashboard do preditor tiver threads em execução
          # if hasattr(self.predictor_dashboard, 'stop_threads'): # Exemplo
          #     self.predictor_dashboard.stop_threads()
          print("Fechando aplicação.")
          self.root.destroy()

# --- Execução Principal ---
if __name__ == "__main__":
    # Define a conscientização de DPI para melhor escala no Windows (opcional)
    try:
       from ctypes import windll
       windll.shcore.SetProcessDpiAwareness(1)
    except ImportError:
       pass # Ignora se ctypes ou windll não estiverem disponíveis (não-Windows)
    except Exception as e:
       print(f"Nota: Não foi possível definir a conscientização de DPI - {e}")

    main_root = tk.Tk()
    app = MainApplication(main_root)
    main_root.mainloop()