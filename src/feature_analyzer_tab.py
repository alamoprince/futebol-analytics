# --- src/feature_analyzer_tab.py ---
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText
import pandas as pd
import numpy as np
import os
import sys
from typing import Optional
import datetime

# Adiciona diretórios ao path para encontrar config e data_handler
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path: sys.path.append(SRC_DIR)
if BASE_DIR not in sys.path: sys.path.append(BASE_DIR)

# Importa o necessário do config e data_handler
try:
    from config import HISTORICAL_DATA_PATH, FEATURE_COLUMNS # Pega path e features
    from data_handler import load_historical_data, calculate_historical_intermediate, calculate_rolling_stats, calculate_derived_features # Funções de carregamento e cálculo inicial
except ImportError as e:
    messagebox.showerror("Erro de Importação", f"Não foi possível importar config/data_handler:\n{e}")
    sys.exit(1)
except Exception as e_init:
     messagebox.showerror("Erro Fatal", f"Erro ao inicializar:\n{e_init}")
     sys.exit(1)

class FeatureAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Analisador de Features - Histórico")
        self.root.geometry("800x600")

        self.df_historical_raw: Optional[pd.DataFrame] = None
        self.df_historical_processed: Optional[pd.DataFrame] = None # Com colunas intermediárias

        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Painel Superior: Ações e Info Básica ---
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))

        load_button = ttk.Button(top_frame, text="Carregar Dados Históricos (Excel)", command=self.load_and_display_data)
        load_button.pack(side=tk.LEFT, padx=(0, 10))

        # Frame para informações básicas lado a lado
        info_container = ttk.Frame(main_frame)
        info_container.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        info_frame = ttk.LabelFrame(info_container, text=" Informações Gerais (df.info) ", padding=5)
        info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.info_text = ScrolledText(info_frame, height=10, state='disabled', wrap=tk.NONE, font=("Consolas", 9))
        self.info_text.pack(fill=tk.BOTH, expand=True)

        head_frame = ttk.LabelFrame(info_container, text=" Amostra dos Dados (df.head) ", padding=5)
        head_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        self.head_text = ScrolledText(head_frame, height=10, state='disabled', wrap=tk.NONE, font=("Consolas", 9))
        self.head_text.pack(fill=tk.BOTH, expand=True)

        # --- Painel Inferior: Estatísticas Descritivas e Alvo ---
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.BOTH, expand=True)

        desc_frame = ttk.LabelFrame(bottom_frame, text=" Estatísticas Descritivas (Features) ", padding=5)
        desc_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.desc_text = ScrolledText(desc_frame, height=15, state='disabled', wrap=tk.NONE, font=("Consolas", 9))
        self.desc_text.pack(fill=tk.BOTH, expand=True)

        target_frame = ttk.LabelFrame(bottom_frame, text=" Distribuição do Alvo (IsDraw) ", padding=5)
        target_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        self.target_text = ScrolledText(target_frame, height=15, state='disabled', wrap=tk.WORD, font=("Consolas", 10))
        self.target_text.pack(fill=tk.BOTH, expand=True)


    def _update_text_widget(self, text_widget: ScrolledText, content: str):
        """Helper para atualizar ScrolledText."""
        try:
            text_widget.config(state='normal')
            text_widget.delete('1.0', tk.END)
            text_widget.insert('1.0', content)
            text_widget.config(state='disabled')
        except tk.TclError: pass # Ignora erro se janela fechada


    def load_and_display_data(self):
        """Carrega os dados históricos e atualiza os displays."""
        self._update_text_widget(self.info_text, "Carregando dados...")
        self._update_text_widget(self.head_text, "")
        self._update_text_widget(self.desc_text, "")
        self._update_text_widget(self.target_text, "")
        self.df_historical_raw = None
        self.df_historical_processed = None

        try:
            # Carrega os dados brutos
            df_raw = load_historical_data(HISTORICAL_DATA_PATH)
            if df_raw is None:
                messagebox.showerror("Erro", f"Não foi possível carregar o arquivo:\n{HISTORICAL_DATA_PATH}")
                self._update_text_widget(self.info_text, "Falha ao carregar.")
                return

            self.df_historical_raw = df_raw.copy() # Guarda cópia bruta
            self.log_to_widget(self.info_text, f"Dados carregados: {self.df_historical_raw.shape[0]} linhas, {self.df_historical_raw.shape[1]} colunas.")

            # Exibe df.info()
            import io
            buffer = io.StringIO()
            self.df_historical_raw.info(buf=buffer)
            info_str = buffer.getvalue()
            self._update_text_widget(self.info_text, info_str)

            # Exibe df.head()
            self._update_text_widget(self.head_text, self.df_historical_raw.head(10).to_string())

            # Calcula colunas intermediárias para análise descritiva das features e alvo
            # Usa a função pública refatorada
            df_processed = calculate_historical_intermediate(self.df_historical_raw)
            # Calcula rolling stats (pode demorar um pouco) - Talvez mover para botão separado?
            # Por enquanto, calculamos aqui para ter as médias disponíveis
            stats_to_roll = ['Ptos', 'VG', 'CG']
            # Define a default rolling window size
            ROLLING_WINDOW = 5
            df_processed = calculate_rolling_stats(df_processed, stats_to_roll, window=ROLLING_WINDOW)
            # Calcula derivadas
            df_processed = calculate_derived_features(df_processed)

            # Guarda a versão processada
            self.df_historical_processed = df_processed

            # Exibe describe() das features que usaremos (se existirem)
            features_to_describe = [f for f in FEATURE_COLUMNS if f in self.df_historical_processed.columns]
            if features_to_describe:
                desc_stats = self.df_historical_processed[features_to_describe].describe().to_string()
                self._update_text_widget(self.desc_text, desc_stats)
            else:
                 self._update_text_widget(self.desc_text, "Nenhuma das features definidas em FEATURE_COLUMNS encontrada após processamento.")


            # Exibe distribuição do alvo 'IsDraw' (se existir)
            if 'IsDraw' in self.df_historical_processed.columns:
                target_dist = self.df_historical_processed['IsDraw'].value_counts(normalize=True).apply("{:.2%}".format).to_string()
                target_counts = self.df_historical_processed['IsDraw'].value_counts().to_string()
                self._update_text_widget(self.target_text, f"Contagem:\n{target_counts}\n\nProporção:\n{target_dist}")
            else:
                self._update_text_widget(self.target_text, "Coluna alvo 'IsDraw' não encontrada/calculada.")


        except Exception as e:
            messagebox.showerror("Erro no Processamento", f"Ocorreu um erro:\n{e}")
            self.log_to_widget(self.info_text, f"Erro: {e}")

    # Função de log adaptada para widget de texto
    def log_to_widget(self, text_widget: ScrolledText, message: str):
         try:
            text_widget.config(state='normal')
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            text_widget.insert(tk.END, f"[{timestamp}] {message}\n")
            text_widget.config(state='disabled')
            text_widget.see(tk.END)
         except tk.TclError: pass


if __name__ == "__main__":
    root = tk.Tk()
    app = FeatureAnalyzerApp(root)
    root.mainloop()