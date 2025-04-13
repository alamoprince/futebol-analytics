# Inside feature_analyzer_tab.py

import tkinter as tk
from tkinter import ttk, messagebox, filedialog # Keep other imports
from tkinter.scrolledtext import ScrolledText
from typing import Optional, List, Dict, Any
import os
import sys
# ... other necessary imports like pandas, os, sys, config, data_handler ...
# --- Ensure Path Setup is correct within this file too if run independently ---
# OR rely on the path setup in app_launcher.py when imported.

# --- Import supporting modules ---
try:
    # Path setup might be needed here if you ever run this file directly
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(CURRENT_DIR) # Assuming it's inside 'src'
    if CURRENT_DIR not in sys.path: sys.path.append(CURRENT_DIR)
    if BASE_DIR not in sys.path: sys.path.append(BASE_DIR)

    from model_trainer import analyze_features
    from config import HISTORICAL_DATA_PATH, FEATURE_COLUMNS, ROLLING_WINDOW # Use config window
    from data_handler import (load_historical_data,
                              calculate_historical_intermediate,
                              calculate_rolling_stats,
                              calculate_derived_features)
except ImportError as e:
     print(f"Import Error in feature_analyzer_tab.py: {e}")
     # Handle error appropriately, maybe raise it or show a message if possible
     raise # Re-raise for the main app to catch if imported

import pandas as pd
import numpy as np
import datetime
import io

class FeatureAnalyzerApp:
    # CHANGE HERE: Accept parent_frame instead of root
    def __init__(self, parent_frame):
        # self.root = root # REMOVE THIS
        self.parent = parent_frame # Store the parent frame

        self.df_historical_raw: Optional[pd.DataFrame] = None
        self.df_historical_processed: Optional[pd.DataFrame] = None

        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.parent, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Top Action Panel ---
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))
        load_button = ttk.Button(top_frame, text="Carregar & Analisar Dados Históricos", command=self.load_and_display_data) # Renamed button text slightly
        load_button.pack(side=tk.LEFT, padx=(0, 10))
        self.status_label = ttk.Label(top_frame, text="Pronto.") # Add a status label
        self.status_label.pack(side=tk.LEFT, padx=10)


        # --- Display Panes ---
        # Use PanedWindow for resizable sections
        paned_window_main = ttk.PanedWindow(main_frame, orient=tk.VERTICAL)
        paned_window_main.pack(fill=tk.BOTH, expand=True)

        # Top Pane (Info/Head)
        pane_top = ttk.PanedWindow(paned_window_main, orient=tk.HORIZONTAL)
        paned_window_main.add(pane_top, weight=1) # Add pane to main PanedWindow

        info_frame = ttk.LabelFrame(pane_top, text=" Infos Gerais (df.info) ", padding=5)
        pane_top.add(info_frame, weight=1) # Add frame to the horizontal pane
        self.info_text = ScrolledText(info_frame, height=10, state='disabled', wrap=tk.NONE, font=("Consolas", 9))
        self.info_text.pack(fill=tk.BOTH, expand=True)

        head_frame = ttk.LabelFrame(pane_top, text=" Amostra Dados (df.head) ", padding=5)
        pane_top.add(head_frame, weight=1)
        self.head_text = ScrolledText(head_frame, height=10, state='disabled', wrap=tk.NONE, font=("Consolas", 9))
        self.head_text.pack(fill=tk.BOTH, expand=True)

        # Middle Pane (Describe/Target)
        pane_middle = ttk.PanedWindow(paned_window_main, orient=tk.HORIZONTAL)
        paned_window_main.add(pane_middle, weight=1)

        desc_frame = ttk.LabelFrame(pane_middle, text=" Describe (Features Finais) ", padding=5)
        pane_middle.add(desc_frame, weight=1)
        self.desc_text = ScrolledText(desc_frame, height=10, state='disabled', wrap=tk.NONE, font=("Consolas", 9))
        self.desc_text.pack(fill=tk.BOTH, expand=True)

        target_frame = ttk.LabelFrame(pane_middle, text=" Distribuição Alvo (IsDraw) ", padding=5)
        pane_middle.add(target_frame, weight=1)
        self.target_text = ScrolledText(target_frame, height=10, state='disabled', wrap=tk.WORD, font=("Consolas", 10))
        self.target_text.pack(fill=tk.BOTH, expand=True)

        # Bottom Pane (Importance/Correlation) - NEW
        pane_bottom = ttk.PanedWindow(paned_window_main, orient=tk.HORIZONTAL)
        paned_window_main.add(pane_bottom, weight=1)

        importance_frame = ttk.LabelFrame(pane_bottom, text=" Importância Features (RF Rápido) ", padding=5)
        pane_bottom.add(importance_frame, weight=1)
        self.importance_text = ScrolledText(importance_frame, height=10, state='disabled', wrap=tk.NONE, font=("Consolas", 9))
        self.importance_text.pack(fill=tk.BOTH, expand=True)

        corr_frame = ttk.LabelFrame(pane_bottom, text=" Correlação Features (com Alvo) ", padding=5)
        pane_bottom.add(corr_frame, weight=1)
        self.corr_text = ScrolledText(corr_frame, height=10, state='disabled', wrap=tk.NONE, font=("Consolas", 9))
        self.corr_text.pack(fill=tk.BOTH, expand=True)

    # --- Helper methods (_update_text_widget, log_to_widget) remain the same ---
    def _update_text_widget(self, text_widget: ScrolledText, content: str):
        """Helper para atualizar ScrolledText."""
        try:
            if text_widget.winfo_exists():
                text_widget.config(state='normal')
                text_widget.delete('1.0', tk.END)
                text_widget.insert('1.0', content)
                text_widget.config(state='disabled')
        except tk.TclError: pass

    def log(self, message: str):
        """Basic log to console and status label."""
        print(f"[FeatureAnalyzer] {message}")
        try:
            if self.status_label.winfo_exists():
                self.status_label.config(text=message[:100]) # Show first 100 chars
                self.parent.update_idletasks() # Update GUI immediately
        except tk.TclError: pass

    def log_to_widget(self, text_widget: ScrolledText, message: str):
         try:
            if text_widget.winfo_exists():
                text_widget.config(state='normal')
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                text_widget.insert(tk.END, f"[{timestamp}] {message}\n")
                text_widget.config(state='disabled')
                text_widget.see(tk.END)
         except tk.TclError: pass

    def load_and_display_data(self):
        """Carrega, processa e analisa dados históricos, exibindo tudo."""
        self.log("Iniciando: Carregando dados...")
        # Clear previous results
        self._update_text_widget(self.info_text, "Carregando...")
        self._update_text_widget(self.head_text, "")
        self._update_text_widget(self.desc_text, "")
        self._update_text_widget(self.target_text, "")
        self._update_text_widget(self.importance_text, "") # Clear new widget
        self._update_text_widget(self.corr_text, "")       # Clear new widget
        self.df_historical_raw = None
        self.df_historical_processed = None

        try:
            # 1. Load Raw Data
            df_raw = load_historical_data(HISTORICAL_DATA_PATH)
            if df_raw is None:
                messagebox.showerror("Erro", f"Falha carregar:\n{HISTORICAL_DATA_PATH}", parent=self.parent)
                self.log("Falha ao carregar.")
                self._update_text_widget(self.info_text, "Falha ao carregar.")
                return
            self.df_historical_raw = df_raw.copy()
            self.log(f"Dados brutos: {self.df_historical_raw.shape}")

            # Display basic info/head
            buffer = io.StringIO()
            self.df_historical_raw.info(buf=buffer)
            self._update_text_widget(self.info_text, buffer.getvalue())
            self._update_text_widget(self.head_text, self.df_historical_raw.head(5).to_string()) # Show fewer rows

            # 2. Process Features (Calculate all required intermediate and final features)
            self.log("Processando features (intermediárias, rolling, derivadas)...")
            df_processed = calculate_historical_intermediate(self.df_historical_raw)
            stats_to_roll = ['Ptos', 'VG', 'CG']
            df_processed = calculate_rolling_stats(df_processed, stats_to_roll, window=ROLLING_WINDOW)
            df_processed = calculate_derived_features(df_processed)
            # Add direct features if missing (e.g., Odd_Over25_FT if used directly)
            direct_odds_features = [f for f in FEATURE_COLUMNS if f in ['Odd_H_FT', 'Odd_D_FT', 'Odd_Over25_FT', 'Odd_BTTS_Yes']]
            for f in direct_odds_features:
                if f not in df_processed.columns: df_processed[f] = np.nan
            self.log("Processamento de features concluído.")
            self.df_historical_processed = df_processed # Store fully processed data

            # 3. Prepare Data for Advanced Analysis (X, y using final features)
            self.log("Preparando dados para análise avançada...")
            target_col = 'IsDraw'
            # Ensure target exists (calculate if needed, as before)
            if target_col not in self.df_historical_processed.columns:
                if 'Goals_H_FT' in self.df_historical_processed.columns and 'Goals_A_FT' in self.df_historical_processed.columns:
                     self.log("Calculando 'IsDraw' a partir dos gols.")
                     h_goals = pd.to_numeric(self.df_historical_processed['Goals_H_FT'], errors='coerce')
                     a_goals = pd.to_numeric(self.df_historical_processed['Goals_A_FT'], errors='coerce')
                     self.df_historical_processed[target_col] = (h_goals == a_goals).astype(int)
                else:
                    self.log(f"Erro: Coluna alvo '{target_col}' ausente e não pode ser calculada.")
                    self._update_text_widget(self.target_text, "Erro: Alvo 'IsDraw' ausente.")
                    # Optionally stop here if target is critical for analysis
                    # return

            # Select final features defined in config and drop rows with NaNs *in those features*
            final_feature_cols = [f for f in FEATURE_COLUMNS if f in self.df_historical_processed.columns]
            cols_for_analysis = final_feature_cols + [target_col]
            if target_col not in self.df_historical_processed.columns: # Check again if calculation failed
                 self.log("Erro: Alvo 'IsDraw' não disponível para análise.")
                 analysis_df = self.df_historical_processed[final_feature_cols].copy()
                 analysis_df = analysis_df.dropna()
                 X = analysis_df
                 y = None # No target available
                 self._update_text_widget(self.target_text, "Alvo 'IsDraw' não disponível.")
            else:
                 analysis_df = self.df_historical_processed[cols_for_analysis].copy()
                 initial_rows = len(analysis_df)
                 analysis_df = analysis_df.dropna() # Drop NaNs for analysis/modeling input
                 rows_dropped = initial_rows - len(analysis_df)
                 self.log(f"Dados para análise: {analysis_df.shape}. ({rows_dropped} linhas removidas por NaNs nas features/alvo).")
                 if analysis_df.empty:
                     self.log("Erro: Nenhum dado restante após remover NaNs para análise.")
                     messagebox.showwarning("Dados Insuficientes", "Nenhum dado completo encontrado para análise avançada.", parent=self.parent)
                     return
                 X = analysis_df[final_feature_cols]
                 y = analysis_df[target_col]


            # 4. Display Describe & Target Distribution
            if not X.empty:
                self._update_text_widget(self.desc_text, X.describe().to_string())
            else:
                 self._update_text_widget(self.desc_text, "Nenhuma feature final para descrever.")

            if y is not None and not y.empty:
                target_dist = y.value_counts(normalize=True).apply("{:.2%}".format).to_string()
                target_counts = y.value_counts().to_string()
                self._update_text_widget(self.target_text, f"Contagem:\n{target_counts}\n\nProporção:\n{target_dist}")
            # (Handled missing 'y' case above)


            # 5. Perform and Display Advanced Analysis (Importance & Correlation)
            if X is not None and y is not None and not X.empty and not y.empty:
                self.log("Executando análise de importância e correlação...")
                try:
                    analysis_results = analyze_features(X, y) # Call the function
                    if analysis_results:
                        df_importance, df_correlation = analysis_results

                        # Display Importance
                        if df_importance is not None and not df_importance.empty:
                            self._update_text_widget(self.importance_text, df_importance.to_string(index=False))
                        else:
                            self._update_text_widget(self.importance_text, "Não foi possível calcular a importância.")

                        # Display Correlation (maybe just with target or top N features?)
                        if df_correlation is not None and not df_correlation.empty:
                            # Option 1: Show correlation with target only
                            # corr_with_target = df_correlation['target_IsDraw'].sort_values(ascending=False).to_string()
                            # self._update_text_widget(self.corr_text, corr_with_target)
                            # Option 2: Show full matrix (can be large)
                            # Limit display precision for readability
                            pd.set_option('display.float_format', '{:.2f}'.format)
                            self._update_text_widget(self.corr_text, df_correlation.to_string())
                            pd.reset_option('display.float_format') # Reset format
                        else:
                            self._update_text_widget(self.corr_text, "Não foi possível calcular a correlação.")
                    else:
                        self.log("Falha na execução de analyze_features.")
                        self._update_text_widget(self.importance_text, "Erro na análise.")
                        self._update_text_widget(self.corr_text, "Erro na análise.")

                except Exception as e_adv_analyze:
                    self.log(f"Erro durante análise avançada: {e_adv_analyze}")
                    messagebox.showerror("Erro Análise", f"Erro calcular importância/correlação:\n{e_adv_analyze}", parent=self.parent)
                    self._update_text_widget(self.importance_text, f"Erro: {e_adv_analyze}")
                    self._update_text_widget(self.corr_text, f"Erro: {e_adv_analyze}")
            else:
                self.log("Não foi possível executar análise avançada (X ou y ausentes/vazios).")
                self._update_text_widget(self.importance_text, "Dados insuficientes.")
                self._update_text_widget(self.corr_text, "Dados insuficientes.")


            self.log("Análise concluída.")

        except FileNotFoundError:
            messagebox.showerror("Erro", f"Arquivo não encontrado:\n{HISTORICAL_DATA_PATH}", parent=self.parent)
            self.log("Falha: Arquivo não encontrado.")
            self._update_text_widget(self.info_text, "Falha: Arquivo não encontrado.")
        except Exception as e:
            self.log(f"Erro geral: {e}")
            messagebox.showerror("Erro Processamento", f"Ocorreu um erro:\n{e}", parent=self.parent)
            self._update_text_widget(self.info_text, f"Erro: {e}")
            import traceback
            traceback.print_exc()
    