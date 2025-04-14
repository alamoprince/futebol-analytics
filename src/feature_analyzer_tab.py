import tkinter as tk
from tkinter import ttk, messagebox, filedialog # Keep other imports
from tkinter.scrolledtext import ScrolledText
from typing import Optional, List, Dict, Any
import os
import sys
import traceback

try:
    # Path setup might be needed here if you ever run this file directly
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(CURRENT_DIR) # Assuming it's inside 'src'
    if CURRENT_DIR not in sys.path: sys.path.append(CURRENT_DIR)
    if BASE_DIR not in sys.path: sys.path.append(BASE_DIR)

    from model_trainer import analyze_features
    from config import HISTORICAL_DATA_PATH, NEW_FEATURE_COLUMNS, FEATURE_COLUMNS, ROLLING_WINDOW # Use config window
    from data_handler import (load_historical_data, calculate_historical_intermediate,
                          calculate_probabilities, calculate_normalized_probabilities,
                          calculate_rolling_stats, calculate_rolling_std,
                          calculate_binned_features)
except ImportError as e:
     print(f"Import Error in feature_analyzer_tab.py: {e}")
     # Handle error appropriately, maybe raise it or show a message if possible
     raise # Re-raise for the main app to catch if imported

import pandas as pd
import numpy as np
import datetime
import io

class FeatureAnalyzerApp:
    
    def __init__(self, parent_frame):
        
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
        """Carrega, processa (com pipeline completo) e analisa dados históricos."""
        self.log("Iniciando: Carregando dados...")
        # Limpa displays
        widgets_to_clear = [
            self.info_text, self.head_text, self.desc_text,
            self.target_text, self.importance_text, self.corr_text
        ]
        for widget in widgets_to_clear:
            self._update_text_widget(widget, "Carregando...") # Limpa todos

        self.df_historical_raw = None
        self.df_historical_processed = None # DataFrame com TUDO calculado

        try:
            # 1. Carregar Dados Brutos
            df_raw = load_historical_data(HISTORICAL_DATA_PATH)
            if df_raw is None:
                messagebox.showerror("Erro", f"Falha carregar:\n{HISTORICAL_DATA_PATH}", parent=self.parent)
                self.log("Falha ao carregar.")
                self._update_text_widget(self.info_text, "Falha ao carregar.")
                return
            self.df_historical_raw = df_raw.copy()
            self.log(f"Dados brutos carregados: {self.df_historical_raw.shape}")

            # Display info/head do RAW
            buffer = io.StringIO()
            self.df_historical_raw.info(buf=buffer)
            self._update_text_widget(self.info_text, buffer.getvalue())
            self._update_text_widget(self.head_text, self.df_historical_raw.head(5).to_string())

            # 2. **EXECUTAR PIPELINE DE CÁLCULO COMPLETO**
            self.log("Executando pipeline completo de cálculo de features...")
            # PASSO A PASSO (espelhando preprocess_and_feature_engineer)
            self.log(" -> Calculando Intermediárias...")
            df_processed = calculate_historical_intermediate(self.df_historical_raw)
            if 'IsDraw' not in df_processed.columns: raise ValueError("'IsDraw' não criada.")

            # Garante probabilidades (pode ser redundante se intermediate já calcula)
            if not all(p in df_processed.columns for p in ['p_H', 'p_D', 'p_A']):
                 self.log(" -> Calculando Probabilidades p_H/D/A...")
                 df_processed = calculate_probabilities(df_processed)

            self.log(" -> Calculando Probabilidades Normalizadas...")
            df_processed = calculate_normalized_probabilities(df_processed)

            self.log(" -> Calculando Médias Rolling (VG, CG)...")
            stats_to_roll_mean = ['VG', 'CG'] # Defina quais médias quer calcular
            df_processed = calculate_rolling_stats(df_processed, stats_to_roll_mean, window=ROLLING_WINDOW)

            self.log(" -> Calculando StDev Rolling (CG)...") # Defina quais StDevs quer calcular
            stats_to_roll_std = ['CG']
            df_processed = calculate_rolling_std(df_processed, stats_to_roll_std, window=ROLLING_WINDOW)

            # Exemplo: Calcular médias e std para PONTOS também
            # self.log(" -> Calculando Médias/StDev Rolling (Ptos)...")
            # df_processed = calculate_rolling_stats(df_processed, ['Ptos'], window=ROLLING_WINDOW)
            # df_processed = calculate_rolling_std(df_processed, ['Ptos'], window=ROLLING_WINDOW)

            self.log(" -> Calculando Binning (Odd_D_Cat)...")
            df_processed = calculate_binned_features(df_processed)

            # Exemplo: Calcular Features Derivadas (se quiser analisá-las também)
            # self.log(" -> Calculando Derivadas (CV_HDA, Diff_Media_CG)...")
            # from data_handler import calculate_derived_features # Importa se necessário
            # df_processed = calculate_derived_features(df_processed) # Recalcula com base nas médias existentes

            self.df_historical_processed = df_processed # Armazena o DF com TUDO calculado
            self.log(f"Pipeline de cálculo concluído. Colunas: {list(self.df_historical_processed.columns)}")

            # 3. Preparar X, y para Análise Avançada
            #    Agora usa self.df_historical_processed que contém TUDO
            self.log("Preparando dados para análise avançada...")
            target_col = 'IsDraw'
            y = self.df_historical_processed[target_col].copy() # Pega o alvo

            # Seleciona TODAS as colunas numéricas calculadas como candidatas para análise
            # Exclui IDs, datas, alvo, intermediárias _raw, etc.
            cols_to_exclude_from_X = [
                target_col, 'Date', 'Home', 'Away', 'FT_Result', 'Profit', # IDs, Alvo, etc.
                'Goals_H_FT', 'Goals_A_FT', # Gols brutos
                'Ptos_H', 'Ptos_A',          # Pontos brutos
                'p_H', 'p_D', 'p_A',       # Probs não normalizadas
                'VG_H_raw', 'VG_A_raw', 'CG_H_raw', 'CG_A_raw' # Stats Raw
            ]
            #candidate_features = [
               # col for col in self.df_historical_processed.columns
                #if pd.api.types.is_numeric_dtype(self.df_historical_processed[col])
                #and col not in cols_to_exclude_from_X
            #]
            # Garante que só seleciona colunas que realmente existem
            candidate_features = [f for f in NEW_FEATURE_COLUMNS if f in self.df_historical_processed.columns]

            if not candidate_features:
                 self.log("Erro: Nenhuma feature numérica candidata encontrada após processamento.")
                 messagebox.showerror("Erro", "Nenhuma feature candidata encontrada para análise.", parent=self.parent)
                 return

            X = self.df_historical_processed[candidate_features].copy()
            self.log(f"Features candidatas para análise: {list(X.columns)}")

            # Combina X e y e remove NaNs para análise
            analysis_df = X.join(y)
            initial_rows = len(analysis_df)
            analysis_df = analysis_df.dropna() # Drop NaNs DEPOIS de calcular TUDO
            rows_dropped = initial_rows - len(analysis_df)
            self.log(f"Dados p/ análise (após dropna): {analysis_df.shape}. ({rows_dropped} linhas removidas).")

            if analysis_df.empty:
                 self.log("Erro: Nenhum dado restante para análise após dropna.")
                 messagebox.showwarning("Dados Insuficientes", "Nenhum dado completo para análise.", parent=self.parent)
                 return

            X_clean = analysis_df[candidate_features]
            y_clean = analysis_df[target_col]

            # 4. Display Describe (AGORA COM TODAS AS FEATURES CALCULADAS) & Target
            self.log("Atualizando displays Describe e Target...")
            self._update_text_widget(self.desc_text, X_clean.describe().to_string())

            target_dist = y_clean.value_counts(normalize=True).apply("{:.2%}".format).to_string()
            target_counts = y_clean.value_counts().to_string()
            self._update_text_widget(self.target_text, f"Contagem:\n{target_counts}\n\nProporção:\n{target_dist}")

            # 5. Análise Avançada (Correlação / Importância com X_clean)
            self.log("Executando análise de importância e correlação...")
            try:
                analysis_results = analyze_features(X_clean, y_clean)
                if analysis_results:
                    df_importance, df_correlation = analysis_results
                    # Display Importance
                    if df_importance is not None and not df_importance.empty:
                        self._update_text_widget(self.importance_text, df_importance.to_string(index=False))
                    else: self._update_text_widget(self.importance_text, "Não calculado.")
                    # Display Correlation
                    if df_correlation is not None and not df_correlation.empty:
                        pd.set_option('display.float_format', '{:.2f}'.format)
                        # Mostra correlação apenas com o alvo para economizar espaço?
                        corr_with_target = df_correlation['target_IsDraw'].sort_values(ascending=False).to_string()
                        self._update_text_widget(self.corr_text, "Correlação c/ Alvo (IsDraw):\n"+ corr_with_target)
                        # Ou mostra tudo: self._update_text_widget(self.corr_text, df_correlation.to_string())
                        pd.reset_option('display.float_format')
                    else: self._update_text_widget(self.corr_text, "Não calculado.")
                else:
                    self.log("Falha na execução de analyze_features.")
                    self._update_text_widget(self.importance_text, "Erro na análise.")
                    self._update_text_widget(self.corr_text, "Erro na análise.")
            except Exception as e_adv_analyze:
                self.log(f"Erro durante análise avançada: {e_adv_analyze}")
                messagebox.showerror("Erro Análise", f"Erro calcular:\n{e_adv_analyze}", parent=self.parent)
                self._update_text_widget(self.importance_text, f"Erro: {e_adv_analyze}")
                self._update_text_widget(self.corr_text, f"Erro: {e_adv_analyze}")

            self.log("Análise concluída.")
            self.status_label.config(text="Análise Concluída.") # Atualiza status final

        except FileNotFoundError:
            messagebox.showerror("Erro", f"Arquivo não encontrado:\n{HISTORICAL_DATA_PATH}", parent=self.parent)
            self.log("Falha: Arquivo não encontrado.")
            self.status_label.config(text="Erro: Arquivo não encontrado.")
            self._update_text_widget(self.info_text, "Falha: Arquivo não encontrado.")
        except Exception as e:
            self.log(f"Erro geral no processamento/análise: {e}")
            self.status_label.config(text=f"Erro: {e}")
            messagebox.showerror("Erro Processamento", f"Ocorreu um erro:\n{e}", parent=self.parent)
            self._update_text_widget(self.info_text, f"Erro: {e}")
            traceback.print_exc()
    