import tkinter as tk
from tkinter import ttk, messagebox, filedialog # Keep other imports
from tkinter.scrolledtext import ScrolledText
from typing import Optional, List, Dict, Any
import os
import sys
import traceback
import logging

try:
    # Path setup might be needed here if you ever run this file directly
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(CURRENT_DIR) # Assuming it's inside 'src'
    if CURRENT_DIR not in sys.path: sys.path.append(CURRENT_DIR)
    if BASE_DIR not in sys.path: sys.path.append(BASE_DIR)

    from model_trainer import analyze_features
    from config import HISTORICAL_DATA_PATH, NEW_FEATURE_COLUMNS, ROLLING_WINDOW # Use config window
    from data_handler import (load_historical_data, calculate_historical_intermediate,
                          calculate_probabilities, calculate_normalized_probabilities,
                          calculate_rolling_stats, calculate_rolling_std,
                          calculate_binned_features, calculate_derived_features)
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
        """Cria os widgets para a aba de Análise de Features."""
        main_frame = ttk.Frame(self.parent, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Painel de Ação Superior (inalterado) ---
        top_frame = ttk.Frame(main_frame); top_frame.pack(fill=tk.X, pady=(0, 10)); load_button = ttk.Button(top_frame, text="Carregar & Analisar Dados Históricos", command=self.load_and_display_data); load_button.pack(side=tk.LEFT, padx=(0, 10)); self.status_label = ttk.Label(top_frame, text="Pronto."); self.status_label.pack(side=tk.LEFT, padx=10);

        # --- Painéis de Display com PanedWindow (inalterado) ---
        paned_window_main = ttk.PanedWindow(main_frame, orient=tk.VERTICAL); paned_window_main.pack(fill=tk.BOTH, expand=True);
        pane_top = ttk.PanedWindow(paned_window_main, orient=tk.HORIZONTAL); paned_window_main.add(pane_top, weight=1);
        pane_middle = ttk.PanedWindow(paned_window_main, orient=tk.HORIZONTAL); paned_window_main.add(pane_middle, weight=1);
        pane_bottom = ttk.PanedWindow(paned_window_main, orient=tk.HORIZONTAL); paned_window_main.add(pane_bottom, weight=2);

        # --- Widgets de Texto com Scroll Duplo ---

        # Função auxiliar para criar um painel de texto com ambas as barras
        def create_text_panel(parent_pane, title, font_size=9, weight=1):
            frame = ttk.LabelFrame(parent_pane, text=f" {title} ", padding=5)
            parent_pane.add(frame, weight=weight) # Adiciona o frame ao PanedWindow

            # Cria as barras de rolagem ANTES do texto
            xscrollbar = ttk.Scrollbar(frame, orient=tk.HORIZONTAL)
            yscrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL)

            # Cria o widget de Texto
            text_widget = tk.Text(
                frame,
                height=8, # Altura padrão (ajuste se necessário)
                wrap=tk.NONE, # <--- MUITO IMPORTANTE: Desabilita quebra de linha
                font=("Consolas", font_size),
                yscrollcommand=yscrollbar.set, # Liga scroll vertical do texto à barra Y
                xscrollcommand=xscrollbar.set, # Liga scroll horizontal do texto à barra X
                state='disabled', # Começa desabilitado
                borderwidth=0, # Remove borda padrão do Text
                highlightthickness=0 # Remove borda de foco
            )

            # Configura as barras para controlar o texto
            yscrollbar.config(command=text_widget.yview)
            xscrollbar.config(command=text_widget.xview)

            # Empacota usando pack (alternativa ao grid)
            yscrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            xscrollbar.pack(side=tk.BOTTOM, fill=tk.X)
            text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            return text_widget # Retorna a referência ao widget de texto

        # -- Topo --
        self.info_text = create_text_panel(pane_top, "Infos Gerais (df.info)", font_size=9, weight=1)
        self.head_text = create_text_panel(pane_top, "Amostra Dados Raw (df.head)", font_size=9, weight=1)

        # -- Meio --
        self.desc_text = create_text_panel(pane_middle, "Describe (Features Calculadas)", font_size=9, weight=1)
        # Target Dist não precisa de scroll horizontal, mantém ScrolledText
        target_frame = ttk.LabelFrame(pane_middle, text=" Distribuição Alvo (IsDraw) ", padding=5)
        pane_middle.add(target_frame, weight=1)
        self.target_text = ScrolledText(target_frame, height=10, state='disabled', wrap=tk.WORD, font=("Consolas", 10), borderwidth=0, highlightthickness=0)
        self.target_text.pack(fill=tk.BOTH, expand=True)

        # -- Fundo --
        self.importance_text = create_text_panel(pane_bottom, "Importância Features (RF Rápido)", font_size=9, weight=1)
        self.corr_text = create_text_panel(pane_bottom, "Correlação Features (com Alvo)", font_size=8, weight=2)

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
        self.log("Iniciando: Carregando & Processando..."); widgets_to_clear = [self.info_text, self.head_text, self.desc_text, self.target_text, self.importance_text, self.corr_text]; [self._update_text_widget(w, "Carregando...") for w in widgets_to_clear]; self.df_historical_raw = None; self.df_historical_processed = None;
       
        try:
            # 1. Load
            df_raw = load_historical_data(HISTORICAL_DATA_PATH);
            if df_raw is None: messagebox.showerror("Erro", f"Falha carregar.", parent=self.parent); self.log("Falha carregar."); self._update_text_widget(self.info_text, "Falha."); return;
            self.df_historical_raw = df_raw.copy(); self.log(f"Dados brutos: {self.df_historical_raw.shape}");
            buffer = io.StringIO(); self.df_historical_raw.info(buf=buffer); self._update_text_widget(self.info_text, buffer.getvalue()); self._update_text_widget(self.head_text, self.df_historical_raw.head(5).to_string());

             # 2. Processamento Completo
            self.log("Processando features..."); df_p=self.df_historical_raw.copy(); # Usa cópia
            logging.info("=== INÍCIO PIPELINE DE CÁLCULO FEATURES (ANALYSIS TAB) ===") # Log de início

            # Etapa 2.1: Intermediárias
            self.log(" -> Calculando Intermediárias..."); df_p=calculate_historical_intermediate(df_p);
            logging.info(f"Após intermediate, cols: {list(df_p.columns)}")
            if 'IsDraw' not in df_p.columns: raise ValueError("Alvo 'IsDraw' ausente pós-intermediate.");
            if not all(c in df_p.columns for c in ['VG_H_raw','VG_A_raw','CG_H_raw','CG_A_raw']): raise ValueError("Stats Raw ausentes pós-intermediate.");
            if not all(p in df_p.columns for p in['p_H','p_D','p_A']): df_p=calculate_probabilities(df_p); logging.info("Probs p_H/D/A recalculadas.");

            # Etapa 2.2: Normalizadas
            self.log(" -> Calculando Normalizadas..."); df_p=calculate_normalized_probabilities(df_p);
            logging.info(f"Após norm probs, cols: {list(df_p.columns)}")
            if 'p_D_norm' not in df_p.columns: raise ValueError("'p_D_norm' ausente pós-normalização.");

            # Etapa 2.3: Rolling Stats (Mean)
            self.log(" -> Calculando Rolling Means..."); df_p=calculate_rolling_stats(df_p,['VG','CG'],window=ROLLING_WINDOW);
            logging.info(f"Após rolling means, cols: {list(df_p.columns)}")
            if 'Media_CG_H' not in df_p.columns: raise ValueError("'Media_CG_H' ausente pós-rolling mean.");

            # Etapa 2.4: Rolling Stats (Std)
            self.log(" -> Calculando Rolling Stds..."); df_p=calculate_rolling_std(df_p,['CG'],window=ROLLING_WINDOW);
            logging.info(f"Após rolling stds, cols: {list(df_p.columns)}")
            if 'Std_CG_H' not in df_p.columns: raise ValueError("'Std_CG_H' ausente pós-rolling std.");

            # Etapa 2.5: Rolling Goals
            self.log(" -> Calculando Rolling Goals...");
            from data_handler import calculate_rolling_goal_stats # Garante import
            df_p=calculate_rolling_goal_stats(df_p,window=ROLLING_WINDOW);
            logging.info(f"Após rolling goals, cols: {list(df_p.columns)}")
            if 'Avg_Gols_Marcados_H' not in df_p.columns: raise ValueError("'Avg_Gols_Marcados_H' ausente pós-rolling goals.");

            # Etapa 2.6: Poisson Draw Prob
            self.log(" -> Calculando Poisson Draw Prob...");
            from data_handler import calculate_poisson_draw_prob # Garante import
            df_p=calculate_poisson_draw_prob(df_p, max_goals=5);
            logging.info(f"Após Poisson Prob, cols: {list(df_p.columns)}")
            if 'Prob_Empate_Poisson' in df_p.columns:
                logging.info(f"  -> Poisson Prob calculada. Amostra:\n{df_p['Prob_Empate_Poisson'].dropna().head().to_string()}")
            else:
                 logging.error(" -> ERRO: Coluna 'Prob_Empate_Poisson' NÃO EXISTE após chamada da função!")
                 # Decide o que fazer: parar? continuar sem a feature?
                 # return # Parar aqui se a feature for essencial

            # Etapa 2.7: Binning
            self.log(" -> Calculando Binning..."); df_p=calculate_binned_features(df_p);
            logging.info(f"Após Binning, cols: {list(df_p.columns)}")
            if 'Odd_D_Cat' not in df_p.columns: logging.warning("'Odd_D_Cat' ausente pós-binning.");

            # Etapa 2.8: Derivadas
            self.log(" -> Calculando Derivadas..."); df_p=calculate_derived_features(df_p);
            logging.info(f"Após Derivadas, cols: {list(df_p.columns)}")
            if 'CV_HDA' not in df_p.columns: logging.warning("'CV_HDA' ausente pós-derivadas.");

            self.df_historical_processed = df_p; self.log("Processamento OK.");
            logging.info("=== FIM PIPELINE DE CÁLCULO FEATURES (ANALYSIS TAB) ===")

            # 3. Preparar X, y para Análise
            self.log("Preparando X, y p/ análise..."); target_col='IsDraw';
            if target_col not in self.df_historical_processed: logging.error("Alvo ausente!"); y=None;
            else: y = self.df_historical_processed[target_col];

            from config import FEATURE_COLUMNS # Usa a lista do config para treino/análise principal
            features_to_analyze = [f for f in FEATURE_COLUMNS if f in self.df_historical_processed.columns]
            missing_in_df = [f for f in FEATURE_COLUMNS if f not in self.df_historical_processed.columns]
            if missing_in_df:
                 self.log(f"AVISO: Features da lista FEATURE_COLUMNS não encontradas no DF processado: {missing_in_df}")

            if not features_to_analyze: self.log("Erro: Nenhuma das FEATURE_COLUMNS encontrada."); return;
            X = self.df_historical_processed[features_to_analyze].copy();
            self.log(f"Shape de X ANTES do dropna (features selecionadas): {X.shape}")

            analysis_df = X if y is None else X.join(y);
            initial_rows=len(analysis_df);
            # LOG ANTES DO DROPNA
            nan_check_before = analysis_df.isnull().sum()
            cols_with_nan_before = nan_check_before[nan_check_before > 0]
            if not cols_with_nan_before.empty:
                self.log(f"NaNs ANTES do dropna final (em {len(cols_with_nan_before)} cols):\n{cols_with_nan_before}")
            # --- DROP NA ---
            analysis_df.dropna(inplace=True);
            rows_dropped=initial_rows-len(analysis_df);
            self.log(f"Dados p/ análise (shape final): {analysis_df.shape} ({rows_dropped} linhas removidas por NaNs).");
            if analysis_df.empty: self.log("Erro: Nenhum dado p/ análise após dropna."); return;

            X_clean = analysis_df[features_to_analyze]; y_clean = analysis_df[target_col] if y is not None else None;

            # 4. Display Describe/Target
            self.log("Atualizando Describe/Target..."); self._update_text_widget(self.desc_text, X_clean.describe().to_string());
            if y_clean is not None: counts=y_clean.value_counts(); dist=y_clean.value_counts(normalize=True); self._update_text_widget(self.target_text, f"Contagem:\n{counts.to_string()}\n\nProporção:\n{dist.apply('{:.2%}'.format).to_string()}");
            else: self._update_text_widget(self.target_text, "Alvo não disponível.");

            # 5. Análise Avançada e Display
            self.log("Executando Análise Avançada...");
            if y_clean is None:
                self.log("Análise avançada pulada (alvo ausente).");
                self._update_text_widget(self.importance_text,"Alvo 'IsDraw' ausente.");
                self._update_text_widget(self.corr_text,"Alvo 'IsDraw' ausente.");
            else:
                try:
                    self.log(f"Chamando analyze_features com X_clean ({X_clean.shape}) e y_clean ({y_clean.shape})...");
                    # Chama a função com logs internos
                    analysis_results = analyze_features(X_clean, y_clean)
                    self.log("analyze_features retornou.") # Confirma que a função terminou

                    if analysis_results is not None:
                        imp_df, corr_df = analysis_results # Desempacota o resultado

                        # Verifica e exibe Importância
                        if isinstance(imp_df, pd.DataFrame) and not imp_df.empty:
                            self.log("Resultado Importância recebido e válido.")
                            self._update_text_widget(self.importance_text, imp_df.to_string(index=False))
                        else:
                            self.log("Resultado Importância é None ou vazio.")
                            self._update_text_widget(self.importance_text, "Não calculado ou resultado vazio.")

                        # Verifica e exibe Correlação
                        if isinstance(corr_df, pd.DataFrame) and not corr_df.empty:
                            self.log("Resultado Correlação recebido e válido.")
                            pd.set_option('display.float_format','{:.2f}'.format)
                            self._update_text_widget(self.corr_text, corr_df.to_string())
                            pd.reset_option('display.float_format')
                        else:
                             self.log("Resultado Correlação é None ou vazio.")
                             self._update_text_widget(self.corr_text, "Não calculado ou resultado vazio.")
                    else:
                        # Se a própria tupla retornada for None (não deveria acontecer com o código atual)
                        self.log("ERRO INESPERADO: analyze_features retornou None em vez de (DataFrame, DataFrame).")
                        self._update_text_widget(self.importance_text, "Erro interno na análise (retorno None).")
                        self._update_text_widget(self.corr_text, "Erro interno na análise (retorno None).")

                except Exception as e_adv_analyze:
                    # Captura erros durante a chamada ou desempacotamento
                    self.log(f"!!! ERRO GRAVE durante chamada/processamento de analyze_features: {e_adv_analyze}")
                    traceback.print_exc()
                    messagebox.showerror("Erro Análise", f"Erro calcular importância/correlação:\n{e_adv_analyze}", parent=self.parent)
                    self._update_text_widget(self.importance_text, f"Erro: {e_adv_analyze}")
                    self._update_text_widget(self.corr_text, f"Erro: {e_adv_analyze}")

            self.log("Análise avançada concluida.");
        except Exception as e:
            self.log(f"!!! ERRO GRAVE durante análise: {e}")
            traceback.print_exc()
        