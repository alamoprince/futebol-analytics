import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText
from typing import Optional, List, Dict, Any, Tuple # Added Tuple
from sklearn.ensemble import RandomForestClassifier # Import RandomForestClassifier
import os
import sys
import traceback
from logger_config import setup_logger
import pandas as pd
import numpy as np # Import numpy
import datetime
import io

logger = setup_logger("FeatureAnalyzerApp")

try:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(CURRENT_DIR)
    if CURRENT_DIR not in sys.path: sys.path.append(CURRENT_DIR)
    if BASE_DIR not in sys.path: sys.path.append(BASE_DIR)

    # Import analysis function correctly
    from model_trainer import analyze_features
    from config import FEATURE_COLUMNS, ROLLING_WINDOW, GOALS_COLS, RANDOM_STATE # Use FEATURE_COLUMNS
    from data_handler import (load_historical_data, calculate_historical_intermediate,
                          calculate_probabilities, calculate_normalized_probabilities,
                          calculate_rolling_stats, calculate_rolling_std,
                          calculate_binned_features, calculate_derived_features,
                          calculate_rolling_goal_stats, calculate_poisson_draw_prob) # Added missing imports
except ImportError as e:
     # Use logger for import errors during development
     logger.critical(f"Import Error in feature_analyzer_tab.py: {e}", exc_info=True)
     # Show a simple message if GUI is possible
     try:
         root_err = tk.Tk(); root_err.withdraw()
         messagebox.showerror("Import Error", f"Failed to import necessary modules:\n{e}\n\nCheck logs for details.")
         root_err.destroy()
     except Exception:
         print(f"CRITICAL Import Error: {e}")
     sys.exit(1) # Exit if imports fail


class FeatureAnalyzerApp:

    def __init__(self, parent_frame):
        self.parent = parent_frame
        self.df_historical_raw: Optional[pd.DataFrame] = None
        self.df_historical_processed: Optional[pd.DataFrame] = None
        # Initialize text widgets as None initially
        self.info_text = None
        self.head_text = None
        self.desc_text = None
        self.target_text = None
        self.importance_text = None
        self.corr_text = None
        self.status_label = None
        self.create_widgets()

    def create_widgets(self):
        """Cria os widgets para a aba de Análise de Features."""
        main_frame = ttk.Frame(self.parent, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        top_frame = ttk.Frame(main_frame); top_frame.pack(fill=tk.X, pady=(0, 10));
        load_button = ttk.Button(top_frame, text="Carregar & Analisar Dados Históricos", command=self.load_and_display_data); load_button.pack(side=tk.LEFT, padx=(0, 10));
        self.status_label = ttk.Label(top_frame, text="Pronto."); self.status_label.pack(side=tk.LEFT, padx=10);

        paned_window_main = ttk.PanedWindow(main_frame, orient=tk.VERTICAL); paned_window_main.pack(fill=tk.BOTH, expand=True);
        pane_top = ttk.PanedWindow(paned_window_main, orient=tk.HORIZONTAL); paned_window_main.add(pane_top, weight=1);
        pane_middle = ttk.PanedWindow(paned_window_main, orient=tk.HORIZONTAL); paned_window_main.add(pane_middle, weight=1);
        pane_bottom = ttk.PanedWindow(paned_window_main, orient=tk.HORIZONTAL); paned_window_main.add(pane_bottom, weight=2);

        def create_text_panel(parent_pane, title, font_size=9, weight=1):
            frame = ttk.LabelFrame(parent_pane, text=f" {title} ", padding=5)
            parent_pane.add(frame, weight=weight)
            xscrollbar = ttk.Scrollbar(frame, orient=tk.HORIZONTAL)
            yscrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL)
            text_widget = tk.Text(
                frame, height=8, wrap=tk.NONE, font=("Consolas", font_size),
                yscrollcommand=yscrollbar.set, xscrollcommand=xscrollbar.set,
                state='disabled', borderwidth=0, highlightthickness=0
            )
            yscrollbar.config(command=text_widget.yview)
            xscrollbar.config(command=text_widget.xview)
            yscrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            xscrollbar.pack(side=tk.BOTTOM, fill=tk.X)
            text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            return text_widget

        # Assign created widgets to instance attributes
        self.info_text = create_text_panel(pane_top, "Infos Gerais (df.info)", font_size=9, weight=1)
        self.head_text = create_text_panel(pane_top, "Amostra Dados Raw (df.head)", font_size=9, weight=1)
        self.desc_text = create_text_panel(pane_middle, "Describe (Features Calculadas)", font_size=9, weight=1)

        target_frame = ttk.LabelFrame(pane_middle, text=" Distribuição Alvo (IsDraw) ", padding=5)
        pane_middle.add(target_frame, weight=1)
        self.target_text = ScrolledText(target_frame, height=10, state='disabled', wrap=tk.WORD, font=("Consolas", 10), borderwidth=0, highlightthickness=0)
        self.target_text.pack(fill=tk.BOTH, expand=True)

        self.importance_text = create_text_panel(pane_bottom, "Importância Features (RF Rápido)", font_size=9, weight=1)
        self.corr_text = create_text_panel(pane_bottom, "Correlação Features (com Alvo)", font_size=9, weight=1)

    def _update_text_widget(self, text_widget: Optional[tk.Text | ScrolledText], content: str):
        """Helper para atualizar tk.Text ou ScrolledText."""
        if text_widget is None: # Check if widget exists
             logger.warning("_update_text_widget: Tentativa de atualizar widget None.")
             return
        try:
            if text_widget.winfo_exists():
                text_widget.config(state='normal')
                text_widget.delete('1.0', tk.END)
                text_widget.insert('1.0', content)
                text_widget.config(state='disabled')
        except tk.TclError as e:
             logger.debug(f"TclError updating text widget: {e}") # Debug level for common error
        except AttributeError as e:
             logger.warning(f"AttributeError updating text widget (likely None): {e}")
        except Exception as e:
             logger.error(f"Unexpected error updating text widget: {e}", exc_info=True)


    def log(self, message: str):
        """Basic log to console and status label."""
        logger.info(f"[GUI] {message}") # Use INFO level for GUI logs
        try:
            if self.status_label and self.status_label.winfo_exists():
                self.status_label.config(text=message[:100])
                # Use update_idletasks only if immediate feedback is crucial
                # self.parent.update_idletasks()
        except tk.TclError: pass
        except Exception as e:
             logger.error(f"Error updating status label: {e}", exc_info=True)


    def log_to_widget(self, text_widget: Optional[tk.Text | ScrolledText], message: str):
         if text_widget is None: return
         try:
            if text_widget.winfo_exists():
                text_widget.config(state='normal')
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                text_widget.insert(tk.END, f"[{timestamp}] {message}\n")
                text_widget.config(state='disabled')
                text_widget.see(tk.END)
         except tk.TclError: pass
         except Exception as e:
             logger.error(f"Error logging to widget: {e}", exc_info=True)


    def load_and_display_data(self):
        self.log("Iniciando: Carregando & Processando dados históricos...")
        widgets_to_clear = [self.info_text, self.head_text, self.desc_text, self.target_text, self.importance_text, self.corr_text]
        for w in widgets_to_clear:
             self._update_text_widget(w, "Carregando...")

        self.df_historical_raw = None
        self.df_historical_processed = None

        try:
            # 1. Load Raw Data (using the corrected load_historical_data)
            self.log("Etapa 1: Carregando dados brutos...")
            df_raw = load_historical_data() # This now handles invalid odds and XG=0 -> NaN
            if df_raw is None or df_raw.empty:
                 errmsg = "Falha ao carregar dados históricos ou nenhum dado encontrado."
                 messagebox.showerror("Erro", errmsg, parent=self.parent)
                 self.log(errmsg)
                 self._update_text_widget(self.info_text, errmsg)
                 return
            self.df_historical_raw = df_raw.copy()
            self.log(f"Dados brutos carregados: {self.df_historical_raw.shape}")

            # Display Raw Info and Head
            buffer = io.StringIO(); self.df_historical_raw.info(buf=buffer); self._update_text_widget(self.info_text, buffer.getvalue());
            # Use pandas display options for better head formatting
            with pd.option_context('display.max_rows', 10, 'display.max_columns', None, 'display.width', 1000):
                 self._update_text_widget(self.head_text, self.df_historical_raw.head(5).to_string());

            # --- PASSO 2: Calcular Médias da Liga (do df_raw, NaN safe) ---
            self.log("Etapa 2: Calculando médias da liga...")
            goals_h_col = GOALS_COLS.get('home', 'Goals_H_FT')
            goals_a_col = GOALS_COLS.get('away', 'Goals_A_FT')
            # Use nanmean to ignore NaNs
            avg_h_league = np.nanmean(self.df_historical_raw[goals_h_col]) if goals_h_col in self.df_historical_raw else 1.0
            avg_a_league = np.nanmean(self.df_historical_raw[goals_a_col]) if goals_a_col in self.df_historical_raw else 1.0
            # Handle case where mean is NaN (e.g., all goals are NaN)
            if pd.isna(avg_h_league): avg_h_league = 1.0
            if pd.isna(avg_a_league): avg_a_league = 1.0
            # Ensure averages are not zero for division robustness
            epsilon = 1e-6
            avg_h_league_safe = max(avg_h_league, epsilon)
            avg_a_league_safe = max(avg_a_league, epsilon)
            self.log(f"Médias Liga (NaN-safe): H={avg_h_league_safe:.3f}, A={avg_a_league_safe:.3f}")

            # --- PASSO 3: Processamento Completo de Features ---
            self.log("Etapa 3: Processando features (Pipeline Completo)...");
            df_p = self.df_historical_raw.copy(); # Start with a fresh copy
            logger.info("=== INÍCIO PIPELINE DE CÁLCULO FEATURES (ANALYSIS TAB) ===")

            # 3.1: Intermediárias (Result, Pts, IsDraw, VG/CG Raw, p_H/D/A)
            self.log(" -> 3.1 Calculando Intermediárias..."); df_p=calculate_historical_intermediate(df_p);
            if 'IsDraw' not in df_p.columns or df_p['IsDraw'].isnull().all(): raise ValueError("Alvo 'IsDraw' ausente ou todo NaN pós-intermediate.");
            # Ensure probs exist or were calculated
            if not all(c in df_p.columns for c in ['p_H','p_D','p_A']):
                 logger.warning("Probabilidades p_H/D/A ausentes após intermediate, recalculando...")
                 df_p=calculate_probabilities(df_p);
                 if not all(c in df_p.columns for c in ['p_H','p_D','p_A']):
                      raise ValueError("Falha ao recalcular p_H/D/A.")
            # Check if VG/CG raw were created (needed for rolling)
            if not all(c in df_p.columns for c in ['VG_H_raw','VG_A_raw','CG_H_raw','CG_A_raw']):
                logger.warning("VG/CG Raw ausentes após intermediate. Rolling VG/CG podem falhar.") # Warn but continue

            # 3.2: Normalizadas
            self.log(" -> 3.2 Calculando Normalizadas..."); df_p=calculate_normalized_probabilities(df_p);
            if 'p_D_norm' not in df_p.columns: logger.warning("'p_D_norm' ausente pós-normalização.");

            # 3.3: Rolling Stats (Mean - VG, CG)
            self.log(" -> 3.3 Calculando Rolling Means (VG, CG)..."); df_p=calculate_rolling_stats(df_p,['VG','CG'],window=ROLLING_WINDOW);
            if 'Media_CG_H' not in df_p.columns: logger.warning("'Media_CG_H' ausente pós-rolling mean.");

            # 3.4: Rolling Stats (Std - CG)
            self.log(" -> 3.4 Calculando Rolling Stds (CG)..."); df_p=calculate_rolling_std(df_p,['CG'],window=ROLLING_WINDOW);
            if 'Std_CG_H' not in df_p.columns: logger.warning("'Std_CG_H' ausente pós-rolling std.");

            # 3.5: Rolling Goals + FA/FD (Passando Médias SAFE da Liga)
            self.log(" -> 3.5 Calculando Rolling Goals e FA/FD...");
            df_p=calculate_rolling_goal_stats(df_p, window=ROLLING_WINDOW,
                                              avg_goals_home_league=avg_h_league_safe, # <<< PASSA MÉDIA SAFE
                                              avg_goals_away_league=avg_a_league_safe) # <<< PASSA MÉDIA SAFE
            if 'FA_H' not in df_p.columns: logger.warning("Coluna FA/FD ('FA_H') não foi criada.");

            # 3.6: Poisson Draw Prob (Passando Médias SAFE da Liga)
            self.log(" -> 3.6 Calculando Poisson Draw Prob...");
            df_p=calculate_poisson_draw_prob(df_p,
                                             avg_goals_home_league=avg_h_league_safe, # <<< PASSA MÉDIA SAFE
                                             avg_goals_away_league=avg_a_league_safe, # <<< PASSA MÉDIA SAFE
                                             max_goals=5);
            if 'Prob_Empate_Poisson' not in df_p.columns: logger.warning("Coluna Poisson não foi criada.");

            # 3.7: Binning (Odd_D_Cat)
            self.log(" -> 3.7 Calculando Binning..."); df_p=calculate_binned_features(df_p);
            if 'Odd_D_Cat' not in df_p.columns: logger.warning("'Odd_D_Cat' ausente pós-binning.");

            # 3.8: Derivadas (CV_HDA, Diff_Media_CG)
            self.log(" -> 3.8 Calculando Derivadas..."); df_p=calculate_derived_features(df_p);
            if 'CV_HDA' not in df_p.columns: logger.warning("'CV_HDA' ausente pós-derivadas.");


            self.df_historical_processed = df_p.copy(); # Store processed data
            self.log("Processamento de features concluído.");
            logger.info("=== FIM PIPELINE DE CÁLCULO FEATURES (ANALYSIS TAB) ===")
            logger.debug(f"Colunas após processamento completo: {list(self.df_historical_processed.columns)}")
            # Log NaN counts after full processing
            nan_counts_processed = self.df_historical_processed.isnull().sum()
            nan_counts_processed = nan_counts_processed[nan_counts_processed > 0]
            if not nan_counts_processed.empty:
                 logger.info(f"Contagem NaNs APÓS processamento completo:\n{nan_counts_processed}")
            else:
                 logger.info("Nenhum NaN detectado após processamento completo.")


            # --- PASSO 4: Preparar X, y para Análise ---
            self.log("Etapa 4: Preparando X, y para análise...");
            target_col='IsDraw';
            if target_col not in self.df_historical_processed.columns:
                 logger.error(f"Coluna alvo '{target_col}' não encontrada no DF processado!");
                 self._update_text_widget(self.target_text, "Erro: Alvo não encontrado.");
                 self._update_text_widget(self.importance_text, "Erro: Alvo não encontrado.");
                 self._update_text_widget(self.corr_text, "Erro: Alvo não encontrado.");
                 return

            y_raw = self.df_historical_processed[target_col];

            # Use FEATURE_COLUMNS from config
            features_to_analyze_config = FEATURE_COLUMNS
            self.log(f"Usando lista de features da config: {features_to_analyze_config}")

            features_present = [f for f in features_to_analyze_config if f in self.df_historical_processed.columns]
            missing_in_df = [f for f in features_to_analyze_config if f not in self.df_historical_processed.columns]

            if not features_present:
                 errmsg = "Erro: Nenhuma das features da lista FEATURE_COLUMNS foi encontrada/calculada no DF processado."
                 self.log(errmsg); messagebox.showerror("Erro Análise", errmsg, parent=self.parent); return;
            if missing_in_df:
                 self.log(f"AVISO: Features da config não encontradas/calculadas: {missing_in_df}. Não serão usadas na análise.")

            X_raw = self.df_historical_processed[features_present].copy();
            self.log(f"Shape de X (features selecionadas) ANTES do dropna: {X_raw.shape}")
            self.log(f"Shape de y ANTES do dropna: {y_raw.shape}")

            # Combine X and y for consistent dropping
            analysis_df = X_raw.join(y_raw);
            initial_rows = len(analysis_df);

            # Log NaNs/Infs BEFORE final dropna
            nan_check_before = analysis_df[features_present].isnull().sum()
            cols_with_nan_before = nan_check_before[nan_check_before > 0]
            if not cols_with_nan_before.empty:
                self.log(f"NaNs ANTES do dropna final (em {len(cols_with_nan_before)} cols das features selecionadas):\n{cols_with_nan_before}")

            inf_check_before = analysis_df[features_present].select_dtypes(include=np.number).isin([np.inf, -np.inf]).sum()
            cols_with_inf_before = inf_check_before[inf_check_before > 0]
            if not cols_with_inf_before.empty:
                self.log(f"INFINITOS ANTES do dropna final (em {len(cols_with_inf_before)} cols das features selecionadas):\n{cols_with_inf_before}")


            # --- FINAL DROP NA (on selected features + target) ---
            self.log(f"Executando dropna final baseado em {features_present + [target_col]}")
            analysis_df.dropna(subset=(features_present + [target_col]), inplace=True);
            rows_dropped = initial_rows - len(analysis_df);
            self.log(f"Dados para análise (shape final): {analysis_df.shape} ({rows_dropped} linhas removidas por NaNs/Infs em features selecionadas ou alvo).");

            if analysis_df.empty:
                 errmsg = "Erro: Nenhum dado restou para análise após remover linhas com NaNs/Infs nas features selecionadas ou no alvo."
                 self.log(errmsg); messagebox.showerror("Erro Análise", errmsg, parent=self.parent);
                 self._update_text_widget(self.desc_text, errmsg);
                 self._update_text_widget(self.target_text, errmsg);
                 self._update_text_widget(self.importance_text, errmsg);
                 self._update_text_widget(self.corr_text, errmsg);
                 return;

            # Separate clean X and y
            X_clean = analysis_df[features_present].copy();
            y_clean = analysis_df[target_col].astype(int); # Ensure target is int

            # Final check for NaN/Inf in clean data (should not happen)
            if X_clean.isnull().values.any() or X_clean.isin([np.inf, -np.inf]).values.any():
                 logger.error("ERRO CRÍTICO: NaNs ou Infs encontrados em X_clean APÓS dropna!"); return
            if y_clean.isnull().values.any():
                 logger.error("ERRO CRÍTICO: NaNs encontrados em y_clean APÓS dropna!"); return


            # --- PASSO 5: Display Describe/Target ---
            self.log("Etapa 5: Atualizando Describe/Target...");
            try:
                 with pd.option_context('display.max_rows', 200, 'display.max_columns', None, 'display.width', 1000):
                      self._update_text_widget(self.desc_text, X_clean.describe(include='all').to_string()); # include='all'
            except Exception as e_desc:
                 self.log(f"Erro ao gerar describe: {e_desc}")
                 self._update_text_widget(self.desc_text, f"Erro: {e_desc}")

            try:
                counts=y_clean.value_counts(); dist=y_clean.value_counts(normalize=True);
                self._update_text_widget(self.target_text, f"Contagem:\n{counts.to_string()}\n\nProporção:\n{dist.apply('{:.2%}'.format).to_string()}");
            except Exception as e_target:
                 self.log(f"Erro ao gerar distribuição do alvo: {e_target}")
                 self._update_text_widget(self.target_text, f"Erro: {e_target}")


            # --- PASSO 6: Análise Avançada (Importância/Correlação) ---
            self.log("Etapa 6: Executando Análise Avançada (Importância/Correlação)...");
            try:
                self.log(f"Chamando analyze_features com X_clean ({X_clean.shape}) e y_clean ({y_clean.shape})...");
                # Chama a função com logs internos (e agora mais robusta contra inf)
                analysis_results: Optional[Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]]
                analysis_results = analyze_features(X_clean, y_clean) # Pass clean data
                self.log("analyze_features retornou.")

                if analysis_results is not None:
                    imp_df, corr_df = analysis_results # Desempacota o resultado

                    # Verifica e exibe Importância
                    if isinstance(imp_df, pd.DataFrame) and not imp_df.empty:
                        self.log("Resultado Importância recebido e válido.")
                        self._update_text_widget(self.importance_text, imp_df.round(4).to_string(index=False))
                    else:
                        msg = "Importância não calculada ou resultado vazio (verifique logs de model_trainer)."
                        self.log(msg)
                        self._update_text_widget(self.importance_text, msg)

                    # Verifica e exibe Correlação
                    if isinstance(corr_df, pd.DataFrame) and not corr_df.empty:
                        self.log("Resultado Correlação recebido e válido.")
                        target_corr = corr_df[['target_IsDraw']].sort_values(by='target_IsDraw', ascending=False)
                        with pd.option_context('display.float_format','{:.3f}'.format): # 3 casas decimais
                             self._update_text_widget(self.corr_text, target_corr.to_string())
                    else:
                         msg = "Correlação não calculada ou resultado vazio (verifique logs de model_trainer)."
                         self.log(msg)
                         self._update_text_widget(self.corr_text, msg)
                else:
                    # Se a própria tupla retornada for None
                    self.log("ERRO INESPERADO: analyze_features retornou None.")
                    self._update_text_widget(self.importance_text, "Erro interno na análise (retorno None).")
                    self._update_text_widget(self.corr_text, "Erro interno na análise (retorno None).")

            except Exception as e_adv_analyze:
                self.log(f"!!! ERRO GRAVE durante chamada/processamento de analyze_features: {e_adv_analyze}")
                logger.error(f"Erro fatal em analyze_features: {e_adv_analyze}", exc_info=True) # Log com traceback
                messagebox.showerror("Erro Análise", f"Erro calcular importância/correlação:\n{e_adv_analyze}", parent=self.parent)
                self._update_text_widget(self.importance_text, f"Erro: {e_adv_analyze}")
                self._update_text_widget(self.corr_text, f"Erro: {e_adv_analyze}")

            self.log("Análise avançada concluida.");
        except Exception as e:
            self.log(f"!!! ERRO GRAVE durante carregamento/análise: {e}")
            logger.error(f"Erro fatal em load_and_display_data: {e}", exc_info=True) # Log com traceback
            messagebox.showerror("Erro Fatal", f"Ocorreu um erro inesperado:\n{e}\n\nVerifique os logs.", parent=self.parent)
            # Limpa widgets em caso de erro grave
            widgets_to_clear = [self.info_text, self.head_text, self.desc_text, self.target_text, self.importance_text, self.corr_text]
            for w in widgets_to_clear: self._update_text_widget(w, f"Erro: {e}")

# --- analyze_features in model_trainer.py (Make more robust) ---
# This function is called by the FeatureAnalyzerApp

# Add this import at the top of model_trainer.py
# import numpy as np

def analyze_features(X: pd.DataFrame, y: pd.Series) -> Optional[Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]]:
    """
    Analisa features: importância (RF rápido) e correlação.
    Retorna (imp_df, corr_df) ou (None, None). Mais robusto a NaNs/Infs.
    """
    logger.info("--- ANÁLISE FEATURES (model_trainer): Iniciando ---")
    imp_df = None
    corr_df = None # Use corr_df instead of corr_matrix to be consistent

    # --- Input Validation ---
    if X is None or y is None:
        logger.error("ANÁLISE FEATURES: Input X ou y é None.")
        return None, None
    if X.empty or y.empty:
        logger.error("ANÁLISE FEATURES: Input X ou y está vazio.")
        return None, None
    if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
         logger.error("ANÁLISE FEATURES: Input X ou y não é do tipo esperado (DataFrame, Series).")
         return None, None

    # --- Alignment (Crucial!) ---
    if not X.index.equals(y.index):
        logger.warning("ANÁLISE FEATURES: Índices X/y não idênticos. Tentando alinhar...")
        try:
            common_index = X.index.intersection(y.index)
            if len(common_index) == 0:
                 logger.error("ANÁLISE FEATURES: Nenhum índice em comum entre X e y após intersection.")
                 return None, None
            X = X.loc[common_index]
            y = y.loc[common_index]
            logger.info(f"ANÁLISE FEATURES: Alinhamento OK. Novo shape X: {X.shape}, y: {y.shape}")
        except Exception as e:
            logger.error(f"ANÁLISE FEATURES: Erro durante alinhamento: {e}", exc_info=True)
            return None, None

    # --- Final check for data after alignment ---
    if X.empty or y.empty:
        logger.error("ANÁLISE FEATURES: X ou y vazio após alinhamento.")
        return None, None

    feature_names = X.columns.tolist()

    # --- 1. Calcular Importância (RF) ---
    logger.info("ANÁLISE FEATURES: Calculando importância RF...")
    try:
        # Check for NaNs/Infs *before* fitting RF
        if X.isnull().values.any():
             nan_cols = X.columns[X.isnull().any()].tolist()
             logger.warning(f"ANÁLISE FEATURES (RF): NaNs encontrados em X (colunas: {nan_cols}). RF pode falhar ou ser impreciso.")
             # OPTIONAL: Impute or drop NaNs here if RF cannot handle them, but dropna should have happened before
             # X_rf = X.fillna(X.median()) # Example imputation
             X_rf = X # Assume RF handles NaNs or prior dropna was sufficient
        else:
             X_rf = X

        if not np.all(np.isfinite(X_rf.select_dtypes(include=np.number).values)):
             logger.warning("ANÁLISE FEATURES (RF): Valores não finitos (inf?) encontrados em X numérico. RF pode falhar.")
             # Replace inf with NaN, then handle NaN if needed
             X_rf = X_rf.replace([np.inf, -np.inf], np.nan)
             # Re-check for NaNs if inf was replaced
             if X_rf.isnull().values.any():
                 logger.warning("ANÁLISE FEATURES (RF): NaNs presentes após substituir inf.")
                 # Apply imputation again if desired
                 # X_rf.fillna(X_rf.median(), inplace=True)


        if y.isnull().values.any():
             logger.error("ANÁLISE FEATURES (RF): NaNs encontrados em y! Não pode treinar RF.")
             # Cannot proceed with importance calculation if target has NaNs
             raise ValueError("Target variable (y) contains NaNs.")


        # Ensure y is integer type for classification
        y_rf = y.astype(int)

        rf_analyzer = RandomForestClassifier(n_estimators=50, max_depth=15, n_jobs=-1, min_samples_leaf=3, random_state=RANDOM_STATE)
        logger.info(f"    -> Fitting RF (X shape: {X_rf.shape}, y shape: {y_rf.shape})")
        rf_analyzer.fit(X_rf, y_rf) # Use potentially cleaned X_rf and integer y_rf
        logger.info("    -> Fit RF concluído.")

        importances = rf_analyzer.feature_importances_
        imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        imp_df = imp_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        logger.info(f"ANÁLISE FEATURES: Importância calculada OK. Shape: {imp_df.shape}")

    except Exception as e:
        logger.error(f"ANÁLISE FEATURES: Erro GRAVE ao calcular importância RF: {e}", exc_info=True)
        imp_df = None # Set to None on failure

    # --- 2. Calcular Correlação ---
    logger.info("ANÁLISE FEATURES: Calculando correlação...")
    try:
        df_temp = X.copy() # Start with the aligned X
        df_temp['target_IsDraw'] = y # Add the aligned y

        # Select only numeric columns for correlation calculation
        numeric_cols = df_temp.select_dtypes(include=np.number).columns
        if 'target_IsDraw' not in numeric_cols:
             logger.warning("ANÁLISE FEATURES (Corr): Coluna alvo não é numérica? Incluindo manualmente.")
             numeric_cols = numeric_cols.tolist() + ['target_IsDraw'] # Ensure target is included

        df_numeric_temp = df_temp[numeric_cols]

        # Check and handle infinities *before* calculating correlation
        if df_numeric_temp.isin([np.inf, -np.inf]).values.any():
            inf_cols = df_numeric_temp.columns[df_numeric_temp.isin([np.inf, -np.inf]).any()].tolist()
            logger.warning(f"ANÁLISE FEATURES (Corr): Valores infinitos encontrados antes de .corr() (colunas: {inf_cols}). Substituindo por NaN.")
            df_numeric_temp = df_numeric_temp.replace([np.inf, -np.inf], np.nan)

        # Check for columns with all NaNs after handling inf (corr fails on these)
        all_nan_cols = df_numeric_temp.columns[df_numeric_temp.isnull().all()].tolist()
        if all_nan_cols:
             logger.warning(f"ANÁLISE FEATURES (Corr): Colunas inteiras com NaN encontradas: {all_nan_cols}. Serão excluídas da correlação.")
             df_numeric_temp = df_numeric_temp.drop(columns=all_nan_cols)
             if 'target_IsDraw' not in df_numeric_temp.columns:
                  logger.error("ANÁLISE FEATURES (Corr): Coluna alvo foi removida (toda NaN?). Não é possível calcular correlação com o alvo.")
                  corr_df = None # Set corr_df to None explicitly
                  raise ValueError("Target column removed due to all NaNs.") # Stop correlation part


        logger.info(f"    -> Calculando corr() em df_numeric_temp (shape: {df_numeric_temp.shape})")
        # Calculate correlation only on the numeric (and finite) data
        corr_matrix = df_numeric_temp.corr() # numeric_only=True is implicit here

        # Extract only the correlation with the target variable
        if 'target_IsDraw' in corr_matrix.columns:
            corr_df = corr_matrix[['target_IsDraw']].sort_values(by='target_IsDraw', ascending=False)
            logger.info(f"ANÁLISE FEATURES: Correlação com o alvo calculada OK. Shape: {corr_df.shape}")
        else:
             logger.error("ANÁLISE FEATURES (Corr): Coluna 'target_IsDraw' não encontrada na matriz de correlação final.")
             corr_df = None


    except Exception as e:
        logger.error(f"ANÁLISE FEATURES: Erro GRAVE ao calcular correlação: {e}", exc_info=True)
        corr_df = None # Set to None on failure

    logger.info("--- ANÁLISE FEATURES (model_trainer): Concluída ---")
    return imp_df, corr_df # Return results (can be None)