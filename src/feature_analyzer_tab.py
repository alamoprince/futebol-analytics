# --- src/feature_analyzer_tab.py ---

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText
import sys
import os
import pandas as pd
import numpy as np
import time # Necessário para sleep, embora não usado diretamente aqui, mas no padrão de parada
import datetime
import io
import traceback
import warnings
import threading # Para SHAP em background
import queue     # Para comunicação GUI-Thread
from typing import Optional, Dict, Tuple, List, Any


# --- Gráficos ---
import matplotlib
matplotlib.use('TkAgg') # Configura backend para Tkinter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns

# --- Modelos e Interpretabilidade ---
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError: lgb = None; LGBM_AVAILABLE = False
# SHAP
try:
    with warnings.catch_warnings(): # Suprime avisos comuns
        warnings.simplefilter("ignore")
        import shap
    SHAP_AVAILABLE = True
except ImportError: shap = None; SHAP_AVAILABLE = False
# PDP/ICE
from sklearn.inspection import PartialDependenceDisplay

# --- Configurar Path ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path: sys.path.insert(0, CURRENT_DIR)
if BASE_DIR not in sys.path: sys.path.insert(0, BASE_DIR)

# --- Importar Módulos do Projeto ---
try:
    from config import (FEATURE_COLUMNS, GOALS_COLS, RANDOM_STATE,
                        BEST_F1_MODEL_SAVE_PATH, BEST_ROI_MODEL_SAVE_PATH,
                        MODEL_ID_F1, MODEL_ID_ROI)
    from data_handler import (load_historical_data, calculate_historical_intermediate,
                              calculate_probabilities, calculate_normalized_probabilities,
                              calculate_rolling_stats, calculate_rolling_std,
                              calculate_binned_features, calculate_derived_features,
                              calculate_rolling_goal_stats, calculate_poisson_draw_prob,
                              calculate_pi_ratings)
    from logger_config import setup_logger
    from predictor import load_model_scaler_features

except ImportError as e:
     import logging # Fallback logger
     logger = logging.getLogger(__name__)
     logger.critical(f"Import Error CRÍTICO em feature_analyzer_tab.py: {e}", exc_info=True)
     try: root_err = tk.Tk(); root_err.withdraw(); messagebox.showerror("Import Error (Analyzer Tab)", f"Failed...\n{e}"); root_err.destroy()
     except Exception: print(f"CRITICAL Import Error: {e}")
     sys.exit(1)

logger = setup_logger("FeatureAnalyzerApp")
if not SHAP_AVAILABLE: logger.warning("Biblioteca 'shap' não encontrada. Gráficos SHAP não disponíveis.")


class FeatureAnalyzerApp:

    def __init__(self, parent_frame, main_root):
        self.parent = parent_frame
        self.main_tk_root = main_root # Referência à janela principal para .after()
        self.gui_queue = queue.Queue()
        self.shap_thread: Optional[threading.Thread] = None

        # DataFrames
        self.df_historical_raw: Optional[pd.DataFrame] = None
        self.df_historical_processed: Optional[pd.DataFrame] = None
        self.X_clean: Optional[pd.DataFrame] = None
        self.y_clean: Optional[pd.Series] = None
        self.current_data_identifier = None # Para validar cache SHAP

        # Widgets de Texto
        self.info_text: Optional[tk.Text] = None
        self.head_text: Optional[tk.Text] = None
        self.desc_text: Optional[tk.Text] = None
        self.target_text: Optional[tk.Text] = None
        self.model_importance_text: Optional[tk.Text] = None
        self.status_label: Optional[ttk.Label] = None

        # Widgets de Seleção
        self.feature_selector_var = tk.StringVar()
        self.feature_selector_combo: Optional[ttk.Combobox] = None
        self.model_selector_var = tk.StringVar()
        self.model_selector_combo: Optional[ttk.Combobox] = None
        self.pdp_feature_selector_var = tk.StringVar()
        self.pdp_feature_selector_combo: Optional[ttk.Combobox] = None
        self.load_trained_models_button: Optional[ttk.Button] = None

        # Containers e Figuras Matplotlib
        self.dist_canvas_widget: Optional[tk.Widget] = None
        self.dist_target_canvas_widget: Optional[tk.Widget] = None
        self.corr_heatmap_canvas_widget: Optional[tk.Widget] = None
        self.shap_summary_canvas_widget: Optional[tk.Widget] = None
        self.pdp_canvas_widget: Optional[tk.Widget] = None
        self.fig_dist: Optional[plt.Figure] = None
        self.fig_dist_target: Optional[plt.Figure] = None
        self.fig_corr: Optional[plt.Figure] = None
        self.fig_shap_summary: Optional[plt.Figure] = None
        self.fig_pdp: Optional[plt.Figure] = None
        self.dist_canvas_container: Optional[ttk.Frame] = None
        self.dist_target_canvas_container: Optional[ttk.Frame] = None
        self.corr_canvas_container: Optional[ttk.Frame] = None
        self.shap_summary_container: Optional[ttk.Frame] = None
        self.pdp_container: Optional[ttk.Frame] = None
        self.model_specific_frame: Optional[ttk.LabelFrame] = None

        # Modelos Carregados e Controle de Fila
        self.loaded_trained_models: Dict[str, Dict] = {}
        self.stop_processing_queue = False # Flag para parada segura da fila

        self.create_widgets()
        self.process_gui_queue() # Inicia o loop da fila da GUI

    def create_widgets(self):
        """Cria a estrutura da GUI com scroll e placeholders."""
        # --- Top Controls (Load Data Button, Status Label) ---
        top_controls_frame = ttk.Frame(self.parent)
        top_controls_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10, 0))
        load_button = ttk.Button(top_controls_frame, text="Carregar & Analisar Dados Históricos", command=self.load_and_display_data); load_button.pack(side=tk.LEFT, padx=(0, 10))
        self.status_label = ttk.Label(top_controls_frame, text="Pronto."); self.status_label.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

        # --- Canvas & Scrollbar ---
        canvas = tk.Canvas(self.parent, borderwidth=0, highlightthickness=0); scrollbar = ttk.Scrollbar(self.parent, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas); self.scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw"); canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=(5, 10)); scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=(5, 10))

        # --- Seções no Scrollable Frame ---
        # 1. Visão Geral
        overview_frame = ttk.LabelFrame(self.scrollable_frame, text=" Visão Geral Dados Brutos ", padding=10); overview_frame.pack(fill=tk.X, padx=10, pady=5)
        self.info_text = self._create_scrolled_text(overview_frame, height=10, title="Infos Gerais (df.info)"); self.head_text = self._create_scrolled_text(overview_frame, height=7, title="Amostra Dados Raw (df.head)")
        # 2. Alvo e Describe
        target_desc_frame = ttk.LabelFrame(self.scrollable_frame, text=" Alvo e Descrição (Pós-Processamento) ", padding=10); target_desc_frame.pack(fill=tk.X, padx=10, pady=5)
        self.desc_text = self._create_scrolled_text(target_desc_frame, height=12, title="Describe (Features Limpas)"); self.target_text = self._create_scrolled_text(target_desc_frame, height=5, title="Distribuição Alvo (IsDraw)")
        # 3. Análise Univariada
        univar_frame = ttk.LabelFrame(self.scrollable_frame, text=" Análise Univariada ", padding=10); univar_frame.pack(fill=tk.X, padx=10, pady=5)
        selector_frame = ttk.Frame(univar_frame); selector_frame.pack(fill=tk.X, pady=(0, 10)); ttk.Label(selector_frame, text="Feature:").pack(side=tk.LEFT, padx=(0, 5))
        self.feature_selector_combo = ttk.Combobox(selector_frame, textvariable=self.feature_selector_var, state="readonly", width=30); self.feature_selector_combo.pack(side=tk.LEFT); self.feature_selector_combo.bind("<<ComboboxSelected>>", self.update_univariate_plots)
        plot_frame_univar = ttk.Frame(univar_frame); plot_frame_univar.pack(fill=tk.BOTH, expand=True)
        self.dist_canvas_container = ttk.LabelFrame(plot_frame_univar, text=" Distribuição Feature ", padding=5); self.dist_canvas_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5); ttk.Label(self.dist_canvas_container, text="Carregue dados...").pack()
        self.dist_target_canvas_container = ttk.LabelFrame(plot_frame_univar, text=" Distribuição vs Alvo ", padding=5); self.dist_target_canvas_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5); ttk.Label(self.dist_target_canvas_container, text="Carregue dados...").pack()
        # 4. Análise Multivariada
        multivar_frame = ttk.LabelFrame(self.scrollable_frame, text=" Análise Multivariada ", padding=10); multivar_frame.pack(fill=tk.X, padx=10, pady=5)
        self.corr_canvas_container = ttk.LabelFrame(multivar_frame, text=" Heatmap Correlação ", padding=5); self.corr_canvas_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5); ttk.Label(self.corr_canvas_container, text="Carregue dados...").pack()
        # 5. Análise Específica do Modelo
        self.model_specific_frame = ttk.LabelFrame(self.scrollable_frame, text=" Análise Específica do Modelo ", padding=10); self.model_specific_frame.pack(fill=tk.X, padx=10, pady=5)
        model_controls_frame = ttk.Frame(self.model_specific_frame); model_controls_frame.pack(fill=tk.X, pady=(0, 5))
        self.load_trained_models_button = ttk.Button(model_controls_frame, text="Carregar Modelos Treinados", command=self.load_trained_models); self.load_trained_models_button.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(model_controls_frame, text="Analisar Modelo:").pack(side=tk.LEFT, padx=(10, 5)); self.model_selector_combo = ttk.Combobox(model_controls_frame, textvariable=self.model_selector_var, state="disabled", width=25); self.model_selector_combo.pack(side=tk.LEFT); self.model_selector_combo.bind("<<ComboboxSelected>>", self.update_model_specific_analysis)
        importance_frame = ttk.LabelFrame(self.model_specific_frame, text=" Importância de Features (Modelo Selecionado) ", padding=5); importance_frame.pack(fill=tk.X, pady=5)
        self.model_importance_text = self._create_scrolled_text(importance_frame, height=12); self._update_text_widget(self.model_importance_text, "Carregue modelos treinados...")
        model_plots_frame = ttk.Frame(self.model_specific_frame); model_plots_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.shap_summary_container = ttk.LabelFrame(model_plots_frame, text=" SHAP Summary Plot ", padding=5); self.shap_summary_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5); ttk.Label(self.shap_summary_container, text="Selecione modelo...").pack()
        if not SHAP_AVAILABLE: ttk.Label(self.shap_summary_container, text="(Biblioteca 'shap' não instalada)", foreground="red").pack()
        pdp_outer_container = ttk.LabelFrame(model_plots_frame, text=" Partial Dependence Plot (PDP) ", padding=5); pdp_outer_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        pdp_controls_frame = ttk.Frame(pdp_outer_container); pdp_controls_frame.pack(fill=tk.X, pady=(0, 5)); ttk.Label(pdp_controls_frame, text="Feature para PDP:").pack(side=tk.LEFT, padx=(0, 5))
        self.pdp_feature_selector_combo = ttk.Combobox(pdp_controls_frame, textvariable=self.pdp_feature_selector_var, state="disabled", width=25); self.pdp_feature_selector_combo.pack(side=tk.LEFT); self.pdp_feature_selector_combo.bind("<<ComboboxSelected>>", self.update_pdp_plot)
        self.pdp_container = ttk.Frame(pdp_outer_container); self.pdp_container.pack(fill=tk.BOTH, expand=True); ttk.Label(self.pdp_container, text="Selecione modelo/feature...").pack()

    def _create_scrolled_text(self, parent, height, title=""):
        outer_frame = ttk.LabelFrame(parent, text=f" {title} ", padding=5); outer_frame.pack(side=tk.TOP, fill=tk.X, pady=5, padx=5)
        text_area_frame = ttk.Frame(outer_frame); text_area_frame.pack(fill=tk.BOTH, expand=True)
        xscrollbar = ttk.Scrollbar(text_area_frame, orient=tk.HORIZONTAL); yscrollbar = ttk.Scrollbar(text_area_frame, orient=tk.VERTICAL)
        widget = tk.Text(text_area_frame, height=height, wrap=tk.NONE, font=("Consolas", 9), relief=tk.FLAT, bd=0, yscrollcommand=yscrollbar.set, xscrollcommand=xscrollbar.set, state='disabled')
        yscrollbar.config(command=widget.yview); xscrollbar.config(command=widget.xview)
        text_area_frame.grid_rowconfigure(0, weight=1); text_area_frame.grid_columnconfigure(0, weight=1)
        widget.grid(row=0, column=0, sticky="nsew"); yscrollbar.grid(row=0, column=1, sticky="ns"); xscrollbar.grid(row=1, column=0, sticky="ew")
        return widget

    def _update_text_widget(self, text_widget: Optional[tk.Text], content: str):
        if text_widget is None: return
        try:
            if text_widget.winfo_exists(): text_widget.config(state='normal'); text_widget.delete('1.0', tk.END); text_widget.insert('1.0', content); text_widget.config(state='disabled')
        except tk.TclError: pass
        except Exception as e: logger.error(f"Erro update text widget: {e}", exc_info=True)

    def _clear_matplotlib_widget(self, container_widget):
        if container_widget:
            for widget in container_widget.winfo_children(): widget.destroy()

    def _embed_matplotlib_figure(self, fig: plt.Figure, container_widget: ttk.Frame):
        self._clear_matplotlib_widget(container_widget)
        try:
            canvas = FigureCanvasTkAgg(fig, master=container_widget); canvas.draw()
            widget = canvas.get_tk_widget(); widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            return canvas
        except Exception as e: logger.error(f"Erro embed matplotlib: {e}", exc_info=True); ttk.Label(container_widget, text=f"Erro gráfico:\n{e}").pack(); return None

    def log(self, message: str):
        logger.info(f"[GUI Analyzer] {message}")
        try:
            if self.status_label and self.status_label.winfo_exists(): self.status_label.config(text=message[:100])
        except tk.TclError: pass
        except Exception as e: logger.error(f"Erro status label: {e}", exc_info=True)

    def load_and_display_data(self):
        self.log("Iniciando: Carregando & Processando dados históricos...")
        widgets_to_clear = [self.info_text, self.head_text, self.desc_text, self.target_text, self.model_importance_text]
        for w in widgets_to_clear: self._update_text_widget(w, "Carregando...")
        self._clear_matplotlib_widget(self.dist_canvas_container); ttk.Label(self.dist_canvas_container, text="...").pack()
        self._clear_matplotlib_widget(self.dist_target_canvas_container); ttk.Label(self.dist_target_canvas_container, text="...").pack()
        self._clear_matplotlib_widget(self.corr_canvas_container); ttk.Label(self.corr_canvas_container, text="...").pack()
        self._clear_matplotlib_widget(self.shap_summary_container); ttk.Label(self.shap_summary_container, text="Carregue dados/modelos...").pack()
        self._clear_matplotlib_widget(self.pdp_container); ttk.Label(self.pdp_container, text="Carregue dados/modelos...").pack()
        self.loaded_trained_models = {}; self.model_selector_var.set(""); self.pdp_feature_selector_var.set("")
        try:
            if self.model_selector_combo: self.model_selector_combo.config(values=[], state="disabled")
            if self.pdp_feature_selector_combo: self.pdp_feature_selector_combo.config(values=[], state="disabled")
        except tk.TclError: pass
        self._update_text_widget(self.model_importance_text, "Carregue dados e modelos...")
        self.df_historical_raw = None; self.df_historical_processed = None; self.X_clean = None; self.y_clean = None; self.current_data_identifier = None

        try:
            self.log("Etapa 1: Carregando dados brutos...")
            df_raw = load_historical_data()
            if df_raw is None or df_raw.empty: raise ValueError("Falha ao carregar dados históricos.")
            self.df_historical_raw = df_raw.copy()
            self.log(f"Dados brutos carregados: {self.df_historical_raw.shape}")
            buffer = io.StringIO(); self.df_historical_raw.info(buf=buffer); self._update_text_widget(self.info_text, buffer.getvalue())
            with pd.option_context('display.max_rows', 10, 'display.max_columns', None, 'display.width', 1000): self._update_text_widget(self.head_text, self.df_historical_raw.head(5).to_string());

            self.log("Etapa 2: Processando features (Pipeline Completo)...");
            df_p = self.df_historical_raw.copy(); logger.info("=== INÍCIO PIPELINE FEAT ENG (ANALYSIS TAB) ===")
            goals_h_col=GOALS_COLS.get('home'); goals_a_col=GOALS_COLS.get('away')
            avg_h_league=np.nanmean(df_p[goals_h_col]) if goals_h_col in df_p else 1.0; avg_a_league=np.nanmean(df_p[goals_a_col]) if goals_a_col in df_p else 1.0
            avg_h_league=1.0 if pd.isna(avg_h_league) else avg_h_league; avg_a_league=1.0 if pd.isna(avg_a_league) else avg_a_league
            epsilon=1e-6; avg_h_league_safe=max(avg_h_league,epsilon); avg_a_league_safe=max(avg_a_league,epsilon)
            self.log(f"Médias Liga: H={avg_h_league_safe:.3f}, A={avg_a_league_safe:.3f}")
            df_p=calculate_historical_intermediate(df_p);
            if 'IsDraw' not in df_p.columns or df_p['IsDraw'].isnull().all(): raise ValueError("Alvo 'IsDraw' ausente/NaN.");
            df_p=calculate_probabilities(df_p); df_p=calculate_normalized_probabilities(df_p); df_p=calculate_pi_ratings(df_p)
            df_p=calculate_rolling_stats(df_p,['VG','CG']); df_p=calculate_rolling_std(df_p,['CG']);
            df_p=calculate_rolling_goal_stats(df_p, avg_goals_home_league=avg_h_league_safe, avg_goals_away_league=avg_a_league_safe)
            df_p=calculate_poisson_draw_prob(df_p, avg_goals_home_league=avg_h_league_safe, avg_goals_away_league=avg_a_league_safe);
            df_p=calculate_binned_features(df_p); df_p=calculate_derived_features(df_p);
            self.df_historical_processed = df_p.copy(); self.log("Processamento features concluído."); logger.info("=== FIM PIPELINE FEAT ENG ===")

            self.log("Etapa 3: Preparando X, y para análise...");
            target_col='IsDraw';
            if target_col not in self.df_historical_processed.columns: raise ValueError(f"Coluna alvo '{target_col}' não encontrada!");
            features_to_analyze=self.df_historical_processed.select_dtypes(include=np.number).columns.tolist()
            cols_to_exclude=list(GOALS_COLS.values())+[target_col,'Ptos_H','Ptos_A']
            features_to_analyze=[f for f in features_to_analyze if f not in cols_to_exclude]
            self.log(f"Features numéricas para análise: {len(features_to_analyze)}")
            features_present=[f for f in features_to_analyze if f in self.df_historical_processed.columns]
            if not features_present: raise ValueError("Nenhuma feature encontrada.")
            X_raw=self.df_historical_processed[features_present].copy(); y_raw=self.df_historical_processed[target_col];
            analysis_df=X_raw.join(y_raw); initial_rows=len(analysis_df)
            analysis_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            analysis_df.dropna(subset=(features_present + [target_col]), inplace=True);
            rows_dropped=initial_rows - len(analysis_df)
            self.log(f"Dados p/ análise (final): {analysis_df.shape} ({rows_dropped} linhas removidas).");
            if analysis_df.empty: raise ValueError("Nenhum dado restou após limpeza.")
            self.X_clean=analysis_df[features_present].copy(); self.y_clean=analysis_df[target_col].astype(int);
            if self.X_clean.isnull().values.any() or self.X_clean.isin([np.inf, -np.inf]).values.any(): raise ValueError("NaNs/Infs persistentes em X_clean")
            if self.y_clean.isnull().values.any(): raise ValueError("NaNs persistentes em y_clean")
            self.current_data_identifier = (self.X_clean.shape, tuple(self.X_clean.columns)); self.log(f"Identificador de dados atualizado: {self.current_data_identifier}")

            self.log("Etapa 4: Atualizando Describe/Target...");
            try:
                 with pd.option_context('display.max_rows', 200, 'display.max_columns', None, 'display.width', 1000): self._update_text_widget(self.desc_text, self.X_clean.describe(include='all').to_string());
            except Exception as e: self.log(f"Erro describe: {e}"); self._update_text_widget(self.desc_text, f"Erro.")
            try:
                counts=self.y_clean.value_counts(); dist=self.y_clean.value_counts(normalize=True); self._update_text_widget(self.target_text, f"Contagem:\n{counts.to_string()}\n\nProporção:\n{dist.apply('{:.2%}'.format).to_string()}");
            except Exception as e: self.log(f"Erro target dist: {e}"); self._update_text_widget(self.target_text, f"Erro.")

            self.log("Etapa 5: Gerando gráficos básicos...");
            numeric_features=self.X_clean.select_dtypes(include=np.number).columns.tolist()
            if numeric_features: self.feature_selector_combo['values']=numeric_features; self.feature_selector_var.set(numeric_features[0] if numeric_features else ""); self.update_univariate_plots()
            else: self.feature_selector_combo['values']=[]; self.feature_selector_var.set(""); self.log("Nenhuma feature numérica.")
            self.generate_correlation_heatmap()
            self.log("Carregamento e análise básica concluídos. Use 'Carregar Modelos Treinados'.");

        except Exception as e:
            errmsg = f"Erro fatal no carregamento/análise: {e}"; self.log(f"!!! ERRO: {errmsg}"); logger.error(errmsg, exc_info=True); messagebox.showerror("Erro Fatal na Análise", f"{errmsg}\n\nVer logs.", parent=self.parent)
            widgets_to_clear=[self.info_text, self.head_text, self.desc_text, self.target_text, self.model_importance_text]
            for w in widgets_to_clear: self._update_text_widget(w, f"Erro: {e}")
            self._clear_matplotlib_widget(self.dist_canvas_container); ttk.Label(self.dist_canvas_container, text="Erro.").pack(); self._clear_matplotlib_widget(self.dist_target_canvas_container); ttk.Label(self.dist_target_canvas_container, text="Erro.").pack(); self._clear_matplotlib_widget(self.corr_canvas_container); ttk.Label(self.corr_canvas_container, text="Erro.").pack(); self._clear_matplotlib_widget(self.shap_summary_container); ttk.Label(self.shap_summary_container, text="Erro.").pack(); self._clear_matplotlib_widget(self.pdp_container); ttk.Label(self.pdp_container, text="Erro.").pack()
            try:
                 if self.model_selector_combo: self.model_selector_combo.config(values=[], state="disabled")
                 if self.pdp_feature_selector_combo: self.pdp_feature_selector_combo.config(values=[], state="disabled")
            except tk.TclError: pass

    def update_univariate_plots(self, event=None):
        if self.X_clean is None or self.y_clean is None: return
        selected_feature = self.feature_selector_var.get()
        if not selected_feature or selected_feature not in self.X_clean.columns: return
        try:
            if self.fig_dist: plt.close(self.fig_dist)
            self.fig_dist, axes = plt.subplots(2, 1, figsize=(6, 5), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
            feature_data = self.X_clean[selected_feature].dropna()
            sns.histplot(feature_data, kde=True, ax=axes[0]); axes[0].set_title(f'Distribuição de {selected_feature}'); axes[0].set_xlabel('')
            sns.boxplot(x=feature_data, ax=axes[1]); axes[1].set_xlabel(selected_feature)
            plt.tight_layout(); self.dist_canvas_widget = self._embed_matplotlib_figure(self.fig_dist, self.dist_canvas_container)
        except Exception as e: logger.error(f"Erro dist plot {selected_feature}: {e}", exc_info=True); self._clear_matplotlib_widget(self.dist_canvas_container); ttk.Label(self.dist_canvas_container, text=f"Erro.").pack()
        try:
            if self.fig_dist_target: plt.close(self.fig_dist_target)
            self.fig_dist_target, ax = plt.subplots(1, 1, figsize=(6, 5))
            plot_data = pd.DataFrame({'feature': self.X_clean[selected_feature], 'target': self.y_clean})
            sns.boxplot(data=plot_data, x='target', y='feature', ax=ax)
            ax.set_title(f'{selected_feature} vs IsDraw'); ax.set_xlabel('IsDraw (0: Não Empate, 1: Empate)'); ax.set_ylabel(selected_feature)
            plt.tight_layout(); self.dist_target_canvas_widget = self._embed_matplotlib_figure(self.fig_dist_target, self.dist_target_canvas_container)
        except Exception as e: logger.error(f"Erro dist vs target plot {selected_feature}: {e}", exc_info=True); self._clear_matplotlib_widget(self.dist_target_canvas_container); ttk.Label(self.dist_target_canvas_container, text=f"Erro.").pack()

    def generate_correlation_heatmap(self):
        if self.X_clean is None: return
        try:
            if self.fig_corr: plt.close(self.fig_corr)
            numeric_features = self.X_clean.select_dtypes(include=np.number)
            if numeric_features.shape[1] < 2: self._clear_matplotlib_widget(self.corr_canvas_container); ttk.Label(self.corr_canvas_container, text="Poucas features.").pack(); return
            corr_matrix = numeric_features.corr()
            self.fig_corr, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax, cbar=True)
            plt.title('Heatmap de Correlação entre Features'); plt.xticks(rotation=60, ha='right', fontsize=7); plt.yticks(rotation=0, fontsize=7); plt.tight_layout(pad=1.5)
            self.corr_heatmap_canvas_widget = self._embed_matplotlib_figure(self.fig_corr, self.corr_canvas_container)
        except Exception as e: logger.error(f"Erro heatmap: {e}", exc_info=True); self._clear_matplotlib_widget(self.corr_canvas_container); ttk.Label(self.corr_canvas_container, text=f"Erro.").pack()

    def load_trained_models(self):
        self.log("Tentando carregar modelos treinados...")
        self.loaded_trained_models = {} # Limpa modelos e cache SHAP
        loaded_ids = []; all_feature_names = set()
        model_paths_to_try = {MODEL_ID_F1: BEST_F1_MODEL_SAVE_PATH, MODEL_ID_ROI: BEST_ROI_MODEL_SAVE_PATH}
        for model_id, model_path in model_paths_to_try.items():
            self.log(f" -> Carregando {model_id}...");
            if not os.path.exists(model_path): self.log(f"    Arquivo não encontrado."); continue
            load_result = load_model_scaler_features(model_path)
            if load_result:
                model, scaler, calib, ev_thr, f1_thr, features, params, metrics, ts = load_result
                if model and features:
                    imp_df = self._calculate_feature_importance(model, features)
                    self.loaded_trained_models[model_id] = {'model': model, 'scaler': scaler, 'calibrator': calib, 'optimal_ev_threshold': ev_thr, 'optimal_f1_threshold': f1_thr, 'features': features, 'params': params, 'metrics': metrics, 'timestamp': ts, 'path': model_path, 'importances': imp_df}
                    loaded_ids.append(model_id); all_feature_names.update(features); self.log(f"    -> Modelo {model_id} carregado.")
                else: self.log(f"    -> Falha (objeto inválido).")
            else: self.log(f"    -> Falha ao carregar.")
        if loaded_ids:
            self.log(f"Modelos carregados: {', '.join(loaded_ids)}"); self.model_selector_combo.config(values=loaded_ids, state="readonly"); self.model_selector_var.set(loaded_ids[0])
            pdp_features = sorted(list(all_feature_names));
            if pdp_features: self.pdp_feature_selector_combo.config(values=pdp_features, state="readonly"); self.pdp_feature_selector_var.set(pdp_features[0] if pdp_features else "")
            else: self.pdp_feature_selector_combo.config(values=[], state="disabled"); self.pdp_feature_selector_var.set("")
            self.update_model_specific_analysis()
        else:
            self.log("Nenhum modelo treinado válido encontrado."); self.model_selector_combo.config(values=[], state="disabled"); self.model_selector_var.set("")
            self.pdp_feature_selector_combo.config(values=[], state="disabled"); self.pdp_feature_selector_var.set("")
            self._update_text_widget(self.model_importance_text, "Nenhum modelo carregado.")
            self._clear_matplotlib_widget(self.shap_summary_container); ttk.Label(self.shap_summary_container, text="Nenhum modelo.").pack()
            self._clear_matplotlib_widget(self.pdp_container); ttk.Label(self.pdp_container, text="Nenhum modelo.").pack()
            messagebox.showinfo("Modelos Não Encontrados", "Não foi possível carregar modelos.", parent=self.parent)

    def _calculate_feature_importance(self, model, feature_names: List[str]) -> Optional[pd.DataFrame]:
        importances = None
        try:
            model_type_name = model.__class__.__name__
            if hasattr(model, 'feature_importances_'): importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                 if model.coef_.shape[0] == 1: importances = np.abs(model.coef_[0])
                 else: importances = np.mean(np.abs(model.coef_), axis=0)
            else: return None
            if importances is not None and len(importances) == len(feature_names):
                 imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                 return imp_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
            else: logger.error(f"Mismatch importância/features para {model_type_name}"); return None
        except Exception as e: logger.error(f"Erro calc importância {model.__class__.__name__}: {e}"); return None

    def update_model_specific_analysis(self, event=None):
        selected_id = self.model_selector_var.get()
        if not selected_id or selected_id not in self.loaded_trained_models:
            self._update_text_widget(self.model_importance_text, "Selecione um modelo válido.")
            self._clear_matplotlib_widget(self.shap_summary_container); ttk.Label(self.shap_summary_container, text="Selecione modelo.").pack()
            self._clear_matplotlib_widget(self.pdp_container); ttk.Label(self.pdp_container, text="Selecione modelo.").pack()
            try:
                if self.pdp_feature_selector_combo: self.pdp_feature_selector_combo.config(values=[], state="disabled"); self.pdp_feature_selector_var.set("")
            except tk.TclError: pass
            return

        self.log(f"Atualizando análise para o modelo: {selected_id}")
        model_data = self.loaded_trained_models[selected_id]
        importances_df = model_data.get('importances')
        if importances_df is not None and not importances_df.empty:
            try: self._update_text_widget(self.model_importance_text, importances_df.round(5).to_string(index=True))
            except Exception as e: logger.error(f"Erro formatar importâncias: {e}"); self._update_text_widget(self.model_importance_text, f"Erro.")
        else: self._update_text_widget(self.model_importance_text, f"Importância não disponível para {selected_id}.")

        # --- Dispara cálculo SHAP (verifica cache ou roda em background/direto) ---
        self._start_shap_calculation(model_data)
        # --- Popula dropdown PDP e dispara PDP inicial ---
        pdp_features = model_data.get('features', [])
        if pdp_features:
            self.pdp_feature_selector_combo.config(values=pdp_features, state="readonly")
            current_pdp_feature = self.pdp_feature_selector_var.get()
            if current_pdp_feature not in pdp_features: self.pdp_feature_selector_var.set(pdp_features[0] if pdp_features else "")
            self.update_pdp_plot() # Atualiza PDP
        else:
            self.pdp_feature_selector_combo.config(values=[], state="disabled"); self.pdp_feature_selector_var.set("")
            self._clear_matplotlib_widget(self.pdp_container); ttk.Label(self.pdp_container, text="Features não encontradas.").pack()

    def _start_shap_calculation(self, model_data: Dict):
        if not SHAP_AVAILABLE: self._clear_matplotlib_widget(self.shap_summary_container); ttk.Label(self.shap_summary_container, text="Biblioteca 'shap' não instalada.").pack(); return
        if self.X_clean is None: self._clear_matplotlib_widget(self.shap_summary_container); ttk.Label(self.shap_summary_container, text="Dados não carregados.").pack(); return

        model = model_data.get('model'); model_id = self.model_selector_var.get(); features_for_model = model_data.get('features')
        if model is None or features_for_model is None: return

        # --- VERIFICAÇÃO DO CACHE COM IDENTIFICADOR ---
        use_cache = False
        if 'shap_values' in model_data and 'shap_data_identifier' in model_data:
            cached_data_id = model_data['shap_data_identifier']; cached_features = model_data.get('shap_model_features')
            if cached_data_id == self.current_data_identifier and cached_features == features_for_model:
                self.log(f"Reutilizando SHAP cacheado para {model_id}."); use_cache = True
            else:
                self.log(f"Cache SHAP para {model_id} invalidado."); model_data.pop('shap_values', None); model_data.pop('X_sample', None); model_data.pop('shap_data_identifier', None); model_data.pop('shap_model_features', None); model_data.pop('shap_used_kernel', None); model_data.pop('shap_n_samples', None)

        if use_cache:
            shap_payload_cached = {'model_id': model_id, 'shap_values': model_data['shap_values'], 'X_sample': model_data['X_sample'], 'model_type_name': model.__class__.__name__, 'used_kernel': model_data.get('shap_used_kernel', False), 'n_samples': model_data.get('shap_n_samples', '?')}
            self._plot_shap_summary(shap_payload_cached); return

        # --- Se não usou cache, continua para o cálculo ---
        model_type_name = model.__class__.__name__; kernel_models = ['SVC', 'KNeighborsClassifier', 'GaussianNB']
        use_kernel_explainer = model_type_name in kernel_models

        if use_kernel_explainer:
            self.log(f"Iniciando cálculo SHAP para {model_type_name} em background..."); self._clear_matplotlib_widget(self.shap_summary_container); ttk.Label(self.shap_summary_container, text=f"Calculando SHAP para {model_type_name}...\n(Pode levar alguns minutos)").pack()
            if self.shap_thread and self.shap_thread.is_alive(): self.log("Cálculo SHAP anterior em andamento..."); return
            # Passa uma CÓPIA do model_data para a thread para evitar race conditions se o usuário trocar de modelo rapidamente
            self.shap_thread = threading.Thread(target=self._run_shap_task, args=(model_data.copy(), True), daemon=True); self.shap_thread.start()
        else:
            self.log(f"Calculando SHAP para {model_type_name} diretamente..."); self._clear_matplotlib_widget(self.shap_summary_container); ttk.Label(self.shap_summary_container, text=f"Calculando SHAP para {model_type_name}...").pack()
            self.parent.update_idletasks(); self._run_shap_task(model_data, False)

    def _run_shap_task(self, model_data_copy: Dict, use_kernel_explainer: bool): # Recebe cópia
        shap_values_for_plot = None; X_shap_sample = None;
        # Pega o ID do modelo a partir da CÓPIA dos dados passados para a thread
        original_model_id_for_task = next((k for k, v in self.loaded_trained_models.items() if v == model_data_copy), None)
        if not original_model_id_for_task:
             logger.error("Não foi possível determinar o model_id original na thread SHAP.")
             # Tenta pegar o ID selecionado na GUI como fallback, mas pode não ser o correto
             original_model_id_for_task = self.model_selector_var.get()
             if not original_model_id_for_task: # Se nem isso funcionar
                 self.gui_queue.put(("shap_error", {'model_id': 'Desconhecido', 'error': "ID do modelo perdido na thread."}))
                 return


        try:
            model = model_data_copy.get('model'); scaler = model_data_copy.get('scaler'); features = model_data_copy.get('features')
            if not all([model, features, self.X_clean is not None]): raise ValueError("Dados insuficientes na thread SHAP")

            X_analysis = self.X_clean[features].copy() # Usa X_clean da instância principal
            if scaler:
                X_analysis = X_analysis.replace([np.inf, -np.inf], np.nan)
                if X_analysis.isnull().values.any(): X_analysis.fillna(X_analysis.median(), inplace=True)
                if X_analysis.isnull().values.any(): raise ValueError("NaNs pós-imputação (thread SHAP)")
                X_scaled_np = scaler.transform(X_analysis); X_shap_ready = pd.DataFrame(X_scaled_np, index=X_analysis.index, columns=features)
            else: X_shap_ready = X_analysis

            N_SHAP_SAMPLES = 50 if use_kernel_explainer else 500
            N_KERNEL_BG = 25 if use_kernel_explainer else 50
            if len(X_shap_ready) > N_SHAP_SAMPLES: X_shap_sample = shap.sample(X_shap_ready, N_SHAP_SAMPLES, random_state=RANDOM_STATE)
            else: X_shap_sample = X_shap_ready

            explainer = None; shap_values = None; model_type_name = model.__class__.__name__
            # --- Lógica para escolher explainer e calcular shap_values ---
            if use_kernel_explainer:
                 X_kernel_bg = shap.sample(X_shap_ready, N_KERNEL_BG, random_state=RANDOM_STATE)
                 def model_predict_proba_pos(data):
                    if not isinstance(data, pd.DataFrame): data_df = pd.DataFrame(data, columns=features)
                    else: data_df = data
                    try: return model.predict_proba(data_df)[:, 1]
                    except Exception as e: logger.error(f"Erro pred kernel wrap: {e}"); return np.full(len(data_df), 0.5)
                 explainer = shap.KernelExplainer(model_predict_proba_pos, X_kernel_bg); shap_values = explainer.shap_values(X_shap_sample)
            elif model_type_name in ['RandomForestClassifier', 'LGBMClassifier', 'GradientBoostingClassifier']: # Inclui LGBM se disponível
                explainer = shap.TreeExplainer(model); shap_values = explainer.shap_values(X_shap_sample)
            elif model_type_name in ['LogisticRegression']:
                 # Passa máscara booleana para LinearExplainer se os dados forem esparsos (improvável aqui)
                 explainer = shap.LinearExplainer(model, X_shap_sample); shap_values = explainer.shap_values(X_shap_sample)
            else:
                # Se não for nenhum dos tipos conhecidos, tenta Kernel como fallback final
                logger.warning(f"Tipo de modelo {model_type_name} não otimizado para SHAP, usando KernelExplainer (lento).")
                use_kernel_explainer = True # Marca que usou kernel
                N_SHAP_SAMPLES = 50; N_KERNEL_BG = 25 # Usa amostras menores
                if len(X_shap_ready) > N_SHAP_SAMPLES: X_shap_sample = shap.sample(X_shap_ready, N_SHAP_SAMPLES, random_state=RANDOM_STATE)
                else: X_shap_sample = X_shap_ready
                X_kernel_bg = shap.sample(X_shap_ready, N_KERNEL_BG, random_state=RANDOM_STATE)
                def model_predict_proba_pos_fb(data):
                    if not isinstance(data, pd.DataFrame): data_df = pd.DataFrame(data, columns=features)
                    else: data_df = data
                    try: return model.predict_proba(data_df)[:, 1]
                    except Exception as e: logger.error(f"Erro pred kernel wrap fb: {e}"); return np.full(len(data_df), 0.5)
                explainer = shap.KernelExplainer(model_predict_proba_pos_fb, X_kernel_bg); shap_values = explainer.shap_values(X_shap_sample)
            # --- Fim da lógica do explainer ---

            # --- Processamento dos shap_values ---
            shap_values_for_plot = None
            if shap_values is not None:
                if isinstance(shap_values, list) and len(shap_values) == 2: shap_values_for_plot = shap_values[1] # Classe 1 para Tree/Linear binário
                elif isinstance(shap_values, np.ndarray):
                     if len(shap_values.shape) == 3: shap_values_for_plot = shap_values[:,:,1] # Formato explainer(X)
                     elif len(shap_values.shape) == 2: shap_values_for_plot = shap_values # Kernel ou Linear binário (já classe 1)
                     else: logger.warning(f"Formato inesperado de shap_values numpy array: {shap_values.shape}"); shap_values_for_plot = shap_values
                else: logger.warning(f"Tipo inesperado de shap_values: {type(shap_values)}"); shap_values_for_plot = shap_values
            # --- Fim do processamento ---

            if shap_values_for_plot is not None and X_shap_sample is not None:
                cache_updated = False
                # --- ATENÇÃO: Acessa o dict principal com thread-safety (idealmente usaria Lock, mas para leitura/escrita simples pode funcionar) ---
                if original_model_id_for_task in self.loaded_trained_models:
                    cache_entry = {'shap_values': shap_values_for_plot, 'X_sample': X_shap_sample, 'shap_data_identifier': self.current_data_identifier, 'shap_model_features': features, 'shap_used_kernel': use_kernel_explainer, 'shap_n_samples': N_SHAP_SAMPLES}
                    try:
                        self.loaded_trained_models[original_model_id_for_task].update(cache_entry)
                        if 'shap_values' in self.loaded_trained_models[original_model_id_for_task]: cache_updated = True; logger.info(f"SHAP cacheado para {original_model_id_for_task}.")
                        else: logger.error(f"ERRO PÓS-CACHE: Chaves SHAP não encontradas para {original_model_id_for_task}!")
                    except Exception as e_cache: logger.error(f"ERRO ao atualizar cache SHAP para {original_model_id_for_task}: {e_cache}", exc_info=True)
                else: logger.warning(f"Modelo {original_model_id_for_task} não no cache ao salvar SHAP.")

                if cache_updated: self.gui_queue.put(("shap_result_calculated", {'model_id': original_model_id_for_task}))
                else: self.gui_queue.put(("shap_error", {'model_id': original_model_id_for_task, 'error': "Falha ao salvar SHAP no cache."}))
            else: raise ValueError("Cálculo SHAP não produziu resultados válidos.")
        except Exception as e: logger.error(f"Erro thread SHAP para {original_model_id_for_task}: {e}", exc_info=True); self.gui_queue.put(("shap_error", {'model_id': original_model_id_for_task, 'error': str(e)}))

    def _plot_shap_summary(self, shap_data: Dict):
        model_id_for_plot = shap_data.get('model_id', 'Desconhecido')
        self._clear_matplotlib_widget(self.shap_summary_container)
        try:
            shap_values = shap_data['shap_values']; X_sample = shap_data['X_sample']
            model_type = shap_data['model_type_name']; used_kernel = shap_data['used_kernel']
            n_samples = shap_data['n_samples']
            if self.fig_shap_summary: plt.close(self.fig_shap_summary)
            self.fig_shap_summary = plt.figure(figsize=(8, 6))
            ax = self.fig_shap_summary.add_subplot(111)
            plot_title = f'SHAP Summary ({model_type})'
            if used_kernel: plot_title += f'\n(KernelExplainer - {n_samples} amostras)'
            # Passa feature_names explicitamente para o plot se X_sample for numpy array
            feature_names_plot = X_sample.columns if isinstance(X_sample, pd.DataFrame) else shap_data.get('features') # Fallback
            shap.summary_plot(shap_values, X_sample, plot_type="dot", show=False, max_display=15, feature_names=feature_names_plot)
            ax = plt.gca(); ax.set_title(plot_title); plt.tight_layout(pad=1.0)
            self.shap_summary_canvas_widget = self._embed_matplotlib_figure(self.fig_shap_summary, self.shap_summary_container)
        except Exception as e: logger.error(f"Erro ao plotar SHAP para {model_id_for_plot}: {e}", exc_info=True); ttk.Label(self.shap_summary_container, text=f"Erro plotar SHAP:\n{e}").pack()

    def update_pdp_plot(self, event=None):
        selected_model_id = self.model_selector_var.get(); selected_pdp_feature = self.pdp_feature_selector_var.get()
        # Adiciona verificação para self.X_clean
        if not selected_model_id or not selected_pdp_feature or self.X_clean is None:
             self._clear_matplotlib_widget(self.pdp_container); ttk.Label(self.pdp_container, text="Selecione modelo/feature...").pack(); return
        model_data = self.loaded_trained_models.get(selected_model_id);
        if not model_data: return
        model = model_data.get('model'); scaler = model_data.get('scaler'); features = model_data.get('features')
        if not all([model, features]) or selected_pdp_feature not in features: self._clear_matplotlib_widget(self.pdp_container); ttk.Label(self.pdp_container, text="Modelo/Features inválidos.").pack(); return

        self._clear_matplotlib_widget(self.pdp_container)
        ttk.Label(self.pdp_container, text=f"Calculando PDP para {selected_pdp_feature}...").pack()
        self.parent.update_idletasks()
        try:
            X_analysis = self.X_clean[features].copy()
            if scaler:
                X_analysis = X_analysis.replace([np.inf, -np.inf], np.nan)
                if X_analysis.isnull().values.any(): X_analysis.fillna(X_analysis.median(), inplace=True)
                if X_analysis.isnull().values.any(): raise ValueError("NaNs pós-imputação (PDP)")
                X_scaled_np = scaler.transform(X_analysis); X_pdp_ready = pd.DataFrame(X_scaled_np, index=X_analysis.index, columns=features)
            else: X_pdp_ready = X_analysis
            if self.fig_pdp: plt.close(self.fig_pdp)
            self.fig_pdp, ax = plt.subplots(figsize=(7, 5))
            # Usa try-except para from_estimator que pode falhar com alguns estimadores/versões
            try:
                PartialDependenceDisplay.from_estimator(estimator=model, X=X_pdp_ready, features=[selected_pdp_feature], kind='average', target=1, ax=ax, line_kw={"color": "green", "linewidth": 2.5})
            except TypeError as te: # Tenta sem o argumento 'target' se der TypeError (versões antigas?)
                 if 'target' in str(te):
                     logger.warning(f"TypeError com argumento 'target' no PDP para {model.__class__.__name__}. Tentando sem ele.")
                     PartialDependenceDisplay.from_estimator(estimator=model, X=X_pdp_ready, features=[selected_pdp_feature], kind='average', ax=ax, line_kw={"color": "orange", "linewidth": 2.5})
                 else: raise te # Relevanta outro TypeError
            except Exception as e_pdp_disp: raise e_pdp_disp # Relevanta outros erros

            ax.set_title(f'Partial Dependence Plot\n({selected_pdp_feature} vs Probabilidade Empate)'); ax.set_xlabel(selected_pdp_feature); ax.set_ylabel('Dependência Parcial (Prob Empate)'); ax.grid(True, linestyle='--', alpha=0.6); plt.tight_layout(pad=1.0)
            self.pdp_canvas_widget = self._embed_matplotlib_figure(self.fig_pdp, self.pdp_container)
        except Exception as e: logger.error(f"Erro ao gerar PDP para {selected_pdp_feature}: {e}", exc_info=True); self._clear_matplotlib_widget(self.pdp_container); ttk.Label(self.pdp_container, text=f"Erro gerar PDP:\n{e}").pack()

    def process_gui_queue(self):
        if self.stop_processing_queue: logger.debug("Parando fila GUI."); return
        try:
            while True:
                try: message_type, payload = self.gui_queue.get_nowait()
                except queue.Empty: break
                except Exception as e: logger.error(f"Erro get fila GUI: {e}", exc_info=True); break
                try:
                    if message_type == "shap_result_calculated":
                        finished_model_id = payload.get('model_id')
                        current_model_id = self.model_selector_var.get()
                        if finished_model_id == current_model_id and finished_model_id in self.loaded_trained_models:
                            model_data = self.loaded_trained_models[finished_model_id]
                            if 'shap_values' in model_data and model_data['shap_values'] is not None and \
                               'X_sample' in model_data and model_data['X_sample'] is not None:
                                shap_plot_payload = {'model_id': finished_model_id, 'shap_values': model_data['shap_values'], 'X_sample': model_data['X_sample'], 'model_type_name': model_data['model'].__class__.__name__, 'used_kernel': model_data.get('shap_used_kernel', False), 'n_samples': model_data.get('shap_n_samples', '?')}
                                self._plot_shap_summary(shap_plot_payload)
                            else: logger.error(f"Erro Plot SHAP: Resultado para {finished_model_id} não encontrado/inválido no cache."); self._clear_matplotlib_widget(self.shap_summary_container); ttk.Label(self.shap_summary_container, text=f"Erro cache SHAP.").pack()
                        else: self.log(f"Ignorando resultado SHAP obsoleto para {finished_model_id}.")
                    elif message_type == "shap_error":
                         error_model_id = payload.get('model_id', 'Desconhecido'); error_msg = payload.get('error', 'Erro desconhecido')
                         self.log(f"Erro cálculo SHAP {error_model_id}: {error_msg}")
                         # Limpa apenas se o erro for para o modelo atualmente selecionado
                         if error_model_id == self.model_selector_var.get():
                             self._clear_matplotlib_widget(self.shap_summary_container); ttk.Label(self.shap_summary_container, text=f"Erro SHAP:\n{error_msg}").pack()
                    else: logger.warning(f"Tipo mensagem GUI desconhecido: {message_type}")
                except Exception as e_proc_msg: logger.error(f"Erro processar msg GUI tipo '{message_type}': {e_proc_msg}", exc_info=True)
        except Exception as e_loop: logger.error(f"Erro CRÍTICO loop fila GUI: {e_loop}", exc_info=True)
        finally:
            if not self.stop_processing_queue:
                 try:
                     if hasattr(self.main_tk_root, 'winfo_exists') and self.main_tk_root.winfo_exists(): self.main_tk_root.after(100, self.process_gui_queue)
                 except Exception as e:
                     if not self.stop_processing_queue: logger.error(f"Erro reagendar fila GUI: {e}")

# --- Bloco de Teste ---
if __name__ == "__main__":
    try: from ctypes import windll; windll.shcore.SetProcessDpiAwareness(1)
    except: pass
    root = tk.Tk(); root.title("Teste Aba Análise - Etapa 3 com Cache"); root.geometry("1200x900")
    tab_frame = ttk.Frame(root); tab_frame.pack(expand=True, fill="both")
    app_tab = FeatureAnalyzerApp(tab_frame, root); root.mainloop()