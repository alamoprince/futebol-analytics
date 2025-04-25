# --- src/feature_analyzer_tab.py ---
# VERSÃO SIMPLIFICADA - APENAS ANÁLISE GERAL

import tkinter as tk
from tkinter import ttk, messagebox
# Removido filedialog pois não é usado aqui
from tkinter.scrolledtext import ScrolledText # Ainda usado para texto
import sys, os, pandas as pd, numpy as np, datetime, io, traceback, warnings
from typing import Optional, List, Dict, Any, Tuple # Adicionadas Tuple, Any

# Gráficos
import matplotlib; matplotlib.use('TkAgg'); import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns

# Path Setup
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path: sys.path.insert(0, CURRENT_DIR)
if BASE_DIR not in sys.path: sys.path.insert(0, BASE_DIR)

# Imports do Projeto
try:
    from config import GOALS_COLS, RANDOM_STATE # RANDOM_STATE não é usado aqui, pode remover se quiser
    from data_handler import (load_historical_data, calculate_historical_intermediate,
                              calculate_probabilities, calculate_normalized_probabilities,
                              calculate_rolling_stats, calculate_rolling_std,
                              calculate_binned_features, calculate_derived_features,
                              calculate_rolling_goal_stats, calculate_poisson_draw_prob,
                              calculate_pi_ratings)
    from logger_config import setup_logger
except ImportError as e:
     import logging; logger = logging.getLogger(__name__)
     logger.critical(f"Import Error CRÍTICO em feature_analyzer_tab.py: {e}", exc_info=True)
     try: root_err = tk.Tk(); root_err.withdraw(); messagebox.showerror("Import Error (Analyzer Tab)", f"Failed...\n{e}"); root_err.destroy()
     except Exception: print(f"CRITICAL Import Error: {e}")
     sys.exit(1)

logger = setup_logger("FeatureAnalyzerAppSimple")

class FeatureAnalyzerApp:

    def __init__(self, parent_frame):
        self.parent = parent_frame
        self.df_historical_raw: Optional[pd.DataFrame] = None
        self.df_historical_processed: Optional[pd.DataFrame] = None
        self.X_clean: Optional[pd.DataFrame] = None
        self.y_clean: Optional[pd.Series] = None
        self.current_data_identifier = None

        self.info_text: Optional[tk.Text] = None
        self.head_text: Optional[tk.Text] = None
        self.desc_text: Optional[tk.Text] = None
        self.target_text: Optional[tk.Text] = None
        self.status_label: Optional[ttk.Label] = None

        self.feature_selector_var = tk.StringVar()
        self.feature_selector_combo: Optional[ttk.Combobox] = None

        self.fig_dist: Optional[plt.Figure] = None; self.axes_dist = None
        self.fig_dist_target: Optional[plt.Figure] = None; self.ax_dist_target = None
        self.fig_corr: Optional[plt.Figure] = None; self.ax_corr = None
        self.dist_canvas_widget: Optional[FigureCanvasTkAgg] = None
        self.dist_target_canvas_widget: Optional[FigureCanvasTkAgg] = None
        self.corr_heatmap_canvas_widget: Optional[FigureCanvasTkAgg] = None
        self.dist_canvas_container: Optional[ttk.Frame] = None
        self.dist_target_canvas_container: Optional[ttk.Frame] = None
        self.corr_canvas_container: Optional[ttk.Frame] = None

        self.univar_update_job = None

        self.create_widgets()

    def create_widgets(self):
        """Cria widgets para análise geral, SEM seção de modelo específico."""
        top_controls_frame = ttk.Frame(self.parent); top_controls_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10, 0))
        load_button = ttk.Button(top_controls_frame, text="Carregar & Analisar Dados Históricos", command=self.load_and_display_data); load_button.pack(side=tk.LEFT, padx=(0, 10))
        self.status_label = ttk.Label(top_controls_frame, text="Pronto."); self.status_label.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

        canvas = tk.Canvas(self.parent, borderwidth=0, highlightthickness=0); scrollbar = ttk.Scrollbar(self.parent, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas); self.scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw"); canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=(5, 10)); scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=(5, 10))

        overview_frame = ttk.LabelFrame(self.scrollable_frame, text=" Visão Geral Dados Brutos ", padding=10); overview_frame.pack(fill=tk.X, padx=10, pady=5)
        self.info_text = self._create_scrolled_text(overview_frame, height=10, title="Infos Gerais (df.info)"); self.head_text = self._create_scrolled_text(overview_frame, height=7, title="Amostra Dados Raw (df.head)")
        target_desc_frame = ttk.LabelFrame(self.scrollable_frame, text=" Alvo e Descrição (Pós-Processamento) ", padding=10); target_desc_frame.pack(fill=tk.X, padx=10, pady=5)
        self.desc_text = self._create_scrolled_text(target_desc_frame, height=12, title="Describe (Features Limpas)"); self.target_text = self._create_scrolled_text(target_desc_frame, height=5, title="Distribuição Alvo (IsDraw)")
        univar_frame = ttk.LabelFrame(self.scrollable_frame, text=" Análise Univariada ", padding=10); univar_frame.pack(fill=tk.X, padx=10, pady=5)
        selector_frame = ttk.Frame(univar_frame); selector_frame.pack(fill=tk.X, pady=(0, 10)); ttk.Label(selector_frame, text="Feature:").pack(side=tk.LEFT, padx=(0, 5))
        self.feature_selector_combo = ttk.Combobox(selector_frame, textvariable=self.feature_selector_var, state="readonly", width=30); self.feature_selector_combo.pack(side=tk.LEFT);
        self.feature_selector_combo.bind("<<ComboboxSelected>>", self.on_univar_feature_select)
        plot_frame_univar = ttk.Frame(univar_frame); plot_frame_univar.pack(fill=tk.BOTH, expand=True)
        self.dist_canvas_container = ttk.LabelFrame(plot_frame_univar, text=" Distribuição Feature ", padding=5); self.dist_canvas_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5); ttk.Label(self.dist_canvas_container, text="...").pack()
        self.dist_target_canvas_container = ttk.LabelFrame(plot_frame_univar, text=" Distribuição vs Alvo ", padding=5); self.dist_target_canvas_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5); ttk.Label(self.dist_target_canvas_container, text="...").pack()
        multivar_frame = ttk.LabelFrame(self.scrollable_frame, text=" Análise Multivariada ", padding=10); multivar_frame.pack(fill=tk.X, padx=10, pady=5)
        self.corr_canvas_container = ttk.LabelFrame(multivar_frame, text=" Heatmap Correlação ", padding=5); self.corr_canvas_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5); ttk.Label(self.corr_canvas_container, text="...").pack()

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
        canvas_attr_name = None
        if container_widget == self.dist_canvas_container: canvas_attr_name = 'dist_canvas_widget'
        elif container_widget == self.dist_target_canvas_container: canvas_attr_name = 'dist_target_canvas_widget'
        elif container_widget == self.corr_canvas_container: canvas_attr_name = 'corr_heatmap_canvas_widget'
        existing_canvas = getattr(self, canvas_attr_name, None) if canvas_attr_name else None
        if existing_canvas and isinstance(existing_canvas, FigureCanvasTkAgg) and plt.fignum_exists(fig.number):
             existing_canvas.figure = fig; existing_canvas.draw_idle(); return existing_canvas
        else:
             self._clear_matplotlib_widget(container_widget)
             try:
                 canvas = FigureCanvasTkAgg(fig, master=container_widget); canvas.draw()
                 widget = canvas.get_tk_widget(); widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
                 if canvas_attr_name: setattr(self, canvas_attr_name, canvas)
                 return canvas
             except Exception as e: logger.error(f"Erro embed matplotlib: {e}", exc_info=True); ttk.Label(container_widget, text=f"Erro gráfico:\n{e}").pack(); return None

    def log(self, message: str):
        logger.info(f"[GUI AnalyzerSimple] {message}")
        try:
            if self.status_label and self.status_label.winfo_exists(): self.status_label.config(text=message[:100])
        except tk.TclError: pass
        except Exception as e: logger.error(f"Erro status label: {e}", exc_info=True)

    # --- load_and_display_data (CORRIGIDO) ---
    def load_and_display_data(self):
        self.log("Iniciando: Carregando & Processando dados históricos...")
        # Limpa widgets
        widgets_to_clear = [self.info_text, self.head_text, self.desc_text, self.target_text]
        for w in widgets_to_clear: self._update_text_widget(w, "Carregando...")
        self._clear_matplotlib_widget(self.dist_canvas_container); ttk.Label(self.dist_canvas_container, text="...").pack()
        self._clear_matplotlib_widget(self.dist_target_canvas_container); ttk.Label(self.dist_target_canvas_container, text="...").pack()
        self._clear_matplotlib_widget(self.corr_canvas_container); ttk.Label(self.corr_canvas_container, text="...").pack()

        # Reseta dados
        self.df_historical_raw = None; self.df_historical_processed = None
        self.X_clean = None; self.y_clean = None; self.current_data_identifier = None

        try:
            # 1. Load Raw Data
            self.log("Etapa 1: Carregando dados brutos...")
            df_raw = load_historical_data()
            if df_raw is None or df_raw.empty: raise ValueError("Falha ao carregar dados históricos.")
            self.df_historical_raw = df_raw.copy()
            self.log(f"Dados brutos carregados: {self.df_historical_raw.shape}")
            # Display Raw Info/Head
            buffer = io.StringIO(); self.df_historical_raw.info(buf=buffer); self._update_text_widget(self.info_text, buffer.getvalue())
            with pd.option_context('display.max_rows', 10, 'display.max_columns', None, 'display.width', 1000):
                 self._update_text_widget(self.head_text, self.df_historical_raw.head(5).to_string())

            # 2. Processamento Completo Features
            self.log("Etapa 2: Processando features (Pipeline Completo)...")
            df_p = self.df_historical_raw.copy() # Começa com cópia dos dados brutos
            logger.info("=== INÍCIO PIPELINE FEAT ENG (ANALYSIS TAB) ===")
            # Calcula médias da liga
            goals_h_col=GOALS_COLS.get('home'); goals_a_col=GOALS_COLS.get('away')
            avg_h_league=np.nanmean(df_p[goals_h_col]) if goals_h_col in df_p else 1.0
            avg_a_league=np.nanmean(df_p[goals_a_col]) if goals_a_col in df_p else 1.0
            avg_h_league=1.0 if pd.isna(avg_h_league) else avg_h_league
            avg_a_league=1.0 if pd.isna(avg_a_league) else avg_a_league
            epsilon=1e-6; avg_h_league_safe=max(avg_h_league,epsilon); avg_a_league_safe=max(avg_a_league,epsilon)
            self.log(f"Médias Liga: H={avg_h_league_safe:.3f}, A={avg_a_league_safe:.3f}")

            # --- Chama TODAS as funções de cálculo ---
            df_p = calculate_historical_intermediate(df_p)
            if 'IsDraw' not in df_p.columns or df_p['IsDraw'].isnull().all(): raise ValueError("Alvo 'IsDraw' ausente/NaN pós-intermediate.")
            df_p = calculate_probabilities(df_p) # Garante probs
            df_p = calculate_normalized_probabilities(df_p)
            df_p = calculate_pi_ratings(df_p)
            df_p = calculate_rolling_stats(df_p,['VG','CG']) # Adicione Ptos se relevante
            df_p = calculate_rolling_std(df_p,['CG'])      # Adicione VG, Ptos se relevante
            df_p = calculate_rolling_goal_stats(df_p, avg_goals_home_league=avg_h_league_safe, avg_goals_away_league=avg_a_league_safe)
            df_p = calculate_poisson_draw_prob(df_p, avg_goals_home_league=avg_h_league_safe, avg_goals_away_league=avg_a_league_safe);
            df_p = calculate_binned_features(df_p)
            df_p = calculate_derived_features(df_p)
            # Adicione outras chamadas 'calculate_*' se tiver mais features
            # -----------------------------------------

            self.df_historical_processed = df_p.copy() # Armazena resultado final do processamento
            self.log("Processamento features concluído.")
            logger.info("=== FIM PIPELINE FEAT ENG ===")

            # 3. Preparar X, y para Análise
            self.log("Etapa 3: Preparando X, y para análise...")
            target_col='IsDraw'
            if target_col not in self.df_historical_processed.columns: raise ValueError(f"Coluna alvo '{target_col}' não encontrada!");

            # Seleciona features NUMÉRICAS do DF processado para análise
            features_to_analyze = self.df_historical_processed.select_dtypes(include=np.number).columns.tolist()
            cols_to_exclude = list(GOALS_COLS.values()) + [target_col, 'Ptos_H', 'Ptos_A'] # Exclui gols, alvo e pontos
            # Garante que só features que realmente existem sejam usadas
            features_present = [f for f in features_to_analyze if f not in cols_to_exclude and f in self.df_historical_processed.columns]
            self.log(f"Features numéricas para análise: {len(features_present)}")
            if not features_present: raise ValueError("Nenhuma feature numérica válida encontrada após processamento.")

            # Cria X_raw e y_raw a partir do DF PROCESSADO
            X_raw = self.df_historical_processed[features_present].copy()
            y_raw = self.df_historical_processed[target_col]

            # Combina para dropna consistente
            analysis_df = X_raw.join(y_raw)
            initial_rows = len(analysis_df)
            analysis_df.replace([np.inf, -np.inf], np.nan, inplace=True) # Trata infinitos
            # Drop baseado nas features presentes + alvo
            analysis_df.dropna(subset=(features_present + [target_col]), inplace=True);
            rows_dropped = initial_rows - len(analysis_df)
            self.log(f"Dados p/ análise (final): {analysis_df.shape} ({rows_dropped} linhas removidas).")
            if analysis_df.empty: raise ValueError("Nenhum dado restou após limpeza final.")

            # Define X_clean e y_clean finais
            self.X_clean = analysis_df[features_present].copy()
            self.y_clean = analysis_df[target_col].astype(int)
            # Validações finais (opcional, mas bom ter)
            if self.X_clean.isnull().values.any() or self.X_clean.isin([np.inf, -np.inf]).values.any(): raise ValueError("NaNs/Infs persistentes em X_clean")
            if self.y_clean.isnull().values.any(): raise ValueError("NaNs persistentes em y_clean")

            # --- ATUALIZA IDENTIFICADOR DOS DADOS ---
            self.current_data_identifier = (self.X_clean.shape, tuple(self.X_clean.columns))
            self.log(f"Identificador de dados atualizado: {self.current_data_identifier}")
            # -----------------------------------------

            # 4. Display Describe/Target (AGORA USA X_clean e y_clean)
            self.log("Etapa 4: Atualizando Describe/Target...");
            try:
                 with pd.option_context('display.max_rows', 200, 'display.max_columns', None, 'display.width', 1000):
                      # Usa self.X_clean que foi definido acima
                      self._update_text_widget(self.desc_text, self.X_clean.describe(include='all').to_string());
            except Exception as e: self.log(f"Erro describe: {e}"); self._update_text_widget(self.desc_text, f"Erro.")
            try:
                # Usa self.y_clean que foi definido acima
                counts=self.y_clean.value_counts(); dist=self.y_clean.value_counts(normalize=True);
                self._update_text_widget(self.target_text, f"Contagem:\n{counts.to_string()}\n\nProporção:\n{dist.apply('{:.2%}'.format).to_string()}");
            except Exception as e: self.log(f"Erro target dist: {e}"); self._update_text_widget(self.target_text, f"Erro.")

            # 5. Gerar Gráficos Básicos (AGORA USA X_clean)
            self.log("Etapa 5: Gerando gráficos básicos...");
            numeric_features = self.X_clean.select_dtypes(include=np.number).columns.tolist() # Usa X_clean
            if numeric_features:
                self.feature_selector_combo['values'] = numeric_features
                self.feature_selector_var.set(numeric_features[0] if numeric_features else "")
                self.on_univar_feature_select() # Dispara debouncer
            else:
                self.feature_selector_combo['values']=[]; self.feature_selector_var.set("")
                self.log("Nenhuma feature numérica para análise univariada.")
            self.generate_correlation_heatmap() # Usa X_clean internamente

            self.log("Carregamento e análise básica concluídos.");

        except Exception as e:
            errmsg = f"Erro fatal no carregamento/análise: {e}"; self.log(f"!!! ERRO: {errmsg}"); logger.error(errmsg, exc_info=True); messagebox.showerror("Erro Fatal na Análise", f"{errmsg}\n\nVer logs.", parent=self.parent)
            widgets_to_clear=[self.info_text, self.head_text, self.desc_text, self.target_text] # Não inclui mais importance
            for w in widgets_to_clear: self._update_text_widget(w, f"Erro: {e}")
            self._clear_matplotlib_widget(self.dist_canvas_container); ttk.Label(self.dist_canvas_container, text="Erro.").pack(); self._clear_matplotlib_widget(self.dist_target_canvas_container); ttk.Label(self.dist_target_canvas_container, text="Erro.").pack(); self._clear_matplotlib_widget(self.corr_canvas_container); ttk.Label(self.corr_canvas_container, text="Erro.").pack()
            try: 
                if self.feature_selector_combo: self.feature_selector_combo.config(values=[], state="disabled"); self.feature_selector_var.set("")
            except tk.TclError: pass


    # --- Funções Gráficos Univar/Multivar ---
    def on_univar_feature_select(self, event=None):
        """Callback com debounce para seleção de feature univariada."""
        # Usa self.parent.after porque esta classe não tem mais self.main_tk_root
        if self.univar_update_job:
            try: self.parent.after_cancel(self.univar_update_job)
            except tk.TclError: pass # Ignora se o job/widget não existe mais
        self.univar_update_job = self.parent.after(400, self._update_univariate_plots_task)


    def _update_univariate_plots_task(self):
        # ... (código idêntico ao anterior, usa X_clean e y_clean) ...
        self.univar_update_job = None
        if self.X_clean is None or self.y_clean is None: return
        selected_feature = self.feature_selector_var.get()
        if not selected_feature or selected_feature not in self.X_clean.columns: return
        # Plot 1: Distribuição
        try:
            if self.fig_dist is None or not plt.fignum_exists(self.fig_dist.number): self.fig_dist, self.axes_dist = plt.subplots(2, 1, figsize=(6, 5), gridspec_kw={'height_ratios': [3, 1]}, sharex=True, constrained_layout=True); self.dist_canvas_widget = self._embed_matplotlib_figure(self.fig_dist, self.dist_canvas_container)
            else: self.axes_dist[0].clear(); self.axes_dist[1].clear()
            feature_data = self.X_clean[selected_feature].dropna(); sns.histplot(feature_data, kde=True, ax=self.axes_dist[0]); self.axes_dist[0].set_title(f'Distribuição de {selected_feature}'); self.axes_dist[0].set_xlabel(''); sns.boxplot(x=feature_data, ax=self.axes_dist[1]); self.axes_dist[1].set_xlabel(selected_feature);
            if self.dist_canvas_widget: self.dist_canvas_widget.draw_idle()
        except Exception as e: logger.error(f"Erro dist plot {selected_feature}: {e}", exc_info=True); self._clear_matplotlib_widget(self.dist_canvas_container); ttk.Label(self.dist_canvas_container, text=f"Erro.").pack()
        # Plot 2: Distribuição vs Alvo
        try:
            if self.fig_dist_target is None or not plt.fignum_exists(self.fig_dist_target.number): self.fig_dist_target, self.ax_dist_target = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True); self.dist_target_canvas_widget = self._embed_matplotlib_figure(self.fig_dist_target, self.dist_target_canvas_container)
            else: self.ax_dist_target.clear()
            plot_data = pd.DataFrame({'feature': self.X_clean[selected_feature], 'target': self.y_clean}); sns.boxplot(data=plot_data, x='target', y='feature', ax=self.ax_dist_target); self.ax_dist_target.set_title(f'{selected_feature} vs IsDraw'); self.ax_dist_target.set_xlabel('IsDraw (0: Não Empate, 1: Empate)'); self.ax_dist_target.set_ylabel(selected_feature);
            if self.dist_target_canvas_widget: self.dist_target_canvas_widget.draw_idle()
        except Exception as e: logger.error(f"Erro dist vs target plot {selected_feature}: {e}", exc_info=True); self._clear_matplotlib_widget(self.dist_target_canvas_container); ttk.Label(self.dist_target_canvas_container, text=f"Erro.").pack()


    def generate_correlation_heatmap(self):
        # ... (código idêntico ao anterior, usa X_clean) ...
        if self.X_clean is None: return
        try:
            if self.fig_corr: plt.close(self.fig_corr)
            numeric_features = self.X_clean.select_dtypes(include=np.number)
            if numeric_features.shape[1] < 2: self._clear_matplotlib_widget(self.corr_canvas_container); ttk.Label(self.corr_canvas_container, text="Poucas features.").pack(); return
            corr_matrix = numeric_features.corr(); self.fig_corr, ax = plt.subplots(figsize=(9, 7), constrained_layout=True); sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax, cbar=True); ax.set_title('Heatmap Correlação'); plt.setp(ax.get_xticklabels(), rotation=60, ha='right', fontsize=7); plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7);
            self.corr_heatmap_canvas_widget = self._embed_matplotlib_figure(self.fig_corr, self.corr_canvas_container)
        except Exception as e: logger.error(f"Erro heatmap: {e}", exc_info=True); self._clear_matplotlib_widget(self.corr_canvas_container); ttk.Label(self.corr_canvas_container, text=f"Erro.").pack()
