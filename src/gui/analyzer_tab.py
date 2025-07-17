import tkinter as tk
import customtkinter as ctk
from tkinter import ttk, messagebox
import sys, os, pandas as pd, numpy as np, io
from typing import Optional

import matplotlib; matplotlib.use('TkAgg'); import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path: sys.path.insert(0, CURRENT_DIR)
if BASE_DIR not in sys.path: sys.path.insert(0, BASE_DIR)

try:
    from config import GOALS_COLS 
    from data_handler import (load_historical_data, preprocess_and_feature_engineer,
                              )
    from logger_config import setup_logger
except ImportError as e:
     import logging; logger = logging.getLogger(__name__)
     logger.critical(f"Import Error CRÍTICO em feature_analyzer_tab.py: {e}", exc_info=True)
     try: root_err = tk.Tk(); root_err.withdraw(); messagebox.showerror("Import Error (Analyzer Tab)", f"Failed...\n{e}"); root_err.destroy()
     except Exception: print(f"CRITICAL Import Error: {e}")
     sys.exit(1)
from strategies.base_strategy import BettingStrategy
logger = setup_logger("FeatureAnalyzerAppSimple")

class FeatureAnalyzerApp:

    def __init__(self, parent_frame, main_root, strategy: BettingStrategy):
        self.parent = parent_frame
        self.main_tk_root = main_root
        self.strategy = strategy

        self.df_historical_raw: Optional[pd.DataFrame] = None
        self.df_historical_processed: Optional[pd.DataFrame] = None
        self.X_clean: Optional[pd.DataFrame] = None
        self.y_clean: Optional[pd.Series] = None
        self.current_data_identifier = None

        self.info_text: Optional[tk.Text] = None
        self.head_text: Optional[tk.Text] = None
        self.desc_text: Optional[tk.Text] = None
        self.target_text: Optional[tk.Text] = None
        self.status_label: Optional[ctk.CTkLabel] = None

        self.feature_selector_var = tk.StringVar()
        self.feature_selector_combo: Optional[ctk.CTkComboBox] = None

        self.fig_dist: Optional[plt.Figure] = None; self.axes_dist = None
        self.fig_dist_target: Optional[plt.Figure] = None; self.ax_dist_target = None
        self.fig_corr: Optional[plt.Figure] = None; self.ax_corr = None
        self.dist_canvas_widget: Optional[FigureCanvasTkAgg] = None
        self.dist_target_canvas_widget: Optional[FigureCanvasTkAgg] = None
        self.corr_heatmap_canvas_widget: Optional[FigureCanvasTkAgg] = None
        self.dist_canvas_container: Optional[ctk.CTkFrame] = None
        self.dist_target_canvas_container: Optional[ctk.CTkFrame] = None
        self.corr_canvas_container: Optional[ctk.CTkFrame] = None

        self.univar_update_job = None

        self.create_widgets()

    def create_widgets(self):

        top_controls_frame = ctk.CTkFrame(self.parent)
        top_controls_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10, 0))
        load_button = ctk.CTkButton(
            top_controls_frame,
            text="Carregar & Analisar Dados Históricos",
            command=self.load_and_display_data
        )
        load_button.pack(side=tk.LEFT, padx=(0, 10))
        self.status_label = ctk.CTkLabel(top_controls_frame, text="Pronto.")
        self.status_label.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

        canvas = tk.Canvas(self.parent, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.parent, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ctk.CTkFrame(canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=(5, 10))

        overview_frame = ctk.CTkFrame(self.scrollable_frame)
        overview_frame.pack(fill=tk.X, padx=10, pady=5)
        ctk.CTkLabel(overview_frame, text="Visão Geral Dados Brutos", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(5, 10))
        self.info_text = self._create_scrolled_text(overview_frame, height=10, title="Infos Gerais (df.info)")
        self.head_text = self._create_scrolled_text(overview_frame, height=7, title="Amostra Dados Raw (df.head)")

        target_label_text = f"Alvo ({self.strategy.get_target_variable_name()}) e Descrição (Features)"
        target_desc_frame = ctk.CTkFrame(self.scrollable_frame)
        target_desc_frame.pack(fill=tk.X, padx=10, pady=5)
        ctk.CTkLabel(target_desc_frame, text=target_label_text, font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(5, 5))

        univar_frame = ctk.CTkFrame(self.scrollable_frame)
        univar_frame.pack(fill=tk.X, padx=10, pady=5)
        ctk.CTkLabel(univar_frame, text="Análise Univariada", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(5, 5))

        selector_frame = ctk.CTkFrame(univar_frame)
        selector_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        ctk.CTkLabel(selector_frame, text="Feature:").pack(side=tk.LEFT, padx=(0, 5))
        self.feature_selector_combo = ctk.CTkComboBox(
            selector_frame,
            variable=self.feature_selector_var,
            state="readonly",
            width=250 
        )
        self.feature_selector_combo.pack(side=tk.LEFT)
        self.feature_selector_combo.bind("<<ComboboxSelected>>", self.on_univar_feature_select)

        plot_frame_univar = ctk.CTkFrame(univar_frame)
        plot_frame_univar.pack(fill=tk.BOTH, expand=True)

        self.dist_canvas_container = ctk.CTkFrame(plot_frame_univar)
        self.dist_canvas_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        ctk.CTkLabel(self.dist_canvas_container, text="...").pack() 

        self.dist_target_canvas_container = ctk.CTkFrame(plot_frame_univar)
        self.dist_target_canvas_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        ctk.CTkLabel(self.dist_target_canvas_container, text="...").pack() 

        multivar_frame = ctk.CTkFrame(self.scrollable_frame)
        multivar_frame.pack(fill=tk.X, padx=10, pady=5)
        ctk.CTkLabel(multivar_frame, text="Análise Multivariada", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(5, 5))

        self.corr_canvas_container = ctk.CTkFrame(multivar_frame)
        self.corr_canvas_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        ctk.CTkLabel(self.corr_canvas_container, text="...").pack()

    def _create_scrolled_text(self, parent, height, title=""):
        container_frame = ctk.CTkFrame(parent)
        container_frame.pack(side=tk.TOP, fill=tk.X, pady=5, padx=5)

        if title:
            title_label = ctk.CTkLabel(container_frame, text=title, font=ctk.CTkFont(weight="bold"))
            title_label.pack(anchor="w", padx=10, pady=(5, 2))  

        text_area_frame = ctk.CTkFrame(container_frame, fg_color="transparent")
        text_area_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))

        xscrollbar = ttk.Scrollbar(text_area_frame, orient=tk.HORIZONTAL)
        yscrollbar = ttk.Scrollbar(text_area_frame, orient=tk.VERTICAL)
        widget = tk.Text(
            text_area_frame,
            height=height,
            wrap=tk.NONE,
            font=("Consolas", 9),
            relief=tk.FLAT,
            bd=0,
            yscrollcommand=yscrollbar.set,
            xscrollcommand=xscrollbar.set,
            state='disabled'
        )
        yscrollbar.config(command=widget.yview)
        xscrollbar.config(command=widget.xview)
        
        text_area_frame.grid_rowconfigure(0, weight=1)
        text_area_frame.grid_columnconfigure(0, weight=1)
        
        widget.grid(row=0, column=0, sticky="nsew")
        yscrollbar.grid(row=0, column=1, sticky="ns")
        xscrollbar.grid(row=1, column=0, sticky="ew")
        
        return widget

    def _update_text_widget(self, text_widget: Optional[tk.Text], content: str):
        if text_widget is None: return
        try:
            if text_widget.winfo_exists(): 
                text_widget.config(state='normal'); 
                text_widget.delete('1.0', tk.END); 
                text_widget.insert('1.0', content); 
                text_widget.config(state='disabled')
        except tk.TclError: 
            pass
        except Exception as e: 
            logger.error(f"Erro update text widget: {e}", exc_info=True)

    def _clear_matplotlib_widget(self, container_widget):
        if container_widget:
            for widget in container_widget.winfo_children(): widget.destroy()

    def _embed_matplotlib_figure(self, fig: plt.Figure, container_widget: ctk.CTkFrame):

        for widget in container_widget.winfo_children():
            widget.destroy()

        try:
            canvas = FigureCanvasTkAgg(fig, master=container_widget)
            canvas.draw()
            
            widget = canvas.get_tk_widget()
            widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            container_widget.canvas = canvas
            
            return canvas

        except Exception as e:
            logger.error(f"Erro ao embutir a figura Matplotlib: {e}", exc_info=True)
            ctk.CTkLabel(container_widget, text=f"Erro ao gerar gráfico:\n{e}", foreground="red").pack()
            return None

    def log(self, message: str):
        logger.info(f"[GUI AnalyzerSimple] {message}")
        try:
            if self.status_label and self.status_label.winfo_exists(): 
                self.status_label.config(text=message[:100])
        except tk.TclError: 
            pass
        except Exception as e: 
            logger.error(f"Erro status label: {e}", exc_info=True)

    def load_and_display_data(self):
        self.log(f"Iniciando: Carregando e Processando dados para a estratégia '{self.strategy.get_display_name()}'...")
        
        for w in [self.info_text, self.head_text, getattr(self, 'desc_text', None), getattr(self, 'target_text', None)]:
            if w: self._update_text_widget(w, "Carregando...")
        for container in [self.dist_canvas_container, self.dist_target_canvas_container, self.corr_canvas_container]:
            if container:
                self._clear_matplotlib_widget(container)
                ctk.CTkLabel(container, text="Aguardando dados...").pack()

        self.df_historical_raw = None
        self.X_clean, self.y_clean = None, None
        
        try:
            self.log("Etapa 1: Carregando arquivo de dados históricos...")
            df_raw = load_historical_data()
            if df_raw is None or df_raw.empty:
                raise ValueError("Falha ao carregar dados históricos.")
            self.df_historical_raw = df_raw
            
            buffer = io.StringIO()
            self.df_historical_raw.info(buf=buffer)
            self._update_text_widget(self.info_text, buffer.getvalue())
            with pd.option_context('display.max_rows', 10, 'display.max_columns', None, 'display.width', 1000):
                self._update_text_widget(self.head_text, self.df_historical_raw.head(5).to_string())

            self.log("Etapa 2: Preparando dados para análise...")
            
            if self.strategy.get_strategy_type() == 'rule_based':
                self.log("... tipo: Estratégia de Regras. Calculando alvo de sucesso...")
                self.y_clean = self.strategy.get_target_for_backtesting(self.df_historical_raw, GOALS_COLS)
                self.y_clean.name = self.strategy.get_target_variable_name()
                
                processed_data = preprocess_and_feature_engineer(self.df_historical_raw, self.strategy)
                if processed_data:
                    self.X_clean, _, _ = processed_data
                else:
                    raise ValueError("Falha ao gerar features para análise da estratégia de regras.")
            else: 
                self.log("... tipo: Estratégia de Machine Learning. Executando pipeline completo...")
                processed_data = preprocess_and_feature_engineer(self.df_historical_raw, self.strategy)
                if processed_data is None:
                    raise ValueError("O pipeline de engenharia de features não retornou dados.")
                self.X_clean, self.y_clean, features_used = processed_data
            
            if self.X_clean is None or self.y_clean is None:
                raise ValueError("Falha ao processar X ou y para análise.")
                
            common_index = self.X_clean.index.intersection(self.y_clean.index)
            self.X_clean = self.X_clean.loc[common_index]
            self.y_clean = self.y_clean.loc[common_index]

            self.df_historical_processed = self.X_clean.join(self.y_clean)
            self.current_data_identifier = (self.X_clean.shape, tuple(self.X_clean.columns))
            logger.info(f"Dados para análise prontos. Shape X: {self.X_clean.shape}, Shape y: {self.y_clean.shape}")

            self.log("Etapa 3: Atualizando painéis de análise visual...")
            
            if hasattr(self, 'desc_text') and self.desc_text:
                with pd.option_context('display.max_rows', 200, 'display.max_columns', None):
                    self._update_text_widget(self.desc_text, self.X_clean.describe(include='all').to_string())
            
            if hasattr(self, 'target_text') and self.target_text:
                counts = self.y_clean.value_counts(dropna=False)
                dist = self.y_clean.value_counts(normalize=True, dropna=False)
                target_name = self.strategy.get_target_variable_name()
                self._update_text_widget(self.target_text, f"Contagem de '{target_name}':\n{counts.to_string()}\n\nProporção:\n{dist.apply('{:.2%}'.format).to_string()}")
            
            numeric_features = sorted(self.X_clean.select_dtypes(include=np.number).columns.tolist())
            if numeric_features:
                self.feature_selector_combo.configure(values=numeric_features) 
                self.feature_selector_var.set(numeric_features[0])
                self.on_univar_feature_select()
            
            self.generate_correlation_heatmap()
            self.log("Carregamento e análise concluídos com sucesso.")

        except Exception as e:
            errmsg = f"Erro fatal no carregamento/análise: {e}"
            self.log(f"!!! ERRO: {errmsg}")
            logger.error(errmsg, exc_info=True)
            messagebox.showerror("Erro Fatal na Análise", f"{errmsg}\n\nVerifique os logs.", parent=self.parent)


    def on_univar_feature_select(self, event=None):

        if self.univar_update_job:
            try: self.parent.after_cancel(self.univar_update_job)
            except tk.TclError: pass 
        self.univar_update_job = self.parent.after(400, self._update_univariate_plots_task)


    def _update_univariate_plots_task(self):
        self.univar_update_job = None
        if self.X_clean is None or self.y_clean is None:
            return
        selected_feature = self.feature_selector_var.get()
        if not selected_feature or selected_feature not in self.X_clean.columns:
            return

        try:
            if self.fig_dist is None or not plt.fignum_exists(self.fig_dist.number):
                self.fig_dist, self.axes_dist = plt.subplots(
                    2, 1, figsize=(6, 5),
                    gridspec_kw={'height_ratios': [3, 1]},
                    sharex=True, constrained_layout=True
                )
            else:
                self.axes_dist[0].clear()
                self.axes_dist[1].clear()

            feature_data = self.X_clean[selected_feature].dropna()
            sns.histplot(feature_data, kde=True, ax=self.axes_dist[0])
            self.axes_dist[0].set_title(f'Distribuição de {selected_feature}')
            self.axes_dist[0].set_xlabel('')
            sns.boxplot(x=feature_data, ax=self.axes_dist[1])
            self.axes_dist[1].set_xlabel(selected_feature)

            self._embed_matplotlib_figure(self.fig_dist, self.dist_canvas_container)

        except Exception as e:
            logger.error(f"Erro ao gerar gráfico de distribuição para {selected_feature}: {e}", exc_info=True)
            self._clear_matplotlib_widget(self.dist_canvas_container)
            ctk.CTkLabel(self.dist_canvas_container, text="Erro ao gerar gráfico.").pack()

        try:
            if self.fig_dist_target is None or not plt.fignum_exists(self.fig_dist_target.number):
                self.fig_dist_target, self.ax_dist_target = plt.subplots(
                    1, 1, figsize=(6, 5), constrained_layout=True
                )
            else:
                self.ax_dist_target.clear()

            target_name = self.strategy.get_target_variable_name()
            plot_data = pd.DataFrame({
                'feature': self.X_clean[selected_feature],
                'target': self.y_clean
            })
            sns.boxplot(data=plot_data, x='target', y='feature', ax=self.ax_dist_target)
            self.ax_dist_target.set_title(f'{selected_feature} vs {target_name}')
            self.ax_dist_target.set_xlabel(f'{target_name} (0: Negativo, 1: Positivo)')
            self.ax_dist_target.set_ylabel(selected_feature)

            self._embed_matplotlib_figure(self.fig_dist_target, self.dist_target_canvas_container)

        except Exception as e:
            logger.error(f"Erro ao gerar gráfico vs alvo para {selected_feature}: {e}", exc_info=True)
            self._clear_matplotlib_widget(self.dist_target_canvas_container)
            ctk.CTkLabel(self.dist_target_canvas_container, text="Erro ao gerar gráfico.").pack()

    def generate_correlation_heatmap(self):
        if self.X_clean is None or self.y_clean is None:
            return
        try:
            if self.fig_corr is None or not plt.fignum_exists(self.fig_corr.number):
                self.fig_corr, self.ax_corr = plt.subplots(figsize=(9, 7), constrained_layout=True)
            else:
                self.ax_corr.clear()

            df_for_corr = self.X_clean.select_dtypes(include=np.number).copy()
            df_for_corr[self.strategy.get_target_variable_name()] = self.y_clean

            if df_for_corr.shape[1] < 2:
                self._clear_matplotlib_widget(self.corr_canvas_container)
                ctk.CTkLabel(self.corr_canvas_container, text="Features insuficientes para correlação.").pack()
                return

            corr_matrix = df_for_corr.corr()
            sns.heatmap(
                corr_matrix, annot=False, cmap='coolwarm', fmt=".2f",
                linewidths=.5, ax=self.ax_corr, cbar=True
            )
            self.ax_corr.set_title('Heatmap de Correlação (Features e Alvo)')
            plt.setp(self.ax_corr.get_xticklabels(), rotation=60, ha='right', fontsize=7)
            plt.setp(self.ax_corr.get_yticklabels(), rotation=0, fontsize=7)

            self._embed_matplotlib_figure(self.fig_corr, self.corr_canvas_container)

        except Exception as e:
            logger.error(f"Erro ao gerar heatmap de correlação: {e}", exc_info=True)
            self._clear_matplotlib_widget(self.corr_canvas_container)
            ctk.CTkLabel(self.corr_canvas_container, text="Erro ao gerar gráfico.").pack()

