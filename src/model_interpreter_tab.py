# --- src/model_interpreter_tab.py ---
# NOVA ABA DEDICADA À INTERPRETAÇÃO DE MODELOS

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText
import sys, os, pandas as pd, numpy as np, time, datetime, io, traceback, warnings, threading, queue, joblib, hashlib
from typing import Optional, List, Dict, Any, Tuple

# Gráficos
import matplotlib; matplotlib.use('TkAgg'); import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns

# Modelos e Interpretabilidade
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression; from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB; from sklearn.neighbors import KNeighborsClassifier
try: import lightgbm as lgb; LGBM_AVAILABLE = True
except ImportError: lgb = None; LGBM_AVAILABLE = False
try:
    with warnings.catch_warnings(): warnings.simplefilter("ignore"); import shap
    SHAP_AVAILABLE = True
except ImportError: shap = None; SHAP_AVAILABLE = False
from sklearn.inspection import PartialDependenceDisplay, permutation_importance # Importa permutation_importance

# Path Setup
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path: sys.path.insert(0, CURRENT_DIR)
if BASE_DIR not in sys.path: sys.path.insert(0, BASE_DIR)

# Imports do Projeto
try:
    from config import (RANDOM_STATE, BEST_F1_MODEL_SAVE_PATH, BEST_ROI_MODEL_SAVE_PATH,
                        MODEL_ID_F1, MODEL_ID_ROI) # Configs de modelo
    from logger_config import setup_logger
    from predictor import load_model_scaler_features # Para carregar modelos
    # Referência à classe da outra aba (para tipagem) - Cuidado com import circular!
    # É melhor evitar a importação direta da classe aqui. Usaremos duck typing.
    # from feature_analyzer_tab import FeatureAnalyzerApp
except ImportError as e:
     import logging; logger = logging.getLogger(__name__); logger.critical(f"Import Error CRÍTICO em model_interpreter_tab.py: {e}", exc_info=True)
     try: root_err = tk.Tk(); root_err.withdraw(); messagebox.showerror("Import Error (Interpreter Tab)", f"Failed...\n{e}"); root_err.destroy()
     except Exception: print(f"CRITICAL Import Error: {e}")
     sys.exit(1)

logger = setup_logger("ModelInterpreterApp") # Logger específico
if not SHAP_AVAILABLE: logger.warning("Biblioteca 'shap' não encontrada.")

# Diretório Cache SHAP
SHAP_CACHE_DIR = os.path.join(BASE_DIR, 'data', 'shap_cache')
os.makedirs(SHAP_CACHE_DIR, exist_ok=True)

class ModelInterpreterApp:

    def __init__(self, parent_frame, main_root, analyzer_app_ref): # Recebe referência
        self.parent = parent_frame
        self.main_tk_root = main_root
        self.analyzer_app = analyzer_app_ref # <<< Armazena referência à Aba 3
        self.gui_queue = queue.Queue()
        self.shap_thread: Optional[threading.Thread] = None
        self.stop_processing_queue = False

        # Widgets Texto & Status
        self.model_importance_text: Optional[tk.Text] = None
        self.status_label: Optional[ttk.Label] = None # Pode ter um status local

        # Widgets Seleção
        self.model_selector_var = tk.StringVar()
        self.model_selector_combo: Optional[ttk.Combobox] = None
        self.pdp_feature_selector_var = tk.StringVar()
        self.pdp_feature_selector_combo: Optional[ttk.Combobox] = None
        self.load_trained_models_button: Optional[ttk.Button] = None

        # Figuras, Eixos, Canvas (SHAP/PDP)
        self.shap_summary_canvas_widget: Optional[FigureCanvasTkAgg] = None
        self.pdp_canvas_widget: Optional[FigureCanvasTkAgg] = None
        self.fig_shap_summary: Optional[plt.Figure] = None; self.ax_shap = None
        self.fig_pdp: Optional[plt.Figure] = None; self.ax_pdp = None
        self.shap_summary_container: Optional[ttk.Frame] = None
        self.pdp_container: Optional[ttk.Frame] = None

        # Modelos Carregados (nesta aba)
        self.loaded_trained_models: Dict[str, Dict] = {}

        # Debounce ID
        self.pdp_update_job = None

        self.create_widgets()
        self.process_gui_queue()

    def create_widgets(self):
        """Cria widgets APENAS para interpretação do modelo."""
        # --- Canvas e Scrollbar (ainda útil se o conteúdo crescer) ---
        canvas = tk.Canvas(self.parent, borderwidth=0, highlightthickness=0); scrollbar = ttk.Scrollbar(self.parent, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas); self.scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw"); canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5); scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=5)

        # --- Frame Principal dentro do Canvas ---
        main_content_frame = ttk.Frame(self.scrollable_frame)
        main_content_frame.pack(fill=tk.BOTH, expand=True)

        # --- Controles de Modelo ---
        model_controls_frame = ttk.Frame(main_content_frame)
        model_controls_frame.pack(fill=tk.X, pady=10, padx=10)
        self.load_trained_models_button = ttk.Button(model_controls_frame, text="Carregar Modelos Treinados", command=self.load_trained_models); self.load_trained_models_button.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(model_controls_frame, text="Analisar Modelo:").pack(side=tk.LEFT, padx=(10, 5)); self.model_selector_combo = ttk.Combobox(model_controls_frame, textvariable=self.model_selector_var, state="disabled", width=25); self.model_selector_combo.pack(side=tk.LEFT); self.model_selector_combo.bind("<<ComboboxSelected>>", self.update_model_specific_analysis)
        self.status_label = ttk.Label(model_controls_frame, text="Carregue os modelos..."); self.status_label.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True) # Status local

        # --- Importância ---
        importance_frame = ttk.LabelFrame(main_content_frame, text=" Importância de Features (Modelo Selecionado) ", padding=10); importance_frame.pack(fill=tk.X, pady=5, padx=10)
        self.model_importance_text = self._create_scrolled_text(importance_frame, height=15); # Aumenta altura talvez
        self._update_text_widget(self.model_importance_text, "Carregue modelos treinados...")

        # --- Gráficos SHAP e PDP ---
        model_plots_frame = ttk.Frame(main_content_frame); model_plots_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0), padx=10)
        # SHAP
        self.shap_summary_container = ttk.LabelFrame(model_plots_frame, text=" SHAP Summary Plot ", padding=5); self.shap_summary_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,5), pady=5); ttk.Label(self.shap_summary_container, text="Selecione modelo...").pack()
        if not SHAP_AVAILABLE: ttk.Label(self.shap_summary_container, text="(Biblioteca 'shap' não instalada)", foreground="red").pack()
        # PDP
        pdp_outer_container = ttk.LabelFrame(model_plots_frame, text=" Partial Dependence Plot (PDP) ", padding=5); pdp_outer_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5,0), pady=5)
        pdp_controls_frame = ttk.Frame(pdp_outer_container); pdp_controls_frame.pack(fill=tk.X, pady=(0, 5)); ttk.Label(pdp_controls_frame, text="Feature para PDP:").pack(side=tk.LEFT, padx=(0, 5))
        self.pdp_feature_selector_combo = ttk.Combobox(pdp_controls_frame, textvariable=self.pdp_feature_selector_var, state="disabled", width=25); self.pdp_feature_selector_combo.pack(side=tk.LEFT); self.pdp_feature_selector_combo.bind("<<ComboboxSelected>>", self.on_pdp_feature_select)
        self.pdp_container = ttk.Frame(pdp_outer_container); self.pdp_container.pack(fill=tk.BOTH, expand=True); ttk.Label(self.pdp_container, text="Selecione modelo/feature...").pack()

    # --- Funções Helper (copiadas/adaptadas de feature_analyzer_tab) ---
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
        if container_widget == self.shap_summary_container: canvas_attr_name = 'shap_summary_canvas_widget'
        elif container_widget == self.pdp_container: canvas_attr_name = 'pdp_canvas_widget'
        else: logger.warning(f"Container desconhecido em _embed_matplotlib_figure: {container_widget}")
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

    def log(self, message: str): # Log local da aba
        logger.info(f"[GUI Interpreter] {message}")
        try:
            if self.status_label and self.status_label.winfo_exists(): self.status_label.config(text=message[:100])
        except tk.TclError: pass
        except Exception as e: logger.error(f"Erro status label (Interpreter): {e}", exc_info=True)

    # --- Funções de Carregamento, Cálculo e Atualização ---
    def load_trained_models(self):
        if not hasattr(self.analyzer_app, 'current_data_identifier') or self.analyzer_app.current_data_identifier is None:
             messagebox.showwarning("Dados Ausentes", "Carregue e processe dados na Aba 'Análise Features' primeiro.", parent=self.parent); return
        self.log("Tentando carregar modelos treinados...")
        self.loaded_trained_models = {} # Limpa cache
        loaded_ids = []; all_feature_names = set()
        model_paths_to_try = {MODEL_ID_F1: BEST_F1_MODEL_SAVE_PATH, MODEL_ID_ROI: BEST_ROI_MODEL_SAVE_PATH}

        for model_id, model_path in model_paths_to_try.items():
            self.log(f" -> Carregando {model_id}...");
            if not os.path.exists(model_path): self.log(f"    Arquivo modelo não encontrado."); continue
            load_result = load_model_scaler_features(model_path)
            if load_result:
                model, scaler, calib, ev_thr, f1_thr, features, params, metrics, ts = load_result
                if model and features:
                    imp_df = self._calculate_feature_importance(model, features)
                    if imp_df is None: logger.warning(f"Não foi possível calcular importância para {model_id}.")
                    model_entry = {'model': model, 'scaler': scaler, 'calibrator': calib, 'optimal_ev_threshold': ev_thr, 'optimal_f1_threshold': f1_thr, 'features': features, 'params': params, 'metrics': metrics, 'timestamp': ts, 'path': model_path, 'importances': imp_df}
                    self.loaded_trained_models[model_id] = model_entry
                    loaded_ids.append(model_id); all_feature_names.update(features); self.log(f"    -> Modelo {model_id} carregado.")
                    shap_cache_path = self._get_shap_cache_path(model_id)
                    if os.path.exists(shap_cache_path):
                        self.log(f"    -> Tentando carregar cache SHAP...");
                        try:
                            shap_cache_data = joblib.load(shap_cache_path)
                            if isinstance(shap_cache_data, dict) and \
                               shap_cache_data.get('shap_data_identifier') == self.analyzer_app.current_data_identifier and \
                               shap_cache_data.get('shap_model_features') == features:
                                self.loaded_trained_models[model_id].update(shap_cache_data); self.log(f"    -> Cache SHAP validado.")
                            else: self.log(f"    -> Cache SHAP inválido.")
                        except Exception as e_load_shap: logger.error(f"Erro carregar cache SHAP: {e_load_shap}", exc_info=True); self.log(f"    -> Erro cache SHAP.")
                    else: self.log(f"    -> Cache SHAP não encontrado.")
                else: self.log(f"    -> Falha (objeto modelo inválido).")
            else: self.log(f"    -> Falha ao carregar {model_id}.")
        if loaded_ids:
            self.log(f"Modelos carregados: {', '.join(loaded_ids)}"); self.model_selector_combo.config(values=loaded_ids, state="readonly"); self.model_selector_var.set(loaded_ids[0])
            pdp_features = sorted(list(all_feature_names));
            if pdp_features: self.pdp_feature_selector_combo.config(values=pdp_features, state="readonly"); self.pdp_feature_selector_var.set(pdp_features[0] if pdp_features else "")
            else: self.pdp_feature_selector_combo.config(values=[], state="disabled"); self.pdp_feature_selector_var.set("")
            self.update_model_specific_analysis()
        else:
             self.log("Nenhum modelo treinado válido."); self.model_selector_combo.config(values=[], state="disabled"); self.model_selector_var.set("")
             self.pdp_feature_selector_combo.config(values=[], state="disabled"); self.pdp_feature_selector_var.set("")
             self._update_text_widget(self.model_importance_text, "Nenhum modelo carregado.")
             self._clear_matplotlib_widget(self.shap_summary_container); ttk.Label(self.shap_summary_container, text="Nenhum modelo.").pack()
             self._clear_matplotlib_widget(self.pdp_container); ttk.Label(self.pdp_container, text="Nenhum modelo.").pack()
             messagebox.showinfo("Modelos Não Encontrados", "Não foi possível carregar modelos.", parent=self.parent)

    def _get_shap_cache_path(self, model_id: str) -> str:
        safe_model_id = model_id.replace(" ", "_").replace("(", "").replace(")", "").lower()
        data_hash = "no_data"
        if hasattr(self.analyzer_app, 'current_data_identifier') and self.analyzer_app.current_data_identifier:
            try: id_str = str(self.analyzer_app.current_data_identifier); data_hash = hashlib.sha1(id_str.encode()).hexdigest()[:8]
            except Exception: data_hash = "error_hash"
        filename = f"shap_cache_{safe_model_id}_data_{data_hash}.joblib"
        return os.path.join(SHAP_CACHE_DIR, filename)

    def _calculate_feature_importance(self, model, model_features: List[str]) -> Optional[pd.DataFrame]:
        importances = None; model_type_name = model.__class__.__name__
        logger.debug(f"Calculando importância para {model_type_name}...")
        if not hasattr(self.analyzer_app, 'X_clean') or self.analyzer_app.X_clean is None or not hasattr(self.analyzer_app, 'y_clean') or self.analyzer_app.y_clean is None: logger.error("Dados X_clean/y_clean não disponíveis."); return None
        if not all(f in self.analyzer_app.X_clean.columns for f in model_features): logger.error("Features modelo ausentes X_clean."); return None
        try: X_model_subset = self.analyzer_app.X_clean[model_features].copy(); y_aligned = self.analyzer_app.y_clean.loc[X_model_subset.index].copy()
        except Exception as e: logger.error(f"Erro criar subset importância: {e}"); return None
        try:
            if hasattr(model, 'feature_importances_'): importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                 if model.coef_.shape[0] == 1: importances = np.abs(model.coef_[0])
                 else: importances = np.mean(np.abs(model.coef_), axis=0)
            else:
                logger.warning(f"Tentando Permutation Importance para {model_type_name}...");
                try: result = permutation_importance(model, X_model_subset, y_aligned, n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1, scoring='accuracy'); importances = result.importances_mean; logger.info(" -> Permutation Importance OK.")
                except Exception as e_perm: logger.error(f"Erro Permutation Importance: {e_perm}", exc_info=True); return None
            if importances is not None and len(importances) == len(model_features): imp_df = pd.DataFrame({'Feature': model_features, 'Importance': importances}); return imp_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
            elif importances is not None: logger.error(f"Mismatch importância/features {model_type_name}"); return None
            else: logger.warning(f"Não foi possível obter importâncias para {model_type_name}."); return None
        except Exception as e: logger.error(f"Erro GERAL calc importância {model_type_name}: {e}", exc_info=True); return None

    def update_model_specific_analysis(self, event=None):
         selected_id = self.model_selector_var.get()
         if not selected_id or selected_id not in self.loaded_trained_models: # ... (limpa GUI e retorna) ...
             self._update_text_widget(self.model_importance_text, "Selecione modelo."); self._clear_matplotlib_widget(self.shap_summary_container); ttk.Label(self.shap_summary_container, text="Selecione.").pack(); self._clear_matplotlib_widget(self.pdp_container); ttk.Label(self.pdp_container, text="Selecione.").pack()
             try: self.pdp_feature_selector_combo.config(values=[], state="disabled"); self.pdp_feature_selector_var.set("")
             except: pass; return
         self.log(f"Atualizando análise para: {selected_id}"); model_data = self.loaded_trained_models[selected_id]; importances_df = model_data.get('importances')
         if importances_df is not None and not importances_df.empty: # Exibe importância
             try: self._update_text_widget(self.model_importance_text, importances_df.round(5).to_string(index=True))
             except Exception as e: logger.error(f"Erro formatar importâncias: {e}"); self._update_text_widget(self.model_importance_text, f"Erro.")
         else: self._update_text_widget(self.model_importance_text, f"Importância não disponível para {selected_id} ({model_data.get('model').__class__.__name__}).")
         self._start_shap_calculation(model_data) # Inicia SHAP (usa cache ou calcula)
         pdp_features = model_data.get('features', []) # Atualiza dropdown PDP
         if pdp_features:
             self.pdp_feature_selector_combo.config(values=pdp_features, state="readonly"); current_pdp_feature = self.pdp_feature_selector_var.get()
             if current_pdp_feature not in pdp_features: self.pdp_feature_selector_var.set(pdp_features[0] if pdp_features else "")
             self.on_pdp_feature_select() # Dispara debouncer PDP
         else: self.pdp_feature_selector_combo.config(values=[], state="disabled"); self.pdp_feature_selector_var.set(""); self._clear_matplotlib_widget(self.pdp_container); ttk.Label(self.pdp_container, text="Features não encontradas.").pack()

    def _start_shap_calculation(self, model_data: Dict):
        if not SHAP_AVAILABLE: self._clear_matplotlib_widget(self.shap_summary_container); ttk.Label(self.shap_summary_container, text="'shap' não instalado.").pack(); return
        if not hasattr(self.analyzer_app, 'X_clean') or self.analyzer_app.X_clean is None: self._clear_matplotlib_widget(self.shap_summary_container); ttk.Label(self.shap_summary_container, text="Dados não carregados.").pack(); return
        model = model_data.get('model'); model_id = self.model_selector_var.get(); features_for_model = model_data.get('features')
        if model is None or features_for_model is None: return
        if model_id not in self.loaded_trained_models or id(self.loaded_trained_models[model_id]) != id(model_data): return

        use_cache = False
        if 'shap_values' in model_data and 'shap_data_identifier' in model_data: # Verifica cache
            cached_data_id = model_data['shap_data_identifier']; cached_features = model_data.get('shap_model_features')
            if cached_data_id == self.analyzer_app.current_data_identifier and cached_features == features_for_model: self.log(f"Reutilizando SHAP cacheado {model_id}."); use_cache = True
            else: 
                self.log(f"Cache SHAP {model_id} invalidado."); # ... (limpa chaves do cache) ...
                for key in ['shap_values', 'X_sample', 'shap_data_identifier', 'shap_model_features', 'shap_used_kernel', 'shap_n_samples']: model_data.pop(key, None)
        if use_cache: # Plota do cache
            shap_payload_cached = {'model_id': model_id, 'shap_values': model_data['shap_values'], 'X_sample': model_data['X_sample'], 'model_type_name': model.__class__.__name__, 'used_kernel': model_data.get('shap_used_kernel', False), 'n_samples': model_data.get('shap_n_samples', '?')}
            self._plot_shap_summary(shap_payload_cached); return

        model_type_name = model.__class__.__name__; kernel_models = ['SVC', 'KNeighborsClassifier', 'GaussianNB']
        use_kernel_explainer = model_type_name in kernel_models
        if use_kernel_explainer: # Roda em background
            self.log(f"Iniciando SHAP {model_type_name} background..."); self._clear_matplotlib_widget(self.shap_summary_container); ttk.Label(self.shap_summary_container, text=f"Calculando SHAP {model_type_name}...\n(Isto pode demorar...)").pack()
            if self.shap_thread and self.shap_thread.is_alive(): self.log("Cálculo SHAP anterior em andamento..."); return
            self.shap_thread = threading.Thread(target=self._run_shap_task, args=(model_id, use_kernel_explainer), daemon=True); self.shap_thread.start()
        else: # Roda direto
            self.log(f"Calculando SHAP {model_type_name} diretamente..."); self._clear_matplotlib_widget(self.shap_summary_container); ttk.Label(self.shap_summary_container, text=f"Calculando SHAP {model_type_name}...").pack()
            self.parent.update_idletasks(); self._run_shap_task(model_id, use_kernel_explainer)

    def _run_shap_task(self, target_model_id: str, use_kernel_explainer: bool):
        shap_values_for_plot = None; X_shap_sample = None;
        if target_model_id not in self.loaded_trained_models: logger.error(f"Thread SHAP: Modelo {target_model_id} não carregado."); self.gui_queue.put(("shap_error", {'model_id': target_model_id, 'error': "Dados modelo não carregados."})); return
        model_data_target = self.loaded_trained_models[target_model_id]
        try:
            model = model_data_target.get('model'); scaler = model_data_target.get('scaler'); features = model_data_target.get('features')
            if not hasattr(self.analyzer_app, 'X_clean') or self.analyzer_app.X_clean is None: raise ValueError("Dados X_clean ausentes (Thread SHAP)")
            if not all([model, features]): raise ValueError("Modelo/Features ausentes (Thread SHAP)")
            X_analysis = self.analyzer_app.X_clean[features].copy();
            if scaler: X_analysis=X_analysis.replace([np.inf,-np.inf],np.nan); X_analysis.fillna(X_analysis.median(),inplace=True); X_scaled_np=scaler.transform(X_analysis); X_shap_ready=pd.DataFrame(X_scaled_np,index=X_analysis.index,columns=features)
            else: X_shap_ready=X_analysis
            N_SHAP_SAMPLES=50 if use_kernel_explainer else 500; N_KERNEL_BG=25 if use_kernel_explainer else 50
            if len(X_shap_ready)>N_SHAP_SAMPLES: X_shap_sample=shap.sample(X_shap_ready, N_SHAP_SAMPLES, random_state=RANDOM_STATE)
            else: X_shap_sample=X_shap_ready
            explainer=None; shap_values=None; model_type_name=model.__class__.__name__
            # --- Lógica explainer ---
            if use_kernel_explainer:
                 X_kernel_bg=shap.sample(X_shap_ready, N_KERNEL_BG, random_state=RANDOM_STATE)
                 def model_predict_proba_pos(data):
                    if not isinstance(data,pd.DataFrame): data_df=pd.DataFrame(data,columns=features)
                    else: data_df=data
                    try: return model.predict_proba(data_df)[:,1]
                    except Exception as e: logger.error(f"Erro pred kernel: {e}"); return np.full(len(data_df),0.5)
                 explainer=shap.KernelExplainer(model_predict_proba_pos,X_kernel_bg); shap_values=explainer.shap_values(X_shap_sample)
            elif model_type_name in ['RandomForestClassifier','LGBMClassifier','GradientBoostingClassifier'] and LGBM_AVAILABLE and lgb is not None: explainer=shap.TreeExplainer(model); shap_values=explainer.shap_values(X_shap_sample)
            elif model_type_name in ['LogisticRegression']: explainer=shap.LinearExplainer(model,X_shap_sample); shap_values=explainer.shap_values(X_shap_sample)
            else: # Fallback
                 logger.warning(f"Tipo {model_type_name} fallback KernelExplainer."); use_kernel_explainer=True; N_SHAP_SAMPLES=50; N_KERNEL_BG=25; #... (resto fallback logic) ...
                 if len(X_shap_ready)>N_SHAP_SAMPLES: X_shap_sample=shap.sample(X_shap_ready, N_SHAP_SAMPLES, random_state=RANDOM_STATE)
                 else: X_shap_sample=X_shap_ready
                 X_kernel_bg=shap.sample(X_shap_ready, N_KERNEL_BG, random_state=RANDOM_STATE)
                 def model_predict_proba_pos_fb(data):
                     if not isinstance(data,pd.DataFrame): data_df=pd.DataFrame(data,columns=features)
                     else: data_df=data
                     try: return model.predict_proba(data_df)[:,1]
                     except Exception as e: logger.error(f"Erro pred kernel fb: {e}"); return np.full(len(data_df),0.5)
                 explainer=shap.KernelExplainer(model_predict_proba_pos_fb, X_kernel_bg); shap_values=explainer.shap_values(X_shap_sample)
            # --- Processamento shap_values ---
            if shap_values is not None:
                if isinstance(shap_values, list) and len(shap_values)==2: shap_values_for_plot=shap_values[1]
                elif isinstance(shap_values, np.ndarray):
                     if len(shap_values.shape)==3: shap_values_for_plot=shap_values[:,:,1]
                     elif len(shap_values.shape)==2: shap_values_for_plot=shap_values
                     else: shap_values_for_plot=shap_values
                else: shap_values_for_plot=shap_values
            # --- Fim processamento ---
            if shap_values_for_plot is not None and X_shap_sample is not None:
                shap_cache_data = {'shap_values': shap_values_for_plot, 'X_sample': X_shap_sample, 'shap_data_identifier': self.analyzer_app.current_data_identifier, 'shap_model_features': features, 'shap_used_kernel': use_kernel_explainer, 'shap_n_samples': N_SHAP_SAMPLES}
                shap_cache_path = self._get_shap_cache_path(target_model_id)
                try: joblib.dump(shap_cache_data, shap_cache_path); self.log(f"Cache SHAP salvo disco {target_model_id}.")
                except Exception as e_save: logger.error(f"Erro salvar cache SHAP disco: {e_save}", exc_info=True)
                cache_updated_mem = False
                if target_model_id in self.loaded_trained_models:
                    try: self.loaded_trained_models[target_model_id].update(shap_cache_data); cache_updated_mem = 'shap_values' in self.loaded_trained_models[target_model_id]; logger.info(f"Cache SHAP memória atualizado {target_model_id}.")
                    except Exception as e_mem: logger.error(f"Erro atualizar cache SHAP memória: {e_mem}", exc_info=True)
                else: logger.warning(f"Modelo {target_model_id} não no dict ao salvar cache mem SHAP.")
                if cache_updated_mem: self.gui_queue.put(("shap_result_calculated", {'model_id': target_model_id}))
                else: self.gui_queue.put(("shap_error", {'model_id': target_model_id, 'error': "Falha atualizar cache SHAP memória."}))
            else: raise ValueError("Cálculo SHAP não produziu resultados válidos.")
        except Exception as e: logger.error(f"Erro thread SHAP {target_model_id}: {e}", exc_info=True); self.gui_queue.put(("shap_error", {'model_id': target_model_id, 'error': str(e)}))

    # --- Funções de Plotagem (SHAP/PDP - Reutilizam Eixos) ---
    def _plot_shap_summary(self, shap_data: Dict):
        model_id_for_plot = shap_data.get('model_id', 'Desconhecido'); self._clear_matplotlib_widget(self.shap_summary_container)
        try:
            shap_values = shap_data['shap_values']; X_sample = shap_data['X_sample']
            model_type = shap_data['model_type_name']; used_kernel = shap_data['used_kernel']; n_samples = shap_data['n_samples']
            model_info = self.loaded_trained_models.get(model_id_for_plot, {}); feature_names_plot = X_sample.columns if isinstance(X_sample, pd.DataFrame) else model_info.get('features')
            if self.fig_shap_summary is None or not plt.fignum_exists(self.fig_shap_summary.number): self.fig_shap_summary, self.ax_shap = plt.subplots(figsize=(8, 6))
            else: self.ax_shap.clear()
            plot_title = f'SHAP Summary ({model_type})';
            if used_kernel: plot_title += f'\n(KernelExplainer - {n_samples} amostras)'
            # Passa eixo para SHAP plot
            shap.summary_plot(shap_values, X_sample, plot_type="dot", show=False, max_display=15, feature_names=feature_names_plot, axes=self.ax_shap)
            self.ax_shap.set_title(plot_title)
            self.fig_shap_summary.tight_layout(pad=1.0)
            if self.shap_summary_canvas_widget is None: # Cria canvas só na primeira vez
                 self.shap_summary_canvas_widget = self._embed_matplotlib_figure(self.fig_shap_summary, self.shap_summary_container)
            elif self.shap_summary_canvas_widget: self.shap_summary_canvas_widget.draw_idle() # Redesenha
        except Exception as e: logger.error(f"Erro plotar SHAP {model_id_for_plot}: {e}", exc_info=True); ttk.Label(self.shap_summary_container, text=f"Erro plotar SHAP:\n{e}").pack()

    def on_pdp_feature_select(self, event=None):
        """Callback com debounce para seleção de feature PDP."""
        if self.pdp_update_job:
            try: self.main_tk_root.after_cancel(self.pdp_update_job)
            except tk.TclError: pass
        self.pdp_update_job = self.main_tk_root.after(400, self._update_pdp_plot_task)

    def _update_pdp_plot_task(self):
        """Tarefa real de atualização do gráfico PDP (chamada pelo debounce)."""
        self.pdp_update_job = None
        selected_model_id = self.model_selector_var.get(); selected_pdp_feature = self.pdp_feature_selector_var.get()
        if not hasattr(self.analyzer_app, 'X_clean') or self.analyzer_app.X_clean is None: return
        if not selected_model_id or not selected_pdp_feature: return
        model_data = self.loaded_trained_models.get(selected_model_id);
        if not model_data: return
        model = model_data.get('model'); scaler = model_data.get('scaler'); features = model_data.get('features')
        if not all([model, features]) or selected_pdp_feature not in features: self._clear_matplotlib_widget(self.pdp_container); ttk.Label(self.pdp_container, text="Modelo/Features inválidos.").pack(); return

        self._clear_matplotlib_widget(self.pdp_container); ttk.Label(self.pdp_container, text=f"Calculando PDP...").pack(); self.parent.update_idletasks()
        try:
            X_analysis = self.analyzer_app.X_clean[features].copy()
            if scaler: X_analysis=X_analysis.replace([np.inf,-np.inf],np.nan); X_analysis.fillna(X_analysis.median(),inplace=True); X_scaled_np=scaler.transform(X_analysis); X_pdp_ready=pd.DataFrame(X_scaled_np, index=X_analysis.index, columns=features)
            else: X_pdp_ready=X_analysis

            if self.fig_pdp is None or not plt.fignum_exists(self.fig_pdp.number): self.fig_pdp, self.ax_pdp = plt.subplots(figsize=(7, 5), constrained_layout=True); self.pdp_canvas_widget = self._embed_matplotlib_figure(self.fig_pdp, self.pdp_container)
            else: self.ax_pdp.clear()

            try: PartialDependenceDisplay.from_estimator(estimator=model, X=X_pdp_ready, features=[selected_pdp_feature], kind='average', target=1, ax=self.ax_pdp, line_kw={"color": "green", "linewidth": 2.5})
            except TypeError as te:
                 if 'target' in str(te): logger.warning(f"TypeError 'target' PDP {model.__class__.__name__}."); PartialDependenceDisplay.from_estimator(estimator=model, X=X_pdp_ready, features=[selected_pdp_feature], kind='average', ax=self.ax_pdp, line_kw={"color": "orange", "linewidth": 2.5})
                 else: raise te
            except Exception as e_pdp_disp: raise e_pdp_disp
            self.ax_pdp.set_title(f'Partial Dependence Plot\n({selected_pdp_feature} vs Prob Empate)'); self.ax_pdp.set_xlabel(selected_pdp_feature); self.ax_pdp.set_ylabel('Dependência Parcial'); self.ax_pdp.grid(True, linestyle='--', alpha=0.6);
            if self.pdp_canvas_widget: self.pdp_canvas_widget.draw_idle()
        except Exception as e: logger.error(f"Erro gerar PDP {selected_pdp_feature}: {e}", exc_info=True); self._clear_matplotlib_widget(self.pdp_container); ttk.Label(self.pdp_container, text=f"Erro gerar PDP:\n{e}").pack()


    # --- process_gui_queue ---
    # (Como na versão anterior, tratando shap_result_calculated e shap_error)
    def process_gui_queue(self):
        if self.stop_processing_queue: logger.debug("Interpreter: Parando fila GUI."); return
        try:
            while True:
                try: message_type, payload = self.gui_queue.get_nowait()
                except queue.Empty: break
                except Exception as e: logger.error(f"Erro get fila GUI (Interpreter): {e}", exc_info=True); break
                try:
                    if message_type == "shap_result_calculated":
                        finished_model_id = payload.get('model_id'); current_model_id = self.model_selector_var.get()
                        if finished_model_id == current_model_id and finished_model_id in self.loaded_trained_models:
                            model_data = self.loaded_trained_models[finished_model_id]
                            if 'shap_values' in model_data and model_data['shap_values'] is not None and 'X_sample' in model_data and model_data['X_sample'] is not None:
                                shap_plot_payload = {'model_id': finished_model_id, 'shap_values': model_data['shap_values'], 'X_sample': model_data['X_sample'], 'model_type_name': model_data['model'].__class__.__name__, 'used_kernel': model_data.get('shap_used_kernel', False), 'n_samples': model_data.get('shap_n_samples', '?')}
                                self._plot_shap_summary(shap_plot_payload)
                            else: logger.error(f"Erro Plot SHAP: Cache inválido {finished_model_id}."); self._clear_matplotlib_widget(self.shap_summary_container); ttk.Label(self.shap_summary_container, text=f"Erro cache SHAP.").pack()
                        else: self.log(f"Ignorando resultado SHAP obsoleto {finished_model_id}.")
                    elif message_type == "shap_error":
                         error_model_id = payload.get('model_id', 'Desconhecido'); error_msg = payload.get('error', 'Erro desconhecido')
                         self.log(f"Erro cálculo SHAP {error_model_id}: {error_msg}")
                         if error_model_id == self.model_selector_var.get(): self._clear_matplotlib_widget(self.shap_summary_container); ttk.Label(self.shap_summary_container, text=f"Erro SHAP:\n{error_msg}").pack()
                    else: logger.warning(f"Tipo msg GUI desconhecido (Interpreter): {message_type}")
                except Exception as e_proc_msg: logger.error(f"Erro processar msg GUI tipo '{message_type}' (Interpreter): {e_proc_msg}", exc_info=True)
        except Exception as e_loop: logger.error(f"Erro CRÍTICO loop fila GUI (Interpreter): {e_loop}", exc_info=True)
        finally:
            if not self.stop_processing_queue:
                 try:
                     if hasattr(self.main_tk_root, 'winfo_exists') and self.main_tk_root.winfo_exists(): self.main_tk_root.after(100, self.process_gui_queue)
                 except Exception as e:
                     if not self.stop_processing_queue: logger.error(f"Erro reagendar fila GUI (Interpreter): {e}")