import tkinter as tk
import customtkinter as ctk
from tkinter import ttk, messagebox
import sys, os, pandas as pd, numpy as np, warnings, threading, queue, joblib, hashlib
from typing import Optional, List, Dict, Any

import matplotlib; matplotlib.use('TkAgg'); import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from PIL import Image, ImageTk

try: import lightgbm as lgb; LGBM_AVAILABLE = True
except ImportError: lgb = None; LGBM_AVAILABLE = False
try:
    with warnings.catch_warnings(): warnings.simplefilter("ignore"); import shap
    SHAP_AVAILABLE = True
except ImportError: shap = None; SHAP_AVAILABLE = False
from sklearn.inspection import PartialDependenceDisplay, permutation_importance

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)); BASE_DIR = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path: sys.path.insert(0, CURRENT_DIR);
if BASE_DIR not in sys.path: sys.path.insert(0, BASE_DIR)

try:
    from config import (RANDOM_STATE,
                        DATA_DIR) 
    from logger_config import setup_logger
    from predictor import load_model_scaler_features
except ImportError as e:
     import logging; logger = logging.getLogger(__name__); logger.critical(f"Import Error CRÍTICO: {e}", exc_info=True)
     try: root_err = tk.Tk(); root_err.withdraw(); messagebox.showerror("Import Error (Interpreter)", f"Failed...\n{e}"); root_err.destroy()
     except Exception: print(f"CRITICAL Import Error: {e}")
     sys.exit(1)

from strategies.base_strategy import BettingStrategy

logger = setup_logger("ModelInterpreterApp")
if not SHAP_AVAILABLE: logger.warning("Biblioteca 'shap' não encontrada.")

SHAP_CACHE_DIR = os.path.join(BASE_DIR, 'data', 'shap_cache'); os.makedirs(SHAP_CACHE_DIR, exist_ok=True)
PDP_CACHE_DIR = os.path.join(BASE_DIR, 'data', 'pdp_cache'); os.makedirs(PDP_CACHE_DIR, exist_ok=True)
N_SHAP_SAMPLES_FAST = 500; N_SHAP_SAMPLES_SLOW = 50; N_KERNEL_BG = 25; N_PDP_SAMPLES = 1000

class ModelInterpreterApp:

    def __init__(self, parent_frame, main_root, analyzer_app_ref, strategy: BettingStrategy):
        self.parent = parent_frame
        self.main_tk_root = main_root
        self.analyzer_app = analyzer_app_ref
        self.strategy = strategy
        self.gui_queue = queue.Queue()
        self.stop_processing_queue = False
        self.shap_thread: Optional[threading.Thread] = None

        self.X_analysis_data: Optional[pd.DataFrame] = None
        self.current_data_identifier = None
        self.loaded_trained_models: Dict[str, Dict] = {}

        self.model_importance_text=None; 
        self.status_label=None;
        self.model_selector_var=tk.StringVar(); 
        self.model_selector_combo=None;
        self.pdp_feature_selector_var=tk.StringVar(); 
        self.pdp_feature_selector_combo=None; 
        self.load_trained_models_button=None;
        self.shap_summary_canvas_widget=None; 
        self.pdp_canvas_widget=None; self.fig_shap_summary=None; self.ax_shap=None;
        self.fig_pdp=None; self.ax_pdp=None; 
        self.shap_summary_container=None; 
        self.pdp_container=None;
        self.pdp_update_job = None

        self.create_widgets()
        self.process_gui_queue()

    def create_widgets(self):

        canvas = tk.Canvas(self.parent, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.parent, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ctk.CTkFrame(canvas)
        self.scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=5)

        main_content_frame = ctk.CTkFrame(self.scrollable_frame)
        main_content_frame.pack(fill=tk.BOTH, expand=True)

        model_controls_frame = ctk.CTkFrame(main_content_frame)
        model_controls_frame.pack(fill=tk.X, pady=10, padx=10)
        self.load_trained_models_button = ctk.CTkButton(model_controls_frame, text="Carregar Modelos e Dados p/ Análise", command=self.load_models_and_data)
        self.load_trained_models_button.pack(side=tk.LEFT, padx=(0, 10))
        ctk.CTkLabel(model_controls_frame, text="Analisar Modelo:").pack(side=tk.LEFT, padx=(10, 5))
        self.model_selector_combo = ctk.CTkComboBox(model_controls_frame, variable=self.model_selector_var, state="disabled", width=200) # Largura ajustada
        self.model_selector_combo.pack(side=tk.LEFT)
        self.model_selector_combo.bind("<<ComboboxSelected>>", self.on_model_select_change)
        self.status_label = ctk.CTkLabel(model_controls_frame, text="Carregue modelos e dados...")
        self.status_label.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

        importance_frame = ctk.CTkFrame(main_content_frame)
        importance_frame.pack(fill=tk.X, pady=5, padx=10)
        ctk.CTkLabel(importance_frame, text="Importância de Features (Modelo Selecionado)", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(5, 5))
        self.model_importance_text = self._create_scrolled_text(importance_frame, height=15)
        self._update_text_widget(self.model_importance_text, "Carregue modelos e dados...")

        model_plots_frame = ctk.CTkFrame(main_content_frame)
        model_plots_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0), padx=10)

        self.shap_summary_container = ctk.CTkFrame(model_plots_frame)
        self.shap_summary_container.pack(fill=tk.X, expand=True, padx=7, pady=(0, 7))
        ctk.CTkLabel(self.shap_summary_container, text="SHAP Summary Plot", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=5)
        ctk.CTkLabel(self.shap_summary_container, text="Selecione modelo...").pack()
        if not SHAP_AVAILABLE:
            ctk.CTkLabel(self.shap_summary_container, text="(Biblioteca 'shap' não instalada)", text_color="orange").pack()

        pdp_outer_container = ctk.CTkFrame(model_plots_frame)
        pdp_outer_container.pack(fill=tk.BOTH, expand=True, padx=7, pady=(7, 0))
        ctk.CTkLabel(pdp_outer_container, text="Partial Dependence Plot (PDP)", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=5)
        
        pdp_controls_frame = ctk.CTkFrame(pdp_outer_container)
        pdp_controls_frame.pack(fill=tk.X, pady=(0, 5), padx=5)
        ctk.CTkLabel(pdp_controls_frame, text="Feature para PDP:").pack(side=tk.LEFT, padx=(0, 5))
        self.pdp_feature_selector_combo = ctk.CTkComboBox(pdp_controls_frame, variable=self.pdp_feature_selector_var, state="disabled", width=200) # Largura ajustada
        self.pdp_feature_selector_combo.pack(side=tk.LEFT)
        self.pdp_feature_selector_combo.bind("<<ComboboxSelected>>", self.on_pdp_feature_select)
        
        self.pdp_container = ctk.CTkFrame(pdp_outer_container)
        self.pdp_container.pack(fill=tk.BOTH, expand=True)
        ctk.CTkLabel(self.pdp_container, text="Selecione modelo/feature...").pack()

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
        if text_widget is None: return; 
        try:
            if text_widget.winfo_exists(): 
                text_widget.config(state='normal'); 
            text_widget.delete('1.0', tk.END); 
            text_widget.insert('1.0', content); 
            text_widget.config(state='disabled')
        except tk.TclError: 
            pass; 
        except Exception as e: 
            logger.error(f"Erro update text widget: {e}", exc_info=True)

    def _clear_matplotlib_widget(self, container_widget):
        if container_widget:
            for widget in container_widget.winfo_children(): widget.destroy()
    
    def _embed_matplotlib_figure(self, fig: plt.Figure, container_widget: ctk.CTkFrame):
        canvas_attr_name = None
        if container_widget == self.shap_summary_container: 
            canvas_attr_name = 'shap_summary_canvas_widget'
        elif container_widget == self.pdp_container: 
            canvas_attr_name = 'pdp_canvas_widget'
        else: 
            logger.warning(f"Container desconhecido embed: {container_widget}")
        existing_canvas = getattr(self, canvas_attr_name, None) if canvas_attr_name else None
        if existing_canvas and isinstance(existing_canvas, FigureCanvasTkAgg) and plt.fignum_exists(fig.number):
            logger.debug(f"Reutilizando canvas {canvas_attr_name}"); 
            existing_canvas.figure = fig; existing_canvas.draw_idle(); return existing_canvas
        else:
            logger.debug(f"Criando novo canvas {canvas_attr_name or 'desconhecido'}")
            self._clear_matplotlib_widget(container_widget);
            try:
                canvas = FigureCanvasTkAgg(fig, master=container_widget); 
                canvas.draw(); widget = canvas.get_tk_widget(); 
                widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
                if canvas_attr_name: 
                    setattr(self, canvas_attr_name, canvas)
                return canvas
            except Exception as e: 
                logger.error(f"Erro embed matplotlib: {e}", exc_info=True); 
                ctk.CTkLabel(container_widget, text=f"Erro gráfico:\n{e}").pack(); 
                return None
    
    def log(self, message: str):
        logger.info(f"[GUI Interpreter] {message}"); 
        try:
            if self.status_label and self.status_label.winfo_exists(): 
                self.status_label.config(text=message[:100])
        except: 
            pass

    def load_models_and_data(self):
        self.log("Carregando modelos e dados...")
        self.loaded_trained_models.clear()
        self.model_selector_var.set("")
        self.pdp_feature_selector_var.set("")
        
        try:
            self.model_selector_combo.configure(values=[], state="disabled")
            self.pdp_feature_selector_combo.configure(values=[], state="disabled")
        except tk.TclError: pass
            
        self._update_text_widget(self.model_importance_text, "Carregando...")
        for container in [self.shap_summary_container, self.pdp_container]:
            self._clear_matplotlib_widget(container)
            ctk.CTkLabel(container, text="Aguardando dados...").pack()
        
        try:
            if not hasattr(self.analyzer_app, 'X_clean') or self.analyzer_app.X_clean is None:
                messagebox.showerror("Erro de Dados", "Carregue os dados na aba 'Análise de Features' primeiro.", parent=self.parent)
                return
                
            self.X_analysis_data = self.analyzer_app.X_clean.copy()
            self.current_data_identifier = self.analyzer_app.current_data_identifier
            
            loaded_ids, all_feature_names = [], set()
            prefix = self.strategy.get_model_config_key_prefix()
            model_paths = {
                f"Melhor F1 ({prefix})": os.path.join(DATA_DIR, f"{prefix}_best_f1.joblib"),
                f"Melhor ROI ({prefix})": os.path.join(DATA_DIR, f"{prefix}_best_roi.joblib")
            }
            
            for model_id, path in model_paths.items():
                self.log(f"-> Carregando: {os.path.basename(path)}...")
                if not os.path.exists(path): continue
                
                load_result = load_model_scaler_features(path)
                if not load_result: continue
                
                model, scaler, calib, _, _, medians, features, _, _, _ = load_result
                if not (model and features): continue
                
                X_prepared = self.X_analysis_data[features].copy()
                if scaler: X_prepared = pd.DataFrame(scaler.transform(X_prepared), index=X_prepared.index, columns=features)
                
                imp_df = self._calculate_feature_importance(model, features, X_prepared)
                
                self.loaded_trained_models[model_id] = {**dict(zip(['model', 'scaler', 'calibrator', 'training_medians', 'features'], [model, scaler, calib, medians, features])), 'importances': imp_df, 'X_prepared': X_prepared}
                loaded_ids.append(model_id)
                all_feature_names.update(features)

            if loaded_ids:
                self.model_selector_combo.configure(values=loaded_ids, state="readonly")
                self.model_selector_var.set(loaded_ids[0])
                pdp_features = sorted(list(all_feature_names))
                self.pdp_feature_selector_combo.configure(values=pdp_features, state="readonly")
                if pdp_features: self.pdp_feature_selector_var.set(pdp_features[0])
                self.update_model_specific_analysis()
            else:
                self.log("Nenhum modelo válido encontrado para esta estratégia.")
        except Exception as e:
            messagebox.showerror("Erro Inesperado", f"Ocorreu um erro ao carregar os modelos:\n{e}")

        except Exception as e:
            logger.error(f"Erro geral em load_models_and_data: {e}", exc_info=True)
            messagebox.showerror("Erro Inesperado", f"Ocorreu um erro ao carregar os modelos:\n{e}")

    def _get_cache_path(self, model_id: str, cache_type: str, cache_dir: str) -> str: 
        safe_model_id = model_id.replace(" ","_").replace("(","").replace(")","").lower();
        data_hash="no_data"
        if self.current_data_identifier:
            try: 
                id_str = str(self.current_data_identifier); 
                data_hash = hashlib.sha1(id_str.encode()).hexdigest()[:8]
            except Exception: data_hash="error_hash"
        filename = f"{cache_type}_cache_{safe_model_id}_data_{data_hash}.joblib"; 
        return os.path.join(cache_dir,filename)
    
    def _get_shap_cache_path(self, model_id: str) -> str: 
        return self._get_cache_path(model_id, "shap", SHAP_CACHE_DIR)

    def _calculate_feature_importance(self, model, model_features: List[str], X_prepared_for_model: pd.DataFrame) -> Optional[pd.DataFrame]:
        importances = None; model_type_name = model.__class__.__name__; 
        logger.debug(f"Calculando importância para {model_type_name}...")
        if not hasattr(self.analyzer_app, 'y_clean') or self.analyzer_app.y_clean is None: 
            logger.error("y_clean não disponível."); 
            return None
        try: y_aligned = self.analyzer_app.y_clean.loc[X_prepared_for_model.index].copy()
        except Exception as e: 
            logger.error(f"Erro alinhar y importância: {e}"); 
            return None
        try:
            if hasattr(model, 'feature_importances_'): 
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                if model.coef_.shape[0] == 1: importances = np.abs(model.coef_[0])
                else: 
                    importances = np.mean(np.abs(model.coef_), axis=0)
            else: 
                logger.warning(f"Tentando Permutation Importance {model_type_name}...");
                try: 
                    result = permutation_importance(model, X_prepared_for_model, y_aligned, n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1, scoring='accuracy'); importances = result.importances_mean; logger.info(" -> Permutation Importance OK.")
                except Exception as e_perm: 
                    logger.error(f"Erro Permutation Importance: {e_perm}", exc_info=True); 
                    return None
            if importances is not None and len(importances) == len(model_features): 
                imp_df = pd.DataFrame({'Feature': model_features, 'Importance': importances}); 
                return imp_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
            elif importances is not None: 
                logger.error(f"Mismatch importância/features {model_type_name}"); 
                return None
            else: 
                logger.warning(f"Não obteve importâncias {model_type_name}."); 
                return None
        except Exception as e: 
            logger.error(f"Erro GERAL calc importância {model_type_name}: {e}", exc_info=True); 
            return None

    def on_model_select_change(self, event=None):

        logger.debug(f"Model selection changed: {self.model_selector_var.get()}")
        self.update_model_specific_analysis()

    def update_model_specific_analysis(self):
        selected_id = self.model_selector_var.get()
        if not selected_id or selected_id not in self.loaded_trained_models:
            self._update_text_widget(self.model_importance_text, "Selecione.")
            self._clear_matplotlib_widget(self.shap_summary_container)
            ctk.CTkLabel(self.shap_summary_container, text="...").pack()
            self._clear_matplotlib_widget(self.pdp_container)
            ctk.CTkLabel(self.pdp_container, text="...").pack()
            try:
                self.pdp_feature_selector_combo.config(values=[], state="disabled")
                self.pdp_feature_selector_var.set("")
            except:
                pass
            return
        self.log(f"Atualizando análise para: {selected_id}")
        model_data = self.loaded_trained_models[selected_id]
        importances_df = model_data.get('importances')
        if importances_df is not None and not importances_df.empty:
            try:
                self._update_text_widget(self.model_importance_text, importances_df.round(5).to_string(index=True))
            except Exception as e:
                logger.error(f"Erro formatar imp: {e}")
                self._update_text_widget(self.model_importance_text, "Erro.")
        else:
            self._update_text_widget(self.model_importance_text, f"Importância não disponível {selected_id}.")
        self._start_shap_calculation(model_data)
        pdp_features = model_data.get('features', [])
        current_pdp_list_str = str(self.pdp_feature_selector_combo.cget('values'))
        new_pdp_list_str = str(tuple(pdp_features))
        if pdp_features and current_pdp_list_str != new_pdp_list_str:
            logger.debug("Atualizando lista PDP.")
            self.pdp_feature_selector_combo.config(values=pdp_features, state="readonly")
            current_pdp_feature = self.pdp_feature_selector_var.get()
            if current_pdp_feature not in pdp_features:
                self.pdp_feature_selector_var.set(pdp_features[0] if pdp_features else "")
            self.on_pdp_feature_select()
        elif pdp_features:
            logger.debug("Lista PDP OK, disparando on_pdp_feature_select.")
            self.on_pdp_feature_select()
        else:
            self.pdp_feature_selector_combo.config(values=[], state="disabled")
            self.pdp_feature_selector_var.set("")
            self._clear_matplotlib_widget(self.pdp_container)
            ctk.CTkLabel(self.pdp_container, text="Features não encontradas.").pack()

    def _start_shap_calculation(self, model_data: Dict):
        if not SHAP_AVAILABLE:
            self._clear_matplotlib_widget(self.shap_summary_container)
            ctk.CTkLabel(self.shap_summary_container, text="'shap' não instalado.").pack()
            return
        if model_data.get('X_prepared') is None:
            self._clear_matplotlib_widget(self.shap_summary_container)
            ctk.CTkLabel(self.shap_summary_container, text="Dados preparados ausentes.").pack()
            return
        model = model_data.get('model')
        model_id = self.model_selector_var.get()
        features_for_model = model_data.get('features')
        if model is None or features_for_model is None:
            return
        if model_id not in self.loaded_trained_models or id(self.loaded_trained_models[model_id]) != id(model_data):
            return

        use_cache = False
        if 'shap_values' in model_data and 'shap_data_identifier' in model_data:
            cached_data_id = model_data['shap_data_identifier']
            cached_features = model_data.get('shap_model_features')
            if cached_data_id == self.current_data_identifier and cached_features == features_for_model:
                self.log(f"Reutilizando SHAP cacheado (memória) {model_id}.")
                use_cache = True
            else:
                self.log(f"Cache SHAP memória {model_id} invalidado.")
                for key in ['shap_values', 'X_sample', 'shap_data_identifier', 'shap_model_features', 'shap_used_kernel', 'shap_n_samples']:
                    model_data.pop(key, None)
        if use_cache:
            shap_payload_cached = {
                'model_id': model_id,
                'shap_values': model_data['shap_values'],
                'X_sample': model_data['X_sample'],
                'model_type_name': model.__class__.__name__,
                'used_kernel': model_data.get('shap_used_kernel', False),
                'n_samples': model_data.get('shap_n_samples', '?')
            }
            self._plot_shap_summary(shap_payload_cached)
            return

        model_type_name = model.__class__.__name__
        kernel_models = ['SVC', 'KNeighborsClassifier', 'GaussianNB']
        use_kernel_explainer = model_type_name in kernel_models
        if use_kernel_explainer:
            self.log(f"Iniciando SHAP {model_type_name} background...")
            self._clear_matplotlib_widget(self.shap_summary_container)
            ctk.CTkLabel(self.shap_summary_container, text=f"Calculando SHAP {model_type_name}...").pack()
            if self.shap_thread and self.shap_thread.is_alive():
                self.log("Cálculo SHAP anterior em andamento...")
                return
            self.shap_thread = threading.Thread(target=self._run_shap_task, args=(model_id, use_kernel_explainer), daemon=True)
            self.shap_thread.start()
        else:
            self.log(f"Calculando SHAP {model_type_name} diretamente...")
            self._clear_matplotlib_widget(self.shap_summary_container)
            ctk.CTkLabel(self.shap_summary_container, text=f"Calculando SHAP {model_type_name}...").pack()
            self.parent.update_idletasks()
            self._run_shap_task(model_id, use_kernel_explainer)

    def _run_shap_task(self, target_model_id: str, use_kernel_explainer: bool):
        shap_values_for_plot = None
        X_shap_sample = None
        if target_model_id not in self.loaded_trained_models:
            logger.error(f"Thread SHAP: Modelo {target_model_id} não carregado.")
            self.gui_queue.put(("shap_error", {'model_id': target_model_id, 'error': "Dados modelo."}))
            return
        model_data_target = self.loaded_trained_models[target_model_id]
        try:
            model = model_data_target.get('model')
            features = model_data_target.get('features')
            X_prepared = model_data_target.get('X_prepared')
            if not all([model, features, X_prepared is not None]):
                raise ValueError("Dados insuficientes thread SHAP")
            X_shap_ready = X_prepared
            N_SHAP_SAMPLES = N_SHAP_SAMPLES_SLOW if use_kernel_explainer else N_SHAP_SAMPLES_FAST
            N_KERNEL_BG_LOCAL = N_KERNEL_BG if use_kernel_explainer else 50
            if len(X_shap_ready) > N_SHAP_SAMPLES:
                X_shap_sample = shap.sample(X_shap_ready, N_SHAP_SAMPLES, random_state=RANDOM_STATE)
            else:
                X_shap_sample = X_shap_ready
            explainer = None
            shap_values = None
            model_type_name = model.__class__.__name__
            if use_kernel_explainer:
                X_kernel_bg = shap.sample(X_shap_ready, N_KERNEL_BG_LOCAL, random_state=RANDOM_STATE)
                def model_predict_proba_pos(data):
                    if not isinstance(data, pd.DataFrame):
                        data_df = pd.DataFrame(data, columns=features)
                    else:
                        data_df = data
                    try:
                        return model.predict_proba(data_df)[:, 1]
                    except Exception as e:
                        logger.error(f"Erro pred kernel: {e}")
                        return np.full(len(data_df), 0.5)
                explainer = shap.KernelExplainer(model_predict_proba_pos, X_kernel_bg)
                shap_values = explainer.shap_values(X_shap_sample, silent=True)
            elif model_type_name in ['RandomForestClassifier', 'LGBMClassifier', 'GradientBoostingClassifier'] and LGBM_AVAILABLE and lgb is not None:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_shap_sample)
            elif model_type_name in ['LogisticRegression']:
                explainer = shap.LinearExplainer(model, X_shap_sample)
                shap_values = explainer.shap_values(X_shap_sample)
            else:
                logger.warning(f"Tipo {model_type_name} fallback Kernel.")
                use_kernel_explainer = True
                N_SHAP_SAMPLES = N_SHAP_SAMPLES_SLOW
                N_KERNEL_BG_LOCAL = N_KERNEL_BG
                if len(X_shap_ready) > N_SHAP_SAMPLES:
                    X_shap_sample = shap.sample(X_shap_ready, N_SHAP_SAMPLES, random_state=RANDOM_STATE)
                else:
                    X_shap_sample = X_shap_ready
                X_kernel_bg = shap.sample(X_shap_ready, N_KERNEL_BG_LOCAL, random_state=RANDOM_STATE)
                def model_predict_proba_pos_fb(data):
                    if not isinstance(data, pd.DataFrame):
                        data_df = pd.DataFrame(data, columns=features)
                    else:
                        data_df = data
                    try:
                        return model.predict_proba(data_df)[:, 1]
                    except Exception as e:
                        logger.error(f"Erro pred kernel fb: {e}")
                        return np.full(len(data_df), 0.5)
                explainer = shap.KernelExplainer(model_predict_proba_pos_fb, X_kernel_bg)
                shap_values = explainer.shap_values(X_shap_sample, silent=True)
            if shap_values is not None:
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_values_for_plot = shap_values[1]
                elif isinstance(shap_values, np.ndarray):
                    if len(shap_values.shape) == 3:
                        shap_values_for_plot = shap_values[:, :, 1]
                    elif len(shap_values.shape) == 2:
                        shap_values_for_plot = shap_values
                    else:
                        shap_values_for_plot = shap_values
                else:
                    shap_values_for_plot = shap_values

            if shap_values_for_plot is not None and X_shap_sample is not None:
                shap_cache_data = {
                    'shap_values': shap_values_for_plot,
                    'X_sample': X_shap_sample,
                    'shap_data_identifier': self.current_data_identifier,
                    'shap_model_features': features,
                    'shap_used_kernel': use_kernel_explainer,
                    'shap_n_samples': N_SHAP_SAMPLES
                }
                shap_cache_path = self._get_shap_cache_path(target_model_id)
                try:
                    joblib.dump(shap_cache_data, shap_cache_path)
                    self.log(f"Cache SHAP salvo disco {target_model_id}.")
                except Exception as e_save:
                    logger.error(f"Erro salvar cache SHAP disco: {e_save}", exc_info=True)
                cache_updated_mem = False
                if target_model_id in self.loaded_trained_models:
                    try:
                        self.loaded_trained_models[target_model_id].update(shap_cache_data)
                        cache_updated_mem = 'shap_values' in self.loaded_trained_models[target_model_id]
                        logger.info(f"Cache SHAP memória {target_model_id} OK.")
                    except Exception as e_mem:
                        logger.error(f"Erro atualizar cache SHAP memória: {e_mem}", exc_info=True)
                else:
                    logger.warning(f"Modelo {target_model_id} não no dict ao salvar cache mem SHAP.")
                if cache_updated_mem:
                    self.gui_queue.put(("shap_result_calculated", {'model_id': target_model_id}))
                else:
                    self.gui_queue.put(("shap_error", {'model_id': target_model_id, 'error': "Falha atualizar cache SHAP memória."}))
            else:
                raise ValueError("Cálculo SHAP não produziu resultados válidos.")
        except Exception as e:
            logger.error(f"Erro thread SHAP {target_model_id}: {e}", exc_info=True)
            self.gui_queue.put(("shap_error", {'model_id': target_model_id, 'error': str(e)}))

    def _plot_shap_summary(self, shap_data: Dict):
        model_id_for_plot = shap_data.get('model_id', 'Desconhecido')
        self.log(f"Gerando gráfico SHAP para {model_id_for_plot}...")
        self._clear_matplotlib_widget(self.shap_summary_container)

        fig_temp = None
        temp_image_path = os.path.join(SHAP_CACHE_DIR, f"temp_shap_summary_{os.getpid()}.png")

        original_autolayout = plt.rcParams['figure.autolayout']
        original_constrained = plt.rcParams.get('figure.constrained_layout.use', False)

        try:
            shap_values = shap_data['shap_values']
            X_sample = shap_data['X_sample']
            model_type = shap_data['model_type_name']
            used_kernel = shap_data['used_kernel']
            n_samples = shap_data['n_samples']
            model_info = self.loaded_trained_models.get(model_id_for_plot, {})
            feature_names_plot = X_sample.columns.tolist() if isinstance(X_sample, pd.DataFrame) else model_info.get('features')
            if not feature_names_plot:
                raise ValueError("Nomes features ausentes SHAP plot.")

            plt.rcParams['figure.autolayout'] = False
            plt.rcParams['figure.constrained_layout.use'] = False

            fig_temp = plt.figure(figsize=(8, 6))
            shap.summary_plot(
                shap_values, X_sample, plot_type="dot",
                show=False, max_display=15, feature_names=feature_names_plot
            )
            ax = fig_temp.gca()
            plot_title = f'SHAP Summary ({model_type})'
            if used_kernel:
                plot_title += f'\n(KernelExplainer - {n_samples} amostras)'
            ax.set_title(plot_title)

            fig_temp.savefig(temp_image_path, bbox_inches='tight', dpi=96)

        except Exception as e_plot_save:
            logger.error(f"Erro ao plotar/salvar SHAP para {model_id_for_plot}: {e_plot_save}", exc_info=True)
            ctk.CTkLabel(self.shap_summary_container, text=f"Erro ao gerar/salvar SHAP:\n{e_plot_save}").pack()
            temp_image_path = None
        finally:
            if fig_temp and plt.fignum_exists(fig_temp.number):
                plt.close(fig_temp)
            plt.rcParams['figure.autolayout'] = original_autolayout
            plt.rcParams['figure.constrained_layout.use'] = original_constrained

        self._clear_matplotlib_widget(self.shap_summary_container)
        if temp_image_path and os.path.exists(temp_image_path):
            try:
                img = Image.open(temp_image_path)
                photo = ImageTk.PhotoImage(img)
                img_label = ctk.CTkLabel(self.shap_summary_container, image=photo)
                img_label.image = photo
                img_label.pack(fill=tk.BOTH, expand=True)
                self.log(f"Gráfico SHAP exibido como imagem para {model_id_for_plot}.")
                try:
                    os.remove(temp_image_path)
                except OSError:
                    logger.warning(f"Não foi possível remover imagem SHAP temp: {temp_image_path}")
            except Exception as e_img:
                logger.error(f"Erro ao carregar/exibir imagem SHAP {temp_image_path}: {e_img}", exc_info=True)
                ctk.CTkLabel(self.shap_summary_container, text=f"Erro exibir imagem SHAP:\n{e_img}").pack()
        elif temp_image_path:
            logger.error(f"Imagem SHAP temporária não encontrada: {temp_image_path}")
            ctk.CTkLabel(self.shap_summary_container, text="Erro: Imagem SHAP não gerada/encontrada.").pack()

    def on_pdp_feature_select(self, event=None):

        logger.debug("--- on_pdp_feature_select triggered ---")
        if self.pdp_update_job:
            try: 
                self.main_tk_root.after_cancel(self.pdp_update_job)
            except: 
                pass 
        self.pdp_update_job = self.main_tk_root.after(300, self._plot_pdp) 


    def _plot_pdp(self):
    
        self.pdp_update_job = None 
        selected_model_id = self.model_selector_var.get()
        selected_pdp_feature = self.pdp_feature_selector_var.get()
        logger.debug(f"--- Iniciando PDP para {selected_model_id} - Feature: {selected_pdp_feature} ---")

        if not selected_model_id or not selected_pdp_feature or selected_model_id not in self.loaded_trained_models:
            self._clear_matplotlib_widget(self.pdp_container)
            ctk.CTkLabel(self.pdp_container, text="Selecione um modelo e uma feature.").pack()
            return
            
        model_data = self.loaded_trained_models[selected_model_id]
        model = model_data.get('model')
        X_prepared = model_data.get('X_prepared')

        if not all([model, X_prepared is not None, selected_pdp_feature in X_prepared.columns]):
            self._clear_matplotlib_widget(self.pdp_container)
            ctk.CTkLabel(self.pdp_container, text="Dados inválidos: Modelo, dados preparados ou feature ausentes.").pack()
            return

        self._clear_matplotlib_widget(self.pdp_container)
        ctk.CTkLabel(self.pdp_container, text=f"Calculando PDP para '{selected_pdp_feature}'...").pack()
        self.parent.update_idletasks()

        try:
            if self.fig_pdp is None or not plt.fignum_exists(self.fig_pdp.number):
                logger.debug("Criando nova figura/eixo para o PDP.")
                self.fig_pdp, self.ax_pdp = plt.subplots(figsize=(7, 5), constrained_layout=True)
            else:
                logger.debug("Limpando eixo PDP existente.")
                self.ax_pdp.clear()

            if len(X_prepared) > N_PDP_SAMPLES:
                logger.debug(f"Usando {N_PDP_SAMPLES} amostras (de {len(X_prepared)}) para o PDP.")
                X_pdp_sample = X_prepared.sample(n=N_PDP_SAMPLES, random_state=RANDOM_STATE)
            else:
                X_pdp_sample = X_prepared

            logger.debug(f"Gerando PDP Display para '{selected_pdp_feature}'...")
            
            PartialDependenceDisplay.from_estimator(
                estimator=model, 
                X=X_pdp_sample, 
                features=[selected_pdp_feature], 
                kind='average', 
                ax=self.ax_pdp,
                line_kw={"color": "darkcyan", "linewidth": 2.5}
            )

            target_name = self.strategy.get_target_variable_name()
            self.ax_pdp.set_title(f'Dependência Parcial\n({selected_pdp_feature} vs. Prob. de {target_name})')
            self.ax_pdp.set_ylabel('Dependência Parcial (na prob. da classe positiva)')
            self.ax_pdp.set_xlabel(f'Valor da Feature: {selected_pdp_feature}')
            self.ax_pdp.grid(True, linestyle='--', alpha=0.6)
            
            sns.rugplot(x=X_pdp_sample[selected_pdp_feature], ax=self.ax_pdp, height=-0.03, clip_on=False, color='black', alpha=0.3)

            self._embed_matplotlib_figure(self.fig_pdp, self.pdp_container)
            logger.debug(f"--- PDP para '{selected_pdp_feature}' finalizado com sucesso. ---")

        except Exception as e:
            logger.error(f"Erro GERAL ao plotar PDP para '{selected_pdp_feature}': {e}", exc_info=True)
            self._clear_matplotlib_widget(self.pdp_container)
            ctk.CTkLabel(self.pdp_container, text=f"Erro ao gerar gráfico PDP:\n{e}", foreground="red").pack()

    def _handle_shap_result(self, payload: Dict[str, Any]):

        finished_model_id = payload.get('model_id')
        current_model_id = self.model_selector_var.get()

        if finished_model_id != current_model_id:
            self.log(f"Ignorando resultado SHAP obsoleto para o modelo '{finished_model_id}'.")
            return

        model_data = self.loaded_trained_models.get(finished_model_id)
        if not model_data or 'shap_values' not in model_data or 'X_sample' not in model_data:
            logger.error(f"Erro ao plotar SHAP: dados ou cache inválidos para o modelo '{finished_model_id}'.")
            self._clear_matplotlib_widget(self.shap_summary_container)
            ctk.CTkLabel(self.shap_summary_container, text=f"Erro: Dados de SHAP para\n{finished_model_id}\não encontrados.").pack()
            return
            
        shap_plot_payload = {
            'model_id': finished_model_id,
            'shap_values': model_data['shap_values'],
            'X_sample': model_data['X_sample'],
            'model_type_name': model_data.get('model', type(None)).__class__.__name__,
            'used_kernel': model_data.get('shap_used_kernel', False),
            'n_samples': model_data.get('shap_n_samples', '?')
        }
        self._plot_shap_summary(shap_plot_payload)

    def _handle_shap_error(self, payload: Dict[str, Any]):
        
        error_model_id = payload.get('model_id', 'desconhecido')
        error_msg = payload.get('error', 'Um erro ocorreu.')
        self.log(f"Erro no cálculo SHAP para o modelo '{error_model_id}': {error_msg}")
        
        if error_model_id == self.model_selector_var.get():
            self._clear_matplotlib_widget(self.shap_summary_container)
            ctk.CTkLabel(
                self.shap_summary_container, 
                text=f"Erro ao calcular SHAP:\n{error_msg}", 
                foreground="red"
            ).pack()

    def process_gui_queue(self):

        if self.stop_processing_queue:
            logger.debug("Interpreter: Parando o processamento da fila da GUI.")
            return

        try:
            while True:
                try:
                    message_type, payload = self.gui_queue.get_nowait()
                    
                    handlers = {
                        "shap_result_calculated": 
                        self._handle_shap_result,
                        "shap_error": 
                        self._handle_shap_error,
                    }
                    
                    handler = handlers.get(message_type)
                    
                    if handler:
                        handler(payload)
                    else:
                        logger.warning(f"Tipo de mensagem GUI desconhecido na Aba Interpreter: '{message_type}'")

                except queue.Empty:
                    break
                except Exception as e:
                    logger.error(f"Erro ao processar uma mensagem da fila GUI (Interpreter): {e}", exc_info=True)

        finally:
            if not self.stop_processing_queue:
                try:
                    if self.main_tk_root and self.main_tk_root.winfo_exists():
                        self.main_tk_root.after(100, self.process_gui_queue)
                except Exception as e_resched:
                    if not self.stop_processing_queue:
                        logger.error(f"Erro crítico ao reagendar a fila da GUI (Interpreter): {e_resched}")