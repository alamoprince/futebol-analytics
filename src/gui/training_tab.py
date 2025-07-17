import tkinter as tk
import customtkinter as ctk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText
import sys, os, threading, datetime, math, numpy as np 
from queue import Queue, Empty
import pandas as pd
from typing import Optional, Dict, List, Any 

from strategies.base_strategy import BettingStrategy
from backtester import RuleBasedBacktester


try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
from typing import Optional, Dict, List, Any
from logger_config import setup_logger

logger = setup_logger("MainTrainingTab")
try:
    GUI_DIR = os.path.dirname(os.path.abspath(__file__))
    SRC_DIR = os.path.dirname(GUI_DIR)
    BASE_DIR = os.path.dirname(SRC_DIR)
    if BASE_DIR not in sys.path: sys.path.insert(0, BASE_DIR)
    if SRC_DIR not in sys.path: sys.path.insert(0, SRC_DIR)
except NameError: 
    pass

try:
    from config import (
        CLASS_NAMES, FIXTURE_FETCH_DAY,MODEL_CONFIG, DATA_DIR, 
        MODEL_TYPE_NAME, ODDS_COLS as CONFIG_ODDS_COLS, MIN_PROB_THRESHOLD_FOR_HIGHLIGHT, 
        BEST_F1_MODEL_SAVE_PATH, BEST_ROI_MODEL_SAVE_PATH, MODEL_ID_F1, MODEL_ID_ROI,
        DEFAULT_F1_THRESHOLD, DEFAULT_EV_THRESHOLD, CALIBRATION_METHOD_DEFAULT )
    
    from data_handler import (
        load_historical_data,
        preprocess_and_feature_engineer, 
        fetch_and_process_fixtures,      
        prepare_fixture_data,            
        _calculate_match_results_and_points 
    )
    from model_trainer import train_evaluate_and_save_best_models as run_training_process
    import predictor 
    import requests 
    from sklearn.model_selection import train_test_split 
    import traceback 
except ImportError as e:
    logger.error(f"Erro import main.py (Treino/Previsão): {e}")
    raise 
except Exception as e_i:
    logger.error(f"Erro geral import main.py (Treino/Previsão): {e_i}")
    raise 

class FootballPredictorDashboard:
    def __init__(self, parent_frame: ctk.CTkFrame, main_root: ctk.CTk, strategy: BettingStrategy, **kwargs):
        self.parent = parent_frame
        self.main_tk_root = main_root
        self.strategy = strategy
        self.strategy_type = self.strategy.get_strategy_type()

        self.gui_queue = Queue()
        self.stop_processing_queue = False
        self.historical_data: Optional[pd.DataFrame] = None
        self.progress_max_value: int = 1
        
        if self.strategy_type == "machine_learning":
            self.loaded_models_data: Dict[str, Dict] = {}
            self.available_model_ids: List[str] = []
            self.selected_model_id: Optional[str] = None
            self.trained_model: Optional[Any] = None
            self.trained_scaler: Optional[Any] = None
            self.trained_calibrator: Optional[Any] = None
            self.training_medians: Optional[pd.Series] = None
            self.feature_columns: Optional[List[str]] = None
            self.optimal_f1_threshold: float = DEFAULT_F1_THRESHOLD
            self.optimal_ev_threshold: float = DEFAULT_EV_THRESHOLD
            self.selected_model_var = tk.StringVar()

        self.create_train_predict_widgets()
        self.main_tk_root.after(100, self.process_gui_queue)
        self.log(f"Aba inicializada para a estratégia: '{self.strategy.get_display_name()}'")
        
        if self.strategy_type == "machine_learning":
            self.load_existing_model_assets()

    def create_train_predict_widgets(self):
        self.parent.grid_columnconfigure(1, weight=1)
        self.parent.grid_rowconfigure(0, weight=1)

        left_panel = ctk.CTkFrame(self.parent)
        left_panel.grid(row=0, column=0, padx=10, pady=10, sticky="ns")
        
        right_panel = ctk.CTkFrame(self.parent, fg_color="transparent")
        right_panel.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="nsew")
        right_panel.grid_rowconfigure(0, weight=2)
        right_panel.grid_rowconfigure(1, weight=1)
        right_panel.grid_columnconfigure(0, weight=1)

        if self.strategy_type == "machine_learning":
            self._create_ml_widgets(left_panel)
        elif self.strategy_type == "rule_based":
            self._create_rule_based_widgets(left_panel)

        self._create_right_panel_widgets(right_panel)

    def start_finding_entries(self):
        threading.Thread(target=self._run_rule_based_pipeline, daemon=True).start()

    def _create_ml_widgets(self, parent: ctk.CTkFrame):
        parent.grid_rowconfigure(1, weight=1)
        control_frame = ctk.CTkFrame(parent)
        control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        self.load_train_button = ctk.CTkButton(control_frame, text="TREINAR MODELOS", command=self.start_training_thread)
        self.load_train_button.pack(fill="x", padx=10, pady=10)

        predict_frame = ctk.CTkFrame(control_frame)
        predict_frame.pack(fill="x", padx=10, pady=(0, 10))
        predict_frame.grid_columnconfigure(1, weight=1)
        self.predict_button = ctk.CTkButton(predict_frame, text=f"Prever ({FIXTURE_FETCH_DAY})", command=self.start_prediction_thread, state="disabled")
        self.predict_button.grid(row=0, column=0, padx=(0, 5))
        self.model_selector_combo = ctk.CTkComboBox(predict_frame, variable=self.selected_model_var, state="readonly", command=self.on_model_select)
        self.model_selector_combo.grid(row=0, column=1, sticky="ew")

        stats_frame = ctk.CTkFrame(parent)
        stats_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        stats_frame.grid_rowconfigure(1, weight=1)
        stats_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(stats_frame, text="Status do Modelo Selecionado", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")
        self.model_stats_text = ScrolledText(stats_frame, wrap="word", font=("Consolas", 9), relief="flat", borderwidth=0, bg="#2B2B2B", fg="white", insertbackground="white")
        self.model_stats_text.grid(row=1, column=0, padx=1, pady=1, sticky="nsew")
        
    def _create_rule_based_widgets(self, parent: ctk.CTkFrame):
        parent.grid_rowconfigure(1, weight=1)
        control_frame = ctk.CTkFrame(parent)
        control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        self.backtest_button = ctk.CTkButton(control_frame, text="Verificar Estratégia no Histórico", command=self.start_backtest_thread)
        self.backtest_button.pack(fill="x", padx=10, pady=10)
        self.find_entries_button = ctk.CTkButton(control_frame, text="Encontrar Entradas Futuras", command=self.start_find_entries_thread)
        self.find_entries_button.pack(fill="x", padx=10, pady=(0, 10))

        results_frame = ctk.CTkFrame(parent)
        results_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        results_frame.grid_rowconfigure(1, weight=1)
        results_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(results_frame, text="Resultado do Backtest", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")
        self.backtest_results_text = ScrolledText(results_frame, wrap="word", font=("Consolas", 9), relief="flat", borderwidth=0, bg="#2B2B2B", fg="white", insertbackground="white")
        self.backtest_results_text.grid(row=1, column=0, padx=1, pady=1, sticky="nsew")
    
    def _create_right_panel_widgets(self, parent: ctk.CTkFrame):
        results_frame = ctk.CTkFrame(parent)
        results_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        results_frame.grid_rowconfigure(0, weight=1)
        results_frame.grid_columnconfigure(0, weight=1)
        
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview", background="#2B2B2B", foreground="white", fieldbackground="#2B2B2B", borderwidth=0)
        style.configure("Treeview.Heading", background="#333333", foreground="white", relief="flat")
        style.map('Treeview.Heading', background=[('active', '#555555')])
        
        self.prediction_tree = ttk.Treeview(results_frame, show='headings', height=10)
        self.prediction_tree.grid(row=0, column=0, sticky="nsew")
        vsb = ttk.Scrollbar(results_frame, orient="vertical", command=self.prediction_tree.yview)
        vsb.grid(row=0, column=1, sticky="ns")
        hsb = ttk.Scrollbar(results_frame, orient="horizontal", command=self.prediction_tree.xview)
        hsb.grid(row=1, column=0, sticky="ew")
        self.prediction_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        log_frame = ctk.CTkFrame(parent)
        log_frame.grid(row=1, column=0, sticky="nsew")
        log_frame.grid_rowconfigure(1, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(log_frame, text="Logs", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.log_area = ScrolledText(log_frame, height=8, wrap="word", font=("Consolas", 9), relief="flat", borderwidth=0, bg="#2B2B2B", fg="white", insertbackground="white")
        self.log_area.grid(row=1, column=0, padx=1, pady=1, sticky="nsew")
        self.log_area.configure(state='disabled')

    def _run_rule_based_pipeline(self):
        fixture_df = fetch_and_process_fixtures()
        if fixture_df is None or fixture_df.empty:
            self.log("Nenhum jogo encontrado para análise.")
            return

        entradas_df = self.strategy.find_entries(fixture_df)
        
        self.gui_queue.put(("prediction_complete", entradas_df))

    def log(self, message: str):
        self.gui_queue.put(("log", message))

    def _reset_model_state(self):
        self.selected_model_id = None
        self.trained_model = None
        self.trained_scaler = None
        self.trained_calibrator = None
        self.training_medians = None
        self.feature_columns = None
        self.model_best_params = None
        self.model_eval_metrics = None
        self.model_file_timestamp = None
        self.optimal_f1_threshold = DEFAULT_F1_THRESHOLD
        self.optimal_ev_threshold = DEFAULT_EV_THRESHOLD
        self.available_model_ids.clear()
        self.loaded_models_data.clear()
        self.selected_model_var.set('')
        self.model_selector_combo.configure(values=[])

    def _update_log_area(self, message: str):
         try:
             if hasattr(self, 'log_area') and self.log_area.winfo_exists():
                 self.log_area.config(state='normal'); ts = datetime.datetime.now().strftime("%H:%M:%S"); self.log_area.insert(tk.END, f"[{ts}] {message}\n"); self.log_area.config(state='disabled'); self.log_area.see(tk.END)
         except tk.TclError: 
             pass
         
    def set_button_state(self, button: ctk.CTkButton, state: str): 
        self.gui_queue.put(("button_state", (button, state)))

    def _update_button_state(self, button_state_tuple):
         button, state = button_state_tuple
         try:
             if button.winfo_exists(): button.configure(state=state) # <<< CORREÇÃO AQUI
         except tk.TclError: pass
         except Exception as e: 
             logger.error(f"Erro interno ao atualizar estado do botão '{button_state_tuple[0].cget('text')}': {e}")

    def _update_model_stats_display_gui(self):

        try:
            if not hasattr(self, 'model_stats_text') or not self.model_stats_text.winfo_exists():
                return
            self.model_stats_text.config(state='normal')
            self.model_stats_text.delete('1.0', tk.END)

            if self.trained_model is None or self.selected_model_id is None:
                stats_content = "Nenhum modelo selecionado ou carregado."
            else:
                model_data = self.loaded_models_data.get(self.selected_model_id, {})
                metrics = model_data.get('metrics', {}) # Métricas salvas do teste/avaliação
                params = model_data.get('params')
                features = model_data.get('features', [])
                timestamp = model_data.get('timestamp') 
                path = model_data.get('path')
                model_obj = model_data.get('model') 
                model_class_name = model_obj.__class__.__name__ if model_obj else metrics.get('model_name', 'N/A') # Usa o nome salvo se objeto não carregou

                optimal_f1_thr = metrics.get('optimal_f1_threshold', DEFAULT_F1_THRESHOLD)
                optimal_ev_thr = metrics.get('optimal_ev_threshold', DEFAULT_EV_THRESHOLD)
                optimal_prec_thr = metrics.get('optimal_precision_threshold', 0.5) 

                calibrator_loaded = model_data.get('calibrator') is not None

                # --- Seção 1: Informações Gerais ---
                stats_content = f"Modelo Selecionado: {self.selected_model_id}\n"
                stats_content += f"  Tipo: {model_class_name}\n"
                if model_class_name == 'VotingClassifier' and params and 'estimators' in params:
                     estimator_names = [e_name.split('_')[0] for e_name in params['estimators']] # Tenta pegar nome base
                     stats_content += f"  Estimadores Base: {', '.join(estimator_names)}\n"
                elif params and model_class_name != 'VotingEnsemble': # Evita mostrar params complexos do voting
                    stats_content += f"  Params Otimizados: {params}\n"

                stats_content += f"  Calibrado ({CALIBRATION_METHOD_DEFAULT}): {'Sim' if calibrator_loaded else 'Não'}\n"
                stats_content += f"  Arquivo: {os.path.basename(path or 'N/A')}\n"
                stats_content += f"  Modificado: {timestamp or 'N/A'}\n"
                if features: stats_content += f"  Features ({len(features)}): {', '.join(features)}\n"
                stats_content += "---\n"

                # --- Seção 2: Limiares Otimizados ---
                stats_content += "Limiares Otimizados (Validação/Teste):\n"
                stats_content += f"- Limiar F1:      {optimal_f1_thr:.4f}\n"
                stats_content += f"- Limiar Precision: {optimal_prec_thr:.4f}\n"
                stats_content += f"- Limiar EV:        {optimal_ev_thr:.4f}\n"
                stats_content += "---\n"

                # --- Seção 3: Métricas de Qualidade (Teste) ---
                stats_content += "Métricas de Qualidade (Conjunto Teste):\n"
                test_n = metrics.get('test_set_size', 'N/A')
                stats_content += f"- Tamanho Teste: {test_n}\n"

                f1_f1 = metrics.get('f1_score_draw') 
                p_f1 = metrics.get('precision_draw_thrF1') 
                r_f1 = metrics.get('recall_draw_thrF1')   
                acc_f1 = metrics.get('accuracy_thrF1')
                stats_content += f"- @ Limiar F1 ({optimal_f1_thr:.3f}):\n"
                stats_content += f"    F1={f1_f1:.4f}" if f1_f1 is not None else "    F1=N/A"
                stats_content += f" | P={p_f1:.4f}" if p_f1 is not None else " | P=N/A"
                stats_content += f" | R={r_f1:.4f}" if r_f1 is not None else " | R=N/A"
                stats_content += f" | Acc={acc_f1:.4f}\n" if acc_f1 is not None else " | Acc=N/A\n"

                # Métricas @ Limiar Precision Otimizado
                f1_p = metrics.get('f1_score_draw_thrPrec')
                p_p = metrics.get('precision_draw_thrPrec')
                r_p = metrics.get('recall_draw_thrPrec')
                acc_p = metrics.get('accuracy_thrPrec')
                stats_content += f"- @ Limiar Prec ({optimal_prec_thr:.3f}):\n"
                stats_content += f"    F1={f1_p:.4f}" if f1_p is not None else "    F1=N/A"
                stats_content += f" | P={p_p:.4f}" if p_p is not None else " | P=N/A"
                stats_content += f" | R={r_p:.4f}" if r_p is not None else " | R=N/A"
                stats_content += f" | Acc={acc_p:.4f}\n" if acc_p is not None else " | Acc=N/A\n"

                # Métricas Probabilísticas (Pós-Calibração)
                auc = metrics.get('roc_auc')
                brier = metrics.get('brier_score')
                logloss = metrics.get('log_loss') 
                stats_content += "- Métricas Probabilísticas:\n"
                stats_content += f"    ROC AUC (Pós-Calib): {auc:.4f}\n" if auc is not None else "    ROC AUC=N/A\n"
                stats_content += f"    Brier Score (Pós-Calib): {brier:.4f}\n" if brier is not None else "    Brier=N/A\n"
                stats_content += f"    Log Loss (Bruto): {logloss:.4f}\n" if logloss is not None else "    Log Loss=N/A\n"

                stats_content += "---\n"

                # --- Seção 4: Estratégia de Aposta EV (Teste) ---
                profit_ev = metrics.get('profit')
                roi_ev = metrics.get('roi')
                n_bets_ev = metrics.get('num_bets') 
                stats_content += f"Estratégia EV (EV > {optimal_ev_thr:.3f} no Teste):\n"
                stats_content += f"- Nº Apostas Sugeridas: {n_bets_ev if n_bets_ev is not None else 'N/A'}\n"
                profit_ev_str = f"{profit_ev:+.2f} u" if profit_ev is not None else "N/A" 
                roi_ev_str = "N/A"
                if isinstance(roi_ev, (int, float, np.number)) and pd.notna(roi_ev) and np.isfinite(roi_ev):
                    roi_ev_str = f"{roi_ev:+.2f} %" 

                stats_content += f"- Lucro/Prejuízo: {profit_ev_str}\n"
                stats_content += f"- ROI Calculado: {roi_ev_str}\n"

            # Atualiza o widget
            self.model_stats_text.insert('1.0', stats_content)
            self.model_stats_text.config(state='disabled')
        except tk.TclError:
            pass 
        except Exception as e:
            logger.error(f"Erro _update_model_stats_display_gui: {e}", exc_info=True)
            try: 
                 if hasattr(self, 'model_stats_text') and self.model_stats_text.winfo_exists():
                      self.model_stats_text.config(state='normal')
                      self.model_stats_text.delete('1.0', tk.END)
                      self.model_stats_text.insert('1.0', f"Erro ao exibir stats:\n{e}")
                      self.model_stats_text.config(state='disabled')
            except: pass

    # --- : Setup Colunas Treeview ---
    def _setup_prediction_columns(self, columns: List[str]):
        """Configura colunas da Treeview, incluindo EV."""
        try:
            if not hasattr(self, 'prediction_tree') or not self.prediction_tree.winfo_exists():
                return

            self.prediction_tree['columns'] = columns
            self.prediction_tree.delete(*self.prediction_tree.get_children())

            # Larguras ajustadas
            col_widths = {
                'Data': 75,
                'Hora': 50,
                'Liga': 130,
                'Casa': 95,
                'Fora': 95,
                'Odd D': 50,
                'P(E) Calib': 75,
                'EV Empate': 70,  
                'Status': 500
            }

            for col in columns:
                self.prediction_tree.heading(col, text='')
                self.prediction_tree.column(col, width=0, minwidth=0, stretch=tk.NO)

                width = col_widths.get(col, 60)
                anchor = tk.W if col in ['Liga', 'Casa', 'Fora', 'Data', 'Hora', 'Status'] else tk.CENTER
                stretch = tk.NO
                header_text = col.replace("P(E) Calib", "P(E) Cal").replace("EV Empate", "EV (E)")

                self.prediction_tree.heading(col, text=header_text, anchor=anchor)
                self.prediction_tree.column(col, anchor=anchor, width=width, stretch=stretch)

            if columns == ['Status']:
                self.prediction_tree.column('Status', stretch=tk.YES)
        except Exception as e:
            logger.error(f"Erro _setup_prediction_columns: {e}")

    # --- : Atualiza Display de Previsões ---
    def _update_prediction_display(self, df: Optional[pd.DataFrame]):
        try:
            for i in self.prediction_tree.get_children():
                self.prediction_tree.delete(i)

            if df is None or df.empty:
                self.log("Nenhuma previsão para exibir.")
                self._setup_prediction_columns(['Status'])
                self.prediction_tree.insert('', 'end', values=['Nenhuma previsão foi gerada ou os jogos não foram encontrados.'])
                return

            strategy_target_name = self.strategy.get_target_variable_name()
            relevant_odd_cols = self.strategy.get_relevant_odds_cols()
            main_odd_col = relevant_odd_cols[0] if relevant_odd_cols else None
            
            prob_positive_raw_col = f'ProbRaw_{strategy_target_name}'
            prob_positive_calib_col = f'Prob_{strategy_target_name}'
            ev_col = f'EV_{strategy_target_name}'

            display_headers = ['Data', 'Hora', 'Liga', 'Casa', 'Fora']
            header_to_col_map = {'Data':'Date_Str', 'Hora':'Time_Str', 'Liga':'League', 'Casa':'Home', 'Fora':'Away'}

            if main_odd_col and main_odd_col in df.columns:
                strategy_name_parts = self.strategy.get_display_name().split(' ')
                odd_display_name = f"Odd ({strategy_name_parts[-2] if len(strategy_name_parts) >= 2 else 'Alvo'})"
                display_headers.append(odd_display_name)
                header_to_col_map[odd_display_name] = main_odd_col

            header_suffix = f"({strategy_target_name})"
            display_headers.extend([f'P Raw {header_suffix}', f'P Calib {header_suffix}', f'EV {header_suffix}'])
            header_to_col_map.update({
                f'P Raw {header_suffix}': prob_positive_raw_col,
                f'P Calib {header_suffix}': prob_positive_calib_col,
                f'EV {header_suffix}': ev_col
            })

            self.log(f"Exibindo {len(df)} previsões para a estratégia '{self.strategy.get_display_name()}'...")
            self._setup_prediction_columns(display_headers)
            
            df_display = df.copy()

            for col in [prob_positive_raw_col, prob_positive_calib_col]:
                if col in df_display.columns:
                    df_display[col] = pd.to_numeric(df_display[col], errors='coerce').apply(lambda x: f"{x:.1%}" if pd.notna(x) else "-")
            
            if ev_col in df_display.columns:
                df_display[ev_col] = pd.to_numeric(df_display[ev_col], errors='coerce').apply(lambda x: f"{x:+.3f}" if pd.notna(x) else "-")
            
            if main_odd_col and main_odd_col in df_display.columns:
                df_display[main_odd_col] = pd.to_numeric(df_display[main_odd_col], errors='coerce').apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
            
            try:
                self.prediction_tree.tag_configure('highlight_suggestion', background='#3a5f3a', foreground='white')
            except tk.TclError:
                pass
                
            prob_col_for_check = prob_positive_calib_col if prob_positive_calib_col in df.columns else prob_positive_raw_col

            for index, row_original in df.iterrows():
                values_list = [
                    str(df_display.loc[index, header_to_col_map.get(header, '')]) if header_to_col_map.get(header) in df_display.columns else "-"
                    for header in display_headers
                ]
                
                tag_to_apply = ()
                try:
                    ev_val = pd.to_numeric(row_original.get(ev_col), errors='coerce')
                    prob_val = pd.to_numeric(row_original.get(prob_col_for_check), errors='coerce')
                    
                    ev_condition = pd.notna(ev_val) and ev_val > self.optimal_ev_threshold
                    prob_condition = pd.notna(prob_val) and prob_val >= MIN_PROB_THRESHOLD_FOR_HIGHLIGHT
                    
                    if ev_condition and prob_condition:
                        tag_to_apply = ('highlight_suggestion',)
                except Exception:
                    pass

                self.prediction_tree.insert('', 'end', values=values_list, tags=tag_to_apply)

        except Exception as e_disp:
            logger.error(f"Erro GERAL em _update_prediction_display: {e_disp}", exc_info=True)
            self._setup_prediction_columns(['Status'])
            self.prediction_tree.insert('', 'end', values=[f'Erro ao exibir previsões: {e_disp}'])

    # --- Callback Seleção Modelo  ---
    def on_model_select(self, event=None):
        """ Handles model selection, loading model, scaler, CALIBRATOR, THRESHOLD. """
        selected_id = self.selected_model_var.get()
        self.log(f"Modelo selecionado: {selected_id}")
        self._update_gui_for_selected_model(selected_id)

    def _update_gui_for_selected_model(self, selected_id: Optional[str]):

        if selected_id and selected_id in self.loaded_models_data:
            model_data = self.loaded_models_data[selected_id]
            self.log(f"Carregando dados para o modelo: {selected_id}")

            self.selected_model_id = selected_id
            self.trained_model = model_data.get('model')
            self.trained_scaler = model_data.get('scaler')
            self.trained_calibrator = model_data.get('calibrator')
            self.training_medians = model_data.get('training_medians')
            self.feature_columns = model_data.get('features')
            self.model_best_params = model_data.get('params')
            self.model_eval_metrics = model_data.get('metrics')
            self.model_file_timestamp = model_data.get('timestamp')
            self.optimal_ev_threshold = model_data.get('optimal_ev_threshold', DEFAULT_EV_THRESHOLD)
            self.optimal_f1_threshold = model_data.get('optimal_f1_threshold', DEFAULT_F1_THRESHOLD)

            is_ready = all([
                self.trained_model is not None,
                self.feature_columns is not None,
                self.historical_data is not None,
                self.training_medians is not None
            ])

            if is_ready:
                self.predict_button.configure(state="normal")
                self.log(f"Modelo '{selected_id}' pronto para previsão.")
            else:
                self.predict_button.configure(state="disabled")
                reasons = []
                if self.trained_model is None: reasons.append("modelo não carregado")
                if self.feature_columns is None: reasons.append("lista de features ausente")
                if self.historical_data is None: reasons.append("dados históricos não carregados")
                if self.training_medians is None: reasons.append("medianas de treino ausentes")
                self.log(f"Previsão desabilitada para '{selected_id}'. Motivo(s): {', '.join(reasons)}.")
        
        else:
            self.log(f"Seleção de modelo inválida ou limpa. Resetando o estado.")
            self._reset_model_state()
            self.predict_button.configure(state="disabled")
        
        self._update_model_stats_display_gui()

    def start_training_thread(self):
        self.log("Iniciando novo processo de treinamento em background...")
        
        self._reset_model_state() 
        self.selected_model_var.set('')
        try:
            self.model_selector_combo.configure(values=[])
        except tk.TclError:
            pass
        self.loaded_models_data.clear()
        self.available_model_ids.clear()
        
        self._update_model_stats_display_gui()
        
        self.load_train_button.configure(state="disabled")
        self.predict_button.configure(state="disabled")

        self.gui_queue.put(("progress_update", (0, "Iniciando...")))

        try:

            train_thread = threading.Thread(
                target=self._run_training_pipeline,
                daemon=True
            )
            train_thread.start()
            
        except Exception as e_thread:
            error_msg = f"Erro crítico ao iniciar a thread de treinamento: {e_thread}"
            self.log(f"ERRO: {error_msg}")
            logger.critical(error_msg, exc_info=True)
            self.gui_queue.put(("error", ("Erro de Aplicação", error_msg)))
            self.gui_queue.put(("progress_end", None))
            self.load_train_button.configure(state="normal")

    def start_prediction_thread(self):
        
        if not all([self.trained_model, self.selected_model_id, self.historical_data, self.feature_columns, self.training_medians is not None]):
            messagebox.showwarning("Dados Incompletos", "Certifique-se de que um modelo, o histórico e as medianas de treino estão carregados.", parent=self.parent)
            return
        if self.trained_calibrator is None:
            self.log("Aviso: Modelo selecionado não possui calibrador. Usando probs brutas.")
        self.log(f"Iniciando previsão com '{self.selected_model_id}' (Limiar={self.optimal_f1_threshold:.3f})...")
        self.set_button_state(self.load_train_button, tk.DISABLED)
        self.set_button_state(self.predict_button, tk.DISABLED)
        try:
            self._setup_prediction_columns(['Status'])
            self.prediction_tree.insert('', tk.END, values=['Buscando jogos...'])
        except Exception as e:
            self.log(f"Erro limpar treeview: {e}")
        predict_thread = threading.Thread(
            target=self._run_prediction_pipeline,
            daemon=True
        )
        predict_thread.start()

    def _run_training_pipeline(self):
        try:
            self.gui_queue.put(("progress_update", (0, "Carregando dados históricos...")))
            df_hist = load_historical_data()
            if df_hist is None: raise ValueError("Falha ao carregar dados históricos.")
            self.historical_data = df_hist
            self.log("Dados históricos carregados.")

            self.gui_queue.put(("progress_update", (1, f"Processando features para '{self.strategy.get_display_name()}'...")))
            processed_data = preprocess_and_feature_engineer(self.historical_data, self.strategy)
            if processed_data is None: raise ValueError("Falha no pré-processamento dos dados.")
            
            X_processed, y_processed, _ = processed_data
            relevant_odds_cols = self.strategy.get_relevant_odds_cols()
            X_with_odds = self.historical_data.loc[X_processed.index, relevant_odds_cols].copy()

            success = run_training_process(
                X=X_processed, y=y_processed, X_with_odds=X_with_odds,
                strategy=self.strategy, progress_callback_stages=self._training_progress_callback,
                **self._get_training_params_as_dict()
            )
            if not success: raise RuntimeError("O processo de treinamento do modelo falhou.")
            self.gui_queue.put(("training_succeeded", None))
        except Exception as e:
            error_msg = f"Erro no pipeline de treino: {e}"
            self.log(f"ERRO: {error_msg}")
            logger.error(error_msg, exc_info=True)
            self.gui_queue.put(("error", ("Erro no Treinamento", error_msg)))
        finally:
            self.gui_queue.put(("progress_end", None))
            self.load_train_button.configure(state="normal")

    def _run_prediction_pipeline(self):

        try:
            self.gui_queue.put(("progress_start", (100,)))
            self.gui_queue.put(("progress_update", (10, "Buscando jogos futuros...")))
            fixture_df = fetch_and_process_fixtures()

            if fixture_df is None or fixture_df.empty:
                self.log("Nenhum jogo encontrado para o dia alvo.")
                self.gui_queue.put(("prediction_complete", pd.DataFrame()))
                return

            self.gui_queue.put(("progress_update", (30, f"Preparando features para {len(fixture_df)} jogos...")))
            
            X_fixtures_prepared = prepare_fixture_data(
                fixture_df,
                self.historical_data,
                self.strategy,
                training_medians=self.training_medians
            )

            if X_fixtures_prepared is None or X_fixtures_prepared.empty:
                self.log("Nenhum jogo restante para prever após a preparação.")
                self.gui_queue.put(("prediction_complete", pd.DataFrame()))
                return

            self.gui_queue.put(("progress_update", (60, f"Realizando previsões para {len(X_fixtures_prepared)} jogos...")))
            
            df_predictions = predictor.make_predictions(
                model=self.trained_model,
                scaler=self.trained_scaler,
                calibrator=self.trained_calibrator,
                strategy=self.strategy, 
                feature_names=self.feature_columns,
                X_fixture_prepared=X_fixtures_prepared,
                fixture_info=fixture_df.loc[X_fixtures_prepared.index]
            )

            if df_predictions is None or df_predictions.empty:
                self.log("Falha ao gerar previsões.")
                self.gui_queue.put(("prediction_complete", pd.DataFrame()))
                return

            self.log(f"Previsões geradas para {len(df_predictions)} jogos.")
            
            target_name = self.strategy.get_target_variable_name()
            prob_col_to_sort_by = f'Prob_{target_name}'
            df_sorted = df_predictions.copy()

            if prob_col_to_sort_by in df_sorted.columns and df_sorted[prob_col_to_sort_by].notna().any():
                self.log(f"Ordenando previsões por '{prob_col_to_sort_by}'...")
                df_sorted = df_sorted.sort_values(by=prob_col_to_sort_by, ascending=False, na_position='last').reset_index(drop=True)

            self.gui_queue.put(("progress_update", (95, "Preparando Exibição...")))
            self.gui_queue.put(("prediction_complete", df_sorted))

        except Exception as e:
            error_msg = f"Erro no Pipeline de Previsão: {e}"
            self.log(f"ERRO: {error_msg}")
            logger.error(error_msg, exc_info=True)
            self.gui_queue.put(("error", ("Erro na Previsão", error_msg)))
            self.gui_queue.put(("prediction_complete", pd.DataFrame()))
        finally:
            self.gui_queue.put(("progress_end", None))
            self.load_train_button.configure(state="normal")
            if self.selected_model_id:
                self.predict_button.configure(state="normal")

    def load_existing_model_assets(self):
        self.log(f"Carregando assets para a estratégia: {self.strategy.get_display_name()}")
        prefix = self.strategy.get_model_config_key_prefix()
        f1_model_path = os.path.join(DATA_DIR, f"{prefix}_best_f1.joblib")
        roi_model_path = os.path.join(DATA_DIR, f"{prefix}_best_roi.joblib")
        
        MODEL_ID_F1 = f"Melhor F1 ({prefix})"
        MODEL_ID_ROI = f"Melhor ROI ({prefix})"
        
        self.loaded_models_data.clear()
        self.available_model_ids.clear()
        
        for model_id, model_path in {MODEL_ID_F1: f1_model_path, MODEL_ID_ROI: roi_model_path}.items():
            if not os.path.exists(model_path): continue
            
            load_result = predictor.load_model_scaler_features(model_path)
            if not load_result: continue
                
            (model, scaler, calibrator, ev_thr, 
             f1_thr, training_medians, features, 
             params, metrics, timestamp) = load_result
            
            if model and features:
                self.loaded_models_data[model_id] = {'model': model, 
                                                     'scaler': scaler, 
                                                     'calibrator': calibrator, 
                                                     'training_medians': training_medians, 
                                                     'features': features, 
                                                     'optimal_ev_threshold': ev_thr, 
                                                     'optimal_f1_threshold': f1_thr, 
                                                     'params': params, 
                                                     'metrics': metrics, 
                                                     'timestamp': timestamp, 
                                                     'path': model_path}
                self.available_model_ids.append(model_id)
                
        if self.available_model_ids:
            self.model_selector_combo.configure(values=self.available_model_ids)
            self.selected_model_var.set(self.available_model_ids[0])
            self.on_model_select(None)
        
        if self.historical_data is None:
            self.log("Carregando dados históricos de fundo...")
            self.historical_data = load_historical_data()
            if self.historical_data is not None:
                self.log("Histórico carregado.")
            else:
                self.log("Falha carregar histórico.")

        if self.selected_model_id and self.historical_data is not None:
            self.set_button_state(self.predict_button, tk.NORMAL)
            self.log("Pronto para previsão.")
        else:
            self.set_button_state(self.predict_button, tk.DISABLED)


    def process_gui_queue(self):
        if self.stop_processing_queue:
            logger.debug("PredictorDashboard: Parando fila GUI.")
            return 

        try:
            while True:
                try:
                    message = self.gui_queue.get_nowait()
                    msg_type, msg_payload = message
                except Empty:
                    break 
                except (ValueError, TypeError):
                    logger.error(f"AVISO GUI (Predictor): Erro unpack msg: {message}")
                    continue
                except Exception as e_get:
                    logger.error(f"Erro get fila GUI (Predictor): {e_get}")
                    continue

                try:
                    if msg_type == "log":
                        self._update_log_area(str(msg_payload))
                    elif msg_type == "button_state":
                        self._update_button_state(msg_payload)
                    elif msg_type == "update_stats_gui":
                        self._update_model_stats_display_gui()
                    elif msg_type == "error":
                        parent_widget = self.parent if hasattr(self, 'parent') and self.parent.winfo_exists() else self.main_tk_root
                        messagebox.showerror(msg_payload[0], msg_payload[1], parent=parent_widget)
                    elif msg_type == "info":
                        parent_widget = self.parent if hasattr(self, 'parent') and self.parent.winfo_exists() else self.main_tk_root
                        messagebox.showinfo(msg_payload[0], msg_payload[1], parent=parent_widget)
                    elif msg_type == "progress_start":
                        if hasattr(self, 'progress_bar') and self.progress_bar.winfo_exists():
                            max_val = msg_payload[0] if isinstance(msg_payload, (tuple, list)) and msg_payload else 100
                            self.progress_bar.config(maximum=max_val, value=0)
                        if hasattr(self, 'progress_label') and self.progress_label.winfo_exists():
                            self.progress_label.config(text="Iniciando...")
                    elif msg_type == "progress_update":
                        if isinstance(msg_payload, tuple) and len(msg_payload) == 2:
                            value, status_text = msg_payload
                            if hasattr(self, 'progress_bar') and self.progress_bar.winfo_exists():
                                self.progress_bar['value'] = value
                            if hasattr(self, 'progress_label') and self.progress_label.winfo_exists():
                                self.progress_label.config(text=str(status_text))
                    elif msg_type == "progress_end":
                        if hasattr(self, 'progress_bar') and self.progress_bar.winfo_exists():
                            self.progress_bar['value'] = 0
                        if hasattr(self, 'progress_label') and self.progress_label.winfo_exists():
                            self.progress_label.config(text="Pronto.")
                    elif msg_type == "training_succeeded":
                        self.log("Treino OK. Recarregando modelos...")
                        self.load_existing_model_assets()
                    elif msg_type == "training_failed":
                        self.log("ERRO: Treino falhou.")
                        self.selected_model_id = None; self.trained_model = None; self.trained_scaler = None
                        self.trained_calibrator = None; self.optimal_f1_threshold=DEFAULT_F1_THRESHOLD;
                        self.feature_columns = None; self.model_best_params = None; self.model_eval_metrics = None
                        self.model_file_timestamp = None; self.selected_model_var.set("")
                        try:
                            if hasattr(self, 'model_selector_combo') and self.model_selector_combo.winfo_exists():
                                self.model_selector_combo.config(values=[])
                        except tk.TclError: pass
                        self._update_model_stats_display_gui()
                        self.set_button_state(self.predict_button, tk.DISABLED) 
                        self.set_button_state(self.load_train_button, tk.NORMAL) 
                    elif msg_type == "prediction_complete":
                        df_preds = msg_payload
                        self.log("Recebidas previsões completas para exibição.")
                        self._update_prediction_display(df_preds)
                        self.set_button_state(self.load_train_button, tk.NORMAL) 
                        if self.selected_model_id: self.set_button_state(self.predict_button, tk.NORMAL)
                    else:
                        self.log(f"AVISO GUI (Predictor): Msg desconhecida: {msg_type}")
                except tk.TclError:
                    pass 
                except Exception as e_proc:
                    logger.error(f"Erro processar msg (Predictor) '{msg_type}': {e_proc}", exc_info=True)

        except Exception as e_loop:
            logger.error(f"Erro CRÍTICO loop fila GUI (Predictor): {e_loop}", exc_info=True)
        finally:
            if not self.stop_processing_queue:
                try:
                    if hasattr(self.main_tk_root, 'winfo_exists') and self.main_tk_root.winfo_exists():
                        self.main_tk_root.after(100, self.process_gui_queue)
                except Exception as e_resched:
                    if not self.stop_processing_queue:
                        logger.error(f"Erro reagendar fila GUI (Predictor): {e_resched}")

    def start_backtest_thread(self):
        self.log("Iniciando backtest da estratégia de regras...")
        self.backtest_button.configure(state="disabled")
        self.find_entries_button.configure(state="disabled")
        threading.Thread(target=self._run_rule_based_backtest, daemon=True).start()
        
    def _run_rule_based_backtest(self):
        try:
            self.gui_queue.put(("progress_update", (0, "Carregando histórico...")))
            df_hist = load_historical_data()
            if df_hist is None or df_hist.empty:
                raise ValueError("Dados históricos não puderam ser carregados.")
            
            backtester = RuleBasedBacktester(self.strategy)
            success = backtester.run(df_hist)
            results_text = backtester.get_results_as_text()
            
            if success:
                self.gui_queue.put(("display_backtest_results", results_text))
            else:
                self.gui_queue.put(("error", ("Erro no Backtest", results_text)))
        except Exception as e:
            self.gui_queue.put(("error", ("Erro Crítico no Backtest", str(e))))
        finally:
            self.gui_queue.put(("progress_end", None))
            self.backtest_button.configure(state="normal")
            self.find_entries_button.configure(state="normal")
            
    def start_find_entries_thread(self):
        self.log("Buscando entradas futuras com a estratégia de regras...")
        self.backtest_button.configure(state="disabled")
        self.find_entries_button.configure(state="disabled")
        threading.Thread(target=self._run_find_entries_pipeline, daemon=True).start()

    def _run_find_entries_pipeline(self):
        try:
            fixture_df = fetch_and_process_fixtures()
            if fixture_df is None or fixture_df.empty:
                self.log("Nenhum jogo futuro encontrado.")
                self.gui_queue.put(("prediction_complete", pd.DataFrame()))
                return

            entradas_df = self.strategy.find_entries(fixture_df)
            self.log(f"Encontradas {len(entradas_df)} entradas futuras.")
            self.gui_queue.put(("prediction_complete", entradas_df))
        except Exception as e:
            self.gui_queue.put(("error", ("Erro ao Buscar Entradas", str(e))))
        finally:
            self.backtest_button.configure(state="normal")
            self.find_entries_button.configure(state="normal")