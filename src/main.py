# --- src/main.py ---
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
# Removidos imports não usados mais nesta classe: Listbox, MULTIPLE, Scrollbar, io
from tkinter.scrolledtext import ScrolledText
import sys, os, threading, datetime, math, numpy as np # Removido io
from queue import Queue, Empty
import pandas as pd
from typing import Optional, Dict, List, Any
from logger_config import setup_logger

logger = setup_logger("Main2")# Adiciona diretórios e importa módulos
# (Mantém o setup de path para caso seja necessário em algum contexto)
try:
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(SRC_DIR)
    if SRC_DIR not in sys.path: sys.path.append(SRC_DIR)
    if BASE_DIR not in sys.path: sys.path.append(BASE_DIR)
except NameError: # Avoid error in environments where __file__ is not defined
    pass

try:
    # Imports necessários para Treino/Previsão
    from config import (
        CLASS_NAMES, FIXTURE_FETCH_DAY,
        MODEL_TYPE_NAME, ODDS_COLS as CONFIG_ODDS_COLS,
        BEST_F1_MODEL_SAVE_PATH, BEST_ROI_MODEL_SAVE_PATH, MODEL_ID_F1, MODEL_ID_ROI,
        FEATURE_COLUMNS, DEFAULT_EV_THRESHOLD # FEATURE_COLUMNS é usado no prepare_fixture_data e pipeline de treino
        # Removido MODEL_CONFIG se optimize_single_model for removido ou não usado aqui
        # Removido ROLLING_WINDOW se não for usado diretamente aqui (é usado em data_handler)
    )
    # Imports de data_handler necessários
    from data_handler import (
        load_historical_data,
        preprocess_and_feature_engineer, # Usado em _run_training_pipeline
        fetch_and_process_fixtures,      # Usado em _run_prediction_pipeline
        prepare_fixture_data,            # Usado em _run_prediction_pipeline
        calculate_historical_intermediate # Usado em _run_training_pipeline (para alinhar dados p/ ROI)
        # Removido calculate_rolling_stats, calculate_derived_features se chamados apenas dentro dos pipelines
    )
    # Imports de model_trainer necessários
    from model_trainer import train_evaluate_and_save_best_models as run_training_process
    # Removido analyze_features, optimize_single_model se não forem chamados aqui
    import predictor # Necessário para load_model_scaler_features e make_predictions
    import requests # Usado em fetch_and_process_fixtures
    from sklearn.model_selection import train_test_split # Usado em _run_training_pipeline
    import traceback # Para log de erros
except ImportError as e:
    logger.error(f"Erro import main.py (Treino/Previsão): {e}")
    raise # Re-levanta erro para app_launcher
except Exception as e_i:
    logger.error(f"Erro geral import main.py (Treino/Previsão): {e_i}")
    raise # Re-levanta erro

from typing import Optional, Dict, List, Any # Garante typing

class FootballPredictorDashboard:
    def __init__(self, parent_frame, main_root):
        self.parent = parent_frame
        self.main_tk_root = main_root

        self.gui_queue = Queue()
        self.historical_data: Optional[pd.DataFrame] = None
        self.loaded_models_data: Dict[str, Dict] = {}
        self.available_model_ids: List[str] = []
        self.selected_model_id: Optional[str] = None
        # --- Novos atributos para Calibrador e Limiar ---
        self.trained_model: Optional[Any] = None
        self.trained_scaler: Optional[Any] = None
        self.trained_calibrator: Optional[Any] = None # NOVO
        self.optimal_threshold: float = 0.5 # NOVO (default)
        # --- Fim Novos Atributos ---
        self.feature_columns: Optional[List[str]] = None
        self.model_best_params: Optional[Dict] = None
        self.model_eval_metrics: Optional[Dict] = None
        self.model_file_timestamp: Optional[str] = None
        self.optimal_ev_threshold: float = DEFAULT_EV_THRESHOLD

        self.create_train_predict_widgets()
        self.main_tk_root.after(100, self.process_gui_queue)
        self.log(f"Aba Treino/Previsão Inicializada ({MODEL_TYPE_NAME})")
        self.log(f"Fonte CSV: '{FIXTURE_FETCH_DAY}'.")
        self.log("Carregando modelos e histórico...")
        self.load_existing_model_assets()

    # --- Widgets (`create_train_predict_widgets`) ---
    # (Nenhuma mudança necessária aqui, apenas no display)
    def create_train_predict_widgets(self):
        # ... (código como antes, cria botões, combobox, progress, stats_text, prediction_tree, log_area) ...
        main_frame = ttk.Frame(self.parent, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Painel esquerdo
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), pady=5)

        # Controles
        control_frame = ttk.LabelFrame(left_panel, text=" Ações ", padding="10")
        control_frame.pack(pady=(0, 5), fill=tk.X)

        self.load_train_button = ttk.Button(
            control_frame, text="TREINAR e Salvar Melhores Modelos",
            command=self.start_training_thread, width=35
        )
        self.load_train_button.pack(pady=5, fill=tk.X)

        predict_frame = ttk.Frame(control_frame)
        predict_frame.pack(fill=tk.X, pady=5)

        self.predict_button = ttk.Button(
            predict_frame, text=f"Prever ({FIXTURE_FETCH_DAY.capitalize()}) com:",
            command=self.start_prediction_thread, width=18
        )
        self.predict_button.pack(side=tk.LEFT, fill=tk.X, expand=False)
        self.predict_button.config(state=tk.DISABLED)

        self.selected_model_var = tk.StringVar()
        self.model_selector_combo = ttk.Combobox(
            predict_frame, textvariable=self.selected_model_var, state="readonly", width=20
        )
        self.model_selector_combo.pack(side=tk.RIGHT, padx=(5, 0), fill=tk.X, expand=True)
        self.model_selector_combo.bind("<<ComboboxSelected>>", self.on_model_select)

        progress_frame = ttk.Frame(control_frame)
        progress_frame.pack(fill=tk.X, pady=(10, 0))

        self.progress_label = ttk.Label(progress_frame, text="Pronto.")
        self.progress_label.pack(side=tk.LEFT, padx=(0, 5))

        self.progress_bar = ttk.Progressbar(
            progress_frame, orient=tk.HORIZONTAL, length=200, mode='determinate'
        )
        self.progress_bar.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        # Status do modelo
        stats_frame = ttk.LabelFrame(left_panel, text=" Status Modelo Selecionado ", padding="10")
        stats_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        self.model_stats_text = ScrolledText(
            stats_frame, height=15, state='disabled', wrap=tk.WORD,
            font=("Consolas", 9), relief=tk.FLAT, bd=0
        )
        self.model_stats_text.pack(fill=tk.BOTH, expand=True)

        # Painel direito
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Previsões
        results_frame = ttk.LabelFrame(right_panel, text=" Previsões ", padding="5")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        cols = [
            'Data', 'Hora', 'Liga', 'Casa', 'Fora', 'Odd H', 'Odd D', 'Odd A',
            'P(Ñ Emp)', 'P(Empate)', 'P(Emp Calib)', 'EV EMPATE'
        ]

        self.prediction_tree = ttk.Treeview(
            results_frame, columns=cols, show='headings', height=10
        )
        vsb = ttk.Scrollbar(results_frame, orient="vertical", command=self.prediction_tree.yview)
        hsb = ttk.Scrollbar(results_frame, orient="horizontal", command=self.prediction_tree.xview)
        self.prediction_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side='right', fill='y')
        hsb.pack(side='bottom', fill='x')
        self.prediction_tree.pack(fill=tk.BOTH, expand=True)
        self._setup_prediction_columns(cols)

        # Logs
        log_frame = ttk.LabelFrame(right_panel, text=" Logs ", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True, side=tk.BOTTOM, pady=(5, 0))

        self.log_area = ScrolledText(
            log_frame, height=8, state='disabled', wrap=tk.WORD, font=("Consolas", 9)
        )
        self.log_area.pack(fill=tk.BOTH, expand=True)

        self._update_model_stats_display_gui()  # Atualiza display inicial


    # --- Métodos Auxiliares GUI (log, _update_log_area, set_button_state, _update_button_state) ---
    # (Sem alterações necessárias aqui)
    def log(self, message: str): self.gui_queue.put(("log", message))
    def _update_log_area(self, message: str):
         try:
             if hasattr(self, 'log_area') and self.log_area.winfo_exists():
                 self.log_area.config(state='normal'); ts = datetime.datetime.now().strftime("%H:%M:%S"); self.log_area.insert(tk.END, f"[{ts}] {message}\n"); self.log_area.config(state='disabled'); self.log_area.see(tk.END)
         except tk.TclError: pass
    def set_button_state(self, button: ttk.Button, state: str): self.gui_queue.put(("button_state", (button, state)))
    def _update_button_state(self, button_state_tuple):
         button, state = button_state_tuple
         try:
             if button.winfo_exists(): button.config(state=state)
         except tk.TclError: pass

    # --- : Atualiza Display de Stats do Modelo ---
    def _update_model_stats_display_gui(self):
        """ Atualiza display de stats focando na estratégia EV final. """
        try:
            if not hasattr(self, 'model_stats_text') or not self.model_stats_text.winfo_exists(): return
            self.model_stats_text.config(state='normal')
            self.model_stats_text.delete('1.0', tk.END)

            if self.trained_model is None or self.selected_model_id is None:
                stats_content = "Nenhum modelo selecionado."
            else:
                model_data = self.loaded_models_data.get(self.selected_model_id, {})
                metrics = model_data.get('metrics', {}) # Métricas salvas do teste
                params = model_data.get('params')
                features = model_data.get('features', [])
                timestamp = self.model_file_timestamp
                path = model_data.get('path')
                model_class_name = self.trained_model.__class__.__name__ if self.trained_model else "N/A"
                optimal_ev_th = model_data.get('optimal_ev_threshold', DEFAULT_EV_THRESHOLD) # Pega limiar EV
                calibrator_loaded = model_data.get('calibrator') is not None

                was_optimized = abs(optimal_ev_th - DEFAULT_EV_THRESHOLD) > 1e-6 if isinstance(optimal_ev_th, (int,float)) else False

                # --- Seção 1: Informações Gerais ---
                stats_content = f"Modelo Selecionado: {self.selected_model_id}\n"
                stats_content += f"  Tipo: {model_class_name}\n"
                stats_content += f"  Calibrado (Isotonic): {'Sim' if calibrator_loaded else 'Não'}\n"
                stats_content += f"  Limiar EV Otimizado: {optimal_ev_th:.4f}\n" if isinstance(optimal_ev_th, (int, float)) else f"  Limiar EV: {optimal_ev_th}\n"
                stats_content += f"  Arquivo: {os.path.basename(path or 'N/A')}\n"
                stats_content += f"  Modificado: {timestamp or 'N/A'}\n"
                if params: stats_content += f"  Params: {params}\n"
                if features: stats_content += f"  Features ({len(features)}): {', '.join(features)}\n"
                stats_content += "---\n"

                # --- Seção 2: Métricas de Qualidade da Previsão (Teste) ---
                acc = metrics.get('accuracy')
                f1_d = metrics.get('f1_score_draw')  # F1 @ 0.5 (ainda útil como ref)
                auc = metrics.get('roc_auc')         # AUC das probs calibradas
                brier = metrics.get('brier_score')   # Brier das probs calibradas
                test_n = metrics.get('test_set_size', 'N/A')
                prec = metrics.get('precision_draw')      # Precision das probs calibradas

                stats_content += "Métricas de Qualidade (Conjunto Teste):\n"
                stats_content += f"- Acurácia: {acc:.4f}\n" if acc is not None else "- Acc: N/A\n"
                stats_content += f"- F1 Empate (@ Thr 0.5): {f1_d:.4f}\n" if f1_d is not None else "- F1@0.5: N/A\n"
                stats_content += f"- ROC AUC (Pós-Calib): {auc:.4f}\n" if auc is not None else "- AUC: N/A\n"
                stats_content += f"- Brier Score (Pós-Calib): {brier:.4f}\n" if brier is not None else "- Brier: N/A\n"
                stats_content += f"- Precision (Pós-Calib): {prec:.4f}\n" if prec is not None else "- Precision: N/A\n"
                stats_content += f"- Tamanho Teste: {test_n}\n"
                stats_content += "---\n"

                # --- Seção 3: Resultado da Estratégia EV (Teste) ---
                profit_ev = metrics.get('profit')   # Profit com Limiar EV
                roi_ev = metrics.get('roi')         # ROI com Limiar EV
                n_bets_ev = metrics.get('num_bets') # Bets com Limiar EV

                stats_content += f"Estratégia de Aposta (EV > {optimal_ev_th:.3f} no Teste):\n" if isinstance(optimal_ev_th,(int,float)) else "Estratégia de Aposta (Teste):\n"
                stats_content += f"- Nº de Apostas Sugeridas: {n_bets_ev if n_bets_ev is not None else 'N/A'}\n"
                profit_ev_str = f"{profit_ev:.2f} u" if profit_ev is not None else "N/A"
                roi_ev_str = "N/A"
                if roi_ev is not None:
                    try:
                         if isinstance(roi_ev, (float, np.number)) and not np.isnan(roi_ev): roi_ev_str = f"{roi_ev:.2f} %"
                         elif isinstance(roi_ev, (int, float)): roi_ev_str = f"{roi_ev:.2f} %"
                    except TypeError: pass
                stats_content += f"- Lucro/Prejuízo: {profit_ev_str}\n"
                stats_content += f"- ROI Calculado: {roi_ev_str}\n"
            
            # Atualiza o widget
            self.model_stats_text.insert('1.0', stats_content)
            self.model_stats_text.config(state='disabled')
        except tk.TclError: pass
        except Exception as e: logger.error(f"Erro _update_model_stats_display_gui: {e}"); traceback.logger.error_exc()

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
                'EV Empate': 70,  # Colunas chave
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
            # ... (verificação inicial da treeview como antes) ...
            if not hasattr(self, 'prediction_tree') or not self.prediction_tree.winfo_exists(): return
        except tk.TclError: return

        self.log(f"--- GUI: Atualizando display previsões (Format + Highlight EV > Thr) ---")
        # HEADERS PARA EXIBIR
        # Definir nomes claros, ex: P(E) Raw, P(E) Calib
        display_headers = ['Data', 'Hora', 'Liga', 'Casa', 'Fora',
                           'Odd D', 'P(E) Raw', 'P(E) Calib', 'EV Empate']

        # Configurar a tag de destaque
        try:
             self.prediction_tree.tag_configure('highlight_ev', background='lightgreen', foreground='black') # Ou outra cor
        except tk.TclError: pass # Ignora se já configurado ou erro

        if df is None or df.empty:
            self.log("GUI: DF vazio/None."); self._setup_prediction_columns(['Status']); self.prediction_tree.insert('', tk.END, values=['Nenhuma previsão gerada.']); return

        self.log(f"GUI: DF {df.shape}. Reconfigurando..."); self._setup_prediction_columns(display_headers)

        # Mapeamento Header -> Coluna Interna (vinda do predictor)
        odds_d_col = CONFIG_ODDS_COLS.get('draw', 'Odd_D_FT');
        # Nomes das colunas Raw e Calib
        prob_draw_raw_col = f'ProbRaw_{CLASS_NAMES[1]}' if CLASS_NAMES and len(CLASS_NAMES)>1 else 'ProbRaw_Empate'
        prob_draw_calib_col = f'ProbCalib_{CLASS_NAMES[1]}' if CLASS_NAMES and len(CLASS_NAMES)>1 else 'ProbCalib_Empate'
        ev_col = 'EV_Empate';

        header_to_col_map = { 'Data': 'Date_Str', 'Hora': 'Time_Str', 'Liga': 'League', 'Casa': 'HomeTeam', 'Fora': 'AwayTeam', 'Odd D': odds_d_col, 'P(E) Raw': prob_draw_raw_col, 'P(E) Calib': prob_draw_calib_col, 'EV Empate': ev_col }
        valid_internal_cols_map = {h: c for h, c in header_to_col_map.items() if c in df.columns}
        if not valid_internal_cols_map or ev_col not in valid_internal_cols_map.values(): self.log(f"ERRO GUI: Cols essenciais (incluindo '{ev_col}') ausentes!"); return

        try:
            self.log(f"GUI: Formatando cols: {list(valid_internal_cols_map.values())}")
            df_display = df[list(valid_internal_cols_map.values())].copy()

            # --- *** ADICIONADO: Formatação de Probabilidades *** ---
            prob_cols_to_format = [prob_draw_raw_col, prob_draw_calib_col]
            for pcol in prob_cols_to_format:
                if pcol in df_display.columns:
                    try:
                        # Converte para numérico, multiplica por 100, arredonda, formata como %
                        numeric_probs = pd.to_numeric(df_display[pcol], errors='coerce')
                        formatted_probs = (numeric_probs * 100).round(1).astype(str) + '%'
                        # Substitui 'nan%' por '-'
                        df_display[pcol] = formatted_probs.replace('nan%', '-', regex=False)
                        # Log para verificar a formatação
                        # self.log(f"DEBUG Format: Coluna {pcol} formatada.")
                    except Exception as e:
                        self.log(f"Aviso format prob {pcol}: {e}")
                        df_display[pcol] = "-" # Fallback
            # --- *** FIM DA FORMATAÇÃO DE PROBS *** ---

            # Formatação EV e Odd D (como antes)
            if ev_col in df_display.columns: 
                try: df_display[ev_col] = pd.to_numeric(df_display[ev_col], errors='coerce').apply(lambda x: f"{x:+.3f}" if pd.notna(x) else "-"); 
                except Exception as e: self.log(f"Aviso format {ev_col}: {e}"); df_display[ev_col] = "-";
            if odds_d_col in df_display.columns: 
                try: df_display[odds_d_col] = pd.to_numeric(df_display[odds_d_col], errors='coerce').apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-"); 
                except Exception as e: self.log(f"Aviso format {odds_d_col}: {e}"); df_display[odds_d_col] = "-";


            # Insere linhas E APLICA TAG
            self.log("GUI: Adicionando linhas formatadas e aplicando tags..."); added_rows = 0
            found_positive_ev = False # Flag para verificar se algum EV positivo foi encontrado
            for index, row in df_display.iterrows():
                values = [str(row.get(valid_internal_cols_map.get(h), '')) for h in display_headers]
                tag_to_apply = ()

                # Lógica para aplicar a tag (como antes)
                try:
                     ev_val_orig = pd.to_numeric(df.loc[index, ev_col], errors='coerce')
                     # Usa limiar EV da classe (self.optimal_ev_threshold)
                     if pd.notna(ev_val_orig) and ev_val_orig > self.optimal_ev_threshold:
                          tag_to_apply = ('highlight_ev',)
                          found_positive_ev = True # Marca que encontramos pelo menos um
                except Exception as e_tag: self.log(f"Aviso tag linha {index}: {e_tag}")

                try:
                    self.prediction_tree.insert('', tk.END, values=values, tags=tag_to_apply)
                    added_rows += 1
                except Exception as e_ins: self.log(f"!! Erro inserir linha {index}: {e_ins}")

            # Log se nenhuma linha foi destacada
            if not found_positive_ev and added_rows > 0:
                 self.log(f"INFO: Nenhuma previsão atingiu o limiar de EV ({self.optimal_ev_threshold:.3f}) para destaque.")

            self.log(f"GUI: {added_rows}/{len(df_display)} linhas adicionadas.")

        except Exception as e: self.log(f"!! Erro GERAL display: {e}"); traceback.logger.error_exc()

    # --- Callback Seleção Modelo  ---
    def on_model_select(self, event=None):
        """ Handles model selection, loading model, scaler, CALIBRATOR, THRESHOLD. """
        selected_id = self.selected_model_var.get()
        self.log(f"Modelo selecionado: {selected_id}")
        self._update_gui_for_selected_model(selected_id)

    def _update_gui_for_selected_model(self, selected_id: Optional[str]):
        """Atualiza estado interno e GUI para o modelo selecionado."""
        if selected_id and selected_id in self.loaded_models_data:
            model_data = self.loaded_models_data[selected_id]
            self.log(f"Carregando dados internos para: {selected_id}")
            self.selected_model_id = selected_id
            self.trained_model = model_data.get('model')
            self.trained_scaler = model_data.get('scaler')
            # --- Carrega Calibrador e Limiar ---
            self.trained_calibrator = model_data.get('calibrator')
            self.optimal_threshold = model_data.get('optimal_threshold', 0.5) # Usa default se não encontrar
            self.feature_columns = model_data.get('features')
            self.model_best_params = model_data.get('params')
            self.model_eval_metrics = model_data.get('metrics')
            self.model_file_timestamp = model_data.get('timestamp')
            self.optimal_ev_threshold = model_data.get('optimal_ev_threshold', DEFAULT_EV_THRESHOLD)

            self._update_model_stats_display_gui() # Mostra stats (incluindo limiar/calib)

            # Habilita prever se tudo OK
            if self.trained_model and self.feature_columns and self.historical_data is not None:
                self.set_button_state(self.predict_button, tk.NORMAL)
                self.log(f"Modelo '{selected_id}' pronto para previsão (Limiar={self.optimal_threshold:.3f}).")
            else:
                # Loga o motivo de desabilitar
                reasons = []
                if not self.trained_model: reasons.append("modelo não carregado")
                if not self.feature_columns: reasons.append("features ausentes")
                if self.historical_data is None: reasons.append("histórico ausente")
                self.log(f"Previsão desabilitada para '{selected_id}'. Motivo(s): {', '.join(reasons)}.")
                self.set_button_state(self.predict_button, tk.DISABLED)
        else:
            # Limpa estado se seleção for inválida
            self.log(f"Seleção inválida/limpa: '{selected_id}'. Resetando.")
            self.selected_model_id = None
            self.trained_model = None; self.trained_scaler = None; self.trained_calibrator = None;
            self.optimal_ev_threshold=DEFAULT_EV_THRESHOLD; 
            self.optimal_threshold = 0.5; self.feature_columns = None; self.model_best_params = None;
            self.model_eval_metrics = None; self.model_file_timestamp = None;
            self.set_button_state(self.predict_button, tk.DISABLED)
            self._update_model_stats_display_gui() # Mostra 'nenhum modelo'

    # --- Funções de Ação (start_training_thread, start_prediction_thread) ---
    # (Sem alterações na lógica de iniciar a thread)
    def start_training_thread(self):
        # ... (código como antes para limpar estado, desabilitar botões, iniciar thread _run_training_pipeline) ...
        self.log("Iniciando processo de treino em background...")
        self.loaded_models_data = {}
        self.available_model_ids = []
        self.selected_model_var.set('')
        try:
            self.model_selector_combo.config(values=[])
        except tk.TclError:
            pass
        self.selected_model_id = None
        self.trained_model = None
        self.trained_scaler = None
        self.trained_calibrator = None
        self.optimal_threshold = 0.5
        self.feature_columns = None
        self.model_best_params = None
        self.model_eval_metrics = None
        self.model_file_timestamp = None
        self.gui_queue.put(("update_stats_gui", None))
        self.set_button_state(self.load_train_button, tk.DISABLED)
        self.set_button_state(self.predict_button, tk.DISABLED)
        self.gui_queue.put(("progress_start", (100,)))
        self.gui_queue.put(("progress_update", (5, "Carregando Histórico...")))
        try:
            df_hist = load_historical_data()
            if df_hist is None:
                raise ValueError("Falha carregar histórico.")
            self.historical_data = df_hist
            self.log("Histórico carregado.")
            self.gui_queue.put(("progress_update", (20, "Iniciando Thread Treino...")))
            train_thread = threading.Thread(
                target=self._run_training_pipeline,
                args=(self.historical_data.copy(),),
                kwargs={'optimize_ev': True},
                daemon=True
            )
            train_thread.start()
        except Exception as e_load:
            error_msg = f"Erro Carregar Histórico: {e_load}"
            self.log(f"ERRO: {error_msg}")
            self.gui_queue.put(("error", ("Erro Carregamento", error_msg)))
            self.gui_queue.put(("progress_end", None))
            self.set_button_state(self.load_train_button, tk.NORMAL)


    def start_prediction_thread(self):
        # (código como antes, mas verifica se calibrador/limiar foram carregados se precisar deles aqui)
        if self.trained_model is None or self.selected_model_id is None:
            messagebox.showwarning("Modelo Não Selecionado", "Selecione um modelo treinado.", parent=self.parent)
            return
        if self.historical_data is None:
            messagebox.showwarning("Histórico Ausente", "Carregue/Treine.", parent=self.parent)
            return
        if not self.feature_columns:
            messagebox.showwarning("Features Ausentes", "Features do modelo não carregadas.", parent=self.parent)
            return
        # Adicional: Avisar se não houver calibrador/limiar? Ou deixar predictor lidar?
        if self.trained_calibrator is None:
            self.log("Aviso: Modelo selecionado não possui calibrador. Usando probs brutas.")
        self.log(f"Iniciando previsão com '{self.selected_model_id}' (Limiar={self.optimal_threshold:.3f})...")
        self.set_button_state(self.load_train_button, tk.DISABLED)
        self.set_button_state(self.predict_button, tk.DISABLED)
        try:
            self._setup_prediction_columns(['Status'])
            self.prediction_tree.insert('', tk.END, values=['Buscando jogos...'])
        except Exception as e:
            self.log(f"Erro limpar treeview: {e}")
        predict_thread = threading.Thread(
            target=self._run_prediction_pipeline,
            kwargs={'odd_draw_col': CONFIG_ODDS_COLS.get('draw', 'Odd_D_FT')},
            daemon=True
        )
        predict_thread.start()


    def _run_training_pipeline(self, df_hist_raw: pd.DataFrame, optimize_ev: bool = True):
        """Pipeline de treinamento, incluindo pré-processamento, treinamento e salvamento."""
        training_successful = False
        try:
            self.gui_queue.put(("progress_update", (30, "Pré-processando...")))
            processed_data = preprocess_and_feature_engineer(df_hist_raw)
            if processed_data is None:
                raise ValueError("Falha pré-processamento.")
            X_processed, y_processed, features_used = processed_data
            self.log(f"Pré-processamento OK. Features: {features_used}")

            self.gui_queue.put(("progress_update", (50, "Alinhando dados com odds...")))
            df_full_data_aligned_for_split = None  # Reset
            try:
                df_hist_intermediate_for_odds = calculate_historical_intermediate(df_hist_raw)
                common_index = X_processed.index.union(y_processed.index)
                df_full_data_aligned_for_split = df_hist_intermediate_for_odds.loc[common_index].copy()
                logger.info(f"DEBUG Main: df_full_data_aligned_for_split criado com shape {df_full_data_aligned_for_split.shape}")
                odd_draw_col_name = CONFIG_ODDS_COLS.get('draw', 'Odd_D_FT')
                if odd_draw_col_name not in df_full_data_aligned_for_split.columns:
                    raise ValueError(f"Coluna '{odd_draw_col_name}' não encontrada p/ ROI.")
            except Exception as e_align_main:
                raise ValueError("Falha alinhar odds.") from e_align_main

            def training_progress_callback(cs, ms, st):
                prog = 60 + int((cs / ms) * 35) if ms > 0 else 95
                self.gui_queue.put(("progress_update", (prog, st)))
            self.log("Iniciando treinamento...")
            self.gui_queue.put(("progress_update", (60, "Treinando Modelos...")))
            success = run_training_process(
                X=X_processed, y=y_processed,
                X_test_with_odds=df_full_data_aligned_for_split,  # Passa o DF completo alinhado
                odd_draw_col_name=CONFIG_ODDS_COLS.get('draw', 'Odd_D_FT'),
                progress_callback=training_progress_callback,
                calibration_method='isotonic',  # Pode vir do config ou ser fixo
                optimize_ev_threshold=optimize_ev,
                default_ev_threshold=DEFAULT_EV_THRESHOLD
            )
            self.gui_queue.put(("progress_update", (95, "Finalizando...")))
            if success:
                self.log("Treino OK.")
                self.gui_queue.put(("training_succeeded", None))
                training_successful = True
            else:
                raise RuntimeError("Falha treino/salvamento.")
        except Exception as e:
            error_msg = f"Erro Treino: {e}"
            self.log(f"ERRO: {error_msg}")
            traceback.logger.error_exc()
            self.gui_queue.put(("error", ("Erro Treino", error_msg)))
            self.gui_queue.put(("training_failed", None))
        finally:
            self.gui_queue.put(("progress_end", None))
            self.set_button_state(self.load_train_button, tk.NORMAL)

    # _run_prediction_pipeline ( para passar calibrador e usar limiar)
    def _run_prediction_pipeline(self, odd_draw_col: str):
        prediction_successful = False
        df_predictions_final_display = None
        try:
            self.gui_queue.put(("progress_start", (100,)))
            self.gui_queue.put(("progress_update", (10, f"Buscando CSV...")))
            fixture_df = fetch_and_process_fixtures()
            if fixture_df is None:
                raise ValueError("Falha buscar CSV.")
            if fixture_df.empty:
                self.log("Nenhum jogo CSV.")
                self.gui_queue.put(("prediction_complete", None))
                return

            self.gui_queue.put(("progress_update", (40, f"Preparando features...")))
            if not self.feature_columns or self.historical_data is None:
                raise ValueError("Features/Histórico ausentes.")
            X_fixtures_prepared = prepare_fixture_data(fixture_df, self.historical_data, self.feature_columns)
            if X_fixtures_prepared is None:
                raise ValueError("Falha preparar features.")
            if X_fixtures_prepared.empty:
                self.log("Nenhum jogo p/ prever.")
                self.gui_queue.put(("prediction_complete", None))
                return

            # --- CHAMADA make_predictions passa odd_draw_col ---
            df_preds_with_ev = predictor.make_predictions(
                model=self.trained_model, 
                scaler=self.trained_scaler, calibrator=self.trained_calibrator,
                feature_names=self.feature_columns, X_fixture_prepared=X_fixtures_prepared,
                fixture_info=fixture_df.loc[X_fixtures_prepared.index],
                odd_draw_col_name=odd_draw_col # Passa nome da coluna
            )
            if df_preds_with_ev is None: raise RuntimeError("Falha gerar previsões com EV.");
            self.log(f"Previsões com EV geradas: {len(df_preds_with_ev)}.");

            df_predictions_final_to_display = df_preds_with_ev

            # --- FILTRO AGORA É POR EV > optimal_ev_threshold ---
            df_to_filter = df_preds_with_ev.copy()
            ev_col_name = 'EV_Empate'  # Coluna criada pelo predictor
            self.log(f"Aplicando filtro EV > {self.optimal_ev_threshold:.3f}...")

            if ev_col_name in df_to_filter.columns:
                # Garante que EV é numérico
                df_to_filter[ev_col_name] = pd.to_numeric(df_to_filter[ev_col_name], errors='coerce')
                df_to_filter.dropna(subset=[ev_col_name], inplace=True)  # Remove onde EV não pôde ser calculado

                initial_rows_f = len(df_to_filter)
                # FILTRA por EV usando o limiar da instância
                df_filtered_f = df_to_filter[df_to_filter[ev_col_name] > self.optimal_ev_threshold].copy()
                rows_kept_f = len(df_filtered_f)
                self.log(f"Filtro EV: {rows_kept_f} de {initial_rows_f} jogos restantes passaram (EV > {self.optimal_ev_threshold:.3f}).")
                df_predictions_final_filtered = df_filtered_f
            else:
                self.log(f"Aviso: Coluna '{ev_col_name}' não encontrada para filtro EV.")
                df_predictions_final_filtered = pd.DataFrame()

            df_preds_with_calib = predictor.make_predictions(
                model=self.trained_model,
                scaler=self.trained_scaler,
                calibrator=self.trained_calibrator,  # <<< PASSA O CALIBRADOR
                feature_names=self.feature_columns,
                X_fixture_prepared=X_fixtures_prepared,
                fixture_info=fixture_df.loc[X_fixtures_prepared.index]
            )

            if df_preds_with_calib is None:
                raise RuntimeError("Falha gerar previsões.")
            self.log(f"Previsões (com probs calibradas se possível) geradas: {len(df_preds_with_calib)}.")

            # --- Filtro AGORA USA o optimal_threshold da classe ---
            df_to_filter = df_preds_with_calib.copy()
            self.log(f"Aplicando filtro de Limiar ({self.optimal_threshold:.3f})...")

            # Usa a coluna de probabilidade CALIBRADA se existir, senão a bruta
            prob_draw_col_calib = f'Prob_{CLASS_NAMES[1]}'  # Nome da coluna calibrada (sem Raw_)
            if prob_draw_col_calib not in df_to_filter.columns:
                # Tenta usar a bruta como fallback se a calibrada não foi gerada
                prob_draw_col_raw = f'ProbRaw_{CLASS_NAMES[1]}'
                if prob_draw_col_raw in df_to_filter.columns:
                    prob_col_to_use = prob_draw_col_raw
                    self.log(f"Aviso: Usando probs brutas ({prob_col_to_use}) para filtro (calibração falhou?).")
                else:
                    self.log(f"Erro: Nenhuma coluna de probabilidade ({prob_draw_col_calib} ou Raw) encontrada.")
                    # Decide o que fazer: pular filtro ou dar erro? Vamos pular por enquanto.
                    prob_col_to_use = None
            else:
                prob_col_to_use = prob_draw_col_calib  # Usa a calibrada!

            df_predictions_final_filtered = df_to_filter  # Default se não puder filtrar
            if prob_col_to_use:
                # Garante que a coluna é numérica
                df_to_filter[prob_col_to_use] = pd.to_numeric(df_to_filter[prob_col_to_use], errors='coerce')
                df_to_filter.dropna(subset=[prob_col_to_use], inplace=True)

                initial_rows_f = len(df_to_filter)
                # FILTRA usando o limiar da instância (self.optimal_threshold)
                df_filtered_f = df_to_filter[df_to_filter[prob_col_to_use] > self.optimal_threshold].copy()
                rows_kept_f = len(df_filtered_f)
                self.log(f"Filtro Limiar: {rows_kept_f} de {initial_rows_f} jogos restantes passaram (P > {self.optimal_threshold:.3f}).")
                df_predictions_final_filtered = df_filtered_f
            else:
                self.log("Filtro de probabilidade não aplicado (coluna não encontrada).")

            # --- Fim do Filtro ---

            # Ordenação (opcional, pelo nome da coluna calibrada/usada)
            if (df_predictions_final_filtered is not None and not df_predictions_final_filtered.empty 
                    and prob_col_to_use and prob_col_to_use in df_predictions_final_filtered.columns):
                self.log(f"Ordenando por '{prob_col_to_use}' descendente...")
                try:
                    df_predictions_final_filtered = df_predictions_final_filtered.sort_values(
                        by=prob_col_to_use, ascending=False
                    ).reset_index(drop=True)
                except Exception as e_sort:
                    self.log(f"Aviso: Erro ordenar: {e_sort}")

            # Envia resultado para GUI
            self.gui_queue.put(("progress_update", (95, "Preparando Exibição...")))
            if df_predictions_final_to_display is not None and not df_predictions_final_to_display.empty:
                 self.log(f"Enviando {len(df_predictions_final_to_display)} previsões para exibição (sem filtro).") # Log atualizado
                 self.gui_queue.put(("prediction_complete", df_predictions_final_to_display)) # Passa o DF COMPLETO
            else:
                 self.log("Nenhuma previsão gerada ou DataFrame vazio.")
                 self.gui_queue.put(("prediction_complete", None))  # Passa None para limpar

        except Exception as e:
            error_msg = f"Erro Pipeline Previsão: {e}"
            self.log(f"ERRO: {error_msg}")
            traceback.logger.error_exc()
            self.gui_queue.put(("error", ("Erro Previsão", error_msg)))
            self.gui_queue.put(("prediction_complete", None))
        finally:
            self.gui_queue.put(("progress_end", None))
            self.set_button_state(self.load_train_button, tk.NORMAL)
            if self.trained_model and self.selected_model_id:
                self.set_button_state(self.predict_button, tk.NORMAL)


    # --- Carregamento Inicial de Modelos  ---
    def load_existing_model_assets(self):
        self.log("--- Carregando Assets Iniciais ---")
        self.loaded_models_data = {}
        self.available_model_ids = []
        default_selection = None
        model_paths_to_try = {
            MODEL_ID_F1: BEST_F1_MODEL_SAVE_PATH,
            MODEL_ID_ROI: BEST_ROI_MODEL_SAVE_PATH,
        }

        for model_id, model_path in model_paths_to_try.items():
            self.log(f"Tentando carregar: {model_id}...")
            # load_model_scaler_features AGORA retorna mais itens
            load_result = predictor.load_model_scaler_features(model_path) # Função atualizada

            if load_result:
                # --- DESEMPACOTA ITENS ---
                model, scaler, calibrator, ev_threshold, features, params, metrics, timestamp = load_result
                if model and features:
                    self.log(f" -> Sucesso: Modelo '{model_id}' carregado (Limiar EV={ev_threshold:.3f}).")
                    # --- ARMAZENA CALIBRADOR E LIMIAR ---
                    self.loaded_models_data[model_id] = {
                        'model': model, 
                        'scaler': scaler, 
                        'calibrator': calibrator, # Armazena calibrador
                        'optimal_ev_threshold': ev_threshold, # Armazena limiar
                        'features': features, 
                        'params': params, 
                        'metrics': metrics,
                        'timestamp': timestamp, 
                        'path': model_path
                    }
                    self.available_model_ids.append(model_id)
                    if default_selection is None: default_selection = model_id
                else: self.log(f" -> Aviso: Arquivo '{model_id}' inválido (sem modelo/features).")

        # Atualiza Combobox e seleciona default
        try:
            if hasattr(self, 'model_selector_combo') and self.model_selector_combo.winfo_exists():
                self.model_selector_combo.config(values=self.available_model_ids)
                if self.available_model_ids:
                    final_selection = default_selection if default_selection in self.available_model_ids else self.available_model_ids[0]
                    self.selected_model_var.set(final_selection)
                    self.on_model_select() # Dispara atualização da GUI com dados do modelo selecionado
                    self.log(f"Modelos disponíveis: {self.available_model_ids}. Selecionado: {final_selection}")
                else:
                    self.selected_model_var.set(""); self.on_model_select(); self.log("Nenhum modelo válido encontrado.")
        except tk.TclError: pass

        # Carrega histórico (se necessário)
        if self.historical_data is None:
            self.log("Carregando dados históricos..."); 
            df_hist = load_historical_data()
            if df_hist is not None: 
                self.historical_data = df_hist; 
                self.log("Histórico carregado.")
            else: self.log("Falha carregar histórico.")

        # Habilita/Desabilita botão de prever
        if self.selected_model_id and self.historical_data is not None:
            self.set_button_state(self.predict_button, tk.NORMAL); self.log("Pronto para previsão.")
        else: self.set_button_state(self.predict_button, tk.DISABLED)


    # --- Processamento da Fila GUI  ---
    def process_gui_queue(self):
        try:
            while True:
                try:
                    message = self.gui_queue.get_nowait()
                    msg_type, msg_payload = message
                except Empty:
                    break
                except (ValueError, TypeError):
                    logger.error(f"AVISO GUI: Erro unpack msg: {message}")
                    continue
                except Exception as e_get:
                    logger.error(f"Erro get fila GUI: {e_get}")
                    continue

                # --- Trata tipos de mensagem ---
                try:
                    if msg_type == "log":
                        self._update_log_area(str(msg_payload))
                    elif msg_type == "button_state":
                        self._update_button_state(msg_payload)
                    elif msg_type == "update_stats_gui":
                        self._update_model_stats_display_gui()
                    elif msg_type == "error":
                        messagebox.showerror(msg_payload[0], msg_payload[1], parent=self.parent)
                    elif msg_type == "info":
                        messagebox.showinfo(msg_payload[0], msg_payload[1], parent=self.parent)
                    elif msg_type == "progress_start":
                        if hasattr(self, 'progress_bar') and self.progress_bar.winfo_exists():
                            self.progress_bar.config(maximum=msg_payload[0] if msg_payload else 100, value=0)
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
                        self.load_existing_model_assets()  # Recarrega para pegar os novos modelos salvos
                    elif msg_type == "training_failed":
                        self.log("ERRO: Treino falhou.")
                        # Limpa estado (código como antes)
                        self.selected_model_id = None
                        self.trained_model = None
                        self.trained_scaler = None
                        self.trained_calibrator = None
                        self.optimal_threshold = 0.5
                        self.feature_columns = None
                        self.model_best_params = None
                        self.model_eval_metrics = None
                        self.model_file_timestamp = None
                        self.selected_model_var.set("")
                        try:
                            self.model_selector_combo.config(values=[])
                        except tk.TclError:
                            pass
                        self._update_model_stats_display_gui()
                        self.set_button_state(self.predict_button, tk.DISABLED)
                    elif msg_type == "prediction_complete":
                        df_preds = msg_payload  # Pode ser DataFrame ou None
                        self.log("Recebidas previsões completas.")
                        self._update_prediction_display(df_preds)  # Atualiza Treeview
                    else:
                        self.log(f"AVISO GUI: Msg desconhecida: {msg_type}")
                except tk.TclError:
                    pass  # Ignora erro se widget destruído
                except Exception as e_proc:
                    logger.error(f"Erro processar msg '{msg_type}': {e_proc}")
                    traceback.logger.error_exc()

        except Exception as e_loop:
            logger.error(f"Erro CRÍTICO loop fila GUI: {e_loop}")
            traceback.logger.error_exc()
        finally:
            # Reagenda
            try:
                if self.main_tk_root and self.main_tk_root.winfo_exists():
                    self.main_tk_root.after(100, self.process_gui_queue)
            except Exception as e_resched:
                logger.error(f"Erro reagendar fila: {e_resched}")