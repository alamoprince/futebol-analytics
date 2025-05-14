# --- src/main.py ---
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
# Removidos imports não usados mais nesta classe: Listbox, MULTIPLE, Scrollbar, io
from tkinter.scrolledtext import ScrolledText
import sys, os, threading, datetime, math, numpy as np # Removido io
from queue import Queue, Empty
import pandas as pd
from typing import Optional, Dict, List, Any # Garante typing


try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
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
        CLASS_NAMES, FIXTURE_FETCH_DAY,MODEL_CONFIG,
        MODEL_TYPE_NAME, ODDS_COLS as CONFIG_ODDS_COLS,
        BEST_F1_MODEL_SAVE_PATH, BEST_ROI_MODEL_SAVE_PATH, MODEL_ID_F1, MODEL_ID_ROI,
        DEFAULT_F1_THRESHOLD, DEFAULT_EV_THRESHOLD, CALIBRATION_METHOD_DEFAULT )
    
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

class FootballPredictorDashboard:
    def __init__(self, parent_frame, main_root):
        self.parent = parent_frame
        self.main_tk_root = main_root

        self.gui_queue = Queue()
        self.stop_processing_queue = False
        self.historical_data: Optional[pd.DataFrame] = None
        self.loaded_models_data: Dict[str, Dict] = {}
        self.available_model_ids: List[str] = []
        self.selected_model_id: Optional[str] = None
        # --- Novos atributos para Calibrador e Limiar ---
        self.trained_model: Optional[Any] = None
        self.trained_scaler: Optional[Any] = None
        self.trained_calibrator: Optional[Any] = None # NOVO
        self.optimal_f1_threshold=DEFAULT_F1_THRESHOLD; # NOVO (default)
        # --- Fim Novos Atributos ---
        self.feature_columns: Optional[List[str]] = None
        self.model_best_params: Optional[Dict] = None
        self.model_eval_metrics: Optional[Dict] = None
        self.model_file_timestamp: Optional[str] = None
        self.optimal_f1_threshold: float = DEFAULT_F1_THRESHOLD # Usa padrão do config
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
        """Atualiza display de stats focando nos diferentes limiares e métricas."""
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
                timestamp = model_data.get('timestamp') # Timestamp do arquivo .joblib
                path = model_data.get('path')
                # Obtém o nome da classe do modelo salvo
                model_obj = model_data.get('model') # Pega o objeto do modelo
                model_class_name = model_obj.__class__.__name__ if model_obj else metrics.get('model_name', 'N/A') # Usa o nome salvo se objeto não carregou

                # Pega os limiares ótimos (com defaults)
                optimal_f1_thr = metrics.get('optimal_f1_threshold', DEFAULT_F1_THRESHOLD)
                optimal_ev_thr = metrics.get('optimal_ev_threshold', DEFAULT_EV_THRESHOLD)
                # Usa get() para o limiar de precision, pois pode não existir em modelos antigos
                optimal_prec_thr = metrics.get('optimal_precision_threshold', 0.5) # Default 0.5 se não encontrado

                calibrator_loaded = model_data.get('calibrator') is not None

                # --- Seção 1: Informações Gerais ---
                stats_content = f"Modelo Selecionado: {self.selected_model_id}\n"
                stats_content += f"  Tipo: {model_class_name}\n"
                # Se for ensemble, mostra estimadores base
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

                # Métricas @ Limiar F1 Otimizado
                f1_f1 = metrics.get('f1_score_draw') # Chave principal F1
                p_f1 = metrics.get('precision_draw_thrF1') # Precision no limiar F1
                r_f1 = metrics.get('recall_draw_thrF1')    # Recall no limiar F1
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
                logloss = metrics.get('log_loss') # Geralmente calculado com probs brutas
                stats_content += "- Métricas Probabilísticas:\n"
                stats_content += f"    ROC AUC (Pós-Calib): {auc:.4f}\n" if auc is not None else "    ROC AUC=N/A\n"
                stats_content += f"    Brier Score (Pós-Calib): {brier:.4f}\n" if brier is not None else "    Brier=N/A\n"
                stats_content += f"    Log Loss (Bruto): {logloss:.4f}\n" if logloss is not None else "    Log Loss=N/A\n"

                stats_content += "---\n"

                # --- Seção 4: Estratégia de Aposta EV (Teste) ---
                profit_ev = metrics.get('profit')
                roi_ev = metrics.get('roi')
                n_bets_ev = metrics.get('num_bets') # Número de apostas sugeridas pelo limiar EV

                stats_content += f"Estratégia EV (EV > {optimal_ev_thr:.3f} no Teste):\n"
                stats_content += f"- Nº Apostas Sugeridas: {n_bets_ev if n_bets_ev is not None else 'N/A'}\n"
                profit_ev_str = f"{profit_ev:+.2f} u" if profit_ev is not None else "N/A" # Adiciona sinal +/-
                roi_ev_str = "N/A"
                # Trata NaN/Inf/None para ROI de forma segura
                if isinstance(roi_ev, (int, float, np.number)) and pd.notna(roi_ev) and np.isfinite(roi_ev):
                    roi_ev_str = f"{roi_ev:+.2f} %" # Adiciona sinal +/-

                stats_content += f"- Lucro/Prejuízo: {profit_ev_str}\n"
                stats_content += f"- ROI Calculado: {roi_ev_str}\n"

            # Atualiza o widget
            self.model_stats_text.insert('1.0', stats_content)
            self.model_stats_text.config(state='disabled')
        except tk.TclError:
            pass # Ignora erro se widget for destruído enquanto atualiza
        except Exception as e:
            logger.error(f"Erro _update_model_stats_display_gui: {e}", exc_info=True)
            try: # Tenta exibir erro no widget
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
            # Adiciona log para ver colunas recebidas
            if df is not None and not df.empty:
                 logger.debug(f"GUI Display: Colunas recebidas no DF: {list(df.columns)}")
                 logger.debug(f"GUI Display: Amostra Time_Str recebida:\n{df['Time_Str'].head() if 'Time_Str' in df.columns else 'Coluna Time_Str AUSENTE'}")
                 logger.debug(f"GUI Display: Amostra Home recebida:\n{df['Home'].head() if 'Home' in df.columns else 'Coluna Home AUSENTE'}")
                 logger.debug(f"GUI Display: Amostra Away recebida:\n{df['Away'].head() if 'Away' in df.columns else 'Coluna Away AUSENTE'}")

            self.log(f"--- GUI: Atualizando display previsões...")
            # Cabeçalhos que você quer na GUI
            display_headers = ['Data', 'Hora', 'Liga', 'Casa', 'Fora',
                               'Odd D', 'P(E) Raw', 'P(E) Calib', 'EV Empate']

            # Tag de destaque
            try:
                self.prediction_tree.tag_configure('highlight_suggestion', background='lightgreen', foreground='black') # Renomear a tag talvez? (Opcional)
            except tk.TclError: pass

            # Limpa e configura Treeview
            if df is None or df.empty:
                self.log("GUI: DF vazio/None."); self._setup_prediction_columns(['Status']); self.prediction_tree.insert('', tk.END, values=['Nenhuma previsão gerada ou falha.']); return
            self.log(f"GUI: DF {df.shape}. Reconfigurando colunas: {display_headers}");
            self._setup_prediction_columns(display_headers)

            # Mapeamento Header GUI -> Coluna Interna DF
            odds_d_col = CONFIG_ODDS_COLS.get('draw', 'Odd_D_FT');
            prob_draw_raw_col = f'ProbRaw_{CLASS_NAMES[1]}' if CLASS_NAMES and len(CLASS_NAMES)>1 else 'ProbRaw_Empate'
            prob_draw_calib_col = f'Prob_{CLASS_NAMES[1]}' if CLASS_NAMES and len(CLASS_NAMES)>1 else 'Prob_Empate'
            ev_col = 'EV_Empate';
            header_to_col_map = {
                'Data': 'Date_Str', 'Hora': 'Time_Str', 'Liga': 'League',
                'Casa': 'Home', 'Fora': 'Away', 'Odd D': odds_d_col,
                'P(E) Raw': prob_draw_raw_col, 'P(E) Calib': prob_draw_calib_col,
                'EV Empate': ev_col
            }

            # Verifica quais colunas mapeadas realmente existem no DF recebido
            valid_internal_cols_map = {h: c for h, c in header_to_col_map.items() if c in df.columns}
            missing_display_cols = [h for h, c in header_to_col_map.items() if c not in df.columns]
            if missing_display_cols: logger.warning(f"GUI Display: Colunas ausentes no DF para cabeçalhos: {missing_display_cols}")


            # Formatação (opcional, pode ser feita aqui ou antes)
            df_display = df.copy() # Trabalha numa cópia para formatação
            # ... (formatação de probs, EV, odds como antes, usando os nomes internos corretos) ...
            prob_cols_to_format = [prob_draw_raw_col, prob_draw_calib_col] # Usa nomes internos
            for pcol in prob_cols_to_format:
                if pcol in df_display.columns:
                     try:
                          numeric_probs = pd.to_numeric(df_display[pcol], errors='coerce'); formatted_probs = (numeric_probs * 100).round(1).astype(str) + '%'; df_display[pcol] = formatted_probs.replace('nan%', '-', regex=False)
                     except Exception: df_display[pcol] = "-"
            if ev_col in df_display.columns:
                 try: df_display[ev_col] = pd.to_numeric(df_display[ev_col], errors='coerce').apply(lambda x: f"{x:+.3f}" if pd.notna(x) else "-")
                 except Exception: df_display[ev_col] = "-"
            if odds_d_col in df_display.columns:
                 try: df_display[odds_d_col] = pd.to_numeric(df_display[odds_d_col], errors='coerce').apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
                 except Exception: df_display[odds_d_col] = "-"


            from config import MIN_PROB_THRESHOLD_FOR_HIGHLIGHT # Importa o novo limiar

            prob_col_to_check = None
            calibrator_exists = self.trained_calibrator is not None
            if calibrator_exists and prob_draw_calib_col in df.columns:
                prob_col_to_check = prob_draw_calib_col
                prob_source_log = "Calibrada"
            elif prob_draw_raw_col in df.columns:
                prob_col_to_check = prob_draw_raw_col # Fallback para bruta
                prob_source_log = "Bruta"
                self.log(f"Aviso GUI: Usando Prob Bruta '{prob_col_to_check}' para verificação de limiar mínimo (calibrada ausente).")
            else:
                prob_source_log = "NENHUMA"
                self.log(f"ERRO GUI: Nenhuma coluna de probabilidade ({prob_draw_calib_col} ou {prob_draw_raw_col}) encontrada para verificação de limiar.")

            # Inserir Linhas
            highlight_criteria_log = f"EV > {self.optimal_ev_threshold:.3f} E Prob ({prob_source_log}) >= {MIN_PROB_THRESHOLD_FOR_HIGHLIGHT:.1%}"
            self.log(f"GUI: Adicionando linhas... (Highlight: {highlight_criteria_log})")
            added_rows = 0
            highlighted_count = 0

            for index, row_original in df.iterrows(): # Itera no DF original para pegar valores numéricos
                values_list = []
                for header in display_headers:
                    internal_col = header_to_col_map.get(header)
                    if internal_col and internal_col in df_display.columns:
                        formatted_value = df_display.loc[index, internal_col]
                        values_list.append(str(formatted_value))
                    else:
                        values_list.append("-") # Melhor que vazio

                # --- Lógica de DESTAQUE MODIFICADA ---
                tag_to_apply = ()
                should_highlight = False # Começa como Falso
                try:
                    # 1. Verifica condição de EV
                    ev_val_orig = pd.to_numeric(row_original.get(ev_col), errors='coerce')
                    ev_condition_met = pd.notna(ev_val_orig) and ev_val_orig > self.optimal_ev_threshold

                    # 2. Verifica condição de Probabilidade Mínima (se coluna existe)
                    prob_condition_met = False
                    if prob_col_to_check: # Só verifica se encontramos uma coluna de prob válida
                        prob_val_orig = pd.to_numeric(row_original.get(prob_col_to_check), errors='coerce')
                        prob_condition_met = pd.notna(prob_val_orig) and prob_val_orig >= MIN_PROB_THRESHOLD_FOR_HIGHLIGHT

                    # 3. Combina as condições
                    if ev_condition_met and prob_condition_met:
                        should_highlight = True

                except Exception as e_highlight:
                     logger.warning(f"GUI Highlight Check Error (Index {index}): {e_highlight}")
                     should_highlight = False # Segurança: não destaca se houver erro

                if should_highlight:
                    tag_to_apply = ('highlight_suggestion',) # Aplica a tag
                    highlighted_count += 1

                # Insere na Treeview
                try:
                    self.prediction_tree.insert('', tk.END, values=values_list, tags=tag_to_apply)
                    added_rows += 1
                except Exception as e_ins:
                    self.log(f"!! Erro inserir linha {index}: {e_ins}")

            self.log(f"GUI: {added_rows}/{len(df)} linhas adicionadas. {highlighted_count} destacadas.")

        except Exception as e_disp:
             logger.error(f"Erro GERAL em _update_prediction_display: {e_disp}", exc_info=True)
             self._setup_prediction_columns(['Status']); self.prediction_tree.insert('', tk.END, values=[f'Erro ao exibir: {e_disp}']);

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
            self.feature_columns = model_data.get('features')
            self.model_best_params = model_data.get('params')
            self.model_eval_metrics = model_data.get('metrics')
            self.model_file_timestamp = model_data.get('timestamp')
            self.optimal_ev_threshold = model_data.get('optimal_ev_threshold', DEFAULT_EV_THRESHOLD)
            self.optimal_f1_threshold = model_data.get('optimal_f1_threshold', DEFAULT_F1_THRESHOLD)

            self._update_model_stats_display_gui() # Mostra stats (incluindo limiar/calib)

            # Habilita prever se tudo OK
            if self.trained_model and self.feature_columns and self.historical_data is not None:
                self.set_button_state(self.predict_button, tk.NORMAL)
                self.log(f"Modelo '{selected_id}' pronto para previsão (Limiar={self.optimal_f1_threshold:.3f}).")
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
            self.trained_model = None; 
            self.trained_scaler = None; 
            self.trained_calibrator = None;
            self.optimal_ev_threshold=DEFAULT_EV_THRESHOLD; 
            self.optimal_f1_threshold=DEFAULT_F1_THRESHOLD; 
            self.feature_columns = None; 
            self.model_best_params = None;
            self.model_eval_metrics = None; 
            self.model_file_timestamp = None;
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
        self.optimal_f1_threshold=DEFAULT_F1_THRESHOLD; 
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
            kwargs={'odd_draw_col': CONFIG_ODDS_COLS.get('draw', 'Odd_D_FT')},
            daemon=True
        )
        predict_thread.start()


    def _run_training_pipeline(self, df_hist_raw: pd.DataFrame, optimize_ev: bool = True, optimize_f1: bool = True): # Adicionado optimize_f1
        """Pipeline de treinamento, incluindo pré-processamento, treinamento e salvamento."""
        training_successful = False
        total_progress_units = 1000 # Valor inicial, será ajustado
        try:
            # Calcula num_models ANTES de qualquer coisa que precise dele
            available_models = {name: config for name, config in MODEL_CONFIG.items() if not (name=='LGBMClassifier' and not LGBM_AVAILABLE)}
            num_models = len(available_models)
            if num_models == 0: raise ValueError("Nenhum modelo configurado.")
            total_progress_units = num_models * 100 # Define o total correto

            # Inicia a barra de progresso com o MÁXIMO correto
            self.gui_queue.put(("progress_start", (total_progress_units,)))
            self.gui_queue.put(("progress_update", (int(total_progress_units*0.05 / num_models), "Pré-processando..."))) # Progresso inicial pequeno

            processed_data = preprocess_and_feature_engineer(df_hist_raw)
            if processed_data is None: raise ValueError("Falha pré-processamento.")
            X_processed, y_processed, features_used = processed_data
            self.log(f"Pré-proc OK. Feats: {features_used}. Shape X: {X_processed.shape}")
            self.gui_queue.put(("progress_update", (int(total_progress_units*0.15 / num_models), "Alinhando odds..."))) # Progresso relativo

            df_full_data_aligned_for_split = None
            try:
                df_hist_intermediate = calculate_historical_intermediate(df_hist_raw)
                common_index = X_processed.index.intersection(df_hist_intermediate.index)
                if len(common_index) < len(X_processed): logger.warning(f"Alinhamento Odds: Perdendo {len(X_processed)-len(common_index)} linhas.")
                X_processed=X_processed.loc[common_index]; y_processed=y_processed.loc[common_index]
                df_full_data_aligned_for_split = df_hist_intermediate.loc[common_index].copy()
                logger.info(f"DEBUG Main: df_full_data_aligned_for_split {df_full_data_aligned_for_split.shape}")
                odd_draw_col_name = CONFIG_ODDS_COLS.get('draw', 'Odd_D_FT')
                if odd_draw_col_name not in df_full_data_aligned_for_split.columns: raise ValueError(f"Coluna Odd Empate '{odd_draw_col_name}' não encontrada.")
            except Exception as e_align: raise ValueError("Falha alinhar odds.") from e_align


            # --- Define o NOVO callback ---
            def training_progress_callback_stages(model_index, status_text):
                base_progress = model_index * 100
                stage_progress = 0; status_lower = status_text.lower()
                if "scaling" in status_lower: stage_progress = 5
                elif "ajustando" in status_lower or "fitting" in status_lower: stage_progress = 10
                elif "calibrando" in status_lower: stage_progress = 60
                elif "otimizando f1" in status_lower: stage_progress = 70
                elif "otimizando ev" in status_lower: stage_progress = 80
                elif "avaliando" in status_lower: stage_progress = 90
                elif "adicionado" in status_lower: stage_progress = 99
                current_total_progress = min(base_progress + stage_progress, total_progress_units)
                self.gui_queue.put(("progress_update", (current_total_progress, status_text)))

            self.log("Iniciando treinamento...")
            self.gui_queue.put(("progress_update", (int(total_progress_units*0.20 / num_models), "Iniciando Treinamento..."))) 

            success = run_training_process(
                X=X_processed, y=y_processed,
                X_test_with_odds=df_full_data_aligned_for_split,
                odd_draw_col_name=odd_draw_col_name, 
                progress_callback_stages=training_progress_callback_stages, 
                num_total_models_expected=num_models,                               
                calibration_method= CALIBRATION_METHOD_DEFAULT,
                optimize_ev_threshold_flag=optimize_ev,
                optimize_f1_threshold_flag=optimize_f1, 
            )

            status_text = "Treino Concluído!" if success else "Treino Falhou."
            self.gui_queue.put(("progress_update", (total_progress_units, status_text))) 
            if success:
                self.log("Treino OK."); self.gui_queue.put(("training_succeeded", None)); training_successful = True
            else:
                self.log("ERRO: Falha treino/salvamento."); self.gui_queue.put(("error", ("Falha Treinamento", "Erro. Ver logs."))); self.gui_queue.put(("training_failed", None))

        except Exception as e:
            error_msg = f"Erro Treino (Preparação/Execução): {e}"
            self.log(f"ERRO: {error_msg}")
            logger.error(f"Erro no pipeline de treino: {e}", exc_info=True) # Logger correto
            self.gui_queue.put(("error", ("Erro Treino", error_msg)))
            self.gui_queue.put(("training_failed", None))
        finally:
            self.gui_queue.put(("progress_end", None))
            self.set_button_state(self.load_train_button, tk.NORMAL)
            if not training_successful:
                self.set_button_state(self.predict_button, tk.DISABLED)

    # _run_prediction_pipeline ( para passar calibrador e usar limiar)
    def _run_prediction_pipeline(self, odd_draw_col: str):
        try:
            self.gui_queue.put(("progress_start", (100,)))
            self.gui_queue.put(("progress_update", (10, f"Buscando CSV de jogos futuros...")))
            fixture_df = fetch_and_process_fixtures()

            if fixture_df is None: # Erro na busca
                self.log("ERRO: Falha ao buscar CSV de jogos futuros.")
                self.gui_queue.put(("error", ("Erro de Rede", "Não foi possível buscar os jogos.")))
                self.gui_queue.put(("prediction_complete", pd.DataFrame())) # Envia DF vazio para limpar treeview
                return # Sai da função
            if fixture_df.empty:
                self.log("Nenhum jogo encontrado no CSV para o dia alvo.")
                self.gui_queue.put(("prediction_complete", pd.DataFrame()))
                return

            self.gui_queue.put(("progress_update", (30, f"Preparando features para {len(fixture_df)} jogos...")))
            if not self.feature_columns or self.historical_data is None:
                self.log("ERRO: Features do modelo ou dados históricos ausentes para preparação.")
                raise ValueError("Features do modelo ou dados históricos não carregados.") # Levanta erro para ser pego pelo except geral

            X_fixtures_prepared = prepare_fixture_data(
                fixture_df,
                self.historical_data,
                self.feature_columns 
            )

            if X_fixtures_prepared is None: 
                self.log("ERRO: Falha ao preparar features para os jogos futuros.")
                raise ValueError("Falha ao preparar features dos jogos.")
            if X_fixtures_prepared.empty:
                self.log("Nenhum jogo restante para prever após preparação de features.")
                self.gui_queue.put(("prediction_complete", pd.DataFrame()))
                return

            self.gui_queue.put(("progress_update", (60, f"Realizando previsões para {len(X_fixtures_prepared)} jogos...")))

            df_predictions_all_info = predictor.make_predictions(
                model=self.trained_model,
                scaler=self.trained_scaler,
                calibrator=self.trained_calibrator, 
                feature_names=self.feature_columns,
                X_fixture_prepared=X_fixtures_prepared,
                fixture_info=fixture_df.loc[X_fixtures_prepared.index], 
                odd_draw_col_name=odd_draw_col
            )

            if df_predictions_all_info is None or df_predictions_all_info.empty:
                self.log("Falha ao gerar previsões ou nenhuma previsão retornada de predictor.make_predictions.")
                self.gui_queue.put(("prediction_complete", pd.DataFrame()))
                return

            self.log(f"Previsões (com probs Raw/Calib e EV) geradas para {len(df_predictions_all_info)} jogos.")

            # --- Opcional: Filtragem antes da exibição ---
            # Se você quiser filtrar os jogos ANTES de mostrá-los na GUI
            # (em vez de apenas destacá-los), você pode aplicar filtros aqui.
            # Por exemplo, para mostrar apenas jogos com EV > X e Prob > Y:

            df_for_display_and_highlight = df_predictions_all_info.copy()
            # Você pode adicionar uma coluna de "Sinal" aqui se quiser
            # Ex: df_for_display_and_highlight['Sinal_Aposta_EV'] = (df_for_display_and_highlight['EV_Empate'] > self.optimal_ev_threshold)
            # Ex: df_for_display_and_highlight['Sinal_Aposta_Prob'] = (df_for_display_and_highlight[prob_col_para_usar] > self.optimal_f1_threshold)

            # --- Lógica de Ordenação (Aplicada ao DataFrame que será exibido) ---
            prob_draw_calib_col_display = f'Prob_{CLASS_NAMES[1]}' if CLASS_NAMES and len(CLASS_NAMES)>1 else 'Prob_Empate'
            df_sorted_for_display = df_for_display_and_highlight.copy()

            # Tenta ordenar pela probabilidade calibrada de empate (se existir e tiver valores)
            if prob_draw_calib_col_display in df_sorted_for_display.columns and \
               df_sorted_for_display[prob_draw_calib_col_display].notna().any():
                self.log(f"Ordenando previsões por '{prob_draw_calib_col_display}' (calibrada) descendente...")
                try:
                    df_sorted_for_display[prob_draw_calib_col_display] = pd.to_numeric(df_sorted_for_display[prob_draw_calib_col_display], errors='coerce')
                    df_sorted_for_display = df_sorted_for_display.sort_values(by=prob_draw_calib_col_display, ascending=False, na_position='last').reset_index(drop=True)
                except Exception as e_sort:
                    self.log(f"Aviso: Erro ao ordenar por prob calibrada: {e_sort}")
            else:
                prob_draw_raw_col_display = f'ProbRaw_{CLASS_NAMES[1]}' if CLASS_NAMES and len(CLASS_NAMES)>1 else 'ProbRaw_Empate'
                if prob_draw_raw_col_display in df_sorted_for_display.columns and \
                   df_sorted_for_display[prob_draw_raw_col_display].notna().any():
                    self.log(f"Aviso: Prob calibrada ausente/NaN. Ordenando por prob bruta '{prob_draw_raw_col_display}'...")
                    try:
                        df_sorted_for_display[prob_draw_raw_col_display] = pd.to_numeric(df_sorted_for_display[prob_draw_raw_col_display], errors='coerce')
                        df_sorted_for_display = df_sorted_for_display.sort_values(by=prob_draw_raw_col_display, ascending=False, na_position='last').reset_index(drop=True)
                    except Exception as e_sort_raw:
                        self.log(f"Aviso: Erro ao ordenar por prob bruta: {e_sort_raw}")
                else:
                    self.log(f"Aviso: Nenhuma coluna de probabilidade (calibrada ou bruta) válida para ordenação.")

            # --- Envio para GUI ---
            self.gui_queue.put(("progress_update", (95, "Preparando Exibição...")))
            if not df_sorted_for_display.empty:
                self.log(f"Enviando {len(df_sorted_for_display)} previsões para exibição.")
                self.gui_queue.put(("prediction_complete", df_sorted_for_display))
                prediction_successful = True 
            else:
                self.log("Nenhuma previsão para exibir após processamento/ordenação.")
                self.gui_queue.put(("prediction_complete", pd.DataFrame()))

        except ValueError as ve: 
            error_msg = f"Erro de Valor no Pipeline de Previsão: {ve}"
            self.log(f"ERRO: {error_msg}")
            logger.error(f"Erro de Valor no pipeline de previsão: {ve}", exc_info=False) 
            self.gui_queue.put(("error", ("Erro de Dados", error_msg)))
            self.gui_queue.put(("prediction_complete", pd.DataFrame()))
        except Exception as e: 
            error_msg = f"Erro Inesperado no Pipeline de Previsão: {e}"
            self.log(f"ERRO CRÍTICO: {error_msg}")
            logger.error(f"Erro completo no pipeline de previsão: {e}", exc_info=True)
            self.gui_queue.put(("error", ("Erro Crítico na Previsão", error_msg)))
            self.gui_queue.put(("prediction_complete", pd.DataFrame())) 
        finally:
            self.gui_queue.put(("progress_end", None))
            self.set_button_state(self.load_train_button, tk.NORMAL) 
            if self.selected_model_id and self.trained_model:
                self.set_button_state(self.predict_button, tk.NORMAL)
            else:
                self.set_button_state(self.predict_button, tk.DISABLED)


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
            load_result = predictor.load_model_scaler_features(model_path) # Função atualizada

            if load_result:
                # --- DESEMPACOTA ITENS ---
                model, scaler, calibrator, ev_threshold, f1_thr, features, params, metrics, timestamp = load_result
                if model and features:
                    self.log(f" -> Sucesso: Modelo '{model_id}' carregado (Limiar EV={ev_threshold:.3f}).")
                    # --- ARMAZENA CALIBRADOR E LIMIAR ---
                    self.loaded_models_data[model_id] = {
                        'model': model, 
                        'scaler': scaler, 
                        'calibrator': calibrator, 
                        'optimal_ev_threshold': ev_threshold, 
                        'optimal_f1_threshold': f1_thr, 
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
                    self.on_model_select() 
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
        # <<< PASSO 1: VERIFICA FLAG NO INÍCIO >>>
        if self.stop_processing_queue:
            logger.debug("PredictorDashboard: Parando fila GUI.")
            return # Não processa e não reagenda

        try:
            while True:
                try:
                    message = self.gui_queue.get_nowait()
                    msg_type, msg_payload = message
                except Empty:
                    break # Sai do loop while se a fila estiver vazia
                except (ValueError, TypeError):
                    logger.error(f"AVISO GUI (Predictor): Erro unpack msg: {message}")
                    continue
                except Exception as e_get:
                    logger.error(f"Erro get fila GUI (Predictor): {e_get}")
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
                        # Garante que messagebox tem um pai válido
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
                        self.set_button_state(self.predict_button, tk.DISABLED) # Garante que está desabilitado
                        self.set_button_state(self.load_train_button, tk.NORMAL) # Reabilita botão de treino
                    elif msg_type == "prediction_complete":
                        df_preds = msg_payload
                        self.log("Recebidas previsões completas para exibição.")
                        self._update_prediction_display(df_preds)
                        self.set_button_state(self.load_train_button, tk.NORMAL) # Reabilita botões pós-previsão
                        if self.selected_model_id: self.set_button_state(self.predict_button, tk.NORMAL)
                    else:
                        self.log(f"AVISO GUI (Predictor): Msg desconhecida: {msg_type}")
                except tk.TclError:
                    pass # Ignora erro se widget destruído
                except Exception as e_proc:
                    # Usa logger importado
                    logger.error(f"Erro processar msg (Predictor) '{msg_type}': {e_proc}", exc_info=True)
                    # Não usa traceback.logger.error_exc() a menos que logger seja o objeto traceback

        except Exception as e_loop:
            logger.error(f"Erro CRÍTICO loop fila GUI (Predictor): {e_loop}", exc_info=True)
        finally:
            # <<< PASSO 2: REAGENDA SÓ SE NÃO FOR PARAR >>>
            if not self.stop_processing_queue:
                try:
                     # Verifica se a janela principal ainda existe
                    if hasattr(self.main_tk_root, 'winfo_exists') and self.main_tk_root.winfo_exists():
                        self.main_tk_root.after(100, self.process_gui_queue)
                except Exception as e_resched:
                      # Evita logar erro se já está parando
                    if not self.stop_processing_queue:
                        logger.error(f"Erro reagendar fila GUI (Predictor): {e_resched}")