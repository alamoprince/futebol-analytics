# --- src/main.py ---
# VERSÃO CORRIGIDA - Focada APENAS na Aba de Treino e Previsão

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
# Removidos imports não usados mais nesta classe: Listbox, MULTIPLE, Scrollbar, io
from tkinter.scrolledtext import ScrolledText
import sys, os, threading, datetime, math, numpy as np # Removido io
from queue import Queue, Empty
import pandas as pd
from typing import Optional, Dict, List, Any
import logging
# Adiciona diretórios e importa módulos
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
        HISTORICAL_DATA_PATH, CLASS_NAMES, ODDS_COLS, FIXTURE_FETCH_DAY,
        MODEL_TYPE_NAME, TEST_SIZE, RANDOM_STATE, ODDS_COLS as CONFIG_ODDS_COLS,
        BEST_F1_MODEL_SAVE_PATH, BEST_ROI_MODEL_SAVE_PATH, MODEL_ID_F1, MODEL_ID_ROI,
        FEATURE_COLUMNS # FEATURE_COLUMNS é usado no prepare_fixture_data e pipeline de treino
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
    logging.error(f"Erro import main.py (Treino/Previsão): {e}")
    raise # Re-levanta erro para app_launcher
except Exception as e_i:
    print(f"Erro geral import main.py (Treino/Previsão): {e_i}")
    raise # Re-levanta erro


# --- src/main.py ---
# ATUALIZADO para Calibrador e Limiar

# ... (imports como na versão anterior corrigida) ...
# ... (Imports necessários como tk, ttk, os, threading, queue, pandas, etc.) ...
# ... (Importa de config, data_handler, model_trainer, predictor) ...
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
        main_frame = ttk.Frame(self.parent, padding="10"); main_frame.pack(fill=tk.BOTH, expand=True)
        left_panel = ttk.Frame(main_frame); left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), pady=5)
        control_frame = ttk.LabelFrame(left_panel, text=" Ações ", padding="10"); control_frame.pack(pady=(0, 5), fill=tk.X)
        self.load_train_button = ttk.Button(control_frame, text="TREINAR e Salvar Melhores Modelos", command=self.start_training_thread, width=35); self.load_train_button.pack(pady=5, fill=tk.X)
        predict_frame = ttk.Frame(control_frame); predict_frame.pack(fill=tk.X, pady=5)
        self.predict_button = ttk.Button(predict_frame, text=f"Prever ({FIXTURE_FETCH_DAY.capitalize()}) com:", command=self.start_prediction_thread, width=18); self.predict_button.pack(side=tk.LEFT, fill=tk.X, expand=False); self.predict_button.config(state=tk.DISABLED)
        self.selected_model_var = tk.StringVar(); self.model_selector_combo = ttk.Combobox(predict_frame, textvariable=self.selected_model_var, state="readonly", width=20); self.model_selector_combo.pack(side=tk.RIGHT, padx=(5, 0), fill=tk.X, expand=True); self.model_selector_combo.bind("<<ComboboxSelected>>", self.on_model_select)
        progress_frame = ttk.Frame(control_frame); progress_frame.pack(fill=tk.X, pady=(10, 0)); self.progress_label = ttk.Label(progress_frame, text="Pronto."); self.progress_label.pack(side=tk.LEFT, padx=(0, 5)); self.progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=200, mode='determinate'); self.progress_bar.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        stats_frame = ttk.LabelFrame(left_panel, text=" Status Modelo Selecionado ", padding="10"); stats_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        self.model_stats_text = ScrolledText(stats_frame, height=15, state='disabled', wrap=tk.WORD, font=("Consolas", 9), relief=tk.FLAT, bd=0); self.model_stats_text.pack(fill=tk.BOTH, expand=True)
        # --- MODIFICADO: Setup inicial da Treeview ---
        right_panel = ttk.Frame(main_frame); right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        results_frame = ttk.LabelFrame(right_panel, text=" Previsões ", padding="5"); results_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        # Adiciona 'Prob Calib' e talvez 'Aposta?'
        cols = ['Data', 'Hora', 'Liga', 'Casa', 'Fora', 'Odd H', 'Odd D', 'Odd A',
                'P(Ñ Emp)', 'P(Empate)', 'P(Emp Calib)'] # <-- Nova coluna Prob Calibrada
        self.prediction_tree = ttk.Treeview(results_frame, columns=cols, show='headings', height=10);
        vsb = ttk.Scrollbar(results_frame, orient="vertical", command=self.prediction_tree.yview); hsb = ttk.Scrollbar(results_frame, orient="horizontal", command=self.prediction_tree.xview); self.prediction_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set); vsb.pack(side='right', fill='y'); hsb.pack(side='bottom', fill='x'); self.prediction_tree.pack(fill=tk.BOTH, expand=True)
        self._setup_prediction_columns(cols) # Chama setup inicial com novas colunas
        # --- FIM MODIFICAÇÃO Treeview Setup ---
        log_frame = ttk.LabelFrame(right_panel, text=" Logs ", padding="5"); log_frame.pack(fill=tk.BOTH, expand=True, side=tk.BOTTOM, pady=(5, 0))
        self.log_area = ScrolledText(log_frame, height=8, state='disabled', wrap=tk.WORD, font=("Consolas", 9)); self.log_area.pack(fill=tk.BOTH, expand=True)
        self._update_model_stats_display_gui() # Atualiza display inicial


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

    # --- MODIFICADO: Atualiza Display de Stats do Modelo ---
    def _update_model_stats_display_gui(self):
        try:
            if not hasattr(self, 'model_stats_text') or not self.model_stats_text.winfo_exists(): return
            self.model_stats_text.config(state='normal')
            self.model_stats_text.delete('1.0', tk.END)

            if self.trained_model is None or self.selected_model_id is None:
                stats_content = "Nenhum modelo selecionado ou carregado."
            else:
                model_data = self.loaded_models_data.get(self.selected_model_id, {})
                metrics = model_data.get('metrics', {})
                params = model_data.get('params')
                features = model_data.get('features', [])
                timestamp = self.model_file_timestamp
                path = model_data.get('path')
                model_class_name = self.trained_model.__class__.__name__ if self.trained_model else "N/A"
                # --- Pega Limiar e Calibrador ---
                optimal_th = model_data.get('optimal_threshold', 'N/A') # Pega do dict carregado
                calibrator_loaded = model_data.get('calibrator') is not None
                # -------------------------------

                stats_content = f"Modelo: {self.selected_model_id}\n"
                stats_content += f"Arquivo: {os.path.basename(path or 'N/A')}\n"
                stats_content += f"Modif.: {timestamp or 'N/A'}\n"
                stats_content += f"Tipo: {model_class_name}\n"
                stats_content += f"Calibrado: {'Sim' if calibrator_loaded else 'Não'}\n" # Mostra se tem calibrador
                stats_content += f"Limiar Ótimo (Val): {optimal_th:.4f}\n" if isinstance(optimal_th, (int, float)) else f"Limiar Ótimo (Val): {optimal_th}\n" # Mostra limiar
                stats_content += "---\n"

                # ... (Restante do display de Features e Params como antes) ...
                if features: stats_content += f"Features ({len(features)}):\n - " + "\n - ".join(features) + "\n---\n"
                else: stats_content += "Features: N/A\n---\n"
                if params: stats_content += "Params:\n"; params_list = [f" - {k}: {v}" for k,v in params.items()]; stats_content += "\n".join(params_list) + "\n---\n"
                else: stats_content += "Params: N/A\n---\n"

                # --- Métricas (AGORA DO CONJUNTO DE TESTE) ---
                stats_content += "Métricas (Teste):\n"
                acc = metrics.get('accuracy')
                f1_d = metrics.get('f1_score_draw') # F1 binário (limiar 0.5)
                brier = metrics.get('brier_score')  # Brier (probs calibradas)
                auc = metrics.get('roc_auc')        # AUC (probs calibradas)
                roi_test = metrics.get('roi')       # ROI (limiar ótimo)
                n_bets_test = metrics.get('num_bets') # Bets (limiar ótimo)
                profit_test = metrics.get('profit') # Profit (limiar ótimo)
                test_n = metrics.get('test_set_size', 'N/A')

                stats_content += f"- Acurácia: {acc:.4f}\n" if acc is not None else "- Acc: N/A\n"
                stats_content += f"- F1 Empate (Thr 0.5): {f1_d:.4f}\n" if f1_d is not None else "- F1 (Thr 0.5): N/A\n"
                stats_content += f"- ROC AUC (Calib): {auc:.4f}\n" if auc is not None else "- ROC AUC: N/A\n"
                stats_content += f"- Brier Score (Calib): {brier:.4f}\n" if brier is not None else "- Brier: N/A\n"
                stats_content += "---\n"
                stats_content += f"Estratégia (Limiar Ótimo={optimal_th:.3f}):\n" if isinstance(optimal_th, (int, float)) else "Estratégia (Limiar Padrão):\n"
                stats_content += f"- Nº Bets (Teste): {n_bets_test if n_bets_test is not None else 'N/A'}\n"
                stats_content += f"- Profit (Teste): {profit_test:.2f} u\n" if profit_test is not None else "- Profit: N/A\n"
                # Formatação segura do ROI do teste
                roi_test_str = "N/A"
                if roi_test is not None:
                     try:
                         if isinstance(roi_test, (float, np.number)) and not np.isnan(roi_test):
                             roi_test_str = f"{roi_test:.2f} %"
                         elif isinstance(roi_test, (int, float)): roi_test_str = f"{roi_test:.2f} %"
                     except TypeError: pass
                stats_content += f"- ROI (Teste): {roi_test_str}\n"
                stats_content += f"\nTamanho Teste: {test_n}"
                # -----------------------------------------

            self.model_stats_text.insert('1.0', stats_content)
            self.model_stats_text.config(state='disabled')
        except tk.TclError: pass
        except Exception as e: print(f"Erro _update_model_stats_display_gui: {e}"); traceback.print_exc()

    # --- MODIFICADO: Setup Colunas Treeview ---
    def _setup_prediction_columns(self, columns: List[str]):
        try:
            if not hasattr(self, 'prediction_tree') or not self.prediction_tree.winfo_exists(): return
            self.prediction_tree['columns'] = columns
            self.prediction_tree.delete(*self.prediction_tree.get_children())

            # Larguras ajustadas para nova coluna
            col_widths = {'Data': 75, 'Hora': 50, 'Liga': 140, 'Casa': 100, 'Fora': 100,
                           'Odd H': 50, 'Odd D': 50, 'Odd A': 50,
                           'P(Ñ Emp)': 70, 'P(Empate)': 70, 'P(Emp Calib)': 75, # Nova coluna
                           'Status': 500}

            for col in columns:
                self.prediction_tree.heading(col, text='')
                self.prediction_tree.column(col, width=0, minwidth=0, stretch=tk.NO) # Limpa

                width = col_widths.get(col, 80)
                anchor = tk.W if col in ['Liga', 'Casa', 'Fora', 'Data', 'Hora', 'Status'] else tk.CENTER
                stretch = tk.NO

                # Renomeia header da prob calibrada
                header_text = col
                if col == 'P(Emp Calib)': header_text = 'P(Emp) Calib.'

                self.prediction_tree.heading(col, text=header_text, anchor=anchor)
                self.prediction_tree.column(col, anchor=anchor, width=width, stretch=stretch)

            if columns == ['Status']: self.prediction_tree.column('Status', stretch=tk.YES)

        except tk.TclError: pass
        except Exception as e: print(f"Erro _setup_prediction_columns: {e}")

    # --- MODIFICADO: Atualiza Display de Previsões ---
    def _update_prediction_display(self, df: Optional[pd.DataFrame]):
        try:
            if not hasattr(self, 'prediction_tree') or not self.prediction_tree.winfo_exists(): return
        except tk.TclError: return

        self.log(f"--- GUI: Atualizando display de previsões ---")
        # Define os headers que QUEREMOS exibir na ordem desejada
        display_headers = ['Data', 'Hora', 'Liga', 'Casa', 'Fora', 'Odd H', 'Odd D', 'Odd A',
                            'P(Ñ Emp)', 'P(Empate)', 'P(Emp Calib)'] # Adiciona Calib

        if df is None or df.empty:
            self.log("GUI: DataFrame vazio/None. Exibindo status.")
            try:
                self._setup_prediction_columns(['Status'])
                self.prediction_tree.insert('', tk.END, values=['Nenhuma previsão válida.'])
            except Exception as e_clear: self.log(f"Erro clear/status treeview: {e_clear}")
            return

        self.log(f"GUI: Recebido DataFrame {df.shape}. Reconfigurando colunas...")
        self._setup_prediction_columns(display_headers) # Reconfigura com os headers certos

        # Mapeamento Header -> Coluna Interna no DataFrame recebido
        # O nome da coluna calibrada agora vem de predictor.py (sem Raw_)
        odds_h_col = CONFIG_ODDS_COLS.get('home', 'Odd_H_FT')
        odds_d_col = CONFIG_ODDS_COLS.get('draw', 'Odd_D_FT')
        odds_a_col = CONFIG_ODDS_COLS.get('away', 'Odd_A_FT')
        # Colunas de Probabilidade SEM Raw_ (assumindo que predictor.py renomeou/adicionou)
        prob_non_draw_col = f'Prob_{CLASS_NAMES[0]}' if CLASS_NAMES and len(CLASS_NAMES) > 0 else 'Prob_Nao_Empate'
        prob_draw_col = f'Prob_{CLASS_NAMES[1]}' if CLASS_NAMES and len(CLASS_NAMES) > 1 else 'Prob_Empate'

        header_to_col_map = {
            'Data': 'Date_Str', 'Hora': 'Time_Str', 'Liga': 'League', 'Casa': 'HomeTeam', 'Fora': 'AwayTeam',
            'Odd H': odds_h_col, 'Odd D': odds_d_col, 'Odd A': odds_a_col,
            # Removido O2.5 e BTTS S dos headers padrão, pode adicionar de volta se quiser
            'P(Ñ Emp)': prob_non_draw_col,
            'P(Empate)': prob_draw_col,
            'P(Emp Calib)': prob_draw_col # A coluna calibrada TERÁ O MESMO NOME da prob da classe positiva
        }

        # Verifica quais colunas do MAPA realmente existem no DF recebido
        valid_internal_cols_map = {h: c for h, c in header_to_col_map.items() if c in df.columns}
        if not valid_internal_cols_map:
             self.log("ERRO GUI: Nenhuma coluna mapeada encontrada no DF para Treeview!");
             # Código de erro na treeview...
             return

        try:
            self.log(f"GUI: Formatando colunas: {list(valid_internal_cols_map.values())}")
            df_display = df[list(valid_internal_cols_map.values())].copy() # Pega só as colunas válidas

            # Formata Probabilidades (Ñ Emp, Emp, Emp Calib)
            prob_cols_to_format = [prob_non_draw_col, prob_draw_col]
            for pcol in prob_cols_to_format:
                if pcol in df_display.columns:
                    try:
                        df_display[pcol] = (pd.to_numeric(df_display[pcol], errors='coerce') * 100).round(1).astype(str) + '%'
                        df_display[pcol] = df_display[pcol].replace('nan%', '-', regex=False)
                    except Exception as e_fmt: self.log(f"Aviso: Erro formatar {pcol}: {e_fmt}")

            # Formata Odds
            odds_cols_to_format = [c for c in valid_internal_cols_map.values() if str(c).startswith('Odd_')]
            for ocol in odds_cols_to_format:
                if ocol in df_display.columns:
                    try: df_display[ocol] = pd.to_numeric(df_display[ocol], errors='coerce').apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
                    except Exception as e_fmt: self.log(f"Aviso: Erro formatar {ocol}: {e_fmt}")

            # Insere linhas
            self.log("GUI: Adicionando linhas formatadas...")
            added_rows = 0
            for index, row in df_display.iterrows():
                values_to_insert = []
                for header in display_headers: # Itera na ordem dos headers da Treeview
                    internal_col = valid_internal_cols_map.get(header) # Pega nome interno do mapa válido
                    values_to_insert.append(str(row.get(internal_col, '')) if internal_col else '') # Pega valor formatado
                try:
                    self.prediction_tree.insert('', tk.END, values=values_to_insert)
                    added_rows += 1
                except Exception as e_ins: self.log(f"!! Erro inserir linha {index}: {e_ins} - Vals: {values_to_insert}")

            self.log(f"GUI: {added_rows}/{len(df_display)} linhas adicionadas.")

        except tk.TclError: self.log("Erro TclError display.")
        except Exception as e: self.log(f"!! Erro GERAL _update_prediction_display: {e}"); traceback.print_exc()


    # --- Callback Seleção Modelo (MODIFICADO) ---
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
            # ---------------------------------
            self.feature_columns = model_data.get('features')
            self.model_best_params = model_data.get('params')
            self.model_eval_metrics = model_data.get('metrics')
            self.model_file_timestamp = model_data.get('timestamp')

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
            self.optimal_threshold = 0.5; self.feature_columns = None; self.model_best_params = None;
            self.model_eval_metrics = None; self.model_file_timestamp = None;
            self.set_button_state(self.predict_button, tk.DISABLED)
            self._update_model_stats_display_gui() # Mostra 'nenhum modelo'

    # --- Funções de Ação (start_training_thread, start_prediction_thread) ---
    # (Sem alterações na lógica de iniciar a thread)
    def start_training_thread(self):
        # ... (código como antes para limpar estado, desabilitar botões, iniciar thread _run_training_pipeline) ...
        self.log("Iniciando processo de treino em background..."); self.loaded_models_data = {}; self.available_model_ids = []; self.selected_model_var.set('');
        try: self.model_selector_combo.config(values=[])
        except tk.TclError: pass
        self.selected_model_id = None; self.trained_model = None; self.trained_scaler = None; self.trained_calibrator = None; self.optimal_threshold = 0.5; self.feature_columns = None; self.model_best_params = None; self.model_eval_metrics = None; self.model_file_timestamp = None;
        self.gui_queue.put(("update_stats_gui", None)); self.set_button_state(self.load_train_button, tk.DISABLED); self.set_button_state(self.predict_button, tk.DISABLED); self.gui_queue.put(("progress_start", (100,))); self.gui_queue.put(("progress_update", (5, "Carregando Histórico...")));
        try:
            df_hist = load_historical_data(HISTORICAL_DATA_PATH);
            if df_hist is None: raise ValueError("Falha carregar histórico.");
            self.historical_data = df_hist; self.log("Histórico carregado."); self.gui_queue.put(("progress_update", (20, "Iniciando Thread Treino...")));
            train_thread = threading.Thread(target=self._run_training_pipeline, args=(self.historical_data.copy(),), daemon=True); train_thread.start()
        except Exception as e_load: error_msg = f"Erro Carregar Histórico: {e_load}"; self.log(f"ERRO: {error_msg}"); self.gui_queue.put(("error", ("Erro Carregamento", error_msg))); self.gui_queue.put(("progress_end", None)); self.set_button_state(self.load_train_button, tk.NORMAL)


    def start_prediction_thread(self):
        # (código como antes, mas verifica se calibrador/limiar foram carregados se precisar deles aqui)
        if self.trained_model is None or self.selected_model_id is None: messagebox.showwarning("Modelo Não Selecionado", "Selecione um modelo treinado.", parent=self.parent); return
        if self.historical_data is None: messagebox.showwarning("Histórico Ausente", "Carregue/Treine.", parent=self.parent); return
        if not self.feature_columns: messagebox.showwarning("Features Ausentes", "Features do modelo não carregadas.", parent=self.parent); return
        # Adicional: Avisar se não houver calibrador/limiar? Ou deixar predictor lidar?
        if self.trained_calibrator is None: self.log("Aviso: Modelo selecionado não possui calibrador. Usando probs brutas.")
        self.log(f"Iniciando previsão com '{self.selected_model_id}' (Limiar={self.optimal_threshold:.3f})...") # Usa o limiar carregado
        self.set_button_state(self.load_train_button, tk.DISABLED); self.set_button_state(self.predict_button, tk.DISABLED)
        try: self._setup_prediction_columns(['Status']); self.prediction_tree.insert('', tk.END, values=['Buscando jogos...'])
        except Exception as e: self.log(f"Erro limpar treeview: {e}")
        predict_thread = threading.Thread(target=self._run_prediction_pipeline, daemon=True); predict_thread.start()

    # --- Funções das Threads (_run_training_pipeline, _run_prediction_pipeline) ---
    # (_run_training_pipeline como antes, já passa df_hist_aligned para X_test_with_odds)
    def _run_training_pipeline(self, df_hist_raw: pd.DataFrame):
        # ... (código como antes, que chama preprocess_and_feature_engineer, alinha df_full_data_aligned_for_split,
        #      e chama run_training_process passando df_full_data_aligned_for_split para X_test_with_odds) ...
        training_successful = False
        try:
            self.gui_queue.put(("progress_update", (30, "Pré-processando...")))
            processed_data = preprocess_and_feature_engineer(df_hist_raw)
            if processed_data is None: raise ValueError("Falha pré-processamento.")
            X_processed, y_processed, features_used = processed_data
            self.log(f"Pré-processamento OK. Features: {features_used}")

            self.gui_queue.put(("progress_update", (50, "Alinhando dados com odds...")))
            df_full_data_aligned_for_split = None # Reset
            try:
                df_hist_intermediate_for_odds = calculate_historical_intermediate(df_hist_raw)
                common_index = X_processed.index.union(y_processed.index)
                df_full_data_aligned_for_split = df_hist_intermediate_for_odds.loc[common_index].copy()
                logging.info(f"DEBUG Main: df_full_data_aligned_for_split criado com shape {df_full_data_aligned_for_split.shape}")
                odd_draw_col_name = CONFIG_ODDS_COLS.get('draw', 'Odd_D_FT')
                if odd_draw_col_name not in df_full_data_aligned_for_split.columns:
                    raise ValueError(f"Coluna '{odd_draw_col_name}' não encontrada p/ ROI.")
            except Exception as e_align_main: raise ValueError("Falha alinhar odds.") from e_align_main

            def training_progress_callback(cs, ms, st): prog = 60 + int((cs / ms) * 35) if ms > 0 else 95; self.gui_queue.put(("progress_update", (prog, st)))
            self.log("Iniciando treinamento..."); self.gui_queue.put(("progress_update", (60, "Treinando Modelos...")))
            success = run_training_process(
                X=X_processed, y=y_processed,
                X_test_with_odds=df_full_data_aligned_for_split, # Passa o DF completo alinhado
                odd_draw_col_name=CONFIG_ODDS_COLS.get('draw', 'Odd_D_FT'),
                progress_callback=training_progress_callback,
                calibration_method='isotonic' # Pode vir do config ou ser fixo
            )
            self.gui_queue.put(("progress_update", (95, "Finalizando...")))
            if success: self.log("Treino OK."); self.gui_queue.put(("training_succeeded", None)); training_successful = True
            else: raise RuntimeError("Falha treino/salvamento.")
        except Exception as e: error_msg = f"Erro Treino: {e}"; self.log(f"ERRO: {error_msg}"); traceback.print_exc(); self.gui_queue.put(("error", ("Erro Treino", error_msg))); self.gui_queue.put(("training_failed", None))
        finally: self.gui_queue.put(("progress_end", None)); self.set_button_state(self.load_train_button, tk.NORMAL)


    # _run_prediction_pipeline (MODIFICADO para passar calibrador e usar limiar)
    def _run_prediction_pipeline(self):
        prediction_successful = False
        df_predictions_final_display = None
        try:
            self.gui_queue.put(("progress_start", (100,)))
            self.gui_queue.put(("progress_update", (10, f"Buscando CSV...")))
            fixture_df = fetch_and_process_fixtures()
            if fixture_df is None: raise ValueError("Falha buscar CSV.")
            if fixture_df.empty: self.log("Nenhum jogo CSV."); self.gui_queue.put(("prediction_complete", None)); return

            self.gui_queue.put(("progress_update", (40, f"Preparando features...")))
            if not self.feature_columns or self.historical_data is None: raise ValueError("Features/Histórico ausentes.")
            X_fixtures_prepared = prepare_fixture_data(fixture_df, self.historical_data, self.feature_columns)
            if X_fixtures_prepared is None: raise ValueError("Falha preparar features.")
            if X_fixtures_prepared.empty: self.log("Nenhum jogo p/ prever."); self.gui_queue.put(("prediction_complete", None)); return

            self.gui_queue.put(("progress_update", (70, "Prevendo...")))
            # --- Passa o calibrador para make_predictions ---
            df_preds_with_calib = predictor.make_predictions(
                model=self.trained_model,
                scaler=self.trained_scaler,
                calibrator=self.trained_calibrator, # <<< PASSA O CALIBRADOR
                feature_names=self.feature_columns,
                X_fixture_prepared=X_fixtures_prepared,
                fixture_info=fixture_df.loc[X_fixtures_prepared.index]
            )
            # ----------------------------------------------
            if df_preds_with_calib is None: raise RuntimeError("Falha gerar previsões.")
            self.log(f"Previsões (com probs calibradas se possível) geradas: {len(df_preds_with_calib)}.")

            # --- Filtro AGORA USA o optimal_threshold da classe ---
            df_to_filter = df_preds_with_calib.copy()
            self.log(f"Aplicando filtro de Limiar ({self.optimal_threshold:.3f})...")

            # Usa a coluna de probabilidade CALIBRADA se existir, senão a bruta
            prob_draw_col_calib = f'Prob_{CLASS_NAMES[1]}' # Nome da coluna calibrada (sem Raw_)
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
                 prob_col_to_use = prob_draw_col_calib # Usa a calibrada!


            df_predictions_final_filtered = df_to_filter # Default se não puder filtrar
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
            if df_predictions_final_filtered is not None and not df_predictions_final_filtered.empty and prob_col_to_use and prob_col_to_use in df_predictions_final_filtered.columns:
                self.log(f"Ordenando por '{prob_col_to_use}' descendente...")
                try: df_predictions_final_filtered = df_predictions_final_filtered.sort_values(by=prob_col_to_use, ascending=False).reset_index(drop=True)
                except Exception as e_sort: self.log(f"Aviso: Erro ordenar: {e_sort}")

            # Envia resultado para GUI
            self.gui_queue.put(("progress_update", (95, "Preparando Exibição...")))
            if df_predictions_final_filtered is not None and not df_predictions_final_filtered.empty:
                self.log(f"Enviando {len(df_predictions_final_filtered)} previsões finais p/ exibição.")
                self.gui_queue.put(("prediction_complete", df_predictions_final_filtered)) # Passa o DF filtrado
            else:
                self.log("Nenhuma previsão restante após filtro p/ exibir.")
                self.gui_queue.put(("prediction_complete", None)) # Passa None para limpar

        except Exception as e: # ... (Tratamento de erro e finally como antes) ...
             error_msg = f"Erro Pipeline Previsão: {e}"; self.log(f"ERRO: {error_msg}"); traceback.print_exc(); self.gui_queue.put(("error", ("Erro Previsão", error_msg))); self.gui_queue.put(("prediction_complete", None))
        finally: self.gui_queue.put(("progress_end", None)); self.set_button_state(self.load_train_button, tk.NORMAL);
        if self.trained_model and self.selected_model_id: self.set_button_state(self.predict_button, tk.NORMAL)


    # --- Carregamento Inicial de Modelos (MODIFICADO) ---
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
                # --- DESEMPACOTA 8 ITENS ---
                model, scaler, calibrator, threshold, features, params, metrics, timestamp = load_result
                # -------------------------
                if model and features:
                    self.log(f" -> Sucesso: Modelo '{model_id}' carregado (Limiar={threshold:.3f}).")
                    # --- ARMAZENA CALIBRADOR E LIMIAR ---
                    self.loaded_models_data[model_id] = {
                        'model': model, 'scaler': scaler, 'calibrator': calibrator, # Armazena calibrador
                        'optimal_threshold': threshold, # Armazena limiar
                        'features': features, 'params': params, 'metrics': metrics,
                        'timestamp': timestamp, 'path': model_path
                    }
                    # -----------------------------------
                    self.available_model_ids.append(model_id)
                    if default_selection is None: default_selection = model_id
                else: self.log(f" -> Aviso: Arquivo '{model_id}' inválido (sem modelo/features).")
            # else: log de erro já feito por predictor.py

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
            self.log("Carregando dados históricos..."); df_hist = load_historical_data(HISTORICAL_DATA_PATH)
            if df_hist is not None: self.historical_data = df_hist; self.log("Histórico carregado.")
            else: self.log("Falha carregar histórico.")

        # Habilita/Desabilita botão de prever
        if self.selected_model_id and self.historical_data is not None:
            self.set_button_state(self.predict_button, tk.NORMAL); self.log("Pronto para previsão.")
        else: self.set_button_state(self.predict_button, tk.DISABLED)


    # --- Processamento da Fila GUI (MODIFICADO) ---
    def process_gui_queue(self):
        try:
            while True:
                try: message = self.gui_queue.get_nowait(); msg_type, msg_payload = message
                except Empty: break
                except (ValueError, TypeError): print(f"AVISO GUI: Erro unpack msg: {message}"); continue
                except Exception as e_get: print(f"Erro get fila GUI: {e_get}"); continue

                # --- Trata tipos de mensagem ---
                try:
                    if msg_type == "log": self._update_log_area(str(msg_payload))
                    elif msg_type == "button_state": self._update_button_state(msg_payload)
                    elif msg_type == "update_stats_gui": self._update_model_stats_display_gui()
                    elif msg_type == "error": messagebox.showerror(msg_payload[0], msg_payload[1], parent=self.parent)
                    elif msg_type == "info": messagebox.showinfo(msg_payload[0], msg_payload[1], parent=self.parent)
                    elif msg_type == "progress_start":
                         if hasattr(self, 'progress_bar') and self.progress_bar.winfo_exists(): self.progress_bar.config(maximum=msg_payload[0] if msg_payload else 100, value=0)
                         if hasattr(self, 'progress_label') and self.progress_label.winfo_exists(): self.progress_label.config(text="Iniciando...")
                    elif msg_type == "progress_update":
                         if isinstance(msg_payload, tuple) and len(msg_payload) == 2:
                            value, status_text = msg_payload
                            if hasattr(self, 'progress_bar') and self.progress_bar.winfo_exists(): self.progress_bar['value'] = value
                            if hasattr(self, 'progress_label') and self.progress_label.winfo_exists(): self.progress_label.config(text=str(status_text))
                    elif msg_type == "progress_end":
                         if hasattr(self, 'progress_bar') and self.progress_bar.winfo_exists(): self.progress_bar['value'] = 0
                         if hasattr(self, 'progress_label') and self.progress_label.winfo_exists(): self.progress_label.config(text="Pronto.")
                    elif msg_type == "training_succeeded":
                        self.log("Treino OK. Recarregando modelos...")
                        self.load_existing_model_assets() # Recarrega para pegar os novos modelos salvos
                    elif msg_type == "training_failed":
                         self.log("ERRO: Treino falhou.")
                         # Limpa estado (código como antes)
                         self.selected_model_id=None; self.trained_model=None; self.trained_scaler=None; self.trained_calibrator=None; self.optimal_threshold=0.5; self.feature_columns=None; self.model_best_params=None; self.model_eval_metrics=None; self.model_file_timestamp=None; self.selected_model_var.set("");
                         try: self.model_selector_combo.config(values=[]);
                         except tk.TclError: pass; self._update_model_stats_display_gui(); self.set_button_state(self.predict_button, tk.DISABLED)
                    elif msg_type == "prediction_complete":
                         df_preds = msg_payload # Pode ser DataFrame ou None
                         self.log("Recebidas previsões completas.")
                         self._update_prediction_display(df_preds) # Atualiza Treeview
                    # Removidos tipos de msg da aba de análise
                    else: self.log(f"AVISO GUI: Msg desconhecida: {msg_type}")
                except tk.TclError: pass # Ignora erro se widget destruído
                except Exception as e_proc: print(f"Erro processar msg '{msg_type}': {e_proc}"); traceback.print_exc()

        except Exception as e_loop: print(f"Erro CRÍTICO loop fila GUI: {e_loop}"); traceback.print_exc()
        finally:
            # Reagenda
            try:
                if self.main_tk_root and self.main_tk_root.winfo_exists(): self.main_tk_root.after(100, self.process_gui_queue)
            except Exception as e_resched: print(f"Erro reagendar fila: {e_resched}")