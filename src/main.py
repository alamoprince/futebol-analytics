# --- src/main.py ---
# ... (imports como antes) ...
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText
import sys, os, threading, datetime, math, numpy as np
from queue import Queue, Empty
import pandas as pd
from typing import Optional, Dict, List, Any
from config import ( HISTORICAL_DATA_PATH, CLASS_NAMES, ODDS_COLS, OTHER_ODDS_NAMES, FIXTURE_FETCH_DAY, MODEL_TYPE_NAME, TEST_SIZE, RANDOM_STATE, ODDS_COLS as CONFIG_ODDS_COLS, BEST_F1_MODEL_SAVE_PATH, BEST_ROI_MODEL_SAVE_PATH, MODEL_ID_F1, MODEL_ID_ROI ) # Importa IDs e Paths
from data_handler import ( load_historical_data, preprocess_and_feature_engineer, fetch_and_process_fixtures, prepare_fixture_data, calculate_historical_intermediate )
from model_trainer import train_evaluate_and_save_best_models as run_training_process # Renomeado
import predictor
import requests
from sklearn.model_selection import train_test_split

# ... (Paths como antes) ...
SRC_DIR = os.path.dirname(os.path.abspath(__file__)); BASE_DIR = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path: sys.path.append(SRC_DIR)
if BASE_DIR not in sys.path: sys.path.append(BASE_DIR)


class FootballPredictorDashboard:
    # ... (__init__, create_widgets  - com Combobox) ...
    def __init__(self, root): # ... (init ) ...
        self.root = root; 
        self.root.title(f"Football Predictor Pro ({MODEL_TYPE_NAME})")
        self.root.geometry("950x750"); 
        self.root.minsize(800, 600); 
        self.gui_queue = Queue(); 
        self.historical_data = None
        self.loaded_models_data: Dict[str, Dict] = {}; 
        self.available_model_ids: List[str] = []
        self.selected_model_id: Optional[str] = None; 
        self.trained_model: Optional[Any] = None; 
        self.trained_scaler: Optional[Any] = None; 
        self.feature_columns: Optional[List[str]] = None; 
        self.model_best_params: Optional[Dict] = None; 
        self.model_eval_metrics: Optional[Dict] = None; 
        self.model_file_timestamp: Optional[str] = None
        self.create_widgets(); self.root.after(100, self.process_gui_queue); 
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.log(f"Bem-vindo ({MODEL_TYPE_NAME})!"); 
        self.log(f"Fonte CSV: '{FIXTURE_FETCH_DAY}'."); 
        self.log(f"Carregando modelos e histórico..."); 
        self.load_existing_model_assets()


    def _run_prediction_pipeline(self):
        """Pipeline completo: busca CSV, prepara features, prevê E FILTRA."""
        prediction_successful = False
        df_predictions_final = None # Variável para guardar resultado filtrado
        try:
            self.gui_queue.put(("progress_start", (100))); self.gui_queue.put(("progress_update", (10, f"Buscando CSV ({FIXTURE_FETCH_DAY})...")));
            fixture_df = fetch_and_process_fixtures() # Contém info básica + odds brutas mapeadas
            if fixture_df is None: raise ValueError("Falha buscar/processar CSV.");
            if fixture_df.empty: self.log("Nenhum jogo no CSV."); self.gui_queue.put(("prediction_complete", None)); return

            self.gui_queue.put(("progress_update", (40, f"Preparando features ({len(fixture_df)})...")));
            # feature_columns deve ser o atributo da classe (as 12 features)
            if not self.feature_columns: raise ValueError("Lista de features do modelo não carregada.")
            X_fixtures_prepared = prepare_fixture_data(fixture_df, self.historical_data, self.feature_columns)
            if X_fixtures_prepared is None or X_fixtures_prepared.empty: raise ValueError("Falha preparar features.")

            self.gui_queue.put(("progress_update", (70, "Realizando previsões...")));
            # Passa scaler (pode ser None), feature_columns corretas
            df_predictions = predictor.make_predictions(self.trained_model, self.trained_scaler, self.feature_columns, X_fixtures_prepared, fixture_df)
            if df_predictions is None: raise RuntimeError("Falha gerar previsões.")
            self.log(f"Previsões brutas geradas: {len(df_predictions)} jogos.")

            # --- FILTRO 1: Remover jogos com odds de INPUT ausentes ---
            # (precisam existir no fixture_df que veio do CSV)
            input_odd_features = ['Odd_H_FT', 'Odd_D_FT', 'Odd_A_FT', 'Odd_Over25_FT', 'Odd_BTTS_Yes']
            cols_to_check_nan = [c for c in input_odd_features if c in df_predictions.columns]
            if cols_to_check_nan:
                initial_rows_f1 = len(df_predictions)
                df_predictions_f1 = df_predictions.dropna(subset=cols_to_check_nan)
                rows_dropped_f1 = initial_rows_f1 - len(df_predictions_f1)
                if rows_dropped_f1 > 0: self.log(f"Filtro 1: Removidos {rows_dropped_f1} jogos com odds de entrada ausentes.")
                df_predictions = df_predictions_f1 # Atualiza o dataframe
            else: self.log("Aviso: Nenhuma coluna de odd encontrada para filtro de NaN.")

            # --- FILTRO 2: Manter apenas jogos com P(Empate) < 87% ---
            prob_col_draw = f'Prob_{CLASS_NAMES[1]}' # Prob_Empate
            if prob_col_draw in df_predictions.columns:
                threshold = 0.87
                initial_rows_f2 = len(df_predictions)
                df_predictions_f2 = df_predictions[df_predictions[prob_col_draw] < threshold]
                rows_dropped_f2 = initial_rows_f2 - len(df_predictions_f2)
                if rows_dropped_f2 > 0: self.log(f"Filtro 2: Removidos {rows_dropped_f2} jogos com P(Empate) >= {threshold*100:.0f}%.")
                df_predictions_final = df_predictions_f2 # Guarda resultado final filtrado
            else:
                self.log(f"Aviso: Coluna '{prob_col_draw}' não encontrada para filtro de probabilidade.")
                df_predictions_final = df_predictions # Usa o resultado do filtro anterior se prob não existe

            if df_predictions_final is not None and not df_predictions_final.empty:
                 self.log(f"Total de {len(df_predictions_final)} previsões após filtros.")
                 prediction_successful = True
            else:
                 self.log("Nenhuma previsão restante após aplicar os filtros.")
                 prediction_successful = False # Considera falha se ficou vazio

            self.gui_queue.put(("progress_update", (95, "Finalizando...")));
            # Envia o DataFrame FINAL FILTRADO para a GUI
            self.gui_queue.put(("prediction_complete", df_predictions_final))

        except Exception as e: # ... (tratamento de erro como antes) ...
            error_msg = f"Erro Previsão Thread: {e}"; self.log(f"ERRO: {error_msg}")
            self.gui_queue.put(("error", ("Erro na Previsão", error_msg))); self.gui_queue.put(("prediction_complete", None))
        finally: # ... (reabilita botões como antes) ...
             self.gui_queue.put(("progress_end", None)); self.set_button_state(self.load_train_button, tk.NORMAL);
             if self.trained_model: self.set_button_state(self.predict_button, tk.NORMAL)
    def create_widgets(self): 

        style = ttk.Style()
        style.theme_use('clam')
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), pady=5, anchor='nw')

        control_frame = ttk.LabelFrame(left_panel, text=" Ações ", padding="10")
        control_frame.pack(pady=(0, 5), fill=tk.X)

        self.load_train_button = ttk.Button(
            control_frame, text="Carregar Histórico e Treinar",
            command=self.start_training_thread, width=30
        )
        self.load_train_button.pack(pady=5, fill=tk.X)

        predict_frame = ttk.Frame(control_frame)
        predict_frame.pack(fill=tk.X, pady=5)

        self.predict_button = ttk.Button(
            predict_frame, text=f"Prever ({FIXTURE_FETCH_DAY.capitalize()})",
            command=self.start_prediction_thread, width=18
        )
        self.predict_button.pack(side=tk.LEFT, fill=tk.X, expand=False)
        self.predict_button.config(state=tk.DISABLED)

        self.selected_model_var = tk.StringVar()
        self.model_selector_combo = ttk.Combobox(
            predict_frame, textvariable=self.selected_model_var,
            state="readonly", width=15
        )
        self.model_selector_combo.pack(side=tk.RIGHT, padx=(5, 0), fill=tk.X, expand=True)
        self.model_selector_combo.bind("<<ComboboxSelected>>", self.on_model_select)

        # Progress bar and label
        progress_frame = ttk.Frame(control_frame)
        progress_frame.pack(fill=tk.X, pady=(10, 0))

        self.progress_label = ttk.Label(progress_frame, text="Pronto.")
        self.progress_label.pack(side=tk.LEFT, padx=(0, 5))

        self.progress_bar = ttk.Progressbar(
            progress_frame, orient=tk.HORIZONTAL, length=200, mode='determinate'
        )
        self.progress_bar.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        stats_frame = ttk.LabelFrame(left_panel, text=" Status Modelo Selecionado ", padding="10")
        stats_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        self.model_stats_text = ScrolledText(
            stats_frame, height=15, state='disabled', wrap=tk.WORD,
            font=("Consolas", 9), relief=tk.FLAT, bd=0
        )
        self.model_stats_text.pack(fill=tk.BOTH, expand=True)
        self._update_model_stats_display_gui()

        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        results_frame = ttk.LabelFrame(right_panel, text=" Previsões ", padding="5")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        cols = ['Data', 'Hora', 'Liga', 'Casa', 'Fora', 'Odd H', 'Odd D', 'Odd A',
            'O2.5', 'BTTS S', 'P(Ñ Emp)', 'P(Empate)']
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

        # Logs frame
        log_frame = ttk.LabelFrame(right_panel, text=" Logs ", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True, side=tk.BOTTOM, pady=(5, 0))

        self.log_area = ScrolledText(
            log_frame, height=8, state='disabled', wrap=tk.WORD,
            font=("Consolas", 9)
        )
        self.log_area.pack(fill=tk.BOTH, expand=True)

    def _setup_prediction_columns(self, columns: List[str]): 
        """Configura as colunas da Treeview de previsões."""
        self.prediction_tree['columns'] = columns
        self.prediction_tree.delete(*self.prediction_tree.get_children())
        col_widths = {
            'Data': 75, 'Hora': 50, 'Liga': 150, 'Casa': 110, 'Fora': 110,
            'Odd H': 50, 'Odd D': 50, 'Odd A': 50, 'O2.5': 50, 'BTTS S': 55,
            'P(Ñ Emp)': 70, 'P(Empate)': 70
        }
        if columns == ['Status']:
            self.prediction_tree.heading('Status', text='Status')
            self.prediction_tree.column('Status', anchor=tk.W, width=500)
            return
        for col in columns:
            width = col_widths.get(col, 80)
            anchor = tk.CENTER if col not in ['Liga', 'Casa', 'Fora', 'Data', 'Hora'] else tk.W
            self.prediction_tree.heading(col, text=col)
            self.prediction_tree.column(col, anchor=anchor, width=width, stretch=False)

    def log(self, message: str):
        """Adiciona uma mensagem ao log."""
        self.gui_queue.put(("log", message))

    def _update_log_area(self, message: str): 
        """Atualiza a área de log com uma nova mensagem."""
        try:
            self.log_area.config(state='normal')
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            self.log_area.insert(tk.END, f"[{ts}] {message}\n")
            self.log_area.config(state='disabled')
            self.log_area.see(tk.END)
        except tk.TclError:
            pass

    def set_button_state(self, button: ttk.Button, state: str):
        """Atualiza o estado de um botão."""
        self.gui_queue.put(("button_state", (button, state)))

    def _update_button_state(self, button_state_tuple): 
        """Aplica o estado a um botão."""
        button, state = button_state_tuple
        try:
            button.config(state=state)
        except tk.TclError:
            pass

    def display_predictions(self, df_predictions: Optional[pd.DataFrame]):
        """Envia previsões para exibição na GUI."""
        self.gui_queue.put(("display_predictions", df_predictions))

    def _update_prediction_display(self, df: Optional[pd.DataFrame]): 
        """Atualiza a exibição de previsões na Treeview."""
        display_headers = [
            'Data', 'Hora', 'Liga', 'Casa', 'Fora', 'Odd H', 'Odd D', 'Odd A',
            'O2.5', 'BTTS S', 'P(Ñ Emp)', 'P(Empate)'
        ]
        self._setup_prediction_columns(display_headers)  # Define colunas corretas

        header_to_col_map = {
            'Data': 'Date_Str', 'Hora': 'Time_Str', 'Liga': 'League',
            'Casa': 'HomeTeam', 'Fora': 'AwayTeam',
            'Odd H': CONFIG_ODDS_COLS['home'], 'Odd D': CONFIG_ODDS_COLS['draw'],
            'Odd A': CONFIG_ODDS_COLS['away'], 'O2.5': 'Odd_Over25_FT',
            'BTTS S': 'Odd_BTTS_Yes', 'P(Ñ Emp)': f'Prob_{CLASS_NAMES[0]}',
            'P(Empate)': f'Prob_{CLASS_NAMES[1]}'
        }

        internal_cols_to_fetch = [
            header_to_col_map.get(h) for h in display_headers if header_to_col_map.get(h)
        ]
        valid_internal_cols = [
            c for c in internal_cols_to_fetch if df is not None and c in df.columns
        ]

        if df is None or df.empty or not valid_internal_cols:
            self._setup_prediction_columns(['Status'])
            self.prediction_tree.insert('', tk.END, values=['Nenhuma previsão válida.'])
            return

        try:
            df_display = df[valid_internal_cols].copy()
            prob_cols = [f'Prob_{CLASS_NAMES[0]}', f'Prob_{CLASS_NAMES[1]}']
            odds_cols_internal = list(CONFIG_ODDS_COLS.values()) + OTHER_ODDS_NAMES

            # Formatação de probabilidades
            for pcol in prob_cols:
                if pcol in df_display:
                    df_display[pcol] = (
                        pd.to_numeric(df_display[pcol], errors='coerce') * 100
                    ).round(1).astype(str) + '%'
                    df_display[pcol] = df_display[pcol].replace('nan%', '-', regex=False)

            # Formatação de odds
            for ocol in odds_cols_internal:
                if ocol in df_display:
                    df_display[ocol] = pd.to_numeric(df_display[ocol], errors='coerce').apply(
                        lambda x: f"{x:.2f}" if pd.notna(x) else "-"
                    )

            # Inserção de valores na Treeview
            for _, row in df_display.iterrows():
                values = [
                    str(row.get(header_to_col_map.get(h), '')) for h in display_headers
                ]
                self.prediction_tree.insert('', tk.END, values=values)

        except Exception as display_error:
            self.log(f"Erro exibir previsões: {display_error}")
            self._setup_prediction_columns(['Status'])
            self.prediction_tree.insert('', tk.END, values=[f'Erro: {display_error}'])

    # --- Callback e Atualização de Stats (CORRIGIDO) ---
    def on_model_select(self, event=None):
        selected_id = self.selected_model_var.get()
        self.log(f"Modelo selecionado via Combobox: {selected_id}")
        self._update_gui_for_selected_model(selected_id)

    def _update_gui_for_selected_model(self, selected_id: Optional[str]):
        """Atualiza estado interno e GUI com base no model_id selecionado."""
        if selected_id and selected_id in self.loaded_models_data:
            model_data = self.loaded_models_data[selected_id]
            self.log(f"Carregando dados internos para: {selected_id}")
            # Atualiza estado ATUAL com dados do modelo selecionado
            self.selected_model_id = selected_id
            self.trained_model = model_data.get('model')
            self.trained_scaler = model_data.get('scaler')
            self.feature_columns = model_data.get('features') # Chave é 'features'
            self.model_best_params = model_data.get('params') # Chave é 'params'
            self.model_eval_metrics = model_data.get('metrics') # Chave é 'metrics'
            self.model_file_timestamp = model_data.get('timestamp')

            # Verifica se o carregamento foi realmente bem-sucedido
            if self.trained_model and self.feature_columns:
                self._update_model_stats_display_gui() # Atualiza o texto das stats
                # Habilita previsão SOMENTE se histórico também estiver carregado
                if self.historical_data is not None:
                     self.set_button_state(self.predict_button, tk.NORMAL)
                else:
                     self.log("Modelo carregado, mas histórico ausente. Previsão desabilitada.")
                     self.set_button_state(self.predict_button, tk.DISABLED)
            else:
                 # Caso onde dados carregados estavam incompletos
                 self.log(f"Erro: Dados carregados para '{selected_id}' estão incompletos (sem modelo ou features).")
                 self.trained_model = None # Garante limpeza
                 self.set_button_state(self.predict_button, tk.DISABLED)
                 self._update_model_stats_display_gui() # Mostra msg de erro/vazio

        else:
             # Caso ID inválido ou nenhum selecionado
             self.log(f"ID de modelo inválido ou não selecionado: '{selected_id}'. Limpando estado.")
             self.selected_model_id = None; self.trained_model = None; self.trained_scaler = None; self.feature_columns = None
             self.model_best_params = None; self.model_eval_metrics = None; self.model_file_timestamp = None
             self.set_button_state(self.predict_button, tk.DISABLED)
             self._update_model_stats_display_gui() # Limpa display de stats

    def _update_model_stats_display_gui(self):
        """Atualiza área de texto com stats do modelo SELECIONADO."""
        try:
            self.model_stats_text.config(state='normal')
            self.model_stats_text.delete('1.0', tk.END)

            # Usa o estado ATUAL (self.trained_model, self.model_eval_metrics)
            if self.trained_model is None or self.selected_model_id is None:
                stats_content = "Nenhum modelo selecionado ou carregado."
            else:
                model_data = self.loaded_models_data.get(self.selected_model_id, {})
                stats_content = (
                    f"Modelo: {self.selected_model_id}\n"
                    f"Arquivo: {os.path.basename(model_data.get('path', 'N/A'))}\n"
                    f"Modif.: {self.model_file_timestamp or 'N/A'}\n"
                    f"Tipo: {self.trained_model.__class__.__name__}\n---\n"
                )

                if self.feature_columns:
                    stats_content += (
                        f"Features ({len(self.feature_columns)}):\n - "
                        + "\n - ".join(self.feature_columns)
                        + "\n---\n"
                    )

                if self.model_best_params:
                    stats_content += "Melhores Parâmetros:\n"
                    params_list = [f" - {k}: {v}" for k, v in self.model_best_params.items()]
                    stats_content += "\n".join(params_list) + "\n---\n"

                # --- VERIFICAÇÃO DAS MÉTRICAS ---
                metrics = self.model_eval_metrics
                if metrics:
                    stats_content += "Métricas Avaliação (Teste):\n"
                    stats_content += (
                        f"- Acurácia: {metrics.get('accuracy'):.4f}\n"
                        if metrics.get('accuracy') is not None
                        else "- Acurácia: N/A\n"
                    )
                    stats_content += (
                        f"- Log Loss: {metrics.get('log_loss'):.4f}\n"
                        if metrics.get('log_loss') is not None and not math.isnan(metrics.get('log_loss'))
                        else "- Log Loss: N/A\n"
                    )
                    stats_content += (
                        f"- ROC AUC: {metrics.get('roc_auc'):.4f}\n"
                        if metrics.get('roc_auc') is not None
                        else "- ROC AUC: N/A\n"
                    )
                    stats_content += "- Métricas 'Empate':\n"
                    stats_content += (
                        f"  - Precision: {metrics.get('precision_draw'):.4f}\n"
                        if metrics.get('precision_draw') is not None
                        else "  - Precision: N/A\n"
                    )
                    stats_content += (
                        f"  - Recall:    {metrics.get('recall_draw'):.4f}\n"
                        if metrics.get('recall_draw') is not None
                        else "  - Recall: N/A\n"
                    )
                    stats_content += (
                        f"  - F1-Score:  {metrics.get('f1_score_draw'):.4f}\n"
                        if metrics.get('f1_score_draw') is not None
                        else "  - F1-Score: N/A\n"
                    )
                    stats_content += "---\nEstratégia BackDraw:\n"
                    stats_content += (
                        f"- Nº Apostas: {metrics.get('num_bets') if metrics.get('num_bets') is not None else 'N/A'}\n"
                    )
                    stats_content += (
                        f"- Profit: {metrics.get('profit'):.2f} u\n"
                        if metrics.get('profit') is not None
                        else "- Profit: N/A\n"
                    )
                    stats_content += (
                        f"- ROI: {metrics.get('roi'):.2f} %\n"
                        if metrics.get('roi') is not None
                        else "- ROI: N/A\n"
                    )
                else:
                    stats_content += "Métricas Avaliação: Não disponíveis neste arquivo.\n"
                    self.log(f"AVISO: Métricas não encontradas para o modelo selecionado '{self.selected_model_id}'")

            self.model_stats_text.insert('1.0', stats_content)
            self.model_stats_text.config(state='disabled')
        except tk.TclError:
            pass
        except Exception as e_stats_disp:
            print(f"Erro update stats GUI: {e_stats_disp}")

    # --- Threads de Ação ---
    def start_training_thread(self): 
        self.log("Carregando históricos..."); 
        self.loaded_models_data = {}; 
        self.available_model_ids = []; 
        self.selected_model_var.set(''); 
        self.model_selector_combo.config(values=[]); 
        self.trained_model = None; 
        self.trained_scaler = None; 
        self.feature_columns = None; 
        self.model_best_params = None; 
        self.model_eval_metrics = None; 
        self.model_file_timestamp = None; 
        self.gui_queue.put(("update_stats_gui", None)); 
        self.set_button_state(self.load_train_button, tk.DISABLED); 
        self.set_button_state(self.predict_button, tk.DISABLED); 
        self.gui_queue.put(("progress_start", (100))); 
        self.gui_queue.put(("progress_update", (5, "Carregando Histórico...")))

        try:
            df_hist = load_historical_data(HISTORICAL_DATA_PATH)
            if df_hist is None:
                raise ValueError("Falha carregar histórico.")
            self.historical_data = df_hist
            self.log("Histórico carregado. Iniciando thread...")
            self.gui_queue.put(("progress_update", (20, "Iniciando Treino...")))
            train_thread = threading.Thread(target=self._run_training_pipeline, args=(self.historical_data,), daemon=True)
            train_thread.start()
        except Exception as e_load:
            error_msg = f"Erro Carregar Histórico: {e_load}"
            self.log(f"ERRO: {error_msg}")
            self.gui_queue.put(("error", ("Erro Carregamento", error_msg)))
            self.gui_queue.put(("progress_end", None))
            self.set_button_state(self.load_train_button, tk.NORMAL)

    def _run_training_pipeline(self, df_hist_raw: pd.DataFrame):
        """Executa o pipeline de treinamento, salvando os dois melhores modelos."""
        training_successful = False
        try:
            # Atualiza progresso e inicia pré-processamento
            self.gui_queue.put(("progress_update", (30, "Pré-processando...")))
            processed = preprocess_and_feature_engineer(df_hist_raw)
            if processed is None:
                raise ValueError("Falha no pré-processamento.")
            X_processed, y_processed, _ = processed

            # Prepara dados de teste para ROI
            self.gui_queue.put(("progress_update", (50, "Preparando teste ROI...")))
            df_hist_interm = calculate_historical_intermediate(df_hist_raw)
            df_hist_aligned = df_hist_interm.loc[X_processed.index.union(y_processed.index)].copy()
            _, X_test_full_data, _, _ = train_test_split(
                df_hist_aligned, y_processed, test_size=TEST_SIZE,
                random_state=RANDOM_STATE, stratify=y_processed
            )

            # Callback para progresso do treinamento
            def training_progress_callback(current_step, max_steps, status_text):
                progress = 60 + int((current_step / max_steps) * 35) if max_steps > 0 else 95
                self.gui_queue.put(("progress_update", (progress, status_text)))

            # Inicia o treinamento
            self.log("Iniciando treino...")
            self.gui_queue.put(("progress_update", (60, "Treinando...")))
            success = run_training_process(
                X_processed, y_processed,
                X_test_with_odds=X_test_full_data,
                odd_draw_col_name=CONFIG_ODDS_COLS['draw'],
                progress_callback=training_progress_callback
            )

            # Finaliza o progresso e verifica sucesso
            self.gui_queue.put(("progress_update", (95, "Finalizando...")))
            if success:
                self.gui_queue.put(("training_succeeded", None))
                training_successful = True
            else:
                raise RuntimeError("Falha no treino, seleção ou salvamento dos modelos.")

        except Exception as e:
            error_msg = f"Erro no Treinamento: {e}"
            self.log(f"ERRO: {error_msg}")
            self.gui_queue.put(("error", ("Erro no Treino", error_msg)))
            self.gui_queue.put(("training_failed", None))
        finally:
            self.gui_queue.put(("progress_end", None))
            self.set_button_state(self.load_train_button, tk.NORMAL)
        # A habilitação do botão de previsão será feita ao receber "training_succeeded" e recarregar os modelos

    def _run_prediction_pipeline(self): 
        """Executa o pipeline de previsão usando o modelo selecionado."""
        prediction_successful = False
        try:
            # Inicia progresso e busca CSV
            self.gui_queue.put(("progress_start", (100)))
            self.gui_queue.put(("progress_update", (10, f"Buscando CSV ({FIXTURE_FETCH_DAY})...")))
            fixture_df = fetch_and_process_fixtures()

            # Verifica se o CSV foi carregado corretamente
            if fixture_df is None:
                raise ValueError("Falha buscar CSV.")
            if fixture_df.empty:
                self.log("Nenhum jogo CSV.")
                self.gui_queue.put(("prediction_complete", None))
                return

            # Prepara features para previsão
            self.gui_queue.put(("progress_update", (40, f"Preparando features ({len(fixture_df)})...")))
            X_fixtures_prepared = prepare_fixture_data(
                fixture_df, self.historical_data, self.feature_columns
            )
            if X_fixtures_prepared is None or X_fixtures_prepared.empty:
                raise ValueError("Falha preparar features.")

            # Realiza previsões
            self.gui_queue.put(("progress_update", (70, "Realizando previsões...")))
            df_predictions = predictor.make_predictions(
                self.trained_model, self.trained_scaler, self.feature_columns,
                X_fixtures_prepared, fixture_df
            )
            if df_predictions is None:
                raise RuntimeError("Falha gerar previsões.")

            # Finaliza progresso e envia previsões para exibição
            self.gui_queue.put(("progress_update", (95, "Finalizando...")))
            self.gui_queue.put(("prediction_complete", df_predictions))
            prediction_successful = True

        except Exception as e:
            error_msg = f"Erro Previsão Thread: {e}"
            self.log(f"ERRO: {error_msg}")
            self.gui_queue.put(("error", ("Erro na Previsão", error_msg)))
            self.gui_queue.put(("prediction_complete", None))

        finally:
            # Finaliza progresso e reabilita botões
            self.gui_queue.put(("progress_end", None))
            self.set_button_state(self.load_train_button, tk.NORMAL)
            if self.trained_model:
                self.set_button_state(self.predict_button, tk.NORMAL)

    def start_prediction_thread(self): # ... (Verifica selected_model_id) ...
        if self.trained_model is None or self.selected_model_id is None: messagebox.showwarning("Modelo Não Selecionado", "Selecione modelo."); return
        if self.historical_data is None: messagebox.showwarning("Histórico Ausente", "Carregue/Treine."); return
        self.log(f"Iniciando previsão com '{self.selected_model_id}'..."); self.set_button_state(self.load_train_button, tk.DISABLED); self.set_button_state(self.predict_button, tk.DISABLED); self._setup_prediction_columns(['Status']); self.prediction_tree.insert('', tk.END, values=['Iniciando...'])
        predict_thread = threading.Thread(target=self._run_prediction_pipeline, daemon=True); predict_thread.start()


    # --- load_existing_model_assets (Adaptado para carregar AMBOS) ---
    def load_existing_model_assets(self):
        """Carrega os dois modelos (Melhor F1, Melhor ROI) se existirem."""
        self.loaded_models_data = {} # Limpa dados antigos
        self.available_model_ids = []
        any_model_loaded = False; default_selection = None

        model_paths_to_try = { MODEL_ID_F1: BEST_F1_MODEL_SAVE_PATH, MODEL_ID_ROI: BEST_ROI_MODEL_SAVE_PATH, }

        for model_id, model_path in model_paths_to_try.items():
            self.log(f"Tentando carregar {model_id} de {model_path}...")
            load_result = predictor.load_model_scaler_features(model_path)
            if load_result:
                model, scaler, features, params, metrics, timestamp = load_result
                if model and features:
                    self.log(f" -> Sucesso: {model_id} ({model.__class__.__name__}). Métricas: {'Sim' if metrics else 'NÃO'}")
                    self.loaded_models_data[model_id] = {'model': model, 'scaler': scaler, 'features': features, 'params': params, 'metrics': metrics, 'timestamp': timestamp, 'path': model_path}
                    self.available_model_ids.append(model_id)
                    any_model_loaded = True
                    if default_selection is None: default_selection = model_id # Pega o primeiro como padrão
                else: self.log(f" -> Falha: Arquivo {model_id} inválido.")
            else: self.log(f" -> Falha: Arquivo {model_id} não encontrado/erro.")

        self.model_selector_combo.config(values=self.available_model_ids) # Atualiza combobox
        if self.available_model_ids:
             self.selected_model_var.set(default_selection) # Define seleção padrão
             self.on_model_select() # Carrega estado e atualiza GUI para o padrão
             self.log(f"Modelos disponíveis: {self.available_model_ids}. Selecionado: {default_selection}")
        else: 
            self.log("Nenhum modelo pré-treinado válido."); 
            self.selected_model_var.set(""); 
            self.on_model_select() # Limpa estado

        self.log("Carregando histórico..."); 
        df_hist = load_historical_data(HISTORICAL_DATA_PATH)
        if df_hist is not None: self.historical_data = df_hist; self.log("Histórico carregado.")
        else: self.log("Falha carregar histórico."); any_model_loaded = False
        # Habilita prever apenas se um modelo foi selecionado E histórico carregado
        if self.selected_model_id and self.historical_data is not None: 
            self.set_button_state(self.predict_button, tk.NORMAL); 
            self.log("Pronto para prever.")
            
        else: self.set_button_state(self.predict_button, tk.DISABLED)

    def process_gui_queue(self):
        """Processa mensagens da fila da GUI (com progresso e correção de erro)."""
        keep_running = True
        while keep_running:
            try:
                # Pega UMA mensagem por vez
                message = self.gui_queue.get_nowait()

                # Processa a mensagem DENTRO de um try/except para isolar erros
                try:
                    if isinstance(message, tuple) and len(message) == 2:
                        msg_type, msg_payload = message

                        # Processa os tipos de mensagem conhecidos
                        if msg_type == "log":
                            self._update_log_area(msg_payload)
                        elif msg_type == "button_state":
                            self._update_button_state(msg_payload)
                        elif msg_type == "display_predictions":
                            self._update_prediction_display(msg_payload)
                        elif msg_type == "update_stats_gui":
                            self._update_model_stats_display_gui()
                        elif msg_type == "error":
                            if isinstance(msg_payload, tuple) and len(msg_payload) == 2:
                                messagebox.showerror(*msg_payload)
                            else:
                                self.log(f"AVISO GUI: Payload inválido p/ error: {msg_payload}")
                        elif msg_type == "info":
                            if isinstance(msg_payload, tuple) and len(msg_payload) == 2:
                                messagebox.showinfo(*msg_payload)
                            else:
                                self.log(f"AVISO GUI: Payload inválido p/ info: {msg_payload}")
                        elif msg_type == "progress_start":
                            max_val = msg_payload if isinstance(msg_payload, int) and msg_payload > 0 else 100
                            self.progress_bar.config(maximum=max_val, value=0)
                            self.progress_label.config(text="Iniciando...")
                        elif msg_type == "progress_update":
                            if isinstance(msg_payload, tuple) and len(msg_payload) == 2:
                                value, status_text = msg_payload
                                self.progress_bar['value'] = value
                                self.progress_label.config(text=status_text)
                            else:
                                self.log(f"AVISO GUI: Payload inválido p/ progress_update: {msg_payload}")
                        elif msg_type == "progress_end":
                            self.progress_bar['value'] = 0
                            self.progress_label.config(text="Pronto.")
                        elif msg_type == "training_succeeded":
                            self.log("Treino concluído. Recarregando modelos...")
                            self.load_existing_model_assets()
                            self.gui_queue.put(("info", ("Treino Concluído", "Melhores modelos salvos.")))
                        elif msg_type == "training_failed":
                            self.trained_model = None
                            self.trained_scaler = None
                            self.feature_columns = None
                            self.model_best_params = None
                            self.model_eval_metrics = None
                            self.model_file_timestamp = None
                            self.log("Pipeline treino falhou.")
                            self._update_model_stats_display_gui()
                            self.set_button_state(self.predict_button, tk.DISABLED)
                            self.set_button_state(self.load_train_button, tk.NORMAL)
                            self._setup_prediction_columns(['Status'])
                            self.prediction_tree.insert('', tk.END, values=['Falha Treinamento.'])
                        elif msg_type == "prediction_complete":
                            df_preds = msg_payload
                            self.log("Pipeline previsão concluído.")
                            self._update_prediction_display(df_preds)
                        else:
                            self.log(f"AVISO GUI: Tipo de mensagem desconhecido: {msg_type}")
                    else:
                        self.log(f"AVISO GUI: Mensagem formato inesperado (não tupla de 2): {message}")

                except Exception as process_error:
                    print(f"Erro ao processar mensagem GUI tipo '{msg_type}': {type(process_error).__name__} - {process_error}")
                    import traceback
                    traceback.print_exc()
                    self.log(f"ERRO GUI: Falha ao processar msg tipo '{msg_type}': {process_error}")

            except Empty:
                keep_running = False
            except Exception as outer_error:
                print(f"Erro fatal no loop da fila GUI: {type(outer_error).__name__} - {outer_error}")
                import traceback
                traceback.print_exc()
                keep_running = False
                try:
                    messagebox.showerror("Erro Crítico GUI", f"Erro inesperado na fila:\n{outer_error}")
                except Exception:
                    pass

            finally:
                if self.root.winfo_exists():
                    self.root.after(100, self.process_gui_queue)

    def on_closing(self): self.log("Fechando..."); self.root.destroy()

if __name__ == "__main__": 
    import requests; import math; import numpy as np; from sklearn.model_selection import train_test_split
    try: from ctypes import windll; windll.shcore.SetProcessDpiAwareness(1)
    except: pass
    root = tk.Tk()
    app = FootballPredictorDashboard(root)
    root.mainloop()