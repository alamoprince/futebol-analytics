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
    print(f"Erro import main.py (Treino/Previsão): {e}")
    raise # Re-levanta erro para app_launcher
except Exception as e_i:
    print(f"Erro geral import main.py (Treino/Previsão): {e_i}")
    raise # Re-levanta erro


class FootballPredictorDashboard:
    # Recebe o frame da aba (parent) e a janela principal (main_root)
    def __init__(self, parent_frame, main_root):
        self.parent = parent_frame # O Frame da aba onde os widgets serão colocados
        self.main_tk_root = main_root 

        self.gui_queue = Queue()
        self.historical_data: Optional[pd.DataFrame] = None
        # Removido self.historical_data_processed - a aba de análise cuida disso
        self.loaded_models_data: Dict[str, Dict] = {}
        self.available_model_ids: List[str] = []
        self.selected_model_id: Optional[str] = None
        self.trained_model: Optional[Any] = None
        self.trained_scaler: Optional[Any] = None
        self.feature_columns: Optional[List[str]] = None 
        self.model_best_params: Optional[Dict] = None
        self.model_eval_metrics: Optional[Dict] = None
        self.model_file_timestamp: Optional[str] = None

        # Chama a criação dos widgets *desta* aba
        self.create_train_predict_widgets() 

        # Inicia o processador da fila da GUI
        # Usa self.main_tk_root que é a janela principal Tk()
        self.main_tk_root.after(100, self.process_gui_queue)

        # --- Inicialização Específica desta Aba ---
        self.log(f"Aba Treino/Previsão Inicializada ({MODEL_TYPE_NAME})")
        self.log(f"Fonte CSV: '{FIXTURE_FETCH_DAY}'.")
        self.log("Carregando modelos e histórico...")
        self.load_existing_model_assets() # Carrega o necessário ao iniciar

    def create_train_predict_widgets(self):
        """Cria os widgets para a aba de Treino e Previsão."""
        # Usa self.parent (o frame da aba) como master do frame principal interno
        main_frame = ttk.Frame(self.parent, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Painel Esquerdo (Controles e Status) ---
        left_panel = ttk.Frame(main_frame) # Pai: main_frame
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), pady=5)

        control_frame = ttk.LabelFrame(left_panel, text=" Ações ", padding="10") # Pai: left_panel
        control_frame.pack(pady=(0, 5), fill=tk.X)

        self.load_train_button = ttk.Button(
            control_frame, text="TREINAR e Salvar Melhores Modelos", # Pai: control_frame
            command=self.start_training_thread, width=35
        )
        self.load_train_button.pack(pady=5, fill=tk.X)

        predict_frame = ttk.Frame(control_frame) # Pai: control_frame
        predict_frame.pack(fill=tk.X, pady=5)

        self.predict_button = ttk.Button(
            predict_frame, text=f"Prever ({FIXTURE_FETCH_DAY.capitalize()}) com:", # Pai: predict_frame
            command=self.start_prediction_thread, width=18
        )
        self.predict_button.pack(side=tk.LEFT, fill=tk.X, expand=False)
        self.predict_button.config(state=tk.DISABLED)

        self.selected_model_var = tk.StringVar()
        self.model_selector_combo = ttk.Combobox(
            predict_frame, textvariable=self.selected_model_var, # Pai: predict_frame
            state="readonly", width=20
        )
        self.model_selector_combo.pack(side=tk.RIGHT, padx=(5, 0), fill=tk.X, expand=True)
        self.model_selector_combo.bind("<<ComboboxSelected>>", self.on_model_select) # on_model_select deve existir

        progress_frame = ttk.Frame(control_frame) # Pai: control_frame
        progress_frame.pack(fill=tk.X, pady=(10, 0))

        self.progress_label = ttk.Label(progress_frame, text="Pronto.") # Pai: progress_frame
        self.progress_label.pack(side=tk.LEFT, padx=(0, 5))

        self.progress_bar = ttk.Progressbar(
            progress_frame, orient=tk.HORIZONTAL, length=200, mode='determinate' # Pai: progress_frame
        )
        self.progress_bar.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        stats_frame = ttk.LabelFrame(left_panel, text=" Status Modelo Selecionado ", padding="10") # Pai: left_panel
        stats_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        self.model_stats_text = ScrolledText(
            stats_frame, height=15, state='disabled', wrap=tk.WORD, # Pai: stats_frame
            font=("Consolas", 9), relief=tk.FLAT, bd=0
        )
        self.model_stats_text.pack(fill=tk.BOTH, expand=True)
        self._update_model_stats_display_gui() # Atualiza ao criar

        # --- Painel Direito (Previsões e Logs) ---
        right_panel = ttk.Frame(main_frame) # Pai: main_frame
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        results_frame = ttk.LabelFrame(right_panel, text=" Previsões ", padding="5") # Pai: right_panel
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        cols = ['Data', 'Hora', 'Liga', 'Casa', 'Fora', 'Odd H', 'Odd D', 'Odd A', 'O2.5', 'BTTS S', 'P(Ñ Emp)', 'P(Empate)']
        self.prediction_tree = ttk.Treeview(
            results_frame, columns=cols, show='headings', height=10 # Pai: results_frame
        )
        vsb = ttk.Scrollbar(results_frame, orient="vertical", command=self.prediction_tree.yview)
        hsb = ttk.Scrollbar(results_frame, orient="horizontal", command=self.prediction_tree.xview)
        self.prediction_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side='right', fill='y')
        hsb.pack(side='bottom', fill='x')
        self.prediction_tree.pack(fill=tk.BOTH, expand=True)
        self._setup_prediction_columns(cols) # Configura ao criar

        log_frame = ttk.LabelFrame(right_panel, text=" Logs ", padding="5") # Pai: right_panel
        log_frame.pack(fill=tk.BOTH, expand=True, side=tk.BOTTOM, pady=(5, 0))

        self.log_area = ScrolledText(
            log_frame, height=8, state='disabled', wrap=tk.WORD, font=("Consolas", 9) # Pai: log_frame
        )
        self.log_area.pack(fill=tk.BOTH, expand=True)

    def on_model_select(self, event=None):
        """Handles the event when a model is selected from the combobox."""
        selected_id = self.selected_model_var.get()
        self.log(f"Modelo selecionado via Combobox: {selected_id}")
        self._update_gui_for_selected_model(selected_id) # Calls helper to update internal state and GUI

    def _update_gui_for_selected_model(self, selected_id: Optional[str]):
        """Atualiza o estado interno e a GUI com base no modelo selecionado."""
        if selected_id and selected_id in self.loaded_models_data:
            model_data = self.loaded_models_data[selected_id]
            self.log(f"Atualizando GUI para: {selected_id}")
            self.selected_model_id = selected_id
            self.trained_model = model_data.get('model')
            self.trained_scaler = model_data.get('scaler')
            self.feature_columns = model_data.get('features')
            self.model_best_params = model_data.get('params')
            self.model_eval_metrics = model_data.get('metrics')
            self.model_file_timestamp = model_data.get('timestamp') # Get timestamp from loaded data

            self._update_model_stats_display_gui() # Atualiza o display de stats

            # Habilita/desabilita botão de prever baseado no modelo E histórico
            if self.trained_model and self.feature_columns and self.historical_data is not None:
                self.set_button_state(self.predict_button, tk.NORMAL)
                self.log(f"Modelo '{selected_id}' pronto para previsão.")
            elif self.historical_data is None:
                self.log(f"Modelo '{selected_id}' selecionado, mas histórico ausente.")
                self.set_button_state(self.predict_button, tk.DISABLED)
            else: # Modelo inválido ou features ausentes
                self.log(f"Modelo '{selected_id}' inválido ou dados ausentes.")
                self.set_button_state(self.predict_button, tk.DISABLED)
        else:
            # Caso onde seleção é limpa ou inválida
            self.log(f"Seleção inválida ou limpa: '{selected_id}'.")
            self.selected_model_id = None
            self.trained_model = None
            self.trained_scaler = None
            self.feature_columns = None
            self.model_best_params = None
            self.model_eval_metrics = None
            self.model_file_timestamp = None
            self.set_button_state(self.predict_button, tk.DISABLED)
            self._update_model_stats_display_gui()

    # --- MÉTODOS AUXILIARES DA GUI (Mantidos) ---
    def log(self, message: str):
        """ Envia mensagem para a fila da GUI para ser exibida na área de log desta aba. """
        try:
            self.gui_queue.put(("log", message))
        except Exception as e:
            print(f"Erro Fila Log: {e}") # Fallback

    def _update_log_area(self, message: str):
         """ Atualiza o widget ScrolledText de log (self.log_area). """
         try:
             if hasattr(self, 'log_area') and self.log_area.winfo_exists():
                self.log_area.config(state='normal')
                ts = datetime.datetime.now().strftime("%H:%M:%S")
                self.log_area.insert(tk.END, f"[{ts}] {message}\n")
                self.log_area.config(state='disabled')
                self.log_area.see(tk.END)
         except tk.TclError: pass # Ignora se widget destruído

    def set_button_state(self, button: ttk.Button, state: str):
        """ Envia comando para a fila para alterar estado de um botão. """
        try:
            self.gui_queue.put(("button_state", (button, state)))
        except Exception as e:
            print(f"Erro Fila Botão: {e}") # Fallback

    def _update_button_state(self, button_state_tuple):
         """ Atualiza o estado de um botão (chamado pela fila). """
         button, state = button_state_tuple
         try:
             if button.winfo_exists():
                 button.config(state=state)
         except tk.TclError: pass


    def _update_model_stats_display_gui(self):
        """ Atualiza o ScrolledText com status do modelo selecionado (self.model_stats_text). """
        # (Código da função _update_model_stats_display_gui aqui - como na resposta anterior,
        # garantindo que acessa self.model_stats_text e usa os atributos da classe como
        # self.selected_model_id, self.loaded_models_data, etc.)
        try:
            # Verifica se o widget existe antes de tentar acessá-lo
            if not hasattr(self, 'model_stats_text') or not self.model_stats_text.winfo_exists():
                # print("Debug: model_stats_text não existe ou foi destruído.") # Debug opcional
                return

            self.model_stats_text.config(state='normal')
            self.model_stats_text.delete('1.0', tk.END)

            if self.trained_model is None or self.selected_model_id is None:
                stats_content = "Nenhum modelo selecionado ou carregado."
            else:
                model_data = self.loaded_models_data.get(self.selected_model_id, {})
                metrics = model_data.get('metrics', {}) # Pega métricas ou dict vazio
                params = model_data.get('params')
                features = model_data.get('features', []) # Pega features ou lista vazia
                timestamp = self.model_file_timestamp # Usa o atributo da classe
                path = model_data.get('path')
                model_class_name = self.trained_model.__class__.__name__ if self.trained_model else "N/A"

                stats_content = f"Modelo: {self.selected_model_id}\n"
                stats_content += f"Arquivo: {os.path.basename(path or 'N/A')}\n"
                stats_content += f"Modif.: {timestamp or 'N/A'}\n"
                stats_content += f"Tipo: {model_class_name}\n---\n"

                if features:
                    stats_content += f"Features ({len(features)}):\n - " + "\n - ".join(features) + "\n---\n"
                else:
                    stats_content += "Features: Não disponíveis\n---\n"

                if params:
                    stats_content += "Params:\n"; params_list = [f" - {k}: {v}" for k,v in params.items()]; stats_content += "\n".join(params_list) + "\n---\n"
                else:
                    stats_content += "Params: Não disponíveis\n---\n"

                # Exibição segura das métricas
                acc = metrics.get('accuracy')
                loss = metrics.get('log_loss')
                auc = metrics.get('roc_auc')
                prec_d = metrics.get('precision_draw')
                rec_d = metrics.get('recall_draw')
                f1_d = metrics.get('f1_score_draw')
                conf_matrix = metrics.get('confusion_matrix')
                profit = metrics.get('profit')
                roi_val = metrics.get('roi')
                n_bets = metrics.get('num_bets')
                train_n = metrics.get('train_set_size', 'N/A')
                test_n = metrics.get('test_set_size', 'N/A')

                stats_content += "Métricas (Teste):\n"
                stats_content += f"- Acurácia: {acc:.4f}\n" if acc is not None else "- Acc: N/A\n"
                stats_content += f"- Log Loss: {loss:.4f}\n" if loss is not None and not math.isnan(loss) else "- Log Loss: N/A\n"
                stats_content += f"- ROC AUC: {auc:.4f}\n" if auc is not None else "- AUC: N/A\n"
                stats_content += "- Métricas Empate:\n"
                stats_content += f"  - Prec: {prec_d:.4f}\n" if prec_d is not None else "  - Prec: N/A\n"
                stats_content += f"  - Rec:  {rec_d:.4f}\n" if rec_d is not None else "  - Rec: N/A\n"
                stats_content += f"  - F1:   {f1_d:.4f}\n" if f1_d is not None else "  - F1: N/A\n"

                if conf_matrix and isinstance(conf_matrix, list) and len(conf_matrix) == 2 and all(isinstance(row, list) and len(row) == 2 for row in conf_matrix):
                   stats_content += "---\nMatriz Confusão:\n"
                   stats_content += f"      Prev:ÑEmp| Prev:Emp\n"
                   stats_content += f"Real:ÑEmp {conf_matrix[0][0]:<6d}| {conf_matrix[0][1]:<6d}\n"
                   stats_content += f"Real:Emp  {conf_matrix[1][0]:<6d}| {conf_matrix[1][1]:<6d}\n"

                stats_content += f"\nAmostras T/T: {train_n} / {test_n}\n---\n"
                stats_content += "BackDraw Simples (Baseado em Previsão Binária):\n" # Nome atualizado
                stats_content += f"- Nº Bets: {n_bets if n_bets is not None else 'N/A'}\n"
                stats_content += f"- Profit: {profit:.2f} u\n" if profit is not None else "- Profit: N/A\n"
                stats_content += f"- ROI: {roi_val:.2f} %\n" if roi_val is not None else "- ROI: N/A\n"

            self.model_stats_text.insert('1.0', stats_content)
            self.model_stats_text.config(state='disabled')
        except tk.TclError:
             pass # Widget pode ter sido destruído
        except Exception as e:
             print(f"Erro _update_model_stats_display_gui: {e}")
             # Tenta mostrar erro no próprio widget se possível
             try:
                 if hasattr(self, 'model_stats_text') and self.model_stats_text.winfo_exists():
                     self.model_stats_text.config(state='normal')
                     self.model_stats_text.delete('1.0', tk.END)
                     self.model_stats_text.insert('1.0', f"Erro ao exibir stats:\n{e}")
                     self.model_stats_text.config(state='disabled')
             except: pass


    def _setup_prediction_columns(self, columns: List[str]):
         """ Configura as colunas da Treeview de previsões (self.prediction_tree). """
         # (Código da função _setup_prediction_columns aqui - como na resposta anterior)
         try:
            if not hasattr(self, 'prediction_tree') or not self.prediction_tree.winfo_exists():
                 return # Sai se a treeview não existe

            self.prediction_tree['columns'] = columns
            self.prediction_tree.delete(*self.prediction_tree.get_children()) # Limpa dados antigos

            # Define larguras (ajuste conforme necessário)
            col_widths = {'Data': 75, 'Hora': 50, 'Liga': 150, 'Casa': 110, 'Fora': 110,
                           'Odd H': 50, 'Odd D': 50, 'Odd A': 50, 'O2.5': 50, 'BTTS S': 55,
                           'P(Ñ Emp)': 70, 'P(Empate)': 70, 'Status': 500} # Largura para coluna 'Status'

            # Configura cada coluna
            for col in columns:
                # Remove a coluna da treeview antes de reconfigurá-la para evitar duplicação
                self.prediction_tree.heading(col, text='') # Limpa heading antigo
                self.prediction_tree.column(col, width=0, minwidth=0, stretch=tk.NO) # Esconde temporariamente

                # Reconfigura heading e coluna
                width = col_widths.get(col, 80) # Pega largura ou usa default
                anchor = tk.W if col in ['Liga', 'Casa', 'Fora', 'Data', 'Hora', 'Status'] else tk.CENTER # Alinhamento
                stretch = tk.NO # Não esticar colunas por padrão

                self.prediction_tree.heading(col, text=col, anchor=anchor) # Define texto e alinhamento do header
                self.prediction_tree.column(col, anchor=anchor, width=width, stretch=stretch) # Define alinhamento, largura e stretch da coluna

            # Caso especial para coluna única 'Status'
            if columns == ['Status']:
                self.prediction_tree.column('Status', stretch=tk.YES) # Permite esticar a coluna Status

         except tk.TclError: pass
         except Exception as e: print(f"Erro _setup_prediction_columns: {e}")


    def _update_prediction_display(self, df: Optional[pd.DataFrame]):
        """ Atualiza a Treeview de previsões (self.prediction_tree) com os dados do DataFrame. """
        # (Código da função _update_prediction_display aqui - como na resposta anterior,
        #  formatando as probabilidades e odds e inserindo as linhas na self.prediction_tree)
        try:
            if not hasattr(self, 'prediction_tree') or not self.prediction_tree.winfo_exists():
                self.log("Widget Treeview de previsão não encontrado.")
                return
        except tk.TclError:
             self.log("Erro ao verificar Treeview de previsão.")
             return

        self.log(f"--- GUI: Atualizando display de previsões ---")
        if df is None or df.empty:
            self.log("GUI: DataFrame de previsões vazio ou None. Exibindo status.")
            try:
                self._setup_prediction_columns(['Status'])
                self.prediction_tree.insert('', tk.END, values=['Nenhuma previsão válida.'])
            except Exception as e_clear: self.log(f"Erro ao limpar/definir status da treeview: {e_clear}")
            return

        self.log(f"GUI: Recebido DataFrame {df.shape}. Preparando exibição...")

        # Colunas a exibir e mapeamento (igual ao anterior)
        display_headers = ['Data', 'Hora', 'Liga', 'Casa', 'Fora', 'Odd H', 'Odd D', 'Odd A', 'O2.5', 'BTTS S', 'P(Ñ Emp)', 'P(Empate)']
        odds_h_col = CONFIG_ODDS_COLS.get('home', 'Odd_H_FT')
        odds_d_col = CONFIG_ODDS_COLS.get('draw', 'Odd_D_FT')
        odds_a_col = CONFIG_ODDS_COLS.get('away', 'Odd_A_FT')
        prob_non_draw_col = f'Prob_{CLASS_NAMES[0]}' if CLASS_NAMES and len(CLASS_NAMES) > 0 else 'Prob_Nao_Empate'
        prob_draw_col = f'Prob_{CLASS_NAMES[1]}' if CLASS_NAMES and len(CLASS_NAMES) > 1 else 'Prob_Empate'

        header_to_col_map = {
            'Data': 'Date_Str', 'Hora': 'Time_Str', 'Liga': 'League', 'Casa': 'HomeTeam', 'Fora': 'AwayTeam',
            'Odd H': odds_h_col, 'Odd D': odds_d_col, 'Odd A': odds_a_col,
            'O2.5': 'Odd_Over25_FT', 'BTTS S': 'Odd_BTTS_Yes', # Nomes fixos assumidos
            'P(Ñ Emp)': prob_non_draw_col, 'P(Empate)': prob_draw_col
        }

        internal_cols_to_fetch = [header_to_col_map.get(h) for h in display_headers if header_to_col_map.get(h)]
        valid_internal_cols = [c for c in internal_cols_to_fetch if c in df.columns]

        if not valid_internal_cols:
            self.log("ERRO GUI: Nenhuma coluna válida encontrada no DF para exibir na Treeview!");
            try:
                self._setup_prediction_columns(['Status'])
                self.prediction_tree.insert('', tk.END, values=['Erro: Colunas de dados ausentes.'])
            except Exception as e_err_col: self.log(f"Erro ao exibir status de colunas ausentes: {e_err_col}")
            return

        try:
            # Reconfigura colunas da Treeview
            self.log(f"GUI: Configurando colunas Treeview: {display_headers}")
            self._setup_prediction_columns(display_headers)

            # Prepara dados para display
            self.log(f"GUI: Selecionando e formatando colunas válidas: {valid_internal_cols}")
            df_display = df[valid_internal_cols].copy()

            # Formata Probabilidades
            prob_cols_format = [prob_non_draw_col, prob_draw_col]
            for pcol in prob_cols_format:
                if pcol in df_display.columns:
                    try:
                        df_display[pcol] = (pd.to_numeric(df_display[pcol], errors='coerce') * 100).round(1).astype(str) + '%'
                        df_display[pcol] = df_display[pcol].replace('nan%', '-', regex=False)
                    except Exception as e_fmt_prob:
                        self.log(f"AVISO GUI: Erro ao formatar prob '{pcol}': {e_fmt_prob}")
                        df_display[pcol] = "-" # Define como '-' se a formatação falhar

            # Formata Odds
            odds_cols_to_format = [c for c in valid_internal_cols if str(c).startswith('Odd_')]
            for ocol in odds_cols_to_format:
                 if ocol in df_display.columns:
                     try:
                         df_display[ocol] = pd.to_numeric(df_display[ocol], errors='coerce').apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
                     except Exception as e_fmt_odd:
                         self.log(f"AVISO GUI: Erro ao formatar odd '{ocol}': {e_fmt_odd}")
                         df_display[ocol] = "-"

            # Insere linhas na Treeview
            self.log("GUI: Adicionando linhas à Treeview...")
            added_rows = 0
            for index, row in df_display.iterrows():
                # Pega valores na ordem dos HEADERS
                values_to_insert = [str(row.get(header_to_col_map.get(header), '')) if header_to_col_map.get(header) else '' for header in display_headers]
                try:
                    self.prediction_tree.insert('', tk.END, values=values_to_insert)
                    added_rows += 1
                except Exception as e_insert:
                     self.log(f"!!!!! ERRO GUI ao inserir linha {index} na Treeview: {e_insert} - Valores: {values_to_insert}")

            self.log(f"GUI: {added_rows} de {len(df_display)} linhas adicionadas à Treeview.")
            if added_rows < len(df_display): self.log("AVISO GUI: Nem todas as linhas foram adicionadas (verificar erros de inserção).")

        except tk.TclError:
            self.log("Erro TclError (provavelmente widget destruído) ao atualizar display.")
        except Exception as e:
             self.log(f"!!!!! ERRO GERAL _update_prediction_display: {e}");
             traceback.print_exc(); # Log completo no console
             try:
                 # Tenta mostrar um erro na Treeview
                 self._setup_prediction_columns(['Status'])
                 self.prediction_tree.insert('', tk.END, values=[f'Erro ao exibir: {e}'])
             except: pass # Ignora erro ao tentar mostrar o erro


    # --- Lógica de Ação Principal (Threads) ---
    # (Manter start_training_thread, _run_training_pipeline,
    #  start_prediction_thread, _run_prediction_pipeline como nas respostas anteriores)
    def start_training_thread(self):
        self.log("Iniciando processo de treino em background...")
        # Limpa estado anterior
        self.loaded_models_data = {}
        self.available_model_ids = []
        self.selected_model_var.set('')
        try:
            self.model_selector_combo.config(values=[])
        except tk.TclError: pass # Ignora se combobox não existe mais

        self.selected_model_id = None
        self.trained_model = None
        self.trained_scaler = None
        self.feature_columns = None
        self.model_best_params = None
        self.model_eval_metrics = None
        self.model_file_timestamp = None
        self.gui_queue.put(("update_stats_gui", None)) # Limpa display de stats

        # Desabilita botões e inicia progresso
        self.set_button_state(self.load_train_button, tk.DISABLED)
        self.set_button_state(self.predict_button, tk.DISABLED)
        self.gui_queue.put(("progress_start", (100,))) # Total de 100 passos (arbitrário)
        self.gui_queue.put(("progress_update", (5, "Carregando Histórico...")))

        # Tenta carregar dados antes de iniciar a thread
        try:
            df_hist = load_historical_data(HISTORICAL_DATA_PATH) # load_historical_data já loga sucesso/erro
            if df_hist is None:
                raise ValueError("Falha ao carregar dados históricos do arquivo.")
            self.historical_data = df_hist # Armazena na instância
            self.log("Dados históricos carregados com sucesso.")
            self.gui_queue.put(("progress_update", (20, "Iniciando Thread de Treino...")))

            # Inicia a thread de treino
            train_thread = threading.Thread(
                target=self._run_training_pipeline,
                args=(self.historical_data.copy(),), # Passa uma cópia para evitar race condition?
                daemon=True # Permite fechar app mesmo se thread estiver rodando
            )
            train_thread.start()

        except Exception as e_load:
            error_msg = f"Erro ao carregar Histórico para Treino: {e_load}"
            self.log(f"ERRO: {error_msg}")
            # Usa a fila da GUI para mostrar o erro
            self.gui_queue.put(("error", ("Erro de Carregamento", error_msg)))
            self.gui_queue.put(("progress_end", None)) # Finaliza barra de progresso
            self.set_button_state(self.load_train_button, tk.NORMAL) # Reabilita botão de treino

    def _run_training_pipeline(self, df_hist_raw: pd.DataFrame):
        """ Executa o pipeline de treino (pré-processamento, treino, avaliação, salvamento) em uma thread. """
        training_successful = False
        try:
            # 1. Pré-processamento e Engenharia de Features
            self.gui_queue.put(("progress_update", (30, "Pré-processando Features...")))
            # Usa a função de data_handler que retorna X, y e a lista de features usadas
            processed_data = preprocess_and_feature_engineer(df_hist_raw)
            if processed_data is None:
                raise ValueError("Falha no pré-processamento ou engenharia de features.")
            X_processed, y_processed, features_used = processed_data
            self.log(f"Pré-processamento concluído. Features: {features_used}")

            # 2. Preparação para Cálculo de ROI no Test Set
            #    Precisamos alinhar os dados completos (com odds) ao índice do conjunto de teste
            self.gui_queue.put(("progress_update", (50, "Preparando dados p/ ROI...")))
            # Recalcula intermediárias no DF bruto para garantir colunas de odds/resultado
            # (Pode ser otimizado se preprocess_and_feature_engineer já retornar tudo)
            df_hist_interm = calculate_historical_intermediate(df_hist_raw)
            # Alinha com os índices de X/y que *passaram* pelo dropna no pré-proc
            common_index = X_processed.index.union(y_processed.index)
            df_hist_aligned = df_hist_interm.loc[common_index].copy()

            # Divide os dados ALINHADOS para obter X_test_full que contém as odds
            _, X_test_full_data, _, y_test_for_roi = train_test_split(
                df_hist_aligned, # Usa o DF alinhado que contém tudo
                y_processed,     # Usa o y processado para estratificação
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                stratify=y_processed
            )
            # X_test_full_data agora tem o mesmo índice que o X_test usado no treino
            # e contém a coluna de Odd_D_FT (e outras) para o cálculo do ROI.

            # 3. Função de Callback para Progresso do Treino
            def training_progress_callback(current_step, max_steps, status_text):
                # Mapeia progresso do treino (0 a max_steps) para a faixa 60-95 da barra total
                progress_percent = 60 + int((current_step / max_steps) * 35) if max_steps > 0 else 95
                self.gui_queue.put(("progress_update", (progress_percent, status_text)))

            # 4. Treinamento, Avaliação e Seleção dos Melhores Modelos
            self.log("Iniciando treinamento dos modelos...")
            self.gui_queue.put(("progress_update", (60, "Treinando Modelos...")))

            # Chama a função de model_trainer que faz tudo
            success = run_training_process( # Nome importado
                X=X_processed,              # Features processadas para treino
                y=y_processed,              # Alvo para treino
                X_test_with_odds=X_test_full_data, # Dados de teste com odds para ROI
                odd_draw_col_name=CONFIG_ODDS_COLS.get('draw', 'Odd_D_FT'), # Nome da coluna de odd de empate
                progress_callback=training_progress_callback # Callback para atualizar GUI
            )
            self.gui_queue.put(("progress_update", (95, "Finalizando Treinamento...")))

            # 5. Resultado
            if success:
                self.log("Treinamento e salvamento concluídos com sucesso.")
                # Envia sinal para recarregar modelos na thread principal da GUI
                self.gui_queue.put(("training_succeeded", None))
                training_successful = True
            else:
                # A função run_training_process deve logar erros específicos
                raise RuntimeError("Falha no processo de treinamento/seleção/salvamento dos modelos.")

        except Exception as e:
            error_msg = f"Erro durante o Pipeline de Treino: {e}"
            self.log(f"ERRO Thread Treino: {error_msg}")
            traceback.print_exc() # Log completo no console
            # Envia erro para a GUI
            self.gui_queue.put(("error", ("Erro no Treino", error_msg)))
            # Sinaliza falha
            self.gui_queue.put(("training_failed", None)) # Sinal específico para falha

        finally:
            # Finaliza a barra de progresso e reabilita botão de treino na GUI
            self.gui_queue.put(("progress_end", None))
            self.set_button_state(self.load_train_button, tk.NORMAL)
            # Não habilita o botão de prever aqui, load_existing_model_assets fará isso se treino der certo

    def start_prediction_thread(self):
        """ Inicia a thread para buscar jogos, preparar features e prever. """
        # Verifica se há um modelo selecionado e dados históricos carregados
        if self.trained_model is None or self.selected_model_id is None:
            messagebox.showwarning("Modelo Não Selecionado", "Selecione um modelo treinado na lista antes de prever.", parent=self.parent) # Adiciona parent
            return
        if self.historical_data is None:
             messagebox.showwarning("Histórico Ausente", "Os dados históricos não foram carregados. Treine ou carregue primeiro.", parent=self.parent) # Adiciona parent
             return
        if not self.feature_columns:
             messagebox.showwarning("Features Ausentes", "A lista de features para o modelo selecionado não foi carregada.", parent=self.parent) # Adiciona parent
             return

        self.log(f"Iniciando previsão com modelo '{self.selected_model_id}'...")
        # Desabilita botões durante a previsão
        self.set_button_state(self.load_train_button, tk.DISABLED)
        self.set_button_state(self.predict_button, tk.DISABLED)

        # Limpa a árvore de previsões e mostra status inicial
        try:
             self._setup_prediction_columns(['Status'])
             self.prediction_tree.insert('', tk.END, values=['Buscando jogos...'])
        except Exception as e_clear: self.log(f"Erro ao limpar treeview: {e_clear}")

        # Inicia a thread de previsão
        predict_thread = threading.Thread(
            target=self._run_prediction_pipeline,
            daemon=True # Permite fechar o app mesmo se a thread travar
        )
        predict_thread.start()

    def _run_prediction_pipeline(self):
        """ Executa o pipeline de previsão (busca CSV, prepara features, prevê, filtra) em uma thread. """
        prediction_successful = False
        df_predictions_final_filtered = None # DataFrame final a ser exibido
        try:
            # 1. Inicia Barra de Progresso
            self.gui_queue.put(("progress_start", (100,))) # Total 100 passos
            self.gui_queue.put(("progress_update", (10, f"Buscando CSV ({FIXTURE_FETCH_DAY})...")))

            # 2. Busca e Processa Jogos Futuros (CSV)
            fixture_df = fetch_and_process_fixtures() # Função de data_handler
            if fixture_df is None:
                # fetch_and_process_fixtures deve logar o erro
                raise ValueError("Falha ao buscar ou processar o arquivo CSV de jogos futuros.")
            if fixture_df.empty:
                self.log("Nenhum jogo encontrado no CSV para o dia selecionado.")
                # Envia None para a GUI limpar a tabela
                self.gui_queue.put(("prediction_complete", None))
                return # Finaliza a execução da thread

            self.log(f"{len(fixture_df)} jogos encontrados no CSV. Preparando features...")
            self.gui_queue.put(("progress_update", (40, f"Preparando {len(fixture_df)} jogos...")))

            # 3. Prepara Features para os Jogos Futuros
            #    Usa a função de data_handler, passando os dados históricos e as features do modelo carregado
            if not self.feature_columns: # Segurança extra
                 raise ValueError("Lista de features do modelo não está disponível.")
            if self.historical_data is None: # Segurança extra
                 raise ValueError("Dados históricos não estão disponíveis para calcular rolling stats.")

            X_fixtures_prepared = prepare_fixture_data(
                fixture_df=fixture_df,
                historical_df=self.historical_data,
                feature_columns=self.feature_columns # Usa as features do modelo SELECIONADO
            )

            if X_fixtures_prepared is None:
                raise ValueError("Falha ao preparar as features para os jogos futuros.")
            if X_fixtures_prepared.empty and not fixture_df.empty:
                 # Isso pode acontecer se NENHUM jogo futuro puder ter features calculadas
                 self.log("Aviso: Nenhum jogo futuro pôde ter features preparadas (talvez times sem histórico?).")
                 self.gui_queue.put(("prediction_complete", None))
                 return
            elif X_fixtures_prepared.empty and fixture_df.empty:
                 self.log("Nenhum jogo futuro para prever após preparação.") # Já tratado acima
                 self.gui_queue.put(("prediction_complete", None))
                 return


            self.log(f"Features preparadas para {len(X_fixtures_prepared)} jogos. Realizando previsões...")
            self.gui_queue.put(("progress_update", (70, f"Prevendo {len(X_fixtures_prepared)} jogos...")))

            # 4. Faz as Previsões
            df_predictions_raw = predictor.make_predictions(
                model=self.trained_model,       # Modelo selecionado
                scaler=self.trained_scaler,     # Scaler associado (pode ser None)
                feature_names=self.feature_columns,# Features do modelo
                X_fixture_prepared=X_fixtures_prepared, # Dados preparados
                fixture_info=fixture_df.loc[X_fixtures_prepared.index] # Info original dos jogos previstos
            )

            if df_predictions_raw is None:
                raise RuntimeError("Falha ao gerar as previsões brutas.")
            self.log(f"Previsões brutas geradas para {len(df_predictions_raw)} jogos.")

            # 5. Aplica Filtros (se necessário) - ADAPTAR CONFORME SUA ESTRATÉGIA
            #    Exemplo: manter apenas jogos com P(Empate) > threshold
            df_to_filter = df_predictions_raw.copy()
            self.log("Aplicando filtros nas previsões...")

            # Exemplo Filtro 1: Remover NaNs em colunas de Odds usadas no display (redundante se prepare_fixture_data tratou)
            # cols_to_check_nan = [CONFIG_ODDS_COLS.get('home'), ...] # Defina as colunas
            # initial_rows_f1 = len(df_to_filter)
            # df_to_filter.dropna(subset=cols_to_check_nan, inplace=True)
            # dropped_f1 = initial_rows_f1 - len(df_to_filter)
            # if dropped_f1 > 0: self.log(f"Filtro 1: Removidos {dropped_f1} jogos (NaNs em Odds Display).")

            # Exemplo Filtro 2: Probabilidade de Empate
            prob_draw_col = f'Prob_{CLASS_NAMES[1]}' if CLASS_NAMES and len(CLASS_NAMES) > 1 else 'Prob_Empate'
            df_predictions_final_filtered = df_to_filter # Default: usa todos que restaram

            if prob_draw_col in df_to_filter.columns:
                threshold = 0.5 # <<< DEFINA SEU LIMIAR DE PROBABILIDADE AQUI >>>
                self.log(f"Aplicando Filtro: Manter jogos com P(Empate) > {threshold*100:.1f}%")

                # Garante que a coluna de probabilidade é numérica
                df_to_filter[prob_draw_col] = pd.to_numeric(df_to_filter[prob_draw_col], errors='coerce')
                # Remove linhas onde a prob não pôde ser convertida para número
                df_to_filter.dropna(subset=[prob_draw_col], inplace=True)

                initial_rows_f2 = len(df_to_filter)
                df_filtered_f2 = df_to_filter[df_to_filter[prob_draw_col] > threshold].copy() # Aplica filtro e copia
                rows_kept_f2 = len(df_filtered_f2)
                self.log(f"Filtro Probabilidade: {rows_kept_f2} de {initial_rows_f2} jogos restantes passaram (P(Empate) > {threshold}).")
                df_predictions_final_filtered = df_filtered_f2 # Atualiza o DF final
            else:
                self.log(f"Aviso: Coluna de probabilidade de empate '{prob_draw_col}' não encontrada para filtro.")

            # --- FIM DOS FILTROS ---

            # 6. Ordenação (Opcional)
            if df_predictions_final_filtered is not None and not df_predictions_final_filtered.empty and prob_draw_col in df_predictions_final_filtered.columns:
                self.log(f"Ordenando resultados por '{prob_draw_col}' descendente...")
                try:
                    df_predictions_final_filtered = df_predictions_final_filtered.sort_values(
                        by=prob_draw_col, ascending=False
                    ).reset_index(drop=True)
                except Exception as e_sort:
                    self.log(f"Aviso: Erro ao ordenar previsões: {e_sort}")

            # 7. Envia Resultado para a GUI
            self.gui_queue.put(("progress_update", (95, "Preparando Exibição...")))
            if df_predictions_final_filtered is not None and not df_predictions_final_filtered.empty:
                self.log(f"Enviando {len(df_predictions_final_filtered)} previsões finais filtradas para exibição.")
                prediction_successful = True
                # Envia o DataFrame filtrado para a GUI
                self.gui_queue.put(("prediction_complete", df_predictions_final_filtered))
            else:
                 self.log("Nenhuma previsão restante após filtros para exibir.")
                 prediction_successful = False
                 # Envia None para a GUI limpar a tabela
                 self.gui_queue.put(("prediction_complete", None))

        except Exception as e:
            error_msg = f"Erro durante Pipeline de Previsão: {e}"
            self.log(f"ERRO Thread Previsão: {error_msg}")
            traceback.print_exc() # Log completo no console
            # Envia erro para a GUI
            self.gui_queue.put(("error", ("Erro na Previsão", error_msg)))
            # Envia None para limpar a tabela da GUI em caso de erro
            self.gui_queue.put(("prediction_complete", None))

        finally:
            # Finaliza barra de progresso e reabilita botões na GUI
            self.gui_queue.put(("progress_end", None))
            self.set_button_state(self.load_train_button, tk.NORMAL)
            # Reabilita botão de prever apenas se um modelo ainda estiver carregado/selecionado
            if self.trained_model and self.selected_model_id:
                self.set_button_state(self.predict_button, tk.NORMAL)

    # --- Carregamento Inicial de Modelos ---
    def load_existing_model_assets(self):
        """ Carrega modelos salvos (F1, ROI) e histórico ao iniciar a aba. """
        self.loaded_models_data = {}
        self.available_model_ids = []
        any_model_loaded = False
        default_selection = None

        model_paths_to_try = {
            MODEL_ID_F1: BEST_F1_MODEL_SAVE_PATH,
            MODEL_ID_ROI: BEST_ROI_MODEL_SAVE_PATH,
        }

        # Carrega os modelos
        for model_id, model_path in model_paths_to_try.items():
            self.log(f"Tentando carregar modelo: {model_id}...")
            # predictor.load_model_scaler_features já loga detalhes
            load_result = predictor.load_model_scaler_features(model_path)
            if load_result:
                model, scaler, features, params, metrics, timestamp = load_result
                if model and features:
                    self.log(f" -> Sucesso: Modelo '{model_id}' carregado.")
                    self.loaded_models_data[model_id] = {
                        'model': model, 'scaler': scaler, 'features': features,
                        'params': params, 'metrics': metrics, 'timestamp': timestamp,
                        'path': model_path
                    }
                    self.available_model_ids.append(model_id)
                    if default_selection is None: # Pega o primeiro carregado como default
                        default_selection = model_id
                    any_model_loaded = True
                else:
                    self.log(f" -> Aviso: Arquivo '{model_id}' inválido (sem modelo/features).")
            # else: predictor.load_model_scaler_features já logou 'Não encontrado'

        # Atualiza o Combobox de seleção de modelo
        try:
            if hasattr(self, 'model_selector_combo') and self.model_selector_combo.winfo_exists():
                 self.model_selector_combo.config(values=self.available_model_ids)
                 if self.available_model_ids:
                     self.selected_model_var.set(default_selection or self.available_model_ids[0]) # Define seleção
                     self.on_model_select() # Dispara atualização da GUI
                     self.log(f"Modelos disponíveis: {self.available_model_ids}. Padrão: {self.selected_model_var.get()}")
                 else:
                     self.selected_model_var.set("") # Limpa seleção
                     self.on_model_select() # Atualiza GUI (mostrará 'nenhum modelo')
                     self.log("Nenhum modelo pré-treinado válido encontrado.")
        except tk.TclError: pass # Ignora erro se widget não existe mais

        # Carrega o histórico (se ainda não carregado)
        if self.historical_data is None:
            self.log("Carregando dados históricos...")
            df_hist = load_historical_data(HISTORICAL_DATA_PATH)
            if df_hist is not None:
                self.historical_data = df_hist
                self.log("Dados históricos carregados.")
            else:
                self.log("Falha ao carregar dados históricos.")
                # Se histórico falhar, desabilita treino/previsão?
                self.set_button_state(self.predict_button, tk.DISABLED)
                # Pode desabilitar treino também ou deixar usuário tentar carregar de novo?
        else:
            self.log("Dados históricos já estavam carregados.")

        # Habilita botão de prever APENAS se modelo E histórico estiverem ok
        if self.selected_model_id and self.historical_data is not None:
            self.set_button_state(self.predict_button, tk.NORMAL)
            self.log("Pronto para previsão.")
        else:
            self.set_button_state(self.predict_button, tk.DISABLED)
            if not self.selected_model_id: self.log("Aguardando seleção de modelo.")
            if self.historical_data is None: self.log("Histórico não carregado.")


    # --- Processamento da Fila da GUI ---
    def process_gui_queue(self):
        """ Processa mensagens da fila para atualizar a GUI de forma segura (thread-safe). """
        try:
            while True: # Processa todas as mensagens na fila de uma vez
                try:
                    message = self.gui_queue.get_nowait()
                    msg_type, msg_payload = message # Desempacota
                except Empty:
                    break # Fila vazia, sai do loop while
                except (ValueError, TypeError) as e_unpack:
                    print(f"AVISO GUI: Erro desempacotar msg: {e_unpack} - Msg: {message}")
                    continue # Pula para próxima mensagem
                except Exception as e_get:
                    print(f"Erro get fila GUI: {e_get}")
                    continue # Pula para próxima mensagem

                # --- Trata diferentes tipos de mensagem ---
                try:
                    if msg_type == "log":
                        self._update_log_area(str(msg_payload)) # Garante string
                    elif msg_type == "button_state":
                        self._update_button_state(msg_payload)
                    elif msg_type == "display_predictions": # Mensagem antiga, usar "prediction_complete"
                        self._update_prediction_display(msg_payload)
                    elif msg_type == "update_stats_gui":
                        self._update_model_stats_display_gui()
                    elif msg_type == "error":
                        if isinstance(msg_payload, tuple) and len(msg_payload) == 2:
                             messagebox.showerror(msg_payload[0], msg_payload[1], parent=self.parent) # Usa self.parent
                    elif msg_type == "info":
                         if isinstance(msg_payload, tuple) and len(msg_payload) == 2:
                              messagebox.showinfo(msg_payload[0], msg_payload[1], parent=self.parent) # Usa self.parent
                    elif msg_type == "progress_start":
                         # Atualiza barra de progresso (inicio)
                         if hasattr(self, 'progress_bar') and self.progress_bar.winfo_exists():
                             max_val = msg_payload[0] if isinstance(msg_payload, tuple) and len(msg_payload) > 0 and isinstance(msg_payload[0], int) else 100
                             self.progress_bar.config(maximum=max_val, value=0)
                         if hasattr(self, 'progress_label') and self.progress_label.winfo_exists():
                             self.progress_label.config(text="Iniciando...")
                    elif msg_type == "progress_update":
                         # Atualiza barra de progresso (valor e texto)
                         if isinstance(msg_payload, tuple) and len(msg_payload) == 2:
                            value, status_text = msg_payload
                            if hasattr(self, 'progress_bar') and self.progress_bar.winfo_exists(): self.progress_bar['value'] = value
                            if hasattr(self, 'progress_label') and self.progress_label.winfo_exists(): self.progress_label.config(text=str(status_text))
                    elif msg_type == "progress_end":
                         # Reseta barra de progresso
                         if hasattr(self, 'progress_bar') and self.progress_bar.winfo_exists(): self.progress_bar['value'] = 0
                         if hasattr(self, 'progress_label') and self.progress_label.winfo_exists(): self.progress_label.config(text="Pronto.")
                    elif msg_type == "training_succeeded":
                        self.log("Treino Concluído. Recarregando modelos...")
                        self.load_existing_model_assets() # Recarrega para atualizar lista e stats
                        # Não precisa mais mostrar info aqui, load_existing_model_assets já loga
                    elif msg_type == "training_failed":
                         self.log("ERRO: Pipeline de treino falhou.")
                         # Limpa estado do modelo na GUI
                         self.selected_model_id = None
                         self.trained_model = None; self.trained_scaler = None; self.feature_columns = None;
                         self.model_best_params = None; self.model_eval_metrics = None; self.model_file_timestamp = None;
                         self.selected_model_var.set("")
                         try: self.model_selector_combo.config(values=[])
                         except tk.TclError: pass
                         self._update_model_stats_display_gui()
                         self.set_button_state(self.predict_button, tk.DISABLED)
                    elif msg_type == "prediction_complete":
                         df_preds = msg_payload
                         self.log("Recebidas previsões completas da thread.")
                         self._update_prediction_display(df_preds) # Atualiza a Treeview
                    
                    else:
                        self.log(f"AVISO GUI: Tipo de mensagem desconhecido recebido: {msg_type}")
                except tk.TclError:
                    # Ignora erros se o widget foi destruído enquanto a msg estava na fila
        
                    pass
                except Exception as e_proc_msg:
                    # Loga outros erros no processamento da mensagem
                    print(f"Erro ao processar mensagem GUI '{msg_type}': {e_proc_msg}")
                    traceback.print_exc()

        except Exception as e_queue_loop:
            # Erro crítico no próprio loop da fila
            print(f"Erro CRÍTICO no loop process_gui_queue: {e_queue_loop}")
            traceback.print_exc()
        finally:
            # Reagenda a verificação da fila se a janela principal ainda existir
            try:
                if self.main_tk_root and self.main_tk_root.winfo_exists():
                    self.main_tk_root.after(100, self.process_gui_queue) # Reagenda
            except Exception as e_reschedule:
                 print(f"Erro ao reagendar process_gui_queue: {e_reschedule}")

   