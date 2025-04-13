# --- src/main.py ---
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText
import sys
import os
import threading
from queue import Queue, Empty
import datetime
import pandas as pd
from typing import Optional, Dict, List, Any
import math
import numpy as np

# Adiciona diretórios ao path e importa módulos
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path: sys.path.append(SRC_DIR)
if BASE_DIR not in sys.path: sys.path.append(BASE_DIR)
try:
    # Importa configs (nomes de odds, fetch day, etc.)
    from config import (
        HISTORICAL_DATA_PATH, MODEL_SAVE_PATH, CLASS_NAMES,
        ODDS_COLS, OTHER_ODDS_NAMES, FIXTURE_FETCH_DAY, MODEL_TYPE_NAME,
        TEST_SIZE, RANDOM_STATE, ODDS_COLS as CONFIG_ODDS_COLS, # Usa alias para evitar conflito
        BEST_F1_MODEL_SAVE_PATH, BEST_ROI_MODEL_SAVE_PATH, MODEL_ID_F1, MODEL_ID_ROI, # Paths/IDs dos modelos
        FEATURE_COLUMNS # Importa a lista de features
    )
    # Importa funções PÚBLICAS necessárias
    from data_handler import (
         load_historical_data,
         preprocess_and_feature_engineer,
         fetch_and_process_fixtures,
         prepare_fixture_data,
         calculate_historical_intermediate # Importa função pública
    )
    # Renomeia função de treino para clareza
    from model_trainer import train_evaluate_and_save_best_models as run_training_process
    import predictor
    import requests
    from sklearn.model_selection import train_test_split # Importa aqui
except ImportError as import_error: # Renomeia variável da exceção
     print(f"Erro import main.py: {import_error}")
     try: root = tk.Tk(); root.withdraw(); messagebox.showerror("Erro Importação", f"Erro módulos necessários:\n{import_error}")
     except Exception: pass
     sys.exit(1)
except Exception as general_import_error: # Captura outros erros de importação
    print(f"Erro geral ao importar módulos em main.py: {general_import_error}")
    try: root = tk.Tk(); root.withdraw(); messagebox.showerror("Erro Fatal", f"Erro ao iniciar aplicação:\n{general_import_error}")
    except Exception: pass
    sys.exit(1)


class FootballPredictorDashboard:
    def __init__(self, root):
        self.root = root; self.root.title(f"Football Predictor Pro ({MODEL_TYPE_NAME})")
        self.root.geometry("950x750"); self.root.minsize(800, 600)
        self.gui_queue = Queue(); self.historical_data = None
        # Armazenamento dos modelos
        self.loaded_models_data: Dict[str, Dict] = {}; self.available_model_ids: List[str] = []
        # Estado do modelo selecionado
        self.selected_model_id: Optional[str] = None; self.trained_model: Optional[Any] = None; self.trained_scaler: Optional[Any] = None; self.feature_columns: Optional[List[str]] = None; self.model_best_params: Optional[Dict] = None; self.model_eval_metrics: Optional[Dict] = None; self.model_file_timestamp: Optional[str] = None

        # CHAMA create_widgets AQUI, as funções auxiliares precisam estar definidas ANTES
        self.create_widgets()

        self.root.after(100, self.process_gui_queue) # Inicia o loop da fila
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.log(f"Bem-vindo ({MODEL_TYPE_NAME})!"); self.log(f"Fonte CSV: '{FIXTURE_FETCH_DAY}'.")
        self.log(f"Carregando modelos e histórico..."); self.load_existing_model_assets()

    # --- FUNÇÕES DE ATUALIZAÇÃO DA GUI (Definidas ANTES de create_widgets) ---

    def _update_model_stats_display_gui(self):
        """Atualiza área de texto com stats do modelo SELECIONADO."""
        try:
            self.model_stats_text.config(state='normal'); self.model_stats_text.delete('1.0', tk.END)
            if self.trained_model is None or self.selected_model_id is None:
                stats_content = "Nenhum modelo selecionado ou carregado."
            else:
                model_data = self.loaded_models_data.get(self.selected_model_id, {})
                stats_content = f"Modelo: {self.selected_model_id}\n"
                stats_content += f"Arquivo: {os.path.basename(model_data.get('path', 'N/A'))}\n"
                stats_content += f"Modif.: {self.model_file_timestamp or 'N/A'}\n"
                stats_content += f"Tipo: {self.trained_model.__class__.__name__}\n---\n"
                if self.feature_columns: stats_content += f"Features ({len(self.feature_columns)}):\n - " + "\n - ".join(self.feature_columns) + "\n---\n"
                if self.model_best_params: stats_content += "Params:\n"; params_list = [f" - {k}: {v}" for k,v in self.model_best_params.items()]; stats_content += "\n".join(params_list) + "\n---\n"
                metrics = self.model_eval_metrics;
                if metrics: stats_content += "Métricas (Teste):\n"; acc = metrics.get('accuracy'); loss = metrics.get('log_loss'); auc = metrics.get('roc_auc'); prec_d = metrics.get('precision_draw'); rec_d = metrics.get('recall_draw'); f1_d = metrics.get('f1_score_draw'); profit = metrics.get('profit'); roi = metrics.get('roi'); n_bets = metrics.get('num_bets'); train_n = metrics.get('train_set_size', 'N/A'); test_n = metrics.get('test_set_size', 'N/A'); stats_content += f"- Acurácia: {acc:.4f}\n" if acc is not None else "- Acc: N/A\n"; stats_content += f"- Log Loss: {loss:.4f}\n" if loss is not None and not math.isnan(loss) else "- Log Loss: N/A\n"; stats_content += f"- ROC AUC: {auc:.4f}\n" if auc is not None else "- AUC: N/A\n"; stats_content += "- Métricas Empate:\n"; stats_content += f"  - Prec: {prec_d:.4f}\n" if prec_d is not None else "  - Prec: N/A\n"; stats_content += f"  - Rec:    {rec_d:.4f}\n" if rec_d is not None else "  - Rec: N/A\n"; stats_content += f"  - F1:  {f1_d:.4f}\n" if f1_d is not None else "  - F1: N/A\n---\n"; stats_content += "BackDraw:\n"; stats_content += f"- Nº Bets: {n_bets if n_bets is not None else 'N/A'}\n"; stats_content += f"- Profit: {profit:.2f} u\n" if profit is not None else "- Profit: N/A\n"; stats_content += f"- ROI: {roi:.2f} %\n" if roi is not None else "- ROI: N/A\n"
                else: stats_content += "Métricas: [yellow]Não disponíveis[/]\n"; self.log(f"AVISO: Métricas não encontradas para '{self.selected_model_id}'")
            self.model_stats_text.insert('1.0', stats_content); self.model_stats_text.config(state='disabled')
        except tk.TclError: pass; 
        except Exception as e: print(f"Erro update stats GUI: {e}")

    def _setup_prediction_columns(self, columns: List[str]):
         """Configura as colunas da Treeview, tratando o caso 'Status'."""
         self.prediction_tree['columns'] = columns; self.prediction_tree.delete(*self.prediction_tree.get_children());
         col_widths = {'Data': 75, 'Hora': 50, 'Liga': 150, 'Casa': 110, 'Fora': 110,'Odd H': 50, 'Odd D': 50, 'Odd A': 50, 'O2.5': 50, 'BTTS S': 55, 'P(Ñ Emp)': 70, 'P(Empate)': 70}
         if columns == ['Status']: self.prediction_tree.heading('Status', text='Status'); self.prediction_tree.column('Status', anchor=tk.W, width=500); return
         else:
             for col in columns: width = col_widths.get(col, 80); anchor = tk.CENTER if col not in ['Liga', 'Casa', 'Fora', 'Data', 'Hora'] else tk.W; self.prediction_tree.heading(col, text=col); self.prediction_tree.column(col, anchor=anchor, width=width, stretch=False)

    def _update_log_area(self, message: str):
        """Atualiza a área de log com uma nova mensagem."""
        try: self.log_area.config(state='normal'); ts = datetime.datetime.now().strftime("%H:%M:%S"); self.log_area.insert(tk.END, f"[{ts}] {message}\n"); self.log_area.config(state='disabled'); self.log_area.see(tk.END)
        except tk.TclError: pass

    def _update_button_state(self, button_state_tuple):
        """Aplica o estado a um botão."""
        button, state = button_state_tuple; 
        try: button.config(state=state); 
        except tk.TclError: pass

    def _update_prediction_display(self, df: Optional[pd.DataFrame]):
         """Atualiza Treeview com odds CSV e probs binárias."""
         self.log(f"--- DEBUG: Iniciando _update_prediction_display ---")
         if df is None or df.empty: # Combina as checagens
             self.log("DEBUG: DataFrame recebido VAZIO ou None.")
             try: # Limpa e mostra status
                 for item in self.prediction_tree.get_children(): self.prediction_tree.delete(item)
                 self._setup_prediction_columns(['Status'])
                 self.prediction_tree.insert('', tk.END, values=['Nenhuma previsão válida após filtros.'])
             except tk.TclError: pass # Ignora se janela fechada
             return

         # Se chegou aqui, df tem dados
         self.log(f"DEBUG: DataFrame recebido com Shape: {df.shape}")
         self.log(f"DEBUG: Amostra recebida:\n{df.head().to_string()}") # Mostra antes de formatar

         # Headers da Treeview
         display_headers = ['Data', 'Hora', 'Liga', 'Casa', 'Fora', 'Odd H', 'Odd D', 'Odd A', 'O2.5', 'BTTS S', 'P(Ñ Emp)', 'P(Empate)']
         self.log(f"DEBUG: Headers a configurar: {display_headers}")

         # Mapeamento header -> coluna interna
         header_to_col_map = {'Data': 'Date_Str', 'Hora': 'Time_Str', 'Liga': 'League', 'Casa': 'HomeTeam', 'Fora': 'AwayTeam','Odd H': CONFIG_ODDS_COLS['home'], 'Odd D': CONFIG_ODDS_COLS['draw'], 'Odd A': CONFIG_ODDS_COLS['away'],'O2.5': 'Odd_Over25_FT', 'BTTS S': 'Odd_BTTS_Yes','P(Ñ Emp)': f'Prob_{CLASS_NAMES[0]}', 'P(Empate)': f'Prob_{CLASS_NAMES[1]}'}
         # Colunas internas que *deveriam* existir no df recebido para popular os headers
         internal_cols_to_fetch = [header_to_col_map.get(h) for h in display_headers if header_to_col_map.get(h)]
         # Colunas que REALMENTE existem no df E que precisamos buscar
         valid_internal_cols = [c for c in internal_cols_to_fetch if c in df.columns]
         self.log(f"DEBUG: Colunas válidas encontradas no DF: {valid_internal_cols}")

         # Limpa e reconfigura a Treeview com os headers corretos
         try:
             for item in self.prediction_tree.get_children(): self.prediction_tree.delete(item)
             self._setup_prediction_columns(display_headers)
             self.log("DEBUG: Treeview limpa e reconfigurada.")
         except tk.TclError: return

         # Verifica se há colunas válidas para exibir
         if not valid_internal_cols:
              self.log("ERRO: Nenhuma coluna válida encontrada no DF após mapeamento!"); self._setup_prediction_columns(['Status']); self.prediction_tree.insert('', tk.END, values=['Erro: Colunas de dados não encontradas.']); return

         try:
             self.log(f"DEBUG: Selecionando colunas válidas para exibição: {valid_internal_cols}")
             df_display = df[valid_internal_cols].copy() # Usa apenas as colunas válidas

             # --- Formatação Robusta ---
             # Formata Probabilidades (se existirem)
             prob_cols = [f'Prob_{CLASS_NAMES[0]}', f'Prob_{CLASS_NAMES[1]}']
             for pcol in prob_cols:
                 if pcol in df_display.columns: # Checa se coluna existe ANTES de formatar
                     self.log(f"DEBUG: Formatando prob: {pcol}")
                     try: df_display[pcol] = (pd.to_numeric(df_display[pcol], errors='coerce') * 100).round(1).astype(str) + '%'; df_display[pcol] = df_display[pcol].replace('nan%', '-', regex=False)
                     except Exception as e_fmt_prob: self.log(f"AVISO: Erro formatar {pcol}: {e_fmt_prob}"); df_display[pcol] = "-"

             # Formata Odds (se existirem)

             odds_cols_to_format = [c for c in valid_internal_cols if c.startswith('Odd_')]
             self.log(f"DEBUG: Formatando odds: {odds_cols_to_format}")
             for ocol in odds_cols_to_format:
                  try: df_display[ocol] = pd.to_numeric(df_display[ocol], errors='coerce').apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
                  except Exception as e_fmt_odd: self.log(f"AVISO: Erro formatar {ocol}: {e_fmt_odd}"); df_display[ocol] = "-"
             # --- Fim Formatação ---

             self.log("DEBUG: Iniciando adição de linhas na Treeview...")
             added_rows = 0
             for index, row in df_display.iterrows():
                 # Monta a lista de valores na ordem dos HEADERS da Treeview
                 values_to_insert = []
                 for header in display_headers:
                      internal_col = header_to_col_map.get(header)
                      # Pega o valor da linha FORMATADA (df_display)
                      values_to_insert.append(str(row.get(internal_col, '')) if internal_col else '') # Garante string

                 # --- Try/Except por linha ---
                 try:
                      self.prediction_tree.insert('', tk.END, values=values_to_insert)
                      added_rows += 1
                 except Exception as e_insert:
                      self.log(f"!!!!! ERRO ao inserir linha {index}: {e_insert}")
                      self.log(f"      Valores problemáticos?: {values_to_insert}")

             self.log(f"DEBUG: {added_rows} de {len(df_display)} linhas adicionadas à Treeview.")
             if added_rows == 0 and not df_display.empty: self.log("ERRO: Nenhuma linha foi adicionada, verificar erros de inserção.")
             elif added_rows < len(df_display): self.log("AVISO: Nem todas as linhas foram adicionadas, verificar erros de inserção.")

         except Exception as e: 
              self.log(f"!!!!! ERRO GERAL _update_prediction_display: {e}"); import traceback; self.log(traceback.format_exc()); self._setup_prediction_columns(['Status']); self.prediction_tree.insert('', tk.END, values=['Erro exibir. Ver Logs.'])

    def create_widgets(self):
        """Cria todos os widgets da GUI."""
        style = ttk.Style(); style.theme_use('clam'); 
        main_frame = ttk.Frame(self.root, padding="10"); 
        main_frame.pack(fill=tk.BOTH, expand=True)
        left_panel = ttk.Frame(main_frame); 
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), pady=5, anchor='nw')
        control_frame = ttk.LabelFrame(left_panel, text=" Ações ", padding="10"); 
        control_frame.pack(pady=(0,5), fill=tk.X)
        self.load_train_button = ttk.Button(control_frame, text="Carregar Histórico e Treinar", command=self.start_training_thread, width=30); 
        self.load_train_button.pack(pady=5, fill=tk.X)
        predict_frame = ttk.Frame(control_frame); 
        predict_frame.pack(fill=tk.X, pady=5); 
        self.predict_button = ttk.Button(predict_frame, text=f"Prever ({FIXTURE_FETCH_DAY.capitalize()})", command=self.start_prediction_thread, width=18); 
        self.predict_button.pack(side=tk.LEFT, fill=tk.X, expand=False); 
        self.predict_button.config(state=tk.DISABLED)
        self.selected_model_var = tk.StringVar(); 
        self.model_selector_combo = ttk.Combobox(predict_frame, textvariable=self.selected_model_var, state="readonly", width=15); 
        self.model_selector_combo.pack(side=tk.RIGHT, padx=(5, 0), fill=tk.X, expand=True); 
        self.model_selector_combo.bind("<<ComboboxSelected>>", self.on_model_select)
        progress_frame = ttk.Frame(control_frame); progress_frame.pack(fill=tk.X, pady=(10, 0)); 
        self.progress_label = ttk.Label(progress_frame, text="Pronto."); 
        self.progress_label.pack(side=tk.LEFT, padx=(0, 5)); 
        self.progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=200, mode='determinate'); 
        self.progress_bar.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        stats_frame = ttk.LabelFrame(left_panel, text=" Status Modelo Selecionado ", padding="10"); 
        stats_frame.pack(pady=10, fill=tk.BOTH, expand=True);
        self.model_stats_text = ScrolledText(stats_frame, height=15, state='disabled', wrap=tk.WORD, font=("Consolas", 9), relief=tk.FLAT, bd=0); 
        self.model_stats_text.pack(fill=tk.BOTH, expand=True);
        self._update_model_stats_display_gui() # Chamada aqui
        right_panel = ttk.Frame(main_frame); 
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True); 
        results_frame = ttk.LabelFrame(right_panel, text=" Previsões ", padding="5");
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        cols = ['Data', 'Hora', 'Liga', 'Casa', 'Fora', 'Odd H', 'Odd D', 
                'Odd A', 'O2.5', 'BTTS S', 'P(Ñ Emp)', 'P(Empate)']
        self.prediction_tree = ttk.Treeview(results_frame, columns=cols, show='headings', height=10); 
        vsb = ttk.Scrollbar(results_frame, orient="vertical", command=self.prediction_tree.yview); 
        hsb = ttk.Scrollbar(results_frame, orient="horizontal", command=self.prediction_tree.xview); 
        self.prediction_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set); 
        vsb.pack(side='right', fill='y'); hsb.pack(side='bottom', fill='x'); 
        self.prediction_tree.pack(fill=tk.BOTH, expand=True);
        self._setup_prediction_columns(cols) # Chamada aqui
        log_frame = ttk.LabelFrame(right_panel, text=" Logs ", padding="5"); 
        log_frame.pack(fill=tk.BOTH, expand=True, side=tk.BOTTOM, pady=(5,0)); 
        self.log_area = ScrolledText(log_frame, height=8, state='disabled', wrap=tk.WORD, font=("Consolas", 9)); 
        self.log_area.pack(fill=tk.BOTH, expand=True)

    # --- Funções de Log, Botão, Display ---
    def log(self, message: str): self.gui_queue.put(("log", message))
    def set_button_state(self, button: ttk.Button, state: str): self.gui_queue.put(("button_state", (button, state)))

    # --- Callback e Atualização de Stats ---
    def on_model_select(self, event=None): 
        selected_id = self.selected_model_var.get(); self.log(f"Modelo selecionado: {selected_id}"); self._update_gui_for_selected_model(selected_id)
    def _update_gui_for_selected_model(self, selected_id: Optional[str]):
        """Atualiza a GUI com base no modelo selecionado."""
        if selected_id and selected_id in self.loaded_models_data:
            model_data = self.loaded_models_data[selected_id]
            self.log(f"Carregando dados internos para: {selected_id}")
            self.selected_model_id = selected_id
            self.trained_model = model_data.get('model')
            self.trained_scaler = model_data.get('scaler')
            self.feature_columns = model_data.get('features')
            self.model_best_params = model_data.get('params')
            self.model_eval_metrics = model_data.get('metrics')
            self.model_file_timestamp = model_data.get('timestamp')

            if self.trained_model and self.feature_columns:
                self._update_model_stats_display_gui()

            if self.historical_data is not None:
                self.set_button_state(self.predict_button, tk.NORMAL)
            else:
                self.log("Histórico ausente.")
                self.set_button_state(self.predict_button, tk.DISABLED)
        else:
            self.log(f"Erro: Dados incompletos ou ID inválido para '{selected_id}'.")
            self.selected_model_id = None
            self.trained_model = None
            self.trained_scaler = None
            self.feature_columns = None
            self.model_best_params = None
            self.model_eval_metrics = None
            self.model_file_timestamp = None
            self.set_button_state(self.predict_button, tk.DISABLED)
            self._update_model_stats_display_gui()

    def start_training_thread(self):
        """Inicia a thread de treinamento."""
        self.log("Carregando históricos...")
        self.loaded_models_data = {}
        self.available_model_ids = []
        self.selected_model_var.set('')
        self.model_selector_combo.config(values=[])
        self.trained_model = None
        self.trained_scaler = None
        self.feature_columns = None
        self.model_best_params = None
        self.model_eval_metrics = None
        self.model_file_timestamp = None
        self.gui_queue.put(("update_stats_gui", None))
        self.set_button_state(self.load_train_button, tk.DISABLED)
        self.set_button_state(self.predict_button, tk.DISABLED)
        self.gui_queue.put(("progress_start", (100)))
        self.gui_queue.put(("progress_update", (5, "Carregando Histórico...")))

        try:
            df_hist = load_historical_data(HISTORICAL_DATA_PATH)
            if df_hist is None:
                raise ValueError("Falha carregar histórico.")
            self.historical_data = df_hist
            self.log("Histórico carregado. Iniciando thread...")
            self.gui_queue.put(("progress_update", (20, "Iniciando Treino...")))
            train_thread = threading.Thread(
                target=self._run_training_pipeline,
                args=(self.historical_data,),
                daemon=True
            )
            train_thread.start()
        except Exception as e_load:
            error_msg = f"Erro Carregar Histórico: {e_load}"
            self.log(f"ERRO: {error_msg}")
            self.gui_queue.put(("error", ("Erro Carregamento", error_msg)))
            self.gui_queue.put(("progress_end", None))
            self.set_button_state(self.load_train_button, tk.NORMAL)

    def _run_training_pipeline(self, df_hist_raw: pd.DataFrame):
        """Executa o pipeline de treinamento."""
        training_successful = False
        try:
            self.gui_queue.put(("progress_update", (30, "Pré-processando...")))
            processed = preprocess_and_feature_engineer(df_hist_raw)
            if processed is None:
                raise ValueError("Falha pré-processamento.")
            X_processed, y_processed, features = processed
            feature_names_for_saving = features

            self.gui_queue.put(("progress_update", (50, "Preparando teste ROI...")))
            df_hist_interm = calculate_historical_intermediate(df_hist_raw)
            df_hist_aligned = df_hist_interm.loc[X_processed.index.union(y_processed.index)].copy()
            _, X_test_full_data, _, _ = train_test_split(
                df_hist_aligned, y_processed, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_processed
            )

            def training_progress_callback(cs, ms, st):
                prog = 60 + int((cs / ms) * 35) if ms > 0 else 95
                self.gui_queue.put(("progress_update", (prog, st)))

            self.log("Iniciando treino...")
            self.gui_queue.put(("progress_update", (60, "Treinando...")))
            success = run_training_process(
                X_processed, y_processed,
                X_test_with_odds=X_test_full_data,
                odd_draw_col_name=CONFIG_ODDS_COLS['draw'],
                progress_callback=training_progress_callback
            )
            self.gui_queue.put(("progress_update", (95, "Finalizando...")))

            if success:
                self.gui_queue.put(("training_succeeded", None))
                training_successful = True
            else:
                raise RuntimeError("Falha treino/seleção/salvamento.")
        except Exception as e:
            error_msg = f"Erro Treino Thread: {e}"
            self.log(f"ERRO: {error_msg}")
            self.gui_queue.put(("error", ("Erro no Treino", error_msg)))
            self.gui_queue.put(("training_failed", None))
        finally:
            self.gui_queue.put(("progress_end", None))
            self.set_button_state(self.load_train_button, tk.NORMAL)

    def start_prediction_thread(self):
        """Inicia a thread de previsão."""
        if self.trained_model is None or self.selected_model_id is None:
            messagebox.showwarning("Modelo Não Selecionado", "Selecione modelo.")
            return
        if self.historical_data is None:
            messagebox.showwarning("Histórico Ausente", "Carregue/Treine.")
            return

        self.log(f"Iniciando previsão com '{self.selected_model_id}'...")
        self.set_button_state(self.load_train_button, tk.DISABLED)
        self.set_button_state(self.predict_button, tk.DISABLED)
        self._setup_prediction_columns(['Status'])
        self.prediction_tree.insert('', tk.END, values=['Iniciando...'])

        predict_thread = threading.Thread(
            target=self._run_prediction_pipeline,
            daemon=True
        )
        predict_thread.start()

    def _run_prediction_pipeline(self):
        """Pipeline completo: busca CSV, prepara, prevê, FILTRA e exibe no log da GUI."""
        prediction_successful = False
        df_predictions_final_filtered = None
        try:
            self.gui_queue.put(("progress_start", (100))); self.gui_queue.put(("progress_update", (10, f"Buscando CSV...")))
            fixture_df = fetch_and_process_fixtures()
            if fixture_df is None: raise ValueError("Falha buscar/processar CSV.");
            if fixture_df.empty: self.log("Nenhum jogo no CSV."); self.gui_queue.put(("prediction_complete", None)); return # Envia None para limpar Treeview

            self.gui_queue.put(("progress_update", (40, f"Preparando features...")))
            if not self.feature_columns: raise ValueError("Features não carregadas.")
            X_fixtures_prepared = prepare_fixture_data(fixture_df, self.historical_data, self.feature_columns)
            if X_fixtures_prepared is None or X_fixtures_prepared.empty: raise ValueError("Falha preparar features.")

            self.gui_queue.put(("progress_update", (70, "Prevendo...")))
            df_predictions_raw = predictor.make_predictions(self.trained_model, self.trained_scaler, self.feature_columns, X_fixtures_prepared, fixture_df)
            if df_predictions_raw is None: raise RuntimeError("Falha gerar previsões.")
            self.log(f"Previsões brutas geradas: {len(df_predictions_raw)} jogos.")

            # --- Aplica Filtros ---
            df_to_filter = df_predictions_raw.copy(); self.log("Aplicando filtros...")
            # Filtro 1 (NaNs)
            input_odd_features = ['Odd_H_FT', 'Odd_D_FT', 'Odd_A_FT', 'Odd_Over25_FT', 'Odd_BTTS_Yes']; cols_to_check_nan = [c for c in input_odd_features if c in df_to_filter.columns]
            if cols_to_check_nan: initial_rows_f1 = len(df_to_filter); df_filtered_f1 = df_to_filter.dropna(subset=cols_to_check_nan); rows_dropped_f1 = initial_rows_f1 - len(df_filtered_f1);
            if rows_dropped_f1 > 0: self.log(f"F1: Removidos {rows_dropped_f1} (odds NaN)."); df_to_filter = df_filtered_f1
            # Filtro 2 (Probabilidade)
            prob_col_draw = f'Prob_{CLASS_NAMES[1]}'; df_predictions_final_filtered = df_to_filter # Default se coluna prob não existe
            if prob_col_draw in df_to_filter.columns:
                threshold = 0.5 # <<< SEU LIMIAR AQUI (ex: 0.5 para >50%)
                self.log(f"Aplicando Filtro 2: Manter P(Empate) > {threshold*100:.0f}%")
                initial_rows_f2 = len(df_to_filter)
                df_to_filter[prob_col_draw] = pd.to_numeric(df_to_filter[prob_col_draw], errors='coerce')
                df_filtered_f2 = df_to_filter[df_to_filter[prob_col_draw] > threshold].copy() # Aplica filtro e copia
                rows_kept_f2 = len(df_filtered_f2); self.log(f"Filtro 2: {rows_kept_f2} de {initial_rows_f2} jogos passaram.")
                df_predictions_final_filtered = df_filtered_f2
            else: self.log(f"Aviso: Coluna '{prob_col_draw}' não encontrada p/ filtro.")
            # --- FIM DOS FILTROS ---
            
            # --- PREPARA E ENVIA DADOS PARA EXIBIÇÃO ---
            self.gui_queue.put(("progress_update", (95, "Preparando Exibição...")))
            if df_predictions_final_filtered is not None and not df_predictions_final_filtered.empty:
                # --- ORDENAÇÃO AQUI ---
                if prob_col_draw in df_predictions_final_filtered.columns:
                    self.log(f"Ordenando resultados por '{prob_col_draw}' descendente...")
                    df_predictions_final_filtered = df_predictions_final_filtered.sort_values(by=prob_col_draw, ascending=False).reset_index(drop=True)
                
                self.log(f"--- {len(df_predictions_final_filtered)} JOGOS FILTRADOS (P(Empate) > {threshold*100:.0f}%) e Filtrados ---") # Log ANTES de enviar p/ GUI Log

                # Seleciona e reordena colunas para exibição textual
                cols_to_show = ['Date_Str', 'Time_Str', 'League', 'HomeTeam', 'AwayTeam',
                                'Odd_H_FT', 'Odd_D_FT', 'Odd_A_FT',
                                f'Prob_{CLASS_NAMES[1]}'] # Apenas P(Empate)
                # Filtra pelas colunas que realmente existem
                cols_to_show_exist = [c for c in cols_to_show if c in df_predictions_final_filtered.columns]
                df_show_text = df_predictions_final_filtered[cols_to_show_exist].copy()

                # Formata Probabilidade para % (arredondado)
                prob_col_draw_show = f'Prob_{CLASS_NAMES[1]}'
                if prob_col_draw_show in df_show_text.columns:
                     df_show_text[prob_col_draw_show] = (df_show_text[prob_col_draw_show] * 100).round(1)

                # Converte DataFrame para string formatada
                try:
                    df_string = df_show_text.to_string(index=False, float_format="%.2f")
                    self.log(f"\n{df_string}\n--------------------") # Envia string formatada para o LOG da GUI
                except Exception as e_tostr:
                    self.log(f"Erro ao formatar DataFrame para log: {e_tostr}")

                prediction_successful = True
            else:
                 self.log("Nenhuma previsão restante após aplicar os filtros.")
                 prediction_successful = False

            # Envia o DataFrame (filtrado ou vazio) para a Treeview tentar exibir
            self.gui_queue.put(("prediction_complete", df_predictions_final_filtered))

        except Exception as e: # ... (tratamento de erro como antes) ...
             error_msg = f"Erro Previsão Thread: {e}"; self.log(f"ERRO: {error_msg}"); self.gui_queue.put(("error", ("Erro na Previsão", error_msg))); self.gui_queue.put(("prediction_complete", None))
        finally: # ... (reabilita botões como antes) ...
             self.gui_queue.put(("progress_end", None)); self.set_button_state(self.load_train_button, tk.NORMAL);
             if self.trained_model: self.set_button_state(self.predict_button, tk.NORMAL)

    def load_existing_model_assets(self):
        """Carrega os modelos existentes."""
        self.loaded_models_data = {}
        self.available_model_ids = []
        any_model_loaded = False
        default_selection = None
        model_paths_to_try = {
            MODEL_ID_F1: BEST_F1_MODEL_SAVE_PATH,
            MODEL_ID_ROI: BEST_ROI_MODEL_SAVE_PATH,
        }

        for model_id, model_path in model_paths_to_try.items():
            self.log(f"Tentando carregar {model_id}...")
            load_result = predictor.load_model_scaler_features(model_path)
            if load_result:
                model, scaler, features, params, metrics, timestamp = load_result
                if model and features:
                    self.log(f" -> Sucesso: {model_id}.")
                    self.loaded_models_data[model_id] = {
                        'model': model,
                        'scaler': scaler,
                        'features': features,
                        'params': params,
                        'metrics': metrics,
                        'timestamp': timestamp,
                        'path': model_path
                    }
                    self.available_model_ids.append(model_id)
                    any_model_loaded = True
                    if default_selection is None:
                        default_selection = model_id
                else:
                    self.log(f" -> Falha: Inválido.")
            else:
                self.log(f" -> Falha: Não encontrado.")

        self.model_selector_combo.config(values=self.available_model_ids)
        if self.available_model_ids:
            self.selected_model_var.set(default_selection)
            self.on_model_select()
            self.log(f"Modelos: {self.available_model_ids}. Selecionado: {default_selection}")
        else:
            self.log("Nenhum modelo válido.")
            self.selected_model_var.set("")
            self.on_model_select()

        self.log("Carregando histórico...")
        df_hist = load_historical_data(HISTORICAL_DATA_PATH)
        if df_hist is not None:
            self.historical_data = df_hist
            self.log("Histórico OK.")
        else:
            self.log("Falha histórico.")
            any_model_loaded = False

        if self.selected_model_id and self.historical_data is not None:
            self.set_button_state(self.predict_button, tk.NORMAL)
            self.log("Pronto.")
        else:
            self.set_button_state(self.predict_button, tk.DISABLED)
    def process_gui_queue(self): 
        try:
            while True:
                try: message = self.gui_queue.get_nowait(); msg_type, msg_payload = message
                except Empty: break
                except (ValueError, TypeError) as e_unpack: self.log(f"AVISO GUI: Erro desempacotar: {e_unpack} - Msg: {message}"); continue
                except Exception as e_get: print(f"Erro get fila: {e_get}"); continue
                if msg_type == "log": self._update_log_area(msg_payload); 
                elif msg_type == "button_state": self._update_button_state(msg_payload); 
                elif msg_type == "display_predictions": self._update_prediction_display(msg_payload); 
                elif msg_type == "update_stats_gui": self._update_model_stats_display_gui()
                elif msg_type == "error":
                    if isinstance(msg_payload, tuple) and len(msg_payload) == 2: messagebox.showerror(*msg_payload)
                elif msg_type == "info":
                     if isinstance(msg_payload, tuple) and len(msg_payload) == 2: messagebox.showinfo(*msg_payload)
                elif msg_type == "progress_start":
                    max_val = msg_payload[0] if isinstance(msg_payload, tuple) and len(msg_payload) > 0 and isinstance(msg_payload[0], int) else 100; self.progress_bar.config(maximum=max_val, value=0); self.progress_label.config(text="Iniciando...")
                elif msg_type == "progress_update":
                    if isinstance(msg_payload, tuple) and len(msg_payload) == 2: value, status_text = msg_payload; self.progress_bar['value'] = value; self.progress_label.config(text=status_text)
                elif msg_type == "progress_end":
                    self.progress_bar['value'] = 0; self.progress_label.config(text="Pronto.")
                elif msg_type == "training_succeeded":
                    self.log("Treino OK. Recarregando..."); self.load_existing_model_assets(); self.gui_queue.put(("info", ("Treino Concluído", "Modelos salvos.")))
                elif msg_type == "training_failed":
                     self.trained_model = None; self.trained_scaler = None; self.feature_columns = None; self.model_best_params = None; self.model_eval_metrics = None; self.model_file_timestamp = None; self.log("Pipeline treino falhou.")
                     self._update_model_stats_display_gui(); self.set_button_state(self.predict_button, tk.DISABLED); self.set_button_state(self.load_train_button, tk.NORMAL); self._setup_prediction_columns(['Status']); self.prediction_tree.insert('', tk.END, values=['Falha Treinamento.'])
                elif msg_type == "prediction_complete":
                     df_preds = msg_payload; self.log("Processando resultado previsão...")
                     self._update_prediction_display(df_preds)
        except Exception as e_queue: print(f"Erro fatal fila GUI: {type(e_queue).__name__} - {e_queue}"); import traceback; traceback.print_exc()
        finally:
             if self.root.winfo_exists(): self.root.after(100, self.process_gui_queue)

    def on_closing(self): self.log("Fechando..."); self.root.destroy()

if __name__ == "__main__": 
    import requests; import math; import numpy as np; from sklearn.model_selection import train_test_split
    try: from ctypes import windll; windll.shcore.SetProcessDpiAwareness(1)
    except: pass
    root = tk.Tk()
    app = FootballPredictorDashboard(root)
    root.mainloop()