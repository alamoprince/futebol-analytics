import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText
import sys, os, threading, datetime, math, numpy as np
from queue import Queue, Empty
import pandas as pd
from typing import Optional, Dict, List, Any
from config import ( HISTORICAL_DATA_PATH, MODEL_SAVE_PATH, CLASS_NAMES, ODDS_COLS, OTHER_ODDS_NAMES, FIXTURE_FETCH_DAY, MODEL_TYPE_NAME, TEST_SIZE, RANDOM_STATE, ODDS_COLS as CONFIG_ODDS_COLS )
from data_handler import (load_historical_data, preprocess_and_feature_engineer, fetch_and_process_fixtures, prepare_fixture_data, calculate_historical_intermediate)
import model_trainer, predictor, requests
from sklearn.model_selection import train_test_split

SRC_DIR = os.path.dirname(os.path.abspath(__file__)); BASE_DIR = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path: sys.path.append(SRC_DIR)
if BASE_DIR not in sys.path: sys.path.append(BASE_DIR)

class FootballPredictorDashboard:
    
    def __init__(self, root): 
        self.root = root; self.root.title(f"Football Predictor Pro ({MODEL_TYPE_NAME})")
        self.root.geometry("950x750"); 
        self.root.minsize(800, 600)
        self.gui_queue = Queue(); 
        self.historical_data = None; 
        self.trained_model = None; 
        self.trained_scaler = None; 
        self.feature_columns = None; 
        self.model_best_params = None; 
        self.model_eval_metrics = None; 
        self.model_file_timestamp = None; 
        self.create_widgets()

        self.root.after(100, self.process_gui_queue); 
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing); 
        self.log(f"Bem-vindo ({MODEL_TYPE_NAME})!"); 
        self.log(f"Fonte CSV: '{FIXTURE_FETCH_DAY}'."); 
        self.log(f"Carregando modelo e histórico..."); 
        self.load_existing_model_assets()

    def create_widgets(self): 
         style = ttk.Style(); 
         style.theme_use('clam');
         main_frame = ttk.Frame(self.root, padding="10"); 
         main_frame.pack(fill=tk.BOTH, expand=True)
         left_panel = ttk.Frame(main_frame); 
         left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), pady=5, anchor='nw'); 
         control_frame = ttk.LabelFrame(left_panel, text=" Ações ", padding="10"); 
         control_frame.pack(pady=(0,5), fill=tk.X)
         self.load_train_button = ttk.Button(control_frame, text="Carregar Histórico e Treinar", command=self.start_training_thread, width=30); 
         self.load_train_button.pack(pady=5, fill=tk.X)
         self.predict_button = ttk.Button(control_frame, text=f"Prever Jogos ({FIXTURE_FETCH_DAY.capitalize()})", command=self.start_prediction_thread, width=30); 
         self.predict_button.pack(pady=5, fill=tk.X); 
         self.predict_button.config(state=tk.DISABLED)
         progress_frame = ttk.Frame(control_frame); 
         progress_frame.pack(fill=tk.X, pady=(10, 0)); 
         self.progress_label = ttk.Label(progress_frame, text="Pronto."); 
         self.progress_label.pack(side=tk.LEFT, padx=(0, 5)); 
         self.progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=200, mode='determinate'); 
         self.progress_bar.pack(side=tk.RIGHT, fill=tk.X, expand=True)
         stats_frame = ttk.LabelFrame(left_panel, text=" Status Modelo ", padding="10"); 
         stats_frame.pack(pady=10, fill=tk.BOTH, expand=True); self.model_stats_text = ScrolledText(stats_frame, height=15, state='disabled', wrap=tk.WORD, font=("Consolas", 9), relief=tk.FLAT, bd=0); 
         self.model_stats_text.pack(fill=tk.BOTH, expand=True); self._update_model_stats_display_gui()
         right_panel = ttk.Frame(main_frame); 
         right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True); results_frame = ttk.LabelFrame(right_panel, text=" Previsões ", padding="5"); 
         results_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
         cols = ['Data', 'Hora', 'Liga', 'Casa', 'Fora', 'Odd H', 'Odd D', 'Odd A', 'O2.5', 'BTTS S', 'P(Ñ Emp)', 'P(Empate)']; 
         self.prediction_tree = ttk.Treeview(results_frame, columns=cols, show='headings', height=10); 
         vsb = ttk.Scrollbar(results_frame, orient="vertical", command=self.prediction_tree.yview); 
         hsb = ttk.Scrollbar(results_frame, orient="horizontal", command=self.prediction_tree.xview); 
         self.prediction_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set); 
         vsb.pack(side='right', fill='y'); hsb.pack(side='bottom', fill='x'); 
         self.prediction_tree.pack(fill=tk.BOTH, expand=True); 
         self._setup_prediction_columns(cols)
         log_frame = ttk.LabelFrame(right_panel, text=" Logs ", padding="5"); 
         log_frame.pack(fill=tk.BOTH, expand=True, side=tk.BOTTOM, pady=(5,0)); 
         self.log_area = ScrolledText(log_frame, height=8, state='disabled', wrap=tk.WORD, font=("Consolas", 9)); 
         self.log_area.pack(fill=tk.BOTH, expand=True)

    def _setup_prediction_columns(self, columns: List[str]): 
         self.prediction_tree['columns'] = columns; 
         self.prediction_tree.delete(*self.prediction_tree.get_children()); 
         col_widths = {'Data': 75, 'Hora': 50, 'Liga': 150, 'Casa': 110, 'Fora': 110,'Odd H': 50, 'Odd D': 50, 
                       'Odd A': 50, 'O2.5': 50, 'BTTS S': 55, 'P(Ñ Emp)': 70, 'P(Empate)': 70}
         for col in columns:
             if col == 'Status': 
                self.prediction_tree.heading('Status', text='Status'); 
                self.prediction_tree.column('Status', anchor=tk.W, width=500); continue 
             width = col_widths.get(col, 80); anchor = tk.CENTER if col not in ['Liga', 'Casa', 'Fora', 'Data', 'Hora'] else tk.W; 
             self.prediction_tree.heading(col, text=col); 
             self.prediction_tree.column(col, anchor=anchor, width=width, stretch=False)
             
    def log(self, message: str): 
        self.gui_queue.put(("log", message))
    def _update_log_area(self, message: str): 
        try: 
            self.log_area.config(state='normal'); 
            ts = datetime.datetime.now().strftime("%H:%M:%S"); 
            self.log_area.insert(tk.END, f"[{ts}] {message}\n"); 
            self.log_area.config(state='disabled'); 
            self.log_area.see(tk.END)

        except tk.TclError: pass

    def set_button_state(self, button: ttk.Button, state: str): 
        self.gui_queue.put(("button_state", (button, state)))
    def _update_button_state(self, button_state_tuple): 
        button, state = button_state_tuple; 
        try: 
            button.config(state=state); 
        except tk.TclError: pass
    def display_predictions(self, df_predictions: Optional[pd.DataFrame]): 
        self.gui_queue.put(("display_predictions", df_predictions))
    def _update_prediction_display(self, df: Optional[pd.DataFrame]):
         """Atualiza Treeview com odds CSV e probs binárias."""
         self.log(f"--- DEBUG: Iniciando _update_prediction_display ---")
         if df is None: self.log("DEBUG: DataFrame recebido é None.")
         elif df.empty: self.log("DEBUG: DataFrame recebido está VAZIO.")
         else: self.log(f"DEBUG: DataFrame recebido com Shape: {df.shape}"); 

         # Limpa a Treeview PRIMEIRO
         try:
             for item in self.prediction_tree.get_children(): self.prediction_tree.delete(item)
             self.log("DEBUG: Treeview limpa.")
         except tk.TclError: return

         # Se DF vazio ou None, configura Status e sai
         if df is None or df.empty:
              self.log("DEBUG: Configurando Treeview para 'Status' (DF Vazio/None).")
              self._setup_prediction_columns(['Status']) # Configura APENAS Status
              self.prediction_tree.insert('', tk.END, values=['Nenhuma previsão gerada ou dados inválidos.'])
              return

         # 1. DEFINE os headers corretos que QUEREMOS exibir
         display_headers = ['Data', 'Hora', 'Liga', 'Casa', 'Fora', 'Odd H', 'Odd D', 'Odd A', 'O2.5', 'BTTS S', 'P(Ñ Emp)', 'P(Empate)']
         self.log(f"DEBUG: Headers que serão configurados: {display_headers}")

         # 2. RECONFIGURA a Treeview com esses headers AGORA
         self._setup_prediction_columns(display_headers)
         self.log("DEBUG: Treeview reconfigurada com headers de dados.")

         # 3. AGORA mapeia headers para colunas internas
         header_to_col_map = { # Recalcula o mapa aqui para clareza
             'Data': 'Date_Str', 'Hora': 'Time_Str', 'Liga': 'League', 'Casa': 'HomeTeam', 'Fora': 'AwayTeam',
             'Odd H': ODDS_COLS['home'], 'Odd D': ODDS_COLS['draw'], 'Odd A': ODDS_COLS['away'],
             'O2.5': 'Odd_Over25_FT', 'BTTS S': 'Odd_BTTS_Yes',
             'P(Ñ Emp)': f'Prob_{CLASS_NAMES[0]}', 'P(Empate)': f'Prob_{CLASS_NAMES[1]}'
         }
         internal_cols_to_fetch = [header_to_col_map.get(h) for h in display_headers if header_to_col_map.get(h)]
         self.log(f"DEBUG: Colunas internas que tentaremos buscar: {internal_cols_to_fetch}")

         # 4. Encontra colunas válidas NO DATAFRAME RECEBIDO (df)
         valid_internal_cols = [c for c in internal_cols_to_fetch if c in df.columns] # Checa em df!
         self.log(f"DEBUG: Colunas internas VÁLIDAS encontradas no DataFrame 'df': {valid_internal_cols}")

         # 5. Continua com a formatação e exibição
         try:
             if not valid_internal_cols: # Checa se HÁ colunas válidas
                  self.log("ERRO: Nenhuma coluna válida encontrada no DF para exibir!")
                  self._setup_prediction_columns(['Status'])
                  self.prediction_tree.insert('', tk.END, values=['Erro: Colunas de dados não encontradas no resultado.'])
                  return

             self.log(f"DEBUG: Selecionando colunas válidas para exibição: {valid_internal_cols}")
             df_display = df[valid_internal_cols].copy()
             prob_cols = [f'Prob_{CLASS_NAMES[0]}', f'Prob_{CLASS_NAMES[1]}']; 
             odds_cols_internal = list(ODDS_COLS.values()) + OTHER_ODDS_NAMES
             for pcol in prob_cols:
                 if pcol in df_display: df_display[pcol] = (pd.to_numeric(df_display[pcol], errors='coerce') * 100).round(1).astype(str) + '%'; df_display[pcol] = df_display[pcol].replace('nan%', '-', regex=False)
             for ocol in odds_cols_internal:
                  if ocol in df_display: df_display[ocol] = pd.to_numeric(df_display[ocol], errors='coerce').apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
             self.log("DEBUG: Iniciando adição de linhas..."); added_rows = 0
             for index, row in df_display.iterrows():
                 values_to_insert = [str(row.get(header_to_col_map.get(h), '')) for h in display_headers];
                 try: self.prediction_tree.insert('', tk.END, values=values_to_insert); added_rows += 1
                 except Exception as e_insert: self.log(f"ERRO inserir linha {index}: {e_insert}")
             self.log(f"DEBUG: {added_rows} linhas adicionadas.")
             if added_rows == 0 and not df.empty: self.log("AVISO: Nenhuma linha adicionada, mas DF não vazio.")

         except Exception as e: 
              self.log(f"!!!!! ERRO GERAL _update_prediction_display: {e}"); import traceback; self.log(traceback.format_exc()); self._setup_prediction_columns(['Status']); self.prediction_tree.insert('', tk.END, values=['Erro exibir. Ver Logs.'])

    def _update_model_stats_display_gui(self): 
        try:
            self.model_stats_text.config(state='normal'); self.model_stats_text.delete('1.0', tk.END)
            if self.trained_model is None: stats_content = "Nenhum modelo treinado."
            else: 
                stats_content = f"Arquivo: {os.path.basename(MODEL_SAVE_PATH)}\nModif.: {self.model_file_timestamp or 'N/A'}\nTipo: {self.trained_model.__class__.__name__} ({MODEL_TYPE_NAME})\n---\n";
                if self.feature_columns: stats_content += f"Features ({len(self.feature_columns)}):\n - " + "\n - ".join(self.feature_columns) + "\n---\n"
                if self.model_best_params: stats_content += "Melhores Parâmetros:\n"; 
                params_list = [f" - {k}: {v}" for k,v in self.model_best_params.items()]; 
                stats_content += "\n".join(params_list) + "\n---\n"
                if self.model_eval_metrics: 
                    stats_content += "Métricas Avaliação (Teste):\n";
                    acc = self.model_eval_metrics.get('accuracy'); 
                    loss = self.model_eval_metrics.get('log_loss'); 
                    auc = self.model_eval_metrics.get('roc_auc'); 
                    prec_d = self.model_eval_metrics.get('precision_draw'); 
                    rec_d = self.model_eval_metrics.get('recall_draw'); 
                    f1_d = self.model_eval_metrics.get('f1_score_draw'); 
                    profit = self.model_eval_metrics.get('profit'); 
                    roi = self.model_eval_metrics.get('roi'); 
                    n_bets = self.model_eval_metrics.get('num_bets'); 
                    stats_content += f"- Acurácia: {acc:.4f}\n" if acc is not None else "- Acurácia: N/A\n"; 
                    stats_content += f"- Log Loss: {loss:.4f}\n" if loss is not None and not math.isnan(loss) else "- Log Loss: N/A\n"; 
                    stats_content += f"- ROC AUC: {auc:.4f}\n" if auc is not None else "- ROC AUC: N/A\n"; 
                    stats_content += "- Métricas 'Empate':\n"; 
                    stats_content += f"  - Precision: {prec_d:.4f}\n" if prec_d is not None else "  - Precision: N/A\n"; 
                    stats_content += f"  - Recall:    {rec_d:.4f}\n" if rec_d is not None else "  - Recall: N/A\n"; 
                    stats_content += f"  - F1-Score:  {f1_d:.4f}\n" if f1_d is not None else "  - F1-Score: N/A\n---\n"; 
                    stats_content += "Estratégia BackDraw:\n"; stats_content += f"- Nº Apostas: {n_bets if n_bets is not None else 'N/A'}\n"; 
                    stats_content += f"- Profit: {profit:.2f} u\n" if profit is not None else "- Profit: N/A\n"; 
                    stats_content += f"- ROI: {roi:.2f} %\n" if roi is not None else "- ROI: N/A\n"
                else: stats_content += "Métricas Avaliação: Não disponíveis.\n"
            self.model_stats_text.insert('1.0', stats_content); 
            self.model_stats_text.config(state='disabled')
        except tk.TclError: pass; 
        except Exception as e_stats_disp: 
            print(f"Erro update stats GUI: {e_stats_disp}")


    # --- Threads de Ação ---
    def start_training_thread(self):
        self.log("Carregando históricos..."); 
        self.trained_model = None; 
        self.trained_scaler = None; 
        self.feature_columns = None; 
        self.model_best_params = None; 
        self.model_eval_metrics = None; 
        self.model_file_timestamp = None; 
        self.gui_queue.put(("update_stats_gui", None)); 
        self.set_button_state(self.load_train_button, tk.DISABLED); 
        self.set_button_state(self.predict_button, tk.DISABLED)
        self.gui_queue.put(("progress_start", (100))) 
        self.gui_queue.put(("progress_update", (5, "Carregando Histórico..."))) 

        try:
            df_hist = load_historical_data(HISTORICAL_DATA_PATH)
            if df_hist is None: raise ValueError("Falha carregar histórico.")
            self.historical_data = df_hist; self.log("Histórico carregado. Iniciando thread...")
            self.gui_queue.put(("progress_update", (20, "Iniciando Treino...")))
            train_thread = threading.Thread(target=self._run_training_pipeline, args=(self.historical_data,), daemon=True); train_thread.start()
        except Exception as e_load: # ... (tratamento erro load como V13) ...
             error_msg = f"Erro Carregar Histórico: {e_load}"; 
             self.log(f"ERRO: {error_msg}"); 
             self.gui_queue.put(("error", ("Erro Carregamento", error_msg))); 
             self.gui_queue.put(("progress_end", None)); 
             self.set_button_state(self.load_train_button, tk.NORMAL)

    def _run_training_pipeline(self, df_hist_raw: pd.DataFrame):
        model_trained_successfully = False
        try:
           
            self.gui_queue.put(("progress_update", (30, "Pré-processando...")))
            processed = preprocess_and_feature_engineer(df_hist_raw)
            if processed is None: raise ValueError("Falha pré-processamento.")
            X_processed, y_processed, features = processed; feature_names_for_saving = features
            self.gui_queue.put(("progress_update", (50, "Preparando teste ROI...")))
            df_hist_interm = calculate_historical_intermediate(df_hist_raw)
            df_hist_aligned = df_hist_interm.loc[X_processed.index.union(y_processed.index)].copy(); _, X_test_full_data, _, _ = train_test_split(df_hist_aligned, y_processed, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_processed)

            def training_progress_callback(current_step, max_steps, status_text):
                progress_percent = 60 + int((current_step / max_steps) * 35) if max_steps > 0 else 95
               
                self.gui_queue.put(("progress_update", (progress_percent, status_text)))
            self.log("Iniciando treino..."); self.gui_queue.put(("progress_update", (60, "Treinando modelos...")))
            train_result = model_trainer.train_evaluate_model(X_processed, y_processed, X_test_with_odds=X_test_full_data, odd_draw_col_name=CONFIG_ODDS_COLS['draw'], progress_callback=training_progress_callback )
            self.gui_queue.put(("progress_update", (95, "Finalizando...")))
            if train_result: model, scaler, _, params, metrics = train_result; model_trainer.save_model_scaler_features(model, scaler, feature_names_for_saving, params, metrics, MODEL_SAVE_PATH); timestamp_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'); payload = (model, scaler, feature_names_for_saving, params, metrics, timestamp_str); self.gui_queue.put(("training_complete", payload)); model_trained_successfully = True
            else: raise RuntimeError("Falha treino (trainer None).")
        except Exception as e: 
            error_msg = f"Erro Treino Thread: {e}"; self.log(f"ERRO: {error_msg}"); self.gui_queue.put(("error", ("Erro no Treino", error_msg))); self.gui_queue.put(("training_failed", None))
        finally:

             self.gui_queue.put(("progress_end", None)) # Payload é None ou irrelevante

             self.set_button_state(self.load_train_button, tk.NORMAL);
             if model_trained_successfully: self.set_button_state(self.predict_button, tk.NORMAL) # Só habilita se sucesso

    def _run_prediction_pipeline(self):
        prediction_successful = False
        try:
             # ---progress_start e update ---
            self.gui_queue.put(("progress_start", (100)))
            self.gui_queue.put(("progress_update", (10, f"Buscando CSV ({FIXTURE_FETCH_DAY})...")))
            fixture_df = fetch_and_process_fixtures()
            if fixture_df is None: raise ValueError("Falha buscar CSV.");
            if fixture_df.empty: 
                self.log("Nenhum jogo no CSV."); 
                self.gui_queue.put(("prediction_complete", None)); return
            self.gui_queue.put(("progress_update", (40, f"Preparando features ({len(fixture_df)} jogos)...")))
            X_fixtures_prepared = prepare_fixture_data(fixture_df, self.historical_data, self.feature_columns)
            if X_fixtures_prepared is None or X_fixtures_prepared.empty: raise ValueError("Falha preparar features.")
            self.gui_queue.put(("progress_update", (70, "Realizando previsões...")))
            df_predictions = predictor.make_predictions(self.trained_model, self.trained_scaler, self.feature_columns, X_fixtures_prepared, fixture_df)
            if df_predictions is None: raise RuntimeError("Falha gerar previsões.")
            self.gui_queue.put(("progress_update", (95, "Finalizando...")))
            
            self.gui_queue.put(("prediction_complete", df_predictions)); prediction_successful = True
        except Exception as e: 
            error_msg = f"Erro Previsão Thread: {e}"; self.log(f"ERRO: {error_msg}"); self.gui_queue.put(("error", ("Erro na Previsão", error_msg))); self.gui_queue.put(("prediction_complete", None))
        finally:

             self.gui_queue.put(("progress_end", None))

             self.set_button_state(self.load_train_button, tk.NORMAL);
             if self.trained_model: self.set_button_state(self.predict_button, tk.NORMAL)

    # --- start_prediction_thread  ---
    def start_prediction_thread(self): 
        if self.trained_model is None: messagebox.showwarning("Modelo Ausente", "Treine/Carregue."); return
        if self.historical_data is None: messagebox.showwarning("Histórico Ausente", "Carregue/Treine."); return
        self.log(f"Iniciando busca CSV ({FIXTURE_FETCH_DAY}) e previsão..."); 
        self.set_button_state(self.load_train_button, tk.DISABLED); 
        self.set_button_state(self.predict_button, tk.DISABLED); 
        self._setup_prediction_columns(['Status']); 
        self.prediction_tree.insert('', tk.END, values=['Iniciando...'])
        predict_thread = threading.Thread(target=self._run_prediction_pipeline, daemon=True); 
        predict_thread.start()

    # --- load_existing_model_assets  ---
    def load_existing_model_assets(self): 
        self.trained_model = None; 
        self.trained_scaler = None; 
        self.feature_columns = None; 
        self.model_best_params = None; 
        self.model_eval_metrics = None; 
        self.model_file_timestamp = None;
        model_loaded_success = False
        load_result = predictor.load_model_scaler_features(MODEL_SAVE_PATH);
        if load_result: model, scaler, features, params, metrics, timestamp = load_result; # Desempacota 6
        if model and features: 
            self.trained_model = model; 
            self.trained_scaler = scaler; 
            self.feature_columns = features; 
            self.model_best_params = params; 
            self.model_eval_metrics = metrics; 
            self.model_file_timestamp = timestamp; 
            self.log(f"Modelo {MODEL_TYPE_NAME} e stats carregados.");
            model_loaded_success = True; self.gui_queue.put(("update_stats_gui", None))
        else: self.log("Arquivo modelo inválido."); 
        self.log("Nenhum modelo pré-treinado.")
        self.log("Carregando históricos..."); 
        df_hist = load_historical_data(HISTORICAL_DATA_PATH)
        if df_hist is not None: 
            self.historical_data = df_hist; 
            self.log("Históricos carregados.")
        else: 
            self.log("Falha carregar históricos."); 
            model_loaded_success = False; 
            self.trained_model = None
        if model_loaded_success and self.historical_data is not None: 
            self.set_button_state(self.predict_button, tk.NORMAL); 
            self.log("Pronto para prever.")
        else: 
            self.set_button_state(self.predict_button, tk.DISABLED); 
            self.gui_queue.put(("update_stats_gui", None))

    # --- process_gui_queue ---
    def process_gui_queue(self):
        """Processa mensagens da fila da GUI (com progresso e correção)."""
        try:
            while True:
                try:
                     message = self.gui_queue.get_nowait()
                     # Garante que é tupla de 2 itens ANTES de desempacotar
                     if isinstance(message, tuple) and len(message) == 2:
                         msg_type, msg_payload = message
                     else:
                          self.log(f"AVISO GUI: Msg formato inesperado: {message}")
                          continue
                except Empty: break # Fila vazia, sai do while interno
                except Exception as e_get: print(f"Erro get fila: {e_get}"); continue

                # Processa mensagens válidas
                if msg_type == "log": self._update_log_area(msg_payload)
                elif msg_type == "button_state": self._update_button_state(msg_payload)
                elif msg_type == "display_predictions": self._update_prediction_display(msg_payload)
                elif msg_type == "update_stats_gui": self._update_model_stats_display_gui()
                elif msg_type == "error": # Espera tupla (title, detail)
                    if isinstance(msg_payload, tuple) and len(msg_payload) == 2: messagebox.showerror(*msg_payload)
                    else: self.log(f"AVISO GUI: Payload inválido p/ error: {msg_payload}")
                elif msg_type == "info": # Espera tupla (title, detail)
                     if isinstance(msg_payload, tuple) and len(msg_payload) == 2: messagebox.showinfo(*msg_payload)
                     else: self.log(f"AVISO GUI: Payload inválido p/ info: {msg_payload}")
                # --- TRATAMENTO PARA PAYLOADS ---
                elif msg_type == "progress_start":
                     # Payload é o max_value (int)
                     max_val = msg_payload if isinstance(msg_payload, int) and msg_payload > 0 else 100
                     self.progress_bar.config(maximum=max_val, value=0); 
                     self.progress_label.config(text="Iniciando...")
                elif msg_type == "progress_update":
                    # Payload é a tupla (value, status_text)
                    if isinstance(msg_payload, tuple) and len(msg_payload) == 2:
                        value, status_text = msg_payload # Desempacota payload
                        self.progress_bar['value'] = value; 
                        self.progress_label.config(text=status_text)
                    else: self.log(f"AVISO GUI: Payload inválido p/ progress_update: {msg_payload}")
                elif msg_type == "progress_end":
                    self.progress_bar['value'] = 0; 
                    self.progress_label.config(text="Pronto.")
                elif msg_type == "training_complete":
                    try: # Payload é a tupla (model, scaler, features, params, metrics, timestamp)
                        if isinstance(msg_payload, tuple) and len(msg_payload) == 6:
                            model, scaler, features, params, metrics, timestamp = msg_payload # Desempacota payload
                            self.trained_model = model; 
                            self.trained_scaler = scaler; 
                            self.feature_columns = features; 
                            self.model_best_params = params; 
                            self.model_eval_metrics = metrics; 
                            self.model_file_timestamp = timestamp + " (Concluído)"; 
                            self.log("Pipeline treino OK.")
                            self._update_model_stats_display_gui(); 
                            self.set_button_state(self.predict_button, tk.NORMAL); 
                            self.set_button_state(self.load_train_button, tk.NORMAL); 
                            self._setup_prediction_columns(self.prediction_tree['columns'])
                        else: 
                            self.log(f"AVISO GUI: Payload inválido p/ training_complete: {msg_payload}")
                    except Exception as e_proc_train: self.log(f"ERRO GUI: Erro processar treino: {e_proc_train}")
                elif msg_type == "training_failed":
                     # ... ( limpar estado, atualizar GUI) ...
                     self.trained_model = None; 
                     self.trained_scaler = None; 
                     self.feature_columns = None; 
                     self.model_best_params = None; 
                     self.model_eval_metrics = None; 
                     self.model_file_timestamp = None; 
                     self.log("Pipeline treino falhou.")
                     self._update_model_stats_display_gui(); 
                     self.set_button_state(self.predict_button, tk.DISABLED); 
                     self.set_button_state(self.load_train_button, tk.NORMAL); 
                     self._setup_prediction_columns(['Status']); 
                     self.prediction_tree.insert('', tk.END, values=['Falha Treinamento.'])
                elif msg_type == "prediction_complete":
                     df_preds = msg_payload; self.log("Pipeline previsão concluído.")
                     self._update_prediction_display(df_preds)

        except Exception as e_queue: # Erro GERAL no processamento da fila
             print(f"Erro fatal fila GUI: {type(e_queue).__name__} - {e_queue}"); import traceback; traceback.print_exc()
             try: messagebox.showerror("Erro Crítico GUI", f"Erro fila eventos:\n{e_queue}")
             except Exception: pass
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