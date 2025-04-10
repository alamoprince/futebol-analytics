# --- src/main.py ---
from config import (TEST_SIZE, RANDOM_STATE,HISTORICAL_DATA_PATH, MODEL_SAVE_PATH, CLASS_NAMES, OTHER_ODDS_NAMES, FIXTURE_FETCH_DAY, MODEL_TYPE_NAME, ODDS_COLS)
from data_handler import (load_historical_data, preprocess_and_feature_engineer, fetch_and_process_fixtures, prepare_fixture_data, calculate_historical_intermediate)
import model_trainer
import predictor
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
import sys
import data_handler
import os
import threading
from queue import Queue, Empty
import datetime
import pandas as pd
from typing import Optional, Dict, List, Any
import math

# Adiciona diretórios ao path e importa módulos
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path: sys.path.append(SRC_DIR)
if BASE_DIR not in sys.path: sys.path.append(BASE_DIR)


class FootballPredictorDashboard:
    def __init__(self, root):
        self.root = root; self.root.title(f"Football Predictor Pro ({MODEL_TYPE_NAME})")
        self.root.geometry("950x750"); self.root.minsize(800, 600) 

        self.gui_queue = Queue()
        self.historical_data: Optional[pd.DataFrame] = None
        self.trained_model: Optional[Any] = None; self.trained_scaler: Optional[Any] = None
        self.feature_columns: Optional[List[str]] = None; self.model_best_params: Optional[Dict] = None
        self.model_eval_metrics: Optional[Dict] = None; self.model_file_timestamp: Optional[str] = None

        self.create_widgets()
        self.root.after(100, self.process_gui_queue)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.log(f"Bem-vindo ({MODEL_TYPE_NAME})!"); self.log(f"Fonte CSV: '{FIXTURE_FETCH_DAY}'.")
        self.log(f"Carregando modelo e histórico..."); self.load_existing_model_assets()

    def create_widgets(self):
        style = ttk.Style(); style.theme_use('clam')
        main_frame = ttk.Frame(self.root, padding="10"); main_frame.pack(fill=tk.BOTH, expand=True)
        left_panel = ttk.Frame(main_frame); left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), pady=5, anchor='nw')
        control_frame = ttk.LabelFrame(left_panel, text=" Ações ", padding="10"); control_frame.pack(pady=(0,5), fill=tk.X)
        self.load_train_button = ttk.Button(control_frame, text="Carregar Histórico e Treinar", command=self.start_training_thread, width=30); self.load_train_button.pack(pady=5, fill=tk.X)
        self.predict_button = ttk.Button(control_frame, text=f"Prever Jogos ({FIXTURE_FETCH_DAY.capitalize()})", command=self.start_prediction_thread, width=30); self.predict_button.pack(pady=5, fill=tk.X)
        self.predict_button.config(state=tk.DISABLED)

        # --- Barra de Progresso e Status ---
        progress_frame = ttk.Frame(control_frame) 
        progress_frame.pack(fill=tk.X, pady=(10, 0))
        self.progress_label = ttk.Label(progress_frame, text="Pronto.")
        self.progress_label.pack(side=tk.LEFT, padx=(0, 5))
        self.progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.progress_bar.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        # ----------------------------------

        stats_frame = ttk.LabelFrame(left_panel, text=" Status Modelo ", padding="10"); stats_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        self.model_stats_text = ScrolledText(stats_frame, height=15, state='disabled', wrap=tk.WORD, font=("Consolas", 9), relief=tk.FLAT, bd=0); self.model_stats_text.pack(fill=tk.BOTH, expand=True); self._update_model_stats_display_gui()
        right_panel = ttk.Frame(main_frame); right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True); results_frame = ttk.LabelFrame(right_panel, text=" Previsões ", padding="5"); results_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        cols = ['Data', 'Hora', 'Liga', 'Casa', 'Fora', 'Odd H', 'Odd D', 'Odd A', 'O2.5', 'BTTS S', 'P(Ñ Emp)', 'P(Empate)']
        self.prediction_tree = ttk.Treeview(results_frame, columns=cols, show='headings', height=10); vsb = ttk.Scrollbar(results_frame, orient="vertical", command=self.prediction_tree.yview); hsb = ttk.Scrollbar(results_frame, orient="horizontal", command=self.prediction_tree.xview); self.prediction_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set); vsb.pack(side='right', fill='y'); hsb.pack(side='bottom', fill='x'); self.prediction_tree.pack(fill=tk.BOTH, expand=True); self._setup_prediction_columns(cols)
        log_frame = ttk.LabelFrame(right_panel, text=" Logs ", padding="5"); log_frame.pack(fill=tk.BOTH, expand=True, side=tk.BOTTOM, pady=(5,0)); self.log_area = ScrolledText(log_frame, height=8, state='disabled', wrap=tk.WORD, font=("Consolas", 9)); self.log_area.pack(fill=tk.BOTH, expand=True)

    # --- Funções _setup_prediction_columns, log, _update_log_area, set_button_state, _update_button_state ---
    def _setup_prediction_columns(self, columns: List[str]):
         self.prediction_tree['columns'] = columns; self.prediction_tree.delete(*self.prediction_tree.get_children()); col_widths = {'Data': 75, 'Hora': 50, 'Liga': 150, 'Casa': 110, 'Fora': 110,'Odd H': 50, 'Odd D': 50, 'Odd A': 50, 'O2.5': 50, 'BTTS S': 55, 'P(Ñ Emp)': 70, 'P(Empate)': 70}
         for col in columns: width = col_widths.get(col, 80); anchor = tk.CENTER if col not in ['Liga', 'Casa', 'Fora', 'Data', 'Hora'] else tk.W; self.prediction_tree.heading(col, text=col); self.prediction_tree.column(col, anchor=anchor, width=width, stretch=False)
         if columns == ['Status']: self.prediction_tree.heading('Status', text='Status'); self.prediction_tree.column('Status', anchor=tk.W, width=400)
    def log(self, message: str): self.gui_queue.put(("log", message))
    def _update_log_area(self, message: str):
        try: self.log_area.config(state='normal'); ts = datetime.datetime.now().strftime("%H:%M:%S"); self.log_area.insert(tk.END, f"[{ts}] {message}\n"); self.log_area.config(state='disabled'); self.log_area.see(tk.END)
        except tk.TclError: pass
    def set_button_state(self, button: ttk.Button, state: str): self.gui_queue.put(("button_state", (button, state)))
    def _update_button_state(self, button_state_tuple):
        button, state = button_state_tuple; 
        try: button.config(state=state); 
        except tk.TclError: pass

    # --- Exibição de Previsões ---
    def display_predictions(self, df_predictions: Optional[pd.DataFrame]): self.gui_queue.put(("display_predictions", df_predictions))
    def _update_prediction_display(self, df: Optional[pd.DataFrame]):
         display_headers = self.prediction_tree['columns']; header_to_col_map = {'Data': 'Date_Str', 'Hora': 'Time_Str', 'Liga': 'League', 'Casa': 'HomeTeam', 'Fora': 'AwayTeam','Odd H': ODDS_COLS['home'], 'Odd D': ODDS_COLS['draw'], 'Odd A': ODDS_COLS['away'],'O2.5': 'Odd_Over25_FT', 'BTTS S': 'Odd_BTTS_Yes','P(Ñ Emp)': f'Prob_{CLASS_NAMES[0]}', 'P(Empate)': f'Prob_{CLASS_NAMES[1]}'}; internal_cols_to_fetch = [header_to_col_map.get(h) for h in display_headers if header_to_col_map.get(h)]; valid_internal_cols = [c for c in internal_cols_to_fetch if df is not None and c in df.columns]
         self.prediction_tree.delete(*self.prediction_tree.get_children());
         if df is None or df.empty: self._setup_prediction_columns(['Status']); self.prediction_tree.insert('', tk.END, values=['Nenhuma previsão.']); return
         self._setup_prediction_columns(display_headers)
         try:
             df_display = df[valid_internal_cols].copy(); prob_cols = [f'Prob_{CLASS_NAMES[0]}', f'Prob_{CLASS_NAMES[1]}']; odds_cols_internal = list(ODDS_COLS.values()) + OTHER_ODDS_NAMES
             for pcol in prob_cols:
                 if pcol in df_display: df_display[pcol] = (df_display[pcol] * 100).round(1).astype(str) + '%'
             for ocol in odds_cols_internal:
                  if ocol in df_display: df_display[ocol] = df_display[ocol].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
             for _, row in df_display.iterrows(): values = [str(row.get(header_to_col_map.get(h), '')) for h in display_headers]; self.prediction_tree.insert('', tk.END, values=values)
         except Exception as e: self.log(f"Erro exibir previsões: {e}"); self._setup_prediction_columns(['Status']); self.prediction_tree.insert('', tk.END, values=[f'Erro: {e}'])

    # --- Área de Stats do Modelo GUI ---
    def _update_model_stats_display_gui(self):
        try:
            self.model_stats_text.config(state='normal'); self.model_stats_text.delete('1.0', tk.END)
            if self.trained_model is None: stats_content = "Nenhum modelo treinado."
            else:
                stats_content = f"Arquivo: {os.path.basename(MODEL_SAVE_PATH)}\nModif.: {self.model_file_timestamp or 'N/A'}\nTipo: {self.trained_model.__class__.__name__} ({MODEL_TYPE_NAME})\n---\n"; 
                if self.feature_columns: stats_content += f"Features ({len(self.feature_columns)}):\n - " + "\n - ".join(self.feature_columns) + "\n---\n"
                if self.model_best_params: stats_content += "Melhores Parâmetros:\n"; params_list = [f" - {k}: {v}" for k,v in self.model_best_params.items()]; stats_content += "\n".join(params_list) + "\n---\n"
                if self.model_eval_metrics: stats_content += "Métricas Avaliação (Teste):\n"; acc = self.model_eval_metrics.get('accuracy'); loss = self.model_eval_metrics.get('log_loss'); auc = self.model_eval_metrics.get('roc_auc'); prec_d = self.model_eval_metrics.get('precision_draw'); rec_d = self.model_eval_metrics.get('recall_draw'); f1_d = self.model_eval_metrics.get('f1_score_draw'); profit = self.model_eval_metrics.get('profit'); roi = self.model_eval_metrics.get('roi'); n_bets = self.model_eval_metrics.get('num_bets'); stats_content += f"- Acurácia: {acc:.4f}\n" if acc is not None else "- Acurácia: N/A\n"; stats_content += f"- Log Loss: {loss:.4f}\n" if loss is not None and not math.isnan(loss) else "- Log Loss: N/A\n"; stats_content += f"- ROC AUC: {auc:.4f}\n" if auc is not None else "- ROC AUC: N/A\n"; stats_content += "- Métricas 'Empate':\n"; stats_content += f"  - Precision: {prec_d:.4f}\n" if prec_d is not None else "  - Precision: N/A\n"; stats_content += f"  - Recall:    {rec_d:.4f}\n" if rec_d is not None else "  - Recall: N/A\n"; stats_content += f"  - F1-Score:  {f1_d:.4f}\n" if f1_d is not None else "  - F1-Score: N/A\n---\n"; stats_content += "Estratégia BackDraw:\n"; stats_content += f"- Nº Apostas: {n_bets if n_bets is not None else 'N/A'}\n"; stats_content += f"- Profit: {profit:.2f} u\n" if profit is not None else "- Profit: N/A\n"; stats_content += f"- ROI: {roi:.2f} %\n" if roi is not None else "- ROI: N/A\n"
                else: stats_content += "Métricas Avaliação: Não disponíveis.\n"
            self.model_stats_text.insert('1.0', stats_content); self.model_stats_text.config(state='disabled')
        except tk.TclError: pass; 
        except Exception as e: print(f"Erro update stats GUI: {e}")

    # --- Threads de Ação (REFATORADAS) ---
    def start_training_thread(self):
        """Carrega histórico e inicia thread de treino."""
        self.log("Carregando dados históricos...")
        # Limpa estado e atualiza GUI antes de carregar
        self.trained_model = None; self.trained_scaler = None; self.feature_columns = None
        self.model_best_params = None; self.model_eval_metrics = None; self.model_file_timestamp = None
        self.gui_queue.put(("update_stats_gui", None))
        self.set_button_state(self.load_train_button, tk.DISABLED)
        self.set_button_state(self.predict_button, tk.DISABLED)
        self.gui_queue.put(("progress_start", 100)) # Inicia progresso (valor max arbitrário)
        self.gui_queue.put(("progress_update", 5, "Carregando Histórico..."))

        try:
            df_hist = load_historical_data(HISTORICAL_DATA_PATH)
            if df_hist is None: raise ValueError("Falha ao carregar histórico do Excel.")
            self.historical_data = df_hist # Guarda o histórico bruto
            self.log("Histórico carregado. Iniciando thread de treino...")
            self.gui_queue.put(("progress_update", 20, "Iniciando Treinamento..."))
            # Inicia a thread, passando o histórico bruto
            train_thread = threading.Thread(target=self._run_training_pipeline, args=(self.historical_data,), daemon=True)
            train_thread.start()
        except Exception as e_load:
             error_msg = f"Erro Carregar Histórico: {e_load}"
             self.log(f"ERRO: {error_msg}")
             self.gui_queue.put(("error", ("Erro Carregamento", error_msg)))
             self.gui_queue.put(("progress_end", None)) # Finaliza progresso
             self.set_button_state(self.load_train_button, tk.NORMAL) # Reabilita botão

    # NOVA função de pipeline que roda na thread
    def _run_training_pipeline(self, df_hist_raw: pd.DataFrame):
        """Pipeline completo de pré-processamento e treino que roda na thread."""
        try:

            # 1. Pré-processamento
            self.gui_queue.put(("progress_update", 30, "Pré-processando..."))
            processed = preprocess_and_feature_engineer(df_hist_raw)
            if processed is None: raise ValueError("Falha no pré-processamento BackDraw.")
            X_processed, y_processed, features = processed
            self.feature_columns = features # Guarda features para salvar depois

            # 2. Preparar X_test_full para ROI
            self.gui_queue.put(("progress_update", 50, "Preparando dados teste p/ ROI..."))
            df_hist_interm = calculate_historical_intermediate(df_hist_raw) # Função pública
            df_hist_aligned = df_hist_interm.loc[X_processed.index.union(y_processed.index)].copy()
            _, X_test_full_data, _, _ = train_test_split(
                 df_hist_aligned, y_processed,
                 test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_processed
            )

            # 3. Treinamento e Avaliação (com callback de progresso)
            # Criar o callback que envia msg para a fila da GUI
            def training_progress_callback(current_step, max_steps, status_text):
                # Calcula progresso entre 60% e 95% da barra total
                progress_percent = 60 + int((current_step / max_steps) * 35)
                self.gui_queue.put(("progress_update", progress_percent, status_text))

            self.log("Iniciando treinamento e avaliação dos modelos...")
            self.gui_queue.put(("progress_update", 60, "Treinando modelos..."))
            train_result = model_trainer.train_evaluate_model(
                X_processed, y_processed,
                X_test_with_odds=X_test_full_data,
                odd_draw_col_name=ODDS_COLS['draw'],
                progress_callback=training_progress_callback # Passa o callback
            )

            # 4. Processar Resultado
            self.gui_queue.put(("progress_update", 95, "Finalizando..."))
            if train_result:
                model, scaler, _, params, metrics = train_result
                model_trainer.save_model_scaler_features(model, scaler, self.feature_columns, params, metrics, MODEL_SAVE_PATH)
                timestamp_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                payload = (model, scaler, self.feature_columns, params, metrics, timestamp_str)
                self.gui_queue.put(("training_complete", payload))
                # Envia resultado completo para a GUI processar
                self.gui_queue.put(("training_complete", (model, scaler, self.feature_columns, params, metrics, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))))
            else:
                raise RuntimeError("Falha no treinamento (train_evaluate_model retornou None).")

        except Exception as e:
            error_msg = f"Erro Treinamento Thread: {e}"
            self.log(f"ERRO: {error_msg}")
            # Envia erro para a GUI
            self.gui_queue.put(("error", ("Erro no Treino", error_msg)))
            # Sinaliza fim do progresso mesmo com erro
            self.gui_queue.put(("progress_end", None))
            # Limpa estado na instância principal (via fila ou direto se seguro?) - Melhor via fila
            self.gui_queue.put(("training_failed", None))

    # Função de pipeline de previsão
    def _run_prediction_pipeline(self):
        """Pipeline completo de busca CSV e previsão."""
        try:
            self.gui_queue.put(("progress_start", 100)) 

            # 1. Buscar CSV
            self.gui_queue.put(("progress_update", 10, f"Buscando CSV ({FIXTURE_FETCH_DAY})..."))
            fixture_df = fetch_and_process_fixtures()
            if fixture_df is None: raise ValueError("Falha ao buscar/processar CSV.")
            if fixture_df.empty:
                 self.log("Nenhum jogo encontrado no CSV.");
                 self.gui_queue.put(("prediction_complete", None)); # Envia None para limpar
                 return # Termina a thread

            self.gui_queue.put(("progress_update", 40, f"Preparando features ({len(fixture_df)} jogos)..."))
            
            # 2. Preparar Features
            X_fixtures_prepared = prepare_fixture_data(fixture_df, self.historical_data, self.feature_columns)
            if X_fixtures_prepared is None or X_fixtures_prepared.empty: raise ValueError("Falha ao preparar features.")

            # 3. Fazer Previsões
            self.gui_queue.put(("progress_update", 70, "Realizando previsões..."))
            df_predictions = predictor.make_predictions(self.trained_model, self.trained_scaler, self.feature_columns, X_fixtures_prepared, fixture_df) # Passa scaler
            if df_predictions is None: raise RuntimeError("Falha ao gerar previsões.")

            # 4. Enviar resultado para GUI
            self.gui_queue.put(("progress_update", 95, "Finalizando..."))
            self.gui_queue.put(("prediction_complete", df_predictions))

        except Exception as e:
            error_msg = f"Erro Previsão Thread: {e}"
            self.log(f"ERRO: {error_msg}")
            self.gui_queue.put(("error", ("Erro na Previsão", error_msg)))
            self.gui_queue.put(("prediction_complete", None)) # Envia None para limpar
        finally:
             # Sinaliza fim do progresso independentemente do resultado
             self.gui_queue.put(("progress_end", None))
             # Reabilita botões (via fila)
             self.set_button_state(self.load_train_button, tk.NORMAL)
             if self.trained_model: self.set_button_state(self.predict_button, tk.NORMAL)

    def start_prediction_thread(self):
       
        """Inicia thread de previsão."""
        # Verifica se o modelo e os dados históricos estão carregados
        if self.trained_model is None: messagebox.showwarning("Modelo Ausente", "Treine/Carregue."); return
        if self.historical_data is None: messagebox.showwarning("Histórico Ausente", "Carregue/Treine."); return
        self.log(f"Iniciando busca CSV ({FIXTURE_FETCH_DAY}) e previsão...")
        self.set_button_state(self.load_train_button, tk.DISABLED)
        self.set_button_state(self.predict_button, tk.DISABLED)
        self._setup_prediction_columns(['Status']); self.prediction_tree.insert('', tk.END, values=['Iniciando...'])
        predict_thread = threading.Thread(target=self._run_prediction_pipeline, daemon=True); predict_thread.start()

    def load_existing_model_assets(self):
        # ... (Carrega modelo e histórico) ...
        self.trained_model = None; self.trained_scaler = None; self.feature_columns = None; self.model_best_params = None; self.model_eval_metrics = None; self.model_file_timestamp = None; model_loaded_success = False
        load_result = predictor.load_model_scaler_features(MODEL_SAVE_PATH)
        if load_result:
            model, scaler, features, params, metrics, timestamp = load_result # Desempacota 6
            if model and features: self.trained_model = model; self.trained_scaler = scaler; self.feature_columns = features; self.model_best_params = params; self.model_eval_metrics = metrics; self.model_file_timestamp = timestamp; self.log(f"Modelo {MODEL_TYPE_NAME} e stats carregados."); model_loaded_success = True; self.gui_queue.put(("update_stats_gui", None))
            else: self.log("Arquivo de modelo inválido.")
        else: self.log("Nenhum modelo pré-treinado encontrado.")
        self.log("Carregando dados históricos..."); df_hist = data_handler.load_historical_data(HISTORICAL_DATA_PATH)
        if df_hist is not None: self.historical_data = df_hist; self.log("Dados históricos carregados.")
        else: self.log("Falha ao carregar históricos."); model_loaded_success = False; self.trained_model = None
        if model_loaded_success and self.historical_data is not None: self.set_button_state(self.predict_button, tk.NORMAL); self.log("Pronto para prever.")
        else: self.set_button_state(self.predict_button, tk.DISABLED); self.gui_queue.put(("update_stats_gui", None))


    def process_gui_queue(self):
        try:
            while True:
                # Desempacota a mensagem principal em tipo e payload
                msg_type, msg_payload = self.gui_queue.get_nowait() # Espera 2 itens aqui

                if msg_type == "log": self._update_log_area(msg_payload)

                # --- VERIFICAR ESTE BLOCO ---
                elif msg_type == "training_complete":
                    try:
                        # Desempacota o PAYLOAD (que é a tupla de 6 itens)
                        model, scaler, features, params, metrics, timestamp = msg_payload # Espera 6 itens AQUI
                        
                        # Atualiza o estado da GUI com os dados recebidos
                        self.trained_model = model; self.trained_scaler = scaler; self.feature_columns = features
                        self.model_best_params = params; self.model_eval_metrics = metrics; self.model_file_timestamp = timestamp + " (Concluído)"
                        self.log("Pipeline de treino concluído com sucesso.")
                        self._update_model_stats_display_gui()
                        self.set_button_state(self.predict_button, tk.NORMAL)
                        self.set_button_state(self.load_train_button, tk.NORMAL)
                        self._setup_prediction_columns(self.prediction_tree['columns']) # Limpa status
                        
                    except ValueError as e_unpack:
                        self.log(f"ERRO GUI: Falha ao desempacotar resultado do treino: {e_unpack}")
                        self.gui_queue.put(("error", ("Erro Interno", "Falha ao processar resultado do treino.")))
                    except Exception as e_proc_train:
                        self.log(f"ERRO GUI: Erro ao processar treino completo: {e_proc_train}")
                        self.gui_queue.put(("error", ("Erro Interno", "Falha ao atualizar GUI pós-treino.")))

                elif msg_type == "training_failed":
                    self.log("Pipeline de treino falhou.")
                elif msg_type == "prediction_complete":
                    
                    df_preds = msg_payload
                    self.log("Pipeline de previsão concluído.")
                    self._update_prediction_display(df_preds)
                    # O payload aqui é apenas df_predictions (ou None)
                    df_preds = msg_payload
                    self.log("Pipeline de previsão concluído.")
                    self._update_prediction_display(df_preds)

        except Empty: pass
        except ValueError as e_msg_unpack: # Captura erro se a MENSAGEM não tiver 2 itens
            print(f"Erro Fila GUI: Mensagem inválida recebida (não é tupla de 2?): {e_msg_unpack}")
            
        except Exception as e: print(f"Erro processar fila GUI: {e}")
        finally: self.root.after(100, self.process_gui_queue)

    def on_closing(self): self.log("Fechando..."); self.root.destroy()

if __name__ == "__main__":
    import requests; import math; import numpy as np; from sklearn.model_selection import train_test_split # Garante imports
    try: from ctypes import windll; windll.shcore.SetProcessDpiAwareness(1)
    except: pass
    root = tk.Tk()
    app = FootballPredictorDashboard(root)
    root.mainloop()