# --- src/main.py ---
# START OF FILE main.py - V15 (Indentation Fix)
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, Listbox, MULTIPLE, END, Scrollbar
from tkinter.scrolledtext import ScrolledText
import sys, os, threading, datetime, math, numpy as np, io
from queue import Queue, Empty
import pandas as pd
from typing import Optional, Dict, List, Any

# Adiciona diretórios e importa módulos
SRC_DIR = os.path.dirname(os.path.abspath(__file__)); BASE_DIR = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path: sys.path.append(SRC_DIR)
if BASE_DIR not in sys.path: sys.path.append(BASE_DIR)
try:
    from config import ( HISTORICAL_DATA_PATH, CLASS_NAMES, ODDS_COLS, OTHER_ODDS_NAMES, FIXTURE_FETCH_DAY, MODEL_TYPE_NAME, TEST_SIZE, RANDOM_STATE, ODDS_COLS as CONFIG_ODDS_COLS, BEST_F1_MODEL_SAVE_PATH, BEST_ROI_MODEL_SAVE_PATH, MODEL_ID_F1, MODEL_ID_ROI, FEATURE_COLUMNS, MODEL_CONFIG, ROLLING_WINDOW # Adiciona ROLLING_WINDOW
    )
    from data_handler import ( load_historical_data, preprocess_and_feature_engineer, fetch_and_process_fixtures, prepare_fixture_data, calculate_historical_intermediate, calculate_rolling_stats, calculate_derived_features )
    from model_trainer import train_evaluate_and_save_best_models as run_training_process, analyze_features, optimize_single_model
    import model_trainer
    import predictor
    import requests
    from sklearn.model_selection import train_test_split
    import traceback
except ImportError as e: print(f"Erro import main.py: {e}"); sys.exit(1)
except Exception as e_i: print(f"Erro geral import main.py: {e_i}"); sys.exit(1)


class FootballPredictorDashboard:

    def __init__(self, parent_frame, main_root=None):
        self.parent_frame = parent_frame
        self.root = main_root if main_root else None
        self.root = self.root; self.root.title(f"Football Predictor Pro ({MODEL_TYPE_NAME}) - Abas")
        self.root.geometry("1050x750"); self.root.minsize(900, 600)
        self.gui_queue = Queue()
        self.historical_data: Optional[pd.DataFrame] = None; self.historical_data_processed: Optional[pd.DataFrame] = None
        self.loaded_models_data: Dict[str, Dict] = {}; self.available_model_ids: List[str] = []
        self.selected_model_id: Optional[str] = None; self.trained_model: Optional[Any] = None; self.trained_scaler: Optional[Any] = None
        self.feature_columns: Optional[List[str]] = None; self.model_best_params: Optional[Dict] = None
        self.model_eval_metrics: Optional[Dict] = None; self.model_file_timestamp: Optional[str] = None

        self.create_widgets_with_tabs() # CHAMA A FUNÇÃO DE CRIAÇÃO (JÁ DEFINIDA ACIMA)

        self.root.after(100, self.process_gui_queue) # Inicia o loop da fila (JÁ DEFINIDA ACIMA)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing) # (JÁ DEFINIDA ABAIXO)
        self.log(f"Bem-vindo!"); self.log(f"Fonte CSV: '{FIXTURE_FETCH_DAY}'.") # log JÁ DEFINIDA ACIMA
        self.log(f"Tentando carregar modelos e histórico..."); self.load_existing_model_assets() # load_existing_model_assets JÁ DEFINIDA ABAIXO

    # --- MÉTODOS AUXILIARES DA GUI (Definidos Primeiro) ---
    def log(self, message: str): 
        self.gui_queue.put(("log", message))

    def _update_log_area(self, message: str): 
         try: 
            self.log_area.config(state='normal'); 
            ts = datetime.datetime.now().strftime("%H:%M:%S"); 
            self.log_area.insert(tk.END, f"[{ts}] {message}\n"); 
            self.log_area.config(state='disabled'); 
            self.log_area.see(tk.END); 
         except tk.TclError: pass

    def set_button_state(self, button: ttk.Button, state: str): 
        self.gui_queue.put(("button_state", (button, state)))

    def _update_button_state(self, button_state_tuple): 
         button, state = button_state_tuple; 
         try: button.config(state=state); 
         except tk.TclError: pass
    def _update_text_widget(self, text_widget: ScrolledText, content: str): 
         try: 
            text_widget.config(state='normal'); 
            text_widget.delete('1.0', tk.END); 
            text_widget.insert('1.0', content); 
            text_widget.config(state='disabled'); 
         except tk.TclError: pass

    def _update_model_stats_display_gui(self): 
        try:
            self.model_stats_text.config(state='normal')
            self.model_stats_text.delete('1.0', tk.END)
            
            if self.trained_model is None or self.selected_model_id is None:
                stats_content = "Nenhum modelo selecionado."
            else:
                model_data = self.loaded_models_data.get(self.selected_model_id, {})
                stats_content = (
                    f"Modelo: {self.selected_model_id}\n"
                    f"Arquivo: {os.path.basename(model_data.get('path', 'N/A'))}\n"
                    f"Modif.: {self.model_file_timestamp or 'N/A'}\n"
                    f"Tipo: {self.trained_model.__class__.__name__}\n"
                    "---\n"
                )
                
            self.model_stats_text.insert('1.0', stats_content)
            self.model_stats_text.config(state='disabled')
        except tk.TclError:
            pass
        except Exception as e:
            print(f"Erro update stats GUI: {e}")

    def _setup_prediction_columns(self, columns: List[str]): 
         self.prediction_tree['columns'] = columns; 
         self.prediction_tree.delete(*self.prediction_tree.get_children()); 
         col_widths = {'Data': 75, 'Hora': 50, 'Liga': 150, 'Casa': 110, 'Fora': 110,
                       'Odd H': 50, 'Odd D': 50, 'Odd A': 50, 'O2.5': 50, 'BTTS S': 55, 
                       'P(Ñ Emp)': 70, 'P(Empate)': 70}
         if columns == ['Status']: 
            self.prediction_tree.heading('Status', text='Status'); 
            self.prediction_tree.column('Status', anchor=tk.W, width=500); return
         else:
             for col in columns: 
                width = col_widths.get(col, 80); 
                anchor = tk.CENTER if col not in ['Liga', 'Casa', 'Fora', 'Data', 'Hora'] else tk.W; 
                self.prediction_tree.heading(col, text=col); 
                self.prediction_tree.column(col, anchor=anchor, width=width, stretch=False)

    def _update_prediction_display(self, df: Optional[pd.DataFrame]): 
         self.log(f"--- DEBUG: Iniciando _update_prediction_display ---"); #... (logs ) ...
         display_headers = ['Data', 'Hora', 'Liga', 'Casa', 'Fora', 'Odd H', 'Odd D', 'Odd A', 'O2.5', 
                            'BTTS S', 'P(Ñ Emp)', 'P(Empate)'];
         try: 
             for item in self.prediction_tree.get_children(): self.prediction_tree.delete(item); self.log("DEBUG: Treeview limpa.")
         except tk.TclError: return
         if df is None or df.empty: 
            self.log("DEBUG: Configurando Status (DF Vazio/None)."); 
            self._setup_prediction_columns(['Status']); 
            self.prediction_tree.insert('', tk.END, values=['Nenhuma previsão válida.']); return
         self.log(f"DEBUG: Reconfigurando colunas p/ dados..."); 
         self._setup_prediction_columns(display_headers); 
         self.log("DEBUG: Treeview reconfigurada.")
         header_to_col_map = {'Data': 'Date_Str', 'Hora': 'Time_Str', 'Liga': 'League', 
                              'Casa': 'HomeTeam', 'Fora': 'AwayTeam','Odd H': CONFIG_ODDS_COLS['home'], 
                              'Odd D': CONFIG_ODDS_COLS['draw'], 'Odd A': CONFIG_ODDS_COLS['away'],
                              'O2.5': 'Odd_Over25_FT', 'BTTS S': 'Odd_BTTS_Yes','P(Ñ Emp)': f'Prob_{CLASS_NAMES[0]}', 
                              'P(Empate)': f'Prob_{CLASS_NAMES[1]}'}; 
         internal_cols_to_fetch = [header_to_col_map.get(h) for h in display_headers if header_to_col_map.get(h)]; 
         valid_internal_cols = [c for c in internal_cols_to_fetch if c in df.columns]; 
         self.log(f"DEBUG: Colunas válidas encontradas: {valid_internal_cols}")
         
         try: # ... (resto da formatação e inserção ) ...
            if not valid_internal_cols: 
                self.log("ERRO: Nenhuma coluna válida!"); 
                self._setup_prediction_columns(['Status']); 
                self.prediction_tree.insert('', tk.END, values=['Erro: Colunas não encontradas.']); return
            df_display = df[valid_internal_cols].copy(); #... (formatação probs e odds) ...
            added_rows = 0; 
            for index, row in df_display.iterrows(): values = [str(row.get(header_to_col_map.get(h), '')) for h in display_headers]; #... (try/except insert) ...
            self.log(f"DEBUG: {added_rows} linhas adicionadas.")
         except Exception as e: self.log(f"!!!!! ERRO GERAL _update_prediction_display: {e}"); 
    
    def _select_all_features(self):
        try: self.feature_listbox.selection_set(0, tk.END); 
        except (tk.TclError, AttributeError): pass
    def _clear_feature_selection(self): 
        try: self.feature_listbox.selection_clear(0, tk.END); 
        except (tk.TclError, AttributeError): pass
    def _populate_analyze_widgets(self): 
        if self.historical_data_processed is None:
            return
        try:
            # Atualiza Info e Head
            buffer = io.StringIO()
            self.historical_data_processed.info(buf=buffer)
            self._update_text_widget(self.analyze_info_text, buffer.getvalue())
            self._update_text_widget(self.analyze_head_text, self.historical_data_processed.head(10).to_string())

            # Atualiza Describe
            features_available = [f for f in FEATURE_COLUMNS if f in self.historical_data_processed.columns]
            if features_available:
                self._update_text_widget(self.analyze_desc_text, self.historical_data_processed[features_available].describe().to_string())
            else:
                self._update_text_widget(self.analyze_desc_text, "N/A")

            # Atualiza Target
            if 'IsDraw' in self.historical_data_processed.columns:
                target_dist = self.historical_data_processed['IsDraw'].value_counts(normalize=True).apply("{:.2%}".format).to_string()
                target_counts = self.historical_data_processed['IsDraw'].value_counts().to_string()
                self._update_text_widget(self.analyze_target_text, f"Contagem:\n{target_counts}\n\nProporção:\n{target_dist}")
            else:
                self._update_text_widget(self.analyze_target_text, "N/A")

            # Atualiza Listbox de Features
            self.feature_listbox.delete(0, tk.END)
            available_cols = sorted(
                [
                    c for c in self.historical_data_processed.columns
                    if pd.api.types.is_numeric_dtype(self.historical_data_processed[c])
                    and c not in ['IsDraw', 'Ptos_H', 'Ptos_A', 'Goals_H_FT', 'Goals_A_FT', 'p_H', 'p_D', 'p_A', 
                                  'VG_H_raw', 'VG_A_raw', 'CG_H_raw', 'CG_A_raw']
                ]
            )
            for col in available_cols:
                self.feature_listbox.insert(tk.END, col)
                if col in FEATURE_COLUMNS:
                    self.feature_listbox.selection_set(tk.END)

            # Atualiza Combobox de Modelos
            self.model_to_optimize_combo.config(values=list(MODEL_CONFIG.keys()))
            if MODEL_CONFIG:
                self.model_to_optimize_combo.current(0)

            # Atualiza Status
            self.analyze_status_label.config(text="Histórico Processado.")
            self.log("Widgets Aba Análise atualizados.")
        except Exception as e:
            self.log(f"Erro popular aba análise: {e}")

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

            # Update the stats display immediately
            self._update_model_stats_display_gui() # Ensure stats are shown

            # Enable/disable predict button based on model AND historical data
            if self.trained_model and self.feature_columns and self.historical_data is not None:
                self.set_button_state(self.predict_button, tk.NORMAL)
                self.log(f"Modelo '{selected_id}' pronto para previsão.")
            elif self.historical_data is None:
                self.log(f"Modelo '{selected_id}' selecionado, mas histórico ausente. Previsão desabilitada.")
                self.set_button_state(self.predict_button, tk.DISABLED)
            else: # Model invalid?
                self.log(f"Modelo '{selected_id}' inválido ou dados ausentes. Previsão desabilitada.")
                self.set_button_state(self.predict_button, tk.DISABLED)

        else:
            # Handle case where selection is cleared or invalid
            self.log(f"Seleção inválida ou limpa: '{selected_id}'. Resetando estado.")
            self.selected_model_id = None
            self.trained_model = None
            self.trained_scaler = None
            self.feature_columns = None
            self.model_best_params = None
            self.model_eval_metrics = None
            self.model_file_timestamp = None
            self.set_button_state(self.predict_button, tk.DISABLED)
            self._update_model_stats_display_gui() # Update stats display (will show 'no model')

    # --- Funções de Ação (Definidas ANTES de create_widgets) ---
    def start_training_thread(self): 
        self.log("Carregando histórico...")
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

    def start_prediction_thread(self): 
        if self.trained_model is None or self.selected_model_id is None:
            messagebox.showwarning("Modelo Não Selecionado", "Selecione.")
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
    def start_feature_analysis_thread(self): 
        if self.historical_data_processed is None: 
            messagebox.showwarning("Dados Ausentes", "Carregue.")
            return

        selected_indices = self.feature_listbox.curselection()
        if not selected_indices: 
            messagebox.showwarning("Seleção Vazia", "Selecione.")
            return

        selected_features = [self.feature_listbox.get(i) for i in selected_indices]
        self.log(f"Análise: Iniciando para: {selected_features}")
        self._update_text_widget(self.analyze_corr_text, "Calculando...")
        self._update_text_widget(self.analyze_importance_text, "Calculando...")

        target_col = 'IsDraw'
        cols_needed = selected_features + [target_col]
        if not all(c in self.historical_data_processed.columns for c in cols_needed): 
            messagebox.showerror("Erro", f"Colunas necessárias não encontradas.")
            return

        df_analysis = self.historical_data_processed[cols_needed].dropna()
        if df_analysis.empty: 
            messagebox.showerror("Erro", "Nenhum dado válido.")
            return

        X_analyze = df_analysis[selected_features]
        y_analyze = df_analysis[target_col]
        thread = threading.Thread(
            target=self._run_feature_analysis, 
            args=(X_analyze, y_analyze), 
            daemon=True
        )
        thread.start()

    def start_optimization_thread(self): messagebox.showinfo("A Fazer", "Otimizar não implementado.")

    # --- Criação dos Widgets com Abas (Definida ANTES de __init__) ---
    def create_widgets_with_tabs(self): 
        style = ttk.Style()
        style.theme_use('clam')
        self.notebook = ttk.Notebook(self.root, padding="5")
        
        self.tab_predict = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.tab_predict, text=' Previsão e Treino Final ')
        self.create_predict_tab_widgets(self.tab_predict)
        
        self.tab_analyze = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.tab_analyze, text=' Análise de Features ')
        self.create_analyze_tab_widgets(self.tab_analyze)
        
        self.notebook.pack(expand=True, fill="both")

    def create_predict_tab_widgets(self, parent_tab): 
        main_frame = ttk.Frame(parent_tab)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), pady=5, anchor='nw')
        
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
            predict_frame, textvariable=self.selected_model_var, 
            state="readonly", width=20
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
        
        cols = ['Data', 'Hora', 'Liga', 'Casa', 'Fora', 'Odd H', 'Odd D', 'Odd A', 'O2.5', 'BTTS S', 'P(Ñ Emp)', 'P(Empate)']
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
        
        log_frame = ttk.LabelFrame(right_panel, text=" Logs ", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True, side=tk.BOTTOM, pady=(5, 0))
        
        self.log_area = ScrolledText(
            log_frame, height=8, state='disabled', wrap=tk.WORD, font=("Consolas", 9)
        )
        self.log_area.pack(fill=tk.BOTH, expand=True)

    def create_analyze_tab_widgets(self, parent_tab): 
        main_frame = ttk.Frame(parent_tab)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        top_frame_analyze = ttk.LabelFrame(main_frame, text=" Análise Inicial ", padding=10)
        top_frame_analyze.pack(fill=tk.X, pady=(0, 5))
        
        load_analyze_button = ttk.Button(
            top_frame_analyze, text="Carregar & Processar Histórico p/ Análise", 
            command=self._load_and_analyze_historical_thread, width=35
        )
        load_analyze_button.pack(side=tk.LEFT, padx=(0, 10), pady=5)
        
        self.analyze_status_label = ttk.Label(top_frame_analyze, text="Status: Não carregado")
        self.analyze_status_label.pack(side=tk.LEFT, pady=5)
        
        mid_frame_analyze = ttk.Frame(main_frame)
        mid_frame_analyze.pack(fill=tk.X, pady=5)
        
        feature_select_frame = ttk.LabelFrame(mid_frame_analyze, text=" Seleção de Features ", padding=5)
        feature_select_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        feature_list_yscroll = Scrollbar(feature_select_frame, orient=tk.VERTICAL)
        feature_list_xscroll = Scrollbar(feature_select_frame, orient=tk.HORIZONTAL)
        
        self.feature_listbox = Listbox(
            feature_select_frame, selectmode=MULTIPLE, height=15, width=30, 
            yscrollcommand=feature_list_yscroll.set, xscrollcommand=feature_list_xscroll.set, 
            exportselection=False
        )
        feature_list_yscroll.config(command=self.feature_listbox.yview)
        feature_list_xscroll.config(command=self.feature_listbox.xview)
        feature_list_yscroll.pack(side=tk.RIGHT, fill=tk.Y)
        feature_list_xscroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.feature_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        feature_buttons_frame = ttk.Frame(feature_select_frame)
        feature_buttons_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))
        
        select_all_button = ttk.Button(
            feature_buttons_frame, text="Todas", width=8, command=self._select_all_features
        )
        select_all_button.pack(side=tk.LEFT, padx=2)
        
        clear_sel_button = ttk.Button(
            feature_buttons_frame, text="Nenhuma", width=8, command=self._clear_feature_selection
        )
        clear_sel_button.pack(side=tk.LEFT, padx=2)
        
        analyze_action_frame = ttk.Frame(mid_frame_analyze)
        analyze_action_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 0))
        
        analyze_features_button = ttk.Button(
            analyze_action_frame, text="Analisar Selecionadas\n(Importância/Correlação)", 
            command=self.start_feature_analysis_thread
        )
        analyze_features_button.pack(pady=5, fill=tk.X)
        
        optimize_frame = ttk.LabelFrame(analyze_action_frame, text=" Otimizar Modelo ", padding=5)
        optimize_frame.pack(pady=10, fill=tk.X)
        
        tk.Label(optimize_frame, text="Modelo:").pack(side=tk.LEFT)
        
        self.model_to_optimize_combo = ttk.Combobox(
            optimize_frame, state="readonly", width=20
        )
        self.model_to_optimize_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        optimize_button = ttk.Button(
            optimize_frame, text="Otimizar", command=self.start_optimization_thread
        )
        optimize_button.pack(side=tk.LEFT, padx=5)
        
        results_container = ttk.Frame(main_frame)
        results_container.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        corr_frame = ttk.LabelFrame(results_container, text=" Correlação ", padding=5)
        corr_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.analyze_corr_text = ScrolledText(
            corr_frame, height=15, state='disabled', wrap=tk.NONE, font=("Consolas", 8)
        )
        self.analyze_corr_text.pack(fill=tk.BOTH, expand=True)
        
        importance_frame = ttk.LabelFrame(results_container, text=" Importância / Otimização ", padding=5)
        importance_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.analyze_importance_text = ScrolledText(
            importance_frame, height=15, state='disabled', wrap=tk.NONE, font=("Consolas", 8)
        )
        self.analyze_importance_text.pack(fill=tk.BOTH, expand=True)
    # --- Funções das Threads e Fila ---
    def _run_training_pipeline(self, df_hist_raw: pd.DataFrame): 
        training_successful = False
        try:
            # Pré-processamento
            self.gui_queue.put(("progress_update", (30, "Pré-processando...")))
            processed = preprocess_and_feature_engineer(df_hist_raw)
            if processed is None:
                raise ValueError("Falha pré-processamento.")
            X_processed, y_processed, features = processed
            feature_names_for_saving = features

            # Preparação do teste ROI
            self.gui_queue.put(("progress_update", (50, "Preparando teste ROI...")))
            df_hist_interm = calculate_historical_intermediate(df_hist_raw)
            df_hist_aligned = df_hist_interm.loc[X_processed.index.union(y_processed.index)].copy()
            _, X_test_full_data, _, _ = train_test_split(
                df_hist_aligned, y_processed, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_processed
            )

            # Callback de progresso
            def cb(cs, ms, st):
                prog = 60 + int((cs / ms) * 35) if ms > 0 else 95
                self.gui_queue.put(("progress_update", (prog, st)))

            # Início do treinamento
            self.log("Iniciando treino...")
            self.gui_queue.put(("progress_update", (60, "Treinando...")))
            success = run_training_process(
                X_processed, y_processed, X_test_with_odds=X_test_full_data,
                odd_draw_col_name=CONFIG_ODDS_COLS['draw'], progress_callback=cb
            )

            # Finalização
            self.gui_queue.put(("progress_update", (95, "Finalizando...")))
            if success:
                self.gui_queue.put(("training_succeeded", None))
                training_successful = True
            else:
                raise RuntimeError("Falha treino/salvamento.")
        except Exception as e:
            error_msg = f"Erro Treino Thread: {e}"
            self.log(f"ERRO: {error_msg}")
            self.gui_queue.put(("error", ("Erro no Treino", error_msg)))
            self.gui_queue.put(("training_failed", None))
        finally:
            self.gui_queue.put(("progress_end", None))
            self.set_button_state(self.load_train_button, tk.NORMAL)
    def _run_prediction_pipeline(self): 
        prediction_successful = False
        df_preds_final = None
        try:
            # ... (fetch, prepare, predict, FILTROS ) ...
            self.gui_queue.put(("progress_start", (100)))
            fixture_df = fetch_and_process_fixtures()
            if fixture_df is None:
                raise ValueError("Falha buscar CSV.")
            if fixture_df.empty:
                self.log("Nenhum jogo CSV.")
                self.gui_queue.put(("prediction_complete", None))
                return

            self.gui_queue.put(("progress_update", (40, f"Preparando features...")))
            if not self.feature_columns:
                raise ValueError("Features não carregadas.")
            X_fixtures_prepared = prepare_fixture_data(fixture_df, self.historical_data, self.feature_columns)
            if X_fixtures_prepared is None:
                raise ValueError("Falha preparar features.")

            self.gui_queue.put(("progress_update", (70, "Prevendo...")))
            df_preds_raw = predictor.make_predictions(
                self.trained_model, self.trained_scaler, self.feature_columns, X_fixtures_prepared, fixture_df
            )
            if df_preds_raw is None:
                raise RuntimeError("Falha gerar previsões.")
            self.log(f"Previsões brutas: {len(df_preds_raw)}.")

            df_to_filter = df_preds_raw.copy()
            self.log("Aplicando filtros...")
            # ... (Filtro 1 NaNs) ...
            df_preds_final = df_to_filter  # Default
            prob_col = f'Prob_{CLASS_NAMES[1]}'
            if prob_col in df_to_filter.columns:
                threshold = 0.5  # <<< SEU LIMIAR AQUI
                self.log(f"Filtro 2: P(Empate) > {threshold*100:.0f}%")
                df_to_filter[prob_col] = pd.to_numeric(df_to_filter[prob_col], errors='coerce')
                df_filtered = df_to_filter[df_to_filter[prob_col] > threshold].copy()
                self.log(f"{len(df_filtered)} jogos passaram.")
                df_preds_final = df_filtered

            if df_preds_final is not None and not df_preds_final.empty:
                self.log(f"{len(df_preds_final)} previsões finais.")
                prediction_successful = True
            else:
                self.log("Nenhuma previsão restante.")
                prediction_successful = False

            self.gui_queue.put(("progress_update", (95, "Finalizando...")))
            self.gui_queue.put(("prediction_complete", df_preds_final))
        except Exception as e:
            error_msg = f"Erro Previsão: {e}"
            self.log(f"ERRO: {error_msg}")
            self.gui_queue.put(("error", ("Erro Previsão", error_msg)))
            self.gui_queue.put(("prediction_complete", None))
        finally:
            self.gui_queue.put(("progress_end", None))
            self.set_button_state(self.load_train_button, tk.NORMAL)
            if self.trained_model:
                self.set_button_state(self.predict_button, tk.NORMAL)

    def _load_and_analyze_historical_thread(self): 
        try: 
            # Carregar e processar dados históricos
            df_raw = load_historical_data(HISTORICAL_DATA_PATH)
            if df_raw is None: 
                raise ValueError("Falha ao carregar os dados históricos.")
            self.gui_queue.put(("log", "Dados históricos brutos carregados."))
            self.gui_queue.put(("analyze_set_status", "Processando features..."))
            
            # Processar features
            df_processed = calculate_historical_intermediate(df_raw)
            stats_to_roll = ['Ptos', 'VG', 'CG']
            df_processed = calculate_rolling_stats(df_processed, stats_to_roll)
            df_processed = calculate_derived_features(df_processed)
            
            # Garantir que todas as features candidatas estejam presentes
            direct_odds = ['Odd_H_FT', 'Odd_D_FT', 'Odd_Over25_FT', 'Odd_BTTS_Yes']
            for f in direct_odds + FEATURE_COLUMNS:
                if f not in df_processed.columns: 
                    df_processed[f] = np.nan
            
            # Enviar dados processados para a fila da GUI
            self.gui_queue.put(("historical_data_processed", df_processed))
        except Exception as e: 
            error_msg = f"Erro ao carregar/analisar: {e}"
            self.log(f"ERRO: {error_msg}")
            self.gui_queue.put(("error", ("Erro na Análise", error_msg)))
            self.gui_queue.put(("analyze_set_status", "Falha."))

        def _run_feature_analysis(self, X: pd.DataFrame, y: pd.Series): 
            try: 
                # Realizar análise de features
                analysis_result = model_trainer.analyze_features(X, y)
                self.gui_queue.put(("feature_analysis_complete", analysis_result))
            except Exception as e: 
                error_msg = f"Erro na análise de features: {e}"
                self.log(f"ERRO: {error_msg}")
                self.gui_queue.put(("error", ("Erro na Análise", error_msg)))
                self.gui_queue.put(("feature_analysis_complete", None))

        # --- Carregamento Inicial ---
    def load_existing_model_assets(self):
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
                        'path': model_path,
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

    # --- Processamento da Fila GUI ---
    def process_gui_queue(self): 
        try:
            while True:
                try: 
                    message = self.gui_queue.get_nowait()
                    msg_type, msg_payload = message
                except Empty: 
                    break
                except (ValueError, TypeError) as e_unpack: 
                    self.log(f"AVISO GUI: Erro desempacotar: {e_unpack} - Msg: {message}")
                    continue
                except Exception as e_get: 
                    print(f"Erro get fila: {e_get}")
                    continue

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
                elif msg_type == "info": 
                    if isinstance(msg_payload, tuple) and len(msg_payload) == 2: 
                        messagebox.showinfo(*msg_payload)
                elif msg_type == "progress_start": 
                    max_val = msg_payload[0] if isinstance(msg_payload, tuple) and len(msg_payload) > 0 and isinstance(msg_payload[0], int) else 100
                    self.progress_bar.config(maximum=max_val, value=0)
                    self.progress_label.config(text="Iniciando...")
                elif msg_type == "progress_update": 
                    if isinstance(msg_payload, tuple) and len(msg_payload) == 2: 
                        value, status_text = msg_payload
                        self.progress_bar['value'] = value
                        self.progress_label.config(text=status_text)
                elif msg_type == "progress_end": 
                    self.progress_bar['value'] = 0
                    self.progress_label.config(text="Pronto.")
                elif msg_type == "training_succeeded": 
                    self.log("Treino OK. Recarregando...")
                    self.load_existing_model_assets()
                    self.gui_queue.put(("info", ("Treino Concluído", "Modelos salvos.")))
                elif msg_type == "training_failed": 
                    self.trained_model = None
                elif msg_type == "prediction_complete": 
                    df_preds = msg_payload
                    self.log("Processando previsão...")
                    self._update_prediction_display(df_preds)
                elif msg_type == "analyze_set_status": 
                    try: 
                        self.analyze_status_label.config(text=f"Status: {msg_payload}")
                    except (tk.TclError, AttributeError): 
                        pass
                elif msg_type == "historical_data_processed": 
                    self.historical_data_processed = msg_payload
                    self.log("Histórico processado p/ análise.")
                    self.gui_queue.put(("update_analyze_tab_widgets", None))
                elif msg_type == "feature_analysis_complete": 
                    self._update_text_widget(self.analyze_corr_text, "Análise Concluída.")
                    self._update_text_widget(self.analyze_importance_text, "")
                    if msg_payload: 
                        imp_df, corr_matrix = msg_payload
                        corr_text = "Correlação c/ Alvo (IsDraw):\n" + corr_matrix['target_IsDraw'].sort_values(ascending=False).to_string() + "\n\nCorr. Features (Top 15):\n" + corr_matrix.iloc[:15, :15].round(2).to_string()
                        self._update_text_widget(self.analyze_corr_text, corr_text)
                        imp_text = "Importância (RF Rápido):\n" + imp_df.to_string(index=False)
                        self._update_text_widget(self.analyze_importance_text, imp_text)
                        self.log("Análise exibida.")
                    else: 
                        self._update_text_widget(self.analyze_corr_text, "Erro correlação.")
                        self._update_text_widget(self.analyze_importance_text, "Erro importância.")
                        self.log("Falha análise.")
                elif msg_type == "update_analyze_tab_widgets": 
                    self._populate_analyze_widgets()
                else: 
                    self.log(f"AVISO GUI: Msg desconhecida: {msg_type}")
        except Exception as e_queue: 
            print(f"Erro fatal fila GUI: {e_queue}")
            traceback.print_exc()
        finally:
            if self.root.winfo_exists(): 
                self.root.after(100, self.process_gui_queue)

    def on_closing(self): self.log("Fechando..."); self.root.destroy()

# ... (bloco if __name__ == "__main__": como antes) ...
if __name__ == "__main__":
    import requests; import math; import numpy as np; from sklearn.model_selection import train_test_split
    try: from ctypes import windll; windll.shcore.SetProcessDpiAwareness(1)
    except: pass
    root = tk.Tk()
    app = FootballPredictorDashboard(root)
    root.mainloop()
# END OF FILE main.py - V15 (Final Order Fix)