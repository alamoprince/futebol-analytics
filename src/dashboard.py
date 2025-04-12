import traceback

# --- src/dashboard.py ---
# ... (imports como na V11 - garantindo math, numpy, etc.) ...
from rich.console import Console; from rich.table import Table; from rich.panel import Panel; from rich.text import Text; from rich.pretty import pretty_repr; from rich.prompt import Prompt, Confirm; from sklearn.model_selection import train_test_split; import pandas as pd; import sys, os, datetime, requests, math, numpy as np; from datetime import date, timedelta
# Adiciona paths
SRC_DIR = os.path.dirname(os.path.abspath(__file__)); BASE_DIR = os.path.dirname(SRC_DIR); #... (add paths) ...
if SRC_DIR not in sys.path: sys.path.append(SRC_DIR); 
if BASE_DIR not in sys.path: sys.path.append(BASE_DIR)
try:
    from config import ( HISTORICAL_DATA_PATH, BEST_F1_MODEL_SAVE_PATH, BEST_ROI_MODEL_SAVE_PATH, MODEL_ID_F1, MODEL_ID_ROI, CLASS_NAMES, ODDS_COLS, OTHER_ODDS_NAMES, FIXTURE_FETCH_DAY, GITHUB_TOKEN, GITHUB_REPO_NAME, GITHUB_PREDICTIONS_PATH, MODEL_TYPE_NAME, TEST_SIZE, RANDOM_STATE )
    from data_handler import ( load_historical_data, preprocess_and_feature_engineer, fetch_and_process_fixtures, prepare_fixture_data, calculate_historical_intermediate )
    from model_trainer import train_evaluate_and_save_best_models as run_training_process # Renomeia
    import predictor; from github_manager import GitHubManager; from typing import Optional, Any, List, Dict
except ImportError as e: print(f"Erro import: {e}"); sys.exit(1)
except NameError as ne: #... (erro GitHub) ...
     if 'GITHUB_TOKEN' in str(ne) or 'GITHUB_REPO_NAME' in str(ne): print("\nERRO CONFIG GITHUB...\n");
     else: print(f"Erro nome: {ne}"); sys.exit(1)

console = Console()

class CLIDashboard:
    # ... (__init__, _init_github_manager, display_model_stats como V11) ...
    def __init__(self): # ... (como V11, inicializa com loaded_models_data) ...
        self.historical_data = None; self.loaded_models_data: Dict[str, Dict] = {}; self.available_model_ids: List[str] = []; self.selected_model_id: Optional[str] = None; self.trained_model = None; self.trained_scaler = None; self.feature_columns = None; self.github_manager = None; self.model_best_params = None; self.model_eval_metrics = None; self.model_file_timestamp = None; console.print(f"[bold green]⚽ Futebol Predictor ({MODEL_TYPE_NAME}) ⚽[/]"); console.print(f"Carregando modelos e histórico..."); self.load_existing_model_assets()
    def _init_github_manager(self) -> bool: # ... (como V11) ...
        if self.github_manager:
            return True
        if not GITHUB_TOKEN or not GITHUB_REPO_NAME:
            return False
        try:
            console.print("[cyan]Conectando GitHub...[/]")
            self.github_manager = GitHubManager()
            return True
        except Exception as e:
            console.print(f"[red]Erro GitHub: {e}[/]")
            self.github_manager = None
            return False

    def display_model_stats(self, model_id_to_display: Optional[str] = None): 
        """Exibe stats do modelo selecionado ou especificado."""
        model_to_show_id = model_id_to_display or self.selected_model_id  # Usa o selecionado se nenhum for passado
        if not model_to_show_id or model_to_show_id not in self.loaded_models_data:
            console.print(f"[yellow]Modelo '{model_to_show_id or 'Nenhum'}' não carregado/selecionado.[/]")
            return

        model_data = self.loaded_models_data[model_to_show_id]
        model = model_data.get('model')
        metrics = model_data.get('metrics')
        params = model_data.get('params')
        features = model_data.get('features')
        timestamp = model_data.get('timestamp')
        path = model_data.get('path')

        content = Text()
        content.append(f"Stats Modelo: {model_to_show_id}\n", style="bold underline")
        content.append(f"Arquivo: {os.path.basename(path or 'N/A')}\n")
        content.append(f"Modif.: {timestamp or 'N/A'}\n")
        content.append(f"Tipo: {model.__class__.__name__ if model else 'N/A'}\n---\n")

        if features:
            content.append(f"Features ({len(features)}):\n - " + "\n - ".join(features) + "\n---\n")

        if params:
            content.append("Params:\n" + pretty_repr(params) + "\n---\n")

        if metrics:
            content.append("Métricas (Teste):\n", style="bold")
            acc = metrics.get('accuracy')
            loss = metrics.get('log_loss')
            auc = metrics.get('roc_auc')
            prec_d = metrics.get('precision_draw')
            rec_d = metrics.get('recall_draw')
            f1_d = metrics.get('f1_score_draw')
            conf_matrix = metrics.get('confusion_matrix')
            profit = metrics.get('profit')
            roi = metrics.get('roi')
            n_bets = metrics.get('num_bets')
            train_n = metrics.get('train_set_size', 'N/A')
            test_n = metrics.get('test_set_size', 'N/A')

            content.append(f"- Acurácia: {acc:.4f}\n" if acc is not None else "- Acurácia: N/A\n")
            content.append(f"- Log Loss: {loss:.4f}\n" if loss is not None and not math.isnan(loss) else "- Log Loss: N/A\n")
            content.append(f"- ROC AUC: {auc:.4f}\n" if auc is not None else "- ROC AUC: N/A\n")
            content.append("- Métricas 'Empate':\n")
            content.append(f"  - Precision: {prec_d:.4f}\n" if prec_d is not None else "  - Precision: N/A\n")
            content.append(f"  - Recall:    {rec_d:.4f}\n" if rec_d is not None else "  - Recall: N/A\n")
            content.append(f"  - F1-Score:  {f1_d:.4f}\n" if f1_d is not None else "  - F1-Score: N/A\n")

            if conf_matrix and isinstance(conf_matrix, list) and len(conf_matrix) == 2 and len(conf_matrix[0]) == 2:
                content.append("- Matriz Confusão:\n")
                content.append(f"      Prev:ÑEmp| Prev:Emp\n")
                content.append(f"Real:ÑEmp {conf_matrix[0][0]:<6d}| {conf_matrix[0][1]:<6d}\n")
                content.append(f"Real:Emp  {conf_matrix[1][0]:<6d}| {conf_matrix[1][1]:<6d}\n")

            content.append(f"- Amostras T/T: {train_n} / {test_n}\n\n")
            content.append("Estratégia BackDraw:\n", style="bold")
            content.append(f"- Nº Apostas: {n_bets if n_bets is not None else 'N/A'}\n")
            content.append(f"- Profit: {profit:.2f} u\n" if profit is not None else "- Profit: N/A\n")
            content.append(f"- ROI: {roi:.2f} %\n" if roi is not None else "- ROI: N/A\n")
        else:
            content.append("Métricas: [yellow]Não disponíveis[/]\n")

        console.print(Panel(content, border_style="cyan", title=f"Status Modelo: {model_to_show_id}"))

    # --- load_existing_model_assets (Adaptado para carregar AMBOS) ---
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
            console.print(f"Tentando carregar {model_id} de {model_path}...")
            load_result = predictor.load_model_scaler_features(model_path)
            if load_result:
                model, scaler, features, params, metrics, timestamp = load_result  # Desempacota 6
                if model and features:
                    console.print(f" -> Sucesso: {model_id} ({model.__class__.__name__}). Métricas: {'Sim' if metrics else 'NÃO'}")
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
                    console.print(f" -> Falha: Arquivo {model_id} inválido.")
            else:
                console.print(f" -> Falha: Arquivo {model_id} não encontrado/erro.")

        if self.available_model_ids:
            self.selected_model_id = default_selection  # Define padrão
            self.log(f"Modelos disponíveis: {self.available_model_ids}. Padrão: {default_selection}")
            # Atualiza estado global com dados do modelo padrão
            self.trained_model = self.loaded_models_data[default_selection].get('model')
            self.trained_scaler = self.loaded_models_data[default_selection].get('scaler')
            self.feature_columns = self.loaded_models_data[default_selection].get('features')
            self.model_best_params = self.loaded_models_data[default_selection].get('params')
            self.model_eval_metrics = self.loaded_models_data[default_selection].get('metrics')
            self.model_file_timestamp = self.loaded_models_data[default_selection].get('timestamp')
            self.display_model_stats(self.selected_model_id)  # Mostra stats do padrão
        else:
            console.print("Nenhum modelo pré-treinado válido encontrado.")
            self.selected_model_id = None
            self.trained_model = None  # Limpa estado

        console.print("Carregando históricos...")
        df_hist = load_historical_data(HISTORICAL_DATA_PATH)
        if df_hist is not None:
            self.historical_data = df_hist
            console.print("[green]Históricos OK.[/]")
        else:
            console.print("Falha carregar históricos.")
            any_model_loaded = False
            self.trained_model = None

        if self.selected_model_id and self.historical_data is not None:
            console.print("Pronto para prever.")
        else:
            console.print("[yellow]Não pronto para prever (sem modelo ou histórico).[/]")

    # --- train_model (Chama função que salva 2) ---
    def train_model(self):
        console.print(f"\n[bold cyan]--- Treinamento Múltiplo ({MODEL_TYPE_NAME}) ---[/]")
        self.trained_model=None; self.model_best_params=None; self.model_eval_metrics=None; self.loaded_models_data = {}; self.available_model_ids = []; self.selected_model_id = None # Limpa tudo

        try:
            console.print(f"Carregando histórico: {HISTORICAL_DATA_PATH}...")
            df_hist_loaded = load_historical_data(HISTORICAL_DATA_PATH)
            if df_hist_loaded is None: raise ValueError("Falha carregar histórico.")
            self.historical_data = df_hist_loaded

            # Callback simples para console
            def cli_progress_callback(current_step, max_steps, status_text):
                console.print(f"  Progresso Treino: {current_step+1}/{max_steps} - {status_text}")

            console.print("Iniciando pré-processamento, treino e seleção...")
            # Chama a função que faz tudo e salva os dois melhores
            success = run_training_process( # Usa alias do import
                self.historical_data, # Passa histórico bruto
                X_test_with_odds=self.historical_data, # Requer ajuste se histórico não tiver todas as colunas pós-cálculo
                                                      # Vamos recalcular X_test_full dentro do trainer por enquanto
                progress_callback=cli_progress_callback
            )

            if success:
                console.print("\n[green]Treinamento e salvamento concluídos. Recarregando modelos...[/]")
                # Recarrega os modelos que foram salvos
                self.load_existing_model_assets()
            else:
                console.print("[bold red]Falha geral no processo de treinamento/salvamento.[/]")

        except Exception as e: console.print(f"[bold red]Erro Treinamento: {e}[/]"); import traceback; traceback.print_exc()


    # --- predict_fixtures (COM FILTROS) ---
    def predict_fixtures(self):
        console.print(f"\n[bold cyan]--- Previsão {MODEL_TYPE_NAME} (Modelo: {self.selected_model_id or 'Nenhum'}) ---[/]")
        if self.trained_model is None or not self.selected_model_id:
            console.print("[yellow]Nenhum modelo selecionado.[/]")
            return
        if self.historical_data is None:
            console.print("[yellow]Histórico ausente.[/]")
            return

        try:
            console.print(f"Buscando CSV ({FIXTURE_FETCH_DAY})...")
            fixture_df = fetch_and_process_fixtures()
            if fixture_df is None:
                console.print("[red]Falha CSV.[/]")
                return
            if fixture_df.empty:
                console.print("[yellow]Nenhum jogo CSV.[/]")
                return

            console.print(f"[green]{len(fixture_df)} jogos processados.")
            console.print("Preparando features...")
            X_fixtures_prepared = prepare_fixture_data(fixture_df, self.historical_data, self.feature_columns)
            if X_fixtures_prepared is None or X_fixtures_prepared.empty:
                raise ValueError("Falha preparar features.")

            console.print(f"Prevendo com {self.trained_model.__class__.__name__}...")
            df_predictions = predictor.make_predictions(
                self.trained_model, self.trained_scaler, self.feature_columns, X_fixtures_prepared, fixture_df
            )
            if df_predictions is None:
                raise RuntimeError("Falha gerar previsões.")
            console.print(f"Previsões brutas: {len(df_predictions)} jogos.")

            # --- FILTROS ---
            console.print("Aplicando filtros...")
            # Filtro 1: Odds de Input
            input_odd_features = ['Odd_H_FT', 'Odd_D_FT', 'Odd_A_FT', 'Odd_Over25_FT', 'Odd_BTTS_Yes']
            cols_to_check_nan = [c for c in input_odd_features if c in df_predictions.columns]
            if cols_to_check_nan:
                initial_rows_f1 = len(df_predictions)
                df_predictions_f1 = df_predictions.dropna(subset=cols_to_check_nan)
                rows_dropped_f1 = initial_rows_f1 - len(df_predictions_f1)
                if rows_dropped_f1 > 0:
                    console.print(f"  Filtro 1: Removidos {rows_dropped_f1} jogos (odds input NaN).")
                df_predictions = df_predictions_f1

            # Filtro 2: Probabilidade Empate < 87%
            prob_col_draw = f'Prob_{CLASS_NAMES[1]}'  # Prob_Empate
            if prob_col_draw in df_predictions.columns:
                threshold = 0.87
                initial_rows_f2 = len(df_predictions)
                df_predictions_f2 = df_predictions[df_predictions[prob_col_draw] < threshold]
                rows_dropped_f2 = initial_rows_f2 - len(df_predictions_f2)
                if rows_dropped_f2 > 0:
                    console.print(f"  Filtro 2: Removidos {rows_dropped_f2} jogos (P(Empate) >= {threshold*100:.0f}%).")
                df_predictions_final = df_predictions_f2
            else:
                console.print(f"Aviso: Coluna '{prob_col_draw}' não encontrada para filtro.")
                df_predictions_final = df_predictions
            # -------------

            if df_predictions_final is not None and not df_predictions_final.empty:
                console.print(f"\n[bold]--- {len(df_predictions_final)} Previsões Filtradas ---[/]")
                # ... (código da tabela Rich como V11) ...
                table = Table(show_header=True, header_style="bold magenta", title="Probabilidades: Não Empate vs Empate")
                display_cols_base = ['Date_Str', 'Time_Str', 'League', 'HomeTeam', 'AwayTeam']
                odds_cols_display_names = [
                    ODDS_COLS['home'], ODDS_COLS['draw'], ODDS_COLS['away'], 'Odd_Over25_FT', 'Odd_BTTS_Yes'
                ]
                odds_cols_display = [c for c in odds_cols_display_names if c in df_predictions_final.columns]
                prob_cols_display = [c for c in df_predictions_final.columns if c.startswith('Prob_')]
                valid_cols_cli = display_cols_base + odds_cols_display + prob_cols_display
                valid_cols_cli = [c for c in valid_cols_cli if c in df_predictions_final.columns]

                for col in valid_cols_cli:
                    header = col.replace('Date_Str', 'Data').replace('Time_Str', 'Hora').replace('HomeTeam', 'Casa').replace('AwayTeam', 'Fora')
                    if col == ODDS_COLS['home']:
                        header = 'Odd H'
                    elif col == ODDS_COLS['draw']:
                        header = 'Odd D'
                    elif col == ODDS_COLS['away']:
                        header = 'Odd A'
                    elif col == 'Odd_Over25_FT':
                        header = 'O2.5'
                    elif col == 'Odd_BTTS_Yes':
                        header = 'BTTS S'
                    elif col == 'Prob_Nao_Empate':
                        header = 'P(Ñ Emp)'
                    elif col == 'Prob_Empate':
                        header = 'P(Empate)'
                    justify = "left" if col in ['League', 'HomeTeam', 'AwayTeam'] else "center"
                    style = "dim" if col in ['Date_Str', 'Time_Str'] else (
                        "cyan" if col.startswith('Odd_') else (
                            "yellow" if col == 'Prob_Nao_Empate' else (
                                "bold green" if col == 'Prob_Empate' else ""
                            )
                        )
                    )
                    table.add_column(header, style=style, justify=justify, overflow="fold", min_width=6)

                df_display_cli = df_predictions_final[valid_cols_cli].copy()
                for pcol in prob_cols_display:
                    df_display_cli[pcol] = (pd.to_numeric(df_display_cli[pcol], errors='coerce') * 100).round(1).astype(str) + '%'
                    df_display_cli[pcol] = df_display_cli[pcol].replace('nan%', '-', regex=False)
                for ocol in odds_cols_display:
                    df_display_cli[ocol] = pd.to_numeric(df_display_cli[ocol], errors='coerce').apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
                for _, row in df_display_cli.iterrows():
                    table.add_row(*[str(row.get(col, '')) for col in valid_cols_cli])
                console.print(table)

                # Salvar no GitHub (opcional, salva PREVISÕES FILTRADAS)
                save_to_github = Confirm.ask("\n[bold]Salvar previsões FILTRADAS no GitHub?[/]", default=False)
                if save_to_github and self._init_github_manager() and self.github_manager:
                    fetch_date = date.today() + timedelta(days=(1 if FIXTURE_FETCH_DAY == 'tomorrow' else 0))
                    fetch_date_str = fetch_date.strftime("%Y-%m-%d")
                    filename = f"previsoes_{MODEL_TYPE_NAME}_filtradas_{fetch_date_str}.csv"
                    github_path = f"{GITHUB_PREDICTIONS_PATH.strip('/')}/{filename}"
                    console.print(f"Salvando em [cyan]{GITHUB_REPO_NAME}/{github_path}[/]...")
                    success = self.github_manager.save_or_update_github(
                        df=df_predictions_final, path=github_path,
                        commit_message_prefix=f"Previsões Filtradas {MODEL_TYPE_NAME} {fetch_date_str}: "
                    )
                    # ... (print sucesso/falha)...

            else:
                console.print("[yellow]Nenhuma previsão restante após filtros.[/]")
        except Exception as e:
            console.print(f"[bold red]Erro Previsão: {e}[/]")
            traceback.print_exc()

    # --- run (menu principal com seleção de modelo para exibir stats) ---
    def run(self):
        while True:
            console.print(f"\n[bold]Menu ({MODEL_TYPE_NAME}):[/]")
            console.print("  [1] Treinar/Retreinar Modelos")
            predict_option = f"  [2] Prever Jogos de '{FIXTURE_FETCH_DAY}'"
            if not self.trained_model: predict_option += " ([yellow]Modelo não carregado/selecionado[/])"
            else: predict_option += f" (Usando: [cyan]{self.selected_model_id}[/])" # Mostra modelo ativo
            console.print(predict_option)
            console.print("  [3] Exibir Estatísticas do Modelo")
            console.print("  [4] Sair")

            default_choice = "2" if self.trained_model else "1"
            choice = Prompt.ask("[bold]Escolha:[/]", choices=["1", "2", "3", "4"], default=default_choice)

            if choice == "1": self.train_model()
            elif choice == "2":
                if self.trained_model: self.predict_fixtures()
                else: console.print("[yellow]Nenhum modelo disponível. Treine (1).[/]")
            elif choice == "3":
                # Permite escolher qual modelo mostrar stats
                if self.available_model_ids:
                    console.print("\nModelos disponíveis para exibir stats:")
                    model_choices = {str(idx+1): model_id for idx, model_id in enumerate(self.available_model_ids)}
                    for idx_str, model_id_str in model_choices.items():
                        console.print(f"  [{idx_str}] {model_id_str}")
                    stat_choice = Prompt.ask("Qual modelo?", choices=list(model_choices.keys()), default="1")
                    self.display_model_stats(model_choices[stat_choice]) # Passa o ID escolhido
                else:
                     console.print("[yellow]Nenhum modelo carregado para exibir stats.[/]")
            elif choice == "4": console.print("[yellow]Saindo...[/]"); break

if __name__ == "__main__": # ... (como V12) ...
    import requests; import math; import numpy as np; from sklearn.model_selection import train_test_split
    cli = CLIDashboard(); 
    try: cli.run()
    except KeyboardInterrupt: console.print("\n[yellow]Interrompido.[/]")
    except Exception as e: console.print(f"\n[red]Erro: {e}[/]"); import traceback; traceback.print_exc()