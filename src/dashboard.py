from config import ( 
        HISTORICAL_DATA_PATH, MODEL_SAVE_PATH, ODDS_COLS, FIXTURE_FETCH_DAY,
        GITHUB_TOKEN, GITHUB_REPO_NAME, GITHUB_PREDICTIONS_PATH, MODEL_TYPE_NAME,
        TEST_SIZE, RANDOM_STATE 
)
from data_handler import (load_historical_data, preprocess_and_feature_engineer, 
        calculate_historical_intermediate, fetch_and_process_fixtures, prepare_fixture_data
)
import model_trainer
import predictor
from github_manager import GitHubManager
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.pretty import pretty_repr
from rich.prompt import Prompt, Confirm
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
import os
import datetime
from datetime import date, timedelta
import math


# Adiciona diretórios ao path

SRC_DIR = os.path.dirname(os.path.abspath(__file__)); BASE_DIR = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path: sys.path.append(SRC_DIR);
if BASE_DIR not in sys.path: sys.path.append(BASE_DIR)

console = Console()

class CLIDashboard:
    def __init__(self):

        """Inicializa variáveis de instância"""

        self.historical_data = None; self.trained_model = None; self.trained_scaler = None; self.feature_columns = None
        self.github_manager = None; self.model_best_params = None; self.model_eval_metrics = None; self.model_file_timestamp = None
        console.print(f"[bold green]⚽ Futebol Predictor ({MODEL_TYPE_NAME}) ⚽[/]"); console.print(f"Carregando modelo e histórico...")
        self.load_existing_model_assets()

    def _init_github_manager(self) -> bool: 

        """Inicializa o GitHubManager se não estiver já inicializado."""

        if self.github_manager: return True;
        if not GITHUB_TOKEN or not GITHUB_REPO_NAME: return False
        try: console.print("[cyan]Conectando GitHub...[/]"); self.github_manager = GitHubManager(); return True
        except Exception as e: console.print(f"[red]Erro GitHub Init: {e}[/]"); self.github_manager = None; return False

    def display_model_stats(self): 

        """Exibe as estatísticas do modelo treinado."""

        if self.trained_model is None: console.print("[yellow]Nenhum modelo carregado.[/]"); return
        content = Text(); content.append(f"Stats Modelo ({MODEL_TYPE_NAME})\n", style="bold underline"); content.append(f"Arquivo: {os.path.basename(MODEL_SAVE_PATH)}\n"); content.append(f"Modif.: {self.model_file_timestamp or 'N/A'}\n"); content.append(f"Tipo: {self.trained_model.__class__.__name__}\n---\n")
        if self.feature_columns: content.append(f"Features ({len(self.feature_columns)}):\n - " + "\n - ".join(self.feature_columns) + "\n---\n")
        if self.model_best_params: content.append("Melhores Parâmetros:\n" + pretty_repr(self.model_best_params) + "\n---\n")
        if self.model_eval_metrics:
            content.append("Métricas Avaliação (Teste):\n", style="bold"); acc = self.model_eval_metrics.get('accuracy'); loss = self.model_eval_metrics.get('log_loss'); auc = self.model_eval_metrics.get('roc_auc'); prec_d = self.model_eval_metrics.get('precision_draw'); rec_d = self.model_eval_metrics.get('recall_draw'); f1_d = self.model_eval_metrics.get('f1_score_draw'); conf_matrix = self.model_eval_metrics.get('confusion_matrix'); profit = self.model_eval_metrics.get('profit'); roi = self.model_eval_metrics.get('roi'); n_bets = self.model_eval_metrics.get('num_bets'); train_n = self.model_eval_metrics.get('train_set_size', 'N/A'); test_n = self.model_eval_metrics.get('test_set_size', 'N/A')
            content.append(f"- Acurácia: {acc:.4f}\n" if acc is not None else "- Acurácia: N/A\n"); content.append(f"- Log Loss: {loss:.4f}\n" if loss is not None and not math.isnan(loss) else "- Log Loss: N/A\n"); content.append(f"- ROC AUC: {auc:.4f}\n" if auc is not None else "- ROC AUC: N/A\n"); content.append("- Métricas 'Empate':\n"); content.append(f"  - Precision: {prec_d:.4f}\n" if prec_d is not None else "  - Precision: N/A\n"); content.append(f"  - Recall:    {rec_d:.4f}\n" if rec_d is not None else "  - Recall: N/A\n"); content.append(f"  - F1-Score:  {f1_d:.4f}\n" if f1_d is not None else "  - F1-Score: N/A\n")
            if conf_matrix and isinstance(conf_matrix, list) and len(conf_matrix)==2 and len(conf_matrix[0])==2: content.append("- Matriz Confusão:\n"); content.append(f"      Prev:ÑEmp| Prev:Emp\n"); content.append(f"Real:ÑEmp {conf_matrix[0][0]:<6d}| {conf_matrix[0][1]:<6d}\n"); content.append(f"Real:Emp  {conf_matrix[1][0]:<6d}| {conf_matrix[1][1]:<6d}\n")
            content.append(f"- Amostras T/T: {train_n} / {test_n}\n\n"); content.append("Estratégia BackDraw:\n", style="bold"); content.append(f"- Nº Apostas: {n_bets if n_bets is not None else 'N/A'}\n"); content.append(f"- Profit: {profit:.2f} u\n" if profit is not None else "- Profit: N/A\n"); content.append(f"- ROI: {roi:.2f} %\n" if roi is not None else "- ROI: N/A\n")
        else: content.append("Métricas: [yellow]Não disponíveis[/]\n")
        console.print(Panel(content, border_style="magenta", title=f"Status Modelo {MODEL_TYPE_NAME}"))

    def load_existing_model_assets(self):

        """Importa o modelo e histórico existentes."""

        model_loaded_success = False; self.trained_model = None; self.trained_scaler = None; self.feature_columns = None; self.model_best_params = None; self.model_eval_metrics = None; self.model_file_timestamp = None
        load_result = predictor.load_model_scaler_features(MODEL_SAVE_PATH)
        if load_result:
            model, scaler, features, params, metrics, timestamp = load_result # Desempacota 6
            if model and features: self.trained_model = model; self.trained_scaler = scaler; self.feature_columns = features; self.model_best_params = params; self.model_eval_metrics = metrics; self.model_file_timestamp = timestamp; console.print(f"[green]Modelo {MODEL_TYPE_NAME} e stats carregados.[/]"); model_loaded_success = True; self.display_model_stats()
            else: console.print("[yellow]Arquivo de modelo inválido.[/]")
        else: console.print("[cyan]Nenhum modelo pré-treinado encontrado.[/]")
        console.print("Carregando histórico..."); 
        try:
             df_hist = load_historical_data(HISTORICAL_DATA_PATH);
             if df_hist is not None: self.historical_data = df_hist; console.print("[green]Histórico carregado.[/]")
             else: raise ValueError("Histórico None.")
        except Exception as e_hist: console.print(f"[red]Erro CRÍTICO Histórico: {e_hist}[/]"); self.trained_model = None
        if model_loaded_success and self.historical_data is not None: console.print(f"[bold green]Pronto para prever ({MODEL_TYPE_NAME}).[/]")


    # --- train_model ---
    def train_model(self):
        console.print(f"\n[bold cyan]--- Treinamento Modelo {MODEL_TYPE_NAME} ---[/]")
        self.trained_model=None; self.model_best_params=None; self.model_eval_metrics=None

        try:
            console.print(f"Carregando histórico: {HISTORICAL_DATA_PATH}...")
            df_hist_loaded = load_historical_data(HISTORICAL_DATA_PATH)
            if df_hist_loaded is None: raise ValueError("Falha ao carregar histórico.")
            X_processed, y_processed, features_used = preprocess_and_feature_engineer(df_hist_loaded)
            self.feature_columns = features_used
            console.print(f"Dados prontos. {len(features_used)} Features.")
            self.historical_data = df_hist_loaded # Guarda ref completa
            

            console.print("Pré-processando (BackDraw)...")
            # Chama a função PÚBLICA de pré-processamento
            df_hist_interm = calculate_historical_intermediate(df_hist_loaded) 
            df_hist_aligned = df_hist_interm.loc[X_processed.index.union(y_processed.index)].copy()
            _, X_test_full, _, _ = train_test_split(df_hist_aligned, y_processed, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_processed)

            console.print("Iniciando treinamento e avaliação...")
            train_result = model_trainer.train_evaluate_model(
                X_processed, y_processed,
                use_time_series_split=False,
                X_test_with_odds=X_test_full,
                odd_draw_col_name=ODDS_COLS['draw']
            )

            if train_result:
                model, scaler, _, params, metrics = train_result
                console.print("[green]Treinamento BackDraw concluído.[/]")
                self.trained_model = model; self.trained_scaler = scaler
                self.model_best_params = params; self.model_eval_metrics = metrics
                self.model_file_timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " (Treinado)"
                model_trainer.save_model_scaler_features(model, scaler, self.feature_columns, params, metrics, MODEL_SAVE_PATH)
                console.print(f"Melhor Modelo ({model.__class__.__name__}) e Stats salvos.")
                self.display_model_stats() # Mostra stats novas
            
            else:
                console.print("[bold red]Falha no treinamento (train_evaluate_model retornou None).[/]")
                # Limpa estado se falhar
                self.trained_model=None; self.model_best_params=None; self.model_eval_metrics=None

        except Exception as e:
             console.print(f"[bold red]Erro Durante Treinamento: {e}[/]")
             import traceback; traceback.print_exc()
             # Limpa estado em caso de erro
             self.trained_model=None; self.model_best_params=None; self.model_eval_metrics=None


    def predict_fixtures(self): 

        """Calcula previsões para jogos futuros."""

        console.print(f"\n[bold cyan]--- Previsão {MODEL_TYPE_NAME} (Fonte CSV) ---[/]");
        if self.trained_model is None: console.print("[yellow]Modelo ausente.[/]"); return;
        if self.historical_data is None: console.print("[yellow]Histórico ausente.[/]"); return
        try:
            console.print(f"Buscando CSV ({FIXTURE_FETCH_DAY})..."); fixture_df = fetch_and_process_fixtures()
            if fixture_df is None: console.print("[red]Falha CSV.[/]"); return;
            if fixture_df.empty: console.print("[yellow]Nenhum jogo CSV.[/]"); return
            console.print(f"[green]{len(fixture_df)} jogos processados."); console.print("Preparando features...")
            X_fixtures_prepared = prepare_fixture_data(fixture_df, self.historical_data, self.feature_columns)
            if X_fixtures_prepared is None or X_fixtures_prepared.empty: raise ValueError("Falha preparar features.")
            console.print(f"Prevendo..."); df_predictions = predictor.make_predictions(self.trained_model, self.trained_scaler, self.feature_columns, X_fixtures_prepared, fixture_df)
            if df_predictions is not None and not df_predictions.empty:
                console.print("\n[bold]--- Previsões ---[/]"); table = Table(show_header=True, header_style="bold magenta", title="Probs: Ñ Emp vs Emp")
                
                # Adiciona colunas de odds e probabilidade
                display_cols_base = ['Date_Str', 'Time_Str', 'League', 'HomeTeam', 'AwayTeam']; odds_cols_display_names = [ODDS_COLS['home'], ODDS_COLS['draw'], ODDS_COLS['away'], 'Odd_Over25_FT', 'Odd_BTTS_Yes']; odds_cols_display = [c for c in odds_cols_display_names if c in df_predictions.columns]; prob_cols_display = [c for c in df_predictions.columns if c.startswith('Prob_')]; valid_cols_cli = display_cols_base + odds_cols_display + prob_cols_display; valid_cols_cli = [c for c in valid_cols_cli if c in df_predictions.columns]
                for col in valid_cols_cli: header=col.replace('Date_Str','Data').replace('Time_Str','Hora').replace('HomeTeam','Casa').replace('AwayTeam','Fora'); #... (resto dos headers) ...
                if col == ODDS_COLS['home']: header='Odd H'; 
                elif col == ODDS_COLS['draw']: header='Odd D'; 
                elif col == ODDS_COLS['away']: header='Odd A'; 
                elif col == 'Odd_Over25_FT': header='O2.5'; 
                elif col == 'Odd_BTTS_Yes': header='BTTS S'; 
                elif col=='Prob_Nao_Empate': header='P(Ñ Emp)'; 
                elif col=='Prob_Empate': header='P(Empate)'
                justify = "left" if col in ['League', 'HomeTeam', 'AwayTeam'] else "center"; style = "dim" if col in ['Date_Str', 'Time_Str'] else ("cyan" if col.startswith('Odd_') else ("yellow" if col=='Prob_Nao_Empate' else ("bold green" if col=='Prob_Empate' else ""))); table.add_column(header, style=style, justify=justify, overflow="fold", min_width=6)
                df_display_cli = df_predictions[valid_cols_cli].copy();
                for pcol in prob_cols_display: df_display_cli[pcol] = (df_display_cli[pcol] * 100).round(1).astype(str) + '%'
                for ocol in odds_cols_display: df_display_cli[ocol] = df_display_cli[ocol].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
                for _, row in df_display_cli.iterrows(): table.add_row(*[str(row.get(col, '')) for col in valid_cols_cli]); console.print(table)
                # Salvar no GitHub (como antes)
                save_to_github = Confirm.ask("\n[bold]Salvar previsões no GitHub?[/]", default=False)
                if save_to_github and self._init_github_manager() and self.github_manager: # ... (código de salvar como V11) ...
                     fetch_date = date.today() + timedelta(days=(1 if FIXTURE_FETCH_DAY == 'tomorrow' else 0)); fetch_date_str = fetch_date.strftime("%Y-%m-%d"); filename = f"previsoes_{MODEL_TYPE_NAME}_{fetch_date_str}.csv"; github_path = f"{GITHUB_PREDICTIONS_PATH.strip('/')}/{filename}"; console.print(f"Salvando em [cyan]{GITHUB_REPO_NAME}/{github_path}[/]..."); success = self.github_manager.save_or_update_github(df=df_predictions, path=github_path, commit_message_prefix=f"Previsões {MODEL_TYPE_NAME} {fetch_date_str}: "); #... (print sucesso/falha)...

            elif df_predictions is not None: console.print("[yellow]Nenhuma previsão gerada.[/]")
            else: raise RuntimeError("Falha ao gerar previsões.")
        except Exception as e: console.print(f"[bold red]Erro Previsão: {e}[/]"); import traceback; traceback.print_exc()


    def run(self):

        """Executa o loop principal do menu interativo."""

        while True:
            console.print("\n[bold]Menu Principal:[/]")
            console.print("  [1] Treinar/Retreinar Modelo")
            console.print(f"  [2] Prever Jogos de '{FIXTURE_FETCH_DAY}'")
            console.print("  [3] Exibir Estatísticas do Modelo")
            console.print("  [4] Sair")

            # --- VERIFICAÇÃO IMPORTANTE ---
            # A sugestão padrão depende se self.trained_model FOI definido no treino
            default_choice = "2" if self.trained_model else "1"
            # -----------------------------

            choice = Prompt.ask(
                "[bold]Escolha:[/]",
                choices=["1", "2", "3", "4"],
                default=default_choice # Usa a variável calculada
            )

            if choice == "1":
                self.train_model()
                # Após treinar, o loop recomeça, e default_choice será recalculado
            elif choice == "2":
                # --- VERIFICAÇÃO IMPORTANTE ---
                # Garante que o modelo existe ANTES de tentar prever
                if self.trained_model:
                    self.predict_fixtures()
                else:
                    console.print("[bold yellow]Nenhum modelo treinado disponível. Treine um modelo primeiro (Opção 1).[/]")
            elif choice == "3":
                self.display_model_stats()
            elif choice == "4":
                console.print("[bold yellow]Saindo...[/]"); break

if __name__ == "__main__": 
    import requests; import math; import numpy as np; from sklearn.model_selection import train_test_split # Garante imports
    cli = CLIDashboard(); 
    try: cli.run()
    except KeyboardInterrupt: console.print("\n[yellow]Interrompido.[/]")
    except Exception as e: console.print(f"\n[red]Erro: {e}[/]"); import traceback; traceback.print_exc()