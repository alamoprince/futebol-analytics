# --- src/model_trainer.py ---
# Código Completo com Correções para Erros de Treino/Avaliação

import pandas as pd
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
try:
    import lightgbm as lgb
    LGBMClassifier = lgb.LGBMClassifier
except ImportError:
    print("AVISO: LightGBM não instalado.")
    lgb = None
    LGBMClassifier = None
from sklearn.metrics import (accuracy_score, classification_report, log_loss,
                             precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import os
import datetime
import numpy as np
import logging
import traceback # Import traceback for detailed error logging

# Tenta importar funções e configs necessárias
try:
    # Removido data_handler import aqui, roi é definido abaixo
    # from data_handler import roi
    from config import (
        RANDOM_STATE, TEST_SIZE, MODEL_CONFIG, CLASS_NAMES,
        BEST_F1_MODEL_SAVE_PATH, BEST_ROI_MODEL_SAVE_PATH, # NOVOS Paths
        CROSS_VALIDATION_SPLITS, N_JOBS_GRIDSEARCH, FEATURE_COLUMNS,
        ODDS_COLS, BEST_MODEL_METRIC # BEST_MODEL_METRIC ainda usado para logs
    )
    from typing import Any, Optional, Dict, Tuple, List, Callable # Garante imports de typing
except ImportError as e:
     print(f"Erro crítico import config/typing em model_trainer.py: {e}")
     raise # Re-levanta o erro, pois é essencial

logging.basicConfig(level=logging.INFO, format='%(asctime)s - TRAINER - %(levelname)s - %(message)s')

# --- Função ROI (Definida localmente ou importada de forma segura) ---
#    Para evitar dependência circular, é mais seguro definir aqui se só for usada aqui.
def roi(y_test: pd.Series, y_pred: np.ndarray, X_test_odds_aligned: Optional[pd.DataFrame], odd_draw_col_name: str) -> Optional[float]:
        """Calcula o Retorno Sobre Investimento (ROI) para apostas em empate."""
        if X_test_odds_aligned is None or odd_draw_col_name not in X_test_odds_aligned.columns:
            # Loga aviso se não puder calcular
            logging.warning(f"ROI não calculado: Dados de odds ('{odd_draw_col_name}') ausentes ou desalinhados.")
            return None # Retorna None se não pode calcular

        # Garante que y_test e X_test_odds_aligned compartilhem o mesmo índice
        common_index = y_test.index.intersection(X_test_odds_aligned.index)
        if len(common_index) != len(y_test):
            logging.warning("ROI não calculado: Índices de y_test e X_test_odds_aligned não correspondem perfeitamente.")
            # Opcional: tentar reindexar, mas é mais seguro retornar None
            return None

        # Filtra previsões e odds para o índice comum
        y_test_common = y_test.loc[common_index]
        # Verifica se y_pred tem o mesmo comprimento que y_test_common (pode não ter se train_test_split mudou)
        # É mais seguro mapear y_pred para o índice de y_test
        try:
            y_pred_series = pd.Series(y_pred, index=y_test.index) # Tenta alinhar y_pred com y_test original
            y_pred_common = y_pred_series.loc[common_index]
        except ValueError:
             logging.error("Erro ao alinhar y_pred com índice para cálculo de ROI.")
             return None


        predicted_draws_indices = common_index[y_pred_common == 1] # Pega indices onde previu empate
        num_bets = len(predicted_draws_indices)

        if num_bets == 0:
            return 0.0 # Retorna 0% se nenhuma aposta foi feita

        actuals = y_test_common.loc[predicted_draws_indices]
        # Pega odds APENAS para os índices onde a aposta foi feita E o índice é comum
        odds = pd.to_numeric(X_test_odds_aligned.loc[predicted_draws_indices, odd_draw_col_name], errors='coerce')

        profit = 0
        valid_bets = 0 # Conta apostas onde a odd era válida
        for idx in predicted_draws_indices:
            odd_d = odds.loc[idx]
            if pd.notna(odd_d) and odd_d > 0:
                profit += (odd_d - 1) if actuals.loc[idx] == 1 else -1
                valid_bets += 1

        if valid_bets == 0:
             logging.warning("ROI calculado como 0: Nenhuma aposta com odd válida encontrada.")
             return 0.0 # Retorna 0% se nenhuma odd era válida

        return (profit / valid_bets) * 100


# --- Função scale_features (Sem alterações necessárias) ---
def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame, scaler_type='standard') -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
    """Aplica scaling aos dados de treino e teste."""
    X_train = X_train.copy() # Evita SettingWithCopyWarning
    X_test = X_test.copy()

    # Converte para float ANTES de escalar
    try:
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
    except ValueError as e:
        logging.error(f"Erro ao converter features para float antes do scaling: {e}")
        # Identifica colunas problemáticas
        for col in X_train.columns:
             if not pd.api.types.is_numeric_dtype(X_train[col]):
                  logging.error(f"  Coluna não numérica em X_train: {col} (tipo: {X_train[col].dtype})")
        for col in X_test.columns:
             if not pd.api.types.is_numeric_dtype(X_test[col]):
                  logging.error(f"  Coluna não numérica em X_test: {col} (tipo: {X_test[col].dtype})")
        raise # Re-levanta o erro original após log detalhado

    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else: # Default é StandardScaler
        scaler = StandardScaler()

    logging.info(f"  Aplicando {scaler.__class__.__name__}...");
    # Trata erros durante fit/transform
    try:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    except Exception as e:
        logging.error(f"Erro durante {scaler.__class__.__name__}.fit_transform/transform: {e}", exc_info=True)
        raise # Re-levanta o erro

    X_train_scaled_df = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)
    logging.info("  Scaling concluído.")
    return X_train_scaled_df, X_test_scaled_df, scaler


# --- Função Principal de Treinamento (COM CORREÇÕES) ---
def train_evaluate_and_save_best_models(
    X: pd.DataFrame, y: pd.Series,
    X_test_with_odds: Optional[pd.DataFrame] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    scaler_type: str = 'standard',
    odd_draw_col_name: str = ODDS_COLS.get('draw', 'Odd_D_FT') # Usa config com fallback

    ) -> bool:
    """
    Treina, otimiza, avalia TODOS os modelos em config.MODEL_CONFIG.
    Seleciona o melhor modelo por F1 e o melhor por ROI (ou 2º melhor F1).
    Salva os dois modelos selecionados em arquivos separados.
    """
    if X is None or y is None: logging.error("Dados X ou y são None."); return False
    if not MODEL_CONFIG: logging.error("MODEL_CONFIG vazio."); return False

    # Filtra modelos disponíveis (ex: se LightGBM não está instalado)
    available_models = {name: config for name, config in MODEL_CONFIG.items()
                        if not (name == 'LGBMClassifier' and lgb is None)}
    if not available_models: logging.error("Nenhum modelo válido/disponível encontrado no config."); return False

    feature_names = list(X.columns)
    all_results = [] # Lista para guardar resultados de TODOS os modelos

    logging.info(f"--- Treinando {len(available_models)} Modelos para Seleção Dupla ---")
    start_time_total = time.time()

    # Divide os dados (features X e alvo y)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        logging.info(f"Split: Treino={len(X_train)}, Teste={len(X_test)}")
    except Exception as e_split:
        logging.error(f"Erro durante train_test_split: {e_split}", exc_info=True)
        return False

    # Prepara dados de teste com odds para cálculo de ROI (alinhados pelo índice)
    X_test_odds_aligned = None
    if X_test_with_odds is not None:
        if odd_draw_col_name in X_test_with_odds.columns:
            try:
                # Garante que X_test_with_odds tem as mesmas colunas que X_test para evitar erros inesperados
                # Pega apenas as linhas de X_test_with_odds que correspondem ao índice de X_test
                common_index_test = X_test.index.intersection(X_test_with_odds.index)
                if len(common_index_test) == len(X_test):
                    X_test_odds_aligned = X_test_with_odds.loc[common_index_test].copy()
                    logging.info("Dados de teste com odds alinhados para cálculo de ROI.")
                else:
                    logging.warning("Índices de X_test e X_test_with_odds não batem completamente. ROI pode ser impreciso ou falhar.")
            except KeyError:
                logging.warning("Falha ao alinhar X_test_with_odds por índice (KeyError). ROI não será calculado.")
            except Exception as e_align:
                 logging.error(f"Erro ao alinhar X_test_with_odds: {e_align}", exc_info=True)
        else:
            logging.warning(f"Coluna de odd de empate '{odd_draw_col_name}' não encontrada em X_test_with_odds. ROI não será calculado.")
    else:
        logging.warning("X_test_with_odds não fornecido. ROI não será calculado.")

    # --- Loop pelos modelos disponíveis ---
    for i, (model_name, config) in enumerate(available_models.items()):
        status_text = f"Modelo {i+1}/{len(available_models)}: {model_name}"
        logging.info(f"\n--- {status_text} ---")
        if progress_callback: progress_callback(i, len(available_models), status_text)
        start_time_model = time.time()

        try:
            model_class = eval(model_name) # Obtém a classe do modelo pelo nome
        except NameError:
            logging.error(f"Classe do modelo '{model_name}' não encontrada/importada.", exc_info=True)
            continue # Pula para o próximo modelo
        except Exception as e_eval:
             logging.error(f"Erro ao obter classe do modelo '{model_name}': {e_eval}", exc_info=True)
             continue

        # Pega configuração do modelo
        model_kwargs = config.get('model_kwargs', {})
        param_grid = config.get('param_grid', {})
        needs_scaling = config.get('needs_scaling', False)

        # Cria cópias dos dados para este modelo (evita modificar os originais)
        X_train_model = X_train.copy()
        X_test_model = X_test.copy()
        current_scaler = None # Scaler específico para este modelo

        # --- Scaling Condicional ---
        if needs_scaling:
            logging.info(f"  Modelo '{model_name}' requer scaling...");
            try:
                X_train_model, X_test_model, current_scaler = scale_features(X_train_model, X_test_model, scaler_type)
                logging.info(f"    ==> DEBUG: Scaling para {model_name} concluído com sucesso.")
            except Exception as scale_err:
                # **CORREÇÃO**: Loga o erro e PULA este modelo se scaling falhar
                logging.error(f"  ERRO GRAVE no scaling para {model_name}: {scale_err}", exc_info=True)
                logging.error(f"  -> PULANDO modelo {model_name} devido a erro no scaling.")
                continue # PULA para o próximo modelo
        else:
            logging.info("  Scaling não requerido.")

        # --- Treinamento (GridSearch ou Padrão) ---
        logging.info(f"  ==> DEBUG: Iniciando criação/treino para {model_name}")
        model_instance_trained = None # Garante que começa como None
        current_best_params = model_kwargs.copy() # Começa com kwargs padrão

        if param_grid: # Tenta GridSearchCV se houver grade de parâmetros
            try:
                # **CORREÇÃO**: Envolve a instanciação do GridSearchCV em try/except
                search_cv = GridSearchCV(
                    estimator=model_class(**model_kwargs), # Cria nova instância AQUI
                    param_grid=param_grid,
                    cv=CROSS_VALIDATION_SPLITS,
                    n_jobs=N_JOBS_GRIDSEARCH,
                    scoring='f1', # Focar em F1 para CV
                    verbose=0,
                    error_score='raise' # Levanta erro se combinação inválida
                )
                logging.info(f"    ==> DEBUG: Instância GridSearchCV para {model_name} criada.")
            except Exception as e_grid_init:
                logging.error(f"    ERRO GRAVE ao criar GridSearchCV para {model_name}: {e_grid_init}", exc_info=True)
                logging.error(f"    -> PULANDO modelo {model_name} devido a erro na config/criação do GridSearchCV.")
                continue # Pula se não conseguir criar o GridSearchCV

            logging.info(f"  Iniciando GridSearchCV (scoring=f1)...");
            try:
                if progress_callback: progress_callback(i, len(available_models), f"Ajustando {model_name}...")
                search_cv.fit(X_train_model, y_train)
                model_instance_trained = search_cv.best_estimator_ # Pega o melhor modelo
                current_best_params = search_cv.best_params_      # Pega os melhores parâmetros
                logging.info(f"    ==> DEBUG: GridSearchCV para {model_name} concluído.")
                logging.info(f"    Melhor CV (f1): {search_cv.best_score_:.4f}");
                logging.info(f"    Params: {current_best_params}")
            except Exception as e_cv:
                # Loga erro detalhado se GridSearchCV.fit falhar
                logging.error(f"    Erro GRAVE no GridSearchCV.fit para {model_name}: {e_cv}", exc_info=True)
                logging.warning(f"    -> Tentando treino com parâmetros padrão após falha do CV...")
                model_instance_trained = None # Garante que vai tentar treino padrão

        else: # Se não há param_grid, treina com kwargs padrão
             logging.info(f"  Nenhuma grade de parâmetros definida. Treinando com params padrão: {model_kwargs}")
             # Vai tentar o treino padrão no bloco abaixo

        # --- Treino Padrão / Fallback ---
        if model_instance_trained is None: # Se GridSearch não foi feito ou falhou
             logging.info(f"  ==> DEBUG: Tentando treino padrão/fallback para {model_name}.")
             try:
                 # Cria instância aqui SE NÃO FOI CRIADA PELO GRIDSEARCH
                 if 'search_cv' not in locals(): # Evita recriar se search_cv existe mas .fit falhou
                      current_model_fallback = model_class(**model_kwargs)
                 else: # Usa o estimador base do gridsearch que falhou
                      current_model_fallback = search_cv.estimator

                 if progress_callback: progress_callback(i, len(available_models), f"Ajustando {model_name} (padrão)...")
                 model_instance_trained = current_model_fallback.fit(X_train_model, y_train)
                 logging.info(f"    ==> DEBUG: Treino padrão/fallback para {model_name} concluído.")
             except Exception as e_fit:
                 # Loga erro detalhado se o treino padrão falhar
                 logging.error(f"    Erro GRAVE no treino padrão/fallback para {model_name}: {e_fit}", exc_info=True)
                 logging.error(f"    -> PULANDO modelo {model_name} devido a erro no fit.")
                 continue # PULA para o próximo modelo se o treino falhar

        # --- Verificação Crítica Pós-Treino ---
        if model_instance_trained is None:
             # Segurança extra: Se por algum motivo chegou aqui sem modelo treinado
             logging.error(f"    ERRO CRÍTICO: Modelo {model_name} é None após tentativas de treino. Pulando avaliação.")
             continue # PULA para o próximo modelo

        # --- Avaliação ---
        logging.info(f"  Avaliando...");
        current_eval_metrics = {} # Zera métricas para este modelo
        evaluation_successful = False # Flag para rastrear sucesso
        try:
            y_pred = model_instance_trained.predict(X_test_model)

            # Métricas básicas (geralmente não falham)
            acc = accuracy_score(y_test, y_pred)
            logging.info(f"    -> DEBUG {model_name}: Accuracy = {acc}")
            current_eval_metrics['accuracy'] = acc

            prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
            logging.info(f"    -> DEBUG {model_name}: Precision (Draw) = {prec}")
            current_eval_metrics['precision_draw'] = prec

            rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
            logging.info(f"    -> DEBUG {model_name}: Recall (Draw) = {rec}")
            current_eval_metrics['recall_draw'] = rec

            f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
            logging.info(f"    -> DEBUG {model_name}: F1 (Draw) = {f1}")
            current_eval_metrics['f1_score_draw'] = f1

            conf_matrix = confusion_matrix(y_test, y_pred).tolist()
            logging.info(f"    -> DEBUG {model_name}: Confusion Matrix = {conf_matrix}")
            current_eval_metrics['confusion_matrix'] = conf_matrix

            # --- Cálculo Robusto LogLoss e ROC AUC ---
            logloss_val = None; roc_auc_val = None
            has_predict_proba = hasattr(model_instance_trained, "predict_proba")
            if has_predict_proba:
                try:
                    y_pred_proba_full = model_instance_trained.predict_proba(X_test_model)
                    try: # Try LogLoss
                        logloss_val = log_loss(y_test, y_pred_proba_full)
                        logging.info(f"    -> DEBUG {model_name}: LogLoss = {logloss_val}")
                    except ValueError as e_logloss: logging.warning(f"    -> AVISO {model_name}: LogLoss não calculado: {e_logloss}")
                    except Exception as e_logloss_o: logging.warning(f"    -> AVISO {model_name}: Erro LogLoss: {e_logloss_o}")

                    try: # Try ROC AUC
                        if y_pred_proba_full.shape[1] > 1:
                            y_pred_proba_draw = y_pred_proba_full[:, 1]
                            # Verifica se y_test tem ambas as classes (0 e 1)
                            if len(np.unique(y_test)) > 1:
                                 roc_auc_val = roc_auc_score(y_test, y_pred_proba_draw)
                                 logging.info(f"    -> DEBUG {model_name}: ROC AUC = {roc_auc_val}")
                            else:
                                 logging.warning(f"    -> AVISO {model_name}: y_test contém apenas uma classe. ROC AUC não calculado.")
                        else: logging.warning(f"    -> AVISO {model_name}: predict_proba tem shape {y_pred_proba_full.shape}. ROC AUC não calculado.")
                    except ValueError as e_rocauc: logging.warning(f"    -> AVISO {model_name}: ROC AUC não calculado: {e_rocauc}") # Comum se só uma classe prevista
                    except Exception as e_rocauc_o: logging.warning(f"    -> AVISO {model_name}: Erro ROC AUC: {e_rocauc_o}")

                except AttributeError: logging.warning(f"    -> AVISO {model_name}: Erro acessando predict_proba.")
                except Exception as e_proba_g: logging.warning(f"    -> AVISO {model_name}: Erro probs: {e_proba_g}")
            else:
                logging.info(f"    -> INFO {model_name}: Modelo não tem predict_proba.")
            # Atribui valores (podem ser None)
            current_eval_metrics['log_loss'] = logloss_val
            current_eval_metrics['roc_auc'] = roc_auc_val

            # Outras métricas
            current_eval_metrics['threshold'] = 0.5 # Limiar padrão usado
            current_eval_metrics['train_set_size'] = len(y_train)
            current_eval_metrics['test_set_size'] = len(y_test)

            # --- Cálculo Robusto ROI ---
            roi_val = None; profit_val = None; num_bets_val = 0
            if X_test_odds_aligned is not None: # Verifica se dados com odds estão disponíveis
                 try:
                     roi_val = roi(y_test, y_pred, X_test_odds_aligned, odd_draw_col_name) # Usa a função roi
                     # Calcula profit e num_bets separadamente se necessário para log (função roi já faz isso)
                     predicted_draws_indices_roi = y_test.index.intersection(X_test_odds_aligned.index)[pd.Series(y_pred, index=y_test.index).loc[y_test.index.intersection(X_test_odds_aligned.index)] == 1]
                     num_bets_val = len(predicted_draws_indices_roi)
                     # Lógica para calcular profit pode ser adicionada aqui se roi() não retornar
                     profit_val = roi_val * num_bets_val / 100 if roi_val is not None and num_bets_val > 0 else 0 # Estimativa
                 except Exception as e_roi_call:
                      logging.error(f"   Erro ao chamar ou processar função roi para {model_name}: {e_roi_call}", exc_info=True)
            # Atribui valores (roi_val pode ser None se dados ausentes, ou 0 se sem apostas/lucro)
            current_eval_metrics['profit'] = profit_val
            current_eval_metrics['roi'] = roi_val
            current_eval_metrics['num_bets'] = num_bets_val

            # --- Log Final Seguro ---
            acc_str = f"{acc:.3f}" if acc is not None else "N/A"
            f1_str = f"{f1:.3f}" if f1 is not None else "N/A"
            roi_log_str = "N/A"
            if roi_val is not None:
                try:
                    # np.isnan só funciona com float, verifica tipo antes
                    if isinstance(roi_val, (float, np.number)) and not np.isnan(roi_val):
                        roi_log_str = f"{roi_val:.2f}%"
                    elif isinstance(roi_val, (int, float)): # Se for 0, por exemplo
                         roi_log_str = f"{roi_val:.2f}%"
                except TypeError: pass
            logging.info(f"    Métricas {model_name}: Acc={acc_str}, F1_Emp={f1_str}, ROI={roi_log_str}") # Log final

            evaluation_successful = True # Marcar como sucesso se chegou aqui

        except Exception as e_eval_outer:
             logging.error(f"    Erro GRAVE DURANTE avaliação {model_name}: {e_eval_outer}", exc_info=True)
             # evaluation_successful continua False

        #Adiciona resultado SE o modelo foi treinado ---
        if model_instance_trained is not None:
             # Adiciona mesmo que evaluation_successful seja False,
             # mas o dicionário 'metrics' pode estar incompleto ou com Nones.
             logging.info(f"    ==> DEBUG: Adicionando resultado para {model_name} (Avaliação Concluída: {evaluation_successful})")
             all_results.append({
                 'model_name': model_name,
                 'model_object': model_instance_trained,
                 'scaler': current_scaler,
                 'params': current_best_params,
                 'metrics': current_eval_metrics # Adiciona o dict (pode ter Nones)
             })
        else:
            logging.error(f"    ==> ERRO LÓGICO: Modelo {model_name} é None, não deveria chegar aqui.")

        logging.info(f"  Tempo p/ {model_name}: {time.time() - start_time_model:.2f} seg.")
    # --- Fim do Loop `for model_name` ---

    # --- Seleção e Salvamento (Fora do Loop) ---
    if progress_callback: progress_callback(len(available_models), len(available_models), "Selecionando e Salvando...")
    end_time_total = time.time()
    logging.info(f"--- Treino concluído ({end_time_total - start_time_total:.2f} seg) ---")

    # Verifica se HÁ resultados para processar
    if not all_results:
        logging.error("Nenhum modelo foi treinado e avaliado com sucesso para gerar resultados.")
        return False # Falha geral

    # --- Cria DataFrame e Ordena ---
    try:
        results_df = pd.DataFrame(all_results)
        # Extrai métricas de forma segura (lidando com possíveis dicts vazios)
        results_df['f1_score_draw'] = results_df['metrics'].apply(lambda m: m.get('f1_score_draw') if isinstance(m, dict) else -1.0).fillna(-1.0).astype(float)
        results_df['roi'] = results_df['metrics'].apply(lambda m: m.get('roi') if isinstance(m, dict) else -np.inf).fillna(-np.inf)
        # Converte ROI para float, tratando possíveis NaNs que sobraram do cálculo
        results_df['roi'] = pd.to_numeric(results_df['roi'], errors='coerce').fillna(-np.inf)

        # Ordena por F1 (desc) e ROI (desc)
        results_df = results_df.sort_values(by=['f1_score_draw', 'roi'], ascending=[False, False]).reset_index(drop=True) # drop=True para não manter index antigo

        print("\n--- Ranking dos Modelos (por F1 Empate) ---")
        # Exibe colunas relevantes, formatando float para legibilidade
        with pd.option_context('display.float_format', '{:.4f}'.format):
             print(results_df[['model_name', 'f1_score_draw', 'roi']])
        print("-" * 40)

    except Exception as e_rank:
         logging.error(f"Erro ao criar DataFrame de resultados ou ordenar: {e_rank}", exc_info=True)
         return False # Falha se não conseguir rankear

    # --- Seleção dos Melhores (requer results_df válido) ---
    if results_df.empty:
         logging.error("DataFrame de resultados está vazio após processamento. Nenhum modelo pode ser selecionado.")
         return False

    # Melhor F1 (primeiro da lista ordenada)
    best_f1_result = results_df.iloc[0].to_dict()
    # Formatação segura para log
    f1_log_best = best_f1_result.get('f1_score_draw', -1)
    f1_log_str = f"{f1_log_best:.4f}" if isinstance(f1_log_best, (int, float)) and not np.isnan(f1_log_best) else "N/A"
    logging.info(f"Melhor Modelo por F1: {best_f1_result['model_name']} (F1={f1_log_str})")

    # Melhor ROI (primeiro com ROI > -infinito)
    # Usa where() para encontrar o primeiro ROI válido
    results_df_roi_valid = results_df.where(results_df['roi'] > -np.inf).dropna(subset=['roi'])
    best_roi_result = None
    if not results_df_roi_valid.empty:
        best_roi_result = results_df_roi_valid.iloc[0].to_dict()
        roi_log_best = best_roi_result.get('roi', -np.inf)
        roi_log_str = f"{roi_log_best:.2f}%" if isinstance(roi_log_best, (int, float)) and roi_log_best > -np.inf else "N/A"
        logging.info(f"Melhor Modelo por ROI: {best_roi_result['model_name']} (ROI={roi_log_str})")
    else:
        logging.warning("Nenhum modelo com ROI válido (> -infinito) encontrado.")

    # --- Lógica de Seleção para Salvamento ---
    model_to_save_f1 = best_f1_result
    model_to_save_roi = None # Default

    if best_roi_result:
        # Verifica se os índices originais (antes do reset_index do sort) são diferentes
        # Se o DataFrame tiver uma coluna 'index' do sort_values anterior:
        if 'index' in best_f1_result and 'index' in best_roi_result and best_f1_result['index'] == best_roi_result['index']:
             # Mesmo modelo é o melhor em F1 e ROI
             if len(results_df) > 1: # Verifica se existe um segundo modelo
                 model_to_save_roi = results_df.iloc[1].to_dict() # Pega o segundo melhor (por F1)
                 logging.info(f"  -> Melhor F1 e ROI são o mesmo ({best_f1_result['model_name']}). Usando 2º Melhor por F1 ({model_to_save_roi['model_name']}) para slot ROI.")
             else: # Só há um modelo no ranking
                 logging.warning(f"  -> Apenas um modelo ({best_f1_result['model_name']}) no ranking. Salvando-o para F1 e ROI.")
                 model_to_save_roi = best_f1_result # Salva o mesmo para ambos
        else:
            # Melhores modelos F1 e ROI são diferentes
            model_to_save_roi = best_roi_result
            logging.info(f"  -> Melhores F1 ({model_to_save_f1['model_name']}) e ROI ({model_to_save_roi['model_name']}) são diferentes.")
    else:
        # Nenhum modelo com ROI válido encontrado
        logging.warning("  -> Nenhum modelo com ROI válido. Salvando melhor F1 para ambos os slots.")
        model_to_save_roi = best_f1_result # Salva o melhor F1 no slot ROI também

    # --- Salvamento ---
    # Salva o melhor F1
    logging.info(f"Salvando Melhor F1 ({model_to_save_f1['model_name']}) em {BEST_F1_MODEL_SAVE_PATH}")
    _save_single_model_object(model_to_save_f1, feature_names, BEST_F1_MODEL_SAVE_PATH)

    # Salva o melhor ROI (ou substituto)
    if model_to_save_roi:
         logging.info(f"Salvando Melhor ROI/2nd F1 ({model_to_save_roi['model_name']}) em {BEST_ROI_MODEL_SAVE_PATH}")
         _save_single_model_object(model_to_save_roi, feature_names, BEST_ROI_MODEL_SAVE_PATH)
    else:
         # Isso não deveria acontecer com a lógica acima, mas por segurança
         logging.error(f"Não foi possível determinar um modelo para salvar no slot 'Melhor ROI' em {BEST_ROI_MODEL_SAVE_PATH}")

    return True # Indica que o processo foi concluído (pode ter tido warnings)


# --- Função Auxiliar para Salvar (_save_single_model_object) ---
def _save_single_model_object(model_result_dict: Dict, feature_names: List[str], file_path: str) -> None:
    """Salva um único modelo e seus dados associados."""
    if not isinstance(model_result_dict, dict):
        logging.error(f"Erro ao salvar: model_result_dict não é um dicionário válido para {file_path}")
        return
    try:
        save_timestamp = datetime.datetime.now().isoformat()
        model_to_save = model_result_dict.get('model_object')
        if model_to_save is None:
             logging.error(f"Erro ao salvar: 'model_object' não encontrado no dicionário para {file_path}")
             return

        save_obj = {
            'model': model_to_save,
            'scaler': model_result_dict.get('scaler'), # Pode ser None
            'feature_names': feature_names,            # Usa as features atuais
            'best_params': model_result_dict.get('params'), # Melhores params do CV ou kwargs
            'eval_metrics': model_result_dict.get('metrics'),# Dict de métricas
            'save_timestamp': save_timestamp,
            'model_class_name': model_to_save.__class__.__name__
        }
        joblib.dump(save_obj, file_path)
        logging.info(f"  -> Modelo {model_to_save.__class__.__name__} salvo com sucesso em {os.path.basename(file_path)}.")
    except Exception as e:
        logging.error(f"  -> Erro GRAVE ao salvar modelo em {file_path}: {e}", exc_info=True)


# Função save_model_scaler_features 
def save_model_scaler_features(model: Any, scaler: Optional[Any], feature_names: List[str],
                               best_params: Optional[Dict], eval_metrics: Optional[Dict],
                               file_path: str) -> None:
     print("AVISO: Função save_model_scaler_features chamada (possivelmente de código antigo?). Usando _save_single_model_object.")
     # Cria dict no formato esperado por _save_single_model_object
     model_result_dict = {
          'model_object': model,
          'scaler': scaler,
          'params': best_params,
          'metrics': eval_metrics
     }
     _save_single_model_object(model_result_dict, feature_names, file_path)


# analyze_features (Mantida como estava)
def analyze_features(X: pd.DataFrame, y: pd.Series) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    # ... (código da função como antes) ...
    logging.info("--- Iniciando Análise de Features (chamado externamente) ---")
    if X is None or y is None or X.empty or y.empty:
        logging.error("Dados inválidos para análise de features.")
        return None
    if not X.index.equals(y.index):
        logging.warning("Índices de X e y não são idênticos para analyze_features. Tentando alinhar.")
        try:
            y = y.reindex(X.index)
            if y.isnull().any():
                 logging.error("Alinhamento de y resultou em NaNs em analyze_features.")
                 return None
        except Exception as e_reindex:
             logging.error(f"Erro ao alinhar y com X em analyze_features: {e_reindex}")
             return None

    feature_names = X.columns.tolist()
    imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': np.nan})
    corr_matrix = pd.DataFrame()

    logging.info("  Calculando importância (RandomForest rápido)...")
    try:
        rf_analyzer = RandomForestClassifier(n_estimators=50, max_depth=15, n_jobs=-1, min_samples_leaf=3, random_state=RANDOM_STATE) # Adiciona random_state
        rf_analyzer.fit(X, y)
        importances = rf_analyzer.feature_importances_
        imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        imp_df = imp_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        logging.info("  Importância calculada.")
    except Exception as e:
        logging.error(f"  Erro ao calcular importância com RF: {e}", exc_info=True)

    logging.info("  Calculando matriz de correlação...")
    try:
        df_temp = X.copy()
        df_temp['target_IsDraw'] = y
        corr_matrix = df_temp.corr()
        logging.info("  Matriz de correlação calculada.")
    except Exception as e:
        logging.error(f"  Erro ao calcular correlação: {e}", exc_info=True)

    logging.info("--- Análise de Features Concluída ---")
    return imp_df, corr_matrix


# optimize_single_model (Mantida como placeholder)
def optimize_single_model(model_name: str, X: pd.DataFrame, y: pd.Series,
                           X_test_with_odds: Optional[pd.DataFrame] = None,
                           progress_callback: Optional[Callable[[int, int, str], None]] = None,
                           scaler_type: str = 'standard',
                           odd_draw_col_name: str = ODDS_COLS.get('draw', 'Odd_D_FT')
                           ) -> Optional[Tuple[str, Dict, Dict]]:
    # ... (código placeholder como antes) ...
    logging.info(f"--- Otimizando Hiperparâmetros para: {model_name} (Placeholder) ---")
    logging.warning("Implementação real de optimize_single_model (GridSearchCV/Avaliação) pendente.")
    # (Resto do código placeholder omitido por brevidade)
    return None # Retorna None pois é placeholder