
import pandas as pd
import time
from sklearn.model_selection import train_test_split, GridSearchCV
try: import lightgbm as lgb; LGBMClassifier = lgb.LGBMClassifier
except ImportError: print("AVISO: LightGBM não instalado."); lgb = None; LGBMClassifier = None
from sklearn.metrics import (
    accuracy_score, log_loss, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib, datetime, logging
from config import ( 
    RANDOM_STATE, TEST_SIZE, MODEL_CONFIG,
    CROSS_VALIDATION_SPLITS, N_JOBS_GRIDSEARCH,
    ODDS_COLS)
from typing import Any, Optional, Dict, Tuple, List, Callable

logging.basicConfig(level=logging.INFO, format='%(asctime)s - TRAINER - %(levelname)s - %(message)s')

# Função scale_features (Adaptada para usar StandardScaler ou MinMaxScaler)
def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame, scaler_type='standard') -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
    
    """ Aplica StandardScaler ou MinMaxScaler aos dados de treino e teste."""

    if scaler_type == 'minmax': scaler = MinMaxScaler(); 
    else: scaler = StandardScaler()
    X_train = X_train.astype(float); X_test = X_test.astype(float)
    logging.info(f"  Aplicando {scaler.__class__.__name__}..."); X_train_scaled = scaler.fit_transform(X_train); X_test_scaled = scaler.transform(X_test)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns); X_test_scaled_df = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)
    logging.info("  Scaling concluído."); return X_train_scaled_df, X_test_scaled_df, scaler

# --- Função de Treinamento (Modificada para Imprimir Importância) ---
def train_evaluate_model(
    X: pd.DataFrame, y: pd.Series,
    X_test_with_odds: Optional[pd.DataFrame] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    scaler_type: str = 'standard',
    odd_draw_col_name: str = ODDS_COLS['draw']
    ) -> Optional[Tuple[Any, Optional[Any], List[str], Optional[Dict], Optional[Dict]]]:
    """
    Treina APENAS os modelos em config.MODEL_CONFIG (agora só RF),
    imprime importância das features e retorna o modelo treinado.
    """
    if X is None or y is None: return None
    if not MODEL_CONFIG: logging.error("MODEL_CONFIG vazio."); return None
    available_models = {name: config for name, config in MODEL_CONFIG.items() if not (name == 'LGBMClassifier' and lgb is None)}
    if not available_models: logging.error("Nenhum modelo válido encontrado."); return None

    feature_names = list(X.columns)

    logging.info(f"--- Treinando Modelo(s) para Análise de Features ---")
    start_time_total = time.time()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    logging.info(f"Split: Treino={len(X_train)}, Teste={len(X_test)}")
    X_test_odds_aligned = None 
    if X_test_with_odds is not None:
        if odd_draw_col_name in X_test_with_odds.columns:
            try: X_test_odds_aligned = X_test_with_odds.loc[X_test.index]
            except KeyError: logging.warning("Índices não batem p/ ROI.")
        else: logging.warning(f"Coluna '{odd_draw_col_name}' não em X_test_with_odds p/ ROI.")
    else: logging.warning("X_test_with_odds não fornecido p/ ROI.")


    # Loop 
    final_model = None
    final_scaler = None
    final_params = None
    final_metrics = None

    for i, (model_name, config) in enumerate(available_models.items()):
        status_text = f"Processando Modelo: {model_name}"
        logging.info(f"\n--- {status_text} ---")
        if progress_callback: progress_callback(i, len(available_models), status_text)
        start_time_model = time.time()
        try: model_class = globals()[model_name]
        except KeyError: logging.error(f"Classe {model_name} não encontrada."); continue
        model_kwargs = config.get('model_kwargs', {}); param_grid = config.get('param_grid', {})
        needs_scaling = config.get('needs_scaling', False)
        X_train_model = X_train.copy(); X_test_model = X_test.copy(); current_scaler = None

        if needs_scaling: 
             logging.info(f"  Modelo '{model_name}' requer scaling..."); 
             try: X_train_model, X_test_model, current_scaler = scale_features(X_train_model, X_test_model, scaler_type); 
             except Exception as scale_err: logging.error(f"  ERRO scaling: {scale_err}. Pulando."); continue
        else: logging.info("  Scaling não requerido.")

        current_model = model_class(**model_kwargs); current_best_params = model_kwargs; model_instance_trained = None
        if param_grid: # Roda GridSearchCV
            grid_verbose_level = 1
            search_cv = GridSearchCV(estimator=current_model, param_grid=param_grid, cv=CROSS_VALIDATION_SPLITS, n_jobs=N_JOBS_GRIDSEARCH, scoring='f1', verbose=grid_verbose_level, error_score='raise')
            logging.info(f"  Iniciando GridSearchCV (verbose={grid_verbose_level}, scoring=f1)...");
            try: logging.info(f"    >>> Fit GridSearchCV {model_name}..."); search_cv.fit(X_train_model, y_train); logging.info(f"    <<< Fit GridSearchCV OK."); model_instance_trained = search_cv.best_estimator_; current_best_params = search_cv.best_params_; logging.info(f"      Melhor CV (f1): {search_cv.best_score_:.4f}"); logging.info(f"      Params: {current_best_params}")
            except Exception as e_cv: logging.warning(f"    Erro CV: {e_cv}. Treinando com padrão...")
        else: logging.info("  Treinando com params padrão...")

        if model_instance_trained is None: # Treino padrão
             try: logging.info(f"    >>> Fit padrão {model_name}..."); model_instance_trained = current_model.fit(X_train_model, y_train); logging.info(f"    <<< Fit padrão OK.")
             except Exception as e_fit: logging.error(f"    Erro treino padrão: {e_fit}. Pulando."); continue

        # Avaliação
        logging.info("  Avaliando..."); current_eval_metrics = {}
        try: # ... (Cálculo de métricas e ROI como V11) ...
             has_predict_proba = hasattr(model_instance_trained, "predict_proba"); y_pred = model_instance_trained.predict(X_test_model); y_pred_proba_draw = None; logloss = None; roc_auc = None
             if has_predict_proba: #... (calcula probs, logloss, roc_auc) ...
                  try: y_pred_proba_full = model_instance_trained.predict_proba(X_test_model); y_pred_proba_draw = y_pred_proba_full[:, 1]; logloss = log_loss(y_test, y_pred_proba_full); roc_auc = roc_auc_score(y_test, y_pred_proba_draw)
                  except Exception as e_proba: logging.warning(f"    Erro probs: {e_proba}")
             current_eval_metrics['accuracy'] = accuracy_score(y_test, y_pred); # ... (resto das métricas) ...
             current_eval_metrics['precision_draw'] = precision_score(y_test, y_pred, pos_label=1, zero_division=0); current_eval_metrics['recall_draw'] = recall_score(y_test, y_pred, pos_label=1, zero_division=0); current_eval_metrics['f1_score_draw'] = f1_score(y_test, y_pred, pos_label=1, zero_division=0); current_eval_metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist(); current_eval_metrics['log_loss'] = logloss; current_eval_metrics['roc_auc'] = roc_auc; current_eval_metrics['threshold'] = 0.5; current_eval_metrics['train_set_size'] = len(y_train); current_eval_metrics['test_set_size'] = len(y_test); current_eval_metrics['profit'] = None; current_eval_metrics['roi'] = None; current_eval_metrics['num_bets'] = 0
             if X_test_odds_aligned is not None: # ... (cálculo ROI) ...
                predicted_draws_indices = X_test.index[y_pred == 1]; num_bets = len(predicted_draws_indices); current_eval_metrics['num_bets'] = num_bets
                if num_bets > 0: #... (calcula profit) ...
                     actuals = y_test.loc[predicted_draws_indices]; odds = X_test_odds_aligned.loc[predicted_draws_indices, odd_draw_col_name].astype(float); profit = 0
                     for idx in predicted_draws_indices: odd_d = odds.loc[idx]; 
                     if pd.notna(odd_d) and odd_d > 0: profit += (odd_d - 1) if actuals.loc[idx] == 1 else -1
                     current_eval_metrics['profit'] = profit; current_eval_metrics['roi'] = (profit / num_bets) * 100 if num_bets > 0 else 0
             logging.info(f"    Métricas {model_name}: Acc={current_eval_metrics['accuracy']:.3f}, F1_Emp={current_eval_metrics['f1_score_draw']:.3f}, ROI={current_eval_metrics.get('roi', 'N/A'):.2f}%")

             # --- ADICIONA ANÁLISE DE IMPORTÂNCIA ---

             if model_name == 'RandomForestClassifier' and hasattr(model_instance_trained, 'feature_importances_'):
                 importances = model_instance_trained.feature_importances_
                 feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                 feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
                 print("\n--- Importância das Features (RandomForest) ---")
                 # Imprime o DataFrame formatado

                 with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
                      print(feature_importance_df)
                 print("--------------------------------------------")
                 # Adiciona ao dict de métricas para salvar
                 current_eval_metrics['feature_importances'] = feature_importance_df.to_dict('records')

             # Guarda o resultado deste modelo 
             final_model = model_instance_trained
             final_scaler = current_scaler
             final_params = current_best_params
             final_metrics = current_eval_metrics

        except Exception as e_eval: logging.error(f"    Erro avaliação {model_name}: {e_eval}")
        logging.info(f"  Tempo p/ {model_name}: {time.time() - start_time_model:.2f} seg.")

    # Fim do Loop
    end_time_total = time.time(); logging.info(f"--- Treino concluído ({end_time_total - start_time_total:.2f} seg) ---")
    if final_model:
        logging.info(f"Modelo Treinado: {final_model.__class__.__name__}")
        # Retorna os resultados do único modelo treinado
        return final_model, final_scaler, feature_names, final_params, final_metrics
    else: logging.error("Nenhum modelo foi treinado."); return None


# Função save_model_scaler_features (Adaptada para usar StandardScaler ou MinMaxScaler)
def save_model_scaler_features(model: Any, scaler: Optional[Any], # Sem discretizer
                               feature_names: List[str], best_params: Optional[Dict], eval_metrics: Optional[Dict],
                               file_path: str) -> None:
    if model is None: print("Erro: Tentando salvar modelo None."); return
    try: save_timestamp = datetime.datetime.now().isoformat(); save_obj = {'model': model, 'scaler': scaler, 'feature_names': feature_names, 'best_params': best_params, 'eval_metrics': eval_metrics, 'save_timestamp': save_timestamp, 'model_class_name': model.__class__.__name__ }; joblib.dump(save_obj, file_path); print(f"Modelo ({model.__class__.__name__}) salvo em: {file_path}")
    except Exception as e: print(f"Erro ao salvar objeto: {e}")