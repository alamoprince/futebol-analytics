import pandas as pd
import time
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
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
from sklearn.metrics import (accuracy_score, log_loss,
                             precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib, datetime, numpy as np, logging
from config import (
    RANDOM_STATE, TEST_SIZE, MODEL_CONFIG,
    CROSS_VALIDATION_SPLITS, N_JOBS_GRIDSEARCH,
    ODDS_COLS, BEST_MODEL_METRIC
)
from typing import Any, Optional, Dict, Tuple, List, Callable

logging.basicConfig(level=logging.INFO, format='%(asctime)s - TRAINER - %(levelname)s - %(message)s')

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame, scaler_type='standard') -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
    if scaler_type == 'minmax': scaler = MinMaxScaler()
    else: scaler = StandardScaler()
    X_train = X_train.astype(float); X_test = X_test.astype(float)
    logging.info(f"  Aplicando {scaler.__class__.__name__}...")
    X_train_scaled = scaler.fit_transform(X_train); X_test_scaled = scaler.transform(X_test)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)
    logging.info("  Scaling concluído.")
    return X_train_scaled_df, X_test_scaled_df, scaler

# --- Função de Treinamento com Callback (Corrigida eval) ---
def train_evaluate_model(
    X: pd.DataFrame, y: pd.Series,
    X_test_with_odds: Optional[pd.DataFrame] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None, # Callback
    scaler_type: str = 'standard',
    odd_draw_col_name: str = ODDS_COLS['draw']
    ) -> Optional[Tuple[Any, Optional[Any], List[str], Optional[Dict], Optional[Dict]]]: # Retorna 5 itens principais
    """
    Treina, otimiza, avalia múltiplos modelos e reporta progresso via callback.
    Retorna o melhor modelo e suas estatísticas.
    """
    if X is None or y is None: return None
    if not MODEL_CONFIG: logging.error("MODEL_CONFIG vazio."); return None

    # Filtra modelos disponíveis
    available_models = {}
    for name, config in MODEL_CONFIG.items():
        if name == 'LGBMClassifier' and lgb is None: continue
        try:

            model_cls_eval = eval(name)
            if model_cls_eval is None: raise NameError(f"Classe {name} não definida ou importada.")
            available_models[name] = config
        except NameError as e: logging.warning(f"{e}. Pulando modelo.")
        except Exception as e_check: logging.warning(f"Erro verificando classe {name}: {e_check}. Pulando.")
    if not available_models: logging.error("Nenhum modelo válido encontrado."); return None

    feature_names = list(X.columns)
    best_overall_model = None; best_scaler = None; best_params = None; best_eval_metrics = None
    higher_is_better = BEST_MODEL_METRIC not in ['log_loss']; best_score = -np.inf if higher_is_better else np.inf
    n_models_to_train = len(available_models)

    logging.info(f"--- Treinando {n_models_to_train} Modelos (Seleção: {BEST_MODEL_METRIC}) ---")
    start_time_total = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    logging.info(f"Split: Treino={len(X_train)}, Teste={len(X_test)}")
    X_test_odds_aligned = None # ... (lógica alinhar X_test_with_odds) ...
    if X_test_with_odds is not None:
        if odd_draw_col_name in X_test_with_odds.columns:
            try: X_test_odds_aligned = X_test_with_odds.loc[X_test.index]
            except KeyError: logging.warning("Índices não batem p/ ROI.")
        else: logging.warning(f"Coluna '{odd_draw_col_name}' não em X_test_with_odds p/ ROI.")
    else: logging.warning("X_test_with_odds não fornecido p/ ROI.")

    # Loop pelos modelos
    for i, (model_name, config) in enumerate(available_models.items()):
        status_text = f"Modelo {i+1}/{n_models_to_train}: {model_name}"
        logging.info(f"\n--- {status_text} ---")
        if progress_callback: progress_callback(i, n_models_to_train, status_text) # Reporta início do modelo
        start_time_model = time.time()
        try: model_class = eval(model_name) # Usa eval
        except NameError: logging.error(f"Classe {model_name} não encontrada (eval)."); continue

        model_kwargs = config.get('model_kwargs', {}); param_grid = config.get('param_grid', {})
        needs_scaling = config.get('needs_scaling', False)
        X_train_model = X_train.copy(); X_test_model = X_test.copy(); current_scaler = None

        if needs_scaling: # Scaling condicional
            logging.info(f"  Modelo '{model_name}' requer scaling...")
            try: X_train_model, X_test_model, current_scaler = scale_features(X_train_model, X_test_model, scaler_type)
            except Exception as scale_err: logging.error(f"  ERRO scaling: {scale_err}. Pulando."); continue
        else: logging.info("  Scaling não requerido.")

        current_model = model_class(**model_kwargs); current_best_params = model_kwargs; model_instance_trained = None
        if param_grid: # GridSearchCV
            grid_verbose_level = 0 if n_models_to_train > 1 else 1
            logging.info(f"  Parâmetros: {param_grid}"); logging.info(f"  Treinando {model_name} (Grid)...")
            search_cv = GridSearchCV(estimator=current_model, param_grid=param_grid, cv=CROSS_VALIDATION_SPLITS, n_jobs=N_JOBS_GRIDSEARCH, scoring='f1', verbose=grid_verbose_level, error_score='raise')
            logging.info(f"  Iniciando GridSearchCV (verbose={grid_verbose_level}, scoring=f1)...");
            try:
                 if progress_callback: progress_callback(i, n_models_to_train, f"Ajustando {model_name} (Grid)...")
                 search_cv.fit(X_train_model, y_train); model_instance_trained = search_cv.best_estimator_; current_best_params = search_cv.best_params_; logging.info(f"    Melhor CV (f1): {search_cv.best_score_:.4f}"); logging.info(f"    Params: {current_best_params}")
            except Exception as e_cv: logging.warning(f"    Erro CV: {e_cv}. Treinando com padrão...")
        else: logging.info("  Treinando com params padrão...")

        if model_instance_trained is None: # Treino padrão
             try:
                 if progress_callback: progress_callback(i, n_models_to_train, f"Ajustando {model_name} (Padrão)...")
                 model_instance_trained = current_model.fit(X_train_model, y_train)
             except Exception as e_fit: logging.error(f"    Erro treino padrão: {e_fit}. Pulando."); continue

        # Avaliação
        logging.info("  Avaliando..."); current_eval_metrics = {}
        if progress_callback: progress_callback(i, n_models_to_train, f"Avaliando {model_name}...")
        try: # ... (Cálculo de métricas e ROI) ...
             has_predict_proba = hasattr(model_instance_trained, "predict_proba"); y_pred = model_instance_trained.predict(X_test_model); y_pred_proba_draw = None; logloss = None; roc_auc = None
             if has_predict_proba: #... (calcula probs, logloss, roc_auc) ...
                 try: y_pred_proba_full = model_instance_trained.predict_proba(X_test_model); y_pred_proba_draw = y_pred_proba_full[:, 1]; logloss = log_loss(y_test, y_pred_proba_full); roc_auc = roc_auc_score(y_test, y_pred_proba_draw)
                 except Exception as e_proba: logging.warning(f"    Erro probs: {e_proba}")
                 
             current_eval_metrics['accuracy'] = accuracy_score(y_test, y_pred); 
             current_eval_metrics['precision_draw'] = precision_score(y_test, y_pred, pos_label=1, zero_division=0); 
             current_eval_metrics['recall_draw'] = recall_score(y_test, y_pred, pos_label=1, zero_division=0); 
             current_eval_metrics['f1_score_draw'] = f1_score(y_test, y_pred, pos_label=1, zero_division=0); 
             current_eval_metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist(); 
             current_eval_metrics['log_loss'] = logloss; 
             current_eval_metrics['roc_auc'] = roc_auc; 
             current_eval_metrics['threshold'] = 0.5; 
             current_eval_metrics['train_set_size'] = len(y_train); 
             current_eval_metrics['test_set_size'] = len(y_test); 
             current_eval_metrics['profit'] = None; 
             current_eval_metrics['roi'] = None; 
             current_eval_metrics['num_bets'] = 0

             if X_test_odds_aligned is not None: # ... (cálculo ROI) ...
                predicted_draws_indices = X_test.index[y_pred == 1]; num_bets = len(predicted_draws_indices); current_eval_metrics['num_bets'] = num_bets
                if num_bets > 0: #... (calcula profit) ...
                     actuals = y_test.loc[predicted_draws_indices]; odds = X_test_odds_aligned.loc[predicted_draws_indices, odd_draw_col_name].astype(float); profit = 0
                     for idx in predicted_draws_indices: odd_d = odds.loc[idx]; 
                     if pd.notna(odd_d) and odd_d > 0: profit += (odd_d - 1) if actuals.loc[idx] == 1 else -1
                     current_eval_metrics['profit'] = profit; current_eval_metrics['roi'] = (profit / num_bets) * 100 if num_bets > 0 else 0
             logging.info(f"    Métricas {model_name}: Acc={current_eval_metrics['accuracy']:.3f}, F1_Emp={current_eval_metrics['f1_score_draw']:.3f}, ROI={current_eval_metrics.get('roi', 'N/A'):.2f}%")

             # Seleção do Melhor Modelo
             current_score = current_eval_metrics.get(BEST_MODEL_METRIC)
             if current_score is not None and not np.isnan(current_score):
                  update_best = False
                  if best_overall_model is None: update_best = True; logging.info(f"    -> Primeiro modelo válido.")
                  elif higher_is_better and current_score > best_score: update_best = True; logging.info(f"    -> Score ({current_score:.4f}) MELHOR que anterior ({best_score:.4f}).")
                  elif not higher_is_better and current_score < best_score: update_best = True; logging.info(f"    -> Score ({current_score:.4f}) MELHOR que anterior ({best_score:.4f}).")
                  
                  if update_best: logging.info(f"  *** Atualizando melhor modelo: {model_name} (Score {BEST_MODEL_METRIC}: {current_score:.4f}) ***"); 
                  best_score = current_score; 
                  best_overall_model = model_instance_trained; 
                  best_scaler = current_scaler; 
                  best_params = current_best_params; 
                  best_eval_metrics = current_eval_metrics

             else: logging.warning(f"  Métrica '{BEST_MODEL_METRIC}' inválida p/ {model_name}.")
        except Exception as e_eval: logging.error(f"    Erro avaliação {model_name}: {e_eval}")
        logging.info(f"  Tempo p/ {model_name}: {time.time() - start_time_model:.2f} seg.")

    # Fim do Loop
    if progress_callback: # Sinaliza fim do processo de treino (antes de retornar)
        progress_callback(n_models_to_train, n_models_to_train, "Seleção Concluída")
    end_time_total = time.time(); logging.info(f"--- Treino concluído ({end_time_total - start_time_total:.2f} seg) ---")
    if best_overall_model:
        logging.info(f"Melhor Modelo: {best_overall_model.__class__.__name__} ({BEST_MODEL_METRIC}: {best_score:.4f})")
        # Retorna 5 itens principais (sem discretizer)
        return best_overall_model, best_scaler, feature_names, best_params, best_eval_metrics
    else: logging.error("Nenhum modelo treinado."); return None


# Função save_model_scaler_features (sem discretizer)
def save_model_scaler_features(model: Any, 
                               scaler: Optional[Any],
                               feature_names: List[str], 
                               best_params: Optional[Dict], 
                               eval_metrics: Optional[Dict],
                               file_path: str) -> None:
    if model is None: print("Erro: Tentando salvar modelo None."); return
    try: save_timestamp = datetime.datetime.now().isoformat(); save_obj = {'model': model, 'scaler': scaler, 'feature_names': feature_names, 'best_params': best_params, 'eval_metrics': eval_metrics, 'save_timestamp': save_timestamp, 'model_class_name': model.__class__.__name__ }; joblib.dump(save_obj, file_path); print(f"Melhor Modelo ({model.__class__.__name__}) salvo em: {file_path}")
    except Exception as e: print(f"Erro ao salvar objeto: {e}")