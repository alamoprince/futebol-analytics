import pandas as pd
import time
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold # Importa StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier # <<< Importa VotingClassifier
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.isotonic import IsotonicRegression
from logger_config import setup_logger
from collections import defaultdict # Para agrupar resultados CV
logger = setup_logger("ModelTrainerApp")
# --- Imblearn Imports ---
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    IMBLEARN_AVAILABLE = True
    logger.info("Biblioteca 'imbalanced-learn' carregada com sucesso.")
except ImportError:
    logger.error("ERRO CRÍTICO: 'imbalanced-learn' não instalado. Sampler desativado.")
    IMBLEARN_AVAILABLE = False
    from sklearn.pipeline import Pipeline as ImbPipeline # Fallback
    SMOTE = None; RandomOverSampler = None

# --- Skopt/BayesSearchCV Imports ---
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Categorical, Integer
    SKOPT_AVAILABLE = True
except ImportError:
    logger.warning("AVISO: skopt não instalado. Usando GridSearchCV fallback.")
    SKOPT_AVAILABLE = False
    from sklearn.model_selection import GridSearchCV # Precisa ser importado para o fallback

# --- LightGBM Imports ---
try:
    import lightgbm as lgb
    LGBMClassifier = lgb.LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    logger.warning("AVISO: LightGBM não instalado.")
    lgb = None; LGBMClassifier = None; LGBM_AVAILABLE = False

# --- Outros Imports ---
from sklearn.metrics import (accuracy_score, classification_report, log_loss,
                             precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, brier_score_loss,
                             precision_recall_curve)
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib, os, datetime, numpy as np, traceback
from sklearn.base import clone # Para clonar estimadores base
try:
    from config import (RANDOM_STATE, MODEL_CONFIG, CLASS_NAMES, TEST_SIZE,
                        BEST_F1_MODEL_SAVE_PATH, BEST_ROI_MODEL_SAVE_PATH,
                        CROSS_VALIDATION_SPLITS, N_JOBS_GRIDSEARCH, DEFAULT_F1_THRESHOLD,
                        ODDS_COLS, BEST_MODEL_METRIC, BEST_MODEL_METRIC_ROI,
                        DEFAULT_EV_THRESHOLD, MIN_RECALL_FOR_PRECISION_OPT,
                        BAYESIAN_OPT_N_ITER, FEATURE_EPSILON)
    from typing import Any, Optional, Dict, Tuple, List, Callable
except ImportError as e: logger.critical(f"Erro crítico import config/typing: {e}", exc_info=True); raise


# --- Funções Auxiliares (roi, calculate_roi_with_threshold, calculate_metrics_with_ev, scale_features) ---
# (Cole as definições completas e corretas aqui)
def roi(y_test: pd.Series, y_pred: np.ndarray, X_test_odds_aligned: Optional[pd.DataFrame], odd_draw_col_name: str) -> Optional[float]:
    if X_test_odds_aligned is None or odd_draw_col_name not in X_test_odds_aligned.columns:
        return None
    try:
        common_index = y_test.index.intersection(X_test_odds_aligned.index)
    except AttributeError:
        return None
    if len(common_index) != len(y_test):
        logger.warning("ROI: Index mismatch")
        return None
    y_test_common = y_test.loc[common_index]
    try:
        y_pred_series = pd.Series(y_pred, index=y_test.index)
        y_pred_common = y_pred_series.loc[common_index]
    except Exception:
        return None
    predicted_draws_indices = common_index[y_pred_common == 1]
    num_bets = len(predicted_draws_indices)
    if num_bets == 0:
        return 0.0
    actuals = y_test_common.loc[predicted_draws_indices]
    odds = pd.to_numeric(X_test_odds_aligned.loc[predicted_draws_indices, odd_draw_col_name], errors='coerce')
    profit = 0
    valid_bets = 0
    for idx in predicted_draws_indices:
        odd_d = odds.loc[idx]
        if pd.notna(odd_d) and odd_d > 1:
            profit += (odd_d - 1) if actuals.loc[idx] == 1 else -1
            valid_bets += 1
    if valid_bets == 0:
        return 0.0
    return (profit / valid_bets) * 100


def calculate_roi_with_threshold(y_true: pd.Series, y_proba: np.ndarray, threshold: float, odds_data: Optional[pd.DataFrame], odd_col_name: str) -> Tuple[Optional[float], int, Optional[float]]:
    profit, roi_value, num_bets, profit_calc, valid_bets_count = None, None, 0, 0, 0
    if odds_data is None or odd_col_name not in odds_data.columns:
        return roi_value, num_bets, profit
    try:
        common_index = y_true.index.intersection(odds_data.index)
        if len(common_index) == 0:
            return 0.0, 0, 0.0
        if len(common_index) != len(y_true):
            logger.warning("ROI Thr: Index mismatch.")
        y_true_common = y_true.loc[common_index]
        odds_common = pd.to_numeric(odds_data.loc[common_index, odd_col_name], errors='coerce')
        try:
            y_proba_series = pd.Series(y_proba, index=y_true.index)
            y_proba_common = y_proba_series.loc[common_index]
        except Exception as e:
            logger.error(f"ROI Thr: Erro alinhar y_proba: {e}")
            return None, 0, None
        bet_indices = common_index[y_proba_common > threshold]
        num_bets = len(bet_indices)
        if num_bets == 0:
            return 0.0, num_bets, 0.0
        actuals = y_true_common.loc[bet_indices]
        odds_selected = odds_common.loc[bet_indices]
        for idx in bet_indices:
            odd_d = odds_selected.loc[idx]
            if pd.notna(odd_d) and odd_d > 1:
                profit_calc += (odd_d - 1) if actuals.loc[idx] == 1 else -1
                valid_bets_count += 1
        profit = profit_calc
        roi_value = (profit / valid_bets_count) * 100 if valid_bets_count > 0 else 0.0
        return roi_value, valid_bets_count, profit
    except Exception as e:
        logger.error(f"ROI Thr: Erro - {e}", exc_info=True)
        return None, 0, None


def calculate_metrics_with_ev(y_true: pd.Series, y_proba_calibrated: np.ndarray, ev_threshold: float, odds_data: Optional[pd.DataFrame], odd_col_name: str) -> Tuple[Optional[float], int, Optional[float]]:
    profit, roi_value, num_bets_suggested, profit_calc, valid_bets_count = None, None, 0, 0, 0
    if odds_data is None or odd_col_name not in odds_data.columns:
        logger.warning(f"EV Metr: Odds ausentes.")
        return roi_value, num_bets_suggested, profit
    try:
        common_index = y_true.index.intersection(odds_data.index)
        if len(common_index) == 0:
            return 0.0, 0, 0.0
        if len(common_index) != len(y_true):
            logger.warning(f"EV Metr: Index mismatch.")
        y_true_common = y_true.loc[common_index]
        odds_common = pd.to_numeric(odds_data.loc[common_index, odd_col_name], errors='coerce')
        try:
            y_proba_common = pd.Series(y_proba_calibrated, index=y_true.index).loc[common_index]
        except Exception as e:
            logger.error(f"EV Metr: Erro alinhar y_proba: {e}")
            return None, 0, None
        valid_mask = odds_common.notna() & y_proba_common.notna() & (odds_common > 1)
        ev = pd.Series(np.nan, index=common_index)
        prob_ok = y_proba_common[valid_mask]
        odds_ok = odds_common[valid_mask]
        ev_calc = (prob_ok * (odds_ok - 1)) - ((1 - prob_ok) * 1)
        ev.loc[valid_mask] = ev_calc
        bet_indices = common_index[ev > ev_threshold]
        num_bets_suggested = len(bet_indices)
        if num_bets_suggested == 0:
            return 0.0, num_bets_suggested, 0.0
        actuals = y_true_common.loc[bet_indices]
        odds_selected = odds_common.loc[bet_indices]
        for idx in bet_indices:
            odd_d = odds_selected.loc[idx]
            if pd.notna(odd_d) and odd_d > 1:
                profit_calc += (odd_d - 1) if actuals.loc[idx] == 1 else -1
                valid_bets_count += 1
        profit = profit_calc
        roi_value = (profit / valid_bets_count) * 100 if valid_bets_count > 0 else 0.0
        logger.debug(f"    -> Métricas EV (Th={ev_threshold:.3f}): ROI={roi_value:.2f}%, Bets Sug={num_bets_suggested}, Bets Vál={valid_bets_count}, Profit={profit:.2f}")
        return roi_value, num_bets_suggested, profit
    except Exception as e:
        logger.error(f"EV Metr: Erro - {e}", exc_info=True)
        return None, 0, None


def scale_features(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, scaler_type='standard') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Any]:
    X_train_c, X_val_c, X_test_c = X_train.copy(), X_val.copy(), X_test.copy()
    scaler = None
    try:
        if scaler_type == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_type == 'standard':
            scaler = StandardScaler()
        else:
            logger.warning(f"Scaler '{scaler_type}' desconhecido.")
            scaler = StandardScaler()
        logger.info(f"  Aplicando {scaler.__class__.__name__}...")
        cols = X_train_c.columns
        X_train_c = X_train_c.replace([np.inf, -np.inf], np.nan)
        if X_train_c.isnull().values.any():
            logger.warning("  NaNs em X_train. Imputando.")
            train_median = X_train_c.median()
            X_train_c.fillna(train_median, inplace=True)
        else:
            train_median = X_train_c.median()
        if X_train_c.isnull().values.any():
            raise ValueError("NaNs persistentes X_train.")
        if X_val_c is not None:
            X_val_c = X_val_c.replace([np.inf, -np.inf], np.nan).fillna(train_median)
        if X_test_c is not None:
            X_test_c = X_test_c.replace([np.inf, -np.inf], np.nan).fillna(train_median)
        scaler.fit(X_train_c)
        X_train_scaled = scaler.transform(X_train_c)
        X_val_scaled = scaler.transform(X_val_c) if X_val_c is not None else None
        X_test_scaled = scaler.transform(X_test_c) if X_test_c is not None else None
        X_train_sc = pd.DataFrame(X_train_scaled, index=X_train.index, columns=cols)
        X_val_sc = pd.DataFrame(X_val_scaled, index=X_val.index, columns=cols) if X_val_scaled is not None else None
        X_test_sc = pd.DataFrame(X_test_scaled, index=X_test.index, columns=cols) if X_test_scaled is not None else None
        logger.info("  Scaling concluído.")
        return X_train_sc, X_val_sc, X_test_sc, scaler
    except Exception as e:
        logger.error(f"Erro GERAL scaling: {e}", exc_info=True)
        raise

# --- Função Principal de Treinamento (COM PIPELINE IMBLEARN) ---
def train_evaluate_and_save_best_models(
    X: pd.DataFrame, y: pd.Series,
    X_test_with_odds: Optional[pd.DataFrame] = None,
    progress_callback_stages: Optional[Callable[[int, str], None]] = None,
    num_total_models: int = 1,
    scaler_type: str = 'standard',
    sampler_type: str = 'smote', # 'smote', 'random', None
    odd_draw_col_name: str = ODDS_COLS.get('draw', 'Odd_D_FT'),
    calibration_method: str = 'isotonic', # 'isotonic' ou 'sigmoid'
    optimize_ev_threshold: bool = True,
    optimize_f1_threshold: bool = True,
    optimize_precision_threshold: bool = True, # Otimiza para precision também
    min_recall_target: float = MIN_RECALL_FOR_PRECISION_OPT,
    bayes_opt_n_iter: int = BAYESIAN_OPT_N_ITER,
    cv_splits: int = CROSS_VALIDATION_SPLITS,
    scoring_metric: str = 'f1', # Métrica para otimização de hiperparâmetros ('f1', 'precision', 'roc_auc', etc.)
    n_ensemble_models: int = 3
    ) -> bool:
    """
    Pipeline otimizado: Treina, calibra, otimiza limiares, avalia, salva.
    """
    # --- Validações Iniciais ---
    if not IMBLEARN_AVAILABLE and sampler_type is not None: logger.error("'imbalanced-learn' não instalado, sampler desativado."); return False
    if X is None or y is None or X.empty or y.empty: logger.error("Dados X ou y inválidos/vazios."); return False
    if not MODEL_CONFIG: logger.error("MODEL_CONFIG vazio."); return False
    available_models = {name: config for name, config in MODEL_CONFIG.items() if not (name=='LGBMClassifier' and not LGBM_AVAILABLE)}
    if not available_models: logger.error("Nenhum modelo válido configurado."); return False
    if num_total_models <= 0: num_total_models = len(available_models); logger.warning("Ajustando num_total_models.")

    feature_names = list(X.columns); all_results = []
    sampler_log = f"Sampler: {sampler_type}" if sampler_type else "Sampler: None"
    opt_log = f"Opt(F1:{optimize_f1_threshold}, EV:{optimize_ev_threshold}, Prec:{optimize_precision_threshold})"
    logger.info(f"--- Treinando {num_total_models} Modelos ({sampler_log}, CV Score: {scoring_metric}, {opt_log}) ---")
    start_time_total = time.time()

    # --- Divisão Estratificada Tripla e Alinhamento de Odds ---
    logger.info("Dividindo dados (Treino/Validação/Teste)...")
    val_size = 0.20; test_size_final = TEST_SIZE; train_val_size_temp = 1.0 - test_size_final
    if train_val_size_temp <= 0: logger.error(f"TEST_SIZE ({TEST_SIZE}) inválido."); return False
    val_size_relative = val_size / train_val_size_temp
    try:
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size_final, random_state=RANDOM_STATE, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size_relative, random_state=RANDOM_STATE, stratify=y_train_val)
        logger.info(f"Split: T={len(X_train)} ({len(X_train)/len(X):.1%}), V={len(X_val)} ({len(X_val)/len(X):.1%}), Ts={len(X_test)} ({len(X_test)/len(X):.1%})")

        X_val_odds, X_test_odds = None, None
        if X_test_with_odds is not None and not X_test_with_odds.empty and odd_draw_col_name in X_test_with_odds.columns:
            common_val = X_val.index.intersection(X_test_with_odds.index); common_test = X_test.index.intersection(X_test_with_odds.index)
            if len(common_val) > 0: X_val_odds = X_test_with_odds.loc[common_val, [odd_draw_col_name]].copy()
            if len(common_test) > 0: X_test_odds = X_test_with_odds.loc[common_test, [odd_draw_col_name]].copy()
            logger.info(f"Odds alinhadas: Val={X_val_odds is not None} ({len(common_val)}), Teste={X_test_odds is not None} ({len(common_test)})")
            if X_val_odds is None or X_test_odds is None: logger.warning("Algumas odds de Val/Teste não puderam ser alinhadas.")
        else: logger.warning("DF de odds ausente, vazio ou sem coluna de empate. ROI/EV não serão calculados.")
    except Exception as e: logger.error(f"Erro divisão/alinhar dados: {e}", exc_info=True); return False

    # Define Stratified K-Fold para CV
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

    # --- Loop principal pelos modelos ---
    for i, (model_name, config) in enumerate(available_models.items()):
        status_text_loop = f"Modelo {i+1}/{num_total_models}: {model_name}"
        logger.info(f"\n--- {status_text_loop} ---")
        if progress_callback_stages: progress_callback_stages(i, f"Iniciando {model_name}...")
        start_time_model = time.time()
        model_trained = None; best_params = None; current_scaler = None; calibrator = None; best_pipeline_object = None;
        current_optimal_f1_threshold = DEFAULT_F1_THRESHOLD
        current_optimal_ev_threshold = DEFAULT_EV_THRESHOLD
        current_optimal_precision_threshold = 0.5 # Default para limiar de precision

        try:
            # --- Setup do Modelo e Scaling ---
            model_class_name = model_name # Para logs
            try: model_class = eval(model_class_name)
            except NameError: logger.error(f"Classe do modelo '{model_class_name}' não encontrada."); continue
            model_kwargs = config.get('model_kwargs', {}); needs_scaling = config.get('needs_scaling', False)
            use_bayes = SKOPT_AVAILABLE and 'search_spaces' in config and BayesSearchCV is not GridSearchCV
            param_space_or_grid = config.get('search_spaces') if use_bayes else config.get('param_grid')

            X_train_m, X_val_m, X_test_m = X_train.copy(), X_val.copy(), X_test.copy()
            if needs_scaling:
                if progress_callback_stages: progress_callback_stages(i, f"Scaling...")
                try: X_train_m, X_val_m, X_test_m, current_scaler = scale_features(X_train_m, X_val_m, X_test_m, scaler_type); logger.info(" -> Scaling OK.")
                except Exception as e_scale: logger.error(f" ERRO scaling p/ {model_class_name}: {e_scale}", exc_info=True); continue

            # --- Criação do Pipeline (Sampler + Classifier) ---
            pipeline_steps = []; sampler_instance = None; sampler_log_name = "None"
            if sampler_type == 'smote' and SMOTE: sampler_instance = SMOTE(random_state=RANDOM_STATE); pipeline_steps.append(('sampler', sampler_instance)); sampler_log_name="SMOTE"
            elif sampler_type == 'random' and RandomOverSampler: sampler_instance = RandomOverSampler(random_state=RANDOM_STATE); pipeline_steps.append(('sampler', sampler_instance)); sampler_log_name="RandomOverSampler"
            pipeline_steps.append(('classifier', model_class(**model_kwargs)))
            pipeline = ImbPipeline(pipeline_steps)
            logger.info(f"  Pipeline criado (Sampler: {sampler_log_name}, Classifier: {model_class_name})")

            # --- Treino com Busca de Hiperparâmetros (CV) ou Padrão ---
            if param_space_or_grid:
                 search_method_name = 'BayesSearchCV' if use_bayes else 'GridSearchCV'
                 search_status_msg = f"Ajustando ({search_method_name[:5]}" + (f"+{sampler_log_name})" if sampler_instance else ")")
                 if progress_callback_stages: progress_callback_stages(i, search_status_msg);
                 logger.info(f"  Iniciando {search_method_name} (Score: {scoring_metric}, CV: {cv_splits})...")
                 # **LEMBRETE**: Parâmetros em config.py DEVEM ter prefixo 'classifier__'
                 try:
                     if use_bayes: search_cv = BayesSearchCV(estimator=pipeline, search_spaces=param_space_or_grid, n_iter=bayes_opt_n_iter, cv=skf, n_jobs=N_JOBS_GRIDSEARCH, scoring=scoring_metric, random_state=RANDOM_STATE, verbose=0)
                     else: search_cv = GridSearchCV(estimator=pipeline, param_grid=param_space_or_grid, cv=skf, n_jobs=N_JOBS_GRIDSEARCH, scoring=scoring_metric, verbose=0, error_score='raise')
                     search_cv.fit(X_train_m, y_train)
                     best_pipeline_object = search_cv.best_estimator_; model_trained = best_pipeline_object.named_steps['classifier']
                     best_params_raw = search_cv.best_params_; best_params = {k.replace('classifier__', ''): v for k, v in best_params_raw.items() if k.startswith('classifier__')}
                     best_cv_score = search_cv.best_score_; logger.info(f"    -> Busca ({search_method_name}) OK. Melhor CV {scoring_metric}: {best_cv_score:.4f}. Params: {best_params}");
                 except Exception as e_search: logger.error(f"    Erro {search_method_name}.fit: {e_search}", exc_info=True); model_trained=None; best_pipeline_object=None; logger.warning("    -> Tentando fallback...");
            if model_trained is None: # Fallback
                 fallback_status = f"Ajustando (Padrão" + (f"+{sampler_log_name})" if sampler_instance else ")")
                 if progress_callback_stages: progress_callback_stages(i, fallback_status)
                 logger.info(f"  Treinando Pipeline com params padrão...");
                 try: pipeline.fit(X_train_m, y_train); best_pipeline_object = pipeline; model_trained = best_pipeline_object.named_steps['classifier']; best_params = {k:v for k,v in model_trained.get_params().items() if k in model_kwargs}; logger.info("    -> Treino padrão OK.")
                 except Exception as e_fall: logger.error(f"    Erro treino fallback: {e_fall}", exc_info=True); continue;
            if model_trained is None: logger.error(" Falha crítica no treino."); continue; # Pula se treino falhou totalmente

            # --- Calibração ---
            y_proba_val_raw_full, y_proba_val_calib, y_proba_val_raw_draw = None, None, None; calibrator = None
            if hasattr(best_pipeline_object, "predict_proba"):
                 if progress_callback_stages: progress_callback_stages(i, "Calibrando..."); logger.info(f"  Calibrando probs ({calibration_method})...");
                 try:
                    y_proba_val_raw_full = best_pipeline_object.predict_proba(X_val_m)
                    if y_proba_val_raw_full.shape[1] >= 2: # Garante pelo menos 2 classes
                        y_proba_val_raw_draw = y_proba_val_raw_full[:, 1] # Classe 1 (Empate)
                        if calibration_method == 'isotonic': calibrator = IsotonicRegression(out_of_bounds='clip')
                        # elif calibration_method == 'sigmoid': calibrator = CalibratedClassifierCV(method='sigmoid', cv='prefit') # Requereria refit do modelo base
                        else: logger.warning(f"Método calibração '{calibration_method}' não suportado aqui. Usando Isotonic."); calibrator = IsotonicRegression(out_of_bounds='clip')
                        calibrator.fit(y_proba_val_raw_draw, y_val) # Ajusta nos dados de validação
                        y_proba_val_calib = calibrator.predict(y_proba_val_raw_draw)
                        logger.info("  -> Calibrador treinado.")
                    else: logger.warning(" predict_proba retornou shape inesperado. Calibração pulada.")
                 except Exception as e_calib: logger.error(f"  Erro durante calibração: {e_calib}", exc_info=True); calibrator=None;
            else: logger.warning(f"  {model_class_name} não tem predict_proba. Calibração pulada.");

            # --- Otimização de Limiares (Validação) ---
            proba_opt_val = y_proba_val_calib if calibrator else y_proba_val_raw_draw
            opt_src_val = 'Calib' if calibrator else ('Raw' if proba_opt_val is not None else 'N/A')

            # Otimização F1
            if optimize_f1_threshold and proba_opt_val is not None:
                if progress_callback_stages: progress_callback_stages(i, f"Otimizando F1 Thr ({opt_src_val})..."); logger.info(f"  Otimizando F1 (Val Probs {opt_src_val})...");
                try: # ... (lógica otimização F1 como antes) ...
                     p,r,t=precision_recall_curve(y_val,proba_opt_val); f1=np.divide(2*p*r,p+r+FEATURE_EPSILON); idx=np.argmax(f1[:-1]); current_optimal_f1_threshold=t[idx]; best_val_f1=f1[idx]; logger.info(f"    Limiar F1 ótimo(Val): {current_optimal_f1_threshold:.4f} (F1={best_val_f1:.4f})")
                except Exception as e_f1: logger.error(f"  Erro otim F1: {e_f1}"); current_optimal_f1_threshold=DEFAULT_F1_THRESHOLD

            # Otimização EV
            if optimize_ev_threshold and proba_opt_val is not None and X_val_odds is not None:
                if progress_callback_stages: progress_callback_stages(i, f"Otimizando EV Thr ({opt_src_val})..."); logger.info(f"  Otimizando EV (Val Probs {opt_src_val})...");
                try: # ... (lógica otimização EV como antes) ...
                     best_val_roi_ev=-np.inf; ev_ths=np.linspace(0.0,0.20,21);
                     for ev_th in ev_ths: 
                        val_roi,_,_=calculate_metrics_with_ev(y_val,proba_opt_val,ev_th,X_val_odds,odd_draw_col_name);
                        if val_roi is not None and val_roi>best_val_roi_ev: best_val_roi_ev=val_roi; current_optimal_ev_threshold=ev_th;
                     if best_val_roi_ev > -np.inf: logger.info(f"    Limiar EV ótimo(Val): {current_optimal_ev_threshold:.3f} (ROI={best_val_roi_ev:.2f}%)")
                     else: logger.warning("    ROI Val inválido."); current_optimal_ev_threshold = DEFAULT_EV_THRESHOLD;
                except Exception as e_ev: logger.error(f"  Erro otim EV: {e_ev}"); current_optimal_ev_threshold = DEFAULT_EV_THRESHOLD;
            elif optimize_ev_threshold: logger.warning(f"  Otim EV pulada (sem probs/odds Val).")

            # Otimização Precision
            if optimize_precision_threshold and proba_opt_val is not None:
                if progress_callback_stages: progress_callback_stages(i, f"Otimizando Prec Thr ({opt_src_val})...")
                logger.info(f"  Otimizando Precision (Val Probs {opt_src_val}, Recall Mín: {min_recall_target:.1%})...");
                # ... (lógica otimização Precision como antes) ...
                best_val_precision=-1.0; found_prec_thr=False; thresholds_test=np.linspace(np.min(proba_opt_val)+1e-4,np.min([np.max(proba_opt_val)-1e-4,0.99]),100)
                for t in thresholds_test: 
                    y_pred_v=(proba_opt_val>=t).astype(int); 
                    prec=precision_score(y_val,y_pred_v,pos_label=1,zero_division=0); 
                    rec=recall_score(y_val,y_pred_v,pos_label=1,zero_division=0);
                    if rec >= min_recall_target and prec >= best_val_precision:
                        if prec > best_val_precision or t > current_optimal_precision_threshold: best_val_precision=prec; current_optimal_precision_threshold=t; found_prec_thr=True;
                if found_prec_thr: final_rec=recall_score(y_val,(proba_opt_val>=current_optimal_precision_threshold).astype(int),pos_label=1,zero_division=0); logger.info(f"    Limiar Prec ótimo(Val): {current_optimal_precision_threshold:.4f} (Prec={best_val_precision:.4f}, Rec={final_rec:.4f})")
                else: logger.warning(f"    Não encontrou limiar p/ Recall Mín {min_recall_target:.1%}."); current_optimal_precision_threshold=0.5

            # --- Avaliação Final no Teste ---
            if progress_callback_stages: progress_callback_stages(i, f"Avaliando..."); logger.info(f"  Avaliando Pipeline/Modelo no Teste...")
            metrics = {'optimal_f1_threshold': current_optimal_f1_threshold, 'optimal_ev_threshold': current_optimal_ev_threshold, 'optimal_precision_threshold': current_optimal_precision_threshold}
            y_proba_test_raw_full, y_proba_test_raw_draw, y_proba_test_calib = None, None, None
            if hasattr(best_pipeline_object, "predict_proba"):
                 try: # ... (predict_proba no teste e aplica calibração) ...
                    y_proba_test_raw_full = best_pipeline_object.predict_proba(X_test_m);
                    if y_proba_test_raw_full.shape[1]>=2: 
                        y_proba_test_raw_draw=y_proba_test_raw_full[:,1];
                        if calibrator: 
                            y_proba_test_calib=calibrator.predict(y_proba_test_raw_draw)
                        else: 
                            y_proba_test_calib=y_proba_test_raw_draw
                 except Exception as e_pred_test: logger.error(f"  Erro predict_proba teste: {e_pred_test}");
            proba_eval_test = y_proba_test_calib if calibrator else y_proba_test_raw_draw
            eval_src_test = 'Calib' if calibrator else ('Raw' if proba_eval_test is not None else 'N/A')
            logger.info(f"  Calculando métricas teste (probs: {eval_src_test})")
            # Calcula métricas para os 3 limiares (0.5, F1, Prec)
            try: # Thr 0.5
                y_pred05=best_pipeline_object.predict(X_test_m); # ... (resto cálculo métricas 0.5)
                metrics.update({'accuracy_thr05':accuracy_score(y_test,y_pred05),'precision_draw_thr05':precision_score(y_test,y_pred05,pos_label=1,zero_division=0),'recall_draw_thr05':recall_score(y_test,y_pred05,pos_label=1,zero_division=0),'f1_score_draw_thr05':f1_score(y_test,y_pred05,pos_label=1,zero_division=0)}); logger.info(f"    -> Métricas @ Thr 0.5: F1={metrics['f1_score_draw_thr05']:.4f}, P={metrics['precision_draw_thr05']:.4f}, R={metrics['recall_draw_thr05']:.4f}")
            except Exception as e: logger.error(f"  Erro métricas @ 0.5: {e}")
            if proba_eval_test is not None:
                 try: # Thr F1
                     y_predF1=(proba_eval_test>=current_optimal_f1_threshold).astype(int); # ... (resto cálculo métricas F1)
                     metrics.update({'accuracy_thrF1':accuracy_score(y_test,y_predF1),'precision_draw_thrF1':precision_score(y_test,y_predF1,pos_label=1,zero_division=0),'recall_draw_thrF1':recall_score(y_test,y_predF1,pos_label=1,zero_division=0),'f1_score_draw':f1_score(y_test,y_predF1,pos_label=1,zero_division=0)}); logger.info(f"    -> Métricas @ Thr F1 ({current_optimal_f1_threshold:.4f}): F1={metrics['f1_score_draw']:.4f}, P={metrics['precision_draw_thrF1']:.4f}, R={metrics['recall_draw_thrF1']:.4f}")
                 except Exception as e: logger.error(f"  Erro métricas @ Thr F1: {e}"); metrics['f1_score_draw']=metrics.get('f1_score_draw_thr05',-1.0);
                 try: # Thr Prec
                     y_predPrec=(proba_eval_test>=current_optimal_precision_threshold).astype(int); # ... (resto cálculo métricas Prec)
                     metrics.update({'accuracy_thrPrec':accuracy_score(y_test,y_predPrec),'precision_draw_thrPrec':precision_score(y_test,y_predPrec,pos_label=1,zero_division=0),'recall_draw_thrPrec':recall_score(y_test,y_predPrec,pos_label=1,zero_division=0),'f1_score_draw_thrPrec':f1_score(y_test,y_predPrec,pos_label=1,zero_division=0)}); logger.info(f"    -> Métricas @ Thr Prec ({current_optimal_precision_threshold:.4f}): F1={metrics['f1_score_draw_thrPrec']:.4f}, P={metrics['precision_draw_thrPrec']:.4f}, R={metrics['recall_draw_thrPrec']:.4f}")
                 except Exception as e: logger.error(f"  Erro métricas @ Thr Prec: {e}")
            else: # Fallback se não houver probs
                 metrics['f1_score_draw']=metrics.get('f1_score_draw_thr05',-1.0); logger.info(" Usando métricas @ 0.5 como F1 final.")
            # Define métricas principais padrão (baseadas no Thr F1)
            metrics['precision_draw']=metrics.get('precision_draw_thrF1',metrics.get('precision_draw_thr05',0.0)); metrics['recall_draw']=metrics.get('recall_draw_thrF1',metrics.get('recall_draw_thr05',0.0));
            # Métricas Probabilísticas
            logloss,auc,brier=None,None,None;
            if proba_eval_test is not None: # ... (cálculo logloss, auc, brier como antes) ...
                try: logloss=log_loss(y_test,y_proba_test_raw_full) if y_proba_test_raw_full is not None else None
                except: pass
                try: auc=roc_auc_score(y_test,proba_eval_test) if len(np.unique(y_test))>1 else None
                except: pass
                try: brier=brier_score_loss(y_test,proba_eval_test)
                except: pass
            metrics.update({'log_loss':logloss,'roc_auc':auc,'brier_score':brier}); logger.info(f"    -> AUC({eval_src_test})={auc:.4f}, Brier({eval_src_test})={brier:.4f}" if auc is not None else f"    -> AUC/Brier({eval_src_test})=N/A")
            # Métricas EV/ROI
            roi_ev,bets_ev,profit_ev = calculate_metrics_with_ev(y_test,proba_eval_test,current_optimal_ev_threshold,X_test_odds,odd_draw_col_name) if proba_eval_test is not None and X_test_odds is not None else (None,0,None);
            metrics.update({'roi':roi_ev,'num_bets':bets_ev,'profit':profit_ev});
            # Tamanhos
            metrics.update({'train_set_size':len(y_train),'val_set_size':len(y_val),'test_set_size':len(y_test)});

            # --- Guarda resultado ---
            all_results.append({
                'model_name': model_class_name,
                'model_object': model_trained, # O classificador ajustado
                'pipeline_object': best_pipeline_object, # <<< ADICIONADO: O pipeline completo ajustado
                'scaler': current_scaler,
                'calibrator': calibrator,
                'params': best_params,
                'metrics': metrics,
                'optimal_f1_threshold': current_optimal_f1_threshold,
                'optimal_ev_threshold': current_optimal_ev_threshold
                # O limiar de precision já está dentro de 'metrics'
            })            

            if progress_callback_stages: progress_callback_stages(i + 1, f"Modelo {model_class_name} OK");
            logger.info(f"    ==> Resultado {model_class_name} adicionado.")
        except Exception as e_outer: logger.error(f"Erro GERAL loop {model_class_name}: {e_outer}", exc_info=True); continue
        logger.info(f"  Tempo p/ {model_class_name}: {time.time() - start_time_model:.2f} seg.")
    # --- Fim do Loop for ---
    # --- CRIAÇÃO E AVALIAÇÃO DO ENSEMBLE (VOTING CLASSIFIER) ---
    if len(all_results) >= 2: # Precisa de pelo menos 2 modelos
        # Ordena resultados individuais por F1 para pegar os melhores
        all_results.sort(key=lambda r: r['metrics'].get('f1_score_draw', -1.0), reverse=True)
        top_n_results = all_results[:n_ensemble_models] # Usa n_ensemble_models do argumento da função
        logger.info(f"\n--- Criando Ensemble com Top {len(top_n_results)} Modelos (por F1): {[r['model_name'] for r in top_n_results]} ---")

        ensemble_estimators = []
        any_needs_scaling_in_ensemble = False # Flag para dados do ensemble

        for result in top_n_results:
            model_name = result['model_name']
            # Pega o PIPELINE treinado do resultado individual
            trained_pipeline = result.get('pipeline_object')
            if trained_pipeline is None:
                 logger.warning(f"Pipeline treinado não encontrado para {model_name}. Pulando para ensemble.")
                 continue
            model_id = f"{model_name}_{all_results.index(result)}" # ID único

            # Clona o pipeline treinado para o ensemble
            pipeline_clone = clone(trained_pipeline)
            ensemble_estimators.append((model_id, pipeline_clone))
            logger.info(f"  Adicionando '{model_name}' (pipeline clonado) ao ensemble.")
            # Verifica se algum pipeline base continha um scaler
            if 'scaler' in pipeline_clone.named_steps:
                 any_needs_scaling_in_ensemble = True

        if not ensemble_estimators:
            logger.error("ENSEMBLE: Nenhum estimador válido para criar ensemble.")
            # Continua para salvar os individuais mesmo se ensemble falhar
        else:
            # Cria e TREINA (refit) o Voting Classifier nos dados de TREINO COMPLETOS
            voting_clf = VotingClassifier(estimators=ensemble_estimators, voting='soft', n_jobs=N_JOBS_GRIDSEARCH)
            logger.info("  Ajustando VotingClassifier final nos dados de treino...")
            try:
                # Usa dados originais (X_train), pois pipelines internos lidam com scaling/sampling
                voting_clf.fit(X_train, y_train)
                logger.info("  VotingClassifier ajustado com sucesso.")

                # Avalia o Ensemble no Teste
                logger.info("  Avaliando Ensemble no Teste...")
                y_pred_ensemble_05 = voting_clf.predict(X_test)
                y_proba_ensemble_full = voting_clf.predict_proba(X_test)
                ensemble_metrics = {'model_name': 'VotingEnsemble'} # Inicializa dict de métricas

                if y_proba_ensemble_full.shape[1] >= 2:
                    y_proba_ensemble_draw = y_proba_ensemble_full[:, 1]
                     # Métricas @ 0.5
                    ensemble_metrics['f1_score_draw_thr05']=f1_score(y_test,y_pred_ensemble_05,pos_label=1,zero_division=0)
                    ensemble_metrics['precision_draw_thr05']=precision_score(y_test,y_pred_ensemble_05,pos_label=1,zero_division=0)
                    ensemble_metrics['recall_draw_thr05']=recall_score(y_test,y_pred_ensemble_05,pos_label=1,zero_division=0)

                     # Calibração e Otimização de Limiares para o ENSEMBLE (no Teste)
                    logger.info("  Calibrando/Otimizando limiares p/ Ensemble (no Teste)...")
                     # (Isso pode ser movido para validação se necessário, mas simplifica aqui)
                    ens_calibrator = IsotonicRegression(out_of_bounds='clip').fit(y_proba_ensemble_draw, y_test)
                    y_proba_ens_calib = ens_calibrator.predict(y_proba_ensemble_draw)

                     # Otimiza F1 para Ensemble
                    try: 
                        p,r,t=precision_recall_curve(y_test,y_proba_ens_calib); 
                        f1=np.divide(2*p*r,p+r+FEATURE_EPSILON); 
                        idx=np.argmax(f1[:-1]); 
                        ens_f1_thr=t[idx];
                    except: 
                        ens_f1_thr = DEFAULT_F1_THRESHOLD
                    y_pred_ens_f1=(y_proba_ens_calib >= ens_f1_thr).astype(int)
                    ensemble_metrics['optimal_f1_threshold']=ens_f1_thr
                    ensemble_metrics['f1_score_draw']=f1_score(y_test,y_pred_ens_f1,pos_label=1,zero_division=0)
                    ensemble_metrics['precision_draw_thrF1']=precision_score(y_test,y_pred_ens_f1,pos_label=1,zero_division=0)
                    ensemble_metrics['recall_draw_thrF1']=recall_score(y_test,y_pred_ens_f1,pos_label=1,zero_division=0)

                     # Otimiza Precision para Ensemble
                    try: 
                        best_prec_ens=-1.0; ens_prec_thr=0.5; found_prec_ens=False; thresholds_test=np.linspace(np.min(y_proba_ens_calib)+1e-4,np.min([np.max(y_proba_ens_calib)-1e-4,0.99]),100);
                        for t in thresholds_test: 
                            y_pred_v=(y_proba_ens_calib>=t).astype(int); prec=precision_score(y_test,y_pred_v,pos_label=1,zero_division=0); rec=recall_score(y_test,y_pred_v,pos_label=1,zero_division=0);
                            if rec>=min_recall_target and prec>=best_prec_ens:
                                if prec>best_prec_ens or t>ens_prec_thr: 
                                    best_prec_ens=prec; ens_prec_thr=t; found_prec_ens=True;
                    except: ens_prec_thr = 0.5
                    y_pred_ens_prec=(y_proba_ens_calib>=ens_prec_thr).astype(int); ensemble_metrics['optimal_precision_threshold']=ens_prec_thr; ensemble_metrics['precision_draw_thrPrec']=precision_score(y_test,y_pred_ens_prec,pos_label=1,zero_division=0); ensemble_metrics['recall_draw_thrPrec']=recall_score(y_test,y_pred_ens_prec,pos_label=1,zero_division=0); ensemble_metrics['f1_score_draw_thrPrec']=f1_score(y_test,y_pred_ens_prec,pos_label=1,zero_division=0);

                     # Métricas Probabilísticas Ensemble
                    try: ensemble_metrics['roc_auc']=roc_auc_score(y_test,y_proba_ens_calib)
                    except: ensemble_metrics['roc_auc']=None
                    try: ensemble_metrics['brier_score']=brier_score_loss(y_test,y_proba_ens_calib)
                    except: ensemble_metrics['brier_score']=None
                    try: ensemble_metrics['log_loss']=log_loss(y_test,y_proba_ensemble_full) # Usa probs brutas
                    except: ensemble_metrics['log_loss']=None

                     # Otimiza EV para Ensemble
                    try: 
                        best_roi_ens=-np.inf; ens_ev_thr=DEFAULT_EV_THRESHOLD; ev_ths=np.linspace(0.0,0.2,21);
                        for ev_th in ev_ths: 
                            val_roi,_,_=calculate_metrics_with_ev(y_test,y_proba_ens_calib,ev_th,X_test_odds,odd_draw_col_name);
                            if val_roi is not None and val_roi>best_roi_ens: best_roi_ens=val_roi; ens_ev_thr=ev_th;
                    except: 
                        ens_ev_thr=DEFAULT_EV_THRESHOLD
                        ensemble_metrics['optimal_ev_threshold']=ens_ev_thr; ens_roi,ens_bets,ens_prof = calculate_metrics_with_ev(y_test,y_proba_ens_calib,ens_ev_thr,X_test_odds,odd_draw_col_name); ensemble_metrics.update({'roi':ens_roi,'num_bets':ens_bets,'profit':ens_prof})

                     # Log das Métricas do Ensemble
                    logger.info("  --- Métricas Ensemble (Teste) ---")
                    logger.info(f"    F1 @ThrF1({ens_f1_thr:.4f}) = {ensemble_metrics.get('f1_score_draw',np.nan):.4f}")
                    logger.info(f"    Prec @ThrPrec({ens_prec_thr:.4f}) = {ensemble_metrics.get('precision_draw_thrPrec',np.nan):.4f} (Recall: {ensemble_metrics.get('recall_draw_thrPrec',np.nan):.4f})")
                    logger.info(f"    AUC = {ensemble_metrics.get('roc_auc',np.nan):.4f}, Brier = {ensemble_metrics.get('brier_score',np.nan):.4f}")
                    roi_ens_log = ensemble_metrics.get('roi', np.nan); roi_str = f"{roi_ens_log:.2f}%" if pd.notna(roi_ens_log) else "N/A"
                    logger.info(f"    ROI @ThrEV({ens_ev_thr:.3f}) = {roi_str} ({ensemble_metrics.get('num_bets')} bets)")
                    logger.info("  ---------------------------------")

                     # Adiciona resultado do ensemble à lista geral (para possível exibição ou análise futura)
                     # Nota: Não salva o objeto ensemble por padrão
                    all_results.append({
                         'model_name': 'VotingEnsemble', 'model_object': voting_clf, 'pipeline_object': None,
                         'scaler': None, 'calibrator': ens_calibrator, 'params': {'estimators': [e[0] for e in ensemble_estimators]},
                         'metrics': ensemble_metrics, 'optimal_f1_threshold': ens_f1_thr, 'optimal_ev_threshold': ens_ev_thr
                     })
                else: logger.warning(" Ensemble predict_proba não retornou 2 colunas.")
            except Exception as e_ens: logger.error(f"Erro treinar/avaliar Voting Ensemble: {e_ens}", exc_info=True)
    else: logger.warning("Modelos insuficientes para ensemble.")
    # --- FIM ENSEMBLE ---

    # --- Seleção e Salvamento dos MELHORES INDIVIDUAIS ---
    # (Lógica de seleção e salvamento como antes, operando sobre a lista filtrada)
    if progress_callback_stages: progress_callback_stages(num_total_models, "Selecionando/Salvando Indiv...") # Ajusta msg progresso
    logger.info(f"--- Processando {len(all_results)} resultados (incluindo ensemble) para salvar melhores individuais ---")
    try:
        # Filtra resultados para operar apenas nos individuais para seleção/salvamento
        individual_results_for_selection = [r for r in all_results if r['model_name'] != 'VotingEnsemble']
        if not individual_results_for_selection: logger.error("Nenhum resultado individual para selecionar."); return False
        results_df = pd.DataFrame(individual_results_for_selection)
    except Exception as e: logger.error(f"Erro ao criar DataFrame de resultados individuais: {e}", exc_info=True)
    

    # --- Seleção e Salvamento Final ---
    if progress_callback_stages: progress_callback_stages(num_total_models, "Selecionando/Salvando...")
    end_time_total=time.time(); logger.info(f"--- Treino concluído ({end_time_total-start_time_total:.2f} seg) ---")
    if not all_results: logger.error("SELEÇÃO: Nenhum resultado válido."); return False
    try:
        results_df = pd.DataFrame(all_results)
        # Extrai métricas para o DataFrame (incluindo as de precision)
        for thr_type in ['thrF1', 'thrPrec']: # Itera para pegar métricas dos dois limiares principais
            results_df[f'precision_draw_{thr_type}'] = results_df['metrics'].apply(lambda m: m.get(f'precision_draw_{thr_type}', 0.0))
            results_df[f'recall_draw_{thr_type}'] = results_df['metrics'].apply(lambda m: m.get(f'recall_draw_{thr_type}', 0.0))
            results_df[f'f1_score_draw_{thr_type}'] = results_df['metrics'].apply(lambda m: m.get(f'f1_score_draw_{thr_type}', -1.0))
        results_df['f1_score_draw']=results_df['metrics'].apply(lambda m: m.get('f1_score_draw', -1.0)) # F1 principal (do thr F1)
        results_df['roi']=results_df['metrics'].apply(lambda m: m.get('roi', -np.inf)); results_df['num_bets']=results_df['metrics'].apply(lambda m: m.get('num_bets', 0));
        results_df['auc']=results_df['metrics'].apply(lambda m: m.get('roc_auc', 0.0)); results_df['brier']=results_df['metrics'].apply(lambda m: m.get('brier_score', 1.0))
        results_df['optimal_f1_threshold'] = results_df['metrics'].apply(lambda m: m.get('optimal_f1_threshold', DEFAULT_F1_THRESHOLD))
        results_df['optimal_ev_threshold'] = results_df['metrics'].apply(lambda m: m.get('optimal_ev_threshold', DEFAULT_EV_THRESHOLD))
        results_df['optimal_precision_threshold'] = results_df['metrics'].apply(lambda m: m.get('optimal_precision_threshold', 0.5))

        # ... (Conversão numérica e fillna como antes, adaptado para novas colunas) ...
        cols_num = ['f1_score_draw','precision_draw_thrF1','recall_draw_thrF1','precision_draw_thrPrec','recall_draw_thrPrec','f1_score_draw_thrPrec','optimal_precision_threshold','roi','num_bets','auc','brier','optimal_f1_threshold','optimal_ev_threshold']
        for col in cols_num: results_df[col]=pd.to_numeric(results_df[col],errors='coerce')
        results_df.fillna({'f1_score_draw': -1.0, 'precision_draw_thrF1': 0.0, 'recall_draw_thrF1': 0.0, 'precision_draw_thrPrec': 0.0, 'recall_draw_thrPrec': 0.0, 'f1_score_draw_thrPrec': -1.0, 'optimal_precision_threshold': 0.5, 'roi': -np.inf, 'num_bets': 0, 'auc': 0.0, 'brier': 1.0, 'optimal_f1_threshold': DEFAULT_F1_THRESHOLD, 'optimal_ev_threshold': DEFAULT_EV_THRESHOLD}, inplace=True)

        if results_df.empty: logger.error("SELEÇÃO: DF resultados vazio."); return False

        # Exibe Comparativo (inclui novas colunas de precision)
        logger.info("--- Comparativo Desempenho Modelos (Teste) ---");
        display_cols = ['model_name','f1_score_draw','precision_draw_thrF1','recall_draw_thrF1','precision_draw_thrPrec','recall_draw_thrPrec','f1_score_draw_thrPrec','auc','brier','roi','num_bets','optimal_f1_threshold','optimal_ev_threshold','optimal_precision_threshold']
        display_cols_exist=[col for col in display_cols if col in results_df.columns]
        results_df_display=results_df[display_cols_exist].round(4) # Arredonda tudo para 4 casas por simplicidade
        if 'num_bets' in results_df_display.columns: results_df_display['num_bets']=results_df_display['num_bets'].astype(int)
        if 'roi' in results_df_display.columns: results_df_display['roi'] = results_df_display['roi'].round(2)
        try: logger.info("\n"+results_df_display.sort_values(by=BEST_MODEL_METRIC,ascending=False).to_markdown(index=False))
        except ImportError: logger.info("\n"+results_df_display.sort_values(by=BEST_MODEL_METRIC,ascending=False).to_string(index=False))
        logger.info("-" * 80)

        # Seleção e Salvamento (Lógica baseada em F1 e ROI como antes)
        # ... (código de seleção e salvamento como antes, _save_model_object já salva o limiar de precision) ...
        results_df_sorted_f1=results_df.sort_values(by=BEST_MODEL_METRIC,ascending=False).reset_index(drop=True); best_f1_result=results_df_sorted_f1.iloc[0].to_dict().copy(); #... (log melhor F1)
        best_roi_result=None; logger.info("--- Ranking ROI ---"); #... (código ranking ROI)
        results_df_sorted_roi=results_df.sort_values(by=BEST_MODEL_METRIC_ROI,ascending=False).reset_index(drop=True); #... (display ranking ROI)
        if not results_df_sorted_roi.empty: #... (lógica find best_roi_result)
             for _,row in results_df_sorted_roi.iterrows():
                 cr=row[BEST_MODEL_METRIC_ROI]; cn=row['model_name']; ev_thr=row.get('optimal_ev_threshold',DEFAULT_EV_THRESHOLD)
                 if isinstance(cr,(int,float,np.number)) and cr > -np.inf: best_roi_result=row.to_dict().copy(); logger.info(f"SELEÇÃO: Melhor ROI Válido: {cn} (ROI={cr:.2f}% @ EV Thr={ev_thr:.3f})"); break;
             if best_roi_result is None: logger.warning("SELEÇÃO: Nenhum ROI válido.")
        else: logger.warning("SELEÇÃO: Ranking ROI vazio.")
        model_to_save_f1=best_f1_result; model_to_save_roi=None; #... (lógica fallback ROI)
        if best_roi_result:
            if best_f1_result.get('model_name')==best_roi_result.get('model_name'):
                 if len(results_df_sorted_f1)>1: model_to_save_roi=results_df_sorted_f1.iloc[1].to_dict().copy(); logger.info(f"  -> Usando 2º F1 ({model_to_save_roi.get('model_name')}) p/ slot ROI.")
                 else: model_to_save_roi=best_f1_result.copy(); logger.warning("  -> Apenas 1 modelo, F1 p/ ambos.")
            else: model_to_save_roi=best_roi_result.copy(); logger.info(f"  -> Usando Melhor ROI ({model_to_save_roi.get('model_name')}) p/ slot ROI.")
        else: model_to_save_roi=best_f1_result.copy(); logger.warning(f"  -> Nenhum ROI válido, F1 p/ ambos.")
        logger.info(f"Salvando Melhor F1 ({model_to_save_f1.get('model_name','ERRO')})..."); _save_model_object(model_to_save_f1, feature_names, BEST_F1_MODEL_SAVE_PATH)
        if model_to_save_roi: logger.info(f"Salvando Melhor ROI/2nd F1 ({model_to_save_roi.get('model_name','ERRO')})..."); _save_model_object(model_to_save_roi, feature_names, BEST_ROI_MODEL_SAVE_PATH)
        else: logger.error("ERRO SALVAR: model_to_save_roi é None."); return False

    except Exception as e_select_save: logger.error(f"Erro GERAL seleção/salvamento: {e_select_save}", exc_info=True); return False;

    logger.info("--- Processo Completo ---"); return True


# --- Função _save_model_object (Sem alterações, já salva o necessário) ---
def _save_model_object(model_result_dict: Dict, feature_names: List[str], file_path: str) -> None:
    if not isinstance(model_result_dict, dict): logger.error(f"Salvar: Dados inválidos p/ {file_path}"); return
    try:
        model_to_save = model_result_dict.get('model_object')
        if model_to_save is None: logger.error(f"Salvar: Modelo ausente p/ {file_path}"); return
        metrics_dict = model_result_dict.get('metrics', {})
        optimal_precision_thr = metrics_dict.get('optimal_precision_threshold', 0.5) # Fallback 0.5
        save_obj = {
            'model': model_to_save, 'scaler': model_result_dict.get('scaler'), 'calibrator': model_result_dict.get('calibrator'),
            'feature_names': feature_names, 'best_params': model_result_dict.get('params'), 'eval_metrics': metrics_dict,
            'optimal_ev_threshold': model_result_dict.get('optimal_ev_threshold', DEFAULT_EV_THRESHOLD),
            'optimal_f1_threshold': model_result_dict.get('optimal_f1_threshold', DEFAULT_F1_THRESHOLD),
            'optimal_precision_threshold': optimal_precision_thr, # <<< SALVO AQUI
            'save_timestamp': datetime.datetime.now().isoformat(), 'model_class_name': model_to_save.__class__.__name__ }
        joblib.dump(save_obj, file_path)
        logger.info(f"  -> Objeto salvo: '{save_obj['model_class_name']}' (F1 Thr={save_obj['optimal_f1_threshold']:.4f}, EV Thr={save_obj['optimal_ev_threshold']:.3f}, Prec Thr={save_obj['optimal_precision_threshold']:.4f}) em {os.path.basename(file_path)}.")
    except Exception as e: logger.error(f"  -> Erro GRAVE ao salvar objeto em {file_path}: {e}", exc_info=True)


# --- Funções Remanescentes (analyze_features, optimize_single_model) ---
# (Mantidas como estavam no seu código original - Copie as definições completas aqui)
def analyze_features(X: pd.DataFrame, y: pd.Series) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """ Analisa features: importância (RF rápido) e correlação. Retorna DFs ou (None, None). """
    logger.info("--- ANÁLISE FEATURES (model_trainer): Iniciando ---")
    imp_df = None
    corr_df = None # Use corr_df instead of corr_matrix to be consistent

    # --- Input Validation ---
    if X is None or y is None:
        logger.error("ANÁLISE FEATURES: Input X ou y é None.")
        return None, None
    if X.empty or y.empty:
        logger.error("ANÁLISE FEATURES: Input X ou y está vazio.")
        return None, None
    if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
         logger.error("ANÁLISE FEATURES: Input X ou y não é do tipo esperado (DataFrame, Series).")
         return None, None

    # --- Alignment (Crucial!) ---
    if not X.index.equals(y.index):
        logger.warning("ANÁLISE FEATURES: Índices X/y não idênticos. Tentando alinhar...")
        try:
            common_index = X.index.intersection(y.index)
            if len(common_index) == 0:
                 logger.error("ANÁLISE FEATURES: Nenhum índice em comum entre X e y após intersection.")
                 return None, None
            X = X.loc[common_index]
            y = y.loc[common_index]
            logger.info(f"ANÁLISE FEATURES: Alinhamento OK. Novo shape X: {X.shape}, y: {y.shape}")
        except Exception as e:
            logger.error(f"ANÁLISE FEATURES: Erro durante alinhamento: {e}", exc_info=True)
            return None, None

    # --- Final check for data after alignment ---
    if X.empty or y.empty:
        logger.error("ANÁLISE FEATURES: X ou y vazio após alinhamento.")
        return None, None

    feature_names = X.columns.tolist()

    # --- 1. Calcular Importância (RF) ---
    logger.info("ANÁLISE FEATURES: Calculando importância RF...")
    try:
        # Check for NaNs/Infs *before* fitting RF
        X_rf = X.copy() # Trabalha em uma cópia
        y_rf = y.copy()

        # Substitui Inf por NaN
        X_rf = X_rf.replace([np.inf, -np.inf], np.nan)

        # Verifica NaNs e imputa se necessário (embora dropna deva ter ocorrido antes)
        if X_rf.isnull().values.any():
             nan_cols = X_rf.columns[X_rf.isnull().any()].tolist()
             logger.warning(f"ANÁLISE FEATURES (RF): NaNs encontrados em X (colunas: {nan_cols}) antes do fit RF. Imputando com mediana.")
             X_rf.fillna(X_rf.median(), inplace=True)
             # Verifica novamente se ainda há NaNs (coluna inteira era NaN?)
             if X_rf.isnull().values.any():
                 raise ValueError("NaNs persistentes em X após imputação para RF.")

        if y_rf.isnull().values.any():
             logger.error("ANÁLISE FEATURES (RF): NaNs encontrados em y! Não pode treinar RF.")
             raise ValueError("Target variable (y) contains NaNs for RF importance.")

        # Ensure y is integer type for classification
        y_rf = y_rf.astype(int)

        rf_analyzer = RandomForestClassifier(n_estimators=50, max_depth=15, n_jobs=-1, min_samples_leaf=3, random_state=RANDOM_STATE)
        logger.info(f"    -> Fitting RF (X shape: {X_rf.shape}, y shape: {y_rf.shape})")
        rf_analyzer.fit(X_rf, y_rf) # Usa dados limpos/imputados
        logger.info("    -> Fit RF concluído.")

        importances = rf_analyzer.feature_importances_
        imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        imp_df = imp_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        logger.info(f"ANÁLISE FEATURES: Importância calculada OK. Shape: {imp_df.shape}")

    except Exception as e:
        logger.error(f"ANÁLISE FEATURES: Erro GRAVE ao calcular importância RF: {e}", exc_info=True)
        imp_df = None # Set to None on failure

    # --- 2. Calcular Correlação ---
    logger.info("ANÁLISE FEATURES: Calculando correlação...")
    try:
        df_temp = X.copy() # Start with the aligned X
        df_temp['target_IsDraw'] = y # Add the aligned y

        # Select only numeric columns for correlation calculation
        # Garante que o alvo (mesmo que não numérico inicialmente) seja incluído se possível
        cols_for_corr = df_temp.select_dtypes(include=np.number).columns.tolist()
        if 'target_IsDraw' not in cols_for_corr and 'target_IsDraw' in df_temp.columns:
            try:
                # Tenta converter o alvo para numérico para correlação
                df_temp['target_IsDraw'] = pd.to_numeric(df_temp['target_IsDraw'], errors='raise')
                cols_for_corr.append('target_IsDraw')
            except (ValueError, TypeError):
                logger.warning("ANÁLISE FEATURES (Corr): Coluna alvo 'target_IsDraw' não é numérica e não pôde ser convertida. Correlação com alvo não será calculada.")

        df_numeric_temp = df_temp[cols_for_corr].copy()


        # Check and handle infinities *before* calculating correlation
        if df_numeric_temp.isin([np.inf, -np.inf]).values.any():
            inf_cols = df_numeric_temp.columns[df_numeric_temp.isin([np.inf, -np.inf]).any()].tolist()
            logger.warning(f"ANÁLISE FEATURES (Corr): Valores infinitos encontrados antes de .corr() (colunas: {inf_cols}). Substituindo por NaN.")
            df_numeric_temp.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Check for columns with all NaNs after handling inf (corr fails on these)
        all_nan_cols = df_numeric_temp.columns[df_numeric_temp.isnull().all()].tolist()
        if all_nan_cols:
             logger.warning(f"ANÁLISE FEATURES (Corr): Colunas inteiras com NaN encontradas: {all_nan_cols}. Serão excluídas da correlação.")
             df_numeric_temp.drop(columns=all_nan_cols, inplace=True)
             if 'target_IsDraw' not in df_numeric_temp.columns and 'target_IsDraw' in all_nan_cols:
                  logger.error("ANÁLISE FEATURES (Corr): Coluna alvo foi removida (toda NaN?). Não é possível calcular correlação com o alvo.")
                  corr_df = None # Set corr_df to None explicitly


        if not df_numeric_temp.empty and 'target_IsDraw' in df_numeric_temp.columns:
            logger.info(f"    -> Calculando corr() em df_numeric_temp (shape: {df_numeric_temp.shape})")
            # Calculate correlation only on the numeric (and finite) data
            corr_matrix = df_numeric_temp.corr(method='pearson') # Ou 'spearman'

            # Extract only the correlation with the target variable
            if 'target_IsDraw' in corr_matrix.columns:
                corr_df = corr_matrix[['target_IsDraw']].sort_values(by='target_IsDraw', ascending=False)
                # Remove a própria correlação do alvo consigo mesmo (que é 1.0)
                corr_df = corr_df.drop('target_IsDraw', errors='ignore')
                logger.info(f"ANÁLISE FEATURES: Correlação com o alvo calculada OK. Shape: {corr_df.shape}")
            else:
                 logger.error("ANÁLISE FEATURES (Corr): Coluna 'target_IsDraw' não encontrada na matriz de correlação final.")
                 corr_df = None
        elif df_numeric_temp.empty:
             logger.error("ANÁLISE FEATURES (Corr): DataFrame numérico vazio após tratamento de NaN/Inf.")
             corr_df = None
        else: # target_IsDraw não está nas colunas para correlação
             logger.warning("ANÁLISE FEATURES (Corr): Coluna alvo não disponível para correlação.")
             corr_df = None


    except Exception as e:
        logger.error(f"ANÁLISE FEATURES: Erro GRAVE ao calcular correlação: {e}", exc_info=True)
        corr_df = None # Set to None on failure

    logger.info("--- ANÁLISE FEATURES (model_trainer): Concluída ---")
    return imp_df, corr_df # Return results (can be None)


def optimize_single_model(*args, **kwargs) -> Optional[Tuple[str, Dict, Dict]]:
    """Placeholder - Esta função não é usada no fluxo principal atual."""
    logger.warning("optimize_single_model não está implementada ou não é usada ativamente.")
    return None