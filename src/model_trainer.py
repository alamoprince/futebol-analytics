import pandas as pd
import time
# MODIFICADO: Adicionado TimeSeriesSplit
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.isotonic import IsotonicRegression
from logger_config import setup_logger
from collections import defaultdict
import warnings # Import warnings
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

try:
    from catboost import CatBoostClassifier # <<< TENTA IMPORTAR A CLASSE
    CATBOOST_AVAILABLE = True
    print("INFO: Biblioteca 'catboost' carregada.")
except ImportError:
    print("AVISO: CatBoost não instalado (pip install catboost).")
    CatBoostClassifier = None # <<< DEFINE COMO NONE SE FALHAR
    CATBOOST_AVAILABLE = False

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
    except AttributeError: return None
    if len(common_index) == 0: return 0.0 # Handle empty intersection
    if len(common_index) != len(y_test):
        logger.warning(f"ROI: Index mismatch, using {len(common_index)} common indices.")
    y_test_common = y_test.loc[common_index]
    try:
        y_pred_series = pd.Series(y_pred, index=y_test.index)
        y_pred_common = y_pred_series.loc[common_index]
    except Exception: return None

    predicted_draws_indices = common_index[y_pred_common == 1]
    num_bets = len(predicted_draws_indices)
    if num_bets == 0: return 0.0

    actuals = y_test_common.loc[predicted_draws_indices]
    # Ensure odds data is aligned and numeric before accessing
    odds_df_aligned = X_test_odds_aligned.loc[common_index]
    odds = pd.to_numeric(odds_df_aligned.loc[predicted_draws_indices, odd_draw_col_name], errors='coerce')

    profit = 0.0
    valid_bets = 0
    for idx in predicted_draws_indices:
        try:
            odd_d = odds.loc[idx]
            if pd.notna(odd_d) and odd_d > 1:
                # Check if actual result is available for this index
                if idx in actuals.index:
                    profit += (odd_d - 1) if actuals.loc[idx] == 1 else -1
                    valid_bets += 1
                else:
                    logger.warning(f"ROI calc: Actual result missing for index {idx}")
        except KeyError:
             logger.warning(f"ROI calc: Index {idx} not found in odds Series.")
        except Exception as e_roi_loop:
            logger.error(f"Error in ROI loop for index {idx}: {e_roi_loop}")

    if valid_bets == 0: return 0.0
    return (profit / valid_bets) * 100.0


def calculate_roi_with_threshold(y_true: pd.Series, y_proba: np.ndarray, threshold: float, odds_data: Optional[pd.DataFrame], odd_col_name: str) -> Tuple[Optional[float], int, Optional[float]]:
    profit, roi_value, num_bets_suggested, profit_calc, valid_bets_count = None, None, 0, 0.0, 0
    if odds_data is None or odd_col_name not in odds_data.columns:
        logger.warning("ROI Thr: Odds data missing or column name invalid.")
        return roi_value, 0, profit # Return 0 bets

    try:
        common_index = y_true.index.intersection(odds_data.index)
        if len(common_index) == 0: return 0.0, 0, 0.0
        if len(common_index) != len(y_true):
            logger.warning(f"ROI Thr: Index mismatch, using {len(common_index)} common indices.")

        y_true_common = y_true.loc[common_index]
        odds_common = pd.to_numeric(odds_data.loc[common_index, odd_col_name], errors='coerce')

        try:
            # Align y_proba using common_index BEFORE filtering by threshold
            y_proba_series = pd.Series(y_proba, index=y_true.index)
            y_proba_common = y_proba_series.loc[common_index]
        except Exception as e:
            logger.error(f"ROI Thr: Error aligning y_proba: {e}")
            return None, 0, None

        # Filter indices AFTER aligning probabilities
        bet_indices = common_index[y_proba_common > threshold]
        num_bets_suggested = len(bet_indices) # Total bets suggested by threshold

        if num_bets_suggested == 0: return 0.0, 0, 0.0 # 0 valid bets placed

        actuals = y_true_common.loc[bet_indices]
        odds_selected = odds_common.loc[bet_indices]

        for idx in bet_indices:
            try:
                odd_d = odds_selected.loc[idx]
                if pd.notna(odd_d) and odd_d > 1:
                    if idx in actuals.index: # Check if actual result exists
                        profit_calc += (odd_d - 1) if actuals.loc[idx] == 1 else -1
                        valid_bets_count += 1
                    else: logger.warning(f"ROI Thr calc: Actual result missing for index {idx}")
            except KeyError: logger.warning(f"ROI Thr calc: Index {idx} not found in odds_selected.")
            except Exception as e_roi_loop: logger.error(f"Error in ROI Thr loop for index {idx}: {e_roi_loop}")

        profit = profit_calc
        roi_value = (profit / valid_bets_count) * 100 if valid_bets_count > 0 else 0.0
        return roi_value, valid_bets_count, profit # Return count of *valid* bets placed

    except Exception as e:
        logger.error(f"ROI Thr: General error - {e}", exc_info=True)
        return None, 0, None

def calculate_metrics_with_ev(y_true: pd.Series, y_proba_calibrated: np.ndarray, 
                              ev_threshold: float, odds_data: Optional[pd.DataFrame], odd_col_name: str) -> Tuple[Optional[float], int, Optional[float]]:
    profit, roi_value, num_bets_suggested, profit_calc, valid_bets_count = None, None, 0, 0.0, 0

    if odds_data is None or odd_col_name not in odds_data.columns:
        logger.warning(f"EV Metr: Odds data missing or column name invalid.")
        return roi_value, 0, profit

    try:
        common_index = y_true.index.intersection(odds_data.index)
        if len(common_index) == 0: return 0.0, 0, 0.0
        if len(common_index) != len(y_true):
            logger.warning(f"EV Metr: Index mismatch, using {len(common_index)} common indices.")

        y_true_common = y_true.loc[common_index]
        odds_common = pd.to_numeric(odds_data.loc[common_index, odd_col_name], errors='coerce')

        try:
            y_proba_common = pd.Series(y_proba_calibrated, index=y_true.index).loc[common_index]
        except Exception as e:
            logger.error(f"EV Metr: Error aligning y_proba: {e}")
            return None, 0, None

        # Calculate EV only for valid odds and probabilities
        valid_mask = odds_common.notna() & y_proba_common.notna() & (odds_common > 1)
        ev = pd.Series(np.nan, index=common_index) # Initialize with NaN
        prob_ok = y_proba_common[valid_mask]
        odds_ok = odds_common[valid_mask]
        if not prob_ok.empty: # Proceed only if there are valid entries
            ev_calc = (prob_ok * (odds_ok - 1)) - ((1 - prob_ok) * 1)
            ev.loc[valid_mask] = ev_calc # Assign calculated EV only where valid

        # Find indices where calculated EV is above threshold
        bet_indices = common_index[ev > ev_threshold] # Relies on NaN comparison being False
        num_bets_suggested = len(bet_indices) # Total bets suggested by EV threshold

        if num_bets_suggested == 0: return 0.0, 0, 0.0 # 0 valid bets placed

        actuals = y_true_common.loc[bet_indices]
        odds_selected = odds_common.loc[bet_indices] # Use already filtered odds

        for idx in bet_indices:
            try:
                odd_d = odds_selected.loc[idx]
                # Check odd validity again (should be valid due to mask, but safe)
                if pd.notna(odd_d) and odd_d > 1:
                    if idx in actuals.index: # Check if actual result exists
                        profit_calc += (odd_d - 1) if actuals.loc[idx] == 1 else -1
                        valid_bets_count += 1
                    else: logger.warning(f"EV Metr calc: Actual result missing for index {idx}")
            except KeyError: logger.warning(f"EV Metr calc: Index {idx} not found in odds_selected.")
            except Exception as e_ev_loop: logger.error(f"Error in EV Metr loop for index {idx}: {e_ev_loop}")

        profit = profit_calc
        roi_value = (profit / valid_bets_count) * 100 if valid_bets_count > 0 else 0.0

        # logger.debug(f"    -> Métricas EV (Th={ev_threshold:.3f}): ROI={roi_value:.2f}%, Bets Sug={num_bets_suggested}, Bets Vál={valid_bets_count}, Profit={profit:.2f}")
        return roi_value, valid_bets_count, profit # Return count of *valid* bets placed

    except Exception as e:
        logger.error(f"EV Metr: General error - {e}", exc_info=True)
        return None, 0, None


def scale_features(X_train: pd.DataFrame, X_val: Optional[pd.DataFrame], X_test: Optional[pd.DataFrame], scaler_type='standard') -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame], Any]:
    """Scales features using StandardScaler or MinMaxScaler."""
    # Check if input DataFrames are valid
    if X_train is None or X_train.empty:
        logger.error("Scaling Error: X_train is None or empty.")
        raise ValueError("X_train cannot be None or empty for scaling.")

    X_train_c = X_train.copy()
    X_val_c = X_val.copy() if X_val is not None else None
    X_test_c = X_test.copy() if X_test is not None else None
    scaler = None

    try:
        if scaler_type == 'minmax': scaler = MinMaxScaler()
        elif scaler_type == 'standard': scaler = StandardScaler()
        else: logger.warning(f"Scaler '{scaler_type}' desconhecido, usando StandardScaler."); scaler = StandardScaler()

        logger.info(f"  Aplicando {scaler.__class__.__name__}...")
        cols = X_train_c.columns

        # Impute NaNs and Infs before fitting/transforming
        # Calculate median based only on the training set
        train_median = X_train_c.replace([np.inf, -np.inf], np.nan).median()

        # Impute Training Data
        X_train_c = X_train_c.replace([np.inf, -np.inf], np.nan)
        if X_train_c.isnull().values.any():
            logger.warning(f"  NaNs/Infs encontrados em X_train ({X_train_c.isnull().sum().sum()}). Imputando com mediana.")
            X_train_c.fillna(train_median, inplace=True)
            # Check if any columns are still all NaN (median might be NaN if column was all NaN)
            if X_train_c.isnull().values.any():
                 nan_cols_after_impute = X_train_c.columns[X_train_c.isnull().all()].tolist()
                 logger.error(f"  ERRO: NaNs persistentes em X_train após imputação (Colunas: {nan_cols_after_impute}). Scaling falhará.")
                 raise ValueError("NaNs persistentes em X_train após imputação com mediana.")

        # Impute Validation Data (if exists)
        if X_val_c is not None:
            X_val_c = X_val_c.replace([np.inf, -np.inf], np.nan)
            if X_val_c.isnull().values.any():
                 # logger.debug(f"  NaNs/Infs encontrados em X_val ({X_val_c.isnull().sum().sum()}). Imputando com mediana do TREINO.")
                 X_val_c.fillna(train_median, inplace=True)

        # Impute Test Data (if exists)
        if X_test_c is not None:
            X_test_c = X_test_c.replace([np.inf, -np.inf], np.nan)
            if X_test_c.isnull().values.any():
                 # logger.debug(f"  NaNs/Infs encontrados em X_test ({X_test_c.isnull().sum().sum()}). Imputando com mediana do TREINO.")
                 X_test_c.fillna(train_median, inplace=True)

        # Fit scaler ONLY on training data
        scaler.fit(X_train_c)

        # Transform all sets
        X_train_scaled = scaler.transform(X_train_c)
        X_val_scaled = scaler.transform(X_val_c) if X_val_c is not None else None
        X_test_scaled = scaler.transform(X_test_c) if X_test_c is not None else None

        # Convert back to DataFrame
        X_train_sc = pd.DataFrame(X_train_scaled, index=X_train.index, columns=cols)
        X_val_sc = pd.DataFrame(X_val_scaled, index=X_val.index, columns=cols) if X_val_scaled is not None else None
        X_test_sc = pd.DataFrame(X_test_scaled, index=X_test.index, columns=cols) if X_test_scaled is not None else None

        logger.info("  Scaling concluído.")
        return X_train_sc, X_val_sc, X_test_sc, scaler

    except Exception as e:
        logger.error(f"Erro GERAL durante scaling: {e}", exc_info=True)
        # Return original data and no scaler on error
        return X_train, X_val, X_test, None

# --- Função Principal de Treinamento (COM PIPELINE IMBLEARN) ---
def train_evaluate_and_save_best_models(
    X: pd.DataFrame, y: pd.Series,
    X_test_with_odds: Optional[pd.DataFrame] = None,
    progress_callback_stages: Optional[Callable[[int, str], None]] = None,
    num_total_models: int = 1, # Should be passed correctly by main.py
    scaler_type: str = 'standard',
    sampler_type: str = 'smote', # 'smote', 'random', None
    odd_draw_col_name: str = ODDS_COLS.get('draw', 'Odd_D_FT'),
    calibration_method: str = 'isotonic', # 'isotonic' or 'sigmoid'
    optimize_ev_threshold: bool = True,
    optimize_f1_threshold: bool = True,
    optimize_precision_threshold: bool = True,
    min_recall_target: float = MIN_RECALL_FOR_PRECISION_OPT,
    bayes_opt_n_iter: int = BAYESIAN_OPT_N_ITER,
    cv_splits: int = CROSS_VALIDATION_SPLITS,
    scoring_metric: str = 'f1', # Metric for hyperparameter tuning
    n_ensemble_models: int = 3
    ) -> bool:
    """
    Pipeline otimizado: Treina com CV Temporal, calibra/otimiza no teste (simplificado), avalia, salva.
    """
    # --- Validações Iniciais ---
    if not IMBLEARN_AVAILABLE and sampler_type is not None: logger.error("'imbalanced-learn' não instalado, sampler desativado."); return False
    if X is None or y is None or X.empty or y.empty: logger.error("Dados X ou y inválidos/vazios."); return False
    if not X.index.equals(y.index): logger.error("Índices de X e y não coincidem ANTES da divisão."); return False # Check initial alignment
    if not MODEL_CONFIG: logger.error("MODEL_CONFIG vazio."); return False
    available_models = {name: config for name, config in MODEL_CONFIG.items() if not (name=='LGBMClassifier' and not LGBM_AVAILABLE) and \
           not (name=='CatBoostClassifier' and not CATBOOST_AVAILABLE)}
    if not available_models: logger.error("Nenhum modelo válido configurado."); return False
    if num_total_models <= 0: num_total_models = len(available_models); logger.warning(f"Ajustando num_total_models para {num_total_models}.")
    if cv_splits <= 1 : logger.error(f"cv_splits ({cv_splits}) deve ser maior que 1."); return False

    feature_names = list(X.columns); all_results = []
    sampler_log = f"Sampler: {sampler_type}" if sampler_type else "Sampler: None"
    opt_log = f"Opt(F1:{optimize_f1_threshold}, EV:{optimize_ev_threshold}, Prec:{optimize_precision_threshold})"
    logger.info(f"--- Treinando {len(available_models)} Modelos ({sampler_log}, CV Temporal: {scoring_metric}, {opt_log}) ---")
    start_time_total = time.time()

    # --- Divisão Temporal Manual ---
    logger.info("Dividindo dados temporalmente (Treino+CV / Teste Final)...")
    n_total = len(X)
    n_test = int(n_total * TEST_SIZE)
    # Ensure test set is large enough for at least one sample after CV splits train on min data
    min_train_per_split = n_total // (cv_splits + 1) # Approx min samples per training fold in TimeSeriesSplit
    if n_test < 1 or (n_total - n_test) < min_train_per_split :
         logger.error(f"Divisão Temporal Inválida: Teste ({n_test}) ou Treino+CV ({n_total-n_test}) muito pequeno para {cv_splits} splits.")
         return False
    n_train_cv = n_total - n_test

    # Assume X, y are already sorted chronologically (crucial!)
    try:
        X_train_cv = X.iloc[:n_train_cv]
        y_train_cv = y.iloc[:n_train_cv]
        X_test     = X.iloc[n_train_cv:]
        y_test     = y.iloc[n_train_cv:]
        logger.info(f"Split Temporal: T+CV={len(X_train_cv)} ({len(X_train_cv)/n_total:.1%}), Teste={len(X_test)} ({len(X_test)/n_total:.1%})")
        if not X_train_cv.index.is_monotonic_increasing or not X_test.index.is_monotonic_increasing:
             logger.warning("Índices não são monotonicamente crescentes após split temporal. Verifique ordenação inicial.")
    except Exception as e_split:
         logger.error(f"Erro durante a divisão temporal manual: {e_split}", exc_info=True)
         return False

    # Alinha Odds com os novos X_test
    X_test_odds = None
    if X_test_with_odds is not None and not X_test_with_odds.empty and odd_draw_col_name in X_test_with_odds.columns:
        try:
            common_test = X_test.index.intersection(X_test_with_odds.index)
            if len(common_test) > 0:
                 X_test_odds = X_test_with_odds.loc[common_test, [odd_draw_col_name]].copy()
                 logger.info(f"Odds alinhadas (Temporal): Teste={X_test_odds is not None} ({len(common_test)} jogos)")
                 # Check if alignment caused loss of test samples needed for metrics
                 if len(X_test_odds) != len(X_test):
                     logger.warning(f"Perda de {len(X_test) - len(X_test_odds)} amostras de teste durante alinhamento de odds.")
                     # Optional: Re-align X_test/y_test to match odds if ROI is critical
                     # X_test = X_test.loc[common_test]
                     # y_test = y_test.loc[common_test]
                     # logger.info(f"Re-alinhado X_test/y_test para {len(X_test)} amostras com odds.")
            else: logger.warning("Nenhum índice em comum entre X_test e X_test_with_odds.")
        except Exception as e_align_odds:
             logger.error(f"Erro ao alinhar odds com X_test: {e_align_odds}", exc_info=True)
             X_test_odds = None # Ensure it's None if alignment fails
    else: logger.warning("DF de odds ausente/vazio/sem coluna. ROI/EV não serão calculados no teste.")
    # --- Fim Divisão Temporal ---

    # Define TimeSeriesSplit para CV
    n_cv_splits_ts = cv_splits
    tscv = TimeSeriesSplit(n_splits=n_cv_splits_ts)

    # --- Loop principal pelos modelos ---
    for i, (model_name, config) in enumerate(available_models.items()):
        status_text_loop = f"Modelo {i+1}/{len(available_models)}: {model_name}"
        logger.info(f"\n--- {status_text_loop} ---")
        if progress_callback_stages: progress_callback_stages(i, f"Iniciando {model_name}...")
        start_time_model = time.time()
        model_trained = None; best_params = None; current_scaler = None; calibrator = None;
        best_pipeline_object = None; final_pipeline_trained = None # Pipeline treinado em T+CV
        current_optimal_f1_threshold = DEFAULT_F1_THRESHOLD
        current_optimal_ev_threshold = DEFAULT_EV_THRESHOLD
        current_optimal_precision_threshold = 0.5

        try:
            # --- Setup do Modelo ---
            try: model_class = eval(model_name)
            except NameError: logger.error(f"Classe do modelo '{model_name}' não encontrada."); continue
            except Exception as e_eval: logger.error(f"Erro ao obter classe para '{model_name}': {e_eval}."); continue
            if model_class is None: logger.error(f"Classe {model_name} é None (importação falhou?)."); continue

            model_kwargs = config.get('model_kwargs', {}); needs_scaling = config.get('needs_scaling', False)
            use_bayes = SKOPT_AVAILABLE and 'search_spaces' in config and config['search_spaces'] is not None
            param_space_or_grid = config.get('search_spaces') if use_bayes else config.get('param_grid')

            # --- Scaling (Usa X_train_cv e X_test) ---
            X_train_cv_m = X_train_cv.copy()
            X_test_m = X_test.copy() # Não há X_val_m
            if needs_scaling:
                if progress_callback_stages: progress_callback_stages(i, f"Scaling...")
                try:
                    X_train_cv_m, _, X_test_m, current_scaler = scale_features(X_train_cv_m, None, X_test_m, scaler_type)
                    if current_scaler is None: raise ValueError("Scaler fitting/transform failed.") # Check if scaler is returned
                    logger.info(" -> Scaling OK.")
                except Exception as e_scale: logger.error(f" ERRO scaling p/ {model_name}: {e_scale}", exc_info=True); continue
            # --- Fim Scaling ---

            # --- Criação do Pipeline ---
            pipeline_steps = []; sampler_instance = None; sampler_log_name = "None"
            if sampler_type == 'smote' and SMOTE: sampler_instance = SMOTE(random_state=RANDOM_STATE); pipeline_steps.append(('sampler', sampler_instance)); sampler_log_name="SMOTE"
            elif sampler_type == 'random' and RandomOverSampler: sampler_instance = RandomOverSampler(random_state=RANDOM_STATE); pipeline_steps.append(('sampler', sampler_instance)); sampler_log_name="RandomOverSampler"
            pipeline_steps.append(('classifier', model_class(**model_kwargs)))
            pipeline = ImbPipeline(pipeline_steps)
            logger.info(f"  Pipeline criado (Sampler: {sampler_log_name}, Classifier: {model_name})")
            # --- Fim Criação Pipeline ---

            # --- Treino com Busca de Hiperparâmetros (CV Temporal) ---
            if param_space_or_grid:
                 search_method_name = 'BayesSearchCV' if use_bayes else 'GridSearchCV'
                 search_status_msg = f"Ajustando CV ({search_method_name[:5]}" + (f"+{sampler_log_name})" if sampler_instance else ")")
                 if progress_callback_stages: progress_callback_stages(i, search_status_msg);
                 logger.info(f"  Iniciando {search_method_name} (Score: {scoring_metric}, CV: Temporal {n_cv_splits_ts} splits)...")
                 try:
                     if param_space_or_grid is None: raise ValueError(f"Espaço de busca/grid é None para {search_method_name} do modelo {model_name}")
                     model_fit_params = config.get('fit_params', {})
                     logger.debug(f"Fit params para {model_name}: {model_fit_params}")
                     if use_bayes: search_cv = BayesSearchCV(estimator=pipeline, search_spaces=param_space_or_grid, n_iter=bayes_opt_n_iter, cv=tscv, n_jobs=N_JOBS_GRIDSEARCH, scoring=scoring_metric, random_state=RANDOM_STATE, verbose=0)
                     else: search_cv = GridSearchCV(estimator=pipeline, param_grid=param_space_or_grid, cv=tscv, n_jobs=N_JOBS_GRIDSEARCH, scoring=scoring_metric, verbose=0, error_score='raise')
                     search_cv.fit(X_train_cv_m, y_train_cv, **model_fit_params) # <<< Fit nos dados de treino+CV
                     best_pipeline_object = search_cv.best_estimator_; # Este é o pipeline com os melhores params encontrados na CV
                     best_params_raw = search_cv.best_params_; best_params = {k.replace('classifier__', ''): v for k, v in best_params_raw.items() if k.startswith('classifier__')}
                     best_cv_score = search_cv.best_score_; logger.info(f"    -> Busca CV ({search_method_name}) OK. Melhor CV {scoring_metric}: {best_cv_score:.4f}. Params: {best_params}");
                 except Exception as e_search: logger.error(f"    Erro {search_method_name}.fit: {e_search}", exc_info=True); model_trained=None; best_pipeline_object=None; logger.warning("    -> Tentando fallback treino padrão...");
            # --- Fim Busca CV ---

            # --- Fallback ou Treino Final ---
            if best_pipeline_object is None: # Se CV falhou ou não foi usado
                fallback_status = f"Ajustando (Padrão" + (f"+{sampler_log_name})" if sampler_instance else ")")
                if progress_callback_stages: progress_callback_stages(i, fallback_status)
                logger.info(f"  Treinando Pipeline com params padrão em TODO X_train_cv...");
                try:
                    pipeline.fit(X_train_cv_m, y_train_cv);
                    final_pipeline_trained = pipeline # O pipeline treinado é o final
                    model_fit_params_final = config.get('fit_params', {})
                    final_pipeline_trained.fit(X_train_cv_m, y_train_cv, **model_fit_params_final)
                    model_trained = final_pipeline_trained.named_steps['classifier']
                    best_params = {k:v for k,v in model_trained.get_params().items() if k in model_kwargs} # Pega params default
                    logger.info("    -> Treino padrão OK.")
                except Exception as e_fall: logger.error(f"    Erro treino fallback: {e_fall}", exc_info=True); continue;
            else: # Se CV funcionou, re-treina o melhor pipeline em todos os dados de treino+CV
                logger.info(f"  Re-ajustando MELHOR pipeline encontrado pela CV em TODO X_train_cv...")
                try:
                    final_pipeline_trained = clone(best_pipeline_object) # Clona o melhor da CV
                    final_pipeline_trained.fit(X_train_cv_m, y_train_cv) # Re-treina
                    model_trained = final_pipeline_trained.named_steps['classifier'] # Modelo final
                    # best_params já foi definido pela CV
                    logger.info("  -> Re-ajuste final OK.")
                except Exception as e_refit: logger.error(f"  ERRO ao re-ajustar pipeline final: {e_refit}", exc_info=True); continue;

            if final_pipeline_trained is None or model_trained is None: logger.error(" Falha crítica no treino final."); continue;
            # --- Fim Treino Final ---

            # --- Previsão, Calibração e Otimização (Simplificado: no Conjunto de Teste) ---
            # (Aviso: Avaliação será otimista, pois calibração/limiares usam dados de teste)
            y_proba_test_raw_full, y_proba_test_raw_draw = None, None
            if hasattr(final_pipeline_trained, "predict_proba"):
                 try:
                     y_proba_test_raw_full = final_pipeline_trained.predict_proba(X_test_m) # Usa pipeline final treinado
                     if y_proba_test_raw_full.shape[1] >= 2: y_proba_test_raw_draw = y_proba_test_raw_full[:, 1]
                 except Exception as e_pred_test: logger.error(f"  Erro predict_proba teste final: {e_pred_test}")

            # Calibrar no Teste
            calibrator = None; y_proba_test_calib = None
            if y_proba_test_raw_draw is not None:
                if progress_callback_stages: progress_callback_stages(i, "Calibrando (Teste)...");
                logger.warning(f"  AVISO: Calibrando probs ({calibration_method}) no CONJUNTO DE TESTE (avaliação otimista)...");
                try:
                    if calibration_method == 'isotonic': calibrator = IsotonicRegression(out_of_bounds='clip')
                    else: logger.warning(f"Método calibração '{calibration_method}' não suportado. Usando Isotonic."); calibrator = IsotonicRegression(out_of_bounds='clip')
                    # Fit calibrator on test set predictions and labels
                    calibrator.fit(y_proba_test_raw_draw, y_test)
                    y_proba_test_calib = calibrator.predict(y_proba_test_raw_draw)
                    logger.info("  -> Calibrador treinado (no Teste).")
                except ValueError as ve:
                    logger.error(f"  Erro (ValueError) durante calibração (no Teste): {ve}. Usando probs brutas.")
                    calibrator = None; y_proba_test_calib = y_proba_test_raw_draw # Fallback
                except Exception as e_calib:
                    logger.error(f"  Erro INESPERADO durante calibração (no Teste): {e_calib}", exc_info=True);
                    calibrator = None; y_proba_test_calib = y_proba_test_raw_draw # Fallback
            else: logger.warning("  Probs brutas de teste ausentes. Calibração pulada.")

            # Probs a serem usadas para otimização e avaliação final
            proba_opt_eval_test = y_proba_test_calib if calibrator and y_proba_test_calib is not None else y_proba_test_raw_draw
            opt_eval_src_test = 'Calib(Teste)' if calibrator and y_proba_test_calib is not None else ('Raw(Teste)' if proba_opt_eval_test is not None else 'N/A')

            # Otimizar Limiares no Teste
            if proba_opt_eval_test is not None:
                logger.warning(f"  AVISO: Otimizando limiares (F1, EV, Prec) no CONJUNTO DE TESTE (avaliação otimista)...")
                # Otimização F1 (no Teste)
                if optimize_f1_threshold:
                    if progress_callback_stages: progress_callback_stages(i, f"Otimizando F1 Thr ({opt_eval_src_test})...")
                    try:
                        p,r,t=precision_recall_curve(y_test,proba_opt_eval_test);
                        # Ensure p and r have the same length as t for f1 calculation
                        min_len = min(len(p), len(r))
                        p, r = p[:min_len], r[:min_len]
                        f1 = np.divide(2*p*r, p+r+FEATURE_EPSILON, where=(p+r)>0, out=np.zeros_like(p+r))
                        # Find threshold corresponding to max f1 (need to handle threshold array length)
                        if len(t) == len(f1): # Standard case
                           idx=np.argmax(f1); best_test_f1=f1[idx]; current_optimal_f1_threshold=t[idx];
                        elif len(t) == len(f1) - 1: # Sometimes thresholds array is one shorter
                           idx=np.argmax(f1[:-1]); best_test_f1=f1[idx]; current_optimal_f1_threshold=t[idx];
                        else: # Unexpected length difference
                            logger.warning(f"Otim F1: Tamanho t ({len(t)}) vs f1 ({len(f1)}) incompatível. Usando F1 máximo.")
                            idx=np.argmax(f1); best_test_f1=f1[idx];
                            # Cannot reliably get threshold, use default or midpoint? Let's use default.
                            current_optimal_f1_threshold=DEFAULT_F1_THRESHOLD
                        logger.info(f"    Limiar F1 ótimo(Teste): {current_optimal_f1_threshold:.4f} (F1={best_test_f1:.4f})")
                    except Exception as e_f1: logger.error(f"  Erro otim F1 (Teste): {e_f1}"); current_optimal_f1_threshold=DEFAULT_F1_THRESHOLD

                # Otimização EV (no Teste)
                if optimize_ev_threshold and X_test_odds is not None:
                    if progress_callback_stages: progress_callback_stages(i, f"Otimizando EV Thr ({opt_eval_src_test})...")
                    try:
                        best_test_roi_ev=-np.inf; ev_ths=np.linspace(0.0,0.20,21);
                        for ev_th in ev_ths:
                           test_roi,_,_=calculate_metrics_with_ev(y_test, proba_opt_eval_test, ev_th, X_test_odds, odd_draw_col_name);
                           if test_roi is not None and test_roi>best_test_roi_ev: best_test_roi_ev=test_roi; current_optimal_ev_threshold=ev_th;
                        if best_test_roi_ev > -np.inf: logger.info(f"    Limiar EV ótimo(Teste): {current_optimal_ev_threshold:.3f} (ROI={best_test_roi_ev:.2f}%)")
                        else: logger.warning("    ROI Teste inválido/negativo em otimização EV."); current_optimal_ev_threshold = DEFAULT_EV_THRESHOLD;
                    except Exception as e_ev: logger.error(f"  Erro otim EV (Teste): {e_ev}"); current_optimal_ev_threshold = DEFAULT_EV_THRESHOLD;
                elif optimize_ev_threshold: logger.warning(f"  Otim EV (Teste) pulada (sem odds Teste).")

                # Otimização Precision (no Teste)
                if optimize_precision_threshold:
                    if progress_callback_stages: progress_callback_stages(i, f"Otimizando Prec Thr ({opt_eval_src_test})...")
                    logger.info(f"  Otimizando Precision (Teste Probs {opt_eval_src_test}, Recall Mín: {min_recall_target:.1%})...");
                    try:
                        best_test_precision=-1.0; found_prec_thr_test=False;
                        # Use thresholds from precision_recall_curve if available, else linspace
                        if 'p' in locals() and 't' in locals() and len(p) == len(t) + 1:
                           thresholds_test = t
                        else:
                           thresholds_test=np.linspace(np.min(proba_opt_eval_test)+1e-4,np.min([np.max(proba_opt_eval_test)-1e-4,0.99]),100)

                        current_optimal_precision_threshold = 0.5 # Reset default before searching
                        for t in thresholds_test:
                            y_pred_v=(proba_opt_eval_test>=t).astype(int);
                            prec=precision_score(y_test,y_pred_v,pos_label=1,zero_division=0);
                            rec=recall_score(y_test,y_pred_v,pos_label=1,zero_division=0);
                            if rec >= min_recall_target and prec >= best_test_precision:
                                if prec > best_test_precision or t > current_optimal_precision_threshold:
                                    best_test_precision=prec; current_optimal_precision_threshold=t; found_prec_thr_test=True;
                        if found_prec_thr_test:
                            final_rec_test=recall_score(y_test,(proba_opt_eval_test>=current_optimal_precision_threshold).astype(int),pos_label=1,zero_division=0);
                            logger.info(f"    Limiar Prec ótimo(Teste): {current_optimal_precision_threshold:.4f} (Prec={best_test_precision:.4f}, Rec={final_rec_test:.4f})")
                        else:
                            logger.warning(f"    Não encontrou limiar p/ Recall Mín {min_recall_target:.1%} no Teste. Usando default 0.5");
                            current_optimal_precision_threshold=0.5
                    except Exception as e_prec:
                        logger.error(f"  Erro otim Precision (Teste): {e_prec}");
                        current_optimal_precision_threshold=0.5
                elif optimize_precision_threshold:
                     logger.warning(f"  Otim Prec (Teste) pulada (sem probs Teste).")
                     current_optimal_precision_threshold=0.5
            else:
                logger.warning("  Otimização de limiares pulada (sem probs de teste).")
                # Use default thresholds if probs aren't available
                current_optimal_f1_threshold = DEFAULT_F1_THRESHOLD
                current_optimal_ev_threshold = DEFAULT_EV_THRESHOLD
                current_optimal_precision_threshold = 0.5
            # --- Fim Otimização ---

            # --- Avaliação Final no Teste ---
            if progress_callback_stages: progress_callback_stages(i, f"Avaliando (Final)..."); logger.info(f"  Avaliando Modelo FINAL no Teste...")
            metrics = {
                'optimal_f1_threshold': current_optimal_f1_threshold,
                'optimal_ev_threshold': current_optimal_ev_threshold,
                'optimal_precision_threshold': current_optimal_precision_threshold,
                'model_name': model_name # Add model name here for clarity in results dict
            }
            eval_src_test = 'Calib(Teste)' if calibrator and y_proba_test_calib is not None else ('Raw(Teste)' if y_proba_test_raw_draw is not None else 'N/A')
            logger.info(f"  Calculando métricas teste final (probs: {eval_src_test})")

            # Métricas @ Thr 0.5 (usando predict do pipeline final)
            try:
                y_pred05 = final_pipeline_trained.predict(X_test_m);
                metrics.update({
                    'accuracy_thr05':accuracy_score(y_test,y_pred05),
                    'precision_draw_thr05':precision_score(y_test,y_pred05,pos_label=1,zero_division=0),
                    'recall_draw_thr05':recall_score(y_test,y_pred05,pos_label=1,zero_division=0),
                    'f1_score_draw_thr05':f1_score(y_test,y_pred05,pos_label=1,zero_division=0)
                });
                logger.info(f"    -> Métricas @ Thr 0.5: F1={metrics['f1_score_draw_thr05']:.4f}, P={metrics['precision_draw_thr05']:.4f}, R={metrics['recall_draw_thr05']:.4f}")
            except Exception as e: logger.error(f"  Erro métricas @ 0.5 (Final): {e}")

            if proba_opt_eval_test is not None:
                 # Métricas @ Thr F1 (Otimizado no Teste)
                 try:
                     y_predF1=(proba_opt_eval_test>=current_optimal_f1_threshold).astype(int);
                     metrics.update({
                         'accuracy_thrF1':accuracy_score(y_test,y_predF1),
                         'precision_draw_thrF1':precision_score(y_test,y_predF1,pos_label=1,zero_division=0),
                         'recall_draw_thrF1':recall_score(y_test,y_predF1,pos_label=1,zero_division=0),
                         'f1_score_draw':f1_score(y_test,y_predF1,pos_label=1,zero_division=0) # Chave principal
                         });
                     logger.info(f"    -> Métricas @ Thr F1 ({current_optimal_f1_threshold:.4f}): F1={metrics['f1_score_draw']:.4f}, P={metrics['precision_draw_thrF1']:.4f}, R={metrics['recall_draw_thrF1']:.4f}")
                 except Exception as e: logger.error(f"  Erro métricas @ Thr F1 (Final): {e}"); metrics['f1_score_draw']=metrics.get('f1_score_draw_thr05',-1.0);

                 # Métricas @ Thr Prec (Otimizado no Teste)
                 try:
                     y_predPrec=(proba_opt_eval_test>=current_optimal_precision_threshold).astype(int);
                     metrics.update({
                         'accuracy_thrPrec':accuracy_score(y_test,y_predPrec),
                         'precision_draw_thrPrec':precision_score(y_test,y_predPrec,pos_label=1,zero_division=0),
                         'recall_draw_thrPrec':recall_score(y_test,y_predPrec,pos_label=1,zero_division=0),
                         'f1_score_draw_thrPrec':f1_score(y_test,y_predPrec,pos_label=1,zero_division=0)
                     });
                     logger.info(f"    -> Métricas @ Thr Prec ({current_optimal_precision_threshold:.4f}): F1={metrics['f1_score_draw_thrPrec']:.4f}, P={metrics['precision_draw_thrPrec']:.4f}, R={metrics['recall_draw_thrPrec']:.4f}")
                 except Exception as e: logger.error(f"  Erro métricas @ Thr Prec (Final): {e}")

                 # Métricas Probabilísticas Finais
                 logloss,auc,brier=None,None,None;
                 try: logloss=log_loss(y_test,y_proba_test_raw_full) if y_proba_test_raw_full is not None else None
                 except Exception as e: logger.warning(f"Erro LogLoss: {e}")
                 try: auc=roc_auc_score(y_test,proba_opt_eval_test) if len(np.unique(y_test))>1 else 0.5 # Use 0.5 if only one class in test
                 except Exception as e: logger.warning(f"Erro AUC: {e}")
                 try: brier=brier_score_loss(y_test,proba_opt_eval_test)
                 except Exception as e: logger.warning(f"Erro Brier: {e}")
                 metrics.update({'log_loss':logloss,'roc_auc':auc,'brier_score':brier});
                 logger.info(f"    -> AUC({eval_src_test})={auc:.4f}, Brier({eval_src_test})={brier:.4f}" if auc is not None and brier is not None else f"    -> AUC/Brier({eval_src_test})=N/A")

                 # Métricas EV/ROI Finais (com limiar EV otimizado no teste)
                 roi_ev,bets_ev,profit_ev = calculate_metrics_with_ev(y_test,proba_opt_eval_test,current_optimal_ev_threshold,X_test_odds,odd_draw_col_name) if X_test_odds is not None else (None,0,None);
                 metrics.update({'roi':roi_ev,'num_bets':bets_ev,'profit':profit_ev});
                 roi_str_final = f"{roi_ev:.2f}%" if roi_ev is not None and np.isfinite(roi_ev) else "N/A"
                 logger.info(f"    -> ROI @ Thr EV ({current_optimal_ev_threshold:.3f}) = {roi_str_final} ({bets_ev} bets)")

            else: # Fallback se não houver probs
                 metrics['f1_score_draw']=metrics.get('f1_score_draw_thr05',-1.0); logger.info(" Usando métricas @ 0.5 como F1 final (sem probs).")
                 metrics.update({'log_loss':None,'roc_auc':None,'brier_score':None, 'roi':None,'num_bets':0,'profit':None})

            # Add dataset sizes to metrics
            metrics.update({'train_set_size':len(y_train_cv),'test_set_size':len(y_test)});

            # --- Guarda resultado FINAL ---
            all_results.append({
                'model_name': model_name,
                'model_object': model_trained, # O classificador final treinado
                'pipeline_object': final_pipeline_trained, # O pipeline final treinado
                'scaler': current_scaler,
                'calibrator': calibrator, # Calibrador ajustado no teste
                'params': best_params, # Melhores params da CV temporal
                'metrics': metrics, # Métricas FINAIS avaliadas no teste
                # Os limiares ótimos (calculados no teste) já estão DENTRO de 'metrics'
                'optimal_f1_threshold': current_optimal_f1_threshold, # Redundante, mas ok
                'optimal_ev_threshold': current_optimal_ev_threshold, # Redundante, mas ok
            })

            if progress_callback_stages: progress_callback_stages(i + 1, f"Modelo {model_name} OK");
            logger.info(f"    ==> Resultado FINAL {model_name} adicionado.")

        except Exception as e_outer: logger.error(f"Erro GERAL loop {model_name}: {e_outer}", exc_info=True); continue # Pula modelo se erro grave
        logger.info(f"  Tempo p/ {model_name}: {time.time() - start_time_model:.2f} seg.")
    # --- Fim do Loop for ---

    if len(all_results) >= 2: # Precisa de pelo menos 2 modelos
        # Ordena resultados individuais por F1 (calculado no teste) para pegar os melhores
        # Filtra para garantir que só modelos com objeto real sejam considerados
        valid_results_for_ensemble = [r for r in all_results if r.get('pipeline_object') or r.get('model_object')]
        if len(valid_results_for_ensemble) < 2:
             logger.warning("Modelos individuais válidos insuficientes para ensemble (<2).")
        else:
            valid_results_for_ensemble.sort(key=lambda r: r['metrics'].get('f1_score_draw', -1.0), reverse=True)
            top_n_results = valid_results_for_ensemble[:n_ensemble_models]
            top_n_model_names = [r['model_name'] for r in top_n_results]
            logger.info(f"\n--- Criando Ensemble com Top {len(top_n_results)} Modelos (por F1 no Teste): {top_n_model_names} ---")

            ensemble_estimators = []
            for result_idx, result in enumerate(top_n_results):
                model_name = result['model_name']
                # Prioriza pegar o pipeline final treinado (que inclui sampler)
                pipeline_to_use = result.get('pipeline_object')
                if pipeline_to_use is None:
                     # Fallback para pegar apenas o model_object se o pipeline não foi salvo (menos ideal)
                     pipeline_to_use = result.get('model_object')
                     if pipeline_to_use is None:
                         logger.warning(f"Pipeline/Modelo treinado não encontrado para {model_name} no índice {result_idx}. Pulando para ensemble.")
                         continue
                     else:
                          logger.warning(f"Usando apenas model_object para {model_name} no ensemble (sampler pode estar faltando).")
                else:
                     logger.info(f"  Usando pipeline_object para {model_name} no ensemble.")


                # Usa um ID único incluindo o índice original do all_results para evitar conflitos
                # Precisa encontrar o índice original em all_results para garantir unicidade se houver modelos iguais
                original_index = next((idx for idx, r in enumerate(all_results) if r['model_name'] == model_name and r['metrics'] == result['metrics']), f"fallback{result_idx}") # Fallback ID if exact match fails

                model_id = f"{model_name}_{original_index}"

                # Clona o pipeline/modelo FINAL treinado para o ensemble
                estimator_clone = clone(pipeline_to_use)
                ensemble_estimators.append((model_id, estimator_clone))
                # logger.info(f"  Adicionando '{model_name}' (ID: {model_id}) ao ensemble.") # Log mais detalhado opcional

            if not ensemble_estimators:
                logger.error("ENSEMBLE: Nenhum estimador válido clonado para criar ensemble.")
            else:
                # Cria o Voting Classifier
                voting_clf = VotingClassifier(estimators=ensemble_estimators, voting='soft', n_jobs=N_JOBS_GRIDSEARCH, verbose=False)

                # <<< CORREÇÃO AQUI: Chamar fit() no VotingClassifier >>>
                logger.info("  Ajustando o wrapper VotingClassifier (não re-treina bases)...")
                try:
                    
                    if 'X_train_cv_m' not in locals() or 'y_train_cv' not in locals():
                         
                         logger.warning("X_train_cv_m não encontrado no escopo local para fit do VotingClassifier, tentando usar X_train_cv original.")
                         
                         if 'X_train_cv_m' not in locals(): raise NameError("X_train_cv_m não está definido para fit do VotingClassifier")

                    voting_clf.fit(X_train_cv_m, y_train_cv)
                    logger.info("  Wrapper VotingClassifier ajustado.")
                except Exception as e_vc_fit:
                     logger.error(f"Erro ao ajustar o wrapper VotingClassifier: {e_vc_fit}", exc_info=True)
                     voting_clf = None # Marca como falho

                # --- Avaliação do Ensemble (só continua se voting_clf foi ajustado) ---
                if voting_clf:
                    logger.info("  Avaliando VotingClassifier (ajustado) no Teste...")
                    try:
                        # Usa X_test_m (dados de teste, potencialmente escalados)
                        y_pred_ensemble_05 = voting_clf.predict(X_test_m)
                        y_proba_ensemble_full = voting_clf.predict_proba(X_test_m)
                        ensemble_metrics = {'model_name': 'VotingEnsemble'}

                        if y_proba_ensemble_full.shape[1] >= 2:
                            y_proba_ensemble_draw = y_proba_ensemble_full[:, 1]

                            # Calibração e Otimização de Limiares para o ENSEMBLE (no Teste)
                            logger.info("  Calibrando/Otimizando limiares p/ Ensemble (no Teste)...")
                            try:
                                ens_calibrator = IsotonicRegression(out_of_bounds='clip').fit(y_proba_ensemble_draw, y_test)
                                y_proba_ens_calib = ens_calibrator.predict(y_proba_ensemble_draw)
                            except Exception as e_ens_calib:
                                logger.error(f"Erro ao calibrar Ensemble: {e_ens_calib}")
                                ens_calibrator = None
                                y_proba_ens_calib = y_proba_ensemble_draw # Fallback para raw

                            proba_opt_eval_ens = y_proba_ens_calib if ens_calibrator else y_proba_ensemble_draw
                            opt_eval_src_ens = 'Calib(Teste)' if ens_calibrator else 'Raw(Teste)'

                            ens_f1_thr, ens_ev_thr, ens_prec_thr = DEFAULT_F1_THRESHOLD, DEFAULT_EV_THRESHOLD, 0.5 # Defaults

                            # Otim F1 Ensemble
                            if optimize_f1_threshold and proba_opt_eval_ens is not None:
                                try:
                                    p, r, t = precision_recall_curve(y_test, proba_opt_eval_ens)
                                    min_len=min(len(p),len(r)); p,r=p[:min_len],r[:min_len]
                                    f1=np.divide(2*p*r,p+r+FEATURE_EPSILON,where=(p+r)>0,out=np.zeros_like(p+r))
                                    valid_f1_indices = np.where(np.isfinite(f1))[0] # Ignora NaN/inf em f1
                                    if len(valid_f1_indices)>0:
                                         f1_valid = f1[valid_f1_indices]
                                         idx = valid_f1_indices[np.argmax(f1_valid)]
                                         # Ajusta índice para threshold t (que pode ser menor)
                                         thr_idx = min(idx, len(t)-1) if len(t)>0 else 0
                                         if len(t)>0 : ens_f1_thr = t[thr_idx]
                                         else: logger.warning("Array de thresholds vazio na otim F1 Ensemble.")
                                    else: logger.warning("Nenhum valor F1 válido encontrado para otimização Ensemble.")
                                except Exception as e_f1_ens: logger.error(f"Erro Otim F1 Ensemble: {e_f1_ens}")

                            # Otim EV Ensemble
                            if optimize_ev_threshold and proba_opt_eval_ens is not None and X_test_odds is not None:
                                try:
                                    best_roi_ens=-np.inf; ev_ths=np.linspace(0.0,0.2,21);
                                    for ev_th in ev_ths:
                                        val_roi,_,_=calculate_metrics_with_ev(y_test,proba_opt_eval_ens,ev_th,X_test_odds,odd_draw_col_name);
                                        if val_roi is not None and np.isfinite(val_roi) and val_roi>best_roi_ens: best_roi_ens=val_roi; ens_ev_thr=ev_th;
                                except Exception as e_ev_ens: logger.error(f"Erro Otim EV Ensemble: {e_ev_ens}")

                            # Otim Prec Ensemble
                            if optimize_precision_threshold and proba_opt_eval_ens is not None:
                                try:
                                    best_prec_ens=-1.0; found_prec_ens=False;
                                    thresholds_test_ens=np.linspace(np.min(proba_opt_eval_ens)+1e-4,np.min([np.max(proba_opt_eval_ens)-1e-4,0.99]),100);
                                    ens_prec_thr=0.5
                                    for t_ens in thresholds_test_ens:
                                        y_pred_v_ens=(proba_opt_eval_ens>=t_ens).astype(int); prec_ens=precision_score(y_test,y_pred_v_ens,pos_label=1,zero_division=0); rec_ens=recall_score(y_test,y_pred_v_ens,pos_label=1,zero_division=0);
                                        if rec_ens>=min_recall_target and prec_ens>=best_prec_ens:
                                            if prec_ens>best_prec_ens or t_ens>ens_prec_thr: best_prec_ens=prec_ens; ens_prec_thr=t_ens; found_prec_ens=True;
                                    if not found_prec_ens: ens_prec_thr = 0.5 # Fallback if no suitable threshold found
                                except Exception as e_prec_ens: logger.error(f"Erro Otim Prec Ensemble: {e_prec_ens}"); ens_prec_thr = 0.5


                            # Calcula Métricas Ensemble
                            y_pred_ens_f1=(proba_opt_eval_ens >= ens_f1_thr).astype(int) if proba_opt_eval_ens is not None else y_pred_ensemble_05
                            y_pred_ens_prec=(proba_opt_eval_ens >= ens_prec_thr).astype(int) if proba_opt_eval_ens is not None else y_pred_ensemble_05
                            ensemble_metrics.update({
                                'optimal_f1_threshold':ens_f1_thr, 'optimal_ev_threshold':ens_ev_thr, 'optimal_precision_threshold':ens_prec_thr,
                                'accuracy_thr05':accuracy_score(y_test,y_pred_ensemble_05),'precision_draw_thr05':precision_score(y_test,y_pred_ensemble_05,pos_label=1,zero_division=0),'recall_draw_thr05':recall_score(y_test,y_pred_ensemble_05,pos_label=1,zero_division=0),'f1_score_draw_thr05':f1_score(y_test,y_pred_ensemble_05,pos_label=1,zero_division=0),
                                'accuracy_thrF1':accuracy_score(y_test,y_pred_ens_f1),'precision_draw_thrF1':precision_score(y_test,y_pred_ens_f1,pos_label=1,zero_division=0),'recall_draw_thrF1':recall_score(y_test,y_pred_ens_f1,pos_label=1,zero_division=0),'f1_score_draw':f1_score(y_test,y_pred_ens_f1,pos_label=1,zero_division=0),
                                'accuracy_thrPrec':accuracy_score(y_test,y_pred_ens_prec),'precision_draw_thrPrec':precision_score(y_test,y_pred_ens_prec,pos_label=1,zero_division=0),'recall_draw_thrPrec':recall_score(y_test,y_pred_ens_prec,pos_label=1,zero_division=0),'f1_score_draw_thrPrec':f1_score(y_test,y_pred_ens_prec,pos_label=1,zero_division=0),
                            })
                            # Métricas Prob Ensemble
                            if proba_opt_eval_ens is not None:
                                try: ensemble_metrics['roc_auc']=roc_auc_score(y_test,proba_opt_eval_ens) if len(np.unique(y_test))>1 else 0.5
                                except: ensemble_metrics['roc_auc']=None
                                try: ensemble_metrics['brier_score']=brier_score_loss(y_test,proba_opt_eval_ens)
                                except: ensemble_metrics['brier_score']=None
                            if y_proba_ensemble_full is not None:
                                try: ensemble_metrics['log_loss']=log_loss(y_test,y_proba_ensemble_full)
                                except: ensemble_metrics['log_loss']=None
                            # Métricas EV Ensemble
                            ens_roi,ens_bets,ens_prof = calculate_metrics_with_ev(y_test,proba_opt_eval_ens,ens_ev_thr,X_test_odds,odd_draw_col_name) if proba_opt_eval_ens is not None and X_test_odds is not None else (None, 0, None)
                            ensemble_metrics.update({'roi':ens_roi,'num_bets':ens_bets,'profit':ens_prof})

                            # Log Métricas Ensemble
                            logger.info("  --- Métricas Ensemble (Teste) ---")
                            logger.info(f"    F1 @ThrF1({ens_f1_thr:.4f}) = {ensemble_metrics.get('f1_score_draw',np.nan):.4f}")
                            logger.info(f"    Prec @ThrPrec({ens_prec_thr:.4f}) = {ensemble_metrics.get('precision_draw_thrPrec',np.nan):.4f} (Recall: {ensemble_metrics.get('recall_draw_thrPrec',np.nan):.4f})")
                            logger.info(f"    AUC = {ensemble_metrics.get('roc_auc',np.nan):.4f}, Brier = {ensemble_metrics.get('brier_score',np.nan):.4f}")
                            roi_ens_log = ensemble_metrics.get('roi', np.nan); roi_str = f"{roi_ens_log:.2f}%" if pd.notna(roi_ens_log) and np.isfinite(roi_ens_log) else "N/A"
                            logger.info(f"    ROI @ThrEV({ens_ev_thr:.3f}) = {roi_str} ({ensemble_metrics.get('num_bets')} bets)")
                            logger.info("  ---------------------------------")

                            # Adiciona resultado do ensemble à lista geral
                            all_results.append({
                                'model_name': 'VotingEnsemble',
                                'model_object': voting_clf, # Salva o objeto VotingClassifier ajustado
                                'pipeline_object': None, # Não há um pipeline único para o ensemble
                                'scaler': None,
                                'calibrator': ens_calibrator, # Calibrador do ensemble
                                'params': {'estimators': top_n_model_names, 'voting': 'soft'},
                                'metrics': ensemble_metrics,
                                'optimal_f1_threshold': ens_f1_thr,
                                'optimal_ev_threshold': ens_ev_thr
                            })
                        else: logger.warning(" Ensemble predict_proba não retornou 2 colunas.")
                    except Exception as e_ens_eval:
                        logger.error(f"Erro avaliar Voting Ensemble: {e_ens_eval}", exc_info=True)
            # --- Fim Avaliação Ensemble ---
    else:
        logger.warning("Modelos individuais válidos insuficientes para ensemble (<2).")

    # --- Seleção e Salvamento Final ---
    if progress_callback_stages: progress_callback_stages(len(available_models), "Selecionando/Salvando...") # Ajusta índice
    end_time_total=time.time(); logger.info(f"--- Treino concluído ({end_time_total-start_time_total:.2f} seg) ---")
    if not all_results: logger.error("SELEÇÃO: Nenhum resultado válido para avaliar."); return False

    try:
        # Cria DataFrame com TODOS os resultados (individuais + ensemble se existiu)
        results_df = pd.DataFrame(all_results)
        # Extrai métricas para o DataFrame
        for thr_type in ['thr05', 'thrF1', 'thrPrec']:
             results_df[f'precision_draw_{thr_type}'] = results_df['metrics'].apply(lambda m: m.get(f'precision_draw_{thr_type}', 0.0))
             results_df[f'recall_draw_{thr_type}'] = results_df['metrics'].apply(lambda m: m.get(f'recall_draw_{thr_type}', 0.0))
             results_df[f'f1_score_draw_{thr_type}'] = results_df['metrics'].apply(lambda m: m.get(f'f1_score_draw_{thr_type}', -1.0))
        results_df['f1_score_draw']=results_df['metrics'].apply(lambda m: m.get('f1_score_draw', -1.0)) # F1 principal (do thr F1 otimizado no teste)
        results_df['roi']=results_df['metrics'].apply(lambda m: m.get('roi', -np.inf));
        results_df['num_bets']=results_df['metrics'].apply(lambda m: m.get('num_bets', 0));
        results_df['auc']=results_df['metrics'].apply(lambda m: m.get('roc_auc', 0.0));
        results_df['brier']=results_df['metrics'].apply(lambda m: m.get('brier_score', 1.0))
        results_df['optimal_f1_threshold'] = results_df['metrics'].apply(lambda m: m.get('optimal_f1_threshold', DEFAULT_F1_THRESHOLD))
        results_df['optimal_ev_threshold'] = results_df['metrics'].apply(lambda m: m.get('optimal_ev_threshold', DEFAULT_EV_THRESHOLD))
        results_df['optimal_precision_threshold'] = results_df['metrics'].apply(lambda m: m.get('optimal_precision_threshold', 0.5))

        # Conversão numérica e fillna
        cols_num = ['f1_score_draw','precision_draw_thrF1','recall_draw_thrF1','precision_draw_thrPrec','recall_draw_thrPrec','f1_score_draw_thrPrec','f1_score_draw_thr05', 'roi','num_bets','auc','brier','optimal_f1_threshold','optimal_ev_threshold','optimal_precision_threshold']
        for col in cols_num: results_df[col]=pd.to_numeric(results_df[col],errors='coerce')
        results_df.fillna({'f1_score_draw': -1.0, 'precision_draw_thrF1': 0.0, 'recall_draw_thrF1': 0.0, 'precision_draw_thrPrec': 0.0, 'recall_draw_thrPrec': 0.0, 'f1_score_draw_thrPrec': -1.0, 'f1_score_draw_thr05': -1.0, 'optimal_precision_threshold': 0.5, 'roi': -np.inf, 'num_bets': 0, 'auc': 0.0, 'brier': 1.0, 'optimal_f1_threshold': DEFAULT_F1_THRESHOLD, 'optimal_ev_threshold': DEFAULT_EV_THRESHOLD}, inplace=True)

        # Exibe Comparativo
        logger.info("--- Comparativo Desempenho Modelos (Teste Final) ---");
        display_cols = ['model_name','f1_score_draw','precision_draw_thrF1','recall_draw_thrF1','precision_draw_thrPrec','recall_draw_thrPrec','f1_score_draw_thrPrec','auc','brier','roi','num_bets','optimal_f1_threshold','optimal_ev_threshold','optimal_precision_threshold']
        display_cols_exist=[col for col in display_cols if col in results_df.columns]
        # Arredonda colunas para exibição
        results_df_display=results_df[display_cols_exist].copy()
        float_cols = results_df_display.select_dtypes(include=['float']).columns
        results_df_display[float_cols] = results_df_display[float_cols].round(4)
        if 'num_bets' in results_df_display.columns: results_df_display['num_bets']=results_df_display['num_bets'].astype(int)
        if 'roi' in results_df_display.columns:
            results_df_display['roi'] = results_df_display['roi'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) and np.isfinite(x) else "N/A")

        try: logger.info("\n"+results_df_display.sort_values(by=BEST_MODEL_METRIC,ascending=False).to_markdown(index=False))
        except ImportError: logger.info("\n"+results_df_display.sort_values(by=BEST_MODEL_METRIC,ascending=False).to_string(index=False))
        logger.info("-" * 80)

        # Seleção e Salvamento (Melhor F1 e Melhor ROI - pode ser o mesmo modelo)
        # Usa o DataFrame original `results_df` que contém os objetos
        results_df_sorted_f1=results_df.sort_values(by=BEST_MODEL_METRIC, ascending=False).reset_index(drop=True)
        best_f1_result_dict = results_df_sorted_f1.iloc[0].to_dict() if not results_df_sorted_f1.empty else None

        best_roi_result_dict = None
        # Ordena por ROI, mas desconsidera infinitos ou NaNs
        results_df_valid_roi = results_df[results_df['roi'].notna() & np.isfinite(results_df['roi'])]
        if not results_df_valid_roi.empty:
            results_df_sorted_roi = results_df_valid_roi.sort_values(by=BEST_MODEL_METRIC_ROI, ascending=False).reset_index(drop=True)
            best_roi_result_dict = results_df_sorted_roi.iloc[0].to_dict()
            logger.info(f"SELEÇÃO: Melhor ROI Válido: {best_roi_result_dict.get('model_name')} (ROI={best_roi_result_dict['metrics'].get('roi'):.2f}%)")
        else:
            logger.warning("SELEÇÃO: Nenhum ROI finito/válido encontrado.")

        if best_f1_result_dict:
             logger.info(f"Salvando Melhor F1 ({best_f1_result_dict.get('model_name','ERRO')})...");
             _save_model_object(best_f1_result_dict, feature_names, BEST_F1_MODEL_SAVE_PATH)
        else: logger.error("ERRO SALVAR: Nenhum modelo encontrado como melhor F1."); return False

        # Decide o que salvar como "Melhor ROI"
        model_to_save_roi = None
        if best_roi_result_dict:
            # Se o melhor ROI é DIFERENTE do melhor F1, salva o melhor ROI
            if best_f1_result_dict and best_roi_result_dict.get('model_name') != best_f1_result_dict.get('model_name'):
                model_to_save_roi = best_roi_result_dict
                logger.info(f"  -> Usando Melhor ROI ({model_to_save_roi.get('model_name')}) p/ slot ROI.")
            # Se o melhor ROI é o MESMO que o melhor F1, tenta pegar o SEGUNDO melhor F1
            elif len(results_df_sorted_f1) > 1:
                 model_to_save_roi = results_df_sorted_f1.iloc[1].to_dict()
                 logger.info(f"  -> Melhor ROI é igual ao Melhor F1. Usando 2º Melhor F1 ({model_to_save_roi.get('model_name')}) p/ slot ROI.")
            # Se só há 1 modelo, usa o melhor F1 para ambos os slots
            else:
                 model_to_save_roi = best_f1_result_dict
                 logger.warning("  -> Apenas 1 modelo avaliado. Usando Melhor F1 para ambos os slots (F1 e ROI).")
        # Se NÃO houve resultado válido de ROI, usa o melhor F1 para ambos
        elif best_f1_result_dict:
             model_to_save_roi = best_f1_result_dict
             logger.warning(f"  -> Nenhum ROI válido. Usando Melhor F1 ({model_to_save_roi.get('model_name')}) p/ slot ROI.")

        if model_to_save_roi:
            logger.info(f"Salvando Melhor ROI/2nd F1 ({model_to_save_roi.get('model_name','ERRO')})...");
            _save_model_object(model_to_save_roi, feature_names, BEST_ROI_MODEL_SAVE_PATH)
        else:
            # Este caso só ocorreria se best_f1_result_dict também fosse None, o que já retornaria False antes.
            logger.error("ERRO SALVAR: Não foi possível determinar modelo para slot ROI."); return False

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