import pandas as pd
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.isotonic import IsotonicRegression
# from sklearn.calibration import CalibratedClassifierCV # Não usado diretamente com Isotonic fora
from logger_config import setup_logger

logger = setup_logger("ModelTrainerApp")


# --- Imblearn Imports ---
try:
    from imblearn.pipeline import Pipeline as ImbPipeline # Renomeado para evitar conflito
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    # from imblearn.under_sampling import NearMiss # Se quiser testar
    # from imblearn.combine import SMOTEENN, SMOTETomek # Se quiser testar
    IMBLEARN_AVAILABLE = True
    logger.info("Biblioteca 'imbalanced-learn' carregada com sucesso.")
except ImportError:
    logger.error("ERRO CRÍTICO: Biblioteca 'imbalanced-learn' não instalada. Instale com 'pip install imbalanced-learn'.")
    IMBLEARN_AVAILABLE = False
    # Define um Pipeline dummy para evitar erros fatais, mas o sampler não funcionará
    from sklearn.pipeline import Pipeline as ImbPipeline
    SMOTE = None # Define como None para verificações posteriores
    RandomOverSampler = None
# -----------------------

try:
    from skopt import BayesSearchCV
    SKOPT_AVAILABLE = True
except ImportError:
    logger.warning("AVISO: scikit-optimize (skopt) não instalado. Usando GridSearchCV como fallback.")
    SKOPT_AVAILABLE = False
    BayesSearchCV = GridSearchCV # Define BayesSearchCV como GridSearchCV se skopt não estiver lá

# Importar BAYESIAN_OPT_N_ITER do config
from config import BAYESIAN_OPT_N_ITER

try:
    import lightgbm as lgb
    LGBMClassifier = lgb.LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    logger.warning("AVISO: LightGBM não instalado.")
    lgb = None; LGBMClassifier = None
    LGBM_AVAILABLE = False

# Importar métricas e numpy
from sklearn.metrics import (accuracy_score, classification_report, log_loss,
                             precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, brier_score_loss,
                             precision_recall_curve)
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib, os, datetime, numpy as np, traceback
try:
    from config import (RANDOM_STATE, MODEL_CONFIG, CLASS_NAMES, TEST_SIZE,
                        BEST_F1_MODEL_SAVE_PATH, BEST_ROI_MODEL_SAVE_PATH,
                        CROSS_VALIDATION_SPLITS, N_JOBS_GRIDSEARCH, DEFAULT_F1_THRESHOLD,
                        ODDS_COLS, BEST_MODEL_METRIC, BEST_MODEL_METRIC_ROI, DEFAULT_EV_THRESHOLD)
    from typing import Any, Optional, Dict, Tuple, List, Callable
except ImportError as e: logger.critical(f"Erro crítico import config/typing: {e}", exc_info=True); raise # Mudado para critical

# --- Funções Auxiliares (roi, calculate_roi_with_threshold, calculate_metrics_with_ev, scale_features) ---
# (Mantidas como estavam no seu código original - Copie as definições completas aqui)
def roi(y_test: pd.Series, y_pred: np.ndarray, X_test_odds_aligned: Optional[pd.DataFrame], odd_draw_col_name: str) -> Optional[float]:
    if X_test_odds_aligned is None or odd_draw_col_name not in X_test_odds_aligned.columns: return None
    try: common_index=y_test.index.intersection(X_test_odds_aligned.index);
    except AttributeError: logger.error("Error accessing index in roi function."); return None
    if len(common_index) != len(y_test):
         logger.warning(f"ROI calc: Index mismatch. y_test={len(y_test)}, common={len(common_index)}. Skipping ROI.")
         return None # Retorna None explicitamente se índices não batem
    y_test_common=y_test.loc[common_index];
    try: y_pred_series=pd.Series(y_pred, index=y_test.index); y_pred_common=y_pred_series.loc[common_index];
    except (ValueError,AttributeError): logger.error("Error aligning y_pred in roi."); return None
    predicted_draws_indices=common_index[y_pred_common == 1]; num_bets=len(predicted_draws_indices);
    if num_bets == 0: return 0.0
    actuals=y_test_common.loc[predicted_draws_indices]; odds=pd.to_numeric(X_test_odds_aligned.loc[predicted_draws_indices, odd_draw_col_name], errors='coerce')
    profit=0; valid_bets=0;
    for idx in predicted_draws_indices:
        odd_d=odds.loc[idx];
        # Verifica se odd é válida (maior que 1, não NaN)
        if pd.notna(odd_d) and odd_d > 1:
             profit += (odd_d - 1) if actuals.loc[idx] == 1 else -1; valid_bets += 1;
        # Opcional: Loggar odds inválidas encontradas
        # elif pd.notna(odd_d):
        #    logger.debug(f"ROI Calc: Invalid odd {odd_d} skipped for index {idx}")

    if valid_bets == 0: return 0.0 # Retorna 0.0 ROI se nenhuma aposta válida foi feita
    return (profit / valid_bets) * 100

def calculate_roi_with_threshold(y_true: pd.Series, y_proba: np.ndarray, threshold: float, odds_data: Optional[pd.DataFrame], odd_col_name: str) -> Tuple[Optional[float], int, Optional[float]]:
    profit, roi_value, num_bets, profit_calc, valid_bets_count = None, None, 0, 0, 0
    if odds_data is None or odd_col_name not in odds_data.columns: return roi_value, num_bets, profit
    try:
        common_index=y_true.index.intersection(odds_data.index);
        if len(common_index)==0: return 0.0, 0, 0.0
        if len(common_index) != len(y_true): logger.warning(f"ROI Thr: Index mismatch. y_true={len(y_true)}, common={len(common_index)}. Using common subset.")
        y_true_common=y_true.loc[common_index]; odds_common=pd.to_numeric(odds_data.loc[common_index,odd_col_name],errors='coerce');
        try:
            # Garante alinhamento da probabilidade com o índice COMUM
            y_proba_series=pd.Series(y_proba, index=y_true.index) # Assume alinhamento original
            y_proba_common=y_proba_series.loc[common_index]
        except Exception as e_align_proba: logger.error(f"ROI Thr: Erro alinhar y_proba: {e_align_proba}"); return None,0,None

        bet_indices=common_index[y_proba_common > threshold]; num_bets=len(bet_indices);
        if num_bets==0: return 0.0,num_bets,0.0
        actuals=y_true_common.loc[bet_indices]; odds_selected=odds_common.loc[bet_indices];
        for idx in bet_indices:
            odd_d=odds_selected.loc[idx];
            if pd.notna(odd_d) and odd_d > 1:
                 profit_calc += (odd_d - 1) if actuals.loc[idx] == 1 else -1; valid_bets_count += 1;
            # elif pd.notna(odd_d): logger.debug(f"ROI Thr Calc: Invalid odd {odd_d} skipped for index {idx}")

        profit = profit_calc; roi_value = (profit / valid_bets_count) * 100 if valid_bets_count > 0 else 0.0;
        return roi_value, num_bets, profit # Retorna valid_bets_count como num_bets efetivas
    except Exception as e: logger.error(f"ROI Thr: Erro - {e}",exc_info=True); return None,0,None

def calculate_metrics_with_ev(y_true: pd.Series, y_proba_calibrated: np.ndarray, ev_threshold: float, odds_data: Optional[pd.DataFrame], odd_col_name: str) -> Tuple[Optional[float], int, Optional[float]]:
    profit, roi_value, num_bets, profit_calc, valid_bets_count = None, None, 0, 0, 0
    if odds_data is None or odd_col_name not in odds_data.columns: logger.warning(f"EV Metr: Odds ausentes ('{odd_col_name}')."); return roi_value,num_bets,profit
    try:
        common_index=y_true.index.intersection(odds_data.index);
        if len(common_index)==0: return 0.0, 0, 0.0
        if len(common_index) != len(y_true): logger.warning(f"EV Metr: Index mismatch. y_true={len(y_true)}, common={len(common_index)}. Using common subset.")

        y_true_common=y_true.loc[common_index]; odds_common=pd.to_numeric(odds_data.loc[common_index,odd_col_name],errors='coerce');
        try:
            # Garante alinhamento da probabilidade com o índice COMUM
            y_proba_common=pd.Series(y_proba_calibrated,index=y_true.index).loc[common_index]
        except Exception as e_align_proba: logger.error(f"EV Metr: Erro alinhar y_proba: {e_align_proba}"); return None,0,None

        valid_mask = odds_common.notna() & y_proba_common.notna() & (odds_common > 1)
        ev = pd.Series(np.nan, index=common_index);

        # Calcula EV apenas onde os dados são válidos
        # ev[valid_mask] = (y_proba_common[valid_mask]*(odds_common[valid_mask]-1)) - ((1-y_proba_common[valid_mask])*1); # Original
        prob_ok = y_proba_common[valid_mask]
        odds_ok = odds_common[valid_mask]
        ev_calc = (prob_ok * (odds_ok - 1)) - ((1 - prob_ok) * 1)
        ev.loc[valid_mask] = ev_calc

        # Filtra pelo limiar de EV (ignorando NaNs resultantes de dados inválidos)
        bet_indices=common_index[ev > ev_threshold]; num_bets=len(bet_indices); # num_bets agora reflete apostas com EV > threshold
        if num_bets==0: return 0.0,num_bets,0.0 # Retorna 0 bets se nenhuma passou no threshold

        actuals=y_true_common.loc[bet_indices]; odds_selected=odds_common.loc[bet_indices];
        for idx in bet_indices:
            odd_d=odds_selected.loc[idx];
            # A verificação > 1 já foi feita no valid_mask, mas é bom garantir
            if pd.notna(odd_d) and odd_d > 1:
                 profit_calc += (odd_d - 1) if actuals.loc[idx] == 1 else -1; valid_bets_count += 1;

        profit = profit_calc; roi_value = (profit / valid_bets_count) * 100 if valid_bets_count > 0 else 0.0;
        # Log usa num_bets (que passaram threshold) e valid_bets_count (que tinham odd válida > 1)
        logger.debug(f"    -> Métricas EV (Th={ev_threshold:.3f}): ROI={roi_value:.2f}%, Bets Sugeridas={num_bets}, Bets Válidas={valid_bets_count}, Profit={profit:.2f}")
        return roi_value, num_bets, profit # Retorna num_bets (sugeridas)
    except Exception as e: logger.error(f"EV Metr: Erro - {e}",exc_info=True); return None,0,None

def scale_features(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, scaler_type='standard') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Any]:
    """ Escala features usando treino, aplica em val e teste. Trata Inf/NaN. """
    X_train_c, X_val_c, X_test_c = X_train.copy(), X_val.copy(), X_test.copy(); scaler = None
    try:
        if scaler_type=='minmax': scaler=MinMaxScaler()
        elif scaler_type=='standard': scaler=StandardScaler()
        else: logger.warning(f"Tipo de scaler '{scaler_type}' desconhecido. Usando StandardScaler."); scaler = StandardScaler()

        logger.info(f"  Aplicando {scaler.__class__.__name__}...");

        # --- Tratamento de Inf/NaN antes de fit/transform ---
        cols = X_train_c.columns
        # Treino: Substitui inf por NaN, depois preenche NaN com mediana
        X_train_c = X_train_c.replace([np.inf, -np.inf], np.nan)
        if X_train_c.isnull().values.any():
             logger.warning("  NaNs encontrados em X_train antes de escalar. Imputando com mediana.")
             train_median = X_train_c.median()
             X_train_c.fillna(train_median, inplace=True)
             # Verifica se ainda há NaNs (caso uma coluna inteira fosse NaN)
             if X_train_c.isnull().values.any():
                 nan_cols = X_train_c.columns[X_train_c.isnull().all()].tolist()
                 logger.error(f"  ERRO FATAL: Colunas inteiras NaN em X_train após imputação: {nan_cols}. Verifique dados originais.")
                 raise ValueError("Colunas inteiras NaN em X_train.")
        else:
            train_median = X_train_c.median() # Calcula mesmo se não houver NaN, para usar em val/test

        # Validação e Teste: Substitui inf por NaN, depois preenche com mediana DO TREINO
        if X_val_c is not None:
            X_val_c = X_val_c.replace([np.inf, -np.inf], np.nan)
            X_val_c.fillna(train_median, inplace=True)
            if X_val_c.isnull().values.any(): logger.warning("NaNs ainda presentes em X_val após imputação!")


        if X_test_c is not None:
            X_test_c = X_test_c.replace([np.inf, -np.inf], np.nan)
            X_test_c.fillna(train_median, inplace=True)
            if X_test_c.isnull().values.any(): logger.warning("NaNs ainda presentes em X_test após imputação!")
        # -----------------------------------------------------

        # Fit no treino (limpo) e transform
        scaler.fit(X_train_c)
        X_train_scaled = scaler.transform(X_train_c)
        X_val_scaled = scaler.transform(X_val_c) if X_val_c is not None else None
        X_test_scaled = scaler.transform(X_test_c) if X_test_c is not None else None

        # Recria DataFrames com índices e colunas originais
        X_train_sc = pd.DataFrame(X_train_scaled, index=X_train.index, columns=cols)
        X_val_sc = pd.DataFrame(X_val_scaled, index=X_val.index, columns=cols) if X_val_scaled is not None else None
        X_test_sc = pd.DataFrame(X_test_scaled, index=X_test.index, columns=cols) if X_test_scaled is not None else None

        logger.info("  Scaling concluído.")
        return X_train_sc, X_val_sc, X_test_sc, scaler # Retorna 4 itens

    except Exception as e:
        logger.error(f"Erro GERAL no scaling: {e}", exc_info=True)
        raise # Re-levanta a exceção para parar o processo


# --- Função Principal de Treinamento (COM PIPELINE IMBLEARN) ---
def train_evaluate_and_save_best_models(
    X: pd.DataFrame, y: pd.Series,
    X_test_with_odds: Optional[pd.DataFrame] = None,
    progress_callback_stages: Optional[Callable[[int, str], None]] = None,
    num_total_models: int = 1,
    scaler_type: str = 'standard',
    sampler_type: str = 'smote', # Adicionado: 'smote', 'random', ou None
    odd_draw_col_name: str = ODDS_COLS.get('draw', 'Odd_D_FT'),
    calibration_method: str = 'isotonic',
    optimize_ev_threshold: bool = True,
    optimize_f1_threshold: bool = True
    ) -> bool:
    """
    Treina (com Sampler + CV), calibra, otimiza limiares, avalia, salva.
    """
    # --- Verificações Iniciais ---
    if not IMBLEARN_AVAILABLE and sampler_type is not None:
        logger.error("Imbalanced-learn não está disponível, mas um sampler foi solicitado. Abortando.")
        return False
    if X is None or y is None: logger.error("Dados X ou y None."); return False
    if not MODEL_CONFIG: logger.error("MODEL_CONFIG vazio."); return False
    available_models = {name: config for name, config in MODEL_CONFIG.items() if not (name=='LGBMClassifier' and not LGBM_AVAILABLE)}
    if not available_models: logger.error("Nenhum modelo válido."); return False
    if num_total_models != len(available_models): num_total_models = len(available_models); logger.warning("Ajustando num_total_models.")

    feature_names = list(X.columns); all_results = []
    sampler_log_msg = f"Sampler: {sampler_type}" if sampler_type else "Sampler: None"
    logger.info(f"--- Treinando {num_total_models} Modelos ({sampler_log_msg}, Opt F1: {optimize_f1_threshold}, Opt EV: {optimize_ev_threshold}) ---")
    start_time_total = time.time()

    # --- Divisão Tripla e Preparo Odds ---
    logger.info("Dividindo dados...")
    val_size=0.20; test_size_final=TEST_SIZE; train_val_size_temp=1.0-test_size_final
    if train_val_size_temp <= 0: logger.error("TEST_SIZE inválido (>= 1.0)."); return False;
    val_size_relative = val_size / train_val_size_temp
    try:
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size_final, random_state=RANDOM_STATE, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size_relative, random_state=RANDOM_STATE, stratify=y_train_val)
        logger.info(f"Split: T={len(X_train)}, V={len(X_val)}, Ts={len(X_test)}")
        # Alinha odds para Validação e Teste
        X_val_odds, X_test_odds = None, None
        if X_test_with_odds is not None and not X_test_with_odds.empty and odd_draw_col_name in X_test_with_odds.columns:
            common_val = X_val.index.intersection(X_test_with_odds.index); common_test = X_test.index.intersection(X_test_with_odds.index)
            if len(common_val) > 0: X_val_odds = X_test_with_odds.loc[common_val, [odd_draw_col_name]].copy()
            if len(common_test) > 0: X_test_odds = X_test_with_odds.loc[common_test, [odd_draw_col_name]].copy()
            logger.info(f"Odds alinhadas: Val={X_val_odds is not None}, Teste={X_test_odds is not None}")
        else: logger.warning("Não foi possível alinhar odds (DataFrame de odds ausente, vazio ou sem a coluna de empate).")
    except Exception as e: logger.error(f"Erro divisão/alinhar dados: {e}", exc_info=True); return False

    # --- Loop pelos modelos ---
    for i, (model_name, config) in enumerate(available_models.items()):
        status_text_loop = f"Modelo {i+1}/{num_total_models}: {model_name}"
        logger.info(f"\n--- {status_text_loop} ---")
        if progress_callback_stages: progress_callback_stages(i, f"Iniciando {model_name}...")
        start_time_model = time.time(); model_trained = None; best_params = None; current_scaler = None; calibrator = None;
        best_pipeline_object = None # Para armazenar o pipeline otimizado
        current_optimal_f1_threshold = DEFAULT_F1_THRESHOLD
        current_optimal_ev_threshold = DEFAULT_EV_THRESHOLD

        try:
            # --- Setup do Modelo e Pipeline ---
            model_class = eval(model_name); model_kwargs = config.get('model_kwargs', {}); needs_scaling = config.get('needs_scaling', False)
            use_bayes_opt = SKOPT_AVAILABLE and 'search_spaces' in config;
            # ATENÇÃO: param_space DEVE ter os nomes prefixados com 'classifier__' no config.py
            param_space = config.get('search_spaces') if use_bayes_opt else config.get('param_grid')
            if param_space is None: logger.warning(f"  Espaço de busca/grid não definido para {model_name}. Usando parâmetros padrão.")

            X_train_m, X_val_m, X_test_m = X_train.copy(), X_val.copy(), X_test.copy() # Cópias para este modelo

            # --- Scaling (Aplicado ANTES do Pipeline/Treino) ---
            if needs_scaling:
                 if progress_callback_stages: progress_callback_stages(i, f"Scaling...")
                 logger.info(f"  Scaling p/ {model_name} com {scaler_type} scaler...")
                 try:
                     X_train_m, X_val_m, X_test_m, current_scaler = scale_features(
                         X_train_m, X_val_m, X_test_m, scaler_type
                     )
                     logger.info("    -> Scaling OK.")
                 except Exception as e_scale:
                     logger.error(f"  ERRO FATAL no scaling para {model_name}: {e_scale}", exc_info=True)
                     continue # Pula para o próximo modelo se scaling falhar

            # --- Criação do Pipeline (Sampler + Classifier) ---
            pipeline_steps = []
            sampler = None
            if sampler_type == 'smote' and SMOTE:
                sampler = SMOTE(random_state=RANDOM_STATE, k_neighbors=5) # k_neighbors pode ser ajustado/otimizado
                pipeline_steps.append(('sampler', sampler))
                logger.info("  Usando SMOTE no pipeline.")
            elif sampler_type == 'random' and RandomOverSampler:
                sampler = RandomOverSampler(random_state=RANDOM_STATE)
                pipeline_steps.append(('sampler', sampler))
                logger.info("  Usando RandomOverSampler no pipeline.")
            elif sampler_type is not None:
                 logger.warning(f"  Tipo de sampler '{sampler_type}' desconhecido ou indisponível. Treinando sem sampler.")

            # Adiciona o classificador ao pipeline
            pipeline_steps.append(('classifier', model_class(**model_kwargs)))
            pipeline = ImbPipeline(pipeline_steps) # Cria o pipeline (imblearn ou sklearn se imblearn falhou)

            # --- Treino (Busca de Hiperparâmetros ou Padrão) ---
            if param_space:
                 method_name = 'BayesSearchCV' if use_bayes_opt else 'GridSearchCV'
                 search_status_msg = f"Ajustando ({method_name[:5]}" + (f"+{sampler_type.upper()})" if sampler else ")")
                 if progress_callback_stages: progress_callback_stages(i, search_status_msg)
                 logger.info(f"  Iniciando {method_name} com Pipeline...")
                 # GARANTA QUE OS PARÂMETROS EM config.py TENHAM O PREFIXO 'classifier__'
                 try:
                     if use_bayes_opt:
                         search_cv = BayesSearchCV(
                             estimator=pipeline, # <<< USA O PIPELINE
                             search_spaces=param_space,
                             n_iter=BAYESIAN_OPT_N_ITER,
                             cv=CROSS_VALIDATION_SPLITS,
                             n_jobs=N_JOBS_GRIDSEARCH,
                             scoring='f1', # F1 da classe positiva (empate)
                             random_state=RANDOM_STATE,
                             verbose=0
                         )
                     else: # GridSearchCV
                         search_cv = GridSearchCV(
                             estimator=pipeline, # <<< USA O PIPELINE
                             param_grid=param_space,
                             cv=CROSS_VALIDATION_SPLITS,
                             n_jobs=N_JOBS_GRIDSEARCH,
                             scoring='f1', # F1 da classe positiva (empate)
                             verbose=0,
                             error_score='raise' # Levanta erro se um combo falhar
                         )

                     # Ajusta nos dados de treino (escalados se necessário)
                     search_cv.fit(X_train_m, y_train)

                     best_pipeline_object = search_cv.best_estimator_ # O melhor estimador é o pipeline
                     model_trained = best_pipeline_object.named_steps['classifier'] # Extrai o classificador treinado
                     best_params_raw = search_cv.best_params_
                     # Remove prefixo para salvar/logar
                     best_params = {k.replace('classifier__', ''): v for k, v in best_params_raw.items() if k.startswith('classifier__')}
                     best_cv_score = search_cv.best_score_
                     logger.info(f"    -> Busca ({method_name}) OK. Melhor CV F1 (Empate): {best_cv_score:.4f}. Params: {best_params}")

                 except Exception as e_search:
                     logger.error(f"    Erro {method_name}.fit com Pipeline: {e_search}", exc_info=True)
                     model_trained = None; best_pipeline_object = None; # Reseta em caso de erro
                     logger.warning("    -> Tentando fallback (sem busca)...")

            # Fallback: Treinar apenas o pipeline com parâmetros padrão (SE a busca falhou OU não havia param_space)
            if model_trained is None:
                 fallback_status_msg = "Ajustando (Padrão" + (f"+{sampler_type.upper()})" if sampler else ")")
                 if progress_callback_stages: progress_callback_stages(i, fallback_status_msg)
                 logger.info(f"  Treinando Pipeline com params padrão...")
                 try:
                     pipeline.fit(X_train_m, y_train) # Treina o pipeline inteiro
                     best_pipeline_object = pipeline # Usa o pipeline treinado
                     model_trained = best_pipeline_object.named_steps['classifier'] # Extrai o classificador
                     best_params = {k: v for k, v in model_trained.get_params().items() if k in model_kwargs} # Pega os params padrão usados
                     logger.info("    -> Treino padrão com Pipeline OK.")
                 except Exception as e_fall:
                     logger.error(f"    Erro treino fallback com Pipeline: {e_fall}", exc_info=True)
                     continue # Pula para o próximo modelo

            # --- Calibração ---
            # Usa o best_pipeline_object (resultado da busca ou fallback) para prever probs na validação
            y_proba_val_raw_full, y_proba_val_calib = None, None
            if hasattr(best_pipeline_object, "predict_proba"):
                if progress_callback_stages: progress_callback_stages(i, "Calibrando...")
                logger.info(f"  Calibrando probs ({calibration_method}) usando previsões do Pipeline...")
                try:
                    # Prever no conjunto de validação (escalado)
                    y_proba_val_raw_full = best_pipeline_object.predict_proba(X_val_m)

                    if y_proba_val_raw_full.shape[1] > 1:
                        y_proba_val_raw_draw = y_proba_val_raw_full[:, 1] # Prob da classe 1 (Empate)
                        calibrator = IsotonicRegression(out_of_bounds='clip')
                        calibrator.fit(y_proba_val_raw_draw, y_val) # Calibra usando probs brutas da validação
                        y_proba_val_calib = calibrator.predict(y_proba_val_raw_draw)
                        logger.info("  -> Calibrador treinado.")
                    else:
                         logger.warning("  predict_proba retornou apenas uma coluna. Não é possível calibrar.")
                except Exception as e_calib:
                    logger.error(f"  Erro durante calibração: {e_calib}", exc_info=True)
                    calibrator = None # Define calibrador como None se falhar
            else:
                logger.warning(f"  Pipeline/Modelo {model_name} não tem predict_proba. Calibração pulada.")

            # --- Otimização Limiares (na Validação) ---
            # Usa probs calibradas (se sucesso) ou brutas da validação
            proba_opt_val = y_proba_val_calib if calibrator else (y_proba_val_raw_draw if 'y_proba_val_raw_draw' in locals() else None)
            opt_src_val = 'Calib' if calibrator else ('Raw' if 'y_proba_val_raw_draw' in locals() else 'N/A')

            if optimize_f1_threshold and proba_opt_val is not None:
                 if progress_callback_stages: progress_callback_stages(i, f"Otimizando F1 Thr ({opt_src_val})...")
                 logger.info(f"  Otimizando F1 (Val Probs {opt_src_val})..."); best_val_f1 = -1.0;
                 try:
                     p, r, t = precision_recall_curve(y_val, proba_opt_val)
                     # Evita divisão por zero e NaNs
                     f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) != 0)
                     # Acha melhor limiar (excluindo o último limiar que corresponde a recall 0)
                     valid_indices = np.where(f1[:-1] == np.max(f1[:-1]))[0]
                     if len(valid_indices) > 0:
                         idx = valid_indices[-1] # Pega o último índice com F1 máximo (geralmente mais conservador)
                         current_optimal_f1_threshold = t[idx]
                         best_val_f1 = f1[idx]
                         logger.info(f"    Limiar F1 ótimo(Val): {current_optimal_f1_threshold:.4f} (F1={best_val_f1:.4f})")
                     else:
                         logger.warning("    Não foi possível encontrar limiar F1 ótimo na validação.")
                         current_optimal_f1_threshold = DEFAULT_F1_THRESHOLD
                 except Exception as e_f1_opt:
                     logger.error(f"  Erro otimização F1: {e_f1_opt}", exc_info=True)
                     current_optimal_f1_threshold = DEFAULT_F1_THRESHOLD

            if optimize_ev_threshold and proba_opt_val is not None and X_val_odds is not None:
                if progress_callback_stages: progress_callback_stages(i, f"Otimizando EV Thr ({opt_src_val})...")
                logger.info(f"  Otimizando EV (Val Probs {opt_src_val})..."); best_val_roi_ev = -np.inf;
                try:
                    ev_ths = np.linspace(0.0, 0.20, 21) # Testar limiares de 0 a 0.20
                    for ev_th in ev_ths:
                        val_roi, _, _ = calculate_metrics_with_ev(y_val, proba_opt_val, ev_th, X_val_odds, odd_draw_col_name)
                        if val_roi is not None and val_roi > best_val_roi_ev:
                            best_val_roi_ev = val_roi; current_optimal_ev_threshold = ev_th;
                    if best_val_roi_ev > -np.inf:
                         logger.info(f"    Limiar EV ótimo(Val): {current_optimal_ev_threshold:.3f} (ROI={best_val_roi_ev:.2f}%)")
                    else:
                        logger.warning("    ROI inválido encontrado durante otimização EV na validação.")
                        current_optimal_ev_threshold = DEFAULT_EV_THRESHOLD
                except Exception as e_ev_opt:
                    logger.error(f"  Erro otimização EV: {e_ev_opt}", exc_info=True)
                    current_optimal_ev_threshold = DEFAULT_EV_THRESHOLD
            elif optimize_ev_threshold:
                logger.warning(f"  Otimização EV pulada (sem probs de validação ou odds de validação).")

            # --- Avaliação no Teste ---
            if progress_callback_stages: progress_callback_stages(i, f"Avaliando...")
            logger.info(f"  Avaliando Pipeline/Modelo no Teste...")
            metrics = {
                'optimal_f1_threshold': current_optimal_f1_threshold,
                'optimal_ev_threshold': current_optimal_ev_threshold
            }
            y_proba_test_raw_full, y_proba_test_raw_draw, y_proba_test_calib = None, None, None

            # Prever probabilidades no teste usando o pipeline/modelo
            if hasattr(best_pipeline_object, "predict_proba"):
                try:
                    y_proba_test_raw_full = best_pipeline_object.predict_proba(X_test_m) # <<< Usa pipeline/modelo nos dados escalados
                    if y_proba_test_raw_full.shape[1] > 1:
                        y_proba_test_raw_draw = y_proba_test_raw_full[:, 1]
                        # Aplica calibrador (se treinado) nas probs brutas do teste
                        if calibrator:
                            y_proba_test_calib = calibrator.predict(y_proba_test_raw_draw)
                    else:
                        logger.warning(" predict_proba no teste retornou apenas uma coluna.")
                except Exception as e_pred_test:
                    logger.error(f"  Erro predict_proba teste: {e_pred_test}", exc_info=True)
            else:
                 logger.warning(f"  Pipeline/Modelo {model_name} não tem predict_proba para avaliação no teste.")

            # --- Calcular Métricas ---
            # Usar probs calibradas se disponíveis, senão as brutas
            proba_eval_test = y_proba_test_calib if calibrator else y_proba_test_raw_draw
            eval_src_test = 'Calib' if calibrator else ('Raw' if y_proba_test_raw_draw is not None else 'N/A')
            logger.info(f"  Calculando métricas de teste usando probs: {eval_src_test}")

            # Métricas baseadas no limiar 0.5 (predição padrão)
            try:
                # Usa predict() do pipeline/modelo para obter predições 0/1 com limiar 0.5
                y_pred05 = best_pipeline_object.predict(X_test_m)
                acc05 = accuracy_score(y_test, y_pred05)
                prec05 = precision_score(y_test, y_pred05, pos_label=1, zero_division=0)
                rec05 = recall_score(y_test, y_pred05, pos_label=1, zero_division=0)
                f1_05 = f1_score(y_test, y_pred05, pos_label=1, zero_division=0)
                matrix05 = confusion_matrix(y_test, y_pred05).tolist()
                metrics.update({
                    'accuracy_thr05': acc05, 'precision_draw_thr05': prec05,
                    'recall_draw_thr05': rec05, 'f1_score_draw_thr05': f1_05,
                    'confusion_matrix_thr05': matrix05
                })
                logger.info(f"    -> Métricas @ Thr 0.5: F1={f1_05:.4f}, P={prec05:.4f}, R={rec05:.4f}")
            except Exception as e_metrics05:
                logger.error(f"  Erro cálculo métricas @ 0.5: {e_metrics05}", exc_info=True)

            # Métricas baseadas no limiar F1 otimizado
            f1_final, prec_final, rec_final = -1.0, 0.0, 0.0
            if proba_eval_test is not None:
                 try:
                     y_predF1 = (proba_eval_test >= current_optimal_f1_threshold).astype(int)
                     accF1 = accuracy_score(y_test, y_predF1)
                     prec_final = precision_score(y_test, y_predF1, pos_label=1, zero_division=0)
                     rec_final = recall_score(y_test, y_predF1, pos_label=1, zero_division=0)
                     f1_final = f1_score(y_test, y_predF1, pos_label=1, zero_division=0)
                     matrixF1 = confusion_matrix(y_test, y_predF1).tolist()
                     metrics.update({
                         'accuracy_thrF1': accF1, 'precision_draw': prec_final, # Nome padrão para Precision final
                         'recall_draw': rec_final,    # Nome padrão para Recall final
                         'f1_score_draw': f1_final,     # Nome padrão para F1 final
                         'confusion_matrix_thrF1': matrixF1
                     })
                     logger.info(f"    -> Métricas @ Thr F1 ({current_optimal_f1_threshold:.4f}): F1={f1_final:.4f}, P={prec_final:.4f}, R={rec_final:.4f}")
                 except Exception as e_metricsF1:
                     logger.error(f"  Erro cálculo métricas @ Thr F1: {e_metricsF1}", exc_info=True)
                     # Fallback para métricas de 0.5 se o cálculo F1 falhar
                     f1_final = metrics.get('f1_score_draw_thr05', -1.0)
                     metrics['f1_score_draw'] = f1_final
                     metrics['precision_draw'] = metrics.get('precision_draw_thr05', 0.0)
                     metrics['recall_draw'] = metrics.get('recall_draw_thr05', 0.0)
            else:
                # Se não houver probs para avaliar F1, usa as de 0.5 como finais
                logger.warning("  Impossível calcular métricas @ Thr F1 (sem probabilidades). Usando métricas @ 0.5 como finais.")
                f1_final = metrics.get('f1_score_draw_thr05', -1.0)
                metrics['f1_score_draw'] = f1_final
                metrics['precision_draw'] = metrics.get('precision_draw_thr05', 0.0)
                metrics['recall_draw'] = metrics.get('recall_draw_thr05', 0.0)

            # Métricas baseadas em probabilidade (AUC, Brier, LogLoss)
            logloss, auc, brier = None, None, None
            # Usa probs calibradas (proba_eval_test) para AUC e Brier
            # Usa probs brutas (y_proba_test_raw_full) para LogLoss (se disponíveis)
            if proba_eval_test is not None:
                 try:
                     logloss = log_loss(y_test, y_proba_test_raw_full) if y_proba_test_raw_full is not None else None
                     logger.debug(f"    -> LogLoss(Raw)={logloss:.4f}" if logloss is not None else "    -> LogLoss(Raw)=N/A")
                 except Exception as e_logloss: logger.warning(f"  Aviso: Erro cálculo LogLoss: {e_logloss}")
                 try:
                     # AUC requer probs da classe positiva
                     auc = roc_auc_score(y_test, proba_eval_test) if len(np.unique(y_test)) > 1 else None
                     logger.info(f"    -> AUC({eval_src_test})={auc:.4f}" if auc is not None else f"    -> AUC({eval_src_test})=N/A")
                 except Exception as e_auc: logger.warning(f"  Aviso: Erro cálculo AUC: {e_auc}")
                 try:
                     # Brier requer probs da classe positiva
                     brier = brier_score_loss(y_test, proba_eval_test)
                     logger.info(f"    -> Brier({eval_src_test})={brier:.4f}" if brier is not None else f"    -> Brier({eval_src_test})=N/A")
                 except Exception as e_brier: logger.warning(f"  Aviso: Erro cálculo Brier: {e_brier}")
            metrics.update({'log_loss': logloss, 'roc_auc': auc, 'brier_score': brier})

            # Métricas de ROI/EV usando limiar EV otimizado
            roi_ev, bets_ev, profit_ev = None, 0, None
            if proba_eval_test is not None and X_test_odds is not None:
                roi_ev, bets_ev, profit_ev = calculate_metrics_with_ev(
                    y_test, proba_eval_test, current_optimal_ev_threshold, X_test_odds, odd_draw_col_name
                )
            else:
                logger.warning("    -> ROI/Profit EV (Teste) não calculado (sem probs ou odds de teste).")
            metrics.update({'roi': roi_ev, 'num_bets': bets_ev, 'profit': profit_ev})

            # Adiciona tamanhos dos datasets
            metrics.update({
                'train_set_size': len(y_train),
                'val_set_size': len(y_val),
                'test_set_size': len(y_test)
            })

            # --- Guarda resultado do modelo ---
            all_results.append({
                'model_name': model_name,
                'model_object': model_trained, # <<< O CLASSIFICADOR treinado
                'scaler': current_scaler,
                'calibrator': calibrator,
                'params': best_params,
                'metrics': metrics, # Dict completo com todas as métricas
                'optimal_f1_threshold': current_optimal_f1_threshold,
                'optimal_ev_threshold': current_optimal_ev_threshold
            })
            if progress_callback_stages: progress_callback_stages(i, f"Resultado adicionado.")
            logger.info(f"    ==> Resultado {model_name} adicionado.")

        except Exception as e_outer:
            logger.error(f"Erro GERAL no loop do modelo {model_name}: {e_outer}", exc_info=True)
            continue # Pula para o próximo modelo

        logger.info(f"  Tempo p/ {model_name} (total): {time.time() - start_time_model:.2f} seg.")
    # --- Fim do Loop for ---

    # --- Seleção e Salvamento ---
    # (Sem alterações necessárias aqui, apenas garante que 'model_object' é o classificador)
    if progress_callback_stages: progress_callback_stages(num_total_models, "Selecionando/Salvando...")
    end_time_total = time.time(); logger.info(f"--- Treino concluído ({end_time_total-start_time_total:.2f} seg) ---")
    if not all_results: logger.error("SELEÇÃO: Nenhum resultado válido."); return False
    logger.info(f"--- Processando {len(all_results)} resultados ---")
    try:
        results_df = pd.DataFrame(all_results)
        # Extrai métricas chave para o DataFrame de resultados para fácil visualização/ordenação
        results_df['f1_score_draw'] = results_df['metrics'].apply(lambda m: m.get('f1_score_draw', -1.0))
        results_df['precision_draw'] = results_df['metrics'].apply(lambda m: m.get('precision_draw', 0.0))
        results_df['recall_draw'] = results_df['metrics'].apply(lambda m: m.get('recall_draw', 0.0))
        results_df['roi'] = results_df['metrics'].apply(lambda m: m.get('roi', -np.inf))
        results_df['num_bets'] = results_df['metrics'].apply(lambda m: m.get('num_bets', 0))
        results_df['auc'] = results_df['metrics'].apply(lambda m: m.get('roc_auc', 0.0))
        results_df['brier'] = results_df['metrics'].apply(lambda m: m.get('brier_score', 1.0))
        # Converte colunas para numérico e trata NaNs/Infs para ordenação segura
        cols_num = ['f1_score_draw','precision_draw','recall_draw','roi','num_bets','auc','brier','optimal_f1_threshold','optimal_ev_threshold']
        for col in cols_num: results_df[col] = pd.to_numeric(results_df[col], errors='coerce')
        results_df['f1_score_draw'].fillna(-1.0, inplace=True); results_df['precision_draw'].fillna(0.0, inplace=True); results_df['recall_draw'].fillna(0.0, inplace=True);
        results_df['roi'].fillna(-np.inf, inplace=True); results_df['num_bets'].fillna(0, inplace=True);
        results_df['auc'].fillna(0.0, inplace=True); results_df['brier'].fillna(1.0, inplace=True);
        results_df['optimal_f1_threshold'].fillna(DEFAULT_F1_THRESHOLD, inplace=True); results_df['optimal_ev_threshold'].fillna(DEFAULT_EV_THRESHOLD, inplace=True)

        if results_df.empty: logger.error("SELEÇÃO: DF de resultados vazio após processamento."); return False

        # Exibe Comparativo
        logger.info("--- Comparativo Desempenho Modelos (Teste) ---")
        display_cols = ['model_name','f1_score_draw','precision_draw','recall_draw','auc','brier','roi','num_bets','optimal_f1_threshold','optimal_ev_threshold']
        display_cols_exist = [col for col in display_cols if col in results_df.columns]
        results_df_display = results_df[display_cols_exist].round({
            'f1_score_draw':4, 'precision_draw':4, 'recall_draw':4, 'auc':4, 'brier':4,
            'roi':2, 'optimal_f1_threshold':4, 'optimal_ev_threshold':3
        })
        if 'num_bets' in results_df_display.columns: results_df_display['num_bets'] = results_df_display['num_bets'].astype(int)
        try: logger.info("\n" + results_df_display.sort_values(by=BEST_MODEL_METRIC, ascending=False).to_markdown(index=False)) # Usa config
        except ImportError: logger.info("\n" + results_df_display.sort_values(by=BEST_MODEL_METRIC, ascending=False).to_string(index=False)) # Fallback sem tabulate
        logger.info("-" * 80)

        # Seleciona Melhor F1
        results_df_sorted_f1 = results_df.sort_values(by=BEST_MODEL_METRIC, ascending=False).reset_index(drop=True)
        if results_df_sorted_f1.empty: logger.error("SELEÇÃO: Nenhum modelo classificado por F1."); return False
        best_f1_result = results_df_sorted_f1.iloc[0].to_dict().copy()
        f1_val = best_f1_result.get(BEST_MODEL_METRIC, -1.0)
        f1_thr = best_f1_result.get('optimal_f1_threshold', DEFAULT_F1_THRESHOLD)
        logger.info(f"SELEÇÃO: Melhor F1 (Teste): {best_f1_result.get('model_name', 'ERRO')} (F1={f1_val:.4f} @ Thr={f1_thr:.4f})")

        # Seleciona Melhor ROI (ou 2º F1 como fallback)
        best_roi_result = None
        logger.info("--- Ranking por ROI (Teste) ---")
        results_df_sorted_roi = results_df.sort_values(by=BEST_MODEL_METRIC_ROI, ascending=False).reset_index(drop=True)
        roi_rank_cols = ['model_name', 'roi', 'num_bets', 'f1_score_draw', 'optimal_ev_threshold']
        roi_rank_cols_exist = [col for col in roi_rank_cols if col in results_df_sorted_roi.columns]
        try: logger.info("\n" + results_df_sorted_roi[roi_rank_cols_exist].round({'roi': 2, 'f1_score_draw': 4, 'optimal_ev_threshold': 3}).to_markdown(index=False))
        except ImportError: logger.info("\n" + results_df_sorted_roi[roi_rank_cols_exist].round({'roi': 2, 'f1_score_draw': 4, 'optimal_ev_threshold': 3}).to_string(index=False))
        logger.info("-" * 30)
        if not results_df_sorted_roi.empty:
             # Itera para encontrar o primeiro com ROI válido (não -inf)
             for idx, row in results_df_sorted_roi.iterrows():
                  current_roi = row[BEST_MODEL_METRIC_ROI]
                  current_name = row['model_name']
                  ev_thr_roi = row.get('optimal_ev_threshold', DEFAULT_EV_THRESHOLD)
                  if isinstance(current_roi, (int, float, np.number)) and current_roi > -np.inf:
                       best_roi_result = row.to_dict().copy()
                       logger.info(f"SELEÇÃO: Melhor ROI Válido (Teste): {current_name} (ROI={current_roi:.2f}% @ EV Thr={ev_thr_roi:.3f})")
                       break # Para no primeiro ROI válido
             if best_roi_result is None: logger.warning("SELEÇÃO: Nenhum ROI válido (> -inf) encontrado.")
        else: logger.warning("SELEÇÃO: Ranking ROI vazio.")

        # Define qual modelo salvar para o slot ROI
        model_to_save_f1 = best_f1_result
        model_to_save_roi = None
        if best_roi_result:
            # Se o melhor F1 e o melhor ROI são o mesmo modelo
            if best_f1_result.get('model_name') == best_roi_result.get('model_name'):
                 # Tenta pegar o segundo melhor F1
                 if len(results_df_sorted_f1) > 1:
                     model_to_save_roi = results_df_sorted_f1.iloc[1].to_dict().copy()
                     logger.info(f"  -> Melhor F1 e ROI são o mesmo. Usando 2º Melhor F1 ({model_to_save_roi.get('model_name')}) para slot ROI.")
                 else: # Só tem um modelo no total
                     model_to_save_roi = best_f1_result.copy()
                     logger.warning("  -> Apenas 1 modelo treinado. Usando Melhor F1 para ambos os slots (F1 e ROI).")
            else: # Melhores F1 e ROI são modelos diferentes
                 model_to_save_roi = best_roi_result.copy()
                 logger.info(f"  -> Usando Melhor ROI ({model_to_save_roi.get('model_name')}) para slot ROI.")
        else: # Nenhum ROI válido encontrado
            model_to_save_roi = best_f1_result.copy() # Usa o melhor F1 como fallback
            logger.warning(f"  -> Nenhum ROI válido encontrado. Usando Melhor F1 ({model_to_save_roi.get('model_name')}) para slot ROI.")

        # Salva os modelos selecionados
        logger.info(f"Salvando Melhor F1 ({model_to_save_f1.get('model_name', 'ERRO')})...")
        _save_model_object(model_to_save_f1, feature_names, BEST_F1_MODEL_SAVE_PATH)

        if model_to_save_roi:
            logger.info(f"Salvando Melhor ROI/2nd F1 ({model_to_save_roi.get('model_name', 'ERRO')})...")
            _save_model_object(model_to_save_roi, feature_names, BEST_ROI_MODEL_SAVE_PATH)
        else:
            # Isso não deveria acontecer por causa dos fallbacks, mas por segurança:
            logger.error("ERRO CRÍTICO SALVAR: Objeto model_to_save_roi é None inesperadamente.")
            return False

    except Exception as e_select_save:
        logger.error(f"Erro GERAL durante seleção/salvamento final: {e_select_save}", exc_info=True)
        return False

    logger.info("--- Processo Completo ---"); return True


# --- Função _save_model_object (Sem alterações, já salva o necessário) ---
def _save_model_object(model_result_dict: Dict, feature_names: List[str], file_path: str) -> None:
    """Salva modelo (classificador), scaler, calibrador, params, métricas, limiares."""
    if not isinstance(model_result_dict, dict):
        logger.error(f"Salvar: Dados inválidos fornecidos para {os.path.basename(file_path)}.")
        return
    try:
        # Pega o CLASSIFICADOR do dicionário (não o pipeline inteiro)
        model_to_save = model_result_dict.get('model_object')
        if model_to_save is None:
            logger.error(f"Salvar: Objeto 'model_object' (classificador) ausente no dicionário para {os.path.basename(file_path)}.")
            return

        # Monta o objeto a ser salvo
        save_obj = {
            'model': model_to_save, # O classificador treinado
            'scaler': model_result_dict.get('scaler'),
            'calibrator': model_result_dict.get('calibrator'),
            'feature_names': feature_names,
            'best_params': model_result_dict.get('params'), # Parâmetros limpos (sem prefixo)
            'eval_metrics': model_result_dict.get('metrics'), # Dicionário completo de métricas
            'optimal_ev_threshold': model_result_dict.get('optimal_ev_threshold', DEFAULT_EV_THRESHOLD),
            'optimal_f1_threshold': model_result_dict.get('optimal_f1_threshold', DEFAULT_F1_THRESHOLD),
            'save_timestamp': datetime.datetime.now().isoformat(),
            'model_class_name': model_to_save.__class__.__name__ # Nome da classe do classificador
        }
        # Salva o objeto
        joblib.dump(save_obj, file_path)
        logger.info(f"  -> Objeto salvo: '{save_obj['model_class_name']}' (F1 Thr={save_obj['optimal_f1_threshold']:.4f}, EV Thr={save_obj['optimal_ev_threshold']:.3f}) em {os.path.basename(file_path)}.")

    except Exception as e:
        logger.error(f"  -> Erro GRAVE ao salvar objeto em {file_path}: {e}", exc_info=True)


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