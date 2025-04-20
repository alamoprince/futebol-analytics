# --- src/model_trainer.py ---
# Código Completo - VERSÃO FINAL CORRIGIDA (v4 - Remoção do Append Duplicado)

import pandas as pd
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from logger_config import setup_logger

logger = setup_logger("ModelTrainerApp")

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
except ImportError:
    logger.warning("AVISO: LightGBM não instalado.")
    lgb = None; LGBMClassifier = None
# Importar métricas e numpy
from sklearn.metrics import (accuracy_score, classification_report, log_loss,
                             precision_score, recall_score, f1_score, # Adicionar f1_score se não estiver
                             roc_auc_score, confusion_matrix, brier_score_loss,
                             precision_recall_curve) # <-- IMPORTANTE: para achar limiar
import numpy as np 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib, os, datetime, numpy as np, traceback
try:
    from config import (RANDOM_STATE, TEST_SIZE, MODEL_CONFIG, CLASS_NAMES,
                        BEST_F1_MODEL_SAVE_PATH, BEST_ROI_MODEL_SAVE_PATH,
                        CROSS_VALIDATION_SPLITS, N_JOBS_GRIDSEARCH, FEATURE_COLUMNS,
                        ODDS_COLS, BEST_MODEL_METRIC, DEFAULT_EV_THRESHOLD)
    from typing import Any, Optional, Dict, Tuple, List, Callable
except ImportError as e: logger.warning(f"Erro crítico import config/typing: {e}"); raise

# --- Funções Auxiliares (roi, calculate_roi_with_threshold, calculate_metrics_with_ev, scale_features) ---
# (Colar as definições corretas dessas funções aqui, como nas respostas anteriores)
def roi(y_test: pd.Series, y_pred: np.ndarray, X_test_odds_aligned: Optional[pd.DataFrame], odd_draw_col_name: str) -> Optional[float]:
    # ... (código completo da função roi) ...
    if X_test_odds_aligned is None or odd_draw_col_name not in X_test_odds_aligned.columns: return None
    try: common_index=y_test.index.intersection(X_test_odds_aligned.index);
    except AttributeError: logger.error("Error accessing index in roi function."); return None
    if len(common_index) != len(y_test): return None
    y_test_common=y_test.loc[common_index];
    try: y_pred_series=pd.Series(y_pred, index=y_test.index); y_pred_common=y_pred_series.loc[common_index];
    except (ValueError,AttributeError): logger.error("Error aligning y_pred in roi."); return None
    predicted_draws_indices=common_index[y_pred_common == 1]; num_bets=len(predicted_draws_indices);
    if num_bets == 0: return 0.0
    actuals=y_test_common.loc[predicted_draws_indices]; odds=pd.to_numeric(X_test_odds_aligned.loc[predicted_draws_indices, odd_draw_col_name], errors='coerce')
    profit=0; valid_bets=0;
    for idx in predicted_draws_indices:
        odd_d=odds.loc[idx];
        if pd.notna(odd_d)and odd_d>0: profit+=(odd_d-1)if actuals.loc[idx]==1 else-1; valid_bets+=1;
    if valid_bets == 0: return 0.0
    return(profit/valid_bets)*100

def calculate_roi_with_threshold(y_true: pd.Series, y_proba: np.ndarray, threshold: float, odds_data: Optional[pd.DataFrame], odd_col_name: str) -> Tuple[Optional[float], int, Optional[float]]:
    # ... (código completo da função calculate_roi_with_threshold) ...
    profit=None;roi_value=None;num_bets=0;profit_calc=0;valid_bets_count=0;
    if odds_data is None or odd_col_name not in odds_data.columns:return roi_value,num_bets,profit
    try:
        common_index=y_true.index.intersection(odds_data.index);
        if len(common_index)!=len(y_true):logger.warning("ROI Thr: Índices não batem.");return roi_value,num_bets,profit;
        y_true_common=y_true.loc[common_index];odds_common=pd.to_numeric(odds_data.loc[common_index,odd_col_name],errors='coerce');
        try:y_proba_common=pd.Series(y_proba,index=y_true.index).loc[common_index];
        except ValueError:logger.error("ROI Thr: Erro alinhar y_proba.");return None,0,None
        bet_indices=common_index[y_proba_common>threshold];num_bets=len(bet_indices);
        if num_bets==0:return 0.0,num_bets,0.0
        actuals=y_true_common.loc[bet_indices];odds_selected=odds_common.loc[bet_indices];
        for idx in bet_indices:
            odd_d=odds_selected.loc[idx];
            if pd.notna(odd_d)and odd_d>0:profit_calc+=(odd_d-1)if actuals.loc[idx]==1 else-1;
            valid_bets_count+=1;
        profit=profit_calc;
        if valid_bets_count>0:
            roi_value=(profit/valid_bets_count)*100;
        else:roi_value=0.0;
        return roi_value,num_bets,profit
    except Exception as e:logger.error(f"ROI Thr: Erro - {e}",exc_info=True);return None,0,None

def calculate_metrics_with_ev(y_true: pd.Series, y_proba_calibrated: np.ndarray, ev_threshold: float, odds_data: Optional[pd.DataFrame], odd_col_name: str) -> Tuple[Optional[float], int, Optional[float]]:
    # ... (código completo da função calculate_metrics_with_ev) ...
    profit=None;roi_value=None;num_bets=0;profit_calc=0;valid_bets_count=0;
    if odds_data is None or odd_col_name not in odds_data.columns:logger.warning(f"EV Metr: Odds ('{odd_col_name}') ausentes.");return roi_value,num_bets,profit
    try:
        common_index=y_true.index.intersection(odds_data.index);
        if len(common_index)!=len(y_true):logger.warning("EV Metr: Índices não batem.");return roi_value,num_bets,profit;
        y_true_common=y_true.loc[common_index];odds_common=pd.to_numeric(odds_data.loc[common_index,odd_col_name],errors='coerce');y_proba_common=pd.Series(y_proba_calibrated,index=y_true.index).loc[common_index];
        ev=(y_proba_common*(odds_common-1))-((1-y_proba_common)*1);# Calcula EV
        bet_indices=common_index[ev>ev_threshold];num_bets=len(bet_indices);# Filtra por EV
        if num_bets==0:return 0.0,num_bets,0.0
        actuals=y_true_common.loc[bet_indices];odds_selected=odds_common.loc[bet_indices];
        for idx in bet_indices:
            odd_d=odds_selected.loc[idx];
            if pd.notna(odd_d)and odd_d>0:
                profit_calc+=(odd_d-1)if actuals.loc[idx]==1 else-1;valid_bets_count+=1;
        profit=profit_calc;
        if valid_bets_count>0:
            roi_value=(profit/valid_bets_count)*100;
        else:
            roi_value=0.0;
        logger.info(f"    -> Métricas EV (Th={ev_threshold:.3f}): ROI={roi_value if roi_value is not None else 'N/A':.2f}%, Bets={num_bets}, Profit={profit if profit is not None else 'N/A'}")
        return roi_value,num_bets,profit
    except Exception as e:logger.error(f"EV Metr: Erro - {e}",exc_info=True);return None,0,None

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame, scaler_type='standard') -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
    # ... (código completo da função scale_features) ...
    X_train=X_train.copy();X_test=X_test.copy();
    try:X_train=X_train.astype(float);X_test=X_test.astype(float)
    except ValueError as e:logger.error(f"Erro converter p/ float scaling:{e}");raise
    if scaler_type=='minmax':scaler=MinMaxScaler()
    else:scaler=StandardScaler()
    logger.info(f"  Aplicando {scaler.__class__.__name__}...");
    try:X_train_scaled=scaler.fit_transform(X_train);X_test_scaled=scaler.transform(X_test)
    except Exception as e:logger.error(f"Erro {scaler.__class__.__name__}.fit/transform:{e}",exc_info=True);raise
    X_train_scaled_df=pd.DataFrame(X_train_scaled,index=X_train.index,columns=X_train.columns);X_test_scaled_df=pd.DataFrame(X_test_scaled,index=X_test.index,columns=X_test.columns);
    logger.info("  Scaling concluído.");return X_train_scaled_df,X_test_scaled_df,scaler

# --- Função Principal de Treinamento (REVISADA FINAL) ---
def train_evaluate_and_save_best_models(
    X: pd.DataFrame, y: pd.Series,
    X_test_with_odds: Optional[pd.DataFrame] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    scaler_type: str = 'standard',
    odd_draw_col_name: str = ODDS_COLS.get('draw', 'Odd_D_FT'),
    calibration_method: str = 'isotonic',
    optimize_ev_threshold: bool = True,
    default_ev_threshold: float = DEFAULT_EV_THRESHOLD,
    # NOVO PARÂMETRO: Controla se otimiza limiar F1
    optimize_f1_threshold: bool = True
    ) -> bool:
    """
    Treina, CALIBRA, opcionalmente otimiza LIMIAR F1 e/ou LIMIAR de EV, avalia modelos.
    Salva 2 melhores baseados nas métricas do CONJUNTO DE TESTE (usando limiar 0.5 ou otimizado).
    """
    # ... (Verificações iniciais X, y, MODEL_CONFIG, available_models) ...
    if X is None or y is None: logger.error("Dados X ou y None."); return False
    if not MODEL_CONFIG: logger.error("MODEL_CONFIG vazio."); return False
    available_models = {name: config for name, config in MODEL_CONFIG.items() if not (name=='LGBMClassifier' and lgb is None)}
    if not available_models: logger.error("Nenhum modelo válido."); return False

    feature_names = list(X.columns); all_results = []
    logger.info(f"--- Treinando/Calibrando {len(available_models)} Modelos (Otimizar F1 Thr: {optimize_f1_threshold}, Otimizar EV Thr: {optimize_ev_threshold}) ---")
    start_time_total = time.time()

    # --- Divisão Tripla e Preparo Odds (com logs) ---
    logger.info("Dividindo dados..."); val_size=0.20; test_size_final=TEST_SIZE; train_val_size_temp = 1.0 - test_size_final;
    if train_val_size_temp <= 0: logger.error("TEST_SIZE inválido."); return False;
    val_size_relative = val_size / train_val_size_temp
    try:
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size_final, random_state=RANDOM_STATE, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size_relative, random_state=RANDOM_STATE, stratify=y_train_val)
        logger.info(f"Split: Treino={len(X_train)}, Val={len(X_val)}, Teste={len(X_test)}")
        # Alinhamento Odds Val/Teste
        X_val_odds=None; X_test_odds=None;
        if X_test_with_odds is not None and not X_test_with_odds.empty and odd_draw_col_name in X_test_with_odds.columns:
             common_val=X_val.index.intersection(X_test_with_odds.index);
             if len(common_val)>0: X_val_odds=X_test_with_odds.loc[common_val,[odd_draw_col_name]].copy()
             common_test=X_test.index.intersection(X_test_with_odds.index);
             if len(common_test)>0: X_test_odds=X_test_with_odds.loc[common_test,[odd_draw_col_name]].copy()
             logger.info(f"Odds alinhadas: Val={X_val_odds is not None}, Teste={X_test_odds is not None}")
        else: logger.warning(f"Não foi possível alinhar odds (X_test_with_odds vazio ou coluna '{odd_draw_col_name}' ausente). ROI/EV não serão calculados.")
    except Exception as e: logger.error(f"Erro divisão/alinhar dados: {e}", exc_info=True); return False

    # --- Loop pelos modelos ---
    for i, (model_name, config) in enumerate(available_models.items()):
        status_text=f"Modelo {i+1}/{len(available_models)}:{model_name}"; logger.info(f"\n--- {status_text} ---");
        if progress_callback: progress_callback(i, len(available_models), f"Iniciando {model_name}...");
        start_time_model = time.time(); model_trained = None; best_params = None; current_scaler = None; calibrator = None;

        try:
            model_class = eval(model_name)
            model_kwargs=config.get('model_kwargs',{}); 
            needs_scaling=config.get('needs_scaling',False);
            use_bayes_opt = SKOPT_AVAILABLE and 'search_spaces' in config
            param_space = config.get('search_spaces') if use_bayes_opt else config.get('param_grid')
            X_train_m, X_val_m, X_test_m = X_train.copy(), X_val.copy(), X_test.copy();

            # Scaling
            if needs_scaling:
                logger.info(f"  Scaling p/ {model_name} (scaler={scaler_type})...");
                try:
                    if scaler_type == 'minmax': scaler_instance = MinMaxScaler()
                    else: scaler_instance = StandardScaler()
                    # Ajusta APENAS no TREINO
                    X_train_m_float = X_train_m.astype(float) # Garante float
                    scaler_instance.fit(X_train_m_float)
                    # Aplica (transform) nos TRÊS conjuntos
                    X_train_scaled_np = scaler_instance.transform(X_train_m_float)
                    X_val_scaled_np = scaler_instance.transform(X_val_m.astype(float))
                    X_test_scaled_np = scaler_instance.transform(X_test_m.astype(float))
                    # Recria DataFrames
                    X_train_m = pd.DataFrame(X_train_scaled_np, index=X_train_m.index, columns=feature_names)
                    X_val_m = pd.DataFrame(X_val_scaled_np, index=X_val_m.index, columns=feature_names)
                    X_test_m = pd.DataFrame(X_test_scaled_np, index=X_test_m.index, columns=feature_names)
                    current_scaler = scaler_instance
                    logger.info(f"    -> Scaling OK para {model_name}.")
                except Exception as scale_err: logger.error(f"  ERRO scaling {model_name}: {scale_err}", exc_info=True); continue

            # Treino (Grid ou Padrão)
            if param_space:
                logger.info(f"  Iniciando busca de Hiperparâmetros (Método: {'BayesSearchCV' if use_bayes_opt else 'GridSearchCV'})...")
                try:
                    if use_bayes_opt:
                        # Configura BayesSearchCV
                        search_cv = BayesSearchCV(
                            estimator=model_class(**model_kwargs),
                            search_spaces=param_space,
                            n_iter=BAYESIAN_OPT_N_ITER, # Número de iterações bayesianas
                            cv=CROSS_VALIDATION_SPLITS,
                            n_jobs=N_JOBS_GRIDSEARCH,
                            scoring='f1', # Métrica a ser otimizada (maximizar F1)
                            random_state=RANDOM_STATE,
                            verbose=0, # Pode aumentar para ver progresso do BayesOpt
                            #optimizer_kwargs={'base_estimator': 'GP'} # Pode experimentar 'GP', 'RF', 'ET'
                        )
                        logger.info(f"    Configurado BayesSearchCV (n_iter={BAYESIAN_OPT_N_ITER}, CV={CROSS_VALIDATION_SPLITS}, Scoring='f1')")
                    else: # Fallback para GridSearchCV (ou para modelos como GNB)
                        search_cv = GridSearchCV(
                            estimator=model_class(**model_kwargs),
                            param_grid=param_space, # Aqui é param_grid
                            cv=CROSS_VALIDATION_SPLITS,
                            n_jobs=N_JOBS_GRIDSEARCH,
                            scoring='f1',
                            verbose=0, error_score='raise'
                        )
                        logger.info(f"    Configurado GridSearchCV (CV={CROSS_VALIDATION_SPLITS}, Scoring='f1')")

                    if progress_callback: progress_callback(i, len(available_models), f"Ajustando {model_name} ({'Bayes' if use_bayes_opt else 'Grid'})...");
                    search_cv.fit(X_train_m, y_train); # Executa a busca
                    model_trained = search_cv.best_estimator_;
                    best_params = search_cv.best_params_; # BayesSearchCV também tem .best_params_
                    # BayesSearchCV retorna o score *negativo* se otimiza, ou positivo se maximiza.
                    # Como scikit-learn > 0.18 maximiza score, usamos best_score_ diretamente.
                    best_cv_score = search_cv.best_score_
                    logger.info(f"    -> Busca OK. Melhor CV F1: {best_cv_score:.4f}. Params: {best_params}");

                except Exception as e: logger.error(f"    Erro {'BayesSearchCV' if use_bayes_opt else 'GridSearchCV'}.fit: {e}", exc_info=True); model_trained = None; logger.warning("    -> Tentando fallback...");
            else: logger.info(f"  Sem grid/space. Treinando c/ params padrão.");

            # --- Calibração (Usando Conjunto de Validação) ---
            logger.info(f"  Calibrando probs com {calibration_method} (usando Val)...")
            y_proba_val_raw_draw = None # Probabilidades brutas no conjunto de VALIDAÇÃO
            y_proba_val_calib = None # Probabilidades calibradas no conjunto de VALIDAÇÃO
            if hasattr(model_trained, "predict_proba"):
                 try:
                     y_proba_val_raw_full = model_trained.predict_proba(X_val_m);
                     if y_proba_val_raw_full.shape[1] > 1:
                          y_proba_val_raw_draw = y_proba_val_raw_full[:, 1] # Pega prob da classe 1 (Empate)
                          if calibration_method == 'isotonic': calibrator = IsotonicRegression(out_of_bounds='clip')
                          else: logger.warning("Calibração não-isotônica não implementada, usando Isotonic."); calibrator = IsotonicRegression(out_of_bounds='clip')
                          calibrator.fit(y_proba_val_raw_draw, y_val) # Ajusta no conjunto de validação
                          y_proba_val_calib = calibrator.predict(y_proba_val_raw_draw) # Calcula probs calibradas NA VALIDAÇÃO
                          logger.info("  -> Calibrador treinado.")
                     else: logger.warning(f"    Predict_proba Val shape {y_proba_val_raw_full.shape}. Calib. pulada.")
                 except Exception as e_calib: logger.error(f"  Erro durante calibração: {e_calib}", exc_info=True); calibrator = None;
            else: logger.warning(f"  {model_name} sem predict_proba. Calib. pulada.")

            # --- Otimização Limiar F1 (Usando Conjunto de Validação) ---
            optimal_f1_threshold = 0.5 # Default
            best_val_f1 = -1.0
            if optimize_f1_threshold and y_proba_val_calib is not None: # Usa probs calibradas se disponíveis
                logger.info("  Otimizando limiar F1 (usando Val com probs calibradas)...")
                try:
                    precision, recall, thresholds = precision_recall_curve(y_val, y_proba_val_calib)
                    # Calcula F1 para cada limiar (evita divisão por zero)
                    f1_scores = (2 * precision * recall) / (precision + recall + 1e-9)
                    # Acha o índice do maior F1 (thresholds tem um elemento a menos que f1_scores)
                    best_f1_idx = np.argmax(f1_scores[:-1]) # Ignora último F1 que pode ser 0
                    optimal_f1_threshold = thresholds[best_f1_idx]
                    best_val_f1 = f1_scores[best_f1_idx]
                    logger.info(f"    Limiar F1 ótimo(Val): {optimal_f1_threshold:.4f} (F1={best_val_f1:.4f})")
                except Exception as e_f1_opt:
                    logger.error(f"  Erro otimizar limiar F1: {e_f1_opt}", exc_info=True)
                    optimal_f1_threshold = 0.5 # Retorna ao default em caso de erro
            elif optimize_f1_threshold and y_proba_val_raw_draw is not None: # Fallback para probs brutas se calibração falhou
                 logger.warning("  Otimizando limiar F1 (usando Val com probs BRUTAS - calibração falhou?)...")
                 try:
                      precision, recall, thresholds = precision_recall_curve(y_val, y_proba_val_raw_draw)
                      f1_scores = (2 * precision * recall) / (precision + recall + 1e-9)
                      best_f1_idx = np.argmax(f1_scores[:-1])
                      optimal_f1_threshold = thresholds[best_f1_idx]
                      best_val_f1 = f1_scores[best_f1_idx]
                      logger.info(f"    Limiar F1 ótimo(Val - Bruta): {optimal_f1_threshold:.4f} (F1={best_val_f1:.4f})")
                 except Exception as e_f1_opt: logger.error(f"  Erro otimizar limiar F1 (Bruta): {e_f1_opt}"); optimal_f1_threshold = 0.5
            elif optimize_f1_threshold:
                 logger.warning("  Otimização Limiar F1 pulada (sem probabilidades de validação).")


            # --- Otimização Limiar EV (Usando Conjunto de Validação) ---
            optimal_ev_threshold = default_ev_threshold; best_val_roi_ev = -np.inf;
            if optimize_ev_threshold and y_proba_val_calib is not None and X_val_odds is not None: # Usa probs CALIBRADAS para EV
                logger.info("  Otimizando limiar EV (usando Val com probs calibradas)...");
                try:
                    ev_ths = np.linspace(0.0, 0.20, 21); # Ou outro range/passo
                    for ev_th in ev_ths:
                        val_roi, _, _ = calculate_metrics_with_ev(y_val, y_proba_val_calib, ev_th, X_val_odds, odd_draw_col_name)
                        if val_roi is not None and val_roi > best_val_roi_ev: best_val_roi_ev=val_roi; optimal_ev_threshold=ev_th;
                    if best_val_roi_ev > -np.inf: logger.info(f"    Limiar EV ótimo(Val): {optimal_ev_threshold:.3f} (ROI={best_val_roi_ev:.2f}%)")
                    else: logger.warning("    ROI Val inválido p/ otimizar EV."); optimal_ev_threshold = default_ev_threshold;
                except Exception as e: logger.error(f"  Erro otimizar EV: {e}", exc_info=True); optimal_ev_threshold = default_ev_threshold;
            elif optimize_ev_threshold:
                logger.warning(f"  Otimização EV pulada (reqs ausentes: probs calib Val ou odds Val).");
            else: logger.info(f"  Otimização EV desabilitada. Usando default {optimal_ev_threshold:.3f}.")


            # --- Avaliação Final no Teste ---
            logger.info(f"  Avaliando no Teste...")
            if progress_callback: progress_callback(i, len(available_models), f"Avaliando {model_name}...");
            metrics = {}; # Reinicia métricas para este modelo

            y_proba_test_raw_full = None
            y_proba_test_raw_draw = None
            y_proba_test_calib = None

            if hasattr(model_trained, "predict_proba"):
                 try:
                      y_proba_test_raw_full = model_trained.predict_proba(X_test_m)
                      if y_proba_test_raw_full.shape[1] > 1:
                           y_proba_test_raw_draw = y_proba_test_raw_full[:, 1]
                           # Aplica calibrador treinado (se existir) nas probs do teste
                           if calibrator:
                                y_proba_test_calib = calibrator.predict(y_proba_test_raw_draw)
                           else:
                                y_proba_test_calib = y_proba_test_raw_draw # Usa bruta se não houver calibrador
                      else: logger.warning(f"  Predict_proba Teste shape {y_proba_test_raw_full.shape}. Métricas baseadas em proba podem falhar.")
                 except Exception as e_proba_test: logger.error(f"  Erro predict_proba no teste: {e_proba_test}");

            # --> Métrica 1: Base @ Limiar 0.5 (Usando predict direto)
            try:
                 y_pred_test_thr05 = model_trained.predict(X_test_m) # Previsão de classe direta
                 acc05 = accuracy_score(y_test, y_pred_test_thr05);
                 prec05 = precision_score(y_test, y_pred_test_thr05, pos_label=1, zero_division=0)
                 rec05 = recall_score(y_test, y_pred_test_thr05, pos_label=1, zero_division=0)
                 f1_05 = f1_score(y_test, y_pred_test_thr05, pos_label=1, zero_division=0)
                 matrix05 = confusion_matrix(y_test, y_pred_test_thr05).tolist();
                 metrics.update({'accuracy_thr05': acc05, 'precision_draw_thr05': prec05, 'recall_draw_thr05': rec05, 'f1_score_draw_thr05': f1_05, 'confusion_matrix_thr05': matrix05});
                 logger.info(f"    -> Métricas @ Limiar 0.5: Acc={acc05:.4f}, F1={f1_05:.4f}, P={prec05:.4f}, R={rec05:.4f}")
            except Exception as e_m05: logger.error(f"    Erro ao calcular métricas @ 0.5: {e_m05}")

            # --> Métrica 2: Base @ Limiar F1 Otimizado (Usa probs calibradas/brutas do teste)
            #     Calcula apenas se otimização F1 foi feita e temos probs no teste
            optimal_f1_threshold_to_use = 0.5 # Default se não otimizou
            if optimize_f1_threshold and (y_proba_test_calib is not None or y_proba_test_raw_draw is not None):
                 optimal_f1_threshold_to_use = optimal_f1_threshold # Usa o limiar encontrado na validação
                 proba_to_use_f1 = y_proba_test_calib if y_proba_test_calib is not None else y_proba_test_raw_draw
                 try:
                      y_pred_test_thrF1 = (proba_to_use_f1 >= optimal_f1_threshold_to_use).astype(int)
                      accF1 = accuracy_score(y_test, y_pred_test_thrF1);
                      precF1 = precision_score(y_test, y_pred_test_thrF1, pos_label=1, zero_division=0)
                      recF1 = recall_score(y_test, y_pred_test_thrF1, pos_label=1, zero_division=0)
                      f1_F1 = f1_score(y_test, y_pred_test_thrF1, pos_label=1, zero_division=0)
                      matrixF1 = confusion_matrix(y_test, y_pred_test_thrF1).tolist();
                      metrics.update({'accuracy_thrF1': accF1, 'precision_draw_thrF1': precF1, 'recall_draw_thrF1': recF1, 'f1_score_draw': f1_F1, 'confusion_matrix_thrF1': matrixF1, 'optimal_f1_threshold': optimal_f1_threshold_to_use}); # Salva o F1 com limiar otimizado como 'f1_score_draw'
                      logger.info(f"    -> Métricas @ Limiar F1 ({optimal_f1_threshold_to_use:.4f}): Acc={accF1:.4f}, F1={f1_F1:.4f}, P={precF1:.4f}, R={recF1:.4f}")
                 except Exception as e_mF1: logger.error(f"    Erro ao calcular métricas @ Limiar F1: {e_mF1}")
            else:
                 # Se não otimizou F1 ou não tem probs, usa F1 de 0.5 como F1 principal
                 metrics['f1_score_draw'] = metrics.get('f1_score_draw_thr05') # Usa F1@0.5 como default
                 metrics['optimal_f1_threshold'] = 0.5
                 logger.info(f"    -> Usando F1 @ 0.5 como métrica principal 'f1_score_draw' (Otimização F1 não realizada/possível).")


            # --> Métrica 3: Métricas baseadas em Probabilidade (AUC, Brier, LogLoss - Usa probs calibradas/brutas do teste)
            logloss=None; auc=None; brier=None;
            proba_to_use_calib = y_proba_test_calib if y_proba_test_calib is not None else y_proba_test_raw_draw
            if proba_to_use_calib is not None:
                try: logloss = log_loss(y_test, y_proba_test_raw_full) if y_proba_test_raw_full is not None else None; logger.info(f"    -> LogLoss(Raw)={logloss:.4f}" if logloss else "LogLoss não calc.")
                except Exception as e: logger.warning(f" Erro LogLoss:{e}")
                try:
                    if len(np.unique(y_test)) > 1: auc = roc_auc_score(y_test, proba_to_use_calib); logger.info(f"    -> AUC(Calib/Bruta)={auc:.4f}" if auc else "AUC não calc.")
                    else: logger.warning(" AUC não calc (1 classe y_test)")
                except Exception as e: logger.warning(f" Erro AUC:{e}")
                try: brier = brier_score_loss(y_test, proba_to_use_calib); logger.info(f"    -> Brier(Calib/Bruta)={brier:.4f}" if brier else "Brier não calc.")
                except Exception as e: logger.warning(f" Erro Brier:{e}")
            metrics.update({'log_loss': logloss, 'roc_auc': auc, 'brier_score': brier})


            # --> Métrica 4: ROI/Profit com Estratégia EV (Usa probs calibradas/brutas do teste e Limiar EV otimizado)
            roi_ev, bets_ev, profit_ev = None, 0, None;
            if proba_to_use_calib is not None and X_test_odds is not None:
                 roi_ev, bets_ev, profit_ev = calculate_metrics_with_ev(y_test, proba_to_use_calib, optimal_ev_threshold, X_test_odds, odd_draw_col_name); # Log já está dentro da função
            else: logger.warning("    -> ROI/Profit EV (Teste) não calculado (sem probs ou odds teste).");
            metrics.update({'roi': roi_ev, 'num_bets': bets_ev, 'profit': profit_ev, 'optimal_ev_threshold': optimal_ev_threshold}); # Salva ROI e limiar EV


            # Adiciona tamanhos dos conjuntos
            metrics.update({'train_set_size':len(y_train), 'val_set_size':len(y_val), 'test_set_size':len(y_test)});

            # Guarda o resultado completo
            all_results.append({
                'model_name': model_name,
                'model_object': model_trained,
                'scaler': current_scaler,
                'calibrator': calibrator,
                'params': best_params,
                'metrics': metrics, # Dict completo com todas as métricas calculadas
                'optimal_f1_threshold': metrics.get('optimal_f1_threshold', 0.5), # Guarda o limiar F1 usado
                'optimal_ev_threshold': metrics.get('optimal_ev_threshold', default_ev_threshold) # Guarda o limiar EV usado
            })
            logger.info(f"    ==> Resultado {model_name} adicionado.")

        except Exception as e_outer:
            logger.error(f"Erro GERAL no loop do modelo {model_name}: {e_outer}", exc_info=True)
            # Tenta continuar para o próximo modelo
            continue

        logger.info(f"  Tempo p/ {model_name} (total): {time.time() - start_time_model:.2f} seg.")
    # --- Fim do Loop for ---

    # --- Seleção e Salvamento (Baseado em F1@LimiarOtimizado ou F1@0.5) ---
    if progress_callback: progress_callback(len(available_models),len(available_models),"Selecionando/Salvando...")
    end_time_total=time.time(); logger.info(f"--- Treino concluído ({end_time_total-start_time_total:.2f} seg) ---")
    if not all_results: logger.error("SELEÇÃO: Nenhum resultado válido para selecionar."); return False

    logger.info(f"--- Processando {len(all_results)} resultados para seleção ---")
    try:
        results_df = pd.DataFrame(all_results)
        # Extrai métricas principais para ordenação e log
        results_df['f1_score_draw'] = results_df['metrics'].apply(lambda m: m.get('f1_score_draw', -1.0) if isinstance(m,dict) else -1.0) # F1 @ Thr F1-Otimizado ou 0.5
        results_df['f1_score_draw_thr05'] = results_df['metrics'].apply(lambda m: m.get('f1_score_draw_thr05', -1.0) if isinstance(m,dict) else -1.0) # F1 @ Thr 0.5
        results_df['precision_draw'] = results_df['metrics'].apply(lambda m: m.get('precision_draw_thrF1', 0.0) if isinstance(m,dict) and 'precision_draw_thrF1' in m else m.get('precision_draw_thr05', 0.0) if isinstance(m,dict) else 0.0) # Precisão no limiar F1 ou 0.5
        results_df['recall_draw'] = results_df['metrics'].apply(lambda m: m.get('recall_draw_thrF1', 0.0) if isinstance(m,dict) and 'recall_draw_thrF1' in m else m.get('recall_draw_thr05', 0.0) if isinstance(m,dict) else 0.0) # Recall no limiar F1 ou 0.5
        results_df['roi'] = results_df['metrics'].apply(lambda m: m.get('roi', -np.inf) if isinstance(m,dict) else -np.inf) # ROI @ Thr EV-Otimizado
        results_df['num_bets'] = results_df['metrics'].apply(lambda m: m.get('num_bets', 0) if isinstance(m,dict) else 0) # Bets @ Thr EV-Otimizado
        results_df['auc'] = results_df['metrics'].apply(lambda m: m.get('roc_auc', 0.0) if isinstance(m,dict) else 0.0) # AUC
        results_df['brier'] = results_df['metrics'].apply(lambda m: m.get('brier_score', 1.0) if isinstance(m,dict) else 1.0) # Brier

        # Converte para numérico com tratamento de erro
        cols_to_numeric = ['f1_score_draw', 'f1_score_draw_thr05', 'precision_draw', 'recall_draw', 'roi', 'num_bets', 'auc', 'brier']
        for col in cols_to_numeric:
             results_df[col] = pd.to_numeric(results_df[col], errors='coerce')
        # Define fill values apropriados para NaNs pós-conversão
        results_df['f1_score_draw'].fillna(-1.0, inplace=True)
        results_df['f1_score_draw_thr05'].fillna(-1.0, inplace=True)
        results_df['precision_draw'].fillna(0.0, inplace=True)
        results_df['recall_draw'].fillna(0.0, inplace=True)
        results_df['roi'].fillna(-np.inf, inplace=True)
        results_df['num_bets'].fillna(0, inplace=True)
        results_df['auc'].fillna(0.0, inplace=True)
        results_df['brier'].fillna(1.0, inplace=True)


        if results_df.empty: logger.error("SELEÇÃO: DF de resultados vazio."); return False;

        # *** LOG REINSERIDO: Tabela Comparativa ***
        logger.info("--- Comparativo de Desempenho dos Modelos (Conjunto de Teste) ---")
        # Seleciona colunas para exibir na tabela
        display_cols = ['model_name', 'f1_score_draw', 'precision_draw', 'recall_draw', 'auc', 'brier', 'roi', 'num_bets', 'optimal_f1_threshold', 'optimal_ev_threshold']
        # Garante que as colunas existem antes de tentar acessá-las
        display_cols_exist = [col for col in display_cols if col in results_df.columns or col in ['optimal_f1_threshold', 'optimal_ev_threshold']] # Limiares vem do dict original
        # Adiciona limiares do dict original se não extraídos antes
        if 'optimal_f1_threshold' not in results_df.columns: results_df['optimal_f1_threshold'] = results_df['metrics'].apply(lambda m: m.get('optimal_f1_threshold', 0.5))
        if 'optimal_ev_threshold' not in results_df.columns: results_df['optimal_ev_threshold'] = results_df['metrics'].apply(lambda m: m.get('optimal_ev_threshold', default_ev_threshold))

        results_df_display = results_df[display_cols_exist].round({ # Arredonda para clareza
            'f1_score_draw': 4, 'precision_draw': 4, 'recall_draw': 4,
            'auc': 4, 'brier': 4, 'roi': 2,
            'optimal_f1_threshold': 4, 'optimal_ev_threshold': 3
        })
        # Converte num_bets para inteiro
        if 'num_bets' in results_df_display.columns: results_df_display['num_bets'] = results_df_display['num_bets'].astype(int)

        # Imprime a tabela formatada
        try:
             # Tenta usar to_markdown se pandas >= 1.0.0
             logger.info("\n" + results_df_display.sort_values(by='f1_score_draw', ascending=False).to_markdown(index=False))
        except AttributeError:
             # Fallback para to_string
             logger.info("\n" + results_df_display.sort_values(by='f1_score_draw', ascending=False).to_string(index=False))
        logger.info("---------------------------------------------------------------------")


        # --- Seleção Melhor F1 (usa a coluna 'f1_score_draw' principal) ---
        results_df_sorted_f1 = results_df.sort_values(by='f1_score_draw', ascending=False).reset_index(drop=True);
        if results_df_sorted_f1.empty: logger.error("SELEÇÃO: Ranking F1 vazio."); return False;
        best_f1_result = results_df_sorted_f1.iloc[0].to_dict().copy();
        f1_val = best_f1_result.get('f1_score_draw', -1.0);
        f1_thr = best_f1_result.get('optimal_f1_threshold', 0.5)
        logger.info(f"SELEÇÃO: Melhor F1 (Teste): {best_f1_result.get('model_name','ERRO')} (F1={f1_val:.4f} @ Thr={f1_thr:.4f})")


        # --- Seleção Melhor ROI (usa ROI calculado com limiar EV otimizado) ---
        best_roi_result = None;
        # *** LOG REINSERIDO: Ranking ROI ***
        logger.info("--- Ranking por ROI (Teste) ---")
        results_df_sorted_roi = results_df.sort_values(by='roi', ascending=False).reset_index(drop=True);
        # Seleciona colunas para exibir no log do ranking ROI
        roi_rank_cols = ['model_name', 'roi', 'num_bets', 'f1_score_draw', 'optimal_ev_threshold']
        roi_rank_cols_exist = [col for col in roi_rank_cols if col in results_df_sorted_roi.columns]
        try:
             logger.info("\n" + results_df_sorted_roi[roi_rank_cols_exist].round({'roi':2, 'f1_score_draw':4, 'optimal_ev_threshold':3}).to_markdown(index=False))
        except AttributeError:
             logger.info("\n" + results_df_sorted_roi[roi_rank_cols_exist].round({'roi':2, 'f1_score_draw':4, 'optimal_ev_threshold':3}).to_string(index=False))
        logger.info("-----------------------------")

        if not results_df_sorted_roi.empty:
             # Procura o primeiro ROI válido (> -infinito)
             for idx, row in results_df_sorted_roi.iterrows():
                  cr=row['roi']; cn=row['model_name']; ev_thr=row.get('optimal_ev_threshold', default_ev_threshold)
                  if isinstance(cr,(int,float,np.number)) and cr > -np.inf:
                       best_roi_result = row.to_dict().copy();
                       logger.info(f"SELEÇÃO: Melhor ROI Válido (Teste): {cn} (ROI={cr:.2f}% @ EV Thr={ev_thr:.3f})"); break;
             if best_roi_result is None: logger.warning("SELEÇÃO: Nenhum ROI válido (> -inf) encontrado no teste.")
        else: logger.warning("SELEÇÃO: Ranking ROI vazio.")

        # --- Lógica Decisão Salvamento (sem alterações) ---
        model_to_save_f1 = best_f1_result; model_to_save_roi = None;
        if best_roi_result:
            if best_f1_result.get('model_name') == best_roi_result.get('model_name'):
                 if len(results_df_sorted_f1) > 1: model_to_save_roi = results_df_sorted_f1.iloc[1].to_dict().copy(); logger.info(f"  -> Usando 2º Melhor F1 ({model_to_save_roi.get('model_name')}) para slot ROI.")
                 else: model_to_save_roi = best_f1_result.copy(); logger.warning("  -> Apenas 1 modelo, salvando F1 para ambos slots.")
            else: model_to_save_roi = best_roi_result.copy(); logger.info(f"  -> Usando Melhor ROI ({model_to_save_roi.get('model_name')}) para slot ROI.")
        else: # Se não achou ROI válido
            model_to_save_roi = best_f1_result.copy(); logger.warning(f"  -> Nenhum ROI válido encontrado, salvando Melhor F1 ({model_to_save_f1.get('model_name')}) para ambos slots.")

        # --- Salvamento (sem alterações, usa _save_model_object) ---
        logger.info(f"Salvando Melhor F1 ({model_to_save_f1.get('model_name','ERRO')})...");
        _save_model_object(model_to_save_f1, feature_names, BEST_F1_MODEL_SAVE_PATH);

        if model_to_save_roi:
             logger.info(f"Salvando Melhor ROI/2nd F1 ({model_to_save_roi.get('model_name','ERRO')})...");
             _save_model_object(model_to_save_roi, feature_names, BEST_ROI_MODEL_SAVE_PATH);
        else: logger.error(f"ERRO CRÍTICO SALVAR: model_to_save_roi é None."); return False

    except Exception as e: logger.error(f"Erro GERAL na seleção/salvamento: {e}", exc_info=True); return False;

    logger.info("--- Processo Completo ---"); return True


# --- Função _save_model_object (MODIFICADA para incluir limiar F1) ---
def _save_model_object(model_result_dict: Dict, feature_names: List[str], file_path: str) -> None:
    """Salva modelo, scaler, calibrador, params, métricas, limiar EV e limiar F1."""
    if not isinstance(model_result_dict, dict): logger.error(f"Salvar: Dados inválidos p/ {file_path}"); return
    try:
        model_to_save = model_result_dict.get('model_object')
        if model_to_save is None: logger.error(f"Salvar: Modelo ausente p/ {file_path}"); return

        save_obj = {
            'model': model_to_save,
            'scaler': model_result_dict.get('scaler'),
            'calibrator': model_result_dict.get('calibrator'),
            'feature_names': feature_names,
            'best_params': model_result_dict.get('params'),
            'eval_metrics': model_result_dict.get('metrics'), # Dict completo
            'optimal_ev_threshold': model_result_dict.get('optimal_ev_threshold', DEFAULT_EV_THRESHOLD), # SALVA LIMIAR EV
            'optimal_f1_threshold': model_result_dict.get('optimal_f1_threshold', 0.5), # <<< SALVA LIMIAR F1
            'save_timestamp': datetime.datetime.now().isoformat(),
            'model_class_name': model_to_save.__class__.__name__
        }
        joblib.dump(save_obj, file_path)
        logger.info(f"  -> Modelo '{save_obj['model_class_name']}' (F1 Thr={save_obj['optimal_f1_threshold']:.4f}, EV Thr={save_obj['optimal_ev_threshold']:.3f}) salvo em {os.path.basename(file_path)}.")
    except Exception as e:
        logger.error(f"  -> Erro GRAVE ao salvar objeto em {file_path}: {e}", exc_info=True)

# --- Funções Remanescentes (analyze_features, optimize_single_model) ---
def analyze_features(X: pd.DataFrame, y: pd.Series) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """ Analisa features: importância (RF rápido) e correlação. Retorna DFs ou (None, None). """
    logger.info("--- ANÁLISE FEATURES: Iniciando ---")
    imp_df = None # Inicia como None
    corr_matrix = None # Inicia como None

    if X is None or y is None or X.empty or y.empty:
        logger.error("ANÁLISE FEATURES: Dados X ou y inválidos/vazios.")
        return imp_df, corr_matrix # Retorna (None, None)

    # Alinhamento (como antes)
    if not X.index.equals(y.index):
        logger.warning("ANÁLISE FEATURES: Índices X/y não idênticos. Tentando alinhar.")
        try: y = y.reindex(X.index);
        except Exception as e: logger.error(f"ANÁLISE FEATURES: Erro alinhar y: {e}"); return imp_df, corr_matrix;
        if y.isnull().any(): logger.error("ANÁLISE FEATURES: NaNs em y após alinhar."); return imp_df, corr_matrix;

    feature_names = X.columns.tolist()

    # 1. Calcular Importância
    logger.info("ANÁLISE FEATURES: Calculando importância RF...")
    try:
        # Verifica se há NaNs/Infs remanescentes em X ou y
        if X.isnull().values.any() or not np.all(np.isfinite(X.values)):
             logger.error("ANÁLISE FEATURES: NaNs ou Infs encontrados em X antes do fit RF!")
             # Opcional: mostrar onde estão os NaNs/Infs
             logger.error(f"Nulos em X:\n{X.isnull().sum()[X.isnull().sum() > 0]}")
             logger.error(f"Infinitos em X:\n{np.isinf(X.values).sum(axis=0)}")
        if y.isnull().values.any():
             logger.error("ANÁLISE FEATURES: NaNs encontrados em y antes do fit RF!")

        rf_analyzer = RandomForestClassifier(n_estimators=50, max_depth=15, n_jobs=-1, min_samples_leaf=3, random_state=RANDOM_STATE)
        logger.info(f"    -> Fitting RF (X shape: {X.shape}, y shape: {y.shape})")
        rf_analyzer.fit(X, y)
        logger.info("    -> Fit RF concluído.")
        importances = rf_analyzer.feature_importances_
        imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        imp_df = imp_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        logger.info(f"ANÁLISE FEATURES: Importância calculada OK. Shape: {imp_df.shape}")
    except Exception as e:
        logger.error(f"ANÁLISE FEATURES: Erro GRAVE ao calcular importância RF: {e}", exc_info=True)
        imp_df = None # Define como None se falhar

    # 2. Calcular Correlação
    logger.info("ANÁLISE FEATURES: Calculando correlação...")
    try:
        df_temp = X.copy()
        df_temp['target_IsDraw'] = y
        # Verifica NaNs/Infs antes de .corr()
        if df_temp.isnull().values.any() or not np.all(np.isfinite(df_temp.values)):
            logger.error("ANÁLISE FEATURES: NaNs ou Infs encontrados em df_temp antes de .corr()!")
            logger.error(f"Nulos em df_temp:\n{df_temp.isnull().sum()[df_temp.isnull().sum() > 0]}")
            logger.error(f"Infinitos em df_temp:\n{np.isinf(df_temp.values).sum(axis=0)}")

        logger.info(f"    -> Calculando corr() em df_temp (shape: {df_temp.shape})")
        corr_matrix = df_temp.corr()
        logger.info(f"ANÁLISE FEATURES: Correlação calculada OK. Shape: {corr_matrix.shape}")
    except Exception as e:
        logger.error(f"ANÁLISE FEATURES: Erro GRAVE ao calcular correlação: {e}", exc_info=True)
        corr_matrix = None # Define como None se falhar

    logger.info("--- ANÁLISE FEATURES: Concluída ---")
    # Retorna os dataframes (podem ser None se houve erro)
    return imp_df, corr_matrix

def optimize_single_model(*args, **kwargs) -> Optional[Tuple[str, Dict, Dict]]:
    logger.warning("optimize_single_model placeholder."); return None