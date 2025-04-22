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
    LGBM_AVAILABLE = True
except ImportError:
    logger.warning("AVISO: LightGBM não instalado.")
    lgb = None; LGBMClassifier = None
    LGBM_AVAILABLE = False
# Importar métricas e numpy
from sklearn.metrics import (accuracy_score, classification_report, log_loss,
                             precision_score, recall_score, f1_score, # Adicionar f1_score se não estiver
                             roc_auc_score, confusion_matrix, brier_score_loss,
                             precision_recall_curve) # <-- IMPORTANTE: para achar limiar
import numpy as np 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib, os, datetime, numpy as np, traceback
try:
    from config import (RANDOM_STATE, MODEL_CONFIG, CLASS_NAMES, TEST_SIZE,
                        BEST_F1_MODEL_SAVE_PATH, BEST_ROI_MODEL_SAVE_PATH,
                        CROSS_VALIDATION_SPLITS, N_JOBS_GRIDSEARCH, DEFAULT_F1_THRESHOLD,
                        ODDS_COLS, BEST_MODEL_METRIC, BEST_MODEL_METRIC_ROI, DEFAULT_EV_THRESHOLD)
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
    profit, roi_value, num_bets, profit_calc, valid_bets_count = None, None, 0, 0, 0
    if odds_data is None or odd_col_name not in odds_data.columns: return roi_value, num_bets, profit
    try:
        common_index=y_true.index.intersection(odds_data.index);
        if len(common_index)==0: return 0.0, 0, 0.0
        y_true_common=y_true.loc[common_index]; odds_common=pd.to_numeric(odds_data.loc[common_index,odd_col_name],errors='coerce');
        try: y_proba_series=pd.Series(y_proba, index=y_true.index); y_proba_common=y_proba_series.loc[common_index];
        except Exception: logger.error("ROI Thr: Erro alinhar y_proba."); return None,0,None
        bet_indices=common_index[y_proba_common>threshold]; num_bets=len(bet_indices);
        if num_bets==0: return 0.0,num_bets,0.0
        actuals=y_true_common.loc[bet_indices]; odds_selected=odds_common.loc[bet_indices];
        for idx in bet_indices:
            odd_d=odds_selected.loc[idx];
            if pd.notna(odd_d)and odd_d>1: profit_calc+=(odd_d-1)if actuals.loc[idx]==1 else-1; valid_bets_count+=1;
        profit=profit_calc; roi_value = (profit/valid_bets_count)*100 if valid_bets_count>0 else 0.0;
        return roi_value,num_bets,profit
    except Exception as e: logger.error(f"ROI Thr: Erro - {e}",exc_info=True); return None,0,None

def calculate_metrics_with_ev(y_true: pd.Series, y_proba_calibrated: np.ndarray, ev_threshold: float, odds_data: Optional[pd.DataFrame], odd_col_name: str) -> Tuple[Optional[float], int, Optional[float]]:
    profit, roi_value, num_bets, profit_calc, valid_bets_count = None, None, 0, 0, 0
    if odds_data is None or odd_col_name not in odds_data.columns: logger.warning(f"EV Metr: Odds ausentes."); return roi_value,num_bets,profit
    try:
        common_index=y_true.index.intersection(odds_data.index);
        if len(common_index)==0: return 0.0, 0, 0.0
        y_true_common=y_true.loc[common_index]; odds_common=pd.to_numeric(odds_data.loc[common_index,odd_col_name],errors='coerce');
        try: y_proba_common=pd.Series(y_proba_calibrated,index=y_true.index).loc[common_index];
        except Exception: logger.error("EV Metr: Erro alinhar y_proba."); return None,0,None
        valid_mask = odds_common.notna() & y_proba_common.notna() & (odds_common > 1)
        ev = pd.Series(np.nan, index=common_index);
        ev.loc[valid_mask] = (y_proba_common[valid_mask]*(odds_common[valid_mask]-1)) - ((1-y_proba_common[valid_mask])*1);
        bet_indices=common_index[ev > ev_threshold]; num_bets=len(bet_indices);
        if num_bets==0: return 0.0,num_bets,0.0
        actuals=y_true_common.loc[bet_indices]; odds_selected=odds_common.loc[bet_indices];
        for idx in bet_indices:
            odd_d=odds_selected.loc[idx];
            if pd.notna(odd_d) and odd_d > 1: profit_calc+=(odd_d-1)if actuals.loc[idx]==1 else-1; valid_bets_count+=1;
        profit=profit_calc; roi_value = (profit/valid_bets_count)*100 if valid_bets_count>0 else 0.0;
        logger.debug(f"    -> Métricas EV (Th={ev_threshold:.3f}): ROI={roi_value:.2f}%, Bets={num_bets}, Profit={profit:.2f}") # DEBUG
        return roi_value,num_bets,profit
    except Exception as e: logger.error(f"EV Metr: Erro - {e}",exc_info=True); return None,0,None


def scale_features(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, scaler_type='standard') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Any]:
    """ Escala features usando treino, aplica em val e teste. """
    X_train_c, X_val_c, X_test_c = X_train.copy(), X_val.copy(), X_test.copy(); scaler = None
    try:
        if scaler_type=='minmax': scaler=MinMaxScaler()
        else: scaler=StandardScaler()
        logger.info(f"  Aplicando {scaler.__class__.__name__}...");
        X_train_c = X_train_c.astype(float).replace([np.inf, -np.inf], np.nan)
        if X_train_c.isnull().values.any():
             logger.warning("NaNs em X_train antes de escalar. Imputando com mediana.")
             train_median = X_train_c.median()
             X_train_c.fillna(train_median, inplace=True)
             if X_val_c is not None: X_val_c = X_val_c.astype(float).replace([np.inf,-np.inf],np.nan).fillna(train_median)
             if X_test_c is not None: X_test_c = X_test_c.astype(float).replace([np.inf,-np.inf],np.nan).fillna(train_median)

        scaler.fit(X_train_c); # Ajusta scaler nos dados de treino limpos
        X_train_scaled=scaler.transform(X_train_c)
        X_val_scaled=scaler.transform(X_val_c.astype(float)) if X_val_c is not None else None
        X_test_scaled=scaler.transform(X_test_c.astype(float)) if X_test_c is not None else None

        X_train_sc = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
        X_val_sc = pd.DataFrame(X_val_scaled, index=X_val.index, columns=X_val.columns) if X_val_scaled is not None else None
        X_test_sc = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns) if X_test_scaled is not None else None
        logger.info("  Scaling concluído.");
        return X_train_sc, X_val_sc, X_test_sc, scaler # Retorna 4 itens
    except Exception as e: logger.error(f"Erro no scaling: {e}", exc_info=True); raise

# --- Função Principal de Treinamento (REVISADA FINAL) ---
def train_evaluate_and_save_best_models(
    X: pd.DataFrame, y: pd.Series,
    X_test_with_odds: Optional[pd.DataFrame] = None,
    progress_callback_stages: Optional[Callable[[int, str], None]] = None, # Nome correto
    num_total_models: int = 1, # Recebe número total
    scaler_type: str = 'standard',
    odd_draw_col_name: str = ODDS_COLS.get('draw', 'Odd_D_FT'),
    calibration_method: str = 'isotonic',
    optimize_ev_threshold: bool = True,
    optimize_f1_threshold: bool = True
    ) -> bool:
    """
    Treina (BayesOpt/Grid), calibra, otimiza limiares F1/EV, avalia, salva.
    Usa callbacks de progresso por etapa.
    """
    # ... (Verificações iniciais) ...
    if X is None or y is None: logger.error("Dados X ou y None."); return False
    if not MODEL_CONFIG: logger.error("MODEL_CONFIG vazio."); return False
    available_models = {name: config for name, config in MODEL_CONFIG.items() if not (name=='LGBMClassifier' and not LGBM_AVAILABLE)}
    if not available_models: logger.error("Nenhum modelo válido."); return False
    if num_total_models != len(available_models): num_total_models = len(available_models); logger.warning("Ajustando num_total_models.")

    feature_names = list(X.columns); all_results = []
    logger.info(f"--- Treinando {num_total_models} Modelos (Opt F1: {optimize_f1_threshold}, Opt EV: {optimize_ev_threshold}) ---"); start_time_total = time.time()

    # --- Divisão Tripla e Preparo Odds ---
    logger.info("Dividindo dados..."); val_size=0.20; test_size_final=TEST_SIZE; train_val_size_temp=1.0-test_size_final
    if train_val_size_temp<=0: logger.error("TEST_SIZE inválido."); return False;
    val_size_relative=val_size/train_val_size_temp
    try:
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size_final, random_state=RANDOM_STATE, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size_relative, random_state=RANDOM_STATE, stratify=y_train_val)
        logger.info(f"Split: T={len(X_train)}, V={len(X_val)}, Ts={len(X_test)}")
        X_val_odds, X_test_odds = None, None
        if X_test_with_odds is not None and not X_test_with_odds.empty and odd_draw_col_name in X_test_with_odds.columns:
            common_val = X_val.index.intersection(X_test_with_odds.index); common_test = X_test.index.intersection(X_test_with_odds.index)
            if len(common_val) > 0: X_val_odds = X_test_with_odds.loc[common_val, [odd_draw_col_name]].copy()
            if len(common_test) > 0: X_test_odds = X_test_with_odds.loc[common_test, [odd_draw_col_name]].copy()
            logger.info(f"Odds alinhadas: Val={X_val_odds is not None}, Teste={X_test_odds is not None}")
        else: logger.warning("Não foi possível alinhar odds.")
    except Exception as e: logger.error(f"Erro divisão/alinhar dados: {e}", exc_info=True); return False

    # --- Loop pelos modelos ---
    for i, (model_name, config) in enumerate(available_models.items()):
        status_text_loop=f"Modelo {i+1}/{num_total_models}: {model_name}"; logger.info(f"\n--- {status_text_loop} ---");
        if progress_callback_stages: progress_callback_stages(i, f"Iniciando {model_name}...");
        start_time_model = time.time(); model_trained = None; best_params = None; current_scaler = None; calibrator = None;
        current_optimal_f1_threshold = DEFAULT_F1_THRESHOLD
        current_optimal_ev_threshold = DEFAULT_EV_THRESHOLD

        try:
            # Setup
            model_class = eval(model_name); model_kwargs=config.get('model_kwargs',{}); needs_scaling=config.get('needs_scaling',False)
            use_bayes_opt=SKOPT_AVAILABLE and 'search_spaces' in config; param_space=config.get('search_spaces') if use_bayes_opt else config.get('param_grid')
            X_train_m, X_val_m, X_test_m = X_train.copy(), X_val.copy(), X_test.copy();

            # Scaling
            if needs_scaling:
                 if progress_callback_stages: progress_callback_stages(i, f"Scaling...");
                 logger.info(f"  Scaling p/ {model_name}...");
                 try:
                     # *** CORREÇÃO APLICADA AQUI ***
                     X_train_m, X_val_m, X_test_m, current_scaler = scale_features(
                         X_train_m, X_val_m, X_test_m, scaler_type
                     )
                     # **********************************
                     logger.info("    -> Scaling OK.")
                 except Exception as e: logger.error(f"  ERRO scaling: {e}"); continue # Pula para próximo modelo

            # Treino (Busca ou Padrão)
            # ... (código como na resposta anterior, sem alterações necessárias aqui) ...
            if param_space:
                 method_name='BayesSearchCV' if use_bayes_opt else 'GridSearchCV'
                 if progress_callback_stages: progress_callback_stages(i, f"Ajustando ({method_name[:5]})...");
                 logger.info(f"  Iniciando {method_name}...");
                 try:
                     if use_bayes_opt: search_cv=BayesSearchCV(estimator=model_class(**model_kwargs),search_spaces=param_space,n_iter=BAYESIAN_OPT_N_ITER,cv=CROSS_VALIDATION_SPLITS,n_jobs=N_JOBS_GRIDSEARCH,scoring='f1',random_state=RANDOM_STATE,verbose=0)
                     else: search_cv=GridSearchCV(estimator=model_class(**model_kwargs),param_grid=param_space,cv=CROSS_VALIDATION_SPLITS,n_jobs=N_JOBS_GRIDSEARCH,scoring='f1',verbose=0, error_score='raise')
                     search_cv.fit(X_train_m, y_train); model_trained=search_cv.best_estimator_; best_params=search_cv.best_params_; best_cv_score=search_cv.best_score_; logger.info(f"    -> Busca OK. CV F1: {best_cv_score:.4f}. Params: {best_params}");
                 except Exception as e: logger.error(f"    Erro {method_name}.fit: {e}"); model_trained = None; logger.warning("    -> Tentando fallback...");
            if model_trained is None:
                 if progress_callback_stages: progress_callback_stages(i, "Ajustando (Padrão)...");
                 logger.info(f"  Treinando {model_name} c/ params padrão...");
                 try: model_trained = model_class(**model_kwargs).fit(X_train_m, y_train); best_params = model_kwargs; logger.info("    -> Treino padrão OK.")
                 except Exception as e_fall: logger.error(f"    Erro treino fallback: {e_fall}"); continue;
            if model_trained is None: continue; # Pula se treino falhou

            # Calibração
            # ... (código como na resposta anterior) ...
            y_proba_val_raw_draw, y_proba_val_calib = None, None
            if hasattr(model_trained, "predict_proba"):
                if progress_callback_stages: progress_callback_stages(i, "Calibrando...");
                logger.info(f"  Calibrando probs com {calibration_method}...");
                try: 
                    y_proba_val_raw_full=model_trained.predict_proba(X_val_m);
                    if y_proba_val_raw_full.shape[1]>1: y_proba_val_raw_draw=y_proba_val_raw_full[:,1]; calibrator=IsotonicRegression(out_of_bounds='clip'); calibrator.fit(y_proba_val_raw_draw,y_val); y_proba_val_calib=calibrator.predict(y_proba_val_raw_draw); logger.info("  -> Calibrador treinado.")
                except Exception as e: logger.error(f"  Erro calibração: {e}"); calibrator=None;
            else: logger.warning(f"  {model_name} sem predict_proba.");


            # Otimização Limiares (na Validação)
            # ... (código como na resposta anterior) ...
            proba_opt = y_proba_val_calib if y_proba_val_calib is not None else y_proba_val_raw_draw; opt_src = 'Calib' if y_proba_val_calib is not None else 'Raw'
            if optimize_f1_threshold and proba_opt is not None:
                 if progress_callback_stages: progress_callback_stages(i, f"Otimizando F1 Thr ({opt_src})...");
                 logger.info(f"  Otimizando F1 (Val Probs {opt_src})..."); best_val_f1=-1.0;
                 try: p,r,t=precision_recall_curve(y_val,proba_opt); f1=(2*p*r)/(p+r+1e-9); idx=np.argmax(f1[:-1]); current_optimal_f1_threshold=t[idx]; best_val_f1=f1[idx]; logger.info(f"    Limiar F1 ótimo(Val): {current_optimal_f1_threshold:.4f} (F1={best_val_f1:.4f})")
                 except Exception as e: logger.error(f"  Erro otim F1: {e}"); current_optimal_f1_threshold=DEFAULT_F1_THRESHOLD
            if optimize_ev_threshold and proba_opt is not None and X_val_odds is not None:
                if progress_callback_stages: progress_callback_stages(i, f"Otimizando EV Thr ({opt_src})...");
                logger.info(f"  Otimizando EV (Val Probs {opt_src})..."); best_val_roi_ev=-np.inf;
                try: 
                    ev_ths=np.linspace(0.0,0.20,21);
                    for ev_th in ev_ths: 
                        val_roi,_,_=calculate_metrics_with_ev(y_val,proba_opt,ev_th,X_val_odds,odd_draw_col_name);
                        if val_roi is not None and val_roi>best_val_roi_ev: 
                            best_val_roi_ev=val_roi; current_optimal_ev_threshold=ev_th;
                    if best_val_roi_ev > -np.inf: logger.info(f"    Limiar EV ótimo(Val): {current_optimal_ev_threshold:.3f} (ROI={best_val_roi_ev:.2f}%)")
                    else: logger.warning("    ROI Val inválido."); current_optimal_ev_threshold = DEFAULT_EV_THRESHOLD;
                except Exception as e: logger.error(f"  Erro otim EV: {e}"); current_optimal_ev_threshold = DEFAULT_EV_THRESHOLD;
            elif optimize_ev_threshold: logger.warning(f"  Otim EV pulada.");

            # Avaliação Teste
            if progress_callback_stages: progress_callback_stages(i, f"Avaliando...");
            # ... (código avaliação como antes, popula 'metrics') ...
            logger.info(f"  Avaliando no Teste...")
            metrics={'optimal_f1_threshold':current_optimal_f1_threshold,'optimal_ev_threshold':current_optimal_ev_threshold}; y_proba_test_raw_full,y_proba_test_raw_draw,y_proba_test_calib = None,None,None
            if hasattr(model_trained,"predict_proba"):
                try: 
                    y_proba_test_raw_full=model_trained.predict_proba(X_test_m);
                    if y_proba_test_raw_full.shape[1]>1: 
                        y_proba_test_raw_draw=y_proba_test_raw_full[:,1];
                        if calibrator: 
                            y_proba_test_calib=calibrator.predict(y_proba_test_raw_draw); 
                    else: y_proba_test_calib=y_proba_test_raw_draw
                except Exception as e: logger.error(f"  Erro predict_proba teste: {e}");
            try: y_pred05=model_trained.predict(X_test_m); acc05=accuracy_score(y_test,y_pred05); prec05=precision_score(y_test,y_pred05,pos_label=1,zero_division=0); rec05=recall_score(y_test,y_pred05,pos_label=1,zero_division=0); f1_05=f1_score(y_test,y_pred05,pos_label=1,zero_division=0); matrix05=confusion_matrix(y_test,y_pred05).tolist(); metrics.update({'accuracy_thr05':acc05,'precision_draw_thr05':prec05,'recall_draw_thr05':rec05,'f1_score_draw_thr05':f1_05,'confusion_matrix_thr05':matrix05}); logger.info(f"    -> Métricas @ Thr 0.5: F1={f1_05:.4f}, P={prec05:.4f}, R={rec05:.4f}")
            except Exception as e: logger.error(f"  Erro métricas @ 0.5: {e}")
            proba_f1_test=y_proba_test_calib if y_proba_test_calib is not None else y_proba_test_raw_draw; f1_final,prec_final,rec_final = -1.0,0.0,0.0
            if proba_f1_test is not None:
                 try: y_predF1=(proba_f1_test>=current_optimal_f1_threshold).astype(int); accF1=accuracy_score(y_test,y_predF1); prec_final=precision_score(y_test,y_predF1,pos_label=1,zero_division=0); rec_final=recall_score(y_test,y_predF1,pos_label=1,zero_division=0); f1_final=f1_score(y_test,y_predF1,pos_label=1,zero_division=0); matrixF1=confusion_matrix(y_test,y_predF1).tolist(); metrics.update({'accuracy_thrF1':accF1,'precision_draw':prec_final,'recall_draw':rec_final,'f1_score_draw':f1_final,'confusion_matrix_thrF1':matrixF1}); logger.info(f"    -> Métricas @ Thr F1 ({current_optimal_f1_threshold:.4f}): F1={f1_final:.4f}, P={prec_final:.4f}, R={rec_final:.4f}")
                 except Exception as e: logger.error(f"  Erro métricas @ Thr F1: {e}"); f1_final = metrics.get('f1_score_draw_thr05', -1.0)
            else: f1_final=metrics.get('f1_score_draw_thr05',-1.0); metrics['f1_score_draw']=f1_final; metrics['precision_draw']=metrics.get('precision_draw_thr05',0.0); metrics['recall_draw']=metrics.get('recall_draw_thr05',0.0); logger.info("    -> Usando métricas @ 0.5 como finais.")
            logloss,auc,brier=None,None,None; proba_calib_test=y_proba_test_calib if y_proba_test_calib is not None else y_proba_test_raw_draw
            if proba_calib_test is not None:
                 try: logloss=log_loss(y_test,y_proba_test_raw_full) if y_proba_test_raw_full is not None else None; logger.debug(f"    -> LogLoss(Raw)={logloss:.4f}" if logloss else "")
                 except Exception:pass
                 try: auc=roc_auc_score(y_test,proba_calib_test) if len(np.unique(y_test))>1 else None; logger.info(f"    -> AUC(Calib/Bruta)={auc:.4f}" if auc else "")
                 except Exception:pass
                 try: brier=brier_score_loss(y_test,proba_calib_test); logger.info(f"    -> Brier(Calib/Bruta)={brier:.4f}" if brier else "")
                 except Exception:pass
            metrics.update({'log_loss':logloss,'roc_auc':auc,'brier_score':brier})
            roi_ev,bets_ev,profit_ev = None,0,None
            if proba_calib_test is not None and X_test_odds is not None: roi_ev,bets_ev,profit_ev=calculate_metrics_with_ev(y_test,proba_calib_test,current_optimal_ev_threshold,X_test_odds,odd_draw_col_name);
            else: logger.warning("    -> ROI/Profit EV (Teste) não calculado.");
            metrics.update({'roi':roi_ev,'num_bets':bets_ev,'profit':profit_ev});
            metrics.update({'train_set_size':len(y_train),'val_set_size':len(y_val),'test_set_size':len(y_test)});


            # Guarda resultado
            all_results.append({ 'model_name': model_name, 'model_object': model_trained, 'scaler': current_scaler, 'calibrator': calibrator, 'params': best_params, 'metrics': metrics, 'optimal_f1_threshold': current_optimal_f1_threshold, 'optimal_ev_threshold': current_optimal_ev_threshold })
            if progress_callback_stages: progress_callback_stages(i, f"Resultado adicionado.");
            logger.info(f"    ==> Resultado {model_name} adicionado.")

        except Exception as e_outer: logger.error(f"Erro GERAL loop {model_name}: {e_outer}", exc_info=True); continue
        logger.info(f"  Tempo p/ {model_name} (total): {time.time() - start_time_model:.2f} seg.")
    # --- Fim do Loop for ---

    # --- Seleção e Salvamento ---
    if progress_callback_stages: progress_callback_stages(num_total_models, "Selecionando/Salvando...")
    # ... (Resto da função como na resposta anterior, usando BEST_MODEL_METRIC/ROI para ordenar) ...
    end_time_total=time.time(); logger.info(f"--- Treino concluído ({end_time_total-start_time_total:.2f} seg) ---")
    if not all_results: logger.error("SELEÇÃO: Nenhum resultado válido."); return False;
    logger.info(f"--- Processando {len(all_results)} resultados ---")
    try:
        results_df=pd.DataFrame(all_results)
        results_df['f1_score_draw']=results_df['metrics'].apply(lambda m: m.get('f1_score_draw', -1.0)); results_df['precision_draw']=results_df['metrics'].apply(lambda m: m.get('precision_draw', 0.0)); results_df['recall_draw']=results_df['metrics'].apply(lambda m: m.get('recall_draw', 0.0)); results_df['roi']=results_df['metrics'].apply(lambda m: m.get('roi', -np.inf)); results_df['num_bets']=results_df['metrics'].apply(lambda m: m.get('num_bets', 0)); results_df['auc']=results_df['metrics'].apply(lambda m: m.get('roc_auc', 0.0)); results_df['brier']=results_df['metrics'].apply(lambda m: m.get('brier_score', 1.0))
        cols_num=['f1_score_draw','precision_draw','recall_draw','roi','num_bets','auc','brier','optimal_f1_threshold','optimal_ev_threshold']
        for col in cols_num: results_df[col]=pd.to_numeric(results_df[col],errors='coerce')
        results_df['f1_score_draw'].fillna(-1.0,inplace=True); results_df['precision_draw'].fillna(0.0,inplace=True); results_df['recall_draw'].fillna(0.0,inplace=True); results_df['roi'].fillna(-np.inf,inplace=True); results_df['num_bets'].fillna(0,inplace=True); results_df['auc'].fillna(0.0,inplace=True); results_df['brier'].fillna(1.0,inplace=True); results_df['optimal_f1_threshold'].fillna(DEFAULT_F1_THRESHOLD,inplace=True); results_df['optimal_ev_threshold'].fillna(DEFAULT_EV_THRESHOLD,inplace=True)
        if results_df.empty: logger.error("SELEÇÃO: DF vazio."); return False;
        logger.info("--- Comparativo Desempenho Modelos (Teste) ---"); display_cols=['model_name','f1_score_draw','precision_draw','recall_draw','auc','brier','roi','num_bets','optimal_f1_threshold','optimal_ev_threshold']; display_cols_exist=[col for col in display_cols if col in results_df.columns]
        results_df_display=results_df[display_cols_exist].round({'f1_score_draw':4,'precision_draw':4,'recall_draw':4,'auc':4,'brier':4,'roi':2,'optimal_f1_threshold':4,'optimal_ev_threshold':3})
        if 'num_bets' in results_df_display.columns: results_df_display['num_bets']=results_df_display['num_bets'].astype(int)
        try: logger.info("\n"+results_df_display.sort_values(by=BEST_MODEL_METRIC,ascending=False).to_markdown(index=False)) # Usa config
        except AttributeError: logger.info("\n"+results_df_display.sort_values(by=BEST_MODEL_METRIC,ascending=False).to_string(index=False))
        logger.info("-" * 80)
        results_df_sorted_f1=results_df.sort_values(by=BEST_MODEL_METRIC,ascending=False).reset_index(drop=True); best_f1_result=results_df_sorted_f1.iloc[0].to_dict().copy(); f1_val=best_f1_result.get(BEST_MODEL_METRIC,-1.0); f1_thr=best_f1_result.get('optimal_f1_threshold',DEFAULT_F1_THRESHOLD); logger.info(f"SELEÇÃO: Melhor F1 (Teste): {best_f1_result.get('model_name','ERRO')} (F1={f1_val:.4f} @ Thr={f1_thr:.4f})")
        best_roi_result=None; logger.info("--- Ranking por ROI (Teste) ---"); results_df_sorted_roi=results_df.sort_values(by=BEST_MODEL_METRIC_ROI,ascending=False).reset_index(drop=True); roi_rank_cols=['model_name','roi','num_bets','f1_score_draw','optimal_ev_threshold']; roi_rank_cols_exist=[col for col in roi_rank_cols if col in results_df_sorted_roi.columns]
        try: logger.info("\n"+results_df_sorted_roi[roi_rank_cols_exist].round({'roi':2,'f1_score_draw':4,'optimal_ev_threshold':3}).to_markdown(index=False))
        except AttributeError: logger.info("\n"+results_df_sorted_roi[roi_rank_cols_exist].round({'roi':2,'f1_score_draw':4,'optimal_ev_threshold':3}).to_string(index=False))
        logger.info("-" * 30)
        if not results_df_sorted_roi.empty:
             for idx,row in results_df_sorted_roi.iterrows():
                  cr=row[BEST_MODEL_METRIC_ROI]; cn=row['model_name']; ev_thr=row.get('optimal_ev_threshold',DEFAULT_EV_THRESHOLD)
                  if isinstance(cr,(int,float,np.number)) and cr > -np.inf: best_roi_result=row.to_dict().copy(); logger.info(f"SELEÇÃO: Melhor ROI Válido (Teste): {cn} (ROI={cr:.2f}% @ EV Thr={ev_thr:.3f})"); break;
             if best_roi_result is None: logger.warning("SELEÇÃO: Nenhum ROI válido.")
        else: logger.warning("SELEÇÃO: Ranking ROI vazio.")
        model_to_save_f1=best_f1_result; model_to_save_roi=None;
        if best_roi_result:
            if best_f1_result.get('model_name')==best_roi_result.get('model_name'):
                 if len(results_df_sorted_f1)>1: model_to_save_roi=results_df_sorted_f1.iloc[1].to_dict().copy(); logger.info(f"  -> Usando 2º F1 ({model_to_save_roi.get('model_name')}) p/ slot ROI.")
                 else: model_to_save_roi=best_f1_result.copy(); logger.warning("  -> Apenas 1 modelo, F1 p/ ambos.")
            else: model_to_save_roi=best_roi_result.copy(); logger.info(f"  -> Usando Melhor ROI ({model_to_save_roi.get('model_name')}) p/ slot ROI.")
        else: model_to_save_roi=best_f1_result.copy(); logger.warning(f"  -> Nenhum ROI válido, F1 p/ ambos.")
        logger.info(f"Salvando Melhor F1 ({model_to_save_f1.get('model_name','ERRO')})..."); _save_model_object(model_to_save_f1,feature_names,BEST_F1_MODEL_SAVE_PATH);
        if model_to_save_roi: logger.info(f"Salvando Melhor ROI/2nd F1 ({model_to_save_roi.get('model_name','ERRO')})..."); _save_model_object(model_to_save_roi,feature_names,BEST_ROI_MODEL_SAVE_PATH);
        else: logger.error("ERRO CRÍTICO SALVAR: model_to_save_roi é None."); return False
    except Exception as e: logger.error(f"Erro GERAL seleção/salvamento: {e}", exc_info=True); return False;

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
            'optimal_f1_threshold': model_result_dict.get('optimal_f1_threshold', DEFAULT_F1_THRESHOLD), # <<< SALVA LIMIAR F1
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