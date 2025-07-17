import pandas as pd
import joblib
import os
import datetime
import numpy as np  
import traceback
from sklearn.exceptions import NotFittedError
from config import CLASS_NAMES, ODDS_COLS, DEFAULT_EV_THRESHOLD, DEFAULT_F1_THRESHOLD
from typing import Optional, Any, List, Dict, Tuple
from calibrator import BaseCalibrator
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from logger_config import setup_logger

logger = setup_logger("PredictorApp")

def load_model_scaler_features(model_path: str) -> Optional[Tuple[Any, Optional[Any], Optional[Any], float, float, Optional[List[str]], Optional[Dict], Optional[Dict], Optional[str]]]:
    """ Carrega modelo, scaler, calibrador, limiar EV, limiar F1, features, params, métricas, timestamp. """
    try:
        load_obj = joblib.load(model_path)
        if isinstance(load_obj, dict) and 'model' in load_obj:
            model = load_obj['model']
            scaler = load_obj.get('scaler'); calibrator = load_obj.get('calibrator')
            optimal_ev_threshold = load_obj.get('optimal_ev_threshold', DEFAULT_EV_THRESHOLD)
            optimal_f1_threshold = load_obj.get('optimal_f1_threshold', DEFAULT_F1_THRESHOLD) # <<< USA DEFAULT DO CONFIG
            training_medians = load_obj.get('training_medians')
            feature_names = load_obj.get('feature_names')
            best_params = load_obj.get('best_params') 
            eval_metrics = load_obj.get('eval_metrics')
            saved_timestamp = load_obj.get('save_timestamp'); model_class_name = load_obj.get('model_class_name', 'N/A'); fts = "N/A"
            try: mtime = os.path.getmtime(model_path); fts = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            except Exception: pass
            logger.info(f"Modelo carregado: {os.path.basename(model_path)} (Modif.: {fts})")
            logger.info(f"  Tipo: {model_class_name} | Calib: {'Sim' if calibrator else 'Não'} | F1 Thr: {optimal_f1_threshold:.4f} | EV Thr: {optimal_ev_threshold:.4f}")
            return model, scaler, calibrator, optimal_ev_threshold, optimal_f1_threshold, training_medians, feature_names, best_params, eval_metrics, fts
        else: 
            logger.warning(f"  Aviso: Formato antigo/inválido em {model_path}.")
            return load_obj, None, None, DEFAULT_EV_THRESHOLD, DEFAULT_F1_THRESHOLD, None, None, None, "N/A"
    except FileNotFoundError: logger.error(f"Erro: Modelo não encontrado: '{model_path}'"); return None
    except Exception as e: logger.error(f"Erro carregar modelo: {e}", exc_info=True); return None

def make_predictions(
    model: Any,
    scaler: Optional[Any],
    calibrator: Optional[BaseCalibrator],
    feature_names: List[str],
    X_fixture_prepared: pd.DataFrame,
    fixture_info: pd.DataFrame,
    odd_draw_col_name: str = ODDS_COLS.get('draw', 'Odd_D_FT')
    ) -> Optional[pd.DataFrame]:

    if model is None or X_fixture_prepared is None or feature_names is None or fixture_info is None: logger.error("Erro make_preds: Args ausentes."); return None
    if X_fixture_prepared.empty or fixture_info.empty : logger.warning("make_preds: Input vazio."); return pd.DataFrame()
    if not set(feature_names).issubset(X_fixture_prepared.columns): missing = list(set(feature_names)-set(X_fixture_prepared.columns)); logger.error(f"Erro: Cols faltando X_prep: {missing}"); return None
    if not X_fixture_prepared.index.equals(fixture_info.index): logger.error("Erro FATAL make_preds: Índices X_prep e fixture_info NÃO COINCIDEM!"); return None

    logger.info(f"Realizando previsões ({model.__class__.__name__}) para {len(X_fixture_prepared)} jogos...")
    try:
        X_pred = X_fixture_prepared[feature_names].copy()
        if scaler:
            logger.info("  Aplicando scaler...");
            X_pred = pd.DataFrame(scaler.transform(X_pred.astype(float)), index=X_pred.index, columns=feature_names)

        if not hasattr(model, "predict_proba"): logger.error("Modelo sem predict_proba."); return None

        logger.info("  Calculando probs brutas...")
        predictions_proba_raw = model.predict_proba(X_pred)
        class_names_local = CLASS_NAMES if CLASS_NAMES and len(CLASS_NAMES) == len(model.classes_) else [f"Class_{c}" for c in model.classes_]
        logger.debug(f"Predictor: Nomes de classe locais: {class_names_local}")
        prob_cols_raw_map={model.classes_[i]:f'ProbRaw_{class_names_local[i]}' for i in range(len(model.classes_))}
        prob_cols_raw_names=list(prob_cols_raw_map.values())
        df_predictions = pd.DataFrame(predictions_proba_raw, columns=prob_cols_raw_names, index=X_pred.index)
        logger.debug(f"Predictor: Colunas probs brutas criadas: {prob_cols_raw_names}")
        logger.debug(f"Predictor: Amostra df_predictions (brutas):\n{df_predictions.head()}")

        # --- Calibração ---
        prob_col_calib_draw = f'Prob_{class_names_local[1]}'  
        prob_col_calib_other = f'Prob_{class_names_local[0]}' 
        prob_col_raw_draw = f'ProbRaw_{class_names_local[1]}' 
        proba_draw_calibrated_series = None
        df_predictions[prob_col_calib_draw] = np.nan
        df_predictions[prob_col_calib_other] = np.nan

        if calibrator and prob_col_raw_draw in df_predictions.columns:
            logger.info(f"  Aplicando calibrador ({calibrator.__class__.__name__}) na coluna '{prob_col_raw_draw}'...")
            try:
                proba_draw_raw_values = df_predictions[prob_col_raw_draw].values # 1D array
                proba_draw_calibrated_array = None

                if isinstance(calibrator, IsotonicRegression): 
                   proba_draw_calibrated_array = calibrator.predict(proba_draw_raw_values)
                elif isinstance(calibrator, LogisticRegression): 
                   proba_draw_calibrated_array = calibrator.predict_proba(proba_draw_raw_values.reshape(-1, 1))[:, 1]
                else:
                   logger.warning(f"  Tipo de calibrador não tratado explicitamente: {calibrator.__class__.__name__}. Tentando .predict_proba() se existir, senão .predict().")
                   if hasattr(calibrator, 'predict_proba'):
                       try:
                           proba_draw_calibrated_array = calibrator.predict_proba(proba_draw_raw_values.reshape(-1, 1))[:, 1]
                       except ValueError as ve_pp:
                           if "Expected 2D array" in str(ve_pp) and proba_draw_raw_values.ndim == 1: 
                               proba_draw_calibrated_array = calibrator.predict_proba(proba_draw_raw_values.reshape(-1,1))[:,1]
                           else: 
                               proba_draw_calibrated_array = calibrator.predict_proba(proba_draw_raw_values)[:,1]
                   elif hasattr(calibrator, 'predict'):
                       proba_draw_calibrated_array = calibrator.predict(proba_draw_raw_values)
                   else:
                       logger.error("  Calibrador não tem método predict_proba nem predict.")

                if proba_draw_calibrated_array is not None:
                        proba_draw_calibrated_array = np.clip(proba_draw_calibrated_array, 0.0, 1.0)
                        proba_draw_calibrated_series = pd.Series(proba_draw_calibrated_array, index=df_predictions.index)
                        df_predictions[prob_col_calib_draw] = proba_draw_calibrated_series
                        logger.info(f"  -> Coluna '{prob_col_calib_draw}' (Calibrada) preenchida.")
            except NotFittedError as nfe:
                logger.error(f"  Calibrador não ajustado: {nfe}")

        # --- Cálculo de EV ---
        df_predictions['EV_Empate'] = np.nan
        calculate_ev = odd_draw_col_name in fixture_info.columns
        proba_ev_base = proba_draw_calibrated_series if proba_draw_calibrated_series is not None else df_predictions.get(prob_col_raw_draw)

        if calculate_ev and proba_ev_base is not None:
            log_msg_ev = f"  Calculando EV usando probs {'calibradas' if proba_draw_calibrated_series is not None else 'brutas'}..."
            logger.info(log_msg_ev)
            try:
                 common_ev_index = fixture_info.index.intersection(proba_ev_base.index)
                 if not common_ev_index.empty:
                      odds_draw=pd.to_numeric(fixture_info.loc[common_ev_index,odd_draw_col_name],errors='coerce')
                      prob_aligned=proba_ev_base.loc[common_ev_index]
                      valid_mask=odds_draw.notna() & prob_aligned.notna() & (odds_draw > 1)
                      prob_ok=prob_aligned[valid_mask]; odds_ok=odds_draw[valid_mask]
                      if not prob_ok.empty: ev_calc=(prob_ok*(odds_ok-1))-((1-prob_ok)*1); df_predictions.loc[prob_ok.index,'EV_Empate']=ev_calc; logger.info(f"  -> EV_Empate calculado para {len(prob_ok)} jogos.")
                      else: logger.info("  -> Nenhuma odd/prob válida para EV.")
                 else: logger.warning("  -> EV não calculado (índices EV não alinhados).")
            except Exception as e_ev: logger.warning(f"Aviso: Erro calcular EV: {e_ev}", exc_info=True)
        elif not calculate_ev: logger.info("  Cálculo EV pulado (sem odd).")
        else: logger.info("  Cálculo EV pulado (sem probs base).")

        # *** JOIN FINAL ***
        logger.debug(f"Predictor: Colunas em fixture_info ANTES do join: {list(fixture_info.columns)}")
        logger.debug(f"Predictor: Colunas em df_predictions ANTES do join: {list(df_predictions.columns)}") # Verifica se Prob_Empate está aqui
        logger.debug(f"Predictor: Índice de fixture_info: {fixture_info.index.dtype}, Primeiros 5: {fixture_info.index.tolist()[:5]}")
        logger.debug(f"Predictor: Índice de df_predictions: {df_predictions.index.dtype}, Primeiros 5: {df_predictions.index.tolist()[:5]}")

        df_results = fixture_info.join(df_predictions)

        if len(df_results) != len(fixture_info): logger.warning(f"Predictor: Tamanho do DF mudou após join!")

        logger.debug(f"Predictor: Colunas em df_results APÓS join: {list(df_results.columns)}") # Verifica se Prob_Empate está aqui
        if 'Time_Str' not in df_results.columns: logger.error("Predictor: Coluna 'Time_Str' PERDIDA após o join!")
        if 'Home' not in df_results.columns: logger.error("Predictor: Coluna 'Home' PERDIDA após o join!")
        if calibrator and prob_col_calib_draw not in df_results.columns: logger.error(f"Predictor: Coluna '{prob_col_calib_draw}' PERDIDA após o join!")
        elif not calibrator and prob_col_calib_draw in df_results.columns and not df_results[prob_col_calib_draw].isnull().all(): logger.warning(f"Predictor: Coluna '{prob_col_calib_draw}' existe mas calibrador era None?") # Inconsistência?
        else: logger.debug(f"Predictor: Amostra '{prob_col_calib_draw}' após join:\n{df_results[prob_col_calib_draw].head() if prob_col_calib_draw in df_results.columns else 'Coluna Ausente'}") # Log final


        logger.info("Previsões (com EV e probs Raw/Calib) geradas.")
        return df_results

    except Exception as e_pred:
         logger.error(f"Erro GERAL em make_predictions: {e_pred}", exc_info=True)
         return None