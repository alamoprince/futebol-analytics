# --- src/predictor.py ---
# ATUALIZADO para Calibrador e Limiar

import pandas as pd
import joblib
import os
import datetime
import numpy as np  # Necessário para isnan
import traceback
from config import CLASS_NAMES, ODDS_COLS, DEFAULT_EV_THRESHOLD
from typing import Optional, Any, List, Dict, Tuple
from logger_config import setup_logger

logger = setup_logger("PredictorApp")

# --- Função de Carregamento (MODIFICADA) ---
def load_model_scaler_features(model_path: str) -> Optional[Tuple[Any, Optional[Any], Optional[Any], float, Optional[List[str]], Optional[Dict], Optional[Dict], Optional[str]]]:
    """ Carrega modelo, scaler, CALIBRADOR, LIMIAR EV, features, params, métricas, timestamp. """
    try:
        load_obj = joblib.load(model_path)
        if isinstance(load_obj, dict) and 'model' in load_obj:
            model = load_obj['model']
            scaler = load_obj.get('scaler')
            calibrator = load_obj.get('calibrator')
            optimal_ev_threshold = load_obj.get('optimal_ev_threshold', DEFAULT_EV_THRESHOLD)
            feature_names = load_obj.get('feature_names')
            best_params = load_obj.get('params')
            eval_metrics = load_obj.get('eval_metrics')
            saved_timestamp = load_obj.get('save_timestamp')
            model_class_name = load_obj.get('model_class_name', 'N/A')
            fts = "N/A"
            try:
                mtime = os.path.getmtime(model_path)
                fts = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            except Exception:
                pass
            logger.info(f"Modelo carregado: {os.path.basename(model_path)} (Modif.: {fts})")
            logger.info(f"  Tipo Modelo: {model_class_name}")
            logger.info(f"  Calibrador: {'Sim' if calibrator else 'Não'}")
            logger.info(f"  Limiar EV Carregado: {optimal_ev_threshold:.4f}")
            return model, scaler, calibrator, optimal_ev_threshold, feature_names, best_params, eval_metrics, fts
        else:
            logger.warning(f"  Aviso: Formato antigo.")
            fts = "N/A"
            return load_obj, None, None, DEFAULT_EV_THRESHOLD, None, None, None, fts
    except FileNotFoundError:
        logger.error(f"Erro: Modelo não encontrado: '{model_path}'")
        return None
    except Exception as e:
        logger.error(f"Erro carregar modelo: {e}")
        logger.debug(traceback.format_exc())
        return None

# --- Função make_predictions (MODIFICADA) ---
def make_predictions(
    model: Any,
    scaler: Optional[Any],
    calibrator: Optional[Any],
    feature_names: List[str],
    X_fixture_prepared: pd.DataFrame, # DataFrame SÓ com features
    fixture_info: pd.DataFrame,       # DataFrame com infos originais (DEVE ter mesmo índice)
    odd_draw_col_name: str = ODDS_COLS.get('draw', 'Odd_D_FT')
    ) -> Optional[pd.DataFrame]:
    # ... (checks iniciais) ...
    if not X_fixture_prepared.index.equals(fixture_info.index):
         logger.error("Erro FATAL make_predictions: Índices de X_fixture_prepared e fixture_info NÃO COINCIDEM!")
         logger.debug(f"  Índice X_prep: {X_fixture_prepared.index[:5]}...")
         logger.debug(f"  Índice fix_inf: {fixture_info.index[:5]}...")
         return None # Não pode continuar se os índices não baterem

    try:
        X_pred = X_fixture_prepared[feature_names].copy() # Usa apenas as features
        if scaler:
            # ... (scaling) ...
            logger.info("  Aplicando scaler...")
            X_pred = pd.DataFrame(scaler.transform(X_pred.astype(float)), index=X_pred.index, columns=feature_names)


        if not hasattr(model, "predict_proba"): logger.error("Modelo sem predict_proba."); return None
        logger.info("  Calculando probs brutas...")
        predictions_proba_raw = model.predict_proba(X_pred)
        # ... (criação de df_predictions com probs raw/calib como antes) ...
        # ... (cálculo de EV como antes) ...
        # Mapa de colunas de probabilidade
        prob_cols_raw_map = {}
        class_names_local = CLASS_NAMES if CLASS_NAMES else ['NaoEmp', 'Empate']
        if len(model.classes_) == len(class_names_local): prob_cols_raw_map = {model.classes_[i]: f'ProbRaw_{class_names_local[i]}' for i in range(len(class_names_local))}
        else: prob_cols_raw_map = {c: f'ProbRaw_Class_{c}' for c in model.classes_}
        prob_cols_raw_names = list(prob_cols_raw_map.values())
        if len(prob_cols_raw_names) != predictions_proba_raw.shape[1]: logger.error("Erro mapear probs raw."); return None
        df_predictions = pd.DataFrame(predictions_proba_raw, columns=prob_cols_raw_names, index=X_pred.index) # Mantém o índice

        # Calibração
        prob_cols_calib_map = {raw_col: raw_col.replace("Raw_", "") for raw_col in prob_cols_raw_names} # Nomeia como Prob_Nao_Empate, Prob_Empate
        prob_col_calib_draw = prob_cols_calib_map.get(prob_cols_raw_names[1])
        proba_draw_calibrated_series = None
        if calibrator and prob_cols_raw_names[1] in df_predictions.columns:
            logger.info(f"  Aplicando calibrador...")
            try:
                 proba_draw_raw = df_predictions[prob_cols_raw_names[1]].values
                 proba_draw_calibrated_array = calibrator.predict(proba_draw_raw)
                 proba_draw_calibrated_series = pd.Series(proba_draw_calibrated_array, index=df_predictions.index)
                 # Adiciona colunas calibradas
                 for raw_col, calib_col in prob_cols_calib_map.items():
                      if raw_col == prob_cols_raw_names[1]: # Coluna do Empate (classe 1)
                           df_predictions[calib_col] = proba_draw_calibrated_series
                      else: # Outras classes (ex: Não Empate - classe 0)
                           # Recalcula baseado na prob do empate para garantir soma 1 (aproximadamente)
                           df_predictions[calib_col] = 1.0 - proba_draw_calibrated_series
                 logger.info("  -> Colunas Prob_* (Calibradas) adicionadas.")
            except Exception as e_calib_pred:
                 logger.warning(f"Aviso: Erro aplicar calibrador: {e_calib_pred}. Colunas Prob_* terão NaN.")
                 for col in prob_cols_calib_map.values(): df_predictions[col] = np.nan
        else:
            logger.info("  Calibrador não disponível/aplicável. Colunas Prob_* (Calibradas) terão NaN.")
            for col in prob_cols_calib_map.values(): df_predictions[col] = np.nan

        # Cálculo de EV (Usa proba_draw_calibrated_series se disponível)
        df_predictions['EV_Empate'] = np.nan
        calculate_ev = odd_draw_col_name in fixture_info.columns
        proba_ev_base = proba_draw_calibrated_series if proba_draw_calibrated_series is not None else df_predictions.get(prob_cols_raw_names[1]) # Fallback para prob bruta

        if calculate_ev and proba_ev_base is not None:
            logger.info(f"  Calculando Valor Esperado (EV) usando probs {'calibradas' if proba_draw_calibrated_series is not None else 'brutas'}...")
            try:
                 # Garante alinhamento para cálculo EV
                 common_ev_index = fixture_info.index.intersection(proba_ev_base.index)
                 if not common_ev_index.empty:
                      odds_draw = pd.to_numeric(fixture_info.loc[common_ev_index, odd_draw_col_name], errors='coerce')
                      prob_aligned = proba_ev_base.loc[common_ev_index]
                      valid_idx_mask = odds_draw.notna() & prob_aligned.notna() & (odds_draw > 1)
                      prob_ok = prob_aligned[valid_idx_mask]; odds_ok = odds_draw[valid_idx_mask]
                      if not prob_ok.empty:
                           ev_calc = (prob_ok * (odds_ok - 1)) - ((1 - prob_ok) * 1)
                           df_predictions.loc[prob_ok.index, 'EV_Empate'] = ev_calc
                           logger.info(f"  -> Coluna EV_Empate calculada para {len(prob_ok)} jogos.")
                      else: logger.info("  -> Nenhuma odd/prob válida para cálculo de EV.")
                 else: logger.warning("  -> EV não calculado (índices não alinhados com fixture_info para EV).")
            except Exception as e_ev: logger.warning(f"Aviso: Erro ao calcular EV: {e_ev}", exc_info=True)
        elif not calculate_ev: logger.info("  Cálculo de EV pulado (coluna odd ausente).")
        else: logger.info("  Cálculo de EV pulado (probs base ausentes).")


        # *** JOIN FINAL ***
        logger.debug(f"Predictor: Colunas em fixture_info ANTES do join: {list(fixture_info.columns)}")
        logger.debug(f"Predictor: Colunas em df_predictions ANTES do join: {list(df_predictions.columns)}")
        logger.debug(f"Predictor: Índice de fixture_info: {fixture_info.index.dtype}, Primeiros 5: {fixture_info.index.tolist()[:5]}")
        logger.debug(f"Predictor: Índice de df_predictions: {df_predictions.index.dtype}, Primeiros 5: {df_predictions.index.tolist()[:5]}")

        # Realiza o join usando o índice (que DEVE ser o mesmo)
        df_results = fixture_info.join(df_predictions)

        # Verifica se o join manteve o tamanho esperado
        if len(df_results) != len(fixture_info):
             logger.warning(f"Predictor: Tamanho do DF mudou após join! Antes: {len(fixture_info)}, Depois: {len(df_results)}")

        logger.debug(f"Predictor: Colunas em df_results APÓS join: {list(df_results.columns)}")
        if 'Time_Str' not in df_results.columns:
             logger.error("Predictor: Coluna 'Time_Str' PERDIDA após o join!")
        elif 'Home' not in df_results.columns:
             logger.error("Predictor: Coluna 'Home' PERDIDA após o join!")
        else:
             logger.debug(f"Predictor: Amostra Time_Str após join:\n{df_results['Time_Str'].head()}")
             logger.debug(f"Predictor: Amostra Home após join:\n{df_results['Home'].head()}")


        logger.info("Previsões (com EV e probs Raw/Calib) geradas.")
        return df_results

    except Exception as e_pred:
         logger.error(f"Erro GERAL em make_predictions: {e_pred}", exc_info=True)
         return None