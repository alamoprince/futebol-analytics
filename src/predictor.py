# --- src/predictor.py ---
# ATUALIZADO para Calibrador e Limiar

import pandas as pd
import joblib
import os
import datetime
import numpy as np # Necessário para isnan
import traceback
# Removido MODEL_SAVE_PATH daqui, caminho é passado como argumento
from config import CLASS_NAMES, ODDS_COLS, DEFAULT_EV_THRESHOLD #, MODEL_SAVE_PATH # Removido MODEL_SAVE_PATH daqui, caminho é passado como argumento               
from typing import Optional, Any, List, Dict, Tuple

# --- Função de Carregamento (MODIFICADA) ---
def load_model_scaler_features(model_path: str) -> Optional[Tuple[Any, Optional[Any], Optional[Any], float, Optional[List[str]], Optional[Dict], Optional[Dict], Optional[str]]]:
    """ Carrega modelo, scaler, CALIBRADOR, LIMIAR EV, features, params, métricas, timestamp. """
    try:
        load_obj = joblib.load(model_path)
        if isinstance(load_obj, dict) and 'model' in load_obj:
            model = load_obj['model']
            scaler = load_obj.get('scaler')
            calibrator = load_obj.get('calibrator')
            optimal_ev_threshold = load_obj.get('optimal_ev_threshold', DEFAULT_EV_THRESHOLD)  # Usa default do config se faltar
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
            print(f"Modelo carregado: {os.path.basename(model_path)} (Modif.: {fts})")
            print(f"  Tipo Modelo: {model_class_name}")
            print(f"  Calibrador: {'Sim' if calibrator else 'Não'}")
            print(f"  Limiar EV Carregado: {optimal_ev_threshold:.4f}")  # Mostra o limiar EV
            # ... (logs de features, metrics como antes, talvez mostre roi vs roi_thr05) ...
            return model, scaler, calibrator, optimal_ev_threshold, feature_names, best_params, eval_metrics, fts
        else:
            # Fallback formato antigo
            print(f"  Aviso: Formato antigo.")
            fts = "N/A"  # ... ;
            return load_obj, None, None, DEFAULT_EV_THRESHOLD, None, None, None, fts  # Retorna limiar default
    except FileNotFoundError:
        print(f"Erro: Modelo não encontrado: '{model_path}'")
        return None
    except Exception as e:
        print(f"Erro carregar modelo: {e}")
        traceback.print_exc()
        return None

# --- Função make_predictions (MODIFICADA) ---
def make_predictions(
    model: Any,
    scaler: Optional[Any],
    calibrator: Optional[Any],
    feature_names: List[str],
    X_fixture_prepared: pd.DataFrame,
    fixture_info: pd.DataFrame,
    odd_draw_col_name: str = ODDS_COLS.get('draw', 'Odd_D_FT')
    ) -> Optional[pd.DataFrame]:
    """ Realiza previsões, aplica scaler/CALIBRADOR, calcula EV. Retorna DF com múltiplas colunas de probs. """
    # ... (Verificações iniciais model, X_fixture_prepared, feature_names como antes) ...
    if model is None or X_fixture_prepared is None or feature_names is None: print("Erro make_preds: Args ausentes."); return None
    if X_fixture_prepared.empty: print("make_preds: Input vazio."); return None
    if not set(feature_names).issubset(X_fixture_prepared.columns): missing=list(set(feature_names)-set(X_fixture_prepared.columns)); print(f"Erro: Cols faltando: {missing}"); return None

    # Verifica se odd para EV existe em fixture_info
    calculate_ev = True
    if odd_draw_col_name not in fixture_info.columns:
        print(f"Aviso: Coluna Odd Empate '{odd_draw_col_name}' não em fixture_info. EV não será calculado.")
        calculate_ev = False

    print(f"Realizando previsões ({model.__class__.__name__}) para {len(X_fixture_prepared)} jogos...")
    try:
        X_pred = X_fixture_prepared[feature_names].copy()
        # 1. Scaling
        if scaler:
            print("  Aplicando scaler..."); X_pred = pd.DataFrame(scaler.transform(X_pred.astype(float)), index=X_pred.index, columns=feature_names)

        # 2. Probabilidades BRUTAS
        if not hasattr(model, "predict_proba"): print(f"Erro: Modelo sem predict_proba."); return None
        print("  Calculando probs brutas..."); predictions_proba_raw = model.predict_proba(X_pred)
        # Mapeia classes para nomes de coluna Raw (como antes)
        prob_cols_raw_map={}; class_names_local=CLASS_NAMES if CLASS_NAMES else ['NaoEmp','Empate'];
        if len(model.classes_) == len(class_names_local): prob_cols_raw_map = {model.classes_[i]: f'ProbRaw_{class_names_local[i]}' for i in range(len(class_names_local))}
        else: prob_cols_raw_map = {c: f'ProbRaw_Class_{c}' for c in model.classes_};
        prob_cols_raw_names = list(prob_cols_raw_map.values());
        if len(prob_cols_raw_names)!=predictions_proba_raw.shape[1]: print("Erro probs raw."); return None;
        df_predictions = pd.DataFrame(predictions_proba_raw, columns=prob_cols_raw_names, index=X_pred.index)

        # 3. Aplica Calibrador -> Cria colunas 'ProbCalib_...'
        prob_cols_calib_map = {raw_col: raw_col.replace("Raw_", "Calib_") for raw_col in prob_cols_raw_names}
        prob_col_calib_draw = prob_cols_calib_map.get(prob_cols_raw_names[1]) # Nome da coluna Calib Empate (assume classe 1 é a segunda)
        prob_col_calib_other = prob_cols_calib_map.get(prob_cols_raw_names[0])# Nome da coluna Calib NaoEmpate
        proba_draw_calibrated_series = None # Reseta

        if calibrator and prob_cols_raw_names[1] in df_predictions.columns:
            print(f"  Aplicando calibrador...");
            try:
                proba_draw_raw = df_predictions[prob_cols_raw_names[1]].values
                proba_draw_calibrated_array = calibrator.predict(proba_draw_raw)
                # ** Cria Série Pandas para manter o índice **
                proba_draw_calibrated_series = pd.Series(proba_draw_calibrated_array, index=df_predictions.index)

                # Adiciona colunas calibradas (com nomes Calib_)
                df_predictions[prob_col_calib_draw] = proba_draw_calibrated_series
                if prob_col_calib_other: # Garante que o nome da outra coluna existe
                      df_predictions[prob_col_calib_other] = 1.0 - proba_draw_calibrated_series
                print("  -> Colunas ProbCalib_* adicionadas.")

            except Exception as e_calib_pred:
                print(f"Aviso: Erro aplicar calibrador: {e_calib_pred}. Colunas ProbCalib_* não geradas.")
                # Adiciona colunas Calib como NaN se falhar
                for col in prob_cols_calib_map.values(): df_predictions[col] = np.nan
        else:
             print("  Calibrador não disponível/aplicável. Colunas ProbCalib_* terão NaN.")
             # Adiciona colunas Calib como NaN
             for col in prob_cols_calib_map.values(): df_predictions[col] = np.nan

        # 4. Calcular EV (usando a SÉRIE calibrada)
        df_predictions['EV_Empate'] = np.nan
        if calculate_ev and proba_draw_calibrated_series is not None: # Usa a SÉRIE que tem índice
             print("  Calculando Valor Esperado (EV)...")
             try:
                 # Usa o índice da SÉRIE para alinhar com fixture_info
                 common_ev_index = fixture_info.index.intersection(proba_draw_calibrated_series.index)
                 if not common_ev_index.empty:
                      odds_draw = pd.to_numeric(fixture_info.loc[common_ev_index, odd_draw_col_name], errors='coerce')
                      prob_calib_aligned = proba_draw_calibrated_series.loc[common_ev_index] # Pega probs alinhadas

                      valid_idx_mask = odds_draw.notna() & prob_calib_aligned.notna() & (odds_draw > 1)
                      prob_ok = prob_calib_aligned[valid_idx_mask]
                      odds_ok = odds_draw[valid_idx_mask]

                      if not prob_ok.empty:
                          ev_calc = (prob_ok * (odds_ok - 1)) - ((1 - prob_ok) * 1)
                          # Atribui de volta usando o índice correto (das máscaras)
                          df_predictions.loc[prob_ok.index, 'EV_Empate'] = ev_calc
                          print(f"  -> Coluna EV_Empate calculada para {len(prob_ok)} jogos.")
                      else: print("  -> Nenhuma odd/prob válida para cálculo de EV.")
                 else: print("  -> EV não calculado (índices não alinhados com fixture_info).")

             except Exception as e_ev: print(f"Aviso: Erro ao calcular EV: {e_ev}", exc_info=True) # Log mais detalhado
        elif not calculate_ev: print("  Cálculo de EV pulado (coluna odd ausente).")
        else: print("  Cálculo de EV pulado (probs calibradas ausentes).")

        # 5. Junta com info original
        common_join_index = fixture_info.index.intersection(df_predictions.index)
        if len(common_join_index) != len(df_predictions): print("Aviso: Nem todas previsões juntadas.")
        df_results = fixture_info.loc[common_join_index].join(df_predictions.loc[common_join_index])

        print("Previsões (com EV e probs Raw/Calib) geradas.")
        return df_results

    except Exception as e: print(f"Erro GERAL make_predictions: {e}"); traceback.print_exc(); return None