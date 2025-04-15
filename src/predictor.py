# --- src/predictor.py ---
# ATUALIZADO para Calibrador e Limiar

import pandas as pd
import joblib
import os
import datetime
import numpy as np # Necessário para isnan
import traceback
# Removido MODEL_SAVE_PATH daqui, caminho é passado como argumento
from config import CLASS_NAMES # Mantém CLASS_NAMES
from typing import Optional, Any, List, Dict, Tuple

# --- Função de Carregamento (MODIFICADA) ---
def load_model_scaler_features(model_path: str) -> Optional[Tuple[Any, Optional[Any], Optional[Any], float, Optional[List[str]], Optional[Dict], Optional[Dict], Optional[str]]]:
    """ Carrega modelo, scaler, CALIBRADOR, LIMIAR, features, params, métricas (inclui roi_thr05), timestamp. """
    try:
        load_obj = joblib.load(model_path)
        if isinstance(load_obj, dict) and 'model' in load_obj:
            # ... (carrega model, scaler, calibrator, threshold, features, params, metrics, timestamp, model_class_name como antes) ...
            model=load_obj['model']; scaler=load_obj.get('scaler'); calibrator=load_obj.get('calibrator'); optimal_threshold=load_obj.get('optimal_threshold', 0.5); feature_names=load_obj.get('feature_names'); best_params=load_obj.get('best_params'); eval_metrics=load_obj.get('eval_metrics'); saved_timestamp=load_obj.get('save_timestamp'); model_class_name=load_obj.get('model_class_name', 'N/A'); fts="N/A"
            try: mtime = os.path.getmtime(model_path); fts = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            except Exception: pass
            print(f"Modelo carregado: {os.path.basename(model_path)} (Modif.: {fts})")
            # ... (logs de tipo, calibrador, limiar, features como antes) ...
            print(f"  Tipo Modelo: {model_class_name}"); print(f"  Calibrador: {'Sim' if calibrator else 'Não'}"); print(f"  Limiar: {optimal_threshold:.4f}")
            # Log das MÉTRICAS carregadas (incluindo as novas se existirem)
            if eval_metrics:
                f1_05 = eval_metrics.get('f1_score_draw', 'N/A')
                roi_opt = eval_metrics.get('roi', 'N/A')
                roi_05 = eval_metrics.get('roi_thr05', 'N/A') # <<< Pega a métrica do limiar 0.5
                f1_05_str = f"{f1_05:.3f}" if isinstance(f1_05, (int,float)) else f1_05
                roi_opt_str = f"{roi_opt:.2f}%" if isinstance(roi_opt, (int,float)) and not np.isnan(roi_opt) else roi_opt if roi_opt is not None else "N/A"
                roi_05_str = f"{roi_05:.2f}%" if isinstance(roi_05, (int,float)) and not np.isnan(roi_05) else roi_05 if roi_05 is not None else "N/A"
                print(f"  Métricas Chave (F1@0.5: {f1_05_str}, ROI@Opt: {roi_opt_str}, ROI@0.5: {roi_05_str})") # <<< Log atualizado

            return model, scaler, calibrator, optimal_threshold, feature_names, best_params, eval_metrics, fts
        else: # Fallback formato antigo
            print(f"  Aviso: Formato antigo detectado."); fts="N/A"
            try: mtime = os.path.getmtime(model_path); fts = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S');
            except: pass
            return load_obj, None, None, 0.5, None, None, None, fts
    # ... (resto do try/except como antes) ...
    except FileNotFoundError: print(f"Erro: Modelo não encontrado: '{model_path}'"); return None
    except Exception as e: print(f"Erro carregar modelo: {e}"); traceback.print_exc(); return None

# --- Função make_predictions (MODIFICADA) ---
def make_predictions(
    model: Any,
    scaler: Optional[Any],
    calibrator: Optional[Any], # <<< NOVO PARÂMETRO
    # optimal_threshold: float, # Limiar não é usado diretamente aqui, mas no filtro posterior
    feature_names: List[str],
    X_fixture_prepared: pd.DataFrame,
    fixture_info: pd.DataFrame # DataFrame original com info dos jogos
    ) -> Optional[pd.DataFrame]:
    """
    Realiza previsões, aplica scaler e CALIBRADOR.

    Args:
        model: O modelo treinado.
        scaler: O scaler ajustado (ou None).
        calibrator: O calibrador treinado (IsotonicRegression ou None).
        feature_names: Lista de nomes das features usadas pelo modelo.
        X_fixture_prepared: DataFrame com features preparadas para os jogos futuros.
        fixture_info: DataFrame com informações originais dos jogos (para juntar resultados).

    Returns:
        DataFrame com informações do jogo, probabilidades brutas e CALIBRADAS, ou None se erro.
    """
    if model is None or X_fixture_prepared is None or feature_names is None:
        print("Erro make_predictions: Modelo, Dados ou Features ausentes.")
        return None
    if X_fixture_prepared.empty:
         print("make_predictions: DataFrame de entrada vazio.")
         # Retorna um DF vazio com as colunas esperadas? Ou None? Vamos retornar None.
         return None
    # Garante que todas as features estão presentes
    if not set(feature_names).issubset(X_fixture_prepared.columns):
        missing_cols = list(set(feature_names) - set(X_fixture_prepared.columns))
        print(f"Erro make_predictions: Colunas não correspondem! Faltando: {missing_cols}")
        return None

    print(f"Realizando previsões ({model.__class__.__name__}) para {len(X_fixture_prepared)} jogos...")
    try:
        # Seleciona apenas as features na ordem correta
        X_pred = X_fixture_prepared[feature_names].copy()

        # 1. Aplica Scaler (se existir)
        if scaler:
            print("  Aplicando scaler carregado...")
            try:
                X_pred_scaled_np = scaler.transform(X_pred.astype(float))
                X_pred = pd.DataFrame(X_pred_scaled_np, index=X_pred.index, columns=feature_names)
            except Exception as e_scale:
                print(f"Aviso: Erro ao aplicar scaler ({e_scale}). Usando dados não escalados.")

        # 2. Obtem Probabilidades BRUTAS
        if not hasattr(model, "predict_proba"):
             print(f"Erro: Modelo {model.__class__.__name__} não tem predict_proba.")
             return None # Não podemos continuar sem probabilidades

        print("  Calculando probabilidades brutas...")
        predictions_proba_raw = model.predict_proba(X_pred)

        # Cria colunas de probabilidade bruta
        prob_cols_raw = [f'ProbRaw_{CLASS_NAMES[i]}' for i in range(len(CLASS_NAMES))] if len(model.classes_) == len(CLASS_NAMES) else [f'ProbRaw_Class_{c}' for c in model.classes_]
        if len(prob_cols_raw) != predictions_proba_raw.shape[1]:
             print(f"Erro: Número de colunas de prob ({predictions_proba_raw.shape[1]}) diferente de CLASS_NAMES ({len(CLASS_NAMES)}).")
             # Usar classes do modelo como fallback
             prob_cols_raw = [f'ProbRaw_Class_{c}' for c in model.classes_]
             if len(prob_cols_raw) != predictions_proba_raw.shape[1]: # Se ainda der erro
                  print("Erro fatal: Inconsistência nas classes do modelo.")
                  return None

        df_predictions = pd.DataFrame(predictions_proba_raw, columns=prob_cols_raw, index=X_pred.index)

        # 3. Aplica Calibrador (se existir)
        if calibrator and prob_cols_raw[1] in df_predictions.columns: # Assume que a classe 1 (Empate) é a segunda
            print(f"  Aplicando calibrador ({calibrator.__class__.__name__})...")
            try:
                # Pega a probabilidade bruta da classe positiva (Empate)
                proba_draw_raw = df_predictions[prob_cols_raw[1]].values
                # Prevê com o calibrador
                proba_draw_calibrated = calibrator.predict(proba_draw_raw)
                # Adiciona como nova coluna (nome sem sufixo "Raw")
                prob_col_calibrated_draw = prob_cols_raw[1].replace("Raw_", "") # Ex: Prob_Empate
                df_predictions[prob_col_calibrated_draw] = proba_draw_calibrated
                # Calcula a prob da outra classe
                prob_col_calibrated_other = prob_cols_raw[0].replace("Raw_", "") # Ex: Prob_Nao_Empate
                df_predictions[prob_col_calibrated_other] = 1.0 - proba_draw_calibrated

            except Exception as e_calib_pred:
                print(f"Aviso: Erro ao aplicar calibrador ({e_calib_pred}). Probabilidades calibradas não geradas.")
                 # Adiciona colunas calibradas como NaN ou cópia das brutas? Melhor deixar como ausentes.
                for col in [prob_col_calibrated_other]: df_predictions[col] = np.nan
                

        else:
            print("  Calibrador não disponível ou coluna de prob bruta não encontrada. Usando probs brutas.")
             # Renomeia colunas brutas para nomes padrão (sem "Raw_") para consistência
            rename_map = {raw_col: raw_col.replace("Raw_", "") for raw_col in prob_cols_raw}
            df_predictions.rename(columns=rename_map, inplace=True)

            for i, raw_col in enumerate(prob_cols_raw): 
                  df_predictions[prob_col_calibrated_other[i]] = df_predictions[raw_col]
                

        # 4. Junta com informações originais do jogo
        # Garante que fixture_info tem o mesmo índice que df_predictions
        common_join_index = fixture_info.index.intersection(df_predictions.index)
        if len(common_join_index) != len(df_predictions):
             print("Aviso: Nem todas as previsões puderam ser juntadas com fixture_info (índices diferentes).")
             # Considera reindexar ou apenas juntar o que for possível? Vamos juntar o possível.
             fixture_info_aligned = fixture_info.loc[common_join_index]
             df_predictions_aligned = df_predictions.loc[common_join_index]
             df_results = fixture_info_aligned.join(df_predictions_aligned)
        else:
             df_results = fixture_info.join(df_predictions) # Junta tudo

        print("Previsões (com probabilidades possivelmente calibradas) geradas com sucesso.")
        return df_results

    except Exception as e:
        print(f"Erro GERAL em make_predictions: {e}")
        import traceback
        traceback.print_exc()
        return None