import pandas as pd
import joblib
import os
import datetime
from config import MODEL_SAVE_PATH, CLASS_NAMES
from typing import Optional, Any, List, Dict, Tuple

# --- Função de Carregamento (Adaptada para mostrar nome do modelo carregado) ---
def load_model_scaler_features(model_path: str = MODEL_SAVE_PATH) -> Optional[Tuple[Any, Optional[Any], Optional[List[str]], Optional[Dict], Optional[Dict], Optional[str]]]:
    """Carrega o MELHOR modelo salvo, scaler, features, params, métricas e timestamp."""
    try:
        load_obj = joblib.load(model_path)
        if isinstance(load_obj, dict) and 'model' in load_obj:
            model = load_obj['model']
            scaler = load_obj.get('scaler'); feature_names = load_obj.get('feature_names')
            best_params = load_obj.get('best_params'); eval_metrics = load_obj.get('eval_metrics')
            saved_timestamp = load_obj.get('save_timestamp')
            model_class_name = load_obj.get('model_class_name', 'N/A') # Pega nome da classe salva
            try: mtime = os.path.getmtime(model_path); fts = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            except Exception: fts = "N/A"

            print(f"Modelo carregado: {model_path} (Modificado: {fts})")
            print(f"  Tipo do Modelo Salvo: {model_class_name}") # Mostra qual modelo foi o melhor
            if feature_names: print(f"  Features ({len(feature_names)}): {feature_names}")
            if best_params: print(f"  Params: {best_params}")
            if eval_metrics: print(f"  Métricas (Acc: {eval_metrics.get('accuracy', -1):.3f}, F1_Draw: {eval_metrics.get('f1_score_draw', -1):.3f}, ROI: {eval_metrics.get('roi', 'N/A'):.2f}%)")
            return model, scaler, feature_names, best_params, eval_metrics, fts
        else: # Modelo antigo
             print(f"  Aviso: Formato antigo detectado."); # ... (resto como antes) ...
             try: mtime = os.path.getmtime(model_path); fts = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
             except: fts = "N/A"
             return load_obj, None, None, None, None, fts
    except FileNotFoundError: print(f"Erro: Modelo não encontrado: '{model_path}'"); return None
    except Exception as e: print(f"Erro ao carregar modelo: {e}"); return None

# --- Função make_predictions (Sem alterações) ---
def make_predictions(model: Any, scaler: Optional[Any], feature_names: List[str],
                     X_fixture_prepared: pd.DataFrame, fixture_info: pd.DataFrame
                     ) -> Optional[pd.DataFrame]:
    # ... (código idêntico ao da V7) ...
    if model is None or X_fixture_prepared is None or feature_names is None: return None
    if not set(feature_names).issubset(X_fixture_prepared.columns): print("Erro: Colunas não correspondem!"); return None
    print(f"Realizando previsões ({model.__class__.__name__}) para {len(X_fixture_prepared)} jogos...")
    try:
        X_pred = X_fixture_prepared[feature_names].copy()
        if scaler: # Aplica scaler SE ele foi salvo com o melhor modelo
            print("  Aplicando scaler carregado...")
            try: X_pred_scaled = scaler.transform(X_pred); X_pred = pd.DataFrame(X_pred_scaled, index=X_pred.index, columns=feature_names)
            except Exception as e_scale: print(f"Aviso: Erro no scaler ({e_scale}). Usando dados não escalados.")
        print("  Calculando probabilidades (Não Empate, Empate)...")
        predictions_proba = model.predict_proba(X_pred)
        prob_cols = [f'Prob_{CLASS_NAMES[i]}' for i in model.classes_] if len(model.classes_) == len(CLASS_NAMES) else [f'Prob_Class_{c}' for c in model.classes_]
        df_predictions = pd.DataFrame(predictions_proba, columns=prob_cols, index=X_pred.index)
        df_results = fixture_info.join(df_predictions); print("Previsões geradas com sucesso.")
        return df_results
    except Exception as e: print(f"Erro predição: {e}"); import traceback; traceback.print_exc(); return None