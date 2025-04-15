# --- src/model_trainer.py ---
# Código Completo com Correções e Debug Detalhado na Seleção

import pandas as pd
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV # Embora não usado ativamente na lógica atual
try:
    import lightgbm as lgb
    LGBMClassifier = lgb.LGBMClassifier
except ImportError:
    print("AVISO: LightGBM não instalado.")
    lgb = None
    LGBMClassifier = None
from sklearn.metrics import (accuracy_score, classification_report, log_loss,
                             precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, brier_score_loss)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import os
import datetime
import numpy as np
import logging
import traceback # Import traceback for detailed error logging

# Tenta importar funções e configs necessárias
try:
    from config import (
        RANDOM_STATE, TEST_SIZE, MODEL_CONFIG, CLASS_NAMES,
        BEST_F1_MODEL_SAVE_PATH, BEST_ROI_MODEL_SAVE_PATH,
        CROSS_VALIDATION_SPLITS, N_JOBS_GRIDSEARCH, FEATURE_COLUMNS,
        ODDS_COLS, BEST_MODEL_METRIC
    )
    from typing import Any, Optional, Dict, Tuple, List, Callable
except ImportError as e:
     print(f"Erro crítico import config/typing em model_trainer.py: {e}")
     raise

logging.basicConfig(level=logging.INFO, format='%(asctime)s - TRAINER - %(levelname)s - %(message)s')

# --- Função ROI (Definida localmente) ---
def roi(y_test: pd.Series, y_pred: np.ndarray, X_test_odds_aligned: Optional[pd.DataFrame], odd_draw_col_name: str) -> Optional[float]:
        """Calcula o Retorno Sobre Investimento (ROI) para apostas em empate."""
        # (Código da função roi como na resposta anterior, incluindo verificações)
        if X_test_odds_aligned is None or odd_draw_col_name not in X_test_odds_aligned.columns:
            logging.warning(f"ROI(Base): Dados de odds ('{odd_draw_col_name}') ausentes.")
            return None
        common_index = y_test.index.intersection(X_test_odds_aligned.index)
        if len(common_index) != len(y_test): logging.warning("ROI(Base): Índices não batem."); return None
        y_test_common = y_test.loc[common_index]
        try: y_pred_series = pd.Series(y_pred, index=y_test.index); y_pred_common = y_pred_series.loc[common_index]
        except ValueError: logging.error("ROI(Base): Erro alinhar y_pred."); return None
        predicted_draws_indices = common_index[y_pred_common == 1]
        num_bets = len(predicted_draws_indices)
        if num_bets == 0: return 0.0
        actuals = y_test_common.loc[predicted_draws_indices]
        odds = pd.to_numeric(X_test_odds_aligned.loc[predicted_draws_indices, odd_draw_col_name], errors='coerce')
        profit = 0; valid_bets = 0
        for idx in predicted_draws_indices:
            odd_d = odds.loc[idx]
            if pd.notna(odd_d) and odd_d > 0:
                profit += (odd_d - 1) if actuals.loc[idx] == 1 else -1
                valid_bets += 1
        if valid_bets == 0: logging.warning("ROI(Base) 0: Nenhuma aposta c/ odd válida."); return 0.0
        return (profit / valid_bets) * 100


# --- Função ROI com Limiar (calculate_roi_with_threshold) ---
def calculate_roi_with_threshold(
    y_true: pd.Series, y_proba: np.ndarray, threshold: float,
    odds_data: Optional[pd.DataFrame], odd_col_name: str
    ) -> Tuple[Optional[float], int, Optional[float]]:
    """Calcula ROI, número de apostas e lucro baseado em um limiar de probabilidade."""
    # (Código da função como na resposta anterior, incluindo verificações e logs)
    profit = None; roi_value = None; num_bets = 0
    if odds_data is None or odd_col_name not in odds_data.columns: logging.warning(f"ROI Thr: Dados odds ('{odd_col_name}') ausentes."); return roi_value, num_bets, profit
    try:
        common_index = y_true.index.intersection(odds_data.index)
        if len(common_index) != len(y_true): logging.warning("ROI Thr: Índices não batem."); return roi_value, num_bets, profit
        y_true_common = y_true.loc[common_index]; odds_common = pd.to_numeric(odds_data.loc[common_index, odd_col_name], errors='coerce')
        try: y_proba_common = pd.Series(y_proba, index=y_true.index).loc[common_index]
        except ValueError: logging.error("ROI Thr: Erro alinhar y_proba."); return None, 0, None
        bet_indices = common_index[y_proba_common > threshold]; num_bets = len(bet_indices)
        if num_bets == 0: return 0.0, num_bets, 0.0
        actuals = y_true_common.loc[bet_indices]; odds_selected = odds_common.loc[bet_indices]
        profit_calc = 0; valid_bets_count = 0
        for idx in bet_indices:
            odd_d = odds_selected.loc[idx]
            if pd.notna(odd_d) and odd_d > 0: profit_calc += (odd_d - 1) if actuals.loc[idx] == 1 else -1; valid_bets_count += 1
        profit = profit_calc
        if valid_bets_count > 0: roi_value = (profit / valid_bets_count) * 100
        else: roi_value = 0.0
        return roi_value, num_bets, profit
    except KeyError as e: logging.error(f"ROI Thr: Erro Key - {e}", exc_info=True); return None, 0, None
    except Exception as e: logging.error(f"ROI Thr: Erro - {e}", exc_info=True); return None, 0, None


# --- Função scale_features ---
def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame, scaler_type='standard') -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
    """Aplica scaling aos dados de treino e teste."""
    # (Código como na resposta anterior, incluindo verificações)
    X_train = X_train.copy(); X_test = X_test.copy()
    try: X_train = X_train.astype(float); X_test = X_test.astype(float)
    except ValueError as e: logging.error(f"Erro converter p/ float scaling: {e}"); raise
    if scaler_type == 'minmax': scaler = MinMaxScaler()
    else: scaler = StandardScaler()
    logging.info(f"  Aplicando {scaler.__class__.__name__}...");
    try:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    except Exception as e: logging.error(f"Erro {scaler.__class__.__name__}.fit/transform: {e}", exc_info=True); raise
    X_train_scaled_df = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)
    logging.info("  Scaling concluído.")
    return X_train_scaled_df, X_test_scaled_df, scaler


# --- Função Principal de Treinamento (COM TODAS AS CORREÇÕES E DEBUG) ---
def train_evaluate_and_save_best_models(
    X: pd.DataFrame, y: pd.Series,
    X_test_with_odds: Optional[pd.DataFrame] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    scaler_type: str = 'standard',
    odd_draw_col_name: str = ODDS_COLS.get('draw', 'Odd_D_FT'),
    calibration_method: str = 'isotonic'
    ) -> bool:
    """
    Treina, CALIBRA, otimiza LIMIAR, avalia modelos. Salva os 2 melhores (F1 e ROI).
    """
    if X is None or y is None: logging.error("Dados X ou y são None."); return False
    if not MODEL_CONFIG: logging.error("MODEL_CONFIG vazio."); return False
    available_models = {name: config for name, config in MODEL_CONFIG.items()
                        if not (name == 'LGBMClassifier' and lgb is None)}
    if not available_models: logging.error("Nenhum modelo válido/disponível."); return False

    feature_names = list(X.columns)
    all_results = []

    logging.info(f"--- Treinando/Calibrando {len(available_models)} Modelos ---")
    start_time_total = time.time()

    # --- Divisão Tripla dos Dados ---
    logging.info("Dividindo dados em Treino, Validação e Teste...")
    val_size = 0.20; test_size_final = TEST_SIZE; train_val_size_temp = 1.0 - test_size_final
    # Proteção contra divisão por zero se test_size_final for 1.0
    if train_val_size_temp <= 0: logging.error("TEST_SIZE inválido (>= 1.0)."); return False
    val_size_relative = val_size / train_val_size_temp

    try:
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size_final, random_state=RANDOM_STATE, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size_relative, random_state=RANDOM_STATE, stratify=y_train_val)
        logging.info(f"Split: Treino={len(X_train)}, Validação={len(X_val)}, Teste={len(X_test)}")

        # Prepara dados com odds para Validação e Teste (COM DEBUG DETALHADO)
        X_val_odds = None; X_test_odds = None
        logging.info(f"--- DEBUG: Verificando X_test_with_odds (Tipo: {type(X_test_with_odds)}) ---")
        if X_test_with_odds is not None and not X_test_with_odds.empty:
            logging.info(f"DEBUG: X_test_with_odds shape {X_test_with_odds.shape}. Col Odd Draw ('{odd_draw_col_name}'): {odd_draw_col_name in X_test_with_odds.columns}")
            if odd_draw_col_name not in X_test_with_odds.columns:
                logging.error(f"Erro crítico: Coluna de Odd '{odd_draw_col_name}' NÃO encontrada em X_test_with_odds.")
                # Decide se continua sem ROI ou para
                # return False # Opção mais segura

            # Alinha VALIDAÇÃO (usando INTERSECÇÃO)
            logging.info("DEBUG: Tentando alinhar odds de VALIDAÇÃO...")
            common_index_val = X_val.index.intersection(X_test_with_odds.index)
            logging.info(f"DEBUG: Índices Val: {len(X_val)}, Histórico c/ Odds: {len(X_test_with_odds.index)}, Intersecção: {len(common_index_val)}")
            if len(common_index_val) > 0:
                 try:
                     # Pega APENAS a coluna de odd necessária para evitar erros de coluna ausente
                     if odd_draw_col_name in X_test_with_odds.columns:
                          X_val_odds = X_test_with_odds.loc[common_index_val, [odd_draw_col_name]].copy()
                          logging.info(f"DEBUG: X_val_odds CRIADO com shape {X_val_odds.shape}.")
                          if len(common_index_val) < len(X_val): logging.warning(f"DEBUG: {len(X_val) - len(common_index_val)} índices de VALIDAÇÃO não encontrados em X_test_with_odds.")
                     else: logging.error("DEBUG: Coluna de Odd Draw não existe em X_test_with_odds para loc de VAL.")
                 except Exception as e_loc_val: logging.error(f"DEBUG: Erro .loc X_val_odds: {e_loc_val}", exc_info=True)
            else: logging.error("DEBUG: NENHUMA INTERSECÇÃO entre Validação e Dados com Odds.")

            # Alinha TESTE (usando INTERSECÇÃO)
            logging.info("DEBUG: Tentando alinhar odds de TESTE...")
            common_index_test = X_test.index.intersection(X_test_with_odds.index)
            logging.info(f"DEBUG: Índices Teste: {len(X_test)}, Histórico c/ Odds: {len(X_test_with_odds.index)}, Intersecção: {len(common_index_test)}")
            if len(common_index_test) > 0:
                 try:
                     if odd_draw_col_name in X_test_with_odds.columns:
                         X_test_odds = X_test_with_odds.loc[common_index_test, [odd_draw_col_name]].copy()
                         logging.info(f"DEBUG: X_test_odds CRIADO com shape {X_test_odds.shape}.")
                         if len(common_index_test) < len(X_test): logging.warning(f"DEBUG: {len(X_test) - len(common_index_test)} índices de TESTE não encontrados em X_test_with_odds.")
                     else: logging.error("DEBUG: Coluna de Odd Draw não existe em X_test_with_odds para loc de TESTE.")
                 except Exception as e_loc_test: logging.error(f"DEBUG: Erro .loc X_test_odds: {e_loc_test}", exc_info=True)
            else: logging.error("DEBUG: NENHUMA INTERSECÇÃO entre Teste e Dados com Odds.")
        else:
             logging.warning("DEBUG: X_test_with_odds é None ou vazio. ROI não será calculado.")

    except Exception as e_split3:
        logging.error(f"Erro durante divisão tripla dos dados: {e_split3}", exc_info=True)
        return False

    # --- Loop pelos modelos disponíveis ---
    for i, (model_name, config) in enumerate(available_models.items()):
        # ... (Bloco try para obter model_class como antes) ...
        status_text = f"Modelo {i+1}/{len(available_models)}: {model_name}"; logging.info(f"\n--- {status_text} ---")
        if progress_callback: progress_callback(i, len(available_models), status_text)
        start_time_model = time.time();
        try: model_class = eval(model_name)
        except NameError: logging.error(f"Classe '{model_name}' não encontrada."); continue
        except Exception as e_eval_cls: logging.error(f"Erro obter classe '{model_name}': {e_eval_cls}"); continue

        model_kwargs = config.get('model_kwargs', {}); param_grid = config.get('param_grid', {}); needs_scaling = config.get('needs_scaling', False)
        X_train_model, X_val_model, X_test_model = X_train.copy(), X_val.copy(), X_test.copy(); current_scaler = None

        # --- Scaling Condicional ---
        if needs_scaling:
            logging.info(f"  Modelo '{model_name}' requer scaling...");
            try:
                # ... (código de scaling como antes, usando fit no treino, transform em val/teste) ...
                if scaler_type == 'minmax': scaler = MinMaxScaler()
                else: scaler = StandardScaler()
                logging.info(f"  Aplicando {scaler.__class__.__name__} (fit no treino)...")
                X_train_model_np = scaler.fit_transform(X_train_model.astype(float))
                X_val_model_np = scaler.transform(X_val_model.astype(float))
                X_test_model_np = scaler.transform(X_test_model.astype(float))
                X_train_model = pd.DataFrame(X_train_model_np, index=X_train.index, columns=feature_names)
                X_val_model = pd.DataFrame(X_val_model_np, index=X_val.index, columns=feature_names)
                X_test_model = pd.DataFrame(X_test_model_np, index=X_test.index, columns=feature_names)
                current_scaler = scaler
                logging.info(f"    ==> DEBUG: Scaling para {model_name} concluído.")
            except Exception as scale_err:
                logging.error(f"  ERRO GRAVE scaling {model_name}: {scale_err}", exc_info=True)
                logging.error(f"  -> PULANDO {model_name}.")
                continue # PULA para o próximo modelo
        else:
            logging.info("  Scaling não requerido.")

        logging.info(f"  ==> DEBUG: CONTINUANDO após bloco de scaling para {model_name}") # LOG DE CHECKPOINT

        # --- Treinamento Modelo Principal ---
        logging.info(f"  ==> DEBUG: Iniciando treino principal para {model_name}")
        model_instance_trained = None; current_best_params = model_kwargs.copy()
        if param_grid:
             try: # Try para criar GridSearchCV
                 search_cv = GridSearchCV(estimator=model_class(**model_kwargs), param_grid=param_grid, cv=CROSS_VALIDATION_SPLITS, n_jobs=N_JOBS_GRIDSEARCH, scoring='f1', verbose=0, error_score='raise')
                 logging.info(f"    ==> DEBUG: Instância GridSearchCV para {model_name} criada.")
             except Exception as e_grid_init: logging.error(f"    ERRO GRAVE criar GridSearchCV {model_name}: {e_grid_init}", exc_info=True); logging.error(f"    -> PULANDO {model_name}."); continue

             logging.info(f"  Iniciando GridSearchCV (scoring=f1)...");
             try: # Try para fit do GridSearchCV
                 if progress_callback: progress_callback(i, len(available_models), f"Ajustando {model_name} (CV)...")
                 search_cv.fit(X_train_model, y_train)
                 model_instance_trained = search_cv.best_estimator_; current_best_params = search_cv.best_params_
                 logging.info(f"    ==> DEBUG: GridSearchCV.fit para {model_name} concluído."); logging.info(f"    Melhor CV (f1): {search_cv.best_score_:.4f}"); logging.info(f"    Params: {current_best_params}")
             except Exception as e_cv: logging.error(f"    Erro GRAVE GridSearchCV.fit {model_name}: {e_cv}", exc_info=True); logging.warning(f"    -> Tentando fallback..."); model_instance_trained = None
        else: logging.info(f"  Sem grid. Treinando com params padrão: {model_kwargs}")

        # Fallback/Treino Padrão
        if model_instance_trained is None:
             logging.info(f"  ==> DEBUG: Tentando treino padrão/fallback.")
             try:
                 # Determina qual instância usar para o fallback
                 if 'search_cv' in locals() and hasattr(search_cv, 'estimator') and model_name not in ['GaussianNB', 'KNeighborsClassifier']: # Reusa estimador base se CV falhou (exceto para alguns modelos onde pode ser problemático)
                     current_model_fallback = search_cv.estimator
                     logging.info(f"    -> Usando estimador base do GridSearchCV falho.")
                 else:
                     current_model_fallback = model_class(**model_kwargs)
                     logging.info(f"    -> Usando nova instância com kwargs.")

                 if progress_callback: progress_callback(i, len(available_models), f"Ajustando {model_name}...")
                 model_instance_trained = current_model_fallback.fit(X_train_model, y_train)
                 logging.info(f"    ==> DEBUG: Treino padrão/fallback concluído.")
             except Exception as e_fit: logging.error(f"    Erro GRAVE treino padrão/fallback {model_name}: {e_fit}", exc_info=True); logging.error(f"    -> PULANDO {model_name}."); continue

        if model_instance_trained is None: logging.error(f"ERRO CRÍTICO: {model_name} NULO após treino."); continue

        # --- Calibração ---
        logging.info(f"  Calibrando probs com {calibration_method}...")
        calibrator = None; y_proba_val_raw_draw = None # Reseta para este modelo
        if hasattr(model_instance_trained, "predict_proba"):
             try:
                 y_proba_val_raw_full = model_instance_trained.predict_proba(X_val_model)
                 if y_proba_val_raw_full.shape[1] > 1:
                      y_proba_val_raw_draw = y_proba_val_raw_full[:, 1]
                      if calibration_method == 'isotonic': calibrator = IsotonicRegression(out_of_bounds='clip')
                      else: calibrator = IsotonicRegression(out_of_bounds='clip'); logging.warning("Usando Isotonic como default.")
                      calibrator.fit(y_proba_val_raw_draw, y_val); logging.info(f"  Calibrador treinado.")
                 else: logging.warning(f"    Predict_proba Val shape {y_proba_val_raw_full.shape}. Calibração pulada.")
             except Exception as e_calib: logging.error(f"  Erro calibração: {e_calib}", exc_info=True); calibrator = None
        else: logging.warning(f"  {model_name} sem predict_proba. Calibração pulada.")

        # --- Otimização Limiar ---
        optimal_threshold = 0.5; best_val_roi = -np.inf
        logging.info("  Otimizando limiar na validação...")
        if calibrator and X_val_odds is not None and y_proba_val_raw_draw is not None: # Precisa das probs brutas p/ calibrar
             try:
                 y_proba_val_calibrated = calibrator.predict(y_proba_val_raw_draw)
                 possible_thresholds = np.linspace(0.15, 0.60, 31)
                 logging.info(f"    Testando {len(possible_thresholds)} limiares [{possible_thresholds.min():.2f}-{possible_thresholds.max():.2f}]...")
                 best_th_details = {}
                 for th in possible_thresholds:
                     val_roi, val_bets, val_profit = calculate_roi_with_threshold(y_val, y_proba_val_calibrated, th, X_val_odds, odd_draw_col_name)
                     if val_roi is not None and val_roi > best_val_roi:
                         best_val_roi = val_roi; optimal_threshold = th; best_th_details = {'th': th, 'roi': val_roi, 'bets': val_bets}
                 if best_val_roi > -np.inf: logging.info(f"    Limiar ótimo (Val): {optimal_threshold:.3f} (ROI={best_val_roi:.2f}%, Bets={best_th_details.get('bets',0)})")
                 else: logging.warning("    ROI inválido na validação. Usando thr=0.5."); optimal_threshold = 0.5
             except Exception as e_thresh: logging.error(f"  Erro otimizar limiar: {e_thresh}", exc_info=True); optimal_threshold = 0.5
        else: logging.warning(f"  Otimização pulada (sem calibrador/odds val/probs val). Usando thr=0.5.")

        # --- Avaliação Final ---
        logging.info(f"  Avaliando no teste (Limiar={optimal_threshold:.3f})...");
        current_eval_metrics = {}; evaluation_successful = False
        try:
            y_pred_test_thr05 = model_instance_trained.predict(X_test_model) # Limiar 0.5
            # Métricas base @ 0.5
            acc = accuracy_score(y_test, y_pred_test_thr05); prec_thr05 = precision_score(y_test, y_pred_test_thr05, pos_label=1, zero_division=0); rec_thr05 = recall_score(y_test, y_pred_test_thr05, pos_label=1, zero_division=0); f1_thr05 = f1_score(y_test, y_pred_test_thr05, pos_label=1, zero_division=0); conf_matrix = confusion_matrix(y_test, y_pred_test_thr05).tolist()
            current_eval_metrics.update({'accuracy': acc, 'precision_draw': prec_thr05, 'recall_draw': rec_thr05, 'f1_score_draw': f1_thr05, 'confusion_matrix': conf_matrix})
            logging.info(f"    -> DEBUG Base Metrics @0.5: Acc={acc:.4f}, F1={f1_thr05:.4f}")

            # ROI @ 0.5
            roi_thr05, num_bets_thr05, profit_thr05 = None, 0, None
            if X_test_odds is not None: roi_thr05, num_bets_thr05, profit_thr05 = calculate_roi_with_threshold(y_test, model_instance_trained.predict_proba(X_test_model)[:, 1], 0.5, X_test_odds, odd_draw_col_name) # Usa predict_proba direto para 0.5
            current_eval_metrics.update({'roi_thr05': roi_thr05, 'profit_thr05': profit_thr05, 'num_bets_thr05': num_bets_thr05})
            logging.info(f"    -> DEBUG ROI @0.5: ROI={roi_thr05 if roi_thr05 is not None else 'N/A':.2f}%, Bets={num_bets_thr05}")

            # Métricas com Probs Calibradas
            logloss_val = None; roc_auc_val = None; brier_val = None; y_proba_test_calibrated = None
            has_predict_proba = hasattr(model_instance_trained, "predict_proba")
            if has_predict_proba:
                try:
                    y_proba_test_raw_full = model_instance_trained.predict_proba(X_test_model)
                    if y_proba_test_raw_full.shape[1] > 1:
                        y_proba_test_raw_draw = y_proba_test_raw_full[:, 1]
                        if calibrator: y_proba_test_calibrated = calibrator.predict(y_proba_test_raw_draw)
                        else: y_proba_test_calibrated = y_proba_test_raw_draw; logging.warning(" -> Usando probs brutas (sem calibrador)")

                        try: logloss_val = log_loss(y_test, y_proba_test_raw_full); logging.info(f"    -> LogLoss(Raw)={logloss_val:.4f}")
                        except Exception as e: logging.warning(f" -> Erro LogLoss: {e}")
                        try:
                             if len(np.unique(y_test)) > 1: roc_auc_val = roc_auc_score(y_test, y_proba_test_calibrated); logging.info(f"    -> ROC AUC(Calib)={roc_auc_val:.4f}")
                             else: logging.warning("    -> ROC AUC não calc (1 classe y_test)")
                        except Exception as e: logging.warning(f" -> Erro ROC AUC: {e}")
                        try: brier_val = brier_score_loss(y_test, y_proba_test_calibrated); logging.info(f"    -> Brier(Calib)={brier_val:.4f}")
                        except Exception as e: logging.warning(f" -> Erro Brier: {e}")
                    else: logging.warning("    -> Shape predict_proba Teste inválido.")
                except Exception as e: logging.error(f"    -> Erro obter/processar probs teste: {e}", exc_info=True)
            else: logging.info(f"    -> {model_name} sem predict_proba.")
            current_eval_metrics.update({'log_loss': logloss_val, 'roc_auc': roc_auc_val, 'brier_score': brier_val})

            # ROI com Limiar Otimizado
            roi_opt, num_bets_opt, profit_opt = None, 0, None
            if y_proba_test_calibrated is not None and X_test_odds is not None:
                 roi_opt, num_bets_opt, profit_opt = calculate_roi_with_threshold(y_test, y_proba_test_calibrated, optimal_threshold, X_test_odds, odd_draw_col_name)
                 roi_opt_str = f"{roi_opt:.2f}%" if roi_opt is not None and not np.isnan(roi_opt) else "N/A"
                 profit_opt_str = f"{profit_opt:.2f}" if profit_opt is not None else "N/A"
                 logging.info(f"    -> Test Set ROI (Th={optimal_threshold:.3f}): {roi_opt_str}, Bets: {num_bets_opt}, Profit: {profit_opt_str}")
            else: logging.warning("    -> ROI Otimizado não calculado (sem probs calib ou odds teste).")
            current_eval_metrics.update({'roi': roi_opt, 'num_bets': num_bets_opt, 'profit': profit_opt, 'optimal_threshold': optimal_threshold})

            # Métricas de tamanho
            current_eval_metrics.update({'train_set_size': len(y_train), 'val_set_size': len(y_val), 'test_set_size': len(y_test)})
            evaluation_successful = True # Marca sucesso

        except Exception as e_eval_outer:
             logging.error(f"    Erro GRAVE DURANTE avaliação final {model_name}: {e_eval_outer}", exc_info=True)

        # Adiciona resultado SE o modelo foi treinado
        if model_instance_trained is not None:
             logging.info(f"    ==> DEBUG: Adicionando resultado {model_name} (Eval OK: {evaluation_successful})")
             all_results.append({ 'model_name': model_name, 'model_object': model_instance_trained, 'scaler': current_scaler, 'calibrator': calibrator, 'params': current_best_params, 'metrics': current_eval_metrics, 'optimal_threshold': optimal_threshold })
        else: logging.error(f"    ==> ERRO LÓGICO: {model_name} nulo, não deveria chegar aqui.")

        logging.info(f"  Tempo p/ {model_name} (total): {time.time() - start_time_model:.2f} seg.")
    # --- Fim do Loop for ---

    # --- SELEÇÃO E SALVAMENTO (COM DEBUG EXAUSTIVO) ---
    if progress_callback: progress_callback(len(available_models), len(available_models), "Selecionando/Salvando...")
    end_time_total = time.time(); logging.info(f"--- Treino concluído ({end_time_total - start_time_total:.2f} seg) ---")
    if not all_results: logging.error("SELEÇÃO: Nenhum resultado válido."); return False
    logging.info(f"--- DEBUG: Iniciando processamento de {len(all_results)} resultados ---")
    try:
        results_df = pd.DataFrame(all_results); logging.info(f"DEBUG SELEÇÃO: DF Inicial (metrics col):\n{results_df[['model_name', 'metrics']]}")
        # Extrai métricas com segurança
        results_df['f1_score_draw'] = results_df['metrics'].apply(lambda m: m.get('f1_score_draw') if isinstance(m,dict) else None)
        results_df['roi'] = results_df['metrics'].apply(lambda m: m.get('roi') if isinstance(m,dict) else None)
        results_df['optimal_threshold'] = results_df.apply(lambda row: row.get('optimal_threshold', 0.5), axis=1)
        logging.info(f"DEBUG SELEÇÃO: DF após extração:\n{results_df[['model_name', 'f1_score_draw', 'roi', 'optimal_threshold']]}")
        # Converte e trata Nulos/Infs
        results_df['f1_score_draw'] = pd.to_numeric(results_df['f1_score_draw'], errors='coerce').fillna(-1.0)
        results_df['roi'] = pd.to_numeric(results_df['roi'], errors='coerce').fillna(-np.inf)
        results_df['optimal_threshold'] = pd.to_numeric(results_df['optimal_threshold'], errors='coerce').fillna(0.5)
        logging.info(f"DEBUG SELEÇÃO: DF após conversão/fillna:\n{results_df[['model_name', 'f1_score_draw', 'roi', 'optimal_threshold']]}")
        logging.info(f"DEBUG SELEÇÃO: Tipos F1/ROI: {results_df['f1_score_draw'].dtype} / {results_df['roi'].dtype}")
        if results_df.empty: logging.error("SELEÇÃO: DF vazio pós-proc."); return False
    except Exception as e: logging.error(f"SELEÇÃO: Erro criar/proc DF: {e}", exc_info=True); return False
    # Ordena e Seleciona F1
    try:
        logging.info("DEBUG SELEÇÃO: Ordenando F1..."); results_df_sorted_f1 = results_df.sort_values(by='f1_score_draw', ascending=False).reset_index(drop=True); logging.info(f"DEBUG SELEÇÃO: Ranking F1:\n{results_df_sorted_f1[['model_name', 'f1_score_draw', 'roi']]}");
        if results_df_sorted_f1.empty: logging.error("SELEÇÃO: Ranking F1 vazio."); return False
        best_f1_result = results_df_sorted_f1.iloc[0].to_dict().copy()
        f1_val = best_f1_result.get('f1_score_draw', 'N/A'); f1_str = f"{f1_val:.4f}" if isinstance(f1_val, (int,float,np.number)) and not np.isnan(f1_val) else "N/A"; logging.info(f"SELEÇÃO: Melhor F1: {best_f1_result.get('model_name', 'ERRO')} (F1={f1_str})")
    except Exception as e: logging.error(f"SELEÇÃO: Erro selecionar F1: {e}", exc_info=True); return False
    # Ordena e Seleciona ROI
    best_roi_result = None;
    try:
        logging.info("DEBUG SELEÇÃO: Ordenando ROI..."); results_df_sorted_roi = results_df.sort_values(by='roi', ascending=False).reset_index(drop=True); logging.info(f"DEBUG SELEÇÃO: Ranking ROI:\n{results_df_sorted_roi[['model_name', 'f1_score_draw', 'roi']]}");
        logging.info("DEBUG SELEÇÃO: Procurando ROI válido (> -inf)...")
        if not results_df_sorted_roi.empty:
            for idx, row in results_df_sorted_roi.iterrows():
                 current_roi = row['roi']; current_name = row['model_name']; logging.info(f"  -> Verificando ROI {current_name}: {current_roi} ({type(current_roi)})")
                 if isinstance(current_roi, (int, float, np.number)) and current_roi > -np.inf: best_roi_result = row.to_dict().copy(); logging.info(f"  -> ENCONTRADO Melhor ROI: {best_roi_result.get('model_name','ERRO')} (ROI={current_roi:.4f})"); break
                 else: logging.info(f"  -> ROI inválido/inf para {current_name}.")
        else: logging.warning("DEBUG SELEÇÃO: Ranking ROI vazio.")
    except Exception as e: logging.error(f"SELEÇÃO: Erro ordenar/encontrar ROI: {e}", exc_info=True)
    if best_roi_result: roi_val = best_roi_result.get('roi','N/A'); roi_str = f"{roi_val:.2f}%" if isinstance(roi_val,(int,float,np.number)) and roi_val > -np.inf else "N/A"; logging.info(f"SELEÇÃO: Melhor ROI Válido: {best_roi_result.get('model_name','ERRO')} (ROI={roi_str})")
    else: logging.warning("SELEÇÃO: Nenhum ROI válido encontrado.")
    # Lógica Decisão Salvamento
    model_to_save_f1 = best_f1_result; model_to_save_roi = None; logging.info(f"--- DEBUG SELEÇÃO: Decidindo slot ROI ---"); logging.info(f"  Best F1: {best_f1_result.get('model_name')}"); logging.info(f"  Best ROI: {best_roi_result.get('model_name') if best_roi_result else 'None'}")
    try:
        if best_roi_result:
            if best_f1_result.get('model_name') == best_roi_result.get('model_name'):
                logging.info("  DEBUG: F1 e ROI mesmo modelo.");
                if len(results_df_sorted_f1) > 1: model_to_save_roi = results_df_sorted_f1.iloc[1].to_dict().copy(); logging.info(f"  -> Usando 2º F1 ({model_to_save_roi.get('model_name')}) p/ ROI.")
                else: logging.warning(f"  -> Só 1 modelo. Salvando F1 p/ ambos."); model_to_save_roi = best_f1_result.copy()
            else: logging.info(f"  DEBUG: F1 e ROI diferentes."); model_to_save_roi = best_roi_result.copy(); logging.info(f"  -> Selecionado Melhor ROI ({model_to_save_roi.get('model_name')}) p/ slot ROI.")
        else: logging.warning("  -> Nenhum ROI válido. Salvando F1 ({best_f1_result.get('model_name')}) p/ slot ROI."); model_to_save_roi = best_f1_result.copy()
    except Exception as e: logging.error(f"SELEÇÃO: Erro decisão: {e}", exc_info=True); model_to_save_roi = best_f1_result.copy(); logging.warning(" -> Fallback: Usando F1 p/ ambos slots.")
    logging.info(f"DEBUG FINAL: Modelo _f1: {model_to_save_f1.get('model_name')}"); logging.info(f"DEBUG FINAL: Modelo _roi: {model_to_save_roi.get('model_name') if model_to_save_roi else 'None'}")
    # Salvamento
    try:
        logging.info(f"Salvando Melhor F1 ({model_to_save_f1.get('model_name', 'ERRO')})..."); _save_calibrated_model_object(model_to_save_f1, feature_names, BEST_F1_MODEL_SAVE_PATH)
        if model_to_save_roi:
            if not isinstance(model_to_save_roi, dict) or 'model_name' not in model_to_save_roi: logging.error(f"ERRO SALVAR ROI: obj inválido: {model_to_save_roi}")
            else: logging.info(f"Salvando Melhor ROI/2nd F1 ({model_to_save_roi.get('model_name', 'ERRO')})..."); _save_calibrated_model_object(model_to_save_roi, feature_names, BEST_ROI_MODEL_SAVE_PATH)
        else: logging.error(f"ERRO CRÍTICO SALVAR: model_to_save_roi é None.")
    except Exception as e: logging.error(f"Erro GRAVE salvamento final: {e}", exc_info=True); return False
    logging.info("--- Processo Completo ---")
    return True


# --- Função _save_calibrated_model_object (MODIFICADA PARA SALVAR CORRETAMENTE) ---
def _save_calibrated_model_object(model_result_dict: Dict, feature_names: List[str], file_path: str) -> None:
    """Salva modelo, scaler, calibrador, params, métricas e limiar."""
    if not isinstance(model_result_dict, dict): logging.error(f"Salvar: Dados inválidos p/ {file_path}"); return
    try:
        # Pega os componentes do dicionário de resultados do modelo específico
        model_to_save = model_result_dict.get('model_object')
        scaler_to_save = model_result_dict.get('scaler')
        calibrator_to_save = model_result_dict.get('calibrator')
        params_to_save = model_result_dict.get('params')
        metrics_to_save = model_result_dict.get('metrics')
        threshold_to_save = model_result_dict.get('optimal_threshold', 0.5) # Pega o limiar específico

        if model_to_save is None: logging.error(f"Salvar: Modelo ausente p/ {file_path}"); return

        # Monta o objeto a ser salvo
        save_obj = {
            'model': model_to_save,
            'scaler': scaler_to_save, # Pode ser None
            'calibrator': calibrator_to_save, # Pode ser None
            'feature_names': feature_names, # Lista de features USADA no treino
            'best_params': params_to_save, # Params do modelo
            'eval_metrics': metrics_to_save, # Dict completo de métricas
            'optimal_threshold': threshold_to_save, # Limiar ótimo calculado para este modelo
            'save_timestamp': datetime.datetime.now().isoformat(),
            'model_class_name': model_to_save.__class__.__name__
        }
        joblib.dump(save_obj, file_path)
        logging.info(f"  -> Modelo '{model_to_save.__class__.__name__}' salvo em {os.path.basename(file_path)}.")

    except Exception as e:
        logging.error(f"  -> Erro GRAVE ao salvar objeto em {file_path}: {e}", exc_info=True)


# --- Funções Remanescentes (analyze_features, optimize_single_model, etc.) ---
# (Manter as definições delas como na resposta anterior)
# ... (código de analyze_features e optimize_single_model) ...
def save_model_scaler_features(model: Any, scaler: Optional[Any], feature_names: List[str],
                               best_params: Optional[Dict], eval_metrics: Optional[Dict],
                               file_path: str) -> None:
     logging.warning("Função save_model_scaler_features chamada (não salva calibrador/limiar). Usando save_calibrated.")
     model_result_dict = {'model_object': model, 'scaler': scaler, 'params': best_params, 'metrics': eval_metrics, 'optimal_threshold': 0.5, 'calibrator': None} # Cria dict aproximado
     _save_calibrated_model_object(model_result_dict, feature_names, file_path)

def analyze_features(X: pd.DataFrame, y: pd.Series) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    logging.info("--- Iniciando Análise de Features ---");
    if X is None or y is None or X.empty or y.empty: logging.error("Dados inválidos p/ análise."); return None;
    if not X.index.equals(y.index):
        logging.warning("Índices X/y não idênticos p/ análise. Tentando alinhar.");
        try: y = y.reindex(X.index);
        except Exception as e: logging.error(f"Erro alinhar y: {e}"); return None;
        if y.isnull().any(): logging.error("NaNs em y após alinhar."); return None;
    feature_names = X.columns.tolist(); imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': np.nan}); corr_matrix = pd.DataFrame();
    logging.info("  Calculando importância (RF)...");
    try:
        rf = RandomForestClassifier(n_estimators=50, max_depth=15, n_jobs=-1, min_samples_leaf=3, random_state=RANDOM_STATE); rf.fit(X, y); importances = rf.feature_importances_; imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False).reset_index(drop=True); logging.info("  Importância OK.");
    except Exception as e: logging.error(f"  Erro RF importance: {e}", exc_info=True);
    logging.info("  Calculando correlação...");
    try: df_temp = X.copy(); df_temp['target_IsDraw'] = y; corr_matrix = df_temp.corr(); logging.info("  Correlação OK.");
    except Exception as e: logging.error(f"  Erro correlação: {e}", exc_info=True);
    logging.info("--- Análise Features Concluída ---"); return imp_df, corr_matrix

# optimize_single_model (Mantida como placeholder)
def optimize_single_model(model_name: str, X: pd.DataFrame, y: pd.Series,
                           X_test_with_odds: Optional[pd.DataFrame] = None,
                           progress_callback: Optional[Callable[[int, int, str], None]] = None,
                           scaler_type: str = 'standard',
                           odd_draw_col_name: str = ODDS_COLS.get('draw', 'Odd_D_FT')
                           ) -> Optional[Tuple[str, Dict, Dict]]:
    # ... (código placeholder como antes) ...
    logging.info(f"--- Otimizando Hiperparâmetros para: {model_name} (Placeholder) ---")
    logging.warning("Implementação real de optimize_single_model (GridSearchCV/Avaliação) pendente.")
    # (Resto do código placeholder omitido por brevidade)
    return None # Retorna None pois é placeholder