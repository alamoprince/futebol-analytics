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
try:
    import lightgbm as lgb
    LGBMClassifier = lgb.LGBMClassifier
except ImportError:
    print("AVISO: LightGBM não instalado.")
    lgb = None; LGBMClassifier = None
from sklearn.metrics import (accuracy_score, classification_report, log_loss,
                             precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, brier_score_loss)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib, os, datetime, numpy as np, logging, traceback
try:
    from config import (RANDOM_STATE, TEST_SIZE, MODEL_CONFIG, CLASS_NAMES,
                        BEST_F1_MODEL_SAVE_PATH, BEST_ROI_MODEL_SAVE_PATH,
                        CROSS_VALIDATION_SPLITS, N_JOBS_GRIDSEARCH, FEATURE_COLUMNS,
                        ODDS_COLS, BEST_MODEL_METRIC, DEFAULT_EV_THRESHOLD)
    from typing import Any, Optional, Dict, Tuple, List, Callable
except ImportError as e: print(f"Erro crítico import config/typing: {e}"); raise

logging.basicConfig(level=logging.INFO, format='%(asctime)s - TRAINER - %(levelname)s - %(message)s')

# --- Funções Auxiliares (roi, calculate_roi_with_threshold, calculate_metrics_with_ev, scale_features) ---
# (Colar as definições corretas dessas funções aqui, como nas respostas anteriores)
def roi(y_test: pd.Series, y_pred: np.ndarray, X_test_odds_aligned: Optional[pd.DataFrame], odd_draw_col_name: str) -> Optional[float]:
    # ... (código completo da função roi) ...
    if X_test_odds_aligned is None or odd_draw_col_name not in X_test_odds_aligned.columns: return None
    try: common_index=y_test.index.intersection(X_test_odds_aligned.index);
    except AttributeError: logging.error("Error accessing index in roi function."); return None
    if len(common_index) != len(y_test): return None
    y_test_common=y_test.loc[common_index];
    try: y_pred_series=pd.Series(y_pred, index=y_test.index); y_pred_common=y_pred_series.loc[common_index];
    except (ValueError,AttributeError): logging.error("Error aligning y_pred in roi."); return None
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
        if len(common_index)!=len(y_true):logging.warning("ROI Thr: Índices não batem.");return roi_value,num_bets,profit;
        y_true_common=y_true.loc[common_index];odds_common=pd.to_numeric(odds_data.loc[common_index,odd_col_name],errors='coerce');
        try:y_proba_common=pd.Series(y_proba,index=y_true.index).loc[common_index];
        except ValueError:logging.error("ROI Thr: Erro alinhar y_proba.");return None,0,None
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
    except Exception as e:logging.error(f"ROI Thr: Erro - {e}",exc_info=True);return None,0,None

def calculate_metrics_with_ev(y_true: pd.Series, y_proba_calibrated: np.ndarray, ev_threshold: float, odds_data: Optional[pd.DataFrame], odd_col_name: str) -> Tuple[Optional[float], int, Optional[float]]:
    # ... (código completo da função calculate_metrics_with_ev) ...
    profit=None;roi_value=None;num_bets=0;profit_calc=0;valid_bets_count=0;
    if odds_data is None or odd_col_name not in odds_data.columns:logging.warning(f"EV Metr: Odds ('{odd_col_name}') ausentes.");return roi_value,num_bets,profit
    try:
        common_index=y_true.index.intersection(odds_data.index);
        if len(common_index)!=len(y_true):logging.warning("EV Metr: Índices não batem.");return roi_value,num_bets,profit;
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
        logging.info(f"    -> Métricas EV (Th={ev_threshold:.3f}): ROI={roi_value if roi_value is not None else 'N/A':.2f}%, Bets={num_bets}, Profit={profit if profit is not None else 'N/A'}")
        return roi_value,num_bets,profit
    except Exception as e:logging.error(f"EV Metr: Erro - {e}",exc_info=True);return None,0,None

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame, scaler_type='standard') -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
    # ... (código completo da função scale_features) ...
    X_train=X_train.copy();X_test=X_test.copy();
    try:X_train=X_train.astype(float);X_test=X_test.astype(float)
    except ValueError as e:logging.error(f"Erro converter p/ float scaling:{e}");raise
    if scaler_type=='minmax':scaler=MinMaxScaler()
    else:scaler=StandardScaler()
    logging.info(f"  Aplicando {scaler.__class__.__name__}...");
    try:X_train_scaled=scaler.fit_transform(X_train);X_test_scaled=scaler.transform(X_test)
    except Exception as e:logging.error(f"Erro {scaler.__class__.__name__}.fit/transform:{e}",exc_info=True);raise
    X_train_scaled_df=pd.DataFrame(X_train_scaled,index=X_train.index,columns=X_train.columns);X_test_scaled_df=pd.DataFrame(X_test_scaled,index=X_test.index,columns=X_test.columns);
    logging.info("  Scaling concluído.");return X_train_scaled_df,X_test_scaled_df,scaler

# --- Função Principal de Treinamento (REVISADA FINAL) ---
def train_evaluate_and_save_best_models(
    X: pd.DataFrame, y: pd.Series,
    X_test_with_odds: Optional[pd.DataFrame] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    scaler_type: str = 'standard',
    odd_draw_col_name: str = ODDS_COLS.get('draw', 'Odd_D_FT'),
    calibration_method: str = 'isotonic',
    optimize_ev_threshold: bool = True, # <<< HABILITADO PARA OTIMIZAR EV
    default_ev_threshold: float = DEFAULT_EV_THRESHOLD
    ) -> bool:
    """
    Treina, CALIBRA, opcionalmente otimiza LIMIAR de EV, avalia modelos. Salva 2 melhores.
    """
    # ... (Verificações iniciais X, y, MODEL_CONFIG, available_models) ...
    if X is None or y is None: logging.error("Dados X ou y None."); return False
    if not MODEL_CONFIG: logging.error("MODEL_CONFIG vazio."); return False
    available_models = {name: config for name, config in MODEL_CONFIG.items() if not (name=='LGBMClassifier' and lgb is None)}
    if not available_models: logging.error("Nenhum modelo válido."); return False

    feature_names = list(X.columns); all_results = []
    logging.info(f"--- Treinando/Calibrando {len(available_models)} Modelos ---"); start_time_total = time.time()

    # --- Divisão Tripla e Preparo Odds (com logs) ---
    logging.info("Dividindo dados..."); val_size=0.20; test_size_final=TEST_SIZE; train_val_size_temp = 1.0 - test_size_final;
    if train_val_size_temp <= 0: logging.error("TEST_SIZE inválido."); return False;
    val_size_relative = val_size / train_val_size_temp
    try:
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size_final, random_state=RANDOM_STATE, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size_relative, random_state=RANDOM_STATE, stratify=y_train_val)
        logging.info(f"Split: Treino={len(X_train)}, Val={len(X_val)}, Teste={len(X_test)}")
        # Alinhamento Odds Val/Teste
        X_val_odds=None; X_test_odds=None; logging.info(f"--- DEBUG: Verificando X_test_with_odds ({type(X_test_with_odds)}) ---")
        if X_test_with_odds is not None and not X_test_with_odds.empty:
             logging.info(f"DEBUG: X_tst_odds shape {X_test_with_odds.shape}. Col Odd ('{odd_draw_col_name}'): {odd_draw_col_name in X_test_with_odds.columns}")
             if odd_draw_col_name not in X_test_with_odds.columns: logging.error(f"Erro: Col Odd '{odd_draw_col_name}' NÃO encontrada.")
             common_val=X_val.index.intersection(X_test_with_odds.index); logging.info(f"DEBUG: Idx Val:{len(X_val)}/Hist:{len(X_test_with_odds.index)} -> Intersec:{len(common_val)}")
             if len(common_val)>0:
                  try:
                      if odd_draw_col_name in X_test_with_odds.columns: X_val_odds=X_test_with_odds.loc[common_val,[odd_draw_col_name]].copy(); logging.info(f"DEBUG: X_val_odds CRIADO {X_val_odds.shape}.")
                      else: logging.error("DEBUG: Col Odd não existe p/ VAL loc.")
                  except Exception as e: logging.error(f"DEBUG: Erro loc Val:{e}")
             else: logging.error("DEBUG: NENHUMA intersecção VAL/Odds.")
             common_test=X_test.index.intersection(X_test_with_odds.index); logging.info(f"DEBUG: Idx Teste:{len(X_test)}/Hist:{len(X_test_with_odds.index)} -> Intersec:{len(common_test)}")
             if len(common_test)>0:
                  try:
                      if odd_draw_col_name in X_test_with_odds.columns: X_test_odds=X_test_with_odds.loc[common_test,[odd_draw_col_name]].copy(); logging.info(f"DEBUG: X_test_odds CRIADO {X_test_odds.shape}.")
                      else: logging.error("DEBUG: Col Odd não existe p/ TESTE loc.")
                  except Exception as e: logging.error(f"DEBUG: Erro loc Teste:{e}")
             else: logging.error("DEBUG: NENHUMA intersecção TESTE/Odds.")
        else: logging.warning("DEBUG: X_test_with_odds None/vazio.")
    except Exception as e: logging.error(f"Erro divisão/alinhar dados: {e}", exc_info=True); return False

    # --- Loop pelos modelos ---
    for i, (model_name, config) in enumerate(available_models.items()):
        status_text=f"Modelo {i+1}/{len(available_models)}:{model_name}"; logging.info(f"\n--- {status_text} ---");
        if progress_callback: progress_callback(i, len(available_models), f"Iniciando {model_name}..."); # Callback Inicial
        start_time_model = time.time();
        try: model_class = eval(model_name)
        except Exception as e: logging.error(f"Erro get classe {model_name}: {e}"); continue

        model_kwargs=config.get('model_kwargs',{}); param_grid=config.get('param_grid',{}); needs_scaling=config.get('needs_scaling',False);
        X_train_m, X_val_m, X_test_m = X_train.copy(), X_val.copy(), X_test.copy(); current_scaler=None;

        # Scaling
        if needs_scaling:
            logging.info(f"  Scaling p/ {model_name} (scaler={scaler_type})...");
            try:
                # 1. Cria o scaler (Standard ou MinMax)
                if scaler_type == 'minmax':
                    scaler_instance = MinMaxScaler()
                else:
                    scaler_instance = StandardScaler()

                # 2. Garante que os dados são float
                X_train_m_float = X_train_m.astype(float)
                X_val_m_float = X_val_m.astype(float)
                X_test_m_float = X_test_m.astype(float)

                # 3. Ajusta o scaler APENAS no TREINO
                logging.info(f"    -> Ajustando {scaler_instance.__class__.__name__} no treino...")
                scaler_instance.fit(X_train_m_float)

                # 4. Aplica (transform) nos TRÊS conjuntos
                logging.info("    -> Aplicando transform em treino, validação e teste...")
                X_train_scaled_np = scaler_instance.transform(X_train_m_float)
                X_val_scaled_np = scaler_instance.transform(X_val_m_float)
                X_test_scaled_np = scaler_instance.transform(X_test_m_float)

                # 5. Recria DataFrames (mantendo índices e colunas)
                X_train_m = pd.DataFrame(X_train_scaled_np, index=X_train_m.index, columns=feature_names)
                X_val_m = pd.DataFrame(X_val_scaled_np, index=X_val_m.index, columns=feature_names)
                X_test_m = pd.DataFrame(X_test_scaled_np, index=X_test_m.index, columns=feature_names)

                # 6. Guarda o scaler AJUSTADO
                current_scaler = scaler_instance
                logging.info(f"    -> Scaling OK para {model_name}.")

            except Exception as scale_err:
                # Loga o erro e pula o modelo se o scaling falhar
                logging.error(f"  ERRO GRAVE no scaling para {model_name}: {scale_err}", exc_info=True)
                logging.error(f"  -> PULANDO {model_name}.")
                continue # PULA para o próximo modelo
        else:
            logging.info("  Scaling não requerido.")

        # Treino (Grid ou Padrão)
        if param_grid:
            try: search_cv = GridSearchCV(estimator=model_class(**model_kwargs), param_grid=param_grid, cv=CROSS_VALIDATION_SPLITS, n_jobs=N_JOBS_GRIDSEARCH, scoring='f1', verbose=0, error_score='raise'); logging.info(f"    -> GridSearch criado.");
            except Exception as e: logging.error(f"    ERRO criar GridSearchCV: {e}", exc_info=True); logging.error(f" -> PULANDO."); continue;
            logging.info(f"  Iniciando GridSearchCV...");
            try:
                if progress_callback: progress_callback(i, len(available_models), f"Ajustando {model_name} (CV)..."); # Callback antes do fit
                search_cv.fit(X_train_m, y_train); model_trained = search_cv.best_estimator_; best_params = search_cv.best_params_; logging.info(f"    -> GridSearch OK. CV F1: {search_cv.best_score_:.4f}. Params: {best_params}");
            except Exception as e: logging.error(f"    Erro GridSearchCV.fit: {e}", exc_info=True); model_trained = None; logging.warning("    -> Tentando fallback...");
        else: logging.info(f"  Sem grid. Treinando c/ params padrão.");

        if model_trained is None: # Fallback
            logging.info("  -> Tentando treino padrão/fallback...");
            try:
                model_inst = model_class(**model_kwargs); # Sempre cria nova instância no fallback
                if progress_callback: progress_callback(i, len(available_models), f"Ajustando {model_name} (Padrão)..."); # Callback antes do fit
                model_trained = model_inst.fit(X_train_m, y_train); logging.info("    -> Treino fallback OK.");
            except Exception as e: logging.error(f"    Erro treino fallback: {e}", exc_info=True); logging.error("    -> PULANDO."); continue;

        if model_trained is None: logging.error(f"ERRO CRITICO: {model_name} NULO pós treino."); continue;

        # Calibração
        logging.info("  Calibrando probs..."); calibrator = None; y_proba_val_raw_draw = None;
        if hasattr(model_trained, "predict_proba"):
             try:
                 y_proba_val_raw_full = model_trained.predict_proba(X_val_m);
                 if y_proba_val_raw_full.shape[1] > 1:
                      y_proba_val_raw_draw = y_proba_val_raw_full[:, 1];
                      if calibration_method == 'isotonic': calibrator = IsotonicRegression(out_of_bounds='clip')
                      else: calibrator = IsotonicRegression(out_of_bounds='clip'); logging.warning("Usando Isotonic.");
                      calibrator.fit(y_proba_val_raw_draw, y_val); logging.info("  -> Calibrador treinado.");
                 else: logging.warning(f" -> Shape predict_proba Val {y_proba_val_raw_full.shape}.");
             except Exception as e: logging.error(f"  Erro calibração: {e}", exc_info=True); calibrator = None;
        else: logging.warning(f"  {model_name} sem predict_proba.");

        # Otimização Limiar EV
        optimal_ev_threshold = default_ev_threshold; best_val_roi_ev = -np.inf;
        if optimize_ev_threshold:
            logging.info("  Otimizando limiar EV (Val)...");
            if calibrator and X_val_odds is not None and y_proba_val_raw_draw is not None:
                 try:
                     y_proba_val_calib = calibrator.predict(y_proba_val_raw_draw); ev_ths = np.linspace(0.0, 0.20, 21); logging.info(f"    Testando {len(ev_ths)} limiares EV...");
                     for ev_th in ev_ths:
                         val_roi, val_bets, _ = calculate_metrics_with_ev(y_val, y_proba_val_calib, ev_th, X_val_odds, odd_draw_col_name)
                         if val_roi is not None and val_roi > best_val_roi_ev: best_val_roi_ev=val_roi; optimal_ev_threshold=ev_th;
                     if best_val_roi_ev > -np.inf: logging.info(f"    Limiar EV ótimo(Val): {optimal_ev_threshold:.3f} (ROI={best_val_roi_ev:.2f}%)")
                     else: logging.warning("    ROI Val inválido p/ otimizar EV."); optimal_ev_threshold = default_ev_threshold;
                 except Exception as e: logging.error(f"  Erro otimizar EV: {e}", exc_info=True); optimal_ev_threshold = default_ev_threshold;
            else: logging.warning(f"  Otimização EV pulada (reqs ausentes).");
        else: logging.info(f"  Otimização EV desabilitada. Usando default {optimal_ev_threshold:.3f}.")

        # Avaliação Final no Teste
        logging.info(f"  Avaliando teste (Estratégia EV > {optimal_ev_threshold:.3f})...");
        if progress_callback: progress_callback(i, len(available_models), f"Avaliando {model_name}..."); # Callback antes da avaliação
        metrics = {}; eval_ok = False;
        try:
            y_pred_test_thr05 = model_trained.predict(X_test_m)
            acc = accuracy_score(y_test, y_pred_test_thr05);
            # **CORREÇÃO** das chamadas de métricas usando keywords
            prec05 = precision_score(y_test, y_pred_test_thr05, pos_label=1, zero_division=0)
            rec05 = recall_score(y_test, y_pred_test_thr05, pos_label=1, zero_division=0)
            f1_05 = f1_score(y_test, y_pred_test_thr05, pos_label=1, zero_division=0)
            matrix = confusion_matrix(y_test, y_pred_test_thr05).tolist();
            metrics.update({'accuracy': acc, 'precision_draw': prec05, 'recall_draw': rec05, 'f1_score_draw': f1_05, 'confusion_matrix': matrix});
            logging.info(f"    -> Base Metrics@0.5: Acc={acc:.4f}, F1={f1_05:.4f}, P={prec05:.4f}, R={rec05:.4f}")

            # ROI @ 0.5
            roi05, bets05, profit05 = None, 0, None; y_proba_test_raw_draw = None
            if X_test_odds is not None and hasattr(model_trained,"predict_proba"):
                try:
                     y_proba_test_raw_full = model_trained.predict_proba(X_test_m)
                     if y_proba_test_raw_full.shape[1]>1:
                          y_proba_test_raw_draw = y_proba_test_raw_full[:,1]
                          roi05, bets05, profit05 = calculate_roi_with_threshold(y_test, y_proba_test_raw_draw, 0.5, X_test_odds, odd_draw_col_name)
                     else: logging.warning("    -> ROI@0.5 não calc (shape proba).")
                except Exception as e: logging.error(f"    Erro calc ROI@0.5: {e}")
            metrics.update({'roi_thr05': roi05, 'profit_thr05': profit05, 'num_bets_thr05': bets05});
            logging.info(f"    -> ROI@0.5: ROI={roi05 if roi05 is not None else 'N/A':.2f}%, Bets={bets05}")

            # Métricas com Probs Calibradas
            logloss=None; auc=None; brier=None; y_prob_calib=None;
            if y_proba_test_raw_draw is not None: # Só prossegue se tivemos probs brutas
                if calibrator:
                    y_prob_calib = calibrator.predict(y_proba_test_raw_draw)
                else:
                    y_prob_calib = y_proba_test_raw_draw  # Usa bruta
                try:
                    logloss = log_loss(y_test, y_proba_test_raw_full)
                    logging.info(f"    -> LogLoss(Raw)={logloss:.4f}")
                except Exception as e:
                    logging.warning(f" Erro LogLoss:{e}")
                try:
                    if len(np.unique(y_test)) > 1:
                        auc = roc_auc_score(y_test, y_prob_calib)
                        logging.info(f"    -> AUC(Calib)={auc:.4f}")
                    else:
                        logging.warning(" AUC não calc (1 classe y_test)")
                except Exception as e:
                    logging.warning(f" Erro AUC:{e}")
                try:
                    brier = brier_score_loss(y_test, y_prob_calib)
                    logging.info(f"    -> Brier(Calib)={brier:.4f}")
                except Exception as e:
                    logging.warning(f" Erro Brier:{e}")
            metrics.update({'log_loss': logloss, 'roc_auc': auc, 'brier_score': brier})

            # ROI COM ESTRATÉGIA EV
            roi_ev, bets_ev, profit_ev = None, 0, None;
            if y_prob_calib is not None and X_test_odds is not None: roi_ev, bets_ev, profit_ev = calculate_metrics_with_ev(y_test, y_prob_calib, optimal_ev_threshold, X_test_odds, odd_draw_col_name); # Log dentro da func
            else: logging.warning("    -> ROI EV (Teste) não calculado.");
            metrics.update({'roi': roi_ev, 'num_bets': bets_ev, 'profit': profit_ev, 'optimal_ev_threshold': optimal_ev_threshold});

            metrics.update({'train_set_size':len(y_train), 'val_set_size':len(y_val), 'test_set_size':len(y_test)});
            eval_ok = True;

        except Exception as e: logging.error(f" Erro GERAL avaliação {model_name}: {e}", exc_info=True); eval_ok = False;

        # --- ADIÇÃO ÚNICA ao all_results ---
        # **CORREÇÃO**: Mover esta seção para ocorrer APENAS UMA VEZ por modelo
        if model_trained is not None:
            logging.info(f"    ==> DEBUG: Adicionando resultado {model_name} (Eval OK: {eval_ok})")
            all_results.append({
                'model_name': model_name,
                'model_object': model_trained,
                'scaler': current_scaler,
                'calibrator': calibrator,
                'params': best_params,
                'metrics': metrics, # Adiciona o dict de métricas como está (pode ter Nones)
                'optimal_ev_threshold': optimal_ev_threshold # Adiciona limiar EV
            })
        else:
            # Este log indica um problema sério no fluxo se ocorrer
            logging.error(f"    ERRO LÓGICO CRÍTICO: {model_name} é None ao tentar adicionar aos resultados.")

        logging.info(f"  Tempo p/ {model_name} (total): {time.time() - start_time_model:.2f} seg.")
    # --- Fim do Loop for ---
    if progress_callback: progress_callback(len(available_models),len(available_models),"Selecionando/Salvando...")
    end_time_total=time.time(); logging.info(f"--- Treino concluído ({end_time_total-start_time_total:.2f} seg) ---")
    if not all_results: logging.error("SELEÇÃO: Nenhum resultado válido."); return False
    logging.info(f"--- DEBUG: Iniciando processamento de {len(all_results)} resultados ---"); # <<< VERIFICAR ESTE NÚMERO
    try: # Processamento DF
        results_df=pd.DataFrame(all_results); #logging.info(f"DEBUG SELEÇÃO: DF Inicial:\n{results_df[['model_name','metrics']]}");
        results_df['f1_score_draw']=results_df['metrics'].apply(lambda m: m.get('f1_score_draw')if isinstance(m,dict)else None);
        results_df['roi']=results_df['metrics'].apply(lambda m: m.get('roi')if isinstance(m,dict)else None);
        results_df['optimal_threshold']=results_df.apply(lambda row:row.get('optimal_ev_threshold',0.5),axis=1);
        #logging.info(f"DEBUG SELEÇÃO: DF pós extração:\n{results_df[['model_name','f1_score_draw','roi','optimal_threshold']]}")
        results_df['f1_score_draw']=pd.to_numeric(results_df['f1_score_draw'],errors='coerce').fillna(-1.0);
        results_df['roi']=pd.to_numeric(results_df['roi'],errors='coerce').fillna(-np.inf);
        results_df['optimal_threshold']=pd.to_numeric(results_df['optimal_threshold'],errors='coerce').fillna(default_ev_threshold);
        logging.info(f"DEBUG SELEÇÃO: DF pós conversão:\n{results_df[['model_name','f1_score_draw','roi','optimal_threshold']]}")
        logging.info(f"DEBUG SELEÇÃO: Tipos F1/ROI:{results_df['f1_score_draw'].dtype}/{results_df['roi'].dtype}");
        if results_df.empty: logging.error("SELEÇÃO: DF vazio pós-proc."); return False;
    except Exception as e: logging.error(f"SELEÇÃO: Erro criar/proc DF:{e}",exc_info=True); return False;
    try: # Seleção F1
        logging.info("DEBUG SELEÇÃO: Ordenando F1..."); results_df_sorted_f1=results_df.sort_values(by='f1_score_draw',ascending=False).reset_index(drop=True); logging.info(f"DEBUG SELEÇÃO: Ranking F1:\n{results_df_sorted_f1[['model_name','f1_score_draw','roi']]}");
        if results_df_sorted_f1.empty: logging.error("SELEÇÃO: Ranking F1 vazio."); return False;
        best_f1_result=results_df_sorted_f1.iloc[0].to_dict().copy(); f1_val=best_f1_result.get('f1_score_draw','N/A'); f1_str=f"{f1_val:.4f}" if isinstance(f1_val,(int,float,np.number))and not np.isnan(f1_val)else"N/A"; logging.info(f"SELEÇÃO: Melhor F1: {best_f1_result.get('model_name','ERRO')} (F1={f1_str})")
    except Exception as e: logging.error(f"SELEÇÃO: Erro selecionar F1:{e}",exc_info=True); return False;
    best_roi_result=None;
    try: # Seleção ROI
        logging.info("DEBUG SELEÇÃO: Ordenando ROI..."); results_df_sorted_roi=results_df.sort_values(by='roi',ascending=False).reset_index(drop=True); logging.info(f"DEBUG SELEÇÃO: Ranking ROI:\n{results_df_sorted_roi[['model_name','f1_score_draw','roi']]}");
        logging.info("DEBUG SELEÇÃO: Procurando ROI válido (> -inf)...")
        if not results_df_sorted_roi.empty:
            for idx,row in results_df_sorted_roi.iterrows():
                cr=row['roi']; cn=row['model_name']; logging.info(f"  -> Verificando ROI {cn}:{cr}({type(cr)})");
                if isinstance(cr,(int,float,np.number))and cr>-np.inf: best_roi_result=row.to_dict().copy(); logging.info(f"  -> ENCONTRADO Melhor ROI:{best_roi_result.get('model_name','ERRO')}(ROI={cr:.4f})"); break;
                else: logging.info(f"  -> ROI inválido/inf p/ {cn}.");
        else: logging.warning("DEBUG SELEÇÃO: Ranking ROI vazio.")
    except Exception as e: logging.error(f"SELEÇÃO: Erro ordenar/encontrar ROI:{e}",exc_info=True);
    if best_roi_result: roi_val=best_roi_result.get('roi','N/A'); roi_str=f"{roi_val:.2f}%" if isinstance(roi_val,(int,float,np.number))and roi_val>-np.inf else"N/A"; logging.info(f"SELEÇÃO: Melhor ROI Válido:{best_roi_result.get('model_name','ERRO')}(ROI={roi_str})")
    else: logging.warning("SELEÇÃO: Nenhum ROI válido encontrado.")
    # Lógica Decisão Salvamento
    model_to_save_f1=best_f1_result; model_to_save_roi=None; logging.info(f"--- DEBUG SELEÇÃO: Decidindo slot ROI ---"); logging.info(f"  Best F1:{best_f1_result.get('model_name')}"); logging.info(f"  Best ROI:{best_roi_result.get('model_name')if best_roi_result else'None'}");
    try: # Decisão
        if best_roi_result:
            if best_f1_result.get('model_name') == best_roi_result.get('model_name'):
                logging.info("  DEBUG: F1 e ROI mesmo modelo.")
                if len(results_df_sorted_f1) > 1:
                    model_to_save_roi = results_df_sorted_f1.iloc[1].to_dict().copy()
                    logging.info(f"  -> Usando 2º F1({model_to_save_roi.get('model_name')}) p/ ROI.")
                else:
                    logging.warning("  -> Só 1 modelo. Salvando F1 p/ ambos.")
                    model_to_save_roi = best_f1_result.copy()
            else:
                logging.info("  DEBUG: F1 e ROI diferentes.")
                model_to_save_roi = best_roi_result.copy()
        else:
            logging.warning(f"  -> Nenhum ROI válido. Salvando F1({best_f1_result.get('model_name')}) p/ slot ROI.")
            model_to_save_roi = best_f1_result.copy()
    except Exception as e:
        logging.error(f"SELEÇÃO: Erro decisão:{e}", exc_info=True)
        model_to_save_roi = best_f1_result.copy()
        logging.warning(" -> Fallback: Usando F1 p/ ambos.")
    logging.info(f"DEBUG FINAL: Modelo _f1:{model_to_save_f1.get('model_name')}"); logging.info(f"DEBUG FINAL: Modelo _roi:{model_to_save_roi.get('model_name')if model_to_save_roi else'None'}");
    try: # Salvamento
        logging.info(f"Salvando Melhor F1 ({model_to_save_f1.get('model_name','ERRO')})..."); _save_ev_model_object(model_to_save_f1,feature_names,BEST_F1_MODEL_SAVE_PATH);
        if model_to_save_roi:
            if not isinstance(model_to_save_roi,dict)or'model_name'not in model_to_save_roi: logging.error(f"ERRO SALVAR ROI: obj inválido:{model_to_save_roi}");
            else: logging.info(f"Salvando Melhor ROI/2nd F1 ({model_to_save_roi.get('model_name','ERRO')})..."); _save_ev_model_object(model_to_save_roi,feature_names,BEST_ROI_MODEL_SAVE_PATH);
        else: logging.error(f"ERRO CRÍTICO SALVAR: model_to_save_roi é None.");
    except Exception as e: logging.error(f"Erro GRAVE salvamento final:{e}",exc_info=True); return False;
    logging.info("--- Processo Completo ---"); return True

# --- Função _save_ev_model_object (REVISADA NOME E CONTEÚDO) ---
def _save_ev_model_object(model_result_dict: Dict, feature_names: List[str], file_path: str) -> None:
    """Salva modelo, scaler, calibrador, params, métricas e limiar EV."""
    if not isinstance(model_result_dict, dict): logging.error(f"Salvar EV: Dados inválidos p/ {file_path}"); return
    try:
        model_to_save = model_result_dict.get('model_object')
        if model_to_save is None: logging.error(f"Salvar EV: Modelo ausente p/ {file_path}"); return

        # Pega os componentes do dicionário para clareza
        scaler_to_save = model_result_dict.get('scaler')
        calibrator_to_save = model_result_dict.get('calibrator')
        params_to_save = model_result_dict.get('params')
        metrics_to_save = model_result_dict.get('metrics')
        # Pega o limiar EV do dict, usando default do config como fallback
        threshold_to_save = model_result_dict.get('optimal_ev_threshold', DEFAULT_EV_THRESHOLD)

        save_obj = {
            'model': model_to_save,
            'scaler': scaler_to_save,
            'calibrator': calibrator_to_save,
            'feature_names': feature_names,
            'best_params': params_to_save,
            'eval_metrics': metrics_to_save, # Dict completo
            'optimal_ev_threshold': threshold_to_save, # SALVA LIMIAR EV
            'save_timestamp': datetime.datetime.now().isoformat(),
            'model_class_name': model_to_save.__class__.__name__
        }
        joblib.dump(save_obj, file_path)
        logging.info(f"  -> Modelo '{model_to_save.__class__.__name__}' (EV Thresh={save_obj['optimal_ev_threshold']:.3f}) salvo em {os.path.basename(file_path)}.")
    except Exception as e:
        logging.error(f"  -> Erro GRAVE ao salvar objeto EV em {file_path}: {e}", exc_info=True)


# --- Funções Antigas (Redirecionadas ou Removidas) ---
def _save_calibrated_model_object(*args, **kwargs): logging.warning("Chamada para _save_calibrated_model_object obsoleta.")
def _save_single_model_object(*args, **kwargs): logging.warning("Chamada para _save_single_model_object obsoleta.")
def save_model_scaler_features(*args, **kwargs): logging.warning("Chamada para save_model_scaler_features obsoleta.")

# --- Funções Remanescentes (analyze_features, optimize_single_model) ---
def analyze_features(X: pd.DataFrame, y: pd.Series) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    # (Código como antes)
    logging.info("--- Análise Features ---"); return None, None # Placeholder

def optimize_single_model(*args, **kwargs) -> Optional[Tuple[str, Dict, Dict]]:
    logging.warning("optimize_single_model placeholder."); return None