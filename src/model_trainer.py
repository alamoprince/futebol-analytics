import pandas as pd
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
try: import lightgbm as lgb; LGBMClassifier = lgb.LGBMClassifier
except ImportError: print("AVISO: LightGBM não instalado."); lgb = None; LGBMClassifier = None
from sklearn.metrics import (accuracy_score, classification_report, log_loss, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib, os, datetime, numpy as np, logging

from data_handler import roi
from config import (
    RANDOM_STATE, TEST_SIZE, MODEL_CONFIG, CLASS_NAMES,
    BEST_F1_MODEL_SAVE_PATH, BEST_ROI_MODEL_SAVE_PATH, # NOVOS Paths
    CROSS_VALIDATION_SPLITS, N_JOBS_GRIDSEARCH, FEATURE_COLUMNS,
    ODDS_COLS, BEST_MODEL_METRIC # BEST_MODEL_METRIC ainda usado para logs
)
from typing import Any, Optional, Dict, Tuple, List, Callable

logging.basicConfig(level=logging.INFO, format='%(asctime)s - TRAINER - %(levelname)s - %(message)s')

# Função scale_features 
def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame, scaler_type='standard') -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
    
    if scaler_type == 'minmax': scaler = MinMaxScaler(); 
    else: scaler = StandardScaler(); X_train = X_train.astype(float); X_test = X_test.astype(float); 
    logging.info(f"  Aplicando {scaler.__class__.__name__}...");
    X_train_scaled = scaler.fit_transform(X_train); 
    X_test_scaled = scaler.transform(X_test); 
    X_train_scaled_df = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns); 
    X_test_scaled_df = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns); 
    logging.info("  Scaling concluído."); 
    return X_train_scaled_df, X_test_scaled_df, scaler

# --- Função de Treinamento (Seleciona e SALVA os 2 melhores) ---
def train_evaluate_and_save_best_models( # Nome da função mudou
    X: pd.DataFrame, y: pd.Series,
    X_test_with_odds: Optional[pd.DataFrame] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    scaler_type: str = 'standard',
    odd_draw_col_name: str = ODDS_COLS['draw']
    
    ) -> bool: 
    """
    Treina, otimiza, avalia TODOS os modelos em config.MODEL_CONFIG.
    Seleciona o melhor modelo por F1 e o melhor por ROI (ou 2º melhor F1).
    Salva os dois modelos selecionados em arquivos separados.
    """
    if X is None or y is None: return False
    if not MODEL_CONFIG: logging.error("MODEL_CONFIG vazio."); return False
    available_models = {name: config for name, config in MODEL_CONFIG.items() if not (name == 'LGBMClassifier' and lgb is None)}
    if not available_models: logging.error("Nenhum modelo válido encontrado."); return False

    feature_names = list(X.columns)
    all_results = [] # Lista para guardar resultados de TODOS os modelos testados

    logging.info(f"--- Treinando {len(available_models)} Modelos para Seleção Dupla ---")
    start_time_total = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    logging.info(f"Split: Treino={len(X_train)}, Teste={len(X_test)}")
    X_test_odds_aligned = None # ... (lógica alinhar X_test_with_odds) ...
    if X_test_with_odds is not None: 
        if odd_draw_col_name in X_test_with_odds.columns:
            try: X_test_odds_aligned = X_test_with_odds.loc[X_test.index]
            except KeyError: logging.warning("Índices não batem p/ ROI.")
        else: logging.warning(f"Coluna '{odd_draw_col_name}' não em X_test_with_odds p/ ROI.")
    else: logging.warning("X_test_with_odds não fornecido p/ ROI.")

    # Loop pelos modelos disponíveis
    for i, (model_name, config) in enumerate(available_models.items()):
        status_text = f"Modelo {i+1}/{len(available_models)}: {model_name}"
        logging.info(f"\n--- {status_text} ---")
        if progress_callback: progress_callback(i, len(available_models), status_text)
        start_time_model = time.time()
        try: model_class = eval(model_name)
        except NameError: logging.error(f"Classe {model_name} não encontrada."); continue

        model_kwargs = config.get('model_kwargs', {}); param_grid = config.get('param_grid', {})
        needs_scaling = config.get('needs_scaling', False)
        X_train_model = X_train.copy(); X_test_model = X_test.copy(); current_scaler = None

        if needs_scaling: # Scaling condicional
             logging.info(f"  Modelo '{model_name}' requer scaling..."); 
             try: X_train_model, X_test_model, current_scaler = scale_features(X_train_model, X_test_model, scaler_type); 
             except Exception as scale_err: logging.error(f"  ERRO scaling: {scale_err}. Pulando."); 
             continue
        else: logging.info("  Scaling não requerido.")

        current_model = model_class(**model_kwargs); current_best_params = model_kwargs; model_instance_trained = None
        if param_grid: # GridSearchCV
            grid_verbose_level = 0; scoring_metric = 'f1'; search_cv = GridSearchCV(estimator=current_model, param_grid=param_grid, cv=CROSS_VALIDATION_SPLITS, n_jobs=N_JOBS_GRIDSEARCH, scoring=scoring_metric, verbose=grid_verbose_level, error_score='raise'); logging.info(f"  Iniciando GridSearchCV (verbose={grid_verbose_level}, scoring={scoring_metric})...");
            try:
                 if progress_callback: progress_callback(i, len(available_models), f"Ajustando {model_name}...")
                 search_cv.fit(X_train_model, y_train); model_instance_trained = search_cv.best_estimator_; current_best_params = search_cv.best_params_; logging.info(f"    Melhor CV ({scoring_metric}): {search_cv.best_score_:.4f}"); logging.info(f"    Params: {current_best_params}")
            except Exception as e_cv: logging.warning(f"    Erro CV: {e_cv}. Treinando com padrão...")
        else: logging.info("  Treinando com params padrão...")

        if model_instance_trained is None: # Treino padrão
             try:
                 if progress_callback: progress_callback(i, len(available_models), f"Ajustando {model_name}...")
                 model_instance_trained = current_model.fit(X_train_model, y_train)
             except Exception as e_fit: logging.error(f"    Erro treino padrão: {e_fit}. Pulando."); continue

        # Avaliação
        logging.info("  Avaliando..."); current_eval_metrics = {}
        if progress_callback: progress_callback(i, len(available_models), f"Avaliando {model_name}...")
        try: # ... (Cálculo de métricas e ROI ) ...
            has_predict_proba = hasattr(model_instance_trained, "predict_proba"); y_pred = model_instance_trained.predict(X_test_model); y_pred_proba_draw = None; logloss = None; roc_auc = None
            if has_predict_proba: #... (calcula probs, logloss, roc_auc) ...
                try: y_pred_proba_full = model_instance_trained.predict_proba(X_test_model); y_pred_proba_draw = y_pred_proba_full[:, 1]; logloss = log_loss(y_test, y_pred_proba_full); roc_auc = roc_auc_score(y_test, y_pred_proba_draw)
                except Exception as e_proba: logging.warning(f"    Erro probs: {e_proba}")
            current_eval_metrics['accuracy'] = accuracy_score(y_test, y_pred); current_eval_metrics['precision_draw'] = precision_score(y_test, y_pred, pos_label=1, zero_division=0); current_eval_metrics['recall_draw'] = recall_score(y_test, y_pred, pos_label=1, zero_division=0); current_eval_metrics['f1_score_draw'] = f1_score(y_test, y_pred, pos_label=1, zero_division=0); current_eval_metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist(); current_eval_metrics['log_loss'] = logloss; current_eval_metrics['roc_auc'] = roc_auc; current_eval_metrics['threshold'] = 0.5; current_eval_metrics['train_set_size'] = len(y_train); current_eval_metrics['test_set_size'] = len(y_test); current_eval_metrics['profit'] = None; current_eval_metrics['roi'] = None; current_eval_metrics['num_bets'] = 0
            if X_test_odds_aligned is not None: # ... (cálculo ROI) ...
                predicted_draws_indices = X_test.index[y_pred == 1]; num_bets = len(predicted_draws_indices); current_eval_metrics['num_bets'] = num_bets
                if num_bets > 0: #... (calcula profit) ...
                    actuals = y_test.loc[predicted_draws_indices]; odds = X_test_odds_aligned.loc[predicted_draws_indices, odd_draw_col_name].astype(float); profit = 0
                    for idx in predicted_draws_indices: odd_d = odds.loc[idx]; 
                    if pd.notna(odd_d) and odd_d > 0: profit += (odd_d - 1) if actuals.loc[idx] == 1 else -1
                    current_eval_metrics['profit'] = profit; current_eval_metrics['roi'] = (profit / num_bets) * 100 if num_bets > 0 else 0
            logging.info(f"    Métricas {model_name}: Acc={current_eval_metrics['accuracy']:.3f}, F1_Emp={current_eval_metrics['f1_score_draw']:.3f}, ROI={current_eval_metrics.get('roi', 'N/A'):.2f}%")
            
            if hasattr(model_instance_trained, 'feature_importances_'):
                importances = model_instance_trained.feature_importances_
                 # Usa 'feature_names' que foi definido no início da função
                feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
                 # Imprime no console/log
                print(f"\n--- Importância das Features ({model_name}) ---")
                with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
                      print(feature_importance_df)
                print("-" * (len(f"--- Importância das Features ({model_name}) ---"))) # Linha separadora
                 # Opcional: Salva no dicionário de métricas (pode deixar grande)
                current_eval_metrics['feature_importances'] = feature_importance_df.round(5).to_dict('records')
                
             # --- Guarda o resultado deste modelo ---
            all_results.append({
                 'model_name': model_name,
                 'model_object': model_instance_trained,
                 'scaler': current_scaler,
                 'params': current_best_params,
                 'metrics': current_eval_metrics
             })
            

        except Exception as e_eval: logging.error(f"    Erro avaliação {model_name}: {e_eval}")
        logging.info(f"  Tempo p/ {model_name}: {time.time() - start_time_model:.2f} seg.")

    # --- Fim do Loop - Seleção e Salvamento ---
    if progress_callback: progress_callback(len(available_models), len(available_models), "Selecionando e Salvando...")
    end_time_total = time.time(); logging.info(f"--- Treino concluído ({end_time_total - start_time_total:.2f} seg) ---")

    if not all_results:
        logging.error("Nenhum modelo foi treinado com sucesso.")
        return False # Retorna False indicando falha

    # Cria DataFrame com os resultados para facilitar ordenação
    results_df = pd.DataFrame(all_results)
    results_df['f1_score_draw'] = results_df['metrics'].apply(lambda m: m.get('f1_score_draw', -1)) # Extrai F1
    # Extrai ROI, tratando None como -infinito para ordenação (queremos maximizar)
    results_df['roi'] = results_df['metrics'].apply(lambda m: m.get('roi') if m.get('roi') is not None else -np.inf)

    # Ordena por F1 (maior primeiro) e depois ROI (maior primeiro)
    results_df = results_df.sort_values(by=['f1_score_draw', 'roi'], ascending=[False, False]).reset_index()

    print("\n--- Ranking dos Modelos (por F1 Empate) ---")
    print(results_df[['model_name', 'f1_score_draw', 'roi']])
    print("-" * 40)

    # Seleciona Melhor por F1
    best_f1_result = results_df.iloc[0].to_dict()
    logging.info(f"Melhor Modelo por F1: {best_f1_result['model_name']} (F1={best_f1_result['f1_score_draw']:.4f})")

    # Seleciona Melhor por ROI (ignorando -infinito)
    results_df_roi_valid = results_df[results_df['roi'] > -np.inf].sort_values(by='roi', ascending=False).reset_index()
    best_roi_result = None
    if not results_df_roi_valid.empty:
        best_roi_result = results_df_roi_valid.iloc[0].to_dict()
        logging.info(f"Melhor Modelo por ROI: {best_roi_result['model_name']} (ROI={best_roi_result['roi']:.2f}%)")
    else:
        logging.warning("Nenhum modelo com ROI válido calculado.")

    # Determina qual modelo salvar para F1 e ROI
    model_to_save_f1 = best_f1_result

    # Se o melhor F1 e melhor ROI são o MESMO modelo, E existe um segundo melhor F1:
    if best_roi_result and best_f1_result['index'] == best_roi_result['index'] and len(results_df) > 1:
        # Pega o SEGUNDO melhor por F1 como substituto para o "melhor ROI"
        model_to_save_roi = results_df.iloc[1].to_dict()
        logging.info(f"  -> Melhor F1 e ROI são o mesmo modelo. Usando 2º Melhor F1 para slot ROI: {model_to_save_roi['model_name']}")
    elif best_roi_result:
        # Se são modelos diferentes, usa o melhor por ROI
        model_to_save_roi = best_roi_result
    else:
        # Se não há modelo com ROI válido, salva o melhor F1 em ambos os slots (ou None para ROI)
        logging.warning("  -> Salvando melhor F1 no slot ROI pois não há ROI válido.")
        model_to_save_roi = best_f1_result # Ou poderia ser None se preferir não duplicar

    # Salva os dois modelos selecionados
    logging.info(f"Salvando Melhor F1 ({model_to_save_f1['model_name']}) em {BEST_F1_MODEL_SAVE_PATH}")
    _save_single_model_object(model_to_save_f1, feature_names, BEST_F1_MODEL_SAVE_PATH)

    if model_to_save_roi: # Só salva se foi definido
         logging.info(f"Salvando Melhor ROI/2nd F1 ({model_to_save_roi['model_name']}) em {BEST_ROI_MODEL_SAVE_PATH}")
         _save_single_model_object(model_to_save_roi, feature_names, BEST_ROI_MODEL_SAVE_PATH)
    else:
         logging.warning(f"Não foi possível salvar um modelo para o slot 'Melhor ROI' em {BEST_ROI_MODEL_SAVE_PATH}")


    return True # Indica que o processo terminou

# --- NOVA Função Auxiliar para Salvar UM objeto de modelo ---
def _save_single_model_object(model_result_dict: Dict, feature_names: List[str], file_path: str) -> None:
    """Salva um único modelo e seus dados associados."""
    try:
        save_timestamp = datetime.datetime.now().isoformat()
        model_to_save = model_result_dict['model_object']
        save_obj = {
            'model': model_to_save,
            'scaler': model_result_dict.get('scaler'), # Pega o scaler específico daquele treino
            'feature_names': feature_names,
            'best_params': model_result_dict.get('params'),
            'eval_metrics': model_result_dict.get('metrics'),
            'save_timestamp': save_timestamp,
            'model_class_name': model_to_save.__class__.__name__
        }
        joblib.dump(save_obj, file_path)
        logging.info(f"  -> Modelo {model_to_save.__class__.__name__} salvo com sucesso.")
    except Exception as e:
        logging.error(f"  -> Erro ao salvar modelo em {file_path}: {e}")

def save_model_scaler_features(model: Any, scaler: Optional[Any], feature_names: List[str],
                               best_params: Optional[Dict], eval_metrics: Optional[Dict],
                               file_path: str) -> None:
     print("AVISO: Função save_model_scaler_features não é mais chamada diretamente pelo pipeline principal.")
     _save_single_model_object({'model_object': model, 'scaler': scaler, 'params': best_params, 'metrics': eval_metrics}, feature_names, file_path)

def analyze_features(X: pd.DataFrame, y: pd.Series) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Analisa features: calcula importância (RF rápido) e correlação.

    Args:
        X: DataFrame com as features selecionadas (deve conter FEATURE_COLUMNS).
        y: Series com o alvo binário ('IsDraw').

    Returns:
        Tupla (df_importancia, df_correlacao) ou None se erro.
        df_importancia: DataFrame com colunas ['Feature', 'Importance']
        df_correlacao: DataFrame da matriz de correlação (incluindo alvo)
    """
    # Ensure logging is configured if you use logging.info/error below
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - ANALYZER - %(levelname)s - %(message)s')

    logging.info("--- Iniciando Análise de Features (chamado externamente) ---")
    if X is None or y is None or X.empty or y.empty:
        logging.error("Dados inválidos para análise de features.")
        return None
    # Ensure y is aligned with X's index if they came from different processing steps
    if not X.index.equals(y.index):
        logging.warning("Índices de X e y não são idênticos. Tentando alinhar y ao índice de X.")
        try:
            y = y.reindex(X.index)
            if y.isnull().any():
                 logging.error("Alinhamento de y resultou em NaNs. Verifique os dados de entrada.")
                 return None
        except Exception as e_reindex:
             logging.error(f"Erro ao alinhar y com X: {e_reindex}")
             return None


    feature_names = X.columns.tolist()
    imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': np.nan}) # Default com NaN
    corr_matrix = pd.DataFrame() # Default vazio

    # 1. Calcular Importância com RandomForest Rápido
    logging.info("  Calculando importância (RandomForest rápido)...")
    try:
        # Usa poucos estimadores e limita profundidade para rapidez
        # Use RANDOM_STATE from config if desired
        rf_analyzer = RandomForestClassifier(n_estimators=50, max_depth=15,
                                             # random_state=RANDOM_STATE, # Optional
                                             n_jobs=-1,
                                             min_samples_leaf=3)
        rf_analyzer.fit(X, y)
        importances = rf_analyzer.feature_importances_
        imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        imp_df = imp_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        logging.info("  Importância calculada.")
    except Exception as e:
        logging.error(f"  Erro ao calcular importância com RF: {e}")
        # imp_df já está com NaNs

    # 2. Calcular Correlação
    logging.info("  Calculando matriz de correlação...")
    try:
        df_temp = X.copy()
        df_temp['target_IsDraw'] = y # Adiciona alvo para correlação direta
        # Calcula correlação de Pearson por padrão
        corr_matrix = df_temp.corr()
        logging.info("  Matriz de correlação calculada.")
    except Exception as e:
        logging.error(f"  Erro ao calcular correlação: {e}")
        # corr_matrix já está vazia

    logging.info("--- Análise de Features Concluída ---")
    return imp_df, corr_matrix

def roi(y_test: pd.Series, y_pred: np.ndarray, X_test_odds_aligned: pd.DataFrame, odd_draw_col_name: str) -> Optional[float]:
        if X_test_odds_aligned is None:
            return None
        predicted_draws_indices = y_test.index[y_pred == 1]
        num_bets = len(predicted_draws_indices)
        if num_bets == 0:
            return 0
        actuals = y_test.loc[predicted_draws_indices]
        odds = X_test_odds_aligned.loc[predicted_draws_indices, odd_draw_col_name].astype(float)
        profit = 0
        for idx in predicted_draws_indices:
            odd_d = odds.loc[idx]
            if pd.notna(odd_d) and odd_d > 0:
                profit += (odd_d - 1) if actuals.loc[idx] == 1 else -1
        return (profit / num_bets) * 100

def optimize_single_model(model_name: str, X: pd.DataFrame, y: pd.Series,
                           X_test_with_odds: Optional[pd.DataFrame] = None, # Requer dados de teste p/ ROI
                           progress_callback: Optional[Callable[[int, int, str], None]] = None, # Para reportar Grid
                           scaler_type: str = 'standard',
                           odd_draw_col_name: str = ODDS_COLS['draw']
                           ) -> Optional[Tuple[str, Dict, Dict]]:
    """
    (Placeholder) Otimiza hiperparâmetros para um único modelo usando GridSearchCV.
    (Implementação real seria similar ao loop dentro de train_evaluate_and_save_best_models)
    """
    logging.info(f"--- Otimizando Hiperparâmetros para: {model_name} (Placeholder) ---")
    logging.warning("Implementação real de optimize_single_model (GridSearchCV/Avaliação) pendente.")

    if X is None or y is None or model_name not in MODEL_CONFIG:
        logging.error("Dados inválidos ou modelo não encontrado no config.")
        return None

    # --- Lógica de exemplo (NÃO EXECUTA TREINO REAL AINDA) ---
    # 1. Pegar config do modelo
    config = MODEL_CONFIG[model_name]
    model_kwargs = config.get('model_kwargs', {})
    param_grid = config.get('param_grid', {})
    needs_scaling = config.get('needs_scaling', False)

    # 2. Separar Treino/Teste (para avaliação final das métricas)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    X_test_odds_aligned = None
    if X_test_with_odds is not None:
        if odd_draw_col_name in X_test_with_odds.columns:
            try: X_test_odds_aligned = X_test_with_odds.loc[X_test.index]
            except KeyError: pass # Ignora se falhar

    # 3. Aplicar Scaling (se necessário)
    X_train_opt = X_train.copy(); X_test_opt = X_test.copy(); scaler_opt = None
    if needs_scaling:
         try: X_train_opt, X_test_opt, scaler_opt = scale_features(X_train_opt, X_test_opt, scaler_type)
         except Exception: print("Erro scaling na otimização"); return None # Falha se scaling der erro

    # 4. RODAR GRIDSEARCHCV 
         search_cv = GridSearchCV(...)
         search_cv.fit(X_train_opt, y_train)
         best_model_opt = search_cv.best_estimator_
         best_params_opt = search_cv.best_params_
         best_cv_score_opt = search_cv.best_score_

    # 5. AVALIAR NO TESTE 
         y_pred = best_model_opt.predict(X_test_opt)
         eval_metrics_opt = {"accuracy": accuracy_score(y_test, y_pred), 
                             "f1_score_draw": f1_score(y_test, y_pred, pos_label=1), 
                             "roi": roi(y_test, y_pred, X_test_odds_aligned, odd_draw_col_name),
                               "num_bets": len(y_test)}

    # --- Simulação de Retorno ---
    best_params_placeholder = {"param_simulado": "valor_simulado", **param_grid} # Usa a grade como exemplo
    best_cv_score_placeholder = 0.65 # Exemplo
    eval_metrics_placeholder = {"accuracy": 0.75, "f1_score_draw": 0.55, "roi": 5.0, "num_bets": 50}

    logging.info("Otimização concluída (Placeholder).")
    logging.info(f"  Melhor Score CV (f1): {best_cv_score_placeholder:.4f}")
    logging.info(f"  Melhores Parâmetros (simulado): {best_params_placeholder}")
    logging.info(f"  Métricas Teste (simulado): {eval_metrics_placeholder}")

    # Retorna nome, best_params, eval_metrics
    return model_name, best_params_placeholder, eval_metrics_placeholder