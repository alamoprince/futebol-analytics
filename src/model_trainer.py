import pandas as pd
import time
# MODIFICADO: Adicionado TimeSeriesSplit
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.isotonic import IsotonicRegression

from logger_config import setup_logger
from collections import defaultdict
import warnings # Import warnings
logger = setup_logger("ModelTrainerApp")
# --- Imblearn Imports ---
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    IMBLEARN_AVAILABLE = True
    logger.info("Biblioteca 'imbalanced-learn' carregada com sucesso.")
except ImportError:
    logger.error("ERRO CRÍTICO: 'imbalanced-learn' não instalado. Sampler desativado.")
    IMBLEARN_AVAILABLE = False
    from sklearn.pipeline import Pipeline as ImbPipeline 
    SMOTE = None; RandomOverSampler = None

# --- Skopt/BayesSearchCV Imports ---
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Categorical, Integer
    SKOPT_AVAILABLE = True
except ImportError:
    logger.warning("AVISO: skopt não instalado. Usando GridSearchCV fallback.")
    SKOPT_AVAILABLE = False
    from sklearn.model_selection import GridSearchCV # Precisa ser importado para o fallback

# --- LightGBM Imports ---
try:
    import lightgbm as lgb
    LGBMClassifier = lgb.LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    logger.warning("AVISO: LightGBM não instalado.")
    lgb = None; LGBMClassifier = None; LGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier # <<< TENTA IMPORTAR A CLASSE
    CATBOOST_AVAILABLE = True
    print("INFO: Biblioteca 'catboost' carregada.")
except ImportError:
    print("AVISO: CatBoost não instalado (pip install catboost).")
    CatBoostClassifier = None # <<< DEFINE COMO NONE SE FALHAR
    CATBOOST_AVAILABLE = False

# --- Outros Imports ---
from sklearn.metrics import (accuracy_score, log_loss,
                             precision_score, recall_score, f1_score,
                             roc_auc_score, brier_score_loss,
                             precision_recall_curve)
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib, os, datetime, numpy as np, traceback
from sklearn.base import clone # Para clonar estimadores base
try:
    from config import (RANDOM_STATE, MODEL_CONFIG, CLASS_NAMES, TEST_SIZE,
                        BEST_F1_MODEL_SAVE_PATH, BEST_ROI_MODEL_SAVE_PATH,
                        CROSS_VALIDATION_SPLITS, N_JOBS_GRIDSEARCH, DEFAULT_F1_THRESHOLD,
                        ODDS_COLS, BEST_MODEL_METRIC, BEST_MODEL_METRIC_ROI,
                        DEFAULT_EV_THRESHOLD, MIN_RECALL_FOR_PRECISION_OPT,
                        BAYESIAN_OPT_N_ITER, FEATURE_EPSILON, CALIBRATION_METHOD_DEFAULT)
    from typing import Any, Optional, Dict, Tuple, List, Callable

    from calibrator import BaseCalibrator, get_calibrator_instance
    from data_handler import BettingMetricsCalculator
except ImportError as e: logger.critical(f"Erro crítico import config/typing/calibrator: {e}", exc_info=True); raise

def scale_features(
        X_train: pd.DataFrame, 
        X_val: Optional[pd.DataFrame], 
        X_test: Optional[pd.DataFrame], 
        scaler_type: str = 'standard'
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[Any]]:
        """
        Ajusta um scaler nos dados de treino e o aplica nos conjuntos de validação e teste.
        
        Args:
            X_train, X_val, X_test: DataFrames de features.
            scaler_type: 'standard' para StandardScaler ou 'minmax' para MinMaxScaler.

        Returns:
            Uma tupla com os DataFrames escalados e a instância do scaler ajustado.
        """
        if scaler_type == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_type == 'standard':
            scaler = StandardScaler()
        else:
            logger.warning(f"Tipo de scaler '{scaler_type}' desconhecido. Não aplicando scaling.")
            return X_train, X_val, X_test, None

        try:
            # Garante que os dados de treino não têm NaNs/Infs antes do fit
            X_train_clean = X_train.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
            
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_clean), index=X_train.index, columns=X_train.columns)
            
            X_val_scaled = None
            if X_val is not None:
                X_val_clean = X_val.replace([np.inf, -np.inf], np.nan).fillna(X_train.median()) # Usa mediana do treino
                X_val_scaled = pd.DataFrame(scaler.transform(X_val_clean), index=X_val.index, columns=X_val.columns)

            X_test_scaled = None
            if X_test is not None:
                X_test_clean = X_test.replace([np.inf, -np.inf], np.nan).fillna(X_train.median()) # Usa mediana do treino
                X_test_scaled = pd.DataFrame(scaler.transform(X_test_clean), index=X_test.index, columns=X_test.columns)
            
            return X_train_scaled, X_val_scaled, X_test_scaled, scaler
        except Exception as e:
            logger.error(f"Erro durante o scaling das features: {e}", exc_info=True)
            # Retorna os dados originais em caso de erro para não quebrar o pipeline
            return X_train, X_val, X_test, None
        
class ModelTrainingOrchestrator:
    """
    Orquestra o pipeline completo de treinamento, avaliação e salvamento de modelos
    de machine learning, incluindo modelos individuais e um ensemble opcional.
    """
    def __init__(self, X: pd.DataFrame, y: pd.Series,
                 X_test_with_odds: Optional[pd.DataFrame],
                 training_params: Dict[str, Any],
                 progress_callback: Optional[Callable[[int, str], None]] = None):

        if X is None or y is None or X.empty or y.empty:
            raise ValueError("Dados X ou y de entrada não podem ser vazios ou None.")
        if not X.index.is_monotonic_increasing or not y.index.is_monotonic_increasing:
            logger.warning("AVISO: Dados de entrada X ou y podem não estar ordenados temporalmente. Isso é crucial para TimeSeriesSplit.")
            # Poderia forçar ordenação aqui se o índice for datetime, mas é melhor garantir antes.
        if not X.index.equals(y.index):
            raise ValueError("Índices de X e y de entrada não coincidem.")

        self.X_full = X
        self.y_full = y
        self.X_test_with_odds_full = X_test_with_odds # DataFrame completo com odds para todo o dataset
        self.progress_callback = progress_callback
        self.training_params = training_params # Armazena todos os parâmetros

        # Desempacota parâmetros com defaults do config.py
        self._unpack_training_params()

        self.original_feature_names: List[str] = list(X.columns)
        self.available_model_configs: Dict[str, Dict] = self._get_available_model_configs()

        if not self.available_model_configs:
            raise ValueError("Nenhum modelo válido configurado em MODEL_CONFIG ou dependências ausentes.")

        # Ajusta o número total de "unidades de progresso"
        num_individual_models = len(self.available_model_configs)
        self.progress_total_units = num_individual_models * 100 # 100 unidades por modelo
        if self.n_ensemble_models > 0 and num_individual_models >= 2:
            self.progress_total_units += 100 # Adiciona unidades para o ensemble

        # Atributos que serão populados durante o processo
        self.X_train_cv_data: Optional[pd.DataFrame] = None
        self.y_train_cv_data: Optional[pd.Series] = None
        self.X_test_data: Optional[pd.DataFrame] = None
        self.y_test_data: Optional[pd.Series] = None
        self.X_test_odds_data_aligned: Optional[pd.DataFrame] = None # Odds alinhadas com o X_test_data
        self.cv_temporal_splitter: Optional[TimeSeriesSplit] = None
        self.all_model_results_list: List[Dict] = []

        logger.info(f"ModelTrainingOrchestrator inicializado para {len(self.available_model_configs)} modelos individuais.")
 
    def _unpack_training_params(self):
        """Desempacota os parâmetros de treinamento do dicionário ou usa defaults."""
        params = self.training_params
        self.scaler_type = params.get('scaler_type', 'standard')
        self.sampler_type = params.get('sampler_type', 'smote' if IMBLEARN_AVAILABLE else None)
        self.odd_draw_col_name = params.get('odd_draw_col_name', ODDS_COLS.get('draw', 'Odd_D_FT'))
        self.calibration_method = params.get('calibration_method', CALIBRATION_METHOD_DEFAULT)
        self.optimize_ev_flag = params.get('optimize_ev_threshold_flag', True)
        self.optimize_f1_flag = params.get('optimize_f1_threshold_flag', True)
        self.optimize_precision_flag = params.get('optimize_precision_threshold_flag', True)
        self.min_recall_prec_opt = params.get('min_recall_target_for_prec_opt', MIN_RECALL_FOR_PRECISION_OPT)
        self.bayes_n_iter = params.get('bayes_opt_n_iter_config', BAYESIAN_OPT_N_ITER)
        self.cv_splits = params.get('cv_splits_config', CROSS_VALIDATION_SPLITS)
        self.cv_scoring = params.get('cv_scoring_metric_config', 'f1') # Usado para CV
        self.n_ensemble_models = params.get('n_ensemble_models_config', 3)
        self.test_size_ratio = params.get('test_size_ratio', TEST_SIZE) # Adicionado

    def _get_available_model_configs(self) -> Dict[str, Dict]:
        """Filtra MODEL_CONFIG para modelos com dependências disponíveis."""
        available = {}
        for name, cfg in MODEL_CONFIG.items():
            if name == 'LGBMClassifier' and not LGBM_AVAILABLE:
                logger.warning("LGBMClassifier configurado mas biblioteca LightGBM não encontrada. Pulando.")
                continue
            if name == 'CatBoostClassifier' and not CATBOOST_AVAILABLE:
                logger.warning("CatBoostClassifier configurado mas biblioteca CatBoost não encontrada. Pulando.")
                continue
            if cfg.get('search_spaces') and not SKOPT_AVAILABLE and not cfg.get('param_grid'):
                logger.warning(f"Modelo {name} tem search_spaces (Bayes) mas skopt não está disponível e não há param_grid (GridSearch) de fallback. Pulando otimização para este modelo ou o modelo em si.")
                # Poderia decidir pular o modelo ou apenas a otimização
            available[name] = cfg
        return available
    
    def _make_progress_call(self, model_idx: int, current_stage_progress_within_model: int, message: str):
        """Helper para calcular progresso total e chamar callback."""
        if self.progress_callback:
            # model_idx é 0-based.
            # Cada modelo individual tem 100 unidades. O ensemble também.
            base_progress_for_model = model_idx * 100
            total_current_progress = min(base_progress_for_model + current_stage_progress_within_model, self.progress_total_units)
            self.progress_callback(total_current_progress, message)
    
    def _prepare_data_splits(self) -> bool:
        """Divide os dados e alinha as odds."""
        # (Implementação de _temporal_train_test_split e _align_odds_with_test_set são as mesmas
        #  definidas como funções globais anteriormente, mas podem ser métodos privados se preferir)
        split_result = _temporal_train_test_split(
            self.X_full, self.y_full, self.test_size_ratio, self.cv_splits
        )
        if split_result is None: return False
        self.X_train_cv_data, self.y_train_cv_data, self.X_test_data, self.y_test_data = split_result

        # Alinha as odds para todo o dataset X_full, depois selecionaremos para o X_test_data
        # Isso é para o caso de X_test_with_odds_full ser o dataset histórico completo antes do split
        if self.X_test_with_odds_full is not None:
            common_indices_full = self.X_full.index.intersection(self.X_test_with_odds_full.index)
            aligned_odds_full_dataset = self.X_test_with_odds_full.loc[common_indices_full]

            self.X_test_odds_data_aligned = _align_odds_with_test_set(
                self.X_test_data, aligned_odds_full_dataset, self.odd_draw_col_name
            )
        else:
            self.X_test_odds_data_aligned = None


        self.cv_temporal_splitter = TimeSeriesSplit(n_splits=self.cv_splits)
        return True
    
    def _process_single_model(self, model_name: str, model_config: Dict, model_idx: int) -> Optional[Dict]:
        log_prefix = f"Mod {model_idx+1}/{len(self.available_model_configs)} ({model_name})"
        logger.info(f"\n--- {log_prefix}: Iniciando ---")
        self._make_progress_call(model_idx, 0, f"{log_prefix}: Iniciando setup...")
        model_start_time = time.time()

        base_pipeline = _setup_model_and_pipeline(model_name, model_config, self.sampler_type)
        if base_pipeline is None: logger.error(f"{log_prefix} Falha setup pipeline."); return None

        needs_scaling = model_config.get('needs_scaling', False)
        X_train_cv_model_input = self.X_train_cv_data.copy()
        X_test_model_input = self.X_test_data.copy()
        fitted_scaler = None

        if needs_scaling:
            self._make_progress_call(model_idx, 5, f"{log_prefix}: Scaling...")
            try:
                X_train_cv_model_input, _, X_test_model_input, fitted_scaler = scale_features(X_train_cv_model_input, None, X_test_model_input, self.scaler_type)
                if fitted_scaler is None: raise ValueError("Scaler não ajustado.")
                logger.info(f"{log_prefix} Scaling OK.")
            except Exception as e: logger.error(f"{log_prefix} ERRO scaling: {e}", exc_info=True); return None

        self._make_progress_call(model_idx, 10, f"{log_prefix}: Otim. Hiperparams...")
        best_cv_pipeline, best_cv_params = _perform_hyperparameter_search(
            base_pipeline, X_train_cv_model_input, self.y_train_cv_data, model_config,
            self.cv_temporal_splitter, self.cv_scoring, self.bayes_n_iter, model_name
        )

        self._make_progress_call(model_idx, 60, f"{log_prefix}: Treino Final...")
        final_pipeline, final_classifier, final_params = _train_final_pipeline(
            base_pipeline, best_cv_pipeline, X_train_cv_model_input, self.y_train_cv_data,
            model_config.get('fit_params',{}), model_name
        )
        if not final_pipeline or not final_classifier: return None

        raw_probas_test_full, raw_probas_test_draw = None, None
        if hasattr(final_pipeline, "predict_proba"):
            try:
                raw_probas_test_full = final_pipeline.predict_proba(X_test_model_input)
                if raw_probas_test_full.shape[1] >= 2: raw_probas_test_draw = raw_probas_test_full[:,1]
            except Exception as e: logger.error(f"{log_prefix} Erro predict_proba teste: {e}")

        logger.warning(f"{log_prefix} AVISO: Calibração/Otim. Limiares no CONJUNTO DE TESTE.")
        # Para o callback dentro de _calibrate_and_optimize_thresholds, precisamos passar
        # o model_idx e o total de modelos para que ele possa calcular o progresso corretamente
        # A função _calibrate_and_optimize_thresholds já tem um progress_callback,
        # passaremos o self._make_progress_call adaptado ou um novo callback.
        # Por simplicidade, vamos deixar _calibrate_and_optimize_thresholds logar internamente
        # e a orquestração principal atualizar o progresso por estágios maiores.

        calibrator, probas_calib_draw, f1_thr, ev_thr, prec_thr = _calibrate_and_optimize_thresholds(
            self.y_test_data, raw_probas_test_draw, self.X_test_odds_data_aligned, self.odd_draw_col_name,
            self.calibration_method, self.optimize_f1_flag, self.optimize_ev_flag, self.optimize_precision_flag,
            self.min_recall_prec_opt, model_name,
            lambda stage_idx, msg: self._make_progress_call(model_idx, 70 + stage_idx*5, f"{log_prefix}: {msg}"), # Callback adaptado
            0 # model_idx_for_callback (0 porque o progresso é relativo ao estágio de calib/optim)
        )
        final_probas_for_eval = probas_calib_draw if calibrator and probas_calib_draw is not None else raw_probas_test_draw

        self._make_progress_call(model_idx, 90, f"{log_prefix}: Avaliando...")
        metrics = _evaluate_model_on_test_set(
            final_pipeline, X_test_model_input, self.y_test_data, raw_probas_test_full,
            final_probas_for_eval, f1_thr, ev_thr, prec_thr,
            self.X_test_odds_data_aligned, self.odd_draw_col_name, model_name
        )
        metrics['train_set_size'] = len(self.y_train_cv_data)

        logger.info(f"{log_prefix} Concluído. Tempo: {time.time() - model_start_time:.2f}s")
        self._make_progress_call(model_idx, 99, f"{log_prefix}: OK.") # Quase 100% para este modelo
        return {
            'model_name': model_name, 'model_object': final_classifier,
            'pipeline_object': final_pipeline, 'scaler': fitted_scaler,
            'calibrator': calibrator, 'params': final_params if final_params else best_cv_params,
            'metrics': metrics
        }
    
    def _train_and_evaluate_individual_models(self):
        for model_idx, (model_name, model_config) in enumerate(self.available_model_configs.items()):
            model_result = self._process_single_model(model_name, model_config, model_idx)
            if model_result:
                self.all_model_results_list.append(model_result)
            # O progresso para 100% deste modelo é feito ao final de _process_single_model
            # ou aqui, se _process_single_model retornar None.
            if model_result is None and self.progress_callback:
                 self._make_progress_call(model_idx, 99, f"Mod {model_idx+1}/{len(self.available_model_configs)} ({model_name}): Falha.")
    
    def _train_evaluate_ensemble(self) -> Optional[Dict]:
        # (Implementação de _train_evaluate_ensemble como na resposta anterior)
        # Certifique-se de chamar self._make_progress_call aqui também para o estágio do ensemble
        # Ex: self._make_progress_call(len(self.available_model_configs), 10, "Ensemble: Iniciando...")
        #      self._make_progress_call(len(self.available_model_configs), 99, "Ensemble: OK.")
        if len(self.all_model_results_list) < 2 or self.n_ensemble_models <= 0:
            logger.info("Ensemble: Modelos individuais insuficientes ou n_ensemble_models <= 0. Pulando.")
            return None

        ensemble_progress_idx = len(self.available_model_configs) # Índice de progresso para o ensemble
        self._make_progress_call(ensemble_progress_idx, 0, "Ensemble: Construindo...")

        logger.info(f"\n--- Construindo Ensemble com Top {self.n_ensemble_models} Modelos ---")
        sorted_results = sorted(self.all_model_results_list, key=lambda r: r['metrics'].get(BEST_MODEL_METRIC, -1.0), reverse=True)
        top_n_base_models_data = sorted_results[:self.n_ensemble_models]
        
        ensemble_estimators, scalers_flags = [], []
        for i, res_ens in enumerate(top_n_base_models_data):
            est_obj = res_ens.get('pipeline_object') or res_ens.get('model_object')
            if est_obj:
                ensemble_estimators.append((f"{res_ens.get('model_name',f'm{i}')}_{i}", clone(est_obj)))
                scalers_flags.append(res_ens.get('scaler') is not None)
            else: logger.warning(f"Ensemble: Estimador base {res_ens.get('model_name')} ausente.")
        
        if not ensemble_estimators: logger.error("Ensemble: Nenhum estimador válido."); return None

        voting_clf = VotingClassifier(estimators=ensemble_estimators, voting='soft', n_jobs=N_JOBS_GRIDSEARCH, verbose=False)
        X_train_ens, X_test_ens_eval = self.X_train_cv_data.copy(), self.X_test_data.copy()
        scaler_ens = None

        if any(scalers_flags):
            self._make_progress_call(ensemble_progress_idx, 10, "Ensemble: Scaling dados...")
            logger.info("Ensemble: Aplicando scaling (usando primeiro scaler encontrado).")
            first_scaler = next((r.get('scaler') for i,r in enumerate(top_n_base_models_data) if scalers_flags[i] and r.get('scaler')), None)
            if first_scaler:
                scaler_ens = clone(first_scaler)
                try:
                    scaler_ens.fit(X_train_ens)
                    X_train_ens = pd.DataFrame(scaler_ens.transform(X_train_ens), index=X_train_ens.index, columns=X_train_ens.columns)
                    X_test_ens_eval = pd.DataFrame(scaler_ens.transform(X_test_ens_eval), index=X_test_ens_eval.index, columns=X_test_ens_eval.columns)
                except Exception as e: logger.error(f"Erro scaling ensemble: {e}", exc_info=True); scaler_ens=None; X_train_ens,X_test_ens_eval = self.X_train_cv_data.copy(),self.X_test_data.copy()
        
        self._make_progress_call(ensemble_progress_idx, 20, "Ensemble: Ajustando wrapper...")
        logger.info("  Ajustando o wrapper VotingClassifier...")
        try:
            voting_clf.fit(X_train_ens, self.y_train_cv_data); logger.info("  -> Wrapper VotingClassifier ajustado.")
            
            raw_probas_f, raw_probas_d = None, None
            if hasattr(voting_clf, "predict_proba"):
                try: raw_probas_f = voting_clf.predict_proba(X_test_ens_eval);
                except Exception as e: logger.error(f"Erro predict_proba Ensemble: {e}")
            if raw_probas_f is not None and raw_probas_f.shape[1]>=2: raw_probas_d = raw_probas_f[:,1]

            logger.warning("Ensemble: AVISO: Calibração/Otim. Limiares no CONJUNTO DE TESTE.")
            ens_cal, ens_cal_probas, ens_f1_t, ens_ev_t, ens_p_t = _calibrate_and_optimize_thresholds(
                self.y_test_data, raw_probas_d, self.X_test_odds_data_aligned, self.odd_draw_col_name, self.calibration_method,
                self.optimize_f1_flag, self.optimize_ev_flag, self.optimize_precision_flag, self.min_recall_prec_opt,
                "VotingEnsemble", lambda stage_idx, msg: self._make_progress_call(ensemble_progress_idx, 70 + stage_idx*5, f"Ensemble: {msg}"), 0
            )
            ens_final_eval_probas = ens_cal_probas if ens_cal and ens_cal_probas is not None else raw_probas_d
            
            self._make_progress_call(ensemble_progress_idx, 90, "Ensemble: Avaliando...")
            ens_metrics = _evaluate_model_on_test_set(
                voting_clf, X_test_ens_eval, self.y_test_data, raw_probas_f, ens_final_eval_probas,
                ens_f1_t, ens_ev_t, ens_p_t, self.X_test_odds_data_aligned, self.odd_draw_col_name, "VotingEnsemble"
            )
            ens_metrics['train_set_size'] = len(self.y_train_cv_data)
            logger.info("  -> Ensemble avaliado.");
            self._make_progress_call(ensemble_progress_idx, 99, "Ensemble: OK.")
            return {'model_name':'VotingEnsemble', 'model_object':voting_clf, 'pipeline_object':None, 'scaler':scaler_ens,
                    'calibrator':ens_cal, 'params':{'estimators':[e[0] for e in ensemble_estimators],'voting':'soft'},
                    'metrics':ens_metrics}
        except Exception as e: logger.error(f"Erro treinar/avaliar Ensemble: {e}", exc_info=True); return None
    
    def _select_and_save_final_models(self) -> bool:
        if self.progress_callback: self._make_progress_call(len(self.available_model_configs), 100, "Selecionando/Salvando...") # -1 removido; usando número de modelos disponíveis para o progresso
        if not self.all_model_results_list: logger.error("SALVAR: Nenhum resultado."); return False
        try:
            results_df = pd.DataFrame(self.all_model_results_list)
            results_df['f1_score_draw_metric'] = results_df['metrics'].apply(lambda m: m.get(BEST_MODEL_METRIC, -1.0))
            results_df['roi_metric'] = results_df['metrics'].apply(lambda m: m.get(BEST_MODEL_METRIC_ROI, -np.inf))
            # ... (logging do comparativo df) ...
            results_df_sorted_f1 = results_df.sort_values(by='f1_score_draw_metric', ascending=False).reset_index(drop=True)
            best_f1_dict = results_df_sorted_f1.iloc[0].to_dict() if not results_df_sorted_f1.empty else None
            if best_f1_dict:
                logger.info(f"Salvando Melhor por F1: {best_f1_dict.get('model_name', 'N/A')}")
                _save_model_object(best_f1_dict, self.original_feature_names, BEST_F1_MODEL_SAVE_PATH) # Usa constante global
            else: logger.error("Nenhum modelo para Melhor F1."); return False
            # ... (lógica para model_for_roi_slot e salvar em BEST_ROI_MODEL_SAVE_PATH) ...
            model_for_roi_slot = None
            results_df_valid_roi = results_df[results_df['roi_metric'].notna() & np.isfinite(results_df['roi_metric']) & (results_df['roi_metric'] > -np.inf)]
            if not results_df_valid_roi.empty:
                best_roi_dict = results_df_valid_roi.sort_values(by='roi_metric', ascending=False).iloc[0].to_dict()
                if best_f1_dict and (best_roi_dict.get('model_name') != best_f1_dict.get('model_name') or len(results_df_sorted_f1) == 1) : model_for_roi_slot = best_roi_dict
                elif len(results_df_sorted_f1) > 1: model_for_roi_slot = results_df_sorted_f1.iloc[1].to_dict()
                else: model_for_roi_slot = best_f1_dict
            elif best_f1_dict: model_for_roi_slot = best_f1_dict; logger.warning("Nenhum ROI válido. Usando melhor F1 para slot 'Melhor ROI'.")
            if model_for_roi_slot:
                logger.info(f"Salvando para Slot Melhor ROI/2nd F1: {model_for_roi_slot.get('model_name', 'N/A')}")
                _save_model_object(model_for_roi_slot, self.original_feature_names, BEST_ROI_MODEL_SAVE_PATH) # Usa constante global
            return True
        except Exception as e: logger.error(f"Erro seleção/salvamento: {e}", exc_info=True); return False


    def run_training_pipeline(self) -> bool:
        logger.info(f"--- ModelTrainingOrchestrator: Iniciando Pipeline ---")
        if self.progress_callback:
            # Informa à GUI o total de unidades de progresso esperado
            self.progress_callback(self.progress_total_units, "progress_max_set") # Sinal especial para a GUI
            self._make_progress_call(0, 1, "Preparando dados...") # Progresso inicial mínimo

        if not self._prepare_data_splits():
            logger.error("Falha na preparação dos splits. Encerrando."); return False
        self._make_progress_call(0, 2, "Splits de dados OK.") # Pequeno avanço

        self._train_and_evaluate_individual_models()
        self._train_evaluate_ensemble() # Já lida com seu próprio progresso
        success = self._select_and_save_final_models()
        
        # Sinaliza conclusão final do progresso (100% do total)
        if self.progress_callback:
             self.progress_callback(self.progress_total_units, "Treinamento Concluído!" if success else "Treinamento Falhou.")

        logger.info(f"--- ModelTrainingOrchestrator: Pipeline Concluído ---")
        return success

# --- Função de Ponto de Entrada (Wrapper) ---
def train_evaluate_and_save_best_models(
    X: pd.DataFrame, y: pd.Series,
    X_test_with_odds: Optional[pd.DataFrame] = None,
    progress_callback_stages: Optional[Callable[[int, str], None]] = None,
    **training_params_kwargs # Captura todos os outros parâmetros de config
    ) -> bool:
    try:
        orchestrator = ModelTrainingOrchestrator(
            X, y, X_test_with_odds,
            training_params_kwargs,
            progress_callback_stages
        )
        return orchestrator.run_training_pipeline()
    except ValueError as ve:
        logger.error(f"Erro ao inicializar ModelTrainingOrchestrator: {ve}", exc_info=True)
        if progress_callback_stages: progress_callback_stages(0, f"Erro Init: {ve}") # Informa GUI
        return False
    except Exception as e:
        logger.error(f"Erro inesperado no processo de treinamento principal: {e}", exc_info=True)
        if progress_callback_stages: progress_callback_stages(0, f"Erro Fatal: {e}")
        return False
    
    
# --- Função Principal de Treinamento ---
def _temporal_train_test_split(
    X: pd.DataFrame, y: pd.Series, test_size_ratio: float, cv_splits_for_min_train: int
) -> Optional[Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]:
    logger.info("Dividindo dados temporalmente (Treino+CV / Teste Final)...")
    n_total = len(X)
    n_test = int(n_total * test_size_ratio)
    min_train_cv_samples_needed = cv_splits_for_min_train + 1 
    if n_total - n_test < min_train_cv_samples_needed :
        logger.error(f"Divisão Temporal Inválida: Treino+CV ({n_total-n_test}) muito pequeno para {cv_splits_for_min_train} splits CV (precisa de >= {min_train_cv_samples_needed}).")
        return None
    if n_test < 1 :
        logger.error(f"Divisão Temporal Inválida: Conjunto de teste teria {n_test} amostras.")
        return None
    n_train_cv = n_total - n_test
    try:
        X_train_cv = X.iloc[:n_train_cv]; y_train_cv = y.iloc[:n_train_cv]
        X_test = X.iloc[n_train_cv:]; y_test = y.iloc[n_train_cv:]
        logger.info(f"Split Temporal: T+CV={len(X_train_cv)} ({len(X_train_cv)/n_total:.1%}), Teste={len(X_test)} ({len(X_test)/n_total:.1%})")
        if not (X_train_cv.index.is_monotonic_increasing and X_test.index.is_monotonic_increasing and \
                y_train_cv.index.is_monotonic_increasing and y_test.index.is_monotonic_increasing):
             logger.warning("Índices não são monotonicamente crescentes após split temporal. Verifique ordenação inicial do DataFrame de entrada.")
        return X_train_cv, y_train_cv, X_test, y_test
    except Exception as e:
        logger.error(f"Erro durante a divisão temporal manual: {e}", exc_info=True)
        return None
    


def _align_odds_with_test_set(
    X_test: pd.DataFrame, X_test_with_odds_full: Optional[pd.DataFrame], odd_draw_col_name: str
) -> Optional[pd.DataFrame]:
    if X_test_with_odds_full is None or X_test_with_odds_full.empty or odd_draw_col_name not in X_test_with_odds_full.columns:
        logger.warning("DF de odds ausente/vazio ou sem coluna de odd de empate. ROI/EV não serão calculados no teste.")
        return None
    try:
        common_test_indices = X_test.index.intersection(X_test_with_odds_full.index)
        if not common_test_indices.empty:
            X_test_odds_aligned = X_test_with_odds_full.loc[common_test_indices, [odd_draw_col_name]].copy()
            X_test_odds_aligned[odd_draw_col_name] = pd.to_numeric(X_test_odds_aligned[odd_draw_col_name], errors='coerce')
            logger.info(f"Odds alinhadas com o conjunto de teste: {len(X_test_odds_aligned)} jogos.")
            if len(X_test_odds_aligned) != len(X_test):
                logger.warning(f"Perda de {len(X_test) - len(X_test_odds_aligned)} amostras de teste durante alinhamento de odds.")
            return X_test_odds_aligned
        else:
            logger.warning("Nenhum índice em comum encontrado entre X_test e X_test_with_odds_full para alinhamento de odds.")
            return None
    except Exception as e:
        logger.error(f"Erro ao alinhar odds com X_test: {e}", exc_info=True)
        return None

def _setup_model_and_pipeline(
    model_name: str, model_config: Dict, sampler_type: Optional[str]
) -> Optional[ImbPipeline]:
    """Configura o pipeline base (com sampler opcional e classificador)."""
    try:
        model_class_module_name = model_name 
        if model_name == 'LGBMClassifier' and LGBM_AVAILABLE: model_class = LGBMClassifier
        elif model_name == 'CatBoostClassifier' and CATBOOST_AVAILABLE: model_class = CatBoostClassifier
        else: model_class = eval(model_name) 

        if model_class is None: raise NameError(f"Classe {model_name} é None.")
    except NameError: logger.error(f"Classe do modelo '{model_name}' não encontrada."); return None
    except Exception as e_eval: logger.error(f"Erro ao obter classe para '{model_name}': {e_eval}."); return None

    model_kwargs = model_config.get('model_kwargs', {})
    pipeline_steps = []
    sampler_instance = None; sampler_log_name = "None"

    if sampler_type == 'smote' and SMOTE and IMBLEARN_AVAILABLE:
        sampler_instance = SMOTE(random_state=RANDOM_STATE); pipeline_steps.append(('sampler', sampler_instance)); sampler_log_name = "SMOTE"
    elif sampler_type == 'random' and RandomOverSampler and IMBLEARN_AVAILABLE:
        sampler_instance = RandomOverSampler(random_state=RANDOM_STATE); pipeline_steps.append(('sampler', sampler_instance)); sampler_log_name = "RandomOverSampler"
    elif sampler_type and not IMBLEARN_AVAILABLE:
        logger.warning(f"Sampler '{sampler_type}' solicitado, mas imblearn não disponível. Sampler não será usado.")

    pipeline_steps.append(('classifier', model_class(**model_kwargs)))
    pipeline = ImbPipeline(pipeline_steps)
    logger.info(f"  Pipeline base criado (Sampler: {sampler_log_name}, Classifier: {model_name})")
    return pipeline


def _perform_hyperparameter_search(
    pipeline_to_tune: ImbPipeline, X_train_cv_processed: pd.DataFrame, y_train_cv: pd.Series,
    model_config: Dict, cv_splitter: TimeSeriesSplit, scoring_metric: str,
    n_bayes_iter: int, model_name_log: str
) -> Tuple[Optional[ImbPipeline], Optional[Dict]]:
    use_bayes = SKOPT_AVAILABLE and 'search_spaces' in model_config and model_config['search_spaces']
    param_space_or_grid = model_config.get('search_spaces') if use_bayes else model_config.get('param_grid')

    if not param_space_or_grid:
        logger.info(f"  Sem config de busca para {model_name_log}. Usando pipeline base com params default."); return None, None

    search_method_name = 'BayesSearchCV' if use_bayes else 'GridSearchCV'
    logger.info(f"  Iniciando {search_method_name} para {model_name_log} (Score: {scoring_metric}, CV: Temporal)...")
    try:
        model_fit_params = model_config.get('fit_params', {})
        if use_bayes:
            search_cv = BayesSearchCV(estimator=pipeline_to_tune, search_spaces=param_space_or_grid,
                                      n_iter=n_bayes_iter, cv=cv_splitter, n_jobs=N_JOBS_GRIDSEARCH,
                                      scoring=scoring_metric, random_state=RANDOM_STATE, verbose=0)
        else:
            search_cv = GridSearchCV(estimator=pipeline_to_tune, param_grid=param_space_or_grid,
                                     cv=cv_splitter, n_jobs=N_JOBS_GRIDSEARCH, scoring=scoring_metric,
                                     verbose=0, error_score='raise')
        search_cv.fit(X_train_cv_processed, y_train_cv, **model_fit_params)
        best_pipeline = search_cv.best_estimator_
        best_params_raw = search_cv.best_params_
        best_classifier_params = {k.replace('classifier__', ''): v for k,v in best_params_raw.items() if k.startswith('classifier__')}
        logger.info(f"    -> Busca CV ({search_method_name}) OK. Melhor CV {scoring_metric}: {search_cv.best_score_:.4f}. Params: {best_classifier_params}")
        return best_pipeline, best_classifier_params
    except Exception as e: logger.error(f"    Erro {search_method_name}.fit: {e}", exc_info=True); return None, None


def _train_final_pipeline(
    base_pipeline: ImbPipeline, best_pipeline_from_cv: Optional[ImbPipeline],
    X_train_full_processed: pd.DataFrame, y_train_full: pd.Series,
    model_fit_params_config: Dict, model_name_log: str
) -> Tuple[Optional[ImbPipeline], Optional[Any], Optional[Dict]]:
    pipeline_for_final_train = clone(best_pipeline_from_cv) if best_pipeline_from_cv else clone(base_pipeline)
    action_log = "Re-ajustando MELHOR pipeline (da CV)" if best_pipeline_from_cv else "Treinando pipeline BASE com params default"
    logger.info(f"  {action_log} para {model_name_log} em todos os dados de treino+CV...")
    try:
        pipeline_for_final_train.fit(X_train_full_processed, y_train_full, **model_fit_params_config)
        trained_classifier_instance = pipeline_for_final_train.named_steps['classifier']
        effective_classifier_params = trained_classifier_instance.get_params()
        logger.info(f"  -> Treino final do pipeline para {model_name_log} OK.")
        return pipeline_for_final_train, trained_classifier_instance, effective_classifier_params
    except Exception as e: logger.error(f"  Erro CRÍTICO ao treinar pipeline final para {model_name_log}: {e}", exc_info=True); return None, None, None

def _calibrate_and_optimize_thresholds(
    y_true_for_opt: pd.Series, y_proba_raw_draw_for_opt: Optional[np.ndarray],
    X_odds_for_opt: Optional[pd.DataFrame], odd_draw_col_name_for_opt: str,
    calibration_method_config: Optional[str], optimize_f1_flag: bool, optimize_ev_flag: bool,
    optimize_precision_flag: bool, min_recall_target_for_prec: float, model_name_log: str,
    progress_callback: Optional[Callable[[int, str], None]], model_idx_for_callback: int
) -> Tuple[Optional[BaseCalibrator], Optional[np.ndarray], float, float, float]:
    calibrator_fitted = None; y_proba_calibrated_draw = None
    optimal_f1_thr = DEFAULT_F1_THRESHOLD; optimal_ev_thr = DEFAULT_EV_THRESHOLD; optimal_prec_thr = 0.5

    if y_proba_raw_draw_for_opt is None:
        logger.warning(f"  Probs brutas ausentes para {model_name_log}. Calibração/Otim. pulada.")
        return None, None, optimal_f1_thr, optimal_ev_thr, optimal_prec_thr

    # 1. Calibração
    calibrator_instance = get_calibrator_instance(calibration_method_config)

    if calibrator_instance:
        if progress_callback: progress_callback(model_idx_for_callback, f"{model_name_log}: Calibrando ({calibration_method_config})...")
        logger.info(f"  Calibrando probs ({calibration_method_config}) para {model_name_log}...")
        try:
            fitted_calibrator_object = calibrator_instance.fit(y_proba_raw_draw_for_opt, y_true_for_opt)
            y_proba_calibrated_draw = fitted_calibrator_object.predict_proba(y_proba_raw_draw_for_opt)
            logger.info(f"  -> Calibrador ({calibration_method_config}) ajustado e probabilidades previstas para {model_name_log}.")
        except Exception as e_calib:
            logger.error(f"  Erro durante calibração ({calibration_method_config}) para {model_name_log}: {e}. Usando probs brutas.", exc_info=True)
            fitted_calibrator_object = None 
            y_proba_calibrated_draw = y_proba_raw_draw_for_opt.copy() 
    else:
        logger.info(f"  Nenhum método de calibração válido especificado ('{calibration_method_config}'). Usando probs brutas para otimização.")
        y_proba_calibrated_draw = y_proba_raw_draw_for_opt.copy() 

    proba_for_thr_opt = y_proba_calibrated_draw 
    opt_src = calibration_method_config if fitted_calibrator_object else 'Brutas'

    if optimize_f1_flag:
        if progress_callback: progress_callback(model_idx_for_callback, f"{model_name_log}: Otim. F1 ({opt_src})...")
        try:
            p, r, t = precision_recall_curve(y_true_for_opt, proba_for_thr_opt, pos_label=1)
            if len(p) > len(t): p,r = p[:-1],r[:-1] 
            f1s = np.divide(2*p*r, p+r+FEATURE_EPSILON, where=(p+r)>0, out=np.zeros_like(p+r))
            valid_f1_indices = np.where(np.isfinite(f1s) & (f1s > 0))[0]
            if valid_f1_indices.size > 0: best_f1_idx = valid_f1_indices[np.argmax(f1s[valid_f1_indices])]; optimal_f1_thr = t[best_f1_idx]; logger.info(f"    Limiar F1 ({opt_src}): {optimal_f1_thr:.4f} (F1={f1s[best_f1_idx]:.4f})")
            else: logger.warning(f"    Não otimizou F1 para {model_name_log}.")
        except Exception as e: logger.error(f"  Erro otim. F1 {model_name_log}: {e}")

    if optimize_ev_flag and X_odds_for_opt is not None:
        if progress_callback: progress_callback(model_idx_for_callback, f"{model_name_log}: Otim. EV ({opt_src})...")
        try:
            best_roi = -np.inf; ev_ths_try = np.linspace(0.15, 0.45, 51)
            for ev_th in ev_ths_try:
                roi_v,_,_ = BettingMetricsCalculator.metrics_with_ev(y_true_for_opt, proba_for_thr_opt, ev_th, X_odds_for_opt, odd_draw_col_name_for_opt)
                if roi_v is not None and np.isfinite(roi_v) and roi_v > best_roi: best_roi = roi_v; optimal_ev_thr = ev_th
            if best_roi > -np.inf: logger.info(f"    Limiar EV ({opt_src}): {optimal_ev_thr:.3f} (ROI={best_roi:.2f}%)")
            else: logger.warning(f"    ROI inválido otim. EV {model_name_log}.")
        except Exception as e: logger.error(f"  Erro otim. EV {model_name_log}: {e}")
    elif optimize_ev_flag: logger.warning(f"  Otim. EV {model_name_log} pulada (sem odds).")

    if optimize_precision_flag:
        if progress_callback: progress_callback(model_idx_for_callback, f"{model_name_log}: Otim. Prec ({opt_src})...")
        try:
            best_p = -1.0; found_p_thr = False
            ps, rs, ts_pr = precision_recall_curve(y_true_for_opt, proba_for_thr_opt, pos_label=1)
            for i in range(len(ts_pr)): 
                p_c, r_c, t_c = ps[i], rs[i], ts_pr[i]
                if r_c >= min_recall_target_for_prec:
                    if p_c > best_p: best_p=p_c; optimal_prec_thr=t_c; found_p_thr=True
                    elif p_c == best_p and t_c > optimal_prec_thr: optimal_prec_thr=t_c; found_p_thr=True 
            if found_p_thr: rec_at_opt_p = recall_score(y_true_for_opt,(proba_for_thr_opt>=optimal_prec_thr).astype(int),pos_label=1,zero_division=0); logger.info(f"    Limiar Prec ({opt_src}): {optimal_prec_thr:.4f} (P={best_p:.4f}, R={rec_at_opt_p:.4f})")
            else: logger.warning(f"    Não encontrou limiar Prec. {model_name_log} (Recall Mín={min_recall_target_for_prec:.1%}).")
        except Exception as e: logger.error(f"  Erro otim. Prec {model_name_log}: {e}")

    return fitted_calibrator_object, y_proba_calibrated_draw, optimal_f1_thr, optimal_ev_thr, optimal_prec_thr


def _evaluate_model_on_test_set(
    final_trained_pipeline: ImbPipeline, X_test_eval_processed: pd.DataFrame, y_test_eval: pd.Series,
    y_proba_raw_test_full_eval: Optional[np.ndarray], y_proba_final_draw_test_eval: Optional[np.ndarray], 
    opt_f1_thr_eval: float, opt_ev_thr_eval: float, opt_prec_thr_eval: float,
    X_test_odds_eval: Optional[pd.DataFrame], odd_draw_col_name_eval: str, model_name_log_eval: str
) -> Dict:
    logger.info(f"  Avaliando Modelo FINAL ({model_name_log_eval}) no Teste...")
    metrics = {'model_name': model_name_log_eval, 'optimal_f1_threshold': opt_f1_thr_eval, 'optimal_ev_threshold': opt_ev_thr_eval, 'optimal_precision_threshold': opt_prec_thr_eval, 'test_set_size': len(y_test_eval)}
    eval_src = 'Calib' if y_proba_final_draw_test_eval is not None and y_proba_raw_test_full_eval is not None and not np.array_equal(y_proba_final_draw_test_eval, y_proba_raw_test_full_eval[:,1] if y_proba_raw_test_full_eval.ndim > 1 and y_proba_raw_test_full_eval.shape[1]>1 else y_proba_raw_test_full_eval) else 'Raw'

    try: y_pred05 = final_trained_pipeline.predict(X_test_eval_processed)
    except Exception as e_pred05: logger.error(f"Erro pred 0.5 {model_name_log_eval}: {e_pred05}"); y_pred05 = np.zeros_like(y_test_eval) 
    metrics.update({'accuracy_thr05': accuracy_score(y_test_eval,y_pred05), 'precision_draw_thr05':precision_score(y_test_eval,y_pred05,pos_label=1,zero_division=0), 'recall_draw_thr05':recall_score(y_test_eval,y_pred05,pos_label=1,zero_division=0), 'f1_score_draw_thr05':f1_score(y_test_eval,y_pred05,pos_label=1,zero_division=0)})
    logger.info(f"    Métricas @0.5: F1={metrics['f1_score_draw_thr05']:.4f}, P={metrics['precision_draw_thr05']:.4f}, R={metrics['recall_draw_thr05']:.4f}")

    if y_proba_final_draw_test_eval is not None:
        y_predF1=(y_proba_final_draw_test_eval>=opt_f1_thr_eval).astype(int)
        metrics.update({'accuracy_thrF1':accuracy_score(y_test_eval,y_predF1), 'precision_draw_thrF1':precision_score(y_test_eval,y_predF1,pos_label=1,zero_division=0), 'recall_draw_thrF1':recall_score(y_test_eval,y_predF1,pos_label=1,zero_division=0), 'f1_score_draw':f1_score(y_test_eval,y_predF1,pos_label=1,zero_division=0)})
        logger.info(f"    Métricas @F1Opt({opt_f1_thr_eval:.4f}): F1={metrics['f1_score_draw']:.4f}, P={metrics['precision_draw_thrF1']:.4f}, R={metrics['recall_draw_thrF1']:.4f}")
        y_predPrec=(y_proba_final_draw_test_eval>=opt_prec_thr_eval).astype(int)
        metrics.update({'accuracy_thrPrec':accuracy_score(y_test_eval,y_predPrec), 'precision_draw_thrPrec':precision_score(y_test_eval,y_predPrec,pos_label=1,zero_division=0), 'recall_draw_thrPrec':recall_score(y_test_eval,y_predPrec,pos_label=1,zero_division=0), 'f1_score_draw_thrPrec':f1_score(y_test_eval,y_predPrec,pos_label=1,zero_division=0)})
        logger.info(f"    Métricas @PrecOpt({opt_prec_thr_eval:.4f}): F1={metrics['f1_score_draw_thrPrec']:.4f}, P={metrics['precision_draw_thrPrec']:.4f}, R={metrics['recall_draw_thrPrec']:.4f}")
        
        auc,brier,logloss_m=None,None,None
        try: auc=roc_auc_score(y_test_eval,y_proba_final_draw_test_eval) if len(np.unique(y_test_eval))>1 else 0.5
        except: pass
        try: brier=brier_score_loss(y_test_eval,y_proba_final_draw_test_eval)
        except: pass
        if y_proba_raw_test_full_eval is not None:
            try: logloss_m=log_loss(y_test_eval,y_proba_raw_test_full_eval)
            except:pass
        metrics.update({'roc_auc':auc,'brier_score':brier,'log_loss':logloss_m})
        logger.info(f"    AUC({eval_src})={auc if auc is not None else 'N/A':.4f}, Brier({eval_src})={brier if brier is not None else 'N/A':.4f}, LogLoss(Raw)={logloss_m if logloss_m is not None else 'N/A':.4f}")

        if X_test_odds_eval is not None:
            roi,bets,prof = BettingMetricsCalculator.metrics_with_ev(y_test_eval,y_proba_final_draw_test_eval,opt_ev_thr_eval,X_test_odds_eval,odd_draw_col_name_eval)
            metrics.update({'roi':roi,'num_bets':bets,'profit':prof})
            roi_s = f"{roi:.2f}%" if roi is not None and np.isfinite(roi) else "N/A"
            logger.info(f"    ROI @EVOpt({opt_ev_thr_eval:.3f}) = {roi_s} ({bets} bets)")
        else: metrics.update({'roi':None,'num_bets':0,'profit':None})
    else: metrics['f1_score_draw']=metrics.get('f1_score_draw_thr05',-1.0); logger.warning(f"  Sem probs {model_name_log_eval}. Usando F1@0.5.") 
    return metrics


def _save_model_object(model_result_dict: Dict, feature_names_list: List[str], file_path_to_save: str) -> None:
    if not isinstance(model_result_dict, dict): logger.error(f"Salvar: Dados inválidos p/ {file_path_to_save}"); return
    try:
        model_obj_to_save = model_result_dict.get('model_object') 
        pipeline_obj_to_save = model_result_dict.get('pipeline_object') 
       
        object_to_actually_save = pipeline_obj_to_save if pipeline_obj_to_save is not None else model_obj_to_save

        if object_to_actually_save is None: logger.error(f"Salvar: Nenhum objeto (pipeline/modelo) para salvar em {file_path_to_save}"); return
        
        metrics_d = model_result_dict.get('metrics', {})
        save_obj_dict = {
            'model': object_to_actually_save, 
            'scaler': model_result_dict.get('scaler'),
            'calibrator': model_result_dict.get('calibrator'),
            'feature_names': feature_names_list, 
            'best_params': model_result_dict.get('params'), 
            'eval_metrics': metrics_d,
            'optimal_ev_threshold': metrics_d.get('optimal_ev_threshold', DEFAULT_EV_THRESHOLD),
            'optimal_f1_threshold': metrics_d.get('optimal_f1_threshold', DEFAULT_F1_THRESHOLD),
            'optimal_precision_threshold': metrics_d.get('optimal_precision_threshold', 0.5),
            'save_timestamp': datetime.datetime.now().isoformat(),
            'model_class_name': model_result_dict.get('model_name', object_to_actually_save.__class__.__name__) # Nome do modelo/pipeline
        }
        joblib.dump(save_obj_dict, file_path_to_save)
        logger.info(f"  -> Objeto salvo: '{save_obj_dict['model_class_name']}' em {os.path.basename(file_path_to_save)}.")
        logger.debug(f"     Detalhes salvos: Features={len(save_obj_dict['feature_names']) if save_obj_dict['feature_names'] else 'N/A'}, Scaler={'Sim' if save_obj_dict['scaler'] else 'Não'}, Calib={'Sim' if save_obj_dict['calibrator'] else 'Não'}")
        logger.debug(f"     Limiares salvos: F1={save_obj_dict['optimal_f1_threshold']:.4f}, EV={save_obj_dict['optimal_ev_threshold']:.3f}, Prec={save_obj_dict['optimal_precision_threshold']:.4f}")

    except Exception as e: logger.error(f"  -> Erro GRAVE ao salvar objeto em {file_path_to_save}: {e}", exc_info=True)

# --- Função Principal ---
def train_evaluate_and_save_best_models(
    X: pd.DataFrame, y: pd.Series,
    X_test_with_odds: Optional[pd.DataFrame] = None,
    progress_callback_stages: Optional[Callable[[int, str], None]] = None,
    num_total_models_expected: int = 1, 
    scaler_type: str = 'standard',
    sampler_type: Optional[str] = 'smote',
    odd_draw_col_name: str = ODDS_COLS.get('draw', 'Odd_D_FT'),
    calibration_method: str = CALIBRATION_METHOD_DEFAULT,
    optimize_ev_threshold_flag: bool = True, 
    optimize_f1_threshold_flag: bool = True,
    optimize_precision_threshold_flag: bool = True,
    min_recall_target_for_prec_opt: float = MIN_RECALL_FOR_PRECISION_OPT, 
    bayes_opt_n_iter_config: int = BAYESIAN_OPT_N_ITER, 
    cv_splits_config: int = CROSS_VALIDATION_SPLITS, 
    cv_scoring_metric_config: str = 'f1', 
    n_ensemble_models_config: int = 3 
    ) -> bool:

    # Validações Iniciais
    if not IMBLEARN_AVAILABLE and sampler_type: logger.error("'imbalanced-learn' não instalado."); return False
    if X is None or y is None or X.empty or y.empty: logger.error("Dados X ou y inválidos."); return False
    if not X.index.is_monotonic_increasing or not y.index.is_monotonic_increasing:
        logger.error("Dados de entrada X ou y não estão ordenados temporalmente. Isso é crucial para TimeSeriesSplit.")

        return False
    if not X.index.equals(y.index): logger.error("Índices X/y não coincidem."); return False
    
    available_model_configs = {
        name: cfg for name, cfg in MODEL_CONFIG.items()
        if not (name=='LGBMClassifier' and not LGBM_AVAILABLE) and \
           not (name=='CatBoostClassifier' and not CATBOOST_AVAILABLE)
    }
    if not available_model_configs: logger.error("Nenhum modelo válido configurado em MODEL_CONFIG."); return False
    if num_total_models_expected <= 0: num_total_models_expected = len(available_model_configs)
    if cv_splits_config <= 1 : logger.error(f"cv_splits ({cv_splits_config}) deve ser > 1."); return False

    original_feature_names = list(X.columns) 
    all_individual_model_results = []
    logger.info(f"--- Iniciando Treinamento Otimizado para {len(available_model_configs)} Modelos Individuais ---")
    total_training_start_time = time.time()

    # 1. Divisão Temporal (uma vez para todos os modelos)
    split_result = _temporal_train_test_split(X, y, TEST_SIZE, cv_splits_config)
    if split_result is None: return False 
    X_train_cv_data, y_train_cv_data, X_test_data, y_test_data = split_result

    # 2. Alinhar Odds com X_test_data (uma vez)
    X_test_odds_data_aligned = _align_odds_with_test_set(X_test_data, X_test_with_odds, odd_draw_col_name)

    # 3. Definir CV Splitter (uma vez)
    cv_temporal_splitter = TimeSeriesSplit(n_splits=cv_splits_config)

    # 4. Loop pelos modelos individuais
    for model_idx, (current_model_name, current_model_config) in enumerate(available_model_configs.items()):
        log_prefix = f"Mod {model_idx+1}/{len(available_model_configs)} ({current_model_name})"
        logger.info(f"\n--- {log_prefix}: Iniciando ---")
        if progress_callback_stages: progress_callback_stages(model_idx, f"Iniciando {current_model_name}...")
        current_model_start_time = time.time()

        # a. Setup do pipeline base para o modelo
        base_model_pipeline = _setup_model_and_pipeline(current_model_name, current_model_config, sampler_type)
        if base_model_pipeline is None: logger.error(f"{log_prefix} Falha no setup do pipeline."); continue
        
        model_requires_scaling = current_model_config.get('needs_scaling', False)
        X_train_cv_for_this_model, X_test_for_this_model = X_train_cv_data.copy(), X_test_data.copy()
        fitted_scaler_for_this_model = None

        if model_requires_scaling:
            if progress_callback_stages:
                progress_callback_stages(model_idx, f"Mod {model_idx+1}/{len(available_model_configs)} ({current_model_name}): Scaling...")
            try:
                X_train_cv_for_this_model, _, X_test_for_this_model, fitted_scaler_for_this_model = \
                    scale_features(X_train_cv_for_this_model, None, X_test_for_this_model, scaler_type)
                if fitted_scaler_for_this_model is None: raise ValueError("Scaler não foi ajustado/retornado.")
                logger.info(f"{log_prefix} Scaling OK.")
            except Exception as e: logger.error(f"{log_prefix} ERRO scaling: {e}", exc_info=True); continue

        # b. Busca de Hiperparâmetros
        if progress_callback_stages: 
            progress_callback_stages(model_idx, f"Mod {model_idx+1}/{len(available_model_configs)} ({current_model_name}): Otim. Hiperparams...")
        best_cv_pipeline, best_cv_params = _perform_hyperparameter_search(
            base_model_pipeline, X_train_cv_for_this_model, y_train_cv_data, current_model_config,
            cv_temporal_splitter, cv_scoring_metric_config, bayes_opt_n_iter_config, current_model_name
        ) 

        # c. Treino Final do Pipeline
        if progress_callback_stages: progress_callback_stages(model_idx, f"{log_prefix}: Treino Final...")
        final_trained_pipeline, final_classifier_instance, effective_final_params = _train_final_pipeline(
            base_model_pipeline, best_cv_pipeline, X_train_cv_for_this_model, y_train_cv_data,
            current_model_config.get('fit_params', {}), current_model_name
        )
        if not final_trained_pipeline or not final_classifier_instance:
            logger.error(f"{log_prefix} Falha crítica no treino final. Pulando."); continue

        # d. Previsão de Probabilidades no Teste
        raw_test_probas_full, raw_test_probas_draw = None, None
        if hasattr(final_trained_pipeline, "predict_proba"):
            try:
                raw_test_probas_full = final_trained_pipeline.predict_proba(X_test_for_this_model) 
                if raw_test_probas_full.shape[1] >= 2: raw_test_probas_draw = raw_test_probas_full[:, 1]
            except Exception as e: logger.error(f"{log_prefix} Erro predict_proba teste: {e}")
        else: logger.warning(f"{log_prefix} Pipeline final sem predict_proba.")

        # e. Calibração e Otimização de Limiares (no teste)
        logger.warning(f"{log_prefix} AVISO: Calibração e Otim. Limiares no CONJUNTO DE TESTE (avaliação otimista).")
        calibrator, probas_calib_draw_test, f1_thr, ev_thr, prec_thr = _calibrate_and_optimize_thresholds(
            y_test_data, raw_test_probas_draw, X_test_odds_data_aligned, odd_draw_col_name, calibration_method,
            optimize_f1_threshold_flag, optimize_ev_threshold_flag, optimize_precision_threshold_flag,
            min_recall_target_for_prec_opt, current_model_name, progress_callback_stages, model_idx
        )
        final_probas_for_eval = probas_calib_draw_test if calibrator and probas_calib_draw_test is not None else raw_test_probas_draw

        # f. Avaliação Final no Teste
        if progress_callback_stages: progress_callback_stages(model_idx, f"{log_prefix}: Avaliando...")
        model_metrics = _evaluate_model_on_test_set(
            final_trained_pipeline, X_test_for_this_model, y_test_data, raw_test_probas_full,
            final_probas_for_eval, f1_thr, ev_thr, prec_thr,
            X_test_odds_data_aligned, odd_draw_col_name, current_model_name
        )
        model_metrics['train_set_size'] = len(y_train_cv_data)

        all_individual_model_results.append({
            'model_name': current_model_name, 'model_object': final_classifier_instance,
            'pipeline_object': final_trained_pipeline, 'scaler': fitted_scaler_for_this_model,
            'calibrator': calibrator, 'params': effective_final_params, 
            'metrics': model_metrics
        })
        if progress_callback_stages: progress_callback_stages(model_idx + 1, f"{current_model_name} OK.")
        logger.info(f"{log_prefix} Concluído. Tempo: {time.time() - current_model_start_time:.2f}s")
    # --- Fim Loop Modelos Individuais ---

    # 5. Lógica do Ensemble 
    if len(all_individual_model_results) >= 2 and n_ensemble_models_config > 0:
        if progress_callback_stages: progress_callback_stages(len(available_model_configs), "Construindo Ensemble...")
        logger.info(f"\n--- Construindo Ensemble com Top {n_ensemble_models_config} Modelos ---")
        
        # Ordena por F1 (chave principal) para selecionar os melhores para o ensemble
        all_individual_model_results.sort(key=lambda r: r['metrics'].get(BEST_MODEL_METRIC, -1.0), reverse=True)
        top_n_for_ensemble = all_individual_model_results[:n_ensemble_models_config]
        
        ensemble_estimators_list = []
        ensemble_needs_scaling_flags = [] # Para verificar se o ensemble precisa de dados escalados
        
        for i_ens, res_ens in enumerate(top_n_for_ensemble):
            estimator_for_ensemble = res_ens.get('pipeline_object')
            if estimator_for_ensemble is None:
                logger.warning(f"Modelo base {res_ens.get('model_name')} sem pipeline_object. Usando model_object (pode faltar sampler).")
                estimator_for_ensemble = res_ens.get('model_object')

            if estimator_for_ensemble:
                unique_estimator_name = f"{res_ens.get('model_name', f'model{i_ens}')}_{i_ens}"
                ensemble_estimators_list.append((unique_estimator_name, clone(estimator_for_ensemble)))
                
                if res_ens.get('scaler') is not None:
                    ensemble_needs_scaling_flags.append(True)
                else:
                    ensemble_needs_scaling_flags.append(False)
            else:
                logger.warning(f"Estimador base {res_ens.get('model_name')} não pôde ser adicionado ao ensemble (objeto ausente).")

        if ensemble_estimators_list:
            voting_clf_ensemble = VotingClassifier(estimators=ensemble_estimators_list, voting='soft', n_jobs=N_JOBS_GRIDSEARCH, verbose=False)
            
            X_train_cv_for_ensemble = X_train_cv_data.copy()
            X_test_for_ensemble_eval = X_test_data.copy()
            scaler_for_ensemble = None

            if any(ensemble_needs_scaling_flags):
                logger.info("Ensemble: Pelo menos um modelo base requer scaling. Aplicando scaling aos dados para o ensemble.")
                first_scaled_model_res = next((r for i, r in enumerate(top_n_for_ensemble) if ensemble_needs_scaling_flags[i]), None)
                if first_scaled_model_res and first_scaled_model_res.get('scaler'):
                    scaler_for_ensemble = clone(first_scaled_model_res.get('scaler')) 
                    X_train_cv_for_ensemble, _, X_test_for_ensemble_eval, _ = scale_features(X_train_cv_for_ensemble, None, X_test_for_ensemble_eval, scaler_type) 
                else:
                    logger.warning("Ensemble: Modelos base precisam de scaling, mas nenhum scaler encontrado. Tentando sem scaling (pode falhar).")

            logger.info("  Ajustando o wrapper VotingClassifier...")
            try:
                voting_clf_ensemble.fit(X_train_cv_for_ensemble, y_train_cv_data) 
                logger.info("  -> Wrapper VotingClassifier ajustado.")

                # Avaliação do Ensemble
                raw_probas_ensemble_full, raw_probas_ensemble_draw = None, None
                if hasattr(voting_clf_ensemble, "predict_proba"):
                    try:
                        raw_probas_ensemble_full = voting_clf_ensemble.predict_proba(X_test_for_ensemble_eval) 
                        if raw_probas_ensemble_full.shape[1] >=2: raw_probas_ensemble_draw = raw_probas_ensemble_full[:,1]
                    except Exception as e_ens_pred: logger.error(f"Erro predict_proba Ensemble: {e_ens_pred}")
                
                logger.warning("Ensemble: AVISO: Calibração e Otim. Limiares no CONJUNTO DE TESTE.")
                ens_calib, ens_probas_calib_draw, ens_f1_thr, ens_ev_thr, ens_prec_thr = _calibrate_and_optimize_thresholds(
                    y_test_data, raw_probas_ensemble_draw, X_test_odds_data_aligned, odd_draw_col_name, calibration_method,
                    optimize_f1_threshold_flag, optimize_ev_threshold_flag, optimize_precision_threshold_flag,
                    min_recall_target_for_prec_opt, "VotingEnsemble", progress_callback_stages, len(available_model_configs) # Progresso para ensemble
                )
                ens_final_probas_eval = ens_probas_calib_draw if ens_calib and ens_probas_calib_draw is not None else raw_probas_ensemble_draw

                ensemble_metrics = _evaluate_model_on_test_set(
                    voting_clf_ensemble, X_test_for_ensemble_eval, y_test_data, raw_probas_ensemble_full,
                    ens_final_probas_eval, ens_f1_thr, ens_ev_thr, ens_prec_thr,
                    X_test_odds_data_aligned, odd_draw_col_name, "VotingEnsemble"
                )
                ensemble_metrics['train_set_size'] = len(y_train_cv_data)

                all_individual_model_results.append({ 
                    'model_name': 'VotingEnsemble', 'model_object': voting_clf_ensemble,
                    'pipeline_object': None, 
                    'scaler': scaler_for_ensemble, 
                    'calibrator': ens_calib, 'params': {'estimators': [e[0] for e in ensemble_estimators_list], 'voting':'soft'},
                    'metrics': ensemble_metrics
                })
                logger.info("  -> Ensemble avaliado e adicionado aos resultados.")
            except Exception as e_ens_fit_eval: logger.error(f"Erro ao treinar/avaliar Ensemble: {e_ens_fit_eval}", exc_info=True)
        else: logger.warning("Nenhum estimador válido para o ensemble.")
    # --- Fim Lógica Ensemble ---

    # 6. Seleção Final e Salvamento
    if progress_callback_stages: progress_callback_stages(num_total_models_expected, "Selecionando/Salvando Melhores...")
    logger.info(f"--- Treinamento Global Concluído ({time.time() - total_training_start_time:.2f} seg) ---")
    if not all_individual_model_results: logger.error("SELEÇÃO FINAL: Nenhum resultado válido."); return False

    try:
        results_df = pd.DataFrame(all_individual_model_results)
        results_df['f1_score_draw_metric'] = results_df['metrics'].apply(lambda m: m.get(BEST_MODEL_METRIC, -1.0)) 
        results_df['roi_metric'] = results_df['metrics'].apply(lambda m: m.get(BEST_MODEL_METRIC_ROI, -np.inf))
        
        # Log do comparativo
        display_cols_log = ['model_name','f1_score_draw_metric','roi_metric'] 
        results_df_log_display = results_df[display_cols_log].copy()
        results_df_log_display['f1_score_draw_metric'] = results_df_log_display['f1_score_draw_metric'].round(4)
        results_df_log_display['roi_metric'] = results_df_log_display['roi_metric'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) and np.isfinite(x) else "N/A")
        logger.info("--- Comparativo Desempenho Modelos (Métrica Principal Teste) ---")
        try: logger.info("\n" + results_df_log_display.sort_values(by='f1_score_draw_metric', ascending=False).to_markdown(index=False))
        except ImportError: logger.info("\n" + results_df_log_display.sort_values(by='f1_score_draw_metric', ascending=False).to_string(index=False))
        logger.info("-" * 80)

        # Salvar Melhor F1
        results_df_sorted_f1 = results_df.sort_values(by='f1_score_draw_metric', ascending=False).reset_index(drop=True)
        best_f1_dict = results_df_sorted_f1.iloc[0].to_dict() if not results_df_sorted_f1.empty else None
        if best_f1_dict:
            logger.info(f"Salvando Melhor por F1: {best_f1_dict.get('model_name', 'N/A')}")
            _save_model_object(best_f1_dict, original_feature_names, BEST_F1_MODEL_SAVE_PATH)
        else: logger.error("Nenhum modelo para salvar como Melhor F1."); return False

        # Salvar Melhor por ROI (ou 2o melhor F1)
        model_for_roi_slot = None
        results_df_valid_roi = results_df[results_df['roi_metric'].notna() & np.isfinite(results_df['roi_metric']) & (results_df['roi_metric'] > -np.inf)] # Garante ROI > -inf
        if not results_df_valid_roi.empty:
            best_roi_dict = results_df_valid_roi.sort_values(by='roi_metric', ascending=False).iloc[0].to_dict()
            if best_f1_dict and (best_roi_dict.get('model_name') != best_f1_dict.get('model_name') or len(results_df_sorted_f1) == 1) :
                model_for_roi_slot = best_roi_dict
            elif len(results_df_sorted_f1) > 1: 
                model_for_roi_slot = results_df_sorted_f1.iloc[1].to_dict() 
            else: 
                model_for_roi_slot = best_f1_dict
        elif best_f1_dict: 
            model_for_roi_slot = best_f1_dict
            logger.warning("Nenhum ROI válido encontrado. Usando melhor F1 para o slot de 'Melhor ROI'.")
        
        if model_for_roi_slot:
            logger.info(f"Salvando para Slot Melhor ROI/2nd F1: {model_for_roi_slot.get('model_name', 'N/A')}")
            _save_model_object(model_for_roi_slot, original_feature_names, BEST_ROI_MODEL_SAVE_PATH)
        else: logger.warning("Nenhum modelo determinado para o slot de 'Melhor ROI'.")

    except Exception as e: logger.error(f"Erro GERAL seleção/salvamento final: {e}", exc_info=True); return False

    logger.info("--- Processo de Treinamento Global Concluído com Sucesso ---")
    return True

def analyze_features(X: pd.DataFrame, y: pd.Series) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """ Analisa features: importância (RF rápido) e correlação. Retorna DFs ou (None, None). """
    logger.info("--- ANÁLISE FEATURES (model_trainer): Iniciando ---")
    imp_df = None
    corr_df = None 

    # --- Input Validation ---
    if X is None or y is None:
        logger.error("ANÁLISE FEATURES: Input X ou y é None.")
        return None, None
    if X.empty or y.empty:
        logger.error("ANÁLISE FEATURES: Input X ou y está vazio.")
        return None, None
    if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
         logger.error("ANÁLISE FEATURES: Input X ou y não é do tipo esperado (DataFrame, Series).")
         return None, None

    # --- Alignment ---
    if not X.index.equals(y.index):
        logger.warning("ANÁLISE FEATURES: Índices X/y não idênticos. Tentando alinhar...")
        try:
            common_index = X.index.intersection(y.index)
            if len(common_index) == 0:
                 logger.error("ANÁLISE FEATURES: Nenhum índice em comum entre X e y após intersection.")
                 return None, None
            X = X.loc[common_index]
            y = y.loc[common_index]
            logger.info(f"ANÁLISE FEATURES: Alinhamento OK. Novo shape X: {X.shape}, y: {y.shape}")
        except Exception as e:
            logger.error(f"ANÁLISE FEATURES: Erro durante alinhamento: {e}", exc_info=True)
            return None, None

    # --- Final check for data after alignment ---
    if X.empty or y.empty:
        logger.error("ANÁLISE FEATURES: X ou y vazio após alinhamento.")
        return None, None

    feature_names = X.columns.tolist()

    # --- 1. Calcular Importância (RF) ---
    logger.info("ANÁLISE FEATURES: Calculando importância RF...")
    try:
        X_rf = X.copy() 
        y_rf = y.copy()

        X_rf = X_rf.replace([np.inf, -np.inf], np.nan)

        if X_rf.isnull().values.any():
             nan_cols = X_rf.columns[X_rf.isnull().any()].tolist()
             logger.warning(f"ANÁLISE FEATURES (RF): NaNs encontrados em X (colunas: {nan_cols}) antes do fit RF. Imputando com mediana.")
             X_rf.fillna(X_rf.median(), inplace=True)
             if X_rf.isnull().values.any():
                 raise ValueError("NaNs persistentes em X após imputação para RF.")

        if y_rf.isnull().values.any():
             logger.error("ANÁLISE FEATURES (RF): NaNs encontrados em y! Não pode treinar RF.")
             raise ValueError("Target variable (y) contains NaNs for RF importance.")

        y_rf = y_rf.astype(int)

        rf_analyzer = RandomForestClassifier(n_estimators=50, max_depth=15, n_jobs=-1, min_samples_leaf=3, random_state=RANDOM_STATE)
        logger.info(f"    -> Fitting RF (X shape: {X_rf.shape}, y shape: {y_rf.shape})")
        rf_analyzer.fit(X_rf, y_rf) 
        logger.info("    -> Fit RF concluído.")

        importances = rf_analyzer.feature_importances_
        imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        imp_df = imp_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        logger.info(f"ANÁLISE FEATURES: Importância calculada OK. Shape: {imp_df.shape}")

    except Exception as e:
        logger.error(f"ANÁLISE FEATURES: Erro GRAVE ao calcular importância RF: {e}", exc_info=True)
        imp_df = None 

    # --- 2. Calcular Correlação ---
    logger.info("ANÁLISE FEATURES: Calculando correlação...")
    try:
        df_temp = X.copy() 
        df_temp['target_IsDraw'] = y 

        cols_for_corr = df_temp.select_dtypes(include=np.number).columns.tolist()
        if 'target_IsDraw' not in cols_for_corr and 'target_IsDraw' in df_temp.columns:
            try:
                df_temp['target_IsDraw'] = pd.to_numeric(df_temp['target_IsDraw'], errors='raise')
                cols_for_corr.append('target_IsDraw')
            except (ValueError, TypeError):
                logger.warning("ANÁLISE FEATURES (Corr): Coluna alvo 'target_IsDraw' não é numérica e não pôde ser convertida. Correlação com alvo não será calculada.")

        df_numeric_temp = df_temp[cols_for_corr].copy()

        if df_numeric_temp.isin([np.inf, -np.inf]).values.any():
            inf_cols = df_numeric_temp.columns[df_numeric_temp.isin([np.inf, -np.inf]).any()].tolist()
            logger.warning(f"ANÁLISE FEATURES (Corr): Valores infinitos encontrados antes de .corr() (colunas: {inf_cols}). Substituindo por NaN.")
            df_numeric_temp.replace([np.inf, -np.inf], np.nan, inplace=True)

        all_nan_cols = df_numeric_temp.columns[df_numeric_temp.isnull().all()].tolist()
        if all_nan_cols:
             logger.warning(f"ANÁLISE FEATURES (Corr): Colunas inteiras com NaN encontradas: {all_nan_cols}. Serão excluídas da correlação.")
             df_numeric_temp.drop(columns=all_nan_cols, inplace=True)
             if 'target_IsDraw' not in df_numeric_temp.columns and 'target_IsDraw' in all_nan_cols:
                  logger.error("ANÁLISE FEATURES (Corr): Coluna alvo foi removida (toda NaN?). Não é possível calcular correlação com o alvo.")
                  corr_df = None 


        if not df_numeric_temp.empty and 'target_IsDraw' in df_numeric_temp.columns:
            logger.info(f"    -> Calculando corr() em df_numeric_temp (shape: {df_numeric_temp.shape})")
            corr_matrix = df_numeric_temp.corr(method='pearson') 

            if 'target_IsDraw' in corr_matrix.columns:
                corr_df = corr_matrix[['target_IsDraw']].sort_values(by='target_IsDraw', ascending=False)
                corr_df = corr_df.drop('target_IsDraw', errors='ignore')
                logger.info(f"ANÁLISE FEATURES: Correlação com o alvo calculada OK. Shape: {corr_df.shape}")
            else:
                 logger.error("ANÁLISE FEATURES (Corr): Coluna 'target_IsDraw' não encontrada na matriz de correlação final.")
                 corr_df = None
        elif df_numeric_temp.empty:
             logger.error("ANÁLISE FEATURES (Corr): DataFrame numérico vazio após tratamento de NaN/Inf.")
             corr_df = None
        else: 
             logger.warning("ANÁLISE FEATURES (Corr): Coluna alvo não disponível para correlação.")
             corr_df = None


    except Exception as e:
        logger.error(f"ANÁLISE FEATURES: Erro GRAVE ao calcular correlação: {e}", exc_info=True)
        corr_df = None 

    logger.info("--- ANÁLISE FEATURES (model_trainer): Concluída ---")
    return imp_df, corr_df 


def optimize_single_model(*args, **kwargs) -> Optional[Tuple[str, Dict, Dict]]:
    """Placeholder - Esta função não é usada no fluxo principal atual."""
    logger.warning("optimize_single_model não está implementada ou não é usada ativamente.")
    return None