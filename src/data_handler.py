# --- src/data_handler.py ---
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import warnings
from io import BytesIO 

from config import (
    FEATURE_COLUMNS,
    ODDS_COLS, GOALS_COLS, ROLLING_WINDOW, SCRAPER_FILTER_LEAGUES,
    TARGET_LEAGUES_INTERNAL_IDS, TARGET_LEAGUES_1,TARGET_LEAGUES_2, HISTORICAL_DATA_PATH_3,
    CSV_HIST_COL_MAP,HISTORICAL_DATA_PATH_1, HISTORICAL_DATA_PATH_2,
    OTHER_ODDS_NAMES, XG_COLS, TARGET_COLUMN, 
    STATS_ROLLING_CONFIG,  STATS_EWMA_CONFIG, 
    PI_RATING_INITIAL, PI_RATING_K_FACTOR, PI_RATING_HOME_ADVANTAGE,
    FEATURE_EPSILON, APPLY_LEAGUE_FILTER_ON_HISTORICAL, 
    FIXTURE_CSV_URL_TEMPLATE, FIXTURE_FETCH_DAY, REQUIRED_FIXTURE_COLS,
    # <<< Importar nomes das novas features >>>
    INTERACTION_P_D_NORM_X_CV_HDA, INTERACTION_P_D_NORM_DIV_CV_HDA,
    INTERACTION_P_D_NORM_X_PIR_DIFF, INTERACTION_P_D_NORM_DIV_PIR_DIFF,
    INTERACTION_ODD_D_X_PIR_DIFF, INTERACTION_ODD_D_DIV_PIR_DIFF,
    PIRATING_MOMENTUM_H, PIRATING_MOMENTUM_A, PIRATING_MOMENTUM_DIFF,
    INTERACTION_PIR_PROBH_X_ODD_H, INTERACTION_AVG_GOLS_MARC_DIFF, INTERACTION_AVG_GOLS_SOFR_DIFF,
    EWMA_VG_H_SHORT, EWMA_VG_A_SHORT, EWMA_CG_H_SHORT, EWMA_CG_A_SHORT,
    EWMA_GolsMarc_H_LONG, EWMA_GolsMarc_A_LONG, EWMA_GolsSofr_H_LONG, EWMA_GolsSofr_A_LONG,
    EWMA_SPAN_SHORT, EWMA_SPAN_LONG,
    
)
from typing import Tuple, Optional, List, Dict, Any
from tqdm import tqdm
import time  # Embora não usado diretamente, pode ser útil para debugging
from datetime import date, timedelta, datetime  # datetime não usado diretamente, mas pode ser útil
import requests
from urllib.error import HTTPError, URLError
from scipy.stats import poisson
from logger_config import setup_logger
import os
import logging

logger = setup_logger('DataHandler')
logger_dh = setup_logger('DataHandlerMetrics')

# --- Funções Auxiliares ---
class BettingMetricsCalculator:

    @staticmethod
    def roi(y_true: pd.Series,
            y_pred_class: np.ndarray, 
            odds_data: Optional[pd.DataFrame],
            odd_col_name: str) -> Optional[float]:

        if odds_data is None or odd_col_name not in odds_data.columns:
            logger_dh.warning("ROI: Dados de odds ou nome da coluna ausentes.")
            return None
        if not isinstance(y_true.index, pd.Index) or not isinstance(odds_data.index, pd.Index):
            logger_dh.warning("ROI: y_true ou odds_data não têm um índice Pandas válido.")
            return None

        try:
            if not isinstance(y_pred_class, pd.Series) or not y_pred_class.index.equals(y_true.index):
                if len(y_pred_class) == len(y_true):
                    y_pred_series = pd.Series(y_pred_class, index=y_true.index)
                else:
                    logger_dh.error(f"ROI: y_pred_class (len {len(y_pred_class)}) não pode ser alinhado com y_true (len {len(y_true)}).")
                    return None
            else:
                y_pred_series = y_pred_class

            common_index = y_true.index.intersection(odds_data.index).intersection(y_pred_series.index)
            if not common_index.any():
                logger_dh.warning("ROI: Nenhum índice em comum entre y_true, odds_data e y_pred_series.")
                return 0.0

            y_true_common = y_true.loc[common_index]
            y_pred_common = y_pred_series.loc[common_index]
            odds_common = pd.to_numeric(odds_data.loc[common_index, odd_col_name], errors='coerce')

            bet_indices = common_index[y_pred_common == 1]
            if not bet_indices.any():
                return 0.0  

            actuals_on_bets = y_true_common.loc[bet_indices]
            odds_on_bets = odds_common.loc[bet_indices]

            profit = 0.0
            valid_bets_count = 0
            for idx in bet_indices:
                odd_val = odds_on_bets.get(idx) 
                actual_val = actuals_on_bets.get(idx)

                if pd.notna(odd_val) and odd_val > 1 and pd.notna(actual_val):
                    profit += (odd_val - 1) if actual_val == 1 else -1
                    valid_bets_count += 1
            
            if valid_bets_count == 0:
                return 0.0
            return (profit / valid_bets_count) * 100.0

        except AttributeError as ae: 
            logger_dh.error(f"ROI AttributeError: {ae}. Verifique os tipos de input.", exc_info=True)
            return None
        except Exception as e:
            logger_dh.error(f"ROI Erro inesperado: {e}", exc_info=True)
            return None

    @staticmethod
    def roi_with_threshold(y_true: pd.Series,
                               y_proba: np.ndarray, 
                               threshold: float,
                               odds_data: Optional[pd.DataFrame],
                               odd_col_name: str) -> Tuple[Optional[float], int, Optional[float]]:

        if odds_data is None or odd_col_name not in odds_data.columns:
            logger_dh.warning("ROI Thr: Dados de odds ou nome da coluna ausentes.")
            return None, 0, None
        try:
            if not isinstance(y_proba, pd.Series) or not y_proba.index.equals(y_true.index):
                if len(y_proba) == len(y_true):
                    y_proba_series = pd.Series(y_proba, index=y_true.index)
                else:
                    logger_dh.error(f"ROI Thr: y_proba (len {len(y_proba)}) não pode ser alinhado com y_true (len {len(y_true)}).")
                    return None, 0, None
            else:
                y_proba_series = y_proba

            common_index = y_true.index.intersection(odds_data.index).intersection(y_proba_series.index)
            if not common_index.any(): return 0.0, 0, 0.0

            y_true_common = y_true.loc[common_index]
            odds_common = pd.to_numeric(odds_data.loc[common_index, odd_col_name], errors='coerce')
            y_proba_common = y_proba_series.loc[common_index]

            bet_indices = common_index[y_proba_common > threshold]

            if not bet_indices.any(): return 0.0, 0, 0.0

            actuals_on_bets = y_true_common.loc[bet_indices]
            odds_on_bets = odds_common.loc[bet_indices]

            profit_calc = 0.0; valid_bets_count = 0
            for idx in bet_indices:
                odd_val = odds_on_bets.get(idx)
                actual_val = actuals_on_bets.get(idx)
                if pd.notna(odd_val) and odd_val > 1 and pd.notna(actual_val):
                    profit_calc += (odd_val - 1) if actual_val == 1 else -1
                    valid_bets_count += 1
            
            profit = profit_calc
            roi_value = (profit / valid_bets_count) * 100.0 if valid_bets_count > 0 else 0.0
            return roi_value, valid_bets_count, profit
        except AttributeError as ae:
            logger_dh.error(f"ROI Thr AttributeError: {ae}. Verifique tipos.", exc_info=True)
            return None, 0, None
        except Exception as e:
            logger_dh.error(f"ROI Thr Erro Geral: {e}", exc_info=True)
            return None, 0, None


    @staticmethod
    def metrics_with_ev(y_true: pd.Series,
                            y_proba_calibrated: np.ndarray, 
                            ev_threshold: float,
                            odds_data: Optional[pd.DataFrame],
                            odd_col_name: str) -> Tuple[Optional[float], int, Optional[float]]:

        if odds_data is None or odd_col_name not in odds_data.columns:
            logger_dh.warning(f"EV Metr: Odds ausentes.")
            return None, 0, None
        try:
            if not isinstance(y_proba_calibrated, pd.Series) or not y_proba_calibrated.index.equals(y_true.index):
                if len(y_proba_calibrated) == len(y_true):
                    y_proba_series = pd.Series(y_proba_calibrated, index=y_true.index)
                else:
                    logger_dh.error(f"EV Metr: y_proba_calibrated (len {len(y_proba_calibrated)}) não pode ser alinhado com y_true (len {len(y_true)}).")
                    return None, 0, None
            else:
                y_proba_series = y_proba_calibrated

            common_index = y_true.index.intersection(odds_data.index).intersection(y_proba_series.index)
            if not common_index.any(): return 0.0, 0, 0.0

            y_true_common = y_true.loc[common_index]
            odds_common = pd.to_numeric(odds_data.loc[common_index, odd_col_name], errors='coerce')
            y_proba_common = y_proba_series.loc[common_index]

            valid_mask = odds_common.notna() & y_proba_common.notna() & (odds_common > 1)
            ev_series = pd.Series(np.nan, index=common_index) 
            if valid_mask.any():
                prob_ok = y_proba_common[valid_mask]; odds_ok = odds_common[valid_mask]
                ev_calculated_values = (prob_ok * (odds_ok - 1)) - ((1 - prob_ok) * 1)
                ev_series.loc[valid_mask] = ev_calculated_values

            bet_indices = common_index[ev_series > ev_threshold]

            if not bet_indices.any(): return 0.0, 0, 0.0

            actuals_on_bets = y_true_common.loc[bet_indices]
            odds_on_bets = odds_common.loc[bet_indices]

            profit_calc = 0.0; valid_bets_count = 0
            for idx in bet_indices:
                odd_val = odds_on_bets.get(idx)
                actual_val = actuals_on_bets.get(idx)
                if pd.notna(odd_val) and odd_val > 1 and pd.notna(actual_val):
                    profit_calc += (odd_val - 1) if actual_val == 1 else -1
                    valid_bets_count += 1
            
            profit = profit_calc
            roi_value = (profit / valid_bets_count) * 100.0 if valid_bets_count > 0 else 0.0
            return roi_value, valid_bets_count, profit
        except AttributeError as ae:
            logger_dh.error(f"EV Metr AttributeError: {ae}. Verifique tipos.", exc_info=True)
            return None, 0, None
        except Exception as e:
            logger_dh.error(f"EV Metr Erro Geral - {e}", exc_info=True)
            return None, 0, None
    
EXPECTED_STAT_CSV_COLS = ['Shots_H', 'Shots_A', 'ShotsOnTarget_H', 'ShotsOnTarget_A', 'Corners_H_FT', 'Corners_A_FT']

def _read_historical_file(file_path: str) -> Optional[pd.DataFrame]:

    base_filename = os.path.basename(file_path)
    file_ext = os.path.splitext(file_path)[1].lower()
    df_part = None

    logger.info(f"Lendo {base_filename} (formato: {file_ext})...")

    if file_ext == '.xlsx':
        try:
            df_part = pd.read_excel(file_path, engine='openpyxl')
        except Exception as e_excel:
            logger.error(f"Falha ao ler arquivo Excel {base_filename}: {e_excel}", exc_info=True)
            return None
    elif file_ext == '.csv':
        try:
            df_part = pd.read_csv(file_path, low_memory=False, encoding='ISO-8859-1')
        except UnicodeDecodeError:
            logger.warning(f"Fallback para UTF-8 em {base_filename}...")
            try:
                df_part = pd.read_csv(file_path, low_memory=False, encoding='utf-8')
            except Exception as e_csv_utf8:
                logger.error(f"Falha ao ler CSV {base_filename} com UTF-8: {e_csv_utf8}", exc_info=True)
                return None
        except Exception as e_csv:
            logger.error(f"Falha ao ler CSV {base_filename}: {e_csv}", exc_info=True)
            return None
    else:
        logger.error(f"Formato de arquivo não suportado para {base_filename}: {file_ext}")
        return None

    if df_part is None or df_part.empty:
        logger.warning(f"DataFrame vazio ou falha na leitura de {base_filename}.")
        return None

    logger.info(f"  -> Lido {df_part.shape[0]} linhas de {base_filename}.")
    logger.debug(f"    Colunas originais: {list(df_part.columns)}")
    return df_part

def _apply_column_mapping_and_league_logic(
    df_input: pd.DataFrame,
    specific_league_map: Dict[str, str], 
    csv_to_internal_col_map: Dict[str, str], 
    apply_league_filter_flag: bool, 
    target_internal_league_ids: List[str], 
    base_filename: str
) -> Optional[pd.DataFrame]:

    df = df_input.copy()
    internal_league_col_name = 'League' 

    raw_goal_col_home = None
    raw_goal_col_away = None
    for k_csv, v_internal in csv_to_internal_col_map.items():
        if v_internal == GOALS_COLS.get('home') and k_csv in df.columns:
            raw_goal_col_home = k_csv
        if v_internal == GOALS_COLS.get('away') and k_csv in df.columns:
            raw_goal_col_away = k_csv

    cols_to_ensure_exist_initially = list(EXPECTED_STAT_CSV_COLS)
    if raw_goal_col_home: cols_to_ensure_exist_initially.append(raw_goal_col_home)
    if raw_goal_col_away: cols_to_ensure_exist_initially.append(raw_goal_col_away)

    for col in cols_to_ensure_exist_initially:
        if col not in df.columns:
            if col not in [raw_goal_col_home, raw_goal_col_away]:
                 df[col] = np.nan

    # --- ETAPA A: Mapeamento de Liga ---
    original_league_col_name_from_csv_map = None
    for csv_col, internal_col in csv_to_internal_col_map.items():
        if internal_col == internal_league_col_name and csv_col in df.columns:
            original_league_col_name_from_csv_map = csv_col
            break

    df_post_league_map = df.copy() 

    if original_league_col_name_from_csv_map:
        logger.info(f"({base_filename}) Mapeando coluna de liga '{original_league_col_name_from_csv_map}'...")
        df_post_league_map[original_league_col_name_from_csv_map] = df_post_league_map[original_league_col_name_from_csv_map].astype(str).str.strip()
        df_post_league_map['_InternalLeagueID_Temp'] = df_post_league_map[original_league_col_name_from_csv_map].map(specific_league_map)
        unmapped_mask = df_post_league_map['_InternalLeagueID_Temp'].isnull()
        original_unmapped_names = df_post_league_map.loc[unmapped_mask, original_league_col_name_from_csv_map].unique()
        if len(original_unmapped_names) > 0:
            logger.warning(f"  ({base_filename}) Ligas NÃO mapeadas para ID Interno: {list(original_unmapped_names)}")
        df_post_league_map['_InternalLeagueID_Temp'].fillna('OTHER_LEAGUE_ID', inplace=True)
    else:
        logger.warning(f"({base_filename}) Coluna original para '{internal_league_col_name}' não mapeada. Criando coluna de Liga placeholder.")
        df_post_league_map['_InternalLeagueID_Temp'] = 'UNMAPPED_LEAGUE' 

    # --- ETAPA B: Filtragem Condicional por Liga ---
    df_to_process_further = df_post_league_map
    if apply_league_filter_flag and '_InternalLeagueID_Temp' in df_to_process_further.columns:
        logger.info(f"  ({base_filename}) APLICANDO FILTRO por TARGET_LEAGUES_INTERNAL_IDS...")
        initial_count_filter = len(df_to_process_further)
        df_to_process_further = df_to_process_further[df_to_process_further['_InternalLeagueID_Temp'].isin(target_internal_league_ids)].copy()
        logger.info(f"    -> Filtro de Liga: {len(df_to_process_further)}/{initial_count_filter} jogos restantes.")
        if df_to_process_further.empty:
            logger.warning(f"  ({base_filename}) Nenhum jogo restante após filtro de liga.")
            return pd.DataFrame()
    elif apply_league_filter_flag:
        logger.warning(f"  ({base_filename}) Filtro de Liga ATIVO, mas coluna _InternalLeagueID_Temp não encontrada. Filtro não aplicado.")


    # --- ETAPA C: Mapeamento Geral de Colunas e Seleção Final ---
    cols_to_rename_dict = {
        k_csv: v_internal for k_csv, v_internal in csv_to_internal_col_map.items()
        if k_csv in df_to_process_further.columns
    }
    original_cols_to_keep = list(cols_to_rename_dict.keys())

    for stat_col in EXPECTED_STAT_CSV_COLS:
        if stat_col in df_to_process_further.columns and stat_col not in original_cols_to_keep:
            original_cols_to_keep.append(stat_col)

    if '_InternalLeagueID_Temp' in df_to_process_further.columns:
        if '_InternalLeagueID_Temp' not in original_cols_to_keep: 
             original_cols_to_keep.append('_InternalLeagueID_Temp')
        cols_to_rename_dict['_InternalLeagueID_Temp'] = internal_league_col_name
    else: 
        if internal_league_col_name not in cols_to_rename_dict.values():
            if internal_league_col_name not in df_to_process_further.columns and \
               not any(v == internal_league_col_name for v in cols_to_rename_dict.values()):
                df_to_process_further[internal_league_col_name] = 'UNMAPPED_LEAGUE_FINAL'
                if internal_league_col_name not in original_cols_to_keep:
                     original_cols_to_keep.append(internal_league_col_name)

    final_original_cols_to_select = [col for col in original_cols_to_keep if col in df_to_process_further.columns]
    final_original_cols_to_select = list(dict.fromkeys(final_original_cols_to_select)) # Mantém ordem e remove duplicatas

    if not final_original_cols_to_select:
        logger.warning(f"({base_filename}) Nenhuma coluna selecionada para manter. Retornando DataFrame vazio.")
        return pd.DataFrame()

    df_selected = df_to_process_further[final_original_cols_to_select].copy()

    actual_rename_map = {k: v for k, v in cols_to_rename_dict.items() if k in df_selected.columns}
    df_processed = df_selected.rename(columns=actual_rename_map)

    logger.debug(f"({base_filename}) Colunas após seleção e rename final: {list(df_processed.columns)}")

    if df_processed.empty:
        logger.warning(f"({base_filename}) DataFrame processado resultou vazio.")
    return df_processed

def _load_single_historical_file(
    file_path: str,
    specific_league_map: Dict[str, str],
    csv_to_internal_col_map: Dict[str, str], 
    apply_league_filter_flag: bool,
    target_internal_league_ids: List[str]
) -> Optional[pd.DataFrame]:

    base_filename = os.path.basename(file_path) 
    df_raw = _read_historical_file(file_path)
    if df_raw is None:
        return None 

    try:
        df_processed_single = _apply_column_mapping_and_league_logic(
            df_raw,
            specific_league_map,
            csv_to_internal_col_map,
            apply_league_filter_flag,
            target_internal_league_ids,
            base_filename 
        )
        return df_processed_single 
    except Exception as e:
        logger.error(f"Erro crítico ao aplicar mapeamentos/filtros em {base_filename}: {e}", exc_info=True)
        return None

def _post_concat_processing(df_input: pd.DataFrame) -> Optional[pd.DataFrame]:

    if df_input.empty:
        logger.error("DF histórico vazio recebido para pós-processamento.")
        return None
    df = df_input.copy()
    logger.info("Iniciando processamento comum pós-concatenação...")

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        if df['Date'].isnull().any():
            logger.warning("Algumas datas não puderam ser convertidas com o formato padrão. Tentando formatos alternativos...")
            df['Date_temp'] = pd.to_datetime(df_input['Date'], errors='coerce', dayfirst=True) 
            df['Date'] = df['Date'].fillna(df['Date_temp'])
            df.drop(columns=['Date_temp'], inplace=True, errors='ignore')

        df.dropna(subset=['Date'], inplace=True)
        logger.info(f"  -> Conversão de 'Date' concluída. {df['Date'].isnull().sum()} NaNs restantes em Date (deveria ser 0).")
    else:
        logger.error("Coluna 'Date' ausente no DataFrame concatenado. Não é possível prosseguir.")
        return None

    if df.empty:
        logger.error("DF vazio após tratamento da coluna 'Date'.")
        return None

    numeric_cols_to_convert = (
        list(GOALS_COLS.values()) + list(ODDS_COLS.values()) +
        OTHER_ODDS_NAMES + list(XG_COLS.values()) + EXPECTED_STAT_CSV_COLS
    )
    for col in numeric_cols_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if col in ODDS_COLS.values() or col in OTHER_ODDS_NAMES:
                df.loc[df[col] <= 1, col] = np.nan

            if col in XG_COLS.values():
                df.loc[df[col] < 0, col] = np.nan
        else:
            logger.debug(f"  Coluna numérica esperada '{col}' ausente no DataFrame concatenado para conversão.")

    essential_cols_for_dropna = ['Date', 'Home', 'Away'] + list(GOALS_COLS.values()) + list(ODDS_COLS.values())
    cols_to_dropna_present = [c for c in essential_cols_for_dropna if c in df.columns]

    logger.info(f"  Verificando NaNs essenciais em: {cols_to_dropna_present}")
    initial_rows = len(df)
    df.dropna(subset=cols_to_dropna_present, inplace=True)
    rows_dropped = initial_rows - len(df)
    if rows_dropped > 0:
        logger.info(f"  Removidas {rows_dropped} linhas devido a NaNs em colunas essenciais.")

    if df.empty:
        logger.error("Nenhum jogo restante após dropna essencial no DataFrame concatenado.")
        return None

    df = df.sort_values(by='Date').reset_index(drop=True)
    logger.info("  DataFrame histórico ordenado por data.")

    logger.info("  Calculando Pi-Ratings finais...")
    df = calculate_pi_ratings(df) 

    logger.info("Processamento pós-concatenação concluído.")
    return df

def load_historical_data() -> Optional[pd.DataFrame]:

    file_configs = [
        # {'path': HISTORICAL_DATA_PATH_1, 'league_map': TARGET_LEAGUES_1}, # Exemplo
        # {'path': HISTORICAL_DATA_PATH_2, 'league_map': TARGET_LEAGUES_2}, # Exemplo
        {'path': HISTORICAL_DATA_PATH_3, 'league_map': TARGET_LEAGUES_1} 
    ]
    global_csv_col_map = CSV_HIST_COL_MAP
    apply_filter_hist = APPLY_LEAGUE_FILTER_ON_HISTORICAL
    target_leagues_ids = TARGET_LEAGUES_INTERNAL_IDS

    log_msg_filter = f"(FILTRO DE LIGA HISTÓRICO ATIVO por TARGET_LEAGUES_INTERNAL_IDS)" if apply_filter_hist else "(FILTRO DE LIGA HISTÓRICO DESATIVADO)"
    logger.info(f"Iniciando carregamento/mapeamento histórico {log_msg_filter}...")

    all_dfs_processed_parts: List[pd.DataFrame] = []

    for config in file_configs:
        file_path = config['path']
        specific_league_mapping = config['league_map'] 

        logger.info(f"-- Processando arquivo: {os.path.basename(file_path)} --")
        df_part_processed = _load_single_historical_file(
            file_path,
            specific_league_mapping,
            global_csv_col_map, 
            apply_filter_hist,
            target_leagues_ids
        )

        if df_part_processed is not None and not df_part_processed.empty:
            all_dfs_processed_parts.append(df_part_processed)
            logger.info(f"  -> Arquivo {os.path.basename(file_path)} processado com sucesso (Shape: {df_part_processed.shape}).")
        elif df_part_processed is not None and df_part_processed.empty:
            logger.warning(f"  -> Arquivo {os.path.basename(file_path)} resultou em DataFrame vazio após processamento (ex: filtro).")
        else:
            logger.error(f"  -> Falha ao processar arquivo {os.path.basename(file_path)}. Pulando.")

    if not all_dfs_processed_parts:
        logger.error("Nenhum arquivo histórico pôde ser carregado ou processado com dados válidos.")
        return None

    logger.info(f"Concatenando {len(all_dfs_processed_parts)} DataFrames processados...")
    try:
        df_combined = pd.concat(all_dfs_processed_parts, ignore_index=True, sort=False)
        logger.info(f"DataFrame histórico combinado antes do pós-processamento: {df_combined.shape}")
    except Exception as e:
        logger.error(f"Erro ao concatenar DataFrames: {e}", exc_info=True)
        return None

    if df_combined.empty:
        logger.error("DataFrame combinado está vazio antes do pós-processamento.")
        return None

    df_final_processed = _post_concat_processing(df_combined)

    if df_final_processed is None or df_final_processed.empty:
        logger.error("Processamento pós-concatenação falhou ou resultou em DataFrame vazio.")
        return None

    logger.info(f"Carregamento e processamento de dados históricos concluídos {log_msg_filter}. Shape Final: {df_final_processed.shape}")
    optional_nan_counts = df_final_processed.isnull().sum()
    optional_nan_counts = optional_nan_counts[optional_nan_counts > 0]
    if not optional_nan_counts.empty:
        logger.info(f"Contagem de NaNs finais (em colunas com algum NaN):\n{optional_nan_counts}")
    else:
        logger.info("Nenhum NaN opcional restante detectado no DataFrame final.")

    return df_final_processed

def _get_final_team_stats(historical_df_processed: pd.DataFrame) -> pd.DataFrame:

    logger.info("Extraindo o estado final das estatísticas dos times do histórico...")
    
    stat_cols = [col for col in historical_df_processed.columns if col.startswith(('Media_', 'Std_', 'EWMA_', 'FA_', 'FD_', 'PiRating_'))]
    
    df_home = historical_df_processed[['Home'] + [c for c in stat_cols if c.endswith('_H')]].rename(columns={'Home': 'Team'})
    df_away = historical_df_processed[['Away'] + [c for c in stat_cols if c.endswith('_A')]].rename(columns={'Away': 'Team'})

    df_home.columns = ['Team'] + [c.replace('_H', '') for c in df_home.columns if c.endswith('_H')]
    df_away.columns = ['Team'] + [c.replace('_A', '') for c in df_away.columns if c.endswith('_A')]

    final_stats = pd.concat([df_home, df_away]).sort_index(kind='stable').drop_duplicates(subset=['Team'], keep='last')
    
    logger.info(f"Estado final extraído para {len(final_stats)} times.")
    return final_stats.set_index('Team')

def _get_training_medians(X_train: pd.DataFrame) -> pd.Series:

    logger.info("Calculando medianas do conjunto de treino para imputação futura.")
    return X_train.median()
   
def calculate_probabilities(df: pd.DataFrame, epsilon=1e-6) -> pd.DataFrame:
    df_calc = df.copy()
    required_odds = list(ODDS_COLS.values())  
    if not all(c in df_calc.columns for c in required_odds):
        logger.warning("Odds 1x2 ausentes para calcular Probabilidades.")
        df_calc[['p_H', 'p_D', 'p_A']] = np.nan
        return df_calc

    for col in required_odds:
        df_calc[col] = pd.to_numeric(df_calc[col], errors='coerce')

    odd_h = df_calc[ODDS_COLS['home']].replace(0, np.nan) 
    odd_d = df_calc[ODDS_COLS['draw']].replace(0, np.nan) 
    odd_a = df_calc[ODDS_COLS['away']].replace(0, np.nan) 

    df_calc['p_H'] = 1 / odd_h
    df_calc['p_D'] = 1 / odd_d
    df_calc['p_A'] = 1 / odd_a
    logger.info("Probabilidades Implícitas (p_H, p_D, p_A) calculadas.")
    return df_calc

def calculate_normalized_probabilities(df: pd.DataFrame, epsilon=1e-6) -> pd.DataFrame:
    df_calc = df.copy()
    prob_cols = ['p_H', 'p_D', 'p_A']
    if not all(c in df_calc.columns for c in prob_cols):
        logger.warning("Probabilidades (p_H/D/A) ausentes para normalização.")
        df_calc[['p_H_norm', 'p_D_norm', 'p_A_norm', 'abs_ProbDiff_Norm']] = np.nan
        return df_calc

    df_calc['Overround'] = df_calc['p_H'] + df_calc['p_D'] + df_calc['p_A']
    df_calc['Overround'] = df_calc['Overround'].replace(0, epsilon)

    df_calc['p_H_norm'] = df_calc['p_H'] / df_calc['Overround']
    df_calc['p_D_norm'] = df_calc['p_D'] / df_calc['Overround']
    df_calc['p_A_norm'] = df_calc['p_A'] / df_calc['Overround']
    df_calc['abs_ProbDiff_Norm'] = abs(df_calc['p_H_norm'] - df_calc['p_A_norm'])
    logger.info("Probabilidades Normalizadas (p_X_norm, abs_ProbDiff_Norm) calculadas.")
    return df_calc.drop(columns=['Overround'], errors='ignore')

def calculate_general_rolling_stats(
    df: pd.DataFrame, 
    stats_configs: List[Dict[str, Any]], 
    default_window: int = ROLLING_WINDOW
) -> pd.DataFrame:

    df_calc = df.copy()
    logger.info(f"Iniciando cálculo OTIMIZADO Geral de Stats Rolling (Janela Padrão={default_window})...")

    df_to_melt = df_calc.reset_index().rename(columns={'index': 'match_idx'})

    for config in stats_configs:
        agg_func = config.get('agg_func')
        stat_prefix = config.get('output_prefix')
        window = config.get('window', default_window)
        min_p = config.get('min_periods', 2 if agg_func == np.std else 1) # Default 2 para std, 1 para outros
        base_h = config.get('base_col_h')
        base_a = config.get('base_col_a')
        
        if not all([agg_func, stat_prefix, base_h, base_a]):
            logger.warning(f"Configuração de stat incompleta: {config}. Pulando.")
            continue
        
        if base_h not in df_to_melt.columns or base_a not in df_to_melt.columns:
            logger.error(f"Erro Rolling Stat: Colunas base '{base_h}'/'{base_a}' não encontradas. Pulando config para '{stat_prefix}'.")
            continue
            
        logger.debug(f"  Processando: '{stat_prefix}' com a função '{agg_func.__name__}'")

        new_col_h = f"{stat_prefix}_H"
        new_col_a = f"{stat_prefix}_A"

        # 1. Melt
        cols_for_this_melt = ['match_idx', 'Home', 'Away', base_h, base_a]
        df_long = pd.melt(
            df_to_melt[cols_for_this_melt],
            id_vars=['match_idx', 'Home', 'Away'],
            value_vars=[base_h, base_a],
            var_name='Stat_Source_Column',
            value_name='Stat_Value'
        )

        # 2. Determinar o time corretamente
        df_long['Team'] = np.where(
            df_long['Stat_Source_Column'] == base_h,
            df_long['Home'],
            df_long['Away']
        )
        
        # 3. Ordena, converte para numérico e calcula a estatística móvel
        df_long = df_long.sort_values(by='match_idx', kind='stable')
        df_long['Stat_Value'] = pd.to_numeric(df_long['Stat_Value'], errors='coerce')
        
        if agg_func == np.std:
            rolling_agg_func = lambda x: x.rolling(window, min_periods=min_p).std(ddof=0)
        else:
            rolling_agg_func = lambda x: x.rolling(window, min_periods=min_p).agg(agg_func)

        df_long['Rolling_Stat'] = (
            df_long.groupby('Team')['Stat_Value']
                   .transform(lambda x: rolling_agg_func(x).shift(1))
        )
        
        # 4. Mapeia os resultados de volta para o DataFrame original
        mapper = df_long.set_index(['match_idx', 'Team'])['Rolling_Stat']
        df_calc[new_col_h] = df_calc.index.map(lambda idx: mapper.get((idx, df_calc.loc[idx, 'Home'])))
        df_calc[new_col_a] = df_calc.index.map(lambda idx: mapper.get((idx, df_calc.loc[idx, 'Away'])))

        logger.info(f"  -> '{new_col_h}' e '{new_col_a}' calculados.")
        
    return df_calc

def calculate_binned_features(df: pd.DataFrame) -> pd.DataFrame:

    df_calc = df.copy()
    odd_d_col = ODDS_COLS['draw']
    if odd_d_col not in df_calc.columns:
        logger.warning("Odd de Empate ausente para binning.")
        df_calc['Odd_D_Cat'] = np.nan
        return df_calc

    odd_d = pd.to_numeric(df_calc[odd_d_col], errors='coerce')

    bins = [-np.inf, 2.90, 3.40, np.inf]
    labels = [1, 2, 3]

    df_calc['Odd_D_Cat'] = pd.cut(odd_d, bins=bins, labels=labels, right=True)
    df_calc['Odd_D_Cat'] = df_calc['Odd_D_Cat'].cat.codes + 1
    df_calc['Odd_D_Cat'] = df_calc['Odd_D_Cat'].replace(0, np.nan)

    logger.info(f"Binning ('Odd_D_Cat') calculado a partir de '{odd_d_col}'.")
    return df_calc


def calculate_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula CV_HDA, Diff_Media_CG e NOVAS features de interação."""
    df_calc = df.copy()
    logger.info("Calculando features derivadas (CV_HDA, Diff_Media_CG, Novas Interações)...")
    epsilon = FEATURE_EPSILON 

    # CV_HDA
    if all(c in df_calc.columns for c in ODDS_COLS.values()):
        odds_matrix = df_calc[list(ODDS_COLS.values())].apply(pd.to_numeric, errors='coerce')
        mean_odds = odds_matrix.mean(axis=1)
        std_odds = odds_matrix.std(axis=1)
        df_calc['CV_HDA'] = std_odds.div(mean_odds.replace(0, epsilon)).fillna(0)
    else: 
        logger.warning("Odds 1x2 ausentes p/ CV_HDA."); 
        df_calc['CV_HDA'] = np.nan

    if 'Media_CG_H' in df_calc.columns and 'Media_CG_A' in df_calc.columns:
        df_calc['Diff_Media_CG'] = df_calc['Media_CG_H'] - df_calc['Media_CG_A']
    else: 
        logger.warning("Médias CG ausentes p/ Diff_Media_CG."); 
        df_calc['Diff_Media_CG'] = np.nan

    if 'p_D_norm' in df_calc.columns and 'CV_HDA' in df_calc.columns:
        df_calc[INTERACTION_P_D_NORM_X_CV_HDA] = df_calc['p_D_norm'] * df_calc['CV_HDA']
        df_calc[INTERACTION_P_D_NORM_DIV_CV_HDA] = df_calc['p_D_norm'] / (df_calc['CV_HDA'] + epsilon)
    else: 
        logger.warning("p_D_norm ou CV_HDA ausente."); 
        df_calc[[INTERACTION_P_D_NORM_X_CV_HDA, INTERACTION_P_D_NORM_DIV_CV_HDA]] = np.nan

    if 'PiRating_Diff' in df_calc.columns:
        abs_pi_rating_diff = df_calc['PiRating_Diff'].abs()
        if 'p_D_norm' in df_calc.columns:
            df_calc[INTERACTION_P_D_NORM_X_PIR_DIFF] = df_calc['p_D_norm'] * abs_pi_rating_diff
            df_calc[INTERACTION_P_D_NORM_DIV_PIR_DIFF] = df_calc['p_D_norm'] / (abs_pi_rating_diff + epsilon)
        else: 
            logger.warning("p_D_norm ausente."); 
        df_calc[[INTERACTION_P_D_NORM_X_PIR_DIFF, INTERACTION_P_D_NORM_DIV_PIR_DIFF]] = np.nan
        odd_d_col = ODDS_COLS['draw']
        if odd_d_col in df_calc.columns:
            df_calc[INTERACTION_ODD_D_X_PIR_DIFF] = df_calc[odd_d_col] * abs_pi_rating_diff
            df_calc[INTERACTION_ODD_D_DIV_PIR_DIFF] = df_calc[odd_d_col] / (abs_pi_rating_diff + epsilon)
        else: 
            logger.warning(f"{odd_d_col} ausente."); 
        df_calc[[INTERACTION_ODD_D_X_PIR_DIFF, INTERACTION_ODD_D_DIV_PIR_DIFF]] = np.nan
    else: 
        logger.warning("PiRating_Diff ausente."); 
    df_calc[[INTERACTION_P_D_NORM_X_PIR_DIFF, INTERACTION_P_D_NORM_DIV_PIR_DIFF, INTERACTION_ODD_D_X_PIR_DIFF, INTERACTION_ODD_D_DIV_PIR_DIFF]] = np.nan

    odd_h_col = ODDS_COLS['home']
    if 'PiRating_Prob_H' in df_calc.columns and odd_h_col in df_calc.columns:
        df_calc[INTERACTION_PIR_PROBH_X_ODD_H] = df_calc['PiRating_Prob_H'] * df_calc[odd_h_col]
    else: 
        logger.warning("PiRating_Prob_H ou Odd_H_FT ausente."); 
    df_calc[INTERACTION_PIR_PROBH_X_ODD_H] = np.nan

    marc_h = EWMA_GolsMarc_H_LONG if EWMA_GolsMarc_H_LONG in df_calc.columns else 'Media_GolsMarcados_H'
    marc_a = EWMA_GolsMarc_A_LONG if EWMA_GolsMarc_A_LONG in df_calc.columns else 'Media_GolsMarcados_A'
    if marc_h in df_calc.columns and marc_a in df_calc.columns:
        df_calc[INTERACTION_AVG_GOLS_MARC_DIFF] = df_calc[marc_h] - df_calc[marc_a]
    else: logger.warning(f"{marc_h} ou {marc_a} ausente."); df_calc[INTERACTION_AVG_GOLS_MARC_DIFF] = np.nan

    sofr_h = EWMA_GolsSofr_H_LONG if EWMA_GolsSofr_H_LONG in df_calc.columns else 'Media_GolsSofridos_H'
    sofr_a = EWMA_GolsSofr_A_LONG if EWMA_GolsSofr_A_LONG in df_calc.columns else 'Media_GolsSofridos_A'
    if sofr_h in df_calc.columns and sofr_a in df_calc.columns:
        df_calc[INTERACTION_AVG_GOLS_SOFR_DIFF] = df_calc[sofr_h] - df_calc[sofr_a]
    else: logger.warning(f"{sofr_h} ou {sofr_a} ausente."); df_calc[INTERACTION_AVG_GOLS_SOFR_DIFF] = np.nan

    logger.info("Features de interação calculadas (incluindo novas).")
    return df_calc

def calculate_poisson_draw_prob(
    df: pd.DataFrame,
    avg_goals_home_league: float,
    avg_goals_away_league: float,
    max_goals: int = 6
    ) -> pd.DataFrame:

    df_calc = df.copy()
    required_cols = ['FA_H', 'FD_A', 'FA_A', 'FD_H']

    if not all(c in df_calc.columns for c in required_cols):
        logger.warning("Colunas de Força FA/FD ausentes para cálculo Poisson. Pulando.")
        df_calc['Prob_Empate_Poisson'] = np.nan
        return df_calc

    logger.info(f"Calculando Prob Empate (Poisson Refinado, max_goals={max_goals})...")

    fa_h = pd.to_numeric(df_calc['FA_H'], errors='coerce').fillna(1.0)
    fd_a = pd.to_numeric(df_calc['FD_A'], errors='coerce').fillna(1.0)
    fa_a = pd.to_numeric(df_calc['FA_A'], errors='coerce').fillna(1.0)
    fd_h = pd.to_numeric(df_calc['FD_H'], errors='coerce').fillna(1.0)

    lambda_h = fa_h * fd_a * avg_goals_home_league
    lambda_a = fa_a * fd_h * avg_goals_away_league

    lambda_h = np.maximum(lambda_h, 1e-6)
    lambda_a = np.maximum(lambda_a, 1e-6)

    prob_empate_total = pd.Series(0.0, index=df_calc.index)
    try:
        k_range = np.arange(max_goals + 1)
        prob_h_k = poisson.pmf(k_range[:, np.newaxis], lambda_h.values[np.newaxis, :])
        prob_a_k = poisson.pmf(k_range[:, np.newaxis], lambda_a.values[np.newaxis, :])
        prob_empate_total = np.sum(prob_h_k * prob_a_k, axis=0)

    except Exception as e:
        logger.error(f"Erro cálculo Poisson PMF: {e}", exc_info=True)
        df_calc['Prob_Empate_Poisson'] = np.nan
        return df_calc

    df_calc['Prob_Empate_Poisson'] = prob_empate_total
    logger.info("Prob_Empate_Poisson (Refinado) calculado.")
    return df_calc

def calculate_pi_ratings(df: pd.DataFrame) -> pd.DataFrame:

    logger.info(f"Calculando Pi-Ratings e Momentum (Janela={ROLLING_WINDOW})...")
    if not df.index.is_monotonic_increasing:
         logger.warning("DataFrame não ordenado por índice (Data?). Ordenando...")
         df = df.sort_index()

    ratings = {}
    rating_history = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW + 1))
    results_pi = []

    goals_h_col = GOALS_COLS.get('home', 'Goals_H_FT')
    goals_a_col = GOALS_COLS.get('away', 'Goals_A_FT')
    if goals_h_col not in df.columns or goals_a_col not in df.columns:
        logger.error(f"Colunas de Gols '{goals_h_col}' ou '{goals_a_col}' ausentes. Não é possível calcular PiRatings.")
        pi_cols = ['PiRating_H', 'PiRating_A', 'PiRating_Diff', 'PiRating_Prob_H', PIRATING_MOMENTUM_H, PIRATING_MOMENTUM_A, PIRATING_MOMENTUM_DIFF]
        for col in pi_cols: df[col] = np.nan
        return df

    df[goals_h_col] = pd.to_numeric(df[goals_h_col], errors='coerce')
    df[goals_a_col] = pd.to_numeric(df[goals_a_col], errors='coerce')

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Calculando Pi-Ratings/Momentum"):
        home_team = row.get('Home'); 
        away_team = row.get('Away')
        if pd.isna(home_team) or pd.isna(away_team): 
             results_pi.append({'Index': index, 'PiRating_H': np.nan, 'PiRating_A': np.nan, 'PiRating_Diff': np.nan, 'PiRating_Prob_H': np.nan, PIRATING_MOMENTUM_H: np.nan, PIRATING_MOMENTUM_A: np.nan, PIRATING_MOMENTUM_DIFF: np.nan})
             continue

        rating_h = ratings.get(home_team, PI_RATING_INITIAL); 
        rating_a = ratings.get(away_team, PI_RATING_INITIAL)

        hist_h = rating_history[home_team]; 
        rating_h_n_ago = None
        if len(hist_h) >= ROLLING_WINDOW: 
            rating_h_n_ago = hist_h[0][1]
        pi_rating_mom_h = (rating_h - rating_h_n_ago) if rating_h_n_ago is not None else np.nan

        hist_a = rating_history[away_team]; 
        rating_a_n_ago = None
        if len(hist_a) >= ROLLING_WINDOW: 
            rating_a_n_ago = hist_a[0][1]
        pi_rating_mom_a = (rating_a - rating_a_n_ago) if rating_a_n_ago is not None else np.nan
        pi_rating_mom_diff = pi_rating_mom_h - pi_rating_mom_a if pd.notna(pi_rating_mom_h) and pd.notna(pi_rating_mom_a) else np.nan

        rating_h_adj = rating_h + PI_RATING_HOME_ADVANTAGE; 
        rating_diff = rating_h_adj - rating_a
        expected_h = 1 / (1 + 10**(-rating_diff / 400))

        match_pi_data = {
            'Index': index,
            'PiRating_H': rating_h, 
            'PiRating_A': rating_a,
            'PiRating_Diff': rating_h - rating_a, 
            'PiRating_Prob_H': expected_h,
            PIRATING_MOMENTUM_H: pi_rating_mom_h, 
            PIRATING_MOMENTUM_A: pi_rating_mom_a,
            PIRATING_MOMENTUM_DIFF: pi_rating_mom_diff
        }
        results_pi.append(match_pi_data)

        score_h = np.nan; 
        gh=row[goals_h_col]; 
        ga=row[goals_a_col]
        if pd.notna(gh) and pd.notna(ga): 
            score_h = 1.0 if gh > ga else (0.5 if gh == ga else 0.0)
        new_rating_h, new_rating_a = rating_h, rating_a
        if pd.notna(score_h):
            new_rating_h = rating_h + PI_RATING_K_FACTOR * (score_h - expected_h)
            new_rating_a = rating_a + PI_RATING_K_FACTOR * ((1 - score_h) - (1 - expected_h))
        ratings[home_team] = new_rating_h; 
        ratings[away_team] = new_rating_a
        rating_history[home_team].append((index, new_rating_h)); 
        rating_history[away_team].append((index, new_rating_a))

    df_pi_ratings = pd.DataFrame(results_pi).set_index('Index')
    df_out = df.copy()
    df_out.update(df_pi_ratings)
    for col in df_pi_ratings.columns:
        if col not in df_out.columns:
            df_out[col] = df_pi_ratings[col]

    logger.info(f"Pi-Ratings/Momentum calculados. Colunas: {list(df_pi_ratings.columns)}")
    nan_counts_new = df_out[list(df_pi_ratings.columns)].isnull().sum()
    logger.debug(f"NaNs PiRating/Momentum:\n{nan_counts_new[nan_counts_new > 0]}")
    return df_out

def calculate_raw_value_cost_goals(df: pd.DataFrame) -> pd.DataFrame:

    df_calc = df.copy()
    logger.info("Calculando VG_raw e CG_raw (Lógica Corrigida)...")
    
    epsilon = FEATURE_EPSILON
    gh = GOALS_COLS.get('home')
    ga = GOALS_COLS.get('away')

    required_for_vcg = ['p_H', 'p_A', gh, ga]
    missing = [col for col in required_for_vcg if col not in df_calc.columns or df_calc[col].isnull().all()]
    if missing:
        logger.warning(f"Inputs para VG/CG Raw ausentes ou todos NaN: {missing}. Colunas VG/CG serão NaN.")
        df_calc[['VG_H_raw', 'VG_A_raw', 'CG_H_raw', 'CG_A_raw']] = np.nan
        return df_calc

    h_g = pd.to_numeric(df_calc[gh], errors='coerce')
    a_g = pd.to_numeric(df_calc[ga], errors='coerce')
    p_H = pd.to_numeric(df_calc['p_H'], errors='coerce')
    p_A = pd.to_numeric(df_calc['p_A'], errors='coerce')

    df_calc['VG_H_raw'] = h_g * p_H
    df_calc['VG_A_raw'] = a_g * p_A
    
    df_calc['CG_H_raw'] = np.where((h_g.notna() & (h_g > epsilon)) & p_H.notna(), p_H / h_g, np.nan)
    df_calc['CG_A_raw'] = np.where((a_g.notna() & (a_g > epsilon)) & p_A.notna(), p_A / a_g, np.nan)
    
    logger.info("-> VG_raw e CG_raw calculados com sucesso (lógica corrigida).")
    return df_calc

def calculate_historical_intermediate(df: pd.DataFrame) -> pd.DataFrame:
    df_calc = df.copy()
    logger.info("Calculando stats intermediárias (Result, IsDraw, Ptos)...")
    gh = GOALS_COLS.get('home'); ga = GOALS_COLS.get('away')
    logger.debug(f"calculate_historical_intermediate: Colunas em df_calc: {df_calc.columns.tolist()}")
    logger.debug(f"calculate_historical_intermediate: Verificando '{gh}' e '{ga}'...")
    if gh in df_calc.columns and ga in df_calc.columns:
        logger.info(f"-> Colunas de Gols encontradas: '{gh}' e '{ga}'.")
        logger.debug(f"  Conteúdo inicial '{gh}' (5): {df_calc[gh].head().tolist()}, dtype: {df_calc[gh].dtype}, NaNs: {df_calc[gh].isnull().sum()}/{len(df_calc)}")
        logger.debug(f"  Conteúdo inicial '{ga}' (5): {df_calc[ga].head().tolist()}, dtype: {df_calc[ga].dtype}, NaNs: {df_calc[ga].isnull().sum()}/{len(df_calc)}")
        h_g = pd.to_numeric(df_calc[gh], errors='coerce'); a_g = pd.to_numeric(df_calc[ga], errors='coerce')
        logger.debug(f"  h_g (pós-numérico) (5): {h_g.head().tolist()}, NaNs: {h_g.isnull().sum()}/{len(h_g)}")
        logger.debug(f"  a_g (pós-numérico) (5): {a_g.head().tolist()}, NaNs: {a_g.isnull().sum()}/{len(a_g)}")
        condlist = [h_g > a_g, h_g == a_g, h_g < a_g]; choicelist_res = ["H", "D", "A"]
        df_calc['FT_Result'] = np.select(condlist, choicelist_res, default=pd.NA)
        logger.debug(f"  Contagem 'FT_Result': \n{df_calc['FT_Result'].value_counts(dropna=False)}")
        df_calc['IsDraw'] = pd.NA; valid_ft_mask = df_calc['FT_Result'].notna()
        df_calc.loc[valid_ft_mask, 'IsDraw'] = (df_calc.loc[valid_ft_mask, 'FT_Result'] == 'D').astype('Int64')
        logger.debug(f"  Contagem 'IsDraw': \n{df_calc['IsDraw'].value_counts(dropna=False)}")
        if df_calc['IsDraw'].isnull().all(): logger.error("ALERTA CRÍTICO (calc_hist_intermed): 'IsDraw' é todo NaN/NA!")
        cl_pts_h = [df_calc['FT_Result']=='H', df_calc['FT_Result']=='D', df_calc['FT_Result']=='A']; ch_pts_h = [3,1,0]
        df_calc['Ptos_H'] = np.select(cl_pts_h, ch_pts_h, default=np.nan)
        cl_pts_a = [df_calc['FT_Result']=='A', df_calc['FT_Result']=='D', df_calc['FT_Result']=='H']; ch_pts_a = [3,1,0]
        df_calc['Ptos_A'] = np.select(cl_pts_a, ch_pts_a, default=np.nan)
        logger.info("->Result/IsDraw/Ptos OK.")
    else:
        logger.warning(f"->Gols '{gh}'/'{ga}' ausentes. Colunas de resultado serão NA.")
        df_calc[['FT_Result', 'IsDraw', 'Ptos_H', 'Ptos_A']] = pd.NA
    logger.info("Cálculo Intermediárias (Result/IsDraw/Ptos) concluído.")
    return df_calc

def calculate_rolling_goal_stats(
    df: pd.DataFrame,
    window: int = ROLLING_WINDOW,
    avg_goals_home_league: Optional[float] = None,
    avg_goals_away_league: Optional[float] = None
) -> pd.DataFrame:

    df_calc = df.copy()
    logger.info(f"Iniciando cálculo OTIMIZADO de Rolling Goals/FA/FD (Janela={window})...")
    
    goals_h_col = GOALS_COLS.get('home', 'Goals_H_FT')
    goals_a_col = GOALS_COLS.get('away', 'Goals_A_FT')
    epsilon = 1e-6

    if goals_h_col not in df_calc.columns or goals_a_col not in df_calc.columns:
        logger.warning("Calc Rolling Goals: Colunas de Gols ausentes.")
        cols_to_add = [
            'Avg_Gols_Marcados_H', 'Avg_Gols_Sofridos_H', 'FA_H', 'FD_H',
            'Avg_Gols_Marcados_A', 'Avg_Gols_Sofridos_A', 'FA_A', 'FD_A'
        ]
        for col in cols_to_add:
            df_calc[col] = np.nan
        return df_calc

    # 1. Calcula as médias da liga
    if avg_goals_home_league is None:
        avg_h_league = np.nanmean(df_calc[goals_h_col]) if pd.notna(np.nanmean(df_calc[goals_h_col])) else 1.0
        logger.warning(f"Média gols casa da liga não fornecida. Calculada como: {avg_h_league:.3f}")
    else:
        avg_h_league = avg_goals_home_league
    if avg_goals_away_league is None:
        avg_a_league = np.nanmean(df_calc[goals_a_col]) if pd.notna(np.nanmean(df_calc[goals_a_col])) else 1.0
        logger.warning(f"Média gols fora da liga não fornecida. Calculada como: {avg_a_league:.3f}")
    else:
        avg_a_league = avg_goals_away_league
    avg_h_league_safe = max(avg_h_league, epsilon)
    avg_a_league_safe = max(avg_a_league, epsilon)

    # 2. Reestrutura os dados para formato longo
    df_home = pd.DataFrame({'Team': df_calc['Home'], 'Gols_Marcados': df_calc[goals_h_col], 'Gols_Sofridos': df_calc[goals_a_col], 'match_idx': df_calc.index})
    df_away = pd.DataFrame({'Team': df_calc['Away'], 'Gols_Marcados': df_calc[goals_a_col], 'Gols_Sofridos': df_calc[goals_h_col], 'match_idx': df_calc.index})
    df_long = pd.concat([df_home, df_away]).sort_values(by='match_idx', kind='stable')
    df_long[['Gols_Marcados', 'Gols_Sofridos']] = df_long[['Gols_Marcados', 'Gols_Sofridos']].apply(pd.to_numeric, errors='coerce')

    # 3. Calcula médias móveis
    df_long['Avg_Gols_Marcados'] = df_long.groupby('Team')['Gols_Marcados'].transform(lambda x: x.rolling(window, min_periods=1).mean().shift(1))
    df_long['Avg_Gols_Sofridos'] = df_long.groupby('Team')['Gols_Sofridos'].transform(lambda x: x.rolling(window, min_periods=1).mean().shift(1))

    # 4. Mapeia os resultados de volta usando MERGE
    stats_to_merge = df_long[['match_idx', 'Team', 'Avg_Gols_Marcados', 'Avg_Gols_Sofridos']]
    df_merged = df_calc.reset_index().rename(columns={'index': 'match_idx'})
    
    home_stats = stats_to_merge.rename(columns={'Team': 'Home', 'Avg_Gols_Marcados': 'Avg_Gols_Marcados_H', 'Avg_Gols_Sofridos': 'Avg_Gols_Sofridos_H'})
    df_merged = df_merged.merge(home_stats, on=['match_idx', 'Home'], how='left')

    away_stats = stats_to_merge.rename(columns={'Team': 'Away', 'Avg_Gols_Marcados': 'Avg_Gols_Marcados_A', 'Avg_Gols_Sofridos': 'Avg_Gols_Sofridos_A'})
    df_merged = df_merged.merge(away_stats, on=['match_idx', 'Away'], how='left')

    df_merged = df_merged.set_index('match_idx')
    df_merged.index.name = df_calc.index.name
    
    cols_to_update = ['Avg_Gols_Marcados_H', 'Avg_Gols_Sofridos_H', 'Avg_Gols_Marcados_A', 'Avg_Gols_Sofridos_A']
    for col in cols_to_update:
        df_calc[col] = df_merged[col]

    # 5. Calcula Força de Ataque (FA) e Defesa (FD)
    df_calc['FA_H'] = df_calc['Avg_Gols_Marcados_H'] / avg_h_league_safe
    df_calc['FD_H'] = df_calc['Avg_Gols_Sofridos_H'] / avg_a_league_safe
    df_calc['FA_A'] = df_calc['Avg_Gols_Marcados_A'] / avg_a_league_safe
    df_calc['FD_A'] = df_calc['Avg_Gols_Sofridos_A'] / avg_h_league_safe
    
    logger.info("-> Rolling Gols/FA/FD calculados com sucesso (otimizado).")
    return df_calc

def calculate_ewma_stats(
    df: pd.DataFrame,
    stats_configs: List[Dict[str, Any]],
    default_span: int = ROLLING_WINDOW
) -> pd.DataFrame:

    df_calc = df.copy()
    logger.info(f"Iniciando cálculo HÍBRIDO de EWMA (Span Padrão={default_span})...")

    df_to_melt = df_calc.reset_index().rename(columns={'index': 'match_idx'})

    for config in stats_configs:
        prefix = config.get('output_prefix')
        base_h = config.get('base_col_h')
        base_a = config.get('base_col_a')
        span = config.get('span', default_span)
        min_p = config.get('min_periods', 1)
        stat_type = config.get('stat_type', 'offensive')
        
        if not all([prefix, base_h, base_a]):
            logger.warning(f"Config EWMA incompleta: {config}. Pulando.")
            continue
            
        logger.debug(f"  Processando EWMA para '{prefix}' com span={span}")
        
        output_col_h = f"{prefix}_s{span}_H"
        output_col_a = f"{prefix}_s{span}_A"

        home_stat_col = base_h if stat_type == 'offensive' else base_a
        away_stat_col = base_a if stat_type == 'offensive' else base_h
        
        # 1. Melt mais eficiente 
        cols_for_this_melt = ['match_idx', 'Home', 'Away', home_stat_col, away_stat_col]
        df_long = pd.melt(
            df_to_melt[cols_for_this_melt],
            id_vars=['match_idx', 'Home', 'Away'],
            value_vars=[home_stat_col, away_stat_col],
            var_name='Stat_Source_Column',
            value_name='Stat_Value'
        )

        # 2. Determinar time corretamente 
        df_long['Team'] = np.where(
            df_long['Stat_Source_Column'] == home_stat_col,
            df_long['Home'],
            df_long['Away']
        )
        
        # 3. Ordena e calcula a EWMA 
        df_long = df_long.sort_values(by='match_idx', kind='stable')
        df_long['Stat_Value'] = pd.to_numeric(df_long['Stat_Value'], errors='coerce')

        df_long['EWMA_Value'] = (
            df_long.groupby('Team')['Stat_Value']
                   .transform(lambda x: x.ewm(span=span, adjust=True, min_periods=min_p).mean().shift(1))
        )
        
        # 4. Mapeia os resultados de volta 
        mapper = df_long.set_index(['match_idx', 'Team'])['EWMA_Value']
        df_calc[output_col_h] = df_calc.index.map(lambda idx: mapper.get((idx, df_calc.loc[idx, 'Home'])))
        df_calc[output_col_a] = df_calc.index.map(lambda idx: mapper.get((idx, df_calc.loc[idx, 'Away'])))

        logger.info(f"  -> '{output_col_h}' e '{output_col_a}' calculados.")
        
    return df_calc

def _select_and_clean_final_features(
    df_with_all_features: pd.DataFrame,
    configured_feature_columns: List[str], 
    target_col_name: str
) -> Optional[Tuple[pd.DataFrame, pd.Series, List[str]]]:

    logger.info("--- Selecionando e Limpando Features Finais ---")

    if df_with_all_features is None or df_with_all_features.empty:
        logger.error("Seleção Final: DataFrame de entrada vazio ou None.")
        return None
    if not configured_feature_columns:
        logger.error("Seleção Final: Nenhuma feature configurada (FEATURE_COLUMNS vazia).")
        return None
    if target_col_name not in df_with_all_features.columns:
        logger.error(f"Seleção Final: Coluna alvo '{target_col_name}' não encontrada no DataFrame processado.")
        return None

    available_features_from_config = [f for f in configured_feature_columns if f in df_with_all_features.columns]
    missing_configured_features = [f for f in configured_feature_columns if f not in df_with_all_features.columns]

    if not available_features_from_config:
        logger.error("Seleção Final: Nenhuma das features configuradas está disponível no DataFrame processado!")
        return None
    if missing_configured_features:
        logger.warning(f"Seleção Final: Features configuradas ausentes do DataFrame: {missing_configured_features}. Elas serão ignoradas.")

    final_features_to_use = available_features_from_config
    logger.info(f"Features finais a serem usadas para o modelo: {final_features_to_use}")

    required_cols_for_output = final_features_to_use + [target_col_name]
    df_selected_subset = df_with_all_features[required_cols_for_output].copy()
    logger.info(f"DataFrame selecionado com features e alvo (antes do dropna): {df_selected_subset.shape}")

    nan_check_before_dropna = df_selected_subset[final_features_to_use].isnull().sum()
    cols_with_nans_before = nan_check_before_dropna[nan_check_before_dropna > 0]
    if not cols_with_nans_before.empty:
        logger.warning(f"NaNs ANTES do dropna final nas features selecionadas:\n{cols_with_nans_before}")
    else:
        logger.info("Nenhum NaN encontrado nas features selecionadas antes do dropna final.")

    initial_inf_count = df_selected_subset[final_features_to_use].isin([np.inf, -np.inf]).sum().sum()
    if initial_inf_count > 0:
        logger.warning(f"Encontrados {initial_inf_count} valores infinitos nas features selecionadas. Convertendo para NaN.")
        df_selected_subset.replace([np.inf, -np.inf], np.nan, inplace=True)

    initial_rows_for_dropna = len(df_selected_subset)
    df_selected_subset.dropna(subset=required_cols_for_output, inplace=True)
    rows_dropped_by_dropna = initial_rows_for_dropna - len(df_selected_subset)

    if rows_dropped_by_dropna > 0:
        logger.info(f"Removidas {rows_dropped_by_dropna} linhas contendo NaNs/Infs nas features finais ou no alvo.")

    if df_selected_subset.empty:
        logger.error("Erro CRÍTICO: Nenhuma linha restante após o dropna final de features e alvo.")
        return None

    X_clean = df_selected_subset[final_features_to_use].copy() 
    y_clean = df_selected_subset[target_col_name].astype(int) 

    if X_clean.isnull().values.any():
        final_nan_check_X = X_clean.isnull().sum()
        logger.error(f"ERRO CRÍTICO: NaNs detectados em X APÓS dropna! Colunas com NaNs:\n{final_nan_check_X[final_nan_check_X > 0]}")
        return None
    if y_clean.isnull().values.any():
        logger.error(f"ERRO CRÍTICO: NaNs detectados em y APÓS dropna!")
        return None

    logger.info(f"--- Seleção e Limpeza de Features Finais Concluída --- Shape X: {X_clean.shape}, Shape y: {y_clean.shape}.")
    logger.debug(f"Features FINAIS efetivamente usadas no modelo: {list(X_clean.columns)}")
    return X_clean, y_clean, list(X_clean.columns) 


def preprocess_and_feature_engineer(df_loaded: pd.DataFrame) -> Optional[Tuple[pd.DataFrame, pd.Series, List[str]]]:
    if df_loaded is None or df_loaded.empty: logger.error("Pré-proc: DF entrada inválido."); return None
    logger.info("--- Iniciando Pipeline Pré-proc e Feature Eng (Histórico) ---"); df_processed = df_loaded.copy()
    gh_col=GOALS_COLS.get('home'); ga_col=GOALS_COLS.get('away')
    avg_h_lg = np.nanmean(df_processed[gh_col]) if gh_col in df_processed and df_processed[gh_col].notna().any() else 1.0
    avg_a_lg = np.nanmean(df_processed[ga_col]) if ga_col in df_processed and df_processed[ga_col].notna().any() else 1.0
    avg_h_lg_safe = max(1.0 if pd.isna(avg_h_lg) else avg_h_lg, FEATURE_EPSILON)
    avg_a_lg_safe = max(1.0 if pd.isna(avg_a_lg) else avg_a_lg, FEATURE_EPSILON)
    logger.info(f"Médias Liga (FA/FD, Poisson): Casa={avg_h_lg_safe:.3f}, Fora={avg_a_lg_safe:.3f}")

    logger.info("=== ETAPA 1: Intermediárias e Probs ===")
    df_processed = calculate_historical_intermediate(df_processed) 
    if TARGET_COLUMN not in df_processed.columns or df_processed[TARGET_COLUMN].isnull().all():
        logger.error(f"Alvo '{TARGET_COLUMN}' ausente ou todos NaNs após stats intermediárias."); return None
    df_processed = calculate_probabilities(df_processed)
    df_processed = calculate_normalized_probabilities(df_processed)
    logger.info("=== ETAPA 1.5: VG_raw e CG_raw ===") 
    df_processed = calculate_raw_value_cost_goals(df_processed) 

    logger.info("=== ETAPA 2: PiRatings ===")
    df_processed = calculate_pi_ratings(df_processed)

    logger.info("=== ETAPA 3: Rolling Stats ===")
    valid_roll_cfg = [
        c for c in STATS_ROLLING_CONFIG 
        if c.get('base_col_h') in df_processed.columns and c.get('base_col_a') in df_processed.columns
    ]
    if valid_roll_cfg:
        df_processed = calculate_general_rolling_stats(df_processed, stats_configs=valid_roll_cfg)
    else:
        logger.warning("Nenhuma config de rolling stat válida. Pulando.")

    logger.info("=== ETAPA 4: EWMA Stats ===")
    valid_ewma_cfg = [c for c in STATS_EWMA_CONFIG if (c.get('base_col_h') and c['base_col_h'] in df_processed.columns and df_processed[c['base_col_h']].notna().any()) or (c.get('base_col_a') and c['base_col_a'] in df_processed.columns and df_processed[c['base_col_a']].notna().any())]
    if valid_ewma_cfg: df_processed = calculate_ewma_stats(df_processed, valid_ewma_cfg) 
    else: logger.warning("Nenhuma config EWMA válida/colunas base são NaN. EWMA stats não calculadas.")

    logger.info("=== ETAPA 5: RollingGoals; FA/FD ===")
    gh_col = GOALS_COLS.get('home')
    ga_col = GOALS_COLS.get('away')
    avg_h_lg = np.nanmean(df_processed[gh_col]) if gh_col in df_processed and df_processed[gh_col].notna().any() else 1.0
    avg_a_lg = np.nanmean(df_processed[ga_col]) if ga_col in df_processed and df_processed[ga_col].notna().any() else 1.0
    
    df_processed = calculate_rolling_goal_stats(df_processed, avg_goals_home_league=avg_h_lg, avg_goals_away_league=avg_a_lg)

    logger.info("=== ETAPA 6: Poisson, Binned, Derivadas ===")
    df_processed = calculate_poisson_draw_prob(df_processed, avg_h_lg_safe, avg_a_lg_safe)
    df_processed = calculate_binned_features(df_processed)
    df_processed = calculate_derived_features(df_processed)

    logger.info("=== FIM PIPELINE FEATURE ENG (HISTÓRICO) ===")
    return _select_and_clean_final_features(df_processed, FEATURE_COLUMNS, TARGET_COLUMN)


def fetch_and_process_fixtures() -> Optional[pd.DataFrame]:

    if FIXTURE_FETCH_DAY == "tomorrow":
        target_date = date.today() + timedelta(days=1)
    else:
        target_date = date.today()
    date_str = target_date.strftime('%Y-%m-%d')
    fixture_url = FIXTURE_CSV_URL_TEMPLATE.format(date_str=date_str)
    logger.info(f"Buscando jogos de {FIXTURE_FETCH_DAY} ({date_str}): {fixture_url}")
    
    df_fix = None
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(fixture_url, headers=headers, timeout=20)
        response.raise_for_status()
        logger.info("Arquivo futuro encontrado. Tentando ler...")
        content_bytes = response.content  

        try:  

            logger.debug("Tentando ler como CSV...")
            df_fix = pd.read_csv(BytesIO(content_bytes), low_memory=False)
            logger.info("Lido com sucesso como CSV.")

        except Exception as e_csv:
            logger.warning(f"Falha ao ler como CSV ({e_csv}), tentando ler como Excel...")
            try:  

                df_fix = pd.read_excel(BytesIO(content_bytes), engine='openpyxl')
                logger.info("Lido com sucesso como Excel.")
            except Exception as e_excel:
                logger.error(f"Falha ao ler como CSV e como Excel: {e_excel}", exc_info=True)
                return None

        if df_fix is None:
            raise ValueError("Falha ao ler dados do arquivo futuro.")
        logger.info(f"Arquivo futuro baixado e lido. Shape: {df_fix.shape}")
    except requests.exceptions.RequestException as e_req:
        logger.error(f"Erro HTTP ao buscar arquivo futuro: {e_req}")
        return None
    except pd.errors.EmptyDataError:
        logger.warning(f"Arquivo futuro {fixture_url} vazio.")
        return pd.DataFrame()
    except Exception as e_load:
        logger.error(f"Erro baixar/ler arquivo futuro: {e_load}", exc_info=True)
        return None

    try:

        logger.info("Processando arquivo de jogos futuros...")
        logger.debug(f"Colunas lidas: {list(df_fix.columns)}")
        cols_to_map_and_rename = {k: v for k, v in CSV_HIST_COL_MAP.items() if k in df_fix.columns}
        if not cols_to_map_and_rename:
            logger.error("Nenhuma coluna mapeável encontrada no arquivo futuro.")
            return None

        df_processed = df_fix[list(cols_to_map_and_rename.keys())].copy()
        df_processed.rename(columns=cols_to_map_and_rename, inplace=True)
        logger.debug(f"Colunas após rename: {list(df_processed.columns)}")

        internal_league_col_name = 'League'
        if internal_league_col_name in df_processed.columns:
            if SCRAPER_FILTER_LEAGUES:
                logger.info(f"Filtrando por ligas alvo: {TARGET_LEAGUES_INTERNAL_IDS}...")
                df_processed[internal_league_col_name] = (
                    df_processed[internal_league_col_name]
                    .astype(str)
                    .str.strip()
                )
                initial_count = len(df_processed)
                df_processed = df_processed[
                    df_processed[internal_league_col_name].isin(TARGET_LEAGUES_INTERNAL_IDS)
                ]
                logger.info(f"Filtro de ligas (Futuro): {len(df_processed)}/{initial_count} jogos restantes.")
                if df_processed.empty:
                    logger.info("Nenhum jogo futuro restante após filtro.")
                    return pd.DataFrame()
            else:
                logger.info("Filtro de liga DESATIVADO para arquivo de jogos futuros.")
        else:
            logger.warning(f"Coluna '{internal_league_col_name}' não encontrada. Filtro de liga não aplicado.")

        try:
            current_required_fixture_cols = REQUIRED_FIXTURE_COLS
        except NameError:
            logger.warning("REQUIRED_FIXTURE_COLS não no config.")
            current_required_fixture_cols = [
                'League', 'Home', 'Away', 'Time_Str', 'Odd_H_FT', 'Odd_D_FT', 'Odd_A_FT'
            ]
        missing_required = [c for c in current_required_fixture_cols if c not in df_processed.columns]
        if missing_required:
            logger.error(f"Colunas essenciais ausentes (Futuro): {missing_required}")
            return None

        if 'Date' in df_processed.columns:
            df_processed['Date_Str'] = pd.to_datetime(df_processed['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
        else:
            logger.warning("Coluna 'Date' ausente para criar 'Date_Str'.")
            df_processed['Date_Str'] = date_str  

        if 'Time' in df_processed.columns:
            df_processed['Time_Str'] = df_processed['Time'].astype(str).str.strip()
        else:
            logger.warning("Coluna 'Time' ausente para criar 'Time_Str'.")
            df_processed['Time_Str'] = "00:00"  

        rows_to_drop = (
            df_processed['Date_Str'].isnull() |
            df_processed['Time_Str'].isnull() |
            (df_processed['Time_Str'] == '')
        )
        if rows_to_drop.any():
            logger.warning(f"Removendo {rows_to_drop.sum()} linhas com Date/Time inválido.")
            df_processed = df_processed[~rows_to_drop]
        if df_processed.empty:
            logger.warning("DF futuro vazio após limpeza Date/Time.")
            return pd.DataFrame()
        df_processed.reset_index(drop=True, inplace=True)
        logger.info(f"Processamento jogos futuros OK. Shape final: {df_processed.shape}")
        logger.debug(f"Colunas finais jogos futuros: {list(df_processed.columns)}")

        for col in list(ODDS_COLS.values()) + OTHER_ODDS_NAMES + list(XG_COLS.values()):
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                if col in list(ODDS_COLS.values()) or col in OTHER_ODDS_NAMES:
                    df_processed.loc[df_processed[col] <= 1, col] = np.nan
                if col in list(XG_COLS.values()):
                    df_processed.loc[df_processed[col] < 0, col] = np.nan

        logger.info(f"Verificando NaNs essenciais (Futuro): {current_required_fixture_cols}")
        initial_rows_fix = len(df_processed)
        cols_to_dropna_present_fix = [c for c in current_required_fixture_cols if c in df_processed.columns]
        df_processed.dropna(subset=cols_to_dropna_present_fix, inplace=True)
        rows_dropped_fix = initial_rows_fix - len(df_processed)
        if rows_dropped_fix > 0:
            logger.info(f"Futuro: Removidas {rows_dropped_fix} linhas com NaNs essenciais.")
        if df_processed.empty:
            logger.info("Nenhum jogo futuro restante após limpeza essencial.")
            return pd.DataFrame()
        df_processed.reset_index(drop=True, inplace=True)
        logger.info(f"Processamento jogos futuros OK. Shape final: {df_processed.shape}")
        logger.debug(f"Colunas finais jogos futuros: {list(df_processed.columns)}")
        return df_processed

    except Exception as e_proc:
        logger.error(f"Erro GERAL no processamento futuro: {e_proc}", exc_info=True)
        return None

def _build_team_history_final_state(
    historical_df_minimal_sorted: pd.DataFrame, 
    all_needed_stat_configs: List[Dict[str, Any]], 
    progress_desc: str = "Construindo Estado Histórico"
) -> Dict[str, Dict[str, deque]]:

    logger.info(f"Construindo estado final histórico...")
    history_maxlen = len(historical_df_minimal_sorted) + 5 
    team_history_final_state = defaultdict(lambda: defaultdict(lambda: deque(maxlen=history_maxlen)))

    unique_base_cols_from_configs = set()
    if all_needed_stat_configs:
        for cfg in all_needed_stat_configs:
            if cfg.get('base_col_h'): unique_base_cols_from_configs.add(cfg['base_col_h'])
            if cfg.get('base_col_a'): unique_base_cols_from_configs.add(cfg['base_col_a'])
    else:
        logger.warning("Nenhuma configuração de stat passada para _build_team_history_final_state. Histórico estará vazio.")
        return team_history_final_state

    logger.debug(f"Colunas base únicas identificadas das configs para popular histórico: {list(unique_base_cols_from_configs)}")

    if not {'Home', 'Away'}.issubset(historical_df_minimal_sorted.columns):
        logger.error("Estado Histórico: Colunas 'Home' ou 'Away' ausentes no DataFrame histórico.")
        return team_history_final_state 

    for _, row in tqdm(historical_df_minimal_sorted.iterrows(), total=len(historical_df_minimal_sorted), desc=progress_desc):
        home_team = row.get('Home')
        away_team = row.get('Away')

        if pd.isna(home_team) or pd.isna(away_team):
            continue

        for base_col_key in unique_base_cols_from_configs:
            if base_col_key not in row: 
                continue

            value_in_row = row[base_col_key]
            if pd.isna(value_in_row): 
                continue

            is_home_stat_indicator = any(s in base_col_key for s in ['_H', '_H_FT', 'Home']) or \
                                     base_col_key in ['Ptos_H', 'VG_H_raw', 'CG_H_raw'] 

            is_away_stat_indicator = any(s in base_col_key for s in ['_A', '_A_FT', 'Away']) or \
                                     base_col_key in ['Ptos_A', 'VG_A_raw', 'CG_A_raw'] 

            # Caso 1: A coluna base parece ser uma estatística do time da casa
            if is_home_stat_indicator and not is_away_stat_indicator: 
                team_history_final_state[home_team][base_col_key].append(value_in_row)

            # Caso 2: A coluna base parece ser uma estatística do time visitante
            elif is_away_stat_indicator and not is_home_stat_indicator:
                team_history_final_state[away_team][base_col_key].append(value_in_row)

            # Caso 3: A coluna base não tem indicador claro de H/A (ex: 'Odd_D_FT', 'XG_Total')
            else:
                team_history_final_state[home_team][base_col_key].append(value_in_row)
                team_history_final_state[away_team][base_col_key].append(value_in_row)

    logger.info(f"Estado histórico final construído para {len(team_history_final_state)} times.")
   
    return team_history_final_state

def _get_final_pi_ratings(df_hist: pd.DataFrame) -> Dict[str, float]:

    ratings = {}
    goals_h_c = GOALS_COLS.get('home')
    goals_a_c = GOALS_COLS.get('away')

    if goals_h_c not in df_hist.columns or goals_a_c not in df_hist.columns:
        logger.error(f"Colunas de Gols ('{goals_h_c}', '{goals_a_c}') ausentes para cálculo de Pi-Rating final.")
        return {}

    df_hist = df_hist.copy()  
    df_hist[goals_h_c] = pd.to_numeric(df_hist[goals_h_c], errors='coerce')
    df_hist[goals_a_c] = pd.to_numeric(df_hist[goals_a_c], errors='coerce')
    df_hist.dropna(subset=[goals_h_c, goals_a_c, 'Home', 'Away'], inplace=True)

    logger.debug(f"Calculando ratings finais em {len(df_hist)} jogos válidos.")

    df_hist = df_hist.sort_values(by='Date')

    for _, row in df_hist.iterrows():
        ht = row['Home']
        at = row['Away']

        rh = ratings.get(ht, PI_RATING_INITIAL)
        ra = ratings.get(at, PI_RATING_INITIAL)
        rh_adj = rh + PI_RATING_HOME_ADVANTAGE
        rd = rh_adj - ra
        exh = 1 / (1 + 10 ** (-rd / 400))

        gh = row[goals_h_c]
        ga = row[goals_a_c]

        score_h = 1.0 if gh > ga else (0.5 if gh == ga else 0.0)

        nrh = rh + PI_RATING_K_FACTOR * (score_h - exh)
        nra = ra + PI_RATING_K_FACTOR * ((1 - score_h) - (1 - exh))
        ratings[ht] = nrh
        ratings[at] = nra

    logger.debug(f"Cálculo de ratings finais concluído. {len(ratings)} times com rating.")
    return ratings

def prepare_fixture_data(
    fixture_df: pd.DataFrame,
    historical_df_full: pd.DataFrame,
    feature_columns_for_model: List[str],
    training_medians: pd.Series 
) -> Optional[pd.DataFrame]:

    if fixture_df is None or historical_df_full is None or training_medians is None or not feature_columns_for_model:
        logger.error("Prep Fix: Argumentos inválidos (fixture_df, historical_df_full, training_medians ou feature_columns)."); return None
    if fixture_df.empty:
        logger.info("Prep Fix: DataFrame de jogos futuros vazio. Nada a preparar."); return pd.DataFrame()
    if historical_df_full.empty:
        logger.error("Prep Fix: DataFrame histórico está vazio. Não é possível preparar features futuras."); return None

    logger.info("--- Iniciando Preparação de Features para Jogos Futuros (Versão Refatorada) ---")
    df_output_fixtures = fixture_df.copy()

    # --- ETAPA 1: Processar o histórico UMA VEZ para obter o estado final ---
    logger.info("Processando histórico completo para extrair o estado final...")
    X_hist, _, _ = preprocess_and_feature_engineer(historical_df_full)
    if X_hist is None:
        logger.error("Falha ao processar o DataFrame histórico para extrair o estado.")
        return None
        
    historical_df_processed = X_hist.join(historical_df_full[['Home', 'Away']])
    historical_df_processed.dropna(subset=['Home', 'Away'], inplace=True)
    
    final_team_stats = _get_final_team_stats(historical_df_processed)
    
    # --- ETAPA 2: Mapear o estado final para os jogos futuros ---
    logger.info("Mapeando o estado final das estatísticas para os jogos futuros...")
    
    home_stats_mapped = df_output_fixtures['Home'].map(final_team_stats.to_dict('index')).apply(pd.Series)
    home_stats_mapped.columns = [f"{col}_H" for col in home_stats_mapped.columns]

    away_stats_mapped = df_output_fixtures['Away'].map(final_team_stats.to_dict('index')).apply(pd.Series)
    away_stats_mapped.columns = [f"{col}_A" for col in away_stats_mapped.columns]

    df_output_fixtures = pd.concat([df_output_fixtures, home_stats_mapped, away_stats_mapped], axis=1)

    # --- ETAPA 3: Calcular features "instantâneas" que dependem apenas dos dados do jogo ---
    logger.info("Calculando features instantâneas (Probs, Derivadas)...")
    df_output_fixtures = calculate_probabilities(df_output_fixtures)
    df_output_fixtures = calculate_normalized_probabilities(df_output_fixtures)
    df_output_fixtures = calculate_binned_features(df_output_fixtures)
    
    goals_h_col = GOALS_COLS.get('home'); goals_a_col = GOALS_COLS.get('away')
    avg_h_league = np.nanmean(historical_df_full[goals_h_col]) if goals_h_col in historical_df_full else 1.0
    avg_a_league = np.nanmean(historical_df_full[goals_a_col]) if goals_a_col in historical_df_full else 1.0
    
    df_output_fixtures['FA_H'] = df_output_fixtures['Media_GolsMarcados_H'] / avg_h_league
    df_output_fixtures['FD_H'] = df_output_fixtures['Media_GolsSofridos_H'] / avg_a_league
    df_output_fixtures['FA_A'] = df_output_fixtures['Media_GolsMarcados_A'] / avg_a_league
    df_output_fixtures['FD_A'] = df_output_fixtures['Media_GolsSofridos_A'] / avg_h_league

    df_output_fixtures = calculate_poisson_draw_prob(df_output_fixtures, avg_h_league, avg_a_league)
    df_output_fixtures = calculate_derived_features(df_output_fixtures)

    # --- ETAPA 4: Seleção final das features e imputação inteligente ---
    logger.info(f"Selecionando features finais e aplicando imputação...")
    
    for col in feature_columns_for_model:
        if col not in df_output_fixtures.columns:
            df_output_fixtures[col] = np.nan
            
    X_fix_prep_final = df_output_fixtures[feature_columns_for_model].copy()

    missing_before = X_fix_prep_final.isnull().sum().sum()
    if missing_before > 0:
        logger.warning(f"{missing_before} valores ausentes encontrados (ex: times novos). Imputando com a mediana do treino.")
        X_fix_prep_final.fillna(training_medians, inplace=True)
    
    if X_fix_prep_final.isnull().values.any():
        logger.warning("Valores ausentes restantes após imputação por mediana. Preenchendo com 0.")
        X_fix_prep_final.fillna(0, inplace=True)

    logger.info(f"--- Preparação de Features para Jogos Futuros Concluída. Shape Final: {X_fix_prep_final.shape} ---")
    return X_fix_prep_final