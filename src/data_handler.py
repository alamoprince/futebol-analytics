# --- src/data_handler.py ---
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import warnings

from config import (
    FEATURE_COLUMNS,
    ODDS_COLS, GOALS_COLS, ROLLING_WINDOW, SCRAPER_FILTER_LEAGUES,
    TARGET_LEAGUES_INTERNAL_IDS, TARGET_LEAGUES_1,TARGET_LEAGUES_2, HISTORICAL_DATA_PATH_3,
    CSV_HIST_COL_MAP,HISTORICAL_DATA_PATH_1, HISTORICAL_DATA_PATH_2,
    OTHER_ODDS_NAMES, XG_COLS,
    PI_RATING_INITIAL, PI_RATING_K_FACTOR, PI_RATING_HOME_ADVANTAGE,
    FEATURE_EPSILON, APPLY_LEAGUE_FILTER_ON_HISTORICAL,
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

# --- Funções Auxiliares ---
def roi(y_test: pd.Series, y_pred: np.ndarray, X_test_odds_aligned: Optional[pd.DataFrame], odd_draw_col_name: str) -> Optional[float]:
    if X_test_odds_aligned is None or odd_draw_col_name not in X_test_odds_aligned.columns:
        return None
    
    try:
        common_index = y_test.index.intersection(X_test_odds_aligned.index)
    except AttributeError:
        return None
        
    if len(common_index) != len(y_test):
        logger.warning("ROI: Index mismatch")
        return None
        
    y_test_common = y_test.loc[common_index]
    
    try:
        y_pred_series = pd.Series(y_pred, index=y_test.index)
        y_pred_common = y_pred_series.loc[common_index]
    except:
        return None
        
    predicted_draws_indices = common_index[y_pred_common == 1]
    num_bets = len(predicted_draws_indices)
    
    if num_bets == 0:
        return 0.0
        
    actuals = y_test_common.loc[predicted_draws_indices]
    odds = pd.to_numeric(X_test_odds_aligned.loc[predicted_draws_indices, odd_draw_col_name], errors='coerce')
    
    profit = 0.0
    valid_bets = 0
    
    for idx in predicted_draws_indices:
        odd_d = odds.loc[idx]
        if pd.notna(odd_d) and odd_d > 1:
            profit += (odd_d - 1) if actuals.loc[idx] == 1 else -1
            valid_bets += 1
            
    if valid_bets == 0:
        return 0.0
        
    return (profit / valid_bets) * 100.0

def calculate_roi_with_threshold(y_true: pd.Series, y_proba: np.ndarray, threshold: float, odds_data: Optional[pd.DataFrame], odd_col_name: str) -> Tuple[Optional[float], int, Optional[float]]:
    profit, roi_value, num_bets_suggested, profit_calc, valid_bets_count = None, None, 0, 0.0, 0
    
    if odds_data is None or odd_col_name not in odds_data.columns:
        return roi_value, num_bets_suggested, profit
        
    try:
        common_index = y_true.index.intersection(odds_data.index)
        if len(common_index) == 0:
            return 0.0, 0, 0.0
            
        if len(common_index) != len(y_true):
            logger.warning("ROI Thr: Index mismatch.")
            
        y_true_common = y_true.loc[common_index]
        odds_common = pd.to_numeric(odds_data.loc[common_index, odd_col_name], errors='coerce')
        
        try:
            y_proba_series = pd.Series(y_proba, index=y_true.index)
            y_proba_common = y_proba_series.loc[common_index]
        except Exception as e:
            logger.error(f"ROI Thr: Erro alinhar y_proba: {e}")
            return None, 0, None
            
        bet_indices = common_index[y_proba_common > threshold]
        num_bets_suggested = len(bet_indices)
        
        if num_bets_suggested == 0:
            return 0.0, num_bets_suggested, 0.0
            
        actuals = y_true_common.loc[bet_indices]
        odds_selected = odds_common.loc[bet_indices]
        
        for idx in bet_indices:
            odd_d = odds_selected.loc[idx]
            if pd.notna(odd_d) and odd_d > 1:
                profit_calc += (odd_d - 1) if actuals.loc[idx] == 1 else -1
                valid_bets_count += 1
                
        profit = profit_calc
        roi_value = (profit / valid_bets_count) * 100 if valid_bets_count > 0 else 0.0
        return roi_value, valid_bets_count, profit  # Retorna bets válidas
        
    except Exception as e:
        logger.error(f"ROI Thr: Erro - {e}", exc_info=True)
        return None, 0, None

def calculate_metrics_with_ev(y_true: pd.Series, y_proba_calibrated: np.ndarray, ev_threshold: float, odds_data: Optional[pd.DataFrame], odd_col_name: str) -> Tuple[Optional[float], int, Optional[float]]:
    profit, roi_value, num_bets_suggested, profit_calc, valid_bets_count = None, None, 0, 0.0, 0
    
    if odds_data is None or odd_col_name not in odds_data.columns:
        logger.warning(f"EV Metr: Odds ausentes.")
        return roi_value, num_bets_suggested, profit
        
    try:
        common_index = y_true.index.intersection(odds_data.index)
        if len(common_index) == 0:
            return 0.0, 0, 0.0
            
        if len(common_index) != len(y_true):
            logger.warning(f"EV Metr: Index mismatch.")
            
        y_true_common = y_true.loc[common_index]
        odds_common = pd.to_numeric(odds_data.loc[common_index, odd_col_name], errors='coerce')
        
        try:
            y_proba_common = pd.Series(y_proba_calibrated, index=y_true.index).loc[common_index]
        except Exception as e:
            logger.error(f"EV Metr: Erro alinhar y_proba: {e}")
            return None, 0, None
            
        valid_mask = odds_common.notna() & y_proba_common.notna() & (odds_common > 1)
        ev = pd.Series(np.nan, index=common_index)
        prob_ok = y_proba_common[valid_mask]
        odds_ok = odds_common[valid_mask]
        ev_calc = (prob_ok * (odds_ok - 1)) - ((1 - prob_ok) * 1)
        ev.loc[valid_mask] = ev_calc
        
        bet_indices = common_index[ev > ev_threshold]
        num_bets_suggested = len(bet_indices)
        
        if num_bets_suggested == 0:
            return 0.0, num_bets_suggested, 0.0
            
        actuals = y_true_common.loc[bet_indices]
        odds_selected = odds_common.loc[bet_indices]
        
        for idx in bet_indices:
            odd_d = odds_selected.loc[idx]
            if pd.notna(odd_d) and odd_d > 1:
                profit_calc += (odd_d - 1) if actuals.loc[idx] == 1 else -1
                valid_bets_count += 1
                
        profit = profit_calc
        roi_value = (profit / valid_bets_count) * 100 if valid_bets_count > 0 else 0.0
        
        logger.debug(f"    -> Métricas EV (Th={ev_threshold:.3f}): ROI={roi_value:.2f}%, Bets Sug={num_bets_suggested}, Bets Vál={valid_bets_count}, Profit={profit:.2f}")
        return roi_value, num_bets_suggested, profit
        
    except Exception as e:
        logger.error(f"EV Metr: Erro - {e}", exc_info=True)
        return None, 0, None
    
# Função load_historical_data
def load_historical_data() -> Optional[pd.DataFrame]:
    """
    Carrega dados históricos, aplica mapeamentos de liga, opcionalmente filtra,
    concatena, e processa.
    """
    all_dfs_processed: List[pd.DataFrame] = []
    file_configs = [
        #{'path': HISTORICAL_DATA_PATH_1, 'league_map': TARGET_LEAGUES_1},
        #{'path': HISTORICAL_DATA_PATH_2, 'league_map': TARGET_LEAGUES_2}
        {'path': HISTORICAL_DATA_PATH_3, 'league_map': TARGET_LEAGUES_1}
    ]
    log_msg_filter = f"(FILTRO ATIVO por TARGET_LEAGUES_INTERNAL_IDS)" if APPLY_LEAGUE_FILTER_ON_HISTORICAL else "(SEM FILTRO DE LIGA)"
    logger.info(f"Iniciando carregamento/mapeamento histórico {log_msg_filter}...")

    # Nomes EXATOS das colunas de stats como nos CSVs
    expected_stat_csv_cols = ['Shots_H', 'Shots_A', 'ShotsOnTarget_H', 'ShotsOnTarget_A', 'Corners_H_FT', 'Corners_A_FT']

    for config in file_configs:
        file_path = config['path']
        current_league_map = config['league_map']
        base_filename = os.path.basename(file_path)
        
        if not os.path.exists(file_path):
            logger.error(f"Histórico não encontrado: {file_path}")
            continue
            
        try:
            logger.info(f"Lendo {base_filename}...")
            df_part = None
            file_ext = os.path.splitext(file_path)[1].lower() # Pega a extensão

            # --- CORREÇÃO: Usa read_excel para .xlsx ---
            if file_ext == '.xlsx':
                try:
                    df_part = pd.read_excel(file_path, engine='openpyxl') # Precisa de openpyxl: pip install openpyxl
                except Exception as e_excel:
                    logger.error(f"Falha ao ler arquivo Excel {base_filename}: {e_excel}", exc_info=True)
                    continue # Pula para o próximo arquivo
            elif file_ext == '.csv': # Mantém lógica para CSV
                try:
                    df_part = pd.read_csv(file_path, low_memory=False, encoding='ISO-8859-1')
                except UnicodeDecodeError:
                    logger.warning(f"Fallback p/ UTF-8 em {base_filename}...")
                    try:
                        df_part = pd.read_csv(file_path, low_memory=False, encoding='utf-8')
                    except Exception as e_csv_utf8:
                         logger.error(f"Falha ao ler CSV {base_filename} com UTF-8: {e_csv_utf8}", exc_info=True)
                         continue
                except Exception as e_csv:
                     logger.error(f"Falha ao ler CSV {base_filename}: {e_csv}", exc_info=True)
                     continue
            else:
                logger.error(f"Formato de arquivo não suportado para {base_filename}: {file_ext}")
                continue

            if df_part is None or df_part.empty: # Verifica se leitura falhou ou retornou vazio
                logger.warning(f"DataFrame vazio ou falha na leitura de {base_filename}.")
                continue

            logger.info(f"  -> Lido {df_part.shape[0]} linhas de {base_filename}.")
            logger.debug(f"    Colunas originais: {list(df_part.columns)}")

            # Garante colunas de stats
            for col in expected_stat_csv_cols:
                if col not in df_part.columns:
                    df_part[col] = np.nan

            # --- ETAPA A: Mapeamento e Identificação da Coluna League ---
            internal_league_col_name = 'League' # Nome interno padrão
            original_league_col_name = None
            # Encontra o nome original da coluna que deve ser mapeada para 'League'
            for csv_col, internal_col in CSV_HIST_COL_MAP.items():
                if internal_col == internal_league_col_name and csv_col in df_part.columns:
                    original_league_col_name = csv_col
                    break

            if not original_league_col_name:
                logger.warning(f"Coluna original para '{internal_league_col_name}' não encontrada/mapeada em {base_filename} via CSV_HIST_COL_MAP. Pulando mapeamento de liga.")
                # Prepara para mapeamento geral sem a coluna League
                cols_to_map_general = {k: v for k, v in CSV_HIST_COL_MAP.items() if k in df_part.columns and k != original_league_col_name}
                cols_to_keep = list(cols_to_map_general.keys()) + expected_stat_csv_cols
                cols_to_keep = [c for c in cols_to_keep if c in df_part.columns]
                df_processed = df_part[cols_to_keep].copy()
                df_processed.rename(columns=cols_to_map_general, inplace=True)
                # Adiciona coluna League vazia para consistência?
                df_processed[internal_league_col_name] = 'UNMAPPED' # Ou pd.NA
            else:
                logger.info(f"  Mapeando coluna '{original_league_col_name}' para ID Interno usando mapa específico do arquivo...")
                df_part[original_league_col_name] = df_part[original_league_col_name].astype(str).str.strip()
                # Aplica o mapeamento (TARGET_LEAGUES_1 ou _2) para criar a coluna de ID interno
                # Usa '_InternalID' temporariamente para não conflitar com mapeamento geral
                df_part['_InternalID'] = df_part[original_league_col_name].map(current_league_map)

                # Log de não mapeadas
                unmapped_mask = df_part['_InternalID'].isnull()
                original_unmapped_names = df_part.loc[unmapped_mask, original_league_col_name].unique()
                if len(original_unmapped_names) > 0:
                    logger.warning(f"    Ligas NÃO mapeadas p/ ID Interno: {list(original_unmapped_names)}")
                # Preenche não mapeadas com um placeholder
                df_part['_InternalID'].fillna('OTHER_LEAGUE', inplace=True)

                # --- ETAPA B: Filtragem CONDICIONAL ---
                df_to_process = df_part # Por padrão, processa tudo
                if APPLY_LEAGUE_FILTER_ON_HISTORICAL:
                    logger.info(f"  APLICANDO FILTRO por TARGET_LEAGUES_INTERNAL_IDS...")
                    initial_count = len(df_part)
                    # Filtra usando a coluna temporária '_InternalID' ANTES do rename final
                    df_to_process = df_part[df_part['_InternalID'].isin(TARGET_LEAGUES_INTERNAL_IDS)].copy()
                    logger.info(f"    -> Filtro: {len(df_to_process)}/{initial_count} jogos restantes.")
                    if df_to_process.empty:
                        logger.warning(f"  Nenhum jogo pós-filtro {base_filename}.")
                        continue
                else:
                    logger.info(f"  Filtro de Liga DESATIVADO.")

                # --- ETAPA C: Mapeamento Geral e Seleção Final ---
                logger.info(f"  Aplicando mapeamento geral de colunas (CSV_HIST_COL_MAP)...")
                # Mapeia todas as colunas DEFINIDAS em CSV_HIST_COL_MAP que existem no df_to_process
                # EXCETO a coluna original da liga (que já foi tratada)
                cols_to_map_general = {k: v for k, v in CSV_HIST_COL_MAP.items() if k in df_to_process.columns and k != original_league_col_name}

                # Colunas a manter: as que serão renomeadas + a coluna de ID interno + stats extras
                cols_to_keep_final = list(cols_to_map_general.keys()) + ['_InternalID']
                # Adiciona colunas de stats que existem no df original
                cols_to_keep_final.extend([c for c in expected_stat_csv_cols if c in df_to_process.columns])
                cols_to_keep_final = list(set(cols_to_keep_final)) # Remove duplicatas
                cols_to_keep_final = [c for c in cols_to_keep_final if c in df_to_process.columns] # Garante que existem

                df_processed = df_to_process[cols_to_keep_final].copy()

                # Renomeia colunas gerais (as chaves de cols_to_map_general)
                df_processed.rename(columns=cols_to_map_general, inplace=True)

                # Renomeia a coluna de ID interno para 'League' (o nome interno padrão)
                df_processed.rename(columns={'_InternalID': internal_league_col_name}, inplace=True)
                logger.debug(f"    Colunas após rename final: {list(df_processed.columns)}")

            # Adiciona à lista
            all_dfs_processed.append(df_processed)
            logger.info(f"  DataFrame {base_filename} processado (Shape: {df_processed.shape}).")

        except Exception as e:
            logger.error(f"Erro CRÍTICO processar {base_filename}: {e}", exc_info=True)
            continue

    # --- Fim do loop ---
    if not all_dfs_processed:
        logger.error("Nenhum CSV histórico processado.")
        return None

    # Concatenação
    logger.info(f"Concatenando {len(all_dfs_processed)} DFs...")
    try:
        df = pd.concat(all_dfs_processed, ignore_index=True, sort=False)
        logger.info(f"DF histórico combinado: {df.shape}")
    except Exception as e:
        logger.error(f"Erro concatenar DFs: {e}")
        return None
    if df.empty:
        logger.error("DF histórico vazio.")
        return None
    logger.debug(f"Colunas DF concatenado: {list(df.columns)}")

    # --- Processamento Comum Pós-Concatenação ---
    # ... (conversão Date, numéricos, dropna essencial, sort, PiRating) ...
    logger.info("Iniciando processamento comum pós-concatenação...")
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
    else:
        logger.error("Coluna 'Date' ausente.")
        return None
    if df.empty:
        logger.error("DF vazio após dropna Date.")
        return None
    numeric_cols_to_convert = (list(GOALS_COLS.values()) + list(ODDS_COLS.values()) + OTHER_ODDS_NAMES + list(XG_COLS.values()) + ['Shots_H', 'Shots_A', 'ShotsOnTarget_H', 'ShotsOnTarget_A', 'Corners_H_FT', 'Corners_A_FT'])
    for col in numeric_cols_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            #... (resto tratamento odds/xg)
            if col in ODDS_COLS.values() or col in OTHER_ODDS_NAMES:
                df.loc[df[col] <= 1, col] = np.nan
            if col in XG_COLS.values():
                df.loc[df[col] == 0, col] = np.nan
        else:
            logger.warning(f"Coluna numérica esperada '{col}' ausente.")
    essential_dropna_cols = ['Date', 'Home', 'Away'] + list(GOALS_COLS.values()) + list(ODDS_COLS.values())
    cols_to_dropna_present = [c for c in essential_dropna_cols if c in df.columns]
    logger.info(f"Verificando NaNs essenciais: {cols_to_dropna_present}")
    initial_rows = len(df)
    df.dropna(subset=cols_to_dropna_present, inplace=True)
    rows_dropped = initial_rows - len(df)
    if rows_dropped > 0:
        logger.info(f"Removidas {rows_dropped} linhas NaNs essenciais.")
    if df.empty:
        logger.error("Nenhum jogo restante após dropna essencial.")
        return None
    df = df.sort_values(by='Date').reset_index(drop=True)
    logger.info("DF histórico ordenado.")
    df = calculate_pi_ratings(df)

    # Log Final
    log_msg_final = "(SEM FILTRO LIGA)" if not APPLY_LEAGUE_FILTER_ON_HISTORICAL else "(FILTRADO)"
    logger.info(f"Carregamento histórico OK {log_msg_final}. Shape Final: {df.shape}")
    optional_nan_counts = df.isnull().sum()
    optional_nan_counts = optional_nan_counts[optional_nan_counts > 0]
    if not optional_nan_counts.empty:
        logger.info(f"Contagem NaNs finais (opcionais):\n{optional_nan_counts}")
    else:
        logger.info("Nenhum NaN opcional restante detectado.")
    return df
    
def calculate_probabilities(df: pd.DataFrame, epsilon=1e-6) -> pd.DataFrame:
    """Calcula probs implícitas p_H, p_D, p_A. Requer Odd_H/D/A_FT."""
    df_calc = df.copy()
    required_odds = list(ODDS_COLS.values())  # Usa config
    if not all(c in df_calc.columns for c in required_odds):
        logger.warning("Odds 1x2 ausentes para calcular Probabilidades.")
        df_calc[['p_H', 'p_D', 'p_A']] = np.nan
        return df_calc

    for col in required_odds:
        # Ensure numeric, but NaNs should already be handled by load_historical_data
        df_calc[col] = pd.to_numeric(df_calc[col], errors='coerce')
        # Keep NaN where odds were invalid (<=1 handled in loading)

    # Evita divisão por zero e lida com NaNs
    odd_h = df_calc[ODDS_COLS['home']].replace(0, np.nan) # Replace 0 with NaN just in case
    odd_d = df_calc[ODDS_COLS['draw']].replace(0, np.nan) # Replace 0 with NaN
    odd_a = df_calc[ODDS_COLS['away']].replace(0, np.nan) # Replace 0 with NaN

    # Inverse will propagate NaNs correctly
    df_calc['p_H'] = 1 / odd_h
    df_calc['p_D'] = 1 / odd_d
    df_calc['p_A'] = 1 / odd_a
    logger.info("Probabilidades Implícitas (p_H, p_D, p_A) calculadas.")
    return df_calc

def calculate_normalized_probabilities(df: pd.DataFrame, epsilon=1e-6) -> pd.DataFrame:
    """Calcula probs normalizadas e diferença H/A. Requer p_H, p_D, p_A."""
    df_calc = df.copy()
    prob_cols = ['p_H', 'p_D', 'p_A']
    if not all(c in df_calc.columns for c in prob_cols):
        logger.warning("Probabilidades (p_H/D/A) ausentes para normalização.")
        df_calc[['p_H_norm', 'p_D_norm', 'p_A_norm', 'abs_ProbDiff_Norm']] = np.nan
        return df_calc

    df_calc['Overround'] = df_calc['p_H'] + df_calc['p_D'] + df_calc['p_A']
    # Evitar divisão por zero no Overround
    df_calc['Overround'] = df_calc['Overround'].replace(0, epsilon)

    df_calc['p_H_norm'] = df_calc['p_H'] / df_calc['Overround']
    df_calc['p_D_norm'] = df_calc['p_D'] / df_calc['Overround']
    df_calc['p_A_norm'] = df_calc['p_A'] / df_calc['Overround']
    df_calc['abs_ProbDiff_Norm'] = abs(df_calc['p_H_norm'] - df_calc['p_A_norm'])
    logger.info("Probabilidades Normalizadas (p_X_norm, abs_ProbDiff_Norm) calculadas.")
    return df_calc.drop(columns=['Overround'], errors='ignore')

def calculate_rolling_std(df: pd.DataFrame, stats_to_calc: List[str], window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Calcula desvio padrão móvel para as estatísticas especificadas."""
    df_calc = df.copy()
    teams = pd.concat([df_calc['Home'], df_calc['Away']]).unique()
    team_history: Dict[str, Dict[str, List[float]]] = {team: {stat: [] for stat in stats_to_calc} for team in teams}
    results_list = []
    rolling_cols_map = {}
    cols_to_calculate = {}

    logger.info(f"Iniciando cálculo Desvio Padrão Rolling (Janela={window})...")
    
    # Mapeia colunas e valida configurações
    for stat_prefix in stats_to_calc:
        std_col_h = f'Std_{stat_prefix}_H'
        std_col_a = f'Std_{stat_prefix}_A'
        skip_h = skip_a = False
        
        # Verifica se colunas já existem
        if std_col_h in df_calc.columns and pd.api.types.is_numeric_dtype(df_calc[std_col_h]):
            skip_h = True
        if std_col_a in df_calc.columns and pd.api.types.is_numeric_dtype(df_calc[std_col_a]):
            skip_a = True

        if skip_h and skip_a:
            logger.warning(f"{std_col_h}/{std_col_a} já existem.")
            continue

        # Determina colunas base
        if stat_prefix == 'Ptos':
            base_h, base_a = 'Ptos_H', 'Ptos_A'
        elif stat_prefix == 'VG':
            base_h, base_a = 'VG_H_raw', 'VG_A_raw'
        elif stat_prefix == 'CG':
            base_h, base_a = 'CG_H_raw', 'CG_A_raw'
        else:
            logger.warning(f"Prefixo StDev '{stat_prefix}' desconhecido.")
            continue

        # Valida existência das colunas base
        if base_h not in df_calc.columns or base_a not in df_calc.columns:
            logger.error(f"Erro StDev: Colunas base '{base_h}'/'{base_a}' não encontradas.")
            continue

        rolling_cols_map[stat_prefix] = {'home': base_h, 'away': base_a}
        if not skip_h:
            cols_to_calculate[stat_prefix + '_H'] = std_col_h
        if not skip_a:
            cols_to_calculate[stat_prefix + '_A'] = std_col_a

    if not cols_to_calculate:
        logger.info("Nenhum StDev Rolling novo a calcular.")
        return df_calc

    logger.info(f"Calculando StDev rolling para: {list(cols_to_calculate.keys())}")

    # Calcula estatísticas
    calculated_stats = []
    for index, row in tqdm(df_calc.iterrows(), total=len(df_calc), desc="Calc. Rolling StDev"):
        home_team = row['Home']
        away_team = row['Away']
        current_match_features = {'Index': index}

        # Calcula stats para time da casa
        for stat_prefix, base_cols in rolling_cols_map.items():
            std_col_h = f'Std_{stat_prefix}_H'
            if stat_prefix + '_H' in cols_to_calculate:
                hist_H = team_history[home_team][stat_prefix]
                recent = hist_H[-window:]
                current_match_features[std_col_h] = np.std(recent) if len(recent) >= 2 else np.nan

        # Calcula stats para time visitante
        for stat_prefix, base_cols in rolling_cols_map.items():
            std_col_a = f'Std_{stat_prefix}_A'
            if stat_prefix + '_A' in cols_to_calculate:
                hist_A = team_history[away_team][stat_prefix]
                recent = hist_A[-window:]
                current_match_features[std_col_a] = np.std(recent) if len(recent) >= 2 else np.nan

        calculated_stats.append(current_match_features)

        # Atualiza histórico
        for stat_prefix, base_cols in rolling_cols_map.items():
            if pd.notna(row[base_cols['home']]):
                team_history[home_team][stat_prefix].append(row[base_cols['home']])
            if pd.notna(row[base_cols['away']]):
                team_history[away_team][stat_prefix].append(row[base_cols['away']])

    # Finaliza e retorna
    df_rolling_stdev = pd.DataFrame(calculated_stats).set_index('Index')
    cols_to_join = [col for col in cols_to_calculate.values() if col in df_rolling_stdev.columns]
    logger.info(f"StDev Rolling calculado. Colunas adicionadas: {cols_to_join}")
    df_final = df_calc.join(df_rolling_stdev[cols_to_join]) if cols_to_join else df_calc
    return df_final

def calculate_rolling_stats(df: pd.DataFrame, stats_to_calc: List[str], window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Calcula médias móveis para as estatísticas especificadas."""
    df_calc = df.copy()
    teams = pd.concat([df_calc['Home'], df_calc['Away']]).astype(str).dropna().unique()
    if len(teams) == 0: logger.warning("Rolling Mean: Nenhum time válido."); return df_calc
    team_history: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=window)))
    results_list = []
    rolling_cols_map = {}
    cols_to_calculate = {}

    logger.info(f"Iniciando cálculo Médias Rolling (Janela={window})...")
    for stat_prefix in stats_to_calc:
        media_col_h = f'Media_{stat_prefix}_H'; media_col_a = f'Media_{stat_prefix}_A'
        base_h, base_a = None, None
        if stat_prefix == 'Ptos': base_h, base_a = 'Ptos_H', 'Ptos_A'
        elif stat_prefix == 'VG': base_h, base_a = 'VG_H_raw', 'VG_A_raw'
        elif stat_prefix == 'CG': base_h, base_a = 'CG_H_raw', 'CG_A_raw'
        else: logger.warning(f"Prefixo Média '{stat_prefix}' desconhecido."); continue
        if (base_h not in df_calc.columns and base_a not in df_calc.columns): logger.error(f"Erro Média: Colunas base '{base_h}'/'{base_a}' não encontradas."); continue
        if base_h not in df_calc.columns: base_h = None
        if base_a not in df_calc.columns: base_a = None
        rolling_cols_map[stat_prefix] = {'home': base_h, 'away': base_a}
        if media_col_h not in df_calc.columns or not pd.api.types.is_numeric_dtype(df_calc[media_col_h]): cols_to_calculate[stat_prefix + '_H'] = media_col_h
        if media_col_a not in df_calc.columns or not pd.api.types.is_numeric_dtype(df_calc[media_col_a]): cols_to_calculate[stat_prefix + '_A'] = media_col_a

    if not cols_to_calculate: logger.info("Nenhuma Média Rolling nova a calcular."); return df_calc
    logger.info(f"Calculando Médias rolling para: {list(cols_to_calculate.keys())}")

    calculated_stats = []
    for index, row in tqdm(df_calc.iterrows(), total=len(df_calc), desc="Calc. Rolling Médias"):
        home_team = row.get('Home'); away_team = row.get('Away')
        current_match_features = {'Index': index}
        if pd.isna(home_team) or pd.isna(away_team):
             for output_col in cols_to_calculate.values(): current_match_features[output_col] = np.nan
             calculated_stats.append(current_match_features); continue

        for stat_prefix, base_cols in rolling_cols_map.items():
            # Home Mean
            media_col_h_calc = f'Media_{stat_prefix}_H'
            if stat_prefix + '_H' in cols_to_calculate:
                hist_H_deque = team_history[home_team][stat_prefix]
                recent_H = list(hist_H_deque)
                current_match_features[media_col_h_calc] = np.nanmean(recent_H) if len(recent_H) > 0 else np.nan
            # Away Mean
            media_col_a_calc = f'Media_{stat_prefix}_A'
            if stat_prefix + '_A' in cols_to_calculate:
                hist_A_deque = team_history[away_team][stat_prefix]
                recent_A = list(hist_A_deque)
                current_match_features[media_col_a_calc] = np.nanmean(recent_A) if len(recent_A) > 0 else np.nan

        calculated_stats.append(current_match_features)

        # Update history
        for stat_prefix, base_cols in rolling_cols_map.items():
            base_h_name = base_cols.get('home')
            base_a_name = base_cols.get('away')
            if base_h_name and pd.notna(row.get(base_h_name)): team_history[home_team][stat_prefix].append(row[base_h_name])
            if base_a_name and pd.notna(row.get(base_a_name)): team_history[away_team][stat_prefix].append(row[base_a_name])

    df_rolling_means = pd.DataFrame(calculated_stats).set_index('Index')
    cols_to_join = [col for col in cols_to_calculate.values() if col in df_rolling_means.columns]
    logger.info(f"Médias Rolling calculadas. Colunas adicionadas/atualizadas: {cols_to_join}")
    df_final = df_calc.copy()
    df_final.update(df_rolling_means[cols_to_join])
    for col in cols_to_join: # Add new columns if they didn't exist
        if col not in df_final.columns: df_final[col] = df_rolling_means[col]
    return df_final

def calculate_binned_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features categóricas (bins). Requer Odd_D_FT."""
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
    epsilon = FEATURE_EPSILON # Use epsilon from config

    # CV_HDA
    if all(c in df_calc.columns for c in ODDS_COLS.values()):
        odds_matrix = df_calc[list(ODDS_COLS.values())].apply(pd.to_numeric, errors='coerce')
        mean_odds = odds_matrix.mean(axis=1)
        std_odds = odds_matrix.std(axis=1)
        df_calc['CV_HDA'] = std_odds.div(mean_odds.replace(0, epsilon)).fillna(0)
    else: logger.warning("Odds 1x2 ausentes p/ CV_HDA."); df_calc['CV_HDA'] = np.nan

    # Diff_Media_CG
    if 'Media_CG_H' in df_calc.columns and 'Media_CG_A' in df_calc.columns:
        df_calc['Diff_Media_CG'] = df_calc['Media_CG_H'] - df_calc['Media_CG_A']
    else: logger.warning("Médias CG ausentes p/ Diff_Media_CG."); df_calc['Diff_Media_CG'] = np.nan

    # --- NOVAS INTERAÇÕES ---
    # Interaction 1: p_D_norm vs CV_HDA (Existing)
    if 'p_D_norm' in df_calc.columns and 'CV_HDA' in df_calc.columns:
        df_calc[INTERACTION_P_D_NORM_X_CV_HDA] = df_calc['p_D_norm'] * df_calc['CV_HDA']
        df_calc[INTERACTION_P_D_NORM_DIV_CV_HDA] = df_calc['p_D_norm'] / (df_calc['CV_HDA'] + epsilon)
    else: logger.warning("p_D_norm ou CV_HDA ausente."); df_calc[[INTERACTION_P_D_NORM_X_CV_HDA, INTERACTION_P_D_NORM_DIV_CV_HDA]] = np.nan

    # Interaction 2: Prob/Odds Empate vs |PiRating_Diff| (Existing)
    if 'PiRating_Diff' in df_calc.columns:
        abs_pi_rating_diff = df_calc['PiRating_Diff'].abs()
        if 'p_D_norm' in df_calc.columns:
            df_calc[INTERACTION_P_D_NORM_X_PIR_DIFF] = df_calc['p_D_norm'] * abs_pi_rating_diff
            df_calc[INTERACTION_P_D_NORM_DIV_PIR_DIFF] = df_calc['p_D_norm'] / (abs_pi_rating_diff + epsilon)
        else: logger.warning("p_D_norm ausente."); df_calc[[INTERACTION_P_D_NORM_X_PIR_DIFF, INTERACTION_P_D_NORM_DIV_PIR_DIFF]] = np.nan
        odd_d_col = ODDS_COLS['draw']
        if odd_d_col in df_calc.columns:
            df_calc[INTERACTION_ODD_D_X_PIR_DIFF] = df_calc[odd_d_col] * abs_pi_rating_diff
            df_calc[INTERACTION_ODD_D_DIV_PIR_DIFF] = df_calc[odd_d_col] / (abs_pi_rating_diff + epsilon)
        else: logger.warning(f"{odd_d_col} ausente."); df_calc[[INTERACTION_ODD_D_X_PIR_DIFF, INTERACTION_ODD_D_DIV_PIR_DIFF]] = np.nan
    else: logger.warning("PiRating_Diff ausente."); df_calc[[INTERACTION_P_D_NORM_X_PIR_DIFF, INTERACTION_P_D_NORM_DIV_PIR_DIFF, INTERACTION_ODD_D_X_PIR_DIFF, INTERACTION_ODD_D_DIV_PIR_DIFF]] = np.nan

    # Interaction 3: PiRating Prob H vs Odd H
    odd_h_col = ODDS_COLS['home']
    if 'PiRating_Prob_H' in df_calc.columns and odd_h_col in df_calc.columns:
        df_calc[INTERACTION_PIR_PROBH_X_ODD_H] = df_calc['PiRating_Prob_H'] * df_calc[odd_h_col]
    else: logger.warning("PiRating_Prob_H ou Odd_H_FT ausente."); df_calc[INTERACTION_PIR_PROBH_X_ODD_H] = np.nan

    # Interaction 4: Diff Gols Marcados (Rolling or EWMA)
    # Prioritize EWMA if available, else use simple rolling
    marc_h = EWMA_GolsMarc_H_LONG if EWMA_GolsMarc_H_LONG in df_calc.columns else 'Media_GolsMarcados_H'
    marc_a = EWMA_GolsMarc_A_LONG if EWMA_GolsMarc_A_LONG in df_calc.columns else 'Media_GolsMarcados_A'
    if marc_h in df_calc.columns and marc_a in df_calc.columns:
        df_calc[INTERACTION_AVG_GOLS_MARC_DIFF] = df_calc[marc_h] - df_calc[marc_a]
    else: logger.warning(f"{marc_h} ou {marc_a} ausente."); df_calc[INTERACTION_AVG_GOLS_MARC_DIFF] = np.nan

    # Interaction 5: Diff Gols Sofridos (Rolling or EWMA)
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
    """Calcula P(Empate) Poisson usando Força de Ataque/Defesa (FA/FD)."""
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
        # Calculate probabilities for all k at once using vectorized operations
        prob_h_k = poisson.pmf(k_range[:, np.newaxis], lambda_h.values[np.newaxis, :])
        prob_a_k = poisson.pmf(k_range[:, np.newaxis], lambda_a.values[np.newaxis, :])
        # Multiply probabilities for k=k and sum along the k axis
        prob_empate_total = np.sum(prob_h_k * prob_a_k, axis=0)

    except Exception as e:
        logger.error(f"Erro cálculo Poisson PMF: {e}", exc_info=True)
        df_calc['Prob_Empate_Poisson'] = np.nan
        return df_calc

    df_calc['Prob_Empate_Poisson'] = prob_empate_total
    logger.info("Prob_Empate_Poisson (Refinado) calculado.")
    return df_calc

def calculate_pi_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula Pi-Ratings e Pi-Rating Momentum. Requer df ORDENADO por data.
    """
    logger.info(f"Calculando Pi-Ratings e Momentum (Janela={ROLLING_WINDOW})...")
    if not df.index.is_monotonic_increasing:
         logger.warning("DataFrame não ordenado por índice (Data?). Ordenando...")
         df = df.sort_index()

    ratings = {}
    rating_history = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW + 1))
    results_pi = []

    goals_h_col = GOALS_COLS.get('home', 'Goals_H_FT')
    goals_a_col = GOALS_COLS.get('away', 'Goals_A_FT')
    # Ensure columns exist before proceeding
    if goals_h_col not in df.columns or goals_a_col not in df.columns:
        logger.error(f"Colunas de Gols '{goals_h_col}' ou '{goals_a_col}' ausentes. Não é possível calcular PiRatings.")
        # Add NaN columns and return
        pi_cols = ['PiRating_H', 'PiRating_A', 'PiRating_Diff', 'PiRating_Prob_H', PIRATING_MOMENTUM_H, PIRATING_MOMENTUM_A, PIRATING_MOMENTUM_DIFF]
        for col in pi_cols: df[col] = np.nan
        return df

    df[goals_h_col] = pd.to_numeric(df[goals_h_col], errors='coerce')
    df[goals_a_col] = pd.to_numeric(df[goals_a_col], errors='coerce')

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Calculando Pi-Ratings/Momentum"):
        home_team = row.get('Home'); away_team = row.get('Away')
        if pd.isna(home_team) or pd.isna(away_team): # Skip if team names are missing
             results_pi.append({'Index': index, 'PiRating_H': np.nan, 'PiRating_A': np.nan, 'PiRating_Diff': np.nan, 'PiRating_Prob_H': np.nan, PIRATING_MOMENTUM_H: np.nan, PIRATING_MOMENTUM_A: np.nan, PIRATING_MOMENTUM_DIFF: np.nan})
             continue

        rating_h = ratings.get(home_team, PI_RATING_INITIAL); rating_a = ratings.get(away_team, PI_RATING_INITIAL)

        hist_h = rating_history[home_team]; rating_h_n_ago = None
        if len(hist_h) >= ROLLING_WINDOW: rating_h_n_ago = hist_h[0][1]
        pi_rating_mom_h = (rating_h - rating_h_n_ago) if rating_h_n_ago is not None else np.nan

        hist_a = rating_history[away_team]; rating_a_n_ago = None
        if len(hist_a) >= ROLLING_WINDOW: rating_a_n_ago = hist_a[0][1]
        pi_rating_mom_a = (rating_a - rating_a_n_ago) if rating_a_n_ago is not None else np.nan
        pi_rating_mom_diff = pi_rating_mom_h - pi_rating_mom_a if pd.notna(pi_rating_mom_h) and pd.notna(pi_rating_mom_a) else np.nan

        rating_h_adj = rating_h + PI_RATING_HOME_ADVANTAGE; rating_diff = rating_h_adj - rating_a
        expected_h = 1 / (1 + 10**(-rating_diff / 400))

        match_pi_data = {
            'Index': index, 'PiRating_H': rating_h, 'PiRating_A': rating_a,
            'PiRating_Diff': rating_h - rating_a, 'PiRating_Prob_H': expected_h,
            PIRATING_MOMENTUM_H: pi_rating_mom_h, PIRATING_MOMENTUM_A: pi_rating_mom_a,
            PIRATING_MOMENTUM_DIFF: pi_rating_mom_diff
        }
        results_pi.append(match_pi_data)

        score_h = np.nan; gh=row[goals_h_col]; ga=row[goals_a_col]
        if pd.notna(gh) and pd.notna(ga): score_h = 1.0 if gh > ga else (0.5 if gh == ga else 0.0)
        new_rating_h, new_rating_a = rating_h, rating_a
        if pd.notna(score_h):
            new_rating_h = rating_h + PI_RATING_K_FACTOR * (score_h - expected_h)
            new_rating_a = rating_a + PI_RATING_K_FACTOR * ((1 - score_h) - (1 - expected_h))
        ratings[home_team] = new_rating_h; ratings[away_team] = new_rating_a
        rating_history[home_team].append((index, new_rating_h)); rating_history[away_team].append((index, new_rating_a))

    df_pi_ratings = pd.DataFrame(results_pi).set_index('Index')
    df_out = df.copy()
    # Use update which handles overlapping columns safely
    df_out.update(df_pi_ratings)
    # Add new columns if they don't exist (first run)
    for col in df_pi_ratings.columns:
        if col not in df_out.columns:
            df_out[col] = df_pi_ratings[col]

    logger.info(f"Pi-Ratings/Momentum calculados. Colunas: {list(df_pi_ratings.columns)}")
    nan_counts_new = df_out[list(df_pi_ratings.columns)].isnull().sum()
    logger.debug(f"NaNs PiRating/Momentum:\n{nan_counts_new[nan_counts_new > 0]}")
    return df_out

def calculate_historical_intermediate(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula stats intermediárias."""
    df_calc = df.copy()
    logger.info("Calculando stats intermediárias...")
    epsilon = FEATURE_EPSILON # Use epsilon from config

    gh = GOALS_COLS.get('home', 'Goals_H_FT'); ga = GOALS_COLS.get('away', 'Goals_A_FT')
    if gh in df_calc.columns and ga in df_calc.columns:
        h_g = pd.to_numeric(df_calc[gh], errors='coerce'); a_g = pd.to_numeric(df_calc[ga], errors='coerce')
        condlist = [h_g > a_g, h_g == a_g, h_g < a_g]; choicelist_res = ["H", "D", "A"]
        df_calc['FT_Result'] = np.select(condlist, choicelist_res, default=pd.NA)
        condlist_pts_h = [df_calc['FT_Result']=='H', df_calc['FT_Result']=='D', df_calc['FT_Result']=='A']; choicelist_pts_h = [3, 1, 0]
        df_calc['Ptos_H'] = np.select(condlist_pts_h, choicelist_pts_h, default=np.nan)
        condlist_pts_a = [df_calc['FT_Result']=='A', df_calc['FT_Result']=='D', df_calc['FT_Result']=='H']; choicelist_pts_a = [3, 1, 0]
        df_calc['Ptos_A'] = np.select(condlist_pts_a, choicelist_pts_a, default=np.nan)
        df_calc['IsDraw'] = (df_calc['FT_Result'] == 'D').astype('Int64'); df_calc.loc[df_calc['FT_Result'].isna(), 'IsDraw'] = pd.NA
        logger.info("->Result/IsDraw/Ptos OK.")
    else: logger.warning(f"->Gols'{gh}'/'{ga}'ausentes."); df_calc[['FT_Result', 'IsDraw', 'Ptos_H', 'Ptos_A']] = pd.NA

    req_odds = list(ODDS_COLS.values())
    if not all(p in df_calc.columns for p in ['p_H', 'p_D', 'p_A']):
        if all(c in df_calc.columns for c in req_odds): logger.info("->Calculando p_H/D/A..."); df_calc = calculate_probabilities(df_calc)
        else: logger.warning("->Odds 1x2 ausentes p/ Probs."); df_calc[['p_H', 'p_D', 'p_A']] = np.nan

    prob_n = ['p_H', 'p_A']; goal_n = [gh, ga]
    if all(c in df_calc.columns for c in prob_n + goal_n):
        h_g = pd.to_numeric(df_calc[gh], errors='coerce'); a_g = pd.to_numeric(df_calc[ga], errors='coerce')
        p_H = pd.to_numeric(df_calc['p_H'], errors='coerce'); p_A = pd.to_numeric(df_calc['p_A'], errors='coerce')
        df_calc['VG_H_raw'] = h_g * p_A; df_calc['VG_A_raw'] = a_g * p_H
        df_calc['CG_H_raw'] = np.where((h_g > epsilon) & p_H.notna(), p_H / h_g, np.nan) # Use > instead of != for float
        df_calc['CG_A_raw'] = np.where((a_g > epsilon) & p_A.notna(), p_A / a_g, np.nan)
        logger.info("->VG/CG Raw OK.")
    else: logger.warning(f"->Inputs VG/CG Raw ausentes."); df_calc[['VG_H_raw', 'VG_A_raw', 'CG_H_raw', 'CG_A_raw']] = np.nan

    logger.info("Cálculo Intermediárias concluído.")
    return df_calc

def calculate_rolling_goal_stats(
    df: pd.DataFrame,
    window: int = ROLLING_WINDOW,
    avg_goals_home_league: Optional[float] = None,
    avg_goals_away_league: Optional[float] = None
    ) -> pd.DataFrame:
    """
    Calcula médias móveis de gols e Força de Ataque/Defesa ajustada pela liga.
    (Divisão FA/FD mais robusta)
    """
    df_calc = df.copy()
    goals_h_col = GOALS_COLS.get('home', 'Goals_H_FT')
    goals_a_col = GOALS_COLS.get('away', 'Goals_A_FT')
    epsilon = 1e-6 # Small number to avoid division by zero

    if goals_h_col not in df_calc.columns or goals_a_col not in df_calc.columns:
        logger.warning("Calc Rolling Goals: Colunas Gols ausentes.")
        cols_to_add = ['Avg_Gols_Marcados_H', 'Avg_Gols_Sofridos_H', 'Avg_Gols_Marcados_A', 'Avg_Gols_Sofridos_A',
                       'FA_H', 'FD_H', 'FA_A', 'FD_A']
        for col in cols_to_add:
            df_calc[col] = np.nan
        return df_calc

    # Use nanmean for calculating league averages if not provided
    if avg_goals_home_league is None:
         avg_h_league_calc = np.nanmean(df_calc[goals_h_col])
         avg_h_league = avg_h_league_calc if pd.notna(avg_h_league_calc) else 1.0 # Fallback if all NaN
         logger.warning(f"Média gols casa da liga não fornecida. Calculada como: {avg_h_league:.3f}")
    else:
         avg_h_league = avg_goals_home_league

    if avg_goals_away_league is None:
         avg_a_league_calc = np.nanmean(df_calc[goals_a_col])
         avg_a_league = avg_a_league_calc if pd.notna(avg_a_league_calc) else 1.0 # Fallback if all NaN
         logger.warning(f"Média gols fora da liga não fornecida. Calculada como: {avg_a_league:.3f}")
    else:
         avg_a_league = avg_goals_away_league

    # Ensure league averages are not zero for division
    avg_h_league_safe = max(avg_h_league, epsilon)
    avg_a_league_safe = max(avg_a_league, epsilon)
    if avg_h_league <= epsilon: logger.warning(f"Média gols casa da liga é <= {epsilon}. Usando {epsilon} para divisão FA/FD.")
    if avg_a_league <= epsilon: logger.warning(f"Média gols fora da liga é <= {epsilon}. Usando {epsilon} para divisão FA/FD.")


    teams = pd.concat([df_calc['Home'], df_calc['Away']]).unique()
    team_history = {
        team: {'scored_home': [], 'conceded_home': [], 'scored_away': [], 'conceded_away': []}
        for team in teams
    }
    results_list = []
    logger.info(f"Calculando Rolling Gols e Forças FA/FD (Janela={window})...")

    for index, row in tqdm(df_calc.iterrows(), total=len(df_calc), desc="Calc Rolling Goals/FA/FD"):
        home_team = row['Home']
        away_team = row['Away']
        current_stats = {'Index': index}

        # --- Home Team Calculations ---
        if home_team in team_history:
            h_scored_hist = team_history[home_team]['scored_home']
            h_conceded_hist = team_history[home_team]['conceded_home']
            a_scored_hist_as_visitor = team_history[home_team]['scored_away'] # Need away goals for FD_H
            a_conceded_hist_as_visitor = team_history[home_team]['conceded_away'] # Need away conceded for FA_H? No.

            # Use combined history for overall form, then specific home/away for FA/FD context
            all_scored = h_scored_hist + a_scored_hist_as_visitor
            all_conceded = h_conceded_hist + a_conceded_hist_as_visitor

            avg_gs_h = np.nanmean(all_scored[-window:]) if all_scored else np.nan
            avg_gc_h = np.nanmean(all_conceded[-window:]) if all_conceded else np.nan

            current_stats['Avg_Gols_Marcados_H'] = avg_gs_h
            current_stats['Avg_Gols_Sofridos_H'] = avg_gc_h

            # FA_H: Team's avg goals scored (home & away) / League's avg HOME goals
            current_stats['FA_H'] = avg_gs_h / avg_h_league_safe if pd.notna(avg_gs_h) else np.nan
            # FD_H: Team's avg goals conceded (home & away) / League's avg AWAY goals
            current_stats['FD_H'] = avg_gc_h / avg_a_league_safe if pd.notna(avg_gc_h) else np.nan
        else: # Home team not in history dict (shouldn't happen with unique())
            current_stats['Avg_Gols_Marcados_H']=np.nan; current_stats['Avg_Gols_Sofridos_H']=np.nan
            current_stats['FA_H']=np.nan; current_stats['FD_H']=np.nan


        # --- Away Team Calculations ---
        if away_team in team_history:
            a_scored_hist = team_history[away_team]['scored_away']
            a_conceded_hist = team_history[away_team]['conceded_away']
            h_scored_hist_as_home = team_history[away_team]['scored_home'] # Need home goals for FD_A
            h_conceded_hist_as_home = team_history[away_team]['conceded_home'] # Need home conceded for FA_A? No.

            all_scored_a = h_scored_hist_as_home + a_scored_hist
            all_conceded_a = h_conceded_hist_as_home + a_conceded_hist

            avg_gs_a = np.nanmean(all_scored_a[-window:]) if all_scored_a else np.nan
            avg_gc_a = np.nanmean(all_conceded_a[-window:]) if all_conceded_a else np.nan

            current_stats['Avg_Gols_Marcados_A'] = avg_gs_a
            current_stats['Avg_Gols_Sofridos_A'] = avg_gc_a

             # FA_A: Team's avg goals scored (home & away) / League's avg AWAY goals
            current_stats['FA_A'] = avg_gs_a / avg_a_league_safe if pd.notna(avg_gs_a) else np.nan
            # FD_A: Team's avg goals conceded (home & away) / League's avg HOME goals
            current_stats['FD_A'] = avg_gc_a / avg_h_league_safe if pd.notna(avg_gc_a) else np.nan
        else: # Away team not in history dict
            current_stats['Avg_Gols_Marcados_A']=np.nan; current_stats['Avg_Gols_Sofridos_A']=np.nan
            current_stats['FA_A']=np.nan; current_stats['FD_A']=np.nan


        results_list.append(current_stats)

        # Update history (only if goals are valid numbers)
        home_goals = pd.to_numeric(row.get(goals_h_col), errors='coerce')
        away_goals = pd.to_numeric(row.get(goals_a_col), errors='coerce')
        if home_team in team_history:
            if pd.notna(home_goals): team_history[home_team]['scored_home'].append(home_goals)
            if pd.notna(away_goals): team_history[home_team]['conceded_home'].append(away_goals) # Home team conceded away goals
        if away_team in team_history:
            if pd.notna(away_goals): team_history[away_team]['scored_away'].append(away_goals)
            if pd.notna(home_goals): team_history[away_team]['conceded_away'].append(home_goals) # Away team conceded home goals


    df_rolling_stats = pd.DataFrame(results_list).set_index('Index')
    logger.info(f"Rolling Gols/FA/FD calculado. Colunas: {list(df_rolling_stats.columns)}")
    # Join calculated stats. Handle potential duplicate columns if called multiple times.
    cols_to_join = [col for col in df_rolling_stats.columns if col not in df_calc.columns]
    df_final = df_calc.join(df_rolling_stats[cols_to_join])
    # Update existing columns if they were already present (overwrite with new calculation)
    cols_to_update = [col for col in df_rolling_stats.columns if col in df_calc.columns]
    if cols_to_update:
        df_final.update(df_rolling_stats[cols_to_update])

    return df_final

def calculate_general_rolling_stats(
    df: pd.DataFrame,
    stats_configs: List[Dict[str, Any]],
    default_window: int = ROLLING_WINDOW
) -> pd.DataFrame:
    if df.empty: return df
    logger.info(f"Calculando {len(stats_configs)} métricas rolling (Janela Padrão={default_window})...")
    df_calc = df.copy()
    teams_home = df_calc['Home'].astype(str).dropna(); teams_away = df_calc['Away'].astype(str).dropna()
    teams = pd.concat([teams_home, teams_away]).unique()
    if len(teams) == 0: logger.warning("Nenhum time válido p/ stats rolling."); return df_calc # Add NaN columns logic removed for brevity, assume valid teams exist
    team_history = defaultdict(lambda: defaultdict(lambda: deque(maxlen=default_window + 5)))
    calculated_stats_list = []; valid_configs = []; output_cols_mapping = {}; base_cols_needed = set(['Home', 'Away'])
    for cfg in stats_configs: # Valida configs
        prefix = cfg.get('output_prefix'); agg_func = cfg.get('agg_func'); base_h = cfg.get('base_col_h'); base_a = cfg.get('base_col_a')
        if not prefix or not agg_func: logger.warning(f"Config inválida: {cfg}."); continue
        h_exists = base_h and base_h in df_calc.columns; a_exists = base_a and base_a in df_calc.columns
        if not h_exists and not a_exists: logger.warning(f"Cols base '{base_h}'/'{base_a}' ausentes p/ {prefix}."); continue
        if h_exists: base_cols_needed.add(base_h); cfg['base_col_h'] = base_h # Ensure correct name
        else: cfg['base_col_h'] = None
        if a_exists: base_cols_needed.add(base_a); cfg['base_col_a'] = base_a # Ensure correct name
        else: cfg['base_col_a'] = None
        context = cfg.get('context', 'all'); output_col_h = f"{prefix}_{context}_H" if context != 'all' else f"{prefix}_H"; output_col_a = f"{prefix}_{context}_A" if context != 'all' else f"{prefix}_A"
        output_cols_mapping[output_col_h] = cfg; output_cols_mapping[output_col_a] = cfg; valid_configs.append(cfg)
    logger.debug(f"Colunas rolling a calcular: {list(output_cols_mapping.keys())}")
    actual_base_cols_needed = [col for col in base_cols_needed if col in df_calc.columns]; logger.debug(f"Colunas base necessárias: {actual_base_cols_needed}")
    if 'Home' not in actual_base_cols_needed: actual_base_cols_needed.append('Home')
    if 'Away' not in actual_base_cols_needed: actual_base_cols_needed.append('Away')
    for index, row_data in tqdm(df_calc[actual_base_cols_needed].iterrows(), total=len(df_calc), desc="Calc Rolling Geral"):
        home_team = row_data.get('Home'); away_team = row_data.get('Away')
        if pd.isna(home_team) or pd.isna(away_team): nan_stats_for_row={'Index':index}; [nan_stats_for_row.update({out_col:np.nan}) for out_col in output_cols_mapping.keys()]; calculated_stats_list.append(nan_stats_for_row); continue
        current_match_stats = {'Index': index}
        for output_col, cfg in output_cols_mapping.items():
            is_home_stat = output_col.endswith("_H"); team_name = home_team if is_home_stat else away_team
            if pd.isna(team_name): current_match_stats[output_col] = np.nan; continue
            prefix=cfg['output_prefix']; agg_func=cfg['agg_func']; window=cfg.get('window', default_window); min_p=cfg.get('min_periods', 1 if agg_func!=np.nanstd else 2);
            base_h=cfg.get('base_col_h'); base_a=cfg.get('base_col_a'); context = cfg.get('context', 'all'); stat_type=cfg.get('stat_type', 'offensive');
            hist_values = []
            try:
                # Acesso direto ao defaultdict do time
                team_data = team_history[team_name]

                key_home = base_h if stat_type == 'offensive' else base_a
                key_away = base_a if stat_type == 'offensive' else base_h

                # --- Select history using .get() on the inner dictionary (team_data) ---
                if context == 'all':
                    if key_home and key_home in team_data:
                         hist_values.extend(list(team_data.get(key_home, deque())))
                    if key_away and key_away in team_data:
                         hist_values.extend(list(team_data.get(key_away, deque())))
                elif context == 'home':
                    col_to_use = key_home
                    if col_to_use and col_to_use in team_data: # <<< MODIFICAÇÃO >>>
                         hist_values = list(team_data.get(col_to_use, deque()))
                elif context == 'away':
                    col_to_use = key_away
                    if col_to_use and col_to_use in team_data: # <<< MODIFICAÇÃO >>>
                         hist_values = list(team_data.get(col_to_use, deque()))
                # --- End history selection ---

            except Exception as e_hist:
                 logger.error(f"Erro inesperado ao acessar histórico {team_name}/{prefix}/{context}: {e_hist}", exc_info=True)
                 current_match_stats[output_col] = np.nan
                 continue
            recent_values = hist_values[-window:] if hist_values else []
            if len(recent_values) >= min_p:
                try:
                    with warnings.catch_warnings(): warnings.simplefilter("ignore", category=RuntimeWarning); calculated_value = agg_func(recent_values)
                    current_match_stats[output_col] = calculated_value if pd.notna(calculated_value) and np.isfinite(calculated_value) else np.nan
                except Exception as e_agg: logger.warning(f"Erro agg func {agg_func.__name__} p/ {output_col}: {e_agg}"); current_match_stats[output_col] = np.nan
            else: current_match_stats[output_col] = np.nan
        calculated_stats_list.append(current_match_stats)
        for cfg_update in valid_configs: # Atualiza histórico
            base_col_h = cfg_update.get('base_col_h'); base_col_a = cfg_update.get('base_col_a');
            val_h = row_data.get(base_col_h); val_a = row_data.get(base_col_a)
            if base_col_h and pd.notna(val_h): team_history[home_team][base_col_h].append(val_h)
            if base_col_a and pd.notna(val_a): team_history[away_team][base_col_a].append(val_a)
    if not calculated_stats_list: logger.warning("Nenhuma stat rolling calculada."); return df_calc
    df_rolling = pd.DataFrame(calculated_stats_list).set_index('Index'); logger.info(f"Métricas rolling calculadas. Shape: {df_rolling.shape}.")
    df_final = df_calc.join(df_rolling, how='left')
    nan_counts_rolling = df_final[list(output_cols_mapping.keys())].isnull().sum(); logger.debug(f"NaNs rolling após join:\n{nan_counts_rolling[nan_counts_rolling > 0]}")
    return df_final

def calculate_ewma_stats(
    df: pd.DataFrame,
    stats_configs: List[Dict[str, Any]],
    default_span: int = ROLLING_WINDOW
) -> pd.DataFrame:
    if df.empty: logger.warning("EWMA: DF vazio."); return df
    logger.info(f"Calculando {len(stats_configs)} métricas EWMA (Span Padrão={default_span})...")
    df_calc = df.copy()
    teams_home = df_calc['Home'].astype(str).dropna(); teams_away = df_calc['Away'].astype(str).dropna()
    teams = pd.concat([teams_home, teams_away]).unique()
    if len(teams) == 0: logger.warning("EWMA: Nenhum time válido."); return df_calc
    team_history = defaultdict(lambda: defaultdict(lambda: deque(maxlen=len(df_calc)))) # Maxlen grande para EWMA
    calculated_stats_list = []; valid_configs_ewma = []; output_cols_mapping_ewma = {}; base_cols_needed_ewma = set(['Home', 'Away'])
    for cfg in stats_configs: # Valida configs
        prefix = cfg.get('output_prefix'); base_h = cfg.get('base_col_h'); base_a = cfg.get('base_col_a'); span = cfg.get('span', default_span)
        if not prefix: logger.warning(f"Config EWMA inválida: {cfg}."); continue
        h_exists = base_h and base_h in df_calc.columns; a_exists = base_a and base_a in df_calc.columns
        if not h_exists and not a_exists: logger.warning(f"Cols base EWMA '{base_h}'/'{base_a}' ausentes p/ {prefix}."); continue
        if h_exists: base_cols_needed_ewma.add(base_h); cfg['base_col_h'] = base_h
        else: cfg['base_col_h'] = None
        if a_exists: base_cols_needed_ewma.add(base_a); cfg['base_col_a'] = base_a
        else: cfg['base_col_a'] = None
        context = cfg.get('context', 'all'); output_col_h = f"{prefix}_s{span}_{context}_H" if context != 'all' else f"{prefix}_s{span}_H"; output_col_a = f"{prefix}_s{span}_{context}_A" if context != 'all' else f"{prefix}_s{span}_A"
        cfg_with_span = cfg.copy(); cfg_with_span['span'] = span
        output_cols_mapping_ewma[output_col_h] = cfg_with_span; output_cols_mapping_ewma[output_col_a] = cfg_with_span; valid_configs_ewma.append(cfg_with_span)
    logger.debug(f"Colunas EWMA a calcular: {list(output_cols_mapping_ewma.keys())}")
    actual_base_cols_needed_ewma = [col for col in base_cols_needed_ewma if col in df_calc.columns]; logger.debug(f"Colunas base EWMA necessárias: {actual_base_cols_needed_ewma}")
    if 'Home' not in actual_base_cols_needed_ewma: actual_base_cols_needed_ewma.append('Home')
    if 'Away' not in actual_base_cols_needed_ewma: actual_base_cols_needed_ewma.append('Away')
    for index, row_data in tqdm(df_calc[actual_base_cols_needed_ewma].iterrows(), total=len(df_calc), desc="Calc EWMA Geral"):
        home_team = row_data.get('Home'); away_team = row_data.get('Away')
        if pd.isna(home_team) or pd.isna(away_team): nan_stats_for_row={'Index':index}; [nan_stats_for_row.update({out_col:np.nan}) for out_col in output_cols_mapping_ewma.keys()]; calculated_stats_list.append(nan_stats_for_row); continue
        current_match_stats = {'Index': index}
        for output_col, cfg in output_cols_mapping_ewma.items():
            is_home_stat = output_col.endswith("_H"); team_name = home_team if is_home_stat else away_team
            if pd.isna(team_name): current_match_stats[output_col] = np.nan; continue
            prefix=cfg['output_prefix']; span=cfg['span']; min_p=cfg.get('min_periods', 1); base_h=cfg.get('base_col_h'); base_a=cfg.get('base_col_a'); context = cfg.get('context', 'all'); stat_type=cfg.get('stat_type', 'offensive');
            hist_values = []
            try:
                team_data = team_history[team_name] # Acesso direto

                key_home = base_h if stat_type == 'offensive' else base_a
                key_away = base_a if stat_type == 'offensive' else base_h

                # --- Select history ---
                if context == 'all':
                    if key_home and key_home in team_data:
                        hist_values.extend(list(team_data.get(key_home, deque())))
                    if key_away and key_away in team_data:
                        hist_values.extend(list(team_data.get(key_away, deque())))
                elif context == 'home':
                    col_to_use = key_home
                    if col_to_use and col_to_use in team_data:
                        hist_values = list(team_data.get(col_to_use, deque()))
                elif context == 'away':
                    col_to_use = key_away
                    if col_to_use and col_to_use in team_data:
                        hist_values = list(team_data.get(col_to_use, deque()))
                if len(hist_values) >= min_p:
                     hist_series = pd.Series(hist_values)
                     ewma_series = hist_series.ewm(span=span, adjust=True, min_periods=min_p).mean()
                     calculated_value = ewma_series.iloc[-1] if not ewma_series.empty else np.nan
                     current_match_stats[output_col] = calculated_value if pd.notna(calculated_value) and np.isfinite(calculated_value) else np.nan
                else: current_match_stats[output_col] = np.nan
            except Exception as e_ewm_calc:
                logger.error(f"Erro calcular EWMA {output_col} p/ {team_name}: {e_ewm_calc}", exc_info=False);
                current_match_stats[output_col] = np.nan
                continue 
        calculated_stats_list.append(current_match_stats)
        for cfg_update in valid_configs_ewma: # Atualiza histórico
            base_col_h = cfg_update.get('base_col_h'); base_col_a = cfg_update.get('base_col_a');
            val_h = row_data.get(base_col_h); val_a = row_data.get(base_col_a)
            if base_col_h and pd.notna(val_h): team_history[home_team][base_col_h].append(val_h)
            if base_col_a and pd.notna(val_a): team_history[away_team][base_col_a].append(val_a)
    if not calculated_stats_list: logger.warning("Nenhuma stat EWMA calculada."); return df_calc
    df_ewma = pd.DataFrame(calculated_stats_list).set_index('Index'); logger.info(f"Métricas EWMA calculadas. Shape: {df_ewma.shape}.")
    df_final = df_calc.join(df_ewma, how='left')
    nan_counts_ewma = df_final[list(output_cols_mapping_ewma.keys())].isnull().sum(); logger.debug(f"NaNs EWMA após join:\n{nan_counts_ewma[nan_counts_ewma > 0]}")
    return df_final

# --- MODIFICADO: preprocess_and_feature_engineer ---
def preprocess_and_feature_engineer(df_loaded: pd.DataFrame) -> Optional[Tuple[pd.DataFrame, pd.Series, List[str]]]:
    if df_loaded is None or df_loaded.empty: logger.error("Pré-proc: DF inválido."); return None
    logger.info("--- Iniciando Pré-proc e Feature Eng (Histórico) ---")
    goals_h_col=GOALS_COLS.get('home'); goals_a_col=GOALS_COLS.get('away'); avg_h_league=np.nanmean(df_loaded[goals_h_col]) if goals_h_col in df_loaded else 1.0; avg_a_league=np.nanmean(df_loaded[goals_a_col]) if goals_a_col in df_loaded else 1.0; avg_h_league=1.0 if pd.isna(avg_h_league) else avg_h_league; avg_a_league=1.0 if pd.isna(avg_a_league) else avg_a_league; avg_h_league_safe=max(avg_h_league,FEATURE_EPSILON); avg_a_league_safe=max(avg_a_league,FEATURE_EPSILON); logger.info(f"Médias Liga: Casa={avg_h_league_safe:.3f}, Fora={avg_a_league_safe:.3f}")
    df_processed = df_loaded.copy(); logger.info("=== INÍCIO PIPELINE FEAT ENG (HISTÓRICO) ===")

    # 1. Stats básicas e probs
    df_processed = calculate_historical_intermediate(df_processed);
    if 'IsDraw' not in df_processed.columns or df_processed['IsDraw'].isnull().all(): logger.error("Alvo 'IsDraw' ausente."); return None
    df_processed = calculate_probabilities(df_processed); df_processed = calculate_normalized_probabilities(df_processed);

    # 2. PiRatings & Momentum
    df_processed = calculate_pi_ratings(df_processed)

    # 3. Rolling Stats Generalizadas (MÉDIAS SIMPLES)
    stats_to_calculate = [
        {'base_col_h': 'VG_H_raw', 'base_col_a': 'VG_A_raw', 'output_prefix': 'Media_VG', 'agg_func': np.nanmean, 'window': ROLLING_WINDOW},
        {'base_col_h': 'CG_H_raw', 'base_col_a': 'CG_A_raw', 'output_prefix': 'Media_CG', 'agg_func': np.nanmean, 'window': ROLLING_WINDOW},
        {'base_col_h': 'CG_H_raw', 'base_col_a': 'CG_A_raw', 'output_prefix': 'Std_CG', 'agg_func': np.nanstd, 'min_periods': 2, 'window': ROLLING_WINDOW},
        {'base_col_h': GOALS_COLS.get('home'), 'base_col_a': GOALS_COLS.get('away'), 'stat_type': 'offensive', 'output_prefix': 'Media_GolsMarcados', 'agg_func': np.nanmean, 'window': ROLLING_WINDOW, 'context': 'all'},
        {'base_col_h': GOALS_COLS.get('away'), 'base_col_a': GOALS_COLS.get('home'), 'stat_type': 'defensive', 'output_prefix': 'Media_GolsSofridos', 'agg_func': np.nanmean, 'window': ROLLING_WINDOW, 'context': 'all'},
        {'base_col_h': 'Shots_H', 'base_col_a': 'Shots_A', 'stat_type': 'offensive', 'output_prefix': 'Media_ChutesTotal', 'agg_func': np.nanmean, 'window': ROLLING_WINDOW, 'context': 'all'},
        {'base_col_h': 'ShotsOnTarget_H', 'base_col_a': 'ShotsOnTarget_A', 'stat_type': 'offensive', 'output_prefix': 'Media_ChutesAlvo', 'agg_func': np.nanmean, 'window': ROLLING_WINDOW, 'context': 'all'},
        {'base_col_h': 'Corners_H_FT', 'base_col_a': 'Corners_A_FT', 'stat_type': 'offensive', 'output_prefix': 'Media_Escanteios', 'agg_func': np.nanmean, 'window': 10, 'context': 'all'},
        {'base_col_h': 'Corners_A_FT', 'base_col_a': 'Corners_H_FT', 'stat_type': 'defensive', 'output_prefix': 'Media_EscanteiosSofridos', 'agg_func': np.nanmean, 'window': 10, 'context': 'all'},
    ]
    valid_stats_to_calculate = [cfg for cfg in stats_to_calculate if (cfg.get('base_col_h') and cfg['base_col_h'] in df_processed.columns) or (cfg.get('base_col_a') and cfg['base_col_a'] in df_processed.columns)]
    df_processed = calculate_general_rolling_stats(df_processed, valid_stats_to_calculate, default_window=ROLLING_WINDOW)

    # 4. EWMA Stats (MÉDIAS PONDERADAS)
    ewma_stats_to_calculate = [
        {'base_col_h': 'VG_H_raw', 'base_col_a': 'VG_A_raw', 'output_prefix': 'EWMA_VG', 'span': EWMA_SPAN_SHORT},
        {'base_col_h': 'CG_H_raw', 'base_col_a': 'CG_A_raw', 'output_prefix': 'EWMA_CG', 'span': EWMA_SPAN_SHORT},
        {'base_col_h': GOALS_COLS.get('home'), 'base_col_a': GOALS_COLS.get('away'), 'stat_type': 'offensive', 'output_prefix': 'EWMA_GolsMarc', 'span': EWMA_SPAN_LONG, 'context': 'all'},
        {'base_col_h': GOALS_COLS.get('away'), 'base_col_a': GOALS_COLS.get('home'), 'stat_type': 'defensive', 'output_prefix': 'EWMA_GolsSofr', 'span': EWMA_SPAN_LONG, 'context': 'all'},
    ]
    valid_ewma_stats_to_calculate = [cfg for cfg in ewma_stats_to_calculate if (cfg.get('base_col_h') and cfg['base_col_h'] in df_processed.columns) or (cfg.get('base_col_a') and cfg['base_col_a'] in df_processed.columns)]
    if valid_ewma_stats_to_calculate:
        df_processed = calculate_ewma_stats(df_processed, valid_ewma_stats_to_calculate, default_span=ROLLING_WINDOW) # Use a default span consistent with rolling window maybe? Or the specific EWMA_SPAN
    else:
        logger.warning("Nenhuma configuração EWMA válida para calcular.")


    # 5. FA/FD (agora pode usar médias simples ou EWMA se disponíveis)
    marc_h_fd = EWMA_GolsMarc_H_LONG if EWMA_GolsMarc_H_LONG in df_processed.columns else 'Media_GolsMarcados_H'
    sofr_a_fd = EWMA_GolsSofr_A_LONG if EWMA_GolsSofr_A_LONG in df_processed.columns else 'Media_GolsSofridos_A'
    marc_a_fd = EWMA_GolsMarc_A_LONG if EWMA_GolsMarc_A_LONG in df_processed.columns else 'Media_GolsMarcados_A'
    sofr_h_fd = EWMA_GolsSofr_H_LONG if EWMA_GolsSofr_H_LONG in df_processed.columns else 'Media_GolsSofridos_H'
    if marc_h_fd in df_processed.columns: df_processed['FA_H'] = df_processed[marc_h_fd]/avg_h_league_safe
    else: df_processed['FA_H']=np.nan; logger.warning(f"FA_H not calculated (missing {marc_h_fd})")
    if sofr_a_fd in df_processed.columns: df_processed['FD_A'] = df_processed[sofr_a_fd]/avg_h_league_safe
    else: df_processed['FD_A']=np.nan; logger.warning(f"FD_A not calculated (missing {sofr_a_fd})")
    if marc_a_fd in df_processed.columns: df_processed['FA_A'] = df_processed[marc_a_fd]/avg_a_league_safe
    else: df_processed['FA_A']=np.nan; logger.warning(f"FA_A not calculated (missing {marc_a_fd})")
    if sofr_h_fd in df_processed.columns: df_processed['FD_H'] = df_processed[sofr_h_fd]/avg_a_league_safe
    else: df_processed['FD_H']=np.nan; logger.warning(f"FD_H not calculated (missing {sofr_h_fd})")

    # 6. Poisson, Binned, Derivadas/Interações
    df_processed = calculate_poisson_draw_prob(df_processed, avg_h_league_safe, avg_a_league_safe, max_goals=6)
    df_processed = calculate_binned_features(df_processed)
    df_processed = calculate_derived_features(df_processed)
    logger.info("=== FIM PIPELINE FEAT ENG (HISTÓRICO) ===")

    # --- Seleção Final e Dropna ---
    df_final_all = df_processed; target_col = 'IsDraw';
    final_feature_cols = FEATURE_COLUMNS;
    logger.info(f"Selecionando features FINAIS para modelo: {final_feature_cols}");
    available_features = [f for f in final_feature_cols if f in df_final_all.columns]
    missing_features = [f for f in final_feature_cols if f not in df_final_all.columns]
    if not available_features: logger.error("Erro FINAIS: Nenhuma feature selecionada disponível!"); return None
    if missing_features:
        logger.warning(f"Aviso FINAIS: Features selecionadas ausentes do DataFrame: {missing_features}. Verifique os cálculos.")
        final_feature_cols = available_features # Usa apenas as que existem
        logger.warning(f"Continuando APENAS com as features disponíveis: {final_feature_cols}")
        if not final_feature_cols: logger.error("Nenhuma feature final disponível após remover ausentes."); return None
    required_cols_for_output = final_feature_cols + [target_col]
    if target_col not in df_final_all.columns: logger.error(f"Erro FINAIS: Alvo '{target_col}' ausente."); return None
    df_select = df_final_all[required_cols_for_output].copy()
    logger.info(f"DF selecionado p/ dropna: {df_select.shape}. Verificando NaNs...")
    initial_rows = len(df_select)
    nan_check_before = df_select[final_feature_cols].isnull().sum(); cols_nan = nan_check_before[nan_check_before > 0];
    if not cols_nan.empty: logger.warning(f"NaNs ANTES dropna final nas features selecionadas:\n{cols_nan}")
    else: logger.info("Nenhum NaN antes dropna final nas features selecionadas.");
    df_select.replace([np.inf, -np.inf], np.nan, inplace=True);
    df_select.dropna(subset=required_cols_for_output, inplace=True);
    rows_dropped = initial_rows - len(df_select);
    if rows_dropped > 0: logger.info(f"Removidas {rows_dropped} linhas contendo NaNs/Infs nas features finais ou no alvo.");
    if df_select.empty: logger.error("Erro CRÍTICO: Nenhuma linha restante pós dropna final."); return None;
    X = df_select[final_feature_cols]; y = df_select[target_col].astype(int);
    if X.isnull().values.any(): final_nan_check = X.isnull().sum(); logger.error(f"ERRO CRÍTICO: NaNs detectados em X PÓS dropna! Colunas:\n{final_nan_check[final_nan_check > 0]}"); return None;
    if y.isnull().values.any(): logger.error("ERRO CRÍTICO: NaNs detectados em y PÓS dropna!"); return None;
    logger.info(f"--- Pré-proc Histórico OK --- Shape Final X:{X.shape}, y:{y.shape}.");
    logger.debug(f"Features FINAIS usadas no modelo: {list(X.columns)}");
    return X, y, list(X.columns)

# --- (fetch_and_process_fixtures - Sem alterações necessárias aqui) ---       
def fetch_and_process_fixtures() -> Optional[pd.DataFrame]:
    from config import REQUIRED_FIXTURE_COLS, TARGET_LEAGUES_INTERNAL_IDS, ODDS_COLS, OTHER_ODDS_NAMES, XG_COLS, FIXTURE_FETCH_DAY, FIXTURE_CSV_URL_TEMPLATE, CSV_HIST_COL_MAP
    if FIXTURE_FETCH_DAY == "tomorrow": target_date = date.today() + timedelta(days=1)
    else: target_date = date.today()
    date_str = target_date.strftime('%Y-%m-%d')
    fixture_url = FIXTURE_CSV_URL_TEMPLATE.format(date_str=date_str)
    logger.info(f"Buscando jogos de {FIXTURE_FETCH_DAY} ({date_str}): {fixture_url}")
    df_fix = None
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(fixture_url, headers=headers, timeout=20)
        response.raise_for_status()
        logger.info("Arquivo futuro encontrado. Tentando ler...")
        from io import BytesIO # Use BytesIO for excel
        content_bytes = response.content # Get content as bytes

        try: # Try CSV first
            logger.debug("Tentando ler como CSV...")
            df_fix = pd.read_csv(BytesIO(content_bytes), low_memory=False)
            logger.info("Lido com sucesso como CSV.")
        except Exception as e_csv:
            logger.warning(f"Falha ao ler como CSV ({e_csv}), tentando ler como Excel...")
            try: # Try Excel as fallback
                df_fix = pd.read_excel(BytesIO(content_bytes), engine='openpyxl')
                logger.info("Lido com sucesso como Excel.")
            except Exception as e_excel:
                 logger.error(f"Falha ao ler como CSV e como Excel: {e_excel}", exc_info=True)
                 return None
        if df_fix is None: raise ValueError("Falha ao ler dados do arquivo futuro.")
        logger.info(f"Arquivo futuro baixado e lido. Shape: {df_fix.shape}")
    except requests.exceptions.RequestException as e_req: logger.error(f"Erro HTTP ao buscar arquivo futuro: {e_req}"); return None
    except pd.errors.EmptyDataError: logger.warning(f"Arquivo futuro {fixture_url} vazio."); return pd.DataFrame()
    except Exception as e_load: logger.error(f"Erro baixar/ler arquivo futuro: {e_load}", exc_info=True); return None

    try:
        logger.info("Processando arquivo de jogos futuros..."); logger.debug(f"Colunas lidas: {list(df_fix.columns)}")
        cols_to_map_and_rename = {k: v for k, v in CSV_HIST_COL_MAP.items() if k in df_fix.columns}
        if not cols_to_map_and_rename: logger.error("Nenhuma coluna mapeável encontrada no arquivo futuro."); return None
        df_processed = df_fix[list(cols_to_map_and_rename.keys())].copy(); df_processed.rename(columns=cols_to_map_and_rename, inplace=True)
        logger.debug(f"Colunas após rename: {list(df_processed.columns)}")
        internal_league_col_name = 'League'
        if internal_league_col_name in df_processed.columns:

            # <<< VERIFICA O NOVO FLAG AQUI >>>
            if SCRAPER_FILTER_LEAGUES:
                logger.info(f"Filtrando por ligas alvo: {TARGET_LEAGUES_INTERNAL_IDS}...")
                df_processed[internal_league_col_name] = df_processed[internal_league_col_name].astype(str).str.strip()
                initial_count = len(df_processed)
                # Filtro agora está DENTRO do if
                df_processed = df_processed[df_processed[internal_league_col_name].isin(TARGET_LEAGUES_INTERNAL_IDS)]
                logger.info(f"Filtro de ligas (Futuro): {len(df_processed)}/{initial_count} jogos restantes.")
                if df_processed.empty:
                    logger.info("Nenhum jogo futuro restante após filtro.")
                    return pd.DataFrame()
            else:
                # Mensagem indicando que o filtro está desativado para o arquivo futuro
                logger.info("Filtro de liga DESATIVADO para arquivo de jogos futuros.")

        else:
            logger.warning(f"Coluna '{internal_league_col_name}' não encontrada. Filtro de liga não aplicado.")
        try: current_required_fixture_cols = REQUIRED_FIXTURE_COLS
        except NameError: logger.warning("REQUIRED_FIXTURE_COLS não no config."); current_required_fixture_cols = ['League', 'Home', 'Away', 'Time_Str', 'Odd_H_FT', 'Odd_D_FT', 'Odd_A_FT']
        missing_required = [c for c in current_required_fixture_cols if c not in df_processed.columns]
        if missing_required: logger.error(f"Colunas essenciais ausentes (Futuro): {missing_required}"); return None
        if 'Date' in df_processed.columns: df_processed['Date_Str'] = pd.to_datetime(df_processed['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
        else: logger.warning("Coluna 'Date' ausente para criar 'Date_Str'."); df_processed['Date_Str'] = date_str # Fallback para data alvo
        if 'Time' in df_processed.columns: df_processed['Time_Str'] = df_processed['Time'].astype(str).str.strip()
        else: logger.warning("Coluna 'Time' ausente para criar 'Time_Str'."); df_processed['Time_Str'] = "00:00" # Fallback
        rows_to_drop = df_processed['Date_Str'].isnull() | df_processed['Time_Str'].isnull() | (df_processed['Time_Str'] == '')
        if rows_to_drop.any(): logger.warning(f"Removendo {rows_to_drop.sum()} linhas com Date/Time inválido."); df_processed = df_processed[~rows_to_drop]
        if df_processed.empty: logger.warning("DF futuro vazio após limpeza Date/Time."); return pd.DataFrame()
        numeric_cols_futures = list(ODDS_COLS.values()) + OTHER_ODDS_NAMES + list(XG_COLS.values())
        for col in numeric_cols_futures:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                if col in list(ODDS_COLS.values()) or col in OTHER_ODDS_NAMES: df_processed.loc[df_processed[col] <= 1, col] = np.nan
                if col in list(XG_COLS.values()): df_processed.loc[df_processed[col] < 0, col] = np.nan
        logger.info(f"Verificando NaNs essenciais (Futuro): {current_required_fixture_cols}")
        initial_rows_fix = len(df_processed); cols_to_dropna_present_fix = [c for c in current_required_fixture_cols if c in df_processed.columns]
        df_processed.dropna(subset=cols_to_dropna_present_fix, inplace=True); rows_dropped_fix = initial_rows_fix - len(df_processed)
        if rows_dropped_fix > 0: logger.info(f"Futuro: Removidas {rows_dropped_fix} linhas com NaNs essenciais.")
        if df_processed.empty: logger.info("Nenhum jogo futuro restante após limpeza essencial."); return pd.DataFrame()
        df_processed.reset_index(drop=True, inplace=True)
        logger.info(f"Processamento jogos futuros OK. Shape final: {df_processed.shape}")
        logger.debug(f"Colunas finais jogos futuros: {list(df_processed.columns)}")
        return df_processed
    except Exception as e_proc: logger.error(f"Erro GERAL no processamento futuro: {e_proc}", exc_info=True); return None

def prepare_fixture_data(fixture_df: pd.DataFrame, historical_df: pd.DataFrame, feature_columns: List[str]) -> Optional[pd.DataFrame]:
    """Prepara features para jogos futuros, incluindo rolling, EWMA e interações."""
    if fixture_df is None or historical_df is None or not feature_columns: logger.error("Prep Fix: Args inválidos."); return None;
    if fixture_df.empty: logger.info("Prep Fix: DF jogos futuros vazio."); return pd.DataFrame()
    if historical_df.empty: logger.error("Prep Fix: DF histórico vazio."); return None

    logger.info("--- Preparando Features Futuras ---")
    logger.debug(f"Prepare Fixture: Colunas recebidas em historical_df: {historical_df.columns.tolist()}")
    logger.debug(f"Prepare Fixture: Shape recebido historical_df: {historical_df.shape}")

    # <<< NOVA ETAPA: Calcular features intermediárias no histórico recebido >>>
    logger.info("Calculando/Garantindo features intermediárias no histórico...")
    try:
        # Calcula P(H)/P(D)/P(A) se não existirem
        if not all(p in historical_df.columns for p in ['p_H', 'p_D', 'p_A']):
            historical_df = calculate_probabilities(historical_df)
        # Calcula VG/CG raw etc.
        historical_df = calculate_historical_intermediate(historical_df)
        # Verifica se as colunas base cruciais foram criadas
        required_bases = ['VG_H_raw', 'VG_A_raw', 'CG_H_raw', 'CG_A_raw', 'Home', 'Away', 'Date']
        missing_bases = [col for col in required_bases if col not in historical_df.columns]
        if missing_bases:
            logger.error(f"Erro Prep Fix: Colunas base {missing_bases} ainda ausentes após calculate_historical_intermediate.")
            return None
        logger.info("Features intermediárias OK.")
    except Exception as e_inter:
        logger.error(f"Erro ao calcular features intermediárias para histórico em prepare_fixture_data: {e_inter}", exc_info=True)
        return None
    # <<< FIM NOVA ETAPA >>>


    goals_h_col=GOALS_COLS.get('home'); goals_a_col=GOALS_COLS.get('away');
    avg_h_league = np.nanmean(historical_df[goals_h_col]) if goals_h_col in historical_df and historical_df[goals_h_col].notna().any() else 1.0
    avg_a_league = np.nanmean(historical_df[goals_a_col]) if goals_a_col in historical_df and historical_df[goals_a_col].notna().any() else 1.0
    avg_h_league_safe=max(avg_h_league,FEATURE_EPSILON); avg_a_league_safe=max(avg_a_league,FEATURE_EPSILON);
    logger.info(f"Usando Médias Liga (Hist): Casa={avg_h_league_safe:.3f}, Fora={avg_a_league_safe:.3f}")

    logger.info("Processando histórico p/ estado final rolling/EWMA...")
    needed_rolling_configs = []
    needed_ewma_configs = []
    base_cols_hist_needed = set(['Date', 'Home', 'Away']) # Base inicial

    # --- Define TODAS as configs possíveis ---
    # (Listas all_possible_rolling_configs e all_possible_ewma_configs como antes)
    all_possible_rolling_configs = [
        {'base_col_h': 'VG_H_raw', 'base_col_a': 'VG_A_raw', 'output_prefix': 'Media_VG', 'agg_func': np.nanmean, 'window': ROLLING_WINDOW},
        {'base_col_h': 'CG_H_raw', 'base_col_a': 'CG_A_raw', 'output_prefix': 'Media_CG', 'agg_func': np.nanmean, 'window': ROLLING_WINDOW},
        {'base_col_h': 'CG_H_raw', 'base_col_a': 'CG_A_raw', 'output_prefix': 'Std_CG', 'agg_func': np.nanstd, 'min_periods': 2, 'window': ROLLING_WINDOW},
        {'base_col_h': GOALS_COLS.get('home'), 'base_col_a': GOALS_COLS.get('away'), 'stat_type': 'offensive', 'output_prefix': 'Media_GolsMarcados', 'agg_func': np.nanmean, 'window': ROLLING_WINDOW, 'context': 'all'},
        {'base_col_h': GOALS_COLS.get('away'), 'base_col_a': GOALS_COLS.get('home'), 'stat_type': 'defensive', 'output_prefix': 'Media_GolsSofridos', 'agg_func': np.nanmean, 'window': ROLLING_WINDOW, 'context': 'all'},
        {'base_col_h': 'Shots_H', 'base_col_a': 'Shots_A', 'stat_type': 'offensive', 'output_prefix': 'Media_ChutesTotal', 'agg_func': np.nanmean, 'window': ROLLING_WINDOW, 'context': 'all'},
        {'base_col_h': 'ShotsOnTarget_H', 'base_col_a': 'ShotsOnTarget_A', 'stat_type': 'offensive', 'output_prefix': 'Media_ChutesAlvo', 'agg_func': np.nanmean, 'window': ROLLING_WINDOW, 'context': 'all'},
        {'base_col_h': 'Corners_H_FT', 'base_col_a': 'Corners_A_FT', 'stat_type': 'offensive', 'output_prefix': 'Media_Escanteios', 'agg_func': np.nanmean, 'window': 10, 'context': 'all'},
        {'base_col_h': 'Corners_A_FT', 'base_col_a': 'Corners_H_FT', 'stat_type': 'defensive', 'output_prefix': 'Media_EscanteiosSofridos', 'agg_func': np.nanmean, 'window': 10, 'context': 'all'},
    ]
    all_possible_ewma_configs = [
        {'base_col_h': 'VG_H_raw', 'base_col_a': 'VG_A_raw', 'output_prefix': 'EWMA_VG', 'span': EWMA_SPAN_SHORT},
        {'base_col_h': 'CG_H_raw', 'base_col_a': 'CG_A_raw', 'output_prefix': 'EWMA_CG', 'span': EWMA_SPAN_SHORT},
        {'base_col_h': GOALS_COLS.get('home'), 'base_col_a': GOALS_COLS.get('away'), 'stat_type': 'offensive', 'output_prefix': 'EWMA_GolsMarc', 'span': EWMA_SPAN_LONG, 'context': 'all'},
        {'base_col_h': GOALS_COLS.get('away'), 'base_col_a': GOALS_COLS.get('home'), 'stat_type': 'defensive', 'output_prefix': 'EWMA_GolsSofr', 'span': EWMA_SPAN_LONG, 'context': 'all'},
    ]
    # --- Fim Definição Configs ---

    final_output_cols = set(feature_columns)

    # Filtra configs ROLLING necessárias
    for cfg in all_possible_rolling_configs:
        # ... (lógica de filtro como antes, adicionando a base_cols_hist_needed) ...
        prefix=cfg['output_prefix']; context=cfg.get('context','all'); output_h=f"{prefix}_{context}_H" if context!='all' else f"{prefix}_H"; output_a=f"{prefix}_{context}_A" if context!='all' else f"{prefix}_A"; is_needed=False
        if output_h in final_output_cols or output_a in final_output_cols: is_needed=True
        elif prefix=='Media_GolsMarcados' and any(f in final_output_cols for f in ['FA_H','FA_A','Prob_Empate_Poisson', INTERACTION_AVG_GOLS_MARC_DIFF]): is_needed=True
        elif prefix=='Media_GolsSofridos' and any(f in final_output_cols for f in ['FD_H','FD_A','Prob_Empate_Poisson', INTERACTION_AVG_GOLS_SOFR_DIFF]): is_needed=True
        elif prefix=='Media_CG' and 'Diff_Media_CG' in final_output_cols: is_needed=True
        if is_needed:
            base_h = cfg.get('base_col_h'); base_a = cfg.get('base_col_a')
            if (base_h and base_h in historical_df.columns) or (base_a and base_a in historical_df.columns): # Check existence in the (now intermediate-calculated) historical_df
                 needed_rolling_configs.append(cfg);
                 if base_h and base_h in historical_df.columns: base_cols_hist_needed.add(base_h)
                 if base_a and base_a in historical_df.columns: base_cols_hist_needed.add(base_a)
                 # Dependencies for VG/CG/StdCG
                 if prefix in ['Media_VG','Media_CG','Std_CG']: base_cols_hist_needed.update([c for c in list(GOALS_COLS.values())+list(ODDS_COLS.values())+['p_H','p_A','VG_H_raw','VG_A_raw','CG_H_raw','CG_A_raw'] if c in historical_df.columns])
            else: logger.warning(f"Rolling Config '{prefix}' ignorada p/ futuro (cols base '{base_h}'/'{base_a}' ausentes no histórico).")

    # Filtra configs EWMA necessárias
    for cfg in all_possible_ewma_configs:
        prefix=cfg['output_prefix']; context=cfg.get('context','all'); span=cfg.get('span', ROLLING_WINDOW); output_h = f"{prefix}_s{span}_{context}_H" if context != 'all' else f"{prefix}_s{span}_H"; output_a = f"{prefix}_s{span}_{context}_A" if context != 'all' else f"{prefix}_s{span}_A"; is_needed = False
        if output_h in final_output_cols or output_a in final_output_cols: is_needed=True
        elif prefix=='EWMA_GolsMarc' and any(f in final_output_cols for f in ['FA_H','FA_A','Prob_Empate_Poisson', INTERACTION_AVG_GOLS_MARC_DIFF]): is_needed=True
        elif prefix=='EWMA_GolsSofr' and any(f in final_output_cols for f in ['FD_H','FD_A','Prob_Empate_Poisson', INTERACTION_AVG_GOLS_SOFR_DIFF]): is_needed=True
        if is_needed:
            base_h = cfg.get('base_col_h'); base_a = cfg.get('base_col_a')
            if (base_h and base_h in historical_df.columns) or (base_a and base_a in historical_df.columns):
                 cfg_with_span = cfg.copy(); cfg_with_span['span'] = span; needed_ewma_configs.append(cfg_with_span)
                 if base_h and base_h in historical_df.columns: base_cols_hist_needed.add(base_h)
                 if base_a and base_a in historical_df.columns: base_cols_hist_needed.add(base_a)
                 if prefix in ['EWMA_VG','EWMA_CG']: base_cols_hist_needed.update([c for c in list(GOALS_COLS.values())+list(ODDS_COLS.values())+['p_H','p_A','VG_H_raw','VG_A_raw','CG_H_raw','CG_A_raw'] if c in historical_df.columns])
            else: logger.warning(f"EWMA Config '{prefix}' ignorada p/ futuro (cols base '{base_h}'/'{base_a}' ausentes no histórico).")
    # --- Fim Filtragem Configs ---

    logger.info(f"Configs rolling necessárias p/ futuro: {[c['output_prefix'] for c in needed_rolling_configs]}")
    logger.info(f"Configs EWMA necessárias p/ futuro: {[c['output_prefix'] for c in needed_ewma_configs]}")

    hist_cols_present = [c for c in base_cols_hist_needed if c in historical_df.columns]
    logger.debug(f"Cols históricas p/ estado final: {hist_cols_present}")
    if not all(c in historical_df.columns for c in ['Date','Home','Away']):
        missing_essentials = [c for c in ['Date','Home','Away'] if c not in historical_df.columns]
        logger.error(f"Erro prep futuro: Cols essenciais {missing_essentials} ainda ausentes APÓS cálculo intermediate."); return None
    historical_df_minimal = historical_df[hist_cols_present].copy()
    historical_df_minimal = historical_df_minimal.sort_values(by='Date', ascending=True)
    logger.info("Construindo estado final histórico rolling/EWMA..."); start_hist = time.time()
    history_maxlen = len(historical_df_minimal) + 5; team_history_final_state = defaultdict(lambda: defaultdict(lambda: deque(maxlen=history_maxlen)))
    all_needed_configs = needed_rolling_configs + needed_ewma_configs
    for _, row in historical_df_minimal.iterrows():
         ht=row.get('Home'); at=row.get('Away');
         if pd.isna(ht) or pd.isna(at): continue
         for cfg in all_needed_configs:
             base_col_h = cfg.get('base_col_h'); base_col_a = cfg.get('base_col_a');
             val_h = row.get(base_col_h) if base_col_h else None; val_a = row.get(base_col_a) if base_col_a else None
             if base_col_h and pd.notna(val_h): team_history_final_state[ht][base_col_h].append(val_h)
             if base_col_a and pd.notna(val_a): team_history_final_state[at][base_col_a].append(val_a)
    logger.info(f"Estado final construído ({(time.time()-start_hist):.2f}s) p/ {len(team_history_final_state)} times.")
    logger.info("Calculando features rolling/EWMA para jogos futuros...")
    fixture_features_list = []; default_window = ROLLING_WINDOW; 
    for index, future_match in tqdm(fixture_df.iterrows(), total=len(fixture_df), desc="Calc. Rolling/EWMA Futuro"):
        ht = future_match.get('Home'); at = future_match.get('Away'); current_match_features = {'Index': index}
        for team_type, team_name in [('H', ht), ('A', at)]:
            if pd.isna(team_name):
                for cfg_inner in all_needed_configs:
                    prefix_inner=cfg_inner['output_prefix']; context_inner=cfg_inner.get('context','all');
                    if 'span' in cfg_inner: span_inner = cfg_inner['span']; output_col_inner = f"{prefix_inner}_s{span_inner}_{context_inner}_H" if context_inner!='all' and team_type=='H' else f"{prefix_inner}_s{span_inner}_{context_inner}_A" if context_inner!='all' and team_type=='A' else f"{prefix_inner}_s{span_inner}_{team_type}"
                    else: output_col_inner = f"{prefix_inner}_{context_inner}_H" if context_inner!='all' and team_type=='H' else f"{prefix_inner}_{context_inner}_A" if context_inner!='all' and team_type=='A' else f"{prefix_inner}_{team_type}"
                    current_match_features[output_col_inner] = np.nan
                continue
            team_hist_state = team_history_final_state[team_name] # Acesso direto

            # Calcula ROLLING Stats
            for cfg in needed_rolling_configs:
                # --- Get config details ---
                prefix = cfg['output_prefix']
                agg_func = cfg['agg_func']
                window = cfg.get('window', default_window)
                min_p = cfg.get('min_periods', 1 if agg_func != np.nanstd else 2)
                base_h = cfg.get('base_col_h')
                base_a = cfg.get('base_col_a')
                context = cfg.get('context', 'all')
                stat_type = cfg.get('stat_type', 'offensive')

                # <<< --- FIX: Define output_col FIRST --- >>>
                # Determine the correct output column name based on team_type ('H' or 'A')
                if context != 'all':
                    output_col = f"{prefix}_{context}_{team_type}" # e.g., Media_GolsMarcados_home_H
                else:
                    output_col = f"{prefix}_{team_type}" # e.g., Media_GolsMarcados_H

                # --- Get historical values for the current team ---
                hist_values = []
                key_home = base_h if stat_type == 'offensive' else base_a
                key_away = base_a if stat_type == 'offensive' else base_h

                # --- Select history based on context (using team_hist_state) ---
                # This logic selects the correct list(s) of historical values to use
                if context == 'all':
                    if key_home and key_home in team_hist_state:
                        hist_values.extend(list(team_hist_state.get(key_home, deque())))
                    if key_away and key_away in team_hist_state:
                        hist_values.extend(list(team_hist_state.get(key_away, deque())))
                elif context == 'home':
                    col_to_use = key_home
                    if col_to_use and col_to_use in team_hist_state:
                        hist_values = list(team_hist_state.get(col_to_use, deque()))
                elif context == 'away':
                    col_to_use = key_away
                    if col_to_use and col_to_use in team_hist_state:
                        hist_values = list(team_hist_state.get(col_to_use, deque()))
                # --- End history selection ---

                # --- Calculate the rolling stat ---
                recent_values = hist_values[-window:] if hist_values else []
                if len(recent_values) >= min_p:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            calculated_value = agg_func(recent_values)
                        # Assign the result using the correctly defined output_col
                        current_match_features[output_col] = calculated_value if pd.notna(calculated_value) and np.isfinite(calculated_value) else np.nan
                    except Exception as e_agg:
                        # Log the error using the correctly defined output_col
                        logger.warning(f"Erro agg {agg_func.__name__} p/ {output_col} (Futuro): {e_agg}")
                        current_match_features[output_col] = np.nan
                else:
                    # Assign NaN if not enough periods, using the correctly defined output_col
                    current_match_features[output_col] = np.nan
            for cfg in needed_ewma_configs:
                prefix=cfg['output_prefix']; span=cfg['span']; min_p=cfg.get('min_periods', 1); base_h=cfg.get('base_col_h'); base_a=cfg.get('base_col_a'); context=cfg.get('context','all'); stat_type=cfg.get('stat_type','offensive');
                output_col = f"{prefix}_s{span}_{context}_H" if context != 'all' and team_type=='H' else f"{prefix}_s{span}_{context}_A" if context != 'all' and team_type=='A' else f"{prefix}_s{span}_{team_type}"
                hist_values = []
                key_home = base_h if stat_type == 'offensive' else base_a
                key_away = base_a if stat_type == 'offensive' else base_h
                if context == 'all':
                    if key_home and key_home in team_hist_state:
                        hist_values.extend(list(team_hist_state.get(key_home, deque())))
                    if key_away and key_away in team_hist_state:
                        hist_values.extend(list(team_hist_state.get(key_away, deque())))
                elif context == 'home':
                    col_to_use = key_home
                    if col_to_use and col_to_use in team_hist_state:
                        hist_values = list(team_hist_state.get(col_to_use, deque()))
                elif context == 'away':
                    col_to_use = key_away
                    if col_to_use and col_to_use in team_hist_state:
                        hist_values = list(team_hist_state.get(col_to_use, deque()))
                if len(hist_values) >= min_p:
                     try:
                         hist_series = pd.Series(hist_values); ewma_series = hist_series.ewm(span=span, adjust=True, min_periods=min_p).mean()
                         calculated_value = ewma_series.iloc[-1] if not ewma_series.empty else np.nan
                         current_match_features[output_col] = calculated_value if pd.notna(calculated_value) and np.isfinite(calculated_value) else np.nan
                     except Exception as e_ewm_calc: logger.error(f"Erro calcular EWMA {output_col} p/ {team_name}: {e_ewm_calc}"); current_match_features[output_col] = np.nan
                else: current_match_features[output_col] = np.nan
        fixture_features_list.append(current_match_features)
    if not fixture_features_list: # Handle case where no features calculated
        logger.warning("Prepare Fixture: Nenhuma feature rolling/EWMA calculada.")
        output_cols_to_add = set(); # Define empty set
        for cfg in all_needed_configs: # Generate expected column names
             prefix=cfg['output_prefix']; context=cfg.get('context','all');
             if 'span' in cfg: span=cfg['span']; output_cols_to_add.add(f"{prefix}_s{span}_{context}_H" if context!='all' else f"{prefix}_s{span}_H"); output_cols_to_add.add(f"{prefix}_s{span}_{context}_A" if context!='all' else f"{prefix}_s{span}_A")
             else: output_cols_to_add.add(f"{prefix}_{context}_H" if context!='all' else f"{prefix}_H"); output_cols_to_add.add(f"{prefix}_{context}_A" if context!='all' else f"{prefix}_A")
        for col in output_cols_to_add: # Add missing columns as NaN
             if col not in fixture_df.columns: fixture_df[col] = np.nan
        df_temp_fixtures = fixture_df
    else:
        df_rolling_ewma_features = pd.DataFrame(fixture_features_list).set_index('Index'); logger.info(f"Rolling/EWMA stats p/ futuro calculadas. Shape: {df_rolling_ewma_features.shape}.")
        df_temp_fixtures = fixture_df.join(df_rolling_ewma_features, how='left')
    logger.info("Calculando Pi-Ratings atuais...");
    try:
        final_ratings=_get_final_pi_ratings(historical_df.sort_values(by='Date')); df_temp_fixtures['PiRating_H']=df_temp_fixtures['Home'].map(final_ratings).fillna(PI_RATING_INITIAL); df_temp_fixtures['PiRating_A']=df_temp_fixtures['Away'].map(final_ratings).fillna(PI_RATING_INITIAL); df_temp_fixtures['PiRating_Diff']=df_temp_fixtures['PiRating_H']-df_temp_fixtures['PiRating_A'];
        logger.warning("PiRating Momentum (Futuro) definido como NaN."); df_temp_fixtures[PIRATING_MOMENTUM_H]=np.nan; df_temp_fixtures[PIRATING_MOMENTUM_A]=np.nan; df_temp_fixtures[PIRATING_MOMENTUM_DIFF]=np.nan
        rating_h_adj_fix=df_temp_fixtures['PiRating_H']+PI_RATING_HOME_ADVANTAGE; rating_diff_fix=rating_h_adj_fix-df_temp_fixtures['PiRating_A']; df_temp_fixtures['PiRating_Prob_H']=1/(1+10**(-rating_diff_fix/400));
    except Exception as e_pi: logger.error(f"Erro PiRatings futuro: {e_pi}", exc_info=True)
    logger.info("Calculando FA/FD, Poisson, Derivadas/Interações futuras...")
    marc_h_fd = EWMA_GolsMarc_H_LONG if EWMA_GolsMarc_H_LONG in df_temp_fixtures.columns else 'Media_GolsMarcados_H'; sofr_a_fd = EWMA_GolsSofr_A_LONG if EWMA_GolsSofr_A_LONG in df_temp_fixtures.columns else 'Media_GolsSofridos_A'
    marc_a_fd = EWMA_GolsMarc_A_LONG if EWMA_GolsMarc_A_LONG in df_temp_fixtures.columns else 'Media_GolsMarcados_A'; sofr_h_fd = EWMA_GolsSofr_H_LONG if EWMA_GolsSofr_H_LONG in df_temp_fixtures.columns else 'Media_GolsSofridos_H'
    if marc_h_fd in df_temp_fixtures.columns: df_temp_fixtures['FA_H']=df_temp_fixtures[marc_h_fd]/avg_h_league_safe
    else: df_temp_fixtures['FA_H']=np.nan; logger.warning(f"FA_H futuro não calculado (faltando {marc_h_fd})")
    if sofr_a_fd in df_temp_fixtures.columns: df_temp_fixtures['FD_A']=df_temp_fixtures[sofr_a_fd]/avg_h_league_safe
    else: df_temp_fixtures['FD_A']=np.nan; logger.warning(f"FD_A futuro não calculado (faltando {sofr_a_fd})")
    if marc_a_fd in df_temp_fixtures.columns: df_temp_fixtures['FA_A']=df_temp_fixtures[marc_a_fd]/avg_a_league_safe
    else: df_temp_fixtures['FA_A']=np.nan; logger.warning(f"FA_A futuro não calculado (faltando {marc_a_fd})")
    if sofr_h_fd in df_temp_fixtures.columns: df_temp_fixtures['FD_H']=df_temp_fixtures[sofr_h_fd]/avg_a_league_safe
    else: df_temp_fixtures['FD_H']=np.nan; logger.warning(f"FD_H futuro não calculado (faltando {sofr_h_fd})")
    df_temp_fixtures=calculate_probabilities(df_temp_fixtures); df_temp_fixtures=calculate_normalized_probabilities(df_temp_fixtures);
    df_temp_fixtures=calculate_binned_features(df_temp_fixtures); df_temp_fixtures=calculate_poisson_draw_prob(df_temp_fixtures,avg_h_league_safe,avg_a_league_safe,max_goals=6);
    df_temp_fixtures=calculate_derived_features(df_temp_fixtures);
    logger.info(f"Selecionando features FINAIS ({feature_columns}) para predição...")
    final_features_expected = feature_columns; final_features_available = [f for f in final_features_expected if f in df_temp_fixtures.columns]; missing_final = [f for f in final_features_expected if f not in final_features_available]
    if not final_features_available: logger.error("Erro FINAIS (Futuro): Nenhuma feature esperada!"); return None
    if missing_final: logger.warning(f"Aviso FINAIS (Futuro): Features ausentes: {missing_final}. Preenchendo com 0."); [df_temp_fixtures.update({col: 0}) for col in missing_final] # Use update or direct assignment
    X_fix_prep = df_temp_fixtures[final_features_expected].copy(); logger.info(f"Shape X_fix_prep: {X_fix_prep.shape}")
    nan_counts = X_fix_prep.isnull().sum(); total_nans = nan_counts.sum();
    if total_nans > 0: logger.warning(f"{total_nans} NaNs restantes. Preenchendo com 0."); X_fix_prep.fillna(0, inplace=True);
    else: logger.info("Nenhum NaN restante.")
    if X_fix_prep.isin([np.inf, -np.inf]).values.any(): logger.warning("Infinitos encontrados. Substituindo por 0."); X_fix_prep.replace([np.inf, -np.inf], 0, inplace=True)
    if X_fix_prep.isnull().values.any(): logger.error("ERRO CRÍTICO (Futuro): NaNs ainda presentes!"); return None
    if not fixture_df.index.equals(X_fix_prep.index):
        logger.warning("Prepare Fixture: Índices divergentes! Reindexando...");
        try: X_fix_prep = X_fix_prep.reindex(fixture_df.index, fill_value=0);
        except Exception as e_reindex: logger.error(f"Falha CRÍTICA ao realinhar índices: {e_reindex}"); return None
    logger.info(f"--- Preparação Features Futuras OK. Shape Final Retornado: {X_fix_prep.shape} ---")
    return X_fix_prep

    # --- Função auxiliar interna _get_final_pi_ratings ---
def _get_final_pi_ratings(df_hist: pd.DataFrame) -> Dict[str, float]:
    """Calculates the final Pi-Ratings based on the entire historical dataset."""
    ratings = {}; goals_h_c=GOALS_COLS.get('home'); goals_a_c=GOALS_COLS.get('away')
    # Ensure required columns exist and are numeric
    if goals_h_c not in df_hist.columns or goals_a_c not in df_hist.columns:
        logger.error(f"Colunas de Gols ('{goals_h_c}', '{goals_a_c}') ausentes para cálculo de Pi-Rating final.")
        return {}
    # Convert to numeric, coercing errors. Drop rows where conversion failed for essential columns.
    df_hist = df_hist.copy() # Avoid SettingWithCopyWarning
    df_hist[goals_h_c] = pd.to_numeric(df_hist[goals_h_c], errors='coerce')
    df_hist[goals_a_c] = pd.to_numeric(df_hist[goals_a_c], errors='coerce')
    df_hist.dropna(subset=[goals_h_c, goals_a_c, 'Home', 'Away'], inplace=True)

    logger.debug(f"Calculando ratings finais em {len(df_hist)} jogos válidos.")
    # Sort again just to be sure, although prepare_fixture_data should have done it
    df_hist = df_hist.sort_values(by='Date')

    for _, row in df_hist.iterrows():
         ht=row['Home']; at=row['Away'];
         # Names should be valid strings after dropna above
         rh=ratings.get(ht, PI_RATING_INITIAL); ra=ratings.get(at, PI_RATING_INITIAL);
         rh_adj=rh+PI_RATING_HOME_ADVANTAGE; rd=rh_adj-ra; exh=1/(1+10**(-rd/400));
         gh=row[goals_h_c]; ga=row[goals_a_c];
         # Goals are guaranteed to be non-NaN here due to dropna
         score_h = 1.0 if gh > ga else (0.5 if gh == ga else 0.0)
         # Update ratings
         nrh = rh + PI_RATING_K_FACTOR*(score_h-exh);
         nra = ra + PI_RATING_K_FACTOR*((1-score_h)-(1-exh));
         ratings[ht]=nrh; ratings[at]=nra
    logger.debug(f"Cálculo de ratings finais concluído. {len(ratings)} times com rating.")
    return ratings