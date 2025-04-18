# --- src/data_handler.py ---
import pandas as pd
import numpy as np
from config import (
    FEATURE_COLUMNS,
    ODDS_COLS, GOALS_COLS, ROLLING_WINDOW,
    FIXTURE_FETCH_DAY, FIXTURE_CSV_URL_TEMPLATE,
    REQUIRED_FIXTURE_COLS, TARGET_LEAGUES, CSV_HIST_COL_MAP,
    HISTORICAL_DATA_PATH_1, HISTORICAL_DATA_PATH_2,
    OTHER_ODDS_NAMES, XG_COLS
    
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

logger = setup_logger('DataHandler')

def roi(y_test: pd.Series, y_pred: np.ndarray, X_test_odds_aligned: pd.DataFrame, odd_draw_col_name: str) -> Optional[float]:
    if X_test_odds_aligned is None:
        return None
    try:
        predicted_draws_indices = y_test.index[y_pred == 1]
        num_bets = len(predicted_draws_indices)
        if num_bets == 0:
            return 0.0 # Return 0.0 for ROI if no bets

        # Ensure alignment before accessing odds
        common_index = y_test.index.intersection(X_test_odds_aligned.index)
        y_test_common = y_test.loc[common_index]
        # Align y_pred with the common index if necessary
        if isinstance(y_pred, np.ndarray):
             y_pred_series = pd.Series(y_pred, index=y_test.index) # Assume original index alignment
             y_pred_common = y_pred_series.loc[common_index]
        elif isinstance(y_pred, pd.Series):
             y_pred_common = y_pred.loc[common_index]
        else:
             logger.error("ROI Error: y_pred type not supported for alignment.")
             return None

        # Find indices for predicted draws within the common subset
        predicted_draws_indices_common = common_index[y_pred_common == 1]
        num_bets_common = len(predicted_draws_indices_common)

        if num_bets_common == 0:
             return 0.0

        actuals = y_test_common.loc[predicted_draws_indices_common]
        # Use the common index to get odds
        odds = pd.to_numeric(X_test_odds_aligned.loc[predicted_draws_indices_common, odd_draw_col_name], errors='coerce')

        profit = 0
        valid_bets = 0
        for idx in predicted_draws_indices_common:
            odd_d = odds.loc[idx]
            if pd.notna(odd_d) and odd_d > 1: # Check odd > 1
                profit += (odd_d - 1) if actuals.loc[idx] == 1 else -1
                valid_bets += 1
            # Optional: Log invalid odds?
            # elif pd.notna(odd_d):
            #    logger.debug(f"ROI Calc: Invalid odd {odd_d} skipped for index {idx}")

        if valid_bets == 0:
            return 0.0 # Return 0.0 ROI if no valid odds were found for bets made

        return (profit / valid_bets) * 100

    except KeyError as e:
        logger.error(f"ROI Calc Error: KeyError - Column '{e}' likely missing or index mismatch.")
        logger.debug(f"y_test index: {y_test.index[:5]}... | X_test_odds_aligned index: {X_test_odds_aligned.index[:5]}... | pred_indices: {predicted_draws_indices[:5]}...")
        return None
    except Exception as e:
        logger.error(f"ROI Calc Error: An unexpected error occurred: {e}", exc_info=True)
        return None
    
# Função load_historical_data
def load_historical_data() -> Optional[pd.DataFrame]:
    """
    Carrega dados históricos de múltiplos arquivos CSV, une, filtra colunas/ligas,
    renomeia, converte tipos e trata NaNs/zeros inválidos (incluindo XG=0).
    """
    all_dfs: List[pd.DataFrame] = []
    file_paths: List[str] = [HISTORICAL_DATA_PATH_1, HISTORICAL_DATA_PATH_2]
    logger.info(f"Iniciando carregamento de dados históricos de: {file_paths}")

    # 1. Leitura
    expected_raw_cols = list(CSV_HIST_COL_MAP.keys())
    logger.info(f"Colunas brutas esperadas (baseado no map): {expected_raw_cols}")

    for file_path in file_paths:
        if not os.path.exists(file_path):
            logger.error(f"Arquivo histórico não encontrado: {file_path}")
            continue

        try:
            logger.info(f"Lendo {os.path.basename(file_path)}...")
            try:
                 df_part = pd.read_csv(file_path, low_memory=False, encoding='ISO-8859-1')
            except UnicodeDecodeError:
                 logger.warning(f"Falha ao ler {os.path.basename(file_path)} com ISO-8859-1. Tentando UTF-8...")
                 try:
                     df_part = pd.read_csv(file_path, low_memory=False, encoding='utf-8')
                 except Exception as e_enc:
                      logger.error(f"Falha ao ler {os.path.basename(file_path)} com UTF-8 também: {e_enc}")
                      continue

            logger.info(f"  -> Lido {df_part.shape[0]} linhas. Colunas no CSV: {list(df_part.columns)}")
            cols_found = [col for col in expected_raw_cols if col in df_part.columns]
            missing_expected = [col for col in expected_raw_cols if col not in df_part.columns]

            if not cols_found:
                logger.error(f"  -> Nenhuma coluna esperada (do CSV_HIST_COL_MAP) encontrada em {os.path.basename(file_path)}. Pulando.")
                continue
            if missing_expected:
                logger.info(f"  -> Colunas esperadas ausentes neste CSV: {missing_expected}")

            all_dfs.append(df_part[cols_found].copy())

        except Exception as e:
            logger.error(f"Erro ao ler/processar {os.path.basename(file_path)}: {e}", exc_info=True)
            continue

    if not all_dfs:
        logger.error("Nenhum CSV histórico carregado com sucesso.")
        return None

    # 2. Concatenação
    logger.info(f"Concatenando {len(all_dfs)} DFs...")
    try:
        df = pd.concat(all_dfs, ignore_index=True, sort=False)
        logger.info(f"DF combinado inicial: {df.shape}")
    except Exception as e:
        logger.error(f"Erro ao concatenar DFs: {e}", exc_info=True)
        return None

    # 3. Mapeamento e Seleção
    logger.info("Renomeando colunas para nomes internos...")
    valid_map = {k: v for k, v in CSV_HIST_COL_MAP.items() if k in df.columns}
    if not valid_map:
        logger.error(f"Nenhuma coluna do CSV_HIST_COL_MAP encontrada no DF concatenado! Cols DF: {list(df.columns)}")
        return None
    df.rename(columns=valid_map, inplace=True)
    internal_cols_to_keep = list(valid_map.values())
    df = df[internal_cols_to_keep].copy()
    logger.info(f"DF após seleção de colunas internas: {df.shape}")
    logger.debug(f"Colunas internas presentes: {list(df.columns)}")

    # 4. Filtro de Liga
    if TARGET_LEAGUES:
        if 'League' in df.columns:
            logger.info(f"Filtrando para as ligas: {list(TARGET_LEAGUES.values())}...")
            df['League'] = df['League'].astype(str).str.strip()
            initial_count = len(df)
            df = df[df['League'].isin(TARGET_LEAGUES.values())]
            logger.info(f"Filtro Liga: {len(df)}/{initial_count} jogos restantes.")
            # No longer returning empty DF here, let subsequent steps handle it
        else:
            logger.warning("Coluna 'League' ausente para filtro de liga.")
    else:
        logger.info("Sem filtro de liga.")

    # 5. Conversão e Tratamento de Zeros/NaNs
    logger.info("Convertendo tipos e tratando NaNs/Zeros inválidos...")
    epsilon = 1e-6 # Small number to avoid division by zero

    # --- Conversão de Data ---
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        date_nan_before = df['Date'].isnull().sum()
        df.dropna(subset=['Date'], inplace=True)
        date_nan_after = date_nan_before - df['Date'].isnull().sum()
        if date_nan_after > 0: logger.info(f"Removidas {date_nan_after} linhas com Datas inválidas.")
        if df.empty: logger.error("Nenhuma linha após remover datas inválidas."); return None;
    else:
        logger.error("Coluna 'Date' interna ausente. Não é possível ordenar/processar.")
        return None

    # --- Colunas Numéricas Essenciais (Gols) ---
    essential_numeric_cols = list(GOALS_COLS.values())
    for col in essential_numeric_cols:
        if col not in df.columns:
            logger.error(f"Erro CRÍTICO: Coluna essencial interna '{col}' ausente APÓS rename!")
            return None
        if not pd.api.types.is_numeric_dtype(df[col]):
             nan_before = df[col].isnull().sum()
             df[col] = pd.to_numeric(df[col], errors='coerce')
             nan_after = df[col].isnull().sum()
             coerced_count = nan_after - nan_before
             if coerced_count > 0: logger.info(f"Coluna '{col}': {coerced_count} valores não numéricos convertidos para NaN.")

    # --- Colunas Numéricas de Odds (1x2 + Outras) ---
    all_odds_internal_names = list(ODDS_COLS.values()) + OTHER_ODDS_NAMES
    for col in all_odds_internal_names:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                 nan_before = df[col].isnull().sum()
                 df[col] = pd.to_numeric(df[col], errors='coerce')
                 nan_after = df[col].isnull().sum()
                 coerced_count = nan_after - nan_before
                 if coerced_count > 0: logger.info(f"Coluna '{col}': {coerced_count} valores não numéricos convertidos para NaN.")

            invalid_odds_count = (df[col] <= 1).sum()
            if invalid_odds_count > 0:
                df.loc[df[col] <= 1, col] = np.nan
                logger.info(f"Coluna '{col}': {invalid_odds_count} odds <= 1 convertidas para NaN.")
        # else: logger.debug(f"Coluna de odd opcional '{col}' não encontrada no DF.")

    # --- Colunas Numéricas Opcionais (xG) ---
    xg_internal_cols = list(XG_COLS.values())
    for col in xg_internal_cols:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                 nan_before = df[col].isnull().sum()
                 df[col] = pd.to_numeric(df[col], errors='coerce')
                 nan_after = df[col].isnull().sum()
                 coerced_count = nan_after - nan_before
                 if coerced_count > 0: logger.info(f"Coluna '{col}': {coerced_count} valores não numéricos convertidos para NaN.")

            # *** NOVO: Converte XG == 0 para NaN ***
            zero_xg = (df[col] == 0).sum()
            if zero_xg > 0:
                df.loc[df[col] == 0, col] = np.nan
                logger.info(f"Coluna '{col}': {zero_xg} XG == 0 convertidos para NaN.")
        # else: logger.debug(f"Coluna xG opcional '{col}' não encontrada no DF.")

    # 6. Dropna Final para colunas essenciais
    essential_dropna_cols = ['Date', 'Home', 'Away'] + list(GOALS_COLS.values()) + list(ODDS_COLS.values())
    cols_to_dropna_present = [c for c in essential_dropna_cols if c in df.columns]
    logger.info(f"Verificando NaNs nas colunas essenciais: {cols_to_dropna_present}")
    initial_rows = len(df)
    df.dropna(subset=cols_to_dropna_present, inplace=True)
    rows_dropped = initial_rows - len(df)
    if rows_dropped > 0: logger.info(f"Removidas {rows_dropped} linhas com NaNs em colunas essenciais.")
    if df.empty: logger.error("Nenhum jogo restante após dropna essencial."); return None

    # 7. Ordenar por Data
    df = df.sort_values(by='Date').reset_index(drop=True)
    logger.info("DF histórico ordenado por Data.")

    # Log final
    final_cols = list(df.columns)
    logger.info(f"Carregamento e tratamento inicial OK. Shape Final: {df.shape}")
    logger.debug(f"Colunas finais no DF histórico: {final_cols}")
    optional_cols_with_nan = df.isnull().sum()
    optional_cols_with_nan = optional_cols_with_nan[optional_cols_with_nan > 0]
    if not optional_cols_with_nan.empty:
        logger.info(f"Contagem de NaNs restantes em colunas opcionais:\n{optional_cols_with_nan}")
    else:
        logger.info("Nenhum NaN restante detectado após carregamento inicial.")


    # --- Adiciona verificação de tipos numéricos ---
    numeric_cols_check = df.select_dtypes(include=np.number).columns.tolist()
    logger.debug(f"Colunas detectadas como numéricas após carregamento: {numeric_cols_check}")
    non_numeric_cols = list(set(df.columns) - set(numeric_cols_check) - {'Date', 'Home', 'Away', 'League', 'FT_Result'}) # Exclui colunas sabidamente não numéricas
    if non_numeric_cols:
         logger.warning(f"Colunas que podem não ser numéricas após carregamento: {non_numeric_cols}")
         # Opcional: Logar os tipos reais dessas colunas
         # logger.warning(f"Tipos dessas colunas:\n{df[non_numeric_cols].dtypes}")


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
    for stat_prefix in stats_to_calc:
        std_col_h = f'Std_{stat_prefix}_H'
        std_col_a = f'Std_{stat_prefix}_A'
        skip_h = skip_a = False
        if std_col_h in df_calc.columns and pd.api.types.is_numeric_dtype(df_calc[std_col_h]):
            skip_h = True
        if std_col_a in df_calc.columns and pd.api.types.is_numeric_dtype(df_calc[std_col_a]):
            skip_a = True

        if skip_h and skip_a:
            logger.warning(f"{std_col_h}/{std_col_a} já existem.")
            continue

        if stat_prefix == 'Ptos':
            base_h, base_a = 'Ptos_H', 'Ptos_A'
        elif stat_prefix == 'VG':
            base_h, base_a = 'VG_H_raw', 'VG_A_raw'
        elif stat_prefix == 'CG':
            base_h, base_a = 'CG_H_raw', 'CG_A_raw'
        else:
            logger.warning(f"Prefixo StDev '{stat_prefix}' desconhecido.")
            continue

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

    calculated_stats = []
    for index, row in tqdm(df_calc.iterrows(), total=len(df_calc), desc="Calc. Rolling StDev"):
        home_team = row['Home']
        away_team = row['Away']
        current_match_features = {'Index': index}

        for stat_prefix, base_cols in rolling_cols_map.items():
            std_col_h = f'Std_{stat_prefix}_H'
            if stat_prefix + '_H' in cols_to_calculate:
                hist_H = team_history[home_team][stat_prefix]
                recent = hist_H[-window:]
                current_match_features[std_col_h] = np.std(recent) if len(recent) >= 2 else np.nan

        for stat_prefix, base_cols in rolling_cols_map.items():
            std_col_a = f'Std_{stat_prefix}_A'
            if stat_prefix + '_A' in cols_to_calculate:
                hist_A = team_history[away_team][stat_prefix]
                recent = hist_A[-window:]
                current_match_features[std_col_a] = np.std(recent) if len(recent) >= 2 else np.nan

        calculated_stats.append(current_match_features)

        for stat_prefix, base_cols in rolling_cols_map.items():
            if pd.notna(row[base_cols['home']]):
                team_history[home_team][stat_prefix].append(row[base_cols['home']])
            if pd.notna(row[base_cols['away']]):
                team_history[away_team][stat_prefix].append(row[base_cols['away']])

    df_rolling_stdev = pd.DataFrame(calculated_stats).set_index('Index')
    cols_to_join = [col for col in cols_to_calculate.values() if col in df_rolling_stdev.columns]
    logger.info(f"StDev Rolling calculado. Colunas adicionadas: {cols_to_join}")
    df_final = df_calc.join(df_rolling_stdev[cols_to_join]) if cols_to_join else df_calc
    return df_final

def calculate_rolling_stats(df: pd.DataFrame, stats_to_calc: List[str], window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Calcula médias móveis para as estatísticas especificadas."""
    df_calc = df.copy()
    teams = pd.concat([df_calc['Home'], df_calc['Away']]).unique()
    team_history: Dict[str, Dict[str, List[float]]] = {team: {stat: [] for stat in stats_to_calc} for team in teams}
    results_list = []
    rolling_cols_map = {}
    cols_to_calculate = {}

    logger.info(f"Iniciando cálculo Médias Rolling (Janela={window})...")
    for stat_prefix in stats_to_calc:
        media_col_h = f'Media_{stat_prefix}_H'
        media_col_a = f'Media_{stat_prefix}_A'
        skip_h = skip_a = False
        if media_col_h in df_calc.columns and pd.api.types.is_numeric_dtype(df_calc[media_col_h]):
            skip_h = True
        if media_col_a in df_calc.columns and pd.api.types.is_numeric_dtype(df_calc[media_col_a]):
            skip_a = True

        if skip_h and skip_a:
            logger.warning(f"{media_col_h}/{media_col_a} já existem.")
            continue

        if stat_prefix == 'Ptos':
            base_h, base_a = 'Ptos_H', 'Ptos_A'
        elif stat_prefix == 'VG':
            base_h, base_a = 'VG_H_raw', 'VG_A_raw'
        elif stat_prefix == 'CG':
            base_h, base_a = 'CG_H_raw', 'CG_A_raw'
        else:
            logger.warning(f"Prefixo Média '{stat_prefix}' desconhecido.")
            continue

        if base_h not in df_calc.columns or base_a not in df_calc.columns:
            logger.error(f"Erro Média: Colunas base '{base_h}'/'{base_a}' não encontradas.")
            continue

        rolling_cols_map[stat_prefix] = {'home': base_h, 'away': base_a}
        if not skip_h:
            cols_to_calculate[stat_prefix + '_H'] = media_col_h
        if not skip_a:
            cols_to_calculate[stat_prefix + '_A'] = media_col_a

    if not cols_to_calculate:
        logger.info("Nenhuma Média Rolling nova a calcular.")
        return df_calc

    logger.info(f"Calculando Médias rolling para: {list(cols_to_calculate.keys())}")

    calculated_stats = []
    for index, row in tqdm(df_calc.iterrows(), total=len(df_calc), desc="Calc. Rolling Médias"):
        home_team = row['Home']
        away_team = row['Away']
        current_match_features = {'Index': index}

        for stat_prefix, base_cols in rolling_cols_map.items():
            media_col_h = f'Media_{stat_prefix}_H'
            if stat_prefix + '_H' in cols_to_calculate:
                hist_H = team_history[home_team][stat_prefix]
                recent = hist_H[-window:]
                current_match_features[media_col_h] = np.mean(recent) if len(recent) > 0 else np.nan

        for stat_prefix, base_cols in rolling_cols_map.items():
            media_col_a = f'Media_{stat_prefix}_A'
            if stat_prefix + '_A' in cols_to_calculate:
                hist_A = team_history[away_team][stat_prefix]
                recent = hist_A[-window:]
                current_match_features[media_col_a] = np.mean(recent) if len(recent) > 0 else np.nan

        calculated_stats.append(current_match_features)

        for stat_prefix, base_cols in rolling_cols_map.items():
            if pd.notna(row[base_cols['home']]):
                team_history[home_team][stat_prefix].append(row[base_cols['home']])
            if pd.notna(row[base_cols['away']]):
                team_history[away_team][stat_prefix].append(row[base_cols['away']])

    df_rolling_means = pd.DataFrame(calculated_stats).set_index('Index')
    cols_to_join = [col for col in cols_to_calculate.values() if col in df_rolling_means.columns]
    logger.info(f"Médias Rolling calculadas. Colunas adicionadas: {cols_to_join}")
    df_final = df_calc.join(df_rolling_means[cols_to_join]) if cols_to_join else df_calc
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
    """Calcula CV_HDA e Diff_Media_CG no DataFrame fornecido."""
    df_calc = df.copy()
    logger.info("Calculando features derivadas (CV_HDA, Diff_Media_CG)...")
    epsilon = 1e-6
    if all(c in df_calc.columns for c in ODDS_COLS.values()):
        odds_matrix = df_calc[list(ODDS_COLS.values())]
        mean_odds = odds_matrix.mean(axis=1)
        std_odds = odds_matrix.std(axis=1)
        df_calc['CV_HDA'] = std_odds.div(mean_odds).fillna(0)
        df_calc.loc[mean_odds <= epsilon, 'CV_HDA'] = 0
    else:
        logger.warning("Odds 1x2 ausentes p/ CV_HDA.")
        df_calc['CV_HDA'] = np.nan
    if 'Media_CG_H' in df_calc.columns and 'Media_CG_A' in df_calc.columns:
        df_calc['Diff_Media_CG'] = df_calc['Media_CG_H'] - df_calc['Media_CG_A']
    else:
        logger.warning("Médias CG ausentes p/ Diff_Media_CG.")
        df_calc['Diff_Media_CG'] = np.nan
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

    logger.info(f"Calculando Prob Empate (Poisson Refinado, max_gols={max_goals})...")

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
        for k in range(max_goals + 1):
            prob_placar_kk = poisson.pmf(k, lambda_h) * poisson.pmf(k, lambda_a)
            prob_empate_total += prob_placar_kk
    except Exception as e:
        logger.error(f"Erro cálculo Poisson PMF: {e}", exc_info=True)
        df_calc['Prob_Empate_Poisson'] = np.nan
        return df_calc

    df_calc['Prob_Empate_Poisson'] = prob_empate_total
    logger.info("Prob_Empate_Poisson (Refinado) calculado.")
    return df_calc

def calculate_historical_intermediate(df: pd.DataFrame) -> pd.DataFrame:
    df_calc=df.copy();logger.info("Calculando stats intermediárias...");epsilon=1e-6;
    gh=GOALS_COLS.get('home','Goals_H_FT');ga=GOALS_COLS.get('away','Goals_A_FT');
    if gh in df_calc.columns and ga in df_calc.columns:
        h_g=pd.to_numeric(df_calc[gh],errors='coerce');a_g=pd.to_numeric(df_calc[ga],errors='coerce');
        condlist = [h_g > a_g, h_g == a_g, h_g < a_g]
        choicelist_res = ["H", "D", "A"]
        df_calc['FT_Result']=np.select(condlist, choicelist_res, default=pd.NA)
        condlist_pts_h = [df_calc['FT_Result']=='H', df_calc['FT_Result']=='D', df_calc['FT_Result']=='A']
        choicelist_pts_h = [3, 1, 0]
        df_calc['Ptos_H']=np.select(condlist_pts_h, choicelist_pts_h, default=np.nan)
        condlist_pts_a = [df_calc['FT_Result']=='A', df_calc['FT_Result']=='D', df_calc['FT_Result']=='H']
        choicelist_pts_a = [3, 1, 0]
        df_calc['Ptos_A']=np.select(condlist_pts_a, choicelist_pts_a, default=np.nan)
        df_calc['IsDraw'] = (df_calc['FT_Result'] == 'D').astype('Int64')
        df_calc.loc[df_calc['FT_Result'].isna(), 'IsDraw'] = pd.NA
        logger.info("->Result/IsDraw/Ptos OK.");
    else:
        logger.warning(f"->Gols'{gh}'/'{ga}'ausentes.");df_calc[['FT_Result','IsDraw','Ptos_H','Ptos_A']]=np.nan;

    req_odds=list(ODDS_COLS.values());
    if not all(p in df_calc.columns for p in['p_H','p_D','p_A']):
        if all(c in df_calc.columns for c in req_odds):
            logger.info("->Calculando p_H/D/A (ausentes)...");
            df_calc = calculate_probabilities(df_calc)
        else:
            logger.warning("->Odds 1x2 ausentes p/ calcular Probs p_H/D/A.");
            df_calc[['p_H','p_D','p_A']]=np.nan;

    prob_n=['p_H','p_A']; goal_n=[gh,ga];
    if all(c in df_calc.columns for c in prob_n+goal_n):
        h_g=pd.to_numeric(df_calc[gh],errors='coerce');a_g=pd.to_numeric(df_calc[ga],errors='coerce');
        p_H=pd.to_numeric(df_calc['p_H'],errors='coerce');p_A=pd.to_numeric(df_calc['p_A'],errors='coerce');
        df_calc['VG_H_raw']=h_g*p_A; df_calc['VG_A_raw']=a_g*p_H;
        # CG: Divisão segura (evita 0 e NaN no denominador)
        df_calc['CG_H_raw']=np.where((pd.notna(h_g)) & (h_g > epsilon) & (pd.notna(p_H)), p_H/h_g, np.nan);
        df_calc['CG_A_raw']=np.where((pd.notna(a_g)) & (a_g > epsilon) & (pd.notna(p_A)), p_A/a_g, np.nan);
        logger.info("->VG/CG Raw OK.");
    else:
        missing_inputs = [c for c in prob_n+goal_n if c not in df_calc.columns]
        logger.warning(f"->Inputs VG/CG Raw ausentes ({missing_inputs}).");
        df_calc[['VG_H_raw','VG_A_raw','CG_H_raw','CG_A_raw']]=np.nan;

    logger.info("Cálculo Intermediárias concluído.");return df_calc

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

def preprocess_and_feature_engineer(df_loaded: pd.DataFrame) -> Optional[Tuple[pd.DataFrame, pd.Series, List[str]]]:
    """Pipeline BackDraw: Calcula/Verifica TODAS as features NOVAS no histórico."""
    if df_loaded is None or df_loaded.empty: # Added empty check
        logger.error("Pré-proc: DataFrame histórico não carregado ou vazio.")
        return None
    logger.info("--- Iniciando Pré-processamento e Engenharia de Features (Histórico) ---")

    # Calculate league averages (handle potential NaNs)
    goals_h_col = GOALS_COLS.get('home', 'Goals_H_FT')
    goals_a_col = GOALS_COLS.get('away', 'Goals_A_FT')
    # Use np.nanmean to ignore NaNs when calculating average
    avg_h_league = np.nanmean(df_loaded[goals_h_col]) if goals_h_col in df_loaded else 1.0
    avg_a_league = np.nanmean(df_loaded[goals_a_col]) if goals_a_col in df_loaded else 1.0
    # Handle case where mean is NaN (e.g., all goals are NaN)
    if pd.isna(avg_h_league): avg_h_league = 1.0
    if pd.isna(avg_a_league): avg_a_league = 1.0
    logger.info(f"Médias Globais Liga (Histórico, NaN-ignored): Casa={avg_h_league:.3f}, Fora={avg_a_league:.3f}")

    # --- Feature Calculation Pipeline ---
    # Start with a copy
    df_processed = df_loaded.copy()

    # 1. Intermediate Stats (Result, Pts, IsDraw, VG/CG Raw, p_H/D/A)
    df_processed = calculate_historical_intermediate(df_processed)
    if 'IsDraw' not in df_processed.columns or df_processed['IsDraw'].isnull().all():
         logger.error("Coluna alvo 'IsDraw' não foi criada ou está toda NaN após intermediate.")
         return None
    # Ensure probabilities were calculated if needed
    if not all(p in df_processed.columns for p in ['p_H', 'p_D', 'p_A']):
        logger.warning("Probabilidades p_H/D/A ausentes. Tentando calcular novamente...")
        df_processed = calculate_probabilities(df_processed)
        if not all(p in df_processed.columns for p in ['p_H', 'p_D', 'p_A']):
             logger.error("Falha ao calcular probabilidades p_H/D/A.")
             # Decide whether to fail or continue without them
             # return None # Fail approach

    # 2. Normalized Probabilities
    df_processed = calculate_normalized_probabilities(df_processed)

    # 3. Rolling Means (VG, CG)
    stats_to_roll_mean = ['VG', 'CG'] # Requires VG_H_raw, VG_A_raw, CG_H_raw, CG_A_raw
    df_processed = calculate_rolling_stats(df_processed, stats_to_roll_mean, window=ROLLING_WINDOW)

    # 4. Rolling Stds (CG)
    stats_to_roll_std = ['CG'] # Requires CG_H_raw, CG_A_raw
    df_processed = calculate_rolling_std(df_processed, stats_to_roll_std, window=ROLLING_WINDOW)

    # 5. Rolling Goals & Forces (Avg Goals, FA/FD)
    df_processed = calculate_rolling_goal_stats(
        df_processed,
        window=ROLLING_WINDOW,
        avg_goals_home_league=avg_h_league, # Pass calculated averages
        avg_goals_away_league=avg_a_league
    )

    # 6. Poisson Draw Probability (Requires FA/FD)
    df_processed = calculate_poisson_draw_prob(df_processed,
                                            avg_goals_home_league=avg_h_league,
                                            avg_goals_away_league=avg_a_league,
                                            max_goals=5) # Configurable max goals

    # 7. Binned Features (Odd_D_Cat)
    df_processed = calculate_binned_features(df_processed) # Requires Odd_D_FT

    # 8. Derived Features (CV_HDA, Diff_Media_CG)
    df_processed = calculate_derived_features(df_processed) # Requires Odds 1x2, Media_CG_H/A

    # --- Final Selection and Dropna ---
    df_final_all = df_processed; target_col='IsDraw';
    logger.info(f"Selecionando features FINAIS da config: {FEATURE_COLUMNS}") # Use FEATURE_COLUMNS from config

    # Check which features are actually available after all calculations
    available_features = [f for f in FEATURE_COLUMNS if f in df_final_all.columns]
    missing_features = [f for f in FEATURE_COLUMNS if f not in df_final_all.columns]

    if not available_features:
        logger.error(f"Erro FINAIS: Nenhuma das features da lista FEATURE_COLUMNS foi calculada! Disponíveis:{list(df_final_all.columns)}");
        return None;
    if missing_features:
        logger.warning(f"Aviso FINAIS: Features da lista FEATURE_COLUMNS não calculadas/encontradas: {missing_features}. Elas não serão usadas.")

    required_cols_for_output = available_features + [target_col] # Use only available features + target

    # Ensure target exists
    if target_col not in df_final_all.columns:
         logger.error(f"Erro FINAIS: Coluna alvo '{target_col}' não encontrada.")
         return None

    # Select only available features and target
    df_select = df_final_all[required_cols_for_output].copy()

    logger.info(f"DF selecionado para dropna final: {df_select.shape}. Verificando NaNs nas features selecionadas...")
    initial_rows = len(df_select)

    # Check NaNs specifically in the selected feature columns before dropping
    nan_check_before = df_select[available_features].isnull().sum()
    cols_nan = nan_check_before[nan_check_before > 0]
    if not cols_nan.empty:
        logger.warning(f"NaNs encontrados ANTES do dropna final (nas features a serem usadas):\n{cols_nan}")
    else:
        logger.info("Nenhum NaN encontrado nas features selecionadas antes do dropna final.")

    # Drop rows where ANY of the SELECTED features OR the target is NaN
    df_select.dropna(subset=required_cols_for_output, inplace=True)
    rows_dropped = initial_rows - len(df_select)

    if rows_dropped > 0:
        logger.info(f"Removidas {rows_dropped} linhas com NaNs (nas features selecionadas ou no alvo).")
    if df_select.empty:
        logger.error("Erro: Nenhuma linha restante após dropna final das features selecionadas.")
        return None

    X = df_select[available_features] # Use only available features
    y = df_select[target_col].astype(int) # Ensure target is integer

    # Final check for NaNs in X and y
    if X.isnull().values.any(): logger.error("ERRO CRÍTICO: NaNs encontrados em X APÓS dropna!"); return None
    if y.isnull().values.any(): logger.error("ERRO CRÍTICO: NaNs encontrados em y APÓS dropna!"); return None

    logger.info(f"--- Pré-proc Histórico OK --- Shape X:{X.shape}, y:{y.shape}.")
    logger.debug(f"Features FINAIS usadas: {list(X.columns)}")
    return X, y, available_features # Return the list of features actually used

def fetch_and_process_fixtures() -> Optional[pd.DataFrame]:
    # Determine target date
    if FIXTURE_FETCH_DAY == "tomorrow":
        target_date = date.today() + timedelta(days=1)
    else:
        target_date = date.today()
    date_str = target_date.strftime('%Y-%m-%d')
    fixture_url = FIXTURE_CSV_URL_TEMPLATE.format(date_str=date_str)
    logger.info(f"Buscando jogos de {FIXTURE_FETCH_DAY} ({date_str}): {fixture_url}")

    # Fetch data
    try:
        # Use requests for better error handling and headers
        headers = {'User-Agent': 'Mozilla/5.0'} # Some sources might require a User-Agent
        response = requests.get(fixture_url, headers=headers, timeout=20) # Added timeout
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
        logger.info("Arquivo CSV encontrado. Lendo...")
        # Read CSV directly from response content using io.StringIO
        from io import StringIO
        csv_content = StringIO(response.text)
        df_fix = pd.read_csv(csv_content)
        logger.info(f"CSV baixado e lido. Shape: {df_fix.shape}")

    except requests.exceptions.RequestException as e_req:
        logger.error(f"Erro na requisição HTTP ao buscar CSV: {e_req}")
        return None
    except pd.errors.EmptyDataError:
        logger.warning(f"Arquivo CSV em {fixture_url} está vazio.")
        return pd.DataFrame() # Return empty DataFrame if CSV is empty
    except Exception as e_load:
        logger.error(f"Erro ao baixar ou ler CSV de jogos futuros: {e_load}", exc_info=True)
        return None

    # Process data
    try:
        logger.info("Processando CSV de jogos futuros...")
        logger.debug(f"Colunas no CSV futuro: {list(df_fix.columns)}")

        # Use the same map as historical data for consistency
        raw_cols_map = CSV_HIST_COL_MAP
        raw_fixture_cols = list(raw_cols_map.keys())

        # Keep only columns present in the CSV that are in our map
        cols_to_keep = [col for col in raw_fixture_cols if col in df_fix.columns]
        if not cols_to_keep:
            logger.error("Nenhuma coluna esperada (do map) encontrada no CSV de jogos futuros.")
            return None
        logger.debug(f"Colunas brutas a serem mantidas/renomeadas: {cols_to_keep}")

        df_processed = df_fix[cols_to_keep].copy()

        # Rename columns to internal names
        rename_map_valid = {k: v for k, v in raw_cols_map.items() if k in df_processed.columns}
        df_processed.rename(columns=rename_map_valid, inplace=True)
        logger.debug(f"Colunas após rename (internas): {list(df_processed.columns)}")

        # Define REQUIRED columns using internal names for fixtures
        # These might differ slightly from historical if some features can't be calculated for fixtures
        # Example: We need at least teams and odds 1x2. Others might be optional.
        # Use REQUIRED_FIXTURE_COLS from config if defined, otherwise define here
        try:
            # Use the config definition if it exists
            current_required_fixture_cols = REQUIRED_FIXTURE_COLS
        except NameError:
            # Fallback definition if not in config
            logger.warning("REQUIRED_FIXTURE_COLS not found in config, using fallback definition.")
            current_required_fixture_cols = ['League', 'Home', 'Away', 'Odd_H_FT', 'Odd_D_FT', 'Odd_A_FT']


        # Check for essential columns *after* renaming
        missing_required = [c for c in current_required_fixture_cols if c not in df_processed.columns]
        if missing_required:
            logger.error(f"Colunas essenciais internas ausentes no CSV futuro: {missing_required}")
            logger.debug(f"Colunas disponíveis após rename: {list(df_processed.columns)}")
            return None

        # Filter leagues (using internal 'League' name)
        if TARGET_LEAGUES:
            if 'League' in df_processed.columns:
                 initial_count = len(df_processed)
                 df_processed['League'] = df_processed['League'].astype(str).str.strip()
                 # Compare with the *values* of the TARGET_LEAGUES dict
                 df_processed = df_processed[df_processed['League'].isin(TARGET_LEAGUES.values())]
                 logger.info(f"Filtro de ligas (Futuro): {len(df_processed)} de {initial_count} jogos restantes.")
                 if df_processed.empty:
                      logger.info("Nenhum jogo futuro nas ligas alvo.")
                      return df_processed # Return empty DF
            else:
                 logger.warning("Coluna 'League' não encontrada para filtro de ligas futuras.")

        # Convert types and handle invalid odds (<=1) for required odds cols
        for col in list(ODDS_COLS.values()): # Only essential 1x2 odds
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                invalid_odds_count = (df_processed[col] <= 1).sum()
                if invalid_odds_count > 0:
                    df_processed.loc[df_processed[col] <= 1, col] = np.nan
                    logger.info(f"Futuro: Coluna '{col}': {invalid_odds_count} odds <= 1 convertidas para NaN.")

        # Convert other optional numeric columns if they exist
        optional_numeric = OTHER_ODDS_NAMES + list(XG_COLS.values())
        for col in optional_numeric:
             if col in df_processed.columns:
                  df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                  # Add specific NaN logic if needed (e.g., odds <= 1, xg < 0)
                  if col in OTHER_ODDS_NAMES:
                      invalid_odds_count = (df_processed[col] <= 1).sum()
                      if invalid_odds_count > 0:
                          df_processed.loc[df_processed[col] <= 1, col] = np.nan
                          logger.info(f"Futuro: Coluna '{col}': {invalid_odds_count} odds <= 1 convertidas para NaN.")


        # Drop rows with NaNs in the absolutely essential columns only
        logger.info(f"Verificando NaNs nas colunas essenciais (Futuro): {current_required_fixture_cols}")
        nan_counts_before_drop = df_processed[current_required_fixture_cols].isnull().sum()
        logger.debug(f"Contagem de NaNs ANTES do dropna essencial (Futuro):\n{nan_counts_before_drop[nan_counts_before_drop > 0]}")

        initial_rows_fix = len(df_processed)
        df_processed.dropna(subset=current_required_fixture_cols, inplace=True)
        rows_dropped_fix = initial_rows_fix - len(df_processed)
        if rows_dropped_fix > 0: logger.info(f"Futuro: Removidas {rows_dropped_fix} linhas com NaNs essenciais.")

        if df_processed.empty:
             logger.info("Nenhum jogo futuro restante após limpeza essencial.")
             return df_processed # Return empty DF

        # Add Date/Time columns if 'Date' exists
        if 'Date' in df_processed.columns:
             df_processed['Date'] = pd.to_datetime(df_processed['Date'], errors='coerce')
             df_processed.dropna(subset=['Date'], inplace=True) # Drop if date conversion failed
             if 'Date' in df_processed.columns: # Check again after dropna
                 df_processed['Date_Str'] = df_processed['Date'].dt.strftime('%Y-%m-%d')
                 df_processed['Time_Str'] = df_processed['Date'].dt.strftime('%H:%M')
             else:
                 df_processed[['Date_Str', 'Time_Str']] = 'N/A'
        else:
             df_processed[['Date_Str', 'Time_Str']] = 'N/A'


        df_processed.reset_index(drop=True, inplace=True) # Reset index
        logger.info(f"Processamento CSV jogos futuros OK. Shape final: {df_processed.shape}")
        logger.debug(f"Colunas finais jogos futuros: {list(df_processed.columns)}")
        return df_processed

    except Exception as e_proc:
        logger.error(f"Erro durante processamento do CSV de jogos futuros: {e_proc}", exc_info=True)
        return None

def prepare_fixture_data(fixture_df: pd.DataFrame, historical_df: pd.DataFrame, feature_columns: List[str]) -> Optional[pd.DataFrame]:
    # ... (initial checks remain the same) ...
    if fixture_df is None or historical_df is None or not feature_columns: logger.error("Prep Fixture: Args inválidos."); return None
    if historical_df.empty: logger.error("Prep Fixture: Histórico vazio."); return None
    if fixture_df.empty: logger.info("Prep Fixture: Nenhum jogo futuro."); return pd.DataFrame(columns=feature_columns)

    logger.info("--- Preparando Features Finais para Jogos Futuros ---")
    logger.info(f"Jogos futuros brutos: {fixture_df.shape}")
    logger.info(f"Features finais esperadas p/ modelo: {feature_columns}")
    epsilon = 1e-6 # Small number

    # --- 1. Calcular Médias da Liga do Histórico (Tratando NaNs) ---
    goals_h_col = GOALS_COLS.get('home','Goals_H_FT')
    goals_a_col = GOALS_COLS.get('away','Goals_A_FT')
    avg_h_league = np.nanmean(historical_df[goals_h_col]) if goals_h_col in historical_df else 1.0
    avg_a_league = np.nanmean(historical_df[goals_a_col]) if goals_a_col in historical_df else 1.0
    if pd.isna(avg_h_league): avg_h_league = 1.0
    if pd.isna(avg_a_league): avg_a_league = 1.0
    # Ensure safe division
    avg_h_league_safe = max(avg_h_league, epsilon)
    avg_a_league_safe = max(avg_a_league, epsilon)
    logger.info(f"Usando Médias Liga (Hist, NaN-ignored): Casa={avg_h_league_safe:.3f}, Fora={avg_a_league_safe:.3f} para cálculo futuro.")

    # --- 2. Processar Histórico para Rolling Stats (APENAS o necessário) ---
    logger.info("Processando histórico APENAS para stats rolling necessárias...")
    start_time = time.time()
    hist_cols_needed = ['Date', 'Home', 'Away'] + list(GOALS_COLS.values())
    needs_probs = any(f.startswith(('Media_VG', 'Media_CG', 'Std_VG', 'Std_CG', 'Diff_Media_CG')) for f in feature_columns)
    if needs_probs: hist_cols_needed.extend(list(ODDS_COLS.values()))

    hist_cols_present = [c for c in hist_cols_needed if c in historical_df.columns]
    historical_df_minimal = historical_df[hist_cols_present].copy()

    if needs_probs:
        if not all(p in historical_df_minimal.columns for p in ['p_H', 'p_A']):
            logger.info(" -> Calculando p_H/p_A para histórico (rolling)...")
            historical_df_minimal = calculate_probabilities(historical_df_minimal)
            needs_vg_cg_raw = any(f.startswith(('Media_VG', 'Media_CG', 'Std_VG', 'Std_CG', 'Diff_Media_CG')) for f in feature_columns)
            if needs_vg_cg_raw:
                 logger.info(" -> Calculando VG/CG Raw para histórico (rolling)...")
                 historical_df_minimal = calculate_historical_intermediate(historical_df_minimal) # Includes VG/CG raw calc

    if 'Date' not in historical_df_minimal: logger.error("Histórico sem 'Date'."); return None
    historical_df_minimal = historical_df_minimal.sort_values(by='Date', ascending=True)
    teams_in_hist = pd.concat([historical_df_minimal['Home'], historical_df_minimal['Away']]).unique()
    logger.info(f"Histórico mínimo processado e ordenado em {time.time()-start_time:.2f}s. {len(teams_in_hist)} times.")

    # --- 3. Calcular Features Rolling para Jogos Futuros ---
    logger.info("Calculando features rolling para jogos futuros...")
    stats_mean_needed = [s for s in ['VG', 'CG'] if any(f.startswith(f'Media_{s}') for f in feature_columns)]
    stats_std_needed = [s for s in ['CG'] if any(f.startswith(f'Std_{s}') for f in feature_columns)] # Add VG/Ptos if needed
    needs_rolling_goals = any(f.startswith('Avg_Gols_') for f in feature_columns)
    needs_fa_fd = any(f.startswith(('FA_', 'FD_', 'Prob_Empate_Poisson')) for f in feature_columns)

    fixture_features_dict = {idx: {} for idx in fixture_df.index}
    team_history_rolling = {team: {} for team in teams_in_hist}
    # Initialize necessary history lists
    for team in teams_in_hist:
        if 'VG' in stats_mean_needed or 'VG' in stats_std_needed: team_history_rolling[team]['VG'] = []
        if 'CG' in stats_mean_needed or 'CG' in stats_std_needed: team_history_rolling[team]['CG'] = []
        if needs_rolling_goals or needs_fa_fd:
            team_history_rolling[team]['scored_home'] = []
            team_history_rolling[team]['conceded_home'] = []
            team_history_rolling[team]['scored_away'] = []
            team_history_rolling[team]['conceded_away'] = []

    # Build final historical state
    logger.info("Construindo estado final do histórico rolling...")
    gh_hist = GOALS_COLS.get('home', 'Goals_H_FT')
    ga_hist = GOALS_COLS.get('away', 'Goals_A_FT')
    for index, row in historical_df_minimal.iterrows():
        ht = row['Home']; at = row['Away']
        # Append VG/CG Raw if needed
        if 'VG' in stats_mean_needed or 'VG' in stats_std_needed:
             if 'VG_H_raw' in row and pd.notna(row['VG_H_raw']): team_history_rolling[ht]['VG'].append(row['VG_H_raw'])
             if 'VG_A_raw' in row and pd.notna(row['VG_A_raw']): team_history_rolling[at]['VG'].append(row['VG_A_raw'])
        if 'CG' in stats_mean_needed or 'CG' in stats_std_needed:
             if 'CG_H_raw' in row and pd.notna(row['CG_H_raw']): team_history_rolling[ht]['CG'].append(row['CG_H_raw'])
             if 'CG_A_raw' in row and pd.notna(row['CG_A_raw']): team_history_rolling[at]['CG'].append(row['CG_A_raw'])
        # Append goals if needed
        if needs_rolling_goals or needs_fa_fd:
            home_goals = pd.to_numeric(row.get(gh_hist), errors='coerce')
            away_goals = pd.to_numeric(row.get(ga_hist), errors='coerce')
            if pd.notna(home_goals):
                team_history_rolling[ht]['scored_home'].append(home_goals)
                team_history_rolling[at]['conceded_away'].append(home_goals)
            if pd.notna(away_goals):
                team_history_rolling[at]['scored_away'].append(away_goals)
                team_history_rolling[ht]['conceded_home'].append(away_goals)

    # Calculate features for future matches
    logger.info("Calculando features rolling para jogos futuros...")
    for index, future_match in tqdm(fixture_df.iterrows(), total=len(fixture_df), desc="Calc. Rolling Futuro"):
        ht = future_match.get('Home')
        at = future_match.get('Away')
        current_match_features = {}
        for team_type, team_name in [('H', ht), ('A', at)]:
            if team_name and team_name in team_history_rolling:
                team_hist = team_history_rolling[team_name]
                # Calculate Mean Stats
                for stat_prefix in stats_mean_needed:
                    hist_values = team_hist.get(stat_prefix, [])
                    recent_values = hist_values[-ROLLING_WINDOW:]
                    current_match_features[f'Media_{stat_prefix}_{team_type}'] = np.nanmean(recent_values) if recent_values else np.nan
                # Calculate Std Stats
                for stat_prefix in stats_std_needed:
                     hist_values = team_hist.get(stat_prefix, [])
                     recent_values = hist_values[-ROLLING_WINDOW:]
                     current_match_features[f'Std_{stat_prefix}_{team_type}'] = np.nanstd(recent_values) if len(recent_values) >= 2 else np.nan
                 # Calculate Avg Goals, FA, FD
                if needs_rolling_goals or needs_fa_fd:
                     home_scored = team_hist.get('scored_home', [])
                     home_conceded = team_hist.get('conceded_home', [])
                     away_scored = team_hist.get('scored_away', [])
                     away_conceded = team_hist.get('conceded_away', [])
                     all_scored = home_scored + away_scored
                     all_conceded = home_conceded + away_conceded
                     recent_scored = all_scored[-ROLLING_WINDOW:]
                     recent_conceded = all_conceded[-ROLLING_WINDOW:]
                     avg_gs = np.nanmean(recent_scored) if recent_scored else np.nan
                     avg_gc = np.nanmean(recent_conceded) if recent_conceded else np.nan
                     if needs_rolling_goals:
                         current_match_features[f'Avg_Gols_Marcados_{team_type}'] = avg_gs
                         current_match_features[f'Avg_Gols_Sofridos_{team_type}'] = avg_gc
                     if needs_fa_fd:
                          # *** USE SAFE AVERAGES FOR DIVISION ***
                          if team_type == 'H':
                              current_match_features['FA_H'] = avg_gs / avg_h_league_safe if pd.notna(avg_gs) else np.nan
                              current_match_features['FD_H'] = avg_gc / avg_a_league_safe if pd.notna(avg_gc) else np.nan
                          else:
                              current_match_features['FA_A'] = avg_gs / avg_a_league_safe if pd.notna(avg_gs) else np.nan
                              current_match_features['FD_A'] = avg_gc / avg_h_league_safe if pd.notna(avg_gc) else np.nan
            else:
                # Team not in history, set NaN
                logger.debug(f"Time '{team_name}' (Tipo {team_type}) não encontrado no histórico. Features rolling serão NaN.")
                for stat_prefix in stats_mean_needed: current_match_features[f'Media_{stat_prefix}_{team_type}'] = np.nan
                for stat_prefix in stats_std_needed: current_match_features[f'Std_{stat_prefix}_{team_type}'] = np.nan
                if needs_rolling_goals: current_match_features[f'Avg_Gols_Marcados_{team_type}'] = np.nan; current_match_features[f'Avg_Gols_Sofridos_{team_type}'] = np.nan
                if needs_fa_fd: current_match_features[f'FA_{team_type}'] = np.nan; current_match_features[f'FD_{team_type}'] = np.nan
        fixture_features_dict[index] = current_match_features

    df_rolling_features = pd.DataFrame.from_dict(fixture_features_dict, orient='index')
    logger.info(f"Rolling stats p/ futuro calculadas. Colunas adicionadas: {list(df_rolling_features.columns)}")
    df_temp_fixtures = fixture_df.join(df_rolling_features, how='left')

    # --- 4. Calcular Features Não-Rolling ---
    logger.info("Calculando probabilidades, binning, Poisson e derivadas para jogos futuros...")
    df_temp_fixtures = calculate_probabilities(df_temp_fixtures)
    df_temp_fixtures = calculate_normalized_probabilities(df_temp_fixtures)
    df_temp_fixtures = calculate_binned_features(df_temp_fixtures)
    # Use safe averages for Poisson as well
    df_temp_fixtures = calculate_poisson_draw_prob(
        df_temp_fixtures,
        avg_goals_home_league=avg_h_league_safe, # Use safe average
        avg_goals_away_league=avg_a_league_safe, # Use safe average
        max_goals=5
    )
    df_temp_fixtures = calculate_derived_features(df_temp_fixtures) # Already handles potential division by zero

    # --- 5. Seleção Final e Tratamento NaN ---
    logger.info(f"Selecionando features FINAIS ({feature_columns}) para modelo...")
    # ... (selection and fillna logic remains the same) ...
    final_features_available = [f for f in feature_columns if f in df_temp_fixtures.columns]
    missing_final = [f for f in feature_columns if f not in df_temp_fixtures.columns]
    if not final_features_available: logger.error("Erro FINAIS (Futuro): Nenhuma feature esperada encontrada."); return None
    if missing_final: logger.warning(f"Aviso FINAIS (Futuro): Features esperadas não encontradas: {missing_final}. Preenchendo com 0.");
    for col in missing_final: df_temp_fixtures[col] = 0
    X_fix_prep = df_temp_fixtures[feature_columns].copy()
    logger.info(f"Shape após seleção final: {X_fix_prep.shape}")
    nan_counts = X_fix_prep.isnull().sum(); total_nans = nan_counts.sum();
    if total_nans > 0: logger.warning(f"{total_nans} NaNs restantes. Preenchendo com 0."); logger.debug(f"NaNs por coluna:\n{nan_counts[nan_counts > 0]}"); X_fix_prep.fillna(0, inplace=True);
    else: logger.info("Nenhum NaN restante nas features finais.")
    if X_fix_prep.isnull().values.any(): logger.error("ERRO CRÍTICO (Futuro): NaNs ainda presentes!"); return None
    logger.info(f"--- Preparação Features Futuras OK. Shape Final: {X_fix_prep.shape} ---")
    return X_fix_prep