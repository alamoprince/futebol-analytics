# --- src/data_handler.py ---
import pandas as pd
import numpy as np
from config import (
    HISTORICAL_DATA_PATH, FEATURE_COLUMNS,
    ODDS_COLS, GOALS_COLS, ROLLING_WINDOW, EXCEL_EXPECTED_COLS,
    FIXTURE_FETCH_DAY, FIXTURE_CSV_URL_TEMPLATE,
    FIXTURE_CSV_COL_MAP, REQUIRED_FIXTURE_COLS, TARGET_LEAGUES, CSV_EXPECTED_COLS,
    OTHER_ODDS_NAMES
)
from typing import Tuple, Optional, List, Dict, Any
from tqdm import tqdm
import time  # Embora não usado diretamente, pode ser útil para debugging
from datetime import date, timedelta, datetime  # datetime não usado diretamente, mas pode ser útil
import requests
from urllib.error import HTTPError, URLError
from scipy.stats import poisson
from logger_config import setup_logger

logger = setup_logger('DataHandler')

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

# Função load_historical_data
def load_historical_data(file_path: str = HISTORICAL_DATA_PATH) -> Optional[pd.DataFrame]:
    """Carrega histórico, verifica colunas base e converte."""
    try:
        df = pd.read_excel(file_path)
        logger.info(f"Histórico carregado: {file_path} (Shape: {df.shape})")
        # Verifica colunas mínimas + as que usamos DIRETAMENTE como features
        required_excel_names = list(set(EXCEL_EXPECTED_COLS))
        missing = [col for col in required_excel_names if col not in df.columns]
        missing_extras = [m for m in missing if m in ['Odd_Over25_FT', 'Odd_BTTS_Yes']]
        missing_minimum = [m for m in missing if m not in ['Odd_Over25_FT', 'Odd_BTTS_Yes']]
        if missing_extras:
            logger.warning(f"Colunas odds extras ({missing_extras}) não encontradas no histórico.")
        if missing_minimum:
            logger.error(f"Colunas mínimas faltando no Histórico: {missing_minimum}")
            return None

        # Renomeia Gols para nomes internos
        df = df.rename(columns={GOALS_COLS['home']: 'Goals_H_FT', GOALS_COLS['away']: 'Goals_A_FT'})
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # Colunas para checar NaNs antes de sort (essenciais p/ cálculos intermediários)
        dropna_check_cols = ['Date', 'Home', 'Away', 'Goals_H_FT', 'Goals_A_FT'] + list(ODDS_COLS.values())
        dropna_check_cols_exist = [col for col in dropna_check_cols if col in df.columns]
        df = df.dropna(subset=dropna_check_cols_exist)
        if 'Date' in df.columns:
            df = df.sort_values(by='Date').reset_index(drop=True)

        # Converte todas as colunas numéricas esperadas (incluindo extras)
        num_cols_convert = ['Goals_H_FT', 'Goals_A_FT'] + list(ODDS_COLS.values()) + ['Odd_Over25_FT', 'Odd_BTTS_Yes']
        for col in num_cols_convert:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        # Dropna SÓ para as essenciais para cálculos
        df = df.dropna(subset=list(ODDS_COLS.values()) + ['Goals_H_FT', 'Goals_A_FT'])
        logger.info("Histórico carregado e colunas essenciais convertidas.")
        return df
    except FileNotFoundError:
        logger.error(f"Histórico NÃO ENCONTRADO: '{file_path}'")
        return None
    except Exception as e:
        logger.error(f"Erro inesperado ao carregar histórico: {e}")
        return None

def calculate_probabilities(df: pd.DataFrame, epsilon=1e-6) -> pd.DataFrame:
    """Calcula probs implícitas p_H, p_D, p_A. Requer Odd_H/D/A_FT."""
    df_calc = df.copy()
    required_odds = list(ODDS_COLS.values())  # Usa config
    if not all(c in df_calc.columns for c in required_odds):
        logger.warning("Odds 1x2 ausentes para calcular Probabilidades.")
        df_calc[['p_H', 'p_D', 'p_A']] = np.nan
        return df_calc

    for col in required_odds:
        df_calc[col] = pd.to_numeric(df_calc[col], errors='coerce')

    # Evita divisão por zero e lida com NaNs
    odd_h = df_calc[ODDS_COLS['home']].replace(0, epsilon)
    odd_d = df_calc[ODDS_COLS['draw']].replace(0, epsilon)
    odd_a = df_calc[ODDS_COLS['away']].replace(0, epsilon)

    df_calc['p_H'] = (1 / odd_h).fillna(0)
    df_calc['p_D'] = (1 / odd_d).fillna(0)
    df_calc['p_A'] = (1 / odd_a).fillna(0)
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
    """Calcula FT_Result, IsDraw, Ptos, Probs, VG/CG raw no DataFrame."""
    df_calc = df.copy()
    logger.info("Calculando stats intermediárias (Resultado, Pontos, VG/CG raw)...")
    epsilon = 1e-6

    goals_h_col = GOALS_COLS.get('home', 'Goals_H_FT')
    goals_a_col = GOALS_COLS.get('away', 'Goals_A_FT')

    if goals_h_col in df_calc.columns and goals_a_col in df_calc.columns:
        h_goals = pd.to_numeric(df_calc[goals_h_col], errors='coerce')
        a_goals = pd.to_numeric(df_calc[goals_a_col], errors='coerce')

        df_calc['FT_Result'] = np.select(
            [h_goals > a_goals, h_goals == a_goals],
            ["H", "D"],
            default="A"
        )
        df_calc['IsDraw'] = (df_calc['FT_Result'] == 'D').astype(int)

        df_calc['Ptos_H'] = np.select(
            [df_calc['FT_Result'] == 'H', df_calc['FT_Result'] == 'D'],
            [3, 1], default=0
        )
        df_calc['Ptos_A'] = np.select(
            [df_calc['FT_Result'] == 'A', df_calc['FT_Result'] == 'D'],
            [3, 1], default=0
        )
        
        logger.info("Resultado (FT_Result, IsDraw) e Pontos (Ptos_H/A) calculados.")
    else:
        logger.warning(f"Colunas de Gols ('{goals_h_col}', '{goals_a_col}') não encontradas. Resultado/Pontos não calculados.")
        df_calc[['FT_Result', 'IsDraw', 'Ptos_H', 'Ptos_A']] = np.nan

    required_odds = list(ODDS_COLS.values())
    if all(c in df_calc.columns for c in required_odds):
        if not all(p in df_calc.columns for p in ['p_H', 'p_D', 'p_A']):
            logger.info("Calculando probabilidades implícitas (p_H/D/A)...")
            for col in required_odds:
                df_calc[col] = pd.to_numeric(df_calc[col], errors='coerce')
            odd_h = df_calc[ODDS_COLS['home']].replace(0, epsilon)
            odd_d = df_calc[ODDS_COLS['draw']].replace(0, epsilon)
            odd_a = df_calc[ODDS_COLS['away']].replace(0, epsilon)
            df_calc['p_H'] = (1 / odd_h).fillna(np.nan)
            df_calc['p_D'] = (1 / odd_d).fillna(np.nan)
            df_calc['p_A'] = (1 / odd_a).fillna(np.nan)
        else:
            logger.info("Probabilidades implícitas (p_H/D/A) já existem.")
    else:
        logger.warning("Odds 1x2 ausentes para calcular Probabilidades.")
        df_calc[['p_H', 'p_D', 'p_A']] = np.nan

    prob_cols_needed = ['p_H', 'p_A']
    goal_cols_needed = [goals_h_col, goals_a_col]
    if all(c in df_calc.columns for c in prob_cols_needed + goal_cols_needed):
         h_goals = pd.to_numeric(df_calc[goals_h_col], errors='coerce')
         a_goals = pd.to_numeric(df_calc[goals_a_col], errors='coerce')
         p_H = pd.to_numeric(df_calc['p_H'], errors='coerce')
         p_A = pd.to_numeric(df_calc['p_A'], errors='coerce')

         df_calc['VG_H_raw'] = h_goals * p_A
         df_calc['VG_A_raw'] = a_goals * p_H

         df_calc['CG_H_raw'] = np.where(h_goals > 0, p_H / h_goals, np.nan)
         df_calc['CG_A_raw'] = np.where(a_goals > 0, p_A / a_goals, np.nan)
         logger.info("Valor/Custo do Gol (VG/CG raw) calculados.")
    else:
         logger.warning("Colunas de Gols ou Probabilidades ausentes para calcular VG/CG raw.")
         df_calc[['VG_H_raw', 'VG_A_raw', 'CG_H_raw', 'CG_A_raw']] = np.nan

    logger.info("Cálculo de stats intermediárias concluído.")
    return df_calc

def calculate_rolling_goal_stats(
    df: pd.DataFrame,
    window: int = ROLLING_WINDOW,
    avg_goals_home_league: Optional[float] = None,
    avg_goals_away_league: Optional[float] = None
    ) -> pd.DataFrame:
    """
    Calcula médias móveis de gols e Força de Ataque/Defesa ajustada pela liga.
    """
    df_calc = df.copy()
    goals_h_col = GOALS_COLS.get('home', 'Goals_H_FT')
    goals_a_col = GOALS_COLS.get('away', 'Goals_A_FT')
    epsilon = 1e-6

    if goals_h_col not in df_calc.columns or goals_a_col not in df_calc.columns:
        logger.warning("Calc Rolling Goals: Colunas Gols ausentes.")
        cols_to_add = ['Avg_Gols_Marcados_H', 'Avg_Gols_Sofridos_H', 'Avg_Gols_Marcados_A', 'Avg_Gols_Sofridos_A',
                       'FA_H', 'FD_H', 'FA_A', 'FD_A']
        for col in cols_to_add:
            df_calc[col] = np.nan
        return df_calc

    avg_h_league = avg_goals_home_league if avg_goals_home_league is not None and avg_goals_home_league > 0 else 1.0
    avg_a_league = avg_goals_away_league if avg_goals_away_league is not None and avg_goals_away_league > 0 else 1.0
    if avg_goals_home_league is None or avg_goals_away_league is None:
         logger.warning(f"Médias de gols da liga não fornecidas. Usando defaults: Casa={avg_h_league:.2f}, Fora={avg_a_league:.2f}")

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

        h_scored_hist = team_history[home_team]['scored_home']
        h_conceded_hist = team_history[home_team]['conceded_home']
        avg_gs_h = np.mean(h_scored_hist[-window:]) if h_scored_hist else np.nan
        avg_gc_h = np.mean(h_conceded_hist[-window:]) if h_conceded_hist else np.nan
        current_stats['Avg_Gols_Marcados_H'] = avg_gs_h
        current_stats['Avg_Gols_Sofridos_H'] = avg_gc_h
        current_stats['FA_H'] = avg_gs_h / avg_h_league if pd.notna(avg_gs_h) else np.nan
        current_stats['FD_H'] = avg_gc_h / avg_a_league if pd.notna(avg_gc_h) else np.nan

        a_scored_hist = team_history[away_team]['scored_away']
        a_conceded_hist = team_history[away_team]['conceded_away']
        avg_gs_a = np.mean(a_scored_hist[-window:]) if a_scored_hist else np.nan
        avg_gc_a = np.mean(a_conceded_hist[-window:]) if a_conceded_hist else np.nan
        current_stats['Avg_Gols_Marcados_A'] = avg_gs_a
        current_stats['Avg_Gols_Sofridos_A'] = avg_gc_a
        current_stats['FA_A'] = avg_gs_a / avg_a_league if pd.notna(avg_gs_a) else np.nan
        current_stats['FD_A'] = avg_gc_a / avg_h_league if pd.notna(avg_gc_a) else np.nan

        results_list.append(current_stats)

        home_goals = pd.to_numeric(row[goals_h_col], errors='coerce')
        away_goals = pd.to_numeric(row[goals_a_col], errors='coerce')
        if pd.notna(home_goals):
            team_history[home_team]['scored_home'].append(home_goals)
            team_history[away_team]['conceded_away'].append(home_goals)
        if pd.notna(away_goals):
            team_history[away_team]['scored_away'].append(away_goals)
            team_history[home_team]['conceded_home'].append(away_goals)

    df_rolling_stats = pd.DataFrame(results_list).set_index('Index')
    logger.info(f"Rolling Gols/FA/FD calculado. Colunas: {list(df_rolling_stats.columns)}")
    return df_calc.join(df_rolling_stats)

def preprocess_and_feature_engineer(df_loaded: pd.DataFrame) -> Optional[Tuple[pd.DataFrame, pd.Series, List[str]]]:
    """Pipeline BackDraw: Calcula/Verifica TODAS as features NOVAS no histórico."""
    if df_loaded is None:
        return None
    logger.info("--- Iniciando Pré-processamento e Engenharia de Features (Histórico) ---")
    logger.info("--- Pré-proc Histórico (v4 - Poisson Refinado) ---")

    goals_h_col = GOALS_COLS.get('home', 'Goals_H_FT')
    goals_a_col = GOALS_COLS.get('away', 'Goals_A_FT')
    avg_h_league = df_loaded[goals_h_col].mean() if goals_h_col in df_loaded else 1.0
    avg_a_league = df_loaded[goals_a_col].mean() if goals_a_col in df_loaded else 1.0
    logger.info(f"Médias Globais Liga (Histórico): Casa={avg_h_league:.3f}, Fora={avg_a_league:.3f}")

    df_interm = calculate_historical_intermediate(df_loaded)
    if 'IsDraw' not in df_interm.columns:
         logger.error("Coluna alvo 'IsDraw' não foi criada por calculate_historical_intermediate.")
         return None
    if not all(p in df_interm.columns for p in ['p_H', 'p_D', 'p_A']):
        df_interm = calculate_probabilities(df_interm)

    df_probs_norm = calculate_normalized_probabilities(df_interm)

    stats_to_roll_mean = ['VG', 'CG']
    df_rolling_mean = calculate_rolling_stats(df_probs_norm, stats_to_roll_mean, window=ROLLING_WINDOW)

    stats_to_roll_std = ['CG']
    df_rolling_std = calculate_rolling_std(df_rolling_mean, stats_to_roll_std, window=ROLLING_WINDOW)

    df_goals_forces = calculate_rolling_goal_stats(
        df_rolling_std,
        window=ROLLING_WINDOW,
        avg_goals_home_league=avg_h_league,
        avg_goals_away_league=avg_a_league
    )

    df_poisson = calculate_poisson_draw_prob(df_goals_forces, 
                                            avg_goals_home_league=avg_h_league,
                                            avg_goals_away_league=avg_a_league,
                                            max_goals=5)

    df_binned = calculate_binned_features(df_poisson)

    logger.info("Calculando features derivadas...")
    df_derived = calculate_derived_features(df_binned)

    df_final = df_derived
    target_col = 'IsDraw'
    logger.info("Selecionando features finais e tratando NaNs...")

    required_final_cols = FEATURE_COLUMNS + [target_col]

    missing_final = [f for f in required_final_cols if f not in df_final.columns]
    if missing_final:
        logger.error(f"Colunas finais ausentes após todos cálculos: {missing_final}. Colunas disponíveis: {list(df_final.columns)}")
        return None

    df_final_selection = df_final[required_final_cols].copy()

    initial_rows = len(df_final_selection)
    df_final_selection = df_final_selection.dropna()
    rows_dropped = initial_rows - len(df_final_selection)
    if rows_dropped > 0:
        logger.info(f"Removidas {rows_dropped} linhas contendo NaNs nas features finais ou alvo.")

    if df_final_selection.empty:
        logger.error("Nenhuma linha restante após remover NaNs.")
        return None

    X = df_final_selection[FEATURE_COLUMNS]
    y = df_final_selection[target_col]

    logger.info(f"--- Pré-processamento e Engenharia (Histórico) OK ---")
    logger.info(f"Shape X final: {X.shape}, Shape y final: {y.shape}")
    logger.info(f"Features finais usadas: {list(X.columns)}")
    return X, y, FEATURE_COLUMNS

def fetch_and_process_fixtures() -> Optional[pd.DataFrame]:
    if FIXTURE_FETCH_DAY == "tomorrow":
        target_date = date.today() + timedelta(days=1)
    else:
        target_date = date.today()
    date_str = target_date.strftime('%Y-%m-%d')
    fixture_url = FIXTURE_CSV_URL_TEMPLATE.format(date_str=date_str)
    logger.info(f"Buscando jogos {FIXTURE_FETCH_DAY} ({date_str}): {fixture_url}")
    try:
        response = requests.head(fixture_url, allow_redirects=True, timeout=10)
        response.raise_for_status()
        logger.info("Arquivo encontrado. Baixando...")
        df_fix = pd.read_csv(fixture_url)
        logger.info(f"CSV baixado. Shape: {df_fix.shape}")
    except Exception as e_load:
        logger.error(f"Erro buscar/ler CSV: {e_load}")
        return None
    try:
        logger.info("Processando CSV...")
        cols_to_keep = list(FIXTURE_CSV_COL_MAP.keys())
        cols_exist_in_df = [c for c in cols_to_keep if c in df_fix.columns]
        if not cols_exist_in_df:
            logger.error("Nenhuma coluna esperada no CSV.")
            return None
        df_processed = df_fix[cols_exist_in_df].copy()
        df_processed.rename(columns=FIXTURE_CSV_COL_MAP, inplace=True)
        current_required_fixture_cols = REQUIRED_FIXTURE_COLS
        missing_required = [c for c in current_required_fixture_cols if c not in df_processed.columns]
        if missing_required:
            logger.error(f"Colunas essenciais pós-map ausentes no CSV: {missing_required}")
            return None
        if TARGET_LEAGUES:
            initial_count = len(df_processed)
            df_processed = df_processed[df_processed['League'].astype(str).isin(TARGET_LEAGUES)]
            logger.info(f"Filtro ligas: {len(df_processed)} de {initial_count} jogos.")
            if df_processed.empty:
                return df_processed
        df_processed.dropna(subset=current_required_fixture_cols, inplace=True)
        df_processed.reset_index(inplace=True, drop=True)
        logger.info(f"Processamento CSV OK. Shape: {df_processed.shape}")
        return df_processed
    except Exception as e_proc:
        logger.error(f"Erro processamento CSV: {e_proc}")
        return None

def prepare_fixture_data(fixture_df: pd.DataFrame, historical_df: pd.DataFrame, feature_columns: List[str]) -> Optional[pd.DataFrame]:
    """Prepara features para jogos futuros, incluindo médias de gols e derivadas."""
    if fixture_df is None or historical_df is None or not feature_columns:
        return None
    if historical_df.empty:
        logger.error("Prep Fixture: Histórico vazio.")
        return None
    if fixture_df.empty:
        logger.info("Prep Fixture: Nenhum jogo futuro.")
        return pd.DataFrame(columns=feature_columns)

    logger.info("--- Preparando Features Finais para Jogos Futuros (v3 - com Derivadas) ---")
    logger.info(f"Jogos futuros brutos: {fixture_df.shape}")
    logger.info(f"Features finais esperadas p/ modelo: {feature_columns}")

    goals_h_col = GOALS_COLS.get('home','Goals_H_FT')
    goals_a_col = GOALS_COLS.get('away','Goals_A_FT')
    avg_h_league = historical_df[goals_h_col].mean() if goals_h_col in historical_df else 1.0
    avg_a_league = historical_df[goals_a_col].mean() if goals_a_col in historical_df else 1.0
    logger.info(f"Usando Médias Liga (Hist): Casa={avg_h_league:.3f}, Fora={avg_a_league:.3f} para cálculo futuro.")

    logger.info("Processando histórico p/ rolling stats...")
    start_time = time.time()
    historical_df_processed = calculate_historical_intermediate(historical_df.copy())
    historical_df_processed = historical_df_processed.sort_values(by='Date', ascending=False)
    teams_in_hist = pd.concat([historical_df_processed['Home'], historical_df_processed['Away']]).unique()
    logger.info(f"Histórico processado em {time.time()-start_time:.2f}s.")

    logger.info("Calculando médias/StDev/Gols/FA/FD rolling para jogos futuros...")
    stats_mean = ['VG', 'CG']
    stats_std = ['CG']
    rolling_features_list = []
    fixture_indices = fixture_df.index
    for index in tqdm(fixture_indices, total=len(fixture_indices), desc="Calc. Rolling Futuro"):
        fm = fixture_df.loc[index]
        ht = fm.get('HomeTeam')
        at = fm.get('AwayTeam')
        mr = {'Index': index}
        for tp, tn in [('H', ht), ('A', at)]:
            if tn and tn in teams_in_hist:
                th = historical_df_processed[(historical_df_processed['Home'] == tn) | (historical_df_processed['Away'] == tn)].head(ROLLING_WINDOW)
                if not th.empty:
                    def get_v(r, b1, b2): return r.get(b1) if r['Home'] == tn else r.get(b2)
                    for sp in stats_mean:
                        b1, b2 = None, None
                        if sp == 'VG':
                            b1, b2 = 'VG_H_raw', 'VG_A_raw'
                        elif sp == 'CG':
                            b1, b2 = 'CG_H_raw', 'CG_A_raw'
                        if b1:
                            v = th.apply(lambda r: get_v(r, b1, b2), axis=1).dropna().tolist()
                            mr[f'Media_{sp}_{tp}'] = np.mean(v) if v else np.nan
                    for sp in stats_std:
                        b1, b2 = None, None
                        if sp == 'CG':
                            b1, b2 = 'CG_H_raw', 'CG_A_raw'
                        if b1:
                            v = th.apply(lambda r: get_v(r, b1, b2), axis=1).dropna().tolist()
                            mr[f'Std_{sp}_{tp}'] = np.std(v) if len(v) >= 2 else np.nan
                    ghc = GOALS_COLS.get('home', 'Goals_H_FT')
                    gac = GOALS_COLS.get('away', 'Goals_A_FT')
                    sv = th.apply(lambda r: r[ghc] if r['Home'] == tn else r[gac], axis=1).dropna().tolist()
                    avg_gs = np.mean(sv) if sv else np.nan
                    mr[f'Avg_Gols_Marcados_{tp}'] = avg_gs
                    cv = th.apply(lambda r: r[gac] if r['Home'] == tn else r[ghc], axis=1).dropna().tolist()
                    avg_gc = np.mean(cv) if cv else np.nan
                    mr[f'Avg_Gols_Sofridos_{tp}'] = avg_gc
                    if tp == 'H':
                         mr['FA_H'] = avg_gs / avg_h_league if pd.notna(avg_gs) else np.nan
                         mr['FD_H'] = avg_gc / avg_a_league if pd.notna(avg_gc) else np.nan
                    else:
                         mr['FA_A'] = avg_gs / avg_a_league if pd.notna(avg_gs) else np.nan
                         mr['FD_A'] = avg_gc / avg_h_league if pd.notna(avg_gc) else np.nan
            else:
                 for sp in stats_mean:
                     mr[f'Media_{sp}_{tp}'] = np.nan
                 for sp in stats_std:
                     mr[f'Std_{sp}_{tp}'] = np.nan
                 mr[f'Avg_Gols_Marcados_{tp}'] = np.nan
                 mr[f'Avg_Gols_Sofridos_{tp}'] = np.nan
                 mr[f'FA_{tp}'] = np.nan
                 mr[f'FD_{tp}'] = np.nan
        rolling_features_list.append(mr)
    
    df_rolling_features = pd.DataFrame(rolling_features_list).set_index('Index')
    logger.info(f"Rolling stats p/ futuro calculadas. Colunas: {list(df_rolling_features.columns)}")
    df_temp_fixtures = fixture_df.join(df_rolling_features, how='left')

    logger.info("Calculando probabilidades e binning para jogos futuros...")
    required_odds_future = list(ODDS_COLS.values())
    missing = [o for o in required_odds_future if o not in df_temp_fixtures.columns]
    if missing:
        logger.error(f"Odds ausentes futuro: {missing}")
        return None
    df_temp_fixtures = calculate_probabilities(df_temp_fixtures)
    df_temp_fixtures = calculate_normalized_probabilities(df_temp_fixtures)
    df_temp_fixtures = calculate_binned_features(df_temp_fixtures)
    df_temp_fixtures = calculate_poisson_draw_prob(
        df_temp_fixtures,
        avg_goals_home_league=avg_h_league,
        avg_goals_away_league=avg_a_league,
        max_goals=5
    )
    logger.info("Calculando features derivadas para jogos futuros...")
    df_temp_fixtures = calculate_derived_features(df_temp_fixtures)

    logger.info("Calculando features derivadas (CV_HDA, Diff_Media_CG) para jogos futuros...")
    df_temp_fixtures_derived = calculate_derived_features(df_temp_fixtures)

    logger.info("Selecionando features FINAIS p/ modelo e tratando NaNs...")
    final_features_for_model = feature_columns
    missing_final_future = [f for f in final_features_for_model if f not in df_temp_fixtures_derived.columns]
    if missing_final_future:
        logger.warning(f"Features do modelo ausentes: {missing_final_future}. Serão NaN.")
        for col in missing_final_future:
            df_temp_fixtures_derived[col] = np.nan

    X_fixture_prepared = df_temp_fixtures_derived[final_features_for_model].copy()
    logger.info(f"Shape APÓS seleção final: {X_fixture_prepared.shape}")

    nan_counts = X_fixture_prepared.isnull().sum()
    total_nans = nan_counts.sum()
    if total_nans > 0:
        logger.info(f"{total_nans} NaNs encontrados. Preenchendo com 0. Contagem por coluna: {nan_counts[nan_counts > 0]}")
        X_fixture_prepared = X_fixture_prepared.fillna(0)

    logger.info(f"--- Preparação Features Futuras OK. Shape final: {X_fixture_prepared.shape} ---")
    return X_fixture_prepared
