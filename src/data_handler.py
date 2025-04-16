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
import time # Embora não usado diretamente, pode ser útil para debugging
import warnings
from datetime import date, timedelta, datetime # datetime não usado diretamente, mas pode ser útil
import requests
from urllib.error import HTTPError, URLError
import logging
from scipy.stats import poisson


warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

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
        df = pd.read_excel(file_path); 
        print(f"Histórico carregado: {file_path} (Shape: {df.shape})")
        # Verifica colunas mínimas + as que usamos DIRETAMENTE como features
        required_excel_names = list(set(EXCEL_EXPECTED_COLS))
        missing = [col for col in required_excel_names if col not in df.columns]
        missing_extras = [m for m in missing if m in ['Odd_Over25_FT', 'Odd_BTTS_Yes']]
        missing_minimum = [m for m in missing if m not in ['Odd_Over25_FT', 'Odd_BTTS_Yes']]
        if missing_extras: 
            print(f"Aviso: Colunas odds extras ({missing_extras}) não encontradas no histórico.")
        if missing_minimum: 
            print(f"Erro CRÍTICO: Colunas mínimas faltando no Histórico: {missing_minimum}"); return None

        # Renomeia Gols para nomes internos
        df = df.rename(columns={GOALS_COLS['home']: 'Goals_H_FT', GOALS_COLS['away']: 'Goals_A_FT'})
        if 'Date' in df.columns: df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # Colunas para checar NaNs antes de sort (essenciais p/ cálculos intermediários)
        dropna_check_cols = ['Date', 'Home', 'Away', 'Goals_H_FT', 'Goals_A_FT'] + list(ODDS_COLS.values())
        dropna_check_cols_exist = [col for col in dropna_check_cols if col in df.columns]; 
        df = df.dropna(subset=dropna_check_cols_exist)
        if 'Date' in df.columns: 
            df = df.sort_values(by='Date').reset_index(drop=True)

        # Converte todas as colunas numéricas esperadas (incluindo extras)
        num_cols_convert = ['Goals_H_FT', 'Goals_A_FT'] + list(ODDS_COLS.values()) + ['Odd_Over25_FT', 'Odd_BTTS_Yes']
        for col in num_cols_convert:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        # Dropna SÓ para as essenciais para cálculos
        df = df.dropna(subset=list(ODDS_COLS.values()) + ['Goals_H_FT', 'Goals_A_FT'])
        print("Histórico carregado e colunas essenciais convertidas.")
        return df
    except FileNotFoundError: print(f"Erro CRÍTICO: Histórico NÃO ENCONTRADO: '{file_path}'"); return None
    except Exception as e: print(f"Erro inesperado ao carregar histórico: {e}"); return None


def calculate_probabilities(df: pd.DataFrame, epsilon=1e-6) -> pd.DataFrame:
    """Calcula probs implícitas p_H, p_D, p_A. Requer Odd_H/D/A_FT."""
    df_calc = df.copy()
    required_odds = list(ODDS_COLS.values()) # Usa config
    if not all(c in df_calc.columns for c in required_odds):
        print("Aviso: Odds 1x2 ausentes para calcular Probabilidades.")
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
    print("  Probabilidades Implícitas (p_H, p_D, p_A) calculadas.")
    return df_calc

def calculate_normalized_probabilities(df: pd.DataFrame, epsilon=1e-6) -> pd.DataFrame:
    """Calcula probs normalizadas e diferença H/A. Requer p_H, p_D, p_A."""
    df_calc = df.copy()
    prob_cols = ['p_H', 'p_D', 'p_A']
    if not all(c in df_calc.columns for c in prob_cols):
        print("Aviso: Probabilidades (p_H/D/A) ausentes para normalização.")
        df_calc[['p_H_norm', 'p_D_norm', 'p_A_norm', 'abs_ProbDiff_Norm']] = np.nan
        return df_calc

    df_calc['Overround'] = df_calc['p_H'] + df_calc['p_D'] + df_calc['p_A']
    # Evitar divisão por zero no Overround
    df_calc['Overround'] = df_calc['Overround'].replace(0, epsilon)

    df_calc['p_H_norm'] = df_calc['p_H'] / df_calc['Overround']
    df_calc['p_D_norm'] = df_calc['p_D'] / df_calc['Overround']
    df_calc['p_A_norm'] = df_calc['p_A'] / df_calc['Overround']
    df_calc['abs_ProbDiff_Norm'] = abs(df_calc['p_H_norm'] - df_calc['p_A_norm'])
    print("  Probabilidades Normalizadas (p_X_norm, abs_ProbDiff_Norm) calculadas.")
    return df_calc.drop(columns=['Overround'], errors='ignore') # Limpa coluna auxiliar

def calculate_rolling_std(df: pd.DataFrame, stats_to_calc: List[str], window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Calcula desvio padrão móvel para as estatísticas especificadas."""
    # Esta função será MUITO similar a calculate_rolling_stats
    # A principal diferença é usar np.std() em vez de np.mean()
    df_calc = df.copy();
    teams = pd.concat([df_calc['Home'], df_calc['Away']]).unique();
    team_history: Dict[str, Dict[str, List[float]]] = {team: {stat: [] for stat in stats_to_calc} for team in teams};
    results_list = [];
    rolling_cols_map = {};
    cols_to_calculate = {} # Quais colunas de desvio padrão REALMENTE calcular

    print(f"  Iniciando cálculo Desvio Padrão Rolling (Janela={window})...")
    # Define mapeamentos e verifica quais colunas calcular
    for stat_prefix in stats_to_calc:
        std_col_h = f'Std_{stat_prefix}_H'; std_col_a = f'Std_{stat_prefix}_A'
        skip_h = skip_a = False
        # Verifica se já existe e é numérico
        if std_col_h in df_calc.columns and pd.api.types.is_numeric_dtype(df_calc[std_col_h]): skip_h = True
        if std_col_a in df_calc.columns and pd.api.types.is_numeric_dtype(df_calc[std_col_a]): skip_a = True

        if skip_h and skip_a: print(f"    Aviso: {std_col_h}/{std_col_a} já existem."); continue

        # Define colunas base ( PRECISAM EXISTIR APÓS calculate_historical_intermediate )
        if stat_prefix == 'Ptos': base_h, base_a = 'Ptos_H', 'Ptos_A'
        elif stat_prefix == 'VG': base_h, base_a = 'VG_H_raw', 'VG_A_raw'
        elif stat_prefix == 'CG': base_h, base_a = 'CG_H_raw', 'CG_A_raw'
        else: print(f"    Aviso: Prefixo StDev '{stat_prefix}' desconhecido."); continue

        if base_h not in df_calc.columns or base_a not in df_calc.columns:
            print(f"    Erro StDev: Colunas base '{base_h}'/'{base_a}' não encontradas."); continue

        rolling_cols_map[stat_prefix] = {'home': base_h, 'away': base_a}
        if not skip_h: cols_to_calculate[stat_prefix + '_H'] = std_col_h
        if not skip_a: cols_to_calculate[stat_prefix + '_A'] = std_col_a

    if not cols_to_calculate:
        print("    Nenhum StDev Rolling novo a calcular.")
        return df_calc

    print(f"    Calculando StDev rolling para: {list(cols_to_calculate.keys())}")

    calculated_stats = []
    # Itera pelas linhas do DataFrame
    for index, row in tqdm(df_calc.iterrows(), total=len(df_calc), desc="Calc. Rolling StDev"):
        home_team = row['Home']; away_team = row['Away']
        current_match_features = {'Index': index}

        # Calcula std para time da casa para cada stat necessária
        for stat_prefix, base_cols in rolling_cols_map.items():
             std_col_h = f'Std_{stat_prefix}_H'
             if stat_prefix + '_H' in cols_to_calculate: # Calcula apenas se necessário
                  hist_H = team_history[home_team][stat_prefix]
                  recent = hist_H[-window:]
                  # Calcula std apenas se houver pelo menos 2 pontos de dados (std de 1 ponto não é definido)
                  current_match_features[std_col_h] = np.std(recent) if len(recent) >= 2 else np.nan # Ou 0 ? NaN parece melhor

        # Calcula std para time visitante
        for stat_prefix, base_cols in rolling_cols_map.items():
            std_col_a = f'Std_{stat_prefix}_A'
            if stat_prefix + '_A' in cols_to_calculate:
                hist_A = team_history[away_team][stat_prefix]
                recent = hist_A[-window:]
                current_match_features[std_col_a] = np.std(recent) if len(recent) >= 2 else np.nan

        calculated_stats.append(current_match_features)

        # Atualiza histórico DEPOIS de calcular para a linha atual
        for stat_prefix, base_cols in rolling_cols_map.items():
            # Adiciona valor base ao histórico *se não for NaN*
            if pd.notna(row[base_cols['home']]): team_history[home_team][stat_prefix].append(row[base_cols['home']])
            if pd.notna(row[base_cols['away']]): team_history[away_team][stat_prefix].append(row[base_cols['away']])

    # Cria DataFrame com os resultados e junta ao original
    df_rolling_stdev = pd.DataFrame(calculated_stats).set_index('Index')

    # Junta apenas as colunas que foram realmente calculadas
    cols_to_join = [col for col in cols_to_calculate.values() if col in df_rolling_stdev.columns]
    print(f"  StDev Rolling calculado. Colunas adicionadas: {cols_to_join}")
    df_final = df_calc.join(df_rolling_stdev[cols_to_join]) if cols_to_join else df_calc
    return df_final

def calculate_rolling_stats(df: pd.DataFrame, stats_to_calc: List[str], window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Calcula médias móveis para as estatísticas especificadas."""
    df_calc = df.copy();
    teams = pd.concat([df_calc['Home'], df_calc['Away']]).unique();
    team_history: Dict[str, Dict[str, List[float]]] = {team: {stat: [] for stat in stats_to_calc} for team in teams};
    results_list = [];
    rolling_cols_map = {};
    cols_to_calculate = {} # Quais colunas de médias REALMENTE calcular

    print(f"  Iniciando cálculo Médias Rolling (Janela={window})...")
    # Define mapeamentos e verifica quais colunas calcular
    for stat_prefix in stats_to_calc:
        media_col_h = f'Media_{stat_prefix}_H'; media_col_a = f'Media_{stat_prefix}_A'
        skip_h = skip_a = False
        # Verifica se já existe e é numérico
        if media_col_h in df_calc.columns and pd.api.types.is_numeric_dtype(df_calc[media_col_h]): skip_h = True
        if media_col_a in df_calc.columns and pd.api.types.is_numeric_dtype(df_calc[media_col_a]): skip_a = True

        if skip_h and skip_a: print(f"    Aviso: {media_col_h}/{media_col_a} já existem."); continue

        # Define colunas base ( PRECISAM EXISTIR APÓS calculate_historical_intermediate )
        if stat_prefix == 'Ptos': base_h, base_a = 'Ptos_H', 'Ptos_A'
        elif stat_prefix == 'VG': base_h, base_a = 'VG_H_raw', 'VG_A_raw'
        elif stat_prefix == 'CG': base_h, base_a = 'CG_H_raw', 'CG_A_raw'
        else: print(f"    Aviso: Prefixo Média '{stat_prefix}' desconhecido."); continue

        if base_h not in df_calc.columns or base_a not in df_calc.columns:
            print(f"    Erro Média: Colunas base '{base_h}'/'{base_a}' não encontradas."); continue

        rolling_cols_map[stat_prefix] = {'home': base_h, 'away': base_a}
        if not skip_h: cols_to_calculate[stat_prefix + '_H'] = media_col_h
        if not skip_a: cols_to_calculate[stat_prefix + '_A'] = media_col_a

    if not cols_to_calculate:
        print("    Nenhuma Média Rolling nova a calcular.")
        return df_calc

    print(f"    Calculando Médias rolling para: {list(cols_to_calculate.keys())}")

    calculated_stats = []
    # Itera pelas linhas do DataFrame
    for index, row in tqdm(df_calc.iterrows(), total=len(df_calc), desc="Calc. Rolling Médias"):
        home_team = row['Home']; away_team = row['Away']
        current_match_features = {'Index': index}

        # Calcula média para time da casa para cada stat necessária
        for stat_prefix, base_cols in rolling_cols_map.items():
             media_col_h = f'Media_{stat_prefix}_H'
             if stat_prefix + '_H' in cols_to_calculate: # Calcula apenas se necessário
                  hist_H = team_history[home_team][stat_prefix]
                  recent = hist_H[-window:]
                  # Calcula média apenas se houver dados recentes
                  current_match_features[media_col_h] = np.mean(recent) if len(recent) > 0 else np.nan # Média!

        # Calcula média para time visitante
        for stat_prefix, base_cols in rolling_cols_map.items():
            media_col_a = f'Media_{stat_prefix}_A'
            if stat_prefix + '_A' in cols_to_calculate:
                hist_A = team_history[away_team][stat_prefix]
                recent = hist_A[-window:]
                current_match_features[media_col_a] = np.mean(recent) if len(recent) > 0 else np.nan # Média!

        calculated_stats.append(current_match_features)

        # Atualiza histórico DEPOIS de calcular para a linha atual
        for stat_prefix, base_cols in rolling_cols_map.items():
            # Adiciona valor base ao histórico *se não for NaN*
            if pd.notna(row[base_cols['home']]): team_history[home_team][stat_prefix].append(row[base_cols['home']])
            if pd.notna(row[base_cols['away']]): team_history[away_team][stat_prefix].append(row[base_cols['away']])

    # Cria DataFrame com os resultados e junta ao original
    df_rolling_means = pd.DataFrame(calculated_stats).set_index('Index')

    # Junta apenas as colunas que foram realmente calculadas
    cols_to_join = [col for col in cols_to_calculate.values() if col in df_rolling_means.columns]
    print(f"  Médias Rolling calculadas. Colunas adicionadas: {cols_to_join}")
    df_final = df_calc.join(df_rolling_means[cols_to_join]) if cols_to_join else df_calc
    return df_final

def calculate_binned_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features categóricas (bins). Requer Odd_D_FT."""
    df_calc = df.copy()
    odd_d_col = ODDS_COLS['draw'] # Usa config
    if odd_d_col not in df_calc.columns:
        print("Aviso: Odd de Empate ausente para binning.")
        df_calc['Odd_D_Cat'] = np.nan
        return df_calc

    odd_d = pd.to_numeric(df_calc[odd_d_col], errors='coerce')

    # Define os limites dos bins (ajuste conforme necessário)
    bins = [-np.inf, 2.90, 3.40, np.inf]
    # Define os rótulos para os bins (1, 2, 3)
    labels = [1, 2, 3]

    df_calc['Odd_D_Cat'] = pd.cut(odd_d, bins=bins, labels=labels, right=True) # right=True significa que 2.90 cai no bin 1
    df_calc['Odd_D_Cat'] = df_calc['Odd_D_Cat'].cat.codes + 1 # Converte para códigos numéricos (1, 2, 3), tratando NaN como 0. Adiciona 1.
    df_calc['Odd_D_Cat'] = df_calc['Odd_D_Cat'].replace(0, np.nan) # Restaura NaN onde era NaN

    print(f"  Binning ('Odd_D_Cat') calculado a partir de '{odd_d_col}'.")
    return df_calc


def calculate_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula CV_HDA e Diff_Media_CG no DataFrame fornecido."""
    df_calc = df.copy(); print("  Calculando features derivadas (CV_HDA, Diff_Media_CG)..."); epsilon = 1e-6
    if all(c in df_calc.columns for c in ODDS_COLS.values()):
        odds_matrix = df_calc[list(ODDS_COLS.values())]
        mean_odds = odds_matrix.mean(axis=1); std_odds = odds_matrix.std(axis=1)
        df_calc['CV_HDA'] = std_odds.div(mean_odds).fillna(0); df_calc.loc[mean_odds <= epsilon, 'CV_HDA'] = 0
    else: 
        print("Aviso: Odds 1x2 ausentes p/ CV_HDA."); 
        df_calc['CV_HDA'] = np.nan
    if 'Media_CG_H' in df_calc.columns and 'Media_CG_A' in df_calc.columns:
        df_calc['Diff_Media_CG'] = df_calc['Media_CG_H'] - df_calc['Media_CG_A']
    else: print("Aviso: Médias CG ausentes p/ Diff_Media_CG."); df_calc['Diff_Media_CG'] = np.nan
    return df_calc

def calculate_poisson_draw_prob(df: pd.DataFrame, max_goals: int = 6) -> pd.DataFrame:
    """
    Calcula a probabilidade de empate usando a distribuição de Poisson.
    Requer as colunas de média de gols: Avg_Gols_Marcados_H/A, Avg_Gols_Sofridos_H/A.
    """
    df_calc = df.copy()
    required_cols = ['Avg_Gols_Marcados_H', 'Avg_Gols_Sofridos_A', # Ataque H vs Defesa A
                     'Avg_Gols_Marcados_A', 'Avg_Gols_Sofridos_H'] # Ataque A vs Defesa H

    if not all(c in df_calc.columns for c in required_cols):
        logging.warning("Colunas de média de gols ausentes para cálculo Poisson. Pulando.")
        df_calc['Prob_Empate_Poisson'] = np.nan
        return df_calc

    logging.info("  Calculando Probabilidade de Empate (Poisson)...")

    # 1. Estimar Gols Esperados (Lambda) para cada time no jogo
    #    Modelo simples: lambda_H = Média Gols Marcados H * Média Gols Sofridos A (ajustado pela média da liga se disponível, mas vamos simplificar por agora)
    #    lambda_A = Média Gols Marcados A * Média Gols Sofridos H
    #    É crucial garantir que as médias não sejam NaN ou zero. Preenche com um valor pequeno.
    lambda_h = (pd.to_numeric(df_calc['Avg_Gols_Marcados_H'], errors='coerce').fillna(0.1) *
                pd.to_numeric(df_calc['Avg_Gols_Sofridos_A'], errors='coerce').fillna(0.1))
    lambda_a = (pd.to_numeric(df_calc['Avg_Gols_Marcados_A'], errors='coerce').fillna(0.1) *
                pd.to_numeric(df_calc['Avg_Gols_Sofridos_H'], errors='coerce').fillna(0.1))
    # Evita lambdas muito baixas (pode causar underflow) ou negativas
    lambda_h = np.maximum(lambda_h, 1e-6)
    lambda_a = np.maximum(lambda_a, 1e-6)


    # 2. Calcular Probabilidade de Empate (Soma de P(k-k) para k=0 até max_goals)
    prob_empate_total = pd.Series(0.0, index=df_calc.index) # Inicializa com zero

    try:
        for k in range(max_goals + 1):
            # P(Gols Casa = k) * P(Gols Fora = k)
            prob_placar_kk = poisson.pmf(k, lambda_h) * poisson.pmf(k, lambda_a)
            prob_empate_total += prob_placar_kk
    except Exception as e_poisson:
        logging.error(f"Erro durante cálculo das probabilidades Poisson: {e_poisson}", exc_info=True)
        # Define como NaN se o cálculo falhar
        df_calc['Prob_Empate_Poisson'] = np.nan
        return df_calc

    df_calc['Prob_Empate_Poisson'] = prob_empate_total
    logging.info("  -> Prob_Empate_Poisson calculado.")

    return df_calc
# --- PIPELINE DE TREINAMENTO (Histórico) ---

def calculate_historical_intermediate(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula FT_Result, IsDraw, Ptos, Probs, VG/CG raw no DataFrame."""
    df_calc = df.copy()
    print("  Calculando stats intermediárias (Resultado, Pontos, VG/CG raw)...")
    epsilon = 1e-6

    # Usa nomes das colunas de gols do config.py
    goals_h_col = GOALS_COLS.get('home', 'Goals_H_FT') # Default se config não tiver
    goals_a_col = GOALS_COLS.get('away', 'Goals_A_FT')

    # --- Cálculo de Resultado e Pontos ---
    if goals_h_col in df_calc.columns and goals_a_col in df_calc.columns:
        # Garante que gols sejam numéricos, tratando erros
        h_goals = pd.to_numeric(df_calc[goals_h_col], errors='coerce')
        a_goals = pd.to_numeric(df_calc[goals_a_col], errors='coerce')

        df_calc['FT_Result'] = np.select(
            [h_goals > a_goals, h_goals == a_goals],
            ["H", "D"],
            default="A"
        )
        # Define IsDraw baseado no resultado calculado
        df_calc['IsDraw'] = (df_calc['FT_Result'] == 'D').astype(int)

        # Calcula Pontos baseado no resultado
        df_calc['Ptos_H'] = np.select(
            [df_calc['FT_Result'] == 'H', df_calc['FT_Result'] == 'D'],
            [3, 1], default=0
        )
        df_calc['Ptos_A'] = np.select(
            [df_calc['FT_Result'] == 'A', df_calc['FT_Result'] == 'D'],
            [3, 1], default=0
        )
        
        print("    -> Resultado (FT_Result, IsDraw) e Pontos (Ptos_H/A) calculados.")
    else:
        print(f"    Aviso: Colunas de Gols ('{goals_h_col}', '{goals_a_col}') não encontradas. Resultado/Pontos não calculados.")
        df_calc[['FT_Result', 'IsDraw', 'Ptos_H', 'Ptos_A']] = np.nan


    # --- Cálculo de Probabilidades Implícitas (pode ser redundante se calculate_probabilities for chamada depois) ---
    #    Mas é bom ter aqui para VG/CG raw dependerem apenas desta função
    required_odds = list(ODDS_COLS.values())
    if all(c in df_calc.columns for c in required_odds):
        if not all(p in df_calc.columns for p in ['p_H', 'p_D', 'p_A']): # Calcula só se já não existirem
             print("    Calculando probabilidades implícitas (p_H/D/A)...")
             for col in required_odds: df_calc[col] = pd.to_numeric(df_calc[col], errors='coerce')
             odd_h = df_calc[ODDS_COLS['home']].replace(0, epsilon)
             odd_d = df_calc[ODDS_COLS['draw']].replace(0, epsilon)
             odd_a = df_calc[ODDS_COLS['away']].replace(0, epsilon)
             df_calc['p_H'] = (1 / odd_h).fillna(np.nan) # Usar NaN se odd for NaN
             df_calc['p_D'] = (1 / odd_d).fillna(np.nan)
             df_calc['p_A'] = (1 / odd_a).fillna(np.nan)
        else:
            print("    -> Probabilidades implícitas (p_H/D/A) já existem.")
    else:
        print("    Aviso: Odds 1x2 ausentes para calcular Probabilidades.")
        df_calc[['p_H', 'p_D', 'p_A']] = np.nan # Garante que colunas existem como NaN se odds faltarem

    # --- Cálculo de VG/CG Raw ---
    prob_cols_needed = ['p_H', 'p_A']
    goal_cols_needed = [goals_h_col, goals_a_col]
    if all(c in df_calc.columns for c in prob_cols_needed + goal_cols_needed):
         # Garante que gols e probs são numéricos
         h_goals = pd.to_numeric(df_calc[goals_h_col], errors='coerce')
         a_goals = pd.to_numeric(df_calc[goals_a_col], errors='coerce')
         p_H = pd.to_numeric(df_calc['p_H'], errors='coerce')
         p_A = pd.to_numeric(df_calc['p_A'], errors='coerce')

         # VG raw - Requer Gols e Prob do Oponente
         df_calc['VG_H_raw'] = h_goals * p_A
         df_calc['VG_A_raw'] = a_goals * p_H

         # CG raw - Requer Gols e Prob Própria. Cuidado com divisão por zero.
         df_calc['CG_H_raw'] = np.where(h_goals > 0, p_H / h_goals, np.nan) # NaN se 0 gols
         df_calc['CG_A_raw'] = np.where(a_goals > 0, p_A / a_goals, np.nan) # NaN se 0 gols
         print("    -> Valor/Custo do Gol (VG/CG raw) calculados.")
    else:
         print("    Aviso: Colunas de Gols ou Probabilidades ausentes para calcular VG/CG raw.")
         df_calc[['VG_H_raw', 'VG_A_raw', 'CG_H_raw', 'CG_A_raw']] = np.nan

    print("  Cálculo de stats intermediárias concluído.")
    return df_calc

def calculate_rolling_goal_stats(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Calcula médias móveis de gols marcados/sofridos em casa/fora."""
    df_calc = df.copy()
    goals_h_col = GOALS_COLS.get('home', 'Goals_H_FT')
    goals_a_col = GOALS_COLS.get('away', 'Goals_A_FT')

    # Verifica se colunas de gols existem
    if goals_h_col not in df_calc.columns or goals_a_col not in df_calc.columns:
        logging.warning("Colunas de Gols não encontradas para calcular médias de gols.")
        # Adiciona colunas com NaN se não puder calcular
        df_calc[['Avg_Gols_Marcados_H', 'Avg_Gols_Sofridos_H', 'Avg_Gols_Marcados_A', 'Avg_Gols_Sofridos_A']] = np.nan
        return df_calc

    teams = pd.concat([df_calc['Home'], df_calc['Away']]).unique()
    # Histórico separado para gols marcados e sofridos
    team_history: Dict[str, Dict[str, List[float]]] = {
        team: {'scored_home': [], 'conceded_home': [], 'scored_away': [], 'conceded_away': []}
        for team in teams
    }
    results_list = []

    logging.info(f"  Calculando Médias Rolling de Gols (Janela={window})...")

    # Itera pelas linhas do DataFrame
    for index, row in tqdm(df_calc.iterrows(), total=len(df_calc), desc="Calc. Rolling Gols"):
        home_team = row['Home']
        away_team = row['Away']
        current_match_stats = {'Index': index}

        # Gols Marcados Casa (média dos últimos 'window' jogos EM CASA)
        h_scored_hist = team_history[home_team]['scored_home']
        recent_h_scored = h_scored_hist[-window:]
        current_match_stats['Avg_Gols_Marcados_H'] = np.mean(recent_h_scored) if recent_h_scored else np.nan

        # Gols Sofridos Casa (média dos últimos 'window' jogos EM CASA)
        h_conceded_hist = team_history[home_team]['conceded_home']
        recent_h_conceded = h_conceded_hist[-window:]
        current_match_stats['Avg_Gols_Sofridos_H'] = np.mean(recent_h_conceded) if recent_h_conceded else np.nan

        # Gols Marcados Fora (média dos últimos 'window' jogos FORA)
        a_scored_hist = team_history[away_team]['scored_away']
        recent_a_scored = a_scored_hist[-window:]
        current_match_stats['Avg_Gols_Marcados_A'] = np.mean(recent_a_scored) if recent_a_scored else np.nan

        # Gols Sofridos Fora (média dos últimos 'window' jogos FORA)
        a_conceded_hist = team_history[away_team]['conceded_away']
        recent_a_conceded = a_conceded_hist[-window:]
        current_match_stats['Avg_Gols_Sofridos_A'] = np.mean(recent_a_conceded) if recent_a_conceded else np.nan

        results_list.append(current_match_stats)

        # Atualiza histórico DEPOIS de calcular para a linha atual
        home_goals = pd.to_numeric(row[goals_h_col], errors='coerce')
        away_goals = pd.to_numeric(row[goals_a_col], errors='coerce')

        if pd.notna(home_goals):
             team_history[home_team]['scored_home'].append(home_goals)
             team_history[away_team]['conceded_away'].append(home_goals) # Gols do H são sofridos pelo A
        if pd.notna(away_goals):
             team_history[away_team]['scored_away'].append(away_goals)
             team_history[home_team]['conceded_home'].append(away_goals) # Gols do A são sofridos pelo H

    # Cria DataFrame com os resultados e junta ao original
    df_rolling_goals = pd.DataFrame(results_list).set_index('Index')
    logging.info(f"  Médias Rolling de Gols calculadas. Colunas: {list(df_rolling_goals.columns)}")
    return df_calc.join(df_rolling_goals)

def preprocess_and_feature_engineer(df_loaded: pd.DataFrame) -> Optional[Tuple[pd.DataFrame, pd.Series, List[str]]]:
    """Pipeline BackDraw: Calcula/Verifica TODAS as features NOVAS no histórico."""
    if df_loaded is None: return None
    print("\n--- Iniciando Pré-processamento e Engenharia de Features (Histórico) ---")

    # PASSO 1: Calcular Intermediárias PRIMEIRO!
    # Isso cria FT_Result, IsDraw, Ptos_H/A, VG_H/A_raw, CG_H/A_raw, e p_H/D/A (se não existirem)
    df_interm = calculate_historical_intermediate(df_loaded)
    if 'IsDraw' not in df_interm.columns: # Verifica se o alvo foi criado
         print("Erro CRÍTICO: Coluna alvo 'IsDraw' não foi criada por calculate_historical_intermediate.")
         return None
    if not all(p in df_interm.columns for p in ['p_H', 'p_D', 'p_A']): 
        df_interm = calculate_probabilities(df_interm) # Garante probs

    # PASSO 2: Calcular Probabilidades Normalizadas (Usa p_H/D/A de df_interm)
    # Cria p_H/D/A_norm, abs_ProbDiff_Norm
    df_probs_norm = calculate_normalized_probabilities(df_interm)

    # PASSO 3: Calcular Médias Rolling (Usa VG/CG_raw de df_interm)
    # Cria Media_VG_H/A, Media_CG_H/A
    stats_to_roll_mean = ['VG', 'CG']
    df_rolling_mean = calculate_rolling_stats(df_probs_norm, stats_to_roll_mean, window=ROLLING_WINDOW)

    # PASSO 4: Calcular Desvio Padrão Rolling (Usa CG_raw de df_interm/df_rolling_mean)
    # Cria Std_CG_H/A
    stats_to_roll_std = ['CG']
    df_rolling_std = calculate_rolling_std(df_rolling_mean, stats_to_roll_std, window=ROLLING_WINDOW)

    # PASSO 5: Calcular Médias Rolling de Gols (Usa FT_Result de df_interm)
    df_goals = calculate_rolling_goal_stats(df_rolling_std, window=ROLLING_WINDOW)

    df_poisson = calculate_poisson_draw_prob(df_goals, max_goals=5) # Calcula P(0-0) até P(5-5)

    # PASSO 6: Calcular Features de Binning (Usa Odd_D_FT de df_rolling_std)
    # Cria Odd_D_Cat
    df_binned = calculate_binned_features(df_poisson)

    # PASSO 6.5: Calcular Features Derivadas
    logging.info("  Calculando features derivadas...")
    df_derived = calculate_derived_features(df_binned) # Calcula CV_HDA, Diff_Media_CG

    # PASSO 7:selecionar Features Finais e Tratar NaNs
    df_final = df_derived # O resultado do último passo contém todas as features
    target_col = 'IsDraw'
    print("  Selecionando features finais e tratando NaNs...")

    # Usa FEATURE_COLUMNS importado do config (garanta que config.py foi atualizado)
    required_final_cols = FEATURE_COLUMNS + [target_col]

    # Verifica se todas as colunas FINAIS existem AGORA
    missing_final = [f for f in required_final_cols if f not in df_final.columns]
    if missing_final:
        print(f"Erro CRÍTICO: Colunas finais ausentes após todos cálculos: {missing_final}")
        print(f"Colunas disponíveis: {list(df_final.columns)}")
        return None

    # Seleciona apenas as colunas finais e o alvo
    df_final_selection = df_final[required_final_cols].copy()

    # Tratamento de NaNs
    initial_rows = len(df_final_selection)
    df_final_selection = df_final_selection.dropna() # Remove linhas com NaN em QUALQUER feature final ou no alvo
    rows_dropped = initial_rows - len(df_final_selection)
    if rows_dropped > 0:
        print(f"  Removidas {rows_dropped} linhas contendo NaNs nas features finais ou alvo.")

    if df_final_selection.empty:
        print("Erro: Nenhuma linha restante após remover NaNs.")
        return None

    X = df_final_selection[FEATURE_COLUMNS]
    y = df_final_selection[target_col]

    print(f"--- Pré-processamento e Engenharia (Histórico) OK ---")
    print(f"    Shape X final: {X.shape}, Shape y final: {y.shape}")
    print(f"    Features finais usadas: {list(X.columns)}")
    return X, y, FEATURE_COLUMNS # Retorna a lista de features usada

# --- Buscar e Processar Jogos Futuros (CSV) ---
def fetch_and_process_fixtures() -> Optional[pd.DataFrame]:
    if FIXTURE_FETCH_DAY == "tomorrow": target_date = date.today() + timedelta(days=1)
    else:   
        target_date = date.today()
    date_str = target_date.strftime('%Y-%m-%d'); 
    fixture_url = FIXTURE_CSV_URL_TEMPLATE.format(date_str=date_str)
    print(f"\nBuscando jogos {FIXTURE_FETCH_DAY} ({date_str}): {fixture_url}")
    try: 
        response = requests.head(fixture_url, allow_redirects=True, timeout=10); 
        response.raise_for_status(); print("Arquivo encontrado. Baixando..."); 
        df_fix = pd.read_csv(fixture_url); print(f"CSV baixado. Shape: {df_fix.shape}")
    except Exception as e_load: print(f"Erro buscar/ler CSV: {e_load}"); return None
    try:
        print("Processando CSV..."); 
        cols_to_keep = list(FIXTURE_CSV_COL_MAP.keys()); 
        cols_exist_in_df = [c for c in cols_to_keep if c in df_fix.columns];
        if not cols_exist_in_df: print("Erro: Nenhuma coluna esperada no CSV."); return None
        df_processed = df_fix[cols_exist_in_df].copy(); 
        df_processed.rename(columns=FIXTURE_CSV_COL_MAP, inplace=True)
        current_required_fixture_cols = REQUIRED_FIXTURE_COLS; 
        missing_required = [c for c in current_required_fixture_cols if c not in df_processed.columns]
        if missing_required: print(f"Erro: Colunas essenciais pós-map ausentes no CSV: {missing_required}"); return None
        if TARGET_LEAGUES:
            initial_count = len(df_processed); 
            df_processed = df_processed[df_processed['League'].astype(str).isin(TARGET_LEAGUES)]
            print(f"Filtro ligas: {len(df_processed)} de {initial_count} jogos.");
            if df_processed.empty: return df_processed
        df_processed.dropna(subset=current_required_fixture_cols, inplace=True); 
        df_processed.reset_index(inplace=True, drop=True); 
        print(f"Processamento CSV OK. Shape: {df_processed.shape}")
        return df_processed
    except Exception as e_proc: print(f"Erro processamento CSV: {e_proc}"); return None


def prepare_fixture_data(fixture_df: pd.DataFrame, historical_df: pd.DataFrame, feature_columns: List[str]) -> Optional[pd.DataFrame]:
    """Prepara features para jogos futuros, incluindo médias de gols E DERIVADAS.""" # Docstring atualizada
    if fixture_df is None or historical_df is None or not feature_columns: return None
    if historical_df.empty: print("Erro Prep Fixture: Histórico vazio."); return None
    if fixture_df.empty: print("Prep Fixture: Nenhum jogo futuro."); return pd.DataFrame(columns=feature_columns)

    print("\n--- Preparando Features Finais para Jogos Futuros (v3 - com Derivadas) ---") # Log atualizado
    print(f"    Jogos futuros brutos: {fixture_df.shape}")
    print(f"    Features finais esperadas p/ modelo: {feature_columns}")

    # --- Etapa 1: Calcular Médias/StDev/Gols Rolling (dependem do HISTÓRICO) ---
    print("  Processando histórico para cálculo de rolling stats...")
    # ... (código para processar histórico, calcular e juntar rolling stats como antes) ...
    # (Garantir que o resultado seja df_temp_fixtures com odds e rolling stats)
    start_time=time.time();historical_df_processed=calculate_historical_intermediate(historical_df.copy());required_hist_cols={'Home','Away','Date','VG_H_raw','VG_A_raw','CG_H_raw','CG_A_raw',GOALS_COLS.get('home','Goals_H_FT'),GOALS_COLS.get('away','Goals_A_FT')};missing_hist=[c for c in required_hist_cols if c not in historical_df_processed.columns];
    if missing_hist:print(f"Erro Prep Fixture: Colunas raw ausentes histórico:{missing_hist}");return None;
    historical_df_processed=historical_df_processed.sort_values(by='Date',ascending=False);teams_in_hist=pd.concat([historical_df_processed['Home'],historical_df_processed['Away']]).unique();print(f"  Histórico processado em {time.time()-start_time:.2f} seg.")
    print("  Calculando médias/StDev/Gols rolling...");stats_to_roll_mean=['VG','CG'];stats_to_roll_std=['CG'];rolling_features_list=[];fixture_indices=fixture_df.index;
    for index in tqdm(fixture_indices, total=len(fixture_indices), desc="Calc. Rolling Futuro"):
        fm=fixture_df.loc[index];ht=fm.get('HomeTeam');at=fm.get('AwayTeam');mr={'Index':index};
        for tp, tn in[('H',ht),('A',at)]:
            if tn and tn in teams_in_hist:
                th=historical_df_processed[ (historical_df_processed['Home']==tn)|(historical_df_processed['Away']==tn) ].head(ROLLING_WINDOW);
                if not th.empty:
                    def get_v(r,b1,b2): return r.get(b1) if r['Home']==tn else r.get(b2);
                    for sp in stats_to_roll_mean:b1,b2=None,None;
                    if sp=='VG':b1,b2='VG_H_raw','VG_A_raw';
                    elif sp=='CG':b1,b2='CG_H_raw','CG_A_raw';
                    if b1:v=th.apply(lambda r:get_v(r,b1,b2),axis=1).dropna().tolist();mr[f'Media_{sp}_{tp}']=np.mean(v) if v else np.nan;
                    for sp in stats_to_roll_std:b1,b2=None,None;
                    if sp=='CG':b1,b2='CG_H_raw','CG_A_raw';
                    if b1:v=th.apply(lambda r:get_v(r,b1,b2),axis=1).dropna().tolist();mr[f'Std_{sp}_{tp}']=np.std(v) if len(v)>=2 else np.nan;
                    ghc=GOALS_COLS.get('home','Goals_H_FT');gac=GOALS_COLS.get('away','Goals_A_FT');sv=th.apply(lambda r:r[ghc] if r['Home']==tn else r[gac],axis=1).dropna().tolist();mr[f'Avg_Gols_Marcados_{tp}']=np.mean(sv) if sv else np.nan;cv=th.apply(lambda r:r[gac] if r['Home']==tn else r[ghc],axis=1).dropna().tolist();mr[f'Avg_Gols_Sofridos_{tp}']=np.mean(cv) if cv else np.nan;
            else: # Default NaNs se time sem histórico
                for sp in stats_to_roll_mean:mr[f'Media_{sp}_{tp}']=np.nan;
                for sp in stats_to_roll_std:mr[f'Std_{sp}_{tp}']=np.nan;
                mr[f'Avg_Gols_Marcados_{tp}']=np.nan;mr[f'Avg_Gols_Sofridos_{tp}']=np.nan;
        rolling_features_list.append(mr);
    df_rolling_features=pd.DataFrame(rolling_features_list).set_index('Index');print(f"  Rolling stats calculadas. Cols:{list(df_rolling_features.columns)}")
    df_temp_fixtures=fixture_df.join(df_rolling_features,how='left');print(f"  Shape após juntar rolling: {df_temp_fixtures.shape}")
    # --- Fim Etapa 1 ---

    # --- Etapa 2: Calcular Features Derivadas das Odds (Probs, Binning) ---
    print("  Calculando probabilidades e binning para jogos futuros...")
    required_odds_future = list(ODDS_COLS.values()); missing = [o for o in required_odds_future if o not in df_temp_fixtures.columns]
    if missing: print(f"Erro: Odds ausentes futuro: {missing}"); return None # Ou preencher com NaN
    df_temp_fixtures = calculate_probabilities(df_temp_fixtures)
    df_temp_fixtures = calculate_normalized_probabilities(df_temp_fixtures)
    df_temp_fixtures = calculate_binned_features(df_temp_fixtures)

    logging.info("  Calculando Prob Empate Poisson para jogos futuros...")
    # Usa df_temp_fixtures que tem as Avg_Gols_...
    df_temp_fixtures = calculate_poisson_draw_prob(df_temp_fixtures, max_goals=5)
    logging.info("  Calculando features derivadas para jogos futuros...")
    # Usa df_temp_fixtures que tem odds e médias (calculadas no passo 1)
    df_temp_fixtures = calculate_derived_features(df_temp_fixtures)

    # --- Fim Etapa 2 ---

    # --- **NOVO**: Etapa 2.5: Calcular Features Derivadas (CV_HDA, Diff_Media_CG) ---
    logging.info("  Calculando features derivadas (CV_HDA, Diff_Media_CG) para jogos futuros...")
    # Certifique-se que a função calculate_derived_features existe e está correta
    # Usa o resultado da etapa anterior (df_temp_fixtures) como input
    df_temp_fixtures_derived = calculate_derived_features(df_temp_fixtures)
    # -----------------------------------------------------------------------------

    # --- Etapa 3: Seleção Final e Tratamento de NaN ---
    print("  Selecionando features FINAIS p/ modelo e tratando NaNs...")
    final_features_for_model = feature_columns
    # **Usa o df_temp_fixtures_derived (que agora tem CV_HDA)**
    missing_final_future = [f for f in final_features_for_model if f not in df_temp_fixtures_derived.columns]
    if missing_final_future:
        print(f"AVISO Prep Fixture: Features do modelo ausentes: {missing_final_future}. Serão NaN.")
        for col in missing_final_future: df_temp_fixtures_derived[col] = np.nan # Adiciona como NaN

    # Seleciona do DataFrame que contém as derivadas
    X_fixture_prepared = df_temp_fixtures_derived[final_features_for_model].copy()
    print(f"    Shape APÓS seleção final: {X_fixture_prepared.shape}")

    # Tratamento final de NaNs
    nan_counts = X_fixture_prepared.isnull().sum(); total_nans = nan_counts.sum()
    if total_nans > 0:
        print(f"    AVISO: {total_nans} NaNs encontrados. Preenchendo com 0.")
        print(f"      Contagem por coluna:\n{nan_counts[nan_counts > 0]}")
        X_fixture_prepared = X_fixture_prepared.fillna(0) # Preenche com 0

    print(f"--- Preparação Features Futuras OK. Shape final: {X_fixture_prepared.shape} ---")
    return X_fixture_prepared