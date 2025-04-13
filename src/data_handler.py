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
        # Remove linhas onde resultado ou pontos não puderam ser calculados (gols NaN)
        # É melhor dropar NaN mais tarde, após calcular tudo que for possível
        # df_calc = df_calc.dropna(subset=['FT_Result']) # REMOVA o dropna daqui
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

def preprocess_and_feature_engineer(df_loaded: pd.DataFrame) -> Optional[Tuple[pd.DataFrame, pd.Series, List[str]]]:
    """Pipeline BackDraw: Calcula/Verifica TODAS as features NOVAS no histórico."""
    if df_loaded is None: return None
    print("\n--- Iniciando Pré-processamento e Engenharia de Features (Histórico) ---")

    # 1. Calcular Stats Intermediárias (Garantir que p_H/D/A são calculados aqui)
    # Placeholder for calculate_historical_intermediate function
    # Replace this with the actual implementation or logic
    df_interm = df_loaded.copy()  # Assuming no intermediate calculations for now
    # Garante probs implícitas se não calculadas em intermediate
    if not all(p in df_interm.columns for p in ['p_H', 'p_D', 'p_A']):
         print("  Recalculando probabilidades p_H/D/A...")
         df_interm = calculate_probabilities(df_interm)

    # 2. Calcular Probabilidades Normalizadas
    df_probs_norm = calculate_normalized_probabilities(df_interm)

    # 3. Calcular Médias Rolling (VG, CG)
    stats_to_roll_mean = ['VG', 'CG'] # Não precisamos mais de Ptos? Verificar.
    # Placeholder for calculate_rolling_stats function
    # Replace this with the actual implementation or logic
    df_rolling_mean = calculate_rolling_stats(df_probs_norm, stats_to_roll_mean, window=ROLLING_WINDOW)
    # 4. Calcular Desvio Padrão Rolling (CG)
    stats_to_roll_std = ['CG'] # Adicionamos apenas Std para CG por agora
    df_rolling_std = calculate_rolling_std(df_rolling_mean, stats_to_roll_std, window=ROLLING_WINDOW)

    # 5. Calcular Features de Binning
    df_binned = calculate_binned_features(df_rolling_std)

    # 6. Selecionar Features Finais e Tratar NaNs
    #    USE A NOVA LISTA DE FEATURES DO config.py!
    df_final = df_binned
    target_col = 'IsDraw' # Alvo
    print("  Selecionando features finais e tratando NaNs...")

    # Usa FEATURE_COLUMNS importado do config
    required_final_cols = FEATURE_COLUMNS + [target_col]

    # Verifica se todas as colunas FINAIS existem
    missing_final = [f for f in required_final_cols if f not in df_final.columns]
    if missing_final:
        print(f"Erro CRÍTICO: Colunas finais ausentes após todos cálculos: {missing_final}")
        print(f"Colunas disponíveis: {list(df_final.columns)}")
        return None

    # Seleciona apenas as colunas finais e o alvo
    df_final_selection = df_final[required_final_cols].copy()

    # Tratamento de NaNs (Importante: Após todas as features calculadas)
    initial_rows = len(df_final_selection)
    df_final_selection = df_final_selection.dropna()
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
    print(f"    Features finais: {list(X.columns)}")
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
    """Prepara TODAS as features NOVAS para jogos futuros, mantendo consistência."""
    if fixture_df is None or historical_df is None or not feature_columns: return None
    if historical_df.empty: print("Erro Prep Fixture: Histórico vazio."); return None
    if fixture_df.empty: print("Prep Fixture: Nenhum jogo futuro para processar."); return pd.DataFrame(columns=feature_columns) # Retorna DF vazio com colunas esperadas

    print("\n--- Preparando Features Finais para Jogos Futuros ---")
    print(f"    Jogos futuros brutos: {fixture_df.shape}")
    print(f"    Features finais esperadas: {feature_columns}")

    # --- Etapa 1: Calcular Médias e StDev Rolling (dependem do HISTÓRICO) ---
    print("  Processando histórico para cálculo de rolling stats...")
    start_time = time.time()
    # Calcula intermediárias (incluindo VG/CG raw) no histórico
    # Placeholder logic for calculate_historical_intermediate
    # Replace this with the actual implementation or logic
    historical_df_processed = historical_df.copy()
    # Example: Add any intermediate calculations here if needed
    # Verifica se colunas raw necessárias existem no histórico processado
    required_hist_cols = {'Home', 'Away', 'Date', 'VG_H_raw', 'VG_A_raw', 'CG_H_raw', 'CG_A_raw'} # Adicione Ptos se usar rolling de pontos
    missing_hist = [c for c in required_hist_cols if c not in historical_df_processed.columns]
    if missing_hist: print(f"Erro Prep Fixture: Colunas raw ausentes no histórico: {missing_hist}"); return None
    # Ordena histórico para pegar os jogos MAIS RECENTES para a janela rolling
    historical_df_processed = historical_df_processed.sort_values(by='Date', ascending=False)
    teams_in_hist = pd.concat([historical_df_processed['Home'], historical_df_processed['Away']]).unique()
    print(f"  Histórico processado em {time.time() - start_time:.2f} seg.")

    print("  Calculando médias e StDev rolling para jogos futuros...")
    stats_to_roll_mean = ['VG', 'CG'] # Quais médias calcular
    stats_to_roll_std = ['CG']   # Quais StDevs calcular
    rolling_features_list = []
    fixture_indices = fixture_df.index

    # Loop para calcular médias e StDevs para cada jogo futuro
    for index in tqdm(fixture_indices, total=len(fixture_indices), desc="Calc. Rolling Futuro"):
        future_match = fixture_df.loc[index]
        home_team = future_match.get('HomeTeam')
        away_team = future_match.get('AwayTeam')
        match_rolling = {'Index': index}

        for team_perspective, team_name in [('H', home_team), ('A', away_team)]:
            # Calcula MÉDIAS
            for stat_prefix in stats_to_roll_mean:
                media_val = np.nan
                base_h, base_a = None, None # Define colunas base
                if stat_prefix == 'VG': base_h, base_a = 'VG_H_raw', 'VG_A_raw'
                elif stat_prefix == 'CG': base_h, base_a = 'CG_H_raw', 'CG_A_raw'
                # Adicionar Ptos se necessário
                if base_h and team_name and team_name in teams_in_hist:
                    team_hist = historical_df_processed[ (historical_df_processed['Home'] == team_name) | (historical_df_processed['Away'] == team_name) ].head(ROLLING_WINDOW)
                    if not team_hist.empty:
                        stat_vals = team_hist.apply( lambda r: r.get(base_h) if r['Home'] == team_name else r.get(base_a), axis=1 ).dropna().tolist()
                        if stat_vals: media_val = np.mean(stat_vals)
                match_rolling[f'Media_{stat_prefix}_{team_perspective}'] = media_val

            # Calcula StDevs
            for stat_prefix in stats_to_roll_std:
                std_val = np.nan
                base_h, base_a = None, None # Define colunas base
                if stat_prefix == 'CG': base_h, base_a = 'CG_H_raw', 'CG_A_raw'
                # Adicionar outros se necessário
                if base_h and team_name and team_name in teams_in_hist:
                     team_hist = historical_df_processed[ (historical_df_processed['Home'] == team_name) | (historical_df_processed['Away'] == team_name) ].head(ROLLING_WINDOW)
                     if not team_hist.empty:
                        stat_vals = team_hist.apply( lambda r: r.get(base_h) if r['Home'] == team_name else r.get(base_a), axis=1 ).dropna().tolist()
                        if len(stat_vals) >= 2: std_val = np.std(stat_vals) # Calcula std só se tiver >= 2 pontos
                match_rolling[f'Std_{stat_prefix}_{team_perspective}'] = std_val

        rolling_features_list.append(match_rolling)

    if not rolling_features_list: print("Erro Prep Fixture: Nenhuma feature rolling calculada."); return None
    df_rolling_features = pd.DataFrame(rolling_features_list).set_index('Index')
    print(f"  Médias e StDevs Rolling calculadas. Colunas: {list(df_rolling_features.columns)}")

    # Junta as features rolling ao DataFrame original dos jogos futuros
    # USA O fixture_df ORIGINAL que contém as odds necessárias para os próximos passos!
    df_temp_fixtures = fixture_df.join(df_rolling_features, how='left')
    print(f"  Shape após juntar rolling stats: {df_temp_fixtures.shape}")

    # --- Etapa 2: Calcular Features Derivadas das Odds (Probs, Binning) ---
    #    Estas features dependem apenas das odds do PRÓPRIO jogo futuro (que estão em df_temp_fixtures)
    print("  Calculando probabilidades e binning para jogos futuros...")

    # Garante que as colunas de Odds existem no df_temp_fixtures (vieram do CSV)
    required_odds_future = list(ODDS_COLS.values())
    missing_odds_future = [o for o in required_odds_future if o not in df_temp_fixtures.columns]
    if missing_odds_future:
        print(f"Erro Prep Fixture: Colunas de Odds ausentes nos dados futuros: {missing_odds_future}")
        # Preenche com NaN para continuar? Ou retorna None? Vamos preencher.
        for col in missing_odds_future: df_temp_fixtures[col] = np.nan
        # return None # Alternativa mais segura

    # Calcula probs implícitas (p_H/D/A)
    df_temp_fixtures = calculate_probabilities(df_temp_fixtures)
    # Calcula probs normalizadas (p_X_norm, abs_ProbDiff_Norm)
    df_temp_fixtures = calculate_normalized_probabilities(df_temp_fixtures)
    # Calcula binning (Odd_D_Cat)
    df_temp_fixtures = calculate_binned_features(df_temp_fixtures)

    # --- Etapa 3: Seleção Final e Tratamento de NaN ---
    print("  Selecionando features finais e tratando NaNs...")

    # Verifica quais das features FINAIS esperadas estão faltando AGORA
    missing_final_future = [f for f in feature_columns if f not in df_temp_fixtures.columns]
    if missing_final_future:
        print(f"    AVISO Prep Fixture: Colunas FINAIS ausentes após todos os cálculos: {missing_final_future}. Serão NaN.")
        # Adiciona colunas faltantes com NaN para garantir a forma correta
        for col in missing_final_future:
            df_temp_fixtures[col] = np.nan

    # Seleciona APENAS as colunas FINAIS na ordem correta
    # Garante que só seleciona colunas que existem para evitar KeyError no momento
    cols_to_select = [f for f in feature_columns if f in df_temp_fixtures.columns]
    print(f"    Colunas que serão selecionadas para X_fixture: {cols_to_select}")
    X_fixture_prepared = df_temp_fixtures[cols_to_select].copy()

    # Garante que todas as colunas FINAIS existem no resultado, mesmo que adicionadas como NaN
    if X_fixture_prepared.shape[1] != len(feature_columns):
        print(f"    ALERTA Prep Fixture: Shape incorreto ({X_fixture_prepared.shape[1]}) vs esperado ({len(feature_columns)}). Readicionando colunas faltantes com NaN.")
        missing_in_selection = [f for f in feature_columns if f not in X_fixture_prepared.columns]
        for f in missing_in_selection:
            X_fixture_prepared[f] = np.nan
        # Reordena para garantir a ordem esperada pelo modelo
        X_fixture_prepared = X_fixture_prepared[feature_columns]

    print(f"    Shape APÓS seleção final: {X_fixture_prepared.shape}")

    # Tratamento final de NaNs (pode haver NaNs de rolling stats para times sem histórico suficiente)
    nan_counts = X_fixture_prepared.isnull().sum()
    total_nans = nan_counts.sum()
    if total_nans > 0:
        print(f"    AVISO Prep Fixture: {total_nans} NaNs encontrados nas features finais.")
        print(f"      Contagem por coluna:\n{nan_counts[nan_counts > 0]}")
        # Estratégia de preenchimento: 0 é simples, mas pode não ser ideal.
        # Pode-se considerar preencher com a média/mediana da coluna calculada no TREINO.
        # Por enquanto, vamos preencher com 0 para permitir a previsão.
        print("      Preenchendo NaNs com 0 para permitir a previsão.")
        X_fixture_prepared = X_fixture_prepared.fillna(0)

    print(f"--- Preparação Features Futuras OK. Shape final: {X_fixture_prepared.shape} ---")
    return X_fixture_prepared