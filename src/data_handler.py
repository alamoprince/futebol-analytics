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


def calculate_historical_intermediate(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula FT_Result, IsDraw, Ptos, Probs, VG/CG raw no DataFrame."""
    df_calc = df.copy(); print("  Calculando stats intermediárias (Ptos, Probs, VG/CG raw)..."); epsilon = 1e-6
    df_calc['FT_Result'] = np.select([df_calc['Goals_H_FT'] > df_calc['Goals_A_FT'], df_calc['Goals_H_FT'] == df_calc['Goals_A_FT']], ["H", "D"], default="A")
    df_calc['IsDraw'] = (df_calc['FT_Result'] == 'D').astype(int)
    df_calc['Ptos_H'] = np.select([df_calc['FT_Result']=='H', df_calc['FT_Result']=='D'], [3, 1], default=0)
    df_calc['Ptos_A'] = np.select([df_calc['FT_Result']=='A', df_calc['FT_Result']=='D'], [3, 1], default=0)
    if all(c in df_calc.columns for c in ODDS_COLS.values()):
        df_calc['p_H'] = 1 / (df_calc[ODDS_COLS['home']] + epsilon); 
        df_calc['p_A'] = 1 / (df_calc[ODDS_COLS['away']] + epsilon); 
        df_calc['p_D'] = 1 / (df_calc[ODDS_COLS['draw']] + epsilon)
    else: 
        print("Aviso: Odds 1x2 ausentes p/ Probs."); 
        df_calc[['p_H', 'p_A', 'p_D']] = np.nan
    if 'p_H' in df_calc.columns and 'p_A' in df_calc.columns:
        df_calc['VG_H_raw'] = df_calc['Goals_H_FT'] * df_calc['p_A']; df_calc['VG_A_raw'] = df_calc['Goals_A_FT'] * df_calc['p_H']
        df_calc['CG_H_raw'] = np.where(df_calc['Goals_H_FT'] > 0, df_calc['p_H'] / df_calc['Goals_H_FT'], np.nan); 
        df_calc['CG_A_raw'] = np.where(df_calc['Goals_A_FT'] > 0, df_calc['p_A'] / df_calc['Goals_A_FT'], np.nan)
    else: 
        print("Aviso: Probs ausentes p/ VG/CG raw."); 
        df_calc[['VG_H_raw', 'VG_A_raw', 'CG_H_raw', 'CG_A_raw']] = np.nan
    return df_calc

def calculate_rolling_stats(df: pd.DataFrame, stats_to_calc: List[str], window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Calcula médias móveis para as estatísticas especificadas."""
    df_calc = df.copy();
    teams = pd.concat([df_calc['Home'], df_calc['Away']]).unique();
    team_history: Dict[str, Dict[str, List[float]]] = {team: {stat: [] for stat in stats_to_calc} for team in teams};
    results_list = [];
    # required_raw_cols = set(); # Não usado fora da função
    rolling_cols_map = {};
    cols_to_calculate = {}

    for stat_prefix in stats_to_calc:
        media_col_h = f'Media_{stat_prefix}_H'; media_col_a = f'Media_{stat_prefix}_A'; skip_h = skip_a = False
        if media_col_h in df_calc.columns and pd.api.types.is_numeric_dtype(df_calc[media_col_h]): skip_h = True
        if media_col_a in df_calc.columns and pd.api.types.is_numeric_dtype(df_calc[media_col_a]): skip_a = True
        if skip_h and skip_a: print(f"  Aviso: {media_col_h}/{media_col_a} já existem."); continue
        if stat_prefix == 'Ptos': base_h, base_a = 'Ptos_H', 'Ptos_A';
        elif stat_prefix == 'VG': base_h, base_a = 'VG_H_raw', 'VG_A_raw';
        elif stat_prefix == 'CG': base_h, base_a = 'CG_H_raw', 'CG_A_raw';
        else: print(f"Aviso: Prefixo '{stat_prefix}' desconhecido."); continue
        if base_h not in df_calc.columns or base_a not in df_calc.columns: 
            print(f"Erro: Colunas base '{base_h}'/'{base_a}' não encontradas."); continue
        # required_raw_cols.add(base_h); required_raw_cols.add(base_a); # Não usado fora
        rolling_cols_map[stat_prefix] = {'home': base_h, 'away': base_a}
        if not skip_h: cols_to_calculate[stat_prefix + '_H'] = media_col_h;
        if not skip_a: cols_to_calculate[stat_prefix + '_A'] = media_col_a

    if not cols_to_calculate: return df_calc
    print(f"  Calculando médias rolling para: {list(cols_to_calculate.keys())} (Janela={window})...")

    # Loop otimizado para cálculo rolling
    calculated_stats = []
    # Itera apenas uma vez, calculando todas as stats necessárias para a linha
    for index, row in tqdm(df_calc.iterrows(), total=len(df_calc), desc="Calc. Rolling Médias"):
        home_team = row['Home']; away_team = row['Away']
        current_match_features = {'Index': index}

        # Calcula médias para time da casa
        for stat_prefix in rolling_cols_map.keys():
             media_col_h = f'Media_{stat_prefix}_H'
             if stat_prefix + '_H' in cols_to_calculate:
                  hist_H = team_history[home_team][stat_prefix]
                  recent = hist_H[-window:]
                  current_match_features[media_col_h] = np.mean(recent) if len(recent) > 0 else np.nan

        # Calcula médias para time visitante
        for stat_prefix in rolling_cols_map.keys():
            media_col_a = f'Media_{stat_prefix}_A'
            if stat_prefix + '_A' in cols_to_calculate:
                hist_A = team_history[away_team][stat_prefix]
                recent = hist_A[-window:]
                current_match_features[media_col_a] = np.mean(recent) if len(recent) > 0 else np.nan

        calculated_stats.append(current_match_features)

        # Atualiza histórico DEPOIS de calcular para a linha atual
        for stat_prefix, base_cols in rolling_cols_map.items():
            if pd.notna(row[base_cols['home']]): team_history[home_team][stat_prefix].append(row[base_cols['home']])
            if pd.notna(row[base_cols['away']]): team_history[away_team][stat_prefix].append(row[base_cols['away']])

    df_rolling_stats = pd.DataFrame(calculated_stats).set_index('Index')
    # Junta apenas as colunas que foram realmente calculadas
    cols_to_join = [col for col in cols_to_calculate.values() if col in df_rolling_stats.columns]
    df_final = df_calc.join(df_rolling_stats[cols_to_join]) if cols_to_join else df_calc
    return df_final


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
def preprocess_and_feature_engineer(df_loaded: pd.DataFrame) -> Optional[Tuple[pd.DataFrame, pd.Series, List[str]]]:
    """Pipeline BackDraw: Calcula/Verifica todas as features necessárias no histórico."""
    if df_loaded is None: return None
    print("Pré-processamento BackDraw (Histórico)...")
    df_intermediate = calculate_historical_intermediate(df_loaded)
    stats_to_roll = ['Ptos', 'VG', 'CG']
    df_with_rolling = calculate_rolling_stats(df_intermediate, stats_to_roll, window=ROLLING_WINDOW)
    df_with_derived = calculate_derived_features(df_with_rolling)
    print("  Selecionando features finais e tratando NaNs..."); target_col = 'IsDraw'
    direct_odds_features = ['Odd_H_FT', 'Odd_D_FT', 'Odd_Over25_FT', 'Odd_BTTS_Yes'];
    for f in direct_odds_features:
        if f not in df_with_derived.columns: df_with_derived[f] = np.nan
    required_final_cols = FEATURE_COLUMNS + [target_col]
    missing_final = [f for f in required_final_cols if f not in df_with_derived.columns]
    if missing_final: print(f"Erro CRÍTICO: Colunas finais ausentes: {missing_final}"); return None
    df_final_selection = df_with_derived[required_final_cols].copy()
    initial_rows = len(df_final_selection); df_final_selection = df_final_selection.dropna()
    rows_dropped = initial_rows - len(df_final_selection)
    if rows_dropped > 0: print(f"  Removidas {rows_dropped} linhas (NaNs).")
    if df_final_selection.empty: print("Erro: Nenhuma linha restante."); return None
    X = df_final_selection[FEATURE_COLUMNS]; y = df_final_selection[target_col]
    print(f"Pré-processamento BackDraw OK. Shape X: {X.shape}, y: {y.shape}"); return X, y, FEATURE_COLUMNS


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


#--- PREPARAR DADOS FUTUROS (BackDraw - Refatorado) ---
def prepare_fixture_data(fixture_df: pd.DataFrame, historical_df: pd.DataFrame, feature_columns: List[str]) -> Optional[pd.DataFrame]:
    """Prepara TODAS as features finais para jogos futuros."""
    if fixture_df is None or historical_df is None or not feature_columns: return None
    if historical_df.empty: print("Erro: Histórico vazio."); return None
    if fixture_df.empty: return pd.DataFrame(columns=feature_columns)

    print("\nPreparando features finais para jogos futuros...")
    print("  Processando histórico p/ cálculo de médias..."); start_time = time.time()
    historical_df_processed = calculate_historical_intermediate(historical_df.copy())
    required_hist_cols = {'Home', 'Away', 'Date', 'Ptos_H', 'Ptos_A', 'VG_H_raw', 'VG_A_raw', 
                          'CG_H_raw', 'CG_A_raw'}; 
    missing_hist = [c for c in required_hist_cols if c not in historical_df_processed.columns]
    if missing_hist: print(f"Erro: Colunas intermediárias ausentes no histórico: {missing_hist}"); return None
    historical_df_processed = historical_df_processed.sort_values(by='Date', ascending=False); 
    teams_in_hist = pd.concat([historical_df_processed['Home'], 
                               historical_df_processed['Away']]).unique()
    print(f"  Histórico processado em {time.time() - start_time:.2f} seg.")

    print("  Calculando médias Ptos, VG, CG para jogos futuros...")
    stats_to_roll = ['Ptos', 'VG', 'CG']
    rolling_features_list = []
    fixture_indices = fixture_df.index
    for index in tqdm(fixture_indices, total=len(fixture_indices), desc="Calc. Rolling Futuro"):
        future_match = fixture_df.loc[index]; home_team = future_match.get('HomeTeam'); away_team = future_match.get('AwayTeam'); match_rolling = {'Index': index}
        for team_perspective, team_name in [('H', home_team), ('A', away_team)]:
            for stat_prefix in stats_to_roll:
                media_val = np.nan
                if team_name and team_name in teams_in_hist:
                    team_hist = historical_df_processed[ (historical_df_processed['Home'] == team_name) | (historical_df_processed['Away'] == team_name) ].head(ROLLING_WINDOW)
                    if stat_prefix == 'Ptos': base_h, base_a = 'Ptos_H', 'Ptos_A';
                    elif stat_prefix == 'VG': base_h, base_a = 'VG_H_raw', 'VG_A_raw';
                    elif stat_prefix == 'CG': base_h, base_a = 'CG_H_raw', 'CG_A_raw';
                    else: continue
                    stat_vals = team_hist.apply(lambda r: r[base_h] if r['Home'] == team_name else r[base_a], axis=1).dropna().tolist()
                    if stat_vals: media_val = np.mean(stat_vals)
                match_rolling[f'Media_{stat_prefix}_{team_perspective}'] = media_val
        rolling_features_list.append(match_rolling) # Movido para fora do loop interno de stats
    df_rolling_features = pd.DataFrame(rolling_features_list).set_index('Index') # Cria DF após o loop principal

    print("  Calculando features derivadas (CV_HDA, Diff_Media_CG)...")
    df_temp = fixture_df.join(df_rolling_features)
    df_with_derived = calculate_derived_features(df_temp)

    print("  Selecionando features finais e tratando NaNs...")
    #print(f"    Colunas disponíveis ANTES da seleção final: {list(df_with_derived.columns)}") # DEBUG
    #print(f"    Features esperadas (config): {feature_columns}") # DEBUG
    missing_final = [f for f in feature_columns if f not in df_with_derived.columns]
    if missing_final: print(f"Aviso: Colunas finais ausentes: {missing_final}. Serão NaN."); #... (add NaNs)...
    # Garante que só seleciona colunas que REALMENTE existem
    cols_to_select = [f for f in feature_columns if f in df_with_derived.columns]
    X_fixture_prepared = df_with_derived[cols_to_select].copy()
    #print(f"    Shape APÓS seleção: {X_fixture_prepared.shape}") # DEBUG
    # Verifica se o shape está correto
    if X_fixture_prepared.shape[1] != len(feature_columns):
        print(f"ERRO: Shape incorreto após seleção! Esperado {len(feature_columns)}, obtido {X_fixture_prepared.shape[1]}")
        # Retorna None para indicar falha
        return None
    nan_counts = X_fixture_prepared.isnull().sum()
    if nan_counts.sum() > 0: print(f"Aviso: {nan_counts.sum()} NaNs. Preenchendo com 0."); 
    X_fixture_prepared = X_fixture_prepared.fillna(0)
    print(f"Features finais preparadas. Shape: {X_fixture_prepared.shape}"); return X_fixture_prepared