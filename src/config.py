# --- src/config.py ---
import os
import numpy as np
from datetime import date, timedelta
import random
import skopt.space as sp
from dotenv import load_dotenv
load_dotenv()

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
SRC_DIR = os.path.join(BASE_DIR, 'src')

# --- Configuração do Modelo "Back Draw" ---
MODEL_TYPE_NAME = "BackDraw_MultiSelect"
MODEL_SUFFIX_F1 = "_backdraw_best_f1" # Sufixo para melhor F1
MODEL_SUFFIX_ROI = "_backdraw_best_roi" # Sufixo para melhor ROI (ou 2nd F1)

# Arquivo de Dados Históricos (CSV) - FootyStats
HISTORICAL_DATA_FILENAME_1 = "Base_de_Dados_FootyStats_(2006_2021).csv"
HISTORICAL_DATA_PATH_1 = os.path.join(DATA_DIR, HISTORICAL_DATA_FILENAME_1)

HISTORICAL_DATA_FILENAME_2 = "Base_de_Dados_FootyStats_(2022_2025).csv"
HISTORICAL_DATA_PATH_2 = os.path.join(DATA_DIR, HISTORICAL_DATA_FILENAME_2)

SCRAPER_BASE_URL = "https://flashscore.com"
SCRAPER_TARGET_DAY =  "tomorrow" # "today" ou "tomorrow"
SCRAPER_TARGET_DATE = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d") 
CHROMEDRIVER_PATH = os.path.join(BASE_DIR, 'chromedriver.exe')
SCRAPER_TIMEOUT = 20
SCRAPER_ODDS_TIMEOUT = 20

# --- Configurações Pi-Rating ---
PI_RATING_INITIAL = 1500  # Rating inicial para times não vistos
PI_RATING_K_FACTOR = 30   # Fator K (ajusta a magnitude da mudança de rating)
PI_RATING_HOME_ADVANTAGE = 65 # Pontos de rating adicionados ao time da casa para cálculo da expectativa (ajuste experimental)

# 1. NOMES INTERNOS PADRÃO (Seus identificadores únicos)
INTERNAL_LEAGUE_NAMES = {
    'ARGENTINA 1': 'Argentina - Primera División', # Nome descritivo opcional
    'SPAIN 2': 'Espanha - Segunda División',
    'ITALY 1': 'Itália - Serie A',
    'ITALY 2': 'Itália - Serie B',
    'BRAZIL 1': 'Brasil - Série A',
    'BRAZIL 2': 'Brasil - Série B',
    'ROMANIA 1': 'Romênia - Liga 1'
    # Adicione mais se precisar
}

# Lista apenas dos identificadores curtos, se preferir usá-los internamente
TARGET_LEAGUES_INTERNAL_IDS = list(INTERNAL_LEAGUE_NAMES.keys())

#database 

TARGET_LEAGUES_1 = {'Argentina Primera División':'ARGENTINA 1','Spain Segunda División':'SPAIN 2', 'Italy Serie A':'ITALY 1', 'Italy Serie B':'ITALY 2', 'Brazil Serie A':'BRAZIL 1','Brazil Serie B':'BRAZIL 2', 'Romania Liga I':'ROMANIA 1'}
TARGET_LEAGUES_2 = {'ARGENTINA 1':'ARGENTINA 1','SPAIN 2':'SPAIN 2', 'ITALY 1':'ITALY 1', 'ITALY 2':'ITALY 2', 'BRAZIL 1':'BRAZIL 1','BRAZIL 2':'BRAZIL 2', 'ROMANIA 1':'ROMANIA 1'}
#database futuro
SCRAPER_TO_INTERNAL_LEAGUE_MAP = {
     # Nome no Scraper : Nome Interno/Histórico
    'ARGENTINA: Torneo Betano - Apertura': 'ARGENTINA 1',
    'SPAIN: LaLiga2': 'SPAIN 2',
    'ITALY: Serie A': 'ITALY 1',
    'ITALY: Serie B': 'ITALY 2',
    'BRAZIL: Serie A Betano': 'BRAZIL 1',
    'BRAZIL: Serie B': 'BRAZIL 2',
    'ROMANIA: Superliga - Relegation Group': 'ROMANIA 1'
}

SCRAPER_SLEEP_BETWEEN_GAMES = 20
SCRAPER_SLEEP_AFTER_NAV = 10

# --- Arquivos dos Modelos Salvos ---
BEST_F1_MODEL_FILENAME = f"best_model{MODEL_SUFFIX_F1}.joblib"
BEST_F1_MODEL_SAVE_PATH = os.path.join(DATA_DIR, BEST_F1_MODEL_FILENAME)

BEST_ROI_MODEL_FILENAME = f"best_model{MODEL_SUFFIX_ROI}.joblib"
BEST_ROI_MODEL_SAVE_PATH = os.path.join(DATA_DIR, BEST_ROI_MODEL_FILENAME)

MODEL_SAVE_PATH = BEST_F1_MODEL_SAVE_PATH # Caminho padrão para salvar o modelo

# Identificadores para exibir na GUI/CLI
MODEL_ID_F1 = "Melhor F1 (Empate)"
MODEL_ID_ROI = "Melhor ROI (Empate)"

# --- Fonte de Dados Futuros (CSV GitHub) ---
FIXTURE_FETCH_DAY = "today"
FIXTURE_CSV_URL_TEMPLATE = "https://raw.githubusercontent.com/alamoprince/data_base_fut_analytics/refs/heads/main/data/raw_scraped/scraped_fixtures_{date_str}.csv"

XG_COLS = {'home': 'XG_H', 'away': 'XG_A'}
XG_COLS['total'] = 'XG_Total' # Adiciona a coluna total (se necessário)
# Colunas base ESPERADAS no CSV (nomes ORIGINAIS do CSV) - VERIFIQUE!
CSV_EXPECTED_COLS_HIST = {
    'Id': 'Id_Jogo',         # Mapeia 'Id' do CSV para 'Id_Jogo' interno (opcional)
    'Date': 'Date',          # Mapeia 'Date' do CSV para 'Date' interno (PODE conter hora?)
    'Time': 'Time_Str',          # Mapeia 'Time' do CSV para 'Time' interno
    'Country': 'Country',    # Mapeia 'Country' para 'Country'
    'League': 'League',      # Mapeia 'League' (que já tem ID interno) para 'League'
    'Home': 'Home',      # Mapeia 'HomeTeam' do CSV para 'Home' interno <<-- IMPORTANTE
    'Away': 'Away',      # Mapeia 'AwayTeam' do CSV para 'Away' interno <<-- IMPORTANTE
    'Odd_H_FT': 'Odd_H_FT',  # Nomes já coincidem
    'Odd_D_FT': 'Odd_D_FT',
    'Odd_A_FT': 'Odd_A_FT',
    'Odd_Over25_FT': 'Odd_Over25_FT',
    'Odd_Under25_FT': 'Odd_Under25_FT',
    'Odd_BTTS_Yes': 'Odd_BTTS_Yes',
    'Odd_BTTS_No': 'Odd_BTTS_No',
    # Adicione mapeamentos para XG se o scraper os salvar com nomes diferentes
    # Ex: 'XG_Home': XG_COLS['home'],
    # Ex: 'XG_Away': XG_COLS['away'],
    # Ex: 'XG_T': XG_COLS['total'],
}

# Mapeamento CSV -> Nomes Internos
CSV_HIST_COL_MAP = {k: k for k in CSV_EXPECTED_COLS_HIST} # Inicia mapeando para si mesmo
CSV_HIST_COL_MAP.update({ # Sobrescreve os que precisam de renomeação
    'Date': 'Date',
    'HomeTeam': 'Home',             # <<< CHAVE é 'Home', VALOR é 'Home'
    'AwayTeam': 'Away',             # <<< CHAVE é 'Away', VALOR é 'Away'
    'Goals_H_FT': 'Goals_H_FT', # <<< Nomes já coincidem
    'Goals_A_FT': 'Goals_A_FT', # <<< Nomes já coincidem
    'Odd_H_FT': 'Odd_H_FT',      # <<< Nomes já coincidem
    'Odd_D_FT': 'Odd_D_FT',      # <<< Nomes já coincidem
    'Odd_A_FT': 'Odd_A_FT',      # <<< Nomes já coincidem
    'League': 'League',          # <<< Nomes já coincidem (ou ajuste se for diferente)
    # Adicione mapeamentos SE os nomes no CSV forem diferentes dos internos desejados
    'Odd_Over25_FT': 'Odd_Over25_FT', # Ex: Nomes já coincidem
    'Odd_Under25_FT': 'Odd_Under25_FT',
    'Odd_BTTS_Yes': 'Odd_BTTS_Yes',
    'Odd_BTTS_No': 'Odd_BTTS_No',
    'XG_Home_Pre': XG_COLS['home'], # Mapeia para 'XG_H'
    'XG_Away_Pre': XG_COLS['away'], # Mapeia para 'XG_A'
    'XG_Total_Pre': 'XG_Total', # Exemplo de nome interno
})
# Colunas internas essenciais após ler e mapear o CSV
REQUIRED_FIXTURE_COLS = ['League','Time', 'Home', 'Away', 'Odd_H_FT', 'Odd_D_FT', 'Odd_A_FT', 
                         'Odd_Over25_FT', 'Odd_BTTS_Yes']


# --- Configurações Gerais do Modelo ---
RANDOM_STATE = 42; TEST_SIZE = 0.2; CROSS_VALIDATION_SPLITS = 5; N_JOBS_GRIDSEARCH = -1; ROLLING_WINDOW = 7 #mudei o rolling_window para 7 dias (1 semana) antes era 10 dias
FEATURE_EPSILON = 1e-6
RESULT_MAPPING = {'D': 0, 'H': 1, 'A': 2} # Usado para Ptos
CLASS_NAMES = ['Nao_Empate', 'Empate'] # Alvo binário

# --- Nomes das Colunas Internas (Odds/Gols) (NOMES INTERNOS)---
ODDS_COLS = {'home': 'Odd_H_FT', 'draw': 'Odd_D_FT', 'away': 'Odd_A_FT'}
GOALS_COLS = {'home': 'Goals_H_FT', 'away': 'Goals_A_FT'} 
OTHER_ODDS_NAMES = [ 'Odd_Over25_FT', 'Odd_Under25_FT', 'Odd_BTTS_Yes', 'Odd_BTTS_No' ]

# Colunas base ESPERADAS no Excel (nomes ORIGINAIS do Excel)
CSV_PATTERN_COLS = [
    'Date', 'Home', 'Away',
    'Goals_H_FT', 'Goals_A_FT', # Nomes que GOALS_COLS usa nas CHAVES
    'Odd_H_FT', 'Odd_D_FT', 'Odd_A_FT', # Nomes que ODDS_COLS usa
    'Odd_Over25_FT', 'Odd_BTTS_Yes' # Outras odds diretas usadas como features
    'XG_Home_Pre', 'XG_Away_Pre', 'XG_Total' # Odds de xG (Expected Goals)
]

# Interações Odds/Prob x CV_HDA
INTERACTION_P_D_NORM_X_CV_HDA = 'pDnorm_x_CVHDA'
INTERACTION_P_D_NORM_DIV_CV_HDA = 'pDnorm_div_CVHDA'
# Interações Odds/Prob x PiRating_Diff
INTERACTION_P_D_NORM_X_PIR_DIFF = 'pDnorm_x_PiRDiffAbs'
INTERACTION_P_D_NORM_DIV_PIR_DIFF = 'pDnorm_div_PiRDiffAbs'
INTERACTION_ODD_D_X_PIR_DIFF = 'OddD_x_PiRDiffAbs'
INTERACTION_ODD_D_DIV_PIR_DIFF = 'OddD_div_PiRDiffAbs'
# Pi-Rating Momentum (Calculado sobre ROLLING_WINDOW jogos anteriores)
PIRATING_MOMENTUM_H = f'PiR_Mom_{ROLLING_WINDOW}G_H'
PIRATING_MOMENTUM_A = f'PiR_Mom_{ROLLING_WINDOW}G_A'
PIRATING_MOMENTUM_DIFF = f'PiR_Mom_{ROLLING_WINDOW}G_Diff'

ALL_CANDIDATE_FEATURES = [
    # Baseadas em Odds / Probabilidades Normalizadas
    'p_D_norm',
    'abs_ProbDiff_Norm',
    'p_H_norm', # Pode adicionar se quiser analisar individualmente
    'p_A_norm', # Pode adicionar se quiser analisar individualmente

    # Médias Rolling (Valor e Custo do Gol)
    'Media_VG_H',
    'Media_VG_A',
    'Media_CG_H',
    'Media_CG_A',

    # Médias Rolling (Pontos) - Adicionar se quiser testar
    'Media_Ptos_H',
    'Media_Ptos_A',

    # Desvios Padrão Rolling (Custo do Gol)
    'Std_CG_H',
    'Std_CG_A',

    # Desvios Padrão Rolling (Valor do Gol) - Adicionar se quiser testar
    'Std_VG_H',
    'Std_VG_A',

    # Desvios Padrão Rolling (Pontos) - Adicionar se quiser testar
    'Std_Ptos_H',
    'Std_Ptos_A',

    # Binning das Odds de Empate
    'Odd_D_Cat',

    # Features Derivadas Originais (Se quiser mantê-las para análise/teste)
    'CV_HDA',
    'Diff_Media_CG', # Redundante se usar Media_CG_H/A? Testar.

    # Odds Diretas (Se quiser comparar diretamente na análise)
    'Odd_H_FT',
    'Odd_D_FT',
    'Odd_A_FT',
    'Odd_Over25_FT', # Se disponível e relevante
    'Odd_BTTS_Yes',  # Se disponível e relevante

    'Avg_Gols_Marcados_H',  # Média Gols Marcados H (Poisson Simples)
    'Avg_Gols_Sofridos_H',  # Média Gols Sofridos H (Poisson Simples)
    'Avg_Gols_Marcados_A',  # Média Gols Marcados A (Poisson Simples)
    'Avg_Gols_Sofridos_A',  # Média Gols Sofridos A (Poisson Simples)

    #Probabilidade Poisson de Empate
    'Prob_Empate_Poisson'

    #xG (Expected Goals) - Se disponível e relevante
    'XG_Home_Pre', 
    'XG_Away_Pre',
    'XG_Total'

    # Novas Features Pi-Rating (Exemplos)
    'PiRating_H',       # Rating do time da casa ANTES do jogo
    'PiRating_A',       # Rating do time visitante ANTES do jogo
    'PiRating_Diff',    # Diferença (PiRating_H - PiRating_A)
    'PiRating_Prob_H',  # Probabilidade de vitória da casa segundo Pi-Ratings

    # Novas Interações (Exemplos - escolha quais testar)
    INTERACTION_P_D_NORM_DIV_CV_HDA,
    INTERACTION_P_D_NORM_X_PIR_DIFF,
    INTERACTION_ODD_D_DIV_PIR_DIFF,
    # Novos Momentums (Exemplos)
    PIRATING_MOMENTUM_H,
    PIRATING_MOMENTUM_A,
    PIRATING_MOMENTUM_DIFF,

]

# --- Lista das Features FINAIS para o Modelo BackDraw ---
FEATURE_COLUMNS = [
    # Derivada das Odds 1x2
    'CV_HDA',

    # Odds Diretas (do CSV ou Histórico)
    'Odd_H_FT',
    'Odd_A_FT',
    'Odd_D_FT',

    # Médias Rolling (Calculadas do Histórico)
    'Media_VG_H',
    'Media_VG_A',
    'Media_CG_H',
    'Media_CG_A',

    # Diferença Rolling (Calculada)
    'Diff_Media_CG',

    # Consistência Custo Gol Casa (Rolling Std)
    'Std_CG_H',   
    'Std_CG_A',    
                   
    
]

NEW_FEATURE_COLUMNS = [
    'p_D_norm',             # Probabilidade Empate Normalizada
    'PiRating_Prob_H',      # Probabilidade de vitória da casa segundo Pi-Ratings
    'CV_HDA',               # Coeficiente de Variação (HDA)
    'Std_CG_A',             # Custo Gol Fora (Rolling Std)
    'Std_CG_H',             # Custo Gol Casa (Rolling Std)
    'Prob_Empate_Poisson',  # Empate de Poisson
    'Odd_D_Cat',            # Binning das Odds de Empate

]

FEATURE_COLUMNS = NEW_FEATURE_COLUMNS

# --- Métrica para Selecionar o Melhor Modelo ---
BEST_MODEL_METRIC = 'f1_score_draw' #antes:'f1_score'
BEST_MODEL_METRIC_ROI = 'roi' # ROI (Expected Value) - Para o modelo de ROI

# --- Configuração dos Modelos a Testar ---

rf_search_spaces = {
    'classifier__n_estimators': sp.Integer(100, 300), # Prefixo!
    'classifier__criterion': sp.Categorical(['gini', 'entropy']), # Prefixo!
    'classifier__max_depth': sp.Integer(5, 20, prior='uniform'), # Prefixo!
    'classifier__min_samples_split': sp.Integer(5, 30), # Prefixo!
    'classifier__min_samples_leaf': sp.Integer(3, 20), # Prefixo!
    'classifier__max_features': sp.Categorical(['sqrt', 'log2']), # Prefixo!
    'classifier__class_weight': sp.Categorical(['balanced', None]), # Prefixo! - 'balanced' pode ser útil aqui
    'classifier__bootstrap': sp.Categorical([True, False]), # Prefixo!
    # Adicione outros parâmetros do SMOTE se quiser otimizá-los, ex:
    # 'sampler__k_neighbors': sp.Integer(3, 11) # Otimizar k do SMOTE
}

# Regressão Logística
lr_search_spaces = {
    'classifier__C': sp.Real(0.01, 100.0, prior='log-uniform'), # Real em escala logarítmica
    'classifier__penalty': sp.Categorical(['l1', 'l2']),
    'classifier__solver': sp.Categorical(['liblinear', 'saga']),
    'classifier__class_weight': sp.Categorical(['balanced', None]),
    'classifier__max_iter': sp.Integer(2000, 5000), # Aumentar se saga não convergir
}

#LightGBM (Se for usar)
lgbm_search_spaces = {
    'classifier__n_estimators': sp.Integer(50, 300),
    'classifier__learning_rate': sp.Real(0.01, 0.2, prior='log-uniform'),
    'classifier__num_leaves': sp.Integer(10, 50),
    'classifier__max_depth': sp.Integer(3, 15),
    'classifier__reg_alpha': sp.Real(0.0, 0.5), # L1 reg
    'classifier__reg_lambda': sp.Real(0.0, 0.5), # L2 reg
    'classifier__colsample_bytree': sp.Real(0.6, 1.0),
 }

# SVC
svc_search_spaces = {
    'classifier__C': sp.Real(0.5, 30.0, prior='log-uniform'),
    'classifier__kernel': sp.Categorical(['rbf']),
    #'degree': sp.Integer(2, 3), # Só para poly
    'classifier__gamma': sp.Real(1e-3, 1.0, prior='log-uniform'), # Mais relevante para RBF
    'classifier__class_weight': sp.Categorical(['balanced', None]),
}

# KNN
knn_search_spaces = {
    'classifier__n_neighbors': sp.Integer(3, 41, prior='uniform'), # Ímpares numa faixa maior
    'classifier__weights': sp.Categorical(['uniform', 'distance']),
    'classifier__metric': sp.Categorical(['minkowski', 'manhattan']),
    'classifier__p': sp.Integer(1, 2), # 1 para manhattan, 2 para minkowski(euclidean)
}

# GaussianNB (Não tem muitos hiperparâmetros para otimizar com Bayes, GridSearchCV é ok)
gnb_search_spaces = { 'classifier__var_smoothing': np.logspace(-9, -2, num=15) }

MODEL_CONFIG = {
    'RandomForestClassifier': {
        'model_kwargs': {'random_state': RANDOM_STATE, 'n_jobs': N_JOBS_GRIDSEARCH},
        'search_spaces': rf_search_spaces, 
        'needs_scaling': False
    },
    'LogisticRegression': {
        'model_kwargs': {'random_state': RANDOM_STATE}, # max_iter está no space
        'search_spaces': lr_search_spaces, 
        'needs_scaling': True
    },
     'LGBMClassifier': { 'model_kwargs': {'random_state': RANDOM_STATE, 'n_jobs': N_JOBS_GRIDSEARCH,
                        'objective': 'binary', 'metric': 'logloss',
                        'is_unbalance': True},
       'search_spaces': lgbm_search_spaces, 
       'needs_scaling': False
    },
     'SVC': {
        'model_kwargs': {'random_state': RANDOM_STATE, 'probability': True},
        'search_spaces': svc_search_spaces, 
        'needs_scaling': True
    },
    'GaussianNB': { # Mantém GridSearchCV para GNB
        'model_kwargs': {},
        'param_grid': gnb_search_spaces,
        'needs_scaling': False
    },
    'KNeighborsClassifier': {
        'model_kwargs': {'n_jobs': N_JOBS_GRIDSEARCH},
        'search_spaces': knn_search_spaces, 
        'needs_scaling': True
    },
}

# Número de iterações para Otimização Bayesiana
BAYESIAN_OPT_N_ITER = 25 # Ajuste conforme necessário (mais iterações = melhor, porém mais lento)

# Número de iterações para GridSearchCV (se não usar Bayes)
DEFAULT_EV_THRESHOLD = 0.05 # Threshold padrão para EV (Expected Value) - Ajuste conforme necessário

DEFAULT_F1_THRESHOLD = 0.7

# --- Configuração GitHub ---

GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')
GITHUB_REPO_NAME = os.environ.get('GITHUB_REPO_NAME')
GITHUB_PREDICTIONS_PATH = f"data/predictions_{MODEL_TYPE_NAME}"