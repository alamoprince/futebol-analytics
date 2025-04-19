# --- src/config.py ---
import os
import numpy as np
from datetime import date, timedelta
import random
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
    'ITÁLY: Serie A': 'ITALY 1',
    'ITÁLY: Serie B': 'ITALY 2',
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
FIXTURE_CSV_URL_TEMPLATE = "https://github.com/alamoprince/data_base_fut_analytics/tree/main/data/raw_scraped/scraped_fixtures_{date_str}.csv"

XG_COLS = {'home': 'XG_H', 'away': 'XG_A'}
XG_COLS['total'] = 'XG_Total' # Adiciona a coluna total (se necessário)
# Colunas base ESPERADAS no CSV (nomes ORIGINAIS do CSV) - VERIFIQUE!
CSV_EXPECTED_COLS_HIST = [
    'Date',
    'Home',             # <<< Nome REAL do CSV
    'Away',             # <<< Nome REAL do CSV
    'Goals_H_FT',       # <<< Nome REAL do CSV
    'Goals_A_FT',       # <<< Nome REAL do CSV
    'Odd_H_FT',         # <<< Nome REAL da Odd Casa (já coincide com interno)
    'Odd_D_FT',         # <<< Nome REAL da Odd Empate (já coincide com interno)
    'Odd_A_FT',         # <<< Nome REAL da Odd Fora (já coincide com interno)
    'League',           # <<< Nome REAL da Liga
    # Opcional - Adicione se quiser usar (nomes REAIS do CSV):
    'Odd_Over25_FT',
    'Odd_Under25_FT',
    'Odd_BTTS_Yes',
    'Odd_BTTS_No',
    'XG_Home_Pre',      # (Se for usar xG pré-jogo)
    'XG_Away_Pre',      # (Se for usar xG pré-jogo)
    'XG_Total_Pre',     # (Se for usar xG pré-jogo)
]

# Mapeamento CSV -> Nomes Internos
CSV_HIST_COL_MAP = {k: k for k in CSV_EXPECTED_COLS_HIST} # Inicia mapeando para si mesmo
CSV_HIST_COL_MAP.update({ # Sobrescreve os que precisam de renomeação
    'Date': 'Date',
    'Home': 'Home',             # <<< CHAVE é 'Home', VALOR é 'Home'
    'Away': 'Away',             # <<< CHAVE é 'Away', VALOR é 'Away'
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
REQUIRED_FIXTURE_COLS = ['League', 'HomeTeam', 'AwayTeam', 'Odd_H_FT', 'Odd_D_FT', 'Odd_A_FT', 
                         'Odd_Over25_FT', 'Odd_BTTS_Yes']


# --- Configurações Gerais do Modelo ---
RANDOM_STATE = 42; TEST_SIZE = 0.2; CROSS_VALIDATION_SPLITS = 5; N_JOBS_GRIDSEARCH = -1; ROLLING_WINDOW = 7 #mudei o rolling_window para 7 dias (1 semana) antes era 10 dias
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
    'Media_CG_A',           # Custo Gol Fora (Rolling Mean)
    'Media_CG_H',           # Custo Gol Casa (Rolling Mean)
    'Media_VG_H',           # Valor Gol Fora (Rolling Mean)
    'Media_VG_A',           # Valor Gol Casa (Rolling Mean)
    'CV_HDA',               # Coeficiente de Variação (HDA)
    'Std_CG_A',             # Custo Gol Fora (Rolling Std)
    'Std_CG_H',             # Custo Gol Casa (Rolling Std)
    'Prob_Empate_Poisson',   # Empate de Poisson
    'XG_Total',         # xG Total (Expected Goals) - Se disponível e relevante
]

FEATURE_COLUMNS = NEW_FEATURE_COLUMNS

# --- Métrica para Selecionar o Melhor Modelo ---
BEST_MODEL_METRIC = 'f1_score' #antes:'f1_score'

# --- Configuração dos Modelos a Testar ---

# RandomForest (Grid Corrigido)
rf_param_grid = {
    'n_estimators': [50, 150], # antes: 100,200
    'max_depth': [5, 8, 12, None], # Adicionado None de volta/ antes:10, 20, None
    'criterion': ["gini", "entropy"],
    'max_features': ["sqrt", "log2"], # None removido (pode ser lento)
    'min_samples_split': [5, 10, 20], # Corrigido: mínimo é 2/ antes: 2, 5, 10
    'min_samples_leaf': [3, 5, 10], # Ajustado/ antes: 1,2
    'bootstrap': [True], # Corrigido: usar booleanos/ antes: [True, False]
    'class_weight': [None, 'balanced']
}

# Regressão Logística (Mantido da versão anterior)
lr_param_grid = {
    'C': [0.05, 0.1, 0.5, 1, 5, 10], #0.1, 1, 10
    'penalty': ['l1', 'l2'], #antes: 'l2'
    'solver': ['liblinear', 'saga'], # antes: 'liblinear'
    'class_weight': [None, 'balanced']
}

# LightGBM (Mantido da versão anterior)
lgbm_param_grid = {
    'n_estimators': [75, 150, 250], #antes:50, 100, 200
    'learning_rate': [0.02, 0.05, 0.1],   #antes:0.05, 0.1
    'num_leaves': [10, 20, 31, 40],     #antes:15, 31, 50
    'max_depth': [4, 6, 8, -1],       #antes:-1, 5, 10
    'reg_alpha': [0, 0.01, 0.1],                   # Regularização L1/ add
    'reg_lambda': [0, 0.01, 0.1],                  # Regularização L2/ add
    'colsample_bytree': [0.7, 0.9, 1.0],           # add
    # 'is_unbalance': [True] # Passar no model_kwargs
}

# SVM (SVC - Grid Adaptado/Corrigido)
svm_param_grid = {
    'C': [0.5, 1, 5, 10],  # Menos opções para C/ antes:1, 10
    'gamma': ['scale','auto',  0.1], # Menos opções para gamma/ antes: 'scale', 0.1
    'kernel': ['poly', 'rbf'], # Sigmoid muitas vezes não performa bem
    'degree': [2, 3, 4], # Apenas relevante para kernel='poly'/antes: 2,3
    'class_weight': [None, 'balanced'],
    'decision_function_shape': ['ovo'],

}

# Gaussian Naive Bayes (Grid Corrigido)
gnb_param_grid = {
    'var_smoothing': np.logspace(-9, -2, num=15),
    
}

# KNN (Grid Adaptado)
knn_param_grid = {
    'n_neighbors': [5, 9, 15, 21, 29, 35], # Ímpares evitam empates na votação/antes:3, 5, 7, 9, 11, 13, 15
    'weights': ['uniform', 'distance'], # Testar pesos diferentes
    'metric': ['minkowski', 'manhattan'], # Métricas comuns/antes:'minkowski', 'euclidean', 'manhattan'
    'p': [1, 2] # p=1 é Manhattan, p=2 é Euclidiana
    # 'algorithm': ['auto'] # 'auto' geralmente é suficiente
    # 'leaf_size': [30] # Menos relevante se usar 'auto' ou 'brute'
}

MODEL_CONFIG = {
    'RandomForestClassifier': { # Random Forest Classifier
        'model_kwargs': {'random_state': RANDOM_STATE, 'n_jobs': N_JOBS_GRIDSEARCH},
        'param_grid': rf_param_grid,
        'needs_scaling': False
    },
    
    'LogisticRegression': { # Regressão Logística
        'model_kwargs': {'random_state': RANDOM_STATE, 'max_iter': 2000},
        'param_grid': lr_param_grid,
        'needs_scaling': True
    },
    #'LGBMClassifier': { # LightGBM Classifier
    #    'model_kwargs': {'random_state': RANDOM_STATE, 'n_jobs': N_JOBS_GRIDSEARCH,
    #                     'objective': 'binary', 'metric': 'logloss',
    #                     'is_unbalance': True}, # Usar is_unbalance para LGBM
    #    'param_grid': lgbm_param_grid,
    #    'needs_scaling': False
    #},
     'SVC': { # Support Vector Classifier
        'model_kwargs': {'random_state': RANDOM_STATE, 'probability': True}, # probability=True é necessário para predict_proba (e log_loss/AUC)
        'param_grid': svm_param_grid,
        'needs_scaling': True # SVM PRECISA de scaling
    },
    'GaussianNB': { # Gaussian Naive Bayes
        'model_kwargs': {}, 
        'param_grid': gnb_param_grid,
        'needs_scaling': False # Geralmente não precisa, mas pode testar
    },
    'KNeighborsClassifier': { # K-Nearest Neighbors
        'model_kwargs': {'n_jobs': N_JOBS_GRIDSEARCH},
        'param_grid': knn_param_grid,
        'needs_scaling': True # KNN PRECISA de scaling
    },
}

DEFAULT_EV_THRESHOLD = 0.05 # Threshold padrão para EV (Expected Value) - Ajuste conforme necessário

# --- Configuração GitHub ---

GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')
GITHUB_REPO_NAME = os.environ.get('GITHUB_REPO_NAME')
GITHUB_PREDICTIONS_PATH = f"data/predictions_{MODEL_TYPE_NAME}"