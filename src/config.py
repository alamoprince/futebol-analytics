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

# Arquivo de Dados Históricos (Excel)
HISTORICAL_DATA_FILENAME = "Brasileirao_A_e_B (1).xlsx"
HISTORICAL_DATA_PATH = os.path.join(DATA_DIR, HISTORICAL_DATA_FILENAME)

# --- Arquivos dos Modelos Salvos ---
BEST_F1_MODEL_FILENAME = f"best_model{MODEL_SUFFIX_F1}.joblib"
BEST_F1_MODEL_SAVE_PATH = os.path.join(DATA_DIR, BEST_F1_MODEL_FILENAME)

BEST_ROI_MODEL_FILENAME = f"best_model{MODEL_SUFFIX_ROI}.joblib"
BEST_ROI_MODEL_SAVE_PATH = os.path.join(DATA_DIR, BEST_ROI_MODEL_FILENAME)

MODEL_SAVE_PATH = BEST_F1_MODEL_SAVE_PATH # Caminho padrão para salvar o modelo

# Identificadores para exibir na GUI/CLI
MODEL_ID_F1 = "Melhor F1 (Empate)"
MODEL_ID_ROI = "Melhor ROI (Empate)"

# --- Configurações da Fonte de Dados de Jogos Futuros (CSV GitHub) ---
FIXTURE_FETCH_DAY = "today"
FIXTURE_CSV_URL_TEMPLATE = "https://github.com/futpythontrader/YouTube/raw/main/Jogos_do_Dia/FootyStats/Jogos_do_Dia_FootyStats_{date_str}.csv"

# --- Fonte de Dados Futuros (CSV GitHub) ---
FIXTURE_FETCH_DAY = "today"
FIXTURE_CSV_URL_TEMPLATE = "https://github.com/futpythontrader/YouTube/raw/main/Jogos_do_Dia/FootyStats/Jogos_do_Dia_FootyStats_{date_str}.csv"

# Colunas base ESPERADAS no CSV (nomes ORIGINAIS do CSV) - VERIFIQUE!
CSV_EXPECTED_COLS = [
    'Date', 'Time', 'League', 'Rodada', 'Home', 'Away',
    'Odd_H_FT', 'Odd_D_FT', 'Odd_A_FT', # Odds 1x2
    'Odd_Over25_FT', 'Odd_Under25_FT',
    'Odd_BTTS_Yes', 'Odd_BTTS_No',
    # Adicione outras que o FICTURE_CSV_COL_MAP use
]
# Mapeamento CSV -> Nomes Internos
FIXTURE_CSV_COL_MAP = {k: k for k in CSV_EXPECTED_COLS} # Inicia mapeando para si mesmo
FIXTURE_CSV_COL_MAP.update({ # Sobrescreve os que precisam de renomeação
    'Date': 'Date_Str', 'Time': 'Time_Str', 'Rodada': 'Round',
    'Home': 'HomeTeam', 'Away': 'AwayTeam',
    # Se os nomes no CSV forem diferentes, ajuste as CHAVES aqui
    # 'CSV_Nome_H': 'Odd_H_FT',
    # 'CSV_Nome_D': 'Odd_D_FT',
    # 'CSV_Nome_A': 'Odd_A_FT',
    # 'CSV_Nome_O25': 'Odd_Over25_FT',
    # 'CSV_Nome_U25': 'Odd_Under25_FT',
    # 'CSV_Nome_BTTSY': 'Odd_BTTS_Yes',
    # 'CSV_Nome_BTTSN': 'Odd_BTTS_No',
})
# Colunas internas essenciais após ler e mapear o CSV
REQUIRED_FIXTURE_COLS = ['League', 'HomeTeam', 'AwayTeam', 'Odd_H_FT', 'Odd_D_FT', 'Odd_A_FT', 
                         'Odd_Over25_FT', 'Odd_BTTS_Yes']

# Lista de Ligas Alvo - AJUSTE COM NOMES DO CSV (coluna 'League')
TARGET_LEAGUES = [ # ... etc ...
]

# --- Configurações Gerais do Modelo ---
RANDOM_STATE = 42; TEST_SIZE = 0.2; CROSS_VALIDATION_SPLITS = 3; N_JOBS_GRIDSEARCH = -1; ROLLING_WINDOW = 10
RESULT_MAPPING = {'D': 0, 'H': 1, 'A': 2} # Usado para Ptos
CLASS_NAMES = ['Nao_Empate', 'Empate'] # Alvo binário

# --- Nomes das Colunas Internas (Odds/Gols) (NOMES INTERNOS)---
ODDS_COLS = {'home': 'Odd_H_FT', 'draw': 'Odd_D_FT', 'away': 'Odd_A_FT'}
GOALS_COLS = {'home': 'Goals_H_FT', 'away': 'Goals_A_FT'} 
OTHER_ODDS_NAMES = [ 'Odd_Over25_FT', 'Odd_Under25_FT', 'Odd_BTTS_Yes', 'Odd_BTTS_No' ]

# Colunas base ESPERADAS no Excel (nomes ORIGINAIS do Excel)
EXCEL_EXPECTED_COLS = [
    'Date', 'Home', 'Away',
    'Goals_H_FT', 'Goals_A_FT', # Nomes que GOALS_COLS usa nas CHAVES
    'Odd_H_FT', 'Odd_D_FT', 'Odd_A_FT', # Nomes que ODDS_COLS usa
    'Odd_Over25_FT', 'Odd_BTTS_Yes' # Outras odds diretas usadas como features
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
    'Media_CG_H',           # Custo Gol Casa (Rolling Mean)
    'Media_CG_A',           # Custo Gol Fora (Rolling Mean)
    'Media_VG_H',           # Valor Gol Fora (Rolling Mean)
    'Odd_H_FT',             # Odds Casa (do CSV ou Histórico)
    'CV_HDA',               # Coeficiente de Variação (HDA)
    'Std_CG_A',             # Custo Gol Fora (Rolling Std)
]

FEATURE_COLUMNS = NEW_FEATURE_COLUMNS

# --- Métrica para Selecionar o Melhor Modelo ---
BEST_MODEL_METRIC = 'f1_score_draw'

# --- Configuração dos Modelos a Testar (SVC e GNB) ---
#svm_param_grid = { 'C': [0.5, 1, 5, 10], 'gamma': ['scale', 0.1, 0.5], 'kernel': ['rbf', 'linear'], 'class_weight': [None, 'balanced'], }
#gnb_param_grid = { 'var_smoothing': np.logspace(-9, -2, num=15) }
# --- Métrica para Selecionar o Melhor Modelo ---
BEST_MODEL_METRIC = 'f1_score_draw' # Ou 'roi'

# --- Configuração dos Modelos a Testar ---

# RandomForest (Grid Corrigido)
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None], # Adicionado None de volta
    'criterion': ["gini", "entropy"],
    'max_features': ["sqrt", "log2"], # None removido (pode ser lento)
    'min_samples_split': [2, 5, 10], # Corrigido: mínimo é 2
    'min_samples_leaf': [1, 2], # Ajustado
    'bootstrap': [True, False], # Corrigido: usar booleanos
    'class_weight': [None, 'balanced']
}

# Regressão Logística (Mantido da versão anterior)
lr_param_grid = {
    'C': [0.1, 1, 10], 'penalty': ['l2'], 'solver': ['liblinear'],
    'class_weight': [None, 'balanced']
}

# LightGBM (Mantido da versão anterior)
lgbm_param_grid = {
    'n_estimators': [50, 100, 200], 'learning_rate': [0.05, 0.1],
    'num_leaves': [15, 31, 50], 'max_depth': [-1, 5, 10],
    # 'is_unbalance': [True] # Passar no model_kwargs
}

# SVM (SVC - Grid Adaptado/Corrigido)
svm_param_grid = {
    'C': [1, 10],  # Menos opções para C
    'gamma': ['scale', 0.1], # Menos opções para gamma
    'kernel': ['poly', 'rbf'], # Sigmoid muitas vezes não performa bem
    'degree': [2, 3], # Apenas relevante para kernel='poly'
    'class_weight': [None, 'balanced'],
    'decision_function_shape': ['ovo'],

}

# Gaussian Naive Bayes (Grid Corrigido)
gnb_param_grid = {
    'var_smoothing': np.logspace(-9, -2, num=15),
    
}

# KNN (Grid Adaptado)
knn_param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15], # Ímpares evitam empates na votação
    'weights': ['uniform', 'distance'], # Testar pesos diferentes
    'metric': ['minkowski', 'euclidean', 'manhattan'] # Métricas comuns
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
    'LGBMClassifier': { # LightGBM Classifier
        'model_kwargs': {'random_state': RANDOM_STATE, 'n_jobs': N_JOBS_GRIDSEARCH,
                         'objective': 'binary', 'metric': 'logloss',
                         'is_unbalance': True}, # Usar is_unbalance para LGBM
        'param_grid': lgbm_param_grid,
        'needs_scaling': False
    },
     'SVC': { # Support Vector Classifier
        'model_kwargs': {'random_state': RANDOM_STATE, 'probability': True}, # probability=True é necessário para predict_proba (e log_loss/AUC)
        'param_grid': svm_param_grid,
        'needs_scaling': True # SVM PRECISA de scaling
    },
    #'GaussianNB': { # Gaussian Naive Bayes
    #    'model_kwargs': {}, 
    #    'param_grid': gnb_param_grid,
    #    'needs_scaling': False # Geralmente não precisa, mas pode testar
    #},
    'KNeighborsClassifier': { # K-Nearest Neighbors
        'model_kwargs': {'n_jobs': N_JOBS_GRIDSEARCH},
        'param_grid': knn_param_grid,
        'needs_scaling': True # KNN PRECISA de scaling
    },
}

# --- Configuração GitHub ---

GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')
GITHUB_REPO_NAME = os.environ.get('GITHUB_REPO_NAME')
GITHUB_PREDICTIONS_PATH = f"data/predictions_{MODEL_TYPE_NAME}"