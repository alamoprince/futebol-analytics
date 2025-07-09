# --- src/config.py ---
import os
import numpy as np
from datetime import date, timedelta
import random

from logger_config import setup_logger

logger = setup_logger("Config")

try:
    from skopt.space import Real, Categorical, Integer
    SKOPT_AVAILABLE_CONFIG = True
except ImportError:
    logger.warning("AVISO (config.py): skopt não instalado. Espaços Bayes não funcionarão.")
    SKOPT_AVAILABLE_CONFIG = False
    # Define classes dummy para evitar erros fatais, mas Bayes não funcionará
    Real = lambda *args, **kwargs: None
    Categorical = lambda *args, **kwargs: None
    Integer = lambda *args, **kwargs: None

try:
    import lightgbm as lgb
    LGBMClassifier = lgb.LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    logger.warning("AVISO: LightGBM não instalado.")
    lgb = None; LGBMClassifier = None; LGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
    logger.info("Biblioteca 'catboost' carregada.")
except ImportError:
    logger.warning("AVISO: CatBoost não instalado (pip install catboost).")
    CatBoostClassifier = None
    CATBOOST_AVAILABLE = False

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

HISTORICAL_DATA_FILENAME_3 = "dados_combinados_2019_2025.xlsx"
HISTORICAL_DATA_PATH_3 = os.path.join(DATA_DIR, HISTORICAL_DATA_FILENAME_3)

SCRAPER_BASE_URL = "https://flashscore.com"
SCRAPER_TARGET_DAY =  "today" # "today" ou "tomorrow"
SCRAPER_TARGET_DATE = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d") 

# --- Fonte de Dados Futuros (CSV GitHub) ---
FIXTURE_FETCH_DAY = "today" # Ou "today"
FIXTURE_CSV_URL_TEMPLATE = "https://raw.githubusercontent.com/alamoprince/data_base_fut_analytics/main/data/raw_scraped/scraped_fixtures_{date_str}.csv" # Atualizado branch

CHROMEDRIVER_PATH = os.path.join(BASE_DIR, 'chromedriver.exe')
SCRAPER_TIMEOUT = 13
SCRAPER_ODDS_TIMEOUT = 15
SCRAPER_SLEEP_BETWEEN_GAMES = 5
SCRAPER_SLEEP_AFTER_NAV = 15

# --- Configurações Pi-Rating ---
PI_RATING_INITIAL = 0  
PI_RATING_K_FACTOR = 30   
PI_RATING_HOME_ADVANTAGE = 65 

# --- Filtro de Ligas do Scraper ---
SCRAPER_FILTER_LEAGUES = False

# --- Filtro de Ligas Alvo para Scraper e Análise ---
APPLY_LEAGUE_FILTER_ON_HISTORICAL = True

# NOMES INTERNOS PADRÃO 
INTERNAL_LEAGUE_NAMES = {
     'ARGENTINA 1': 'Argentina - Primera División', # Nome descritivo opcional
     'SPAIN 2': 'Espanha - Segunda División',
     'ITALY 1': 'Itália - Serie A',
     'ITALY 2': 'Itália - Serie B',
     'BRAZIL 1': 'Brasil - Série A',
     'BRAZIL 2': 'Brasil - Série B',
     'ROMANIA 1': 'Romênia - Liga 1',
     'SERBIA 1': 'Serbia - SuperLiga', 
     'FRANCE 1': 'France - Ligue 1',
      # --- Argentina ---
    'ARGENTINA: PRIMERA': 'ARGENTINA 2', # Suposição: Seria a Primera B Nacional (Segunda Divisão) - VERIFICAR!

    # --- Ásia ---
    'ASIA: AFC CHAMPION': 'ASIA AFC CHAMPIONS', # Torneio Continental

    # --- Austrália ---
    'AUSTRALIA: A-LEAGUE': 'AUSTRALIA 1', # Principal liga

    # --- Polônia ---
    'POLAND: DIVISION 1': 'POLAND 2', # Division 1 na Polônia é a segunda divisão (Ekstraklasa é a 1ª)
    'POLAND: CENTRAL YOUTH LEAGUE': 'POLAND YOUTH',
    'POLAND: EKSTRALIGA WOMEN': 'POLAND WOMEN 1', # Principal liga feminina

    # --- Portugal ---
    'PORTUGAL: LIGA PORTUGAL': 'PORTUGAL 1', # Primeira Liga
    'PORTUGAL: LIGA PORTUGAL 2': 'PORTUGAL 2', # Segunda Liga
    'PORTUGAL: LIGA 3 - PROMOTION GROUP': 'PORTUGAL 3', # Terceira Liga (fase)

    # --- Romênia ---
    'ROMANIA 1': 'ROMANIA 1', # Já parece padronizado (Superliga)
    'ROMANIA: SUPERLIGA - CHAMPIONSHIP GROUP': 'ROMANIA 1', # Mapeia fase para liga principal
    'ROMANIA: LIGA 2 - RELEGATION GROUP': 'ROMANIA 2', # Segunda Liga (fase)
    'ROMANIA: SUPERLIGA WOMEN - CHAMPIONSHIP GROUP': 'ROMANIA WOMEN 1', # Principal liga feminina (fase)

    # --- Rússia ---
    'RUSSIA: PREMIER LEAGUE': 'RUSSIA 1',
    'RUSSIA: FNL': 'RUSSIA 2', # Football National League (Segunda Divisão)

    # --- San Marino ---
    'SAN MARINO: CAMPIONATO SAMMARINESE - PLAY OFFS - QUARTER-FINALS': 'SAN MARINO 1 PLAYOFF', # Liga principal (fase)

    # --- São Tomé e Príncipe ---
    'SÃO TOMÉ AND PRÍNCIPE: CAMPEONATO NACIONAL': 'SAO TOME 1', # Principal liga

    # --- Escócia ---
    'SCOTLAND: PREMIERSHIP - RELEGATION GROUP': 'SCOTLAND 1',
    'SCOTLAND: PREMIERSHIP - CHAMPIONSHIP GROUP': 'SCOTLAND 1',
    'SCOTLAND: LEAGUE ONE': 'SCOTLAND 3', # Terceira divisão
    'SCOTLAND: LEAGUE TWO': 'SCOTLAND 4', # Quarta divisão
    'SCOTLAND: LEAGUE TWO - RELEGATION - PLAY OFFS - SEMI-FINALS': 'SCOTLAND 4 PLAYOFF',

    # --- Sérvia ---
    'SERBIA: SUPER LIGA - RELEGATION GROUP': 'SERBIA 1',
    'SERBIA: SUPER LIGA - CHAMPIONSHIP GROUP': 'SERBIA 1',
    'SERBIA: PRVA LIGA - RELEGATION GROUP': 'SERBIA 2', # Segunda divisão (fase)
    'SERBIA: PRVA LIGA - CHAMPIONSHIP GROUP': 'SERBIA 2', # Segunda divisão (fase)

    # --- Sierra Leone ---
    'SIERRA LEONE: PREMIER LEAGUE': 'SIERRA LEONE 1',

    # --- Singapura ---
    'SINGAPORE: PREMIER LEAGUE': 'SINGAPORE 1',

    # --- Eslováquia ---
    'SLOVAKIA: NIKE LIGA - RELEGATION GROUP': 'SLOVAKIA 1', # Principal liga (fase)
    'SLOVAKIA: NIKE LIGA - CHAMPIONSHIP GROUP': 'SLOVAKIA 1', # Principal liga (fase)
    'SLOVAKIA: 2. LIGA': 'SLOVAKIA 2',
    'SLOVAKIA: 1. LIGA WOMEN - RELEGATION GROUP': 'SLOVAKIA WOMEN 1', # Principal liga feminina (fase)
    'SLOVAKIA: 1. LIGA WOMEN - CHAMPIONSHIP GROUP': 'SLOVAKIA WOMEN 1', # Principal liga feminina (fase)

    # --- Eslovênia ---
    'SLOVENIA: PRVA LIGA': 'SLOVENIA 1',
    'SLOVENIA: 2. SNL': 'SLOVENIA 2',

    # --- Somália ---
    'SOMALIA: NATIONAL LEAGUE': 'SOMALIA 1',

    # --- África do Sul ---
    'SOUTH AFRICA: PREMIERSHIP': 'SOUTH AFRICA 1',
    'SOUTH AFRICA: MOTSEPE FOUNDATION CHAMPIONSHIP': 'SOUTH AFRICA 2', # Segunda divisão

    # --- América do Sul (Torneio específico) ---
    'SOUTH AMERICA: SOUTH AMERICAN CHAMPIONSHIP WOMEN U17': 'SOUTH AMERICA U17 WOMEN',

    # --- Coreia do Sul ---
    'SOUTH KOREA: K LEAGUE 1': 'SOUTH KOREA 1',

    # --- Espanha ---
    'SPAIN 2': 'SPAIN 2', # Já parece padronizado (Segunda División)
    'SPAIN: PRIMERA RFEF': 'SPAIN 3', # Terceira divisão
    'SPAIN: LIGA F WOMEN': 'SPAIN WOMEN 1', # Principal liga feminina

    # --- Suriname ---
    'SURINAME: SML': 'SURINAME 1', # Surinaamse Major League

    # --- Suécia ---
    'SWEDEN: ALLSVENSKAN': 'SWEDEN 1', # Principal liga
    'SWEDEN: SUPERETTAN': 'SWEDEN 2', # Segunda divisão
    'SWEDEN: ALLSVENSKAN WOMEN': 'SWEDEN WOMEN 1', # Principal liga feminina

    # --- Suíça ---
    'SWITZERLAND: SUPER LEAGUE - RELEGATION GROUP': 'SWITZERLAND 1',
    'SWITZERLAND: SUPER LEAGUE - CHAMPIONSHIP GROUP': 'SWITZERLAND 1',
    'SWITZERLAND: CHALLENGE LEAGUE': 'SWITZERLAND 2', # Segunda divisão
    'SWITZERLAND: PROMOTION LEAGUE': 'SWITZERLAND 3', # Terceira divisão
    'SWITZERLAND: SUPER LEAGUE WOMEN - PLAY OFFS - SEMI-FINALS': 'SWITZERLAND WOMEN 1 PLAYOFF',
    'SWITZERLAND: SUPER LEAGUE WOMEN - RELEGATION': 'SWITZERLAND WOMEN 1 RELEGATION',
    'SWITZERLAND: SUPER LEAGUE WOMEN - PLACEMENT PLAY OFFS - FINAL': 'SWITZERLAND WOMEN 1 PLAYOFF',

    # --- Tajiquistão ---
    'TAJIKISTAN: VYSSHAYA LIGA': 'TAJIKISTAN 1', # Principal liga

    # --- Tailândia ---
    'THAILAND: THAI FA CUP - QUARTER-FINALS': 'THAILAND CUP', # Copa Nacional

    # --- Togo ---
    'TOGO: CHAMPIONNAT NATIONAL': 'TOGO 1', # Principal liga

    # --- Trinidad e Tobago ---
    'TRINIDAD AND TOBAGO: TT PREMIER LEAGUE': 'TRINIDAD 1',

    # --- Tunísia ---
    'TUNISIA: LIGUE PROFESSIONNELLE 1': 'TUNISIA 1',

    # --- Turquia ---
    'TURKEY: SUPER LIG': 'TURKEY 1',

    # --- Turcomenistão ---
    'TURKMENISTAN: YOKARY LIGA': 'TURKMENISTAN 1', # Principal liga

    # --- Uganda ---
    'UGANDA: UGANDA CUP - SEMI-FINALS': 'UGANDA CUP', # Copa Nacional

    # --- Ucrânia ---
    'UKRAINE: PREMIER LEAGUE': 'UKRAINE 1',
    'UKRAINE: PERSHA LIGA - PROMOTION GROUP': 'UKRAINE 2', # Segunda divisão (fase)
    'UKRAINE: PERSHA LIGA - RELEGATION GROUP': 'UKRAINE 2', # Segunda divisão (fase)

    # --- Emirados Árabes Unidos ---
    'UNITED ARAB EMIRATES: UAE LEAGUE': 'UAE 1',

    # --- Uruguai ---
    'URUGUAY: LIGA AUF URUGUAYA - APERTURA': 'URUGUAY 1', # Principal liga (fase)

    # --- EUA ---
    'USA: MLS': 'USA 1', # Major League Soccer
    'USA: USL CHAMPIONSHIP': 'USA 2', # Segunda divisão (geralmente considerada)
    'USA: NWSL WOMEN': 'USA WOMEN 1', # Principal liga feminina
    'USA: USL SUPER LEAGUE WOMEN': 'USA WOMEN 2', # Nova segunda liga feminina (a verificar nível exato)

    # --- Uzbequistão ---
    'UZBEKISTAN: SUPER LEAGUE': 'UZBEKISTAN 1',

    # --- Venezuela ---
    'VENEZUELA: LIGA FUTVE - APERTURA - QUADRANGULAR': 'VENEZUELA 1', # Principal liga (fase)

    # --- Vietnã ---
    'VIETNAM: V.LEAGUE 1': 'VIETNAM 1',

    # --- Mundo (Amistosos) ---
    'WORLD: FRIENDLY INTERNATIONAL WOMEN': 'WORLD FRIENDLY WOMEN',

    # --- Zâmbia ---
    'ZAMBIA: SUPER LEAGUE': 'ZAMBIA 1',

    # --- Zimbábue ---
    'ZIMBABWE: PREMIER SOCCER LEAGUE': 'ZIMBABWE 1',
}

# Lista apenas dos identificadores curtos, se preferir usá-los internamente
TARGET_LEAGUES_INTERNAL_IDS = list(INTERNAL_LEAGUE_NAMES.keys())

#database historica
TARGET_LEAGUES_1 = {'Argentina Primera División':'ARGENTINA 1','Spain Segunda División':'SPAIN 2', 'Serbia SuperLiga':'SERBIA 1', 'France: Ligue 1':'FRANCE 1','Italy Serie A':'ITALY 1', 'Italy Serie B':'ITALY 2', 'Brazil Serie A':'BRAZIL 1','Brazil Serie B':'BRAZIL 2', 'Romania Liga I':'ROMANIA 1'}
TARGET_LEAGUES_2 = {'ARGENTINA 1':'ARGENTINA 1','SPAIN 2':'SPAIN 2', 'SERBIA 1':'SERBIA 1', 'FRANCE 1':'FRANCE 1','ITALY 1':'ITALY 1', 'ITALY 2':'ITALY 2', 'BRAZIL 1':'BRAZIL 1','BRAZIL 2':'BRAZIL 2', 'ROMANIA 1':'ROMANIA 1'}

#database futuro
SCRAPER_TO_INTERNAL_LEAGUE_MAP = {
     # Nome no Scraper : Nome Interno/Histórico
    'ARGENTINA: Torneo Betano - Apertura': 'ARGENTINA 1',
    'SPAIN: LaLiga2': 'SPAIN 2',
    'ITALY: Serie A': 'ITALY 1',
    'ITALY: Serie B': 'ITALY 2',
    'BRAZIL: Serie A Betano': 'BRAZIL 1',
    'BRAZIL: Serie B Superbet': 'BRAZIL 2',
    'ROMANIA: Superliga - Relegation Group': 'ROMANIA 1'
}

# --- Arquivos dos Modelos Salvos ---
BEST_F1_MODEL_FILENAME = f"best_model{MODEL_SUFFIX_F1}.joblib"
BEST_F1_MODEL_SAVE_PATH = os.path.join(DATA_DIR, BEST_F1_MODEL_FILENAME)

BEST_ROI_MODEL_FILENAME = f"best_model{MODEL_SUFFIX_ROI}.joblib"
BEST_ROI_MODEL_SAVE_PATH = os.path.join(DATA_DIR, BEST_ROI_MODEL_FILENAME)

MODEL_SAVE_PATH = BEST_F1_MODEL_SAVE_PATH # Caminho padrão para salvar o modelo

APPLY_LEAGUE_FILTER_ON_HISTORICAL = True

# --- Arquivos dos Modelos Salvos ---
BEST_F1_MODEL_FILENAME = f"best_model{MODEL_SUFFIX_F1}.joblib"
BEST_F1_MODEL_SAVE_PATH = os.path.join(DATA_DIR, BEST_F1_MODEL_FILENAME)
BEST_ROI_MODEL_FILENAME = f"best_model{MODEL_SUFFIX_ROI}.joblib"
BEST_ROI_MODEL_SAVE_PATH = os.path.join(DATA_DIR, BEST_ROI_MODEL_FILENAME)
MODEL_SAVE_PATH = BEST_F1_MODEL_SAVE_PATH # Padrão
MODEL_ID_F1 = "Melhor F1 (Empate)"
MODEL_ID_ROI = "Melhor ROI (Empate)"

# --- Nomes Colunas Internas ---
XG_COLS = {'home': 'XG_H', 'away': 'XG_A', 'total': 'XG_Total'}
ODDS_COLS = {'home': 'Odd_H_FT', 'draw': 'Odd_D_FT', 'away': 'Odd_A_FT'}
GOALS_COLS = {'home': 'Goals_H_FT', 'away': 'Goals_A_FT'}
OTHER_ODDS_NAMES = [ 'Odd_Over25_FT', 'Odd_Under25_FT', 'Odd_BTTS_Yes', 'Odd_BTTS_No' ]

# Mapeamento CSV/Excel -> Nomes Internos (AJUSTADO E SIMPLIFICADO)
CSV_HIST_COL_MAP = {
    # Coluna no Arquivo : Nome Interno Desejado
    # Essenciais (Baseado no seu CSV)
    'Date': 'Date',
    'Time': 'Time',
    'Time_Str': 'Time_Str',             
    'League': 'League',
    'Home': 'Home',        
    'Away': 'Away',  
    'HomeTeam': 'Home',
    'AwayTeam': 'Away',       
    'Goals_H_FT': 'Goals_H_FT',
    'Goals_A_FT': 'Goals_A_FT', 
    'Odd_H_FT': 'Odd_H_FT',
    'Odd_D_FT': 'Odd_D_FT',
    'Odd_A_FT': 'Odd_A_FT',
    # Outras Odds
    'Odd_Over25_FT': 'Odd_Over25_FT',
    'Odd_Under25_FT': 'Odd_Under25_FT',
    'Odd_BTTS_Yes': 'Odd_BTTS_Yes',
    'Odd_BTTS_No': 'Odd_BTTS_No',
     # xG (Ajuste os nomes das colunas do SEU arquivo aqui, se diferentes)
    'XG_Home': XG_COLS['home'],
    'XG_Away': XG_COLS['away'],
    'XG_Total': XG_COLS['total'],
    # Stats (Ajuste os nomes das colunas do SEU arquivo aqui, se diferentes)
    'Shots_H': 'Shots_H',
    'Shots_A': 'Shots_A',
    'ShotsOnTarget_H': 'ShotsOnTarget_H',
    'ShotsOnTarget_A': 'ShotsOnTarget_A',
    'Corners_H_FT': 'Corners_H_FT',
    'Corners_A_FT': 'Corners_A_FT',

}
# Colunas internas essenciais APÓS mapeamento para buscar jogos futuros
# Ajustado para incluir Time_Str, remover Time (que será usado para criar Time_Str)
REQUIRED_FIXTURE_COLS = ['League','Time_Str', 'Home', 'Away', 'Odd_H_FT', 'Odd_D_FT', 'Odd_A_FT'] # Mínimo necessário

# --- Configurações Gerais do Modelo ---
RANDOM_STATE = 42
TEST_SIZE = 0.15 # Fração para teste final
CROSS_VALIDATION_SPLITS = 5 # Número de splits para TimeSeriesSplit na CV
N_JOBS_GRIDSEARCH = -1 # Usar todos os processadores
ROLLING_WINDOW = 5 # <<< AJUSTE AQUI a janela padrão para médias simples e momentum PiRating
FEATURE_EPSILON = 1e-9 # Valor pequeno para evitar divisão por zero


# --- Nomes das Features ---
# Interações (Manter nomes)
INTERACTION_P_D_NORM_X_CV_HDA = 'pDnorm_x_CVHDA'
INTERACTION_P_D_NORM_DIV_CV_HDA = 'pDnorm_div_CVHDA'
INTERACTION_P_D_NORM_X_PIR_DIFF = 'pDnorm_x_PiRDiffAbs'
INTERACTION_P_D_NORM_DIV_PIR_DIFF = 'pDnorm_div_PiRDiffAbs'
INTERACTION_ODD_D_X_PIR_DIFF = 'OddD_x_PiRDiffAbs'
INTERACTION_ODD_D_DIV_PIR_DIFF = 'OddD_div_PiRDiffAbs'
INTERACTION_PIR_PROBH_X_ODD_H = 'PiRProbH_x_OddH'
INTERACTION_AVG_GOLS_MARC_DIFF = 'AvgGolsMarc_Diff'
INTERACTION_AVG_GOLS_SOFR_DIFF = 'AvgGolsSofr_Diff'

# Momentum (Nome baseado em ROLLING_WINDOW)
PIRATING_MOMENTUM_H = f'PiR_Mom_{ROLLING_WINDOW}G_H'
PIRATING_MOMENTUM_A = f'PiR_Mom_{ROLLING_WINDOW}G_A'
PIRATING_MOMENTUM_DIFF = f'PiR_Mom_{ROLLING_WINDOW}G_Diff'

# Nomes EWMA
EWMA_SPAN_SHORT = 3  # <<< AJUSTE AQUI o span curto
EWMA_SPAN_LONG = 5 # <<< AJUSTE AQUI o span longo
EWMA_VG_H_SHORT = f'EWMA_VG_H_s{EWMA_SPAN_SHORT}'
EWMA_VG_A_SHORT = f'EWMA_VG_A_s{EWMA_SPAN_SHORT}'
EWMA_CG_H_SHORT = f'EWMA_CG_H_s{EWMA_SPAN_SHORT}'
EWMA_CG_A_SHORT = f'EWMA_CG_A_s{EWMA_SPAN_SHORT}'
EWMA_GolsMarc_H_LONG = f'EWMA_GolsMarc_H_s{EWMA_SPAN_LONG}'
EWMA_GolsMarc_A_LONG = f'EWMA_GolsMarc_A_s{EWMA_SPAN_LONG}'
EWMA_GolsSofr_H_LONG = f'EWMA_GolsSofr_H_s{EWMA_SPAN_LONG}'
EWMA_GolsSofr_A_LONG = f'EWMA_GolsSofr_A_s{EWMA_SPAN_LONG}'

ALL_CANDIDATE_FEATURES = [
    'p_D_norm', 'abs_ProbDiff_Norm', 'p_H_norm', 'p_A_norm', 'Media_VG_H',
    'Media_VG_A', 'Media_CG_H', 'Media_CG_A', 'Media_Ptos_H', 'Media_Ptos_A',
    'Std_CG_H', 'Std_CG_A', 'Std_VG_H', 'Std_VG_A', 'Std_Ptos_H', 'Std_Ptos_A',
    'Odd_D_Cat', 'CV_HDA', 'Diff_Media_CG', 'Odd_H_FT', 'Odd_D_FT', 'Odd_A_FT',
    'Odd_Over25_FT', 'Odd_BTTS_Yes', 'Avg_Gols_Marcados_H', 'Avg_Gols_Sofridos_H',
    'Avg_Gols_Marcados_A', 'Avg_Gols_Sofridos_A', 'Prob_Empate_Poisson',
    'XG_H', 'XG_A', 'XG_Total', 'PiRating_H', 'PiRating_A',
    'PiRating_Diff', 'PiRating_Prob_H', PIRATING_MOMENTUM_H, PIRATING_MOMENTUM_A,
    PIRATING_MOMENTUM_DIFF, INTERACTION_P_D_NORM_DIV_CV_HDA,
    INTERACTION_P_D_NORM_X_PIR_DIFF, INTERACTION_ODD_D_DIV_PIR_DIFF,
    INTERACTION_PIR_PROBH_X_ODD_H, INTERACTION_AVG_GOLS_MARC_DIFF,
    INTERACTION_AVG_GOLS_SOFR_DIFF, EWMA_VG_H_SHORT, EWMA_VG_A_SHORT,
    EWMA_CG_H_SHORT, EWMA_CG_A_SHORT, EWMA_GolsMarc_H_LONG, EWMA_GolsMarc_A_LONG,
    EWMA_GolsSofr_H_LONG, EWMA_GolsSofr_A_LONG,
]
ALL_CANDIDATE_FEATURES = sorted(list(set(ALL_CANDIDATE_FEATURES)))
logger.debug(f"Total de features candidatas: {len(ALL_CANDIDATE_FEATURES)}")

# --- Lista das Features FINAIS para o Modelo BackDraw ---
FEATURE_COLUMNS = [
    
]

NEW_FEATURE_COLUMNS = [
    'Prob_Empate_Poisson',
    'abs_ProbDiff_Norm',
    'Odd_Over25_FT', 
    'CV_HDA',
    PIRATING_MOMENTUM_A,
    'Media_VG_H',
    'Media_VG_A',
]   

FEATURE_COLUMNS = NEW_FEATURE_COLUMNS

# --- Métrica para Selecionar o Melhor Modelo ---
BEST_MODEL_METRIC = 'f1_score_draw' #antes:'f1_score'
BEST_MODEL_METRIC_ROI = 'roi' # ROI (Expected Value) - Para o modelo de ROI

# --- Métrica Principal e Limiares Default ---
BEST_MODEL_METRIC = 'f1_score_draw'
BEST_MODEL_METRIC_ROI = 'roi'
DEFAULT_F1_THRESHOLD = 0.25
DEFAULT_EV_THRESHOLD = 0.1
MIN_RECALL_FOR_PRECISION_OPT = 0.25
MIN_PROB_THRESHOLD_FOR_HIGHLIGHT = 0.2

# --- Configuração Otimização Bayesiana/GridSearch ---
BAYESIAN_OPT_N_ITER = 50 # Mantido

categorical_feature_names = ['Odd_D_Cat']
categorical_features_indices = [
    i for i, col in enumerate(NEW_FEATURE_COLUMNS)
    if col in [categorical_feature_names]
    ]

# --- Configuração dos Modelos a Testar ---


rf_search_spaces = {
    'classifier__n_estimators': Integer(50, 500), # Aumentado limite superior
    'classifier__max_depth': Integer(5, 35),       # Aumentado limite superior
    'classifier__min_samples_split': Integer(5, 80),# Aumentado limite superior
    'classifier__min_samples_leaf': Integer(3, 50), # Aumentado limite superior
    'classifier__max_features': Categorical(['sqrt', 'log2', 0.6, 0.8]),
    'classifier__class_weight': Categorical(['balanced', 'balanced_subsample', None]),
    'classifier__bootstrap': Categorical([True, False]),
    # 'sampler__k_neighbors': Integer(3, 11) # Exemplo otimização SMOTE
}

lr_search_spaces = {
    'classifier__C': Real(1e-4, 1e4, prior='log-uniform'), # Range bem maior
    'classifier__penalty': Categorical(['l1', 'l2']),
    'classifier__solver': Categorical(['liblinear', 'saga']), # 'saga' suporta l1/l2
    'classifier__class_weight': Categorical(['balanced', None]),
    'classifier__max_iter': Integer(1000, 7000), # Aumentado um pouco
}

lgbm_search_spaces = {
    'classifier__n_estimators': Integer(100, 1000), # Mais estimadores
    'classifier__learning_rate': Real(0.005, 0.15, prior='log-uniform'), # Learning rate menor
    'classifier__num_leaves': Integer(10, 100),
    'classifier__max_depth': Integer(3, 18), # Um pouco mais raso talvez
    'classifier__reg_alpha': Real(1e-3, 5.0, prior='log-uniform'), # Maior range L1
    'classifier__reg_lambda': Real(1e-3, 5.0, prior='log-uniform'),# Maior range L2
    'classifier__colsample_bytree': Real(0.4, 0.9), # Range um pouco maior
    'classifier__subsample': Real(0.4, 0.9),       
    'classifier__boosting_type': Categorical(['gbdt', 'dart']),
}

svc_search_spaces = {
    'classifier__C': Real(1e-2, 1e3, prior='log-uniform'), # Range bem maior para C
    'classifier__kernel': Categorical(['rbf']), # Mantem RBF por enquanto
    'classifier__gamma': Real(1e-4, 1.0, prior='log-uniform'), # Range um pouco maior
    'classifier__class_weight': Categorical(['balanced', None]),
}

knn_search_spaces = {
    'classifier__n_neighbors': Integer(5, 81, prior='uniform'), # Range maior, ímpares
    'classifier__weights': Categorical(['uniform', 'distance']),
    'classifier__metric': Categorical(['minkowski', 'manhattan']), #'euclidean' é minkowski com p=2
    'classifier__p': Integer(1, 3),
}

# GNB usa param_grid para GridSearchCV (se Bayes não estiver ativo ou se preferir)
gnb_param_grid = { 'classifier__var_smoothing': np.logspace(-11, 0, num=20) } # Aumenta range

catboost_search_spaces = {
    'classifier__iterations': Integer(100, 1000), # Pode aumentar mais depois
    'classifier__learning_rate': Real(0.008, 0.15, prior='log-uniform'),
    'classifier__depth': Integer(4, 12),
    'classifier__l2_leaf_reg': Real(1, 20, prior='uniform'),
    'classifier__border_count': Integer(32, 255), 
     'classifier__subsample': Real(0.5, 1.0),       
    'classifier__boosting_type': Categorical(['Plain', 'Ordered']),
}

# --- Configuração Final dos Modelos ---
MODEL_CONFIG = {}

if SKOPT_AVAILABLE_CONFIG: # Só adiciona se skopt estiver disponível (para BayesSearch)
    MODEL_CONFIG['RandomForestClassifier'] = {
        'model_kwargs': {'random_state': RANDOM_STATE, 'n_jobs': N_JOBS_GRIDSEARCH, 'class_weight': 'balanced_subsample' if 'balanced_subsample' in rf_search_spaces['classifier__class_weight'].categories else None}, # Define um default razoável
        'search_spaces': rf_search_spaces, 'param_grid': None, 'needs_scaling': False
    }
    MODEL_CONFIG['LogisticRegression'] = {
        'model_kwargs': {'random_state': RANDOM_STATE, 'solver':'saga', 'max_iter': 3000}, # Saga suporta L1/L2
        'search_spaces': lr_search_spaces, 'param_grid': None, 'needs_scaling': True
    }
    if LGBM_AVAILABLE:
        MODEL_CONFIG['LGBMClassifier'] = {
            'model_kwargs': { # Kwargs que NÃO estão no search_spaces
                'random_state': RANDOM_STATE, 'n_jobs': N_JOBS_GRIDSEARCH,
                'objective': 'binary',
                #'fit_params': {'classifier__callbacks': [lgb.early_stopping(100, verbose=False)]}
                # 'metric': 'logloss', # Métrica interna de avaliação (não de otimização CV)
                # early stopping é passado via fit_params no model_trainer
             },
            'search_spaces': lgbm_search_spaces, 'param_grid': None, 'needs_scaling': False
        }
    MODEL_CONFIG['SVC'] = {
        'model_kwargs': {'random_state': RANDOM_STATE, 'probability': True},
        'search_spaces': svc_search_spaces, 'param_grid': None, 'needs_scaling': True
    }
    MODEL_CONFIG['KNeighborsClassifier'] = {
        'model_kwargs': {'n_jobs': N_JOBS_GRIDSEARCH},
        'search_spaces': knn_search_spaces, 'param_grid': None, 'needs_scaling': True
    }
    if CATBOOST_AVAILABLE:
        MODEL_CONFIG['CatBoostClassifier'] = {
            'model_kwargs': {
                'random_state': RANDOM_STATE, 'verbose': 0,
                'eval_metric': 'F1', # Métrica para monitorar (ex: em early stopping)
                'loss_function': 'Logloss',
                # Early stopping é passado via fit_params no model_trainer
                'cat_features': categorical_features_indices if categorical_features_indices else None # Passa índices se houver features categóricas
             },
            'search_spaces': catboost_search_spaces, 'param_grid': None, 'needs_scaling': False,
            #'fit_params': {'classifier__early_stopping_rounds': 100}
        }
else: 
    logger.warning("Usando GridSearchCV como fallback. Defina 'param_grid' em MODEL_CONFIG se necessário.")
   
# Adiciona GaussianNB (que sempre usa GridSearchCV ou nenhum param)
MODEL_CONFIG['GaussianNB'] = {
    'model_kwargs': {},
    'search_spaces': None, # GNB não usa BayesSearch
    'param_grid': gnb_param_grid, # Usa GridSearchCV com este grid
    'needs_scaling': False
}

# --- Em config.py ---
HEURISTIC_FILTER_RULES = {
    'CV_HDA': {'min': 0.3, 'max': 0.4},
    'Odd_H_FT': {'min': 1.01, 'max': 2.01},
    'Odd_D_FT': {'min': 3.0, 'max': None},
    'Odd_Over25_FT': {'min': 1.01, 'max': 2.01},
    'Odd_BTTS_Yes': {'min': 1.45, 'max': None}, # Ajuste o nome da coluna se necessário
    'Media_CG_H': {'min': None, 'max': 0.31}
}
HEURISTIC_FILTER_MODEL_NAME = "Filtro Heurístico A"

# Coluna alvo
TARGET_COLUMN = 'IsDraw' # Certifique-se que é esta
CLASS_NAMES = ['Nao_Empate', 'Empate']
CALIBRATION_METHOD_DEFAULT = 'sigmoid'

STATS_ROLLING_CONFIG = [
    # VG (Value Goals)
    {'base_col_h': 'VG_H_raw', 'base_col_a': 'VG_A_raw', 'output_prefix': 'Media_VG',
     'agg_func': np.nanmean, 'window': ROLLING_WINDOW},
     
    # CG (Cost Goals)
    {'base_col_h': 'CG_H_raw', 'base_col_a': 'CG_A_raw', 'output_prefix': 'Media_CG',
     'agg_func': np.nanmean, 'window': ROLLING_WINDOW},
    {'base_col_h': 'CG_H_raw', 'base_col_a': 'CG_A_raw', 'output_prefix': 'Std_CG',
     'agg_func': np.nanstd, 'min_periods': 2, 'window': ROLLING_WINDOW},
     
    # Chutes Totais
    {'base_col_h': 'Shots_H', 'base_col_a': 'Shots_A', 'output_prefix': 'Media_ChutesTotal',
     'agg_func': np.nanmean, 'window': ROLLING_WINDOW},
     
    # Chutes ao Alvo
    {'base_col_h': 'ShotsOnTarget_H', 'base_col_a': 'ShotsOnTarget_A', 'output_prefix': 'Media_ChutesAlvo',
     'agg_func': np.nanmean, 'window': ROLLING_WINDOW},
     
    # Escanteios
    {'base_col_h': 'Corners_H_FT', 'base_col_a': 'Corners_A_FT', 'output_prefix': 'Media_Escanteios',
     'agg_func': np.nanmean, 'window': 10}, # Janela específica
     
    # Exemplo de Ptos
    {'base_col_h': 'Ptos_H', 'base_col_a': 'Ptos_A', 'output_prefix': 'Media_Ptos',
     'agg_func': np.nanmean, 'window': ROLLING_WINDOW},
]

# --- Configurações para Estatísticas EWMA (Média Móvel Exponencial Ponderada) ---
STATS_EWMA_CONFIG = [
    # VG
    {'base_col_h': 'VG_H_raw', 'base_col_a': 'VG_A_raw', 'output_prefix': 'EWMA_VG', 'span': EWMA_SPAN_SHORT, 'context': 'all'},
    # CG
    {'base_col_h': 'CG_H_raw', 'base_col_a': 'CG_A_raw', 'output_prefix': 'EWMA_CG', 'span': EWMA_SPAN_SHORT, 'context': 'all'},
    # Gols Marcados
    {'base_col_h': GOALS_COLS.get('home'), 'base_col_a': GOALS_COLS.get('away'), 'stat_type': 'offensive',
     'output_prefix': 'EWMA_GolsMarc', 'span': EWMA_SPAN_LONG, 'context': 'all'},
    # Gols Sofridos
    {'base_col_h': GOALS_COLS.get('away'), 'base_col_a': GOALS_COLS.get('home'), 'stat_type': 'defensive',
     'output_prefix': 'EWMA_GolsSofr', 'span': EWMA_SPAN_LONG, 'context': 'all'},
    # Você pode adicionar EWMA para Shots, Corners, etc., se fizerem sentido
]
# --- Configurações do GitHub ---
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')
GITHUB_REPO_NAME = os.environ.get('GITHUB_REPO_NAME')
GITHUB_PREDICTIONS_PATH = f"data/predictions_{MODEL_TYPE_NAME}"