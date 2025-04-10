# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

# --- Ajuste o Caminho Base ---
# Obtém o diretório onde o .spec está localizado (geralmente a raiz do projeto)
basedir = os.path.dirname(__file__)
srcdir = os.path.join(basedir, 'src')

# Adiciona o diretório src ao path para PyInstaller encontrar os módulos
pathex = [basedir, srcdir]

# --- Coleta de Arquivos de Dados ---
# Arquivos não-Python que precisam ser incluídos no executável

# 1. Arquivo de Dados Históricos (Excel)
#    Assume que o nome está definido em config.py, mas é mais seguro definir aqui explicitamente
#    ou copiar o arquivo para um local padrão antes de construir.
#    Vamos usar o nome padrão por enquanto.
historical_data_filename = "Brasileirao_A_e_B (1).xlsx"
datas = [(os.path.join(basedir, 'data', historical_data_filename), 'data')]

# 2. Modelo Salvo (.joblib) - Ponto de Atenção!
#    O nome do arquivo do modelo pode mudar. A forma mais robusta seria
#    copiar o .joblib DESEJADO para um nome fixo (ex: 'model_to_bundle.joblib')
#    na pasta 'data/' ANTES de rodar o PyInstaller, e referenciar esse nome fixo aqui.
#    Ou, usar o nome padrão definido no config (menos flexível se o melhor modelo mudar).
#    Usaremos o nome padrão por enquanto, mas esteja ciente disso.
#    (Importar do config aqui pode ser complexo para o PyInstaller)
model_filename = "best_model_backdraw_best_v3_more_features.joblib" # <<< AJUSTE ESTE NOME se necessário
model_path_src = os.path.join(basedir, 'data', model_filename)
if os.path.exists(model_path_src):
    datas.append((model_path_src, 'data'))
else:
    print(f"AVISO: Arquivo de modelo '{model_filename}' não encontrado em 'data/'. Não será incluído no build.")

# 3. (Opcional) Arquivo .env - NÃO RECOMENDADO para bundling, pois expõe segredos.
#    É melhor que o usuário configure variáveis de ambiente.
# if os.path.exists(os.path.join(basedir, '.env')):
#     datas.append((os.path.join(basedir, '.env'), '.'))

# --- Coleta de Dados de Bibliotecas (Ex: sklearn, pandas) ---
# PyInstaller geralmente lida bem com sklearn e pandas, mas às vezes precisa de ajuda.
# Adiciona dados que podem ser necessários para scikit-learn e pandas
datas += collect_data_files('sklearn')
datas += collect_data_files('pandas')
# datas += collect_data_files('numpy') # Geralmente não precisa explicitamente

# --- Hidden Imports ---
# Módulos que o PyInstaller pode não detectar automaticamente.
hiddenimports = [
    'sklearn.utils._typedefs',
    'sklearn.utils._heap',
    'sklearn.utils._sorting',
    'sklearn.neighbors._typedefs',
    'sklearn.neighbors._quad_tree',
    'sklearn.tree',
    'sklearn.tree._utils',
    'sklearn.ensemble._forest',
    'pandas._libs.tslibs.timedeltas',
    'pandas._libs.tslibs.nattype',
    'pandas._libs.tslibs.np_datetime',
    'pandas._libs.tslibs.offsets',
    'pandas._libs.tslibs.period',
    'pandas._libs.tslibs.timestamps',
    'pandas._libs.tslibs.timezones',
    'scipy._lib.messagestream', # Dependência comum do SciPy/Sklearn
    'joblib.externals.loky', # Para paralelismo do joblib/sklearn
    'dotenv', # Se usar python-dotenv
    'github', # Para PyGithub
    'rich', # Para o dashboard (mesmo que não seja o entry point principal)
    # Adicione 'lightgbm' se você o instalou e está usando
    # 'lightgbm'
]

# --- Análise Principal ---
a = Analysis(
    ['src\\main.py'], # Script principal da GUI
    pathex=pathex,
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False
)

# --- Criação do Arquivo PYZ (Bytecode) ---
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# --- Criação do Executável ---
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='FootballPredictorPro_BackDraw', # Nome do arquivo .exe
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True, # Usa UPX para comprimir (requer UPX instalado e no PATH)
    console=False, # <<< IMPORTANTE: False para aplicação GUI (sem console)
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon='SeuPath/futebol_analytics/Utilitarios/icon.ico' 
)

# --- Coleta de Arquivos (Dependências e Dados) ---
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='FootballPredictorPro_BackDraw' # Nome da PASTA que será criada em dist/
)