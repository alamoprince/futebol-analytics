import logging
import os
from datetime import datetime
import sys

# Tenta obter o diretório base de forma segura
try:
    # Assume que logger_config.py está em src
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(SRC_DIR)
except NameError:
    # Fallback se __file__ não estiver definido (ex: alguns modos interativos)
    BASE_DIR = os.path.abspath(".")
    print("Warning: Could not determine BASE_DIR reliably. Using current directory for logs.")

LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = datetime.now().strftime("futebol_analytics_log_%Y-%m-%d_%H-%M-%S.log") # Mudado para .log
log_filepath = os.path.join(LOG_DIR, log_filename)

# Formato do Log
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Configura o logger raiz (menos recomendado) ou loggers específicos
# Configuração para um logger específico (recomendado)
def setup_logger(logger_name='FutebolAnalyticsApp', level=logging.INFO):
    """Configura e retorna um logger com handlers de arquivo e console."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False # Evita que mensagens subam para o logger raiz se ele também tiver handlers

    # Evita adicionar handlers duplicados
    if not logger.handlers:
        # Handler de Arquivo
        try:
            file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
            file_handler.setLevel(level)
            formatter_file = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
            file_handler.setFormatter(formatter_file)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Erro ao configurar file handler do log: {e}", file=sys.stderr)


        # Handler de Console
        try:
            console_handler = logging.StreamHandler(sys.stdout) # Direciona para stdout
            console_handler.setLevel(level) # Console pode ter nível diferente se desejado
            formatter_console = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
            console_handler.setFormatter(formatter_console)
            logger.addHandler(console_handler)
        except Exception as e:
            print(f"Erro ao configurar console handler do log: {e}", file=sys.stderr)

        print(f"Logging setup for '{logger_name}' completed. Log file: {log_filepath}") # Confirmação

    return logger