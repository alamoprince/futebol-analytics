import sys
import os
import pandas as pd
from datetime import date, timedelta

# --- Configurar Path ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(APP_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if APP_DIR not in sys.path: # Adiciona diretório pai também
    sys.path.insert(0, APP_DIR)

# --- Importar Módulos ---
try:
    from src.scraper_predictor import scrape_upcoming_fixtures # Importa a função do scraper
    from src.github_manager import GitHubManager  # Importa a classe do GitHub Manager
    from src.config import (
        SCRAPER_TARGET_DAY, GITHUB_REPO_NAME, GITHUB_PREDICTIONS_PATH, MODEL_TYPE_NAME
    ) # Importa configs relevantes
    from dotenv import load_dotenv # Para carregar GITHUB_TOKEN do .env
except ImportError as e:
    print(f"Erro ao importar módulos: {e}")
    print("Certifique-se de que este script está no diretório correto e que os módulos existem em 'src'.")
    sys.exit(1)

# --- Função Principal ---
def main():
    print("--- Iniciando Coleta e Upload para GitHub ---")

    # 1. Carregar variáveis de ambiente (necessário para GitHubManager)
    print("Carregando variáveis de ambiente (se .env existir)...")
    dotenv_path = os.path.join(APP_DIR, '.env') # Procura .env no mesmo nível do script
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path, override=True)
        print(".env carregado.")
    else:
        print(".env não encontrado, usando variáveis de ambiente existentes.")


    # 2. Executar o Scraper
    print(f"\nExecutando scraper para '{SCRAPER_TARGET_DAY}'...")
    # Executar em modo headless (sem janela visível) para automação
    # Passe o caminho do chromedriver se necessário/definido no config
    from src.config import CHROMEDRIVER_PATH # Importa dentro da função para pegar valor atualizado
    df_scraped_fixtures = scrape_upcoming_fixtures(headless=True, chromedriver_path=CHROMEDRIVER_PATH)

    # 3. Verificar Resultado do Scraper
    if df_scraped_fixtures is None:
        print("\nERRO: Scraper falhou ou não retornou dados.")
        return # Termina se não houver dados

    if df_scraped_fixtures.empty:
        print("\nAVISO: Scraper rodou, mas não encontrou jogos para as ligas alvo.")
        return # Termina se o DataFrame estiver vazio

    print(f"\nSucesso: {len(df_scraped_fixtures)} jogos coletados.")
    print("Amostra:")
    print(df_scraped_fixtures.head())

    # 4. Preparar para Upload no GitHub
    try:
        print("\nInicializando GitHub Manager...")
        gh_manager = GitHubManager() # Inicializa a conexão
    except Exception as e:
        print(f"ERRO: Falha ao inicializar GitHub Manager: {e}")
        return # Termina se não conectar ao GitHub

    # Define o nome do arquivo e o caminho no repositório
    target_day = SCRAPER_TARGET_DAY
    if target_day == "today":
        file_date_str = date.today().strftime("%Y-%m-%d")
    elif target_day == "tomorrow":
        file_date_str = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        file_date_str = target_day # Assume que já está no formato YYYY-MM-DD

    # --- NOME DO ARQUIVO CSV ---
    # Usar um nome que descreva a origem (scraper) e a data
    csv_filename = f"scraped_fixtures_{file_date_str}.csv"
    # --- CAMINHO NO GITHUB ---
    # Usar um subdiretório para dados brutos do scraper
    github_path = f"data/raw_scraped/{csv_filename}" # Exemplo de caminho

    print(f"\nPreparando para salvar em: {GITHUB_REPO_NAME}/{github_path}")

    # 5. Salvar/Atualizar no GitHub
    commit_message = f"Atualiza/Cria jogos raspados para {file_date_str}"
    success = gh_manager.save_or_update_github(
        df=df_scraped_fixtures,
        path=github_path,
        commit_message_prefix=commit_message # Adiciona um prefixo claro
    )

    if success:
        print("\n--- Upload para GitHub concluído com sucesso! ---")
    else:
        print("\n--- Falha no Upload para GitHub. Verifique os logs do GitHubManager. ---")

# --- Executar a função principal ---
if __name__ == "__main__":
    main()