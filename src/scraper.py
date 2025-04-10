import pandas as pd
import time
import os
import logging
import warnings
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException, ElementNotInteractableException, StaleElementReferenceException
from config import (
    SCRAPER_BASE_URL, SCRAPER_TARGET_DAY, CHROMEDRIVER_PATH,
    SCRAPER_TIMEOUT, SCRAPER_ODDS_TIMEOUT, ODDS_COLS, OTHER_ODDS_OUTPUT_NAMES, # Nomes das odds que queremos
    TARGET_LEAGUES, SCRAPER_SLEEP_BETWEEN_GAMES, SCRAPER_SLEEP_AFTER_NAV
)
from typing import Dict, List, Optional, Any
from tqdm import tqdm
import random

warnings.filterwarnings('ignore')
# Configuração básica do logging para erros do scraper
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - SCRAPER - %(levelname)s - %(message)s')

# --- Lista de User-Agents (Mantida da versão anterior para robustez) ---
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0',
]

# --- Funções Auxiliares (Adaptadas do seu script e do projeto) ---
def _safe_float(text: Optional[str]) -> Optional[float]:
    """Converte texto para float de forma segura."""
    if text is None: return None
    try: return float(text)
    except (ValueError, TypeError): return None

def _initialize_driver(chromedriver_path: Optional[str], headless: bool) -> Optional[webdriver.Chrome]:
    """Inicializa o WebDriver do Chrome com opções anti-detecção."""
    options = webdriver.ChromeOptions()
    user_agent = random.choice(USER_AGENTS)
    # print(f"[Scraper Init] Usando User-Agent: {user_agent}") # Debug User Agent
    options.add_argument(f'user-agent={user_agent}')
    if headless: options.add_argument('--headless'); options.add_argument('--window-size=1920,1080')
    options.add_argument('--no-sandbox'); options.add_argument('--disable-dev-shm-usage'); options.add_argument('--disable-gpu')
    options.add_argument('log-level=3'); options.add_experimental_option('excludeSwitches', ['enable-logging'])
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument('--disable-blink-features=AutomationControlled')

    driver: Optional[webdriver.Chrome] = None
    print("Iniciando WebDriver...")
    try:
        if chromedriver_path and os.path.exists(chromedriver_path):
            service = ChromeService(executable_path=chromedriver_path)
            driver = webdriver.Chrome(service=service, options=options)
            # print(f"  WebDriver iniciado usando: {chromedriver_path}") # Debug Path
        else:
            # print(f"  Aviso: Chromedriver não encontrado. Tentando PATH.") # Debug Path
            driver = webdriver.Chrome(options=options)
            # print("  WebDriver iniciado a partir do PATH.") # Debug Path

        if driver:
            # Tenta aplicar script anti-detecção (pode não ser necessário com undetected_chromedriver)
             try:
                 driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
                     'source': '''Object.defineProperty(navigator, 'webdriver', {get: () => undefined})'''
                 })
             except Exception: pass # Ignora se falhar (ex: CDP não disponível)
        print("WebDriver inicializado.")
        return driver
    except WebDriverException as e:
        logging.error(f"Erro CRÍTICO ao iniciar WebDriver: {e}. Verifique instalação/PATH.")
        return None
    except Exception as e_init:
         logging.error(f"Erro inesperado na inicialização do WebDriver: {e_init}")
         return None

def _close_cookies(driver, timeout=10):
    """Tenta fechar o banner de cookies."""
    try:
        cookie_button_selector = 'button#onetrust-accept-btn-handler'
        button_cookies = WebDriverWait(driver, timeout).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, cookie_button_selector))
        )
        button_cookies.click()
        # print("  Banner de cookies fechado.") # Debug
        time.sleep(0.5)
    except Exception:
        # print("  Aviso: Não foi possível fechar cookies (normal se já fechado).") # Debug
        pass # Ignora se não encontrar ou der erro

def _select_target_day(driver, target_day, timeout):
    """Clica no botão do dia alvo (today/tomorrow)."""
    day_selector = f'button.calendar__navigation--{target_day}'
    print(f"Selecionando dia '{target_day}'...")
    try:
        day_button = WebDriverWait(driver, timeout).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, day_selector))
        )
        # Tenta clicar com JS como fallback
        try:
             day_button.click()
        except ElementNotInteractableException:
             driver.execute_script("arguments[0].click();", day_button)
        print(f"Dia '{target_day}' selecionado. Aguardando carregamento...")
        time.sleep(SCRAPER_SLEEP_AFTER_NAV)
        return True
    except Exception as e:
        logging.error(f"Erro CRÍTICO ao clicar no botão do dia '{target_day}': {e}")
        return False

def _get_match_ids(driver, timeout):
    """Extrai os IDs dos jogos visíveis na página."""
    match_ids = []
    match_selector = 'div.event__match[id^="g_1_"]' # Seletor mais específico para jogos
    print("Procurando IDs dos jogos...")
    try:
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, match_selector))
        )
        match_elements = driver.find_elements(By.CSS_SELECTOR, match_selector)
        for element in match_elements:
            match_id = element.get_attribute("id")
            if match_id and match_id.startswith("g_1_") and len(match_id) > 4:
                 match_ids.append(match_id[4:])
        print(f"{len(match_ids)} IDs de jogos encontrados.")
        return match_ids
    except TimeoutException:
        print("Nenhum jogo encontrado na página ou timeout.")
        return []
    except Exception as e:
        logging.error(f"Erro ao extrair IDs dos jogos: {e}")
        return []

# --- Funções Adaptadas do Seu Script para Coleta de Dados por Jogo ---

def get_basic_info(driver, id_jogo) -> Optional[Dict[str, Any]]:
    """Captura informações básicas: data, hora, país, liga, times."""
    info = {'Id': id_jogo, 'Date': None, 'Time': None, 'Country': None, 'League': None, 'HomeTeam': None, 'AwayTeam': None}
    try:
        summary_url = f"{SCRAPER_BASE_URL}/match/{id_jogo}/#/match-summary"
        driver.get(summary_url)
        # Espera um elemento chave carregar para evitar erros com página vazia
        WebDriverWait(driver, SCRAPER_TIMEOUT).until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, 'div.duelParticipant__startTime'))
        )
        # Adiciona uma pequena pausa extra para renderização completa
        time.sleep(0.5)

        # Extrai os dados com seletores atualizados (VERIFICAR!)
        datetime_str = driver.find_element(By.CSS_SELECTOR,'div.duelParticipant__startTime').text
        info['Date'] = datetime_str.split(' ')[0].replace('.', '/') # Formata data
        info['Time'] = datetime_str.split(' ')[1]

        country_el = driver.find_element(By.CSS_SELECTOR, 'span.tournamentHeader__country')
        info['Country'] = country_el.text.split(':')[0].strip()

        league_el = driver.find_element(By.CSS_SELECTOR, 'span.tournamentHeader__country > a')
        info['League'] = league_el.text # Pega nome completo da liga
        # Constrói nome completo para filtro (fora desta função)
        # info['FullLeagueName'] = f"{info['Country']}: {info['League']}"

        # Extrai times (usando nomes consistentes HomeTeam/AwayTeam)
        home_el = driver.find_element(By.CSS_SELECTOR,'div.duelParticipant__home div.participant__participantName')
        info['HomeTeam'] = home_el.text
        away_el = driver.find_element(By.CSS_SELECTOR,'div.duelParticipant__away div.participant__participantName')
        info['AwayTeam'] = away_el.text

        # Validação mínima
        if not info['HomeTeam'] or not info['AwayTeam'] or not info['League']:
            logging.warning(f"Informações básicas incompletas para Jogo ID: {id_jogo}")
            return None # Retorna None se dados essenciais faltarem

        return info

    except (TimeoutException, NoSuchElementException) as e_basic:
        logging.error(f"Erro (Timeout/Não Encontrado) ao buscar info básica Jogo ID {id_jogo}: {e_basic}")
        # Tenta pegar o título e URL para debug em caso de erro
        try: current_title = driver.title; current_url = driver.current_url
        except: current_title = "N/A"; current_url = "N/A"
        logging.error(f"  --> Título: '{current_title}', URL: '{current_url}'")
        return None
    except Exception as e_basic_other:
        logging.error(f"Erro inesperado em get_basic_info Jogo ID {id_jogo}: {e_basic_other}")
        return None


def get_odds_1x2_ft(driver, id_jogo) -> Dict[str, Optional[float]]:
    """Captura odds 1x2 FT."""
    odds = {ODDS_COLS['home']: None, ODDS_COLS['draw']: None, ODDS_COLS['away']: None}
    try:
        url = f"{SCRAPER_BASE_URL}/match/{id_jogo}/#/odds-comparison/1x2-odds/full-time"
        driver.get(url)
        time.sleep(SCRAPER_SLEEP_AFTER_NAV) # Pausa crucial
        # Espera pela primeira linha da tabela de odds
        WebDriverWait(driver, SCRAPER_ODDS_TIMEOUT).until(EC.visibility_of_element_located((By.CSS_SELECTOR, 'div.ui-table__row')))
        linhas = driver.find_elements(By.CSS_SELECTOR,'div.ui-table__row')
        if linhas:
            # Tenta pegar odds da primeira linha (geralmente média ou principal bookie)
            odds_cells = linhas[0].find_elements(By.CSS_SELECTOR, 'a.oddsCell__odd, span.oddsCell__odd') # Tenta 'a' ou 'span'
            if len(odds_cells) >= 3:
                odds[ODDS_COLS['home']] = _safe_float(odds_cells[0].text)
                odds[ODDS_COLS['draw']] = _safe_float(odds_cells[1].text)
                odds[ODDS_COLS['away']] = _safe_float(odds_cells[2].text)
            else: logging.warning(f"Não encontrou 3 células de odds 1x2 na primeira linha. Jogo ID: {id_jogo}")
        else: logging.warning(f"Nenhuma linha de odds 1x2 encontrada. Jogo ID: {id_jogo}")
    except (TimeoutException, NoSuchElementException):
        logging.warning(f"Timeout ou elemento não encontrado para odds 1x2. Jogo ID: {id_jogo}")
    except Exception as e:
        logging.error(f"Erro inesperado ao buscar odds 1x2 Jogo ID {id_jogo}: {e}")
    return odds


def get_odds_ou25_ft(driver, id_jogo) -> Dict[str, Optional[float]]:
    """Captura odds Over/Under 2.5 FT."""
    odds = {'Odd_Over25': None, 'Odd_Under25': None}
    try:
        url = f"{SCRAPER_BASE_URL}/match/{id_jogo}/#/odds-comparison/over-under/full-time"
        driver.get(url)
        time.sleep(SCRAPER_SLEEP_AFTER_NAV)
        # Espera por uma linha que contenha o span com '2.5'
        xpath_selector = "//div[contains(@class, 'ui-table__row')]//span[contains(@class, 'oddsCell__noOddsCell') and normalize-space(.)='2.5']"
        WebDriverWait(driver, SCRAPER_ODDS_TIMEOUT).until(EC.visibility_of_element_located((By.XPATH, xpath_selector)))
        # Encontra a linha específica do 2.5
        linha_25 = driver.find_element(By.XPATH, xpath_selector).find_element(By.XPATH, "./ancestor::div[contains(@class, 'ui-table__row')]")

        odds_cells = linha_25.find_elements(By.CSS_SELECTOR, 'a.oddsCell__odd, span.oddsCell__odd') # Tenta 'a' ou 'span'
        if len(odds_cells) >= 2:
            odds['Odd_Over25'] = _safe_float(odds_cells[0].text)
            odds['Odd_Under25'] = _safe_float(odds_cells[1].text)
        else: logging.warning(f"Não encontrou 2 células de odds O/U 2.5. Jogo ID: {id_jogo}")

    except (TimeoutException, NoSuchElementException):
        logging.warning(f"Timeout ou elemento não encontrado para odds O/U 2.5. Jogo ID: {id_jogo}")
    except Exception as e:
        logging.error(f"Erro inesperado ao buscar odds O/U 2.5 Jogo ID {id_jogo}: {e}")
    return odds


def get_odds_btts_ft(driver, id_jogo) -> Dict[str, Optional[float]]:
    """Captura odds BTTS Yes/No FT."""
    odds = {'Odd_BTTS_Yes': None, 'Odd_BTTS_No': None}
    try:
        url = f"{SCRAPER_BASE_URL}/match/{id_jogo}/#/odds-comparison/both-teams-to-score/full-time"
        driver.get(url)
        time.sleep(SCRAPER_SLEEP_AFTER_NAV)
        WebDriverWait(driver, SCRAPER_ODDS_TIMEOUT).until(EC.visibility_of_element_located((By.CSS_SELECTOR, 'div.ui-table__row')))
        linhas = driver.find_elements(By.CSS_SELECTOR,'div.ui-table__row')
        if linhas:
            odds_cells = linhas[0].find_elements(By.CSS_SELECTOR, 'a.oddsCell__odd, span.oddsCell__odd') # Tenta 'a' ou 'span'
            if len(odds_cells) >= 2:
                odds['Odd_BTTS_Yes'] = _safe_float(odds_cells[0].text)
                odds['Odd_BTTS_No'] = _safe_float(odds_cells[1].text)
            else: logging.warning(f"Não encontrou 2 células de odds BTTS. Jogo ID: {id_jogo}")
        else: logging.warning(f"Nenhuma linha de odds BTTS encontrada. Jogo ID: {id_jogo}")
    except (TimeoutException, NoSuchElementException):
        logging.warning(f"Timeout ou elemento não encontrado para odds BTTS. Jogo ID: {id_jogo}")
    except Exception as e:
        logging.error(f"Erro inesperado ao buscar odds BTTS Jogo ID {id_jogo}: {e}")
    return odds

# --- Função Principal de Scraping (Integrada) ---
def scrape_upcoming_fixtures(chromedriver_path: Optional[str] = CHROMEDRIVER_PATH, headless: bool = True) -> Optional[pd.DataFrame]:
    """
    Coleta jogos do dia alvo, filtra por ligas, raspa odds (1x2, O/U 2.5, BTTS)
    e retorna um DataFrame pandas.
    """
    driver = _initialize_driver(chromedriver_path, headless)
    if not driver: return None

    all_scraped_data = []

    try:
        driver.get(SCRAPER_BASE_URL)
        driver.maximize_window()
        _close_cookies(driver)

        if not _select_target_day(driver, SCRAPER_TARGET_DAY, SCRAPER_TIMEOUT):
            return None

        match_ids = _get_match_ids(driver, SCRAPER_TIMEOUT)
        if not match_ids:
            print(f"Nenhum jogo encontrado para {SCRAPER_TARGET_DAY}.")
            return pd.DataFrame()

        print(f"\nIniciando scraping detalhado para {len(match_ids)} jogos encontrados...")

        # Loop principal de coleta
        for match_id in tqdm(match_ids, total=len(match_ids), desc=f"Scraping {SCRAPER_TARGET_DAY}"):
            # 1. Obter Informações Básicas e Filtrar Liga
            basic_info = get_basic_info(driver, match_id)

            if not basic_info:
                time.sleep(0.5) # Pequena pausa antes do próximo ID
                continue # Pula se não conseguiu info básica

            # Constrói nome completo da liga para filtro
            full_league_name = f"{basic_info.get('Country', '')}: {basic_info.get('League', '')}"

            # Aplica o filtro de ligas
            if TARGET_LEAGUES and full_league_name not in TARGET_LEAGUES:
                # print(f"  Jogo ID {match_id} ({full_league_name}) fora do alvo. Pulando.") # Debug Filtro
                continue # Pula se a liga não está na lista alvo

            # Se passou pelo filtro, coleta as odds
            # print(f"  Jogo ID {match_id} ({full_league_name}) DENTRO do alvo. Coletando odds...") # Debug Filtro

            # 2. Coletar Odds (1x2, O/U 2.5, BTTS) - FT apenas
            odds_1x2 = get_odds_1x2_ft(driver, match_id)
            odds_ou25 = get_odds_ou25_ft(driver, match_id)
            odds_btts = get_odds_btts_ft(driver, match_id)

            # Combina todas as informações em um dicionário
            final_jogo_data = basic_info.copy() # Começa com info básica
            final_jogo_data.update(odds_1x2)
            final_jogo_data.update(odds_ou25)
            final_jogo_data.update(odds_btts)

            all_scraped_data.append(final_jogo_data)

            # Pausa entre jogos
            time.sleep(SCRAPER_SLEEP_BETWEEN_GAMES)

        # --- Fim do Loop ---

        print(f"\nScraping concluído. {len(all_scraped_data)} jogos das ligas alvo coletados.")
        if not all_scraped_data: return pd.DataFrame()

        df_fixtures = pd.DataFrame(all_scraped_data)

        # Log de odds ausentes
        missing_1x2 = df_fixtures[ODDS_COLS['home']].isnull().sum()
        missing_ou = df_fixtures['Odd_Over25'].isnull().sum()
        missing_btts = df_fixtures['Odd_BTTS_Yes'].isnull().sum()
        print(f"  Resumo de Odds Ausentes no resultado final: 1x2 FT={missing_1x2}, O/U 2.5 FT={missing_ou}, BTTS FT={missing_btts}")

        # Garante que as colunas de odds extras definidas no config existam, mesmo que vazias
        expected_cols = list(ODDS_COLS.values()) + OTHER_ODDS_OUTPUT_NAMES
        for col in expected_cols:
            if col not in df_fixtures.columns:
                 df_fixtures[col] = None # Adiciona a coluna com None/NaN

        return df_fixtures

    except Exception as e_global:
        logging.error(f"Erro GERAL INESPERADO durante scraping: {e_global}", exc_info=True)
        return None
    finally:
        if driver:
            driver.quit()
            print("WebDriver fechado.")

# --- Bloco para testar o scraper isoladamente ---
if __name__ == '__main__':
    print("--- Testando Scraper V6 (Baseado no seu script + Filtro) ---")
    # Teste visível é melhor para verificar seletores
    df_test = scrape_upcoming_fixtures(headless=False)

    if df_test is not None:
        print("\n--- DataFrame Coletado e Filtrado (Amostra) ---")
        # Mostra mais colunas
        pd.set_option('display.max_rows', 50)
        pd.set_option('display.max_columns', 20)
        pd.set_option('display.width', 150)
        print(df_test.head(10))
        print(f"\nTotal de jogos filtrados: {len(df_test)}")
        if not df_test.empty:
             print("\nLigas presentes no resultado:")
             print(df_test['League'].value_counts())
             print("\nInfo das colunas de Odds:")
             odds_cols_to_check = list(ODDS_COLS.values()) + OTHER_ODDS_OUTPUT_NAMES
             # Filtra para mostrar apenas colunas que existem no DF
             cols_exist = [c for c in odds_cols_to_check if c in df_test.columns]
             print(df_test[cols_exist].info())
             print(df_test[cols_exist].head())
        else:
             print("Nenhum jogo passou pelo filtro ou foi coletado.")
    else:
        print("\n--- Falha GERAL ao coletar dados ---")
    print("--- Fim do Teste do Scraper V6 ---")