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
    SCRAPER_TIMEOUT, SCRAPER_ODDS_TIMEOUT, ODDS_COLS, OTHER_ODDS_NAMES, # Nomes das odds que queremos
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

logger = logging.getLogger('Scraper') # Usar logger específico é melhor
logger.setLevel(logging.INFO) # Ajustar nível conforme necessidade
# Adicionar handlers se ainda não configurado globalmente...

def get_basic_info(driver, id_jogo) -> Optional[Dict[str, Any]]:
    """Captura informações básicas: data, hora, país, liga, times."""
    info = {'Id': id_jogo, 'Date': None, 'Time': None, 'Country': None, 'League': None, 'HomeTeam': None, 'AwayTeam': None}
    logger.info(f"  Buscando info básica para Jogo ID: {id_jogo}")
    try:
        summary_url = f"{SCRAPER_BASE_URL}/match/{id_jogo}/#/match-summary/match-summary"
        logger.info(f"    Navegando para: {summary_url}")
        driver.get(summary_url)
        wait = WebDriverWait(driver, SCRAPER_TIMEOUT) # Usar timeout do config

        # 1. Esperar por elemento chave (ex: container do cabeçalho)
        header_container_selector = "div.duelParticipant" # AJUSTE SE NECESSÁRIO
        logger.info(f"    Aguardando container principal: '{header_container_selector}'")
        # --- CORRIGIDO: Passa tupla para EC ---
        wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, header_container_selector)))
        # ------------------------------------
        logger.info(f"    Container principal encontrado.")
        time.sleep(0.5) # Pausa extra

        # 2. Extrair Data e Hora
        try:
            datetime_selector = 'div.duelParticipant__startTime'
            logger.info(f"    Buscando data/hora: '{datetime_selector}'")
            # --- CORRIGIDO: Passa tupla ---
            datetime_el = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, datetime_selector)))
            # ------------------------------
            datetime_str = datetime_el.text
            parts = datetime_str.split(' ')
            if len(parts) >= 2:
                 info['Date'] = parts[0].replace('.', '/')
                 info['Time'] = parts[1]
                 logger.info(f"      Data/Hora: {info['Date']} {info['Time']}")
            else: logger.warning(f"Formato data/hora inesperado: '{datetime_str}' ID: {id_jogo}")
        except (TimeoutException, NoSuchElementException): logger.warning(f"Data/hora não encontrada ID: {id_jogo}")
        except Exception as e: logger.error(f"Erro extrair data/hora ID {id_jogo}: {e}")

        # 3. Extrair País e Liga
        breadcrumb_list_selector = "ol.wcl-breadcrumbList_m5Npe" # CONFIRME ESTE SELETOR
        try:
            logger.info(f"    Buscando breadcrumbs: '{breadcrumb_list_selector}'")
            # --- CORRIGIDO: Passa tupla ---
            breadcrumb_list = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, breadcrumb_list_selector)))
            # ------------------------------
            logger.info("      Lista de breadcrumbs encontrada.")

            # Tenta País (2º item)
            try:
                 pais_selector_relative = "li[itemprop='itemListElement']:nth-child(2) span[itemprop='name']" # Seletor relativo à lista
                 pais_el = breadcrumb_list.find_element(By.CSS_SELECTOR, pais_selector_relative)
                 info['Country'] = pais_el.text.strip()
                 logger.info(f"        País: {info['Country']}")
            except NoSuchElementException: logger.warning(f"País (2º breadcrumb) não encontrado ID: {id_jogo}")

            # Tenta Liga (3º item)
            try:
                 liga_selector_relative = "li[itemprop='itemListElement']:nth-child(3) span[itemprop='name']" # Seletor relativo à lista
                 liga_el = breadcrumb_list.find_element(By.CSS_SELECTOR, liga_selector_relative)
                 info['League'] = liga_el.text.strip()
                 logger.info(f"        Liga: {info['League']}")
            except NoSuchElementException: logger.warning(f"Liga (3º breadcrumb) não encontrada ID: {id_jogo}")

        except (TimeoutException, NoSuchElementException): logger.warning(f"Lista breadcrumbs não encontrada ID: {id_jogo}")
        except Exception as e: logger.error(f"Erro processar breadcrumbs ID {id_jogo}: {e}")


        # 4. Extrair Times
        try:
            home_selector = 'div.duelParticipant__home div.participant__participantName' # CONFIRME
            logger.info(f"    Buscando Time Casa: '{home_selector}'")
            # --- CORRIGIDO: Passa tupla ---
            home_el = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, home_selector)))
            # ------------------------------
            info['HomeTeam'] = home_el.text
            logger.info(f"      Casa: {info['HomeTeam']}")
        except (TimeoutException, NoSuchElementException): logger.error(f"Time Casa não encontrado ID: {id_jogo}")
        except Exception as e: logger.error(f"Erro extrair time casa ID {id_jogo}: {e}")

        try:
            away_selector = 'div.duelParticipant__away div.participant__participantName' # CONFIRME
            logger.info(f"    Buscando Time Fora: '{away_selector}'")
            # --- CORRIGIDO: Passa tupla ---
            away_el = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, away_selector)))
            # ------------------------------
            info['AwayTeam'] = away_el.text
            logger.info(f"      Fora: {info['AwayTeam']}")
        except (TimeoutException, NoSuchElementException): logger.error(f"Time Fora não encontrado ID: {id_jogo}")
        except Exception as e: logger.error(f"Erro extrair time fora ID {id_jogo}: {e}")


        # Validação Mínima Essencial
        if not info.get('HomeTeam') or not info.get('AwayTeam') or not info.get('League') or not info.get('Date'):
            logger.error(f"Falha Crítica: Info básica ausente ID: {id_jogo}. Pulando jogo. Info coletada: {info}")
            return None

        logger.info(f"  -> Info básica OK para Jogo ID: {id_jogo}")
        return info

    # Tratamento de erro geral movido para o final para capturar qualquer exceção
    except Exception as e_general:
        logger.error(f"Erro inesperado GERAL em get_basic_info Jogo ID {id_jogo}: {e_general}", exc_info=True)
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
    """Captura odds Over/Under 2.5 FT usando data-testid."""
    odds = {'Odd_Over25_FT': None, 'Odd_Under25_FT': None}
    logger.info(f"  Buscando odds O/U 2.5 para Jogo ID: {id_jogo}")
    try:
        url = f"{SCRAPER_BASE_URL}/match/{id_jogo}/#/odds-comparison/over-under/full-time"
        logger.info(f"    Navegando para: {url}")
        driver.get(url)
        wait = WebDriverWait(driver, SCRAPER_ODDS_TIMEOUT)

        # 1. Esperar e encontrar a LINHA que contém o span específico do 2.5
        xpath_linha_25 = "//span[@data-testid='wcl-oddsValue' and normalize-space(.)='2.5']/ancestor::div[contains(@class, 'ui-table__row')]"
        logger.info(f"    Aguardando linha O/U 2.5 com XPath: {xpath_linha_25}")

        # Espera que a linha se torne visível
        linha_25_encontrada = wait.until(EC.visibility_of_element_located((By.XPATH, xpath_linha_25)))
        logger.info("    Linha O/U 2.5 encontrada.")
        time.sleep(0.2) # Pequena pausa opcional

        # 2. Extrair Odds de dentro da linha encontrada
        if linha_25_encontrada:
            try:
                # Busca os elementos de odd DENTRO da linha encontrada
                odds_cells = linha_25_encontrada.find_elements(By.CSS_SELECTOR, 'a.oddsCell__odd, span.oddsCell__odd') # Mantém este seletor SE ele estiver correto para as odds
                logger.info(f"      Encontradas {len(odds_cells)} células de odds na linha 2.5.")
                if len(odds_cells) >= 2:
                    # A primeira geralmente é OVER, a segunda UNDER
                    odds['Odd_Over25_FT'] = _safe_float(odds_cells[0].text)
                    odds['Odd_Under25_FT'] = _safe_float(odds_cells[1].text)
                    logger.info(f"      Odds O/U 2.5 extraídas: O={odds['Odd_Over25_FT']}, U={odds['Odd_Under25_FT']}")
                else:
                    logging.warning(f"Não encontrou 2+ células de odds ('a.oddsCell__odd' ou 'span.oddsCell__odd') na linha 2.5. Verifique o seletor de odds. Jogo ID: {id_jogo}")
            except Exception as e_extract:
                 logging.error(f"Erro ao extrair odds da linha 2.5 encontrada. Jogo ID {id_jogo}: {e_extract}")
        # else: O wait.until() já daria TimeoutException se não encontrasse

    except TimeoutException:
        logging.warning(f"Timeout ao esperar/encontrar linha O/U 2.5. Jogo ID: {id_jogo}")
    except Exception as e:
        logging.error(f"Erro inesperado ao buscar odds O/U 2.5 Jogo ID {id_jogo}: {e}", exc_info=True)

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
        missing_ou = df_fixtures['Odd_Over25_FT'].isnull().sum()
        missing_btts = df_fixtures['Odd_BTTS_Yes'].isnull().sum()
        print(f"  Resumo de Odds Ausentes no resultado final: 1x2 FT={missing_1x2}, O/U 2.5 FT={missing_ou}, BTTS FT={missing_btts}")

        # Garante que as colunas de odds extras definidas no config existam, mesmo que vazias
        expected_cols = list(ODDS_COLS.values()) + OTHER_ODDS_NAMES
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
             odds_cols_to_check = list(ODDS_COLS.values()) + OTHER_ODDS_NAMES
             # Filtra para mostrar apenas colunas que existem no DF
             cols_exist = [c for c in odds_cols_to_check if c in df_test.columns]
             print(df_test[cols_exist].info())
             print(df_test[cols_exist].head())
        else:
             print("Nenhum jogo passou pelo filtro ou foi coletado.")
    else:
        print("\n--- Falha GERAL ao coletar dados ---")
    print("--- Fim do Teste do Scraper V1 ---")