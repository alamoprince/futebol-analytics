import pandas as pd
import time
import os
import warnings
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException, ElementNotInteractableException, StaleElementReferenceException
from config import (
    SCRAPER_BASE_URL, SCRAPER_TARGET_DAY, CHROMEDRIVER_PATH,
    SCRAPER_TIMEOUT, SCRAPER_ODDS_TIMEOUT, ODDS_COLS, OTHER_ODDS_NAMES, 
    SCRAPER_TO_INTERNAL_LEAGUE_MAP, SCRAPER_SLEEP_BETWEEN_GAMES, SCRAPER_SLEEP_AFTER_NAV, TARGET_LEAGUES_INTERNAL_IDS, 
    SCRAPER_FILTER_LEAGUES
)
from typing import Dict, List, Optional, Any, Callable
from tqdm import tqdm
import random

from logger_config import setup_logger
logger = setup_logger("ScrapperPredictor")
warnings.filterwarnings('ignore')

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0',
]

def _safe_float(text: Optional[str]) -> Optional[float]:
    """Converte texto para float de forma segura."""
    if text is None: return None
    try: return float(text)
    except (ValueError, TypeError): return None

def _initialize_driver(chromedriver_path: Optional[str], headless: bool) -> Optional[webdriver.Chrome]:
    """Inicializa o WebDriver do Chrome com opções anti-detecção."""
    options = webdriver.ChromeOptions()
    user_agent = random.choice(USER_AGENTS)
    options.add_argument(f'user-agent={user_agent}')
    if headless: 
        options.add_argument('--headless'); 
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--no-sandbox'); 
        options.add_argument('--disable-dev-shm-usage'); 
        options.add_argument('--disable-gpu')
        options.add_argument('log-level=3'); 
        options.add_experimental_option('excludeSwitches', ['enable-logger'])
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        options.add_argument('--disable-blink-features=AutomationControlled')

    driver: Optional[webdriver.Chrome] = None
    logger.info("Iniciando WebDriver...")
    try:
        if chromedriver_path and os.path.exists(chromedriver_path):
            service = ChromeService(executable_path=chromedriver_path)
            driver = webdriver.Chrome(service=service, options=options)
        else:
            driver = webdriver.Chrome(options=options)

        if driver:
             try:
                driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
                    'source': '''Object.defineProperty(navigator, 'webdriver', {get: () => undefined})'''
                })
             except Exception: pass 
        logger.info("WebDriver inicializado.")
        return driver
    except WebDriverException as e:
        logger.error(f"Erro CRÍTICO ao iniciar WebDriver: {e}. Verifique instalação/PATH.")
        return None
    except Exception as e_init:
        logger.error(f"Erro inesperado na inicialização do WebDriver: {e_init}")
        return None

def _close_cookies(driver, timeout=10):
    """Tenta fechar o banner de cookies."""
    try:
        cookie_button_selector = 'button#onetrust-accept-btn-handler'
        button_cookies = WebDriverWait(driver, timeout).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, cookie_button_selector))
        )
        button_cookies.click()
        time.sleep(0.5)
    except Exception:
        pass 

def _select_target_day(driver, target_day, timeout):
    """Clica no botão do dia alvo (today/tomorrow)."""
    if target_day == "today":
        logger.info("Dia 'today' já selecionado por padrão.")
        return True  
    elif target_day == "tomorrow":
        day_selector = 'button[data-day-picker-arrow="next"]'
        num_clicks = 1
        logger.info(f"Selecionando dia '{target_day}' clicando na seta 'próximo dia' {num_clicks} vez(es)...")
    else:
        logger.error(f"Dia alvo '{target_day}' não suportado. Apenas 'today' e 'tomorrow' são válidos.")
        return False

    try:
        for i in range(num_clicks):
            logger.debug(f"  Clique #{i+1} na seta 'próximo dia'.") 
            arrow_button = WebDriverWait(driver, timeout).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, day_selector))
            )
            driver.execute_script("arguments[0].scrollIntoView(true);", arrow_button)
            time.sleep(0.5)
            driver.execute_script("arguments[0].click();", arrow_button)
            if i < num_clicks - 1:
                time.sleep(1) 
        logger.info(f"Dia '{target_day}' selecionado com sucesso. Aguardando carregamento da página...")
        time.sleep(SCRAPER_SLEEP_AFTER_NAV) 
        return True

    except TimeoutException:
        logger.error(f"Erro CRÍTICO: Timeout ao esperar pelo botão de seta '{day_selector}'. O seletor pode ter mudado.")
        return False
    except Exception as e:
        logger.error(f"Erro CRÍTICO ao clicar no botão de seta para '{target_day}': {e}", exc_info=True)
        return False

def _get_match_ids(driver, timeout):
    """Extrai os IDs dos jogos visíveis na página."""
    match_ids = []
    match_selector = 'div.event__match[id^="g_1_"]' 
    logger.info("Procurando IDs dos jogos...")
    try:
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, match_selector))
        )
        match_elements = driver.find_elements(By.CSS_SELECTOR, match_selector)
        for element in match_elements:
            match_id = element.get_attribute("id")
            if match_id and match_id.startswith("g_1_") and len(match_id) > 4:
                 match_ids.append(match_id[4:])
        logger.debug(f"{len(match_ids)} IDs de jogos encontrados.")
        return match_ids
    except TimeoutException:
        logger.error("Nenhum jogo encontrado na página ou timeout.")
        return []
    except Exception as e:
        logger.error(f"Erro ao extrair IDs dos jogos: {e}")
        return []

def get_basic_info(driver, id_jogo, from_summary_page=False) -> Optional[Dict[str, Any]]:
    """Captura informações básicas: data, hora, país, liga, times."""
    info = {'Id': id_jogo, 'Date': None, 'Time': None, 'Country': None, 'League': None, 'HomeTeam': None, 'AwayTeam': None}
    logger.info(f"  Buscando info básica para Jogo ID: {id_jogo}")
    try:
        if not from_summary_page:
            summary_url = f"{SCRAPER_BASE_URL}/match/{id_jogo}/#/match-summary/match-summary"
            logger.info(f"    Navegando para: {summary_url}")
            driver.get(summary_url)
        wait = WebDriverWait(driver, SCRAPER_TIMEOUT)

        header_container_selector = "div.duelParticipant"
        logger.info(f"    Aguardando container principal: '{header_container_selector}'")
        wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, header_container_selector)))
        logger.info(f"    Container principal encontrado.")
        time.sleep(0.5)

        try:
            datetime_selector = 'div.duelParticipant__startTime'
            datetime_el = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, datetime_selector)))
            datetime_str = datetime_el.text
            parts = datetime_str.split(' ')
            if len(parts) >= 2: info['Date'] = parts[0].replace('.', '/'); info['Time'] = parts[1]
            else: logger.warning(f"Formato data/hora inesperado: '{datetime_str}' ID: {id_jogo}")
        except Exception as e: logger.error(f"Erro extrair data/hora ID {id_jogo}: {e}")

        breadcrumb_list_selector = "ol.wcl-breadcrumbList_m5Npe"
        try:
            logger.info(f"    Buscando breadcrumbs: '{breadcrumb_list_selector}'")
            breadcrumb_list = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, breadcrumb_list_selector)))
            logger.info("      Lista de breadcrumbs encontrada.")

            try:
                 pais_selector_relative = "li[itemprop='itemListElement']:nth-child(2) span[itemprop='name']"
                 pais_el = breadcrumb_list.find_element(By.CSS_SELECTOR, pais_selector_relative)
                 info['Country'] = pais_el.text.strip().upper() 
                 logger.info(f"        País: {info['Country']}")
            except NoSuchElementException: logger.warning(f"País (2º breadcrumb) não encontrado ID: {id_jogo}")

            try:
                 liga_selector_relative = "li[itemprop='itemListElement']:nth-child(3) span[itemprop='name']"
                 liga_el = breadcrumb_list.find_element(By.CSS_SELECTOR, liga_selector_relative)
                 league_text_raw = liga_el.text.strip()
                 logger.info(f"        Liga (Texto Bruto): {league_text_raw}")

                 league_text_cleaned = re.sub(r'\s*-\s*(ROUND|GROUP|PLAYOFFS?)\b.*', '', league_text_raw, flags=re.IGNORECASE).strip()

                 info['League'] = league_text_cleaned 
                 logger.info(f"        Liga (Limpo): {info['League']}")
            except NoSuchElementException: logger.warning(f"Liga (3º breadcrumb) não encontrada ID: {id_jogo}")
            except Exception as e_league: logger.error(f"Erro ao limpar nome da liga ID {id_jogo}: {e_league}")

        except Exception as e: logger.error(f"Erro processar breadcrumbs ID {id_jogo}: {e}")

        try:
            home_selector = 'div.duelParticipant__home div.participant__participantName'
            home_el = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, home_selector)))
            info['HomeTeam'] = home_el.text
        except Exception as e: logger.error(f"Time Casa não encontrado ID: {id_jogo}: {e}")

        try:
            away_selector = 'div.duelParticipant__away div.participant__participantName'
            away_el = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, away_selector)))
            info['AwayTeam'] = away_el.text
        except Exception as e: logger.error(f"Time Fora não encontrado ID: {id_jogo}: {e}")

        if not info.get('HomeTeam') or not info.get('AwayTeam') or not info.get('Country') or not info.get('League') or not info.get('Date'):
            logger.error(f"Falha Crítica: Info básica ausente/inválida ID: {id_jogo}. Pulando jogo. Info coletada: {info}")
            return None

        info['ScraperLeagueName'] = f"{info.get('Country', '')}: {info.get('League', '')}"
        logger.info(f"    Nome Completo Scraper para Map: {info['ScraperLeagueName']}")

        logger.info(f"  -> Info básica OK para Jogo ID: {id_jogo}")
        return info

    except Exception as e_general:
        logger.error(f"Erro inesperado GERAL em get_basic_info Jogo ID {id_jogo}: {e_general}", exc_info=True)
        return None

def get_all_odds_for_match(driver, id_jogo) -> Dict[str, Optional[float]]:
    """
    Coleta todos os mercados de odds necessários navegando diretamente para a URL de cada mercado.
    Esta abordagem é mais robusta contra erros de sincronização de JavaScript.
    """
    all_odds = {
        ODDS_COLS['home']: None, ODDS_COLS['draw']: None, ODDS_COLS['away']: None,
        'Odd_Over25_FT': None, 'Odd_Under25_FT': None,
        'Odd_BTTS_Yes': None, 'Odd_BTTS_No': None
    }
    logger.info(f"  Buscando TODAS as odds para Jogo ID: {id_jogo} (Método Direto)")
    wait = WebDriverWait(driver, SCRAPER_ODDS_TIMEOUT)

    # 1. Coleta Odds 1x2
    try:
        url_1x2 = f"{SCRAPER_BASE_URL}/match/{id_jogo}/#/odds-comparison/1x2-odds/full-time"
        driver.get(url_1x2)
        row_selector = 'div.ui-table__row'
        first_row = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, row_selector)))
        odds_cells = first_row.find_elements(By.CSS_SELECTOR, 'a.oddsCell__odd, span.oddsCell__odd')
        if len(odds_cells) >= 3:
            all_odds[ODDS_COLS['home']] = _safe_float(odds_cells[0].text)
            all_odds[ODDS_COLS['draw']] = _safe_float(odds_cells[1].text)
            all_odds[ODDS_COLS['away']] = _safe_float(odds_cells[2].text)
    except Exception as e:
        logger.warning(f"Não foi possível coletar odds 1x2 para o jogo {id_jogo}: {type(e).__name__}")

    # 2. Coleta Odds Over/Under 2.5
    try:
        url_ou = f"{SCRAPER_BASE_URL}/match/{id_jogo}/#/odds-comparison/over-under/full-time"
        driver.get(url_ou)
        xpath_linha_25 = "//div[contains(@class, 'ui-table__row')]//span[normalize-space(.)='2.5']"
        linha_25_container = wait.until(EC.visibility_of_element_located((By.XPATH, f"{xpath_linha_25}/ancestor::div[contains(@class, 'ui-table__row')]")))
        odds_cells = linha_25_container.find_elements(By.CSS_SELECTOR, 'a.oddsCell__odd, span.oddsCell__odd')
        if len(odds_cells) >= 2:
            all_odds['Odd_Over25_FT'] = _safe_float(odds_cells[0].text)
            all_odds['Odd_Under25_FT'] = _safe_float(odds_cells[1].text)
    except Exception as e:
        logger.warning(f"Não foi possível coletar odds O/U 2.5 para o jogo {id_jogo}: {type(e).__name__}")

    # 3. Coleta Odds BTTS
    try:
        url_btts = f"{SCRAPER_BASE_URL}/match/{id_jogo}/#/odds-comparison/both-teams-to-score/full-time"
        driver.get(url_btts)
        row_selector = 'div.ui-table__row'
        first_btts_row = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, row_selector)))
        odds_cells = first_btts_row.find_elements(By.CSS_SELECTOR, 'a.oddsCell__odd, span.oddsCell__odd')
        if len(odds_cells) >= 2:
            all_odds['Odd_BTTS_Yes'] = _safe_float(odds_cells[0].text)
            all_odds['Odd_BTTS_No'] = _safe_float(odds_cells[1].text)
    except Exception as e:
        logger.warning(f"Não foi possível coletar odds BTTS para o jogo {id_jogo}: {type(e).__name__}")

    collected_markets = [k for k, v in all_odds.items() if v is not None]
    logger.info(f"  -> Coleta de odds concluída para Jogo ID: {id_jogo}. Mercados obtidos: {len(collected_markets)}/{len(all_odds)}")
    return all_odds

# --- Função Principal de Scraping ---
def scrape_upcoming_fixtures(
    chromedriver_path: Optional[str] = CHROMEDRIVER_PATH,
    headless: bool = True,
    progress_callback: Optional[Callable[[str, Any], None]] = None
) -> Optional[pd.DataFrame]:
    driver = _initialize_driver(chromedriver_path, headless)
    if not driver:
        if progress_callback:
            progress_callback("update", (0, "Erro ao iniciar o WebDriver."))
        return None

    all_scraped_data = []
    try:
        driver.get(SCRAPER_BASE_URL)
        _close_cookies(driver)

        if not _select_target_day(driver, SCRAPER_TARGET_DAY, SCRAPER_TIMEOUT):
            if progress_callback:
                progress_callback("update", (0, f"Falha ao selecionar o dia {SCRAPER_TARGET_DAY}."))
            return None
            
        match_ids = _get_match_ids(driver, SCRAPER_TIMEOUT)
        if not match_ids:
            logger.warning(f"Nenhum jogo encontrado para {SCRAPER_TARGET_DAY}.")
            if progress_callback:
                progress_callback("update", (0, "Nenhum jogo encontrado."))
            return pd.DataFrame()

        total_matches = len(match_ids)
        if progress_callback:
            progress_callback("start", (total_matches, f"Encontrados {total_matches} jogos..."))

        if SCRAPER_FILTER_LEAGUES:
            logger.info(f"\nIniciando scraping detalhado para {total_matches} jogos (FILTRANDO por ligas alvo)...")
        else:
            logger.info(f"\nIniciando scraping detalhado para {total_matches} jogos (SEM FILTRO de ligas)...")

        processed_count = 0
        skipped_count = 0

        for i, match_id in enumerate(match_ids):
            # Atualiza o progresso antes de iniciar a coleta do jogo
            if progress_callback:
                progress_callback("update", (i, f"Coletando jogo {i + 1}/{total_matches}..."))

            summary_url = f"{SCRAPER_BASE_URL}/match/{match_id}/#/match-summary/match-summary"
            driver.get(summary_url)
            basic_info = get_basic_info(driver, id_jogo=match_id, from_summary_page=True) # Passa um flag
            
            if not basic_info:
                logger.warning(f"Jogo ID {match_id}: Falha ao obter info básica. Pulando.")
                skipped_count += 1
                continue

            # 2. Lógica de filtro da liga
            full_league_name_scraped = basic_info['ScraperLeagueName']
            internal_league_id = SCRAPER_TO_INTERNAL_LEAGUE_MAP.get(full_league_name_scraped)
            
            should_process = False
            final_league_name = None

            if SCRAPER_FILTER_LEAGUES:
                if internal_league_id and internal_league_id in TARGET_LEAGUES_INTERNAL_IDS:
                    should_process = True
                    final_league_name = internal_league_id
                else:
                    skipped_count += 1
                    continue
            else:
                should_process = True
                final_league_name = internal_league_id if internal_league_id else full_league_name_scraped
            
            # 3. Se o jogo deve ser processado, coleta as odds
            if should_process:
                processed_count += 1
                basic_info['League'] = final_league_name
                if 'ScraperLeagueName' in basic_info:
                    del basic_info['ScraperLeagueName']

                all_odds = get_all_odds_for_match(driver, match_id)
                
                final_jogo_data = {**basic_info, **all_odds}
                all_scraped_data.append(final_jogo_data)
                
                time.sleep(SCRAPER_SLEEP_BETWEEN_GAMES)

        # Atualização final da barra de progresso
        if progress_callback:
            progress_callback("update", (total_matches, f"Coleta finalizada. {processed_count} jogos processados."))

        logger.info(f"\nScraping concluído.")
        logger.info(f"- Jogos processados (odds coletadas): {processed_count}")
        if SCRAPER_FILTER_LEAGUES:
            logger.debug(f"- Jogos pulados (fora do filtro): {skipped_count}")
        else:
            logger.debug(f"- Jogos pulados (erro info básica): {skipped_count}")
        
        if not all_scraped_data:
            logger.warning("Nenhum dado de jogo válido foi coletado.")
            return pd.DataFrame()

        df_fixtures = pd.DataFrame(all_scraped_data)

        expected_odds_cols = list(ODDS_COLS.values()) + OTHER_ODDS_NAMES
        for col in expected_odds_cols:
            if col not in df_fixtures.columns:
                df_fixtures[col] = None

        df_fixtures.rename(columns={'HomeTeam': 'Home', 'AwayTeam': 'Away'}, inplace=True)
        if 'Time' in df_fixtures.columns and 'Time_Str' not in df_fixtures.columns:
            df_fixtures['Time_Str'] = df_fixtures['Date'] + ' ' + df_fixtures['Time']

        logger.info(f"DataFrame final com {len(df_fixtures)} jogos gerado.")
        return df_fixtures

    except KeyboardInterrupt:
        logger.info("\nScraping interrompido pelo usuário.")
        if all_scraped_data: return pd.DataFrame(all_scraped_data)
        else: return pd.DataFrame()
    except Exception as e_global:
        logger.error(f"Erro GERAL INESPERADO durante scraping: {e_global}", exc_info=True)
        if all_scraped_data: return pd.DataFrame(all_scraped_data)
        else: return None
    finally:
        if driver:
            try:
                driver.quit()
                logger.info("WebDriver fechado.")
            except Exception as e_quit:
                logger.error(f"Erro ao fechar WebDriver: {e_quit}")
            