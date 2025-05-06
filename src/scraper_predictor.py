import pandas as pd
import time
import os
import logging
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
    SCRAPER_TIMEOUT, SCRAPER_ODDS_TIMEOUT, ODDS_COLS, OTHER_ODDS_NAMES, # Nomes das odds que queremos
    SCRAPER_TO_INTERNAL_LEAGUE_MAP, SCRAPER_SLEEP_BETWEEN_GAMES, SCRAPER_SLEEP_AFTER_NAV, TARGET_LEAGUES_INTERNAL_IDS, 
    SCRAPER_FILTER_LEAGUES
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
        else:
            driver = webdriver.Chrome(options=options)

        if driver:
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
        time.sleep(0.5)
    except Exception:
        pass # Ignora se não encontrar ou der erro

def _select_target_day(driver, target_day, timeout):
    """Clica no botão do dia alvo (today/tomorrow)."""
    # Se for today, não precisa fazer nada pois o site já abre nesse dia
    if target_day == "today":
        print("Dia 'today' já selecionado por padrão.")
        return True
        
    # Para tomorrow, precisa clicar no botão
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
        wait = WebDriverWait(driver, SCRAPER_TIMEOUT)

        header_container_selector = "div.duelParticipant"
        logger.info(f"    Aguardando container principal: '{header_container_selector}'")
        wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, header_container_selector)))
        logger.info(f"    Container principal encontrado.")
        time.sleep(0.5)

        # Extrair Data e Hora (sem mudanças)
        try:
            datetime_selector = 'div.duelParticipant__startTime'
            datetime_el = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, datetime_selector)))
            datetime_str = datetime_el.text
            parts = datetime_str.split(' ')
            if len(parts) >= 2: info['Date'] = parts[0].replace('.', '/'); info['Time'] = parts[1]
            else: logger.warning(f"Formato data/hora inesperado: '{datetime_str}' ID: {id_jogo}")
        except Exception as e: logger.error(f"Erro extrair data/hora ID {id_jogo}: {e}")

        # Extrair País e Liga (COM LIMPEZA)
        breadcrumb_list_selector = "ol.wcl-breadcrumbList_m5Npe"
        try:
            logger.info(f"    Buscando breadcrumbs: '{breadcrumb_list_selector}'")
            breadcrumb_list = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, breadcrumb_list_selector)))
            logger.info("      Lista de breadcrumbs encontrada.")

            # País (sem mudanças)
            try:
                 pais_selector_relative = "li[itemprop='itemListElement']:nth-child(2) span[itemprop='name']"
                 pais_el = breadcrumb_list.find_element(By.CSS_SELECTOR, pais_selector_relative)
                 info['Country'] = pais_el.text.strip().upper() # Padroniza para MAIÚSCULAS
                 logger.info(f"        País: {info['Country']}")
            except NoSuchElementException: logger.warning(f"País (2º breadcrumb) não encontrado ID: {id_jogo}")

            # Liga (COM LIMPEZA)
            try:
                 liga_selector_relative = "li[itemprop='itemListElement']:nth-child(3) span[itemprop='name']"
                 liga_el = breadcrumb_list.find_element(By.CSS_SELECTOR, liga_selector_relative)
                 league_text_raw = liga_el.text.strip()
                 logger.info(f"        Liga (Texto Bruto): {league_text_raw}")

                 league_text_cleaned = re.sub(r'\s*-\s*(ROUND|GROUP|PLAYOFFS?)\b.*', '', league_text_raw, flags=re.IGNORECASE).strip()

                 info['League'] = league_text_cleaned # Salva o nome limpo
                 logger.info(f"        Liga (Limpo): {info['League']}")
            except NoSuchElementException: logger.warning(f"Liga (3º breadcrumb) não encontrada ID: {id_jogo}")
            except Exception as e_league: logger.error(f"Erro ao limpar nome da liga ID {id_jogo}: {e_league}")


        except Exception as e: logger.error(f"Erro processar breadcrumbs ID {id_jogo}: {e}")


        # Extrair Times (sem mudanças)
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


        # Validação Mínima Essencial (verifica se Country e League foram preenchidos após limpeza)
        if not info.get('HomeTeam') or not info.get('AwayTeam') or not info.get('Country') or not info.get('League') or not info.get('Date'):
            logger.error(f"Falha Crítica: Info básica ausente/inválida ID: {id_jogo}. Pulando jogo. Info coletada: {info}")
            return None

        # *** CONSTRÓI O NOME COMPLETO PARA MAPEAMENTO ***
        # Usa o País (padronizado) e a Liga (limpa)
        info['ScraperLeagueName'] = f"{info.get('Country', '')}: {info.get('League', '')}"
        logger.info(f"    Nome Completo Scraper para Map: {info['ScraperLeagueName']}")

        logger.info(f"  -> Info básica OK para Jogo ID: {id_jogo}")
        return info

    except Exception as e_general:
        logger.error(f"Erro inesperado GERAL em get_basic_info Jogo ID {id_jogo}: {e_general}", exc_info=True)
        return None


def get_odds_1x2_ft(driver, id_jogo) -> Dict[str, Optional[float]]:
    """Captura odds 1x2 FT."""
    odds = {ODDS_COLS['home']: None, ODDS_COLS['draw']: None, ODDS_COLS['away']: None}
    h_col, d_col, a_col = ODDS_COLS['home'], ODDS_COLS['draw'], ODDS_COLS['away'] # Nomes internos
    logger.debug(f"  Buscando odds 1x2 FT para Jogo ID: {id_jogo}") # Já existia, talvez mudar para debug
    try:
        url = f"{SCRAPER_BASE_URL}/match/{id_jogo}/#/odds-comparison/1x2-odds/full-time"
        driver.get(url)
        # REMOVIDO sleep, confiar no WebDriverWait
        wait = WebDriverWait(driver, SCRAPER_ODDS_TIMEOUT) # Usa timeout específico para odds
        row_selector = 'div.ui-table__row'
        wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, row_selector))) # Espera primeira linha
        time.sleep(0.3) # Pequena pausa após elemento aparecer, pode ajudar em renderizações lentas

        linhas = driver.find_elements(By.CSS_SELECTOR, row_selector)
        if linhas:
            # Tenta pegar odds da primeira linha
            odds_cells_selectors = 'a.oddsCell__odd, span.oddsCell__odd' # Tenta 'a' ou 'span'
            odds_cells = linhas[0].find_elements(By.CSS_SELECTOR, odds_cells_selectors)
            if len(odds_cells) >= 3:
                odds[h_col] = _safe_float(odds_cells[0].text)
                odds[d_col] = _safe_float(odds_cells[1].text)
                odds[a_col] = _safe_float(odds_cells[2].text)
                # *** NOVO LOG DEBUG ***
                logger.debug(f"    Odds 1x2 FT Capturadas ID {id_jogo}: H={odds[h_col]}, D={odds[d_col]}, A={odds[a_col]}")
            else:
                logging.warning(f"Não encontrou 3+ células de odds ({odds_cells_selectors}) 1x2 FT na primeira linha. Jogo ID: {id_jogo}")
                logger.debug(f"    Conteúdo HTML da primeira linha (1x2): {linhas[0].get_attribute('outerHTML')[:500]}...") # Log HTML para debug
        else:
            logging.warning(f"Nenhuma linha de odds ({row_selector}) 1x2 FT encontrada. Jogo ID: {id_jogo}")

    except TimeoutException:
        logging.warning(f"Timeout ao esperar odds 1x2 FT. Jogo ID: {id_jogo}")
    except NoSuchElementException:
         logging.warning(f"Elemento não encontrado para odds 1x2 FT. Jogo ID: {id_jogo}")
    except Exception as e:
        logging.error(f"Erro inesperado ao buscar odds 1x2 FT Jogo ID {id_jogo}: {e}", exc_info=True) # Log com traceback

    # *** NOVO LOG DEBUG (Final) ***
    # Log mesmo se falhou, para ver os Nones
    logger.debug(f"  Resultado Final Odds 1x2 FT ID {id_jogo}: {odds}")
    return odds

def get_odds_ou25_ft(driver, id_jogo) -> Dict[str, Optional[float]]:
    """Captura odds Over/Under 2.5 FT."""
    odds = {'Odd_Over25_FT': None, 'Odd_Under25_FT': None}
    o_col, u_col = 'Odd_Over25_FT', 'Odd_Under25_FT' # Nomes internos
    logger.debug(f"  Buscando odds O/U 2.5 FT para Jogo ID: {id_jogo}") # Mudar para debug
    try:
        url = f"{SCRAPER_BASE_URL}/match/{id_jogo}/#/odds-comparison/over-under/full-time"
        driver.get(url)
        # REMOVIDO sleep, confiar no WebDriverWait
        wait = WebDriverWait(driver, SCRAPER_ODDS_TIMEOUT)

        xpath_linha_25 = "//span[@data-testid='wcl-oddsValue' and normalize-space(.)='2.5']/ancestor::div[contains(@class, 'ui-table__row')]"
        logger.debug(f"    Aguardando linha O/U 2.5 com XPath: {xpath_linha_25}")

        linha_25_encontrada = wait.until(EC.visibility_of_element_located((By.XPATH, xpath_linha_25)))
        logger.debug("    Linha O/U 2.5 encontrada.")
        time.sleep(0.3) # Pausa opcional

        if linha_25_encontrada:
            try:
                odds_cells_selectors = 'a.oddsCell__odd, span.oddsCell__odd'
                odds_cells = linha_25_encontrada.find_elements(By.CSS_SELECTOR, odds_cells_selectors)
                logger.debug(f"      Encontradas {len(odds_cells)} células de odds ({odds_cells_selectors}) na linha 2.5.")
                if len(odds_cells) >= 2:
                    odds[o_col] = _safe_float(odds_cells[0].text)
                    odds[u_col] = _safe_float(odds_cells[1].text)
                    # *** NOVO LOG DEBUG ***
                    logger.debug(f"      Odds O/U 2.5 FT Capturadas ID {id_jogo}: O={odds[o_col]}, U={odds[u_col]}")
                else:
                    logging.warning(f"Não encontrou 2+ células de odds O/U 2.5 FT na linha encontrada. Verifique o seletor. Jogo ID: {id_jogo}")
                    logger.debug(f"      Conteúdo HTML da linha O/U 2.5: {linha_25_encontrada.get_attribute('outerHTML')[:500]}...") # Log HTML para debug
            except Exception as e_extract:
                 logging.error(f"Erro ao extrair odds da linha 2.5 encontrada. Jogo ID {id_jogo}: {e_extract}", exc_info=True)

    except TimeoutException:
        logging.warning(f"Timeout ao esperar/encontrar linha O/U 2.5. Jogo ID: {id_jogo}")
    except NoSuchElementException:
         logging.warning(f"Elemento linha O/U 2.5 não encontrado. Jogo ID: {id_jogo}")
    except Exception as e:
        logging.error(f"Erro inesperado ao buscar odds O/U 2.5 FT Jogo ID {id_jogo}: {e}", exc_info=True)

    # *** NOVO LOG DEBUG (Final) ***
    logger.debug(f"  Resultado Final Odds O/U 2.5 FT ID {id_jogo}: {odds}")
    return odds


def get_odds_btts_ft(driver, id_jogo) -> Dict[str, Optional[float]]:
    """Captura odds BTTS Yes/No FT."""
    odds = {'Odd_BTTS_Yes': None, 'Odd_BTTS_No': None}
    y_col, n_col = 'Odd_BTTS_Yes', 'Odd_BTTS_No' # Nomes internos
    logger.debug(f"  Buscando odds BTTS FT para Jogo ID: {id_jogo}") # Mudar para debug
    try:
        url = f"{SCRAPER_BASE_URL}/match/{id_jogo}/#/odds-comparison/both-teams-to-score/full-time"
        driver.get(url)
        # REMOVIDO sleep, confiar no WebDriverWait
        wait = WebDriverWait(driver, SCRAPER_ODDS_TIMEOUT)
        row_selector = 'div.ui-table__row'
        wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, row_selector)))
        time.sleep(0.3) # Pausa opcional

        linhas = driver.find_elements(By.CSS_SELECTOR, row_selector)
        if linhas:
            odds_cells_selectors = 'a.oddsCell__odd, span.oddsCell__odd'
            odds_cells = linhas[0].find_elements(By.CSS_SELECTOR, odds_cells_selectors)
            if len(odds_cells) >= 2:
                odds[y_col] = _safe_float(odds_cells[0].text)
                odds[n_col] = _safe_float(odds_cells[1].text)
                # *** NOVO LOG DEBUG ***
                logger.debug(f"    Odds BTTS FT Capturadas ID {id_jogo}: Yes={odds[y_col]}, No={odds[n_col]}")
            else:
                logging.warning(f"Não encontrou 2+ células de odds ({odds_cells_selectors}) BTTS FT. Jogo ID: {id_jogo}")
                logger.debug(f"    Conteúdo HTML da primeira linha (BTTS): {linhas[0].get_attribute('outerHTML')[:500]}...") # Log HTML para debug
        else:
            logging.warning(f"Nenhuma linha de odds ({row_selector}) BTTS FT encontrada. Jogo ID: {id_jogo}")

    except TimeoutException:
        logging.warning(f"Timeout ao esperar odds BTTS FT. Jogo ID: {id_jogo}")
    except NoSuchElementException:
         logging.warning(f"Elemento não encontrado para odds BTTS FT. Jogo ID: {id_jogo}")
    except Exception as e:
        logging.error(f"Erro inesperado ao buscar odds BTTS FT Jogo ID {id_jogo}: {e}", exc_info=True)

    # *** NOVO LOG DEBUG (Final) ***
    logger.debug(f"  Resultado Final Odds BTTS FT ID {id_jogo}: {odds}")
    return odds

# --- Função Principal de Scraping (Integrada) ---
def scrape_upcoming_fixtures(chromedriver_path: Optional[str] = CHROMEDRIVER_PATH, headless: bool = True) -> Optional[pd.DataFrame]:
    driver = _initialize_driver(chromedriver_path, headless)
    if not driver: return None

    all_scraped_data = []
    try:
        driver.get(SCRAPER_BASE_URL)
        # driver.maximize_window() # Pode causar problemas em headless, remover se necessário
        _close_cookies(driver)

        if not _select_target_day(driver, SCRAPER_TARGET_DAY, SCRAPER_TIMEOUT): return None
        match_ids = _get_match_ids(driver, SCRAPER_TIMEOUT)
        if not match_ids:
            print(f"Nenhum jogo encontrado para {SCRAPER_TARGET_DAY}.")
            return pd.DataFrame() # Retorna DF vazio

        # --- LOG INDICANDO MODO DE FILTRAGEM ---
        if SCRAPER_FILTER_LEAGUES:
            print(f"\nIniciando scraping detalhado para {len(match_ids)} jogos encontrados (FILTRANDO por ligas alvo)...")
            print(f"Ligas Alvo (IDs Internos): {TARGET_LEAGUES_INTERNAL_IDS}")
        else:
            print(f"\nIniciando scraping detalhado para {len(match_ids)} jogos encontrados (SEM FILTRO de ligas)...")

        processed_count = 0
        skipped_count = 0

        for match_id in tqdm(match_ids, total=len(match_ids), desc=f"Scraping {SCRAPER_TARGET_DAY}"):
            # 1. Obter Informações Básicas (inclui 'ScraperLeagueName')
            basic_info = get_basic_info(driver, match_id)
            if not basic_info or 'ScraperLeagueName' not in basic_info:
                logger.warning(f"Jogo ID {match_id}: Falha ao obter info básica ou ScraperLeagueName ausente. Pulando.")
                skipped_count += 1
                time.sleep(0.2) # Pequena pausa antes do próximo
                continue

            # 2. Tentar Mapear a Liga (SEMPRE TENTAR)
            full_league_name_scraped = basic_info['ScraperLeagueName'] # Nome construído: "PAÍS: Liga Limpa"
            internal_league_id = None
            scraped_name_norm = full_league_name_scraped.strip().lower()

            # Procura por uma correspondência exata primeiro (case-insensitive)
            for scraper_map_key, internal_id_map in SCRAPER_TO_INTERNAL_LEAGUE_MAP.items():
                if scraper_map_key.strip().lower() == scraped_name_norm:
                    internal_league_id = internal_id_map
                    logger.debug(f"  Jogo ID {match_id}: Mapeamento EXATO encontrado para '{full_league_name_scraped}' -> '{internal_league_id}'")
                    break

            # Fallback: Tentar correspondência parcial se exata falhar (Opcional, pode ser impreciso)

            # 3. Aplicar Filtro ou Definir Nome da Liga Final
            should_process = False
            final_league_name = None

            if SCRAPER_FILTER_LEAGUES:
                # Modo Filtro: Só processa se mapeado E na lista alvo
                if internal_league_id is not None and internal_league_id in TARGET_LEAGUES_INTERNAL_IDS:
                    should_process = True
                    final_league_name = internal_league_id # Usa ID interno
                    logger.info(f"  Jogo ID {match_id} ({full_league_name_scraped}): DENTRO do alvo ('{internal_league_id}'). Coletando odds...")
                else:
                    # Se não mapeado ou não na lista alvo, PULA o jogo
                    logger.debug(f"  Jogo ID {match_id} ({full_league_name_scraped}): Fora do alvo ou não mapeado (ID: {internal_league_id}). Pulando.")
                    skipped_count += 1
                    continue # Pula para o próximo jogo
            else:
                # Modo Sem Filtro: Processa TODOS os jogos
                should_process = True
                if internal_league_id is not None:
                    final_league_name = internal_league_id # Usa ID interno se mapeado
                    logger.info(f"  Jogo ID {match_id} ({full_league_name_scraped}): Mapeado para '{internal_league_id}' (sem filtro). Coletando odds...")
                else:
                    final_league_name = full_league_name_scraped # Usa nome raspado se não mapeado
                    logger.info(f"  Jogo ID {match_id} ({full_league_name_scraped}): NÃO MAPEADO (sem filtro). Usando nome raspado. Coletando odds...")

            # 4. Se deve processar, coleta as odds e adiciona
            if should_process:
                processed_count += 1
                # Atualiza a coluna 'League' com o nome final definido
                basic_info['League'] = final_league_name
                # Remove a chave auxiliar
                if 'ScraperLeagueName' in basic_info:
                    del basic_info['ScraperLeagueName']

                # Coletar Odds
                odds_1x2 = get_odds_1x2_ft(driver, match_id)
                odds_ou25 = get_odds_ou25_ft(driver, match_id)
                odds_btts = get_odds_btts_ft(driver, match_id)

                # Combinar dados
                final_jogo_data = basic_info.copy()
                final_jogo_data.update(odds_1x2)
                final_jogo_data.update(odds_ou25)
                final_jogo_data.update(odds_btts)
                all_scraped_data.append(final_jogo_data)

                time.sleep(SCRAPER_SLEEP_BETWEEN_GAMES) # Pausa entre jogos processados

        # --- Fim do Loop ---
        print(f"\nScraping concluído.")
        print(f"  - Jogos processados (odds coletadas): {processed_count}")
        if SCRAPER_FILTER_LEAGUES:
            print(f"  - Jogos pulados (fora do filtro ou erro info básica): {skipped_count}")
        else:
             print(f"  - Jogos pulados (erro info básica): {skipped_count}")

        if not all_scraped_data:
            print("Nenhum dado de jogo válido foi coletado.")
            return pd.DataFrame()

        df_fixtures = pd.DataFrame(all_scraped_data)

        # Log de odds ausentes
        missing_1x2 = df_fixtures[ODDS_COLS['home']].isnull().sum() + df_fixtures[ODDS_COLS['draw']].isnull().sum() + df_fixtures[ODDS_COLS['away']].isnull().sum()
        missing_ou = df_fixtures['Odd_Over25_FT'].isnull().sum() + df_fixtures['Odd_Under25_FT'].isnull().sum()
        missing_btts = df_fixtures['Odd_BTTS_Yes'].isnull().sum() + df_fixtures['Odd_BTTS_No'].isnull().sum()
        print(f"  Resumo de Odds Ausentes (células): 1x2={missing_1x2}, O/U 2.5={missing_ou}, BTTS={missing_btts}")

        # Garantir que todas as colunas de odds esperadas existam (mesmo que vazias)
        expected_odds_cols = list(ODDS_COLS.values()) + OTHER_ODDS_NAMES
        for col in expected_odds_cols:
             if col not in df_fixtures.columns: df_fixtures[col] = None

        # Renomear colunas básicas para o padrão interno (se necessário - get_basic_info já usa os nomes corretos)
        df_fixtures.rename(columns={'HomeTeam': 'Home', 'AwayTeam': 'Away'}, inplace=True) # Ajuste se necessário

        # Adicionar 'Time_Str' se não existir (embora get_basic_info deva criar 'Time')
        if 'Time' in df_fixtures.columns and 'Time_Str' not in df_fixtures.columns:
             df_fixtures['Time_Str'] = df_fixtures['Date'] + ' ' + df_fixtures['Time'] # Exemplo

        print(f"DataFrame final com {len(df_fixtures)} jogos gerado.")
        return df_fixtures

    except KeyboardInterrupt:
        print("\nScraping interrompido pelo usuário.")
        # Retorna o que foi coletado até agora, se houver
        if all_scraped_data: return pd.DataFrame(all_scraped_data)
        else: return pd.DataFrame()
    except Exception as e_global:
        logging.error(f"Erro GERAL INESPERADO durante scraping: {e_global}", exc_info=True)
        # Tenta retornar o que foi coletado
        if all_scraped_data: return pd.DataFrame(all_scraped_data)
        else: return None # Indica falha maior
    finally:
        if driver:
            try:
                driver.quit()
                print("WebDriver fechado.")
            except Exception as e_quit:
                 print(f"Erro ao fechar WebDriver: {e_quit}")

# --- Bloco de Teste (opcional) ---
if __name__ == '__main__':
    print("--- Iniciando Teste do Scraper ---")
    # Forçar SEM filtro para testar a nova opção
    SCRAPER_FILTER_LEAGUES = False
    print(f"Modo de Filtro: {SCRAPER_FILTER_LEAGUES}")
    # Reduzir pausas para teste rápido
    SCRAPER_SLEEP_BETWEEN_GAMES = 0.5
    SCRAPER_SLEEP_AFTER_NAV = 2

    # Definir nível de log para DEBUG para ver mais detalhes
    logger.setLevel(logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG) # Define para o root logger também, se necessário

    # Executar o scraper
    df_scraped = scrape_upcoming_fixtures(headless=True) # Use headless=False para ver o navegador

    if df_scraped is not None:
        print("\n--- Resultado do Scraping ---")
        print(f"Total de jogos coletados: {len(df_scraped)}")
        if not df_scraped.empty:
            print("Tipos de dados:")
            print(df_scraped.info())
            print("\nAmostra dos dados:")
            print(df_scraped.head())
            print("\nContagem de Ligas:")
            print(df_scraped['League'].value_counts())

            # Salvar localmente para inspeção (opcional)
            try:
                save_path = os.path.join(os.path.dirname(__file__), '..', 'data', f'test_scraped_{SCRAPER_TARGET_DAY}.csv')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                df_scraped.to_csv(save_path, index=False)
                print(f"\nDados salvos em: {save_path}")
            except Exception as e_save:
                 print(f"\nErro ao salvar CSV de teste: {e_save}")

        else:
            print("Scraper retornou um DataFrame vazio.")
    else:
        print("\nScraper falhou e retornou None.")

    print("--- Teste do Scraper Concluído ---")