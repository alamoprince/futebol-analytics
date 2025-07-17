import pandas as pd
import numpy as np
from typing import Dict, Optional

# Importações do projeto
from strategies.base_strategy import RuleBasedBettingStrategy
from config import GOALS_COLS
from logger_config import setup_logger

logger = setup_logger("Backtester")

class RuleBasedBacktester:
    """
    Executa um backtest para uma dada estratégia baseada em regras em um
    conjunto de dados históricos.
    """
    def __init__(self, strategy: RuleBasedBettingStrategy):
        if not isinstance(strategy, RuleBasedBettingStrategy):
            raise TypeError("A estratégia fornecida deve ser do tipo RuleBasedBettingStrategy.")
        self.strategy = strategy
        self.results: Dict[str, any] = {}

    def run(self, historical_df: pd.DataFrame) -> bool:
        """
        Executa o pipeline completo de backtesting.

        Args:
            historical_df: DataFrame com todos os jogos históricos.

        Returns:
            True se o backtest foi bem-sucedido, False caso contrário.
        """
        logger.info(f"Iniciando backtest para a estratégia: {self.strategy.get_display_name()}")
        try:
            # 1. Encontrar as entradas de acordo com as regras da estratégia
            logger.info("Encontrando entradas no histórico...")
            entries_df = self.strategy.find_entries(historical_df)

            if entries_df.empty:
                logger.warning("Backtest concluído: Nenhuma entrada encontrada para esta estratégia no histórico.")
                self.results = {"total_entries": 0}
                return True
            
            num_entries = len(entries_df)
            logger.info(f"{num_entries} entradas encontradas. Calculando resultados...")

            # 2. Obter a coluna de odds relevante para esta aposta
            odds_col = self.strategy.get_odds_col_for_backtesting()
            if odds_col not in entries_df.columns:
                raise ValueError(f"A coluna de odds necessária '{odds_col}' não foi encontrada nos dados.")
            
            odds = pd.to_numeric(entries_df[odds_col], errors='coerce')

            # 3. Determinar o resultado de cada aposta (1 para vitória, 0 para derrota)
            y_true = self.strategy.get_target_for_backtesting(entries_df, GOALS_COLS)

            # Valida se os resultados e odds estão alinhados
            if len(y_true) != len(odds):
                raise ValueError("O número de resultados (y_true) e odds não coincidem.")
            
            # Remove entradas onde não foi possível calcular o resultado ou a odd era inválida
            valid_mask = y_true.notna() & odds.notna()
            y_true = y_true[valid_mask]
            odds = odds[valid_mask]
            num_valid_bets = len(y_true)

            if num_valid_bets == 0:
                logger.warning("Nenhuma aposta válida encontrada após a limpeza de NaNs.")
                self.results = {"total_entries": num_entries, "valid_bets": 0}
                return True

            # 4. Calcular métricas de desempenho
            # A lógica assume uma aposta LAY. Se a aposta for ganha (y_true=1), o lucro é de +1 unidade (a stake).
            # Se a aposta for perdida (y_true=0), o prejuízo é a responsabilidade: -(odd - 1).
            profit_per_bet = np.where(y_true == 1, 1.0, -(odds - 1))
            
            total_profit = np.nansum(profit_per_bet)
            accuracy = y_true.mean() * 100 if num_valid_bets > 0 else 0
            
            # Para o ROI de uma aposta LAY, o "investimento" total é a soma da responsabilidade
            total_liability = np.nansum(np.where(y_true == 0, odds - 1, 0))
            roi = (total_profit / total_liability) * 100 if total_liability > 0 else float('inf') if total_profit > 0 else 0

            # Armazena os resultados
            self.results = {
                "strategy_name": self.strategy.get_display_name(),
                "total_entries": num_entries,
                "valid_bets": num_valid_bets,
                "accuracy_pct": accuracy,
                "total_profit_units": total_profit,
                "roi_pct": roi
            }
            logger.info(f"Backtest concluído com sucesso para '{self.strategy.get_display_name()}'.")
            return True

        except Exception as e:
            logger.error(f"Falha durante a execução do backtest: {e}", exc_info=True)
            self.results = {"error": str(e)}
            return False

    def get_results(self) -> Dict[str, any]:
        """Retorna o dicionário com os resultados do backtest."""
        return self.results
        
    def get_results_as_text(self) -> str:
        """Formata os resultados do backtest como uma string para exibição na GUI."""
        if not self.results:
            return "Nenhum resultado de backtest disponível."
        if "error" in self.results:
            return f"Erro durante o backtest:\n{self.results['error']}"
        if self.results.get("total_entries", 0) == 0:
            return f"Backtest para '{self.strategy.get_display_name()}':\n\nNenhuma oportunidade de aposta foi encontrada no histórico."
        
        text = f"--- Backtest da Estratégia: {self.results['strategy_name']} ---\n\n"
        text += f"Total de Entradas Encontradas: {self.results['total_entries']}\n"
        text += f"Apostas Válidas Analisadas: {self.results['valid_bets']}\n"
        text += f"Taxa de Acerto da Estratégia: {self.results['accuracy_pct']:.2f}%\n"
        text += f"Lucro/Prejuízo Total: {self.results['total_profit_units']:+.2f} unidades\n"
        
        roi_str = f"{self.results['roi_pct']:.2f}%" if np.isfinite(self.results['roi_pct']) else "N/A (sem perdas)"
        text += f"ROI (baseado na responsabilidade): {roi_str}\n"
        
        return text