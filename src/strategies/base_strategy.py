from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Optional, Dict, Literal

StrategyType = Literal["machine_learning", "rule_based"]


class BettingStrategy(ABC):

    @abstractmethod
    def get_display_name(self) -> str:
        """Retorna o nome amigável da estratégia para ser exibido na GUI."""
        pass
    
    @abstractmethod
    def get_strategy_type(self) -> StrategyType:
        """Indica se a estratégia usa ML ou é baseada em regras."""
        pass

    @abstractmethod
    def get_target_variable_name(self) -> str: 
        """Retorna o nome da coluna que representa o alvo da estratégia (para análise e/ou treino)."""
        pass

class MLBettingStrategy(BettingStrategy, ABC):
    """
    Classe base para todas as estratégias que usam um modelo de Machine Learning.
    Ela implementa o tipo e define os novos métodos abstratos necessários para o ML.
    """
    
    def get_strategy_type(self) -> StrategyType:
        return "machine_learning"

    @abstractmethod
    def preprocess_target(self, df: pd.DataFrame, goals_cols: Dict) -> pd.Series:
        """Processa o DataFrame histórico e retorna uma Série com o alvo binário."""
        pass
    
    @abstractmethod
    def get_feature_columns(self, all_features: List[str]) -> List[str]:
        """Retorna a lista de nomes de features que o modelo usará."""
        pass
    
    @abstractmethod
    def get_relevant_odds_cols(self) -> List[str]:
        """Retorna a(s) coluna(s) de odds para o cálculo de EV."""
        pass

    def get_model_config_key_prefix(self) -> str:
        """Gera um prefixo para nomes de arquivos (pode ser sobrescrito)."""
        return self.__class__.__name__.replace("Strategy", "")

class RuleBasedBettingStrategy(BettingStrategy, ABC):
    """
    Classe base para estratégias baseadas em regras estatísticas/heurísticas.
    """

    def get_strategy_type(self) -> StrategyType:
        return "rule_based"

    @abstractmethod
    def find_entries(self, df_data: pd.DataFrame) -> pd.DataFrame:
        """Aplica as regras e retorna um DataFrame com as entradas."""
        pass

    @abstractmethod
    def get_odds_col_for_backtesting(self) -> str:
        """Retorna a coluna de odd usada para calcular o resultado da aposta."""
        pass
        
    @abstractmethod
    def get_target_for_backtesting(self, df_entries: pd.DataFrame, goals_cols: Dict) -> pd.Series:
        """A partir das entradas, determina o resultado da aposta (1=ganha, 0=perdida)."""
        pass