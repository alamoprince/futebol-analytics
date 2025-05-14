from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Optional, Dict

class BettingStrategy(ABC):
    
    @abstractmethod
    def get_target_variable_name(self) -> str:
        """Returns the name of the target variable for the betting strategy."""
        pass
    
    @abstractmethod
    def get_preprocess_target(self, df_historical: pd.DataFrame, goals_cols: Dict) -> pd.DataFrame:
        """Preprocess the target variable for the betting strategy."""
        pass
    
    @abstractmethod
    def get_relevant_odds_cols(self) -> List[str]:
        """Returns the relevant odds columns for the betting strategy."""
        pass
    
    def get_model_config_key_prefix(self) -> str: 
        return self.__class__.__name__.replace("Strategy", "") 

    @abstractmethod
    def get_display_name(self) -> str:
        pass

    def get_feature_columns(self, all_candidate_features: List[str]) -> List[str]:
        from config import FEATURE_COLUMNS 
        return FEATURE_COLUMNS

class BackDrawStrategy(BettingStrategy):
    def get_target_variable_name(self) -> str:
        return "IsDraw"

    def get_preprocess_target(self, df_historical: pd.DataFrame, goals_cols: Dict) -> pd.Series:
        gh = goals_cols.get('home')
        ga = goals_cols.get('away')
        if gh not in df_historical.columns or ga not in df_historical.columns:
            raise ValueError("Colunas de gols ausentes para BackDrawStrategy")
        h_g = pd.to_numeric(df_historical[gh], errors='coerce')
        a_g = pd.to_numeric(df_historical[ga], errors='coerce')
        return (h_g == a_g).astype(int) 

    def get_relevant_odds_cols(self) -> List[str]:
        from config import ODDS_COLS
        return [ODDS_COLS['draw']]

    def get_display_name(self) -> str:
        return "Apostar no Empate (Back Draw)"

class BackAwayStrategy(BettingStrategy):
    def get_target_variable_name(self) -> str:
        return "IsAwayWin"

    def preprocess_target(self, df_historical: pd.DataFrame, goals_cols: Dict) -> pd.Series:
        gh = goals_cols.get('home')
        ga = goals_cols.get('away')
        if gh not in df_historical.columns or ga not in df_historical.columns:
            raise ValueError("Colunas de gols ausentes para BackAwayStrategy")
        h_g = pd.to_numeric(df_historical[gh], errors='coerce')
        a_g = pd.to_numeric(df_historical[ga], errors='coerce')
        return (a_g > h_g).astype(int)

    def get_relevant_odds_columns(self) -> List[str]:
        from config import ODDS_COLS
        return [ODDS_COLS['away']]

    def get_display_name(self) -> str:
        return "Apostar no Visitante (Back Away)"