from .base_strategy import BettingStrategy
from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Optional, Dict

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