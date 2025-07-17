from typing import Dict
from .base_strategy import RuleBasedBettingStrategy
import pandas as pd
import numpy as np

class LayAwayStrategy(RuleBasedBettingStrategy):
    
    def get_display_name(self) -> str:
        return "Lay ao Visitante (Regras Estatísticas)"
    
    def get_target_variable_name(self) -> str:
        """Retorna o nome do alvo para esta estratégia de regras."""
        return "LayAwaySuccess"

    def find_entries(self, df_data: pd.DataFrame) -> pd.DataFrame:

        jogos = df_data.copy()
        
        required_cols = ['Odd_H_Back', 'Odd_A_Back', 'Odd_D_Back', 'Odd_A_Lay']
        for col in required_cols:
            if col not in jogos.columns: return pd.DataFrame() 
            jogos[col] = pd.to_numeric(jogos[col], errors='coerce')
        
        jogos.dropna(subset=required_cols, inplace=True)
        if jogos.empty: return pd.DataFrame()

        jogos['VAR1'] = np.sqrt((jogos['Odd_H_Back'] - jogos['Odd_A_Back'])**2)
        jogos['VAR2'] = np.degrees(np.arctan((jogos['Odd_A_Back'] - jogos['Odd_H_Back']) / 2))
        jogos['VAR3'] = np.degrees(np.arctan((jogos['Odd_D_Back'] - jogos['Odd_A_Back']) / 2))

        flt = (jogos.VAR1 >= 4) & (jogos.VAR2 >= 60) & (jogos.VAR3 <= -60) & \
              (jogos.Odd_A_Lay >= 2) & (jogos.Odd_A_Lay <= 50)
              
        return jogos[flt]

    def get_odds_col_for_backtesting(self) -> str:

        return 'Odd_A_Lay'

    def get_target_for_backtesting(self, df_entries: pd.DataFrame, goals_cols: Dict) -> pd.Series:

        home_goals_col = goals_cols.get('home')
        away_goals_col = goals_cols.get('away')
        
        home_goals = pd.to_numeric(df_entries[home_goals_col], errors='coerce')
        away_goals = pd.to_numeric(df_entries[away_goals_col], errors='coerce')

        away_did_not_win = (away_goals <= home_goals)
        
        return away_did_not_win.astype('Int64')