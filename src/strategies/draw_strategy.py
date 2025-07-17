from .base_strategy import MLBettingStrategy, StrategyType
import pandas as pd
import numpy as np
from typing import List, Dict
import os

class BackDrawStrategy(MLBettingStrategy):
    def __init__(self):
        super().__init__()
        
    def get_display_name(self) -> str:
        return "Apostar no Empate (Back Draw)"

    def get_target_variable_name(self) -> str:
        return "IsDraw"

    def get_model_config_key_prefix(self) -> str:
        return "BackDraw"

    def preprocess_target(self, df_historical: pd.DataFrame, goals_cols: Dict) -> pd.Series:
        home_goals_col = goals_cols.get('home')
        away_goals_col = goals_cols.get('away')
        if home_goals_col not in df_historical.columns or away_goals_col not in df_historical.columns:
            raise ValueError("Colunas de gols essenciais ('home', 'away') não encontradas no DataFrame.")
        home_goals = pd.to_numeric(df_historical[home_goals_col], errors='coerce')
        away_goals = pd.to_numeric(df_historical[away_goals_col], errors='coerce')
        is_draw_series = (home_goals == away_goals)
        return is_draw_series.astype('Int64')

    def get_relevant_odds_cols(self) -> List[str]:
        from config import ODDS_COLS
        if 'draw' not in ODDS_COLS:
            raise KeyError("A chave 'draw' não foi encontrada no dicionário ODDS_COLS do config.")
        return [ODDS_COLS['draw']]

    def get_feature_columns(self, all_candidate_features: List[str]) -> List[str]:
        from config import (
            PIRATING_MOMENTUM_A,
            INTERACTION_P_D_NORM_DIV_CV_HDA,
            INTERACTION_P_D_NORM_X_PIR_DIFF
        )
        draw_model_features = [
            'Prob_Empate_Poisson',
            'p_D_norm',
            'abs_ProbDiff_Norm',
            'CV_HDA',
            'Media_VG_H',
            'Media_VG_A',
            'Media_CG_H',
            'Media_CG_A',
            'Std_CG_H',
            'Std_CG_A',
            'PiRating_Diff',
            PIRATING_MOMENTUM_A,
            INTERACTION_P_D_NORM_DIV_CV_HDA,
            INTERACTION_P_D_NORM_X_PIR_DIFF,
        ]

        available_features = [f for f in draw_model_features if f in all_candidate_features]
        
        missing_features = set(draw_model_features) - set(available_features)
        if missing_features:
            print(f"AVISO (BackDrawStrategy): As seguintes features esperadas não foram encontradas nos dados: {missing_features}")

        return available_features