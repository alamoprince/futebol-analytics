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
