# --- START OF FILE analyzer.py ---

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List

class MatchAnalyzer:
    """
    Realiza análises estatísticas sobre dados de partidas de futebol.
    Esta classe é projetada para ser stateless, operando sobre os DataFrames fornecidos.
    """

    def _parse_score(self, score_str: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
        """
        Analisa uma string de placar no formato 'Casa:Fora' (ex: '2:1').

        Args:
            score_str: A string do placar.

        Returns:
            Uma tupla (gols_casa, gols_fora). Retorna (None, None) se o placar for inválido.
        """
        if not score_str or ':' not in score_str:
            # print(f"⚠️ Warning: Formato de placar inválido ou ausente: '{score_str}'")
            return (None, None)
        try:
            # Considera apenas a parte principal antes de espaços (ex: ignora '(AET)')
            main_score = score_str.strip().split(' ')[0]
            home_goals, away_goals = map(int, main_score.split(':'))
            return (home_goals, away_goals)
        except (ValueError, IndexError):
            # print(f"⚠️ Warning: Não foi possível parsear o placar: '{score_str}'")
            return (None, None)

    def calculate_win_rate(self, df: pd.DataFrame, team_name: str) -> float:
        """Calcula a taxa de vitórias para um time específico no DataFrame fornecido."""
        wins = 0
        valid_matches = 0
        if df.empty:
            return 0.0

        for _, row in df.iterrows():
            home_team = row.get('home_team') 
            away_team = row.get('away_team')
            score = row.get('score')

            home_goals, away_goals = self._parse_score(score)

            if home_goals is None or away_goals is None:
                continue # Ignora partidas com placar inválido

            valid_matches += 1
            if (home_team == team_name and home_goals > away_goals) or \
               (away_team == team_name and away_goals > home_goals):
                wins += 1

        return (wins / valid_matches) if valid_matches > 0 else 0.0

    def calculate_draw_rate(self, df: pd.DataFrame, team_name: str) -> float:
        """Calcula a taxa de empates para um time específico."""
        draws = 0
        valid_matches = 0
        if df.empty:
            return 0.0

        for _, row in df.iterrows():
             # Verifica se o time participou da partida
            if row.get('home_team') != team_name and row.get('away_team') != team_name:
                continue

            home_goals, away_goals = self._parse_score(row.get('score'))

            if home_goals is None or away_goals is None:
                continue # Ignora partidas com placar inválido

            valid_matches += 1
            if home_goals == away_goals:
                draws += 1

        return (draws / valid_matches) if valid_matches > 0 else 0.0

    def calculate_average_goals(self, df: pd.DataFrame, team_name: str) -> Tuple[float, float]:
       
        """Calcula a média de gols marcados e sofridos por um time."""
        
        goals_scored = 0
        goals_conceded = 0
        valid_matches = 0
        if df.empty:
            return (0.0, 0.0)

        for _, row in df.iterrows():
            home_team = row.get('Home')
            away_team = row.get('Away')
            score = row.get('score')
            home_goals, away_goals = self._parse_score(score)

            if home_goals is None or away_goals is None:
                continue # Ignora placar inválido

            # Verifica se o time participou
            if home_team != team_name and away_team != team_name:
                 continue

            valid_matches += 1
            if home_team == team_name:
                goals_scored += home_goals
                goals_conceded += away_goals
            elif away_team == team_name:
                goals_scored += away_goals
                goals_conceded += home_goals

        avg_scored = (goals_scored / valid_matches) if valid_matches > 0 else 0.0
        avg_conceded = (goals_conceded / valid_matches) if valid_matches > 0 else 0.0
        return (avg_scored, avg_conceded)

    def calculate_team_stats(self, df: pd.DataFrame, team_name: str) -> Dict[str, float]:
        """
        Calcula um conjunto de estatísticas agregadas para um time com base em seu histórico.

        Args:
            df: DataFrame com o histórico de jogos do time.
            team_name: Nome do time a ser analisado.

        Returns:
            Dicionário com as estatísticas calculadas.
        """
        if df.empty or team_name not in pd.concat([df.get('home_team', pd.Series()), df.get('away_team', pd.Series())]).unique():
             print(f"ℹ️ Dataframe vazio ou time '{team_name}' não encontrado nos jogos fornecidos.")
             # Retorna um dicionário com zeros para evitar erros posteriores
             return {
                 'avg_goals_scored': 0.0, 'avg_goals_conceded': 0.0,
                 'win_rate': 0.0, 'draw_rate': 0.0, 'loss_rate': 0.0,
                 # Adicione outras estatísticas com valor padrão 0.0 aqui se necessário
             }

        stats: Dict[str, float] = {}

        # Estatísticas básicas de gols
        avg_scored, avg_conceded = self.calculate_average_goals(df, team_name)
        stats['avg_goals_scored'] = avg_scored
        stats['avg_goals_conceded'] = avg_conceded

        # Taxas de Resultado (Vitória, Empate, Derrota)
        stats['win_rate'] = self.calculate_win_rate(df, team_name)
        stats['draw_rate'] = self.calculate_draw_rate(df, team_name)
        # Taxa de derrota é 1 - (vitorias + empates)
        stats['loss_rate'] = 1.0 - (stats['win_rate'] + stats['draw_rate'])

        return stats

    def compare_stats(self, stats1: Dict[str, float], stats2: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
        """
        Compara dicionários de estatísticas de dois times.

        Args:
            stats1: Dicionário de estatísticas do time 1.
            stats2: Dicionário de estatísticas do time 2.

        Returns:
            Dicionário onde as chaves são os nomes das estatísticas e os valores
            são tuplas (valor_time1, valor_time2).
        """
        comparison: Dict[str, Tuple[float, float]] = {}
        # Garante que apenas estatísticas presentes em ambos sejam comparadas
        common_keys = set(stats1.keys()) & set(stats2.keys())
        for key in common_keys:
            comparison[key] = (stats1[key], stats2[key])

        return comparison

