import numpy as np

from datetime import datetime, timedelta
from typing import List

from ..objects import Game
from ..databases import PlayerDatabase, GameDatabase
from .rating_system import RatingSystem

class ChessMetrics(RatingSystem):
    def __init__(self, default_change=850, too_old=2 * 365, epsilon=1e-2, 
                 home_advantage=20, delta_expected=2, weighted_average=0.99, **kwargs) -> 'ChessMetrics':
        """
        The ChessMetrics rating system implementation. See https://en.wikipedia.org/wiki/Chessmetrics
        Also partially based on https://www.kaggle.com/competitions/chess/discussion/193.

        We adjust the rating system to ensure it always converges. For this purpose, we perform several
        iterations of the process described in the Wikipedia page over all games with a decay factor (weighted_average). 

        This class represents the ChessMetrics rating system used to calculate the ratings of chess players.
        It inherits from the `RatingSystem` class.

        Attributes:
            - default_change (float): The default rating change value.
            - too_old (int): The number of days after which a game is considered too old.
            - epsilon (float): The max change allowed until convergence.
            - home_advantage (int): The advantage given to the home player.
            - delta_expected (float): The weighting to do in the logistic curve formula.
            - weighted_average (float): The weighted_average coefficient used in updated the ratings.

        Args:
            - default_change (float, optional): The default change value. Defaults to 850.
            - too_old (int, optional): The number of days after which a game is considered too old. Defaults to 2 years.
            - epsilon (float, optional): The max change allowed until convergence. Defaults to 50.
            - home_advantage (int, optional): The advantage given to the home player. Defaults to 20.
            - delta_expected (float, optional): The weighting to do in the logistic curve formula. Defaults to 2.
            - weighted_average (float, optional): The weighted_average coefficient used in updated the ratings. Defaults to 0.9.
        """
        super().__init__(**kwargs, default_change=default_change, too_old=too_old, epsilon=epsilon, 
                         home_advantage=home_advantage, delta_expected=delta_expected, 
                         weighted_average=weighted_average)
        self.too_old_timedelta = timedelta(days=self.too_old)

    def compute_weights(self, games : List[Game], last_period_date : datetime):
        """
        Computes the weights of the games.

        Args:
            - games (list): A list of game objects.
            - last_period_date (datetime): The last date of the period.
        
        Returns:
            - list: A list of weights for the games.
        """
        weights = []
        for game in games:
            date = game.get_date()
            age = (last_period_date - date).days
            weight = max(0, self.too_old - age) / self.too_old
            weights.append(game.weight * weight)
        return weights

    def period_update(self, player_database : PlayerDatabase, 
                      game_database : GameDatabase, 
                      period_dates : List[datetime], **kwargs):
        games = list(game_database.get_games_between_dates(period_dates[-1], period_dates[-1] - self.too_old_timedelta))
        weights = self.compute_weights(games, period_dates[-1])
        ratings, old_ratings = [], []
        for id in range(player_database.get_max_id() + 1):
            player = player_database[id]
            if player is None:
                ratings.append(0)
                old_ratings.append(0)
            else:
                ratings.append(player.get_rating().default_rating.rating)
                old_ratings.append(player.get_rating().default_rating.rating)
        
        while True:
            player_averages = [(0, 0) for _ in range(player_database.get_max_id() + 1)]
            for game, weight in zip(games, weights):
                result = game.get_result()

                player_averages[game.home] = (player_averages[game.home][0] + weight * (ratings[game.out] - self.home_advantage + (result - 0.5) * self.default_change), 
                                            player_averages[game.home][1] + weight)
                player_averages[game.out] = (player_averages[game.out][0] + weight * (ratings[game.home] + self.home_advantage - (result - 0.5) * self.default_change), 
                                            player_averages[game.out][1] + weight)
            for i in range(len(player_averages)):
                ratings[i] *= (1 - self.weighted_average)
                if player_averages[i][1] > 0:
                    ratings[i] += self.weighted_average * player_averages[i][0] / player_averages[i][1]
                else:
                    ratings[i] += self.weighted_average * ratings[i]
            if np.max(np.abs(np.array(ratings) - np.array(old_ratings))) < self.epsilon:
                break
            old_ratings = ratings[:]

        for player in player_database:
            player.get_rating().update(ratings[player.id])

    def compute_expected_score_rating(self, rating : float, opponent_rating : float, is_home : bool = False) -> float:
        if is_home:
            return 1 / (1 + 10 ** (self.delta_expected * (opponent_rating.rating - rating.rating - 2 * self.home_advantage) / self.default_change))
        return 1 / (1 + 10 ** (self.delta_expected * (opponent_rating.rating - rating.rating   + 2 * self.home_advantage) / self.default_change))
