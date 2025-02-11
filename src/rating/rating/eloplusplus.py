import numpy as np

from datetime import datetime
from typing import List

from ..objects import Player, Game, Rating
from ..databases import GameDatabase, PlayerDatabase
from .rating_system import RatingSystem


class EloPlusPlus(RatingSystem):
    def __init__(self, gamma=0.2, lambda_=0.77, iterations=50,
                 factor=0.602, default_change=200 * np.log(10), lr_change_coef=0.1, 
                 linearized : int = None):
        """
        EloPlusPlus algorithm based on https://arxiv.org/abs/1012.4571, the winner of the Kaggle chess rating competition.

        Attributes:
            - gamma (float): The extra value associated to playing home for the EloPlusPlus algorithm.
            - lambda_ (float): The regularization parameter for the EloPlusPlus algorithm.
            - iterations (int): The number of iterations to run the SGD for.
            - factor (float): The factor that is used in updating the learning rate.
            - default_change (float): The default standard unit of change in ratings.
            - lr_change_coef (float): The learning rate change coefficient used in updated the learning rate.
            - linearized (int): The number of periods to linearize the ratings for. If None, the ratings are not linearized.

        Args:
            - gamma (float): The extra value associated to playing home for the EloPlusPlus algorithm. Default is 0.2.
            - lambda_ (float): The regularization parameter for the EloPlusPlus algorithm. Default is 0.77.
            - iterations (int): The number of iterations to run the SGD for. Default is 50.
            - factor (float): The factor that is used in updating the learning rate. Default is 0.602.
            - default_change (float): The default standard unit of change in ratings. Default is 200 * np.log(10).
            - lr_change_coef (float): The learning rate change coefficient used in updated the learning rate. Default is 0.1.
            - linearized (int): The number of periods to linearize the ratings for. If None, the ratings are not linearized. Default is None.
        """
        super().__init__(gamma=gamma, lambda_=lambda_, iterations=iterations,
                         factor=factor, default_change=default_change, lr_change_coef=lr_change_coef, 
                         linearized=linearized)
        
    def compute_weights(self, games : List[Game], period_dates : List[datetime]):
        """
        Computes the weights of the games.

        Args:
            - games (list): A list of game objects.
            - period_dates (list): A list of dates for each period.
        
        Returns:
            - list: A list of weights for the games.
        """
        periods = [np.argmax([date >= game.get_date() for date in period_dates]) for game in games]
        min_period = min(periods)
        max_period = max(periods)
        weights = []
        for game, period in zip(games, periods):
            weight = (1 + period - min_period) / (1 + max_period - min_period)
            weights.append(weight ** 2)
        return weights
        
    def period_update(self, player_database : PlayerDatabase, 
                      game_database : GameDatabase, 
                      period_dates : List[datetime], **kwargs):
        """
        Updates the ratings of the players for a specific period.

        Args:
            - player_database (PlayerDatabase): The player database object.
            - game_database (GameDatabase): The game database object.
            - period_dates (list): The list of dates for each period.
        """
        period_dates_last = None
        if len(period_dates) > 1 and self.linearized is not None:
            period_dates_last = period_dates[max(0, len(period_dates) - self.linearized - 1)]
        games = list(game_database.get_games_between_dates(period_dates[-1], period_dates_last))
        weights = self.compute_weights(games, period_dates)
        ratings = []
        for id in range(player_database.get_max_id() + 1):
            player = player_database[id]
            if player is None:
                ratings.append(0)
            else:
                ratings.append(player.get_rating().rating)

        for iteration in range(self.iterations):
            player_averages = [(0, 0, 0) for _ in range(player_database.get_max_id() + 1)]
            # shuffle games
            indices = np.random.permutation(len(games))
            games = [games[i] for i in indices]
            weights = [weights[i] for i in indices]
            for game, weight in zip(games, weights):
                player_averages[game.home] = (player_averages[game.home][0] + game.weight * weight * ratings[game.out], 
                                               player_averages[game.home][1] + game.weight * weight, 
                                               player_averages[game.home][2] + game.weight)
                player_averages[game.out] = (player_averages[game.out][0] + game.weight * weight * ratings[game.home], 
                                               player_averages[game.out][1] + game.weight * weight,
                                               player_averages[game.out][2] + game.weight)
            for i in range(len(player_averages)):
                if player_averages[i][2] > 0:
                    player_averages[i] = (player_averages[i][0] / player_averages[i][1], player_averages[i][2])
                else:
                    player_averages[i] = (0, 0)

            lr = (1 + self.lr_change_coef * self.iterations) / (1 + iteration + self.lr_change_coef * self.iterations)
            lr = lr ** self.factor

            for game, weight in zip(games, weights):
                expected_outcome = self.compute_expected_score_float_rating(ratings[game.home], ratings[game.out])
                ratings[game.home] -= lr * weight * (expected_outcome - game.get_result()) * expected_outcome * (1 - expected_outcome) * self.default_change
                ratings[game.out] += lr * weight * (expected_outcome - game.get_result()) * expected_outcome * (1 - expected_outcome) * self.default_change
                ratings[game.home] -= lr * self.lambda_ / player_averages[game.home][1] * (ratings[game.home] - player_averages[game.home][0])
                ratings[game.out] -= lr * self.lambda_ / player_averages[game.out][1] * (ratings[game.out] - player_averages[game.out][0])
        
        for player in player_database:
            player.get_rating().update(ratings[player.id])

    def compute_expected_score_float_rating(self, rating1 : float, rating2 : float, rating_1_is_home : bool = True) -> float:
        """
        Computes the expected score of a player against an opponent.

        Args:
            - rating1 (float): The rating of the first player.
            - rating2 (float): The rating of the second player.
            - rating_1_is_home (bool, optional): If True, the first player is home. If False, the second player is home. Defaults to True.

        Returns:
            - float: The expected score of the first player against the second player.
        """
        if rating_1_is_home:
            return 1 / (1 + np.exp((rating2 - rating1) / self.default_change - self.gamma))
        else:
            return 1 / (1 + np.exp((rating2 - rating1) / self.default_change + self.gamma))

    def compute_expected_score_rating(self, rating1 : Rating, rating2 : Rating, rating_1_is_home : bool = True) -> float:
        """
        Computes the expected score of a player against an opponent.

        Args:
            - rating1 (Rating): The rating of the first player.
            - rating2 (Rating): The rating of the second player.
            - rating_1_is_home (bool, optional): If True, the first player is home. If False, the second player is home. Defaults to True.

        Returns:
            - float: The expected score of the first player against the second player.
        """
        return self.compute_expected_score_float_rating(rating1.rating, rating2.rating, rating_1_is_home)

    def compute_expected_score(self, player : Player, 
                               games : List[Game], 
                               player_database : PlayerDatabase,
                               date : datetime, 
                               next : bool = False) -> float:
        opponents = [game.home if game.home != player.id else game.out for game in games]
        opponents = [player_database[opponent] for opponent in opponents]
        is_home = [game.home == player.id for game in games]
        ratings = [opponent.get_rating_at_date(date, next=next) for opponent in opponents]
        player_rating = player.get_rating_at_date(date, next=next)
        expected_scores = []
        for i, rating in enumerate(ratings):
            expected_scores.append(games[i].weight * self.compute_expected_score_rating(player_rating, rating, is_home[i]))
        return np.sum(expected_scores)