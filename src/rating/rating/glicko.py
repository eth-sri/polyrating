import numpy as np

from datetime import datetime
from typing import List

from ..objects import Rating, Player, Game
from ..objects.rating import DefaultRating
from ..databases import PlayerDatabase, GameDatabase
from .rating_system import RatingSystem

class Glicko(RatingSystem):
    def __init__(self, default_change : float = 400, C : float = 50, 
                 max_exp_value=10, iterations=1, **kwargs) -> 'Glicko':
        """
        Implementation of the Glicko rating system.

        The Glicko rating system is a method for calculating the ratings of players in games such as chess. It takes into account the ratings of opponents and the outcomes of games to update a player's rating.

        Attributes:
            - default_change (float): The default rating change value.
            - C (float): The constant value used in the rating update formula.
            - max_exp_value (int): The maximum value for the exponent in the expected score formula. Prevents overflow errors.

        Args:
            - default_change (float, optional): The default change value. Defaults to 400.
            - C (float, optional): The C value. Defaults to 50.
            - max_exp_value (int, optional): The maximum value for the exponent in the expected score formula. Prevents overflow errors. Defaults to 10.
        """
        super().__init__(default_change=default_change, C=C, max_exp_value=max_exp_value, iterations=iterations, **kwargs)

    def period_update(self, player_database : PlayerDatabase, 
                      game_database : GameDatabase, 
                      period_dates : List[datetime], **kwargs):
        period_dates = period_dates[:]
        period_dates.insert(0, None)
        for iteration in range(self.iterations):
            updated_ratings = []
            player_scores_and_opponents = dict()
            for game in game_database.get_games_between_dates(period_dates[-1], period_dates[-2]):
                for _ in range(game.weight): # TODO: this can be done more efficiently for weight > 1
                    player_scores_and_opponents[game.home] = player_scores_and_opponents.get(game.home, []) + [(game.get_result(), game.out)]
                    player_scores_and_opponents[game.out] = player_scores_and_opponents.get(game.out, []) + [(1 - game.get_result(), game.home)]
            for player in player_database:
                scores_and_opponents = player_scores_and_opponents.get(player.id, [])
                opponents = [player_database[opponent] for score, opponent in scores_and_opponents]
                scores = [score for score, opponent in scores_and_opponents]
                if len(opponents) == 0 and player.get_rating().deviation >= player.get_rating().default_rating.deviation:
                    continue
                new_rating = self.update_player(player, opponents, scores)
                updated_ratings.append((player, new_rating))
            for player, new_rating in updated_ratings:
                player.get_rating().update(new_rating[0], new_rating[1], new_rating[2])

    def update_player(self, player : Player, opponents : List[Player], scores : List[float]) -> tuple:
        """
        Updates the rating of a player based on the opponents and scores.

        Args:
            - player (Player): The player to update the rating for.
            - opponents (list): The list of opponents.
            - scores (list): The list of scores of the player against the opponents.

        Returns:
            - tuple: A tuple containing the updated rating, deviation, and volatility of the player.
        """
        new_deviation = min(np.sqrt(player.get_rating().deviation ** 2 + self.C ** 2), player.get_rating().default_rating.deviation)
        rating_update = 0
        player_rating = player.get_rating()
        for opponent, score in zip(opponents, scores):
            opponent_rating = opponent.get_rating()
            E = self.E(player_rating.rating, opponent_rating.rating, opponent_rating.deviation)
            g = self.g(opponent_rating.deviation)
            rating_update += g * (score - E)
        
        if len(opponents) > 0:
            d_2 = 1 / self.d_squared(player.get_rating().rating, opponents)
        else:
            d_2 = 0
        denom = (1 / new_deviation ** 2 + d_2)
        update = self.q / denom * rating_update
        new_rating = player.get_rating().rating + update
        new_deviation = np.sqrt(1 / denom)
        return (new_rating, new_deviation, player.get_rating().default_rating.volatility)
    
    @property
    def q(self) -> float:
        return np.log(10) / self.default_change

    def d_squared(self, rating : float, opponents : List[Player]) -> float:
        """
        Computes the squared denominator value used in the rating update formula.

        Args:
            - rating (float): The rating of the player.
            - opponents (list): The list of opponents.

        Returns:
            - float: The squared denominator value.
        """
        expected_d = 0
        for opponent in opponents:
            opponent_rating = opponent.get_rating()
            E = self.E(rating, opponent_rating.rating, opponent_rating.deviation)
            g = self.g(opponent_rating.deviation)
            expected_d += g ** 2 * E * (1 - E)
        denominator = (self.q) ** 2 * expected_d
        return 1 / denominator
    
    def g(self, deviation : float) -> float:
        """
        Computes the g value used in the rating update formula.

        Args:
            - deviation (float): The deviation value.

        Returns:
            - float: The g value.
        """
        return 1 / np.sqrt(1 + 3 * ((self.q) ** 2 * deviation ** 2) / (np.pi ** 2))
    
    def E(self, rating : float, opponent_rating : float, opponent_deviation : float) -> float:
        """
        Computes the expected score of a player against an opponent.

        Args:
            - rating (float): The rating of the player.
            - opponent_rating (float): The rating of the opponent.
            - opponent_deviation (float): The deviation of the opponent.

        Returns:
            - float: The expected score of the player against the opponent.
        """
        exponent = -self.g(opponent_deviation) * (rating - opponent_rating) / self.default_change
        exponent = np.clip(exponent, -self.max_exp_value, self.max_exp_value)
        return 1 / (1 + 10 ** (exponent))

    def convert_rating(self, rating : float, deviation : float, volatility : float, 
                       default_rating : DefaultRating) -> tuple:
        """
        Converts a rating to the appropriate format.

        Args:
            - rating (float): The rating to be converted.
            - deviation (float): The deviation to be converted.
            - volatility (float): The volatility to be converted.
            - default_rating (DefaultRating): The default rating value.

        Returns:
            - tuple: A tuple containing the converted rating, deviation, and volatility.
        """
        return (rating, deviation, volatility)
    
    def compute_expected_score(self, player : Player, games : List[Game], player_database : PlayerDatabase, date : datetime, 
                               next : bool = False) -> float:
        opponents = [game.home if game.home != player.id else game.out for game in games]
        opponents = [player_database[opponent] for opponent in opponents]
        ratings = [opponent.get_rating_at_date(date, next=next) for opponent in opponents]
        player_rating = player.get_rating_at_date(date, next=next)
        ratings = [self.convert_rating(rating.rating, rating.deviation, rating.volatility, rating.default_rating) for rating in ratings]
        converted_rating = self.convert_rating(player_rating.rating, player_rating.deviation, player_rating.volatility, player_rating.default_rating)
        expected_score = sum([game.weight * self.E(converted_rating[0], rating[0], rating[1]) for game, rating in zip(games, ratings)])
        return expected_score
    
    def compute_expected_score_rating(self, player_rating: Rating, opponent_rating: Rating, is_home: bool) -> float:
        player_rating = self.convert_rating(player_rating.rating, player_rating.deviation, player_rating.volatility, player_rating.default_rating)
        opponent_rating = self.convert_rating(opponent_rating.rating, opponent_rating.deviation, opponent_rating.volatility, opponent_rating.default_rating)
        return self.E(player_rating[0], opponent_rating[0], opponent_rating[1])