from datetime import datetime
from typing import List

from ..objects import Player, Game, Rating
from ..databases import PlayerDatabase, GameDatabase
from .rating_system import RatingSystem

class Elo(RatingSystem):
    def __init__(self, K : float = 64, change : float = 400, iterations=1, **kwargs) -> 'Elo':
        """
        The Elo rating system implementation.

        This class represents the Elo rating system used to calculate the ratings of chess players.
        It inherits from the `RatingSystem` class.

        Attributes:
            - K (float): The K-factor used in the Elo rating calculation.
            - change (float): The change parameter used in the Elo rating calculation.

        Args:
            - K (float, optional): The K-factor used in Elo rating calculations. Defaults to 64.
            - change (float, optional): The change in rating used in the expected score formula. Defaults to 400.
        """
        super().__init__(**kwargs, K=K, change=change, iterations=iterations)

    def period_update(self, player_database : PlayerDatabase, 
                      game_database : GameDatabase, 
                      period_dates : List[datetime], **kwargs):
        period_dates = period_dates[:]
        period_dates.insert(0, None)
        for iteration in range(self.iterations):
            differences_per_player = {}
            for game in game_database.get_games_between_dates(period_dates[-1], period_dates[-2]):
                home = player_database[game.home]
                out = player_database[game.out]
                if home.id not in differences_per_player:
                    differences_per_player[home.id] = []
                if out.id not in differences_per_player:
                    differences_per_player[out.id] = []
                
                home_rating = home.get_rating()
                out_rating = out.get_rating()
                expected_home = self.compute_expected_score_rating(home_rating, out_rating)
                expected_out = self.compute_expected_score_rating(out_rating, home_rating)
                differences_per_player[home.id].append(game.weight * self.K * (game.get_result() - expected_home))
                differences_per_player[out.id].append(game.weight * self.K * ((1 - game.get_result()) - expected_out))

            for player_id in differences_per_player:
                player = player_database[player_id]
                new_rating = player.get_rating().rating + sum(differences_per_player[player_id])
                player.get_rating().update(new_rating)

    def compute_expected_score(self, player : Player, 
                               games : List[Game], player_database : PlayerDatabase, date : datetime,
                               next : bool = False) -> float:
        expected_score = 0
        opponents = [game.home if game.home != player.id else game.out for game in games]
        opponents = [player_database[opponent] for opponent in opponents]
        player_rating = player.get_rating_at_date(date, next=next)
        for i, opponent in enumerate(opponents):
            opponent_rating = opponent.get_rating_at_date(date, next=next)
            expected_score += games[i].weight * self.compute_expected_score_rating(player_rating, opponent_rating)
        return expected_score

    def compute_expected_score_rating(self, rating : Rating, opponent_rating : Rating, is_home : bool = False) -> float:
        return 1 / (1 + 10 ** ((opponent_rating.rating - rating.rating) / self.change))