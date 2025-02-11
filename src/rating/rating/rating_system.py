from datetime import datetime
from typing import List

from ..base import BaseClass
from ..objects import Player, Rating, Game
from ..databases import GameDatabase, PlayerDatabase


class RatingSystem(BaseClass):
    def __init__(self, **kwargs) -> 'RatingSystem':
        """
        Represents a rating system for calculating chess ratings.

        This class provides methods for updating player ratings for a specific period,
        computing the expected score of a player against an opponent, and computing
        the performance of a player in a tournament.

        Args:
            - **kwargs: Additional keyword arguments that can be passed to the base class.
        """
        super().__init__(**kwargs)

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
        raise NotImplementedError

    def compute_expected_score(self, player : Player, 
                               games : List[Game], 
                               player_database : PlayerDatabase,
                               date : datetime, 
                               next : bool = False) -> float:
        """
        Computes the expected score of a player against an opponent.

        Args:
            - player (Player): The player object.
            - games (list): A list of games played by the player.
            - player_database (PlayerDatabase): The player database object.
            - date (datetime): The date of the game.
            - next (bool): If True, the expected score is computed for the next game. If False, the expected score is computed for the current game.

        Returns:
            - float: The expected score of the player against the opponent.
        """
        expected_score = 0
        opponents = [game.home if game.home != player.id else game.black for game in games]
        is_home = [game.home == player.id for game in games]
        opponents = [player_database[opponent] for opponent in opponents]
        for game, opponent, is_whit in zip(games, opponents, is_home):
            player_rating = player.get_rating_at_date(date, next=next)
            opponent_rating = opponent.get_rating_at_date(date, next=next)
            expected_score += game.weight * self.compute_expected_score_rating(player_rating, opponent_rating, is_whit)
        return expected_score
    
    def compute_expected_score_rating(self, player_rating : Rating, 
                                      opponent_rating : Rating, 
                                      is_home : bool) -> float:
        """
        Computes the expected score of a player against an opponent.

        Args:
            - player_rating (Rating): The rating of the player.
            - opponent_rating (Rating): The rating of the opponent.
            - is_home (bool): If True, the player is home, otherwise the player is black.

        Returns:
            - float: The expected score of the player against the opponent.
        """
        raise NotImplementedError

    def compute_tournament_performance(self, player : Player, 
                                       tournament_id : int, 
                                       tournament_date : datetime, 
                                       game_database : GameDatabase, 
                                       player_database : PlayerDatabase, 
                                       next : bool = False, rating_check : int = 10000) -> Rating:
        """
        Computes the performance of a player in a tournament.

        Args:
            - player (Player): The player object.
            - tournament_id (int): The id of the tournament object.
            - tournament_date (datetime): The date of the tournament.
            - game_database (GameDatabase): The game database object.
            - player_database (PlayerDatabase): The player database object.
            - next (bool): If True, ratings for after this period are used for computing the tournament performance. 
            - rating_check (int): The maximum rating difference between the minimum and maximum rating.
       Returns:
            - Rating: The performance of the player in the tournament.
        """
        opponents = []
        is_home = []
        actual_score = 0
        for game in game_database.get_games_per_tournament(tournament_id):
            if player.id in [game.home, game.black]:
                is_home.append(game.home == player.id)
                if game.home == player.id:
                    opponents.append(player_database[game.black])
                    actual_score += game.get_result()
                else:
                    opponents.append(player_database[game.home])
                    actual_score += 1 - game.get_result()

        min_rating, max_rating = -rating_check, rating_check
        while max_rating - min_rating > 0.0001:
            rating = Rating((max_rating + min_rating) / 2)
            expected_score = 0
            for is_whit, opponent in zip(is_home, opponents):
                opponent_rating = opponent.get_rating_at_date(tournament_date, next=next)
                expected_score += self.compute_expected_score_rating(rating, opponent_rating, is_whit)
            if expected_score <= actual_score:
                min_rating = rating.rating
            else:
                max_rating = rating.rating
        
        return Rating((max_rating + min_rating) / 2)