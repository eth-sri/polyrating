from datetime import datetime
from loguru import logger
from typing import List, Generator
import numpy as np

from .object import Object
from .rating import Rating
from .player import Player
from ..databases import GameDatabase, PlayerDatabase
from ..rating import RatingSystem

class Tournament(Object):
    def __init__(self, name: str, date: datetime, rounds: int = None, time_control: str = None, 
                 id: int = None, results: dict = None, byes: List = None, 
                 performance_diff : float = 400, tie_breaks=None, tie_break_names=None) -> 'Tournament':
        """
        Represents a chess tournament.

        Attributes:
            - name (str): The name of the tournament.
            - date (datetime): The date of the tournament.
            - rounds (int): The number of rounds in the tournament.
            - time_control (str): The time control used in the tournament.
            - id (str): The ID of the tournament (optional).
            - results (dict): A list of results for each player (optional).
            - byes (list): A list of tuples representing players who received a bye in a specific round (optional).
            - performance_diff (float): The difference in performance between players used to compute tournament performance.

        Args:
            - name (str): The name of the tournament.
            - date (datetime): The date of the tournament in the format "dd/mm/YYYY".
            - rounds (int): The number of rounds in the tournament.
            - time_control (str): The time control used in the tournament.
            - id (int, optional): The ID of the tournament. Defaults to None.
            - results (dict, optional): The results of the tournament. Defaults to an empty dictionary.
            - byes (List, optional): The list of players who received a bye in the tournament. Defaults to an empty list.
            - performance_diff (float, optional): The difference in performance between players used to compute tournament performance. Defaults to 400.
        """
        if results is None:
            results = dict()
        if byes is None:
            byes = []
        
        if tie_breaks is None:
            tie_breaks = dict()
        self.set_tie_breaks(tie_breaks, tie_break_names)
        super().__init__(id, name=name, date=date, rounds=rounds, 
                         time_control=time_control, results=results, 
                         byes=byes, performance_diff=performance_diff, tie_breaks=tie_breaks, 
                         tie_break_names=self.tie_break_names)
        self.results = {int(key): value for key, value in results.items()} if results else None

    def generate_settings(self) -> dict:
        settings = super().generate_settings()
        settings['date'] = self.date.strftime("%d/%m/%Y - %H:%M:%S")
        return settings
    
    def set_tie_breaks(self, tie_breaks : dict, tie_break_names : List[str] = None):
        self.tie_breaks = tie_breaks
        if tie_break_names is not None:
            self.tie_break_names = tie_break_names
        else:
            self.tie_break_names = set()
            for player in tie_breaks:
                self.tie_break_names = self.tie_break_names.union(set(tie_breaks[player].keys()))
            self.tie_break_names = list(self.tie_break_names)
    
    @classmethod
    def load_from_settings(cls, settings : dict) -> 'Tournament':
        settings = super().get_input_parameters(settings)
        settings['date'] = datetime.strptime(settings['date'], "%d/%m/%Y - %H:%M:%S")
        return cls(**settings)

    def get_date(self) -> datetime:
        """
        Get the date of the tournament.

        Returns:
            - datetime: The date of the tournament.
        """
        return self.date

    def add_bye(self, player : int, round : int):
        """
        Add a bye for a player in a specific round.

        Args:
            - player (int): The ID of the player who received the bye.
            - round (int): The round number in which the bye is received.
        """
        self.byes.append((player, round))

    def get_string_date(self) -> str:
        """
        Get the date of the tournament as a string in the format "dd/mm/YYYY".

        Returns:
            - str: The date of the tournament as a string.
        """
        return self.date
    
    def compute_tournament_results(self, game_database : GameDatabase, 
                                   player_database : PlayerDatabase, 
                                   rating_system : RatingSystem):
        """
        Compute the results for the tournament.

        Args:
            - game_database (GameDatabase): The game database containing the game objects.
            - player_database (PlayerDatabase): The player database containing the player objects.
            - rating_system (RatingSystem): The rating system used to compute the results.
        """
        logger.debug(f"Computing results for tournament {self.name}...")
        results = self.init_results(game_database)

        # Basic statistics
        for player in results:
            results[player]['was_bye'] = sum([int(player == bye[0]) for bye in self.byes])
            results[player]['score'] = sum(results[player]['scores']) + results[player]['was_bye']
            results[player]['player'] = player_database[player]
            results[player]['n_games'] = len(results[player]['scores'])
            for tie_break in self.tie_break_names:
                results[player][tie_break] = self.tie_breaks.get(player, dict()).get(tie_break, 0)
        
        # More advanced statistics
        for player in results:
            performance = self.compute_performance(results, player)
            if len(performance) == 0:
                results[player]['performance'] = 0
                results[player]['rating_performance'] = Rating(0)
            else:
                results[player]['performance'] = sum(performance) / len(performance)
                results[player]['rating_performance'] = rating_system.compute_tournament_performance(results[player]['player'], 
                                                                                                 self.id, self.get_date(),
                                                                                                 game_database, 
                                                                                                 player_database, 
                                                                                                 next=False)
            games = []
            for forfeit, game in zip(results[player]['forfeit'], results[player]['games']):
                if not forfeit:
                    games.append(game)

            if len(games) == 0:
                results[player]['expected_score'] = 0
            else:
                expected_score = rating_system.compute_expected_score(results[player]['player'], games, player_database,
                                                                    self.get_date(), next=True)
                results[player]['expected_score'] = expected_score + results[player]['was_bye']

    
        for player_id in results:
            results[player_id]['tournament'] = self.id
            results[player_id]['player'] = results[player_id]['player'].id
            del results[player_id]['scores'], results[player_id]['opponents']

        self.results = results

    def init_results(self, game_database : GameDatabase) -> dict:
        """
        Initializes and returns a dictionary containing the results of the tournament.

        Args:
            - game_database: An instance of the game database containing the games played in the tournament.

        Returns:
            - results: A dictionary containing the results of the tournament. The keys are player names, and the values are dictionaries with two keys:
                - 'scores': A list of game results for the player.
                - 'opponents': A list of opponent names for the player.

        """
        results = dict()
        for game in game_database.get_games_per_tournament(self.id, allow_forfeit=True):
            if game.home not in results:
                results[game.home] = {'scores': [], 'opponents': [], 'games': [], 'forfeit': []}
            if game.out not in results:
                results[game.out] = {'scores': [], 'opponents': [], 'games': [], 'forfeit': []}

            home_points, out_points = game.get_points(home=True), game.get_points(home=False)
            results[game.out]['opponents'].append(game.home)
            results[game.out]['scores'].append(out_points)
            results[game.out]['games'].append(game)
            results[game.out]['forfeit'].append(game.is_forfeit)
            results[game.home]['opponents'].append(game.out)
            results[game.home]['scores'].append(home_points)
            results[game.home]['games'].append(game)
            results[game.home]['forfeit'].append(game.is_forfeit)
        return results

    def compute_performance(self, results : dict, player : Player) -> List[float]:
        """
        Compute the performance of a player based on the results of their games using the old-fashioned system.
        For each game, the performance is computed as the rating of the opponent plus or minus the default deviation (depending on result).

        Args:
            - results (dict): A dictionary containing the results of all players' games.
            - player (Player): The name of the player for whom to compute the performance.

        Returns:
            - list: A list of performance values for each game played by the player.

        """
        performance = []
        for game in range(len(results[player]['scores'])):
            if results[player]['forfeit'][game]:
                continue
            opponent = results[results[player]['opponents'][game]]['player']
            opponent_rating = opponent.get_rating_at_date(self.get_date(), next=True)
            if results[player]['scores'][game] == 1:
                performance.append(opponent_rating.rating + self.performance_diff)
            elif results[player]['scores'][game] == 0:
                performance.append(opponent_rating.rating - self.performance_diff)
            else:
                performance.append(opponent_rating.rating)
        return performance

    def get_results(self, game_database : GameDatabase = None, 
                    player_database : PlayerDatabase = None, 
                    rating_system : RatingSystem = None) -> Generator[dict, None, None]:
        """
        Compute the results for the tournament.

        Args:
            - game_database (GameDatabase): The game database containing the game objects.
            - player_database (PlayerDatabase): The player database containing the player objects.

        Returns:
            - iterator: An iterator of dictionaries representing the results for each player in the tournament.
        """
        if not self.result_is_computed():
            self.compute_tournament_results(game_database, player_database, rating_system)
        
        for player_id in self.results:
            yield self.results[player_id]
    
    def get_player_performance(self, player_id : int, 
                               game_database : GameDatabase = None, 
                               player_database : PlayerDatabase = None, 
                               rating_system : RatingSystem = None) -> dict:
        """
        Get the results for a specific player in the tournament.

        Args:
            - player_id (str): The ID of the player.
            - game_database (GameDatabase): The game database containing the game objects.
            - player_database (PlayerDatabase): The player database containing the player objects.

        Returns:
            - dict: A dictionary representing the results for the player in the tournament.
        """
        if not self.result_is_computed():
            self.compute_tournament_results(game_database, player_database, rating_system)
        
        return self.results.get(player_id, None)
    
    def get_players(self, player_database : PlayerDatabase,
                    game_database : GameDatabase = None,
                    rating_system : RatingSystem = None) -> Generator[Player, None, None]:
        """
        Get the players in the tournament.

        Args:
            - player_database (PlayerDatabase): The player database containing the player objects.
            - game_database (GameDatabase): The game database containing the game objects.
            - rating_system (RatingSystem): The rating system used to compute the results.
        
        Returns:
            - iterator: An iterator of players in the tournament.
        """
        if not self.result_is_computed():
            self.compute_tournament_results(game_database, player_database, rating_system)
        
        for player_id in self.results:
            yield player_database[player_id]
    
    def result_is_computed(self) -> bool:
        """
        Check if the results for the tournament have been computed.

        Returns:
            - bool: True if the results have been computed, False otherwise.
        """
        return self.results is not None

    def __str__(self) -> str:
        """
        Get a string representation of the tournament.

        Returns:
            - str: A string representation of the tournament.
        """
        return f"{self.name} - {self.get_string_date()}"

    def __eq__(self, __value: object) -> bool:
        """
        Check if the tournament is equal to another tournament.

        Args:
            - __value (object): The object to compare with.

        Returns:
            - bool: True if the tournaments are equal, False otherwise.
        """
        if isinstance(__value, Tournament):
            if self.name != __value.name:
                return False
            if self.date != __value.date:
                return False
            if self.rounds != __value.rounds:
                return False
            if self.time_control != __value.time_control:
                return False
            return True
        return False
