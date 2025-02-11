import os

from loguru import logger

from ..base import BaseClass
from ..databases import GameDatabase, PlayerDatabase, TournamentDatabase
from .statistics import *


class StatManager(BaseClass):
    def __init__(self) -> 'StatManager':
        """
        The StatManager class is responsible for managing the computation of statistics for players, games, and tournaments.
        """
        super().__init__()

    def start_round(self, base_folder : str, history_subfolder : str):
        """
        Starts a new round of computing statistics.

        Args:
            - base_folder (str): The base folder where the subfolders are located.
            - history_subfolder (str): The name of the history subfolder.
        """
        logger.info("Starting new round of computing statistics...")
        os.makedirs(os.path.join(base_folder, history_subfolder), exist_ok=True)

    def compute_statistics(self, player_database : PlayerDatabase, 
                           game_database : GameDatabase, 
                           tournament_database : TournamentDatabase, 
                            rating_system : RatingSystem,
                           base_folder : str, current_subfolder : str,
                           tournament_subfolder : str):
        """
        Compute statistics for players, games, and tournaments.

        Args:
            - player_database (PlayerDatabase): The player database.
            - game_database (GameDatabase): The game database.
            - tournament_database (TournamentDatabase): The tournament database.
            - rating_system (RatingSystem): The rating system used to compute the ratings.
            - base_folder (str): The base folder path.
            - current_subfolder (str): The current subfolder path.
            - tournament_subfolder (str): The tournament subfolder path.
        """
        # get every class subclassed from Statistic, not including the class TournamentStatistic
        normal_stats = [cls() for cls in Statistic.__subclasses__() if cls != TournamentStatistic]
        tournament_stats = [cls() for cls in TournamentStatistic.__subclasses__()]

        # compute normal stats
        for stat in normal_stats:
            stat.compute(player_database, game_database, tournament_database, rating_system,
                         os.path.join(base_folder, current_subfolder))
        
        # compute tournament stats
        tournament = tournament_database.get_last()
        tournament_format = tournament.get_date().strftime("%Y_%m_%d") + '_' + tournament.name
        tournament_folder = os.path.join(base_folder, tournament_subfolder, tournament_format)
        os.makedirs(tournament_folder, exist_ok=True)
        for stat in tournament_stats:
            stat.compute(player_database, game_database, tournament, rating_system, tournament_folder)
    
    def run(self, player_database : PlayerDatabase, game_database : GameDatabase, 
            tournament_database : TournamentDatabase, rating_system : RatingSystem,
            base_folder : str, history_subfolder : str, tournament_subfolder : str, date_folder : str):
        """
        Runs the statistics manager.

        Args:
            - player_database (str): Path to the player database.
            - game_database (str): Path to the game database.
            - tournament_database (str): Path to the tournament database.
            - rating_system (RatingSystem): The rating system used to compute the ratings.
            - base_folder (str): Path to the base folder.
            - history_subfolder (str): Path to the history subfolder.
            - tournament_subfolder (str): Path to the tournament subfolder.
            - date_folder (str): Path to the date folder.
        """
        self.start_round(base_folder, os.path.join(history_subfolder, date_folder))
        self.compute_statistics(player_database, game_database, tournament_database, rating_system,
                               base_folder, os.path.join(history_subfolder, date_folder), tournament_subfolder)

        
