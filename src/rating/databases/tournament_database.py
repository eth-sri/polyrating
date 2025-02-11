from typing import List

from .database import Database

class TournamentDatabase(Database):
    def __init__(self, objects : dict = None) -> 'TournamentDatabase':
        """
        A class representing a tournament database.

        This class inherits from the `Database` class and provides functionality to store and manage tournaments.

        Attributes:
            - objects (dict): A dictionary to store the tournaments.

        Args:
            - objects (dict, optional): A dict object containing tournaments. Defaults to an empty dict.
        """
        super().__init__(objects)

    def get_player_performances(self, player_id : int) -> List[dict]:
        """
        Get a list over tournament performances for a specific player.

        Args:
            - player_id (str): The ID of the player.

        Returns:
            - list: A list of tournament performances.
        """
        performances = []
        for tournament in self.objects.values():
            if tournament.result_is_computed():
                result = tournament.get_player_performance(player_id)
                if result is not None:
                    performances.append(result)
        return performances