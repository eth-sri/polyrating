from loguru import logger

from ..objects import Player
from .database import Database
from .game_database import GameDatabase

class PlayerDatabase(Database):
    def __init__(self, objects : dict = None, strict : bool = True) -> 'PlayerDatabase':
        """
        Represents a database of chess players.

        Attributes:
            - objects (dict): A dictionary containing player objects.
            - strict (bool): A flag indicating whether to use exact name comparisons when searching for players.

        Args:
            - objects (dict, optional): A dict object containing players. Defaults to an empty dict.
        """
        super().__init__(objects, strict=strict)
        self.objects_by_name = {player.name: player for player in self.objects.values()}
    
    def search_database(self, name : str) -> Player:
        """
        Searches the player database for a player with the given name. If strict is False, the search is case-insensitive 
        and also matches if only the first letter of the last name is given for a player.

        Args:
            - name (str): The name of the player to search for.

        Returns:
            - Player or None: The player object if found, None otherwise.
        """
        if self.strict:
            return self.objects_by_name.get(name, None)

        # Check for exact matches
        for player in self.objects.values():
            if player.name.lower() == name.lower():
                return player
        
        # allow last name and first name to be swapped
        for player in self.objects.values():
            if ' '.join(player.name.split(' ')[::-1]).lower() == name.lower():
                # change the player name to have first name first (likely smallest of the two)
                return player

        # allow for only the first letter of the last name to be given
        for player in self.objects.values():
            returned_player = self.check_condition(player, name)
            if returned_player is not None:
                return returned_player
        
        # allow for only the first letter of the last name to be given, swapped
        for player in self.objects.values():
            name_here = ' '.join(name.split(' ')[::-1])
            returned_player = self.check_condition(player, name_here)
            if returned_player is not None:
                return returned_player
        return None
    
    def check_condition(self, player : Player, name : str) -> Player:
        """
        Checks the condition for matching player names. This condition allows for a name
        match if only the first letter of the last name is given for a player.

        Args:
            - player (Player): The player object to compare.
            - name (str): The name to compare with the player's name.

        Returns:
            - Player or None: If the condition is met, returns the player object. Otherwise, returns None.
        """
        if ' ' not in player.name or ' ' not in name: # check if both have last names
            return None
        one_has_only_one_letter = (len(player.name.split(' ')[1]) == 1 or len(name.split(' ')[1]) == 1)
        if not one_has_only_one_letter:
            return None
        condition_2 = player.name.split(' ')[1][0].lower() == name.split(' ')[1][0].lower()
        if player.name.split(' ')[0].lower() == name.split(' ')[0].lower() and condition_2:
            # change the player name to include the full last name
            logger.debug(f"Found players with likely the same name: {player.name} to {name}")
            if len(player.name) < len(name):
                player.set_name(name)
            return player
        return None
    
    def get_player_by_name(self, name : str) -> Player:
        """
        Retrieves a player from the database based on their name.

        Args:
            - name (str): The name of the player to search for.

        Returns:
            - Player: The player object matching the given name, or None if no player is found.
        """
        return self.search_database(name)

    def clear_empty(self, game_database : GameDatabase):
        """
        Removes players from the database who have no recorded games.

        Args:
            - game_database (GameDatabase): The game database to check for player games.
        """
        self.objects = {id: player for id, player in self.objects.items() 
                        if game_database.get_n_games_per_player(player.id, allow_forfeit=True) > 0}
        self.objects_by_name = {player.name: player for player in self.objects.values()}
        
    def add(self, player : Player):
        """
        Adds a player to the database.

        Args:
            - player (Player): The player to add to the database.
        """
        if not self.search_database(player.name):
            self.objects[player.id] = player
            self.objects_by_name[player.name] = player

    def remove(self, player : Player):
        """
        Removes a player from the database.

        Args:
            - player (Player): The player to remove from the database.
        """
        if player.id in self.objects:
            del self.objects[player.id]
            del self.objects_by_name[player.name]