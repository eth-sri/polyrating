from typing import List, Generator

from .object import Object
from .rating import Rating
from .game import Game
from .rating_history import RatingHistory

class Player(RatingHistory, Object):
    def __init__(self, name : str, id : int = None, rating : Rating = None, 
                 rating_history : List[Rating] = None, other_info=dict()) -> 'Player':
        """
        Represents a chess player.

        Attributes:
            - name (str): The name of the player.
            - id (int): The ID of the player.
            - rating (Rating): The rating of the player.
            - rating_history (list): The list of rating history for the player.
            - other_info (dict): A dictionary containing additional information about the player.

        Args:
            - name (str): The name of the player.
            - id (int, optional): The ID of the player. Defaults to None.
            - rating (Rating, optional): The rating of the player.
            - rating_history (list, optional): The list of rating history for the player. Each element is a tuple consisting of the rating at that time and the date. Defaults to an empty list.
            - other_info (dict, optional): A dictionary containing additional information about the player. Defaults to an empty dictionary.
        """
        if rating is None:
            rating = Rating()
        if rating_history is None:
            rating_history = []

        # call super RatingHistory and Object
        RatingHistory.__init__(self, rating=rating, rating_history=rating_history)
        Object.__init__(self, id=id, name=name, other_info=other_info)
        self.add_kwargs(rating=rating, rating_history=rating_history)
        
    def get_info(self) -> dict:
        """
        Returns the information about the player.

        Returns:
            - dict: The information about the player.
        """
        return {
            'name': self.name,
            'id': self.id,
            'rating': self.rating,
            'rating_history': self.rating_history,
            **self.other_info
        }

    def set_name(self, name : str):
        """
        Sets the name of the player.

        Args:
            - name (str): The name of the player.
        """
        self.name = name

    def get_number_of_wins(self, games : Generator[Game, None, None]) -> int:
        """
        Returns the number of wins for the player.

        Args:
            - games (iterator): An iterator over games played by the player.

        Returns:
            - int: The number of wins for the player.
        """
        wins = 0
        for game in games:
            if game.get_winner() == self.id:
                wins += game.weight
        return wins
    
    def get_number_of_losses(self, games : Generator[Game, None, None]) -> int:
        """
        Returns the number of losses for the player.

        Args:
            - games (iterator): An iterator over games played by the player.

        Returns:
            - int: The number of losses for the player.
        """
        losses = 0
        for game in games:
            if game.get_winner() != self.id and game.get_winner() is not None:
                losses += game.weight
        return losses
    
    def get_number_of_draws(self, games : Generator[Game, None, None]) -> int:
        """
        Returns the number of draws in a list of games.

        Args:
            - games (iterator): An iterator over games played by the player.

        Returns:
            - int: The number of draws in the list of games.
        """
        draws = 0
        for game in games:
            if game.get_winner() is None:
                draws += game.weight
        return draws
    
    def __str__(self) -> str:
        """
        Returns a string representation of the player.

        Returns:
            - str: The string representation of the player.

        """
        return f"{self.name} ({self.rating})"
    
    def __eq__(self, __value: object) -> bool:
        """
        Check if the current player is equal to another player.

        Args:
            - __value (object): The object to compare with.

        Returns:
            - bool: True if the players are equal, False otherwise.
        """
        if isinstance(__value, Player):
            return self.name == __value.name
        return False
