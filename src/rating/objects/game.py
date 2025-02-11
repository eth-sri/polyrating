import re
from datetime import datetime
from typing import Dict
import numpy as np

from .object import Object
from .advantage import Advantage

class Game(Object):
    def __init__(self, home : int, out : int, result : str, date : datetime, tournament_id : int = None, 
                 id : int = None, points_for_win : float = 1, points_for_tie : float = 0.5, 
                 points_for_loss : float = 0, advantages_home : Dict[str, float] = None, 
                 advantages_out : Dict[str, float] = None, shared_advantages : Dict[str, float] = None,
                 add_home_advantage : bool = False, forfeit_keep_points : bool = True, max_point_difference : float = 1.0,
                 weight : int = 1) -> 'Game':
        """
        Represents a chess game between two players.

        Attributes:
            - home (int): The rating of the home player.
            - out (int): The rating of the out player.
            - result (str): The result of the game. Can be 1/2-1/2, x-y, xF-y, x-yF, xF-yF where x,y are any floating numbers.
            - id (int, optional): The ID of the game. Defaults to None.
            - date (str, datetime): The date of the game in the format "dd/mm/YYYY".
            - tournament_id (int, optional): The ID of the tournament. Defaults to None (no tournament).
            - points_for_win (float, optional): The number of points awarded for a win. Defaults to 1.
            - points_for_tie (float, optional): The number of points awarded for a tie. Defaults to 0.5.
            - points_for_loss (float, optional): The number of points awarded for a loss. Defaults to 0.
            - advantages_home (dict, optional): The advantages of the home player. Maps the advantage name to a float indicating how present the advantage is. Defaults to an empty dictionary.
            - advantages_out (dict, optional): The advantages of the out player. Maps the advantage name to a float indicating how present the advantage is. Defaults to an empty dictionary.
            - forfeit_keep_points (bool, optional): If True, points associated with each player are counted as the given points in case of a forfeit. This allows for custom match results that do not fit the normal points system.
            - max_point_difference (float, optional): The maximum point difference between the two players. When complex result is used, this is the point difference at which a result is considered completely won/lost. Defaults to 1.0.
            - weight (int, optional): The weight of the game. If the game occurred multiple times, then you can set the weight to the number of times the game occurred. Defaults to 1.
        """
        if advantages_home is None:
            advantages_home = dict()
        if advantages_out is None:
            advantages_out = dict()

        if shared_advantages is not None:
            for key, value in shared_advantages.items():
                advantages_home[key] = value
                advantages_out[key] = value

        if add_home_advantage:
            advantages_home[Advantage.HOME_ADVANTAGE] = 1
            advantages_out[Advantage.HOME_ADVANTAGE] = -1

        result_without_F = re.sub(r"F", "", result)
        home_score_str = result_without_F.split("-")[0]
        self.home_score = float(home_score_str) if home_score_str != "1/2" else 0.5
        out_score_str = result_without_F.split("-")[1]
        self.out_score = float(out_score_str) if out_score_str != "1/2" else 0.5
        super().__init__(id, home=home, out=out, result=result, 
                         date=date, tournament_id=tournament_id, 
                         points_for_win=points_for_win,
                         points_for_tie=points_for_tie,
                         points_for_loss=points_for_loss, 
                         advantages_home=advantages_home,
                         advantages_out=advantages_out,
                         forfeit_keep_points=forfeit_keep_points, max_point_difference=max_point_difference, weight=weight)

        self.is_forfeit = "F" in result
        
    def generate_settings(self) -> dict:
        """
        Generates settings for the Game object.

        Returns:
            - dict: The generated settings.
        """
        settings = super().generate_settings()
        settings['date'] = self.date.strftime("%d/%m/%Y - %H:%M:%S")
        return settings
    
    @classmethod
    def load_from_settings(cls, settings : dict) -> 'Game':
        """
        Loads a Game object from settings.

        Args:
            - settings (dict): The settings to load from.

        Returns:
            - Game: The loaded Game object.
        """
        settings = super().get_input_parameters(settings)
        settings['date'] = datetime.strptime(settings['date'], "%d/%m/%Y - %H:%M:%S")
        return cls(**settings)
    
    def get_advantages(self, home : bool) -> Dict[str, float]:
        """
        Get the advantages of a player.

        Args:
            - home (bool): True if the player is the home player, False if the player is the out player.

        Returns:
            - dict: The advantages of the player.
        """
        return self.advantages_home if home else self.advantages_out

    def get_date(self) -> datetime:
        """
        Get the date of the game.

        Returns:
            - datetime.datetime: The date of the game.
        """
        return self.date

    def get_winner(self) -> int:
        """
        Returns the winner of the game.
        If the game has no winner, there is no winner and None is returned.

        Returns:
            - str or None: The winner of the game or None if there is no winner.
        """
        if self.home_score > self.out_score:
            return self.home
        elif self.home_score < self.out_score:
            return self.out
        else:
            return None
    
    def get_result(self, complex_result : bool = False) -> float:
        """
        Get the result of the game.

        Args:
            - complex_result (bool, optional): If True, returns a float between 0 and 1 indicating how much the game was won by. Defaults to False.

        Returns:
            - float: The result of the game. Returns 1 if the result is "1(F)-0",
                0 if the result is "0-1(F)", and 0.5 for draw result. Is None if the result is 0F-0F.
        """
        if complex_result:
            res = np.clip(self.home_score - self.out_score, -self.max_point_difference, self.max_point_difference)
            return (res + self.max_point_difference) / (2 * self.max_point_difference)
        if self.home_score > self.out_score:
            return 1
        elif self.out_score > self.home_score:
            return 0
        elif self.home_score == self.out_score and not self.is_forfeit:
            return 0.5
        else:
            return None
        
    def get_points(self, home : bool) -> float:
        """
        Get the number of points awarded to a player based on the game result.

        Args:
            - home (bool): True if the player is the home player, False if the player is the out player.

        Returns:
            - float: The number of points awarded to the player.
        """
        result = self.get_result()
        if self.is_forfeit and self.forfeit_keep_points:
            return self.home_score if home else self.out_score
        if result == 1:
            return self.points_for_win if home else self.points_for_loss
        elif result == 0 or result is None:
            return self.points_for_loss if home else self.points_for_win
        elif result == 0.5:
            return self.points_for_tie
    
    def merge(self, game : 'Game') -> 'Game':
        """
        Merge two Game objects.

        Args:
            - game (Game): The game to merge with.
        """
        assert self == game and self.id != game.id
        self.weight += game.weight

    def __str__(self) -> str:
        """
        Returns a string representation of the game.

        The string includes the names of the home and out players,
        as well as the result of the game.

        Returns:
            - str: A string representation of the game.
        """
        return f"{self.home} vs {self.out} - {self.result} ({self.date.strftime('%d/%m/%Y')})"
    
    def __eq__(self, __value: object) -> bool:
        """
        Check if two Game objects are equal.

        Args:
            - __value (object): The object to compare with.

        Returns:
            - bool: True if the objects are equal, False otherwise.
        """
        if isinstance(__value, Game):
            for key in self.__dict__:
                # if key is id or kwargs, skip
                if key == "id" or key == "kwargs" or key == "weight":
                    continue
                if self.__dict__[key] != __value.__dict__[key]:
                    return False
            return True
        return False
