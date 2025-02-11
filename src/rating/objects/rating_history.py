from datetime import datetime
from typing import List

from ..base import BaseClass
from .rating import Rating


class RatingHistory(BaseClass):
    def __init__(self, rating : Rating = None, rating_history : List[Rating] = None) -> 'RatingHistory':
        """
        Represents a rating history.

        Attributes:
            - rating (Rating): The rating of the player.
            - rating_history (list): The list of rating history for the player.

        Args:
            - rating (Rating, optional): The rating of the player.
            - rating_history (list, optional): The list of rating history for the player. Each element is a tuple consisting of the rating at that time and the date. Defaults to an empty list.
        """
        if rating is None:
            rating = Rating()
        if rating_history is None:
            rating_history = []
        super().__init__(rating=rating, rating_history=rating_history)
        
    def generate_settings(self) -> dict:
        """
        Generates settings for the RatingHistory object.

        Returns:
            - dict: The generated settings.
        """
        settings = super().generate_settings()
        settings['rating_history'] = [(rating.generate_settings(), date.strftime("%d/%m/%Y - %H:%M:%S")) for rating, date in self.rating_history]
        return settings
    
    @classmethod
    def load_from_settings(cls, settings : dict) -> 'RatingHistory':
        """
        Loads a RatingHistory object from settings.

        Args:
            - settings (dict): The settings to load from.

        Returns:
            - RatingHistory: The loaded RatingHistory object.
        """
        settings = super().get_input_parameters(settings)
        settings['rating_history'] = [(rating, datetime.strptime(date, "%d/%m/%Y - %H:%M:%S")) for rating, date in settings['rating_history']]
        return cls(**settings)

    def get_rating(self) -> Rating:
        """
        Returns the current rating of the RatingHistory.

        Returns:
            - float: The current rating of the RatingHistory.
        """
        return self.rating
    
    def clear_rating_history(self):
        """
        Clears the rating history of the RatingHistory.
        """
        self.rating_history = []

    def get_rating_history(self) -> List[Rating]:
        """
        Returns the rating history of the RatingHistory.

        Returns:
            - list: the rating history.
        """
        return self.rating_history
    
    def set_rating_history(self, rating_history : List[Rating]):
        """
        Sets the rating history of the RatingHistory.

        Args:
            - rating_history (list): A list of rating values representing the RatingHistory's rating history.
        """
        self.rating_history = rating_history

    def store_rating(self, date : datetime):
        """
        Stores the current rating, deviation, volatility, and date in the rating history.

        Args:
            - date (str): The date when the rating was recorded.
        """
        if len(self.rating_history) == 0 or not self.rating.equal_rating_and_advantages(self.rating_history[-1][0]):
            self.rating_history.append((self.rating.copy(), date))

    def get_rating_at_date(self, date : datetime, next : bool = False) -> Rating:
        """
        Returns the rating of the rating history at a specific date.

        Args:
            - date (datetime): The date to get the rating for.
            - next (bool, optional): If True, returns the rating at the next date, otherwise the previous date. Defaults to False.

        Returns:
            - float: The rating of the rating history at the specified date. If no rating is found for the date, returns the current rating.
        """
        for i, rating in enumerate(self.rating_history):
            if rating[-1] >= date:
                if next:
                    return rating[0]
                elif i == 0:
                    copied_rating = rating[0].copy()
                    copied_rating.reset()
                    return copied_rating
                else:
                    return self.rating_history[i-1][0]
        return self.rating
    
    def rating_boost(self) -> List[Rating]:
        """
        Calculate the maximum rating boost for the rating history. 
        I.e. the difference between the current rating and the previous rating, divided by the deviation.

        Returns:
            - list: A list containing the maximum rating boost, the current rating, the previous rating, and the date of the rating.
        """
        max_boost = 0
        rating_info = None
        for i, rating in enumerate(self.rating_history[1:]):
            rating_diff = rating[0].rating - self.rating_history[i][0].rating
            rating_boost = rating_diff / self.rating_history[i][0].deviation
            if rating_boost > max_boost:
                max_boost = rating_boost
                rating_info = [rating_boost, rating[0], self.rating_history[i][0], rating[1]]
        return rating_info
