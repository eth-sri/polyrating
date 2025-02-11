from datetime import datetime
from typing import List, Generator

from ..base import BaseClass


class RatingPeriod(BaseClass):
    def __init__(self, rating_periods : List[datetime] = None, **kwargs) -> 'RatingPeriod':
        """
        Represents a rating period for chess ratings.

        Attributes:
            - rating_periods (list): List of datetime objects representing the rating periods.

        Args:
            - rating_periods (list, optional): List of datetime objects representing the rating periods.
            - **kwargs: Additional keyword arguments to be passed to the base class constructor.
        """
        if rating_periods is None:
            rating_periods = []
        super().__init__(rating_periods=rating_periods, **kwargs)

    def trigger_new_period(self, datetime : datetime = datetime.now()) -> bool:
        """
        Triggers a new rating period.

        Args:
            - datetime (datetime, optional): The datetime object representing the new rating period.
        """
        if len(self.rating_periods) == 0 or datetime > self.rating_periods[-1]:
            self.rating_periods.append(datetime)
        elif datetime not in self.rating_periods:
            self.rating_periods.append(datetime)
            self.rating_periods.sort()
        
    def generate_settings(self) -> dict:
        """
        Generates settings for the RatingPeriod object.

        Returns:
            - dict: The generated settings.
        """
        settings = super().generate_settings()
        settings['rating_periods'] = [period.strftime("%Y/%m/%d - %H:%M:%S") for period in self.rating_periods]
        return settings
    
    @classmethod
    def load_from_settings(cls, settings : dict) -> 'RatingPeriod':
        """
        Loads a RatingPeriod object from settings.

        Args:
            - settings (dict): The settings to load from.

        Returns:
            - RatingPeriod: The loaded RatingPeriod object.
        """
        settings = super().get_input_parameters(settings)
        settings['rating_periods'] = [datetime.strptime(period, "%Y/%m/%d - %H:%M:%S") for period in settings['rating_periods']]
        return cls(**settings)
        
    def __len__(self) -> int:
        """
        Returns the number of rating periods.

        Returns:
            - int: The number of rating periods.
        """
        return len(self.rating_periods)
    
    def __getitem__(self, index : int) -> datetime:
        """
        Returns the rating period at the specified index.

        Args:
            - index (int): The index of the rating period.

        Returns:
            - datetime: The rating period at the specified index.
        """
        return self.rating_periods[index]
    
    def __iter__(self) -> Generator[datetime, None, None]:
        """
        Returns an iterator for the rating periods.

        Returns:
            - iterator: An iterator for the rating periods.
        """
        return iter(self.rating_periods)

    def get_last_period(self) -> datetime:
        """
        Returns the last rating period.

        Returns:
            - datetime: The last rating period.
        """
        return self.rating_periods[-1]

    def n_new_rating_periods(self, last_date : datetime = None) -> int:
        """
        Returns the number of new rating periods.

        Args:
            - last_date (datetime, optional): The last date to compare against.

        Returns:
            - int: The number of new rating periods.
        """
        if last_date is None:
            return len(self.rating_periods)
        return len([p for p in self.rating_periods if p > last_date])
    
    def get_period_of_date(self, date : datetime, next : bool = True) -> datetime:
        """
        Returns the rating period of a given date.

        Args:
            - date (datetime): The date to find the rating period for.
            - next (bool, optional): If True, returns the rating period after the given date. If False, returns the rating period before the given date.

        Returns:
            - datetime: The rating period of the given date.
        """
        for i in range(len(self.rating_periods)):
            if date < self.rating_periods[i]:
                if i > 0:
                    return self.rating_periods[i] if next else self.rating_periods[i-1]
                return self.rating_periods[i]
        return self.rating_periods[-1]
    
    def iterate_periods(self, last_date : datetime = None) -> Generator[datetime, None, None]:
        """
        Iterates over the rating periods.

        Args:
            - last_date (datetime, optional): The last date to start iterating from.

        Yields:
            - list: A list of rating periods.
        """
        for i in range(len(self.rating_periods)):
            if last_date is None or self.rating_periods[i] > last_date:
                yield self.rating_periods[:i+1]

# create an enum for the rating period
class RatingPeriodEnum:
    TOURNAMENT = 0 # rating period is determined by tournaments: a new rating period is triggered when a new tournament is created
    TIMEDELTA = 1 # rating period is determined by a timedelta: a new rating period is triggered after a certain amount of time has passed
    MANUAL = 2 # rating period is determined manually: a new rating period is triggered manually