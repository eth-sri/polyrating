from typing import Dict, List

from ..base import BaseClass

class DefaultRating(BaseClass):
    all_instances = []
    current_id = 0

    @staticmethod
    def create_or_get_default(rating : float = 1500, deviation : float = 500, volatility : float = 0.09, 
                              id : int = None) -> 'DefaultRating':
        """
        Creates a new DefaultRating object or returns an existing one with the same rating, deviation, and volatility.

        Args:
            - rating (float): The default rating value.
            - deviation (float): The default deviation value.
            - volatility (float): The default volatility value.
            - id (int): The ID of the DefaultRating object.

        Returns:
            - DefaultRating: The DefaultRating object.
        """
        default = DefaultRating.check_instances(rating, deviation, volatility, id)
        if default:
            return default
        return DefaultRating(rating, deviation, volatility, id)

    def __init__(self, rating : float = 1500, deviation : float = 500, volatility : float = 0.09, id : int = None):
        """
        Initialize a Rating object.

        Args:
            rating (float): The initial rating value. Defaults to 1500.
            deviation (float): The initial deviation value. Defaults to 500.
            volatility (float): The initial volatility value. Defaults to 0.09.
            id (int): The ID of the rating object. Defaults to None.
        """
        self.rating = rating
        self.deviation = deviation
        self.volatility = volatility
        if id is None:
            id = DefaultRating.current_id            
        DefaultRating.current_id = max(DefaultRating.current_id, id + 1)
        self.id = id
        super().__init__(rating=rating, deviation=deviation, volatility=volatility, id=id)
        self.all_instances.append(self)

    @classmethod
    def load_from_settings(cls, settings : dict) -> 'DefaultRating':
        """
        Loads a DefaultRating object from settings.

        Args:
            - settings (dict): The settings to load from.

        Returns:
            - DefaultRating: The loaded DefaultRating object.
        """
        settings = super().get_input_parameters(settings)
        return DefaultRating.create_or_get_default(**settings)

    @staticmethod
    def check_instances(rating, deviation, volatility, id):
        """
        Check if there is an instance in the list of DefaultRating objects that matches the given rating, deviation, volatility, and id.

        Args:
            - rating (float): The rating value to match.
            - deviation (float): The deviation value to match.
            - volatility (float): The volatility value to match.
            - id (str): The id value to match.

        Returns:
            - DefaultRating or None: The matching instance if found, otherwise None.
        """
        for instance in DefaultRating.all_instances:
            if instance.rating == rating and instance.deviation == deviation and instance.volatility == volatility and instance.id == id:
                return instance
        return None

    def set_default(self, rating : float = None, deviation : float = None, volatility : float = None):
        """
        Sets the default values for the rating, deviation, and volatility.

        Args:
            - rating (float): The default rating value.
            - deviation (float): The default deviation value.
            - volatility (float): The default volatility value.
        """
        instance = DefaultRating.check_instances(rating, deviation, volatility, self.id)
        if instance and instance != self:
            instance.set_default(rating, deviation, volatility)
        if rating:
            self.rating = rating
        if deviation:
            self.deviation = deviation
        if volatility:
            self.volatility = volatility

DEFAULT_RATING = DefaultRating(1500, 500, 0.09)

class BaseRating(BaseClass):
    def __init__(self, rating : float = None, deviation : float = None, volatility : float = None, 
                 default_rating : DefaultRating = None) -> 'Rating':
        """
        Represents a rating object for a player in a chess game.

        Attributes:
            - rating (float): The rating of the player.
            - deviation (float): The deviation of the player's rating.
            - volatility (float): The volatility of the player's rating.
            - default_rating (DefaultRating): The default rating values.

        Args:
            - rating (float): The initial rating value. If not provided, the default rating will be used.
            - deviation (float): The initial deviation value. If not provided, the default deviation will be used.
            - volatility (float): The initial volatility value. If not provided, the default volatility will be used.
            - default_rating (DefaultRating): The default rating values. If not provided, the DEFAULT_RATING will be used.
        """
        if default_rating is None:
            default_rating = DEFAULT_RATING

        if rating is None:
            rating = default_rating.rating
        if deviation is None:
            deviation = default_rating.deviation
        if volatility is None:
            volatility = default_rating.volatility
        super().__init__(rating=rating, deviation=deviation, 
                         volatility=volatility, default_rating=default_rating)

    def set_default(self, rating : float = None, deviation : float = None, volatility : float = None):
        """
        Sets the default values for the rating, deviation, and volatility.

        Args:
            - rating (float): The default rating value.
            - deviation (float): The default deviation value.
            - volatility (float): The default volatility value.
        """
        self.default_rating.set_default(rating, deviation, volatility)
        
    def copy(self) -> 'BaseRating':
        """
        Creates a copy of the Rating object.

        Returns:
            - Rating: A new Rating object with the same attribute values as the original.
        """
        return BaseRating(rating=self.rating, deviation=self.deviation, volatility=self.volatility,
                          default_rating=self.default_rating)

    def update(self, new_rating : float, new_deviation : float = None, new_volatility : float = None):
        """
        Updates the player's rating, deviation, and volatility.

        Args:
            - new_rating (float): The new rating value.
            - new_deviation (float): The new deviation value.
            - new_volatility (float): The new volatility value.
        """
        self.rating = new_rating
        if new_deviation:
            self.deviation = new_deviation
        if new_volatility:
            self.volatility = new_volatility

    def reset(self):
        """
        Resets the rating, deviation, and volatility to their default values.
        """
        self.rating = self.default_rating.rating
        self.deviation = self.default_rating.deviation
        self.volatility = self.default_rating.volatility

    def set(self, rating : 'BaseRating'):
        """
        Sets the rating, deviation, and volatility of the object.

        Args:
            - rating (Rating): The Rating object containing the new rating, deviation, and volatility.
        """
        self.rating = rating.rating
        self.deviation = min(rating.deviation, self.default_rating.deviation)
        self.volatility = rating.volatility

    def __eq__(self, o: object) -> bool:
        """
        Check if two Rating objects are equal.

        Args:
            - o (object): The object to compare with.

        Returns:
            - bool: True if the objects are equal, False otherwise.
        """
        if isinstance(o, BaseRating):
            if self.rating != o.rating:
                return False
            if self.deviation != o.deviation:
                return False
            if self.volatility != o.volatility:
                return False
            return True
        return False
    
    def __str__(self) -> str:
        """
        Returns a string representation of the Rating object.
        
        The string is formatted as "{rating} ± {deviation}".
        
        Returns:
            - str: The string representation of the Rating object.
        """
        return f"{self.rating} ± {self.deviation}"

class Rating(BaseRating):
    def __init__(self, rating: float = None, deviation: float = None, volatility: float = None,
                 advantages: Dict[str, BaseRating] = None, default_rating: DefaultRating = None):
        """
        A rating object for a player in a chess game. This object can have ratings for different advantages.

        Attributes:
            - rating (float): The rating of the player.
            - deviation (float): The deviation of the player's rating.
            - volatility (float): The volatility of the player's rating.
            - advantages (Dict[str, BaseRating]): A dictionary of advantages. The keys are the names of the advantages.

        Args:
            rating (float): The rating value.
            deviation (float): The deviation value.
            volatility (float): The volatility value.
            advantages (Dict[str, BaseRating]): A dictionary of advantages. The keys are the names of the advantages.

        """
        if advantages is None:
            advantages = dict()
        super().__init__(rating=rating, deviation=deviation, volatility=volatility, 
                         default_rating=default_rating)
        self.add_kwargs(advantages=advantages)

    def copy(self) -> 'Rating':
        """
        Create a deep copy of the Rating object.

        Returns:
            - Rating: A new Rating object with the same attribute values as the original.
        """
        advantages = {
            key: value.copy() for key, value in self.advantages.items()
        }
        return Rating(rating=self.rating, deviation=self.deviation, volatility=self.volatility, 
                      advantages=advantages, default_rating=self.default_rating)
    
    def add_advantage(self, key : str, rating : BaseRating):
        """
        Adds an advantage to the rating object.

        Args:
            - key (str): The key to identify the advantage.
            - rating (BaseRating): The rating object representing the advantage.
        """
        self.advantages[key] = rating

    def remove_advantage(self, key: str):
        """
        Removes the advantage associated with the given key.

        Args:
            - key (str): The key of the advantage to be removed.
        """
        if key in self.advantages:
            del self.advantages[key]

    def update_advantage(self, advantage : str, new_rating : float, 
                         new_deviation : float = None, new_volatility : float = None, 
                         default_rating : DefaultRating = None):
        """
        Updates the player's rating, deviation, and volatility for a specific advantage.

        Args:
            - advantage (str): The advantage to update.
            - new_rating (float): The new rating value.
            - new_deviation (float): The new deviation value.
            - new_volatility (float): The new volatility value.
            - default_rating (DefaultRating): The default rating values.
        """
        if advantage in self.advantages:
            self.advantages[advantage].update(new_rating, new_deviation, new_volatility)
        else:
            if default_rating is None:
                default_rating = DefaultRating()
            self.advantages[advantage] = BaseRating(new_rating, new_deviation, new_volatility, default_rating)

    def reset_advantage(self, key : str):
        """
        Resets the advantage associated with the given key.

        Args:
            - key (str): The key representing the advantage to be reset.
        """
        if key in self.advantages:
            self.advantages[key].reset()

    def get_advantage(self, key: str) -> BaseRating:
        """
        Retrieve the advantage associated with the given key.

        Args:
            - key (str): The key to look up the advantage.

        Returns:
            - BaseRating: The advantage associated with the given key, or None if the key is not found.
        """
        return self.advantages.get(key, None)
    
    def has_advantage(self, key : str) -> bool:
        """
        Checks if the given key is present in the advantages dictionary.

        Args:
            - key (str): The key to check for in the advantages dictionary.

        Returns:
            - bool: True if the key is present in the advantages dictionary, False otherwise.
        """
        return key in self.advantages
    
    def get_advantage_names(self) -> List[str]:
        """
        Returns a list of names of all the advantages.

        Returns:
            - List[str]: A list of advantage names.
        """
        return list(self.advantages.keys())

    def get_rating(self, advantages : Dict[str, float] = None) -> float:
        """
        Calculates the overall rating of the player, taking into account the advantages.

        Args:
            - advantages (Dict[str, float], optional): A dictionary of advantages and their corresponding weights. Defaults to None.

        Returns:
            - float: The calculated overall rating of the player.
        """
        rating = self.rating
        for key, value in self.advantages.items():
            rating += value.rating * advantages.get(key, 0)
        return rating

    def reset(self):
        """
        Resets the rating object by calling the reset method of the base class
        and resetting all the advantages.
        """
        super().reset()
        for key in self.advantages:
            self.advantages[key].reset()

    def set(self, rating : 'Rating'):
        """
        Sets the rating object with the values from another rating object.

        Args:
            - rating (Rating): The rating object to set values from.
        """
        super().set(rating)
        for key in self.advantages:
            if key in rating.advantages:
                self.advantages[key].set(rating.advantages[key])
        
    def __eq__(self, o: object) -> bool:
        """
        Check if two Rating objects are equal.

        Args:
            - o (object): The object to compare with.

        Returns:
            - bool: True if the objects are equal, False otherwise.
        """
        if isinstance(o, Rating):
            if not super().__eq__(o):
                return False
            if self.advantages != o.advantages:
                return False
            return True
        return False
    
    def equal_rating_and_advantages(self, rating : 'Rating') -> bool:
        """
        Check if the rating and advantages of two Rating objects are equal.

        Args:
            - rating (Rating): The Rating object to compare with.

        Returns:
            - bool: True if the rating and advantages are equal, False otherwise.
        """
        if self.rating != rating.rating:
            return False
        if self.deviation != rating.deviation:
            return False
        if self.volatility != rating.volatility:
            return False
        if self.advantages != rating.advantages:
            return False
        return True
    
    def __str__(self) -> str:
        """
        Returns a string representation of the Rating object.

        The string includes the superclass string representation and the advantages of the rating.

        Returns:
            - str: The string representation of the Rating object.
        """
        return f"{super().__str__()} {', '.join([f'{key}: {value}' for key, value in self.advantages.items()])}"
