import re
from typing import List

from ..base import BaseClass


class Matching(BaseClass):
    def __init__(self, **kwargs):
            """
            Initializes the Matching object. This object is used to match a dictionary with a set of conditions.
            This superclass matches any dictionary.

            Args:
                **kwargs: Additional keyword arguments.

            Returns:
                None
            """
            super().__init__(**kwargs)

    def match(self, dict_ : dict) -> bool:
        """
        Matches the object with the given dictionary.

        Args:
            - dict_ (dict): The dictionary to match with.

        Returns:
            - bool: True if the dictionary matches the conditions given by the object, False otherwise.
        """
        return True
    
    def __str__(self) -> str:
        return 'True'


class Equal(Matching):
    def __init__(self, attribute : str, should_equal : str):
        """
        Initializes the EqualMatching object. This object matches a dictionary if the value of the
        given attribute is equal to the given value.

        Args:
            attribute (str): The attribute to match.
            should_equal (Union(str, float)): The value that the attribute should be equal to.

        Returns:
            None
        """
        super().__init__(attribute=attribute, should_equal=should_equal)

    def match(self, dict_ : dict) -> bool:
        return dict_.get(self.attribute) == self.should_equal
    
    def __str__(self):
        return f"{self.attribute} == {self.should_equal}"
    

class GreaterThan(Matching):
    def __init__(self, attribute : str, should_be_greater_than : float, equal : bool = False):
        """
        Initializes the GreaterThanMatching object. This object matches a dictionary if the value of the
        given attribute is greater than the given value.

        Args:
            attribute (str): The attribute to match against.
            should_be_greater_than (float): The value that the attribute should be greater than.
            equal (bool, optional): Whether the attribute can be equal to the given value. Defaults to False.
        """
        super().__init__(attribute=attribute, should_be_greater_than=should_be_greater_than, equal=equal)
    def __init__(self, attribute, should_be_greater_than, equal=False):
        super().__init__(attribute=attribute, should_be_greater_than=should_be_greater_than, 
                         equal=equal)

    def match(self, dict_ : dict) -> bool:
        if self.equal:
            return dict_.get(self.attribute) >= self.should_be_greater_than
        return dict_.get(self.attribute) > self.should_be_greater_than
    
    def __str__(self):
        if self.equal:
            return f"({self.attribute} >= {self.should_be_greater_than})"
        return f"{self.attribute} > {self.should_be_greater_than}"
    

class LessThan(Matching):
    def __init__(self, attribute: str, should_be_less_than: float, equal: bool = False):
        """
        Initializes the LessThanMatching object. This object matches a dictionary if the value of the
        given attribute is less than the given value. 

        Args:
            attribute (str): The attribute to match against.
            should_be_less_than (float): The value that the attribute should be less than.
            equal (bool, optional): Whether the attribute can be equal to the given value. Defaults to False.
        """
        super().__init__(attribute=attribute, should_be_less_than=should_be_less_than, equal=equal)
    def __init__(self, attribute : str, should_be_less_than : float, equal : bool = False):
        super().__init__(attribute=attribute, should_be_less_than=should_be_less_than, 
                         equal=equal)

    def match(self, dict_ : dict) -> bool:
        if self.equal:
            return dict_.get(self.attribute) <= self.should_be_less_than
        return dict_.get(self.attribute) < self.should_be_less_than
    
    def __str__(self):
        if self.equal:
            return f"({self.attribute} <= {self.should_be_less_than})"
        return f"({self.attribute} < {self.should_be_less_than})"
    

class IsIn(Matching):
    def __init__(self, attribute : str, should_be_in : list):
        """
        Initializes the InMatching object. This object matches a dictionary if the value of the
        given attribute is in the given list.

        Args:
            attribute (str): The attribute to match.
            should_be_in (list): The list of values that the attribute should be in.
        """
        super().__init__(attribute=attribute, should_be_in=should_be_in)

    def match(self, dict_ : dict) -> bool:
        return dict_.get(self.attribute) in self.should_be_in
    
    def __str__(self):
        return f"({self.attribute} in {self.should_be_in})"
    

class Regex(Matching):
    def __init__(self, attribute : str, regex : str):
        """
        Initializes the RegexMatching object. This object matches a dictionary if the value of the
        given attribute matches the given regular expression pattern.

        Args:
            attribute (str): The attribute to match against.
            regex (str): The regular expression pattern to match.

        Returns:
            None
        """
        super().__init__(attribute=attribute, regex=regex)

    def match(self, dict_ : dict) -> bool:
        return re.match(self.regex, dict_.get(self.attribute)) is not None
    
    def __str__(self):
        return f"({self.attribute} matches {self.regex})"


class Or(Matching):
    def __init__(self, matchings : List[Matching]):
        """
        Initializes the Or object. This object matches a dictionary if any of the given matchings match.

        Args:
            matchings (list): A list of matchings.

        Returns:
            None
        """
        super().__init__(matchings=matchings)

    def match(self, dict_ : dict) -> bool:
        return any(matching.match(dict_) for matching in self.matchings)
    
    def __str__(self):
        return "(" + " or ".join(str(matching) for matching in self.matchings) + ")"
    

class And(Matching):
    def __init__(self, matchings : List[Matching]):
        """
        Initializes the And object. This object matches a dictionary if all of the given matchings match.

        Args:
            matchings (list): A list of matchings.

        Returns:
            None
        """
        super().__init__(matchings=matchings)

    def match(self, dict_ : dict) -> bool:
        return all(matching.match(dict_) for matching in self.matchings)
    
    def __str__(self):
        return "(" + " and ".join(str(matching) for matching in self.matchings) + ")"


class Not(Matching):
    def __init__(self, matching : Matching):
        """
        Initializes the Not object. This object matches a dictionary if the given matching does not match.

        Args:
            matching (Matching): The matching value for the object.

        Returns:
            None
        """
        super().__init__(matching=matching)

    def match(self, dict_ : dict) -> bool:
        return not self.matching.match(dict_)
    
    def __str__(self):
        return f"not {self.matching}"
