from ..base import BaseClass

class Object(BaseClass):
    id_counter = 0
    def __init_subclass__(cls, **kwargs):
        """
        This method is called automatically when a subclass is created.
        It initializes a separate id_counter for each subclass.
        """
        super().__init_subclass__(**kwargs)
        cls.id_counter = 0  # Initialize a separate id_counter for each subclass

    def __init__(self, id : int = None, **kwargs) -> 'Object':
        """
        A base class representing an object.

        This class provides a unique ID for each instance and a string representation of the object.

        Attributes:
            - id (int): The ID of the instance.

        Args:
            - id (int, optional): The ID of the instance. If not provided, a unique ID will be assigned.

        """
        self.id = id
        if id is None:
            self.id = self.__class__.id_counter
            self.__class__.id_counter += 1
        else:
            self.__class__.id_counter = max(self.__class__.id_counter, id + 1)
        
        super().__init__(id=self.id, **kwargs)
    
    def __str__(self) -> str:
        """
        Returns a string representation of the object.
        
        The string representation includes the class name and the object's id.
        
        Returns:
            - str: The string representation of the object.
        """
        return f"{self.__class__.__name__} {self.id}"
