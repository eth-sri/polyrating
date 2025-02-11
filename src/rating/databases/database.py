import numpy as np
from typing import Generator

from ..base import BaseClass
from ..objects import Object

class Database(BaseClass):
    def __init__(self, objects : dict = None, **kwargs) -> 'Database':
        """
        A class representing a database.

        This class provides methods to add, remove, and retrieve objects from the database.

        Attributes:
            - objects (dict): A dictionary containing objects. The keys are the IDs of the objects.

        Args:
            - objects (dict, optional): A dict object containing the initial objects. Defaults to an empty dict.
        """
        if objects is None:
            objects = dict()
        self.objects : dict = objects
        # convert the keys to ints
        self.objects = {int(key): value for key, value in self.objects.items()}
        super().__init__(objects=self.objects, **kwargs)
    
    def add(self, obj : Object):
        """
        Adds an object to the collection.

        Args:
            - obj: The object to be added. Must have an ID attribute.
        """
        self.objects[obj.id] = obj
    
    def remove(self, obj : Object):
        """
        Removes an object from the collection.

        Args:
            - obj: The object to be removed. Must have an ID attribute.

        Returns:
            - The removed object, or None if the object was not found.
        """
        return self.objects.pop(obj.id, None)
    
    def check_duplicate(self, obj : Object) -> bool:
        """
        Checks if an object already exists in the database.

        Args:
            - obj: The object to check for duplicates.

        Returns:
            - bool: True if a duplicate object is found, False otherwise.
        """
        for existing_obj in self.objects.values():
            if existing_obj == obj:
                return True
        return False
    
    def __len__(self) -> int:
        """
        Returns the number of objects in the container.
        
        Returns:
            - int: The number of objects in the container.
        """
        return len(self.objects)
    
    def __getitem__(self, id_ : int) -> Object:
        """
        Retrieve an object from the collection by its ID.

        Args:
            - id_ (str): The ID of the object to retrieve.

        Returns:
            - object: The object with the specified ID, or None if it doesn't exist.
        """
        return self.objects.get(id_, None)
    
    def get_max_id(self) -> int:
        """
        Returns the highest ID in the database.

        Returns:
            - int: The highest ID in the database, or -1 if the database is empty.
        """
        if len(self.objects) == 0:
            return -1
        return max(self.objects.keys())
    
    def get_last(self) -> Object:
        """
        Retrieves the last object in the database.

        Returns:
            - object: The last object in the database, or None if the database is empty.
        """
        if len(self.objects) == 0:
            return None
        # get the object with the highest id
        return self.objects[max(self.objects.keys())]

    def get_random(self) -> Object:
        """
        Retrieves a random object from the database.

        Returns:
            - object: A random object from the database, or None if the database is empty.
        """
        if len(self.objects) == 0:
            return None
        # get a random object
        return self.objects[np.random.choice(list(self.objects.keys()))]
    
    def empty(self):
        """
        Empties the database.
        """
        values = list(self.objects.values())
        for obj in values:
            self.remove(obj)
    
    def __iter__(self) -> Generator[Object, None, None]:
        """
        Retrieves objects from the database.

        Returns:
            - generator: A generator that yields objects from the database.
        """
        objects = list(self.objects.values())
        for obj in objects:
            yield obj