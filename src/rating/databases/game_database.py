
from datetime import datetime
from sortedcontainers import SortedList
from typing import List, Generator
from tqdm import tqdm
from loguru import logger

from ..objects import Game
from .database import Database


class GameDatabase(Database):
    def __init__(self, objects : dict = None) -> 'GameDatabase':
        """
        A class representing a game database.

        Inherits from the `Database` class.

        Attributes:
            - objects (dict): A sorted dictionary containing the games.
            - games_per_player (dict): A dictionary mapping player IDs to a list of game IDs.
            - games_per_tournament (dict): A dictionary mapping tournament IDs to a list of game IDs.

        Args:
            - objects (dict): A sorted dictionary containing the games.
        """
        super().__init__(objects)
        self.games_per_player = dict()  # {player_id: [game_id, game_id, ...]}
        self.games_per_tournament = dict()  # {tournament_id: [game_id, game_id, ...]}
        self.games_by_date = SortedList([], key=lambda x: x[1])
        self.update_dicts(self.objects.values())
        self.earliest_date = None
        self.latest_date = None

    def get_latest_date(self) -> datetime:
        """
        Get the latest date of the games in the database.

        Returns:
            - datetime: The latest date of the games in the database.
        """
        if self.latest_date is None and len(self.objects) > 0:
            self.latest_date = max(game.get_date() for game in self.objects.values())
        return self.latest_date

    def get_earliest_date(self) -> datetime:
        """
        Get the earliest date of the games in the database.

        Returns:
            - datetime: The earliest date of the games in the database.
        """
        if self.earliest_date is None and len(self.objects) > 0:
            self.earliest_date = min(game.get_date() for game in self.objects.values())
        return self.earliest_date

    def update_dicts(self, games : List[Game], 
                     remove : bool = False):
        """
        Updates the dictionaries `games_per_player` and `games_per_tournament` based on the given list of games.

        Args:
            - games (list): A list of game objects.
            - remove (bool, optional): If True, removes the game IDs from the dictionaries. Defaults to False.
        """
        for game in games:
            self.update_player_dict_single_game(game, game.home, remove)
            self.update_player_dict_single_game(game, game.out, remove)

            game_id_date = (game.id, game.get_date())
            if not remove:
                # add to the list, keeping it sorted
                self.games_by_date.add(game_id_date)
            else:
                self.games_by_date.remove(game_id_date)

            if game.tournament_id is None:
                continue

            if game.tournament_id not in self.games_per_tournament and not remove:
                self.games_per_tournament[game.tournament_id] = [game.id]
            elif not remove:
                self.games_per_tournament[game.tournament_id].append(game.id)
            else:
                self.games_per_tournament[game.tournament_id].remove(game.id)


    def update_player_dict_single_game(self, game : Game, player_id : int, remove : bool):
        """
        Updates the player dictionary with a single game.

        Args:
            - game (Game): The game object to be added or removed.
            - player_id (int): The ID of the player.
            - remove (bool): Indicates whether to remove the game from the player's list.
        """
        if player_id not in self.games_per_player and not remove:
            self.games_per_player[player_id] = {game.id}
        elif not remove:
            self.games_per_player[player_id].add(game.id)
        else:
            self.games_per_player[player_id].remove(game.id)

    def get_games_no_forfeit(self) -> Generator[Game, None, None]:
        """
        Get an iterator of games without forfeits.

        Returns:
            - iterator: An iterator of games without forfeits.

        """
        games = list(self.objects.values())
        for game in games:
            if not game.is_forfeit:
                yield game

    def get_games_per_player(self, player_id : int, allow_forfeit : bool = False) -> Generator[Game, None, None]:
        """
        Get an iterator of games for a specific player.

        Args:
            - player_id (int): The ID of the player.
            - allow_forfeit (bool, optional): If True, includes forfeits. Defaults to False.

        Returns:
            - iterator: An iterator of games for the player.
        """
        games = list(self.games_per_player.get(player_id, []))
        for game in games:
            if allow_forfeit or not self[game].is_forfeit:
                yield self[game]

    def get_n_games_per_player(self, player_id : int, allow_forfeit : bool = False) -> int:
        """
        Get the number of games for a specific player.

        Args:
            - player_id (int): The ID of the player.
            - allow_forfeit (bool, optional): If True, includes forfeits. Defaults to False.

        Returns:
            - int: The number of games for the player.
        """
        n_games = 0
        for game in self.games_per_player.get(player_id, []):
            if allow_forfeit or not self[game].is_forfeit:
                n_games += 1
        return n_games

    def get_games_per_tournament(self, tournament_id : int, allow_forfeit : bool = False) -> Generator[Game, None, None]:
        """
        Get an iterator of games for a specific tournament.

        Args:
            - tournament_id (int): The ID of the tournament.
            - allow_forfeit (bool, optional): If True, includes forfeits. Defaults to False.

        Returns:
            - iterator: An iterator of games for the tournament.
        """
        games = list(self.games_per_tournament.get(tournament_id, []))
        for game in games:
            if allow_forfeit or not self[game].is_forfeit:
                yield self[game]

    def get_n_games_per_tournament(self, tournament_id : int, allow_forfeit : bool = False) -> int:
        """
        Get the number of games for a specific tournament.

        Args:
            - tournament_id (int): The ID of the tournament.
            - allow_forfeit (bool, optional): If True, includes forfeits. Defaults to False.

        Returns:
            - int: The number of games for the tournament.
        """
        n_games = 0
        for game in self.games_per_tournament.get(tournament_id, []):
            if allow_forfeit or not self[game].is_forfeit:
                n_games += 1
        return n_games
    
    def get_games_between_dates(self, date_before : datetime, date_after : datetime = None, 
                                allow_forfeit : bool = False) -> Generator[Game, None, None]:
        """
        Get an iterator of games between two dates.

        Args:
            - date_before (datetime): The last possible date.
            - date_after (datetime, optional): The earliest possible date. Defaults to None.
            - allow_forfeit (bool, optional): If True, includes forfeits. Defaults to False.

        Returns:
            - iterator: An iterator of games between the two dates.
        """
        left = self.games_by_date.bisect_right((0, date_after)) if date_after is not None else 0
        right = self.games_by_date.bisect_right((0, date_before))
        games = self.games_by_date[left:right]
        for game_id, _ in games:
            game = self.objects[game_id]
            if allow_forfeit or not game.is_forfeit:
                yield game

    def get_n_games_between_dates(self, date_before : datetime, date_after : datetime = None, 
                                  allow_forfeit : bool = False) -> int:
        """
        Get the number of games between two dates.

        Args:
            - date_before (datetime): The last possible date.
            - date_after (datetime, optional): The earliest possible date. Defaults to None.
            - allow_forfeit (bool, optional): If True, includes forfeits. Defaults to False.

        Returns:
            - int: The number of games between the two dates.
        """
        left = self.games_by_date.bisect_right((0, date_after)) if date_after is not None else 0
        right = self.games_by_date.bisect_right((0, date_before))
        n_games = 0
        for game_id, _ in self.games_by_date[left:right]:
            game = self.objects[game_id]
            if allow_forfeit or not game.is_forfeit:
                n_games += 1
        return n_games
    
    def add(self, obj : Game):
        """
        Add a game to the database.

        Args:
            - obj (Game): The game object to be added.
        """
        super().add(obj)
        self.update_dicts([obj])
        self.earliest_date = min(self.earliest_date, obj.get_date()) if self.earliest_date is not None else obj.get_date()
        self.latest_date = max(self.latest_date, obj.get_date()) if self.latest_date is not None else obj.get_date()

    def remove(self, obj : Game):
        """
        Remove a game from the database.

        Parameters:
            - obj (Game): The game object to be removed.
        """
        super().remove(obj)
        self.update_dicts([obj], remove=True)
        if obj.get_date() == self.earliest_date:
            self.earliest_date = None
        if obj.get_date() == self.latest_date:
            self.latest_date = None

    def compact(self):
        """
        Compact the database by removing duplicate games. 
        Instead, games that are duplicates are merged using the weight of the game as an indicator of how frequently it occurred.
        """
        logger.debug(f"Compacting the game database of size {len(self)}.")
        all_ids = set([game.id for game in self.objects.values()])
        # tqdm bar with extra info on the combos done
        tqdm_bar = tqdm(total=len(all_ids), desc="Compacting games", unit="games", leave=False, dynamic_ncols=True, bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}], combos done: {postfix}")
        combos_done = 0
        while len(all_ids) > 0:
            combos_done += 1
            tqdm_bar.update(1)
            game = self.objects[all_ids.pop()]
            game_ids_home = self.games_per_player[game.home]
            game_ids_out = self.games_per_player[game.out]
            intersection_games = game_ids_home.intersection(game_ids_out)
            for other_game_id in intersection_games:
                other_game = self.objects[other_game_id]
                if game == other_game and game.id != other_game_id:
                    game.merge(other_game)
                    self.remove(other_game)
                    all_ids.remove(other_game_id)
                    tqdm_bar.update(1)
            tqdm_bar.set_postfix_str(str(combos_done))
        tqdm_bar.close()
        logger.debug(f"Database compacted to {len(self)}.")
