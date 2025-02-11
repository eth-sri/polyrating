from loguru import logger
from datetime import timedelta, datetime

from .base import BaseClass
from .objects import Player, Game, Matching, DefaultRating, Tournament, Advantage, RatingPeriod, RatingPeriodEnum
from .databases import PlayerDatabase, GameDatabase, TournamentDatabase
from .rating import PolyratingCrossEntropy, RatingSystem, TrueSkillThroughTime
from .utils import extract_tournament, extract_players, extract_games
from .statistics import StatManager


class Manager(BaseClass):
    def __init__(self, rating_system : RatingSystem = None, 
                 player_database : PlayerDatabase = None, 
                 game_database : GameDatabase = None, 
                 tournament_database : TournamentDatabase = None, 
                 stat_manager : StatManager = None, 
                 rating_period : RatingPeriod = None, 
                 rating_period_type : int = RatingPeriodEnum.TOURNAMENT, 
                 custom_timedelta : timedelta = timedelta(days=7), 
                 last_date_update : datetime = None, do_recompute : bool = True, 
                 recompute : bool = False, 
                 add_home_advantage : bool = True, 
                 forfeit_keep_points : bool = True) -> 'Manager':
        """
        The Manager class is responsible for managing the player database, game database, tournament database,
        rating system, and the statistics manager.

        Attributes:
            - rating_system (RatingSystem): The rating system to use.
            - player_database (PlayerDatabase): The player database.
            - game_database (GameDatabase): The game database.
            - tournament_database (TournamentDatabase): The tournament database.
            - stat_manager (StatManager): The statistics manager.
            - rating_period (RatingPeriod): The rating period.
            - rating_period_type (RatingPeriodEnum): The rating period type.
            - custom_timedelta (timedelta): The custom time delta for rating periods.
            - last_date_update (datetime): The last date of rating update.
            - do_recompute (bool): Whether to recompute ratings if games/players/tournaments are removed/added before the last update.
            - recompute (bool): Whether to recompute ratings.
            - add_home_advantage (bool): Whether to add a home advantage to the games when added.
            - forfeit_keep_points (bool, optional): If True, points associated with each player are counted as the given points in case of a forfeit. This allows for custom match results that do not fit the normal points system.

        Args:
            - rating_system (RatingSystem, optional): The rating system to use. Defaults to PolyratingCrossEntropy().
            - player_database (PlayerDatabase, optional): The player database. Defaults to PlayerDatabase().
            - game_database (GameDatabase, optional): The game database. Defaults to GameDatabase().
            - tournament_database (TournamentDatabase, optional): The tournament database. Defaults to TournamentDatabase().
            - stat_manager (StatManager, optional): The statistics manager. Defaults to StatManager().
            - rating_period (RatingPeriod, optional): The rating period. Defaults to RatingPeriod().
            - rating_period_type (RatingPeriodEnum, optional): The rating period type. Defaults to RatingPeriodEnum.TOURNAMENT.
            - custom_timedelta (timedelta, optional): The custom time delta for rating periods. Defaults to timedelta(days=7).
            - last_date_update (datetime, optional): The last date of rating update. Defaults to None.
            - recompute (bool, optional): Whether to recompute ratings. Defaults to False.
            - do_recompute (bool): Whether to recompute ratings if games/players/tournaments are removed/added before the last update.
            - add_home_advantage (bool, optional): Whether to add a home advantage to the games when added. Defaults to True.
            - forfeit_keep_points (bool, optional): If True, points associated with each player are counted as the given points in case of a forfeit. This allows for custom match results that do not fit the normal points system.
        """
        if rating_system is None:
            rating_system = PolyratingCrossEntropy(
                linearized=10,
                epsilon=1e-2
            )
        if player_database is None:
            player_database = PlayerDatabase()
        if game_database is None:
            game_database = GameDatabase()
        if tournament_database is None:
            tournament_database = TournamentDatabase()
        if stat_manager is None:
            stat_manager = StatManager()
        if rating_period is None:
            rating_period = RatingPeriod()

        super().__init__(rating_system=rating_system, player_database=player_database,
                         game_database=game_database, tournament_database=tournament_database, 
                         stat_manager=stat_manager, rating_period=rating_period,
                         rating_period_type=rating_period_type, custom_timedelta=custom_timedelta, 
                         last_date_update=last_date_update, recompute=recompute, do_recompute=do_recompute, 
                         add_home_advantage=add_home_advantage, forfeit_keep_points=forfeit_keep_points)
        
    def generate_settings(self) -> dict:
        """
        Generate the settings dictionary for the manager.

        Returns:
            - dict: The generated settings dictionary.
        """
        settings = super().generate_settings()
        settings['custom_timedelta'] = self.custom_timedelta.total_seconds()
        settings['last_date_update'] = None
        if self.last_date_update is not None:
            settings['last_date_update'] = self.last_date_update.strftime("%Y-%m-%d - %H:%M:%S")
        return settings
    
    def clone(self) -> 'Manager':
        """
        Clone the manager.

        Returns:
            - Manager: The cloned manager.
        """
        return Manager.load_from_settings(self.generate_settings())
    
    def reset_and_recompute(self, rating_system : RatingSystem = None, rating_period_type : int = None, custom_timedelta : timedelta = None):
        """
        Reset the manager and recompute the ratings.

        Args:
            - rating_system (RatingSystem): The rating system to use. Defaults to None.
            - rating_period_type (int, optional): The rating period type. Defaults to None.
            - custom_timedelta (timedelta, optional): The custom time delta for rating periods. Defaults to None.
        """
        if rating_system is not None:
            self.rating_system = rating_system
        if custom_timedelta is not None:
            self.custom_timedelta = custom_timedelta
        if rating_period_type is not None:
            self.rating_period_type = rating_period_type
            if self.rating_period_type == RatingPeriodEnum.TOURNAMENT:
                self.rating_period = RatingPeriod()
                for tournament in self.tournament_database:
                    self.rating_period.trigger_new_period(tournament.get_date())
            elif self.rating_period_type == RatingPeriodEnum.TIMEDELTA:
                self.rating_period = RatingPeriod()
                self.rating_period.trigger_new_period(self.game_database.get_earliest_date())
        
        self.recompute = True
        self.update_rating()
    
    @classmethod
    def load_from_settings(cls, settings : dict) -> 'Manager':
        """
        Load the chess rating manager from the given settings.

        Args:
            - cls (class): The class of the chess rating manager.
            - settings (dict): The settings to load from.

        Returns:
            - Manager: An instance of the chess rating manager.

        """
        kwargs = super().get_input_parameters(settings)
        kwargs['custom_timedelta'] = timedelta(seconds=kwargs['custom_timedelta'])
        if kwargs['last_date_update'] is not None:
            kwargs['last_date_update'] = datetime.strptime(kwargs['last_date_update'], "%Y-%m-%d - %H:%M:%S")
        return cls(**kwargs)
        
    def trigger_new_period(self, tournament : Tournament = None):
        """
        Triggers a new rating period based on the specified tournament or timedelta.

        Args:
            - tournament (Tournament, optional): The tournament object representing the new period. Defaults to None.
        """
        if self.rating_period_type == RatingPeriodEnum.TOURNAMENT and tournament is not None:
            self.rating_period.trigger_new_period(tournament.get_date())
        elif self.rating_period_type == RatingPeriodEnum.TIMEDELTA:
            if self.last_date_update is None or len(self.rating_period) == 0:
                self.rating_period.trigger_new_period(self.game_database.get_earliest_date() + self.custom_timedelta)
            
            while self.rating_period.get_last_period() < self.game_database.get_latest_date():
                self.rating_period.trigger_new_period(self.rating_period.get_last_period() + self.custom_timedelta)
        else:
            self.rating_period.trigger_new_period(datetime.now())

    def compute_statistics(self, tournament : Tournament = None, 
                           data_folder : str = "data", 
                           history_folder : str = "history", 
                           tournament_folder : str = "tournaments"):
        """
        Computes tournament results and statistics.

        Args:
            - tournament (Tournament): The tournament object.
            - data_folder (str, optional): The folder where the data is stored. Defaults to "data".
            - history_folder (str, optional): The folder where the historical tournament data is stored. Defaults to "history".
            - tournament_folder (str, optional): The folder where the tournament data is stored. Defaults to "tournaments".
        """
        logger.info("Computing tournament results...")
        if tournament is not None:
            tournament.compute_tournament_results(self.game_database, self.player_database, self.rating_system)

        logger.info("Computing statistics...")
        latest_date_str = self.last_date_update.strftime("%Y_%m_%d") if self.last_date_update is not None else "no_date"
        self.stat_manager.run(self.player_database, self.game_database, self.tournament_database, self.rating_system,
                              data_folder, history_folder, tournament_folder, latest_date_str)

    def update_rating(self):
        """
        Perform one iteration of the rating update process.
        """
        logger.info("Updating ratings...")
        
        if len(self.rating_period) == 0:
            logger.info("No rating period set. Automatically triggering a new period.")
            self.trigger_new_period()
        if self.recompute:
            for player in self.player_database:
                player.clear_rating_history()
                player.get_rating().reset()
                self.last_date_update = None
            self.recompute = False
        for period_dates in self.rating_period.iterate_periods(self.last_date_update):
            logger.info(f"Updating ratings for period {period_dates[-1]}...")
            if (len(period_dates) == 1 and self.game_database.get_n_games_between_dates(period_dates[0]) == 0):
                continue
            elif len(period_dates) > 1 and self.game_database.get_n_games_between_dates(period_dates[-1], period_dates[-2]) == 0:
                continue
            self.rating_system.period_update(self.player_database, self.game_database, period_dates)
            for player in self.player_database:
                player.store_rating(period_dates[-1])
        self.last_date_update = self.rating_period[-1]

    def add_tournament(self, tournament_path : str = None, tournament_name : str = None, force : bool = False, tournament : Tournament = None) -> Tournament:
        """
        Adds a tournament to the chess rating manager. Also computes updated ratings and statistics.

        Args:
            - tournament_path (str): The path to the tournament file. Defaults to None.
            - tournament_name (str, optional): The name of the tournament. If not provided, the name will be extracted from the file.
            - force (bool, optional): If set to True, allows adding a tournament with the same name as an existing one. Defaults to False.
            - tournament (Tournament, optional): The tournament object to add. Defaults to None.
        """
        logger.info(f"Adding tournament from {tournament_path}...")
        if tournament is None:
            tournament = extract_tournament(tournament_path)
            if tournament_name is not None:
                tournament.name = tournament_name
        if tournament is None:
            raise ValueError(f"No tournament found in {tournament_path}.")
        
        if self.tournament_database.check_duplicate(tournament) and not force:
            raise ValueError(f"Tournament {tournament.name} already exists in the database. If you think this is a mistake, use the force option to add as a new tournament.")
        
        self.tournament_database.add(tournament)

        if tournament_path is not None:
            logger.info(f"Extracting players from {tournament_path}")
            players, tie_breaks, tie_break_names = extract_players(tournament_path)
            for player in players:
                existing_player = self.player_database.get_player_by_name(player.name)
                if existing_player is None:
                    self.player_database.add(player)
                else:
                    tie_breaks[existing_player.id] = tie_breaks[player.id]

            tournament.set_tie_breaks(tie_breaks, tie_break_names)

            logger.info(f"Extracting games from {tournament_path}")
            extract_games(tournament_path, tournament, self.game_database, self.player_database, 
                          add_home_advantage=self.add_home_advantage)
        self.player_database.clear_empty(self.game_database)
        self.trigger_new_period(tournament)
        was_false = not self.recompute
        self.recompute = self.last_date_update is not None and tournament.get_date() <= self.last_date_update and self.do_recompute
        if self.recompute and was_false:
            logger.warning(f"You have added a tournament from {tournament.get_date()} which is earlier than the last tournament in the rating period. Next recomputation of the ratings will need a full recompute.")
        return tournament
    
    def remove_tournament(self, tournament_name : str = None, tournament : Tournament = None):
        """
        Remove a tournament from the chess rating manager. Also removes all games associated with the tournament, and players that only have played in that tournament.

        Args:
            - tournament_name (str): The name of the tournament to remove. Defaults to None.
            - tournament (Tournament, optional): The tournament object to remove. Defaults to None.
        """
        if tournament is None:
            tournament = self.tournament_database.get_tournament_by_name(tournament_name)
        if tournament is None:
            raise ValueError(f"Tournament {tournament_name} not found in the database.")
        for game in self.game_database.get_games_per_tournament(tournament.id):
            if self.last_date_update is not None and game.get_date() < self.last_date_update and self.do_recompute:
                self.recompute = True
                logger.warning(f"You have removed a game from {game.get_date()} which is earlier than the last game in the rating period. Next recomputation of the ratings will need a full recompute.")
            self.game_database.remove(game)
        self.tournament_database.remove(tournament)
        self.player_database.clear_empty(self.game_database)

    def remove_player(self, player_name : str = None, player : Player = None):
        """
        Remove a player from the player database. Also removes all games associated with the player.

        Args:
            - player_name (str): The name of the player to remove. Defaults to None.
            - player (Player, optional): The player object to remove. Defaults to None.
        """
        if player is None:
            player = self.player_database.get_player_by_name(player_name)
        if player is None:
            raise ValueError(f"Player {player_name} not found in the database.")
        for game in self.game_database.get_games_per_player(player.id):
            if self.last_date_update is not None and game.get_date() < self.last_date_update and self.do_recompute:
                self.recompute = True
                logger.warning(f"You have removed a game from {game.get_date()} which is earlier than the last game in the rating period. Next recomputation of the ratings will need a full recompute.")
            self.game_database.remove(self.game_database[game.id])
        self.player_database.remove(player)

    def add_player(self, player_name : str = None, player : Player = None) -> Player:
        """
        Add a player to the player database.

        Args:
            - player_name (str): The name of the player to add.
            - player (Player, optional): The player object to add. Defaults to None.

        Returns:
            - Player: The player object.
        """
        if player is None:
            player = self.player_database.get_player_by_name(player_name)
            if player is None:
                player = Player(player_name)
                self.player_database.add(player)
        else:
            self.player_database.add(player)
        return player

    def remove_game(self, game_id : int = None, game : Game = None):
        """
        Remove a game from the game database.

        Args:
            - game_id (int): The ID of the game to remove. Defaults to None.
            - game (Game, optional): The game object to remove. Defaults to None.
        """
        if game is None:
            game = self.game_database[game_id]
        if game is None:
            raise ValueError(f"Game {game_id} not found in the database.")
        if self.last_date_update is not None and game.get_date() < self.last_date_update and self.do_recompute:
            self.recompute = True
            logger.warning(f"You have removed a game from {game.get_date()} which is earlier than the last game in the rating period. Next recomputation of the ratings will need a full recompute.")
        self.game_database.remove(game)

    def add_game(self, home_name : str = None, out_name : str = None, result_str : str = None, 
                 date : datetime = None, tournament_id : int = None, 
                 allow_new_players : bool = True, force_new_players : bool = False, 
                 game : Game = None) -> Game:
        """
        Adds a game to the manager's databases.

        Args:
            - home_name (str): The name of the home player.
            - out_name (str): The name of the out player.
            - result_str (str): The result of the game. Either 1-0, 0-1, 1/2-1/2, 1F-0, 0-1F, 0F-0F.
            - date (datetime, optional): The date of the game. Defaults to None.
            - tournament_id (int, optional): The ID of the tournament. Defaults to None.
            - allow_new_players (bool, optional): If set to True, allows adding new players to the database. Defaults to True.
            - force_new_players (bool, optional): If set to True, forces that the players are new players. Defaults to False.
            - game (Game, optional): The game object to add. Defaults to None.
        Returns:
            - Game: The game object.
        """
        if game is not None:
            self.game_database.add(game)
            return game
        logger.debug(f"Adding game {home_name} vs {out_name} with result {result_str}...")
        home = self.player_database.get_player_by_name(home_name)
        out = self.player_database.get_player_by_name(out_name)
        if home is None:
            if self.player_database.get_player_by_name(home_name) is None:
                if not allow_new_players:
                    raise ValueError(f"Player {home_name} not found in the database.")
                home = Player(home_name)
                self.player_database.add(home)
            elif force_new_players:
                raise ValueError(f"Player {home_name} found in the database.")
        if out is None:
            if self.player_database.get_player_by_name(out_name) is None:
                if not allow_new_players:
                    raise ValueError(f"Player {out_name} not found in the database.")
                out = Player(out_name)
                self.player_database.add(out)
            elif force_new_players:
                raise ValueError(f"Player {out_name} found in the database.")
        game = Game(home=home.id, out=out.id, result=result_str, date=date, tournament_id=tournament_id,
                    add_home_advantage=self.add_home_advantage, forfeit_keep_points=self.forfeit_keep_points)
        self.game_database.add(game)
        was_false = not self.recompute
        self.recompute = self.last_date_update is not None and game.get_date() <= self.last_date_update and self.do_recompute
        if self.recompute and was_false:
            logger.warning(f"You have added a game from {game.get_date()} which is earlier than the last game in the rating period. Next recomputation of the ratings will need a full recompute.")
        return game