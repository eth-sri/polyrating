import time
import numpy as np
import pandas as pd

from loguru import logger
from datetime import datetime, timedelta
from hyperopt import hp, Trials
from hyperopt import fmin, tpe, space_eval

from .rating import PolyratingDavidson, RatingSystem, PolyratingRao, PolyratingCrossEntropy, ChessMetrics, Glicko2, Glicko, Elo, EloPlusPlus, TrueSkillThroughTime
from .databases import GameDatabase, PlayerDatabase
from .objects import Player, Game, Rating, DefaultRating, DEFAULT_RATING, RatingPeriod, Advantage


class Optimizer:
    def __init__(self, game_database : GameDatabase, 
                 player_database : PlayerDatabase, 
                 rating_period : RatingPeriod, max_evals : int = 100, 
                 exclude_default : bool = True, 
                 epsilon : float = 1e-3, 
                 optimize_default_rating : bool = True, 
                 no_omega : bool = False, 
                 before : bool = True, **kwargs) -> 'Optimizer':
        """
        The Optimizer class is responsible for optimizing the rating system parameters based on game data.

        Attributes:
            - game_database (GameDatabase): The database containing the game data.
            - player_database (PlayerDatabase): The database containing the player data.
            - rating_period (RatingPeriod): The rating period.
            - max_evals (int): The maximum number of evaluations for the optimization process.
            - exclude_default (bool): Flag indicating whether to exclude default ratings from counting.
            - epsilon (float): A small value used for numerical stability in computations.
            - optimize_default_rating (bool): Flag indicating whether to optimize the default rating.
            - no_omega (bool): Flag indicating whether to exclude the omega parameter from optimization.
            - games_to_not_count (list): List of game IDs to not count in the optimization process.
            - param_history (list): List of dictionaries containing the parameter history during optimization.
            - before (bool): Flag indicating whether to compute the accuracy before or after the games.

        Args:
            - game_database (GameDatabase): The database containing chess game data.
            - player_database (PlayerDatabase): The database containing player data.
            - rating_period (RatingPeriod): The rating period.
            - max_evals (int, optional): The maximum number of evaluations for the optimization algorithm. Defaults to 100.
            - exclude_default (bool, optional): Whether to exclude games where one player has the default rating from the optimization. Defaults to True.
            - epsilon (float, optional): The convergence threshold for the optimization algorithm. Defaults to 1e-3.
            - optimize_default_rating (bool, optional): Whether to optimize the default rating. Defaults to True.
            - no_omega (bool, optional): Whether to exclude omega values from the optimization. Defaults to False.
            - before (bool, optional): Whether to compute the accuracy before or after the games. Defaults to True.
        """
        self.max_evals = max_evals
        self.game_database = game_database
        self.player_database = player_database
        self.rating_period = rating_period
        self.exclude_default = exclude_default
        self.epsilon = epsilon
        self.optimize_default_rating = optimize_default_rating
        self.no_omega = no_omega
        self.games_to_not_count = dict()
        self.param_history = []
        self.before = before

        if exclude_default:
            dates = dict()
            for player in self.player_database:
                dates[player.id] = [game.get_date() for game in self.game_database.get_games_per_player(player.id)]
            for game in self.game_database.get_games_no_forfeit():
                # check if this is the first game for one of the players
                game_period = self.rating_period.get_period_of_date(game.get_date(), next=False) - timedelta(seconds=1)
                home_game_dates = [date for date in dates[game.home] if date <= game_period]
                out_game_dates = [date for date in dates[game.out] if date <= game_period]
                if len(home_game_dates) == 0 or len(out_game_dates) == 0:
                    self.games_to_not_count[game.id] = True
                else:
                    self.games_to_not_count[game.id] = False

    def extend_database(self, target_n_games : int, target_n_players : int, target_rating_periods : int):
        """
        Randomly samples players and games to extend the game, player, and rating period databases.

        Args:
            - target_n_games (int): The target number of games to add to the database.
            - target_n_players (int): The target number of players to add to the database.
            - target_rating_periods (int): The target number of rating periods to add.
        """
        logger.info(f"Extending database to {target_n_games} games, {target_n_players} players, and {target_rating_periods} rating periods.")
        player_database = PlayerDatabase()
        player_matching = dict()

        # Create the player database
        for i, player in enumerate(self.player_database):
            if i >= target_n_players:
                break
            new_player = Player(str(i), i)
            player_database.add(new_player)
            player_matching[player.id] = [new_player.id]
        for i in range(target_n_players - len(self.player_database)):
            player = self.player_database.get_random()
            new_player = Player(str(i + len(self.player_database)), i + len(self.player_database))
            # add player matching
            player_matching[player.id].append(new_player.id)
            player_database.add(new_player)

        logger.info(f"Player database extended to {len(player_database)} players.")

        rating_periods = RatingPeriod()
        # Create the rating period database
        for i in range(target_rating_periods):
            rating_periods.trigger_new_period(datetime.now() - timedelta(days=i))

        logger.info(f"Rating period extended to {len(rating_periods)} periods.")

        game_database = GameDatabase()

        # Create the game database
        for i in range(target_n_games):
            game = self.game_database.get_random()
            # choose a random home and out player
            home = player_matching[game.home][np.random.randint(0, len(player_matching[game.home]))]
            out = player_matching[game.out][np.random.randint(0, len(player_matching[game.out]))]
            # choose a random date:
            date = rating_periods.get_last_period() - timedelta(days=np.random.randint(0, target_rating_periods))
            new_game = Game(home, out, game.result, i, date.strftime("%d/%m/%Y"))
            game_database.add(new_game)

        logger.info(f"Game database extended to {len(game_database)} games.")
        
        self.game_database = game_database
        self.player_database = player_database
        self.rating_period = rating_periods

    def store_param_history(self, file : str):
        """
        Stores the parameter history as a CSV file.

        Args:
            - file (str): The file path to save the parameter history.
        """
        param_history = pd.DataFrame(self.param_history)
        param_history.to_csv(file, index=False)

    def compute_game_differences(self, games : GameDatabase, players : PlayerDatabase, rating_system : RatingSystem) -> float:
        """
        Computes the accuracy of predictions for a given set of games, players, and rating system.

        Args:
            - games (GameDatabase): A list of game objects.
            - players (PlayerDatabase): A list of player objects.
            - rating_system (object): An instance of the rating system to be evaluated.

        Returns:
            - float: The computed game differences.

        """
        if self.before:
            delta = timedelta(seconds=1)
        else:
            delta = timedelta(seconds=-1)
        log_loss = 0
        accuracy = 0
        accuracy_count = 0
        log_loss_count = 0
        for game in games.get_games_no_forfeit():
            if self.games_to_not_count[game.id]:
                continue
            home = players[game.home]
            out = players[game.out]
            expected_score = rating_system.compute_expected_score(home, [game], players, game.get_date() - delta)
            result = game.get_result()
            log_loss += -result * np.log(expected_score + self.epsilon)
            log_loss +=  - (1 - result) * np.log(1 - expected_score + self.epsilon)
            log_loss_count += 1
            if result == 1 and expected_score >= 0.5:
                accuracy += 1
                accuracy_count += 1
            elif result == 0 and expected_score <= 0.5:
                accuracy += 1
                accuracy_count += 1
            elif result in [0, 1]:
                accuracy_count += 1

        if np.isnan(log_loss):
            return 10 ** 6

        return log_loss / log_loss_count
        # return - accuracy / accuracy_count
    
    def update_rating(self, rating_system : RatingSystem, **kwargs):
        """
        Updates the rating of each player in the player database using the specified rating system.

        Args:
            - rating_system (RatingSystem): The rating system to use for updating the ratings.
            - **kwargs: Additional keyword arguments that can be passed to the rating system.
        """
        for player in self.player_database:
            player.clear_rating_history()
            player.get_rating().reset()
        
        for period_dates in self.rating_period.iterate_periods():
            logger.info(f"Updating ratings for period {period_dates[-1]}")
            rating_system.period_update(self.player_database, self.game_database, period_dates)
            for player in self.player_database:
                player.store_rating(period_dates[-1])

    def objective(self, params : dict) -> float:
        """
        Compute the objective value for the optimization problem.

        Args:
            - params (dict): A dictionary containing the parameters for the rating system.

        Returns:
            - float: The objective value representing the difference between predicted and actual game outcomes.
        """
        logger.info(f"Optimizing with params: {params}")
        params_without_rating_class = {k: v for k, v in params.items() if k != 'rating_class'}
        if 'advantages' in params_without_rating_class:
            params_without_rating_class['advantages'] = {Advantage.HOME_ADVANTAGE: DefaultRating(0, params_without_rating_class['advantages'], 
                                                                                                 id=Advantage.HOME_ADVANTAGE)}
        if 'omegas_advantages' in params_without_rating_class:
            params_without_rating_class['omegas_advantages'] = {Advantage.HOME_ADVANTAGE: params_without_rating_class['omegas_advantages']}
        
        advantages = dict()
        omegas_advantages = dict() 
        keys = list(params_without_rating_class.keys())
        for param in keys:
            if 'advantage' in param and 'omega' not in param:
                advantages[param] = Rating(0, params_without_rating_class[param])
                del params_without_rating_class[param]
            if 'omega' in param:
                omegas_advantages[param.replace('omegas_', '')] = params_without_rating_class[param]
                del params_without_rating_class[param]

        if len(advantages) > 0:
            params_without_rating_class['advantages'] = advantages
            params_without_rating_class['omegas_advantages'] = omegas_advantages

        rating_system = params['rating_class'](**params_without_rating_class)
        DEFAULT_RATING.set_default(params.get('default_rating'), 
                            params.get('default_deviation'), 
                            params.get('default_volatility'))
        try:
            self.update_rating(rating_system)
            difference = self.compute_game_differences(self.game_database, self.player_database, rating_system)
        except Exception as e:
            logger.error(f"Error in optimization: {e}")
            difference = 10 ** 6
        logger.info(f"Difference: {difference}")
        self.param_history.append({**params, 'difference': difference})
        return difference
    
    def time_vs_performance(self, save_file : str, max_max_iters : int = 100) -> pd.DataFrame:
        """
        Computes the time taken and log loss for different rating classes and parameters.

        Args:
            - save_file (str): The file path to save the results as a CSV file.
            - max_max_iters (int, optional): The maximum number of iterations. Defaults to 100.

        Returns:
            - pandas.DataFrame: A DataFrame containing the time taken and log loss for each rating class and parameter combination.
        """
        updates = []
        rating_classes = [PolyratingDavidson, PolyratingRao, PolyratingCrossEntropy]
        a = time.time()
        glicko = Glicko2()
        self.update_rating(glicko)
        b = time.time()
        log_loss = self.compute_game_differences(self.game_database, self.player_database, glicko, before=True)
        time_difference = b - a
        updates.append({'time': time_difference, 'log_loss': log_loss, 'rating_class': 'Glicko2'})
        logger.info(f"Time: {time_difference}, Log Loss: {log_loss}, Rating Class: Glicko2")
        for polyrating in rating_classes:
            for linearized in [False, True]:
                for max_iters in [1, 2, 5, 10, 20, 50, 100]:
                    if max_iters > max_max_iters:
                        continue
                    rating_system = polyrating(linearized=linearized, max_iterations=max_iters)
                    a = time.time()
                    self.update_rating(rating_system)
                    b = time.time()
                    log_loss = self.compute_game_differences(self.game_database, self.player_database,
                                                             rating_system, before=True)
                    time_difference = b - a
                    updates.append({'time': time_difference, 
                                    'log_loss': log_loss, 
                                    'rating_class': str(polyrating), 
                                    'linearized': linearized, 
                                    'max_iters': max_iters})
                    logger.info(f"Time: {time_difference}, Log Loss: {log_loss}, Rating Class: {polyrating}, Linearized: {linearized}, Max Iters: {max_iters}")

        df = pd.DataFrame(updates)
        df.to_csv(save_file, index=False)
        return df
    
    def optimize_with_defined_space(self, space : dict) -> dict:
        """
        Optimize the objective function over the defined space.

        Args:
            - space (dict): The search space for optimization.

        Returns:
            - dict: The best set of parameters found by the optimization algorithm.
        """
        if not self.optimize_default_rating:
            if 'default_deviation' in space:
                del space['default_deviation']
            if 'default_volatility' in space:
                del space['default_volatility']
            if 'sigma' in space:
                del space['sigma']
        if self.no_omega:
            if 'omega' in space:
                del space['omega']
            if 'default_change' in space:
                del space['default_change']

        # minimize the objective over the space
        trials = Trials()
        best = fmin(self.objective, space, algo=tpe.suggest, max_evals=self.max_evals, trials=trials)
        
        best_params = space_eval(space, best)
        return best_params, min(trials.losses())

    def optimize(self):
        """
        Optimize the rating parameters using hyperparameter optimization.

        This method defines different search spaces for various rating classes and performs hyperparameter optimization
        using the defined spaces. The `optimize_with_defined_space` method is called for each search space.
        """
        # define the search space
        space_davidson = {
            'rating_class': hp.choice('rating_class', [PolyratingDavidson]),
            'default_deviation': hp.uniform('default_deviation', 100, 1500),
            'theta': hp.uniform('theta', 0, 1),
            'epsilon': hp.loguniform('epsilon', -3, 0),
            'linearized': hp.choice('linearized', [None, 1]),
            'default_change': hp.uniform('default_change', 100, 1500),
            'omega': hp.uniform('omega', 1, 100),
            Advantage.HOME_ADVANTAGE: hp.uniform(Advantage.HOME_ADVANTAGE, 0, 100), 
            'omegas_' + Advantage.HOME_ADVANTAGE: hp.uniform('omegas_home_advantage', 0, 50)
        }
        best_params_davidson, loss_davidson = self.optimize_with_defined_space(space_davidson)
        space_rao = {
            'rating_class': hp.choice('rating_class', [PolyratingRao]),
            'default_deviation': hp.uniform('default_deviation', 100, 1500),
            'theta': hp.uniform('theta', 1, 2),
            'epsilon': hp.loguniform('epsilon', -3, 0),
            'linearized': hp.choice('linearized', [None, 1]),
            'default_change': hp.uniform('default_change', 100, 1500),
            'omega': hp.uniform('omega', 1, 100),
            Advantage.HOME_ADVANTAGE: hp.uniform(Advantage.HOME_ADVANTAGE, 0, 100), 
            'omegas_' + Advantage.HOME_ADVANTAGE: hp.uniform('omegas_home_advantage', 0, 50)
        }
        best_params_rao, loss_rao = self.optimize_with_defined_space(space_rao)
        space_crossentropy = {
            'rating_class': hp.choice('rating_class', [PolyratingCrossEntropy]),
            'default_deviation': hp.uniform('default_deviation', 100, 1500),
            'epsilon': hp.loguniform('epsilon', -3, 0),
            'linearized': hp.choice('linearized', [None, 1]),
            'default_change': hp.uniform('default_change', 100, 1500),
            Advantage.HOME_ADVANTAGE: hp.uniform(Advantage.HOME_ADVANTAGE, 0, 100), 
            'omegas_' + Advantage.HOME_ADVANTAGE: hp.uniform('omegas_home_advantage', 0, 50)
        }
        best_params_cross, loss_cross = self.optimize_with_defined_space(space_crossentropy)
        space_glicko2 = {
            'rating_class': hp.choice('rating_class', [Glicko2]),
            'default_deviation': hp.uniform('default_deviation', 100, 1500),
            'default_volatility': hp.uniform('default_volatility', 0.01, 0.2),
            'tau': hp.uniform('tau', 0, 1),
            'conversion_constant': hp.uniform('conversion_constant', 50, 500),
        }
        best_params_glicko2, loss_glicko2 = self.optimize_with_defined_space(space_glicko2)

        space_glicko = {
            'rating_class': hp.choice('rating_class', [Glicko]),
            'default_deviation': hp.uniform('default_deviation', 100, 1500),
            'default_change': hp.uniform('default_change', 100, 800),
            'C': hp.uniform('tau', 1, 200),
        }
        best_params_glicko, loss_glicko = self.optimize_with_defined_space(space_glicko)

        space_elo = {
            'rating_class': hp.choice('rating_class', [Elo]),
            'K': hp.uniform('K', 10, 500),
        }
        best_params_elo, loss_elo = self.optimize_with_defined_space(space_elo)
        space_eloplusplus = {
            'rating_class': hp.choice('rating_class', [EloPlusPlus]),
            'gamma': hp.uniform('gamma', 0, 1),
            'lambda_': hp.uniform('lambda_', 0, 1),
            'factor': hp.uniform('factor', 0, 1),
            'default_change': hp.uniform('default_change', 100, 1500),
            'lr_change_coef': hp.uniform('lr_change_coef', 0, 1),
        }
        best_params_eloplusplus, loss_eloplusplus = self.optimize_with_defined_space(space_eloplusplus)
        space_chessmetrics = {
            'rating_class': hp.choice('rating_class', [ChessMetrics]),
            'default_change': hp.uniform('default_change', 500, 1500),
            'too_old': hp.uniform('too_old', 100, 2 * 365),
            'epsilon': hp.loguniform('epsilon', -3, 0),
            'home_advantage': hp.uniform('home_advantage', 0, 100),
            'delta_expected': hp.uniform('delta_expected', 0.1, 5),
            'weighted_average': hp.uniform('weighted_average', 0.5, 0.99999),
        }
        best_params_chessmetrics, loss_chessmetrics = self.optimize_with_defined_space(space_chessmetrics)

        space_trueskill = {
            'rating_class': hp.choice('rating_class', [TrueSkillThroughTime]),
            'p_draw': hp.uniform('p_draw', 0.01, 0.2),
            'beta': hp.uniform('beta', 0.5, 1.5),
            'gamma': hp.uniform('gamma', 0, 0.1),
            'linearized': hp.choice('linearized', [None, 1, 3, 5]),
            'sigma': hp.uniform('sigma', 0.5, 12),
        }
        best_params_trueskill, loss_trueskill = self.optimize_with_defined_space(space_trueskill)