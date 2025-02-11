import numpy as np

from loguru import logger
from datetime import datetime
from typing import List, Dict, Tuple

from ..objects import Player, Game
from ..objects.rating import Rating, DefaultRating, DEFAULT_RATING
from ..objects.matching import Matching
from ..objects.rating_history import RatingHistory
from ..databases import PlayerDatabase, GameDatabase
from .rating_system import RatingSystem


class Polyrating(RatingSystem):
    def __init__(self, default_change : float = 400, epsilon : float = 1e-2, max_iterations : float = 100, 
                 omega : float = 10, linearized : int = None, 
                 omegas_advantages : Dict[str, float] = None, advantages : Dict[str, 'DefaultRating'] = None, 
                 shared_advantages : List[Tuple[str, Matching, DefaultRating, float]] = None, 
                 sigma_armijo : float = 0.5, min_omega=1e-2, allow_complex_result : bool = False, max_variance=10 ** 8,
                 shared_m=10, **kwargs) -> 'Polyrating':
        """
        A rating system that computes the expected score and tournament performance
        based on the whole history of a player's ratings. 
        The base is based on on https://link.springer.com/chapter/10.1007/978-3-540-87608-3_11

        We additionally allow for the use of advantages: these are separate (smaller) ratings that are added 
        to the main rating and can differ for the same player in different games. Since the advantages are just
        added to the main rating, it is rather simple to implement them.

        Attributes:
            - default_change (float): The default change in rating for a player.
            - epsilon (float): The convergence threshold for the rating update algorithm.
            - max_iterations (int): The maximum number of iterations for the rating update algorithm.
            - omega (float): Deviation of the Markov chain. Roughly is the number of rating points a player is expected to gain or lose in a single period.
            - linearized (int): The number of periods to linearize the rating history.
            - omegas_advantages (dict): The deviation of the Markov chain for each advantage.
            - shared_omegas_advantages (dict): The deviation of the Markov chain for each shared advantage. Shared means shared by the players that match the given Matching.
            - advantages (dict): The advantages with their default ratings to compute the prior.
            - shared_advantages (list): The shared advantages with their default ratings to compute the prior.  Shared means shared by the players that match the given Matching.
            - sigma_armijo (float): The sigma value for the Armijo rule, i.e. f(x+ad) <= f(x) + sigma * alpha * d where alpha is the chosen step size and d is the descent direction given by Newton's method. This is used to determine the step size in the Armijo rule. Only affects convergence rate, do not set this unless you know what you are doing.
            - min_omega (float): The minimum value for the deviation of the Markov chain. Lower than this value, and the rating is forced to be constant over time. For the purpose of numerical stability, do not set this lower.
            - allow_complex_result (bool): If True, the result of the game can be any number between 0 and 1 (depending on the margin with which it was won). Defaults to False.
            - max_variance (float): The maximum variance allowed for a rating. If the variance exceeds this value, the prior is assumed to be uniform
            - shared_m (int): the amount of samples to keep in the history of the shared ratings for the LBFGS optimization.
        Args:
            - default_change (float): The default change in rating for a player.
            - epsilon (float): The convergence threshold for the rating calculation.
            - max_iterations (int): The maximum number of iterations for the rating calculation.
            - omega (float): Deviation of the Markov chain. Roughly is the number of rating points a player is expected to gain or lose in a single period.
            - linearized (int): The number of periods to linearize the rating history. If None, the whole history is used.
            - omegas_advantages (dict): The deviation of the Markov chain for each advantage.
            - shared_omegas_advantages (dict): The deviation of the Markov chain for each shared advantage. Shared means shared by the players that match the given Matching.
            - advantages (dict): The advantages with their default ratings to compute the prior.
            - shared_advantages (list): The shared advantages with their default ratings to compute the prior. Shared means shared by the players that match the given Matching.
            - sigma_armijo (float): The sigma value for the Armijo rule, i.e. f(x+ad) <= f(x) + sigma * alpha * d where alpha is the chosen step size and d is the descent direction given by Newton's method. This is used to determine the step size in the Armijo rule. Only affects convergence rate, do not set this unless you know what you are doing.
            - min_omega (float): The minimum value for the deviation of the Markov chain. Lower than this value, and the rating is forced to be constant over time. For the purpose of numerical stability, do not set this lower.
            - allow_complex_result (bool): If True, the result of the game can be any number between 0 and 1 (depending on the margin with which it was won). Defaults to False.
            - max_variance (float): The maximum variance allowed for a rating. If the variance exceeds this value, the prior is assumed to be uniform
            - shared_m (int): the amount of samples to keep in the history of the shared ratings for the LBFGS optimization.
            - **kwargs: Additional keyword arguments.
        """
        if advantages is None:
            advantages = dict()
        if omegas_advantages is None:
            omegas_advantages = dict()
        if shared_advantages is None:
            shared_advantages = []

        omega = max(omega, min_omega)
        omegas_advantages = {key: max(omega, min_omega) for key, omega in omegas_advantages.items()}
        shared_advantages = [(key, match, rating, max(omega, min_omega)) for key, match, rating, omega in shared_advantages]
        self.omegas = [omega] + [omegas_advantages.get(key, omega) for key in advantages.keys()]
        self.default_ratings = [DEFAULT_RATING] + list(advantages.values())
        self.shared_omegas = [adv[3] for adv in shared_advantages]
        self.shared_default_ratings = [adv[2] for adv in shared_advantages]
        self.shared_s_list = []
        self.shared_y_list = []
        self.shared_prev_grad = None
        self.shared_prev_x = None
        self.shared_rho_list = []
        self.shared_alpha_list = []


        if 'shared_rating_histories' not in kwargs:
            self.shared_rating_histories = [RatingHistory(Rating(default_rating=self.shared_default_ratings[i])) for i in range(len(self.shared_default_ratings))]
            kwargs['shared_rating_histories'] = self.shared_rating_histories
        else:
            self.shared_rating_histories = kwargs['shared_rating_histories']

        super().__init__(default_change=default_change, epsilon=epsilon, linearized=linearized, 
                         max_iterations=max_iterations, omega=omega, 
                         advantages=advantages, omegas_advantages=omegas_advantages, 
                         shared_advantages=shared_advantages, allow_complex_result=allow_complex_result,
                         sigma_armijo=sigma_armijo,
                         min_omega=min_omega, max_variance=max_variance,
                         shared_m=shared_m, **kwargs)

    def compute_expected_score_rating(self, player_rating : Rating, 
                                      opponent_rating : Rating, 
                                      is_home : bool) -> float:
        """
        Computes the expected score of a player against an opponent.

        Args:
            - player_rating (Rating): The rating of the player.
            - opponent_rating (Rating): The rating of the opponent.
            - is_home (bool): If True, the player is home, otherwise the player is out.

        Returns:
            - float: The expected score of the player against the opponent.
        """
        win = self.win_prob(player_rating.rating, opponent_rating.rating)
        tie = self.tie_prob(player_rating.rating, opponent_rating.rating)
        return win + 1 / 2 * tie

    def compute_expected_score(self, player : Player, games : List[Game],
                               player_database : PlayerDatabase, 
                               date : datetime, 
                               next : bool = False) -> float:
        expected_score = 0
        for game in games:
            home_rating = player_database[game.home].get_rating_at_date(date, next=next)
            out_rating = player_database[game.out].get_rating_at_date(date, next=next)
            actual_rating_home = home_rating.get_rating(game.get_advantages(True))
            for i, advantage in enumerate(self.shared_advantages):
                if advantage[1].match(player_database[game.home].get_info()):
                    game_advantage = game.get_advantages(True).get(advantage[0], 0)
                    rating = self.shared_rating_histories[i].get_rating_at_date(date, next=next)
                    actual_rating_home += rating.rating * game_advantage

            actual_rating_out = out_rating.get_rating(game.get_advantages(False))
            for i, advantage in enumerate(self.shared_advantages):
                if advantage[1].match(player_database[game.out].get_info()):
                    game_advantage = game.get_advantages(False).get(advantage[0], 0)
                    rating = self.shared_rating_histories[i].get_rating_at_date(date, next=next)
                    actual_rating_out += rating.rating * game_advantage
            
            if game.home == player.id:
                player_rating = actual_rating_home
                opponent_rating = actual_rating_out
            else:
                player_rating = actual_rating_out
                opponent_rating = actual_rating_home
            win = self.win_prob(player_rating, opponent_rating)
            tie = self.tie_prob(player_rating, opponent_rating)
            expected_score += game.weight * (win + 1 / 2 * tie)
        return expected_score
    
    def compute_tournament_performance(self, player : Player, 
                                       tournament_id : int, tournament_date : datetime, 
                                       game_database : GameDatabase, 
                                       player_database : PlayerDatabase, 
                                       next : bool = False) -> Rating:
        games_in_tournament = game_database.get_games_per_tournament(tournament_id)
        games_in_tournament = [game for game in games_in_tournament if game.home == player.id or game.out == player.id]
        rating_info = player.get_rating().copy()
        rating_history = player.get_rating_history()
        player.set_rating_history([])
        player.get_rating().reset()
        ratings, variations, _, _ = self.compute_period_update(player_database, games_in_tournament, 
                                                [tournament_date], 
                                                player_id=player.id)
        player.get_rating().set(rating_info)
        player.set_rating_history(rating_history)
        return Rating(ratings[player.id, -1, 0], variations[player.id, -1, 0] ** 0.5)

    def period_update(self, player_database : PlayerDatabase, game_database : GameDatabase, 
                      period_dates : List[datetime], **kwargs):
        # for each game, the period in which it takes place is the earliest date which is higher or equal to the game date
        ratings, variances, shared_ratings, shared_variances = self.compute_period_update(player_database, 
                                                        list(game_database.get_games_between_dates(period_dates[-1])), 
                                                        period_dates, **kwargs)
        for player in player_database:
            has_played_before = False
            for game in game_database.get_games_per_player(player.id):
                if game.get_date() <= period_dates[-1]:
                    has_played_before = True
                    break
            if not has_played_before:
                continue
            rating = player.get_rating()
            rating.update(ratings[player.id, -1, 0], variances[player.id, -1, 0] ** 0.5)

            for i, advantage in enumerate(self.advantages):
                rating.update_advantage(advantage, ratings[player.id, -1, i + 1], 
                                        variances[player.id, -1, i + 1] ** 0.5, 
                                        default_rating=self.default_ratings[i + 1])
        
        for i, shared_advantage in enumerate(self.shared_advantages):
            self.shared_rating_histories[i].get_rating().update(shared_ratings[-1, i], shared_variances[-1, i] ** 0.5)
            self.shared_rating_histories[i].store_rating(period_dates[-1])

    def compute_period_update(self, player_database : PlayerDatabase, 
                              games : List[Game],
                              period_dates : List[datetime],
                              player_id : int = None, **kwargs) -> tuple:
        """
        Computes the period update for player ratings.

        Args:
            - player_database (PlayerDatabase): The database of players.
            - games (Generator[Game])): The database of games.
            - period_dates (list): A list of period dates.
            - player_id (int, optional): The ID of the player to compute the update for. Defaults to None (all players).
            - **kwargs: Additional keyword arguments.

        Returns:
            - tuple: A tuple containing the ratings and variances arrays. Note that the variances are calculated as if the ratings were the ones of the next to last Newton-optimization step performed to avoid extra computations. However,
            due to the epsilon value, the ratings from that step are quite close to the final ones.
        """
        # Initialize ratings and variances
        self.shared_s_list = []
        self.shared_y_list = []
        self.shared_prev_grad = None
        self.shared_prev_x = None
        self.shared_rho_list = []
        self.shared_alpha_list = []
        linear = self.linearized
        if player_id is not None:
            linear = 1

        periods = self.extract_periods(games, period_dates)

        size_matrix = max(periods) + 1
        if linear:
            size_matrix = min(linear, size_matrix)
            linear = min(linear, size_matrix)
        # set proper initial ratings and variances
        last_dimension = 1 + len(self.advantages)
        ratings = np.zeros((player_database.get_max_id() + 1, size_matrix, last_dimension))
        variances = np.zeros((player_database.get_max_id() + 1, size_matrix, last_dimension))
        last_dimension = len(self.shared_advantages)
        shared_ratings = np.zeros((size_matrix, last_dimension))
        if player_id is not None:
            # initialize shared ratings to the actual values, since they dont get updated
            for dim in range(last_dimension):
                for i, date in enumerate(period_dates[-linear:]):
                    shared_rating = self.shared_rating_histories[dim].get_rating_at_date(date, next=True)
                    shared_ratings[i, dim] = shared_rating.rating

        shared_variances = np.zeros((size_matrix, last_dimension))

        date = period_dates[0]
        if linear is not None:
            date = period_dates[-min(linear, len(period_dates))]
        self.initialize_ratings(player_database, player_id, ratings, variances, date)

        # get games per player, and store in efficient format
        games_per_player = self.initialize_games_per_players(player_database, games, linear, periods)
        
        old_ratings = np.zeros(ratings.shape)
        old_shared_ratings = np.zeros(shared_ratings.shape)
        earliest_player_periods = {
            id_: min([game[1] for game in games]) if len(games) > 0 else 0 for id_, games in games_per_player.items()
        }

        default_ratings = dict()
        if linear is not None:
            self.get_default_ratings(player_database, period_dates, player_id, linear, default_ratings)

        for iteration in range(self.max_iterations):
            shared_diff = 0
            if len(self.shared_advantages) > 0 and player_id is None:
                shared_diff = np.mean(np.abs(shared_ratings - old_shared_ratings))
            logger.debug(f"Starting iteration {iteration} of period update. Current difference: {np.mean(np.abs(ratings - old_ratings)):.5f}, {shared_diff:.5f}")

            if self.break_condition_player_not_none(player_id, ratings, shared_ratings, old_ratings, old_shared_ratings, iteration):
                break
            elif self.break_condition_player_none(player_id, ratings, shared_ratings, old_ratings, old_shared_ratings, iteration):
                break

            old_ratings = ratings.copy()

            for player in player_database:
                if len(games_per_player.get(player.id, [])) == 0 and player.get_rating().deviation >= player.get_rating().default_rating.deviation:
                    continue
                default_rating = np.array([rating.rating for rating in self.default_ratings])
                default_deviation = np.array([rating.deviation for rating in self.default_ratings])
                if linear is not None:
                    default_rating, default_deviation = default_ratings[player.id]

                if player_id is None or player.id == player_id:
                    ratings[player.id], variances[player.id], _, _ = self.update_player(ratings[player.id], 
                                                                                  variances[player.id], 
                                                                                  games_per_player.get(player.id, []), ratings, 
                                                                                  earliest_player_periods.get(player.id, 0), 
                                                                                  default_rating, default_deviation, 
                                                                                  shared_ratings)
                    
            old_shared_ratings = shared_ratings.copy()
            derivative_shared_ratings, second_derivative_shared_ratings = self.initialize_shared_derivatives(size_matrix, last_dimension, shared_ratings)
            if len(self.shared_advantages) > 0 and player_id is None:
                for player in player_database:
                    if len(games_per_player.get(player.id, [])) == 0:
                        continue
                    _, _, der_shared, sec_shared = self.update_player(ratings[player.id], 
                                                                    variances[player.id], 
                                                                    games_per_player.get(player.id, []), ratings, 
                                                                    earliest_player_periods.get(player.id, 0), 
                                                                    None, None, 
                                                                    shared_ratings, True)
                    
                    derivative_shared_ratings += der_shared
                    second_derivative_shared_ratings += sec_shared
                shared_variances = self.update_shared_ratings(player_id, shared_ratings, derivative_shared_ratings, second_derivative_shared_ratings)
            if iteration == self.max_iterations:
                logger.debug("Maximum number of iterations reached in period update.")

        for player in player_database:
            if len(games_per_player.get(player.id, [])) == 0 and player.get_rating().deviation >= player.get_rating().default_rating.deviation:
                    continue
            default_rating = np.array([rating.rating for rating in self.default_ratings])
            default_deviation = np.array([rating.deviation for rating in self.default_ratings])
            if linear is not None:
                default_rating, default_deviation = default_ratings[player.id]
            if player_id is None or player.id == player_id:
                final_variances = self.compute_final_variances_player(ratings[player.id], variances[player.id], games_per_player.get(player.id, []), ratings, 
                                                                  earliest_player_periods.get(player.id, 0), default_rating, default_deviation, shared_ratings, 
                                                                  second_derivative_shared_ratings)
                variances[player.id, -1, :] = final_variances
        
        return ratings, variances, shared_ratings, shared_variances

    def break_condition_player_none(self, player_id: int, ratings: np.ndarray, shared_ratings: np.ndarray,
                                    old_ratings: np.ndarray, old_shared_ratings: np.ndarray, iteration: int):
        """
        Check if the break condition for the player being None is met.

        Args:
            - player_id (int): The ID of the player.
            - ratings (np.ndarray): The current ratings of the players.
            - shared_ratings (np.ndarray): The shared ratings of the players.
            - old_ratings (np.ndarray): The previous ratings of the players.
            - old_shared_ratings (np.ndarray): The previous shared ratings of the players.
            - iteration (int): The current iteration number.

        Returns:
            - bool: True if the break condition is met, False otherwise.
        """
        if player_id is not None:
            return False
        if iteration == 0:
            return False
        if np.mean(np.abs(ratings - old_ratings)) >= self.epsilon:
            return False
        return (old_shared_ratings.size == 0 or np.mean(np.abs(shared_ratings - old_shared_ratings)) < self.epsilon)

    def break_condition_player_not_none(self, player_id: int, ratings: np.ndarray, shared_ratings: np.ndarray,
                                        old_ratings: np.ndarray, old_shared_ratings: np.ndarray, iteration: int):
        """
        Check if the break condition for the player is met.

        Args:
            - player_id (int): The ID of the player.
            - ratings (np.ndarray): The current ratings of all players.
            - shared_ratings (np.ndarray): The shared ratings among all players.
            - old_ratings (np.ndarray): The previous ratings of all players.
            - old_shared_ratings (np.ndarray): The previous shared ratings among all players.
            - iteration (int): The current iteration number.

        Returns:
            - bool: True if the break condition is met, False otherwise.
        """
        if player_id is None:
            return False
        if iteration == 0:
            return False
        return np.mean(np.abs(ratings[player_id] - old_ratings[player_id])) < self.epsilon
    
    def initialize_shared_derivatives(self, size_matrix: int, last_dimension: int, shared_ratings: np.ndarray):
        """
        Initializes the shared derivatives for the given size matrix and last dimension.

        Args:
            - size_matrix (int): The size of the matrix.
            - last_dimension (int): The last dimension of the matrix.
            - shared_ratings (np.ndarray): The shared ratings.

        Returns:
            - tuple: A tuple containing the derivative shared ratings and second derivative shared ratings.
        """
        derivative_shared_ratings = np.zeros((size_matrix * last_dimension,))
        second_derivative_shared_ratings = np.zeros((size_matrix * last_dimension, size_matrix * last_dimension))
        rating_array = np.array([rating.rating for rating in self.shared_default_ratings])
        variance_array = np.array([rating.deviation for rating in self.shared_default_ratings]) ** 2
        derivative_shared_ratings[:len(shared_ratings[0])] += self.derivative_prior(shared_ratings[0], rating_array, variance_array).reshape(-1)
        prior_sec_der = self.second_derivative_prior(shared_ratings[0], rating_array, variance_array).reshape(-1)
        for i in range(len(prior_sec_der)):
            second_derivative_shared_ratings[i, i] += prior_sec_der[i]
        derivative_shared_ratings += self.derivative_markov(shared_ratings, omegas=self.shared_omegas).reshape(-1)
        sec_der_markov = self.second_derivative_markov(shared_ratings, omegas=self.shared_omegas)
        for i in range(sec_der_markov.shape[2]):
            second_derivative_shared_ratings[i :: shared_ratings.shape[1], i :: shared_ratings.shape[1]] += sec_der_markov[:, :, i]
        return derivative_shared_ratings, second_derivative_shared_ratings

    def update_shared_ratings(self, player_id: int, shared_ratings: np.ndarray,
                              derivative_shared_ratings: np.ndarray, second_derivative_shared_ratings: np.ndarray):
        """
        Update the shared ratings based on the player ID, derivative shared ratings, and second derivative shared ratings.

        Args:
            - player_id (int): The ID of the player.
            - shared_ratings (np.ndarray): The shared ratings.
            - derivative_shared_ratings (np.ndarray): The derivative shared ratings.
            - second_derivative_shared_ratings (np.ndarray): The second derivative shared ratings.

        Returns:
            - np.ndarray: The updated shared variances.
        """
        if player_id is None and second_derivative_shared_ratings.shape[0] > 0:
            if np.linalg.det(second_derivative_shared_ratings) != 0:
                second_derivative_shared_inv = np.linalg.inv(second_derivative_shared_ratings)
                if self.shared_prev_x is not None:
                    s = (shared_ratings - self.shared_prev_x).reshape(-1)
                    y = (-derivative_shared_ratings - self.shared_prev_grad).reshape(-1)
                    rho = 1 / np.dot(s, y)
                    self.shared_rho_list.append(rho)
                    self.shared_s_list.append(s)
                    self.shared_y_list.append(y)
                    if len(self.shared_rho_list) > self.shared_m:
                        self.shared_rho_list.pop(0)
                        self.shared_s_list.pop(0)
                        self.shared_y_list.pop(0)

                if len(self.shared_s_list) == 0:
                    step = -derivative_shared_ratings
                else:
                    q = (-derivative_shared_ratings.copy()).reshape(-1)
                    alpha_list = []
                    for i in range(len(self.shared_s_list) - 1, -1, -1):
                        alpha = np.dot(self.shared_s_list[i], q) * self.shared_rho_list[i]
                        q -= alpha * self.shared_y_list[i]
                        alpha_list.append(alpha)
                    gamma = np.dot(self.shared_s_list[-1], self.shared_y_list[-1]) / np.dot(self.shared_y_list[-1], self.shared_y_list[-1])
                    step = gamma * q

                    for i in range(len(self.shared_s_list)):
                        beta = np.dot(self.shared_y_list[i], step) * self.shared_rho_list[i]
                        step += (alpha_list[-(i+1)] - beta) * self.shared_s_list[i]
                
                step = step.reshape(shared_ratings.shape)
                self.shared_prev_grad = - derivative_shared_ratings
                self.shared_prev_x = shared_ratings.copy()
                shared_ratings -= step
                
                for i, omega in enumerate(self.shared_omegas):
                    if omega < self.min_omega: # if the deviation is too small, we force the rating to be constant
                        shared_ratings[:, i] = np.mean(shared_ratings[:, i])
                diag = np.diag(second_derivative_shared_inv).reshape(shared_ratings.shape)
                shared_variances = - diag
            else:
                shared_variances = np.zeros(shared_ratings.shape)
            return shared_variances
        return np.zeros(shared_ratings.shape)

    def initialize_games_per_players(self, player_database: PlayerDatabase, games: List[Game], 
                                     linear: int, periods: List[int]):
        """
        Initializes a dictionary that maps each player to a list of games they have played.

        Args:
            - player_database (PlayerDatabase): The database containing player information.
            - games (List[Game]): The list of games to be processed.
            - linear (int): The number of most recent periods to consider.
            - periods (List[int]): The list of periods corresponding to each game.

        Returns:
            - dict: A dictionary mapping each player to a list of games they have played. Each game is represented as a tuple
                  containing the result, period, opponent, home advantages, out advantages, home shared advantages, and
                  out shared advantages.
        """
        games_per_player = dict()
        max_periods = max(periods)
        for game, period in zip(games, periods):
            # get result
            if linear is None or period >= max_periods - linear + 1:
                if linear is not None:
                    period = period - max_periods - 1 + linear
                result = game.get_result(complex_result=self.allow_complex_result)
                home = player_database[game.home]
                out = player_database[game.out]
                home_advantages = game.get_advantages(True)
                home_advantages = {key: home_advantages.get(key, 0) for key in self.advantages}
                out_advantages = game.get_advantages(False)
                out_advantages = {key: out_advantages.get(key, 0) for key in self.advantages}
                home_shared_advantages = dict()
                out_shared_advantages = dict()
                for i, advantage in enumerate(self.shared_advantages):
                    if advantage[1].match(home.get_info()):
                        home_shared_advantages[advantage[0]] = (game.get_advantages(True).get(advantage[0], 0), i)
                    if advantage[1].match(out.get_info()):
                        out_shared_advantages[advantage[0]] = (game.get_advantages(False).get(advantage[0], 0), i)
                games_per_player[game.home] = games_per_player.get(game.home, []) + \
                                                    [(result, period, game.out, home_advantages, out_advantages, 
                                                    home_shared_advantages, out_shared_advantages, game.weight)]
                games_per_player[game.out] = games_per_player.get(game.out, []) + \
                                                    [(1 - result, period, game.home, out_advantages, home_advantages,
                                                    out_shared_advantages, home_shared_advantages, game.weight)]
                                                    
        return games_per_player

    def initialize_ratings(self, player_database: PlayerDatabase, player_id: int, 
                           ratings: np.ndarray, variances: np.ndarray, date: datetime):
        """
        Initializes the ratings and variances arrays for all players in the player database.

        Args:
            - player_database (PlayerDatabase): The database of players.
            - player_id (int): The ID of the player for whom the ratings are being initialized.
            - ratings (np.ndarray): The ratings array to be initialized.
            - variances (np.ndarray): The variances array to be initialized.
            - date (datetime): The date at which the ratings are being initialized.
        """
        for player in player_database:
            player_rating = player.get_rating_at_date(date, next=player_id is not None)
            ratings[player.id, :, 0] = player_rating.rating
            variances[player.id, :, 0] = player_rating.deviation ** 2
            for i, advantage in enumerate(self.advantages):
                if player_rating.has_advantage(advantage):
                    advantage_rating = player_rating.get_advantage(advantage)
                    ratings[player.id, :, i + 1] = advantage_rating.rating
                    variances[player.id, :, i + 1] = advantage_rating.deviation ** 2
                else:
                    ratings[player.id, :, i + 1] = self.default_ratings[i + 1].rating
                    variances[player.id, :, i + 1] = self.default_ratings[i + 1].deviation ** 2

    def extract_periods(self, games: List[Game], period_dates: List[datetime]):
        """
        Extracts the periods for each game based on the provided period dates.

        Args:
            - games (List[Game]): A list of Game objects representing the chess games.
            - period_dates (List[datetime]): A list of datetime objects representing the period dates.

        Returns:
            - List[int]: A list of integers representing the periods for each game.
        """
        periods = []
        for game in games:
            for i, date in enumerate(period_dates):
                if date >= game.get_date():
                    periods.append(i)
                    break
        return periods

    def get_default_ratings(self, player_database : PlayerDatabase, period_dates : List[datetime], 
                            player_id : int, linear : int, default_ratings : dict):
        """
        Get the default ratings for players based on their historical ratings.

        Args:
            - player_database (PlayerDatabase): A list of Player objects representing the player database.
            - period_dates (list): A list of datetime objects representing the period dates.
            - player_id (int): The ID of the player that is currently being updated in tournament performance.
            - linear (int): The number of period dates to consider when calculating the default ratings.
            - default_ratings (dict): A dictionary to store the default ratings for each player.
        """
        for player in player_database:
            date = period_dates[-min(linear, len(period_dates))]
            player_rating = player.get_rating_at_date(date, next=player_id is not None)
            default_rating, default_deviation = [player_rating.rating], [player_rating.deviation + self.omegas[0]]
            for i, advantage in enumerate(self.advantages):
                if player_rating.has_advantage(advantage):
                    advantage_rating = player_rating.get_advantage(advantage)
                    default_rating.append(advantage_rating.rating)
                    default_deviation.append(advantage_rating.deviation + self.omegas[i+1])
                else:
                    default_rating.append(self.default_ratings[i+1].rating)
                    default_deviation.append(self.default_ratings[i+1].deviation + self.omegas[i+1])
            default_rating = np.array(default_rating)
            default_deviation = np.array(default_deviation)
            default_ratings[player.id] = (default_rating, default_deviation)

    def compute_likelihood(self, player_rating_ : np.ndarray, 
                           games_with_ratings : List, 
                           all_ratings : np.ndarray, 
                           earliest_playing_period : int, 
                           default_rating : np.ndarray, 
                           default_deviation : np.ndarray,
                           shared_ratings : np.ndarray) -> np.ndarray:
        """
        Computes the output value based of the likelihood function for the given player

        Args:
            - player_rating_ (np.ndarray): The player's rating history.
            - games_with_ratings (list): List of games with ratings.
            - all_ratings (list): All ratings.
            - earliest_playing_period (int): The earliest playing period.
            - default_rating (np.ndarray): The default rating.
            - default_deviation (np.ndarray): The default deviation.
            - shared_ratings (np.ndarray): The shared ratings.

        Returns:
            - output (float): The computed output value.
        """
        output = 0
        output += self.markov(player_rating_)
        output += sum(self.log_prior(player_rating_[0], default_rating, default_deviation ** 2))
        for game in games_with_ratings:
            index_game = game[1] - earliest_playing_period
            p_rating = player_rating_[index_game]
            opp_rating = all_ratings[game[2], game[1]]
            combined_rating_player = p_rating[0] + sum([p_rating[1 + i] * game[3][key] for i, key in enumerate(game[3])])
            combined_rating_player += sum([shared_ratings[game[1], i] * game[5][key][0] for i, key in enumerate(game[5])])
            combined_rating_opp = opp_rating[0] + sum([opp_rating[1 + i] * game[4][key] for i, key in enumerate(game[4])])
            combined_rating_opp += sum([shared_ratings[game[1], i] * game[6][key][0] for i, key in enumerate(game[6])])
            if game[0] == 1:
                output += game[7] * self.log_win(combined_rating_player, combined_rating_opp)
            elif game[0] == 0:
                output += game[7] * self.log_win(combined_rating_opp, combined_rating_player)
            else:
                output += game[7] * self.log_tie(combined_rating_player, combined_rating_opp, game[0])
        return output

    def update_player(self, player_rating : np.ndarray, player_variance : np.ndarray, 
                      games_with_ratings : List, all_ratings : np.ndarray, 
                      earliest_playing_period : int, default_rating : float, default_deviation : float, 
                      shared_ratings : np.ndarray, shared : bool = False) -> tuple:
        """
        Computes the new player's rating and variance based on the provided inputs.

        Args:
            - player_rating (numpy.ndarray): The player's current rating.
            - player_variance (numpy.ndarray): The player's current variance.
            - games_with_ratings (list): A list of games with ratings.
            - all_ratings (numpy.ndarray): All ratings for all players.
            - earliest_playing_period (int): The earliest playing period.
            - default_rating (float): The default rating.
            - default_deviation (float): The default deviation.
            - shared_ratings (numpy.ndarray): The shared ratings.
            - shared (bool): If True, the update is done for the shared things and not for the other player specific ratings. Defaults to False.

        Returns:
            - tuple: A tuple containing the updated player rating, player variance, derivative shared ratings, and second derivative shared ratings.
        """
        player_rating_ = player_rating.copy()
        player_rating_ = player_rating_[earliest_playing_period:]
        derivative_shared_ratings = np.zeros((shared_ratings.shape[0] * shared_ratings.shape[1],))
        second_derivative_shared_ratings = np.zeros((derivative_shared_ratings.shape[0], derivative_shared_ratings.shape[0]))
        derivative = np.zeros((player_rating_.shape[0] * player_rating_.shape[1],))
        second_derivative = np.zeros((derivative.shape[0], derivative.shape[0]))
        if not shared:
            derivative += self.derivative_markov(player_rating_).reshape(-1)
            # note that the derivative of the prior is always 0 since we explicitly model the first rating as the prior
            derivative[:player_rating_.shape[1]] += self.derivative_prior(player_rating_[0], 
                                                    default_rating, 
                                                    default_deviation ** 2).reshape(-1)
            sec_der_markov = self.second_derivative_markov(player_rating_)
            for i in range(sec_der_markov.shape[2]):
                second_derivative[i :: player_rating_.shape[1], i :: player_rating_.shape[1]] += sec_der_markov[:, :, i]
            # the second derivative with respect to the prior is not 0, since it still appears in the second derivative of the log likelihood
            sec_der_prior = self.second_derivative_prior(player_rating_[0], 
                                                                        default_rating, 
                                                                        default_deviation ** 2)
            for i in range(sec_der_prior.shape[0]):
                second_derivative[i, i] += sec_der_prior[i]
        
        for game in games_with_ratings:
            index_game = game[1] - earliest_playing_period
            p_rating = player_rating[game[1]]
            opp_rating = all_ratings[game[2], game[1]]
            
            combined_rating_player = p_rating[0] + sum([p_rating[1 + i] * game[3][key] for i, key in enumerate(game[3])])
            combined_rating_player += sum([shared_ratings[game[1], i] * game[5][key][0] for i, key in enumerate(game[5])])
            combined_rating_opp = opp_rating[0] + sum([opp_rating[1 + i] * game[4][key] for i, key in enumerate(game[4])])
            combined_rating_opp += sum([shared_ratings[game[1], i] * game[6][key][0] for i, key in enumerate(game[6])])
            if game[0] == 1:
                derivative_here = game[7] * self.derivative_log_win(combined_rating_player, combined_rating_opp)
                second_derivative_here = game[7] * self.second_derivative_log_win(combined_rating_player, combined_rating_opp)
            elif game[0] == 0:
                derivative_here = game[7] * self.derivative_log_loss(combined_rating_player, combined_rating_opp)
                second_derivative_here = game[7] * self.second_derivative_log_loss(combined_rating_player, combined_rating_opp)
            else:
                derivative_here = game[7] * self.derivative_log_tie(combined_rating_player, combined_rating_opp, game[0])
                second_derivative_here = game[7] * self.second_derivative_log_tie(combined_rating_player, combined_rating_opp, game[0])
            
            if not shared:
                multiplication_array_player = np.array([1] + [game[3][key] for key in game[3]])
                index_0 = index_game * player_rating_.shape[1]
                index_1 = index_0 + player_rating_.shape[1]
                derivative[index_0 : index_1] += derivative_here * multiplication_array_player
                mult_array = np.matmul(multiplication_array_player.reshape(-1, 1), multiplication_array_player.reshape(1, -1))
                second_derivative[index_0 : index_1, index_0 : index_1] += second_derivative_here * mult_array
            else:
                shared_multiplication_array = np.zeros(shared_ratings.shape[1])
                for i, key in enumerate(game[5]):
                    shared_multiplication_array[game[5][key][1]] += game[5][key][0]
                index_0 = game[1] * len(self.shared_advantages)
                index_1 = index_0 + len(self.shared_advantages)
                derivative_shared_ratings[index_0 : index_1] += derivative_here * shared_multiplication_array
                mult_array = np.matmul(shared_multiplication_array.reshape(-1, 1), shared_multiplication_array.reshape(1, -1))
                second_derivative_shared_ratings[index_0 : index_1, index_0 : index_1] += second_derivative_here * mult_array
        
        if shared:
            return None, None, derivative_shared_ratings, second_derivative_shared_ratings

        # check invertibility
        if np.linalg.det(second_derivative) != 0:
            second_derivative_inv = np.linalg.inv(second_derivative)
            # take the inverse of the double derivative
            step = np.matmul(second_derivative_inv, derivative).reshape(player_rating_.shape)
            step_size = self.apply_armijos_rule(player_rating_, step, derivative,
                                                games_with_ratings, all_ratings, earliest_playing_period, 
                                                    default_rating, default_deviation, shared_ratings)

            
            if len(games_with_ratings) > 0:
                # update the rating
                player_rating_ -= step_size * step
                # put the default rating back in the beginning
                player_rating[earliest_playing_period:] = player_rating_
                for i, omega in enumerate(self.omegas):
                    if omega < self.min_omega: # force the rating to be the same
                        player_rating[earliest_playing_period:, i] = np.mean(player_rating[earliest_playing_period:, i])
            # Note that second_derivative_inv is the inverse of the fisher information matrix, an appropriate estimate of the variance of an MLE
            # is this inverse, so we use it as the variance of the rating
            diag = np.diag(second_derivative_inv).reshape(player_rating_.shape)
            player_variance[earliest_playing_period:] = - diag

        return player_rating, player_variance, derivative_shared_ratings, second_derivative_shared_ratings

    def apply_armijos_rule(self, x, descent_direction, derivative, *args):
        """
        Apply the Armijo's rule to determine the step size for gradient descent.

        Parameters:
            x (numpy.ndarray): The current position in the optimization space.
            descent_direction (numpy.ndarray): The direction of descent.
            derivative (numpy.ndarray): The derivative of the function at the current position.
            *args: Additional arguments to be passed to the compute_likelihood function.

        Returns:
            float: The step size determined by the Armijo's rule.
        """
        step_size = 1
        step_size_sigma_0 = None
        func = self.compute_likelihood(x, *args)

        step_func = self.compute_likelihood(x - step_size * descent_direction, *args)
        step_element = self.sigma_armijo * np.dot(derivative.reshape(-1), descent_direction.reshape(-1))
        # use Armijo Rule to ensure convergence
        while func - step_size * step_element >= step_func:
            if func <= step_func and step_size_sigma_0 is None:
                step_size_sigma_0 = step_size
            step_size *= 0.9
            if step_size < 1e-3:
                step_size = 0
                break
            step_func = self.compute_likelihood(x - step_size * descent_direction, *args)
        if step_size > 0 or step_size_sigma_0 is None:
            return step_size
        return step_size_sigma_0
    
    def assign_submatrix(self, matrix : np.ndarray, a : int, b : int, t : int, other_matrix : np.ndarray):
        """
        Assigns a submatrix to a larger matrix based on given parameters.

        Parameters:
        matrix (numpy.ndarray): The larger matrix to assign the submatrix to.
        a (int): The starting index for the submatrix assignment.
        b (int): The size of each submatrix block.
        t (int): The step size for iterating over the larger matrix.
        other_matrix (numpy.ndarray): The submatrix to assign.

        Raises:
            ValueError: If the shape of other_matrix does not match the required dimensions for the submatrix assignment.
        """

        # Calculate the indices
        max_index = matrix.shape[0]
        indices = np.array([i for j in range(a, max_index, t) for i in range(j, min(j + b, max_index))])

        # Check if other_matrix has the appropriate shape
        if other_matrix.shape == (len(indices), len(indices)):
            matrix[np.ix_(indices, indices)] = other_matrix
        else:
            raise ValueError("other_matrix does not match the required dimensions for the submatrix assignment")
        
    def compute_full_variance_matrix(self, all_ratings : np.ndarray, shared_ratings : np.ndarray, default_ratings, default_deviations, games_per_player, 
                                     max_period):
        shape = all_ratings.shape[0] * all_ratings.shape[2] + shared_ratings.shape[1]
        variance_inverse_matrix = np.zeros((shape, shape))

        
    def compute_final_variances_player(self, player_rating : np.ndarray, player_variance : np.ndarray, 
                      games_with_ratings : List, all_ratings : np.ndarray, 
                      earliest_playing_period : int, default_rating : float, default_deviation : float, 
                      shared_ratings : np.ndarray, shared_ratings_second_derivative : np.ndarray) -> np.ndarray:
        """
        Computes the final variances for the player based on the provided inputs. Apart from the player rating and correlations between them, this
        function also takes into account the correlations between the player ratings and the shared ratings for the computation of the inverse.

        Args:
            - player_rating (numpy.ndarray): The player's current rating.
            - player_variance (numpy.ndarray): The player's current variance.
            - games_with_ratings (list): A list of games with ratings.
            - all_ratings (np.ndarray): All ratings for all players.
            - earliest_playing_period (int): The earliest playing period.
            - default_rating (float): The default rating.
            - default_deviation (float): The default deviation.
            - shared_ratings (np.ndarray): The shared ratings.
            - shared_ratings_second_derivative (np.ndarray): The second derivative of the shared ratings.

        Returns:
            - tuple: The computed variances
        """
        player_rating_ = player_rating.copy()
        player_rating_ = player_rating_[earliest_playing_period:]

        size_per_time = len(player_rating_[0]) + len(self.shared_advantages)
        size = size_per_time * len(player_rating_)
        variance_inverse_matrix = np.zeros((size, size))

        prior = self.second_derivative_prior(player_rating_[0], default_rating, default_deviation ** 2).reshape(-1)

        for i in range(len(prior)):
            variance_inverse_matrix[i, i] += prior[i]

        markov = self.second_derivative_markov(player_rating_)

        for i in range(len(player_rating_[0])):
            variance_inverse_matrix[i :: size_per_time, i :: size_per_time] += markov[:, :, i]

        if len(self.shared_advantages) > 0:
            size_per_time_here = len(self.shared_advantages)
            self.assign_submatrix(
                variance_inverse_matrix,
                len(player_rating_[0]),
                len(self.shared_advantages),
                size_per_time,
                shared_ratings_second_derivative[size_per_time_here * earliest_playing_period:, size_per_time_here * earliest_playing_period:]
            )
        B_vals = dict()
        for game in games_with_ratings:
            index_game = game[1] - earliest_playing_period
            p_rating = player_rating[game[1]]
            opp_rating = all_ratings[game[2], game[1]]
            
            combined_rating_player = p_rating[0] + sum([p_rating[1 + i] * game[3][key] for i, key in enumerate(game[3])])
            combined_rating_player += sum([shared_ratings[game[1], i] * game[5][key][0] for i, key in enumerate(game[5])])
            combined_rating_opp = opp_rating[0] + sum([opp_rating[1 + i] * game[4][key] for i, key in enumerate(game[4])])
            combined_rating_opp += sum([shared_ratings[game[1], i] * game[6][key][0] for i, key in enumerate(game[6])])
            if game[0] == 1:
                second_derivative_here = game[7] * self.second_derivative_log_win(combined_rating_player, combined_rating_opp)
            elif game[0] == 0:
                second_derivative_here = game[7] * self.second_derivative_log_loss(combined_rating_player, combined_rating_opp)
            else:
                second_derivative_here = game[7] * self.second_derivative_log_tie(combined_rating_player, combined_rating_opp, game[0])

            multiplication_array = np.array([1] + [game[3][key] for key in game[3]])
            shared_multiplication_array = np.zeros(shared_ratings.shape[1])

            B_vals[game[2]] = B_vals.get(game[2], 0) + second_derivative_here
            for i, key in enumerate(game[5]):
                shared_multiplication_array[game[5][key][1]] += game[5][key][0]
            multiplication_array = np.concatenate([multiplication_array, shared_multiplication_array])
            mult_variances = np.matmul(multiplication_array.reshape(-1, 1), multiplication_array.reshape(1, -1))
            variances_here = second_derivative_here * mult_variances
            variances_here[len(player_rating_[0]):, len(player_rating_[0]):] = 0 # has already been included
            variance_inverse_matrix[index_game * size_per_time : (index_game + 1) * size_per_time, index_game * size_per_time : (index_game + 1) * size_per_time] += second_derivative_here * mult_variances

        if np.linalg.det(variance_inverse_matrix) != 0:
            variance_matrix = np.linalg.inv(variance_inverse_matrix)
        else:
            variance_matrix = np.zeros(variance_inverse_matrix.shape)

        diag = np.diag(variance_matrix)
        if len(self.shared_advantages) > 0:
            return - diag[-size_per_time:-size_per_time + len(player_rating_[0])]
        return - diag[-size_per_time:]


    def gamma(self, rating : float) -> float:
        """
        Calculates the gamma value based on the given rating. Gamma is the normalized strength of the player.

        Args:
            - rating (float): The rating value.

        Returns:
            - float: The calculated gamma value.
        """
        return 10 ** (rating / self.default_change)

    def derivative_gamma(self, rating : float) -> float:
        """
        Calculates the derivative of the gamma function with respect to the rating.

        Args:
            - rating (float): The rating value.

        Returns:
            - float: The derivative of the gamma function with respect to the rating.
        """
        return self.gamma(rating) * np.log(10) / self.default_change
    
    def second_derivative_gamma(self, rating : float) -> float:
        """
        Calculates the second derivative of the gamma function with respect to the rating.

        Args:
            - rating (float): The rating value.

        Returns:
            - float: The second derivative of the gamma function.
        """
        return self.gamma(rating) * (np.log(10) ** 2) / (self.default_change ** 2)
    
    def log_prior(self, rating : np.ndarray, default_rating : np.ndarray, variance : np.ndarray) -> np.ndarray:
        """
        Calculates the log prior probability of a rating given a default rating and variance.

        Args:
            - rating (float): The rating value.
            - default_rating (float): The default rating value.
            - variance (float): The variance value.

        Returns:
            - float: The log prior probability.
        """
        return np.where(variance < self.max_variance, - (rating - default_rating) ** 2 / (2 * variance), 0)
    
    def derivative_prior(self, rating : np.ndarray, default_rating : np.ndarray, variance : np.ndarray) -> np.ndarray:
        """
        Calculates the derivative of the prior distribution.

        Args:
            - rating (float): The player's current rating.
            - default_rating (float): The default rating for the player.
            - variance (float): The variance of the prior distribution.

        Returns:
            - float: The derivative of the prior distribution.
        """
        return np.where(variance < self.max_variance, (default_rating - rating) / variance, 0)
    
    def second_derivative_prior(self, rating : np.ndarray, default_rating : np.ndarray, variance : np.ndarray) -> np.ndarray:
        """
        Calculate the second derivative of the prior distribution.

        Args:
            - rating (float): The player's rating.
            - default_rating (float): The default rating.
            - variance (float): The variance of the prior distribution.

        Returns:
            - float: The second derivative of the prior distribution.
        """
        return np.where(variance < self.max_variance, - 1 / variance, 0)
    
    def markov(self, rating_history : np.ndarray, omegas : List[float] = None) -> float:
        """
        Calculates the Markov chain likelihood for a given rating history.

        The Markov score measures the smoothness of the rating progression over time.
        It is calculated by summing the squared differences between consecutive ratings,
        weighted by the inverse of the square of the smoothing factor (omega).

        Args:
            - rating_history (list): A list of rating values over time.

        Returns:
            - markov (float): The Markov score for the given rating history.
        """
        if omegas is None:
            omegas = self.omegas
        markov = 0
        for j in range(rating_history.shape[-1]):
            for i in range(1, len(rating_history)):
                markov += 1 / 2  * (rating_history[i, j] - rating_history[i - 1, j]) ** 2 / (omegas[j] ** 2)
        return markov
    
    def derivative_markov(self, rating_history : np.ndarray, omegas : List[float] = None) -> float:
        """
        Calculates the derivative of the rating history using the Markov method.

        Args:
            - rating_history (list): A list of rating values over time.

        Returns:
            - numpy.ndarray: An array containing the derivative of the rating history.
        """
        if omegas is None:
            omegas = self.omegas
        derivative = np.zeros(rating_history.shape)
        for j in range(rating_history.shape[-1]):
            for i in range(1, len(rating_history)):
                derivative[i - 1, j] += (rating_history[i, j] - rating_history[i - 1, j]) / (omegas[j] ** 2)
                derivative[i, j] -= (rating_history[i, j] - rating_history[i - 1, j]) / (omegas[j] ** 2)
        return derivative
    
    def second_derivative_markov(self, rating_history : np.ndarray, omegas : List[float] = None) -> float:
        """
        Calculates the second derivative matrix for a given rating history.

        Args:
            - rating_history (numpy.ndarray): The rating history array.

        Returns:
            - numpy.ndarray: The second derivative matrix.

        """
        if omegas is None:
            omegas = self.omegas
        second_derivative = np.zeros((len(rating_history), len(rating_history), rating_history.shape[-1]))
        for j in range(rating_history.shape[-1]):
            for i in range(1, len(rating_history)):
                second_derivative[i - 1, i - 1, j] -= 1 / (omegas[j] ** 2)
                second_derivative[i, i, j] -= 1 / (omegas[j] ** 2)
                second_derivative[i - 1, i, j] += 1 / (omegas[j] ** 2)
                second_derivative[i, i - 1, j] += 1 / (omegas[j] ** 2)
        return second_derivative
    
    def log_win(self, rating1 : float, rating2 : float) -> float:
        """
        Computes the log probability of a win between two players.

        Args:
            - rating1 (int): The rating of player 1.
            - rating2 (int): The rating of player 2.

        Returns:
            - float: The log probability of a win.
        """
        raise NotImplementedError
    
    def log_tie(self, rating1 : float, rating2 : float, result : float = 0.5) -> float:
        """
        Computes the log probability of a tie between two players.

        Args:
            - rating1 (int): The rating of player 1.
            - rating2 (int): The rating of player 2.
            - result (float): The result of the game.

        Returns:
            - float: The log probability of a tie.
        """
        raise NotImplementedError
    
    def win_prob(self, rating1 : float, rating2 : float) -> float:
        """
        Calculates the win probability of a player with rating1 against a player with rating2.

        Args:
            - rating1 (float): The rating of the first player.
            - rating2 (float): The rating of the second player.

        Returns:
            - float: The win probability of the first player.
        """
        raise NotImplementedError
    
    def tie_prob(self, rating1 : float, rating2 : float) -> float:
        """
        Calculates the probability of a tie between two players with the given ratings.

        Args:
            - rating1 (float): The rating of the first player.
            - rating2 (float): The rating of the second player.

        Returns:
            - float: The probability of a tie between the two players.
        """
        raise NotImplementedError
    
    def derivative_log_win(self, rating1 : float, rating2 : float) -> float:
        """
        Calculates the derivative of the logarithmic win probability between two players.

        Args:
            - rating1 (float): The rating of player 1.
            - rating2 (float): The rating of player 2.

        Returns:
            - float: The derivative of the logarithmic win probability.
        """
        raise NotImplementedError
    
    def second_derivative_log_win(self, rating1 : float, rating2 : float) -> float:
        """
        Calculates the second derivative of the logarithmic win probability
        between two chess players based on their ratings.

        Args:
            - rating1 (float): The rating of the first player.
            - rating2 (float): The rating of the second player.

        Returns:
            - float: The second derivative of the logarithmic win probability.
        """
        raise NotImplementedError
    
    def derivative_log_loss(self, rating1 : float, rating2 : float) -> float:
        """
        Calculates the derivative of the log loss function with respect to the rating difference.

        Args:
            - rating1 (float): The rating of player 1.
            - rating2 (float): The rating of player 2.

        Returns:
            - float: The derivative of the log loss function.
        """
        raise NotImplementedError
    
    def second_derivative_log_loss(self, rating1 : float, rating2 : float) -> float:
        """
        Calculate the second derivative of the log loss between two ratings.

        Args:
            - rating1 (float): The first rating.
            - rating2 (float): The second rating.

        Returns:
            - float: The second derivative of the log loss.
        """
        raise NotImplementedError
    
    def derivative_log_tie(self, rating1 : float, rating2 : float, result : float = 0.5) -> float:
        """
        Calculates the derivative of the log-likelihood function for a tie outcome.

        Args:
            - rating1 (float): The rating of player 1.
            - rating2 (float): The rating of player 2.
            - result (float): The result of the game.

        Returns:
            - float: The derivative of the log-likelihood function for a tie outcome.
        """
        raise NotImplementedError
    
    def second_derivative_log_tie(self, rating1 : float, rating2 : float, result : float = 0.5) -> float:
        """
        Calculate the second derivative of the log-likelihood function for a tie outcome.

        Args:
            - rating1 (float): The rating of player 1.
            - rating2 (float): The rating of player 2.
            - result (float): The result of the game.

        Returns:
            - float: The second derivative of the log-likelihood function.
        """
        raise NotImplementedError