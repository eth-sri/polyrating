import numpy as np
from typing import List

from ..objects import Rating, Player
from ..objects.rating import DefaultRating
from .glicko import Glicko


class Glicko2(Glicko):
    def __init__(self, tau : float = 0.7, conversion_constant : float = 173.7178, 
                 max_exp_value=10, **kwargs) -> 'Glicko2':
        """
        Glicko2 rating system implementation.

        This class extends the base Glicko class and provides methods for converting ratings, updating player ratings based on match results,
        and calculating various values used in the Glicko2 rating system.

        Attributes:
            - tau (float): The system constant.
            - conversion_constant (float): The conversion constant used to convert ratings.
            - max_exp_value (int): The maximum value for the exponent in the expected score formula. Prevents overflow errors.

        Args:
            - tau (float, optional): The system constant. Defaults to 0.7.
            - conversion_constant (float, optional): The conversion constant used to convert ratings. Defaults to 173.7178.
            - max_exp_value (int, optional): The maximum value for the exponent in the expected score formula. Prevents overflow errors. Defaults to 10.
            **kwargs: Additional keyword arguments to be passed to the base class.

        """
        super().__init__(tau=tau, conversion_constant=conversion_constant, 
                         max_exp_value=max_exp_value, **kwargs)

    def convert_rating(self, rating : float, deviation : float, volatility : float, 
                       default_rating : DefaultRating) -> tuple:
        """
        Converts a rating to a standardized value.

        Args:
            - rating (float): The rating to be converted.
            - deviation (float): The rating deviation.
            - volatility (float): The volatility of the rating.
            - default_rating (DefaultRating): The default rating value.

        Returns:
            - tuple: A tuple containing the converted rating, deviation, and volatility.

        """
        new_rating = (rating - default_rating.rating) / self.conversion_constant
        new_deviation = deviation / self.conversion_constant
        return new_rating, new_deviation, volatility
    
    def inverse_convert_rating(self, rating : float, deviation : float, 
                               volatility : float, default_rating : DefaultRating) -> tuple:
        """
        Converts a rating back to its original value.

        Args:
            - rating (float): The converted rating.
            - deviation (float): The converted deviation.
            - volatility (float): The volatility of the rating.
            - default_rating (DefaultRating): The default rating value.

        Returns:
            - tuple: A tuple containing the original rating, deviation, and volatility.
        """
        new_rating = rating * self.conversion_constant + default_rating.rating
        new_deviation = deviation * self.conversion_constant
        return new_rating, new_deviation, volatility

    def update_player(self, player : Player, opponents : List[Player], scores : List[float]) -> Rating:
        """
        Returns the updated rating, deviation, and volatility of a player based on the results of their matches.

        Args:
            - player (Player): The player whose rating, deviation, and volatility are being updated.
            - opponents (list): A list of opponents the player has played against.
            - scores (list): A list of scores corresponding to the matches against the opponents.

        Returns:
            - tuple: A tuple containing the updated rating, deviation, and volatility of the player.
        """
        rating_opponents = [self.convert_rating(opponent.get_rating().rating, 
                                                opponent.get_rating().deviation,
                                                opponent.get_rating().volatility,
                                                opponent.get_rating().default_rating) for opponent in opponents]
        converted_rating = self.convert_rating(player.get_rating().rating, 
                                               player.get_rating().deviation,
                                               player.get_rating().volatility,
                                               player.get_rating().default_rating)
        if len(opponents) > 0:
            delta = self.delta(converted_rating[0], rating_opponents, scores)
            v = self.v(converted_rating[0], rating_opponents)
            new_sigma = self.solve_for_x(delta, v, converted_rating[1], converted_rating[2])
            denom = (converted_rating[1] ** 2 + new_sigma ** 2)
            new_phi = 1 / np.sqrt(1 / denom + 1 / v)
            sum_new_rating = delta / v
            new_rating = converted_rating[0] + new_phi ** 2 * sum_new_rating
        else:
            new_sigma = converted_rating[2]
            new_phi = np.sqrt((converted_rating[1] ** 2 + new_sigma ** 2))
            new_rating = converted_rating[0]
        
        new_rating = self.inverse_convert_rating(new_rating, new_phi, new_sigma, 
                                                 player.get_rating().default_rating)
        return new_rating

    def v(self, mu : float, rating_opponents : List[tuple]) -> float:
        """
        Calculates the estimated variance of the player's rating.

        Args:
            - mu (float): The player's rating.
            - rating_opponents (list): List of opponents' ratings.

        Returns:
            - float: The estimated variance of the player's rating.
        """
        result = 0
        for rating in rating_opponents:
            E = self.E(mu, rating[0], rating[1])
            g = self.g(rating[1])
            result += g ** 2 * E * (1 - E)
        
        return (result) ** -1
    
    def delta(self, mu : float, rating_opponents : List[tuple], scores : List[float]) -> float:
        """
        Calculates the delta value for the Glicko rating system.

        Args:
            - mu (float): The player's rating.
            - rating_opponents (list): List of opponents' ratings.
            - scores (list): List of scores (0 for a loss, 0.5 for a draw, 1 for a win).

        Returns:
            - float: The calculated delta value.

        """
        result = 0
        for rating, s in zip(rating_opponents, scores):
            g = self.g(rating[1])
            E = self.E(mu, rating[0], rating[1])
            result += g * (s - E)
        return self.v(mu, rating_opponents) * result

    def g(self, phi : float) -> float:
        """
        Calculate the g factor for the Glicko rating system.

        Args:
            - phi (float): The rating deviation.

        Returns:
            - float: The g factor.
        """
        return 1 / np.sqrt(1 + 3 * phi ** 2 / np.pi ** 2)
    
    def E(self, mu : float, mu_j : float, phi_j : float) -> float:
        """
        Calculates the expected outcome of a match between two players.

        Args:
            - mu (float): The rating of the player.
            - mu_j (float): The rating of the opponent player.
            - phi_j (float): The rating deviation of the opponent player.

        Returns:
            - float: The expected outcome of the match between the two players.
        """
        exponent = -self.g(phi_j) * (mu - mu_j)
        exponent = np.clip(exponent, -self.max_exp_value, self.max_exp_value)
        return 1 / (1 + np.exp(exponent))
    
    def f(self, x : float, delta : float, v : float, phi : float, sigma : float) -> float:
        """
        Calculate the value of the function f.

        Args:
            - x: The input value.
            - delta: The delta value.
            - v: The v value.
            - phi: The phi value.
            - sigma: The sigma value.

        Returns:
            - result: The calculated result of the function f.
        """
        e_x = np.exp(x)
        result = (e_x * (delta ** 2 - phi ** 2 - v - e_x) / (2 * (phi ** 2 + v + e_x) ** 2) - (x - np.log(sigma ** 2)) / self.tau ** 2)
        return result
    
    def solve_for_x(self, delta : float, v : float, phi : float, sigma : float, epsilon : float = 1e-6) -> float:
        """
        Find the 0 of the function f using the algorithm proposed by the Glicko2 system.

        Args:
            - delta (float): The difference between the actual outcome and the expected outcome.
            - v (float): The variance of the actual outcome.
            - phi (float): The rating deviation of the player.
            - sigma (float): The system constant.
            - epsilon (float, optional): The desired level of precision for the solution. Defaults to 1e-6.

        Returns:
            - x (float): The value of x calculated using the Glicko algorithm.
        """
        A = np.log(sigma ** 2)
        if delta ** 2 > phi ** 2 + v:
            B = np.log(delta ** 2 - phi ** 2 - v)
        else:
            k = 1
            while self.f(-phi * k + A, delta, v, phi, sigma) < 0:
                k += 1
            B = - phi * k + A
        
        f_a = self.f(A, delta, v, phi, sigma)
        f_b = self.f(B, delta, v, phi, sigma)
        while np.abs(B - A) > epsilon:
            C = A + (A - B) * f_a / (f_b - f_a)
            f_c = self.f(C, delta, v, phi, sigma)
            if f_c * f_b < 0:
                A = B
                f_a = f_b
            else:
                f_a = f_a / 2
            B = C
            f_b = f_c
        return np.exp(A / 2)