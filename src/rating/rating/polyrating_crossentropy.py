import numpy as np

from .polyrating_base import Polyrating

class PolyratingCrossEntropy(Polyrating):
    def __init__(self, *args, **kwargs) -> 'PolyratingCrossEntropy':
        """
        Class representing the Whole History Rating Cross Entropy.

        This class extends the Polyrating class and provides methods for calculating win probabilities,
        logarithmic win values, tie probabilities, logarithmic tie values, derivatives, and second derivatives
        for the cross entropy rating system. In this system, the cross entropy loss of the games is minimized,
        where a tie counts as half a win and half a loss.

        Attributes:
            - Inherits attributes from the Polyrating class.
        """
        if 'allow_complex_result' in kwargs:
            del kwargs['allow_complex_result']
        super().__init__(*args, **kwargs, allow_complex_result=True)

    def win_prob(self, rating1 : float, rating2 : float) -> float:
        gamma1 = self.gamma(rating1)
        gamma2 = self.gamma(rating2)
        return gamma1 / (gamma1 + gamma2)
    
    def log_win(self, rating1 : float, rating2 : float) -> float:
        gamma1 = self.gamma(rating1)
        gamma2 = self.gamma(rating2)
        return np.log(gamma1) - np.log(gamma1 + gamma2)
    
    def tie_prob(self, rating1 : float, rating2 : float) -> float:
        return 0 # important for the expected score
    
    def log_tie(self, rating1 : float, rating2 : float, result : float = 0.5) -> float:
        return result * self.log_win(rating1, rating2) + (1 - result) * self.log_win(rating2, rating1)
    
    def derivative_log_win(self, rating1 : float, rating2 : float) -> float:
        gamma1 = self.gamma(rating1)
        gamma2 = self.gamma(rating2)
        derivative1 = self.derivative_gamma(rating1)
        term1 = 1 / gamma1 * derivative1
        term2 = - 1 / (gamma1 + gamma2) * derivative1
        return term1 + term2
    
    def second_derivative_log_win(self, rating1 : float, rating2 : float) -> float:
        gamma1 = self.gamma(rating1)
        gamma2 = self.gamma(rating2)
        derivative1 = self.derivative_gamma(rating1)
        second_derivative1 = self.second_derivative_gamma(rating1)
        #  term1 = -1 / gamma1 ** 2 * derivative1 ** 2 + 1 / gamma1 * second_derivative1 # this is actually always 0
        term2 = 1 / (gamma1 + gamma2) ** 2 * derivative1 ** 2 - 1 / (gamma1 + gamma2) * second_derivative1
        return term2
    
    def derivative_log_loss(self, rating1 : float, rating2 : float) -> float:
        gamma1 = self.gamma(rating1)
        gamma2 = self.gamma(rating2)
        derivative1 = self.derivative_gamma(rating1)
        return - 1 / (gamma1 + gamma2) * derivative1
    
    def second_derivative_log_loss(self, rating1 : float, rating2 : float) -> float:
        return self.second_derivative_log_win(rating1, rating2)
        # gamma1 = self.gamma(rating1)
        # gamma2 = self.gamma(rating2)
        # derivative1 = self.derivative_gamma(rating1)
        # second_derivative1 = self.second_derivative_gamma(rating1)
        # term2 = 1 / (gamma1 + gamma2) ** 2 * derivative1 ** 2 - 1 / (gamma1 + gamma2) * second_derivative1
        # return term2
    
    def derivative_log_tie(self, rating1 : float, rating2 : float, result : float = 0.5) -> float:
        return result * self.derivative_log_win(rating1, rating2) + (1 - result) * self.derivative_log_loss(rating1, rating2)
    
    def second_derivative_log_tie(self, rating1 : float, rating2 : float, result : float = 0.5) -> float:
        return self.second_derivative_log_win(rating1, rating2)
        # return 1 / 2 * self.second_derivative_log_win(rating1, rating2) + 1 / 2 * self.second_derivative_log_loss(rating1, rating2)
    
