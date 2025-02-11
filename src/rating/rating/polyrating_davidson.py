import numpy as np

from .polyrating_base import Polyrating

class PolyratingDavidson(Polyrating):
    def __init__(self, theta: float = 1, *args, **kwargs) -> 'PolyratingDavidson':
        """
        A class that represents the Whole History Rating system based on the Davidson method. Davidson 
        extends the Bradley-Terry model to accommodate the possibility of ties. 
        Based on https://www.semanticscholar.org/paper/On-Extending-the-Bradley-Terry-Model-to-Accommodate-Davidson/3833525afb515fbf56c616b5ae5470247fc38fa5

        Attributes:
            - theta (float): A parameter that controls the weight of the tie probability in the rating calculation.

        Args:
            - theta (float): The value of theta parameter.
        """
        if 'allow_complex_result' in kwargs:
            del kwargs['allow_complex_result']
        super().__init__(*args, **kwargs, theta=theta, allow_complex_result=False)
    
    def denominator(self, gamma1 : float, gamma2 : float) -> float:
        return (gamma1 + gamma2 + self.theta * np.sqrt(gamma1 * gamma2))

    def win_prob(self, rating1 : float, rating2 : float) -> float:
        gamma1 = self.gamma(rating1)
        gamma2 = self.gamma(rating2)
        return gamma1 / self.denominator(gamma1, gamma2)
    
    def log_win(self, rating1 : float, rating2 : float) -> float:
        gamma1 = self.gamma(rating1)
        gamma2 = self.gamma(rating2)
        return np.log(gamma1) - np.log(self.denominator(gamma1, gamma2))
    
    def tie_prob(self, rating1 : float, rating2 : float) -> float:
        gamma1 = self.gamma(rating1)
        gamma2 = self.gamma(rating2)
        return self.theta * np.sqrt(gamma1 * gamma2) / self.denominator(gamma1, gamma2)
    
    def log_tie(self, rating1 : float, rating2 : float, result : float = 0.5) -> float:
        gamma1 = self.gamma(rating1)
        gamma2 = self.gamma(rating2)
        return np.log(self.theta * np.sqrt(gamma1 * gamma2)) - np.log(self.denominator(gamma1, gamma2))
    
    def derivative_log_win(self, rating1 : float, rating2 : float) -> float:
        gamma1 = self.gamma(rating1)
        gamma2 = self.gamma(rating2)
        derivative1 = self.derivative_gamma(rating1)
        term1 = 1 / gamma1 * derivative1
        term2 = - 1 / self.denominator(gamma1, gamma2) * (1 + self.theta / 2 * np.sqrt(gamma2 / gamma1)) * derivative1
        return term1 + term2
    
    def second_derivative_log_win(self, rating1 : float, rating2 : float) -> float:
        gamma1 = self.gamma(rating1)
        gamma2 = self.gamma(rating2)
        derivative1 = self.derivative_gamma(rating1)
        second_derivative1 = self.second_derivative_gamma(rating1)
        term1 = -1 / gamma1 ** 2 * derivative1 ** 2 + 1 / gamma1 * second_derivative1
        term2 = 1 / (self.denominator(gamma1, gamma2) ** 2) * (1 + self.theta / 2 * np.sqrt(gamma2 / gamma1)) ** 2 * derivative1 ** 2
        term3 = - 1 / self.denominator(gamma1, gamma2) * (1 + self.theta / 2 * np.sqrt(gamma2 / gamma1)) * second_derivative1
        term4 = 1 / self.denominator(gamma1, gamma2) * self.theta / 4 * np.sqrt(gamma2) * (1 / gamma1) ** (3 / 2) * derivative1 ** 2
        return term1 + term2 + term3 + term4
    
    def derivative_log_loss(self, rating1 : float, rating2 : float) -> float:
        gamma1 = self.gamma(rating1)
        gamma2 = self.gamma(rating2)
        derivative1 = self.derivative_gamma(rating1)
        term2 = - 1 / self.denominator(gamma1, gamma2) * (1 + self.theta / 2 * np.sqrt(gamma2 / gamma1)) * derivative1
        return term2
    
    def second_derivative_log_loss(self, rating1 : float, rating2 : float) -> float:
        gamma1 = self.gamma(rating1)
        gamma2 = self.gamma(rating2)
        derivative1 = self.derivative_gamma(rating1)
        second_derivative1 = self.second_derivative_gamma(rating1)
        term2 = 1 / (self.denominator(gamma1, gamma2) ** 2) * (1 + self.theta / 2 * np.sqrt(gamma2 / gamma1)) ** 2 * derivative1 ** 2
        term3 = - 1 / self.denominator(gamma1, gamma2) * (1 + self.theta / 2 * np.sqrt(gamma2 / gamma1)) * second_derivative1
        term4 = 1 / self.denominator(gamma1, gamma2) * self.theta / 4 * np.sqrt(gamma2) * (1 / gamma1) ** (3 / 2) * derivative1 ** 2
        return term2 + term3 + term4
    
    def derivative_log_tie(self, rating1 : float, rating2 : float, result : float = 0.5) -> float:
        gamma1 = self.gamma(rating1)
        gamma2 = self.gamma(rating2)
        derivative1 = self.derivative_gamma(rating1)
        term1 = 1 / (2 * gamma1) * derivative1
        term2 = - 1 / self.denominator(gamma1, gamma2) * (1 + self.theta / 2 * np.sqrt(gamma2 / gamma1)) * derivative1
        return term1 + term2
    
    def second_derivative_log_tie(self, rating1 : float, rating2 : float, result : float = 0.5) -> float:
        gamma1 = self.gamma(rating1)
        gamma2 = self.gamma(rating2)
        derivative1 = self.derivative_gamma(rating1)
        second_derivative1 = self.second_derivative_gamma(rating1)
        term1 = -1 / gamma1 ** 2 * derivative1 ** 2 + 1 / gamma1 * second_derivative1
        term2 = 1 / (self.denominator(gamma1, gamma2) ** 2) * (1 + self.theta / 2 * np.sqrt(gamma2 / gamma1)) ** 2 * derivative1 ** 2
        term3 = - 1 / self.denominator(gamma1, gamma2) * (1 + self.theta / 2 * np.sqrt(gamma2 / gamma1)) * second_derivative1
        term4 = 1 / self.denominator(gamma1, gamma2) * self.theta / 4 * np.sqrt(gamma2) * (1 / gamma1) ** (3 / 2) * derivative1 ** 2
        return term1 / 2 + term2 + term3 + term4