import numpy as np

from .polyrating_base import Polyrating

class PolyratingRao(Polyrating):
    def __init__(self, theta: float = 2 ** 0.5, *args, **kwargs) -> 'PolyratingRao':
        """
        This class represents a rating system based on the Whole History Rating (WHR) algorithm,
        extended with Rao's modification. Rao's modification extends the Taylor-Bradley model for ties
        to accommodate the possibility of ties. Based on https://www.jstor.org/stable/2282923

        Attributes:
            - theta (float): The scaling factor for the rating difference. Default is 2 ** 0.5.

        Args:
            - theta (float): The value of theta. Default is 2 ** 0.5.
        """
        if 'allow_complex_result' in kwargs:
            del kwargs['allow_complex_result']
        super().__init__(*args, **kwargs, theta=theta, allow_complex_result=False)
    
    def win_prob(self, rating1 : float, rating2 : float) -> float:
        gamma1 = self.gamma(rating1)
        gamma2 = self.gamma(rating2)
        return gamma1 / (gamma1 + self.theta * gamma2)
    
    def log_win(self, rating1 : float, rating2 : float) -> float:
        gamma1 = self.gamma(rating1)
        gamma2 = self.gamma(rating2)
        return np.log(gamma1) - np.log(gamma1 + self.theta * gamma2)
    
    def log_tie(self, rating1 : float, rating2 : float, result : float = 0.5) -> float:
        gamma1 = self.gamma(rating1)
        gamma2 = self.gamma(rating2)
        return np.log((self.theta ** 2 - 1) * gamma1 * gamma2) - np.log((self.theta * gamma1 + gamma2)) - np.log((self.theta * gamma2 + gamma1))
    
    def tie_prob(self, rating1 : float, rating2 : float) -> float:
        gamma1 = self.gamma(rating1)
        gamma2 = self.gamma(rating2)
        return (self.theta ** 2 - 1) * gamma1 * gamma2 / ((self.theta * gamma1 + gamma2) * (self.theta * gamma2 + gamma1))
    
    def derivative_log_win(self, rating1 : float, rating2 : float) -> float:
        gamma1 = self.gamma(rating1)
        gamma2 = self.gamma(rating2)
        derivative1 = self.derivative_gamma(rating1)
        return 1 / gamma1 * derivative1 - 1 / (gamma1 + self.theta * gamma2) * derivative1
    
    def second_derivative_log_win(self, rating1 : float, rating2 : float) -> float:
        gamma1 = self.gamma(rating1)
        gamma2 = self.gamma(rating2)
        derivative1 = self.derivative_gamma(rating1)
        second_derivative1 = self.second_derivative_gamma(rating1)
        first_terms = 1 / gamma1 * second_derivative1 - 1 / gamma1 ** 2 * derivative1 ** 2
        second_terms = - 1 / (gamma1 + self.theta * gamma2) * second_derivative1 + 1 / (gamma1 + self.theta * gamma2) ** 2 * derivative1 ** 2
        return first_terms + second_terms
    
    def derivative_log_loss(self, rating1 : float, rating2 : float) -> float:
        gamma1 = self.gamma(rating1)
        gamma2 = self.gamma(rating2)
        derivative1 = self.derivative_gamma(rating1)
        return - self.theta / (gamma2 + self.theta * gamma1) * derivative1
    
    def second_derivative_log_loss(self, rating1 : float, rating2 : float) -> float:
        gamma1 = self.gamma(rating1)
        gamma2 = self.gamma(rating2)
        derivative1 = self.derivative_gamma(rating1)
        second_derivative1 = self.second_derivative_gamma(rating1)
        return - self.theta / (gamma2 + self.theta * gamma1) * second_derivative1 + self.theta ** 2 / (gamma2 + self.theta * gamma1) ** 2 * derivative1 ** 2
    
    def derivative_log_tie(self, rating1 : float, rating2 : float, result : float = 0.5) -> float:
        gamma1 = self.gamma(rating1)
        gamma2 = self.gamma(rating2)
        derivative1 = self.derivative_gamma(rating1)
        return 1 / gamma1 * derivative1 - self.theta / (self.theta * gamma1 + gamma2) * derivative1 - 1 / (self.theta * gamma2 + gamma1) * derivative1
    
    def second_derivative_log_tie(self, rating1 : float, rating2 : float, result : float = 0.5) -> float:
        gamma1 = self.gamma(rating1)
        gamma2 = self.gamma(rating2)
        derivative1 = self.derivative_gamma(rating1)
        second_derivative1 = self.second_derivative_gamma(rating1)
        first_terms = 1 / gamma1 * second_derivative1 - 1 / gamma1 ** 2 * derivative1 ** 2
        second_terms = - self.theta / (self.theta * gamma1 + gamma2) * second_derivative1 + self.theta ** 2 / (self.theta * gamma1 + gamma2) ** 2 * derivative1 ** 2
        third_terms = - 1 / (self.theta * gamma2 + gamma1) * second_derivative1 + 1 / (self.theta * gamma2 + gamma1) ** 2 * derivative1 ** 2
        return first_terms + second_terms + third_terms