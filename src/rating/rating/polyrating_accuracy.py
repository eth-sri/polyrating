import numpy as np

from .polyrating_base import Polyrating

class PolyratingAccuracy(Polyrating):
    def __init__(self, clip_value=1e-15, *args, **kwargs) -> 'PolyratingAccuracy':
        """
        Whole history rating system with the accuracy function.
        """
        if 'allow_complex_result' in kwargs:
            del kwargs['allow_complex_result']
        super().__init__(*args, **kwargs, clip_value=clip_value, allow_complex_result=True)
    
    def win_prob(self, rating1 : float, rating2 : float) -> float:
        return np.clip(1 / 2 * (1 + rating1 - rating2), self.clip_value, 1 - self.clip_value)
    
    def log_win(self, rating1 : float, rating2 : float) -> float:
        return np.log(self.win_prob(rating1, rating2))
    
    def log_tie(self, rating1 : float, rating2 : float, result : float = 0.5) -> float:
        return result * np.log(self.win_prob(rating1, rating2)) + (1 - result) * np.log(self.win_prob(rating2, rating1))
    
    def tie_prob(self, rating1 : float, rating2 : float) -> float:
        return 0
    
    def derivative_log_win(self, rating1 : float, rating2 : float) -> float:
        win_prob = self.win_prob(rating1, rating2)
        if win_prob == self.clip_value or win_prob == 1 - self.clip_value:
            return 0
        return 1 / (1 + rating1 - rating2)
    
    def second_derivative_log_win(self, rating1 : float, rating2 : float) -> float:
        win_prob = self.win_prob(rating1, rating2)
        if win_prob == self.clip_value or win_prob == 1 - self.clip_value:
            return 0
        return - 1 / (1 + rating1 - rating2) ** 2
    
    def derivative_log_loss(self, rating1 : float, rating2 : float) -> float:
        return  - self.derivative_log_win(rating2, rating1)
    
    def second_derivative_log_loss(self, rating1 : float, rating2 : float) -> float:
        return self.second_derivative_log_win(rating2, rating1)
    
    def derivative_log_tie(self, rating1 : float, rating2 : float, result : float = 0.5) -> float:
        return result * self.derivative_log_win(rating1, rating2) + (1 - result) * self.derivative_log_loss(rating1, rating2)
    
    def second_derivative_log_tie(self, rating1 : float, rating2 : float, result : float = 0.5) -> float:
        return result * self.second_derivative_log_win(rating1, rating2) + (1 - result) * self.second_derivative_log_loss(rating1, rating2)