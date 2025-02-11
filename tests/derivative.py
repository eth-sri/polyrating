import unittest
from rating import PolyratingCrossEntropy, PolyratingDavidson, PolyratingRao, Polyrating, PolyratingAccuracy
import numpy as np

def first_derivative_numerical(f, x, h=1e-6):
    """Numerically approximate the first derivative of function f at point x."""
    return (f(x + h) - f(x - h)) / (2 * h)

class TestDerivatives(unittest.TestCase):

    def check_first_derivative(self, f, f_prime, x0):
        """Helper method to test the first derivative."""
        numerical_derivative = first_derivative_numerical(f, x0)
        analytical_derivative = f_prime(x0)
        if abs(numerical_derivative) < 5:
            self.assertAlmostEqual(numerical_derivative, analytical_derivative, places=6)
        else:
            self.assertAlmostEqual(1, analytical_derivative / numerical_derivative, places=6)

    def whole_rating_derivatives(self, rating_system : Polyrating, limits=(1000, 2500, 100)):
        """Test the first and second derivatives of the whole history rating system."""
        opponent_ratings = np.arange(limits[0], limits[1], limits[2])
        x0 = np.arange(limits[0], limits[1], limits[2])
        for x_value in x0:
            for opponent in opponent_ratings:
                f_win = lambda x: rating_system.log_win(x, opponent)
                f_loss = lambda x: rating_system.log_win(opponent, x)
                f_tie = lambda x: rating_system.log_tie(x, opponent)
                f_win_prime = lambda x: rating_system.derivative_log_win(x, opponent)
                f_lose_prime = lambda x: rating_system.derivative_log_loss(x, opponent)
                f_tie_prime = lambda x: rating_system.derivative_log_tie(x, opponent)
                f_win_double_prime = lambda x: rating_system.second_derivative_log_win(x, opponent)
                f_lose_double_prime = lambda x: rating_system.second_derivative_log_loss(x, opponent)
                f_tie_double_prime = lambda x: rating_system.second_derivative_log_tie(x, opponent)
                self.check_first_derivative(f_win, f_win_prime, x_value)
                self.check_first_derivative(f_loss, f_lose_prime, x_value)
                self.check_first_derivative(f_tie, f_tie_prime, x_value)
                self.check_first_derivative(f_win_prime, f_win_double_prime, x_value)
                self.check_first_derivative(f_lose_prime, f_lose_double_prime, x_value)
                self.check_first_derivative(f_tie_prime, f_tie_double_prime, x_value)
    
    def test_rao(self):
        self.whole_rating_derivatives(PolyratingRao())

    def test_davidson(self):
        self.whole_rating_derivatives(PolyratingDavidson())

    def test_cross_entropy(self):
        self.whole_rating_derivatives(PolyratingCrossEntropy())

    def test_accuracy(self):
        self.whole_rating_derivatives(PolyratingAccuracy(), limits=(0, 1, 0.01))


if __name__ == '__main__':
    unittest.main()