import trueskillthroughtime as ttt

from datetime import datetime
from typing import List

from ..objects import Player, Rating
from ..databases import GameDatabase, PlayerDatabase
from .rating_system import RatingSystem


class TrueSkillThroughTime(RatingSystem):
    def __init__(self, p_draw : float = 0.05, beta : float = 1.0, gamma : float = 0.03, sigma : float = 6.0,
                 linearized : int = None) -> 'TrueSkillThroughTime':
        """
        TrueSkillThroughTime algorithm: https://www.microsoft.com/en-us/research/wp-content/uploads/2008/01/NIPS2007_0931.pdf
        Note: because the trueskill library returns some errors when the prior mean and deviation are not (close to) 0 and 1, we ignore the DEFAULT_RATING values.

        Attributes:
            - p_draw (float): The probability of a draw.
            - beta (float): The standard deviation of the performance.
            - gamma (float): The standard deviation of the draw probability.
            - sigma (float): The standard deviation of the prior.
            - linearized (int): The number of periods to use in an update. If None, all periods are used.

        Args:
            - p_draw (float): The probability of a draw. Default is 0.05.
            - beta (float): The standard deviation of the performance. Default is 1.0.
            - gamma (float): The standard deviation of the draw probability. Default is 0.03.
            - sigma (float): The standard deviation of the prior. Default is 1.
            - linearized (int): The number of periods to use in an update. If None, all periods are used.
        """
        super().__init__(p_draw=p_draw, beta=beta, gamma=gamma, linearized=linearized, sigma=sigma)

    def period_update(self, player_database : PlayerDatabase, 
                      game_database : GameDatabase, 
                      period_dates : List[datetime], **kwargs):
        if not self.linearized is not None or len(period_dates) < self.linearized + 1:
            games = list(game_database.get_games_between_dates(period_dates[-1]))
        else:
            games = list(game_database.get_games_between_dates(period_dates[-1], period_dates[-1-self.linearized]))
        
        if self.linearized is not None:
            linear = min(self.linearized, len(period_dates))
        else:
            linear = len(period_dates)
        first_date = None
        composition, results = [], []
        for game in games:
            for _ in range(game.weight):
                if first_date is None or game.get_date() >= first_date:
                    composition.append([[game.home], [game.out]])
                    results.append([game.get_result(), 1 - game.get_result()])
        priors = dict()
        for player in player_database:
            rating = player.get_rating_at_date(period_dates[-linear], next=False)
            if rating.rating == player.get_rating().default_rating.rating and rating.deviation == player.get_rating().default_rating.deviation:
                priors[player.id] = ttt.Player(ttt.Gaussian(0, self.sigma), self.beta, self.gamma)
            else:
                priors[player.id] = ttt.Player(ttt.Gaussian(rating.rating, rating.deviation), self.beta, self.gamma)

        history = ttt.History(composition=composition, results=results, beta=self.beta, gamma=self.gamma, 
                              p_draw=self.p_draw, priors=priors)
        history.convergence()
        
        lcs = history.learning_curves()
        for learning_curve in lcs:
            player = player_database[learning_curve]
            mu, sigma = lcs[learning_curve][-1][1].mu, lcs[learning_curve][-1][1].sigma
            player.get_rating().update(mu, sigma)
    
    def compute_expected_score_rating(self, player_rating : Rating, 
                                      opponent_rating : Rating, 
                                      is_home : bool) -> float:
        p1 = ttt.Player(ttt.Gaussian(player_rating.rating, player_rating.deviation), beta=self.beta, gamma=self.gamma, prior_draw=self.p_draw)
        p2 = ttt.Player(ttt.Gaussian(opponent_rating.rating, opponent_rating.deviation), beta=self.beta, gamma=self.gamma, prior_draw=self.p_draw)
        game = ttt.Game([[p1], [p2]])
        return game.evidence
    
    def compute_tournament_performance(self, player : Player, 
                                       tournament_id : int, 
                                       tournament_date : datetime, 
                                       game_database : GameDatabase, 
                                       player_database : PlayerDatabase, 
                                       next : bool = False, rating_check : int = 20) -> Rating:
        return super().compute_tournament_performance(player, tournament_id, tournament_date, game_database, player_database, next, rating_check)