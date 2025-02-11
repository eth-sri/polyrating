from rating import Manager, DefaultRating, PolyratingCrossEntropy, RatingPeriodEnum, DetailedLeaderboard
from rating import DEFAULT_RATING
from helper import prepare_lmsys_data, add_games_lmsys

from bootstrap import bootstrap

def compute_function(data):
    """
    Compute the function for evaluating data. Used for bootstrapping

    Parameters:
    - data: The input data to be evaluated.

    Returns:
    - The evaluation result.
    - None
    """
    DEFAULT_RATING.set_default(deviation=10 ** 5)
    return evaluate(data, ['is_hard', 'is_english', 'is_chinese', 'is_code']), None

def evaluate(data, categories, std=50):
    """
    Evaluates the performance of players based on the given data and categories.

    Parameters:
    - data (list): A list of game data.
    - categories (list): A list of categories to evaluate the players.
    - std (int): The standard deviation for the rating system. Default is 50.

    Returns:
    - leaderboard (DetailedLeaderboard): The computed leaderboard.

    """
    advantages = {category: DefaultRating(0, std) for category in categories}
    category_manager = Manager(rating_system=PolyratingCrossEntropy(advantages=advantages, epsilon=1e-3), 
                                rating_period_type=RatingPeriodEnum.MANUAL)
    add_games_lmsys(data, category_manager, categories)
    category_manager.update_rating()
    leaderboard = DetailedLeaderboard.compute(category_manager.player_database, category_manager.game_database, 
                                              category_manager.tournament_database, category_manager.rating_system)
    return leaderboard

if __name__ == '__main__':
    import argparse

    # argparse with bootstrap arguments

    parser = argparse.ArgumentParser()
    parser.add_argument('--n-bootstrap', type=int, default=1000)
    parser.add_argument('--cores', type=int, default=1)

    args = parser.parse_args()


    DEFAULT_RATING.set_default(deviation=10 ** 5)
    data = prepare_lmsys_data()
    results, _ = bootstrap(data, compute_function, args.n_bootstrap, args.cores)

    results.to_csv(f'results/leaderboard_polyrating.csv', index=False)