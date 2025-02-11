from rating import Manager, PolyratingCrossEntropy, PolyratingAccuracy, RatingPeriodEnum, DEFAULT_RATING, PolyratingDavidson, PolyratingRao
import numpy as np
import pandas as pd
from loguru import logger
from helper import prepare_lmsys_data, get_games_lmsys, optimal_log_loss_lmsys, log_loss_rating_system

def train(rating_system, train_data, test_data):
    """
    Trains a rating system using the provided training data and evaluates its performance on the test data.

    Args:
        rating_system (RatingSystem): The rating system to train.
        train_data (DataFrame): The training data used to train the rating system.
        test_data (DataFrame): The test data used to evaluate the performance of the rating system.

    Returns:
        float: The log loss of the rating system on the test data.
    """
    manager = Manager(
        rating_system=rating_system,
        rating_period_type=RatingPeriodEnum.MANUAL,
    )
    games = get_games_lmsys(train_data, manager)

    for game in games:
        manager.game_database.add(game)
    manager.trigger_new_period()

    manager.update_rating()

    test_games = get_games_lmsys(test_data, manager)

    return log_loss_rating_system(manager, test_games)

if __name__ == '__main__':
    DEFAULT_RATING.set_default(deviation=10 ** 5, rating=0.5)

    data = prepare_lmsys_data()

    rating_systems = {
        'accuracy': PolyratingAccuracy(epsilon=1e-4),
        'cross_entropy': PolyratingCrossEntropy(epsilon=1e-3),
        'davidson': PolyratingDavidson(epsilon=1e-3),
        'rao': PolyratingRao(epsilon=1e-3),
    }

    lengths = list(np.arange(10000, 200001, 10000))
    lengths = [4000, 6000, 8000] + lengths

    # shuffle data
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    test_data = data[10 ** 6:]
    results = dict()

    optimal = optimal_log_loss_lmsys(test_data)
    logger.info(f"Optimal: {optimal}")
    optimal_results = {
        key: train(val, test_data, test_data) for key, val in rating_systems.items()
    }

    logger.info(f"Optimal results: {optimal_results}")

    for length in lengths:
        train_data = data[:length]
        for name, rating_system in rating_systems.items():
            results.setdefault(name, []).append(train(rating_system, train_data, test_data))
            logger.info(f"Length: {length}, Rating System: {name}, Result: {results[name][-1]}")

    results = pd.DataFrame(results)
    results['length'] = lengths
    # add optimal results
    for key, val in optimal_results.items():
        results[key + '_optimal'] = val

    results['optimal'] = optimal
    results.to_csv('results/alternatives.csv', index=False)
