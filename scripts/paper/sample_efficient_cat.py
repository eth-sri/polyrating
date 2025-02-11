from rating import Manager, DefaultRating, PolyratingCrossEntropy, RatingPeriodEnum, DetailedLeaderboard
from rating import DEFAULT_RATING

from helper import log_loss_rating_system, prepare_lmsys_data, get_games_lmsys, optimal_log_loss_lmsys, add_games_lmsys
import numpy as np
import pandas as pd
from loguru import logger
from concurrent.futures import ProcessPoolExecutor

def execute_cv_function(rating_system, train_data, test_data, category1, category2=None):
    """
    Function used in cross-validation to evaluate the performance of a rating system.

    Args:
        rating_system (str): The rating system to be used.
        train_data (list): The training data.
        test_data (list): The test data.
        category1 (str): The first category name
        category2 (str, optional): The second category name. If None, no second category is used. Defaults to None.

    Returns:
        float: The log loss of the rating system on the test data.

    """
    manager = Manager(rating_period_type=RatingPeriodEnum.MANUAL)
    if category2 is not None:
        add_games_lmsys(train_data, manager, [category1, category2, 'combined_category'])
    else:
        add_games_lmsys(train_data, manager, [category1])
    manager.trigger_new_period()
    manager.reset_and_recompute(rating_system=rating_system)
    if category2 is not None:
        return log_loss_rating_system(manager, get_games_lmsys(test_data, manager, [category1, category2, 'combined_category']))
    return log_loss_rating_system(manager, get_games_lmsys(test_data, manager, [category1]))

def cv(data, rating_systems, category1, category2=None, n_cv=5, n_cores=10):
    """
    Perform cross-validation on the given data using the specified rating systems.

    Args:
        data (pandas.DataFrame): The input data to perform cross-validation on.
        rating_systems (list): A list of rating systems to evaluate.
        category1 (str): The first category for evaluation.
        category2 (str, optional): The second category for evaluation. Defaults to None.
        n_cv (int, optional): The number of cross-validation folds. Defaults to 5.
        n_cores (int, optional): The number of CPU cores to use for parallel execution. Defaults to 10.

    Returns:
        tuple: A tuple containing the index of the best performing rating system and its corresponding mean result.
    """
    data = data.sample(frac=1, random_state=42)
    n_samples = len(data)
    n_samples_per_fold = n_samples // n_cv
    results = []
    for i in range(n_cv):
        logger.info(f'Cross validation fold {i}')
        test_data = data.iloc[i * n_samples_per_fold: (i + 1) * n_samples_per_fold]
        train_data = pd.concat([data.iloc[:i * n_samples_per_fold], data.iloc[(i + 1) * n_samples_per_fold:]])
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            results_cv = list(executor.map(execute_cv_function, rating_systems, [train_data] * len(rating_systems), 
                                           [test_data] * len(rating_systems), [category1] * len(rating_systems), [category2] * len(rating_systems)))
        results.append(results_cv)
    results = np.array(results)
    best_manager = np.argmin(results.mean(axis=0))
    return best_manager, results.mean(axis=0)[best_manager]


def evaluate_categories(data, category1, category2=None, n_test_samples=10000, n_times=50, 
                        max_train_samples=50000, max_cores=20):
    """
    Evaluate the performance of different rating systems for data from various categories

    Args:
        data (pandas.DataFrame): The input data containing the games between models.
        category1 (str): The name of the first category.
        category2 (str, optional): The name of the second category. Defaults to None.
        n_test_samples (int, optional): The number of samples to use for testing. Defaults to 10000.
        n_times (int, optional): The number of iterations to run. Defaults to 50.
        max_train_samples (int, optional): The maximum number of training samples to use for training. Defaults to 50000.
        max_cores (int, optional): The maximum number of CPU cores to use. Defaults to 20.

    Returns:
        pandas.DataFrame: A DataFrame containing the evaluation results for each iteration.
    """
    category_name = category1
    if category2 is not None:
        data['combined_category'] = data[category1] & data[category2]
        category_name = 'combined_category'

    category_data = data[data[category_name]]
    logger.info(f'Category {category1} & {category2} has {len(category_data)} samples')
    # shuffle code_data
    category_data = category_data.sample(frac=1, random_state=42)
    no_category_data = data[~data[category_name]]

    train_data = pd.concat([category_data.iloc[:-n_test_samples], no_category_data])
    test_data = category_data.iloc[-n_test_samples:]

    max_train_samples = min(max_train_samples, len(category_data) - len(test_data))

    optimal_loss = optimal_log_loss_lmsys(test_data, [category_name])
    
    current_std = 1

    results = []

    for i in range(n_times):
        n_train_category_samples = i * max_train_samples // n_times

        train_data = pd.concat([category_data.iloc[:n_train_category_samples], no_category_data])

        default_manager = Manager(rating_system=PolyratingCrossEntropy(epsilon=1e-3),
                                    rating_period_type=RatingPeriodEnum.MANUAL)
        
        std_range = list(range(max(1, current_std - 10), current_std + 41, 10))
        if n_train_category_samples == 0:
            std_range = [1]
            cv_result = 0
        else:
            if category2 is not None:
                rating_systems = [PolyratingCrossEntropy(advantages={category1: DefaultRating(0, std), category2: DefaultRating(0, std), 
                                                'combined_category': DefaultRating(0, std)}, epsilon=1e-2) 
                                                for std in std_range]
            else:
                rating_systems = [PolyratingCrossEntropy(advantages={category1: DefaultRating(0, std)}, epsilon=1e-2) 
                                                for std in std_range]
            cv_result, loss_  = cv(train_data, rating_systems, category1, category2, n_cores=min(max_cores, len(std_range)))

            logger.info(f'Best std: {std_range[cv_result]}, loss: {loss_}')
            current_std = std_range[cv_result]

        if category2 is not None:
            category_manager = Manager(rating_system=PolyratingCrossEntropy(
                                    advantages={category1: DefaultRating(0, std_range[cv_result]), category2: DefaultRating(0, std_range[cv_result]), 
                                                'combined_category': DefaultRating(0, std_range[cv_result])}, epsilon=1e-3), 
                                                rating_period_type=RatingPeriodEnum.MANUAL)
        else:
            category_manager = Manager(rating_system=PolyratingCrossEntropy(
                                    advantages={category1: DefaultRating(0, std_range[cv_result])}, epsilon=1e-3), 
                                    rating_period_type=RatingPeriodEnum.MANUAL)

        if category2 is not None:
            add_games_lmsys(category_data.iloc[:n_train_category_samples], default_manager, [category1, category2, 'combined_category'])
            add_games_lmsys(train_data, category_manager, [category1, category2, 'combined_category'])
        else:
            add_games_lmsys(category_data.iloc[:n_train_category_samples], default_manager, [category1])
            add_games_lmsys(train_data, category_manager, [category1])

        default_manager.update_rating()
        category_manager.update_rating()

        if category2 is not None:
            test_games_categories = get_games_lmsys(test_data, category_manager, [category1, category2, 'combined_category'])
            test_games_default = get_games_lmsys(test_data, default_manager, [category1, category2, 'combined_category'])
        else:
            test_games_categories = get_games_lmsys(test_data, category_manager, [category1])
            test_games_default = get_games_lmsys(test_data, default_manager, [category1])

        results.append({
            'default': log_loss_rating_system(default_manager, test_games_default),
            'category': log_loss_rating_system(category_manager, test_games_categories),
            'optimal': optimal_loss,
            'n_train': n_train_category_samples
        })
        logger.info(f'Iteration {i} - Default: {results[-1]["default"]}, Category: {results[-1]["category"]}, Optimal: {results[-1]["optimal"]}')
    
    return pd.DataFrame(results)

if __name__ == '__main__':

    DEFAULT_RATING.set_default(deviation=10 ** 5)
    data = prepare_lmsys_data()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--category1', type=str, required=True)
    parser.add_argument('--category2', type=str, default=None)
    parser.add_argument('--cores', type=int, default=15)

    args = parser.parse_args()

    if args.category2 is not None:
        results = evaluate_categories(data, args.category1, args.category2, max_cores=args.cores)
        results.to_csv(f'results/sample_efficient_{args.category1}_{args.category2}.csv', index=False)
    else:
        results = evaluate_categories(data, args.category1, args.category2, max_cores=args.cores, n_test_samples=50000)
        results.to_csv(f'results/sample_efficient_{args.category1}.csv', index=False)