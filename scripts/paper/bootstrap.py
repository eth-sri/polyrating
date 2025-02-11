import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

def compute_pivot_leaderboard(real_leaderboard, leaderboards, alpha=0.05):
    """
    Compute the leaderboard with pivotal intervals based on the real leaderboard and a list of leaderboards.

    Parameters:
        real_leaderboard (DataFrame): The real leaderboard DataFrame.
        leaderboards (list): A list of leaderboards DataFrames obtained from doing bootstrapping.
        alpha (float, optional): The significance level for computing the pivotal interval. Defaults to 0.05.

    Returns:
        DataFrame: The pivot leaderboard DataFrame.
    """
    if real_leaderboard is None:
        return None
    stds_per_model = dict()
    for model in real_leaderboard['Name']:
        stds_model = {'model': model}
        for column in real_leaderboard.columns:
            if 'Rating' in column:
                stds_model[column] = real_leaderboard[real_leaderboard['Name'] == model][column].values[0]
                values = [df[df['Name'] == model][column].values[0] for df in leaderboards]
                stds_model[column + ' std'] = np.std(values)
                pivot_small = np.percentile(values, 100 * alpha / 2)
                pivot_large = np.percentile(values, 100 * (1 - alpha / 2))
                stds_model[column + ' alpha'] = [stds_model[column] - pivot_large, stds_model[column] - pivot_small]
        stds_per_model[model] = stds_model
    return pd.DataFrame(stds_per_model).T


def compute_pivot_shared(real_shared_ratings, shared_ratings, alpha=0.05):
    """
    Compute the shared ratings along with their pivotal intervals.

    Parameters:
    - real_shared_ratings (dict): A dictionary containing the shared ratings.
    - shared_ratings (list): A list of dataframes containing the shared ratings obtained from bootstrapping.
    - alpha (float, optional): The significance level for computing the pivot. Default is 0.05.

    Returns:
    - pandas.DataFrame: A dataframe containing the computed shared ratings with their pivotal intervals.
    """
    if real_shared_ratings is None:
        return None
    shared_ratings_new = dict()
    for rating in real_shared_ratings.keys():
        values = [df[rating][0] for df in shared_ratings]
        pivot_small = np.percentile(values, 100 * alpha / 2)
        pivot_large = np.percentile(values, 100 * (1 - alpha / 2))
        shared_ratings_new[rating] = [
                                        real_shared_ratings[rating][0], 
                                        np.std(values),
                                        [
                                            real_shared_ratings[rating][0] - pivot_large, 
                                            real_shared_ratings[rating][0] - pivot_small
                                        ]
                                    ]
    return pd.DataFrame(shared_ratings_new).T

def single_bootstrap(seed, data, compute_function):
    """
    Perform a single bootstrap iteration.

    Args:
        seed (int): The seed value for random number generation.
        data (pandas.DataFrame): The data to be sampled.
        compute_function (function): The function to compute the leaderboard and shared rating.

    Returns:
        tuple: A tuple containing the leaderboard and shared rating.
    """
    np.random.seed(seed)
    if seed == 0:
        sample = data
    else:
        sample = data.sample(frac=1, replace=True, random_state=seed)
    leaderboard, shared_rating = compute_function(sample)
    return leaderboard, shared_rating

def bootstrap(data, compute_function, n_bootstrap=1000, cores=1):
    """
    Perform bootstrap resampling on the given data.

    Args:
        data: The data to be resampled.
        compute_function: The function used to compute the resampled data.
        n_bootstrap: The number of bootstrap iterations to perform (default: 1000).
        cores: The number of CPU cores to use for parallel execution (default: 1).

    Returns:
        The leaderboard and shared ratings with their pivotal intervals computed from the bootstrap resampling.
    """
    with ProcessPoolExecutor(max_workers=cores) as executor:
        results = list(executor.map(single_bootstrap, range(n_bootstrap + 1), 
                                    [data] * (n_bootstrap + 1), 
                                    [compute_function] * (n_bootstrap + 1)))

    leaderboards = [result[0] for result in results]
    shared_ratings = [result[1] for result in results]
    
    real_leaderboard, real_shared_rating = leaderboards[0], shared_ratings[0]

    leaderboards, shared_ratings = leaderboards[1:], shared_ratings[1:]

    pivot_leaderboard = compute_pivot_leaderboard(real_leaderboard, leaderboards)
    pivot_shared = compute_pivot_shared(real_shared_rating, shared_ratings)

    return pivot_leaderboard, pivot_shared
