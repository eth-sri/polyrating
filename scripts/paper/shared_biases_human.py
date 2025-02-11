from rating import Manager, PolyratingCrossEntropy, RatingPeriodEnum, DefaultRating, Matching, DetailedLeaderboard, Game, DEFAULT_RATING
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger
from bootstrap import bootstrap
import textstat
from collections import Counter
import re

def remove_redundant_backslashes(text):
    """
    Removes redundant backslashes from the given text.

    Args:
        text (str): The input text.

    Returns:
        str: The cleaned text with redundant backslashes removed.
    """
    cleaned_text = ""
    for i, char in enumerate(text):
        if char == "\\" and i < len(text) - 1 and text[i + 1] not in ['n', 't', 'r', 'u']:
            continue
        cleaned_text += char
    
    return cleaned_text

def literal_eval_better(x):
    """
    Improved version of the literal_eval function that evaluates a string containing a Python literal.

    Args:
        x (str): The string to be evaluated.

    Returns:
        list: A list of strings obtained from evaluating the input string.

    """
    strings = []
    x = x[1:-1]
    while len(x) > 0:
        if x[0] == "'":
            end = x[1:].find("',", 1)
        else:
            end = x[1:].find('",', 1)
        if end == -1:
            strings.append(x[1:-1])
            break
        strings.append(x[1:end + 1])
        x = x[end+3:].strip()
    return [remove_redundant_backslashes(s) for s in strings]

def compute_repetitiveness(text):
    """
    Computes the repetitiveness of a given text.

    The repetitiveness is calculated by counting the number of repeated words
    in the text and dividing it by the total number of words. The result is
    then multiplied by 5.

    Args:
        text (str): The input text to calculate repetitiveness for.

    Returns:
        float: The repetitiveness score of the text.
    """
    words = re.findall(r'\b\w+\b', text.lower())
    word_counts = Counter(words)
    repeated_words = sum(1 for count in word_counts.values() if count > 1)
    total_words = len(words)
    repetitiveness = repeated_words / total_words if total_words > 0 else 0
    return repetitiveness * 5

def flesch_reading_ease(text):
    """
    Calculates the Flesch Reading Ease score for a given text.

    The Flesch Reading Ease score is a measure of how easy a text is to read.
    It takes into account the average number of syllables per word and the average number of words per sentence.

    Args:
        text (str): The text for which to calculate the Flesch Reading Ease score.

    Returns:
        float: The Flesch Reading Ease score, ranging from 0 (difficult) to 100 (easy).
    """
    return max(min(1, textstat.flesch_reading_ease(text) / 100), 0)

def print_info(games):
    """
    Print information about the attributes and attribute differences in the given list of games.

    Args:
        games (list): A list of game objects.

    Returns:
        None
    """
    attributes_values = dict()
    attribute_difference_values = dict()
    for game in games:
        for attribute in game.advantages_home.keys():
            if attribute not in attributes_values:
                attributes_values[attribute] = []
            attributes_values[attribute].append(game.advantages_home[attribute])
            attributes_values[attribute].append(game.advantages_out[attribute])
            if attribute not in attribute_difference_values:
                attribute_difference_values[attribute] = []
            attribute_difference_values[attribute].append(abs(game.advantages_home[attribute] - game.advantages_out[attribute]))
    
    for attribute in attributes_values.keys():
        logger.info(f'{attribute}: {np.mean(attributes_values[attribute])} +- {np.std(attributes_values[attribute])}')
        quantile_095 = np.percentile(attributes_values[attribute], 95)
        quantile_005 = np.percentile(attributes_values[attribute], 5)
        logger.info(f'5% quantile: {quantile_005}, 95% quantile: {quantile_095}')
        logger.info(f'{attribute} difference: {np.mean(attribute_difference_values[attribute])}')


def add_games(data, manager):
    """
    Add games to the game database based on the provided data.

    Args:
        data (pandas.DataFrame): The data containing information about the games.
        manager (GameManager): The game manager object.

    Returns:
        None
    """
    date_games = datetime.now()
    for i, row in tqdm(data.iterrows()):
        player_home = manager.add_player(row['model_a'])
        player_out = manager.add_player(row['model_b'])
        result = "1-0"
        if row['winner_model_b'] == 1:
            result = "0-1"
        elif row['winner_tie'] == 1:
            result = "1/2-1/2"
        home_length = np.log10(sum([len(message) for message in row['response_a']]) + 1)
        out_length = np.log10(sum([len(message) for message in row['response_b']]) + 1)
        home_formality = row['formality_a']
        out_formality = row['formality_b']
        home_sentiment = row['sentiment_a']
        out_sentiment = row['sentiment_b']
        home_flesch = flesch_reading_ease(row['response_a'][0])
        out_flesch = flesch_reading_ease(row['response_b'][0])
        home_repetitive = compute_repetitiveness(row['response_a'][0])
        out_repetitive = compute_repetitiveness(row['response_b'][0])
        home_order = 1
        out_order = 0
        game = Game(
            player_home.id, player_out.id, result, date_games, add_home_advantage=False,
            advantages_home={'formality': home_formality, 
                              'length': home_length, 'order': home_order, 
                              'sentiment': home_sentiment, 
                              'flesch': home_flesch, 'repetitive': home_repetitive},
            advantages_out={'formality': out_formality, 
                              'length': out_length, 'order': out_order, 
                              'sentiment': out_sentiment, 
                              'flesch': out_flesch, 'repetitive': out_repetitive}
        )
        manager.game_database.add(game)
    print_info(manager.game_database)
    manager.trigger_new_period()

def evaluate_manager(data, manager):
    """
    Evaluates the manager by adding games, updating ratings, and computing the leaderboard and shared ratings.

    Args:
        data: The data to be used for evaluation.
        manager: The manager object.

    Returns:
        A tuple containing the leaderboard and shared ratings.
    """
    add_games(data, manager)
    manager.update_rating()
    leaderboard = DetailedLeaderboard.compute(
        manager.player_database, manager.game_database, manager.tournament_database, manager.rating_system
    )
    shared_ratings = dict()
    for i, history in enumerate(manager.rating_system.shared_rating_histories):
        category = manager.rating_system.shared_advantages[i][0]
        shared_ratings[category] = (history.rating_history[-1][0].rating, history.rating_history[-1][0].deviation)

    return leaderboard, shared_ratings

def compute_function_shared(data):
    """
    Computes the function for evaluating the shared biases in the data. Used for bootstrapping.

    Args:
        data: The input data for evaluation.

    Returns:
        leaderboard: The leaderboard containing the rankings of the evaluated data.
        shared_ratings: The shared ratings for the evaluated data.
    """
    DEFAULT_RATING.set_default(deviation=10 ** 4)
    shared_manager = Manager(
        rating_system=PolyratingCrossEntropy(
            shared_advantages=[
                ('formality', Matching(), DefaultRating(0, 50), 0.1),
                ('length', Matching(), DefaultRating(0, 50), 0.1),
                ('order', Matching(), DefaultRating(0, 50), 0.1),
                ('sentiment', Matching(), DefaultRating(0, 50), 0.1),
                ('flesch', Matching(), DefaultRating(0, 50), 0.1),
                ('repetitive', Matching(), DefaultRating(0, 50), 0.1),
            ],
            epsilon=1e-3, max_iterations=1000
        ),
        rating_period_type=RatingPeriodEnum.MANUAL,
    )

    leaderboard, shared_ratings = evaluate_manager(data, shared_manager)
    return leaderboard, shared_ratings


def load_lmsys_local():
    """
    Load the lmsys data from a CSV file and perform necessary data transformations.

    Returns:
        pandas.DataFrame: The loaded and transformed data.
    """
    data = pd.read_csv('data/lmsys.csv')
    data['response_a'] = data['response_a'].apply(lambda x: x.replace('[null,', '["",').replace(',null]', ',""]').replace('[null]', '[""]').replace(',null,', ',"",'))
    data['response_b'] = data['response_b'].apply(lambda x: x.replace('[null,', '["",').replace(',null]', ',""]').replace('[null]', '[""]').replace(',null,', ',"",'))
    data['response_a'] = data['response_a'].apply(literal_eval_better)
    data['response_b'] = data['response_b'].apply(literal_eval_better)
    data['prompt'] = data['prompt'].apply(literal_eval_better)
    return data
    
if __name__ == '__main__':
    import argparse

    # argparse with bootstrap arguments
    parser = argparse.ArgumentParser(description='Run the lmsys released script')
    parser.add_argument('--n-bootstrap', type=int, default=1000)
    parser.add_argument('--cores', type=int, default=1)
    args = parser.parse_args()


    DEFAULT_RATING.set_default(deviation=10 ** 4)
    data = load_lmsys_local()

    leaderboard, shared_ratings = bootstrap(data, compute_function_shared, args.n_bootstrap, args.cores)

    shared_ratings = pd.DataFrame(shared_ratings)
    shared_ratings.to_csv('results/lmsys_released_shared_ratings.csv')