from rating import Game, DEFAULT_RATING, DetailedLeaderboard, Manager, PolyratingCrossEntropy, RatingPeriodEnum, Matching, DefaultRating
from shared_biases_human import flesch_reading_ease, print_info, compute_repetitiveness
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from bootstrap import bootstrap

def add_games(data, manager):
    """
    Add games to the game database.

    Args:
        data (DataFrame): The data containing information about the games.
        manager (GameManager): The game manager object.

    Returns:
        None
    """
    date_games = datetime.now()
    primary_tags = np.unique(data['primary_tag'])
    for i, row in tqdm(data.iterrows()):
        player_home = manager.add_player(row['model_A'])
        player_out = manager.add_player(row['model_B'])
        result = "1-0"
        if row['extent'] <= 1:
            result = '1/2-1/2'
        elif row['winner'] == row['model_B']:
            result = "0-1"
        home_length = np.log10(len(row['model_outputs'][row['model_A']]) + 1)
        out_length = np.log10(len(row['model_outputs'][row['model_B']]) + 1)
        home_formality = row['formality_a']
        out_formality = row['formality_b']
        home_sentiment = row['sentiment_a']
        out_sentiment = row['sentiment_b']
        home_flesch = flesch_reading_ease(row['model_outputs'][row['model_A']])
        out_flesch = flesch_reading_ease(row['model_outputs'][row['model_B']])
        home_repetitiveness = compute_repetitiveness(row['model_outputs'][row['model_A']])
        out_repetitiveness = compute_repetitiveness(row['model_outputs'][row['model_B']])
        home_order = 1
        out_order = 0
        advantages_home = {'formality': home_formality, 
                              'length': home_length, 'order': home_order, 'sentiment': home_sentiment,
                              'flesch': home_flesch, 
                              'repetitive': home_repetitiveness}
        advantages_out = {'formality': out_formality,
                                'length': out_length, 'order': out_order, 'sentiment': out_sentiment,
                                'flesch': out_flesch, 
                                'repetitive': out_repetitiveness}
        for primary_tag in primary_tags:
            if primary_tag == row['primary_tag']:
                advantages_home[primary_tag] = 1
                advantages_out[primary_tag] = 1
            else:
                advantages_home[primary_tag] = 0
                advantages_out[primary_tag] = 0
        game = Game(
            player_home.id, player_out.id, result, date_games, add_home_advantage=False,
            advantages_home=advantages_home,
            advantages_out=advantages_out
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

    
if __name__ == '__main__':
    import argparse

    # argparse with bootstrap arguments
    parser = argparse.ArgumentParser(description='Run the lmsys released script')
    parser.add_argument('--n-bootstrap', type=int, default=1000)
    parser.add_argument('--cores', type=int, default=1)
    args = parser.parse_args()

    DEFAULT_RATING.set_default(deviation=10 ** 4)

    data = pd.read_json('data/wildbench_pairwise_v2.json')

    data['model_outputs'] = data['model_outputs'].apply(lambda x: {key: val if val != '[This model response is empty.]' else '' for key, val in x.items()})

    leaderboard, shared_ratings = bootstrap(data, compute_function_shared, args.n_bootstrap, args.cores)

    shared_ratings = pd.DataFrame(shared_ratings)
    shared_ratings.to_csv('results/wildbench_released_shared_ratings.csv')