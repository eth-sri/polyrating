import numpy as np
from datetime import timedelta, datetime
import pandas as pd
from tqdm import tqdm
from rating import Game, Manager, PolyratingCrossEntropy, RatingPeriodEnum

def log_loss_rating_system(manager, games):
    """
    Calculates the log loss for a rating system based on the given manager and games.

    Parameters:
    - manager: The rating system manager.
    - games: A list of games.

    Returns:
    - The log loss value.

    """
    system = manager.rating_system
    players = manager.player_database
    log_loss = 0
    total = 0
    for game in games:
        expected_score = system.compute_expected_score(players[game.home], [game], players, game.get_date() + timedelta(days=1))
        expected_score = np.clip(expected_score / game.weight, 1e-15, 1 - 1e-15)
        log_loss += game.weight * (game.get_result() * np.log(expected_score) + (1 - game.get_result()) * np.log(1 - expected_score))
        total += game.weight
    return -log_loss / total

def prepare_lmsys_data():
    """
    Reads the data from 'data/clean_battle_20240629_public.json' and prepares it for the LMSYS analysis.

    Returns:
        pandas.DataFrame: The prepared data for LMSYS analysis.
    """
    data = pd.read_json('data/clean_battle_20240629_public.json')
    data['is_chinese'] = data['language'] == 'Chinese'
    data['is_english'] = data['language'] == 'English'

    for extra_col in data['criteria_tag'].iloc[0]:
        data[extra_col] = data['criteria_tag'].apply(lambda x: x[extra_col] if extra_col in x else None)
    for extra_col in data['dedup_tag'].iloc[0]:
        data[extra_col] = data['dedup_tag'].apply(lambda x: x[extra_col] if extra_col in x else None)

    data['is_hard'] = data['specificity'].astype(float) + data['domain_knowledge'].astype(float) + data['complexity'].astype(float) + data['problem_solving'].astype(float) + data['creativity'].astype(float) + data['technical_accuracy'].astype(float) + data['real_world'].astype(float) >= 6
    return data

def get_games_lmsys(data, manager, category_rows=[]):
    """
    Retrieves a list of Game objects based on the provided data.

    Args:
        data (pandas.DataFrame): The input data containing game information.
        manager (PlayerManager): The player manager object used to add players.
        category_rows (list, optional): A list of category rows to include in the game objects. Defaults to an empty list.

    Returns:
        list: A list of Game objects.
    """
    cols_we_keeps = ['winner', 'model_a', 'model_b'] + category_rows
    data_here = data[cols_we_keeps].copy()
    # add a row called weight that counts the number of times that row appears. Only keep those unique rows once
    data_here['weight'] = 1
    data_here = data_here.groupby(cols_we_keeps).count().reset_index()
    games = []

    date_games = datetime.now()

    for i, row in tqdm(data_here.iterrows()):
        player_home = manager.add_player(row['model_a'])
        player_out = manager.add_player(row['model_b'])
        result = "1-0"
        if row['winner'] == 'model_b':
            result = "0-1"
        elif 'tie' in row['winner']:
            result = "1/2-1/2"
        advantages_home = {category: int(row[category]) for category in category_rows}
        advantages_out = {category: int(row[category]) for category in category_rows}
        game = Game(
            player_home.id, player_out.id, result, date_games, add_home_advantage=False, weight=row['weight'],
            advantages_home=advantages_home,
            advantages_out=advantages_out
        )
        games.append(game)
    return games

def add_games_lmsys(train_data, manager, categories=[]):
    """
    Adds games from the LMSYS system to the game database managed by the manager.

    Args:
        train_data (list): List of training data.
        manager (object): The game manager object.
        categories (list, optional): List of categories to filter the games. Defaults to an empty list.

    Returns:
        None
    """
    for game in get_games_lmsys(train_data, manager, categories):
        manager.game_database.add(game)
    manager.trigger_new_period()

def optimal_log_loss_lmsys(test_data, categories=[]):
    """
    Calculates the optimal log loss for a given test data and categories using the LMSYS rating system.

    Parameters:
    - test_data: The test data to be used for calculating the log loss.
    - categories: Optional list of categories to filter the test data.

    Returns:
    - The optimal log loss for the LMSYS rating system.

    """
    default_manager = Manager(rating_system=PolyratingCrossEntropy(epsilon=1e-3),
                                    rating_period_type=RatingPeriodEnum.MANUAL)
    add_games_lmsys(test_data, default_manager, categories)
    default_manager.update_rating()
    return log_loss_rating_system(default_manager, get_games_lmsys(test_data, default_manager, categories))