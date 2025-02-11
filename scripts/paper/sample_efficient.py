
from rating import Manager, DefaultRating, PolyratingCrossEntropy, RatingPeriodEnum, DetailedLeaderboard, Matching
from rating import Game, DEFAULT_RATING

from datetime import datetime, timedelta
from helper import prepare_lmsys_data, get_games_lmsys, optimal_log_loss_lmsys, log_loss_rating_system
import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def get_games_mixeval(data, manager, multiplier=1):
    """
    Generate a list of Game objects based on the provided data for the MixEval data.

    Args:
        data (pandas.DataFrame): The input data containing information about the games.
        manager (PlayerManager): The player manager object used to add players.
        multiplier (float, optional): A multiplier to adjust the game results. Defaults to 1.

    Returns:
        list: A list of Game objects.

    """
    data_here = data.copy()
    rows_we_keeps = ['model_A', 'model_B', 'score_a', 'score_b']
    data_here = data_here[rows_we_keeps]    
    data_here = data_here.groupby(rows_we_keeps).count().reset_index()
    games = []

    date_games = datetime.now()

    for i, row in tqdm(data_here.iterrows()):
        player_home = manager.add_player(row['model_A'])
        player_out = manager.add_player(row['model_B'])
        res = 1 / 200 * (100 + row['score_a'] - row['score_b'])
        res = min(1, res * multiplier)
        result = [res, 1-res]
        result = f"{result[0]}-{result[1]}"
        advantage_home = {'llm': 1}
        advantage_out = {'llm': 1}
        game = Game(
            player_home.id, player_out.id, result, date_games, add_home_advantage=False, weight=1000,
            advantages_home=advantage_home,
            advantages_out=advantage_out,
        )
        games.append(game)
    return games

def get_games_wildbench(data, manager):
    """
    Generate a list of Game objects based on the given data and manager for Wildbench data.

    Args:
        data (DataFrame): The input data containing information about the games.
        manager (Manager): The manager object used to add players.

    Returns:
        list: A list of Game objects.

    """
    data_here = data.copy()
    length_a = []
    length_b = []
    for i, row in data_here.iterrows():
        length_a.append(np.log10(len(row['model_outputs'][row['model_A']]) + 1))
        length_b.append(np.log10(len(row['model_outputs'][row['model_B']]) + 1))
    data_here['llm_length_a'] = length_a
    data_here['llm_length_b'] = length_b
    rows_we_keeps = ['extent', 'winner', 'model_A', 'model_B', 'llm_length_a', 'llm_length_b']
    data_here = data_here[rows_we_keeps]    
    data_here['weight'] = 1
    data_here = data_here.groupby(rows_we_keeps).count().reset_index()
    games = []

    date_games = datetime.now()

    for i, row in tqdm(data_here.iterrows()):
        player_home = manager.add_player(row['model_A'])
        player_out = manager.add_player(row['model_B'])
        extent = row['extent']
        if extent == 0:
            result = "1/2-1/2"
        elif extent == 1:
            result = [0.5, 0.5]
        else:
            result = [1, 0]
        if row['winner'] == row['model_B']:
            result = f"{result[1]}-{result[0]}"
        elif row['winner'] == row['model_A']:
            result = f"{result[0]}-{result[1]}"
        game = Game(
            player_home.id, player_out.id, result, date_games, add_home_advantage=False, weight=row['weight'],
            advantages_home={'llm': 1, 'llm_length': row['llm_length_a']},
            advantages_out={'llm': 1, 'llm_length': row['llm_length_b']},
        )
        games.append(game)
    return games

def optimize_multiplier(train_data, data_other, range_=[0.95, 1.0, 1.05, 1.1, 1.15, 1.2]):
    """
    Optimize the multiplier parameter for the rating system.
    Parameters:
    - train_data: The training data from LMSYS to be used as test data here
    - data_other: The data used to train the rating system for the multiplier optimization
    - range_: The range of multiplier values to iterate over. Default is [0.8, 0.9, 1.0, 1.1, 1.2].

    Returns:
    - The multiplier value that minimizes the log loss.

    """
    losses = []
    for multiplier in range_:
        default_manager = Manager(rating_system=PolyratingCrossEntropy(epsilon=1e-3),
                                    rating_period_type=RatingPeriodEnum.MANUAL)
        add_games(data_other, default_manager, False, False, multiplier)
        default_manager.trigger_new_period()
        default_manager.update_rating()
        losses.append(log_loss_rating_system(default_manager, get_games_lmsys(train_data, default_manager)))
    return range_[np.argmin(losses)]

def add_games(train_data, manager, lmsys=True, wildbench=False, multiplier=1):
    """
    Add games to the game database.

    Parameters:
    - train_data: The training data.
    - manager: The game manager.
    - lmsys: A boolean indicating whether to use the lmsys strategy (default: True).
    - wildbench: A boolean indicating whether to use the wildbench strategy (default: False).
    - multiplier: The multiplier for the mixeval strategy (default: 1).

    Returns:
    None
    """
    if wildbench:
        for game in get_games_wildbench(train_data, manager):
            manager.game_database.add(game)
    elif not lmsys:
        for game in get_games_mixeval(train_data, manager, multiplier):
            manager.game_database.add(game)
    else:
        for game in get_games_lmsys(train_data, manager):
            manager.game_database.add(game)

def execute_cv_function(rating_system, train_data, train_data_mixeval, test_data, multiplier=1, wildbench=True):
    """
    Executes the cross-validation function.

    Args:
        rating_system (str): The rating system to be used.
        train_data (list): The training data from lmsys.
        train_data_mixeval (list): The training data from mixeval or wildbench.
        test_data (list): The test data.
        multiplier (int, optional): The multiplier value. Defaults to 1.
        wildbench (bool, optional): Flag indicating whether wildbench is used. Defaults to True.

    Returns:
        float: The log loss rating system.

    """
    manager = Manager(rating_period_type=RatingPeriodEnum.MANUAL)
    add_games(train_data, manager)
    add_games(train_data_mixeval, manager, lmsys=False, wildbench=wildbench, multiplier=multiplier)
    manager.trigger_new_period()
    manager.reset_and_recompute(rating_system=rating_system)
    return log_loss_rating_system(manager, get_games_lmsys(test_data, manager))

def cv(data, train_data_mixeval, rating_systems, n_cv=5, multiplier=1, wildbench=True):
    """
    Perform cross-validation on rating systems. Used to optimize the hyperparameters.

    Args:
        data (pd.DataFrame): The input data for cross-validation.
        train_data_mixeval (pd.DataFrame): The training data for MixEval or Wildbench.
        rating_systems (list): List of rating systems to evaluate.
        n_cv (int, optional): Number of cross-validation folds. Defaults to 5.
        multiplier (int, optional): Multiplier for rating updates. Defaults to 1.
        wildbench (bool, optional): Flag indicating whether to use WildBench evaluation. Defaults to True.

    Returns:
        tuple: A tuple containing the index of the best rating system and its corresponding mean score.
    """
    data = data.sample(frac=1, random_state=0)
    n_samples = len(data)
    n_samples_per_fold = n_samples // n_cv
    results = []
    for i in range(n_cv):
        logger.info(f'Cross validation fold {i}')
        test_data = data.iloc[i * n_samples_per_fold: (i + 1) * n_samples_per_fold]
        train_data = pd.concat([data.iloc[:i * n_samples_per_fold], data.iloc[(i + 1) * n_samples_per_fold:]])
        results_cv = [
            execute_cv_function(rating_system, train_data, train_data_mixeval, test_data, multiplier, wildbench=wildbench) for rating_system in rating_systems
        ]
        results.append(results_cv)
    results = np.array(results)
    best_manager = np.argmin(results.mean(axis=0))
    return best_manager, results.mean(axis=0)[best_manager]

def evaluate_category(data, data_mixeval, n_test_samples=300000, n_times=50, max_train_samples=50000, max_cores=50, 
                      wildbench=True):
    """
    Evaluate the category based on the given data.

    Args:
        data (pandas.DataFrame): The input data for evaluation from lmsys.
        data_mixeval (pandas.DataFrame): The data for evaluation from wildbench or mixeval.
        n_test_samples (int, optional): The number of test samples. Defaults to 300000.
        n_times (int, optional): The number of times to run the evaluation. Defaults to 50.
        max_train_samples (int, optional): The maximum number of training samples. Defaults to 50000.
        max_cores (int, optional): The maximum number of CPU cores to use. Defaults to 50.
        wildbench (bool, optional): Whether to use wildbench. Defaults to True.

    Returns:
        pandas.DataFrame: The evaluation results.
    """
    data = data.sample(frac=1, random_state=0)
    data_mixeval = data_mixeval.sample(frac=1, random_state=0)
    test_data = data.iloc[-n_test_samples:]

    optimal_loss = optimal_log_loss_lmsys(test_data)

    max_train_samples = min(max_train_samples, len(data) - len(test_data))

    # Define a partial function with fixed arguments except for the iteration index
    single_loop_partial = partial(single_loop, data, data_mixeval, n_times, max_train_samples, test_data, optimal_loss=optimal_loss, wildbench=wildbench)

    # Parallelize the execution of the for loop
    with ProcessPoolExecutor(max_workers=max_cores) as executor:
        results = list(executor.map(single_loop_partial, range(n_times)))

    return pd.DataFrame(results)

def single_loop(data, data_mixeval, n_times, max_train_samples, test_data, i, optimal_loss, wildbench=True):
    """
    Perform a single loop of the rating system evaluation.

    Args:
        data (pandas.DataFrame): The training data.
        data_mixeval (pandas.DataFrame): The training data from wildbench or mixeval.
        n_times (int): The number of iterations.
        max_train_samples (int): The maximum number of training samples.
        test_data (pandas.DataFrame): The test data.
        i (int): The current iteration index.
        optimal_loss (float): The optimal loss value.
        wildbench (bool, optional): Whether to use wildbench. Defaults to True.

    Returns:
        dict: A dictionary containing the evaluation results for the default, category, and optimal rating systems.

    """
    n_train_category_samples = i * max_train_samples // n_times

    default_manager = Manager(rating_system=PolyratingCrossEntropy(epsilon=1e-3),
                                    rating_period_type=RatingPeriodEnum.MANUAL)
        
    
    if n_train_category_samples > 0 and not wildbench:
        multiplier = optimize_multiplier(data.iloc[:n_train_category_samples], data_mixeval)
    else:
        multiplier = 1

    if n_train_category_samples == 0:
        std_range = [1]
        cv_result = 0
    else:
        std_range = list(range(1, 52, 5))
        if wildbench:
            rating_systems = [PolyratingCrossEntropy(advantages={'llm': DefaultRating(0, std)}, 
                                                    shared_advantages=[('llm_length', Matching(), DefaultRating(0, 100), 0.1)], 
                                                    epsilon=1e-2)  for std in std_range]
        else:
            rating_systems = [PolyratingCrossEntropy(advantages={'llm': DefaultRating(0, std)}, epsilon=1e-3) for std in std_range]
        cv_result, loss_  = cv(data.iloc[:n_train_category_samples], data_mixeval, rating_systems, 
                               multiplier=multiplier, wildbench=wildbench)

        logger.info(f'Best std: {std_range[cv_result]}, loss: {loss_}')

    if wildbench:
        rating_system = PolyratingCrossEntropy(advantages={'llm': DefaultRating(0, std_range[cv_result])}, 
                                           shared_advantages=[('llm_length', Matching(), DefaultRating(0, 100), 0.1)], 
                                           epsilon=1e-3)
    else:
        rating_system = PolyratingCrossEntropy(advantages={'llm': DefaultRating(0, std_range[cv_result])}, epsilon=1e-3)
    category_manager = Manager(rating_system=rating_system, rating_period_type=RatingPeriodEnum.MANUAL)
    
    logger.info(f'Optimal multiplier: {multiplier}')
    add_games(data.iloc[:n_train_category_samples], default_manager)
    add_games(data.iloc[:n_train_category_samples], category_manager)
    add_games(data_mixeval, category_manager, lmsys=False, wildbench=wildbench, multiplier=multiplier)
    default_manager.trigger_new_period()
    category_manager.trigger_new_period()

    default_manager.update_rating()
    category_manager.update_rating()

    test_games_categories = get_games_lmsys(test_data, category_manager)
    test_games_default = get_games_lmsys(test_data, default_manager)
    result = {
            'default': log_loss_rating_system(default_manager, test_games_default),
            'category': log_loss_rating_system(category_manager, test_games_categories),
            'optimal': optimal_loss,
            'n_train': n_train_category_samples,
            'std': std_range[cv_result]
        }
    logger.info(f'Iteration {i} - Default: {result["default"]}, Category: {result["category"]}, Optimal: {result["optimal"]}')
    return result

def do_wildbench(data, max_cores):
    """Performs the evaluation for the Wildbench data.

    Args:
        data (pd.DataFrame): The input data for evaluation.
        max_cores (int): The maximum number of CPU cores to use.
    """
    data_wildbench = pd.read_json('data/wildbench_pairwise_v2.json')

    model_mapper = {
        '01-ai/Yi-1.5-34B-Chat': 'yi-1.5-34b-chat', 
        '01-ai/Yi-1.5-6B-Chat': None,
       '01-ai/Yi-1.5-9B-Chat': None,
       'Magpie-Align/Llama-3-8B-Magpie-Pro-SFT-v0.1': None,
       'Nexusflow/Starling-LM-7B-beta': 'starling-lm-7b-beta',
       'NousResearch/Hermes-2-Theta-Llama-3-8B': None,
       'NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO': 'nous-hermes-2-mixtral-8x7b-dpo',
       'Qwen/Qwen1.5-72B-Chat': 'qwen1.5-72b-chat', 
       'Qwen/Qwen1.5-7B-Chat@together': 'qwen1.5-7b-chat',
       'Qwen/Qwen2-72B-Instruct': 'qwen2-72b-instruct', 
       'ZhangShenao/SELM-Zephyr-7B-iter-3': None,
       'allenai/tulu-2-dpo-70b': 'tulu-2-dpo-70b', 
       'anthropic/claude-3-haiku-20240307': 'claude-3-haiku-20240307',
       'anthropic/claude-3-opus-20240229': 'claude-3-opus-20240229',
       'anthropic/claude-3-sonnet-20240229': 'claude-3-sonnet-20240229',
       'chujiezheng/Llama-3-Instruct-8B-SimPO-ExPO': None,
       'chujiezheng/Starling-LM-7B-beta-ExPO': None, 
       'cohere/command-r': 'command-r',
       'cohere/command-r-plus': 'command-r-plus', 
       'databricks/dbrx-instruct@together': 'dbrx-instruct-preview',
       'deepseek/deepseekv2-chat': None, 
       'google/gemini-1.5-flash': 'gemini-1.5-flash-api-0514',
       'google/gemini-1.5-pro': 'gemini-1.5-pro-api-0514', 
       'google/gemma-2b-it': 'gemma-2b-it',
       'google/gemma-7b-it': 'gemma-7b-it', 
       'm-a-p/neo_7b_instruct_v0.1': None,
       'meta-llama/Llama-2-70b-chat-hf': 'llama-2-70b-chat', 
       'meta-llama/Llama-2-7b-chat-hf': 'llama-2-7b-chat',
       'meta-llama/Meta-Llama-3-70B-Instruct': 'llama-3-70b-instruct',
       'meta-llama/Meta-Llama-3-8B-Instruct': 'llama-3-8b-instruct',
       'microsoft/Phi-3-medium-128k-instruct': 'phi-3-medium-4k-instruct',
       'microsoft/Phi-3-mini-128k-instruct': 'phi-3-mini-128k-instruct', 
       'mistral/mistral-large-2402': 'mistral-large-2402',
       'mistralai/Mistral-7B-Instruct-v0.2': 'mistral-7b-instruct-v0.2',
       'mistralai/Mixtral-8x7B-Instruct-v0.1': 'mixtral-8x7b-instruct-v0.1',
       'openai/gpt-3.5-turbo-0125': 'gpt-3.5-turbo-0125', 
       'openai/gpt-4-0125-preview': 'gpt-4-0125-preview',
       'openai/gpt-4-turbo-2024-04-09': 'gpt-4-turbo-2024-04-09', 
       'openai/gpt-4o-2024-05-13': 'gpt-4o-2024-05-13',
       'princeton-nlp/Llama-3-Instruct-8B-SimPO': None,
       'reka/reka-core-20240501': 'reka-core-20240501', 
       'reka/reka-edge': None,
       'reka/reka-flash-20240226': 'reka-flash-21b-20240226', 
       'yi/yi-large': 'yi-large',
    }
    models_wildbench = [key for key in model_mapper if model_mapper[key] is not None]
    data_wildbench = data_wildbench[data_wildbench['model_A'].isin(models_wildbench) & data_wildbench['model_B'].isin(models_wildbench)]
    models_data = [value for key, value in model_mapper.items() if value is not None]
    data = data[data['model_a'].isin(models_data) & data['model_b'].isin(models_data)]
    data_wildbench['model_A'] = data_wildbench['model_A'].apply(lambda x: model_mapper[x])
    data_wildbench['model_B'] = data_wildbench['model_B'].apply(lambda x: model_mapper[x])
    data_wildbench['winner'] = data_wildbench['winner'].apply(lambda x: model_mapper[x] if x in model_mapper else x)
    data_wildbench['model_outputs'] = data_wildbench['model_outputs'].apply(lambda x: {model_mapper[key]: value for key, value in x.items() if key in model_mapper})
    results = evaluate_category(data, data_wildbench, max_cores=max_cores)
    results.to_csv(f'results/sample_efficient_wildbench.csv', index=False)

def do_mixeval(data, max_cores):
    """Performs the evaluation for the MixEval data.

    Args:
        data (pd.DataFrame): The input data for evaluation.
        max_cores (int): The maximum number of CPU cores to use
    """
    mixeval_score = {
        'Claude 3.5 Sonnet-0620': 68.05,
        'GPT-4o-2024-05-13': 64.7,
        'Claude 3 Opus': 63.5,
        'GPT-4-Turbo-2024-04-09': 62.6,
        'Gemini 1.5 Pro-API-0409': 58.7,
        'Gemini 1.5 Pro-API-0514': 58.3,
        'Yi-Large-preview': 56.8,
        'LLaMA-3-70B-Instruct': 55.9,
        'Qwen-Max-0428': 55.8,
        'Claude 3 Sonnet': 54.0,
        'Reka Core-20240415': 52.9,
        'MAmmoTH2-8x7B-Plus': 51.8,
        'DeepSeek-V2': 51.7,
        'GPT-4o mini': 51.6,
        'Command R+': 51.4,
        'Yi-1.5-34B-Chat': 51.2,
        'Mistral-Large': 50.3,
        'Qwen1.5-72B-Chat': 48.3,
        'Mistral-Medium': 47.8,
        'Gemini 1.0 Pro': 46.4,
        'Reka Flash-20240226': 46.2,
        'Mistral-Small': 46.2,
        'LLaMA-3-8B-Instruct': 45.6,
        'Command R': 45.2,
        'Qwen1.5-32B-Chat': 43.3,
        'GPT-3.5-Turbo-0125': 43.0,
        'Claude 3 Haiku': 42.8,
        'Yi-34B-Chat': 42.6,
        'Mixtral-8x7B-Instruct-v0.1': 42.5,
        'Starling-LM-7B-beta': 41.8,
        'Yi-1.5-9B-Chat': 40.9,
        'Gemma-1.1-7B-IT': 39.1,
        'Vicuna-33B-v1.3': 38.7,
        'LLaMA-2-70B-Chat': 38.0,
        'MAP-Neo-Instruct-v0.1': 37.8,
        'Mistral-7B-Instruct-v0.2': 36.2,
        'Qwen1.5-7B-Chat': 35.5,
        'Reka Edge-20240208': 32.2,
        'Zephyr-7B-β': 31.6,
        'LLaMA-2-7B-Chat': 30.8,
        'Yi-6B-Chat': 30.1,
        'Qwen1.5-MoE-A2.7B-Chat': 29.1,
        'Gemma-1.1-2B-IT': 28.4,
        'Vicuna-7B-v1.5': 27.8,
        'OLMo-7B-Instruct': 26.7,
        'Qwen1.5-4B-Chat': 24.6,
        'JetMoE-8B-Chat': 24.3,
        'MPT-7B-Chat': 23.8
    }
    model_mapper = {
        'Claude 3.5 Sonnet-0620': 'claude-3-5-sonnet-20240620',
        'GPT-4o-2024-05-13': 'gpt-4o-2024-05-13',
        'Claude 3 Opus': 'claude-3-opus-20240229',
        'GPT-4-Turbo-2024-04-09': 'gpt-4-turbo-2024-04-09',
        'Gemini 1.5 Pro-API-0409': 'gemini-1.5-pro-api-0409-preview',
        'Gemini 1.5 Pro-API-0514': 'gemini-1.5-flash-api-0514',
        'Yi-Large-preview': 'yi-large-preview',
        'LLaMA-3-70B-Instruct': 'llama-3-70b-instruct',
        'Qwen-Max-0428': 'qwen-max-0428',
        'Claude 3 Sonnet': 'claude-3-sonnet-20240229',
        'Reka Core-20240415': 'reka-core-20240501',
        'MAmmoTH2-8x7B-Plus': None,
        'DeepSeek-V2': None,
        'GPT-4o mini': None,
        'Command R+': 'command-r-plus',
        'Yi-1.5-34B-Chat': 'yi-1.5-34b-chat',
        'Mistral-Large': 'mistral-large-2402',
        'Qwen1.5-72B-Chat': 'qwen1.5-72b-chat',
        'Mistral-Medium': 'mistral-medium',
        'Gemini 1.0 Pro': 'gemini-pro',
        'Reka Flash-20240226': 'reka-flash-21b-20240226',
        'Mistral-Small': None,
        'LLaMA-3-8B-Instruct': 'llama-3-8b-instruct',
        'Command R': 'command-r',
        'Qwen1.5-32B-Chat': 'qwen1.5-32b-chat',
        'GPT-3.5-Turbo-0125': 'gpt-3.5-turbo-0125',
        'Claude 3 Haiku': 'claude-3-haiku-20240307',
        'Yi-34B-Chat': 'yi-34b-chat',
        'Mixtral-8x7B-Instruct-v0.1': 'mixtral-8x7b-instruct-v0.1',
        'Starling-LM-7B-beta': 'starling-lm-7b-beta',
        'Yi-1.5-9B-Chat': None,
        'Gemma-1.1-7B-IT': 'gemma-1.1-7b-it',
        'Vicuna-33B-v1.3': 'vicuna-33b',
        'LLaMA-2-70B-Chat': 'llama-2-70b-chat',
        'MAP-Neo-Instruct-v0.1': None,
        'Mistral-7B-Instruct-v0.2': 'mistral-7b-instruct-v0.2',
        'Qwen1.5-7B-Chat': 'qwen1.5-7b-chat',
        'Reka Edge-20240208': None,
        'Zephyr-7B-β': 'zephyr-7b-beta',
        'LLaMA-2-7B-Chat': 'llama-2-7b-chat',
        'Yi-6B-Chat': None,
        'Qwen1.5-MoE-A2.7B-Chat': None,
        'Gemma-1.1-2B-IT': 'gemma-1.1-2b-it',
        'Vicuna-7B-v1.5': 'vicuna-7b',
        'OLMo-7B-Instruct': 'olmo-7b-instruct',
        'Qwen1.5-4B-Chat': 'qwen1.5-4b-chat',
        'JetMoE-8B-Chat': None,
        'MPT-7B-Chat': 'mpt-7b-chat'
    }

    mixeval_data = []
    for i, model in enumerate(mixeval_score):
        for j, model2 in enumerate(mixeval_score):
            if j > i:
                mixeval_data.append({'model_A': model, 'model_B': model2, 
                                     'score_a': mixeval_score[model], 'score_b': mixeval_score[model2]})
                
    data_mixeval = pd.DataFrame(mixeval_data)
    models_mixeval = [key for key in model_mapper if model_mapper[key] is not None]
    data_mixeval = data_mixeval[data_mixeval['model_A'].isin(models_mixeval) & data_mixeval['model_B'].isin(models_mixeval)]
    models_data = [value for key, value in model_mapper.items() if value is not None]
    data = data[data['model_a'].isin(models_data) & data['model_b'].isin(models_data)]
    data_mixeval['model_A'] = data_mixeval['model_A'].apply(lambda x: model_mapper[x])
    data_mixeval['model_B'] = data_mixeval['model_B'].apply(lambda x: model_mapper[x])
    results = evaluate_category(data, data_mixeval, max_cores=max_cores, wildbench=False)

    results.to_csv(f'results/sample_efficient_mixeval.csv', index=False)

    
if __name__ == '__main__':
    import argparse 

    parser = argparse.ArgumentParser()
    parser.add_argument('--cores', type=int, default=40)
    parser.add_argument('--wildbench', action='store_true')

    args = parser.parse_args()

    DEFAULT_RATING.set_default(deviation=10 ** 5)
    data = prepare_lmsys_data()

    if args.wildbench:
        do_wildbench(data, args.cores)
    else:
        do_mixeval(data, args.cores)
    