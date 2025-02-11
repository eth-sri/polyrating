import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from loguru import logger
from datetime import datetime, timedelta
from scipy.stats import norm
from typing import List, Generator, Any

from ..objects import Tournament, Game, Rating, DEFAULT_RATING
from ..databases import GameDatabase, PlayerDatabase, TournamentDatabase
from ..rating import RatingSystem


class Statistic:
    """
    Represents a statistic calculation for chess ratings.

    Attributes:
        - default_color (str): The default color for the statistic.
    """
    default_color = sns.color_palette("Greens")[4]
    second_color = sns.color_palette('Blues')[4]

    def __init__(self):
        pass

    @staticmethod
    def compute(player_database : PlayerDatabase = None, game_database : GameDatabase = None, 
                tournament_database : TournamentDatabase = None, rating_system: RatingSystem = None, save_folder : str = None, file_name : str = None, 
                **kwargs) -> Any:
        """
        Computes the statistic based on the provided databases.

        Args:
            - player_database (PlayerDatabase): The player database.
            - game_database (GameDatabase): The game database.
            - tournament_database (TournamentDatabase): The tournament database.
            - rating_system (RatingSystem): The rating system used to compute the statistic.
            - save_folder (str, optional): The folder to save the computed statistic. Defaults to None.
            - file_name (str, optional): The name of the file to save the computed statistic. Defaults to None.
            - **kwargs: Additional keyword arguments for the computation.

        Returns:
            - Any: The computed statistic. Can take any format, depending on the computed statistic
        """
        raise NotImplementedError
    
class TournamentStatistic(Statistic):
    """
    Represents a statistic related to a chess tournament.
    """
    @staticmethod
    def compute(player_database : PlayerDatabase = None, game_database : GameDatabase = None, 
                tournament : Tournament = None, rating_system: RatingSystem = None, 
                save_folder : str = None, file_name : str = None, 
                **kwargs) -> Any:
        """
        Computes the statistic based on the provided player and game databases,
        and the specified tournament.

        Args:
            - player_database (PlayerDatabase): The database containing player information.
            - game_database (GameDatabase): The database containing game information.
            - tournament (Tournament): The tournament for which the statistic is computed.
            - rating_system (RatingSystem): The rating system used to compute the statistic.
            - save_folder (str, optional): The folder where the computed statistic will be saved.
            - file_name (str, optional): The name of the file to save the computed statistic.
            - **kwargs: Additional keyword arguments for the computation.

        Returns:
            - Any: The computed statistic. Can take any format, depending on the computed statistic.
        """
        raise NotImplementedError
    
class DetailedLeaderboard(Statistic):
    """
    Represents a detailed leaderboard for chess players.
    """
    @staticmethod
    def compute_leaderboard(player_database : PlayerDatabase, game_database : GameDatabase) -> pd.DataFrame:
        """
        Computes the leaderboard based on the provided player, game, and tournament databases.

        Args:
            - player_database (PlayerDatabase): The database containing player information.
            - game_database (GameDatabase): The database containing game information.

        Returns:
            - pandas.DataFrame: The computed leaderboard as a pandas DataFrame.
        """
        leaderboard = []
        advantage_names = set()
        for player in player_database:
            advantage_names = advantage_names.union(set(player.get_rating().get_advantage_names()))
        advantage_names = list(advantage_names)
        all_info_names = ["Name", "Rating", "Deviation", "Wins", "Losses", "Draws"]
        for name in advantage_names:
            all_info_names.append(f"{name} Rating")
            all_info_names.append(f"{name} Deviation")
        for player in player_database:
            wins = player.get_number_of_wins(game_database.get_games_per_player(player.id))
            losses = player.get_number_of_losses(game_database.get_games_per_player(player.id))
            draws = player.get_number_of_draws(game_database.get_games_per_player(player.id))
            rating = player.get_rating()
            all_info = [player.name, rating.rating, rating.deviation, wins, losses, draws]
            for name in advantage_names:
                advantage_rating = rating.get_advantage(name)
                if advantage_rating is None:
                    all_info.append(None)
                    all_info.append(None)
                else:
                    all_info.append(advantage_rating.rating)
                    all_info.append(advantage_rating.deviation)
            leaderboard.append(all_info)
        leaderboard.sort(key=lambda x: x[1], reverse=True)
        leaderboard = pd.DataFrame(leaderboard, columns=all_info_names)
        return leaderboard

    @staticmethod
    def compute(player_database : PlayerDatabase = None, game_database : GameDatabase = None, 
                tournament_database : TournamentDatabase = None, rating_system: RatingSystem = None, 
                save_folder : str = None, file_name : str = 'detailed_leaderboard.csv') -> pd.DataFrame:
        leaderboard = DetailedLeaderboard.compute_leaderboard(player_database, game_database)
        if save_folder and file_name:
            leaderboard.to_csv(os.path.join(save_folder, file_name), index=False)
        return leaderboard
    
class Leaderboard(Statistic):
    """
    Represents a restricted leaderboard for chess players.
    """

    @staticmethod
    def compute_leaderboard(player_database : PlayerDatabase = None, game_database : GameDatabase = None,
                            tournament_database : TournamentDatabase = None, restricted : bool = False) -> pd.DataFrame:
        """
        Computes the leaderboard based on the provided player, game, and tournament databases.

        Args:
            - player_database (PlayerDatabase): The database containing player information.
            - game_database (GameDatabase): The database containing game information.
            - tournament_database (TournamentDatabase): The database containing tournament information.
            - restricted (bool, optional): Whether to compute the restricted leaderboard. Defaults to False.

        Returns:
            - pandas.DataFrame: The computed leaderboard as a pandas DataFrame.
        """
        leaderboard = []
        for player in player_database:
            wins = player.get_number_of_wins(game_database.get_games_per_player(player.id))
            losses = player.get_number_of_losses(game_database.get_games_per_player(player.id))
            draws = player.get_number_of_draws(game_database.get_games_per_player(player.id))
            rating = player.get_rating()
            last_game_date = [game.get_date() for game in game_database.get_games_per_player(player.id, allow_forfeit=True)]
            if len(last_game_date) == 0:
                continue
            last_game_date = max(last_game_date)
            condition_met = wins + losses + draws >= 12 and last_game_date >= datetime.now() - timedelta(days=365)
            if not restricted or condition_met:
                leaderboard.append((player.name, int(rating.rating), wins, losses, draws))
        leaderboard.sort(key=lambda x: x[1], reverse=True)
        leaderboard = pd.DataFrame(leaderboard, columns=["Name", "Rating", "Wins", "Losses", "Draws"])
        return leaderboard

    @staticmethod
    def compute(player_database : PlayerDatabase = None, game_database : GameDatabase = None, 
                tournament_database : TournamentDatabase = None, rating_system: RatingSystem = None, 
                save_folder : str = None, file_name : str = 'leaderboard.csv') -> pd.DataFrame:
        leaderboard = Leaderboard.compute_leaderboard(player_database, game_database, tournament_database, restricted=True)
        if save_folder and file_name:
            leaderboard.to_csv(os.path.join(save_folder, file_name), index=False)
        return leaderboard
    

class AnonymousLeaderboard(Statistic):
    """
    Represents a restricted leaderboard for chess players.
    """

    @staticmethod
    def compute_leaderboard(player_database : PlayerDatabase, game_database : GameDatabase,
                            tournament_database : TournamentDatabase = None, anonymous_date=datetime(2024, 4, 28), 
                            anon_file='anonymous.txt', not_anom_file='not_anonymous.txt') -> pd.DataFrame:
        """
        Computes the leaderboard based on the provided player, game, and tournament databases.

        Args:
            - player_database (PlayerDatabase): The database containing player information.
            - game_database (GameDatabase): The database containing game information.
            - tournament_database (TournamentDatabase): The database containing tournament information.
            - anonymous_date (datetime, optional): The date before which players are only added anonymously. Defaults to datetime(2024, 4, 28).
            - anon_file (str, optional): The file containing the names of players wishing to remain anonymous. Defaults to 'anonymous.txt'.
            - not_anom_file (str, optional): The file containing the names of players not wishing to remain anonymous. Defaults to 'not_anonymous.txt'.

        Returns:
            - pandas.DataFrame: The computed leaderboard as a pandas DataFrame.
        """
        anon_names = []
        with open(os.path.join('data', anon_file), 'r') as f:
            for line in f:
                anon_names.append(line.strip())
        not_anon_names = []
        with open(os.path.join('data', not_anom_file), 'r') as f:
            for line in f:
                not_anon_names.append(line.strip())
        leaderboard = []
        for player in player_database:
            wins = player.get_number_of_wins(game_database.get_games_per_player(player.id))
            losses = player.get_number_of_losses(game_database.get_games_per_player(player.id))
            draws = player.get_number_of_draws(game_database.get_games_per_player(player.id))
            rating = player.get_rating()
            last_game_date = [game.get_date() for game in game_database.get_games_per_player(player.id, allow_forfeit=True)]
            if len(last_game_date) == 0:
                continue
            last_game_date = max(last_game_date)
            condition_met = last_game_date >= datetime.now() - timedelta(days=300)
            is_anon = (last_game_date < anonymous_date and player.name not in not_anon_names) or player.name in anon_names
            n_games = wins + losses + draws
            question_mark = ' (?)' if n_games < 12 else ''
            if condition_met:
                if not is_anon:
                    leaderboard.append((player.name, str(int(rating.rating)) + question_mark, wins, losses, draws))
                else:
                    leaderboard.append(("Anonymous", str(int(rating.rating)), '', '', ''))
        leaderboard.sort(key=lambda x: int(x[1].split(' ')[0]), reverse=True)
        leaderboard = pd.DataFrame(leaderboard, columns=["Name", "Rating", "Wins", "Losses", "Draws"])
        leaderboard['Rank'] = [i + 1 for i in range(len(leaderboard))]
        return leaderboard

    @staticmethod
    def compute(player_database : PlayerDatabase = None, game_database : GameDatabase = None, 
                tournament_database : TournamentDatabase = None, rating_system: RatingSystem = None, save_folder : str = None, 
                file_name : str = 'anonymized_leaderboard.csv', anonymous_date=datetime(2024, 4, 28), 
                anon_file='anonymous.txt', not_anom_file='not_anonymous.txt') -> pd.DataFrame:
        leaderboard = AnonymousLeaderboard.compute_leaderboard(player_database, game_database, tournament_database, 
                                                               anonymous_date=anonymous_date, 
                                                               anon_file=anon_file, not_anom_file=not_anom_file)
        if save_folder and file_name:
            leaderboard.to_csv(os.path.join(save_folder, file_name), index=False)
        return leaderboard

class TournamentRanking(TournamentStatistic):
    """
    Represents the ranking statistics for a chess tournament.
    """

    @staticmethod
    def compute(player_database : PlayerDatabase = None, game_database : GameDatabase = None, 
                tournament : Tournament = None, rating_system: RatingSystem = None, save_folder : str = None, 
                file_name : str = "leaderboard.csv") -> pd.DataFrame:
        ranking = []
        for player_stats in tournament.get_results():
            player = player_database[player_stats['player']]
            player_rating = player.get_rating_at_date(tournament.get_date(), next=True)
            ranking_info = [player.name, int(player_rating.rating), player_stats['score']]
            for tie_break_name in tournament.tie_break_names:
                ranking_info.append(player_stats[tie_break_name])
            ranking_info.append(int(player_stats['rating_performance'].rating))
            ranking.append(ranking_info)
        ranking.sort(key=lambda x: tuple(x[2:]), reverse=True)
        ranking = pd.DataFrame(ranking, columns=["Name", "Rating", "Score"] + tournament.tie_break_names + ["Performance"])
        ranking['Rank'] = np.arange(1, len(ranking) + 1)
        if save_folder and file_name:
            ranking.to_csv(os.path.join(save_folder, file_name), index=False)
        return ranking
    
class WinRateByHome(Statistic):
    """
    A class representing the win rate by home statistic.
    """

    @staticmethod
    def create_figure(wins : List[int], save_path : str) -> tuple:
        """
        Creates a bar plot figure showing the win rate by home.

        Args:
            - wins (list): A list containing the number of wins for both home and out.
            - save_path (str): The path to save the figure.
        
        Returns:
            - tuple: A tuple containing the figure and axis objects.
        """
        fig, ax = plt.subplots()

        ax = sns.barplot(x=["Home", "Out", "Draw"], y=wins, color=Statistic.default_color, alpha=0.7)
        ax.set_title("Win Rate by Color", fontsize=14)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Home", "Out", "Draw"], fontsize=12)
        ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
        ax.set_yticklabels(["0%", "10%", "20%", "30%", "40%", "50%"], fontsize=12)
        sns.despine(left=True, bottom=True)
        ax.set_facecolor((0.95, 0.95, 0.95))
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path)
        return fig, ax

    @staticmethod
    def compute_wins(games : Generator[Game, None, None]) -> np.ndarray:
        """
        Computes the number of wins for each color.

        Args:
            - games (Generator[Game, None, None]): The database containing game information.

        Returns:
            - wins (numpy.ndarray): An array containing the number of wins for each color.
        """
        wins = [0, 0, 0]
        for game in games:
            if game.get_result() == 1:
                wins[0] += 1
            elif game.get_result() == 0:
                wins[1] += 1
            else:
                wins[2] += 1
        wins = np.array(wins)
        wins = wins / np.sum(wins)
        return wins

    @staticmethod
    def compute(player_database : PlayerDatabase = None, game_database : GameDatabase = None, 
                tournament_database : TournamentDatabase = None, rating_system: RatingSystem = None, 
                save_folder : str = None, file_name : str = "win_rate_by_color.png") -> tuple:
        wins = WinRateByHome.compute_wins(game_database.get_games_no_forfeit())
        save_path = None
        if save_folder and file_name:
            save_path = os.path.join(save_folder, file_name)
        return WinRateByHome.create_figure(wins, save_path)
    
class WinRateByHomeTournament(TournamentStatistic):
    """
    A class representing the win rate by color for a tournament.
    """

    @staticmethod
    def compute(player_database : PlayerDatabase = None, game_database : GameDatabase = None, 
                tournament : Tournament = None, rating_system: RatingSystem = None, 
                save_folder=None, file_name : str = "win_rate_by_color.png") -> tuple:
        games = game_database.get_games_per_tournament(tournament.id)
        wins = WinRateByHome.compute_wins(games)
        save_path = None
        if save_folder and file_name:
            save_path = os.path.join(save_folder, file_name)
        return WinRateByHome.create_figure(wins, save_path)
    
class WinRatingDifference(Statistic):
    """
    A class that computes and visualizes win chances based on rating difference between players.
    """
    @staticmethod
    def compute_rating_differences(game_database : GameDatabase, 
                                   player_database : PlayerDatabase) -> pd.DataFrame:
        """
        Computes the rating differences and results for each game.

        Args:
            - game_database (GameDatabase): Game database object.
            - player_database (PlayerDatabase): Player database object.

        Returns:
            - pandas.DataFrame: DataFrame containing the rating differences and results.
        """
        win_chances_home = []
        win_chances_out = []
        for game in game_database.get_games_no_forfeit():
            home = player_database[game.home]
            out = player_database[game.out]
            rating_difference = home.get_rating().rating - out.get_rating().rating
            win_chances_home.append((rating_difference, game.get_result()))
            win_chances_out.append((-rating_difference, 1 - game.get_result()))
        win_chances_home = pd.DataFrame(win_chances_home, 
                                         columns=["Rating Difference", "Win Chance"])
        win_chances_out = pd.DataFrame(win_chances_out, 
                                         columns=["Rating Difference", "Win Chance"])
        win_chances = pd.concat([win_chances_home, win_chances_out])
        return win_chances
    
    @staticmethod
    def create_figure(win_chances : pd.DataFrame, save_path : str, rating_system : RatingSystem, mean_deviation : float, 
                      mean_rating : float, max_rating : float, min_rating : float) -> tuple:
        """
        Creates a line plot of win chances by rating difference.

        Args:
            - win_chances (pandas.DataFrame): DataFrame containing the rating differences and win chances.
            - save_path (str): Path to save the figure.
            - rating_system (RatingSystem): Rating System used in the computation of the ratings
            - mean_deviation (float): mean deviation across all players

        Returns:
            - tuple: A tuple containing the figure and axis objects.
        """
        # plot the mean score in each bin
        bins = np.linspace(min_rating - mean_rating, max_rating - mean_rating, 20)
        win_chances["Rating Difference"] = pd.cut(win_chances["Rating Difference"], bins)
        # set rating difference to the middle of the bin
        win_chances["Rating Difference"] = win_chances["Rating Difference"].apply(lambda x: x.mid)

        fig, ax = plt.subplots()
        # use groupby to get the mean and std of each bin
        win_chances = win_chances.groupby("Rating Difference", observed=False).mean()
        win_chances = win_chances.reset_index()
        ax = sns.lineplot(data=win_chances, x="Rating Difference", y="Win Chance", 
                          color=Statistic.default_color, label="Observed", ax=ax)
        x_theory = np.linspace(min_rating - mean_rating, max_rating - mean_rating, 500)
        x_ratings = [(Rating(mean_rating, mean_deviation), Rating(mean_rating - x, mean_deviation)) for x in x_theory]
        y_theory = [0.5 *(rating_system.compute_expected_score_rating(x1, x2, True) + rating_system.compute_expected_score_rating(x1, x2, False)) for (x1, x2) in x_ratings]
        sns.lineplot(x=x_theory, y=y_theory, label='Theoretical', ax=ax)
        xlim = min(mean_rating - min_rating, max_rating - mean_rating)
        ax.set_xlim(-xlim,xlim)
        ax.set_title("Win Chances by Rating Difference", fontsize=14)
        # remove y label
        ax.set_ylabel("")
        ax.set_xlabel("Rating Difference", fontsize=12)
        # set y ticks from 0 to 1
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"], fontsize=12)
        # remove axis lines
        sns.despine(left=True, bottom=True)
        ax.set_facecolor((0.95, 0.95, 0.95))
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path)
        return fig, ax
    
    @staticmethod
    def compute(player_database : PlayerDatabase = None, game_database : GameDatabase = None, 
                tournament_database : TournamentDatabase = None, rating_system: RatingSystem = None, save_folder : str = None, 
                file_name : str = "win_rating_difference.png") -> tuple:
        mean_deviation = np.sqrt(np.mean([player.get_rating().deviation ** 2 for player in player_database if player.get_rating().rating != DEFAULT_RATING.rating]))
        max_rating = max([player.get_rating().rating for player in player_database if player.get_rating().rating != DEFAULT_RATING.rating])
        min_rating = min([player.get_rating().rating for player in player_database if player.get_rating().rating != DEFAULT_RATING.rating])
        mean_rating = np.mean([player.get_rating().rating for player in player_database if player.get_rating().rating != DEFAULT_RATING.rating])
        win_chances = WinRatingDifference.compute_rating_differences(game_database, player_database)
        save_path = None
        if save_folder and file_name:
            save_path = os.path.join(save_folder, file_name)
        return WinRatingDifference.create_figure(win_chances, save_path, rating_system, mean_deviation, mean_rating, max_rating, min_rating)

class RankingByMaxTournamentPerformance(Statistic):
    """
    A class that computes the ranking of players based on their maximum tournament performance.
    """

    @staticmethod
    def max_performances(player_database : PlayerDatabase, tournament_database : TournamentDatabase, 
                         performance_name : str) -> pd.DataFrame:
        """
        Computes the maximum performance for each player.

        Args:
            - player_database (PlayerDatabase): The database of players.
            - game_database (GameDatabase): The database of games.
            - tournament_database (TournamentDatabase): The database of tournaments.
            - performance_name (str): The name of the performance to compute.

        Returns:
            - pandas.DataFrame: The computed maximum performances.
        """
        max_performances = []
        for player in player_database:
            player_performances_ = tournament_database.get_player_performances(player.id)
            # only count performances were player played all games
            player_performances = []
            for performance in player_performances_:
                if performance['n_games'] == tournament_database[performance['tournament']].rounds:
                    player_performances.append(performance)
            
            if len(player_performances) > 0:
                # get the maximum performance and the tournament
                max_performance = max(player_performances, key=lambda x: x[performance_name].rating)
                max_ = max_performance[performance_name].rating
                tournament = tournament_database[max_performance['tournament']]
                max_performances.append((player.name, int(player.get_rating().rating),
                                         int(max_), 
                                         tournament.name, tournament.get_date().strftime("%d/%m/%Y")))
        max_performances.sort(key=lambda x: x[2], reverse=True)
        max_performances = pd.DataFrame(max_performances, 
                                        columns=["Name", "Rating", "Max Performance", "Tournament", "Date"])
        return max_performances

    @staticmethod
    def compute(player_database : PlayerDatabase = None, game_database : GameDatabase = None, 
                tournament_database : TournamentDatabase = None, rating_system: RatingSystem = None, save_folder : str = None,
                file_name : str = "max_performances.csv") -> pd.DataFrame:
        max_performances = RankingByMaxTournamentPerformance.max_performances(player_database, 
                                                                              tournament_database, 
                                                                              "rating_performance")
        if save_folder and file_name:
            max_performances.to_csv(os.path.join(save_folder, file_name), 
                                    index=False)
        return max_performances


class TournamentsPerPlayer(Statistic):
    """
    A class representing the statistic of the number of tournaments per player.
    """

    @staticmethod
    def create_figure(tournaments : List[int], save_path : str):
        """
        Create a figure showing the histogram of the number of tournaments per player.

        Args:
            - tournaments (list): A list of integers representing the number of tournaments for each player.
            - save_path (str): The path to save the figure.
        """
        fig, ax = plt.subplots()
        ax = sns.histplot(tournaments, color=Statistic.default_color, discrete=True, 
                          shrink=0.95, alpha=0.7, linewidth=0)
        ax.set_title("Number of Tournaments per Player", fontsize=14)
        ax.set_xlabel("Number of Tournaments", fontsize=12)
        ax.set_ylabel("")
        ax.set_facecolor((0.95, 0.95, 0.95))
        sns.despine(left=True, bottom=True)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path)
        return fig, ax
    
    @staticmethod
    def compute(player_database : PlayerDatabase = None, game_database : GameDatabase = None, 
                tournament_database : TournamentDatabase = None, rating_system: RatingSystem = None, save_folder : str = None, 
                file_name : str = "tournaments_per_player.png") -> tuple:
        tournaments = [len(tournament_database.get_player_performances(player.id)) for player in player_database]
        save_path = None
        if save_folder and file_name:
            save_path = os.path.join(save_folder, file_name)
        return TournamentsPerPlayer.create_figure(tournaments, save_path)
    

class MostSurprisingGames(Statistic):
    """
    A class representing the statistic for computing the most surprising games.
    """
    @staticmethod
    def compute_surprises(games : Generator[Game, None, None], player_database : PlayerDatabase, 
                          rating_system : RatingSystem,
                          n_games : int) -> pd.DataFrame:
        """
        Computes how surprising the result each of the given games is.

        Args:
            - games (Generator[Game, None, None]): The game database.
            - player_database (PlayerDatabase): The player database.
            - rating_system (RatingSystem): The rating system used to compute the ratings.
            - n_games (int): The number of most surprising games to compute.

        Returns:
            - pandas.DataFrame: A DataFrame containing the most surprising games.

        """
        games_rating_difference = []
        for game in games:
            home = player_database[game.home]
            out = player_database[game.out]
            if game.get_date() is not None:
                home_rating = home.get_rating_at_date(game.get_date(), next=True)
                out_rating = out.get_rating_at_date(game.get_date(), next=True)
            else:
                home_rating = home.get_rating()
                out_rating = out.get_rating()

            expected_score = rating_system.compute_expected_score(
                home, [game], player_database, game.get_date(), next=True
            )

            loss = game.get_result() * np.log(expected_score) + (1 - game.get_result()) * np.log(1 - expected_score)
            games_rating_difference.append((home.name, int(home_rating.rating),
                                            out.name, int(out_rating.rating),
                                            game.result, game.get_date().strftime("%d/%m/%Y"), loss, expected_score))
        games_rating_difference.sort(key=lambda x: x[-2])
        surprises = pd.DataFrame(games_rating_difference[:n_games], 
                                 columns=["Home", "Home Rating", "Out", "Out Rating",
                                          "Result", "Date", "Loss", "Expected Score"])
        return surprises
    
    @staticmethod
    def compute(player_database : PlayerDatabase = None, game_database : GameDatabase = None, 
                tournament_database : TournamentDatabase = None, rating_system: RatingSystem = None, 
                save_folder : str = None, file_name : str = 'most_surprising_games.csv', n_games : int = 50) -> pd.DataFrame:
        surprises = MostSurprisingGames.compute_surprises(game_database.get_games_no_forfeit(), 
                                                          player_database, rating_system,
                                                          n_games)
        if save_folder and file_name:
            surprises.to_csv(os.path.join(save_folder, file_name), index=False)
        return surprises
    

class MostSurprisingGamesTournament(TournamentStatistic):
    """
    A class representing the statistic of the most surprising games in a tournament.
    """
    @staticmethod
    def compute(player_database : PlayerDatabase = None, 
                game_database : GameDatabase = None, 
                tournament : Tournament = None, rating_system: RatingSystem = None, save_folder=None, 
                file_name : str = "most_surprising_games.csv", n_games : int = 10) -> pd.DataFrame:
        games =  game_database.get_games_per_tournament(tournament.id)
        surprises = MostSurprisingGames.compute_surprises(games, player_database, rating_system, 
                                                          n_games)
        if save_folder and file_name:
            surprises.to_csv(os.path.join(save_folder, file_name), index=False)
        return surprises
    

class NumberOfGamesLeaderboard(Statistic):
    """
    Represents a leaderboard that computes the number of games played by each player.
    """

    @staticmethod
    def compute(player_database : PlayerDatabase = None, game_database : GameDatabase = None, 
                tournament_database : TournamentDatabase = None, rating_system: RatingSystem = None, save_folder : str = None, 
                file_name : str = "number_of_games.csv") -> pd.DataFrame:
        leaderboard = []
        for player in player_database:
            leaderboard.append((player.name, int(player.get_rating().rating),
                                game_database.get_n_games_per_player(player.id)))
        leaderboard.sort(key=lambda x: x[2], reverse=True)
        leaderboard = pd.DataFrame(leaderboard, columns=["Name", "Rating", "Number of Games"])
        if save_folder and file_name:
            leaderboard.to_csv(os.path.join(save_folder, file_name), index=False)
        return leaderboard
    
class RatingDistribution(Statistic):
    """
    A class representing the rating distribution statistic.

    This statistic computes and visualizes the distribution of ratings for a given set of players.
    """

    @staticmethod
    def create_figure(ratings : List[float], save_path : str) -> tuple:
        """
        Create a figure of the rating distribution.

        Args:
            - ratings (list): A list of ratings.
            - save_path (str): The path to save the figure.
        
        Returns:
            - tuple: A tuple containing the figure and axis objects.
        """
        fig, ax = plt.subplots()
        ax = sns.histplot(ratings, color=Statistic.default_color, kde=False, 
                          shrink=0.95, alpha=0.7, linewidth=0)
        ax.set_title("Rating Distribution", fontsize=14)
        ax.set_xlabel("Rating", fontsize=12)
        ax.set_ylabel("")
        ax.set_facecolor((0.95, 0.95, 0.95))
        sns.despine(left=True, bottom=True)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path)
        return fig, ax
    
    @staticmethod
    def compute(player_database : PlayerDatabase = None, game_database : GameDatabase = None, 
                tournament_database : TournamentDatabase = None, rating_system: RatingSystem = None, save_folder : str = None, 
                file_name : str = "rating_distribution.png") -> tuple:
        ratings = [player.get_rating().rating for player in player_database if player.get_rating().rating != DEFAULT_RATING.rating]
        save_path = None
        if save_folder and file_name:
            save_path = os.path.join(save_folder, file_name)
        return RatingDistribution.create_figure(ratings, save_path)
    

class MostSurprisingPerformance(Statistic):
    """
    A class representing the statistic for the most surprising rating increases in a single round update.
    """

    @staticmethod
    def compute_surprises(player_database : PlayerDatabase, n_performances : int) -> pd.DataFrame:
        """
        Computes the most surprising performances based on player and tournament databases.

        Args:
            - player_database (PlayerDatabase): The database containing player information.
            - n_performances (int): The number of performances to consider.

        Returns:
            - pandas.DataFrame: A DataFrame containing the most surprising performances.
        """
        players_boosts = [(player, player.rating_boost()) for player in player_database]
        players_boosts = [boost for boost in players_boosts if boost[1] is not None]
        players_boosts = sorted(players_boosts, key=lambda x: x[1][0], reverse=True)
        players_boosts = players_boosts[:n_performances]
        players_boosts = [(player.name, int(player.get_rating().rating), 1 - norm.cdf(performance[0]),
                           performance[2].rating, performance[1].rating, performance[3]) 
                          for player, performance in players_boosts]
        # to dataframe
        players_boosts = pd.DataFrame(players_boosts, 
                                      columns=["Name", "Rating", "Probability", "Start", "End", "Date"])
        return players_boosts
    
    @staticmethod
    def compute(player_database : PlayerDatabase = None, game_database : GameDatabase = None, 
                tournament_database : TournamentDatabase = None, rating_system: RatingSystem = None, save_folder : str = None, 
                file_name : str = "most_surprising_performances.csv", n_performances : int = 50) -> pd.DataFrame:
        players_boosts = MostSurprisingPerformance.compute_surprises(player_database, n_performances)
        if save_folder and file_name:
            players_boosts.to_csv(os.path.join(save_folder, file_name), index=False)
        return players_boosts


class MostSurprisingTournamentPerformance(TournamentStatistic):
    """
    A class representing the statistic for the most surprising performance in a tournament.
    """

    @staticmethod
    def compute_surprises(player_database : PlayerDatabase, game_database : GameDatabase, 
                          tournament : Tournament, restrict : int = None) -> pd.DataFrame:
        """
        Computes the most surprising performances based on player and tournament databases.

        Args:
            - player_database (PlayerDatabase): The database containing player information.
            - game_database (GameDatabase): The database containing game information.
            - tournament (Tournament): The tournament for which to compute the most surprising performances.
            - restrict (int): Minimum number of games to play before being included in the leaderboard.

        Returns:
            - pandas.DataFrame: A DataFrame containing the most surprising performances.
        """
        player_boosts = []
        for player in tournament.get_players(player_database):
            rating_before_tournament = player.get_rating_at_date(tournament.get_date(), next=False)
            tournament_performance = tournament.get_player_performance(player.id)
            performance_tournament = tournament_performance['rating_performance']
            dev = np.sqrt(rating_before_tournament.deviation ** 2 + performance_tournament.deviation ** 2)
            games = [game for game in game_database.get_games_per_player(player.id) if game.get_date() < tournament.get_date()]
            if restrict is not None and len(games) < restrict:
                continue
            player_boosts.append((player, (performance_tournament.rating - rating_before_tournament.rating) / dev, 
                                  performance_tournament.rating))
        player_boosts = sorted(player_boosts, key=lambda x: x[1], reverse=True)
        player_boosts = [(player.name, int(player.get_rating().rating), 1 - norm.cdf(performance), tournament_perf,
                           tournament.name, tournament.get_date().strftime("%d/%m/%Y"))
                          for player, performance, tournament_perf in player_boosts]
        # to dataframe
        player_boosts = pd.DataFrame(player_boosts, 
                                      columns=["Name", "Rating", "Probability", "Tournament Performance", "Tournament", "Date"])
        return player_boosts
    
    @staticmethod
    def compute(player_database : PlayerDatabase = None, game_database : GameDatabase = None, 
                tournament : Tournament = None, rating_system: RatingSystem = None, save_folder : str = None, 
                file_name : str = "most_surprising_performances.csv") -> pd.DataFrame:
        players_boosts = MostSurprisingTournamentPerformance.compute_surprises(player_database, game_database, tournament)
        if save_folder and file_name:
            players_boosts.to_csv(os.path.join(save_folder, file_name), index=False)
        return players_boosts


class MostSurprisingRestrictedTournamentPerformance(TournamentStatistic):
    """
    A class representing the statistic for the most surprising performance in a tournament, restricted to players with at least some number of games.
    """
    
    @staticmethod
    def compute(player_database : PlayerDatabase = None, game_database : GameDatabase = None, 
                tournament : Tournament = None, rating_system: RatingSystem = None, save_folder : str = None, 
                file_name : str = "most_surprising_performances_restricted.csv", minimum_games : int = 10) -> pd.DataFrame:
        players_boosts = MostSurprisingTournamentPerformance.compute_surprises(player_database, game_database, 
                                                                               tournament, restrict=minimum_games)
        if save_folder:
            players_boosts.to_csv(os.path.join(save_folder, file_name), 
                                  index=False)
        return players_boosts
            

class SharedRatingStatistic(Statistic):
    """A class representing the shared ratings statistic. This statistic returns the shared ratings of the rating system, if it has any."""


    @staticmethod
    def compute(player_database : PlayerDatabase = None, game_database : GameDatabase = None, 
                tournament_database : TournamentDatabase = None, rating_system: RatingSystem = None, save_folder : str = None, 
                file_name : str = "shared_ratings.csv") -> pd.DataFrame:
        shared_ratings = []
        if hasattr(rating_system, "shared_rating_histories"):
            for advantage, rating in zip(rating_system.shared_advantages, rating_system.shared_rating_histories):
                shared_ratings.append((advantage[0], str(advantage[1]), rating.get_rating().rating, rating.get_rating().deviation))
        shared_ratings = pd.DataFrame(shared_ratings, columns=["Name", "Matching", "Rating", "Deviation"])
        if save_folder:
            shared_ratings.to_csv(os.path.join(save_folder, file_name), index=False)
        return shared_ratings