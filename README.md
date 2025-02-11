# Polyrating: A Cost-Effective and Bias-Aware Rating System for LLM Evaluation

This repository contains a general-purpose library for generating ratings using well-known and new rating systems, such as Elo, Glicko, TrueSkill, and Polyrating. Specifically, it currently serves the following purposes:
- It contains the code to reproduce the results from our paper *Polyrating: A Multivariate Rating System for Language Model Evaluation*. 
- It can be used to compute rating systems for people's projects.

This README is structured as follows:
- [Installation and Basic Use](#installation-and-basic-use): Provides details on how to install and use the code.
- [Further Use](#further-use): Provides a more in-depth look at various features of the code that can be useful for more advanced use cases. How to use our rating system, `Polyrating`, is also explained here in further detail.
- [Reproducing Results](#reproducing-results): Provides a step-by-step overview of how to reproduce the results from our paper.

Feel free to open an issue when encountering any bug, having a feature request, or for any other questions regarding this repository.

## Installation and Basic Use

You can install the code by installing [Conda](https://docs.anaconda.com/free/miniconda/) and running the following in your command prompt:

```bash
cd path/to/this/folder
conda create -n ratings python=3.11 -y
conda activate ratings
python -m pip install -e .
```

This allows you to run the program.

### Basic Use

This repository provides an easy way to manage games, select a rating system, and obtain a leaderboard for the players in your database. For example, the following code sets up a manager and adds a few games to the database. Note that every rating system requires a rating period to be set, where a rating period is a period of time where ratings remain static.

```python
from rating import Manager, Elo, RatingPeriodEnum, DetailedLeaderboard
from datetime import timedelta, datetime

manager = Manager(
    rating_system=Elo(),
    rating_period_type=RatingPeriodEnum.TIMEDELTA, # sets the constant rating period to be defined by some timedelta
    custom_timedelta=timedelta(days=30), # use monthly rating periods
)

game = manager.add_game(
    "Jasper Dekoninck", # home player
    "Magnus Carlsen", # out player
    "1-0", # result, either 1-0, 0-1, 1/2-1/2
 datetime.now(), # data of the game
)
manager.add_game("Jasper Dekoninck", "Hikaru Nakamura", 
                 "1-0", datetime.now())
manager.add_game("Magnus Carlsen", "Hikaru Nakamura", 
                 "1/2-1/2", datetime.now())
manager.update_rating() # updates ratings
leaderboard = DetailedLeaderboard.compute(manager.player_database, manager.game_database) # computes leaderboard
manager.save('data/databases.json') # saves database
manager = Manager.load('data/databases.json') # load database
```

This basic script should be sufficient for most use cases. Note that you can use all of the following rating systems: `Glicko`, `Glicko2`, `TrueSkillThroughTime`, `EloPlusPLus`, `ChessMetrics` and of course, several variants of Polyrating: `PolyratingCrossEntropy`, `PolyRatingRao`, `PolyRatingDavidson`, and `PolyratingAccuracy`. The hyperparameters for each of these options are explained in the documentation. 

## Further Use
This section explains several further features implemented in this library.

### Tournaments
You can also manage tournaments within your rating system.

```python
from rating import Manager, Tournament
from datetime import datetime

manager = Manager()
tournament = Tournament("Our awesome tournament", datetime.now(), rounds=7, time_control='5+3') # rounds and time_control are completely optional
manager.add_tournament(tournament=tournament)
game = manager.add_game("Jasper Dekoninck", "Hikaru Nakamura", 
                 "1-0", datetime.now(), tournament_id=tournament.id) # adds the game to the tournament
```
By using the tournament object, one can compute tournament-specific statistics, as explained further in the section on [statistics](#statistics).

The ETH Chess Club uses the [VegaChess](https://www.vegachess.com/ns/) software to administer our tournaments. Therefore, we have implemented additional features that interact with this software for the automation of our pipeline.

```python
from rating import Manager, Tournament

manager = Manager()
manager.add_tournament("path/to/tournament", "Our awesome tournament") # automatically adds all games and players from the tournament that used VegaChess
```

### Default Rating
By default, player ratings are initialized at 1500 with a standard deviation of 500. To change this, simply do:
```python
from rating import DEFAULT_RATING
DEFAULT_RATING.set_default(rating=1000, deviation=250)
```

### Object Handling

Apart from adding games to the database, you can also invoke the following functions separately:
```python
from rating import Manager, Tournament
from datetime import datetime

manager = Manager()
tournament = Tournament("Our awesome tournament", datetime.now())
manager.add_tournament(tournament=tournament)
game = manager.add_game("Jasper Dekoninck", "Hikaru Nakamura", 
                 "1-0", datetime.now())
manager.add_player("Magnus Carlsen") # only adds a player
manager.remove_game(game=game)
manager.remove_player('Jasper Dekoninck') # removes the player and all games associated with that player
manager.remove_tournament(tournament=tournament) # removes the tournament and all games associated with the tournament
```

### Complex Results
It is possible to add more complex results to games. The following results are all valid:
- `1-0`, `0-1`, `1/2-1/2`: Standard results.
- `1-0F`, `1F-0`: Any `F` in a result will be counted as a forfeit. Forfeits are not used when computing ratings.
- `0.6-0.4`, `3-1`, `4-1`: More complex results. Some rating systems are able to leverage these results to obtain improved ratings.

Furthermore, a `Game` object can be instantiated with various point systems for computing the winner of a tournament. By default, each player gets 1 point for a win, 0.5 for a draw, and 0 for a loss. You can change these defaults as follows:
```python
from rating import Manager, Game
from datetime import datetime

manager = Manager()
player1 = manager.add_player('Manchester United')
player2 = manager.add_player('Manchester City')

game = Game(player1.id, player2.id, "2-2", datetime.now(), 
            points_for_win=3, points_for_tie=1, points_for_loss=0)
manager.add_game(game=game)
```

### Statistics
You can compute a variety of statistics on your data. The easiest way to compute these statistics is by simply calling `manager.compute_statistics()`. All statistics will then be automatically computed and stored in the `data` folder. In the following table, you can find the description of all statistics that you can compute.

| Statistic      | Description | Stored in |
| ----------- | ----------- | ----------- |
| Leaderboard | Contains the rating for each player. Removes players that have played 12 games or less. Used for the ETH Chess Club. | `leaderboard.csv` |
| DetailedLeaderboard | Contains the rating for each player. | `detailed_leaderboard.csv` |
| AnonymousLeaderboard | Contains the rating for each player. Player names are anonymized if they only played before May 2024. Used for the ETH Chess Club. | `anonymized_leaderboard.csv` |
| Tournament Ranking | Computes the ranking for a specific tournament, along with tournament performances. |`ranking.csv`|
| Win Rate by Color | Computes how often the home player wins/loses/draws. | `win_rate_by_color.png`|
| Rating Distribution | Computes a plot of the rating distribution over all players | `rating_distribution.png`|
| Tournaments per player | Shows a histogram of the amount of tournaments a player has played. | `tournaments_per_player.png`|
| Win Rating Difference | Shows a plot of the win probability over various rating differences | `win_rating_difference.png`|
| Games per Player | Shows the sum of the number of games per player. | `number_of_games.csv`|
| Most Surprising Games | Shows the games where a player beats the odds by beating (or drawing) a much stronger player. | `most_surprising_games.csv`|
| Most Surprising Performances | Shows the highest rating increases for one tournament, normalized by the deviation. | `most_surprising_performances.csv` |

### Polyrating and Multivariate rating
Finally, we explain how to use our own `Polyrating` system using this repository. To compute a multivariate rating, you first need to add metadata to each game and player. In this repository, we call this metadata `advantages`. For example, the following code shows how to add some basic advantages to a game:

```python
from rating import Manager, Game
from datetime import datetime

manager = Manager()
player1 = manager.add_player('Jasper Dekoninck')
player2 = manager.add_player('Hikaru Nakamura')

game = Game(player1.id, player2.id, "1-0", datetime.now(), 
            advantages_home={'home': 1, 'blitz': 1}, advantages_out={'home': 0, 'blitz': 1})
manager.add_game(game=game)
```
Only the rating systems based on `Polyrating` can use these advantages to fit a better rating. The following code shows a typical way to use advantages:
```python
from rating import Manager, Game, PolyratingCrossEntropy, Matching, DefaultRating

rating_system = PolyratingCrossEntropy(
    advantages={'blitz': DefaultRating(0, 50)}, # each player will have an additional blitz rating, initialized as 0 with deviation of 50
    shared_advantages=[('home', Matching(), DefaultRating(0, 50), 5)] # shared over all players
)
manager = Manager(rating_system=rating_system)
...
```
`Matching` is a class that enables you to match each shared advantage with a specific set of players. An empty matching means it matches all players (more than that you will likely never need). The 5 at the end indicates the extra deviation that is added between rating periods. More technically, it is the deviation between consecutive time steps of the Markov Chain for that rating. This ensures it does not need to stay constant over time, but can change a bit between rating periods.

This extra deviation between rating periods can also be added for the non-shared advantages as follows:
```python
rating_system = PolyratingCrossEntropy(
    advantages={'blitz': DefaultRating(0, 50)},
    omegas_advantages={'blitz': 5}, # extra deviation for the advantage
    omega=5, # extra deviation for the base ratings
    shared_advantages=[('home', Matching(), DefaultRating(0, 50), 5)]
)
```

Finally, the `linearized` parameter allows you to introduce an approximation in the rating system since it is quite expensive to run. Essentially, if you set this parameter to `k`, the rating system will only optimize the rating over the last `k` rating periods and use the rating of each player from before this period to initialize its ratings instead of the default rating. A bit complicated, but all that matters is that the lower you set this, the faster the algorithm works, but the more approximate it becomes.

## Reproducing Results

To reproduce the results, you first need to install the package as described above and then run:
```bash
python -m pip install -r requirements.txt
bash scripts/paper/main.sh
```
This will download all datasets and run the code. We note that each file in this bash script is run consecutively on a single core, which will take a lot of time. You can adjust the number of cores used in the file manually to run things quicker. Furthermore, each line in the script is annotated with the name of the figure or table that it generates the data for. Results are stored in the `results` folder in interpretable csv files. In the following list, we mention to which results in the paper the csv files correspond.

- Table 1a / Table 6a: `lmsys_released_shared_ratings.csv`
- Table 1b / Table 6b: `wildbench_released_shared_ratings.csv`
- Fig 2a: `sample_efficient_is_chinese.csv`
- Fig 2b: `sample_efficient_is_code.csv`
- Fig 2c: `sample_efficient_is_hard.csv`
- Fig 3a: `sample_efficient_wildbench.csv`
- Fig 3b: `sample_efficient_mixeval.csv`
- Fig 3c: `sample_efficient_is_code_is_chinese.csv`
- Table 2 / table 7: `leaderboard_polyrating.csv`
- Table 3 / Table 8: `leaderboard_univariate*.csv` (all files that start with `leaderboard_univariate`)
- Fig 4:  `alternatives.csv`

## Citation

```
@article{dekoninck2024polyrating,
      title={Polyrating: A Cost-Effective and Bias-Aware Rating System for LLM Evaluation}, 
      author={Jasper Dekoninck and Maximilian Baader and Martin Vechev},
      year={2024},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```