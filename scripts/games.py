from rating import Manager, Game
from datetime import datetime
import pandas as pd
import os

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Rating calculation for ETH chess players")
    parser.add_argument("--file", type=str, help="Input file name", default='data/games.csv')
    parser.add_argument("--database_path", type=str, help="Path to the database folder", default='data/databases.json')
    args = parser.parse_args()
    if os.path.isfile(args.database_path):
        manager = Manager.load(args.database_path)
    else:
        manager = Manager()
    
    games = pd.read_csv(args.file)

    for i, game in games.iterrows():
        # Jun 13, 2024 @ 1:17 PM
        date = datetime.strptime(game["Submission Time"], "%b %d, %Y @ %I:%M %p")
        name_white = game['First Name (White)'] + ' ' + game['Last Name (White)']
        name_black = game['First Name (Black)'] + ' ' + game['Last Name (Black)']
        result = game['Result of the game']
        if result == 'White won':
            result = '1-0'
        elif result == 'Black won':
            result = '0-1'
        else:
            result = '1/2-1/2'
        time_control = game['Time Control']

        manager.add_game(
            home_name=name_white,
            out_name=name_black,
            result_str=result,
            date=date,
        )

    manager.save(args.database_path)
