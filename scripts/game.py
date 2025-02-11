from rating import Manager
from datetime import datetime
import os

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Rating calculation for ETH chess players")
    parser.add_argument("--white_name", type=str, help="Name of the white player", default=None)
    parser.add_argument("--black_name", type=str, help="Name of the black player", default=None)
    parser.add_argument("--result", type=str, help="Result of the game (1-0, 0-1 or 1/2-1/2)", default=None)
    parser.add_argument("--date", type=str, help="Date of the game dd/mm/YYYY", default=None)
    parser.add_argument("--tournament_id", type=int, help="Id of the tournament", default=None)
    parser.add_argument("--database_path", type=str, help="Path to the database folder", default='data/databases.json')

    args = parser.parse_args()
    if os.path.isfile(args.database_path):
        manager = Manager.load(args.database_path)
    else:
        manager = Manager()
    assert args.result in ["1-0", "0-1", "1/2-1/2"], "Invalid result"
    date = datetime.strptime(args.date, "%d/%m/%Y")
    manager.add_game(args.white_name, args.black_name, args.result, date, args.tournament_id)
    manager.save(args.database_path)