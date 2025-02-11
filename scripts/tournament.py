from rating import Manager, PlayerDatabase
import os

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Rating calculation for ETH chess players")
    parser.add_argument("--tournament_path", type=str, help="Path to the tournament folder", default=None)
    parser.add_argument("--tournament_name", type=str, help="Name of the tournament, if different than loaded", default=None)
    parser.add_argument("--force", action="store_true", help="Force the addition of the tournament, even if it already exists")
    parser.add_argument("--data_folder", type=str, help="Path to the data folder", default='data')
    parser.add_argument("--database_path", type=str, help="Path to the database folder", default='databases.json')
    parser.add_argument("--history_folder", type=str, help="Path to the history folder", default='history')
    parser.add_argument("--tournament_folder", type=str, help="Path to the tournament folder", default='tournaments')
    parser.add_argument('--reset', action='store_true', help='Reset the database')


    args = parser.parse_args()
    if args.reset:
        # remove current, history, databases and tournaments folder
        os.system(f"rm -rf {os.path.join(args.data_folder, args.history_folder)}")
        os.system(f"rm {os.path.join(args.data_folder, args.database_path)}")
        os.system(f"rm -rf {os.path.join(args.data_folder, args.tournament_folder)}")

    if not os.path.isfile(os.path.join(args.data_folder, args.database_path)):
        manager = Manager(player_database=PlayerDatabase(strict=False))
    else:
        manager = Manager.load(os.path.join(args.data_folder, args.database_path))

    if args.tournament_path is not None:
        tournament = manager.add_tournament(args.tournament_path, args.tournament_name, args.force)
        manager.update_rating()
        manager.compute_statistics(tournament, args.data_folder, args.history_folder, args.tournament_folder)
        manager.save(os.path.join(args.data_folder, args.database_path))
