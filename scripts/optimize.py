from rating import Optimizer, Manager

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-evals", type=int, default=100)
    parser.add_argument('--include-default', action='store_true')
    parser.add_argument('--store-file', type=str, default='data/optimization.csv')
    parser.add_argument('--database-file', type=str, default='data/databases.json')
    parser.add_argument('--optimize-default-rating', action='store_true')
    parser.add_argument('--do-time', action='store_true')
    parser.add_argument('--no-omega', action='store_true')
    parser.add_argument('--n-players', type=int, default=1000)
    parser.add_argument('--n-games', type=int, default=10000)
    parser.add_argument('--n-rating-periods', type=int, default=30)
    parser.add_argument('--max-max-iters', type=int, default=100)
    parser.add_argument('--after', action='store_true')
    args = parser.parse_args()

    manager = Manager.load(args.database_file)
    game_database = manager.game_database
    player_database = manager.player_database
    tournament_database = manager.tournament_database
    optimizer = Optimizer(game_database, player_database, 
                          manager.rating_period, max_evals=args.max_evals, 
                          exclude_default=not args.include_default, 
                          optimize_default_rating=args.optimize_default_rating, 
                          no_omega=args.no_omega, before=not args.after)
    if not args.do_time:
        optimizer.optimize()
        optimizer.store_param_history(args.store_file)
    else:
        optimizer.extend_database(args.n_games, args.n_players, args.n_rating_periods)
        optimizer.time_vs_performance(args.store_file, args.max_max_iters)
    