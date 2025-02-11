import unittest
from rating import Manager, RatingPeriodEnum, Tournament
from datetime import datetime, timedelta
import os

class TestBasic(unittest.TestCase):
    def test_manager_add_remove_players(self):
        manager = Manager()
        manager.add_player("player1")
        manager.add_player("player2")
        self.assertEqual(len(manager.player_database), 2)
        manager.add_player("player1")
        self.assertEqual(len(manager.player_database), 2)
        manager.remove_player("player1")
        self.assertEqual(len(manager.player_database), 1)

    def test_manager_add_remove_games(self):
        manager = Manager()
        manager.add_game(
            "player1", "player2", "1-0", datetime(2021, 1, 1)
        )
        self.assertEqual(len(manager.game_database), 1)
        self.assertEqual(len(manager.player_database), 2)
        player1 = manager.player_database.get_player_by_name("player1")
        self.assertNotEqual(player1, None)
        self.assertEqual(len(list(manager.game_database.get_games_per_player(player1.id))), 1)
        manager.remove_player("player1")
        self.assertEqual(len(manager.game_database), 0)
        self.assertEqual(len(manager.player_database), 1)
        game = manager.add_game(
            "player1", "player2", "1-0", datetime(2021, 1, 1)
        )
        self.assertEqual(game.get_result(), 1)
        manager.remove_game(game.id)
        self.assertEqual(len(manager.game_database), 0)

    def test_trigger_period(self):
        manager = Manager(rating_period_type=RatingPeriodEnum.TIMEDELTA, custom_timedelta=timedelta(days=7))
        manager.add_game(
            "player1", "player2", "1-0", datetime(2021, 1, 1)
        )
        manager.add_game(
            "player1", "player2", "1-0", datetime(2021, 1, 8)
        )
        manager.trigger_new_period()
        self.assertEqual(len(manager.rating_period), 1)

        manager = Manager(rating_period_type=RatingPeriodEnum.TOURNAMENT)
        self.assertEqual(len(manager.rating_period), 0)
        tournament = Tournament("tournament1", datetime(2021, 1, 1), 7, '5+3')
        manager.add_tournament(tournament=tournament)
        self.assertEqual(len(manager.tournament_database), 1)
        manager.trigger_new_period(tournament)
        self.assertEqual(len(manager.rating_period), 1)

        manager = Manager(rating_period_type=RatingPeriodEnum.MANUAL)

        manager.trigger_new_period()
        manager.trigger_new_period()
        self.assertEqual(len(manager.rating_period), 2)

    def test_save_and_load(self):
        manager = Manager(rating_period_type=RatingPeriodEnum.TIMEDELTA, custom_timedelta=timedelta(days=7))
        manager.add_game(
            "player1", "player2", "1-0", datetime(2021, 1, 1)
        )
        manager.add_game(
            "player1", "player2", "1-0", datetime(2021, 1, 8)
        )
        manager.trigger_new_period()
        manager.save("test.json")
        manager2 = Manager.load("test.json")
        self.assertEqual(len(manager2.rating_period), 1)
        self.assertEqual(len(manager2.game_database), 2)
        self.assertEqual(len(manager2.player_database), 2)
        os.remove("test.json")

if __name__ == '__main__':
    unittest.main()