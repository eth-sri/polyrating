import unittest
import os
from rating import Database, PlayerDatabase, GameDatabase, TournamentDatabase, Game, Player, Tournament, Object
from datetime import datetime, timedelta

class TestDatabases(unittest.TestCase):
    def test_default(self):
        db = Database()
        self.assertEqual(len(db), 0)
        object1 = Object()
        self.assertEqual(object1.id, 0)
        object2 = Object()
        self.assertEqual(object2.id, 1)
        db.add(object1)
        db.add(object2)
        self.assertEqual(len(db), 2)
        self.assertEqual(db[0], object1)
        self.assertEqual(db[1], object2)
        object3 = Object()
        self.assertEqual(db.check_duplicate(object3), False)
        self.assertEqual(db.check_duplicate(object1), True)
        self.assertEqual(db.get_max_id(), 1)
        self.assertEqual(db.get_last(), object2)
        for object in db:
            self.assertTrue(object in [object1, object2])
        db.remove(object1)
        self.assertEqual(len(db), 1)
        self.assertEqual(db[1], object2)

        db.save("test.json")
        db2 = Database.load("test.json")
        self.assertEqual(len(db2), 1)
        os.remove("test.json")

    def test_player(self):
        db = PlayerDatabase(strict=False)
        player1 = Player("Player First")
        player2 = Player("Player Second")
        db.add(player1)
        db.add(player2)
        self.assertEqual(len(db), 2)
        self.assertEqual(db.get_player_by_name("Player First"), player1)
        self.assertEqual(db.get_player_by_name("Player Second"), player2)
        self.assertEqual(db.get_player_by_name("Player F"), player1)
        self.assertEqual(db.get_player_by_name("first player"), player1)
        self.assertEqual(db.get_player_by_name("Player Fi"), None)
        self.assertEqual(db.get_player_by_name("Player Third"), None)
        game_db = GameDatabase()
        db.clear_empty(game_db)
        self.assertEqual(len(db), 0)

    def test_game(self):
        db = GameDatabase()
        game1 = Game(0, 1, "1-0", datetime(2021, 1, 1))
        game2 = Game(0, 2, "1-0", datetime(2021, 1, 2))
        db.add(game1)
        db.add(game2)
        self.assertEqual(len(db), 2)
        for game in db.get_games_per_player(0):
            self.assertTrue(game in [game1, game2])
        for game in db.get_games_per_player(1):
            self.assertTrue(game in [game1])
        self.assertEqual(db.get_n_games_per_player(0), 2)
        self.assertEqual(db.get_n_games_per_player(1), 1)
        self.assertEqual(db.get_n_games_per_player(2), 1)
        self.assertEqual(db.get_n_games_per_player(3), 0)
        db.remove(game1)
        self.assertEqual(len(db), 1)
        self.assertEqual(db.get_n_games_per_player(0), 1)
        tournament = Tournament("tournament1", datetime(2021, 1, 1), 7, '5+3')
        tournament2 = Tournament("tournament2", datetime(2021, 1, 1), 7, '5+3')
        game3 = Game(0, 1, "1-0", datetime(2021, 1, 1), tournament_id=tournament.id)
        game4 = Game(0, 1, "1-0", datetime(2021, 1, 1), tournament_id=tournament2.id)
        db.add(game3)
        db.add(game4)
        self.assertEqual(len(db), 3)
        for game in db.get_games_per_tournament(0):
            self.assertTrue(game in [game3])
        for game in db.get_games_per_tournament(1):
            self.assertTrue(game in [game4])
        self.assertEqual(db.get_n_games_per_tournament(0), 1)
        self.assertEqual(db.get_n_games_per_tournament(1), 1)
        db.remove(game3)
        self.assertEqual(len(db), 2)
        self.assertEqual(db.get_n_games_per_tournament(0), 0)


if __name__ == '__main__':
    unittest.main()