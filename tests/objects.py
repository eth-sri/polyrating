import unittest
import os
from rating import Game, Player, Tournament, Object, Rating, GameDatabase, DEFAULT_RATING
from datetime import datetime, timedelta

class TestDatabases(unittest.TestCase):
    def test_game(self):
        game = Game(0, 1, "1-0", datetime(2021, 1, 1))
        self.assertEqual(game.get_result(), 1)
        self.assertEqual(game.get_date(), datetime(2021, 1, 1))
        self.assertEqual(game.get_winner(), 0)

        game = Game(0, 1, "0-1", datetime(2021, 1, 1))
        self.assertEqual(game.get_result(), 0)
        self.assertEqual(game.get_date(), datetime(2021, 1, 1))
        self.assertEqual(game.get_winner(), 1)

        game = Game(0, 1, "1/2-1/2", datetime(2021, 1, 1))
        self.assertEqual(game.get_result(), 0.5)
        self.assertEqual(game.get_date(), datetime(2021, 1, 1))
        self.assertEqual(game.get_winner(), None)

    def test_object(self):
        object1 = Object()
        object2 = Object()
        self.assertEqual(object2.id, object1.id + 1)

    def test_rating(self):
        rating = Rating()
        self.assertEqual(rating.rating, DEFAULT_RATING.rating)
        self.assertEqual(rating.deviation, DEFAULT_RATING.deviation)
        self.assertEqual(rating.volatility, DEFAULT_RATING.volatility)
        copied_rating = rating.copy()
        copied_rating.update(100, 10, 0.06)
        self.assertEqual(copied_rating.rating, 100)
        self.assertEqual(copied_rating.deviation, 10)
        self.assertEqual(copied_rating.volatility, 0.06)
        self.assertEqual(rating.rating, DEFAULT_RATING.rating)
        self.assertEqual(rating.deviation, DEFAULT_RATING.deviation)
        self.assertEqual(rating.volatility, DEFAULT_RATING.volatility)
        rating.set(copied_rating)
        self.assertEqual(rating.rating, 100)
        self.assertEqual(rating.deviation, 10)
        self.assertEqual(rating.volatility, 0.06)
        copied_rating.reset()
        self.assertEqual(copied_rating.rating, DEFAULT_RATING.rating)
        self.assertEqual(copied_rating.deviation, DEFAULT_RATING.deviation)
        self.assertEqual(copied_rating.volatility, DEFAULT_RATING.volatility)
        DEFAULT_RATING.set_default(100, 10, 0.06)
        copied_rating.reset()
        self.assertEqual(copied_rating.rating, 100)
        self.assertEqual(copied_rating.deviation, 10)
        self.assertEqual(copied_rating.volatility, 0.06)

    def test_player(self):
        player = Player("Player First")
        self.assertEqual(player.name, "Player First")
        self.assertEqual(player.get_rating().rating, DEFAULT_RATING.rating)
        self.assertEqual(player.get_rating().deviation, DEFAULT_RATING.deviation)
        self.assertEqual(player.get_rating().volatility, DEFAULT_RATING.volatility)
        player.store_rating(datetime(2021, 1, 1))
        player.get_rating().update(100, 10, 0.06)
        self.assertEqual(player.get_rating_at_date(datetime(2021, 1, 1)).rating, DEFAULT_RATING.rating)
        self.assertEqual(player.get_rating_at_date(datetime(2021, 2, 1)).rating, 100)
        self.assertEqual(len(player.get_rating_history()), 1)
        db = GameDatabase()
        player2 = Player("Player Second")
        game = Game(player.id, player2.id, "1-0", datetime(2021, 1, 1))
        db.add(game)
        game2 = Game(player.id, player2.id, "1-0", datetime(2021, 1, 2))
        db.add(game2)
        game = Game(player.id, player2.id, "0-1", datetime(2021, 1, 3))
        db.add(game)
        game = Game(player.id, player2.id, "1/2-1/2", datetime(2021, 1, 4))
        db.add(game)
        self.assertEqual(player.get_number_of_draws(db), 1)
        self.assertEqual(player.get_number_of_wins(db), 2)
        self.assertEqual(player.get_number_of_losses(db), 1)

    def setUp(self) -> None:
        self.defaults = DEFAULT_RATING.rating, DEFAULT_RATING.deviation, DEFAULT_RATING.volatility

    def tearDown(self):
        DEFAULT_RATING.set_default(*self.defaults)




if __name__ == '__main__':
    unittest.main()