import unittest
from rating import Manager, RatingPeriodEnum, DefaultRating, DEFAULT_RATING, PolyratingRao, Elo, ChessMetrics, EloPlusPlus, Glicko, Glicko2, PolyratingCrossEntropy, PolyratingDavidson, Rating, TrueSkillThroughTime
from datetime import datetime, timedelta

class TestBasic(unittest.TestCase):
    def basic_test(self, rating_system, do_deviation=True, symmetric=True, add_second_bigger=True):
        manager = Manager(rating_system=rating_system, rating_period_type=RatingPeriodEnum.TIMEDELTA, 
                          custom_timedelta=timedelta(days=7))
        time = datetime(2021, 1, 1)
        time2 = datetime(2021, 1, 8)
        game = manager.add_game(
            "player1", "player2", "1-0", time
        )
        game2 = manager.add_game(
            "player1", "player2", "1-0", time2
        )

        player1, player2 = manager.player_database.get_player_by_name("player1"), manager.player_database.get_player_by_name("player2")
        manager.trigger_new_period()
        manager.update_rating()
        if add_second_bigger:
            self.assertTrue(player1.get_rating().rating > DEFAULT_RATING.rating)
            self.assertTrue(player2.get_rating().rating < DEFAULT_RATING.rating)
        self.assertEqual(len(player1.get_rating_history()), 1)
        self.assertEqual(len(player2.get_rating_history()), 1)

        self.assertTrue(player1.get_rating_at_date(time).rating <= player1.get_rating_at_date(time2).rating)
        self.assertTrue(player2.get_rating_at_date(time).rating >= player2.get_rating_at_date(time2).rating)
        if do_deviation:
            self.assertTrue(player1.get_rating().deviation < DEFAULT_RATING.deviation)
            self.assertTrue(player2.get_rating().deviation < DEFAULT_RATING.deviation)

        previous_rating, previous_deviation = player1.get_rating().rating, player1.get_rating().deviation

        manager.remove_game(game=game)
        manager.update_rating()
        if add_second_bigger:
            self.assertTrue(previous_rating > player1.get_rating().rating)
        if do_deviation:
            self.assertTrue(previous_deviation < player1.get_rating().deviation)


        manager = Manager(rating_system=rating_system, rating_period_type=RatingPeriodEnum.TIMEDELTA)
        for i in range(10):
            for j in range(i):
                manager.add_game(
                    f"player{i}", f"player{j}", "1-0", time
                )
        manager.trigger_new_period()
        manager.update_rating()
        ratings = [manager.player_database.get_player_by_name(f"player{i}").get_rating().rating for i in range(10)]
        self.assertEqual(sorted(ratings), ratings)
        if symmetric:
            for i in range(10):
                for j in range(i):
                    manager.add_game(
                        f"player{j}", f"player{i}", "1-0", time
                    )
            manager.update_rating()
            ratings = [manager.player_database.get_player_by_name(f"player{i}").get_rating().rating for i in range(10)]
            # assert all equal, up to a certain precision
            self.assertTrue(all([abs(ratings[i] - ratings[i+1]) < 1 for i in range(9)]))

            for i in range(10):
                for j in range(i):
                    manager.add_game(
                        f"player{i}", f"player{j}", "1/2-1/2", time
                    )
                    manager.add_game(
                        f"player{j}", f"player{i}", "1/2-1/2", time
                    )
            manager.update_rating()
            ratings = [manager.player_database.get_player_by_name(f"player{i}").get_rating().rating for i in range(10)]
            # assert all equal, up to a certain precision
            self.assertTrue(all([abs(ratings[i] - ratings[i+1]) < 1 for i in range(9)]))

    def test_polyrating_rating_rao(self):
        self.basic_test(PolyratingRao())

    def test_polyrating_rating_cross_entropy(self):
        self.basic_test(PolyratingCrossEntropy())

    def test_polyrating_rating_davidson(self):
        self.basic_test(PolyratingDavidson())

    def test_elo(self):
        self.basic_test(Elo(), False)

    def test_glicko(self):
        self.basic_test(Glicko())

    def test_glicko2(self):
        self.basic_test(Glicko2())
    
    def test_elo_plus_plus(self):
        self.basic_test(EloPlusPlus(), False, False)

    def test_chess_metrics(self):
        self.basic_test(ChessMetrics(), False, True, False)

    def test_true_skill_through_time(self):
        self.basic_test(TrueSkillThroughTime(), False, True, False)

if __name__ == '__main__':
    unittest.main()