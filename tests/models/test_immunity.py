from unittest import TestCase

from models.immunity import immunity_by_year_day


class ImmunityTests(TestCase):
    test_days = [0, 30, 91, 182, 273, 334, 365]

    def test_immunity_by_year_0_to_1(self):
        expected_results = [0.0, 0.07, 0.5, 1.0, 0.51, 0.07, 0.0]
        results = []
        for day in self.test_days:
            immunity = immunity_by_year_day(day=day, output_low=0.0, output_high=1.0)
            results.append(immunity)

        results = [round(r, 2) for r in results]

        assert results == expected_results

    def test_immunity_year_change_stability(self):
        expected_results = [0.0, 0.07, 0.5, 1.0, 0.51, 0.07, 0.0]
        results = []
        for day in self.test_days:
            immunity = immunity_by_year_day(day=day+365, output_low=0.0, output_high=1.0)
            results.append(immunity)

        results = [round(r, 2) for r in results]

        assert results == expected_results

    def test_immunity_by_year_02_to_08(self):
        expected_results = [0.2, 0.24, 0.5, 0.8, 0.5, 0.24, 0.2]
        results = []
        for day in self.test_days:
            immunity = immunity_by_year_day(day=day, output_low=0.2, output_high=0.8)
            results.append(immunity)

        results = [round(r, 2) for r in results]

        assert results == expected_results
