from unittest import TestCase

from models.age import sample_age


class AgeSamplingTests(TestCase):
    age_ranges = {0: [0, 9],
                  1: [10, 19],
                  2: [20, 29],
                  3: [30, 39],
                  4: [40, 49],
                  5: [50, 59],
                  6: [60, 69],
                  7: [70, 79],
                  8: [80, 89],
                  9: [90, 100]}
    age_probs = {0: 0.1125, 1: 0.15, 2: 0.2, 3: 0.15, 4: 0.2, 5: 0.1, 6: 0.05, 7: 0.025, 8: 0.00625, 9: 0.00625}
    immunity_reduction_factors = {0: 0.2, 1: 0.05, 2: 0.0, 3: 0.0, 4: 0.05, 5: 0.1, 6: 0.15, 7: 0.2, 8: 0.25, 9: 0.3}

    age_params = {
        "age_ranges": age_ranges,
        "age_probs": age_probs,
        "immunity_reduction_factors": immunity_reduction_factors,
    }

    def test_age_sampling(self):
        for target_idx, factor in [(3, 0.0), (4, 0.05)]:
            probs_adjusted = {idx: 0 if idx != target_idx else 1.0
                              for idx, p
                              in self.age_params['age_probs'].items()}
            age_range_key, age = sample_age(age_ranges=self.age_params['age_ranges'],
                                            age_probs=probs_adjusted)
            immunity_by_reduction_factor = self.age_params['immunity_reduction_factors'][age_range_key]

            assert age_range_key == target_idx
            assert self.age_ranges[target_idx][0] <= age <= self.age_ranges[target_idx][1]
            assert immunity_by_reduction_factor == factor
