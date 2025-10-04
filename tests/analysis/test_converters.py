from unittest import TestCase

import pandas as pd

from src.analysis.converters import convert_transitions_to_state_counts

transitions_data = [
    {'time': 1, 'source_state': 'susceptible', 'dest_state': 'infected'},
    {'time': 3, 'source_state': 'susceptible', 'dest_state': 'infected'},
    {'time': 3, 'source_state': 'susceptible', 'dest_state': 'infected'},
    {'time': 5, 'source_state': 'infected', 'dest_state': 'recovered'},
    {'time': 6, 'source_state': 'infected', 'dest_state': 'recovered'},
    {'time': 6, 'source_state': 'infected', 'dest_state': 'recovered'},
]
transitions_df = pd.DataFrame(transitions_data)
test_data = [
    {'time': 0, 'susceptible': 3, 'infected': 0, 'recovered': 0},
    {'time': 1, 'susceptible': 2, 'infected': 1, 'recovered': 0},
    {'time': 2, 'susceptible': 2, 'infected': 1, 'recovered': 0},
    {'time': 3, 'susceptible': 0, 'infected': 3, 'recovered': 0},
    {'time': 4, 'susceptible': 0, 'infected': 3, 'recovered': 0},
    {'time': 5, 'susceptible': 0, 'infected': 2, 'recovered': 1},
    {'time': 6, 'susceptible': 0, 'infected': 0, 'recovered': 3},
    {'time': 7, 'susceptible': 0, 'infected': 0, 'recovered': 3}
]
test_df = pd.DataFrame(test_data)


class Test(TestCase):
    def test_convert_transitions_to_state_counts(self):
        result_df = convert_transitions_to_state_counts(transitions_df=transitions_df,
                                                        states=['susceptible', 'infected', 'recovered'],
                                                        initial_state='susceptible',
                                                        n_agents=3,
                                                        sim_duration=8)
        self.assertTrue(test_df.equals(result_df))
