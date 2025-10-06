from unittest import TestCase

import pandas as pd

from src.analysis.converters import convert_transitions_to_state_counts, convert_transitions_to_agent_states

transitions_data = [
    {'time': 0, 'agent': 0, 'source_state': 'susceptible', 'dest_state': 'infected'},
    {'time': 2, 'agent': 1, 'source_state': 'susceptible', 'dest_state': 'infected'},
    {'time': 3, 'agent': 2, 'source_state': 'susceptible', 'dest_state': 'infected'},
    {'time': 5, 'agent': 0, 'source_state': 'infected', 'dest_state': 'recovered'},
    {'time': 6, 'agent': 1, 'source_state': 'infected', 'dest_state': 'recovered'},
    {'time': 6, 'agent': 2, 'source_state': 'infected', 'dest_state': 'recovered'},
]
transitions_df = pd.DataFrame(transitions_data)

state_counts_test_data = [
    {'time': 0, 'susceptible': 2, 'infected': 1, 'recovered': 0},
    {'time': 1, 'susceptible': 2, 'infected': 1, 'recovered': 0},
    {'time': 2, 'susceptible': 1, 'infected': 2, 'recovered': 0},
    {'time': 3, 'susceptible': 0, 'infected': 3, 'recovered': 0},
    {'time': 4, 'susceptible': 0, 'infected': 3, 'recovered': 0},
    {'time': 5, 'susceptible': 0, 'infected': 2, 'recovered': 1},
    {'time': 6, 'susceptible': 0, 'infected': 0, 'recovered': 3},
    {'time': 7, 'susceptible': 0, 'infected': 0, 'recovered': 3}
]
state_counts_test_df = pd.DataFrame(state_counts_test_data)

agent_states_test_data = [{'time': 0, 'agent': 0, 'state': 1},
                          {'time': 0, 'agent': 1, 'state': 0},
                          {'time': 0, 'agent': 2, 'state': 0},
                          {'time': 1, 'agent': 0, 'state': 1},
                          {'time': 1, 'agent': 1, 'state': 0},
                          {'time': 1, 'agent': 2, 'state': 0},
                          {'time': 2, 'agent': 0, 'state': 1},
                          {'time': 2, 'agent': 1, 'state': 1},
                          {'time': 2, 'agent': 2, 'state': 0},
                          {'time': 3, 'agent': 0, 'state': 1},
                          {'time': 3, 'agent': 1, 'state': 1},
                          {'time': 3, 'agent': 2, 'state': 1},
                          {'time': 4, 'agent': 0, 'state': 1},
                          {'time': 4, 'agent': 1, 'state': 1},
                          {'time': 4, 'agent': 2, 'state': 1},
                          {'time': 5, 'agent': 0, 'state': 2},
                          {'time': 5, 'agent': 1, 'state': 1},
                          {'time': 5, 'agent': 2, 'state': 1},
                          {'time': 6, 'agent': 0, 'state': 2},
                          {'time': 6, 'agent': 1, 'state': 2},
                          {'time': 6, 'agent': 2, 'state': 2},
                          {'time': 7, 'agent': 0, 'state': 2},
                          {'time': 7, 'agent': 1, 'state': 2},
                          {'time': 7, 'agent': 2, 'state': 2}]

agent_states_test_df = pd.DataFrame(agent_states_test_data)
agent_states_test_df['state'] = agent_states_test_df['state'].astype('uint8')
agent_states_test_df['agent'] = agent_states_test_df['agent'].astype('uint64')


class Test(TestCase):
    def test_convert_transitions_to_state_counts(self):
        result_df = convert_transitions_to_state_counts(transitions_df=transitions_df,
                                                        states=['susceptible', 'infected', 'recovered'],
                                                        initial_state='susceptible',
                                                        n_agents=3,
                                                        sim_duration=8)
        self.assertTrue(state_counts_test_df.equals(result_df))

    def test_convert_transitions_to_agent_states(self):
        result_df = convert_transitions_to_agent_states(transitions_df=transitions_df,
                                                        states=['susceptible', 'infected', 'recovered'],
                                                        initial_state='susceptible',
                                                        n_agents=3,
                                                        sim_duration=8)

        self.assertTrue(agent_states_test_df.equals(result_df))
