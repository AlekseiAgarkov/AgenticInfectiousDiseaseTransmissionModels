from copy import deepcopy
from typing import List

import numpy as np
import pandas as pd


def convert_transitions_to_state_counts(transitions_df: pd.DataFrame,
                                        states: List[str],
                                        initial_state: str,
                                        n_agents: int,
                                        sim_duration: int) -> pd.DataFrame:
    transitions_data = transitions_df.sort_values("time").reset_index(drop=True).to_dict("records")
    previous_time = 0
    data = {0: {state: 0 for state in states}}
    data[0][initial_state] = n_agents
    for row in transitions_data:
        source_state = row['source_state']
        dest_state = row['dest_state']
        current_time = row['time']
        if current_time not in data.keys():
            previous_state = data[previous_time]
            data[current_time] = deepcopy(previous_state)
            previous_time = current_time

        data[current_time][source_state] -= 1
        data[current_time][dest_state] += 1

    sim_timeline = pd.DataFrame(range(sim_duration), columns=['time'])
    result_df = pd.merge(sim_timeline,
                         pd.DataFrame([{"time": k, **v} for k, v in data.items()]),
                         how="left",
                         on="time").ffill()
    for column in result_df.columns:
        result_df[column] = result_df[column].astype(int)
    return result_df


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def convert_transitions_to_agent_states(transitions_df: pd.DataFrame,
                                        states: List[str],
                                        initial_state: str,
                                        n_agents: int,
                                        sim_duration: int) -> pd.DataFrame:
    states_numeric = {state: i for i, state in enumerate(states)}

    time = pd.DataFrame(range(sim_duration), columns=['time'])

    agents_at_time_0 = transitions_df[transitions_df['time'] == 0].agent.tolist()
    missing_agents_at_time_0 = [a for a in range(n_agents) if a not in agents_at_time_0]
    zeroes = np.zeros(len(missing_agents_at_time_0), dtype=int)
    missing_agents_at_time_0 = pd.DataFrame(
        {'time': zeroes,
         'agent': missing_agents_at_time_0,
         'dest_state': initial_state})

    transitions_data = (
        pd.concat([missing_agents_at_time_0, transitions_df[['time', 'agent', 'dest_state']]])
        .assign(dest_state = lambda _df: pd.Categorical(_df.dest_state, states, ordered=True))
        .sort_values(['time', 'agent', 'dest_state'])
        .pivot_table(index='time', columns='agent', values='dest_state', aggfunc='last')
        .reset_index()
        .rename_axis(mapper=None, axis=1)
        .merge(time, on='time', how='right')
        .ffill()
        .melt(id_vars='time', var_name='agent', value_name='state')
        .astype({'agent': 'uint64'})
        .sort_values(by=['time', 'agent'])
        .reset_index(drop=True)
    )

    return transitions_data
