from copy import deepcopy
from typing import List

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
