import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Generic, TypeVar, get_args

import pandas as pd
from tqdm import tqdm

from analysis.converters import convert_transitions_to_state_counts, convert_transitions_to_agent_states
from models.seir import SEIRNeighborsFSMExtended


@dataclass
class SimulationDataPaths:
    id: str
    path: str
    transitions: str
    params: str


def list_simulations(path: str, id_len: int) -> List[SimulationDataPaths]:
    simulations = {}
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_file():
                filename = entry.name
                filename, extension = os.path.splitext(filename)
                sim_id = filename[-id_len:]
                simulations[sim_id] = simulations.get(sim_id, {})
                if "Transitions" in filename:
                    simulations[sim_id]['Transitions'] = entry.name
                if "Params" in filename:
                    simulations[sim_id]['Params'] = entry.name

    return [SimulationDataPaths(id=k, path=path, transitions=v['Transitions'], params=v['Params'])
            for k, v in
            simulations.items()]


SEIRAgentT = TypeVar('SEIRAgentT')


class SimulationProcessor(Generic[SEIRAgentT], ABC):
    def __init__(self, simulation_data_paths: SimulationDataPaths):
        self.simulation_data_paths = simulation_data_paths
        self.simulation_parameters: Dict[str, Any] = self._process_params()
        self.simulation_transitions = pd.read_csv(
            os.path.join(self.simulation_data_paths.path, self.simulation_data_paths.transitions))
        self.agent_states = self._get_states()
        self.state_counts = self._transitions_to_state_counts()
        self.observed_agent_states = self._transitions_to_agent_states()
        self.metrics = self._extract_metrics()
        self.sim_data = self._join_metrics_and_params()

    def _process_params(self) -> Dict[str, Any]:
        with open(os.path.join(os.path.join(self.simulation_data_paths.path, self.simulation_data_paths.params)),
                  "r") as file:
            p = json.load(file)

        return self._extract_params(p)

    @abstractmethod
    def _get_states(self) -> List[str]:
        pass

    @abstractmethod
    def _extract_params(self, p: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _extract_metrics(self) -> Dict[str, Any]:
        pass

    def _transitions_to_state_counts(self) -> pd.DataFrame:
        return convert_transitions_to_state_counts(transitions_df=self.simulation_transitions,
                                                   states=self.agent_states,
                                                   initial_state=self.agent_states[0],
                                                   n_agents=self.simulation_parameters['n_agents'],
                                                   sim_duration=self.simulation_parameters['sim_duration'])

    def _transitions_to_agent_states(self):
        return convert_transitions_to_agent_states(transitions_df=self.simulation_transitions,
                                                   states=self.agent_states,
                                                   initial_state=self.agent_states[0],
                                                   n_agents=self.simulation_parameters['n_agents'],
                                                   sim_duration=self.simulation_parameters['sim_duration'])

    def _join_metrics_and_params(self) -> Dict[str, Any]:
        return {**self.simulation_parameters, **self.metrics}


class SEIRNeighborsFSMExtendedProcessor(SimulationProcessor[SEIRNeighborsFSMExtended]):
    def _get_states(self) -> List[str]:
        return [state.name for state in SEIRNeighborsFSMExtended.states]

    def _extract_params(self, p) -> Dict[str, Any]:
        return {
            "simulation_id": self.simulation_data_paths.id,
            "random_seed": p["random_seed"],
            "n_agents": p["n_agents"],
            "sim_duration": p["sim_duration"],
            "initially_infected": p["initially_infected"],
            "lowest_immunity": p["immunity_params"]["lowest_immunity"],
            "highest_immunity": p["immunity_params"]["highest_immunity"],
            "mask_beta_penalty": p["immunity_params"]["mask_beta_penalty"],
            "mask_discipline_worst": p["immunity_params"]["mask_discipline_worst"],
            "mask_discipline_best": p["immunity_params"]["mask_discipline_best"],
            "beta": p["beta"],
            "e1": p["e1"],
            "e2": p["e2"],
            "t1": p["t1"],
            "t2": p["t2"],
            "pollutant_immunity_reduction": p["pollutants_config"]["pollutant_immunity_reduction"],
            "agent_class": p["agent_class"],
            "scenario_name": p["scenario_name"]
        }

    def _extract_metrics(self):
        last_sim_day = self.observed_agent_states['time'].max()
        state_at_last_sim_date = self.observed_agent_states[self.observed_agent_states['time'] == last_sim_day]
        increments_30d = list(range(30, self.simulation_parameters['sim_duration'], 30))
        data = {}
        for inc in increments_30d:
            inc_till_date = self.observed_agent_states[self.observed_agent_states['time'] <= inc]
            infected = inc_till_date[inc_till_date['state'] == "infected"]["agent"].nunique()
            exposed = inc_till_date[inc_till_date['state'] == "exposed"]["agent"].nunique()
            recovered = inc_till_date[inc_till_date['state'] == "recovered"]["agent"].nunique()
            data[f"infected_{inc}d"] = infected
            data[f"exposed_{inc}d"] = exposed
            data[f"recovered_{inc}d"] = recovered

        return {**data,
                "max_infected": self.state_counts['infected'].max(),
                "max_exposed": self.state_counts['exposed'].max(),
                "max_recovered": self.state_counts['recovered'].max(),
                "total_infected": self.observed_agent_states[self.observed_agent_states['state'] == 'infected'][
                    'agent'].nunique(),
                "total_exposed": self.observed_agent_states[self.observed_agent_states['state'] == 'exposed'][
                    'agent'].nunique(),
                "total_recovered": (state_at_last_sim_date['state'] == 'recovered').sum(),
                }
