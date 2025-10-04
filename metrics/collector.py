from enum import StrEnum
from typing import List

import pandas as pd
import simpy
from transitions import State

from models.agents import BaseAgent


class Metric(StrEnum):
    STATE_COUNTS = "state_counts"
    TRANSITIONS = "transitions"


class MetricsCollector:
    def __init__(self,
                 env: simpy.Environment,
                 entities: List[BaseAgent],
                 states: List[State]
                 ):
        self.env = env
        self.entities = entities
        self.states: List[State] = states
        for metric in Metric:
            self._init_container(kind=metric, value=list())

    def run(self):
        while True:
            metrics = self._collect_metrics()
            self._get_container(kind=Metric.STATE_COUNTS).append(metrics)
            yield self.env.timeout(1)

    def _collect_metrics(self) -> dict:
        return {
            "time": self.env.now,
            **{
                state.name: sum([e.__getattribute__(f"is_{state.name}")() for e in self.entities])
                for state in self.states
            }
        }

    def append_transition_record(self, agent_name: str, time: int, from_state: str, to_state: str):
        self._get_container(kind=Metric.TRANSITIONS).append({"time": time,
                                                             "agent_name": agent_name,
                                                             "from_state": from_state,
                                                             "to_state": to_state})

    def _get_container(self, kind: Metric):
        return self.__getattribute__(f"_{kind.value}_container")

    def _init_container(self, kind: str, value):
        return self.__setattr__(f"_{kind}_container", value)

    def _get_metrics(self, kind: Metric) -> pd.DataFrame:
        return pd.DataFrame(self.__getattribute__(f"_{kind.value}_container"))

    def metrics_to_csv(self, kind: Metric, filename: str) -> None:
        df: pd.DataFrame = self._get_metrics(kind=kind)
        df.to_csv(filename, index=False)
