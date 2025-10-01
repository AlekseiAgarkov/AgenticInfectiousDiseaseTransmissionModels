from typing import List

import pandas as pd
import simpy
from transitions import State

from models.agents import BaseAgent


class MetricsCollector:
    def __init__(self,
                 env: simpy.Environment,
                 entities: List[BaseAgent],
                 states: List[State]
                 ):
        self.env = env
        self.entities = entities
        self.states: List[State] = states
        self._metrics_container: List = []

    def run(self):
        while True:
            metrics = self._collect_metrics()
            self._metrics_container.append(metrics)
            yield self.env.timeout(1)

    def _collect_metrics(self) -> dict:
        return {
            "time": self.env.now,
            **{
                state.name: sum([e.__getattribute__(f"is_{state.name}")() for e in self.entities])
                for state in self.states
            }
        }

    def get_metrics(self) -> pd.DataFrame:
        return pd.DataFrame(self._metrics_container)

    def to_csv(self, filename: str) -> None:
        df: pd.DataFrame = self.get_metrics()
        df.to_csv(filename, index=False)
