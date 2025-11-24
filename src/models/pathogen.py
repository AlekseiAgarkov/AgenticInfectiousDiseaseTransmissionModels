from abc import ABC, abstractmethod
from typing import Any, Union

import numpy as np
import simpy


class Pathogen(ABC):

    def __init__(self, env: simpy.Environment, name: str):
        self.env = env
        self.name = name

    def __call__(self):
        return self._get_beta(sim_time=self.env.now)

    @abstractmethod
    def _get_beta(self, sim_time: int) -> Union[float, np.float64]:
        pass


class LinearPathogen(Pathogen):
    pathogen_values: np.ndarray[tuple[Any, ...], np.dtype[np.float64]]

    def __init__(self,
                 env: simpy.Environment,
                 name: str,
                 sim_duration: int,
                 base_beta: float,
                 max_beta: float):
        super().__init__(env, name)
        self.sim_duration = sim_duration
        self.base_beta = base_beta
        self.max_beta = max_beta

        self.pathogen_values = np.linspace(start=base_beta, num=sim_duration + 1, stop=max_beta)

    def _get_beta(self, sim_time: int) -> Union[float, np.float64]:
        return self.pathogen_values[sim_time]


class DiscretePredefinedPathogen(Pathogen):
    pathogen_values: np.ndarray[tuple[Any, ...], np.dtype[np.float64]]

    def __init__(self,
                 env: simpy.Environment,
                 name: str,
                 pathogen_values: np.ndarray[tuple[Any, ...], np.dtype[np.float64]]):
        super().__init__(env, name)

        self.pathogen_values = pathogen_values

    def _get_beta(self, sim_time: int) -> Union[float, np.float64]:
        return self.pathogen_values.item(sim_time)
