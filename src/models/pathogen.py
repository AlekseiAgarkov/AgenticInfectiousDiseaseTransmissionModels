from abc import ABC, abstractmethod

import simpy


class Pathogen(ABC):

    def __init__(self, env: simpy.Environment, name: str):
        self.env = env
        self.name = name

    def __call__(self):
        return self._get_beta(sim_time=self.env.now)

    @abstractmethod
    def _get_beta(self, sim_time: int) -> float:
        pass
