from abc import ABC, abstractmethod
from transitions import State
from typing import List

import simpy


class BaseAgent(ABC):
    states: List[State]

    def __init__(self, env: simpy.Environment, name: str):
        self.env = env
        self.name = name

    @abstractmethod
    def run(self):
        pass
