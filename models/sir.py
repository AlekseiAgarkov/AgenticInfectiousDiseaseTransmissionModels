from typing import List

import numpy as np
import simpy
from transitions import State, Machine

from models.agents import BaseAgent


class SIRBasicFSMAgent(BaseAgent):
    states: List[State] = [
        State(name='susceptible'),
        State(name='infected'),
        State(name='resistant'),
    ]

    def __init__(self, env: simpy.Environment, name: str, beta: float, gamma: float, sim_duration: int):
        super().__init__(env=env, name=name)
        self.beta: float = beta
        self.gamma: float = gamma
        self.sim_duration: int = sim_duration

        self.machine: Machine = Machine(model=self, states=self.states, initial=self.states[0])
        self.machine.add_transition(trigger="try_get_infected", source="susceptible", dest="infected", conditions="got_infected")
        self.machine.add_transition(trigger="recover", source="infected", dest="resistant", conditions="got_resistant")

    def got_infected(self):
        return bool(np.random.binomial(n=1, p=self.beta))

    def got_resistant(self):
        return bool(np.random.binomial(n=1, p=self.gamma))

    def run(self):
        while True:
            if self.is_susceptible():
                self.try_get_infected()
                yield self.env.timeout(1)

            if self.is_infected():
                self.recover()
                yield self.env.timeout(1)

            if self.is_resistant():
                till_end_of_simulation: int = (self.sim_duration - self.env.now) + 1
                yield self.env.timeout(till_end_of_simulation)
