from typing import List

import numpy as np
import simpy
from transitions import State, Machine

from models.agents import BaseAgent


class SEIRBasicFSMAgent(BaseAgent):
    states: List[State] = [
        State(name='susceptible'),
        State(name='exposed'),
        State(name='infected'),
        State(name='recovered'),
    ]

    def __init__(self,
                 env: simpy.Environment,
                 name: str,
                 beta: float,
                 gamma: float,
                 sigma: float,
                 sim_duration: int):
        super().__init__(env=env, name=name)
        self.beta: float = beta
        self.sigma: float = sigma
        self.gamma: float = gamma
        self.sim_duration: int = sim_duration

        self.machine: Machine = Machine(model=self, states=self.states, initial=self.states[0])
        self.machine.add_transition(trigger="try_get_exposed",
                                    source="susceptible",
                                    dest="exposed",
                                    conditions="got_exposed")
        self.machine.add_transition(trigger="try_get_infected",
                                    source="exposed",
                                    dest="infected",
                                    conditions="got_infected")
        self.machine.add_transition(trigger="try_recover",
                                    source="infected",
                                    dest="recovered",
                                    conditions="got_recovered")

    def got_exposed(self) -> bool:
        return bool(np.random.binomial(n=1, p=self.beta))

    def got_infected(self) -> bool:
        return bool(np.random.binomial(n=1, p=self.sigma))

    def got_recovered(self) -> bool:
        return bool(np.random.binomial(n=1, p=self.gamma))

    def run(self):
        while True:
            if self.is_susceptible():
                yield self.env.timeout(1)
                self.try_get_exposed()

            if self.is_exposed():
                yield self.env.timeout(1)
                self.try_get_infected()

            if self.is_infected():
                yield self.env.timeout(1)
                self.try_recover()

            if self.is_recovered():
                till_end_of_simulation: int = (self.sim_duration - self.env.now) + 1
                yield self.env.timeout(till_end_of_simulation)
