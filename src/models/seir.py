import random
from typing import List, Union, Optional

import numpy as np
import simpy
from transitions import State, EventData
from transitions.extensions import GraphMachine

from metrics.collector import TransitionMetricsCollector
from models.agents import BaseAgent
from models.immunity import immunity_by_year_day


class SEIRFSMBase(BaseAgent):
    states: List[State] = [
        State(name='susceptible'),
        State(name='exposed'),
        State(name='infected'),
        State(name='recovered'),
    ]

    def __init__(self,
                 env: simpy.Environment,
                 name: Union[str, int],
                 sim_duration: int):
        super().__init__(env=env, name=name)
        self.sim_duration: int = sim_duration

        self.machine: GraphMachine = GraphMachine(model=self,
                                                  states=self.states,
                                                  initial=self.states[0],
                                                  graph_engine="graphviz",
                                                  finalize_event='finalize',
                                                  send_event=True,
                                                  show_conditions=True,
                                                  show_state_attributes=True,
                                                  title=f"{self.__class__.__name__} State Machine")

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

    def finalize(self, event: EventData):
        pass

    def got_exposed(self, event: EventData) -> bool:
        return True

    def got_infected(self, event: EventData) -> bool:
        return True

    def got_recovered(self, event: EventData) -> bool:
        return True

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


class SEIRClassicFSMAgent(SEIRFSMBase):
    def __init__(self,
                 env: simpy.Environment,
                 name: Union[str, int],
                 beta: float,
                 gamma: float,
                 sigma: float,
                 sim_duration: int):
        super().__init__(env=env, name=name, sim_duration=sim_duration)
        self.beta: float = beta
        self.sigma: float = sigma
        self.gamma: float = gamma

    def got_exposed(self, event: EventData) -> bool:
        return bool(np.random.binomial(n=1, p=self.beta))

    def got_infected(self, event: EventData) -> bool:
        return bool(np.random.binomial(n=1, p=self.sigma))

    def got_recovered(self, event: EventData) -> bool:
        return bool(np.random.binomial(n=1, p=self.gamma))


class SEIRNeighborsFSMAgent(SEIRFSMBase):
    def __init__(self,
                 env: simpy.Environment,
                 metrics_collector: TransitionMetricsCollector,
                 name: Union[str, int],
                 beta: float,
                 sim_duration: int,
                 x: float,
                 y: float,
                 e1: int,
                 e2: int,
                 t1: int,
                 t2: int):
        super().__init__(env=env, name=name, sim_duration=sim_duration)
        self.neighbors = Optional[List['SEIRNeighborsFSMAgent']]
        self.beta = beta
        self.y = y
        self.x = x
        self.e1 = e1
        self.e2 = e2
        self.t2 = t2
        self.t1 = t1
        self.metrics_collector: TransitionMetricsCollector = metrics_collector

    def finalize(self, event: EventData):
        if event.result:
            transition_record = {"time": self.env.now,
                                 "agent": self.name,
                                 "event": "transition",
                                 "source_state": event.transition.source,
                                 "dest_state": event.transition.dest}
            self.metrics_collector.append_transition_record(transition_record)

    def got_exposed(self, event: EventData) -> bool:
        p = min([sum(n.beta * n.is_infected() for n in self.neighbors), 1.])
        return bool(np.random.binomial(n=1, p=p))

    def set_neighbors(self, neighbors: List['SEIRNeighborsFSMAgent']):
        self.neighbors: Optional[List['SEIRNeighborsFSMAgent']] = neighbors

    def run(self):
        while True:
            if self.is_susceptible():
                yield self.env.timeout(1)
                self.try_get_exposed()

            if self.is_exposed():
                timeout = random.randint(self.e1, self.e2)
                yield self.env.timeout(timeout)
                self.try_get_infected()

            if self.is_infected():
                timeout = random.randint(self.t1, self.t2)
                yield self.env.timeout(timeout)
                self.try_recover()

            if self.is_recovered():
                till_end_of_simulation: int = (self.sim_duration - self.env.now) + 1
                yield self.env.timeout(till_end_of_simulation)


class SEIRNeighborsFSMExtended(SEIRNeighborsFSMAgent):
    def __init__(self,
                 env: simpy.Environment,
                 metrics_collector: TransitionMetricsCollector,
                 name: Union[str, int],
                 beta: float,
                 sim_duration: int,
                 x: float,
                 y: float,
                 e1: int,
                 e2: int,
                 t1: int,
                 t2: int,
                 age: int,
                 immunity_lower_bound: float = 0.0,
                 immunity_upper_bound: float = 1.0):
        super().__init__(env=env,
                         metrics_collector=metrics_collector,
                         name=name,
                         beta=beta,
                         sim_duration=sim_duration,
                         x=x,
                         y=y,
                         e1=e1,
                         e2=e2,
                         t1=t1,
                         t2=t2)

        self.age = age

        assert 0.0 <= immunity_lower_bound <= 1.0
        assert 0.0 <= immunity_upper_bound <= 1.0
        assert immunity_lower_bound <= immunity_upper_bound

        self.immunity_lower_bound = immunity_lower_bound
        self.immunity_upper_bound = immunity_upper_bound

    def probability_to_infect(self):
        return self.beta * self.is_infected()

    def current_immunity(self, decimals: int = 2) -> float:
        return round(immunity_by_year_day(day=self.env.now,
                                          low=self.immunity_lower_bound,
                                          high=self.immunity_upper_bound), decimals)

    def got_exposed(self, event: EventData) -> bool:
        neighbor_infect_probability = sum(n.probability_to_infect() for n in self.neighbors)
        adjusted_for_immunity = neighbor_infect_probability * (1 - self.current_immunity())
        p = min([adjusted_for_immunity, 1.])
        return bool(np.random.binomial(n=1, p=p))
