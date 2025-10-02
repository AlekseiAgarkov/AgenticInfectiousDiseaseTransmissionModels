from typing import Type, Dict

import simpy

from metrics.collector import MetricsCollector
from models.agents import BaseAgent


def configure_simulation(environment: simpy.Environment,
                         agent_cls: Type[BaseAgent],
                         agent_params: Dict,
                         n_agents: int) -> MetricsCollector:
    agents: Dict[int, BaseAgent] = {n: agent_cls(env=environment,
                                                 name=f"Agent_{n}",
                                                 **agent_params)
                                    for n in range(n_agents)}

    metrics: MetricsCollector = MetricsCollector(
        env=environment,
        entities=list(agents.values()),
        states=agent_cls.states)

    for a in agents.values():
        environment.process(a.run())

    environment.process(metrics.run())

    return metrics
