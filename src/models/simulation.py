import random
from logging import Logger
from typing import Type, Dict

import pandas as pd
import simpy

from metrics.collector import StateCountMetricsCollector, MetricsCollector, TransitionMetricsCollector
from models.agents import BaseAgent
from models.seir import SEIRNeighborsFSMAgent


def configure_simulation(environment: simpy.Environment,
                         agent_cls: Type[BaseAgent],
                         agent_params: Dict,
                         n_agents: int) -> StateCountMetricsCollector:
    agents: Dict[int, BaseAgent] = {n: agent_cls(env=environment,
                                                 name=f"Agent_{n}",
                                                 **agent_params)
                                    for n in range(n_agents)}

    metrics: StateCountMetricsCollector = StateCountMetricsCollector(
        env=environment,
        entities=list(agents.values()),
        states=agent_cls.states)

    for a in agents.values():
        environment.process(a.run())

    environment.process(metrics.run())

    return metrics


def configure_neighbors_simulation(log: Logger,
                                   environment: simpy.Environment,
                                   agent_params: Dict,
                                   neighbors_data: pd.DataFrame,
                                   initially_infected: int) -> TransitionMetricsCollector:

    log.info("Initializing Metrics Collector")
    metrics_collector: TransitionMetricsCollector = TransitionMetricsCollector()

    log.info("Initializing Agents")
    agents: Dict[int, SEIRNeighborsFSMAgent] = {}
    for n in range(neighbors_data.shape[0]):
        agents[n] = SEIRNeighborsFSMAgent(env=environment,
                                          metrics_collector=metrics_collector,
                                          name=n,
                                          **agent_params,
                                          x=float(neighbors_data['x'].iloc[n]),
                                          y=float(neighbors_data['y'].iloc[n]))

    log.info("Linking Neighbors")
    for i, a in agents.items():
        neighbors = [agents[n] for n in neighbors_data['neighbors'].iloc[i]]
        a.set_neighbors(neighbors)

    log.info("Initializing Infected Agents")
    for agent in random.sample(list(agents.keys()), initially_infected):
        agents[agent].to_infected()

    log.info("Submitting Agents to Environment")
    for a in agents.values():
        environment.process(a.run())

    return metrics_collector
