import random
from logging import Logger
from typing import Type, Dict, List, Optional

import numpy as np
import pandas as pd
import simpy
from tqdm import tqdm

from metrics.collector import StateCountMetricsCollector, TransitionMetricsCollector
from models.age import sample_age
from models.agents import BaseAgent
from models.immunity import adjust_immunity_by_mid_proportional
from models.seir import SEIRNeighborsFSMAgent, SEIRNeighborsFSMExtended


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

    for a in tqdm(agents.values()):
        environment.process(a.run())

    environment.process(metrics.run())

    return metrics


def configure_neighbors_simulation(log: Logger,
                                   environment: simpy.Environment,
                                   agent_params: Dict,
                                   neighbors_data: pd.DataFrame,
                                   initially_infected: int,
                                   initially_infected_indices: Optional[List] = None) -> TransitionMetricsCollector:
    log.info("Initializing Metrics Collector")
    metrics_collector: TransitionMetricsCollector = TransitionMetricsCollector()

    log.info("Initializing Agents")
    agents: Dict[int, SEIRNeighborsFSMAgent] = {}
    for n in tqdm(range(neighbors_data.shape[0])):
        agents[n] = SEIRNeighborsFSMAgent(env=environment,
                                          metrics_collector=metrics_collector,
                                          name=n,
                                          **agent_params,
                                          x=float(neighbors_data['x'].iloc[n]),
                                          y=float(neighbors_data['y'].iloc[n]))

    log.info("Linking Neighbors")
    for i, a in tqdm(agents.items()):
        neighbors = [agents[n] for n in neighbors_data['neighbors'].iloc[i]]
        a.set_neighbors(neighbors)

    log.info("Initializing Infected Agents")
    agents_to_infect = initially_infected_indices or list(random.sample(list(agents.keys()), initially_infected))

    for agent in tqdm(agents_to_infect):
        agents[agent].to_infected()

    log.info("Submitting Agents to Environment")
    for a in tqdm(agents.values()):
        environment.process(a.run())

    return metrics_collector


def configure_extended_neighbors_simulation(
        log: Logger, environment: simpy.Environment,
        agent_params: Dict,
        neighbors_data: pd.DataFrame,
        initially_infected: int,
        age_params: Dict,
        immunity_params: Dict,
        initially_infected_indices: Optional[List] = None) -> TransitionMetricsCollector:
    log.info("Initializing Metrics Collector")
    metrics_collector: TransitionMetricsCollector = TransitionMetricsCollector()

    log.info("Initializing Agents")

    agents: Dict[int, SEIRNeighborsFSMExtended] = {}
    for n in tqdm(range(neighbors_data.shape[0])):
        age_range_key, age = sample_age(age_ranges=age_params['age_ranges'],
                                        age_probs=age_params['age_probs'])
        immunity_by_reduction_factor = age_params['immunity_reduction_factors'][age_range_key]

        (agent_immunity_lower_bound,
         agent_immunity_upper_bound) = adjust_immunity_by_mid_proportional(
            lowest_immunity=immunity_params['lowest_immunity'],
            highest_immunity=immunity_params['highest_immunity'],
            reduction_factor=immunity_by_reduction_factor)

        wears_mask_at_contact_p = np.random.uniform(low=immunity_params['mask_discipline_worst'],
                                                    high=immunity_params['mask_discipline_best'])

        agents[n] = SEIRNeighborsFSMExtended(env=environment,
                                             metrics_collector=metrics_collector,
                                             name=n,
                                             **agent_params,
                                             age=age,
                                             immunity_lower_bound=agent_immunity_lower_bound,
                                             immunity_upper_bound=agent_immunity_upper_bound,
                                             mask_beta_penalty=immunity_params['mask_beta_penalty'],
                                             wears_mask_at_contact_p=wears_mask_at_contact_p,
                                             x=float(neighbors_data['x'].iloc[n]),
                                             y=float(neighbors_data['y'].iloc[n]))

    log.info("Linking Neighbors")
    for i, a in tqdm(agents.items()):
        neighbors = [agents[n] for n in neighbors_data['neighbors'].iloc[i]]
        a.set_neighbors(neighbors)

    log.info("Initializing Infected Agents")
    agents_to_infect = initially_infected_indices or list(random.sample(list(agents.keys()), initially_infected))

    for agent in tqdm(agents_to_infect):
        agents[agent].to_infected()

    log.info("Submitting Agents to Environment")
    for a in tqdm(agents.values()):
        environment.process(a.run())

    return metrics_collector
