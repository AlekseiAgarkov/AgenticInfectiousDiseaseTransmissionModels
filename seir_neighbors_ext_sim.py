import ast
import json
import logging
import random
import tomllib
from argparse import ArgumentParser
from logging import Logger
from pathlib import Path
from typing import List

import pandas as pd
import simpy

from metrics.collector import Metric
from models.seir import SEIRNeighborsFSMAgent
from models.simulation import configure_extended_neighbors_simulation
from simulation_utils.logging import configure_logging
from simulation_utils.time import msk_now, DATETIME_FORMAT

if __name__ == '__main__':
    p = ArgumentParser()
    paths = p.add_argument_group('paths')
    paths.add_argument('-o', '--output_path', type=str, nargs="?", help="Simulation data output folder")
    paths.add_argument('-c', '--config_path', type=str, nargs="?", default=None,
                       help="Config path")
    paths.add_argument('--neighbors_data_path', type=str, nargs="?", default=None,
                       help="Path to Neighbors Data")

    simulation_params = p.add_argument_group("simulation_params")
    simulation_params.add_argument('-r', '--random_seed', type=int, nargs="?", help="Random seed")
    simulation_params.add_argument('-n', '--n_agents', type=int, nargs="?", help="Number of agents")
    simulation_params.add_argument('-t', '--sim_duration', type=int, nargs="?", help="Duration of Simulation, units")
    simulation_params.add_argument('-i', '--initially_infected', type=int, nargs="?",
                                   help="Number of initially infected")

    agent_params = p.add_argument_group("agent_params")
    agent_params.add_argument('-b', '--beta', type=float, nargs="?", help="Model parameter Beta")
    agent_params.add_argument('--e1', type=float, nargs="?", help="Min Exposed State Duration")
    agent_params.add_argument('--e2', type=float, nargs="?", help="Max Exposed State Duration")
    agent_params.add_argument('--t1', type=float, nargs="?", help="Min Infected State Duration")
    agent_params.add_argument('--t2', type=float, nargs="?", help="Max Infected State Duration")

    args = p.parse_args()
    config = None
    if args.config_path is not None:
        config = tomllib.loads(Path(args.config_path).read_text(encoding="utf-8"))

    simulation_start = msk_now()
    simulation_start_str = simulation_start.strftime(DATETIME_FORMAT)

    SIMULATOR_NAME = config['simulator']['name']

    AGENT_PARAMS: dict = {
        "beta": args.beta or config['agents']['beta'],
        "sim_duration": args.sim_duration or config['simulation']['sim_duration'],
        "e1": args.e1 or config['agents']['e1'],
        "e2": args.e2 or config['agents']['e2'],
        "t1": args.t1 or config['agents']['t1'],
        "t2": args.t2 or config['agents']['t2']
    }

    RANDOM_SEED = args.random_seed or config['simulation']['random_seed']

    NEIGHBORS_DATA: pd.DataFrame = pd.read_csv(config['paths']['neighbors_data_path'],
                                               converters={'neighbors': ast.literal_eval})

    N_AGENTS: int = args.n_agents or config['simulation']['n_agents']
    INITIALLY_INFECTED: int = args.initially_infected or config['simulation']['initially_infected']
    INITIALLY_INFECTED_INDICES: List[int] = config['simulation'].get('initially_infected_indices')
    LOWEST_IMMUNITY: float = config['simulation'].get('lowest_immunity')
    HIGHEST_IMMUNITY: float = config['simulation'].get('highest_immunity')

    assert N_AGENTS == NEIGHBORS_DATA.shape[0]

    AGE_PARAMS: dict = {
        "age_ranges": config['age'].get('age_ranges'),
        "age_probs": config['age'].get('age_probs'),
        "immunity_reduction_factors": config['age'].get('immunity_reduction_factors'),
    }

    SIM_DURATION: int = args.sim_duration or config['simulation']['sim_duration']
    OUTPUT_PATH: str = args.output_path or config['paths']['output_path']
    LOG_OUTPUT: str = f'{config['paths']['log_output']}/{SIMULATOR_NAME}_Log-{simulation_start_str}.log'

    simulation_params = {
        "random_seed": RANDOM_SEED,
        "n_agents": N_AGENTS,
        "sim_duration": SIM_DURATION,
        "output_path": OUTPUT_PATH,
        "initially_infected": INITIALLY_INFECTED,
        "lowest_immunity": LOWEST_IMMUNITY,
        "highest_immunity": HIGHEST_IMMUNITY,
        "age_params": AGE_PARAMS,
        **AGENT_PARAMS
    }

    log: Logger = logging.getLogger(SIMULATOR_NAME)
    configure_logging(log_output=LOG_OUTPUT)

    log.info(f'Simulation has started with params: {simulation_params}')

    random.seed(RANDOM_SEED)
    env = simpy.Environment()
    metrics = configure_extended_neighbors_simulation(log=log,
                                                      environment=env,
                                                      agent_params=AGENT_PARAMS,
                                                      neighbors_data=NEIGHBORS_DATA,
                                                      lowest_immunity=LOWEST_IMMUNITY,
                                                      highest_immunity=HIGHEST_IMMUNITY,
                                                      age_params=AGE_PARAMS,
                                                      initially_infected=INITIALLY_INFECTED,
                                                      initially_infected_indices=INITIALLY_INFECTED_INDICES)

    log.info(f"Running Simulation: {config['title']}")
    env.run(until=SIM_DURATION)

    simulation_end = msk_now()
    simulation_end_str = simulation_end.strftime(DATETIME_FORMAT)
    elapsed = (simulation_end - simulation_start).total_seconds()

    simulation_params['simulation_start'] = simulation_start_str
    simulation_params['simulation_end'] = simulation_end_str
    simulation_params['simulation_duration_seconds'] = elapsed
    simulation_params['agent_class'] = SEIRNeighborsFSMAgent.__name__
    simulation_params['scenario_name'] = config['title']

    transitions_path = f"{OUTPUT_PATH}/{SIMULATOR_NAME}_Transitions-{simulation_end_str}.csv"
    params_path = f"{OUTPUT_PATH}/{SIMULATOR_NAME}_Params-{simulation_end_str}.json"

    log.info(f'Simulation has finished. Elapsed time {elapsed} seconds')
    log.info(f"Saving transitions to {transitions_path}")
    log.info(f"Saving params to {params_path}")

    metrics.metrics_to_csv(kind=Metric.TRANSITIONS, filename=transitions_path)

    with open(params_path, 'w') as f:
        json.dump(simulation_params, f)
