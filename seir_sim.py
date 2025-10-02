import json
import logging
import random
import tomllib
from argparse import ArgumentParser
from pathlib import Path

import simpy

from models.simulation import configure_simulation
from simulation_utils.logging import configure_logging
from simulation_utils.time import msk_now, DATETIME_FORMAT

if __name__ == '__main__':
    p = ArgumentParser()
    paths = p.add_argument_group('paths')
    paths.add_argument('-o', '--output_path', type=str, nargs="?", help="Simulation data output folder")
    paths.add_argument('-c', '--config_path', type=str, nargs="?", default=None,
                       help="Config path")

    simulation_params = p.add_argument_group("simulation_params")
    simulation_params.add_argument('-r', '--random_seed', type=int, nargs="?", help="Random seed")
    simulation_params.add_argument('-n', '--n_agents', type=int, nargs="?", help="Number of agents")
    simulation_params.add_argument('-t', '--sim_duration', type=int, nargs="?", help="Duration of Simulation, units")

    agent_params = p.add_argument_group("agent_params")
    agent_params.add_argument('-b', '--beta', type=float, nargs="?", help="Model parameter Beta")
    agent_params.add_argument('-g', '--gamma', type=float, nargs="?", help="Model parameter Gamma")
    agent_params.add_argument('-s', '--sigma', type=float, nargs="?", help="Model parameter Sigma")

    args = p.parse_args()
    config = None
    if args.config_path is not None:
        config = tomllib.loads(Path(args.config_path).read_text(encoding="utf-8"))

    simulation_start = msk_now()
    simulation_start_str = simulation_start.strftime(DATETIME_FORMAT)

    SIMULATOR_NAME = config['simulator']['name']

    RANDOM_SEED = args.random_seed or config['simulation']['random_seed']
    AGENT_PARAMS: dict = {
        "beta": args.beta or config['agents']['beta'],
        "gamma": args.gamma or config['agents']['gamma'],
        "sigma": args.gamma or config['agents']['sigma'],
        "sim_duration": args.sim_duration or config['simulation']['sim_duration']
    }

    N_AGENTS: int = args.n_agents or config['simulation']['n_agents']
    SIM_DURATION: int = args.sim_duration or config['simulation']['sim_duration']
    OUTPUT_PATH: str = args.output_path or config['paths']['output_path']
    LOG_OUTPUT: str = f'{config['paths']['log_output']}/{SIMULATOR_NAME}_Log-{simulation_start_str}.log'

    simulation_params = {
        "random_seed": RANDOM_SEED,
        "n_agents": N_AGENTS,
        "sim_duration": SIM_DURATION,
        "output_path": OUTPUT_PATH,
        **AGENT_PARAMS
    }

    log: logging.Logger = logging.getLogger(SIMULATOR_NAME)
    configure_logging(log_output=LOG_OUTPUT)

    log.info(f'Simulation has started with params: {simulation_params}')

    random.seed(RANDOM_SEED)
    env = simpy.Environment()
    metrics = configure_simulation(environment=env, agent_params=AGENT_PARAMS, n_agents=N_AGENTS)
    env.run(until=SIM_DURATION)

    simulation_end = msk_now()
    simulation_end_str = simulation_end.strftime(DATETIME_FORMAT)
    elapsed = (simulation_end - simulation_start).total_seconds()

    simulation_params['simulation_start'] = simulation_start_str
    simulation_params['simulation_end'] = simulation_end_str
    simulation_params['simulation_duration_seconds'] = elapsed

    log.info(f'Simulation has finished. Elapsed time {elapsed} seconds')

    metrics.to_csv(f"{OUTPUT_PATH}/{SIMULATOR_NAME}_Data-{simulation_end_str}.csv")

    with open(f'{OUTPUT_PATH}/{SIMULATOR_NAME}_Params-{simulation_end_str}.json', 'w') as f:
        json.dump(simulation_params, f)
