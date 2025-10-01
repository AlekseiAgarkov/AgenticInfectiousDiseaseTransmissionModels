import json
import random
import tomllib
import zoneinfo
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import simpy

from models.seir import configure_simulation

MSK = zoneinfo.ZoneInfo("Europe/Moscow")


def msk_now_str():
    return datetime.now(MSK).strftime("%Y-%m-%d-%H%M%SZ")


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

    simulation_params = {
        "random_seed": RANDOM_SEED,
        "n_agents": N_AGENTS,
        "sim_duration": SIM_DURATION,
        "output_path": OUTPUT_PATH,
        **AGENT_PARAMS
    }

    random.seed(RANDOM_SEED)
    env = simpy.Environment()
    metrics = configure_simulation(environment=env, agent_params=AGENT_PARAMS, n_agents=N_AGENTS)
    env.run(until=SIM_DURATION)

    simulation_ts = msk_now_str()
    metrics.to_csv(f"{OUTPUT_PATH}/SEIR_SimulationData-{simulation_ts}.csv")

    with open(f'{OUTPUT_PATH}/SEIR_SimulationParams-{simulation_ts}.json', 'w') as f:
        json.dump(simulation_params, f)
