from argparse import ArgumentParser
import json
import random
import zoneinfo
from datetime import datetime
from typing import Dict

import simpy

from metrics.collector import MetricsCollector
from models.sir import SIRBasicFSMAgent

MSK = zoneinfo.ZoneInfo("Europe/Moscow")


def msk_now_str():
    return datetime.now(MSK).strftime("%Y-%m-%d-%H%M%SZ")


if __name__ == '__main__':
    p = ArgumentParser()
    paths = p.add_argument_group('paths')
    paths.add_argument('-o', '--output_path', type=str, nargs="?", default='simulation_data',
                       help="Simulation data output folder")
    paths.add_argument('-c', '--config_path', type=str, nargs="?", default="simulation_config",
                       help="Config path")

    simulation_params = p.add_argument_group("simulation_params")
    simulation_params.add_argument('-r', '--random_seed', type=int, nargs="?", default=42, help="Random seed")
    simulation_params.add_argument('-n', '--n_agents', type=int, nargs="?", default=100, help="Number of agents")
    simulation_params.add_argument('-t', '--sim_duration', type=int, nargs="?", default=365,
                                   help="Duration of Simulation, units")

    agent_params = p.add_argument_group("agent_params")
    agent_params.add_argument('-b', '--beta', type=float, nargs="?", default=0.025, help="Model parameter beta")
    agent_params.add_argument('-g', '--gamma', type=float, nargs="?", default=0.01, help="Model parameter gamma")

    args = p.parse_args()
    RANDOM_SEED = args.random_seed
    AGENT_PARAMS: dict = {k:  v for k, v in vars(args).items() if k in ["beta", "gamma", "sim_duration"]}
    N_AGENTS: int = args.n_agents
    SIM_DURATION: int = args.sim_duration
    OUTPUT_PATH: str = args.output_path
    CONFIG_PATH: str = args.config_path

    random.seed(RANDOM_SEED)
    env = simpy.Environment()
    agents: Dict[int, SIRBasicFSMAgent] = {
        n: SIRBasicFSMAgent(env=env, name=f"A_{n}", **AGENT_PARAMS)
        for n in range(N_AGENTS)}

    collector: MetricsCollector = MetricsCollector(
        env=env,
        entities=list(agents.values()),
        states=SIRBasicFSMAgent.states)

    for a in agents.values():
        env.process(a.run())

    env.process(collector.run())
    env.run(until=SIM_DURATION)

    simulation_ts = msk_now_str()
    collector.to_csv(f"{OUTPUT_PATH}/SIR_SimulationData-{simulation_ts}.csv")

    simulation_params = vars(p.parse_args())

    with open(f'{OUTPUT_PATH}/SIR_SimulationParams-{simulation_ts}.json', 'w') as f:
        json.dump(simulation_params, f)
