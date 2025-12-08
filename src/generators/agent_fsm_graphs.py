from simpy import Environment

from metrics.collector import TransitionMetricsCollector
from models.seir import SEIRClassicFSMAgent, SEIRNeighborsFSMAgent, SEIRNeighborsFSMExtended
from models.sir import SIRBasicFSMAgent


def generate_graph(agent, path):
    agent.machine.get_graph().draw(path, format="png", prog='dot')


if __name__ == '__main__':
    env = Environment()

    sir_basic_fsm_agent = SIRBasicFSMAgent(env=env, name="", beta=0.01, gamma=0.01, sim_duration=100)
    generate_graph(sir_basic_fsm_agent, path=f"docs/img/{SIRBasicFSMAgent.__name__}.png")

    seir_basic_fsm_agent = SEIRClassicFSMAgent(env=env, name="", beta=0.01, gamma=0.01, sigma=0.11, sim_duration=100)
    generate_graph(seir_basic_fsm_agent, path=f"docs/img/{SEIRClassicFSMAgent.__name__}.png")

    metrics_collector = TransitionMetricsCollector()

    seir_neighbors_fsm_agent = SEIRNeighborsFSMAgent(
        env=env,
        metrics_collector=metrics_collector,
        name="",
        beta=0.01,
        sim_duration=100,
        x=0.01, y=0.01,
        e1=1, e2=5,
        t1=5, t2=7)
    generate_graph(seir_neighbors_fsm_agent, path=f"docs/img/{SEIRNeighborsFSMAgent.__name__}.png")

    seir_neighbors_fsm_extended_agent = SEIRNeighborsFSMExtended(
        env=env,
        metrics_collector=metrics_collector,
        name="",
        beta=0.01,
        sim_duration=100,
        x=0.01, y=0.01,
        e1=1, e2=5,
        t1=5, t2=7,
        age=42)
    generate_graph(seir_neighbors_fsm_extended_agent, path=f"docs/img/{SEIRNeighborsFSMExtended.__name__}.png")
