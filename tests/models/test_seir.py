from unittest import TestCase
from unittest.mock import MagicMock

from metrics.collector import TransitionMetricsCollector
from models.seir import SEIRFSMBase, SEIRClassicFSMAgent, SEIRNeighborsFSMAgent


class SEIRFSMBaseTests(TestCase):
    def setUp(self):
        self.seir = SEIRFSMBase(env=None, name=0, sim_duration=0)

    def tearDown(self):
        self.seir = None

    def test_try_get_exposed(self):
        self.seir.try_get_exposed()
        assert self.seir.state == 'exposed'

    def test_try_get_infected(self):
        self.seir.to_exposed()
        self.seir.try_get_infected()
        assert self.seir.state == 'infected'

    def test_try_recover(self):
        self.seir.to_infected()
        self.seir.try_recover()
        assert self.seir.state == 'recovered'


class SEIRClassicFSMAgentTransitionAlwaysTests(TestCase):
    def setUp(self):
        self.seir = SEIRClassicFSMAgent(env=None,
                                        name=0,
                                        sim_duration=0,
                                        beta=1.0,
                                        gamma=1.0,
                                        sigma=1.0)

    def tearDown(self):
        self.seir = None

    def test_try_get_exposed(self):
        self.seir.try_get_exposed()
        assert self.seir.state == 'exposed'

    def test_try_get_infected(self):
        self.seir.to_exposed()
        self.seir.try_get_infected()
        assert self.seir.state == 'infected'

    def test_try_recover(self):
        self.seir.to_infected()
        self.seir.try_recover()
        assert self.seir.state == 'recovered'


class SEIRClassicFSMAgentTransitionNeverTests(TestCase):
    def setUp(self):
        self.seir = SEIRClassicFSMAgent(env=None,
                                        name=0,
                                        sim_duration=0,
                                        beta=0.0,
                                        gamma=0.0,
                                        sigma=0.0)

    def tearDown(self):
        self.seir = None

    def test_try_get_exposed(self):
        self.seir.try_get_exposed()
        assert self.seir.state == 'susceptible'

    def test_try_get_infected(self):
        self.seir.to_exposed()
        self.seir.try_get_infected()
        assert self.seir.state == 'exposed'

    def test_try_recover(self):
        self.seir.to_infected()
        self.seir.try_recover()
        assert self.seir.state == 'infected'


class SEIRNeighborsFSMAgentNoInfectedNeighbors(TestCase):
    def setUp(self):
        env = MagicMock()
        env.now = 0

        self.metrics_collector = TransitionMetricsCollector()
        self.seir_params = dict(env=env,
                                name='test_agent',
                                sim_duration=0,
                                beta=0.0,
                                x=0,
                                y=0,
                                e1=0,
                                e2=0,
                                t1=0,
                                t2=0,
                                metrics_collector=self.metrics_collector)
        self.seir = SEIRNeighborsFSMAgent(**self.seir_params)

    def tearDown(self):
        self.seir = None
        self.seir_params = None
        self.metrics_collector = None

    def test_try_get_exposed_from_susceptible(self):
        susceptible_neighbor = SEIRNeighborsFSMAgent(**{**self.seir_params, "beta": 1.0, "name": 'neigbor'})
        assert susceptible_neighbor.state == 'susceptible'

        self.seir.set_neighbors(neighbors=[susceptible_neighbor])

        self.seir.try_get_exposed()
        assert self.seir.state == 'susceptible'

    def test_try_get_exposed_from_highly_infective(self):
        highly_infective_neighbor = SEIRNeighborsFSMAgent(**{**self.seir_params, "beta": 1.0, "name": 'neigbor'})
        highly_infective_neighbor.to_infected()
        assert highly_infective_neighbor.state == 'infected'

        self.seir.set_neighbors(neighbors=[highly_infective_neighbor])

        self.seir.try_get_exposed()
        assert self.seir.state == 'exposed'

    def test_try_get_exposed_from_non_infective(self):
        non_infective_neighbor = SEIRNeighborsFSMAgent(**{**self.seir_params, "beta": 0.0, "name": 'neigbor'})
        non_infective_neighbor.to_infected()
        assert non_infective_neighbor.state == 'infected'

        self.seir.set_neighbors(neighbors=[non_infective_neighbor])
        self.seir.try_get_exposed()
        assert self.seir.state == 'susceptible'
