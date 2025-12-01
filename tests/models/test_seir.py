from copy import deepcopy
from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np

from metrics.collector import TransitionMetricsCollector
from models.seir import SEIRFSMBase, SEIRClassicFSMAgent, SEIRNeighborsFSMAgent, SEIRNeighborsFSMExtended


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


class SEIRNeighborsFSMAgentNoInfectedNeighborsTests(TestCase):
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
        non_infective_neighbor = SEIRNeighborsFSMAgent(**{**self.seir_params, "beta": 0.0, "name": 'neighbor'})
        non_infective_neighbor.to_infected()
        assert non_infective_neighbor.state == 'infected'

        self.seir.set_neighbors(neighbors=[non_infective_neighbor])
        self.seir.try_get_exposed()
        assert self.seir.state == 'susceptible'


class SEIRNeighborsFSMExtendedTests(TestCase):
    def setUp(self):
        env = MagicMock()
        env.now = 0

        self.metrics_collector = TransitionMetricsCollector()
        self.seir_params = dict(env=env,
                                sim_duration=0,
                                beta=0.0,
                                x=0,
                                y=0,
                                e1=0,
                                e2=0,
                                t1=0,
                                t2=0,
                                age=np.random.randint(low=0, high=90),
                                metrics_collector=self.metrics_collector)
        self.highly_infective_neighbor = SEIRNeighborsFSMExtended(**{**self.seir_params,
                                                                     "beta": 1.0,
                                                                     "name": 'neighbor',
                                                                     "immunity_lower_bound": 0.0,
                                                                     "immunity_upper_bound": 1.0})
        self.highly_infective_neighbor.to_infected()
        assert self.highly_infective_neighbor.state == 'infected'

    def tearDown(self):
        self.seir_params = None
        self.metrics_collector = None
        self.highly_infective_neighbor = None

    def test_try_get_exposed_with_high_immunity(self):
        test_agent = SEIRNeighborsFSMExtended(**{**self.seir_params,
                                                 "name": "test_agent",
                                                 "immunity_lower_bound": 1.0,
                                                 "immunity_upper_bound": 1.0})

        test_agent.set_neighbors(neighbors=[self.highly_infective_neighbor])

        test_agent.try_get_exposed()
        assert test_agent.state == 'susceptible'

    def test_try_get_exposed_with_low_immunity(self):
        test_agent = SEIRNeighborsFSMExtended(**{**self.seir_params,
                                                 "name": "test_agent",
                                                 "immunity_lower_bound": 0.0,
                                                 "immunity_upper_bound": 0.0})

        test_agent.set_neighbors(neighbors=[self.highly_infective_neighbor])

        test_agent.try_get_exposed()
        assert test_agent.state == 'exposed'

    def test_try_get_exposed_with_infected(self):
        test_agent = SEIRNeighborsFSMExtended(**{**self.seir_params,
                                                 "name": "test_agent",
                                                 "immunity_lower_bound": 0.50,
                                                 "immunity_upper_bound": 0.50})

        test_agent.set_neighbors(neighbors=[self.highly_infective_neighbor])

        for _ in range(100):
            test_agent.try_get_exposed()
            if test_agent.state == 'exposed':
                break

        assert test_agent.state == 'exposed', "Test is probabilistic, please retry"

    def test_try_get_exposed_bounded_immunity_low(self):
        env = MagicMock()
        env.now = 0
        immunity_lower_bound = 0.0
        immunity_upper_bound = 1.0
        test_agent = SEIRNeighborsFSMExtended(**{**self.seir_params,
                                                 "name": "test_agent",
                                                 "immunity_lower_bound": immunity_lower_bound,
                                                 "immunity_upper_bound": immunity_upper_bound})

        test_agent.set_neighbors(neighbors=[self.highly_infective_neighbor])

        assert test_agent.current_immunity() == immunity_lower_bound
        test_agent.try_get_exposed()
        assert test_agent.state == 'exposed'

    def test_try_get_exposed_bounded_immunity_high(self):
        env = MagicMock()
        env.now = 182
        immunity_lower_bound = 0.0
        immunity_upper_bound = 1.0
        test_agent = SEIRNeighborsFSMExtended(**{**self.seir_params,
                                                 "name": "test_agent",
                                                 "immunity_lower_bound": immunity_lower_bound,
                                                 "immunity_upper_bound": immunity_upper_bound,
                                                 "env": env})

        test_agent.set_neighbors(neighbors=[self.highly_infective_neighbor])

        assert test_agent.current_immunity() == immunity_upper_bound
        test_agent.try_get_exposed()
        assert test_agent.state == 'susceptible'

    def test_mask_beta_penalty(self):
        env = MagicMock()
        env.now = 0

        test_agent = SEIRNeighborsFSMExtended(**{**self.seir_params,
                                                 "name": "test_agent",
                                                 "immunity_lower_bound": 0.0,
                                                 "immunity_upper_bound": 0.0,
                                                 "env": env})

        highly_infective_neighbor_heavily_masked = deepcopy(self.highly_infective_neighbor)
        highly_infective_neighbor_heavily_masked.mask_beta_penalty = 1.0
        highly_infective_neighbor_heavily_masked.wears_mask_at_contact_p = 1.0
        test_agent.set_neighbors(neighbors=[highly_infective_neighbor_heavily_masked])

        assert highly_infective_neighbor_heavily_masked.beta_f() == 1.0
        assert highly_infective_neighbor_heavily_masked.probability_to_infect() == 0.0
        assert test_agent.current_immunity() == 0.0
        test_agent.try_get_exposed()
        assert test_agent.state == 'susceptible'

    def test_pollutant_beta_penalty(self):
        test_agent = SEIRNeighborsFSMExtended(**{**self.seir_params,
                                                 "name": "test_agent",
                                                 "immunity_lower_bound": 1.0,
                                                 "immunity_upper_bound": 1.0,
                                                 "exposed_to_pollutant": True,
                                                 "pollutant_beta_penalty": 1.0})

        test_agent.set_neighbors(neighbors=[self.highly_infective_neighbor])

        test_agent.try_get_exposed()
        assert test_agent.state == 'exposed'

    def test_wears_mask_always(self):
        test_agent = SEIRNeighborsFSMExtended(**{**self.seir_params,
                                                 "name": "test_agent",
                                                 "beta": 1.0,
                                                 "wears_mask_at_contact_p": 1.0,
                                                 "mask_beta_penalty": 1.0})
        test_agent.to_infected()
        assert test_agent.state == 'infected'

        assert test_agent.probability_to_infect() == 0.0

    def test_wears_mask_sometimes(self):
        test_agent = SEIRNeighborsFSMExtended(**{**self.seir_params,
                                                 "name": "test_agent",
                                                 "beta": 1.0,
                                                 "wears_mask_at_contact_p": 0.5,
                                                 "mask_beta_penalty": 0.25})
        test_agent.to_infected()
        assert test_agent.state == 'infected'

        assert 0.0 <= test_agent.probability_to_infect() <= 1.0

    def test_wears_mask_never(self):
        test_agent = SEIRNeighborsFSMExtended(**{**self.seir_params,
                                                 "name": "test_agent",
                                                 "beta": 1.0,
                                                 "wears_mask_at_contact_p": 0.0,
                                                 "mask_beta_penalty": 1.0})

        test_agent.to_infected()
        assert test_agent.state == 'infected'

        assert test_agent.probability_to_infect() == 1.0
