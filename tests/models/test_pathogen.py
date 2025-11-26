import tomllib
from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np

from models.pathogen import LinearPathogen, DiscretePredefinedPathogen, init_pathogen_from_config


class LinearPathogenTests(TestCase):

    def test_linear_pathogen(self):
        env = MagicMock()

        base_beta = 0.0
        increment = 0.1
        sim_duration = 10
        max_beta = sim_duration * increment

        linear_pathogen = LinearPathogen(env=env,
                                         name='linear_pathogen',
                                         sim_duration=sim_duration,
                                         start_beta=base_beta,
                                         end_beta=max_beta)

        for sim_step in range(sim_duration + 1):
            env.now = sim_step
            assert linear_pathogen() == (increment * sim_step + base_beta)


class DiscretePredefinedPathogenTests(TestCase):

    def test_discrete_predefined_pathogen(self):
        env = MagicMock()

        pathogen_values = np.array([0.1, 0.5, 0.2, 0.15, 0.4])
        sim_duration = len(pathogen_values)

        discrete_predefined_pathogen = DiscretePredefinedPathogen(env=env,
                                                                  name='linear_pathogen',
                                                                  pathogen_values=pathogen_values)

        for sim_step in range(sim_duration):
            env.now = sim_step
            assert discrete_predefined_pathogen() == pathogen_values[sim_step]

    def test_discrete_predefined_pathogen_circular(self):
        env = MagicMock()

        pathogen_values = np.array([0.1, 0.5, 0.2, 0.15, 0.4])
        target_pathogen_values = np.concatenate((pathogen_values, pathogen_values))
        sim_duration = len(pathogen_values) * 2

        discrete_predefined_pathogen = DiscretePredefinedPathogen(env=env,
                                                                  name='linear_pathogen',
                                                                  pathogen_values=pathogen_values,
                                                                  is_circular=True)

        for sim_step in range(sim_duration):
            env.now = sim_step
            assert discrete_predefined_pathogen() == target_pathogen_values[sim_step]


class InitPathogensTests(TestCase):

    def setUp(self):
        self.toml_config = """
        [linear_pathogen]
        class = "LinearPathogen"
        name = "LinearPathogen"
        start_beta = 0.0
        end_beta = 1.0
    
        [discrete_predefined_pathogen]
        class = "DiscretePredefinedPathogen"
        name = "DiscretePredefinedPathogen"
        pathogen_values = [0.1, 0.2]
        is_circular = true
        """
        self.config = tomllib.loads(self.toml_config)
        self.env = MagicMock()

    def tearDown(self):
        self.toml_config = None
        self.config = None
        self.env = None

    def test_init_linear_pathogen(self):
        linear_pathogen = init_pathogen_from_config(env=self.env,
                                                    config_data={**self.config["linear_pathogen"],
                                                                 "sim_duration": 10})

        assert isinstance(linear_pathogen, LinearPathogen)

    def test_init_discrete_predefined_pathogen(self):
        discrete_predefined_pathogen = init_pathogen_from_config(
            env=self.env,
            config_data=self.config["discrete_predefined_pathogen"])

        assert isinstance(discrete_predefined_pathogen, DiscretePredefinedPathogen)

    def test_init_ignore_extra_params(self):
        discrete_predefined_pathogen = init_pathogen_from_config(
            env=self.env,
            config_data={**self.config["discrete_predefined_pathogen"], "extra_param": 1})

        assert isinstance(discrete_predefined_pathogen, DiscretePredefinedPathogen)
