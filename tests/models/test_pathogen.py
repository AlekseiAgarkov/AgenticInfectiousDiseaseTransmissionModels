from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np

from models.pathogen import LinearPathogen, DiscretePredefinedPathogen


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
                                         base_beta=base_beta,
                                         max_beta=max_beta)

        for sim_step in range(sim_duration + 1):
            env.now = sim_step
            assert linear_pathogen() == (increment * sim_step + base_beta)


class DiscretePredefinedPathogenTests(TestCase):

    def test_discrete_predefined_pathogen(self):
        env = MagicMock()

        pathogen_values = np.array([0.1, 0.5, 0.2, 0.15, 0.4])
        sim_duration = len(pathogen_values)

        linear_pathogen = DiscretePredefinedPathogen(env=env,
                                                     name='linear_pathogen',
                                                     pathogen_values=pathogen_values)

        for sim_step in range(sim_duration):
            env.now = sim_step
            assert linear_pathogen() == pathogen_values[sim_step]
