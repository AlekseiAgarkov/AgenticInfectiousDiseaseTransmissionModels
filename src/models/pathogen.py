from abc import ABC, abstractmethod
from typing import Any, Union, Dict

import numpy as np
import simpy


class Pathogen(ABC):

    def __init__(self, env: simpy.Environment, name: str, **kwargs):
        self.env = env
        self.name = name

    def __call__(self):
        return self._get_beta(sim_time=self.env.now)

    @abstractmethod
    def _get_beta(self, sim_time: int) -> Union[float, np.float64]:
        pass


PATHOGEN_REGISTRY: Dict[str, Pathogen] = {}


def register_pathogen(cls):
    """Decorator to register classes"""
    PATHOGEN_REGISTRY[cls.__name__] = cls
    return cls


@register_pathogen
class LinearPathogen(Pathogen):
    pathogen_values: np.ndarray[tuple[Any, ...], np.dtype[np.float64]]

    def __init__(self,
                 env: simpy.Environment,
                 name: str,
                 sim_duration: int,
                 base_beta: float,
                 max_beta: float):
        super().__init__(env, name)
        self.sim_duration = sim_duration
        self.base_beta = base_beta
        self.max_beta = max_beta

        self.pathogen_values = np.linspace(start=base_beta, num=sim_duration + 1, stop=max_beta)

    def _get_beta(self, sim_time: int) -> Union[float, np.float64]:
        return self.pathogen_values.item(sim_time)


@register_pathogen
class DiscretePredefinedPathogen(Pathogen):
    pathogen_values: np.ndarray[tuple[Any, ...], np.dtype[np.float64]]

    def __init__(self,
                 env: simpy.Environment,
                 name: str,
                 pathogen_values: np.ndarray[tuple[Any, ...], np.dtype[np.float64]],
                 is_circular: bool = False):
        super().__init__(env, name)

        self.pathogen_values = pathogen_values
        self.is_circular = is_circular

    def _get_beta(self, sim_time: int) -> Union[float, np.float64]:
        idx = sim_time % len(self.pathogen_values) if self.is_circular else sim_time
        return self.pathogen_values.item(idx)


def init_pathogen_from_config(env: simpy.Environment, config_data: Dict[str, Any]):
    """Create class instance from config dictionary"""
    class_name: str = config_data.get("class")
    if not class_name or class_name not in PATHOGEN_REGISTRY:
        raise ValueError(f"Unknown class: {class_name}")

    cls: Pathogen = PATHOGEN_REGISTRY[class_name]
    # Remove class key from kwargs
    kwargs = {k: v for k, v in config_data.items() if k != "class"}
    return cls(env=env, **kwargs)
