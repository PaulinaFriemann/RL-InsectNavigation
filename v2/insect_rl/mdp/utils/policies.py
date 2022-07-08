from typing import Iterable
from collections.abc import Iterable
from collections.abc import Mapping
from typing import TypeVar
import numpy as np


T = TypeVar('T')


def boltzmann(actions: Iterable[T], utilities: Mapping[T, float]):
    # calculate Boltzmann distribution.
    # TODO add Beta
    boltzman_distribution = []
    for a in actions:
        boltzman_distribution.append(np.exp(utilities[a]))
    boltzman_distribution = np.array(boltzman_distribution)
    boltzman_distribution /= np.sum(boltzman_distribution)
    return boltzman_distribution
