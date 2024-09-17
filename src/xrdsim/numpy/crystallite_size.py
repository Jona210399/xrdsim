from typing import Protocol

import numpy as np


class CrystalliteSizeSampler(Protocol):
    def sample() -> float: ...


class UniformCrystalliteSampler:
    def __init__(self, crystallite_size_range: tuple[float, float]):
        self.crystallite_size_range = crystallite_size_range

    def sample(self) -> float:
        return np.random.uniform(*self.crystallite_size_range)
