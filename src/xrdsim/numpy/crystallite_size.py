from typing import Protocol

import numpy as np


class CrystalliteSizeProvider(Protocol):
    def get_crystallite_size() -> float: ...


class UniformCrystalliteSampler(CrystalliteSizeProvider):
    def __init__(self, crystallite_size_range: tuple[float, float]):
        self.crystallite_size_range = crystallite_size_range

    def get_crystallite_size(self) -> float:
        return np.random.uniform(*self.crystallite_size_range)


class ConstantCrystalliteSize(CrystalliteSizeProvider):
    def __init__(self, crystallite_size: float):
        self.crystallite_size = crystallite_size

    def get_crystallite_size(self):
        return self.crystallite_size
