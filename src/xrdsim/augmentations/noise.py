import numpy as np
from numpy.typing import NDArray


def add_gaussian_noise(
    x: NDArray,
    scale: float = 0.0025,
) -> NDArray:
    noise = np.random.normal(loc=0, scale=scale, size=x.shape)
    return x + noise
