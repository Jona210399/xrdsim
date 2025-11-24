from typing import Callable

import numpy as np
from numpy.typing import NDArray


def minmax_scale(x: NDArray) -> NDArray:
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x


def add_background(
    x: NDArray,
    y: NDArray,
    background_function: Callable[[NDArray], NDArray],
) -> NDArray:
    background = background_function(x)
    return y + background


def sinusodal_background(
    x: NDArray,
    amplitude: float = 0.025,
    frequency: float = 0.05,
) -> NDArray:
    return amplitude * np.sin(frequency * x)


def scale_invariant_linear_background(
    x: NDArray,
    slope: float = -0.1,
    intercept: float = 0.0,
) -> NDArray:
    return slope * x / np.max(x) + intercept


def constant_background(
    x: NDArray,
    offset: float = 0.0,
) -> NDArray:
    return np.full_like(x, offset)


def polynomial_background(
    x: NDArray,
    coefficients: list[float] | NDArray = None,
) -> NDArray:
    """Polynomial background of arbitrary order.

    Args:
        x: Two-theta angles
        coefficients: Polynomial coefficients [c0, c1, c2, ...] for c0 + c1*x + c2*x^2 + ...
                     Defaults to [0.0, -0.001, 0.00001] (quadratic)
    """
    if coefficients is None:
        coefficients = [0.0, -0.001, 0.00001]

    coefficients = np.array(coefficients)
    background = np.zeros_like(x)

    for i, coef in enumerate(coefficients):
        background += coef * x**i

    return background


def exponential_background(
    x: NDArray,
    amplitude: float = 0.5,
    decay_rate: float = 0.05,
    offset: float = 0.0,
) -> NDArray:
    """Exponential decay background, typical for fluorescence.

    Args:
        x: Two-theta angles
        amplitude: Initial amplitude
        decay_rate: Decay rate (higher = faster decay)
        offset: Constant offset
    """
    return amplitude * np.exp(-decay_rate * x) + offset


def air_scattering_background(
    x: NDArray,
    amplitude: float = 0.1,
    offset: float = 0.0,
) -> NDArray:
    """1/sin(θ) background for air scattering at low angles.

    Args:
        x: Two-theta angles (in degrees)
        amplitude: Scaling factor
        offset: Constant offset
    """
    # Convert to radians and divide by 2 (two-theta to theta)
    theta = np.deg2rad(x / 2.0)
    # Avoid division by zero at very small angles
    sin_theta = np.sin(theta)
    sin_theta = np.where(sin_theta < 1e-6, 1e-6, sin_theta)
    return amplitude / sin_theta + offset


def amorphous_hump_background(
    x: NDArray,
    center: float = 25.0,
    width: float = 10.0,
    amplitude: float = 0.2,
) -> NDArray:
    """Gaussian hump representing amorphous content.

    Args:
        x: Two-theta angles
        center: Center position of the hump
        width: Standard deviation (width) of the hump
        amplitude: Height of the hump
    """
    return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)


def compose_background_functions(
    background_functions: list[Callable[[NDArray], NDArray]],
) -> Callable[[NDArray], NDArray]:
    def composed_function(x: NDArray) -> NDArray:
        total_background = np.zeros_like(x)
        for func in background_functions:
            total_background += func(x)
        return total_background

    return composed_function
