import numba
import numpy as np
from numpy.typing import NDArray

from xrdsim.numpy.crystallite_size import CrystalliteSizeSampler


class GaussianProfile:
    def __init__(self, x_range: tuple[float, float, int]):
        self.x_range = x_range

    def convolute_peaks(
        self,
        peaks_x: NDArray,
        peaks_y: NDArray,
        sigmas: NDArray,
    ):

        return self._convolute_peaks(
            peaks_x,
            peaks_y,
            sigmas,
            self.x_range,
        )

    @staticmethod
    @numba.njit(cache=True)
    def _convolute_peaks(
        peaks_x: NDArray,
        peaks_y: NDArray,
        sigmas: NDArray,
        x_range: tuple[float, float, int],
    ):
        xs = np.linspace(*x_range)
        ys = np.zeros(len(xs))

        for peak_x, peak_y, sigma in zip(peaks_x, peaks_y, sigmas):
            y: np.ndarray = (
                peak_y
                / (sigma * np.sqrt(2 * np.pi))
                * np.exp(-1 / (2 * sigma**2) * (xs - peak_x) ** 2)
            )
            ys += y

        return ys


class GaussianScherrerProfile:
    def __init__(
        self,
        gaussian_profile: GaussianProfile,
        shape_factor: float,
        wavelength: float,
        crystallite_size_sampler: CrystalliteSizeSampler,
    ) -> None:
        self.gaussian_profile = gaussian_profile
        self.shape_factor = shape_factor
        self.wavelength = wavelength
        self.crystallite_size_sampler = crystallite_size_sampler

    def convolute_peaks(
        self,
        peaks_x: NDArray,
        peaks_y: NDArray,
    ):

        crystallite_size = self.crystallite_size_sampler.sample()

        sigmas = self.scherrer_equation(
            peaks_x,
            crystallite_size,
            self.shape_factor,
            self.wavelength,
        )

        return self.gaussian_profile.convolute_peaks(peaks_x, peaks_y, sigmas)

    @staticmethod
    @numba.njit(cache=True)
    def scherrer_equation(
        peak_two_thetas: np.ndarray,
        crystallite_size: float,
        shape_factor: float,
        wavelength: float,
    ):
        thetas: np.ndarray = np.radians(peak_two_thetas) / 2

        betas: np.ndarray = (shape_factor * wavelength) / (
            np.cos(thetas) * crystallite_size
        )

        sigmas = np.sqrt(1 / (2 * np.log(2))) * 0.5 * np.degrees(betas)

        return sigmas
