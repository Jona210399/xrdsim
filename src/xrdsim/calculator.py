from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from pymatgen.core.structure import Structure

from xrdsim.constants import (
    DEFAULT_ANGLE_RANGE,
    DEFAULT_CRYSTALLITE_SIZE_RANGE,
    DEFAULT_RESCALE_INTENSITY,
    DEFAULT_SHAPEFACTOR,
    DEFAULT_WAVELENGTH,
)
from xrdsim.numpy.crystallite_size import UniformCrystalliteSampler
from xrdsim.numpy.peak_calculator import NumbaXRDPeakCalculator
from xrdsim.numpy.peak_profiles import GaussianProfile, GaussianScherrerProfile


class PeakCalculator(Protocol):
    def calculate(structure: Structure) -> tuple[NDArray, NDArray]: ...


class PeakProfile(Protocol):
    def convolute_peaks(
        peak_x: NDArray,
        peak_y: NDArray,
        *args,
        **kwargs,
    ) -> tuple[NDArray, NDArray]: ...


class XRDCalculator:
    def __init__(
        self,
        peak_calculator: PeakCalculator,
        peak_profile: PeakProfile,
        rescale_intensity: bool,
    ):
        self.rescale_intensity = rescale_intensity
        self.peak_calculator = peak_calculator
        self.peak_profile = peak_profile

    def calculate(self, structure: Structure) -> tuple[NDArray, NDArray]:
        peak_two_thetas, peak_intensities = self.peak_calculator.calculate(structure)

        two_thetas, intensities = self.peak_profile.convolute_peaks(
            peak_two_thetas,
            peak_intensities,
        )

        if self.rescale_intensity:
            intensities = intensities / np.max(intensities)

        return two_thetas, intensities


def get_default_numpy_xrd_calculator() -> XRDCalculator:
    return XRDCalculator(
        peak_calculator=NumbaXRDPeakCalculator(
            wavelength=DEFAULT_WAVELENGTH,
            angle_range=DEFAULT_ANGLE_RANGE,
        ),
        peak_profile=GaussianScherrerProfile(
            gaussian_profile=GaussianProfile(DEFAULT_ANGLE_RANGE),
            shape_factor=DEFAULT_SHAPEFACTOR,
            wavelength=DEFAULT_WAVELENGTH,
            crystallite_size_sampler=UniformCrystalliteSampler(
                DEFAULT_CRYSTALLITE_SIZE_RANGE
            ),
        ),
        rescale_intensity=DEFAULT_RESCALE_INTENSITY,
    )
