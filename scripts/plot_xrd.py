import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pymatgen.core import Structure

from xrdsim.calculator import XRDCalculator
from xrdsim.constants import DEFAULT_SHAPEFACTOR, DEFAULT_WAVELENGTH
from xrdsim.numpy.crystallite_size import ConstantCrystalliteSize
from xrdsim.numpy.peak_calculator import NumbaXRDPeakCalculator
from xrdsim.numpy.peak_profiles import GaussianProfile, GaussianScherrerProfile
from xrdsim.tools.plotter import XRDPlotter

ANGLE_RANGE = (10, 160, 15000)


def get_xrd_calculator() -> XRDCalculator:
    return XRDCalculator(
        peak_calculator=NumbaXRDPeakCalculator(
            wavelength=DEFAULT_WAVELENGTH,
            angle_range=ANGLE_RANGE,
        ),
        peak_profile=GaussianScherrerProfile(
            gaussian_profile=GaussianProfile(ANGLE_RANGE),
            shape_factor=DEFAULT_SHAPEFACTOR,
            wavelength=DEFAULT_WAVELENGTH,
            crystallite_size_provider=ConstantCrystalliteSize(100.0),
        ),
        rescale_intensity=True,
    )


def background_function(x):
    return 0.025 * np.sin(0.05 * x) + 0.1 * x / np.max(x)


def main():

    path = Path.cwd().joinpath("data", "mp-1183076.cif")

    structure = Structure.from_file(path)
    print(structure)
    calculator = get_xrd_calculator()
    two_thetas, intensities, meta = calculator.calculate(structure)

    background = intensities + background_function(two_thetas[::-1])

    intensities = (background - np.min(background)) / (
        np.max(background) - np.min(background)
    )

    intensity_noise = np.random.normal(loc=0, scale=0.0025, size=ANGLE_RANGE[2])
    intensities = intensities + intensity_noise
    plotter = XRDPlotter(two_thetas, intensities)
    fig, ax = plotter.plot()
    plt.show()


if __name__ == "__main__":
    main()
