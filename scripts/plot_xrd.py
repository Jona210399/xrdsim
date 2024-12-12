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

path = Path.cwd().joinpath("data", "mp-1183076.cif")

structure = Structure.from_file(path)
print(structure)
angle_range = (10, 160, 15000)


def get_xrd_calculator() -> XRDCalculator:
    return XRDCalculator(
        peak_calculator=NumbaXRDPeakCalculator(
            wavelength=DEFAULT_WAVELENGTH,
            angle_range=angle_range,
        ),
        peak_profile=GaussianScherrerProfile(
            gaussian_profile=GaussianProfile(angle_range),
            shape_factor=DEFAULT_SHAPEFACTOR,
            wavelength=DEFAULT_WAVELENGTH,
            crystallite_size_provider=ConstantCrystalliteSize(100.0),
        ),
        rescale_intensity=True,
    )


def generate_xrd_background(two_theta, coefficients=(1e-4, 1e-10, 1e-3)):
    background = np.polyval(coefficients, two_theta[::-1])
    # background = np.clip(background, 0, None)
    return background


def background_function(x):
    return 0.025 * np.sin(0.05 * x) + 0.1 * x / np.max(x)  # Sine + linear background


calculator = get_xrd_calculator()
two_thetas, intensities, meta = calculator.calculate(structure)


# Add the background to the simulated intensities
background = intensities + background_function(two_thetas[::-1])

# Normalize the intensities to keep them in the range [0, 1]
intensities = (background - np.min(background)) / (
    np.max(background) - np.min(background)
)


intensity_noise = np.random.normal(loc=0, scale=0.0025, size=15000)
# intensities = intensities + intensity_noise
# intensities = intensities + background
intensities = intensities + intensity_noise
plotter = XRDPlotter(two_thetas, intensities)
fig, ax = plotter.plot()
plt.show()
