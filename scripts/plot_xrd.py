from pathlib import Path

import matplotlib.pyplot as plt
from pymatgen.core import Structure

from xrdsim.augmentations.background import (
    add_background,
    compose_background_functions,
    minmax_scale,
    scale_invariant_linear_background,
    sinusodal_background,
)
from xrdsim.augmentations.noise import add_gaussian_noise
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


def main():
    path = Path.cwd().joinpath("data", "mp-1183076.cif")

    structure = Structure.from_file(path)
    print(structure)
    calculator = get_xrd_calculator()
    two_thetas, intensities, meta = calculator.calculate(structure)

    intensities = add_background(
        two_thetas,
        intensities,
        compose_background_functions(
            [
                sinusodal_background,
                scale_invariant_linear_background,
            ]
        ),
    )

    intensities = add_gaussian_noise(intensities, scale=0.0025)
    intensities = minmax_scale(intensities)

    plotter = XRDPlotter(two_thetas, intensities)
    fig, ax = plotter.plot()
    plt.show()


if __name__ == "__main__":
    main()
