from dataclasses import asdict

import numba
import numpy as np
from numpy.typing import NDArray
from pymatgen.core import Lattice, Structure

from xrdsim.constants import SCALED_INTENSITY_TOL, TWO_THETA_TOL
from xrdsim.scattering_information import ScatteringInformation


class NumbaXRDPeakCalculator:
    def __init__(
        self,
        wavelength: float,
        angle_range: tuple[float, float, int],
    ):
        self.wavelength = wavelength
        self.angle_range = angle_range

    def calculate(
        self,
        structure: Structure,
    ) -> tuple[
        NDArray,
        NDArray,
    ]:
        si = ScatteringInformation.from_structure(
            structure,
        )
        hkls, g_hkls = self.get_hkls(structure.lattice)

        peak_two_thetas, peak_intensities = self.get_peaks(
            self.wavelength,
            **si.to_ndarrays(),
            hkls=hkls,
            g_hkls=g_hkls,
        )

        return peak_two_thetas, peak_intensities

    def get_hkls(self, lattice: Lattice) -> tuple[NDArray, NDArray]:
        angles = np.deg2rad(np.array(*self.angle_range[:2], dtype=np.float64) / 2)
        min_r, max_r = 2 * np.sin(angles) / self.wavelength
        recip_latt = lattice.reciprocal_lattice_crystallographic

        hkls, g_hkls, _, _ = recip_latt.get_points_in_sphere(
            [[0, 0, 0]],
            [0, 0, 0],
            max_r,
            zip_results=False,
        )
        valid_mask = g_hkls >= min_r
        hkls, g_hkls = hkls[valid_mask], g_hkls[valid_mask]

        sorted_indices = np.argsort(g_hkls)
        hkls, g_hkls = hkls[sorted_indices], g_hkls[sorted_indices]

        non_zero_mask = np.nonzero(g_hkls)
        hkls, g_hkls = hkls[non_zero_mask], g_hkls[non_zero_mask]

        return np.round(hkls), g_hkls

    @staticmethod
    @numba.njit(cache=True, fastmath=True)
    def get_peaks(
        wavelength: float,
        atomic_numbers: NDArray,
        scattering_parameters: NDArray,
        fractional_coordinates: NDArray,
        site_occupations: NDArray,
        debyewaller_factors: NDArray,
        hkls: NDArray,
        g_hkls: NDArray,
    ) -> tuple[NDArray, NDArray]:

        thetas = np.arcsin(wavelength * g_hkls / 2)
        s = g_hkls / 2
        s2 = s**2

        lorentz_factors = (1 + np.cos(2 * thetas) ** 2) / (
            np.sin(thetas) ** 2 * np.cos(thetas)
        )
        two_thetas = np.degrees(2 * thetas)

        intensities = np.zeros(len(two_thetas))

        for i in range(0, len(hkls)):
            g_dot_r = np.dot(
                hkls[i],
                fractional_coordinates.T,
            )

            fs = atomic_numbers - 41.78214 * s2[i] * np.sum(
                scattering_parameters[:, :, 0]
                * np.exp(-scattering_parameters[:, :, 1] * s2[i]),
                axis=1,
            )

            dw_correction = np.exp(-debyewaller_factors * s2[i])

            f_hkl: NDArray = np.sum(
                fs * site_occupations * np.exp(2j * np.pi * g_dot_r) * dw_correction
            )

            i_hkl = (f_hkl * f_hkl.conjugate()).real

            i_hkl = i_hkl * lorentz_factors[i]

            ind = np.where(np.abs(two_thetas - two_thetas[i]) < TWO_THETA_TOL)

            if len(ind[0]) > 0:
                intensities[ind[0][0]] += i_hkl
            else:
                intensities[ind] = i_hkl

        max_intensity = np.max(intensities)

        mask = intensities / max_intensity * 100 > SCALED_INTENSITY_TOL

        two_thetas = two_thetas[mask]
        intensities = intensities[mask]

        return two_thetas, intensities
