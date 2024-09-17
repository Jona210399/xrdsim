from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pymatgen.core.structure import Structure

from xrdsim.constants import ATOMIC_SCATTERING_PARAMS


@dataclass
class ScatteringInformation:
    atomic_numbers: NDArray  # Shape: (m)
    scattering_parameters: NDArray  # Shape: (m, 4, 2)
    fractional_coordinates: NDArray  # Shape: (m, 3)
    site_occupations: NDArray  # Shape: (m)
    debyewaller_factors: NDArray  # Shape: (m)
    hkls: NDArray  # Shape: (p, 3)
    g_hkls: NDArray  # Shape: (p)

    @classmethod
    def from_structure(
        cls,
        structure: Structure,
        wavelength: float,
        angle_range: tuple[float, float, int],
    ):
        start_angle, end_angle, _ = angle_range
        lattice = structure.lattice
        debyewaller_factors = {}

        angles = np.deg2rad(np.array([start_angle, end_angle], dtype=np.float64) / 2)
        min_r, max_r = 2 * np.sin(angles) / wavelength
        recip_latt = lattice.reciprocal_lattice_crystallographic
        recip_pts = recip_latt.get_points_in_sphere(
            [[0, 0, 0]],
            [0, 0, 0],
            max_r,
            zip_results=False,
        )
        hkls, g_hkls, _, _ = recip_pts
        mask = g_hkls >= min_r
        hkls = hkls[mask]
        g_hkls = g_hkls[mask]
        sorted_indices = np.argsort(g_hkls)
        hkls = hkls[sorted_indices]
        g_hkls = g_hkls[sorted_indices]

        atomic_numbers = []
        scattering_parameters = []
        site_occupations = []
        dw_factors = []
        fractional_coordinates = []

        for site in structure:
            for sp, occu in site.species.items():
                atomic_numbers.append(sp.Z)
                try:
                    c = ATOMIC_SCATTERING_PARAMS[sp.symbol]
                except KeyError:
                    return None

                scattering_parameters.append(c)
                dw_factors.append(debyewaller_factors.get(sp.symbol, 0))
                fractional_coordinates.append(site.frac_coords)
                site_occupations.append(occu)

        mask = np.nonzero(g_hkls)
        g_hkls = g_hkls[mask]
        hkls = hkls[mask]
        hkls = np.round(hkls)

        return cls(
            np.array(atomic_numbers),
            np.array(scattering_parameters),
            np.array(fractional_coordinates),
            np.array(site_occupations),
            np.array(dw_factors),
            hkls,
            g_hkls,
        )