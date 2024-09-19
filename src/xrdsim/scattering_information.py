from dataclasses import asdict, dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from pymatgen.core import Element, Site, Structure

from xrdsim.constants import ATOMIC_SCATTERING_PARAMS


@dataclass
class ScatteringInformation:
    atomic_numbers: list  # Shape: (m)
    scattering_parameters: list[list[list[float]]]  # Shape: (m, 4, 2)
    fractional_coordinates: list[list[float]]  # Shape: (m, 3)
    site_occupations: list  # Shape: (m)
    debyewaller_factors: list  # Shape: (m)

    @classmethod
    def from_structure(
        cls,
        structure: Structure,
    ):
        debyewaller_factors = {}

        atomic_numbers = []
        scattering_parameters = []
        site_occupations = []
        dw_factors = []
        fractional_coordinates = []

        for site in structure:
            site: Site
            for sp, occu in site.species.items():
                sp: Element
                atomic_numbers.append(sp.Z)
                try:
                    c = ATOMIC_SCATTERING_PARAMS[sp.symbol]
                except KeyError:
                    return None

                scattering_parameters.append(c)
                dw_factors.append(debyewaller_factors.get(sp.symbol, 0))
                fractional_coordinates.append(site.frac_coords)
                site_occupations.append(occu)

        return cls(
            atomic_numbers,
            scattering_parameters,
            fractional_coordinates,
            site_occupations,
            dw_factors,
        )

    def to_tensors(
        self, device: torch.device, dtype: torch.dtype
    ) -> dict[str, torch.Tensor]:

        return {
            k: torch.tensor(v, device=device, dtype=dtype)
            for k, v in asdict(self).items()
        }

    def to_ndarrays(self) -> dict[str, NDArray]:
        return {k: np.array(v) for k, v in asdict(self).items()}
