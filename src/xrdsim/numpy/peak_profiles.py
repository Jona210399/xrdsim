from typing import Any

import numba
import numpy as np
from numpy.typing import NDArray

from xrdsim.numpy.crystallite_size import CrystalliteSizeProvider


class PeaksOnlyProfile:
    def __init__(self):
        pass

    @staticmethod
    def convolute_peaks(peaks_x: NDArray, peaks_y: NDArray) -> tuple[NDArray, NDArray]:
        return peaks_x, peaks_y

    def get_constant_metadata(self):
        return {"profile": "peaks_only"}

    def get_metadata(self):
        return {}


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

    def get_constant_metadata(self):
        return {
            "profile": "gaussian",
            "x_range": self.x_range,
        }

    def get_metadata(self):
        return {}

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

        return xs, ys


class GaussianScherrerProfile:
    def __init__(
        self,
        gaussian_profile: GaussianProfile,
        shape_factor: float,
        wavelength: float,
        crystallite_size_provider: CrystalliteSizeProvider,
    ) -> None:
        self.gaussian_profile = gaussian_profile
        self.shape_factor = shape_factor
        self.wavelength = wavelength
        self.crystallite_size_provider = crystallite_size_provider

    def convolute_peaks(
        self,
        peaks_x: NDArray,
        peaks_y: NDArray,
    ):
        crystallite_size = self.crystallite_size_provider.get_crystallite_size()
        self.crystallite_size = crystallite_size

        sigmas = self.scherrer_equation(
            peaks_x,
            crystallite_size,
            self.shape_factor,
            self.wavelength,
        )

        return self.gaussian_profile.convolute_peaks(peaks_x, peaks_y, sigmas)

    def get_metadata(self) -> dict[str, Any]:
        return {
            **self.gaussian_profile.get_metadata(),
            **{"crystallite_size": self.crystallite_size},
        }

    def get_constant_metadata(self) -> dict[str, Any]:
        return {
            **self.gaussian_profile.get_constant_metadata(),
            **{
                "profile": "gaussian_scherrer",
                "shape_factor": self.shape_factor,
                "wavelength": self.wavelength,
            },
        }

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


class LorentzianProfile:
    def __init__(self, x_range: tuple[float, float, int]):
        self.x_range = x_range

    def convolute_peaks(
        self,
        peaks_x: NDArray,
        peaks_y: NDArray,
        gammas: NDArray,
    ) -> NDArray:
        return self._convolute_peaks(peaks_x, peaks_y, gammas, self.x_range)

    def get_metadata(self):
        return {}

    def get_constant_metadata(self):
        return {
            "profile": "lorentzian",
            "x_range": self.x_range,
        }

    @staticmethod
    @numba.njit(cache=True)
    def _convolute_peaks(
        peaks_x: NDArray,
        peaks_y: NDArray,
        gammas: NDArray,
        x_range: tuple[float, float, int],
    ) -> NDArray:
        x_values = np.linspace(*x_range)
        profile = np.zeros_like(x_values)

        for x0, y0, gamma in zip(peaks_x, peaks_y, gammas):
            profile += LorentzianProfile.lorentzian(x_values, x0, y0, gamma)

        return profile

    @staticmethod
    def lorentzian(
        x: NDArray,
        x0: float,
        y0: float,
        gamma: float,
    ) -> NDArray:
        return y0 / (1 + ((x - x0) / gamma) ** 2)


class LorentzianStrainProfile:
    def __init__(
        self,
        lorentzian_profile: LorentzianProfile,
        strain_sampler: Any,
    ) -> None:
        self.lorentzian_profile = lorentzian_profile
        self.strain_sampler = strain_sampler

    def convolute_peaks(
        self,
        peaks_x: NDArray,
        peaks_y: NDArray,
    ) -> NDArray:
        strain = self.strain_sampler.sample()
        self.strain = strain

        fwhms = self.strain_broadening(
            peaks_x,
            strain,
        )
        gammas = fwhms / 2

        return self.lorentzian_profile.convolute_peaks(peaks_x, peaks_y, gammas)

    def get_metadata(self) -> dict[str, Any]:
        return {
            **self.lorentzian_profile.get_metadata(),
            **{"strain": self.strain},
        }

    def get_constant_metadata(self) -> dict[str, Any]:
        return {
            **self.lorentzian_profile.get_constant_metadata(),
            **{"profile": "lorentzian_strain"},
        }

    @staticmethod
    @numba.njit(cache=True)
    def strain_broadening(
        peak_two_thetas: np.ndarray,
        strain: float,
    ) -> np.ndarray:
        thetas: np.ndarray = np.radians(peak_two_thetas) / 2

        # Strain broadening formula: β_L = 4 * strain * tan(θ)
        fwhms: np.ndarray = 4 * strain * np.tan(thetas)
        return np.degrees(fwhms)


class PseudoVoigtProfile:
    def __init__(
        self,
        gaussian_profile: GaussianProfile,
        lorentzian_profile: LorentzianProfile,
        mixing_factor: float,
    ) -> None:
        if not (0 <= mixing_factor <= 1):
            raise ValueError("Mixing factor must be between 0 and 1.")

        self.gaussian_profile = gaussian_profile
        self.lorentzian_profile = lorentzian_profile
        self.mixing_factor = mixing_factor

    def convolute_peaks(
        self,
        peaks_x: NDArray,
        peaks_y: NDArray,
        gaussian_sigmas: NDArray,
        lorentzian_fwhms: NDArray,
    ) -> NDArray:
        gaussian_profile = self.gaussian_profile.convolute_peaks(
            peaks_x, peaks_y, gaussian_sigmas
        )
        lorentzian_profile = self.lorentzian_profile.convolute_peaks(
            peaks_x, peaks_y, lorentzian_fwhms
        )

        pseudo_voigt_profile = (
            self.mixing_factor * lorentzian_profile
            + (1 - self.mixing_factor) * gaussian_profile
        )

        return pseudo_voigt_profile

    def get_metadata(self) -> dict[str, Any]:
        return {
            **self.gaussian_profile.get_metadata(),
            **self.lorentzian_profile.get_metadata(),
        }

    def get_constant_metadata(self) -> dict[str, Any]:
        return {
            **self.gaussian_profile.get_constant_metadata(),
            **self.lorentzian_profile.get_constant_metadata(),
            **{
                "profile": "pseudo_voigt",
                "mixing_factor": self.mixing_factor,
            },
        }


class PseudoVoigtCagliotiProfile:
    def __init__(
        self,
        pseudo_voigt: PseudoVoigtProfile,
    ) -> None:
        self.pseudo_voigt = pseudo_voigt

    def convolute_peaks(
        self,
        peaks_x: NDArray,
        peaks_y: NDArray,
        caglioti_params: tuple[float, float, float],
    ) -> NDArray:
        fwhms_squared = self.caglioti_equation(peaks_x, *caglioti_params)
        fwhms = np.sqrt(fwhms_squared)

        gaussian_sigmas = fwhms / (2 * np.sqrt(2 * np.log(2)))
        lorentzian_gammas = fwhms / 2

        pseudo_voigt_profile = self.pseudo_voigt.convolute_peaks(
            peaks_x,
            peaks_y,
            gaussian_sigmas,
            lorentzian_gammas,
        )

        return pseudo_voigt_profile

    def get_metadata(self) -> dict[str, Any]:
        return {**self.pseudo_voigt.get_metadata()}

    def get_constant_metadata(self) -> dict[str, Any]:
        return {
            **self.pseudo_voigt.get_constant_metadata(),
            **{"profile": "pseudo_voigt_caglioti"},
        }

    @staticmethod
    @numba.njit(cache=True)
    def caglioti_equation(
        peak_two_thetas: np.ndarray,
        U: float,
        V: float,
        W: float,
    ) -> np.ndarray:
        thetas = np.radians(peak_two_thetas) / 2
        tan_thetas = np.tan(thetas)

        return U * tan_thetas**2 + V * tan_thetas + W
