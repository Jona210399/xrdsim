import json
import os

SCALED_INTENSITY_TOL: float = 0.001
TWO_THETA_TOL: float = 1e-05
ATOMIC_SCATTERING_PARAMS_FILE: str = "atomic_scattering_params.json"

DEFAULT_WAVELENGTH: float = 1.5406
DEFAULT_SHAPEFACTOR: float = 0.9
DEFAULT_CRYSTALLITE_SIZE_RANGE: tuple[float, float] = (200, 1000)
DEFAULT_ANGLE_RANGE: tuple[float, float, int] = (5, 90, 8501)
DEFAULT_RESCALE_INTENSITY: bool = True

with open(
    os.path.join(os.path.dirname(__file__), ATOMIC_SCATTERING_PARAMS_FILE)
) as file:
    ATOMIC_SCATTERING_PARAMS = json.load(file)
