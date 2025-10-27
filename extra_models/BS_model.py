import numpy as np
from extra_models.extended_model_base import (
    ExtendedLensModel,
    load_and_interpolate_mass_function,
)

# Load and interpolate Boson Star mass function
m_boson_inner, dm_boson_inner = load_and_interpolate_mass_function(
    "extra_models/data/mt_boson_list.csv"
)


def m_boson(t, t_m):
    """Boson Star mass distribution function."""
    return np.where(np.abs(t) < t_m, m_boson_inner(t / t_m), 1)


def dm_boson(t, t_m):
    """Derivative of Boson Star mass distribution."""
    return np.where(np.abs(t) < t_m, dm_boson_inner(t / t_m) / t_m, 0)


class BSmodel(ExtendedLensModel):
    """Boson Star lensing model."""

    @property
    def model_mass_functions(self):
        """Return mass function tuple for Boson Star model."""
        return (m_boson, dm_boson)

    @property
    def _model_type(self):
        """Return model type identifier."""
        return "BS"
