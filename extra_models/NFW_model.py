import numpy as np
from extra_models.extended_model_base import (
    ExtendedLensModel,
    load_and_interpolate_mass_function,
)

# Load and interpolate NFW mass function
# NFW uses drop_row=0 to handle the mirrored data differently than Boson Star
m_nfw_inner, dm_nfw_inner = load_and_interpolate_mass_function(
    "extra_models/data/mt_nfw_list.csv", drop_row=0
)


def m_nfw(t, t_m):
    """NFW mass distribution function."""
    return np.where(np.abs(t) < t_m, m_nfw_inner(t / t_m), 1)


def dm_nfw(t, t_m):
    """Derivative of NFW mass distribution."""
    return np.where(np.abs(t) < t_m, dm_nfw_inner(t / t_m) / t_m, 0)


class NFWmodel(ExtendedLensModel):
    """NFW (Navarro-Frenk-White) lensing model."""

    @property
    def model_mass_functions(self):
        """Return mass function tuple for NFW model."""
        return (m_nfw, dm_nfw)

    @property
    def _model_type(self):
        """Return model type identifier."""
        return "NFW"
