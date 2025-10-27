"""
Base module for extended lens models (Boson Star, NFW, etc.).

This module provides shared functionality for models that use extended mass
distributions, reducing code duplication across different model implementations.
"""

import numpy as np
import pandas as pd
from collections import OrderedDict
from scipy.interpolate import make_interp_spline
from scipy.optimize import root_scalar
from pyLIMA.magnification.impact_parameter import impact_parameter
from pyLIMA.models.PSPL_model import PSPLmodel


def load_and_interpolate_mass_function(csv_path, drop_row=None):
    """
    Load mass function data from CSV and create interpolation splines.

    Parameters
    ----------
    csv_path : str
        Path to CSV file with columns 't' and 'mt'
    drop_row : int, optional
        Row index to drop after reversing (used for NFW model)

    Returns
    -------
    tuple
        (m_inner_spline, dm_inner_spline) - interpolation and its derivative
    """
    # Load data
    mt_data = pd.read_csv(csv_path, header=None, names=["t", "mt"])
    mt_data["t"] = mt_data["t"].apply(eval)

    # Mirror the data around t=0
    mt_data_2 = mt_data.iloc[::-1]
    mt_data_2["t"] = -1 * mt_data_2["t"]

    # Handle model-specific adjustments
    if drop_row is not None:
        mt_data_2 = mt_data_2.drop(drop_row, axis=0)
    else:
        # For Boson Star: add [0.0, 0.0] row
        mt_data_2.loc[len(mt_data_2)] = [0.0, 0.0]

    # Combine mirrored and original data
    mt_both = pd.concat([mt_data_2, mt_data], axis=0, ignore_index=True)

    # Create spline interpolations
    m_inner = make_interp_spline(mt_both["t"].values, mt_both["mt"].values)
    dm_inner = m_inner.derivative()

    return m_inner, dm_inner


def calculate_magnification(tau, beta, t_m, m, dm, return_impact_parameter=False):
    """
    Calculate magnification for extended lens models.

    This function computes the total magnification by finding image positions
    through the lens equation and summing magnifications over all images.

    Parameters
    ----------
    tau : array-like
        x-coordinates of the source trajectory
    beta : array-like
        y-coordinates of the source trajectory
    t_m : float
        Scale radius of the extended mass distribution
    m : callable
        Mass distribution function m(t, t_m)
    dm : callable
        Derivative of mass distribution dm(t, t_m)
    return_impact_parameter : bool, optional
        Whether to return impact parameter (default: False)

    Returns
    -------
    array
        Total magnification values for each observation time
    """

    def find_roots(u_t, t_m):
        """Find roots of the lens equation for extended objects."""
        _t = np.arange(-1e2, 1e2, 1e-1)

        # Mask out _t = 0 to prevent division by zero
        mask_nonzero = _t != 0
        _t = _t[mask_nonzero]

        _m = m(_t, t_m)
        _f = -u_t + _t - _m / _t

        mask_sign_change = _f[:-1] * _f[1:] < 0
        indices_of_changed_sign = np.nonzero(mask_sign_change)[0]

        roots = []
        for idx in indices_of_changed_sign:
            _sol = root_scalar(
                lambda t: -u_t + t - m(t, t_m) / t, bracket=[_t[idx], _t[idx + 1]]
            )
            roots.append(_sol.root)
        return np.array(roots)

    def mu_function(t, t_m):
        """Calculate magnification for a single image position."""
        return 1 / (
            np.abs(1 - m(t, t_m) / (t**2))
            * np.abs(1 + m(t, t_m) / (t**2) - dm(t, t_m) / t)
        )

    u_t = impact_parameter(tau, beta)

    # Find all image positions for each observation time
    utt = []
    for ut in u_t:
        sol = find_roots(ut, t_m)
        utt.append([ut, sol])

    # Calculate total magnification as sum over all images
    magnification = []
    for ut, t in utt:
        mus = []
        for tt in t:
            mu = mu_function(tt, t_m)
            mus.append(mu)
        magnification.append(np.sum(np.abs(mus)))
    magnification = np.array(magnification)

    return magnification


class ExtendedLensModel(PSPLmodel):
    """
    Base class for extended lens models (e.g., Boson Star, NFW).

    This class provides common functionality for models that extend the
    Point Source Point Lens (PSPL) model with an additional parameter
    for extended mass distribution.

    Subclasses should define:
    - model_mass_functions: property returning (m_func, dm_func) tuple
    - _model_type: str attribute for model identification
    """

    @property
    def model_mass_functions(self):
        """Return tuple of (m_func, dm_func) for this model.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement model_mass_functions")

    @property
    def _model_type(self):
        """Return model type identifier (e.g., 'BS', 'NFW').

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _model_type")

    def paczynski_model_parameters(self):
        """
        [t0, u0, tE, t_m]
        Extended model parameters: standard PSPL + mass scale parameter
        """
        model_dictionary = {"t0": 0, "u0": 1, "tE": 2, "t_m": 3}
        self.Jacobian_flag = "Analytical"
        return model_dictionary

    def define_pyLIMA_standard_parameters(self):
        """
        Override to handle t_m parameter boundaries.
        """
        # Build the model dictionary
        model_dictionnary = self.paczynski_model_parameters()
        model_dictionnary_updated = self.astrometric_model_parameters(model_dictionnary)
        self.second_order_model_parameters(model_dictionnary_updated)
        self.telescopes_fluxes_model_parameters(model_dictionnary_updated)

        self.pyLIMA_standards_dictionnary = OrderedDict(
            sorted(model_dictionnary_updated.items(), key=lambda x: x[1])
        )

        # Set parameter boundaries
        t0_bounds = (2400000, 2500000)
        u0_bounds = (-1.0, 1.0)
        tE_bounds = (0.1, 500)
        tm_bounds = (0.1, 10.0)  # t_m bounds for extended mass parameter
        piEN_bounds = (-0.5, 0.5)
        piEE_bounds = (-0.5, 0.5)

        self.standard_parameters_boundaries = []
        for param_name, param_index in sorted(
            self.pyLIMA_standards_dictionnary.items(), key=lambda x: x[1]
        ):
            if param_name == "t0":
                self.standard_parameters_boundaries.append(t0_bounds)
            elif param_name == "u0":
                self.standard_parameters_boundaries.append(u0_bounds)
            elif param_name == "tE":
                self.standard_parameters_boundaries.append(tE_bounds)
            elif param_name == "t_m":
                self.standard_parameters_boundaries.append(tm_bounds)
            elif param_name == "piEN":
                self.standard_parameters_boundaries.append(piEN_bounds)
            elif param_name == "piEE":
                self.standard_parameters_boundaries.append(piEE_bounds)
            else:
                # For flux parameters and others, use default bounds
                self.standard_parameters_boundaries.append((0, np.inf))

    def model_type(self):
        """Return model type identifier."""
        return self._model_type

    def model_magnification(
        self, telescope, pyLIMA_parameters, return_impact_parameter=False
    ):
        """
        Calculate magnification for extended lens model.

        Handles both single and binary source systems with blending.
        """
        if telescope.lightcurve is not None:
            (
                source1_trajectory_x,
                source1_trajectory_y,
                source2_trajectory_x,
                source2_trajectory_y,
                dseparation,
                dalpha,
            ) = self.sources_trajectory(
                telescope, pyLIMA_parameters, data_type="photometry"
            )

            m_func, dm_func = self.model_mass_functions

            source1_magnification = calculate_magnification(
                source1_trajectory_x,
                source1_trajectory_y,
                t_m=pyLIMA_parameters["t_m"],
                m=m_func,
                dm=dm_func,
            )

            if source2_trajectory_x is not None:
                source2_magnification = calculate_magnification(
                    source2_trajectory_x,
                    source2_trajectory_y,
                    t_m=pyLIMA_parameters["t_m"],
                    m=m_func,
                    dm=dm_func,
                )

                blend_magnification_factor = pyLIMA_parameters[
                    "q_flux_" + telescope.filter
                ]
                effective_magnification = (
                    source1_magnification
                    + source2_magnification * blend_magnification_factor
                )
                magnification = effective_magnification
            else:
                magnification = source1_magnification
        else:
            magnification = None

        return magnification
