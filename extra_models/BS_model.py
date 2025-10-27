import numpy as np
import pandas as pd
import pyLIMA
from pyLIMA.models.PSPL_model import PSPLmodel
from pyLIMA.magnification.impact_parameter import impact_parameter
from scipy.interpolate import make_interp_spline
from scipy.optimize import root_scalar
from collections import OrderedDict

boson_mt = pd.read_csv(
    "extra_models/data/mt_boson_list.csv", header=None, names=["t", "mt"]
)
boson_mt["t"] = boson_mt["t"].apply(eval)
boson_mt_2 = boson_mt.iloc[::-1]
boson_mt_2["t"] = -1 * boson_mt_2["t"]
boson_mt_2.loc[len(boson_mt_2)] = [0.0, 0.0]
boson_mt_both = pd.concat([boson_mt_2, boson_mt], axis=0, ignore_index=True)

m_boson_inner = make_interp_spline(
    boson_mt_both["t"].values, boson_mt_both["mt"].values
)


def m_boson(t, t_m):
    return np.where(np.abs(t) < t_m, m_boson_inner(t / t_m), 1)


dm_boson_inner = m_boson_inner.derivative()


def dm_boson(t, t_m):
    return np.where(np.abs(t) < t_m, dm_boson_inner(t / t_m) / t_m, 0)


def magnification_BS(tau, beta, t_m, m, dm, return_impact_parameter=False):
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


class BSmodel(PSPLmodel):
    def paczynski_model_parameters(self):
        """
        [t0, u0, tE, t_m]
        Boson star model parameters: standard PSPL + boson mass parameter
        """
        model_dictionary = {'t0': 0, 'u0': 1, 'tE': 2, 't_m': 3}
        self.Jacobian_flag = 'Analytical'

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
            sorted(model_dictionnary_updated.items(), key=lambda x: x[1]))

        # Manually set boundaries for all parameters including t_m
        t0_bounds = (2400000, 2500000)
        u0_bounds = (-1.0, 1.0)
        tE_bounds = (0.1, 500)
        tm_bounds = (0.1, 10.0)  # t_m bounds for boson star parameter
        piEN_bounds = (-0.5, 0.5)
        piEE_bounds = (-0.5, 0.5)

        self.standard_parameters_boundaries = []
        for param_name, param_index in sorted(self.pyLIMA_standards_dictionnary.items(), key=lambda x: x[1]):
            if param_name == 't0':
                self.standard_parameters_boundaries.append(t0_bounds)
            elif param_name == 'u0':
                self.standard_parameters_boundaries.append(u0_bounds)
            elif param_name == 'tE':
                self.standard_parameters_boundaries.append(tE_bounds)
            elif param_name == 't_m':
                self.standard_parameters_boundaries.append(tm_bounds)
            elif param_name == 'piEN':
                self.standard_parameters_boundaries.append(piEN_bounds)
            elif param_name == 'piEE':
                self.standard_parameters_boundaries.append(piEE_bounds)
            else:
                # For flux parameters and others, use default bounds
                self.standard_parameters_boundaries.append((0, np.inf))

    def model_type(self):
        return "BS"

    def model_magnification(
        self, telescope, pyLIMA_parameters, return_impact_parameter=False
    ):

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

            source1_magnification = magnification_BS(
                source1_trajectory_x,
                source1_trajectory_y,
                t_m=pyLIMA_parameters["t_m"],
                m=m_boson,
                dm=dm_boson,
            )

            if source2_trajectory_x is not None:

                source2_magnification = magnification_BS(
                    source2_trajectory_x,
                    source2_trajectory_y,
                    t_m=pyLIMA_parameters["t_m"],
                    m=m_boson,
                    dm=dm_boson,
                )

                blend_magnification_factor = pyLIMA_parameters[
                    "q_flux_" + telescope.filter
                ]
                effective_magnification = (
                    source1_magnification
                    + source2_magnification * blend_magnification_factor
                )

                magnification = effective_magnification
                # breakpoint()
            else:

                magnification = source1_magnification

        else:

            magnification = None

        return magnification
