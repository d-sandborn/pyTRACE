import numpy as np
import warnings
from seawater import satO2, ptmp, dens, pres
from scipy.stats import invgauss
from scipy.spatial import Delaunay


def equation_check(equation):
    """Check equation inputs and assigns them to be [1] regardless.
    Largely a carry-over from ESPER, which accepted more options."""
    match equation:
        case []:
            equation = [1]
        case [0]:
            equation = [1]
        case [1]:
            equation = [1]
        case _:
            warnings.warn(
                "Input 'equations' could not be parsed. Setting to [1]."
            )
            equation = [1]
    return equation


def units_check(per_kg_sw_tf):
    """Check for per_kg_sw_tf input and setting default if not given.
    This input is not needed for TRACE, currently"""
    # Checking for per_kg_sw_tf input and setting default if not given. This
    # input is not needed for TRACE, currently
    if not per_kg_sw_tf:
        warnings.warn(
            "Optional argument per_kg_sw_tf is not in use. Setting to True."
        )
        per_kg_sw_tf = True
    return per_kg_sw_tf


def preindustrial_check(preindustrial_xco2):
    """Check for preindustrial_xco2 input and setting default if not given"""
    if not isinstance(preindustrial_xco2, float) and not isinstance(
        preindustrial_xco2, int
    ):
        warnings.warn(
            "preindustrial_xco2 could not be parsed as str or float. Setting to 280."
        )
        preindustrial_xco2 = 280
    return preindustrial_xco2


def uncerts_check(meas_uncerts, predictor_measurements, predictor_types):
    """Checks the meas_uncerts argument.  This also deals with the
    possibility that the user has provided a single set of uncertainties
    for all estimates."""
    if meas_uncerts is not None:
        use_default_uncertainties = False

        if (
            (not np.max(np.shape(meas_uncerts)) == len(predictor_measurements))
            and not np.min(
                np.shape(meas_uncerts) == len(predictor_measurements)
            )
            and not np.max(np.size(meas_uncerts)) == 0
        ):
            warnings.warn(
                "meas_uncerts must be undefined, a vector with the same number of elements as predictor_measurements has columns, [] (for default values), or an array of identical dimension to predictor_measurements. Setting meas_uncerts to None."
            )
            meas_uncerts == None
        elif (
            not np.min(np.shape(meas_uncerts) == len(predictor_measurements))
            and not np.max(np.shape(meas_uncerts)) == 0
        ):
            if not np.shape(meas_uncerts)[1] == len(predictor_measurements):
                warnings.warn(
                    "There are different numbers of columns of input uncertainties and input measurements."
                )
            input_u = (
                np.ones(len(predictor_measurements)) * meas_uncerts
            )  # Copying uncertainty estimates for all estimates if only singular values are provided
            # not sure that works as intended
        if not np.shape(predictor_types)[1] == len(predictor_measurements):
            raise ValueError(
                "The predictor_types input does not have the same number of columns as the predictor_measurements input.  This means it is unclear which measurement is in which column."
            )
    else:
        use_default_uncertainties = True
        input_u = None

    return meas_uncerts, input_u, use_default_uncertainties


def depth_check(output_coordinates, valid_indices):
    """This step checks for negative depths.  If found, it changes them to
    positive depths and issues a warning."""
    if np.any(output_coordinates[valid_indices, 2] < 0):
        warnings.warn(
            "Negative depths were detected and changed to positive values."
        )
        output_coordinates[valid_indices, 2] = np.abs(
            output_coordinates[valid_indices, 2]
        )
    return output_coordinates


def coordinate_check(output_coordinates, valid_indices):
    """Book-keeping with coordinate inputs and adjusting negative
    longitudes."""
    if np.any(np.abs(output_coordinates[:, 1]) > 90):
        raise ValueError(
            "A latitude >90 degrees (N or S) has been detected.  Verify latitude is in the 2nd colum of the coordinate input."
        )
    C = output_coordinates[valid_indices, :].copy()
    C[:, 0] = np.mod(C[:, 0], 360)
    C[C[:, 0] < 0, 0] += 360
    return output_coordinates, C


def prepare_uncertainties(
    predictor_measurements,
    predictor_types,
    valid_indices,
    use_default_uncertainties,
    input_u,
):
    """Preparing full predictor_measurement uncertainty grid.
    Maybe vestigial??"""
    default_uncertainties = np.diag([1, 1, 0.02, 0.02, 0.02, 0.01])
    default_u_all = np.zeros([len(predictor_measurements), 6])
    default_u_all[:, predictor_types - 1] = (
        predictor_measurements
        * default_uncertainties[predictor_types - 1, predictor_types - 1]
    )  # Setting multiplicative default uncertainties for P, N, O2, and Si.
    default_u_all[:, np.argwhere(predictor_types == 1)] = (
        0.003  # Then setting additive default uncertainties for T
    )
    default_u_all[:, np.argwhere(predictor_types == 2)] = (
        0.003  # Then setting additive default uncertainties for S
    )
    default_u_all = default_u_all[valid_indices, :]
    input_u_all = default_u_all  # [valid_indices, :]
    if not use_default_uncertainties:  # if user supplied uncertainties
        # input_u_all = np.zeros_like(predictor_measurements)
        input_u_all[:, predictor_types - 1] = input_u
        input_u_all = input_u_all[valid_indices, :]
        # Overriding user provided uncertainties that are smaller than the
        # (minimum) default uncertainties
        input_u_all = np.max(np.stack([input_u_all, default_u_all]), 0)
    return default_u_all, input_u_all


def inverse_gaussian_wrapper(x, gamma=1, delta=1.3):
    """Calculate ventilation distributions (assumed probability
    distribution). lambda should perhaps be 1/1.3 from He et al.
    Note that invgauss calls are different in pyTRACE and TRACE!
    Also note that TRACE approximates mu as 3.4 instead of ~3.38, as
    commented text below indicates."""
    nu = 3.4  # gamma #default 1
    lam = 1  # gamma**3 / 2 / delta**2 #default 0.29585798816568043
    y = invgauss.pdf(x, mu=nu / lam, scale=lam, loc=0)
    return y


def inpolygon(xq, yq, xv, yv):
    """Probably checks if points in polygon."""
    try:
        points = np.array([xq, yq]).T
        path = np.array([xv, yv]).T
        tri = Delaunay(path)
        return tri.find_simplex(points) >= 0
    except:
        return np.array([False] * len(xq))


def say_hello():
    """It's only polite."""
    print(
        """
▒▓████████▓▒░▒▓███████▓▒░C░▒▓██████▓▒░C░▒▓██████▓▒░░▒▓████████▓▒
CC░▒▓█▓▒░CCC░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░CCCCCC
CC░▒▓█▓▒░CCC░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░CCCCCC░▒▓█▓▒░CCCCCC
CC░▒▓█▓▒░CCC░▒▓███████▓▒░░▒▓████████▓▒░▒▓█▓▒░CCCCCC░▒▓██████▓▒░C
CC░▒▓█▓▒░CCC░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░CCCCCC░▒▓█▓▒░CCCCCC
CC░▒▓█▓▒░CCC░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░CCCCCC
CC░▒▓█▓▒░CCC░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░░▒▓████████▓▒
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

                         Python v0.0.1 alpha
                    
Carter, B.; Sandborn D. 2025.
MATLAB - github.com/BRCScienceProducts/TRACEv1
Python - github.com/d-sandborn/pyTRACE"""
    )
