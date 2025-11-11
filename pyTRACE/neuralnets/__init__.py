"""Functions for neural network estimation in pyTRACE."""

import numpy as np
import warnings
from tqdm import tqdm
import os
from scipy import io
from os.path import join as joinpath
import pickle
from numba import njit
from pyTRACE.utils import (
    equation_check,
    units_check,
    uncerts_check,
    coordinate_check,
    prepare_uncertainties,
    inpolygon,
)
from gsw import pt0_from_t, rho_t_exact, p_from_z, SA_from_SP

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from seawater import ptmp, dens, pres


def trace_nn(
    desired_variables,
    output_coordinates,
    predictor_measurements,
    predictor_types,
    DATADIR,
    equations=[],
    meas_uncerts=None,
    error_codes=[-999, -9, -1e20],
    per_kg_sw_tf=True,
    verbose_tf=True,
    eos="seawater",
    delta_over_gamma=None,
):
    """
    Implement ESPER NN estimation of properties for pyTRACE.

    Input/Output dimensions:
    .........................................................................
    p: Integer number of desired property estimate types (e.g., TA, pH, NO3-)
    n: Integer number of desired estimate locations
    e: Integer number of equations used at each location
    y: Integer number of parameter measurement types provided by the user.
    n*e: Total number of estimates returned as an n by e array

    Parameters
    ----------
    desired_variables: list
        List of ints. Specifies which variables will be returned.
        Outputs in umol/kg for 1-5.
        1. pTA (preformed total titration seawater alkalinity)
        2. pP (preformed phosphate)
        3. pN (preformed nitrate)
        4. pSi (prefomred silicate)
        5. pO (preformed dissolved molecular oxygen,i.e., O2<aq>)
        6. SF (scale factors for age distribution)
        7. EstT (temperature estimated from S, coordinates)
    output_coordinates : numpy.ndarray
        n by 3 array of coordinates (longitude degrees E, latitude
        degrees N, depth m) at which estimates are desired.
    predictor_measurements : numpy.ndarray
        n by y array of parameter measurements (salinity, temperature,
        etc.) The column order (y columns) is arbitrary, but specified
        by predictor_types. Temperature should be expressed as degrees
        C and salinity should be specified with the unitless convention.
        nan inputs are acceptable, but will lead to nan estimates for
        any equations that depend on that parameter. If temperature is
        not provided then it will be estimated from salinity.
    predictor_types : numpy.array
        1 by y array indicating which
        parameter is in each column of 'predictor_measurements'.
        Note that salinity is required for all equations. Input
        parameter key:
            1. Salinity
            2. Temperature
    equations : list, optional
        Indicates which predictors will be used to estimate properties.
        This input should always be omitted because there is only
        one possible equation, but this functionality is retained in the
        code to allow for eventual generalization of the TRACE NN to
        operate with more predictor combinations.
        (S=salinity, Theta=potential temperature, N=nitrate,
         Si=silicate, T=temperature, O2=dissolved oxygen molecule...
         see 'predictor_measurements' for units).
        Options:
            1.  S, T
        The default is [1].
    meas_uncerts : list, optional
        List of measurement uncertainties presented in order indicated
        by 'predictor_types'. Providing these estimates will improve
        estimate uncertainties in 'UncertaintyEstimates'. Measurement
        uncertainties are a small part of TRACE estimate uncertainties
        for WOCE-quality measurements. However, estimate uncertainty
        scales with measurement uncertainty, so it is recommended that
        measurement uncertainties be specified for sensor measurements.
        If this optional input argument is not provided, the default
        WOCE-quality uncertainty is assumed.  If a list is provided
        then the uncertainty estimates are assumed to apply uniformly
        to all input parameter measurements.
        The default is None.
    error_codes : list, optional
        Error codes to be parsed as np.nan in input parameter arrays.
        The default is [-999, -9, -1e20].
    per_kg_sw_tf : bool, optional
        Retained for future development (allowing for flexible units
        for currently-unsupported predictors). The default is True.
    verbose_tf : bool, optional
        Flag to control output verbosity. Setting this to False will
        make TRACE stop printing updates to the command line.  Warnings
        and errors, if any, will be given regardless.
        The default is True.
    eos: str, optional
        Choice of seawater equation of state to use for potential
        temperature, density, and depth conversions. Available choices
        are 'seawater' (EOS-80) and 'gsw' (TEOS-10). 'seawater' will
        be deprecated, but is kept for compatibility with TRACEv1.
        The default is 'seawater'.

    Raises
    ------
    ValueError
        Input parameter issues reported to the user.
    FileNotFoundError
        File issues reported to the user.

    Returns
    -------
    Estimates: dict
        Dictionary of parameters estimated from temperature, salinity,
        or depth.

    """
    equations = equation_check(equations)
    per_kg_sw_tf = units_check(per_kg_sw_tf)
    (
        meas_uncerts,
        input_u,
        use_default_uncertainties,
        predictor_measurements,
        predictor_types,
    ) = uncerts_check(meas_uncerts, predictor_measurements, predictor_types)
    valid_indices = ~np.isnan(output_coordinates).any(axis=1)
    valid_indices = np.argwhere(valid_indices > 0)[:, 0]

    n = len(valid_indices)
    p = len(desired_variables)

    for i in error_codes:
        if i in predictor_measurements:
            warnings.warn(
                "A common non-NaN missing data indicator (e.g. -999, -9, -1e20) was detected in the input measurements provided.  Missing data should be replaced with np.nan, otherwise, PyTRACE will interpret your inputs at face value and give terrible estimates."
            )

    output_coordinates, C = coordinate_check(output_coordinates, valid_indices)
    default_u_all, input_u_all = prepare_uncertainties(
        predictor_measurements,
        predictor_types,
        valid_indices,
        use_default_uncertainties,
        input_u,
    )

    if len(predictor_types) != predictor_measurements.shape[1]:
        raise ValueError(
            "predictor_types and predictor_measurements must have the same number of columns."
        )
    m_all = np.full((n, 6), np.nan)
    u_all = np.full((n, 6), np.nan)
    m_all[:, predictor_types - 1] = predictor_measurements[valid_indices, :]
    u_all[:, predictor_types - 1] = input_u_all[:, predictor_types]
    vname_dict = {
        1: "Preformed_TA",
        2: "Preformed_P",
        3: "Preformed_N",
        4: "Preformed_Si",
        5: "Preformed_O",
        6: "SFs",
        7: "Temperature",
    }
    Estimates = {
        vname_dict[name]: np.full(n, np.nan) for name in desired_variables
    }
    for i in tqdm(range(p), disable=(not verbose_tf), desc="Pref Props"):
        prop = desired_variables[i]
        have_vars = [False] * 6
        match prop:
            case 1 | 2 | 3 | 4 | 5:  # preformed properties
                needed_for_property = [0, 1, 2, 5, 4]
                need_vars = np.array([True, True, True, False, False])
            case 6:  # SF
                needed_for_property = [0, 1, 2, 3, 4, 5]
                need_vars = np.array([True, True, True, False, False, False])
            case 7:  # EstT
                needed_for_property = [0, 1, 2, 5, 4]
                need_vars = np.array([True, True, False, False, False])

        for k in predictor_types:
            have_vars[k] = True

        # Adding one because depth is provided
        have_vars[0] = True
        # Making sure all needed variables are present
        if not per_kg_sw_tf:
            need_vars[1] = True
        if (  # there are definitely cleaner ways to do this
            not all([have_vars[k] for k in np.argwhere(need_vars)[:, 0]])
            and verbose_tf
        ):
            warnings.warn(
                "One or more regression equations for the current property require one or more input parameters that are either not provided or not labeled correctly. These equations will return NaN for all estimates.  All 16 equations are used by default unless specific equations are specified.  Temperature is also required when density or carbonate system calculations are called."
            )  # They should already be NaN
        # Limiting measurements to the subset needed for this property estimate
        m = m_all[:, needed_for_property]

        if need_vars[1]:
            if eos == "seawater":
                m[:, 1] = ptmp(m[:, 0], m[:, 1], pres(C[:, 2], C[:, 1]), 0)
            else:
                m[:, 1] = pt0_from_t(
                    SA_from_SP(
                        m[:, 0], p_from_z(C[:, 2], C[:, 1]), C[:, 0], C[:, 1]
                    ),
                    m[:, 1],
                    p_from_z(C[:, 2], C[:, 1]),
                )

        # Converting units to molality if they are provided as molarity.
        if not per_kg_sw_tf:
            if eos == "seawater":
                densities = (
                    dens(m[:, 0], m[:, 1], pres(C[:, 2], C[:, 1])) / 1000
                )
            else:
                densities = (
                    rho_t_exact(  # avoid calculating CT
                        SA_from_SP(
                            m[:, 0],
                            p_from_z(C[:, 2], C[:, 1]),
                            C[:, 0],
                            C[:, 1],
                        ),
                        m[:, 1],
                        p_from_z(C[:, 2], C[:, 1]),
                    )
                    / 1000
                )
            m[:, 2] = m[:, 2] / densities
            m[:, 3] = m[:, 3] / densities
            m[:, 4] = m[:, 4] / densities

        match prop:
            case 1:
                VName = "Preformed_TA"
            case 2:
                VName = "Preformed_P"
            case 3:
                VName = "Preformed_N"
            case 4:
                VName = "Preformed_Si"
            case 5:
                VName = "Preformed_O"
            case 6:  # SF
                VName = "SFs"
            case 7:  # EstT
                VName = "Temperature"
            case _:
                raise ValueError(
                    "A property identifier >8 or <1 was supplied, but this routine only has 2 possible property estimates.  The property identifier is the first input."
                )
        # Loading the data, with an error message if not found
        fn = joinpath(DATADIR, "Polys.mat")
        # Making sure you downloaded the needed file and put it somewhere it
        # can be found
        if not os.path.isfile(fn):
            raise FileNotFoundError(
                "TRACE could not find Polys.mat.  These mandatory file(s) should be distributed from the same website as TRACE.  Contact the corresponding author if you cannot find it there.  "
            )
        L = io.loadmat(fn)

        output_estimates = np.full((output_coordinates.shape[0]), np.nan)
        m = np.concatenate([C[:, 2][:, None], m[:, :]], axis=1)

        est_atl = np.full((C.shape[0], 4), np.nan)
        est_other = np.full((C.shape[0], 4), np.nan)

        Equation = 1  # equations[eq]
        if all([have_vars[k] for k in np.argwhere(need_vars)[:, 0]]):
            P = np.hstack(
                (
                    np.cos(np.radians(C[:, 0] - 20)).reshape(-1, 1),
                    np.sin(np.radians(C[:, 0] - 20)).reshape(-1, 1),
                    C[:, 1].reshape(-1, 1),
                    m[:, 0:3].reshape(m.shape[0], -1),
                )
            )
            for Net in tqdm(
                range(4), disable=(not verbose_tf), leave=False, desc="Nets"
            ):
                # Separate neural networks are used for the Arctic/Atlantic and
                # the rest of the ocean.
                """
                est_other[:, Net] = execute_nn(
                    P,
                    VName,
                    "Other",
                    Equation,
                    Net,
                    DATADIR,
                    verbose_tf=verbose_tf,
                )
                est_atl[:, Net] = execute_nn(
                    P,
                    VName,
                    "Atl",
                    Equation,
                    Net,
                    DATADIR,
                    verbose_tf=verbose_tf,
                )
                """
                est_atl[:, Net], est_other[:, Net] = (
                    execute_nn_combined_vectorized(
                        P,
                        VName,
                        Equation,
                        Net,
                        DATADIR,
                        verbose_tf=verbose_tf,
                        delta_over_gamma=delta_over_gamma,
                    )
                )

        # Averaging across neural network committee members
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            est_atl = np.nanmean(est_atl, axis=1)
            est_other = np.nanmean(est_other, axis=1)

        # We do not want information to propagate across the Panama Canal
        # (for instance), so data is carved into two segments... the
        # Atlantic/Arctic (AA) and everything else.

        AAInds = np.logical_or(
            inpolygon(
                C[:, 0],
                C[:, 1],
                L["Polys"]["LNAPoly"][0, 0][:, 0],
                L["Polys"]["LNAPoly"][0, 0][:, 1],
            ),
            np.logical_or(
                inpolygon(
                    C[:, 0],
                    C[:, 1],
                    L["Polys"]["LSAPoly"][0, 0][:, 0],
                    L["Polys"]["LSAPoly"][0, 0][:, 1],
                ),
                np.logical_or(
                    inpolygon(
                        C[:, 0],
                        C[:, 1],
                        L["Polys"]["LNAPolyExtra"][0, 0][:, 0],
                        L["Polys"]["LNAPolyExtra"][0, 0][:, 1],
                    ),
                    np.logical_or(
                        inpolygon(
                            C[:, 0],
                            C[:, 1],
                            L["Polys"]["LSAPolyExtra"][0, 0][:, 0],
                            L["Polys"]["LSAPolyExtra"][0, 0][:, 1],
                        ),
                        inpolygon(
                            C[:, 0],
                            C[:, 1],
                            L["Polys"]["LNOPoly"][0, 0][:, 0],
                            L["Polys"]["LNOPoly"][0, 0][:, 1],
                        ),
                    ),
                ),
            ),
        )

        BeringInds = inpolygon(
            C[:, 0],
            C[:, 1],
            L["Polys"]["Bering"][0, 0][:, 0],
            L["Polys"]["Bering"][0, 0][:, 1],
        )
        SoAtlInds = np.logical_and(
            np.logical_or(C[:, 0] > 290, C[:, 0] < 20),
            np.logical_and(C[:, 1] > -44, C[:, 1] < -34),
        )
        SoAfrInds = np.logical_and(
            np.logical_and(C[:, 0] > 19, C[:, 0] < 27),
            np.logical_and(C[:, 1] > -44, C[:, 1] < -34),
        )
        Est = est_other.copy()

        if Est[AAInds].size > 0:
            Est[AAInds] = est_atl[AAInds]
        if Est[BeringInds].size > 0:
            Est[BeringInds] = est_atl[BeringInds] * (
                (C[BeringInds, 1] - 62.5) / 7.5
            ) + est_other[BeringInds] * ((70 - C[BeringInds, 1]) / 7.5)
        if Est[SoAtlInds].size > 0:
            Est[SoAtlInds] = est_atl[SoAtlInds] * (
                (C[SoAtlInds, 1] + 44) / 10
            ) + est_other[SoAtlInds] * ((-34 - C[SoAtlInds, 1]) / 10)
        if Est[SoAfrInds].size > 0:
            Est[SoAfrInds] = Est[SoAfrInds] * (
                (27 - C[SoAfrInds, 0]) / 8
            ) + est_other[SoAfrInds] * ((C[SoAfrInds, 0] - 19) / 8)
        output_estimates[valid_indices] = Est
        if VName == "SFs":
            output_estimates = 10.0**output_estimates
        Estimates[VName] = output_estimates

    return Estimates


@njit
def mapminmax_apply(x, xoffset, gain, ymin):
    """Map Minimum and Maximum Input Processing Function."""
    y = ((x - xoffset.T) * gain.T) + ymin.T
    return y


@njit
def tansig_apply(n):
    """Sigmoid Symmetric Transfer Function."""
    return 2 / (1 + np.exp(-2 * n)) - 1


@njit
def mapminmax_reverse(y, xoffset, gain, ymin):
    """Map Minimum and Maximum Output Reverse-Processing Function."""
    x = ((y - ymin.T) / gain.T) + xoffset.T
    return x


@njit
def tile_n_dot(tileable, dota, dotb):
    return tileable + np.dot(dota, dotb)


def execute_nn(X, VName, Location, Equation, Net, DATADIR, verbose_tf=True):
    """Execute neural network by calling pickle file with weights
    determined using MATLAB machine learning routines, followed by
    linear algebra replicating neural network architecture exactly."""
    with open(joinpath(DATADIR, "nn_params.pkl"), "rb") as f:
        dill = pickle.load(f)
    if VName == "Temperature":
        VName = "EstT_Temperature"
        X = X[:, 0:5]
    X = np.array(X)
    dill = dill[
        "TRACE_"
        + VName
        + "_"
        + str(Equation)
        + "_"
        + str(Location)
        + "_"
        + str(Net + 1)
    ]
    x1_step1 = dill[0]
    b1 = np.array(dill[1], dtype=np.float64)
    IW1_1 = np.array(dill[2], dtype=np.float64)
    b2 = dill[3]
    LW2_1 = np.array(dill[4], dtype=np.float64)
    b3 = np.array([[dill[5]]], dtype=np.float64)
    LW3_2 = np.array(dill[6], dtype=np.float64)
    y1_step1 = dill[7]

    x1_step1xoffset = np.array(x1_step1["xoffset"])
    x1_step1gain = np.array(x1_step1["gain"])
    x1_step1ymin = np.array(x1_step1["ymin"])

    y1_step1ymin = np.array(y1_step1["ymin"])
    y1_step1ygain = np.array(y1_step1["gain"])
    y1_step1xoffset = np.array(y1_step1["xoffset"])

    TS = len(X)
    Y = np.full(TS, np.nan)  # [None] * TS
    for ts in tqdm(
        range(TS), disable=(not verbose_tf), leave=False, desc="Locations"
    ):
        if Net == 0:
            Xp1 = mapminmax_apply(
                X[ts], x1_step1xoffset, x1_step1gain, x1_step1ymin
            )
            a1 = tansig_apply(tile_n_dot(b1, IW1_1, Xp1.T))
            a2 = tile_n_dot(np.array([[b2]]), LW2_1, a1)
            Y[ts] = mapminmax_reverse(
                a2, y1_step1xoffset, y1_step1ygain, y1_step1ymin
            )[0][0]
        else:
            Xp1 = mapminmax_apply(
                X[ts], x1_step1xoffset, x1_step1gain, x1_step1ymin
            )
            a1 = tansig_apply(tile_n_dot(b1, IW1_1, Xp1.T))
            a2 = tansig_apply(tile_n_dot(np.array(b2), LW2_1, a1))
            a3 = tile_n_dot(b3, LW3_2, a2)
            Y[ts] = mapminmax_reverse(
                a3, y1_step1xoffset, y1_step1ygain, y1_step1ymin
            )[0][0]

    return Y


def execute_nn_combined(X, VName, Equation, Net, DATADIR, verbose_tf=True):
    """
    Execute neural network by calling pickle file with weights
    determined using MATLAB machine learning routines, followed by
    linear algebra replicating neural network architecture exactly.
    """
    with open(joinpath(DATADIR, "nn_params.pkl"), "rb") as f:
        dill = pickle.load(f)
    if VName == "Temperature":
        VName = "EstT_Temperature"
        X = X[:, 0:5]
    X = np.array(X)
    dillAtl = dill[
        "TRACE_"
        + VName
        + "_"
        + str(Equation)
        + "_"
        + "Atl"
        + "_"
        + str(Net + 1)
    ]
    dillOther = dill[
        "TRACE_"
        + VName
        + "_"
        + str(Equation)
        + "_"
        + "Other"
        + "_"
        + str(Net + 1)
    ]
    x1_step1Atl = dillAtl[0]
    b1Atl = np.array(dillAtl[1], dtype=np.float64)
    IW1_1Atl = np.array(dillAtl[2], dtype=np.float64)
    b2Atl = dillAtl[3]
    LW2_1Atl = np.array(dillAtl[4], dtype=np.float64)
    b3Atl = np.array([[dillAtl[5]]], dtype=np.float64)
    LW3_2Atl = np.array(dillAtl[6], dtype=np.float64)
    y1_step1Atl = dillAtl[7]

    x1_step1xoffsetAtl = np.array(x1_step1Atl["xoffset"])
    x1_step1gainAtl = np.array(x1_step1Atl["gain"])
    x1_step1yminAtl = np.array(x1_step1Atl["ymin"])

    y1_step1yminAtl = np.array(y1_step1Atl["ymin"])
    y1_step1ygainAtl = np.array(y1_step1Atl["gain"])
    y1_step1xoffsetAtl = np.array(y1_step1Atl["xoffset"])

    x1_step1Other = dillOther[0]
    b1Other = np.array(dillOther[1], dtype=np.float64)
    IW1_1Other = np.array(dillOther[2], dtype=np.float64)
    b2Other = dillOther[3]
    LW2_1Other = np.array(dillOther[4], dtype=np.float64)
    b3Other = np.array([[dillOther[5]]], dtype=np.float64)
    LW3_2Other = np.array(dillOther[6], dtype=np.float64)
    y1_step1Other = dillOther[7]

    x1_step1xoffsetOther = np.array(x1_step1Other["xoffset"])
    x1_step1gainOther = np.array(x1_step1Other["gain"])
    x1_step1yminOther = np.array(x1_step1Other["ymin"])

    y1_step1yminOther = np.array(y1_step1Other["ymin"])
    y1_step1ygainOther = np.array(y1_step1Other["gain"])
    y1_step1xoffsetOther = np.array(y1_step1Other["xoffset"])

    TS = len(X)
    YAtl = np.full(TS, np.nan)
    YOther = np.full(TS, np.nan)
    for ts in tqdm(
        range(TS), disable=(not verbose_tf), leave=False, desc="Locations"
    ):
        if Net == 0:
            # Atl
            Xp1 = mapminmax_apply(
                X[ts], x1_step1xoffsetAtl, x1_step1gainAtl, x1_step1yminAtl
            )
            a1 = tansig_apply(tile_n_dot(b1Atl, IW1_1Atl, Xp1.T))
            a2 = tile_n_dot(np.array([[b2Atl]]), LW2_1Atl, a1)
            YAtl[ts] = mapminmax_reverse(
                a2, y1_step1xoffsetAtl, y1_step1ygainAtl, y1_step1yminAtl
            )[0][0]
            # Other
            Xp1 = mapminmax_apply(
                X[ts],
                x1_step1xoffsetOther,
                x1_step1gainOther,
                x1_step1yminOther,
            )
            a1 = tansig_apply(tile_n_dot(b1Other, IW1_1Other, Xp1.T))
            a2 = tile_n_dot(np.array([[b2Other]]), LW2_1Other, a1)
            YOther[ts] = mapminmax_reverse(
                a2, y1_step1xoffsetOther, y1_step1ygainOther, y1_step1yminOther
            )[0][0]
        else:
            # Atl
            Xp1 = mapminmax_apply(
                X[ts], x1_step1xoffsetAtl, x1_step1gainAtl, x1_step1yminAtl
            )
            a1 = tansig_apply(tile_n_dot(b1Atl, IW1_1Atl, Xp1.T))
            a2 = tansig_apply(tile_n_dot(np.array(b2Atl), LW2_1Atl, a1))
            a3 = tile_n_dot(b3Atl, LW3_2Atl, a2)
            YAtl[ts] = mapminmax_reverse(
                a3, y1_step1xoffsetAtl, y1_step1ygainAtl, y1_step1yminAtl
            )[0][0]
            # Other
            Xp1 = mapminmax_apply(
                X[ts],
                x1_step1xoffsetOther,
                x1_step1gainOther,
                x1_step1yminOther,
            )
            a1 = tansig_apply(tile_n_dot(b1Other, IW1_1Other, Xp1.T))
            a2 = tansig_apply(tile_n_dot(np.array(b2Other), LW2_1Other, a1))
            a3 = tile_n_dot(b3Other, LW3_2Other, a2)
            YOther[ts] = mapminmax_reverse(
                a3, y1_step1xoffsetOther, y1_step1ygainOther, y1_step1yminOther
            )[0][0]

    return YAtl, YOther


def execute_nn_combined_vectorized(
    X,
    VName,
    Equation,
    Net,
    DATADIR,
    verbose_tf=True,
    delta_over_gamma=None,
):
    """
    Execute neural network by calling pickle file with weights
    determined using MATLAB machine learning routines, followed by
    linear algebra replicating neural network architecture exactly.
    """
    if delta_over_gamma is None:  # default 1.3
        with open(joinpath(DATADIR, "nn_params.pkl"), "rb") as f:
            dill = pickle.load(f)
        if VName == "Temperature":
            VName = "EstT_Temperature"
            X = X[:, 0:5]
        X = np.array(X)
        dillAtl = dill[
            "TRACE_"
            + VName
            + "_"
            + str(Equation)
            + "_"
            + "Atl"
            + "_"
            + str(Net + 1)
        ]
        dillOther = dill[
            "TRACE_"
            + VName
            + "_"
            + str(Equation)
            + "_"
            + "Other"
            + "_"
            + str(Net + 1)
        ]
    else:  # user-specified, pick nearest 0.2<=D/G<=1.8
        with open(
            joinpath(DATADIR, "nn_params_variable_igttd.pkl"), "rb"
        ) as f:
            dill = pickle.load(f)
        X = np.array(X)

        dillAtl = dill[
            "TRACE_"
            + VName
            + "_"
            + str(Equation)
            + "_"
            + "Atl"
            + "_"
            + str(Net + 1)
            + "_Iter_"
            + pickle_picker(delta_over_gamma)
        ]
        dillOther = dill[
            "TRACE_"
            + VName
            + "_"
            + str(Equation)
            + "_"
            + "Other"
            + "_"
            + str(Net + 1)
            + "_Iter_"
            + pickle_picker(delta_over_gamma)
        ]
    x1_step1Atl = dillAtl[0]
    b1Atl = np.array(dillAtl[1], dtype=np.float64)
    IW1_1Atl = np.array(dillAtl[2], dtype=np.float64)
    b2Atl = dillAtl[3]
    LW2_1Atl = np.array(dillAtl[4], dtype=np.float64)
    b3Atl = np.array([[dillAtl[5]]], dtype=np.float64)
    LW3_2Atl = np.array(dillAtl[6], dtype=np.float64)
    y1_step1Atl = dillAtl[7]

    x1_step1xoffsetAtl = np.array(x1_step1Atl["xoffset"])
    x1_step1gainAtl = np.array(x1_step1Atl["gain"])
    x1_step1yminAtl = np.array(x1_step1Atl["ymin"])

    y1_step1yminAtl = np.array(y1_step1Atl["ymin"])
    y1_step1ygainAtl = np.array(y1_step1Atl["gain"])
    y1_step1xoffsetAtl = np.array(y1_step1Atl["xoffset"])

    x1_step1Other = dillOther[0]
    b1Other = np.array(dillOther[1], dtype=np.float64)
    IW1_1Other = np.array(dillOther[2], dtype=np.float64)
    b2Other = dillOther[3]
    LW2_1Other = np.array(dillOther[4], dtype=np.float64)
    b3Other = np.array([[dillOther[5]]], dtype=np.float64)
    LW3_2Other = np.array(dillOther[6], dtype=np.float64)
    y1_step1Other = dillOther[7]

    x1_step1xoffsetOther = np.array(x1_step1Other["xoffset"])
    x1_step1gainOther = np.array(x1_step1Other["gain"])
    x1_step1yminOther = np.array(x1_step1Other["ymin"])

    y1_step1yminOther = np.array(y1_step1Other["ymin"])
    y1_step1ygainOther = np.array(y1_step1Other["gain"])
    y1_step1xoffsetOther = np.array(y1_step1Other["xoffset"])

    TS = len(X)
    YAtl = np.full(TS, np.nan)
    YOther = np.full(TS, np.nan)

    if Net == 0:
        # Atl
        Xp1 = mapminmax_apply(
            X, x1_step1xoffsetAtl, x1_step1gainAtl, x1_step1yminAtl
        )
        a1 = tansig_apply(tile_n_dot(b1Atl, IW1_1Atl, Xp1.T))
        a2 = tile_n_dot(np.array([[b2Atl]]), LW2_1Atl, a1)
        YAtl = mapminmax_reverse(
            a2, y1_step1xoffsetAtl, y1_step1ygainAtl, y1_step1yminAtl
        )[
            0
        ]  # [0]
        # Other
        Xp1 = mapminmax_apply(
            X,
            x1_step1xoffsetOther,
            x1_step1gainOther,
            x1_step1yminOther,
        )
        a1 = tansig_apply(tile_n_dot(b1Other, IW1_1Other, Xp1.T))
        a2 = tile_n_dot(np.array([[b2Other]]), LW2_1Other, a1)
        YOther = mapminmax_reverse(
            a2, y1_step1xoffsetOther, y1_step1ygainOther, y1_step1yminOther
        )[
            0
        ]  # [0]
    else:
        # Atl
        Xp1 = mapminmax_apply(
            X, x1_step1xoffsetAtl, x1_step1gainAtl, x1_step1yminAtl
        )
        a1 = tansig_apply(tile_n_dot(b1Atl, IW1_1Atl, Xp1.T))
        a2 = tansig_apply(tile_n_dot(np.array(b2Atl), LW2_1Atl, a1))
        a3 = tile_n_dot(b3Atl, LW3_2Atl, a2)
        YAtl = mapminmax_reverse(
            a3, y1_step1xoffsetAtl, y1_step1ygainAtl, y1_step1yminAtl
        )[
            0
        ]  # [0]
        # Other
        Xp1 = mapminmax_apply(
            X,
            x1_step1xoffsetOther,
            x1_step1gainOther,
            x1_step1yminOther,
        )
        a1 = tansig_apply(tile_n_dot(b1Other, IW1_1Other, Xp1.T))
        a2 = tansig_apply(tile_n_dot(np.array(b2Other), LW2_1Other, a1))
        a3 = tile_n_dot(b3Other, LW3_2Other, a2)
        YOther = mapminmax_reverse(
            a3, y1_step1xoffsetOther, y1_step1ygainOther, y1_step1yminOther
        )[
            0
        ]  # [0]

    return YAtl, YOther


def pickle_picker(delta_over_gamma):
    mu = 2 * (delta_over_gamma) ** 2
    available_mu = np.array(  # 2*(0.2<=D/G<=1.8)^2, round to 2 s.f.
        [
            0.080,
            0.18,
            0.32,
            0.50,
            0.72,
            0.98,
            1.3,
            1.6,
            2.0,
            2.4,
            2.9,
            3.4,
            3.9,
            4.5,
            5.1,
            5.8,
            6.5,
        ]
    )
    nearest_mu = np.argmin(np.abs(available_mu - mu)) + 1
    return str(nearest_mu)
