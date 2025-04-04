"""
Combined function for NN estimation in pyTRACE
"""

import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
import os
from scipy import io
import pickle

from pyTRACE.utils import (
    equation_check,
    units_check,
    preindustrial_check,
    uncerts_check,
    depth_check,
    coordinate_check,
    prepare_uncertainties,
    inverse_gaussian_wrapper,
    inpolygon,
)
from seawater import satO2, ptmp, dens, pres


def trace_nn(
    desired_variables,
    output_coordinates,
    predictor_measurements,
    predictor_types,
    equations=[],
    meas_uncerts=None,
    error_codes=[-999, -9, -1e20],
    per_kg_sw_tf=True,
    verbose_tf=True,
):
    """
    Implements ESPER NN estimation of properties for pyTRACE.

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
    meas_uncerts, input_u, use_default_uncertainties = uncerts_check(
        meas_uncerts, predictor_measurements, predictor_types
    )
    valid_indices = ~np.isnan(output_coordinates).any(axis=1)
    valid_indices = np.argwhere(valid_indices > 0)[:, 0]

    n = len(valid_indices)
    e = np.size(equations)
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
    for i in tqdm(range(p), disable=(not verbose_tf)):
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
            m[:, 1] = ptmp(m[:, 0], m[:, 1], pres(C[:, 2], C[:, 1]), 0)

        # Checking to see whether O2 is needed. Defining AOU and subbing in for
        # O2 if yes (see above).
        # if need_vars[5]:
        #    m[:, 0, 3] = satO2(m[:, 0, 0], m[:, 0, 1]) * 44.64 - m[:, 0, 3]

        # Converting units to molality if they are provided as molarity.
        if not per_kg_sw_tf:
            densities = dens(m[:, 0], m[:, 1], pres(C[:, 2], C[:, 1])) / 1000
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
        fn = "./pyTRACE/Polys.mat"
        # Making sure you downloaded the needed file and put it somewhere it
        # can be found
        if not os.path.isfile(fn):
            raise FileNotFoundError(
                "TRACE could not find Polys.mat.  These mandatory file(s) should be distributed from the same website as TRACE.  Contact the corresponding author if you cannot find it there.  "
            )
        L = io.loadmat(fn)

        output_estimates = np.full((output_coordinates.shape[0]), np.nan)
        est = np.full(n, np.nan)
        m = np.concat([C[:, 2][:, None], m[:, :]], axis=1)

        est_atl = np.full((C.shape[0], 4), np.nan)
        est_other = np.full((C.shape[0], 4), np.nan)

        eq = 0
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
            for Net in tqdm(range(4), disable=(not verbose_tf), leave=False):
                # Separate neural networks are used for the Arctic/Atlantic and
                # the rest of the ocean.
                est_other[:, Net - 1] = execute_nn(
                    P, VName, "Other", Equation, Net, verbose_tf=verbose_tf
                )
                est_atl[:, Net - 1] = execute_nn(
                    P, VName, "Atl", Equation, Net, verbose_tf=verbose_tf
                )

                # est_atl[:, eq, Net] = function_atl(P.T).T
                # est_other[:, eq, Net] = function_other(P.T).T

        # Averaging across neural network committee members
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
        if VName == "SFs":
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
                ) + est_other[SoAfrInds, :] * ((C[SoAfrInds, 0] - 19) / 8)
            output_estimates[valid_indices] = Est
            output_estimates = 10.0**output_estimates
        else:
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
                ) + est_other[SoAfrInds, :] * ((C[SoAfrInds, 0] - 19) / 8)
            output_estimates[valid_indices] = Est
        Estimates[VName] = output_estimates

    return Estimates


def mapminmax_apply(x, settings={}):
    """Map Minimum and Maximum Input Processing Function"""
    y = np.subtract(x, np.array(settings["xoffset"]).T)
    y = np.multiply(y, np.array(settings["gain"]).T)
    y = np.add(y, np.array(settings["ymin"]).T)
    return y


def tansig_apply(n):
    """igmoid Symmetric Transfer Function"""
    return 2 / (1 + np.exp(-2 * n)) - 1


def mapminmax_reverse(y, settings={}):
    """Map Minimum and Maximum Output Reverse-Processing Function"""
    x = np.subtract(y, np.array(settings["ymin"]).T)
    x = np.divide(x, np.array(settings["gain"]).T)
    x = np.add(x, np.array(settings["xoffset"]).T)
    return x


def execute_nn(X, VName, Location, Equation, Net, verbose_tf=True):
    """Execute neural network by calling pickle file with weights
    determined using MATLAB machine learning routines, followed by
    linear algebra replicating neural network architecture exactly."""
    with open("./pyTRACE/nn_params.pkl", "rb") as f:
        dill = pickle.load(f)
    if VName == "Temperature":
        VName = "EstT_Temperature"
        X = X[:, 0:5]

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
    b1 = dill[1]
    IW1_1 = dill[2]
    b2 = dill[3]
    LW2_1 = dill[4]
    b3 = dill[5]
    LW3_2 = dill[6]
    y1_step1 = dill[7]

    TS = len(X)
    if len(X) != 0:
        Q = 1  # X[0][None, :].shape[1] if isinstance(X[0], np.ndarray) else len(X[0])
    else:
        Q = 1
    Y = np.full(TS, np.nan)  # [None] * TS
    for ts in tqdm(range(TS), disable=(not verbose_tf)):
        if Net == 0:
            Xp1 = mapminmax_apply(X[ts], x1_step1)
            a1 = tansig_apply(np.tile(b1, (1, Q)) + np.dot(IW1_1, Xp1.T))
            a2 = np.tile(b2, (1, Q)) + np.dot(LW2_1, a1)
            Y[ts] = mapminmax_reverse(a2, y1_step1)[0][0]
        else:
            Xp1 = mapminmax_apply(X[ts], x1_step1)
            a1 = tansig_apply(np.tile(b1, (1, Q)) + np.dot(IW1_1, Xp1.T))
            a2 = tansig_apply(np.tile(b2, (1, Q)) + np.dot(LW2_1, a1))
            a3 = np.tile(b3, (1, Q)) + np.dot(LW3_2, a2)
            Y[ts] = mapminmax_reverse(a3, y1_step1)[0][0]

    return Y
