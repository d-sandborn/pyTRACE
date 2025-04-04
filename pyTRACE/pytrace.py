import numpy as np
import pandas as pd
from scipy.stats import invgauss
from scipy.interpolate import interp1d
import warnings
from os import path
from seawater import satO2, ptmp, dens, pres  # TODO replace with gsw
import xarray as xr
from pyTRACE.neuralnets import trace_nn
import PyCO2SYS as pyco2
from pyTRACE.utils import (
    equation_check,
    units_check,
    preindustrial_check,
    uncerts_check,
    depth_check,
    coordinate_check,
    prepare_uncertainties,
    inverse_gaussian_wrapper,
    say_hello,
)


def trace(
    output_coordinates,
    dates,
    predictor_measurements,
    predictor_types,
    atm_co2_trajectory,
    preindustrial_xco2=280,
    equations=[1],
    meas_uncerts=None,
    per_kg_sw_tf=True,
    verbose_tf=True,
    error_codes=[-999, -9, -1e20],
    canth_diseq=1,
):
    """
    Generates etimates of ocean anthropogenic carbon content from
    user-supplied inputs of coordinates (lat, lon, depth), salinity,
    temperature, and date.
    ==================================================================

    Information is also needed about the historical and/or future CO2
    trajectory.  This information can be provided or default values can
    be asssumed. Missing data should be indicated with np.nan
    A nan coordinate will yield nan estimates for all equations at
    that coordinate. A nan parameter value will yield nan estimates for
    all equations that require that parameter. Please send questions or
    related requests to sandborn@uw.edu and brendan.carter@gmail.com.

    Parameters
    ----------
    output_coordinates : numpy.ndarray
        n by 3 array of coordinates (longitude degrees E, latitude
        degrees N, depth m) at which estimates are desired.
    dates : numpy.array
        Array of years c.e. for which estimates
        are desired. Decimal digits ignored I think?
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
    atm_co2_trajectory : int
        Integer between 1 and 9 specifying the
        atmospheric CO2 trajectory:
            1. Histyorical/Linear (modify historical CO2 file for updates)
            2. SSP1_1.9
            3. SSP1_2.6
            4. SSP2_4.5
            5. SSP3_7.0
            6. SSP3_7.0_lowNTCF
            7. SSP4_3.4
            8. SSP4_6.0
            9. SSP5_3.4_over
    preindustrial_xco2 : float, optional
        Preindustrial reference xCO2 value. T
        he default is 280.
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
    per_kg_sw_tf : bool, optional
        Retained for future development (allowing for flexible units
        for currently-unsupported predictors). The default is True.
    verbose_tf : bool, optional
        Flag to control output verbosity. Setting this to False will
        make TRACE stop printing updates to the command line.  Warnings
        and errors, if any, will be given regardless.
        The default is True.
    error_codes : list, optional
        Error codes to be parsed as np.nan in input parameter arrays.
        The default is [-999, -9, -1e20].
    canth_diseq : int, optional
        Air-sea carbon dioxide equilibrium assumed for calculation of
        pCO2 as a function of atmospheric CO2. A value of 1 indicates
        full equilibrium.
        The default is 1.

    Raises
    ------
    ValueError
        Input parameter issues reported to the user.

    Returns
    -------
    output : xarray.Dataset
        Dataset containing input parameters and corresponding estimated
        Canth, preformed properties, and associated metadata.

    """
    package_dir = path.dirname(__file__)
    equations = equation_check(equations)
    per_kg_sw_tf = units_check(per_kg_sw_tf)
    preindustrial_xco2 = preindustrial_check(preindustrial_xco2)
    meas_uncerts, input_u, use_default_uncertainties = uncerts_check(
        meas_uncerts, predictor_measurements, predictor_types
    )

    # PyTRACE requires non-NaN coordinates to provide an estimate.  This step
    # eliminates NaN coordinate combinations prior to estimation.  NaN estimates
    # will be returned for these coordinates.
    # nan_grid_coords = np.any(np.isnan(output_coordinates), axis=1) or np.isnan(dates)  # WHAT??
    valid_indices = ~np.logical_or(
        np.isnan(output_coordinates).any(axis=1).reshape(-1, 1),
        np.isnan(dates).reshape(-1, 1),
    )
    valid_indices = np.argwhere(valid_indices > 0)[:, 0]
    output_coordinates = depth_check(output_coordinates, valid_indices)

    # Doing a size check for the coordinates.
    if np.shape(output_coordinates)[1] != 3:
        raise ValueError(
            "output_coordinates has either too many or two few columns.  This version only allows 3 columns with the first being longitude (deg E), the second being latitude (deg N), and the third being depth (m)."
        )

    # Figuring out how many estimates are required
    n = len(valid_indices)

    # Checking for common missing data indicator flags and warning if any are found.
    # Consider adding your commonly-used flags.
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

    # Ensure all predictors are identified
    if len(predictor_types) != predictor_measurements.shape[1]:
        raise ValueError(
            "predictor_types and predictor_measurements must have the same number of columns."
        )

    # Estimate temperature if not provided
    if 2 not in predictor_types:
        warnings.warn(
            "Temperature is being estimated from salinity and coordinate information."
        )
        ests = trace_nn(
            [7],
            output_coordinates,
            predictor_measurements[:, predictor_types == 1],
            np.array([1]),
            package_dir,
            verbose_tf=verbose_tf,
        )
        predictor_measurements = np.hstack(
            (predictor_measurements, ests["Temperature"][:, None])
        )
        predictor_types = np.append(predictor_types, 2)

    # Reorder predictors
    m_all = np.full((n, 2), np.nan)
    u_all = np.full((n, 2), np.nan)
    m_all[:, predictor_types - 1] = predictor_measurements[valid_indices, :]
    u_all[:, predictor_types - 1] = input_u_all[:, predictor_types]

    # Reshape Dates if necessary
    # if dates.ndim == 1:
    #    dates = np.tile(dates, (len(output_coordinates), 1))
    dates = dates[valid_indices]

    # Estimate preformed properties using a neural network
    if verbose_tf:
        print("\nEstimating preformed properties.")
    pref_props_sub = trace_nn(
        [1, 2, 4],
        C,
        m_all,
        np.array([1, 2]),
        package_dir,
        verbose_tf=verbose_tf,
    )

    # Remap the scale factors using another neural network
    if verbose_tf:
        print("\nEstimating scale factors.")
    sfs = trace_nn(
        [6], C, m_all, np.array([1, 2]), package_dir, verbose_tf=verbose_tf
    )

    # Load CO2 history
    # Note, this history has been modified to
    # reflect the values that would be expected in the surface ocean given the
    # slow response of the surface ocean to a rapidly changing atmospheric
    # value. "Adjusted" can be deleted in the following line to use the
    # original atmospheric values.  If this approach is used, then users should
    # consider altering CanthDiseq below to modulate the degree of equilibrium.
    co2_rec = np.loadtxt(package_dir + "/CO2TrajectoriesAdjusted.txt")
    co2_rec = np.vstack([co2_rec[0, :], co2_rec])
    co2_rec[0, 0] = -1e10  # Set ancient CO2 to preindustrial placeholder

    y = inverse_gaussian_wrapper(
        x=np.arange(0.01, 5.01, 0.01), gamma=1, delta=1.3
    )
    ventilation = y / y.sum()

    # Interpolate CO2 based on ventilation and atmospheric trajectory
    co2_set = interp1d(co2_rec[:, 0], co2_rec[:, 0 + atm_co2_trajectory])
    co2_set = co2_set(
        dates[:, None] - sfs["SFs"].reshape(-1, 1) * np.arange(1, 501)
    )
    co2_set = co2_set.dot(ventilation.T)

    # Calculate transit times (assumed based on ventilation)
    age = (sfs["SFs"].reshape(-1, 1) * np.arange(1, 501)).dot(
        ventilation.T
    )  # weird subset - check after NNs are working

    # Calculate vapor pressure correction term (assumed equation)
    vpwp = np.exp(
        24.4543
        - 67.4509 * (100 / (293.15 + m_all[:, 1]))
        - 4.8489 * np.log((293.15 + m_all[:, 1]))
    )
    vpcorr_wp = np.exp(-0.000544 * m_all[:, 0])
    vpswwp = vpwp * vpcorr_wp
    vpfac = 1 - vpswwp
    # vpfac=1; % This overrides the commented code above

    # This allows the user to arbitrarily change the degree of equilibration with
    # the anthropogenic transient (recommended value of 1 if using adjusted CO2
    # trajectories, which is default).

    # Calculate equilibrium DIC with and without anthropogenic CO2
    if verbose_tf:
        print("\nInitializing PyCO2SYS calculation.")
    out = pyco2.sys(
        alkalinity=pref_props_sub["Preformed_TA"],
        pCO2=vpfac * (canth_diseq * (co2_set.T - 280) + 280),
        salinity=m_all[:, 0],
        temperature=m_all[:, 1],
        pressure=0,
        total_silicate=pref_props_sub["Preformed_Si"],
        total_phosphate=pref_props_sub["Preformed_P"],
        opt_pH_scale=1,
        opt_k_carbonic=10,  # LDK00
        opt_k_HSO4=1,  # D90a
        opt_total_borate=2,  # LKB10
    )
    out = out["dic"]
    out_ref = pyco2.sys(
        alkalinity=pref_props_sub["Preformed_TA"],
        pCO2=preindustrial_xco2 * vpfac,
        salinity=m_all[:, 0],
        temperature=m_all[:, 1],
        pressure=0,
        total_silicate=pref_props_sub["Preformed_Si"],
        total_phosphate=pref_props_sub["Preformed_P"],
        opt_pH_scale=1,
        opt_k_carbonic=10,  # LDK00
        opt_k_HSO4=1,  # D90a
        opt_total_borate=2,  # LKB10
    )
    out_ref = out_ref["dic"]

    # Calculate anthropogenic carbon content
    canth_sub = out - out_ref

    output = xr.Dataset(
        data_vars=dict(
            dic=(["loc"], out),
            dic_ref=(["loc"], out_ref),
            canth=(["loc"], canth_sub),
            age=(["loc"], age),
            preformed_ta=(["loc"], pref_props_sub["Preformed_TA"]),
            preformed_si=(["loc"], pref_props_sub["Preformed_Si"]),
            preformed_p=(["loc"], pref_props_sub["Preformed_P"]),
            temperature=(["loc"], m_all[:, 1]),
            salinity=(["loc"], m_all[:, 0]),
            uncertainty=(
                ["loc"],
                np.sqrt(4.4**2 + 2**2 + (0.15 * canth_sub) ** 2),
            ),
        ),
        coords=dict(
            year=(["loc"], dates),
            lon=(["loc"], C[:, 0]),
            lat=(["loc"], C[:, 1]),
        ),
        attrs=dict(description="pyTRACE output"),
    )
    # Return results
    if verbose_tf:
        print("\npyTRACE completed.")
    return output
