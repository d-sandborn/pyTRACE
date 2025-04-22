"""
Top level module of pyTRACE.
trace()
    Generates etimates of ocean anthropogenic carbon content from
    user-supplied inputs of coordinates (lat, lon, depth), salinity,
    temperature, and date.
No other functions presently implemented.
"""

import sys
import numpy as np
from scipy.interpolate import interp1d
import warnings
import xarray as xr
from pyTRACE.neuralnets import trace_nn
import PyCO2SYS as pyco2
from os.path import dirname, join as joinpath
import datetime
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


DATADIR = joinpath(dirname(__file__), "data")


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
            1. Historical/Linear (modify historical CO2 file for updates)
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
    # package_dir = path.dirname(__file__)
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
            DATADIR,
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
        DATADIR,
        verbose_tf=verbose_tf,
    )

    # Remap the scale factors using another neural network
    if verbose_tf:
        print("\nEstimating scale factors.")
    sfs = trace_nn(
        [6], C, m_all, np.array([1, 2]), DATADIR, verbose_tf=verbose_tf
    )

    # Load CO2 history
    # Note, this history has been modified to
    # reflect the values that would be expected in the surface ocean given the
    # slow response of the surface ocean to a rapidly changing atmospheric
    # value. "Adjusted" can be deleted in the following line to use the
    # original atmospheric values.  If this approach is used, then users should
    # consider altering CanthDiseq below to modulate the degree of equilibrium.
    co2_rec = np.loadtxt(joinpath(DATADIR, "CO2TrajectoriesAdjusted.txt"))
    co2_rec = np.vstack([co2_rec[0, :], co2_rec])  # redundant??
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

    # function for rebuilding full-length output arrays

    def create_vector_with_values(target_length, index_vector, value_vector):
        result = np.full(target_length, np.nan)
        result[index_vector] = value_vector
        return result

    output = xr.Dataset(
        data_vars=dict(
            canth=(
                ["loc"],
                create_vector_with_values(
                    len(output_coordinates), valid_indices, canth_sub
                ),
                {
                    "units": "micro_mol_carbon_per_kg",
                    "long_name": "anthropogenic carbon",
                    "standard_name": "moles_of_anthropogenic_carbon_per_unit_mass_in_sea_water",
                    "ancillary_variables": "canth",
                },
            ),
            age=(
                ["loc"],
                create_vector_with_values(
                    len(output_coordinates), valid_indices, age
                ),
                {
                    "units": "year",
                    "long_name": "mean water mass age",
                    "standard_name": "age_of_water_mass",
                },
            ),
            dic=(
                ["loc"],
                create_vector_with_values(
                    len(output_coordinates), valid_indices, out
                ),
                {
                    "units": "micro_mol_carbon_per_kg",
                    "long_name": "dissolved inorganic carbon",
                    "standard_name": "moles_of_dissolved_inorganic_carbon_per_unit_mass_in_sea_water",
                },
            ),
            dic_ref=(
                ["loc"],
                create_vector_with_values(
                    len(output_coordinates), valid_indices, out_ref
                ),
                {
                    "units": "micro_mol_carbon_per_kg",
                    "long_name": "preindustrial dissolved inorganic carbon",
                    "standard_name": "preindustrial_moles_of_dissolved_inorganic_carbon_per_unit_mass_in_sea_water",
                },
            ),
            pco2=(
                ["loc"],
                create_vector_with_values(
                    len(output_coordinates),
                    valid_indices,
                    vpfac * (canth_diseq * (co2_set.T - 280) + 280),
                ),
                {
                    "units": "micro_atm",
                    "long_name": "partial pressure of carbon dioxide",
                    "standard_name": "partial_pressure_of_carbon_dioxide_in_sea_water",
                },
            ),
            pco2_ref=(
                ["loc"],
                create_vector_with_values(
                    len(output_coordinates),
                    valid_indices,
                    preindustrial_xco2 * vpfac,
                ),
                {
                    "units": "micro_atm",
                    "long_name": "preindustrial partial pressure of carbon dioxide",
                    "standard_name": "preindustrial_partial_pressure_of_carbon_dioxide_in_sea_water",
                },
            ),
            preformed_ta=(
                ["loc"],
                create_vector_with_values(
                    len(output_coordinates),
                    valid_indices,
                    pref_props_sub["Preformed_TA"],
                ),
                {
                    "units": "micro_mol_per_kg",
                    "long_name": "preformed alkalinity",
                    "standard_name": "sea_water_preformed_alkalinity_per_unit_mass_expressed_as_mole_equivalent",
                },
            ),
            preformed_si=(
                ["loc"],
                create_vector_with_values(
                    len(output_coordinates),
                    valid_indices,
                    pref_props_sub["Preformed_Si"],
                ),
                {
                    "units": "micro_mol_per_kg",
                    "long_name": "preformed total silicate",
                    "standard_name": "moles_of_silicate_per_unit_mass_in_sea_water",
                },
            ),
            preformed_p=(
                ["loc"],
                create_vector_with_values(
                    len(output_coordinates),
                    valid_indices,
                    pref_props_sub["Preformed_P"],
                ),
                {
                    "units": "micro_mol_per_kg",
                    "long_name": "preformed total phosphate",
                    "standard_name": "moles_of_phosphate_per_unit_mass_in_sea_water",
                },
            ),
            temperature=(
                ["loc"],
                create_vector_with_values(
                    len(output_coordinates), valid_indices, m_all[:, 1]
                ),
                {
                    "units": "degree_C",
                    "long_name": "in-situ temperature",
                    "standard_name": "sea_water_temperature",
                },
            ),
            salinity=(
                ["loc"],
                create_vector_with_values(
                    len(output_coordinates), valid_indices, m_all[:, 0]
                ),
                {
                    "units": 1,
                    "long_name": "spractical alinity",
                    "standard_name": "sea_water_practical_salinity",
                },
            ),
            u_canth=(
                ["loc"],
                create_vector_with_values(
                    len(output_coordinates),
                    valid_indices,
                    np.sqrt(4.4**2 + 2**2 + (0.15 * canth_sub) ** 2),
                ),
                {
                    "units": "micro_mol_carbon_per_kg",
                    "long_name": "estimated uncertainty of anthropogenic carbon",
                    "standard_name": "uncertainty_moles_of_anthropogenic_carbon_per_unit_mass_in_sea_water",
                },
            ),
        ),
        coords=dict(
            year=(
                ["loc"],
                create_vector_with_values(
                    len(output_coordinates), valid_indices, dates
                ),
                {
                    "units": "years since 1-1-1 0:0:0",
                    "long_name": "calendar year c.e.",
                    "standard_name": "mole_concentration_of_anthropogenic_carbon_in_seawater",
                },
            ),
            lon=(
                ["loc"],
                output_coordinates[:, 0],
                {
                    "units": "degrees_east",
                    "long_name": "longitude",
                    "standard_name": "longitude",
                },
            ),
            lat=(
                ["loc"],
                output_coordinates[:, 1],
                {
                    "units": "degrees_north",
                    "long_name": "latitude",
                    "standard_name": "latitude",
                },
            ),
            depth=(
                ["loc"],
                output_coordinates[:, 2],
                {
                    "units": "m",
                    "long_name": "depth",
                    "standard_name": "depth_below_sea_surface",
                    "positive": "down",
                    "valid_min": 0,
                    "valid_max": 10936,
                },
            ),
        ),
        attrs=dict(
            Conventions="CF-1.12",
            description="Results of Tracer-based Rapid Anthropogenic Carbon Estimation (TRACE).",
            history=str(datetime.datetime.now()) + " " + sys.version,
            references="doi.org/10.5194/essd-2024-560",
        ),
    )
    # Return results
    if verbose_tf:
        print("\nTRACE completed.")
    return output
