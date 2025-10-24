"""
Top level module of pyTRACE.

trace()
    Generates etimates of ocean anthropogenic carbon content from
    user-supplied inputs of coordinates (lat, lon, depth), salinity,
    temperature, and year.
No other functions presently implemented.
"""

import sys
import numpy as np
import numpy.typing as npt
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
    decimal_year_to_iso_timestamp,
    _integrate_column,
)
import platform


DATADIR = joinpath(dirname(__file__), "data")


def trace(
    output_coordinates: npt.ArrayLike,
    dates: npt.ArrayLike,
    predictor_measurements: npt.ArrayLike,
    predictor_types: npt.ArrayLike,
    atm_co2_trajectory: int = 1,
    preindustrial_xco2: float = 280.0,
    output_filename: str = None,
    delta_over_gamma: float = None,
    verbose_tf=True,
    error_codes: list = [-999, -9, -1e20],
    canth_diseq: float = 1.0,
    eos: str = "seawater",
    opt_pH_scale: int = 1,
    opt_k_carbonic: int = 10,  # LDK00
    opt_k_HSO4: int = 1,  # D90a
    opt_total_borate: int = 2,
    preformed_p: npt.ArrayLike = None,
    preformed_si: npt.ArrayLike = None,
    preformed_ta: npt.ArrayLike = None,
    scale_factors: npt.ArrayLike = None,
    meas_uncerts: npt.ArrayLike = None,
    per_kg_sw_tf: bool = True,
):
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

                             Python v0.2.0 beta

    Carter, B. R., Sandborn D. E. 2025.
    https://doi.org/10.5194/essd-17-3073-2025
    MATLAB - github.com/BRCScienceProducts/TRACEv1
    Python - github.com/d-sandborn/pyTRACE

    Generates etimates of ocean anthropogenic carbon content from
    user-supplied inputs of coordinates (lat, lon, depth), salinity,
    (optionally) temperature, and year.

    Information is also needed about the historical and/or future CO2
    trajectory.  This information can be provided or default values can
    be asssumed. Missing data should be indicated with np.nan
    A nan coordinate will yield nan estimates for all equations at
    that coordinate. A nan parameter value will yield nan estimates for
    all equations that require that parameter. Please send questions or
    related requests to sandborn@uw.edu and brendan.carter@gmail.com.
    ==================================================================

    This module can be installed as an editable or installation or as a
    package in a virtual environment. It references its necessary data files
    (found in the pyTRACE/data directory) internally, but can output analysis
    results into the directory specified by output_filename using absolute
    or relative reference conventions.

    Parameters
    ----------
    output_coordinates : ArrayLike
        n by 3 array of coordinates (longitude decimal degrees E, latitude
        decimal degrees N, depth m) at which estimates are desired.
    dates : ArrayLike
        n by 1 array of years c.e. for which estimates are desired.
    predictor_measurements : ArrayLike
        n by y array of y parameter measurements (salinity, temperature)
        The column order (y columns) is specified by predictor_types.
        Temperature should be expressed as degrees C and salinity should be
        specified on the practical scale with the unitless convention.
        nan inputs are acceptable, but will lead to nan estimates for
        any equations that depend on that parameter. If temperature is not
        provided it will be estimated from salinity (not recommended).
    predictor_types : ArrayLike
        1 by y array indicating which
        parameter is in each column of 'predictor_measurements'.
        Note that salinity is required for all equations. This applies to all
        n estimates. Input parameter key:
            1. Salinity
            2. Temperature
    atm_co2_trajectory : int
        Integer between 1 and 9 specifying the atmospheric xCO2 trajectory:
            1. Historical/Linear
            2. SSP1_1.9
            3. SSP1_2.6
            4. SSP2_4.5
            5. SSP3_7.0
            6. SSP3_7.0_lowNTCF
            7. SSP4_3.4
            8. SSP4_6.0
            9. SSP5_3.4_over
        Custom columns can be added to the data/CO2ATrajectoreisAdjusted.txt
        file and referenced here.
    preindustrial_xco2 : float, optional
        Preindustrial reference xCO2 value. The default is 280.
    output_filename: str, optional
        filename for TRACE output to be saved in current working directory.
        If no filename is given, no file will be saved. Presently only NETCDF4
        (.nc) files can be saved.
        The default is None.
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
        pCO2 as a function of atmospheric CO2. This should only be used if
        user-provided atmospheric trajectories not otherwise modified for
        anthropogenic carbon disequilibrium are being supplied.
        The default is 1.
    eos: str, optional
        Choice of seawater equation of state to use for temperature, density,
        and depth conversions. Available choices are 'seawater' (EOS-80)
        and 'gsw' (TEOS-10). 'seawater' will be deprecated, but is kept
        for compatibility with TRACEv1.
        The default is 'seawater'.
    delta_over_gamma: float, optional
        Ratio of second to first moments of inverse gaussian distribution
        used to convolute surface and interior histories of anthropogenic
        carbon.
        The default is 1.3038404810405297 to match TRACEv1 s.t.
        pf=makedist(‘InverseGaussian’,‘mu’,1,‘lambda’,3.4) is identical.
    opt_pH_scale: int, optional
        PyCO2SYS option for pH scale.
        The default is 1 (Total scale).
    opt_k_carbonic: int, optional
        PyCO2SYS option for carbonic acid dissociation constants.
        The default is 10 (LDK00).
    opt_k_HSO4: int, optional
        PyCO2SYS option for bisulfate dissociation constant.
        The default is 1 (D90a).
    opt_total_borate: int, optional
        PyCO2SYS option for borate:salinity relationship to use to estimate
        total borate.
        The default is 2.
    preformed_p: ArrayLike, optional
        n by 1 array of preformed P. When given along with preformed_ta and
        preformed_si, neural network estimation will be skipped.
        The default is None.
    preformed_si: ArrayLike, optional
        n by 1 array of preformed Si. When given along with preformed_ta and
        preformed_p, neural network estimation will be skipped.
        The default is None.
    preformed_ta: ArrayLike, optional
        n by 1 array of preformed TA. When given along with preformed_p and
        preformed_si, neural network estimation will be skipped.
        The default is None.
    scale_factors: ArrayLike, optional
        n by 1 array of scale factors for the inverse gaussian
        parameterization. When given neural network estimation will be skipped.
        The default is None.
    meas_uncerts : ArrayLike, optional
        ArrayLike object of measurement uncertainties presented in order
        indicated by 'predictor_types'. Providing these estimates may alter
        estimated uncertainties. Measurement uncertainties are a small part
        of TRACE estimate uncertainties for WOCE-quality measurements.
        However, estimate uncertainty scales with measurement uncertainty,
        so it is recommended that measurement uncertainties be specified
        for sensor measurements. If this optional input argument is not
        provided, the default WOCE-quality uncertainty is assumed.
        If values provided then the uncertainty estimates are assumed to
        apply uniformly to all input parameter measurements.
        The default is None.
    per_kg_sw_tf : bool, optional
        Retained for future development (allowing for flexible units
        for currently-unsupported predictors). The default is True.

    Returns
    -------
    output : xarray.Dataset
        CF-compliant ataset containing input parameters, estimated
        Canth, preformed properties, and associated metadata. This dataset
        is saved to the directory indicated by output_filename if provided.

    """
    equations = [1]  # leftover from ESPER
    equations = equation_check(equations)
    per_kg_sw_tf = units_check(per_kg_sw_tf)  # dummy placeholder

    # set at default 280 if not int or float
    preindustrial_xco2 = preindustrial_check(preindustrial_xco2)

    (
        meas_uncerts,
        input_u,
        use_default_uncertainties,
        predictor_measurements,
        predictor_types,
    ) = uncerts_check(  # prep user-provided uncerts, also check array format
        meas_uncerts,
        predictor_measurements,
        predictor_types,
    )

    # PyTRACE requires non-NaN coordinates to provide an estimate.  This step
    # eliminates NaN coordinate combinations prior to estimation.  NaN estimates
    # will be returned for these coordinates.
    valid_indices = ~np.logical_or(
        np.isnan(output_coordinates).any(axis=1).reshape(-1, 1),
        np.isnan(predictor_measurements)
        .all(axis=1)
        .reshape(-1, 1),  # True if both S and T are present
        np.isnan(dates).reshape(-1, 1),
    )
    valid_indices = np.argwhere(valid_indices > 0)[:, 0]

    # all depths made positive, also check array format
    output_coordinates = depth_check(output_coordinates, valid_indices)

    # Doing a size check for the coordinates.
    if np.shape(output_coordinates)[1] != 3:
        raise ValueError(
            "output_coordinates has too many or two few columns.  This version only allows 3 columns with the first being longitude (deg E), the second being latitude (deg N), and the third being depth (m)."
        )

    # Figuring out how many estimates are required
    n = len(valid_indices)

    # Checking for common missing data indicator flags and warning if any are
    # found.
    for i in error_codes:
        if i in predictor_measurements:
            warnings.warn(
                "A common non-NaN missing data indicator (e.g. -999, -9, -1e20) was detected in the input measurements provided.  Missing data should be replaced with np.nan, otherwise, PyTRACE will interpret your inputs at face value and give terrible estimates."
            )

    # Flag weird latitudes. Convert longitudes to 0-360.
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
            eos=eos,
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
    try:
        dates = np.asarray(dates)
    except Exception as e:
        print(f"{e}\nCould not convert at dates to a numpy array.")
    dates = dates[valid_indices]

    # Estimate preformed properties using a neural network or reuse old ones
    # which saves a considerable amount of time in loops
    if (
        (preformed_p is not None)
        and (preformed_ta is not None)
        and (preformed_si is not None)
    ):
        try:
            pref_props_sub = {  # eliminates indices without all req'd info
                "Preformed_P": preformed_p[valid_indices],
                "Preformed_Si": preformed_si[valid_indices],
                "Preformed_TA": preformed_ta[valid_indices],
            }
            if verbose_tf:
                print("\nUsing provided preformed properties.")
        except Exception as e:
            print("\ne\nDefaulting to estimating preformed properties.")
            pref_props_sub = trace_nn(
                [1, 2, 4],
                C,
                m_all,
                np.array([1, 2]),
                DATADIR,
                verbose_tf=verbose_tf,
                eos=eos,
            )
    else:
        if verbose_tf:
            print("\nEstimating preformed properties.")
        pref_props_sub = trace_nn(
            [1, 2, 4],
            C,
            m_all,
            np.array([1, 2]),
            DATADIR,
            verbose_tf=verbose_tf,
            eos=eos,
        )

    # Remap the scale factors using another neural network or reuse old ones
    # which saves a considerable amount of time in loops
    if scale_factors is not None:
        try:
            sfs = {  # eliminates indices without all req'd info
                "SFs": scale_factors[valid_indices]
            }
            if verbose_tf:
                print("\nUsing provided scale factors.")
        except Exception as e:
            print("\ne\nDefaulting to estimating scale factors.")
            sfs = trace_nn(
                [6],
                C,
                m_all,
                np.array([1, 2]),
                DATADIR,
                verbose_tf=verbose_tf,
                eos=eos,
            )
    else:
        if verbose_tf:
            print("\nEstimating scale factors.")
        sfs = trace_nn(
            [6],
            C,
            m_all,
            np.array([1, 2]),
            DATADIR,
            verbose_tf=verbose_tf,
            eos=eos,
            delta_over_gamma=delta_over_gamma,
        )

    # Load CO2 history
    # Note, this history has been modified to
    # reflect the values that would be expected in the surface ocean given the
    # slow response of the surface ocean to a rapidly changing atmospheric
    # value. "Adjusted" can be deleted in the following line to use the
    # original atmospheric values.  If this approach is used, then users should
    # consider altering canth_diseq below to modulate the degree of equilibrium.
    if verbose_tf:
        print(
            "\nLoading CO2 History:"
            + joinpath(DATADIR, "CO2TrajectoriesAdjusted.txt")
        )
    co2_rec = np.loadtxt(joinpath(DATADIR, "CO2TrajectoriesAdjusted.txt"))
    co2_rec = np.vstack([co2_rec[0, :], co2_rec])  # redundant??
    co2_rec[0, 0] = -1e10  # Set ancient CO2 to preindustrial placeholder

    if delta_over_gamma is None:  # if no D/G specified
        delta_over_gamma = (
            1.3038404810405297  # take default value == sqrt(3.4/2)
        )

    ventilation = inverse_gaussian_wrapper(
        x=np.arange(0.01, 5.01, 0.01), delta_over_gamma=delta_over_gamma
    )

    # Interpolate CO2 based on ventilation and atmospheric trajectory
    co2_set = interp1d(co2_rec[:, 0], co2_rec[:, atm_co2_trajectory])
    co2_set = co2_set(
        dates[:, None] - sfs["SFs"].reshape(-1, 1) * np.arange(1, 501)
    )
    co2_set = co2_set.dot(ventilation.T)

    # Calculate transit times (assumed based on ventilation)
    age = (sfs["SFs"].reshape(-1, 1) * np.arange(1, 501)).dot(ventilation.T)
    mode_age = (sfs["SFs"].reshape(-1, 1) * np.arange(1, 501))[
        :, np.argmax(ventilation)
    ]

    # Calculate vapor pressure correction term
    vpwp = np.exp(
        24.4543
        - 67.4509 * (100 / (293.15 + m_all[:, 1]))
        - 4.8489 * np.log((293.15 + m_all[:, 1]) / 100)
    )
    vpcorr_wp = np.exp(-0.000544 * m_all[:, 0])
    vpswwp = vpwp * vpcorr_wp
    vpfac = 1 - vpswwp

    # Calculate equilibrium DIC with and without anthropogenic CO2
    if verbose_tf:
        print("\nInitializing PyCO2SYS calculation.")
    out = pyco2.sys(
        alkalinity=pref_props_sub["Preformed_TA"],
        pCO2=vpfac
        * (
            canth_diseq * (co2_set.T - preindustrial_xco2) + preindustrial_xco2
        ),
        salinity=m_all[:, 0],
        temperature=m_all[:, 1],
        pressure=0,
        total_silicate=pref_props_sub["Preformed_Si"],
        total_phosphate=pref_props_sub["Preformed_P"],
        opt_pH_scale=opt_pH_scale,
        opt_k_carbonic=opt_k_carbonic,
        opt_k_HSO4=opt_k_HSO4,
        opt_total_borate=opt_total_borate,
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
        opt_pH_scale=opt_pH_scale,
        opt_k_carbonic=opt_k_carbonic,
        opt_k_HSO4=opt_k_HSO4,
        opt_total_borate=opt_total_borate,
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
                    "units": "micromole kg-1",
                    "long_name": "anthropogenic carbon",
                    "standard_name": "moles_of_anthropogenic_carbon_per_unit_mass_in_sea_water",
                },
            ),
            mean_age=(
                ["loc"],
                create_vector_with_values(
                    len(output_coordinates), valid_indices, age
                ),
                {
                    "units": "year",
                    "long_name": "mean water mass age",
                    "standard_name": "mean_age_of_water_mass",
                },
            ),
            mode_age=(
                ["loc"],
                create_vector_with_values(
                    len(output_coordinates), valid_indices, mode_age
                ),
                {
                    "units": "year",
                    "long_name": "mode water mass age",
                    "standard_name": "mode_age_of_water_mass",
                },
            ),
            dic=(
                ["loc"],
                create_vector_with_values(
                    len(output_coordinates), valid_indices, out
                ),
                {
                    "units": "micromole kg-1",
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
                    "units": "micromole kg-1",
                    "long_name": "preindustrial dissolved inorganic carbon",
                    "standard_name": "preindustrial_moles_of_dissolved_inorganic_carbon_per_unit_mass_in_sea_water",
                },
            ),
            pco2=(
                ["loc"],
                create_vector_with_values(
                    len(output_coordinates),
                    valid_indices,
                    vpfac
                    * (
                        canth_diseq * (co2_set.T - preindustrial_xco2)
                        + preindustrial_xco2
                    ),
                ),
                {
                    "units": "microatmosphere",
                    "long_name": "partial pressure of carbon dioxide",
                    "standard_nme": "partial_pressure_of_carbon_dioxide_in_sea_water",
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
                    "units": "microatmosphere",
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
                    "units": "micromole kg-1",
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
                    "units": "micromole kg-1",
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
                    "units": "micromole kg-1",
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
                    "units": "degree_Celsius",
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
                    "units": "1",
                    "long_name": "practical salinity",
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
                    "units": "micromole kg-1",
                    "long_name": "estimated uncertainty of anthropogenic carbon",
                    "standard_name": "uncertainty_moles_of_anthropogenic_carbon_per_unit_mass_in_sea_water",
                },
            ),
            delta_over_gamma=(
                ["loc"],
                create_vector_with_values(
                    len(output_coordinates),
                    valid_indices,
                    delta_over_gamma,
                ),
                {
                    "units": "1",
                    "long_name": "ratio of second to first moment of inverse gaussian distribution",
                    "standard_name": "ratio_of_second_to_first_moment_of_inverse_gaussian_distribution",
                },
            ),
            scale_factors=(
                ["loc"],
                create_vector_with_values(
                    len(output_coordinates),
                    valid_indices,
                    sfs["SFs"],
                ),
                {
                    "units": "1",
                    "long_name": "scaling factors of inverse gaussian distribution",
                    "standard_name": "scaling_factors_of_inverse_gaussian_distribution",
                },
            ),
        ),
        coords=dict(
            year=(  # make cftime-decodable
                ["loc"],
                decimal_year_to_iso_timestamp(
                    create_vector_with_values(
                        len(output_coordinates),
                        valid_indices,
                        dates,
                    )
                ),
                {
                    "units": "days since 0001-01-01 00:00:00",
                    "long_name": "year",
                    "standard_name": "year",
                    "calendar": "proleptic_gregorian",
                },
            ),
            lon=(
                ["loc"],
                output_coordinates[:, 0],
                {
                    "units": "degrees_east",
                    "long_name": "longitude",
                    "standard_name": "longitude",
                    "valid_min": -360,
                    "valid_max": 360,
                },
            ),
            lat=(
                ["loc"],
                output_coordinates[:, 1],
                {
                    "units": "degrees_north",
                    "long_name": "latitude",
                    "standard_name": "latitude",
                    "valid_min": -90,
                    "valid_max": 90,
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
            Conventions="CF-1.10",
            description="Results of Tracer-based Rapid Anthropogenic Carbon Estimation (TRACE)",
            history="TRACE version 0.2.0 (beta), "
            + str(datetime.datetime.now())
            + " Python "
            + sys.version
            + " "
            + platform.platform(),
            date_created=str(datetime.datetime.now()),
            references="doi.org/10.5194/essd-2024-560",
            co2sys_parameters=f"opt_pH_scale: {opt_pH_scale}, opt_k_carbonic: {opt_k_carbonic}, opt_k_HSO4: {opt_k_HSO4}, opt_total_borate: {opt_total_borate}",
            trace_parameters=f"per_kg_sw_tf: {per_kg_sw_tf}, canth_diseq: {canth_diseq}, eos: {eos}, delta_over_gamma: {delta_over_gamma}",
        ),
    )
    # Return results
    if verbose_tf:
        print("\nTRACE completed.")
    if output_filename is not None:
        try:
            output.to_netcdf(output_filename)
        except Exception as e:
            print("File " + output_filename + " could not be saved")
            print(e)
    return output


def integrate_column(
    integrand,
    salinity,
    temperature,
    depth,
    lat: float,
    bottom: float,
    top: float = 0,
    romb_resolution: int = 10,
):
    """
    Integrates anthropogenic carbon estimates at given geographic location.

    This function may be looped to construct basin or global inventories.

    Parameters
    ----------
    integrand : numpy.ndarray
        Array with length n of quantities to integrate.
        Must be in per kg seawater units.
    salinity : numpy.ndarray
        Array with length n of salinity values associated with integrand.
        Must be practical scale.
    temperature : numpy.ndarray
        Array with length n of temperature values associated with integrand.
        Must be in-situ temperature.
    depth : numpy.ndarray
        Array with length n of depth values associated with integrand.
        Must be units of meters bsl (i.e. positive values).
    lat : float
        Latitude N of column inventory location in degrees. Used for depth
        to pressure conversion.
    bottom : float
        Maximum depth of integration in meters bsl.
    top : float, optional
        Minimum depth of integration in meters bsl.
        The default is 0.
    romb_resolution : int, optional
        Controls of points over which pchip interpolation interpolates
        integrand, using formula points = 2^romb_resolution-1.
        The default is 10.

    Returns
    -------
    column_inventory : float
        Column integration of integrand, given in input units transformed to
        per square meter.

    """
    warnings.warn(
        "This integration is performed without checking for discontinuities, bathymetric boundaries, and other factors which may bias a column integration. This function should be used with caution. "
    )

    column_inventory = _integrate_column(
        integrand=integrand,
        salinity=salinity,
        temperature=temperature,
        depth=depth,
        lat=lat,
        bottom=bottom,
        top=top,
        romb_resolution=romb_resolution,
    )

    return column_inventory
