import numpy as np
import warnings
import datetime
from scipy.stats import invgauss

# from scipy.spatial import Delaunay
from gsw import pt0_from_t, rho_t_exact, p_from_z, SA_from_SP, CT_from_t
from tqdm import tqdm
from scipy.interpolate import PchipInterpolator
from scipy.integrate import romb

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from seawater import ptmp, dens, pres
# from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import geopandas as gpd

# import pandas as pd


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
    This input is not needed for TRACE, currently."""
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
            "preindustrial_xco2 could not be parsed as int or float. Setting to 280."
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
        if not np.shape(predictor_types)[0] == len(predictor_measurements):
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


def inverse_gaussian_wrapper(x, delta_over_gamma=1.3038404810405297):
    """Calculate ventilation distributions (assumed probability
    distribution). lambda should perhaps be 1/1.3 from He et al.
    Note that invgauss calls are different in pyTRACE and TRACE!
    Also note that TRACE approximates mu as 3.4 instead of ~3.38,
    leading to the default delta_over_gamma = sqrt(3.4/2)."""
    nu = 2 * (delta_over_gamma) ** 2  # default 3.4
    lam = 1
    y = invgauss.pdf(x, mu=nu / lam, scale=lam, loc=0)
    y = y / y.sum()
    return y


def inpolygon(xq, yq, xv, yv):
    """New test for points in polygon."""
    polygon_geom = Polygon(zip(xv, yv))
    polygon = gpd.GeoDataFrame(
        index=[0], crs="epsg:4326", geometry=[polygon_geom]
    )
    # df = pd.DataFrame({"lon": xq, "lat": yq})
    geo = gpd.points_from_xy(xq, yq)
    points = gpd.GeoDataFrame(geometry=geo, crs=polygon.crs)
    pointInPolys = points.intersects(polygon.union_all())
    return pointInPolys


# try: #this mistakenly identifies points within convex boundaries
#    points = np.array([xq, yq]).T
#    path = np.array([xv, yv]).T
#    tri = Delaunay(path)
#    return tri.find_simplex(points) >= 0
# except:
#    return np.array([False] * len(xq))


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

                         Python v0.2.0 beta
                    
Carter, B.; Sandborn D. 2025.
https://doi.org/10.5194/essd-17-3073-2025
MATLAB - github.com/BRCScienceProducts/TRACEv1
Python - github.com/d-sandborn/pyTRACE"""
    )


def decimal_year_to_iso_timestamp(  # for CF Conventions
    decimal_year_input: np.ndarray | float,
) -> np.ndarray | str:
    def _convert_single_decimal_year(decimal_year: float) -> str:
        if np.isnan(decimal_year):
            return "NaT"
        year = int(decimal_year)
        fraction = decimal_year - year

        days_in_year = (
            366
            if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
            else 365
        )

        total_seconds_in_year = days_in_year * 24 * 60 * 60
        offset_seconds = fraction * total_seconds_in_year

        start_of_year_utc = datetime.datetime(
            year, 1, 1, 0, 0, 0, 0, tzinfo=datetime.timezone.utc
        )

        dt_object_utc = start_of_year_utc + datetime.timedelta(
            seconds=offset_seconds
        )

        iso_timestamp = dt_object_utc.isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        )
        return iso_timestamp

    if isinstance(decimal_year_input, np.ndarray):
        vectorized_converter = np.vectorize(
            _convert_single_decimal_year, otypes=[str]
        )
        return vectorized_converter(decimal_year_input)
    else:
        return _convert_single_decimal_year(decimal_year_input)


def _integrate_column(
    integrand,
    salinity,
    temperature,
    depth,
    lat: float,
    bottom: float,
    top: float = 0,
    romb_resolution: int = 10,
):
    shapes = {v.shape for v in [integrand, salinity, temperature, depth]}
    if not len(shapes) == 1:
        raise ValueError("The shapes of the input vectors do not match.")
    num_target_points_for_romb = (2**romb_resolution) + 1
    # depthgrid, latgrid = np.meshgrid(ds.depth.data, ds.lat.data)
    pressure = p_from_z(-depth, lat * np.ones(len(depth)))
    profile = integrand * rho_t_exact(
        salinity, temperature, pressure
    )  # micromol/kg to micromol/m^3

    valid_indices = np.logical_and(
        (depth <= bottom),
        (~np.isnan(profile)),
    )

    if np.sum(valid_indices) < 1:  # nada if no water
        raise ValueError(
            "No valid indices to integrate. Check that at least one depth is less than the bottom depth, and that other values are reasonable."
        )
    elif np.sum(valid_indices) < 2:  # simple average if only one block
        warnings.warn(
            "Only one valid index to integrate! Assuming that this is the average value throughout the column. "
        )
        column_inventory = np.nansum(profile * (bottom - top))
    elif np.sum(valid_indices) >= 2:  # pchip/romb
        valid_original_depths = depth[valid_indices]
        valid_profile = profile[valid_indices]

        pchip_interpolator = PchipInterpolator(
            valid_original_depths,
            valid_profile,
            extrapolate=True,
        )
        dynamic_target_depth_points = np.linspace(
            top, bottom, num_target_points_for_romb
        )
        h = dynamic_target_depth_points[1] - dynamic_target_depth_points[0]
        # Perform interpolation
        interpolated_values = pchip_interpolator(dynamic_target_depth_points)
        try:
            column_inventory = romb(interpolated_values, dx=h)
        except ValueError as e:
            print(f"Error during Romberg integration.")

    return column_inventory
