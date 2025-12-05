# Using TRACE

TRACE may be used to generate point estimates of anthropogenic carbon at one or more locations under default assumptions. This is demonstrated under **General Use**, below. More options are explained in the following sections. 

## General Use

Call TRACE within a Python script or iPython console by running 

```python
from tracepy import trace
```

which will make available the top-level function for anthropogenic carbon estimation. For details on its input and output parameters, see the API description below, or call

```python
?trace
```

To estimate anthropogenic carbon (C<sub>anth</sub>) at the surface ocean at the equator/prime meridian in the years 2000 and 2200 assuming IPCC pathway SSP5_3.4_over:

```python
output = trace(
    output_coordinates=np.array([[0, 0, 0], [0, 0, 0]]),
    dates=np.array([2000, 2200]),
    predictor_measurements=np.array([[35, 20], [35, 20]]),
    predictor_types=np.array([1, 2]),
    atm_co2_trajectory=9
)
```

which returns an [xarray dataset](https://docs.xarray.dev/en/latest/generated/xarray.Dataset.html) containing C<sub>anth</sub> at the specified dates and times, its uncertainties, and associated metadata. More examples can be found in the ```demos``` folder. Attributes of the [CF-compliant](https://cfconventions.org/) dataset describe the variables and their units.

```python
output

<xarray.Dataset> Size: 448B
Dimensions:           (loc: 2)
Coordinates:
    year              (loc) <U20 160B '2000-01-01T00:00:00Z' '2200-01-01T00:0...
    lon               (loc) int64 16B 0 0
    lat               (loc) int64 16B 0 0
    depth             (loc) int64 16B 0 0
Dimensions without coordinates: loc
Data variables:
    canth             (loc) float64 16B 47.79 79.87
    mean_age          (loc) float64 16B 7.224 7.224
    mode_age          (loc) float64 16B 1.697 1.697
    dic               (loc) float64 16B 1.994e+03 2.026e+03
    dic_ref           (loc) float64 16B 1.946e+03 1.946e+03
    pco2              (loc) float64 16B 325.1 380.7
    pco2_ref          (loc) float64 16B 260.0 260.0
    preformed_ta      (loc) float64 16B 2.296e+03 2.296e+03
    preformed_si      (loc) float64 16B 2.167 2.167
    preformed_p       (loc) float64 16B 0.5108 0.5108
    temperature       (loc) float64 16B 20.0 20.0
    salinity          (loc) float64 16B 35.0 35.0
    u_canth           (loc) float64 16B 8.645 12.92
    delta_over_gamma  (loc) float64 16B 1.3 1.3
    scale_factors     (loc) float64 16B 0.05143 0.05143
Attributes:
    Conventions:        CF-1.10
    description:        Results of Tracer-based Rapid Anthropogenic Carbon Es...
    history:            TRACE version 1.0.0, ...
    date_created:       2025-07-16 19:36:23.452010
    references:         doi.org/10.5194/essd-2024-560
    co2sys_parameters:  opt_pH_scale: 1, opt_k_carbonic: 10, opt_k_HSO4: 1, o...
    trace_parameters:   per_kg_sw_tf: True, canth_diseq: 1.0, eos: seawater, ...

output.canth.data

array([47.78685407, 79.87492991])

```

This result above agrees exactly with TRACEv1, which gives C<sub>anth</sub> = ```[47.7869 79.8749]``` for the same inputs.

## Estimation without temperature

Calling ```trace``` without a temperature input will cause it to estimate temperature from salinity and coordinates via a neural network. This is **not** recommended, but will yield results (with a warning) as in TRACEv1:

```python
output = trace(
    output_coordinates=np.array([[0, 0, 0], [0, 0, 0]]),
    dates=np.array([2000, 2010]),
    predictor_measurements=np.array([[35], [35]]),
    predictor_types=np.array([1]),
    atm_co2_trajectory=1
)

UserWarning: Temperature is being estimated from salinity and coordinate information.

output

<xarray.Dataset> Size: 448B
Dimensions:           (loc: 2)
Coordinates:
    year              (loc) <U20 160B '2000-01-01T00:00:00Z' '2010-01-01T00:0...
    lon               (loc) int64 16B 0 0
    lat               (loc) int64 16B 0 0
    depth             (loc) int64 16B 0 0
Dimensions without coordinates: loc
Data variables:
    canth             (loc) float64 16B 56.06 66.46
    mean_age          (loc) float64 16B 1.883 1.883
    mode_age          (loc) float64 16B 0.4424 0.4424
    dic               (loc) float64 16B 1.925e+03 1.935e+03
    dic_ref           (loc) float64 16B 1.869e+03 1.869e+03
    pco2              (loc) float64 16B 322.4 337.9
    pco2_ref          (loc) float64 16B 252.0 252.0
    preformed_ta      (loc) float64 16B 2.282e+03 2.282e+03
    preformed_si      (loc) float64 16B 1.787 1.787
    preformed_p       (loc) float64 16B 0.08551 0.08551
    temperature       (loc) float64 16B 26.47 26.47
    salinity          (loc) float64 16B 35.0 35.0
    u_canth           (loc) float64 16B 9.699 11.08
    delta_over_gamma  (loc) float64 16B 1.3 1.3
    scale_factors     (loc) float64 16B 0.01341 0.01341
Attributes:
    Conventions:        CF-1.10
    description:        Results of Tracer-based Rapid Anthropogenic Carbon Es...
    history:            TRACE version 0.2.0 (beta), 2025-07-16 19:36:55.79208...
    date_created:       2025-07-16 19:36:55.792097
    references:         doi.org/10.5194/essd-2024-560
    co2sys_parameters:  opt_pH_scale: 1, opt_k_carbonic: 10, opt_k_HSO4: 1, o...
    trace_parameters:   per_kg_sw_tf: True, canth_diseq: 1.0, eos: seawater, ...

output.canth.data

array([56.059132, 66.45668126])

```

The same result was obtained in TRACEv1: C<sub>anth</sub> = ```[56.0591 66.4567]``` for the same inputs.

## Other Options

Each argument to `trace` can either be a single scalar value (float or int), or an array given as a list or Numpy array containing a series of values. `trace` remains regrettably sensitive to argument formatting, so please check your input array dimensions and reach out with any problems (or solutions).

!!! inputs "`trace` Arguments"

    Necessary input arguments include `output_coordinates`, `dates`, `predictor_measurements`, and `predictor_types`. All other parameters are optional, with their default values listed below. 

    #### Coordinates

    Each of n rows of output coordinates and dates indicates a single location in space/time at which an estimation is made. `trace` cannot presently handle multidimensional (e.g. latitude/longitude/depth) arrays of coordinates: they must be [flattened](https://numpy.org/doc/stable/reference/generated/numpy.ravel.html) to one-dimensional vectors, then concatenated into the columns as described below. 

    * `output_coordinates`: n by 3 array of coordinates (longitude decimal degrees E, latitude decimal degrees N, depth m, in that order) at which estimates are desired.

    * `dates`: n by 1 array of years c.e. for which estimates are desired. 

    #### Predictors

    Unlike for ESPER or PyESPER, only salinity and temperature are accepted predictors for age and other preformed properties. 

    * `predictor_measurements`: n by y array of y parameter measurements (salinity, temperature). The column order (y columns) is specified by predictor_types. Temperature should be expressed as degrees C and salinity should be specified on the practical scale with the unitless convention. nan inputs are acceptable, but will lead to nan estimates for any equations that depend on that parameter. If temperature is not provided it will be estimated from salinity (not recommended).

    * `predictor_types` : 1 by y array indicating which parameter is in each column of 'predictor_measurements'. Note that salinity is required for all equations. This applies to all n estimates. Input parameter key:
        * `1`. Salinity
        * `2`. Temperature

    #### Atmospheric CO2 

    * `atm_co2_trajectory` : Integer between 1 and 9 specifying the atmospheric xCO2 trajectory:
        * `1`. Historical/Linear **(default)**
        * `2`. SSP1_1.9
        * `3`. SSP1_2.6
        * `4`. SSP2_4.5
        * `5`. SSP3_7.0
        * `6`. SSP3_7.0_lowNTCF
        * `7`. SSP4_3.4
        * `8`. SSP4_6.0
        * `9`. SSP5_3.4_over
    
        Custom columns can be added to the data/CO2TrajectoreisAdjusted.txt file and referenced here.
    
    * `preindustrial_xco2` : Optional preindustrial reference xCO2 value. The default is `280`.

    * `canth_diseq` : Air-sea carbon dioxide equilibrium assumed for calculation of pCO2 as a function of atmospheric CO2. This should only be used if user-provided atmospheric trajectories not otherwise modified for anthropogenic carbon disequilibrium are being supplied. The default is `1`.

    #### Output Options

    `trace` returns a CF-compliant dataset, which may be directly saved to a file for ease of data archival and scientific replicability. 

    * `output_filename`: Filename for `trace` output to be saved in current working directory. If no filename is given, no file will be saved. Presently only NETCDF4 (.nc) files can be saved.  The default is `None`.

    * `verbose_tf` : Flag to control output verbosity. Setting this to False will make `trace` stop printing updates to the command line.  Warnings and errors, if any, will be given regardless. The default is `True`.

    #### PyCO2SYS Options

    *The descriptions here derive in part from the [PyCO2SYS docs](https://pyco2sys.readthedocs.io/en/latest/co2sys_nd/)* which the interested reader should consult for further explanation. 

    * `opt_pH_scale`: PyCO2SYS option for pH scale:
        * `1`. Total, i.e. $\mathrm{pH} = -\log_{10} ([\mathrm{H}^+] + [\mathrm{HSO}_4^-])$ **(default)**
        * `2`. Seawater, i.e. $\mathrm{pH} = -\log_{10} ([\mathrm{H}^+] + [\mathrm{HSO}_4^-] + [\mathrm{HF}])$
        * `3`. Free, i.e. $\mathrm{pH} = -\log_{10} [\mathrm{H}^+]$
        * `4`. NBS, i.e. relative to [NBS/NIST](https://www.nist.gov/history/nist-100-foundations-progress/nbs-nist) reference standards

    * `opt_k_carbonic`: PyCO2SYS option for carbonic acid dissociation constants:
        * `1`: RRV93 (0 < *T* < 45 °C, 5 < *S* < 45, Total scale, artificial seawater)
        * `2`: GP89 (−1 < *T* < 40 °C, 10 < *S* < 50, Seawater scale, artificial seawater)
        * `3`: H73a and H73b refit by DM87 (2 < *T* < 35 °C, 20 < *S* < 40, Seawater scale, artificial seawater)
        * `4`: MCHP73 refit by DM87 (2 < *T* < 35 °C, 20 < *S* < 40, Seawater scale, real seawater)
        * `5`: H73a, H73b and [MCHP73](../refs/#m) refit by [DM87](../refs/#d) (2 < *T* < 35 °C, 20 < *S* < 40, Seawater scale, artificial seawater)
        * `6`: MCHP73 aka "GEOSECS" (2 < *T* < 35 °C, 19 < *S* < 43, NBS scale, real seawater)
        * `7`: MCHP73 without certain species aka "Peng" (2 < *T* < 35 °C, 19 < *S* < 43, NBS scale, real seawater)
        * `8`: M79 (0 < *T* < 50 °C, *S* = 0, freshwater only)
        * `9`: CW98 (2 < *T* < 30 °C, 0 < *S* < 40, NBS scale, real estuarine seawater)
        * `10`: LDK00 (2 < *T* < 35 °C, 19 < *S* < 43, Total scale, real seawater) **(default)**
        * `11`: MM02 (0 < *T* < 45 °C, 5 < *S* < 42, Seawater scale, real seawater)
        * `12`: MPL02 (−1.6 < *T* < 35 °C, 34 < *S* < 37, Seawater scale, field measurements)
        * `13`: MGH06 (0 < *T* < 50 °C, 1 < *S* < 50, Seawater scale, real seawater)
        * `14`: M10 (0 < *T* < 50 °C, 1 < *S* < 50, Seawater scale, real seawater)
        * `15`: WMW14 (0 < *T* < 45 °C, 0 < *S* < 45, Seawater scale, real seawater)
        * `16`: SLH20  (−1.67 < *T* < 31.80 °C, 30.73 < *S* < 37.57, Total scale, field measurements)
        * `17`: SB21 (15 < *T* < 35 °C, 19.6 < *S* < 41, Total scale, real seawater)
        * `18`: PLR18 (–6 < *T* < 25 °C, 33 < *S* < 100, Total scale, real seawater)

    The brackets above show the valid temperature (*T*) and salinity (*S*) ranges, original pH scale, and type of material measured to derive each set of constants.

    * `opt_k_HSO4`: PyCO2SYS option for bisulfate dissociation constant:

        * `1`: D90a **(default)**
        * `2`: KRCB77
        * `3`: WM13/WMW14

    * `opt_total_borate`: PyCO2SYS option for borate:salinity relationship to use to estimate total borate:

        * `1`: U74 **(default)**
        * `2`: LKB10
        * `3`: KSK18

    #### Preformed Properties

    These inputs are particularly useful for looped estimations, e.g. for successive re-estimation over time. As preformed properties are time-invariant, feeding `trace` the previously-estimated preformed properties for the same locations saves the vast majority of the run time. 
    
    * `preformed_p`: n by 1 array of preformed P. When given along with preformed_ta and preformed_si, neural network estimation will be skipped. The default is `None`.

    * `preformed_si`: n by 1 array of preformed Si. When given along with preformed_ta and preformed_p, neural network estimation will be skipped. The default is `None`.

    * `preformed_ta`: n by 1 array of preformed TA. When given along with preformed_p and preformed_si, neural network estimation will be skipped. The default is `None`.

    * `scale_factors`: n by 1 array of scale factors for the inverse gaussian parameterization. When given neural network estimation will be skipped. The default is `None`.

    #### Miscellaneous

    * `eos`: Choice of seawater equation of state to use for temperature, density, and depth conversions. Available choices are `seawater` (EOS-80) and `gsw` (TEOS-10). `seawater` will be deprecated, but is kept for compatibility with TRACEv1. The default is `seawater`.

    * `delta_over_gamma`: Ratio of second to first moments of inverse gaussian distribution used to convolute surface and interior histories of anthropogenic carbon. The default is `1.3038404810405297` to match TRACEv1 such that `pf=makedist(‘InverseGaussian’,‘mu’,1,‘lambda’,3.4)` is identical.

    * `meas_uncerts` : ArrayLike object of measurement uncertainties presented in order indicated by 'predictor_types'. Providing these estimates may alter estimated uncertainties. Measurement uncertainties are a small part of TRACE estimate uncertainties for WOCE-quality measurements. However, estimate uncertainty scales with measurement uncertainty, so it is recommended that measurement uncertainties be specified for sensor measurements. If this optional input argument is not provided, the default WOCE-quality uncertainty is assumed. If values provided then the uncertainty estimates are assumed to apply uniformly to all input parameter measurements. The default is `None`.

    * `per_kg_sw_tf` : Retained for future development (allowing for flexible units for currently-unsupported predictors). The default is `True`.

    * `error_codes` : List of error codes to be parsed as np.nan in input parameter arrays. The default is `[-999, -9, -1e20]`.

# Column Integration

A column integration function is available:

```python

from tracepy import column_integration

```

This function integrates concentrations (e.g. anthropogenic carbon) for a single location between user-provided depths via Piecewise Cubic Hermite Interpolating Polynomial ([PCHIP](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html)) followed by [Romberg](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.romb.html) numerical integration. While this function only calculates a column inventory at a single location, it is designed to be looped to produce regional or global inventories. 
