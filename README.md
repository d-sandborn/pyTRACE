# Tracer-based Rapid Anthropogenic Carbon Estimation (TRACE)

[![Python application](https://github.com/d-sandborn/pyTRACE/actions/workflows/python-app.yml/badge.svg)](https://github.com/d-sandborn/pyTRACE/actions/workflows/python-app.yml) 
[![DOI](https://zenodo.org/badge/931694885.svg)](https://doi.org/10.5281/zenodo.15597122)

After [TRACEv1](https://github.com/BRCScienceProducts/TRACEv1) and being developed in parallel with [ESPER](https://github.com/BRCScienceProducts/ESPER) and [PyESPER](https://github.com/LarissaMDias/PyESPER). Please reference the TRACEv1 [publication](https://doi.org/10.5194/essd-17-3073-2025) for further details. This work is the subject of a manuscript in preparation or review, and should be considered preliminary. This repository will be updated with any preprints and final published paper, and a new release will be produced pending publication. 

This routine generates estimates of ocean anthropogenic carbon content from user-supplied inputs of coordinates (lon, lat, depth), salinity, temperature, and year. Information is also needed about the historical and/or future atmospheric CO<sub>2</sub> trajectory.  This information can be provided or default values can be assumed.  This tool is a multi-platform implementation of the inverse gaussian transit time distribution method aimed at increasing the accessibility of ocean anthropogenic carbon content estimation.

## Setup

Clone TRACE to your machine or download and unzip a [release](https://github.com/d-sandborn/pyTRACE/releases).  Ensure pip and python are installed in a virtual environment (we suggest [this method](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)). TRACE can then be installed (as an editable install) by navigating to the unzipped directory of TRACE and running the following command in a terminal emulator
```bash
python -m pip install -e .
```
TRACE is not yet available via conda, but this is a target for future development if interest warrants it. Additionally, PyCO2SYS >= v2 is required for speed and stability purposes, and is installed by default using the command above. More information on that package can be found [here](https://mvdh.xyz/PyCO2SYS/).

## Use

Call TRACE within a Python script or iPython console by running 

```python
from TRACE import trace
```

which will make available the top-level function for anthropogenic carbon estimation. For details on its input and output parameters, run

```python
?trace
```

Further documentation may be produced at a later date if interest warrants it. See also the comments and docstrings in the source code. Interested users are welcome to reach out to the development team with questions. 

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
    history:            TRACE version 0.2.0 (beta), 2025-07-16 19:36:23.45199...
    date_created:       2025-07-16 19:36:23.452010
    references:         doi.org/10.5194/essd-2024-560
    co2sys_parameters:  opt_pH_scale: 1, opt_k_carbonic: 10, opt_k_HSO4: 1, o...
    trace_parameters:   per_kg_sw_tf: True, canth_diseq: 1.0, eos: seawater, ...

output.canth.data

array([47.78685407, 79.87492991])

```

This result above agrees exactly with TRACEv1, which gives C<sub>anth</sub> = ```[47.7869 79.8749]``` for the same inputs.

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

### Column Integration

A column integration function is available:

```python
from TRACE import column_integration
```

This function integrates concentrations (e.g. anthropogenic carbon) for a single location between user-provided depths via Piecewise Cubic Hermite Interpolating Polynomial ([PCHIP](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html) followed by [Romberg](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.romb.html) numerical integration. While this function only calculates a column inventory at a single location, it is designed to be looped to produce regional or global inventories. 


## Disclaimer

The material embodied in this software is provided to you "as-is" and without warranty of any kind, express, implied or otherwise, including without limitation, any warranty of fitness for a particular purpose.In no event shall the authors be liable to you or anyone else for any direct, special, incidental, indirect or consequential damages of any kind, or any damages whatsoever, including without limitation, loss of profit, loss of use, savings or revenue, or the claims of third parties, whether or not the authors have been advised of the possibility of such loss, however caused and on any theory of liability, arising out of or in connection with the possession, use or performance of this software.
