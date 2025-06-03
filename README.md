# Tracer-based Rapid Anthropogenic Carbon Estimation (TRACE)
Python version 0.1.0 (beta)

After [TRACEv1](https://github.com/BRCScienceProducts/TRACEv1) and being developed in parallel with [ESPER](https://github.com/BRCScienceProducts/ESPER) and [PyESPER](https://github.com/LarissaMDias/PyESPER).

Please reference the [preprint](https://essd.copernicus.org/preprints/essd-2024-560/) and forthcoming publication on TRACEv1 for further details.

This code generates estimates of ocean anthropogenic carbon content from user-supplied inputs of coordinates (lon, lat, depth), salinity, temperature, and year. Information is also needed about the historical and/or future CO<sub>2</sub> trajectory.  This information can be provided or default values can be assumed.  This tool is a multi-platform implementation of the inverse gaussian transit time distribution method aimed at increasing the accessibility of ocean anthropogenic carbon content estimation.

## Setup

Clone TRACE to your machine or download and unzip a [release](https://github.com/d-sandborn/pyTRACE/releases).  Ensure pip is installed, preferably in a virtual environment (we suggest [this method](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)). pyTRACE can then be installed (as an editable install) by navigating to the unzipped directory of TRACE and running the following command in a terminal emulator
```
python -m pip install -e .
```
Additionally, PyCO2SYS > v2 is required for speed and stability purposes. Instructions to install the newest public beta for that package can be found [here](https://mvdh.xyz/PyCO2SYS/).

## Use

Call TRACE within a Python script or iPython console by running 

```
from pyTRACE import trace
```

which will make available the top-level function for anthropogenic carbon estimation. For details on its input and output parameters, run

```
?trace
```

To estimate anthropogenic carbon (C<sub>anth</sub>) at the surface ocean at the equator/prime meridian in the years 2000 and 2200 assuming SSP5_3.4_over:

```
>>> output = trace(
        output_coordinates=np.array([[0, 0, 0], [0, 0, 0]]),
        dates=np.array([2000, 2200]),
        predictor_measurements=np.array([[35, 20], [35, 20]]),
        predictor_types=np.array([1, 2]),
        atm_co2_trajectory=9
    )
```

which returns an [xarray dataset](https://docs.xarray.dev/en/latest/generated/xarray.Dataset.html) containing C<sub>anth</sub> at the specified dates and times, its uncertainties, and associated metadata. More examples can be found in the ```demos``` folder. Attributes of the [CF-compliant](https://cfconventions.org/) dataset describe the variables and their units.

```
>>> output

<xarray.Dataset> Size: 320B
Dimensions:       (loc: 2)
Coordinates:
    year          (loc) float64 16B 2e+03 2.2e+03
    lon           (loc) int64 16B 0 0
    lat           (loc) int64 16B 0 0
    depth         (loc) int64 16B 0 0
Dimensions without coordinates: loc
Data variables:
    canth         (loc) float64 16B 47.79 79.87
    mean_age      (loc) float64 16B 7.224 7.224
    mode_age      (loc) float64 16B 1.697 1.697
    dic           (loc) float64 16B 1.994e+03 2.026e+03
    dic_ref       (loc) float64 16B 1.946e+03 1.946e+03
    pco2          (loc) float64 16B 325.1 380.7
    pco2_ref      (loc) float64 16B 260.0 260.0
    preformed_ta  (loc) float64 16B 2.296e+03 2.296e+03
    preformed_si  (loc) float64 16B 2.167 2.167
    preformed_p   (loc) float64 16B 0.5108 0.5108
    temperature   (loc) float64 16B 20.0 20.0
    salinity      (loc) float64 16B 35.0 35.0
    u_canth       (loc) float64 16B 8.645 12.92
    gamma         (loc) float64 16B 1.0 1.0
    delta         (loc) float64 16B 1.3 1.3
    scale_factor  (loc) float64 16B 0.05143 0.05143
Attributes:
    Conventions:        CF-1.12
    description:        Results of Tracer-based Rapid Anthropogenic Carbon Es...
    history:            2025-06-02 15:14:07.415470 3.12.8 | packaged by conda...
    references:         doi.org/10.5194/essd-2024-560
    co2sys_parameters:  {'pressure': 0, 'opt_pH_scale': 1, 'opt_k_carbonic': ...
    trace_parameters:   {'meas_uncerts': None, 'per_kg_sw_tf': True, 'canth_d...

>>> output.canth.data

array([47.78685407, 79.87492991])

```

This result above agrees exactly with TRACEv1, which gives C<sub>anth</sub> = ```[47.7869 79.8749]``` for the same inputs.

Calling ```trace``` without a temperature input will cause it to estimate temperature from salinity and coordinates via a neural network. This is **not** recommended, but will yield results (with a warning) as in TRACEv1:

```
>>> trace(
         output_coordinates=np.array([[0, 0, 0], [0, 0, 0]]),
         dates=np.array([2000, 2010]),
         predictor_measurements=np.array([[35], [35]]),
         predictor_types=np.array([1]),
         atm_co2_trajectory=9
    )

UserWarning: Temperature is being estimated from salinity and coordinate information.

<xarray.Dataset> Size: 320B
Dimensions:       (loc: 2)
Coordinates:
    year          (loc) float64 16B 2e+03 2.01e+03
    lon           (loc) int64 16B 0 0
    lat           (loc) int64 16B 0 0
    depth         (loc) int64 16B 0 0
Dimensions without coordinates: loc
Data variables:
    canth         (loc) float64 16B 56.06 66.46
    mean_age      (loc) float64 16B 1.883 1.883
    mode_age      (loc) float64 16B 0.4424 0.4424
    dic           (loc) float64 16B 1.925e+03 1.935e+03
    dic_ref       (loc) float64 16B 1.869e+03 1.869e+03
    pco2          (loc) float64 16B 322.4 337.9
    pco2_ref      (loc) float64 16B 252.0 252.0
    preformed_ta  (loc) float64 16B 2.282e+03 2.282e+03
    preformed_si  (loc) float64 16B 1.787 1.787
    preformed_p   (loc) float64 16B 0.08551 0.08551
    temperature   (loc) float64 16B 26.47 26.47
    salinity      (loc) float64 16B 35.0 35.0
    u_canth       (loc) float64 16B 9.699 11.08
    gamma         (loc) float64 16B 1.0 1.0
    delta         (loc) float64 16B 1.3 1.3
    scale_factor  (loc) float64 16B 0.01341 0.01341
Attributes:
    Conventions:        CF-1.12
    description:        Results of Tracer-based Rapid Anthropogenic Carbon Es...
    history:            2025-06-02 15:55:17.630342 3.12.8 | packaged by conda...
    references:         doi.org/10.5194/essd-2024-560
    co2sys_parameters:  {'pressure': 0, 'opt_pH_scale': 1, 'opt_k_carbonic': ...
    trace_parameters:   {'meas_uncerts': None, 'per_kg_sw_tf': True, 'canth_d...
>>> output.canth.data

array([56.059132, 66.45668126])

```

The same result was obtained in TRACEv1: C<sub>anth</sub> = ```[56.0591 66.4567]``` for the same inputs.

## Disclaimer

The material embodied in this software is provided to you "as-is" and without warranty of any kind, express, implied or otherwise, including without limitation, any warranty of fitness for a particular purpose.In no event shall the authors be liable to you or anyone else for any direct, special, incidental, indirect or consequential damages of any kind, or any damages whatsoever, including without limitation, loss of profit, loss of use, savings or revenue, or the claims of third parties, whether or not the authors have been advised of the possibility of such loss, however caused and on any theory of liability, arising out of or in connection with the possession, use or performance of this software.
