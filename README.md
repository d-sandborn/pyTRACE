# Tracer-based Rapid Anthropogenic Carbon Estimation (TRACE)
version 0.0.1 (alpha)

After [TRACEv1](https://github.com/BRCScienceProducts/TRACEv1) and being developed in parallel with [ESPER](https://github.com/BRCScienceProducts/ESPER) and [PyESPER](https://github.com/LarissaMDias/PyESPER).

Please reference the [preprint](https://essd.copernicus.org/preprints/essd-2024-560/) and forthcoming publication on TRACEv1 for further details.

This code generates estimates of ocean anthropogenic carbon content from user-supplied inputs of coordinates (lon, lat, depth), salinity, temperature, and year. Information is also needed about the historical and/or future CO<sub>2</sub> trajectory.  This information can be provided or default values can be assumed.  This tool is a multi-platform implementation of the transit time distribution method aimed at increasing the accessibility of ocean anthropogenic carbon content estimation.

## Setup

Clone TRACE to your computer or download and extract a zipped file.  Ensure Python, pip, and the dependencies listed in requirements.txt are installed, preferably in a virtual environment. pyTRACE can then be installed (as an editable install) by navigating to the base directory of TRACE and running the following command in a terminal emulator
```
python -m pip install -e .
```
Additionally, PyCO2SYS > v2 is required for speed and stability purposes. Please contact the authors for help installing it. 

## Use

Call TRACE within Python by running 

```
from pyTRACE import trace
```

Which will make available the top-level function for anthropogenic carbon estimation. For details on its input and output parameters, run

```
?trace
```

To estimate anthropogenic carbon (C<sub>anth</sub>) at the surface ocean at the equator/prime meridian in the years 2000 and 2200 assuming SSP5_3.4_over:
```
output = trace(
    output_coordinates=np.array([[0, 0, 0], [0, 0, 0]]),
    dates=np.array([2000, 2200]),
    predictor_measurements=np.array([[35, 20], [35, 20]]),
    predictor_types=np.array([1, 2]),
    atm_co2_trajectory=9
)
```
which returns an xarray dataset containing C<sub>anth</sub> at the specified dates and times, its uncertainties, and associated dic, age, and preformed properties. Note that the result below doesn't agree exactly with TRACEv1, which gives C<sub>anth</sub> = [47.7869, 79.8749] for the same inputs. The reasons for this disagreement being investigated.

```
>>> output

<xarray.Dataset> Size: 208B
Dimensions:       (loc: 2)
Coordinates:
    year          (loc) int64 16B 2000 2200
    lon           (loc) int64 16B 0 0
    lat           (loc) int64 16B 0 0
Dimensions without coordinates: loc
Data variables:
    dic           (loc) float64 16B 2.011e+03 2.04e+03
    dic_ref       (loc) float64 16B 1.962e+03 1.962e+03
    canth         (loc) float64 16B 49.31 77.96
    age           (loc) float64 16B 4.316 4.316
    preformed_ta  (loc) float64 16B 2.296e+03 2.296e+03
    preformed_si  (loc) float64 16B 2.167 2.167
    preformed_p   (loc) float64 16B 0.5108 0.5108
    temperature   (loc) float64 16B 20.0 20.0
    salinity      (loc) float64 16B 35.0 35.0
    uncertainty   (loc) float64 16B 8.835 12.65
Attributes:
    description:  pyTRACE output


```

More examples can be found in the ```demos``` folder.

## Disclaimer

The material embodied in this software is provided to you "as-is" and without warranty of any kind, express, implied or otherwise, including without limitation, any warranty of fitness for a particular purpose.In no event shall the authors be liable to you or anyone else for any direct, special, incidental, indirect or consequential damages of any kind, or any damages whatsoever, including without limitation, loss of profit, loss of use, savings or revenue, or the claims of third parties, whether or not the authors have been advised of the possibility of such loss, however caused and on any theory of liability, arising out of or in connection with the possession, use or performance of this software.
