# pyTRACE
version 0.0.1 (pre-alpha)

Tracer-based Rapid Anthropogenic Carbon Estimation (TRACE) converted to Python.

After https://github.com/BRCScienceProducts/TRACEv1

Please reference the manuscript for TRACEv1 (in print) for further details.

This code generates estimates of ocean anthropogenic carbon content from user-supplied inputs of coordinates (lat, lon, depth), salinity, temperature, and date. Information is also needed about the historical and/or future CO<sub>2</sub> trajectory.  This information can be provided or default values can be assumed.  

## Setup

Clone pyTRACE to your computer or download and extract a zipped file.  Ensure Python, pip, and the dependencies listed in requirements.txt are installed, preferably in a virtual environment. pyTRACE can then be installed (as an editable install) by navigating to the base directory of pyTRACE and running the following command in a terminal emulator
```
python -m pip install -e .
```

## Use

Call pyTRACE within Python by running 

```
from pyTRACE.main import pyTRACE
```

Which will make available the top-level function anthropogenic carbon estimation. For details on its input and output parameters, run

```
>?pyTRACE
```

## Disclaimer

The material embodied in this software is provided to you "as-is" and without warranty of any kind, express, implied or otherwise, including without limitation, any warranty of fitness for a particular purpose.In no event shall the authors be liable to you or anyone else for any direct, special, incidental, indirect or consequential damages of any kind, or any damages whatsoever, including without limitation, loss of profit, loss of use, savings or revenue, or the claims of third parties, whether or not the authors have been advised of the possibility of such loss, however caused and on any theory of liability, arising out of or in connection with the possession, use or performance of this software.
