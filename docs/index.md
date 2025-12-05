# Tracer-based Rapid Anthropogenic Carbon Estimation (TRACE)

[![Python application](https://github.com/d-sandborn/pyTRACE/actions/workflows/python-app.yml/badge.svg)](https://github.com/d-sandborn/pyTRACE/actions/workflows/python-app.yml) 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17822675.svg)](https://doi.org/10.5281/zenodo.17822675)

After [TRACEv1](https://github.com/BRCScienceProducts/TRACEv1) and being developed in parallel with [ESPER](https://github.com/BRCScienceProducts/ESPER) and [PyESPER](https://github.com/LarissaMDias/PyESPER). This work is the subject of a manuscript in preparation or review, and should be considered preliminary. This repository will be updated with any preprints and final published paper, and a new release will be produced pending publication. 

This routine generates estimates of ocean anthropogenic carbon content from user-supplied inputs of coordinates (lon, lat, depth), salinity, temperature, and year. Information is also needed about the historical and/or future atmospheric CO<sub>2</sub> trajectory.  This information can be provided or default values can be assumed.  This tool is a multi-platform implementation of the inverse gaussian transit time distribution method aimed at increasing the accessibility of ocean anthropogenic carbon content estimation.

## Setup

Clone TRACE to your machine or download and unzip a [release](https://github.com/d-sandborn/pyTRACE/releases).  Ensure pip and python are installed in a virtual environment (we suggest [this method](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)). TRACE can then be installed (as an editable install) by navigating to the unzipped directory of TRACE and running the following command in a terminal emulator
```bash
python -m pip install -e .
```
TRACE will be made available via pip once its dependencies are all available there. TRACE is not yet available via conda, but this is a target for future development if interest warrants it. PyCO2SYS >= v2 is required for speed and stability purposes, and is installed by default using the command above. More information on that package can be found [here](https://mvdh.xyz/PyCO2SYS/).

## Citation

A paper describing TRACEv1 is freely available:

!!! note "TRACEv1 manuscript" Carter, B. R., Schwinger, J., Sonnerup, R., Fassbender, A. J., Sharp, J. D., Dias, L. M., & Sandborn, D. E. (2025). Tracer-based rapid anthropogenic carbon estimation (TRACE). Earth System Science Data, 17(6), 3073–3088. https://doi.org/10.5194/essd-17-3073-2025

To cite the original TRACEv1 software:

!!! note "TRACEv1 software" Carter, B. R. (2025). BRCScienceProducts/TRACEv1: TRACEv1_publication. Zenodo. https://doi.org/10.5281/zenodo.15692788

To cite the Python implementation of TRACE:

!!! note "TRACE-Python software" Sandborn, D. E., Barrett, R., & Carter, B. R. (2025). d-sandborn/TRACE: Tracer-based Rapid Anthropogenic Carbon Estimation (TRACE) (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.17822675

A publication describing the Python implementation of TRACE is presently in review:

!!! note "TRACE-Python manuscript" Sandborn, D.E., Carter, B. R., Warner, M. J., & Dias, L. M. TRACE-Python: Tracer-based Rapid Anthropogenic Carbon Estimation Implemented in Python (version 1.0). In review.

## Disclaimer

The material embodied in this software is provided to you "as-is" and without warranty of any kind, express, implied or otherwise, including without limitation, any warranty of fitness for a particular purpose.In no event shall the authors be liable to you or anyone else for any direct, special, incidental, indirect or consequential damages of any kind, or any damages whatsoever, including without limitation, loss of profit, loss of use, savings or revenue, or the claims of third parties, whether or not the authors have been advised of the possibility of such loss, however caused and on any theory of liability, arising out of or in connection with the possession, use or performance of this software.