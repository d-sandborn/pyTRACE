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
TRACE will be made available via pip once its dependencies are all available there. TRACE is not yet available via conda, but this is a target for future development if interest warrants it. PyCO2SYS >= v2 is required for speed and stability purposes, and is installed by default using the command above. More information on that package can be found [here](https://mvdh.xyz/PyCO2SYS/).