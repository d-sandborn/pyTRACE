#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="trace-python",
    python_requires=">=3.12",
    version="1.0.0",
    description="Tracer-based Rapid Anthropogenic Carbon Estimation (TRACE) in Python",
    long_description="After [TRACEv1](https://github.com/BRCScienceProducts/TRACEv1) and being developed in parallel with [ESPER](https://github.com/BRCScienceProducts/ESPER) and [PyESPER](https://github.com/LarissaMDias/PyESPER). Please reference the TRACEv1 [publication](https://doi.org/10.5194/essd-17-3073-2025) for further details. This work is the subject of a manuscript in preparation or review, and should be considered preliminary. This repository will be updated with any preprints and final published paper, and a new release will be produced pending publication. This routine generates estimates of ocean anthropogenic carbon content from user-supplied inputs of coordinates (lon, lat, depth), salinity, temperature, and year. Information is also needed about the historical and/or future atmospheric CO<sub>2</sub> trajectory.  This information can be provided or default values can be assumed.  This tool is a multi-platform implementation of the inverse gaussian transit time distribution method aimed at increasing the accessibility of ocean anthropogenic carbon content estimation.",
    author="Daniel Sandborn & Brendan Carter",
    author_email="sandborn@uw.edu",
    url="https://github.com/d-sandborn/TRACE",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "netcdf4",
        "xarray",
        "scipy",
        "setuptools",
        "matplotlib",
        "pyyaml",
        "tqdm",
        "seawater",  # deprecated, kept for comparability with TRACEv1
        "gsw",  # replacement for seawater
        "PyCO2SYS @ git+https://github.com/mvdh7/PyCO2SYS@v2.0.0-b4",  # to be replaced with production version
        "numba",
        "shapely",
        "geopandas",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.13",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Oceanography",
    ],
)
