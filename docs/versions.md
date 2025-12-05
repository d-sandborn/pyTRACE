# Version History

## v1.0.0 **(current)**

This release is in preparation for a peer-reviewed publication.

* First production release!
* Added R.B. as contributor.
* Changed module/function calls to reduce conflict with builtin trace.py.
* Other bug fixes and speed improvements.

## v0.3.0-beta

This release is in preparation for a peer-reviewed publication.

* Changed name to TRACE to simplify calls and illustrate continuity with other implementations.
* Updated README and docstrings.
* Improved input/output tests.
* Vectorized preformed property estimation for significant speed boost.
* Improved variable Δ/Γ with new NN weights.
* Set up publication pipeline for PyPI.
* Other bug fixes and speed improvements.

## v0.2.0-beta

This release is in preparation for a peer-reviewed publication.

* Increased robustness to user input, but in need of more demonstrations on how to break it.
* More tests implemented via Github.
* Reached CF-1.10 compliance.
* Implemented automatic output file creation for better archiving and repeatability.
* Greatly expanded documentation in docstrings, comments. True docs are yet to come.
* Other bug fixes and speed improvements.

## v0.1.0-beta

* Agreement with TRACEv1 (MATLAB) reached for check values (it was the vapor pressure correction's fault).
* Most under-the-hood parameters made accessible at top-level function, including inverse gaussian shape, CO2SYS options, and atmospheric disequilibrium.
* Equation of state choice implemented with both CSIRO and GSW options.
* Preformed properties may be user-provided, which saves time on recalculating them for time series.
* Demo scripts updated and converted from Jupyter to Marimo notebooks, with an eye towards future web apps.
* Implemented github action for check values and imports.
* Brought output closer to CF compliance.
* Installation instructions made clearer (but not perfect).
* Various other bug fixes.

## v0.0.2-alpha

pyTRACE is now working reliably across geographic locations and oceanographic conditions and gives results in good (but not perfect) agreement with TRACEv1. Other highlights:

* Two demos now outline basic functions, with plots for ease of reference.
* Much-expanded documentation of functions; still no docs pages.
* Increasing compliance with locally-installable flat package format.
* Moving towards regularizing user experience between TRACE/ESPER applications, as well as under-the-hood functions.
* Input parameters, especially need for numpy matrix format, has proven to be too fragile. Need to make user input more robust, with eyes on common objects like Pandas DataFrames and Xarray dataarrays/sets.
* Several issues found which will need to be addressed before beta, including transition to Python-GSW and expanding user access to parameters like IG fit, vapor pressure options and CO2SYS options.


## v0.0.1-alpha

A first attempt Pre-release

* It works mostly. Probably still throws errors when going between ocean basins? Still slightly off TRACEv1.
