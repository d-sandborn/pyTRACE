# pyTRACE (for Python)
Tracer-based Rapid Anthropogenic Carbon Estimation (TRACE)
After https://github.com/BRCScienceProducts/TRACEv1
Version 0.0

ALL TEXT BELOW IS PLACEHOLDER
NEEDS UPDATING TO PYTHONIC FORMAT

This code generates estimates of ocean anthropogenic carbon content from
user-supplied inputs of coordinates (lat, lon, depth), salinity,
temperature, and date. Information is also needed about the historical
and/or future CO2 trajectory.  This information can be provided or
default values can be assumed.  

**Setup:** download the TRACEv1 archive to your computer and extract to a location 
on your MATLAB path (or add the location to your path).  Ensure the TRACEv1.m 
function and the /private directory are in the same folder.  Call the function 
using the guidelines below (and in the function description).

A detailed description of the methodology will be provided in a yet-to-be submitted 
manuscript.  The following is a summary.  The approach takes several steps:

**Steps taken during the formulation of this code: **
(1) transit time distributions are fit to CFC-11, CFC-12, and SF6
measurements made since ~2000 c.e. predominantly on repeat hydrography
cruises. These transit time distributions are inverse gaussians with one
scale factor that sets the mean age and width of the distribution. (2) A
neural network is constructed that relates the scale factor to
coordinate information and measured salinity and temperature. (3) A
separate neural network is uses an identical approach to relate the same
information to preformed property distributions estimated in a previous
research effort (Carter et al. 2021a:Preformed Properties for Marine
Organic Matter and Carbonate Mineral Cycling Quantification)

**Steps taken when the code is called: **
(1) Both neural networks are called, generating information that is used 
herein to construct preformed property information along with a transit 
time distribution.  (2) The user-provided dates are then combined with 
default, built-in, or user-provided atmospheric CO2 trajectories to 
produce records of the atmospheric CO2 exposure for each parcel of water. 
(3) This information is combined with estimated equilibrium values for 
the given CO2 exposure along with preindustrial (i.e., 280 uatm pCO2) 
CO2 exposure.  (4) The difference between the two is computed as the 
anthropogenic carbon estimate.

Updated 2025.02.12

Citation information: 
TBD

Related citations: (related to seawater property estimation)
ESPER_LIR and ESPER_NN, Carter et al., 2021: https://doi.org/10.1002/lom3.10461
LIARv1*: Carter et al., 2016, doi: 10.1002/lom3.10087
LIARv2*, LIPHR*, LINR* citation: Carter et al., 2018, https://doi.org/10.1002/lom3.10232
LIPR*, LISIR*, LIOR*, first described/used: Carter et al., 2021, https://doi.org/10.1029/2020GB006623
* deprecated in favor of ESPERs

ESPER_NN and TRACE are inspired by CANYON-B, which also uses neural networks: 
Bittig et al. 2018: https://doi.org/10.3389/fmars.2018.00328

This function needs the CSIRO seawater package to run.  Scale
differences from TEOS-10 are a negligible component of estimate errors
as formulated.

Example calls:

_This first example asks for estimates at the surface ocean at the equator/prime meridian in the years 2000 and 2200 assuming SSP5_3.4_over is followed_
```
[Canth]=TRACEv1([0 0 0;0 0 0],[2000;2200],[35 20;35 20],[1 2],[9],[0])
```
_Results in_
```
Canth =

   49.2636
   79.6467
```
_This second example demonstrates a function call performed without providing temperature information, which is not recommended and should result in a warning_
```
[Canth]=TRACEv1([0 0 0;0 0 0],[2000;2010],[35;35],[1],[1],[0])
```
_Results in_

Warning: TRACE was called either without providing temperature or without
specifying which column of PredictorMeasurements contains temperature.
Temperature is therefore being estimated from salinity and coordinate
information, but this is not optimal and the validation for TRACE should
not be considered appropriate for the estimates returned from this
function call. 
> In TRACEv1 (line 325) 
```
Canth =

   56.4181
   66.7979
```
