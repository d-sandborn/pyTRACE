import marimo

__generated_with = "0.13.14"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # TRACE Demo

    The code below is intended to demonstrate the essential features of TRACE, compare it with its MATLAB predecessor TRACEv1, and establish guidelines for use and interpretation of its output. The code and output here are preliminary and subject to breaking changes.

    ## Overview

    TRACE is organized as a Python application rather than a published package, and should be downloaded or cloned from its Github repository at [github.com/d-sandborn/pyTRACE](https://github.com/d-sandborn/pyTRACE) and installed locally as per the directions on that page. 

    If you have questions, comments, or suggestion, please reach out to Daniel Sandborn at sandborn (at) uw.edu and Brendan Carter at brendan.carter (at) gmail.com. 

    ## Check Values

    The first demonstration mirrors that on the TRACEv1 repo and outputs the same check values. pyTRACE in its present form is a functionally-identical version of the TRACE method producing the same results for the same input, to machine precision.
    """
    )
    return


@app.cell
def _(np, trace):
    output = trace(output_coordinates=np.array([[0, 0, 0], [0, 0, 0]]),
                   dates=np.array([2000, 2200]),
                   predictor_measurements=np.array([[35, 20], [35, 20]]),
                   predictor_types=np.array([1, 2]),
                   atm_co2_trajectory=9,
                   verbose_tf = False #to remove loading bars from the notebook
                  )

    output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Speed Test""")
    return


@app.cell
def _(np, pd):
    _input_df = pd.DataFrame({'lat': np.ones(10000), 'lon': np.linspace(-80, 80, 10000), 'depth': np.random.normal(loc=1000, size=10000), 'year': np.ones(10000) * 2020, 'sal': np.random.normal(loc=35, size=10000), 'temp': np.random.normal(loc=15, size=10000)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Regional Scale""")
    return


@app.cell
def _(np, pd, trace, xr):
    ersst = xr.open_dataset(
        "http://psl.noaa.gov/thredds/dodsC/Datasets/noaa.ersst.v5/sst.mnmean.nc?lat[0:1:88],lon[0:1:179],time[0:1:2055],sst[0:1:2055][0:1:88][0:1:179]"
    )
    gridx, gridy = np.meshgrid(ersst.lon.data, ersst.lat.data)
    _input_df = pd.DataFrame(
        {
            "lon": gridx.ravel(),
            "lat": gridy.ravel(),
            "depth": np.ones(len(gridx.ravel())) * 50,
            "year": np.ones(len(gridx.ravel())) * 2020,
            "sal": np.ones(len(gridx.ravel())) * 35,
            "temp": ersst.sst[0, :, :].data.ravel(),
        }
    )
    output_1 = trace(
        output_coordinates=_input_df[["lon", "lat", "depth"]].to_numpy(),
        dates=_input_df.year.to_numpy(),
        predictor_measurements=_input_df[["sal", "temp"]].to_numpy(),
        predictor_types=np.array([1, 2]),
        atm_co2_trajectory=5,
        verbose_tf=False,
    )
    output_1
    return gridx, gridy, output_1


@app.cell
def _(gridx, gridy, np, output_1, plt):
    fig, ax = plt.subplots()
    p = ax.pcolormesh(gridx, gridy, np.reshape(output_1.canth.data, gridx.shape))
    fig.colorbar(p, label='C_anth (micro mol/kg)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Canth at 50m depth assuming SST from ERSSTv5')
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    from TRACE import trace
    import numpy as np
    import pandas as pd
    import xarray as xr
    return mo, np, pd, plt, trace, xr


if __name__ == "__main__":
    app.run()
