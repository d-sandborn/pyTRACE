#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 13:29:39 2025

@author: des
"""
import pandas as pd
import numpy as np
import xarray as xr
from pyTRACE import trace
from seawater import dpth
import matplotlib.pyplot as plt
import scienceplots

import cmocean.cm as cm
from scipy.interpolate import griddata
from sklearn.neighbors import LocalOutlierFactor

from matplotlib.tri import (
    TriAnalyzer,
    Triangulation,
    UniformTriRefiner,
    LinearTriInterpolator,
)

plt.style.use("science")
a16n = xr.open_dataset("33RO20130803_bottle.nc")
a16s = xr.open_dataset("33RO20131223_bottle.nc")
combined_length = len(a16n.N_PROF) * len(a16n.N_LEVELS) + len(
    a16s.N_PROF
) * len(a16s.N_LEVELS)
latlist = np.concat(
    [
        (
            np.ones((len(a16n.N_PROF), len(a16n.N_LEVELS)))
            * a16n.latitude.data[:, None]
        ).ravel(),
        (
            np.ones((len(a16s.N_PROF), len(a16s.N_LEVELS)))
            * a16s.latitude.data[:, None]
        ).ravel(),
    ]
)
lonlist = np.concat(
    [
        (
            np.ones((len(a16n.N_PROF), len(a16n.N_LEVELS)))
            * a16n.longitude.data[:, None]
        ).ravel(),
        (
            np.ones((len(a16s.N_PROF), len(a16s.N_LEVELS)))
            * a16s.longitude.data[:, None]
        ).ravel(),
    ]
)
input_df = pd.DataFrame(
    {
        "lat": latlist,
        "lon": lonlist,
        "pressure": np.concat(
            [a16n.pressure.data.ravel(), a16s.pressure.data.ravel()]
        ),
        "year": np.ones((combined_length)) * 2013,
        "sal": np.concat(
            [
                a16n.bottle_salinity.data.ravel(),
                a16s.bottle_salinity.data.ravel(),
            ]
        ),
        "temp": np.concat(
            [
                a16n.ctd_temperature.data.ravel(),
                a16s.ctd_temperature.data.ravel(),
            ]
        ),
    }
)
# input_df = input_df.dropna(how="any")
input_df["depth"] = dpth(input_df.pressure, input_df.lat)  # pressure to depth

output = trace(
    output_coordinates=input_df[["lon", "lat", "depth"]].to_numpy(),
    dates=input_df.year.to_numpy(),
    predictor_measurements=input_df[["sal", "temp"]].to_numpy(),
    predictor_types=np.array([1, 2]),
    atm_co2_trajectory=5,
    verbose_tf=True,
)
# %%
output_df = pd.DataFrame(
    dict(
        lat=output.lat.data,
        depth=output.depth.data,
        canth=output.canth.data,
        age=output.age.data,
        sal=output.salinity.data,
        tc=output.temperature.data,
        ta=output.preformed_ta.data,
        dic=output.dic.data,
        uncert=output.uncertainty.data,
    )
).dropna(how="any")

bottom_depths = output_df.groupby(by="lat").max()
bottom_depths = bottom_depths.reset_index()

# %%
fig, ax = plt.subplots(dpi=200)
# ax.tricontour(output_df.lon, output_df.depth, output_df.canth, levels = 10, colors = 'k')
tc = ax.tricontourf(
    output_df.lat, output_df.depth, output_df.tc, levels=10, cmap="RdBu_r"
)
ax.scatter(
    output_df.lat,
    output_df.depth,
    c="k",
    s=1,
    marker=".",
    linewidths=0,
    edgecolors=None,
)
fig.colorbar(tc, label="CTD Temperature (C)")
ax.set_xlabel("Latitude")
ax.set_ylabel("Depth (m)")
ax.set_title("A16 c. 2013")
ax.fill_between(
    bottom_depths.lat, bottom_depths.depth, bottom_depths.depth.max(), color="k"
)

ax.invert_yaxis()
fig.savefig("a16_temp_demo.png")
# %%
fig, ax = plt.subplots(dpi=200)
# ax.tricontour(output_df.lon, output_df.depth, output_df.canth, levels = 10, colors = 'k')
tc = ax.tricontourf(
    output_df.lat, output_df.depth, output_df.sal, levels=10, cmap="RdBu_r"
)
ax.scatter(
    output_df.lat,
    output_df.depth,
    c="k",
    s=1,
    marker=".",
    linewidths=0,
    edgecolors=None,
)
fig.colorbar(tc, label="Bottle Salinity")
ax.set_xlabel("Latitude")
ax.set_ylabel("Depth (m)")
ax.set_title("A16 c. 2013")
ax.fill_between(
    bottom_depths.lat, bottom_depths.depth, bottom_depths.depth.max(), color="k"
)

ax.invert_yaxis()
fig.savefig("a16_sal_demo.png")

# %%
fig, ax = plt.subplots(dpi=200)
# ax.tricontour(output_df.lon, output_df.depth, output_df.canth, levels = 10, colors = 'k')
tc = ax.tricontourf(
    output_df.lat, output_df.depth, output_df.uncert, levels=10, cmap="RdBu_r"
)
ax.scatter(
    output_df.lat,
    output_df.depth,
    c="k",
    s=1,
    marker=".",
    linewidths=0,
    edgecolors=None,
)
fig.colorbar(tc, label="C_anth uncertainty (micromol/kg)")
ax.set_xlabel("Latitude")
ax.set_ylabel("Depth (m)")
ax.set_title("A16 c. 2013")
ax.fill_between(
    bottom_depths.lat, bottom_depths.depth, bottom_depths.depth.max(), color="k"
)

ax.invert_yaxis()
fig.savefig("a16_u_demo.png")
# %%
fig, ax = plt.subplots(dpi=200)
# ax.tricontour(output_df.lon, output_df.depth, output_df.canth, levels = 10, colors = 'k')
tc = ax.tricontourf(
    output_df.lat, output_df.depth, output_df.dic, levels=10, cmap="RdBu_r"
)
ax.scatter(
    output_df.lat,
    output_df.depth,
    c="k",
    s=1,
    marker=".",
    linewidths=0,
    edgecolors=None,
)
fig.colorbar(tc, label="Predicted DIC (micromol/kg)")
ax.set_xlabel("Latitude")
ax.set_ylabel("Depth (m)")
ax.set_title("A16 c. 2013")
ax.fill_between(
    bottom_depths.lat, bottom_depths.depth, bottom_depths.depth.max(), color="k"
)

ax.invert_yaxis()
fig.savefig("a16_dic_demo.png")

# %%
fig, ax = plt.subplots(dpi=200)
# ax.tricontour(output_df.lon, output_df.depth, output_df.canth, levels = 10, colors = 'k')
tc = ax.tricontourf(
    output_df.lat, output_df.depth, output_df.ta, levels=10, cmap="RdBu_r"
)
ax.scatter(
    output_df.lat,
    output_df.depth,
    c="k",
    s=1,
    marker=".",
    linewidths=0,
    edgecolors=None,
)
fig.colorbar(tc, label="Preformed TA (micromol/kg)")
ax.set_xlabel("Latitude")
ax.set_ylabel("Depth (m)")
ax.set_title("A16 c. 2013")
ax.fill_between(
    bottom_depths.lat, bottom_depths.depth, bottom_depths.depth.max(), color="k"
)

ax.invert_yaxis()
fig.savefig("a16_ta_demo.png")

# %%
xmin = output_df.lat.min()
xmax = output_df.lat.max()
ymin = output_df.depth.min()
ymax = output_df.depth.max()
cmap = cm.dense
vmin = 0
vmax = None
ybreak = 1000
X, Y = np.meshgrid(output_df.lat, output_df.depth)
Xn, Yn = np.mgrid[xmin:xmax:0.5, ymin:ymax:10]
Z = griddata(
    (output_df.lat, output_df.depth),
    output_df.canth,
    (Xn, Yn),
    method="linear",
    rescale=True,
)

fig, ax = plt.subplots(2, figsize=(9, 6))
mesh0 = ax[0].pcolormesh(Xn, Yn, Z, cmap=cmap, vmin=vmin, vmax=vmax)
mesh1 = ax[1].pcolormesh(Xn, Yn, Z, cmap=cmap, vmin=vmin, vmax=vmax)
ax[1].scatter(
    output_df.lat,
    output_df.depth,
    c="k",
    s=1,
    marker=".",
    linewidths=0,
    edgecolors=None,
)
ax[0].scatter(
    output_df.lat,
    output_df.depth,
    c="k",
    s=1,
    marker=".",
    linewidths=0,
    edgecolors=None,
)
fig.colorbar(
    mesh1,
    ax=ax.ravel().tolist(),
    label="C$_{anth}$ (micromol/kg)",
    location="right",
    anchor=(0.5, 1),
)
ax[1].set_xlabel("Latitude")
ax[1].set_ylabel("Depth (m)")
ax[0].set_title("TRACE-reconstructed A16 c. 2013")
ax[1].fill_between(
    bottom_depths.lat, bottom_depths.depth, bottom_depths.depth.max(), color="k"
)
ax[1].set_ylim(ymax, ybreak)
ax[0].set_ylim(ybreak, ymin)
ax[0].set_xlim(xmin, xmax)
ax[1].set_xlim(xmin, xmax)

plt.subplots_adjust(
    left=0.1, bottom=0.1, right=0.825, top=0.9, wspace=0.4, hspace=0
)

fig.savefig("a16_canth_demo.png")

# %%
xmin = output_df.lat.min()
xmax = output_df.lat.max()
ymin = output_df.depth.min()
ymax = output_df.depth.max()
cmap = cm.dense
vmin = None
vmax = None
ybreak = 1000
X, Y = np.meshgrid(output_df.lat, output_df.depth)
Xn, Yn = np.mgrid[xmin:xmax:0.5, ymin:ymax:10]
Z = griddata(
    (output_df.lat, output_df.depth),
    output_df.age,
    (Xn, Yn),
    method="linear",
    rescale=True,
)

fig, ax = plt.subplots(2, figsize=(9, 6))
mesh0 = ax[0].pcolormesh(Xn, Yn, Z, cmap=cmap, vmin=vmin, vmax=vmax)
mesh1 = ax[1].pcolormesh(Xn, Yn, Z, cmap=cmap, vmin=vmin, vmax=vmax)
ax[1].scatter(
    output_df.lat,
    output_df.depth,
    c="k",
    s=1,
    marker=".",
    linewidths=0,
    edgecolors=None,
)
ax[0].scatter(
    output_df.lat,
    output_df.depth,
    c="k",
    s=1,
    marker=".",
    linewidths=0,
    edgecolors=None,
)
fig.colorbar(
    mesh1,
    ax=ax.ravel().tolist(),
    label="Mean Age (yr)",
    location="right",
    anchor=(0.5, 1),
)
ax[1].set_xlabel("Latitude")
ax[1].set_ylabel("Depth (m)")
ax[0].set_title("TRACE-reconstructed A16 c. 2013")
ax[1].fill_between(
    bottom_depths.lat, bottom_depths.depth, bottom_depths.depth.max(), color="k"
)
ax[1].set_ylim(ymax, ybreak)
ax[0].set_ylim(ybreak, ymin)
ax[0].set_xlim(xmin, xmax)
ax[1].set_xlim(xmin, xmax)

plt.subplots_adjust(
    left=0.1, bottom=0.1, right=0.825, top=0.9, wspace=0.4, hspace=0
)

fig.savefig("a16_age_demo.png")

# %%
xmin = output_df.lat.min()
xmax = output_df.lat.max()
ymin = output_df.depth.min()
ymax = output_df.depth.max()
cmap = cm.dense
vmin = None
vmax = None
ybreak = 1000
X, Y = np.meshgrid(output_df.lat, output_df.depth)
Xn, Yn = np.mgrid[xmin:xmax:0.5, ymin:ymax:10]
param_list = ["canth", "age", "sal", "tc", "ta", "dic", "uncert"]
label_list = [
    "C$_{anth}$ ($\mu$mol kg$^{-1}$)",
    "Mean Age (yr)",
    "Salinity",
    "Temperature ($^{\circ}$C)",
    "A$_T$ ($\mu$mol kg$^{-1}$)",
    "DIC ($\mu$mol kg$^{-1}$)",
    "u(C$_{anth}$) ($\mu$mol kg$^{-1}$)",
]
for i in range(len(param_list)):
    Z = griddata(
        (output_df.lat, output_df.depth),
        output_df[param_list[i]],
        (Xn, Yn),
        method="linear",
        rescale=True,
    )
    vmin = np.nanmin(Z)
    vmax = np.nanmax(Z)
    fig, ax = plt.subplots(2, figsize=(9, 6))
    mesh0 = ax[0].contourf(Xn, Yn, Z, cmap=cmap, vmin=vmin, vmax=vmax)
    lines0 = ax[0].contour(Xn, Yn, Z, colors="k", vmin=vmin, vmax=vmax)
    mesh1 = ax[1].pcolormesh(Xn, Yn, Z, cmap=cmap, vmin=vmin, vmax=vmax)
    lines1 = ax[1].contour(Xn, Yn, Z, colors="k", vmin=vmin, vmax=vmax)
    ax[1].scatter(
        output_df.lat,
        output_df.depth,
        c="k",
        s=1,
        marker=".",
        linewidths=0,
        edgecolors=None,
    )
    ax[0].scatter(
        output_df.lat,
        output_df.depth,
        c="k",
        s=1,
        marker=".",
        linewidths=0,
        edgecolors=None,
    )
    fig.colorbar(
        mesh0,
        ax=ax.ravel().tolist(),
        label=label_list[i],
        location="right",
        anchor=(0.5, 1),
    )
    ax[1].set_xlabel("Latitude")
    ax[1].set_ylabel("Depth (m)")
    ax[0].set_title("TRACE-reconstructed A16 c. 2013")
    ax[1].fill_between(
        bottom_depths.lat,
        bottom_depths.depth,
        bottom_depths.depth.max(),
        color="k",
    )
    ax[1].set_ylim(ymax, ybreak)
    ax[0].set_ylim(ybreak, ymin)
    ax[0].set_xlim(xmin, xmax)
    ax[1].set_xlim(xmin, xmax)

    plt.subplots_adjust(
        left=0.1, bottom=0.1, right=0.825, top=0.9, wspace=0.4, hspace=0
    )

    fig.savefig("a16_" + param_list[i] + "_demo.png")
