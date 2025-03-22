from __future__ import annotations

import numpy as np
import pykrige
import xarray as xr

from mergeplg import interpolate

ds_cmls = xr.Dataset(
    data_vars={
        "R": (("cml_id", "time"), np.reshape(np.arange(1, 13), (3, 4))),
    },
    coords={
        "cml_id": ("cml_id", ["cml1", "cml2", "cml3"]),
        "time": ("time", np.arange(0, 4)),
        "site_0_x": ("cml_id", [-1, 0, 0]),
        "site_0_y": ("cml_id", [-1, -1, 2]),
        "site_1_x": ("cml_id", [1, 2, 2]),
        "site_1_y": ("cml_id", [1, 1, 0]),
        "site_0_lon": ("cml_id", [-1, 0, 0]),
        "site_0_lat": ("cml_id", [-1, -1, 2]),
        "site_1_lon": ("cml_id", [1, 2, 2]),
        "site_1_lat": ("cml_id", [1, 1, 0]),
        "x": ("cml_id", [0, 1, 1]),
        "y": ("cml_id", [0, 0, 1]),
        "length": ("cml_id", [2 * np.sqrt(2), 2 * np.sqrt(2), 2 * np.sqrt(2)]),
    },
)

ds_gauges = xr.Dataset(
    data_vars={
        "R": (("id", "time"), np.reshape(np.arange(1, 13), (3, 4))),
    },
    coords={
        "id": ("id", ["gauge1", "gauge2", "gauge3"]),
        "time": ("time", np.arange(0, 4)),
        "lon": ("id", [1, 0, 2]),
        "lat": ("id", [1, 1, 2]),
        "x": ("id", [1, 0, 2]),
        "y": ("id", [1, 1, 2]),
    },
)

ds_rad = xr.Dataset(
    data_vars={"R": (("time", "y", "x"), np.ones([4, 4, 4]))},
    coords={
        "x": ("x", [-1, 0, 1, 2]),
        "y": ("y", [-1, 0, 1, 2]),
        "time": ("time", [0, 1, 2, 3]),
        "lon": (("y", "x"), np.meshgrid([-1, 0, 1, 2], [-1, 0, 1, 2])[0]),
        "lat": (("y", "x"), np.meshgrid([-1, 0, 1, 2], [-1, 0, 1, 2])[1]),
        "xs": (("y", "x"), np.meshgrid([-1, 0, 1, 2], [-1, 0, 1, 2])[0]),
        "ys": (("y", "x"), np.meshgrid([-1, 0, 1, 2], [-1, 0, 1, 2])[1]),
    },
)


def test_blockkriging_vs_pykrige():
    # CML and rain gauge overlapping sets
    da_cml_t1 = ds_cmls.isel(cml_id=[2, 1], time=[0]).R
    da_gauges_t1 = ds_gauges.isel(id=[2, 1], time=[0]).R

    da_grid = ds_rad.isel(time=[0]).R

    # Initialize highlevel-class
    interpolate_krig = interpolate.InterpolateBlockKriging()

    variogram_model = "spherical"
    variogram_parameters = {"sill": 0.9, "range": 2, "nugget": 0.1}

    # Interpolate field
    interp_field = interpolate_krig.interpolate(
        da_grid,
        da_cml=da_cml_t1,
        da_gauge=da_gauges_t1,
        variogram_model=variogram_model,
        variogram_parameters=variogram_parameters,
        nnear=8,
        full_line=False,
    )
    print(np.round(interp_field.data, 5))
    # Interpolate field
    interp_field = interpolate_krig.interpolate(
        da_grid,
        da_cml=da_cml_t1,
        da_gauge=da_gauges_t1,
        variogram_model=variogram_model,
        variogram_parameters=variogram_parameters,
        nnear=False,
        full_line=False,
    )
    print(np.round(interp_field.data, 5))
    print("aiai")
    # Get ground observations and x0 geometry
    obs, x0 = interpolate_krig.get_obs_x0_(da_cml=da_cml_t1, da_gauge=da_gauges_t1)

    # Setup pykrige using midpoint of CMLs as reference
    ok = pykrige.OrdinaryKriging(
        x0[:, 1, int(x0.shape[2] / 2)],  # x-midpoint coordinate
        x0[:, 0, int(x0.shape[2] / 2)],  # y-midpoint coordinate
        obs.ravel(),
        variogram_model=variogram_model,
        variogram_parameters=variogram_parameters,
        pseudo_inv=True,
    )

    z, ss = ok.execute(
        "points",
        da_grid.xs.data.ravel().astype(float),
        da_grid.ys.data.ravel().astype(float),
    )
    interp_field_pykrige = [z.reshape(da_grid.xs.shape)]
    print(np.round(interp_field.data, 5))
    print(np.round(interp_field_pykrige, 5))
