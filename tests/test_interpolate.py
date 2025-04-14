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
        "x_grid": (("y", "x"), np.meshgrid([-1, 0, 1, 2], [-1, 0, 1, 2])[0]),
        "y_grid": (("y", "x"), np.meshgrid([-1, 0, 1, 2], [-1, 0, 1, 2])[1]),
    },
)


def test_blockkriging_vs_pykrige():
    # CML and rain gauge overlapping sets
    ds_cml_t1 = ds_cmls.isel(cml_id=[2, 1], time=1)
    ds_gauges_t1 = ds_gauges.isel(id=[2, 1], time=0)

    ds_grid = ds_rad.isel(time=0)

    variogram_model = "exponential"
    variogram_parameters = {"sill": 1, "range": 2, "nugget": 0.5}

    # Initialize highlevel-class
    interpolate_krig = interpolate.InterpolateOrdinaryKriging(
        ds_grid=ds_rad,
        ds_cmls=ds_cml_t1,
        full_line=False,
        variogram_model=variogram_model,
        variogram_parameters=variogram_parameters,
        nnear=12,        
        min_observations=1
    )

    # Interpolate field
    interp_field = interpolate_krig(
        ds_grid,
        ds_cmls=ds_cml_t1,
    )

    x_mid = 0.5*(ds_cml_t1.site_0_x + ds_cml_t1.site_1_x).data
    y_mid = 0.5*(ds_cml_t1.site_0_y + ds_cml_t1.site_1_y).data
    obs = ds_cml_t1.R.data
    
    # Setup pykrige using midpoint of CMLs as reference
    ok = pykrige.OrdinaryKriging(
        x_mid.ravel(),
        y_mid.ravel(),  
        obs.ravel(),
        variogram_model=variogram_model,
        variogram_parameters=variogram_parameters,
        pseudo_inv=True,
        exact_values=False,  # Account for nugget when predicting
    )

    z, ss = ok.execute(
        "points",
        ds_grid.x_grid.data.ravel().astype(float),
        ds_grid.y_grid.data.ravel().astype(float),
    )

    interp_field_pykrige = z.reshape(ds_grid.x_grid.shape)

    np.testing.assert_almost_equal(interp_field_pykrige, interp_field)

