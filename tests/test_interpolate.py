from __future__ import annotations

import numpy as np
import pykrige
import pytest
import xarray as xr

from mergeplg import interpolate, merge

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
        "lon": ("id", [-1, 0, 2]),
        "lat": ("id", [1, 1, 2]),
        "x": ("id", [-1, 0, 2]),
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


def test_interpolator_OK_exact_at_points():
    # Init data and interpolator, select CML and gauge data that do not overlap
    ds_cmls_t1 = ds_cmls.isel(cml_id=[0, 1], time=1)
    ds_gauges_t1 = ds_gauges.isel(id=[0, 1], time=1)

    interpolator = interpolate.InterpolateOrdinaryKriging(
        ds_grid=ds_rad,
        ds_cmls=ds_cmls_t1,
        ds_gauges=ds_gauges_t1,
        min_observations=1,
        variogram_parameters={"sill": 1, "range": 1, "nugget": 0},
        full_line=False,
    )

    # Test that providing only RG works
    interpolated = interpolator(
        da_gauges=ds_gauges_t1.R,
    )
    for gauge_id in ds_gauges_t1.id:
        merge_r = interpolated.sel(
            x=ds_gauges_t1.sel(id=gauge_id).x.data,
            y=ds_gauges_t1.sel(id=gauge_id).y.data,
        ).data
        gauge_r = ds_gauges_t1.R.sel(id=gauge_id).data
        np.testing.assert_almost_equal(
            merge_r,
            gauge_r,
            decimal=6,
        )

    # Test that providing only CML works
    interpolated = interpolator(
        da_cmls=ds_cmls_t1.R,
    )
    for cml_id in ds_cmls_t1.cml_id:
        merge_r = interpolated.sel(
            x=ds_cmls_t1.sel(cml_id=cml_id).x.data,
            y=ds_cmls_t1.sel(cml_id=cml_id).y.data,
        ).data
        gauge_r = ds_cmls_t1.R.sel(cml_id=cml_id).data
        np.testing.assert_almost_equal(
            merge_r,
            gauge_r,
            decimal=6,
        )

    # Test that providing RG and CML works
    interpolated = interpolator(
        da_cmls=ds_cmls_t1.R,
        da_gauges=ds_gauges_t1.R,
    )
    for cml_id in ds_cmls_t1.cml_id:
        merge_r = interpolated.sel(
            x=ds_cmls_t1.sel(cml_id=cml_id).x.data,
            y=ds_cmls_t1.sel(cml_id=cml_id).y.data,
        ).data
        gauge_r = ds_cmls_t1.R.sel(cml_id=cml_id).data
        np.testing.assert_almost_equal(
            merge_r,
            gauge_r,
            decimal=6,
        )
    for gauge_id in ds_gauges_t1.id:
        merge_r = interpolated.sel(
            x=ds_gauges_t1.sel(id=gauge_id).x.data,
            y=ds_gauges_t1.sel(id=gauge_id).y.data,
        ).data
        gauge_r = ds_gauges_t1.R.sel(id=gauge_id).data
        np.testing.assert_almost_equal(
            merge_r,
            gauge_r,
            decimal=6,
        )


def test_interpolator_IDW_exact_at_points():
    # Init data and interpolator, select CML and gauge that do not overlap
    ds_cmls_t1 = ds_cmls.isel(cml_id=[0, 1], time=1)
    ds_gauges_t1 = ds_gauges.isel(id=[0, 1], time=1)

    interpolator = interpolate.InterpolateIDW(
        ds_grid=ds_rad,
        ds_cmls=ds_cmls_t1,
        ds_gauges=ds_gauges_t1,
        min_observations=1,
    )

    # Test that providing only RG works
    interpolated = interpolator(
        da_gauges=ds_gauges_t1.R,
    )
    for gauge_id in ds_gauges_t1.id:
        merge_r = interpolated.sel(
            x=ds_gauges_t1.sel(id=gauge_id).x.data,
            y=ds_gauges_t1.sel(id=gauge_id).y.data,
        ).data
        gauge_r = ds_gauges_t1.R.sel(id=gauge_id).data
        np.testing.assert_almost_equal(
            merge_r,
            gauge_r,
            decimal=6,
        )

    # Test that providing only CML works
    interpolated = interpolator(
        da_cmls=ds_cmls_t1.R,
    )
    for cml_id in ds_cmls_t1.cml_id:
        merge_r = interpolated.sel(
            x=ds_cmls_t1.sel(cml_id=cml_id).x.data,
            y=ds_cmls_t1.sel(cml_id=cml_id).y.data,
        ).data
        gauge_r = ds_cmls_t1.R.sel(cml_id=cml_id).data
        np.testing.assert_almost_equal(
            merge_r,
            gauge_r,
            decimal=6,
        )

    # Test that providing RG and CML works
    interpolated = interpolator(
        da_cmls=ds_cmls_t1.R,
        da_gauges=ds_gauges_t1.R,
    )
    for cml_id in ds_cmls_t1.cml_id:
        merge_r = interpolated.sel(
            x=ds_cmls_t1.sel(cml_id=cml_id).x.data,
            y=ds_cmls_t1.sel(cml_id=cml_id).y.data,
        ).data
        gauge_r = ds_cmls_t1.R.sel(cml_id=cml_id).data
        np.testing.assert_almost_equal(
            merge_r,
            gauge_r,
            decimal=6,
        )
    for gauge_id in ds_gauges_t1.id:
        merge_r = interpolated.sel(
            x=ds_gauges_t1.sel(id=gauge_id).x.data,
            y=ds_gauges_t1.sel(id=gauge_id).y.data,
        ).data
        gauge_r = ds_gauges_t1.R.sel(id=gauge_id).data
        np.testing.assert_almost_equal(
            merge_r,
            gauge_r,
            decimal=6,
        )


def test_interpolator_update():
    # CML and rain gauge overlapping sets
    ds_cml_t1 = ds_cmls.isel(cml_id=[2, 1], time=1)
    ds_cml_t1_2 = ds_cmls.isel(cml_id=[0, 1], time=1)

    # Initialize highlevel-class
    interpolate_krig = interpolate.InterpolateOrdinaryKriging(
        ds_grid=ds_rad,
        ds_cmls=ds_cml_t1,
    )

    # Test that interpolator do not change
    interpolator_1 = interpolate_krig._interpolator

    # Interpolate field
    interpolate_krig(
        da_cmls=ds_cml_t1.R,
    )
    interpolator_2 = interpolate_krig._interpolator
    assert interpolator_1 is interpolator_2

    # Test that interpolator change when new data is present
    interpolate_krig(
        da_cmls=ds_cml_t1_2.R,
    )
    interpolator_3 = interpolate_krig._interpolator
    assert interpolator_3 is not interpolator_1


def test_no_data():
    # Test that providing no data raises ValueError
    ds_cml_t1 = ds_cmls.isel(cml_id=[1, 2], time=1)
    ds_cml_t1_2 = ds_cmls.isel(cml_id=[], time=1)

    interpolator = interpolate.InterpolateOrdinaryKriging(
        ds_grid=ds_rad,
        ds_cmls=ds_cml_t1,
        min_observations=1,
    )
    # Check that no data causes ValueError
    msg = "Please provide CML or rain gauge data"
    with pytest.raises(ValueError, match=msg):
        interpolator(
            da_cmls=ds_cml_t1_2.R,
        )


def test_kedpoint_vs_pykrige():
    # Select rain gauges
    da_gauges_t = ds_gauges.isel(id=[0, 1, 2], time=1).R

    # Select radar timestep
    da_rad_t = ds_rad.isel(time=1).R
    da_rad_t.data = np.array(
        [
            [1.0, 3.0, 3.0, 3.0],
            [2.0, 3.0, 4.0, 4.0],
            [4.0, 3.0, 4.0, 5.0],
            [4.0, 4.0, 6.0, 6.0],
        ]
    )

    variogram_model = "exponential"
    variogram_parameters = {"sill": 1, "range": 2, "nugget": 0}

    # Initialize highlevel-class
    interpolate_ked = merge.MergeKrigingExternalDrift(
        ds_rad=ds_rad,
        ds_gauges=ds_gauges,
        variogram_model=variogram_model,
        variogram_parameters=variogram_parameters,
        nnear=12,
        min_observations=1,
        full_line=False,
    )

    # Interpolate field
    interp_field = interpolate_ked(
        da_rad=da_rad_t,
        da_gauges=da_gauges_t,
    )

    # Setup pykrige
    ked = pykrige.UniversalKriging(
        da_gauges_t.x.data.ravel(),
        da_gauges_t.y.data.ravel(),
        da_gauges_t.data.ravel(),
        drift_terms=["external_Z"],
        external_drift=da_rad_t.data,
        external_drift_x=da_rad_t.x.data.ravel(),
        external_drift_y=da_rad_t.y.data.ravel(),
        variogram_model=variogram_model,
        variogram_parameters=variogram_parameters,
        pseudo_inv=True,
        exact_values=False,  # Account for nugget when predicting
    )

    z, ss = ked.execute(
        "points",
        da_rad_t.x_grid.data.ravel().astype(float),
        da_rad_t.y_grid.data.ravel().astype(float),
    )

    interp_field_pykrige = z.reshape(da_rad_t.x_grid.shape)
    np.testing.assert_almost_equal(interp_field_pykrige, interp_field)


def test_kedpoint_vs_pykrige_radarisone():
    # In this example the external drift term is all ones, causing the
    # interpolator to default to ordinary kriging.

    # Select rain gauges
    da_gauges_t = ds_gauges.isel(id=[0, 1, 2], time=1).R

    # Select radar timestep
    da_rad_t = ds_rad.isel(time=1).R

    variogram_model = "exponential"
    variogram_parameters = {"sill": 1, "range": 2, "nugget": 0.5}

    # Initialize highlevel-class
    interpolate_ked = merge.MergeKrigingExternalDrift(
        ds_rad=ds_rad,
        ds_gauges=ds_gauges,
        variogram_model=variogram_model,
        variogram_parameters=variogram_parameters,
        nnear=12,
        min_observations=1,
        full_line=False,
    )

    # Interpolate field
    interp_field = interpolate_ked(
        da_rad=da_rad_t,
        da_gauges=da_gauges_t,
    )

    # Get value at gauges
    rad = da_rad_t.sel(x=da_gauges_t.x, y=da_gauges_t.y).data

    # Setup pykrige using midpoint of CMLs as reference
    ok = pykrige.UniversalKriging(
        da_gauges_t.x.data.ravel(),
        da_gauges_t.y.data.ravel(),
        da_gauges_t.data.ravel(),
        specified_drift=rad.ravel(),
        variogram_model=variogram_model,
        variogram_parameters=variogram_parameters,
        pseudo_inv=True,
        exact_values=False,  # Account for nugget when predicting
    )

    z, ss = ok.execute(
        "points",
        da_rad_t.x_grid.data.ravel().astype(float),
        da_rad_t.y_grid.data.ravel().astype(float),
    )

    interp_field_pykrige = z.reshape(da_rad_t.x_grid.shape)

    np.testing.assert_almost_equal(interp_field_pykrige, interp_field)


def test_kud_ked():
    # CML and rain gauge overlapping sets
    ds_cmls_t = ds_cmls.isel(cml_id=[1], time=0)
    ds_gauges_t = ds_gauges.isel(id=[1], time=0)
    ds_grid_t = ds_rad.isel(time=0)

    variogram_model = "spherical"
    variogram_parameters = {"sill": 1, "range": 1, "nugget": 0.5}

    # Initialize highlevel-class
    interpolate_krig = merge.MergeKrigingExternalDrift(
        ds_rad=ds_grid_t,
        ds_cmls=ds_cmls_t,
        ds_gauges=ds_gauges_t,
        full_line=True,
        variogram_model=variogram_model,
        variogram_parameters=variogram_parameters,
        nnear=12,
        discretization=2,
        min_observations=1,
    )

    # Set sigma and update R CML
    ds_cmls_t["sigma"] = xr.full_like(ds_cmls_t.R, 0)
    ds_cmls_t["R"] = xr.full_like(ds_cmls_t.R, 1)
    ds_gauges_t["sigma"] = xr.full_like(ds_gauges_t.R, 0)

    # Interpolate field
    interp_field = interpolate_krig(
        da_rad=ds_grid_t.R,
        da_cmls=ds_cmls_t.R,
        da_gauges=ds_gauges_t.R,
        da_cmls_sigma=ds_cmls_t.sigma,
        da_gauges_sigma=ds_gauges_t.sigma,
    )

    # Test CML exact at 3 gridpoints (discretization = 2)
    cml_avrg = ds_cmls_t.R.data
    point1 = interp_field.sel(x=0, y=-1)
    point2 = interp_field.sel(x=1, y=0)
    point3 = interp_field.sel(x=2, y=1)

    np.testing.assert_almost_equal(point1, cml_avrg)
    np.testing.assert_almost_equal(point2, cml_avrg)
    np.testing.assert_almost_equal(point3, cml_avrg)

    # Test gauge at 1 gridpoint
    gauge_avrg = ds_gauges_t.R.data
    point1 = interp_field.sel(x=0, y=1)
    np.testing.assert_almost_equal(gauge_avrg, point1)


def test_kud_ok():
    # CML and rain gauge overlapping sets
    ds_cmls_t = ds_cmls.isel(cml_id=[1], time=0)
    ds_gauges_t = ds_gauges.isel(id=[1], time=0)
    ds_grid_t = ds_rad.isel(time=0)

    variogram_model = "spherical"
    variogram_parameters = {"sill": 1, "range": 1, "nugget": 0.5}

    # Initialize highlevel-class
    interpolate_krig = interpolate.InterpolateOrdinaryKriging(
        ds_grid=ds_grid_t,
        ds_cmls=ds_cmls_t,
        ds_gauges=ds_gauges_t,
        full_line=True,
        variogram_model=variogram_model,
        variogram_parameters=variogram_parameters,
        nnear=12,
        discretization=2,
        min_observations=1,
    )

    # Set sigma and update R CML
    ds_cmls_t["sigma"] = xr.full_like(ds_cmls_t.R, 0)
    ds_cmls_t["R"] = xr.full_like(ds_cmls_t.R, 1)
    ds_gauges_t["sigma"] = xr.full_like(ds_gauges_t.R, 0)

    # Interpolate field
    interp_field = interpolate_krig(
        da_cmls=ds_cmls_t.R,
        da_gauges=ds_gauges_t.R,
        da_cmls_sigma=ds_cmls_t.sigma,
        da_gauges_sigma=ds_gauges_t.sigma,
    )

    # Test CML exact at 3 gridpoints (discretization = 2)
    cml_avrg = ds_cmls_t.R.data
    point1 = interp_field.sel(x=0, y=-1)
    point2 = interp_field.sel(x=1, y=0)
    point3 = interp_field.sel(x=2, y=1)

    np.testing.assert_almost_equal(point1, cml_avrg)
    np.testing.assert_almost_equal(point2, cml_avrg)
    np.testing.assert_almost_equal(point3, cml_avrg)

    # Test gauge at 1 gridpoint
    gauge_avrg = ds_gauges_t.R.data
    point1 = interp_field.sel(x=0, y=1)
    np.testing.assert_almost_equal(gauge_avrg, point1)


def test_blockkriging_vs_pykrige():
    # CML and rain gauge overlapping sets
    ds_cml_t1 = ds_cmls.isel(cml_id=[2, 1], time=1)

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
        min_observations=1,
    )

    # Interpolate field
    interp_field = interpolate_krig(
        da_cmls=ds_cml_t1.R,
    )

    x_mid = 0.5 * (ds_cml_t1.site_0_x + ds_cml_t1.site_1_x).data
    y_mid = 0.5 * (ds_cml_t1.site_0_y + ds_cml_t1.site_1_y).data
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
