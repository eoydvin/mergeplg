from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from mergeplg import merge

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


def test_max_distance():
    # Rain gauge
    da_gauges_t1 = ds_gauges.isel(id=[0, 1], time=0).R

    # Select radar timestep
    da_rad_t = ds_rad.isel(time=0).R

    # Additive
    merger = merge.MergeDifferenceOrdinaryKriging(
        ds_rad=ds_rad,
        ds_gauges=ds_gauges,
        full_line=False,
        variogram_parameters={"sill": 1, "range": 1, "nugget": 0},
        method="additive",
        max_distance=2,
        fill_radar=False,
    )

    # Test that providing only RG works
    merged = merger(
        da_rad_t,
        da_gauges=da_gauges_t1,
    )

    # Gauges located at (1, 1) and (0, 1) are within all cells
    # (by 2 units) for all gridcells, except the lower row y=-1
    assert np.isnan(merged.isel(y=0).data).all()


def test_multiplicative_additiveKriging():
    # CML and rain gauge not overlapping sets
    da_gauges_t1 = ds_gauges.isel(id=[0, 1], time=0).R

    # Select radar timestep
    da_rad_t = ds_rad.isel(time=0).R

    # Additive
    merger = merge.MergeDifferenceOrdinaryKriging(
        ds_rad=ds_rad,
        ds_gauges=ds_gauges,
        full_line=False,
        variogram_parameters={"sill": 1, "range": 1, "nugget": 0},
        method="additive",
    )

    # Test that providing only RG works
    merged = merger(
        da_rad_t,
        da_gauges=da_gauges_t1,
    )
    for gauge_id in da_gauges_t1.id:
        merge_r = merged.sel(
            x=da_gauges_t1.sel(id=gauge_id).x.data,
            y=da_gauges_t1.sel(id=gauge_id).y.data,
        ).data
        gauge_r = da_gauges_t1.sel(id=gauge_id).data
        np.testing.assert_almost_equal(
            merge_r,
            gauge_r,
            decimal=6,
        )

    # Multiplicative
    merger = merge.MergeDifferenceOrdinaryKriging(
        ds_rad=ds_rad,
        ds_gauges=ds_gauges,
        method="multiplicative",
        variogram_parameters={"sill": 1, "range": 1, "nugget": 0},
        full_line=False,
    )

    # Test that providing only RG works
    merged = merger(
        da_rad_t,
        da_gauges=da_gauges_t1,
    )
    for gauge_id in da_gauges_t1.id:
        merge_r = merged.sel(
            x=da_gauges_t1.sel(id=gauge_id).x.data,
            y=da_gauges_t1.sel(id=gauge_id).y.data,
        ).data
        gauge_r = da_gauges_t1.sel(id=gauge_id).data
        np.testing.assert_almost_equal(
            merge_r,
            gauge_r,
            decimal=6,
        )

    # Check that invalid merger causes ValueError
    merger = merge.MergeDifferenceOrdinaryKriging(
        ds_rad=ds_rad,
        ds_gauges=ds_gauges,
        method="not_valid_method",
    )
    msg = "Method must be multiplicative or additive"
    with pytest.raises(ValueError, match=msg):
        merged = merger(
            da_rad_t,
            da_gauges=da_gauges_t1,
        )


def test_multiplicative_additiveIDW():
    # CML and rain gauge not overlapping sets
    da_gauges_t1 = ds_gauges.isel(id=[0, 1], time=0).R

    # Select radar timestep
    da_rad_t = ds_rad.isel(time=0).R

    # Additive
    merger = merge.MergeDifferenceIDW(
        ds_rad=ds_rad,
        ds_gauges=ds_gauges,
        method="additive",
    )

    # Test that providing only RG works
    merged = merger(
        da_rad_t,
        da_gauges=da_gauges_t1,
    )
    for gauge_id in da_gauges_t1.id:
        merge_r = merged.sel(
            x=da_gauges_t1.sel(id=gauge_id).x.data,
            y=da_gauges_t1.sel(id=gauge_id).y.data,
        ).data
        gauge_r = da_gauges_t1.sel(id=gauge_id).data
        np.testing.assert_almost_equal(
            merge_r,
            gauge_r,
            decimal=6,
        )

    # Multiplicative
    merger = merge.MergeDifferenceIDW(
        ds_rad=ds_rad,
        ds_gauges=ds_gauges,
        method="multiplicative",
    )

    # Test that providing only RG works
    merged = merger(
        da_rad_t,
        da_gauges=da_gauges_t1,
    )
    for gauge_id in da_gauges_t1.id:
        merge_r = merged.sel(
            x=da_gauges_t1.sel(id=gauge_id).x.data,
            y=da_gauges_t1.sel(id=gauge_id).y.data,
        ).data
        gauge_r = da_gauges_t1.sel(id=gauge_id).data
        np.testing.assert_almost_equal(
            merge_r,
            gauge_r,
            decimal=6,
        )

    # Check that invalid merger causes ValueError
    merger = merge.MergeDifferenceIDW(
        ds_rad=ds_rad,
        ds_gauges=ds_gauges,
        method="not_valid_method",
    )
    msg = "Method must be multiplicative or additive"
    with pytest.raises(ValueError, match=msg):
        merged = merger(
            da_rad_t,
            da_gauges=da_gauges_t1,
        )


def test_obk_filter():
    # Test that large differences from radar and obs is removved
    # CML and rain gauge overlapping sets
    da_gauges_t = ds_gauges.isel(id=[1, 2], time=0).R

    # Set one unrealistically high rain gauge value
    da_gauges_t.data = np.array([1, 40])

    # Select radar timestep
    da_rad_t = ds_rad.isel(time=0).R

    # Initialize highlevel-class
    merger = merge.MergeDifferenceOrdinaryKriging(
        ds_rad=da_rad_t,
        ds_gauges=da_gauges_t,
        nnear=8,
        min_observations=1,
        method="additive",
        range_checks={"diff_check": 10.0},
    )

    # Adjust field
    adjusted = merger(
        da_rad=da_rad_t,
        da_gauges=da_gauges_t,
    )

    # Test that first obs is accounted for
    merge_r = adjusted.sel(
        x=da_gauges_t.isel(id=0).x,
        y=da_gauges_t.isel(id=0).y,
    ).data
    cml_r = da_gauges_t.isel(id=0).data
    assert cml_r == merge_r

    # Test that second obs is ignored
    merge_r = adjusted.sel(
        x=da_gauges_t.isel(id=1).x,
        y=da_gauges_t.isel(id=1).y,
    ).data
    cml_r = da_gauges_t.isel(id=1).data
    assert cml_r != merge_r

    # Initialize highlevel-class
    merger = merge.MergeDifferenceIDW(
        ds_rad=da_rad_t,
        ds_gauges=da_gauges_t,
        nnear=8,
        min_observations=1,
        method="multiplicative",
        range_checks={"ratio_check": (0.1, 15)},
    )

    # Adjust field
    adjusted = merger(
        da_rad=da_rad_t,
        da_gauges=da_gauges_t,
    )

    # Test that first obs is accounted for
    merge_r = adjusted.sel(
        x=da_gauges_t.isel(id=0).x,
        y=da_gauges_t.isel(id=0).y,
    ).data
    cml_r = da_gauges_t.isel(id=0).data
    assert cml_r == merge_r

    # Test that second obs is ignored
    merge_r = adjusted.sel(
        x=da_gauges_t.isel(id=1).x,
        y=da_gauges_t.isel(id=1).y,
    ).data
    cml_r = da_gauges_t.isel(id=1).data
    assert cml_r != merge_r


def test_idw_filter():
    # Test that large differences from radar and obs is removved
    # CML and rain gauge overlapping sets
    da_gauges_t = ds_gauges.isel(id=[1, 2], time=0).R

    # Set one unrealistically high rain gauge value
    da_gauges_t.data = np.array([1, 40])

    # Select radar timestep
    da_rad_t = ds_rad.isel(time=0).R

    # Initialize highlevel-class
    merger = merge.MergeDifferenceIDW(
        ds_rad=da_rad_t,
        ds_gauges=da_gauges_t,
        nnear=8,
        min_observations=1,
        method="additive",
        range_checks={"diff_check": 10.0},
    )

    # Adjust field
    adjusted = merger(
        da_rad=da_rad_t,
        da_gauges=da_gauges_t,
    )

    # Test that first obs is accounted for
    merge_r = adjusted.sel(
        x=da_gauges_t.isel(id=0).x,
        y=da_gauges_t.isel(id=0).y,
    ).data
    cml_r = da_gauges_t.isel(id=0).data
    assert cml_r == merge_r

    # Test that second obs is ignored
    merge_r = adjusted.sel(
        x=da_gauges_t.isel(id=1).x,
        y=da_gauges_t.isel(id=1).y,
    ).data
    cml_r = da_gauges_t.isel(id=1).data
    assert cml_r != merge_r

    # Initialize highlevel-class
    merger = merge.MergeDifferenceIDW(
        ds_rad=da_rad_t,
        ds_gauges=da_gauges_t,
        nnear=8,
        min_observations=1,
        method="multiplicative",
        range_checks={"ratio_check": (0.1, 15)},
    )

    # Adjust field
    adjusted = merger(
        da_rad=da_rad_t,
        da_gauges=da_gauges_t,
    )

    # Test that first obs is accounted for
    merge_r = adjusted.sel(
        x=da_gauges_t.isel(id=0).x,
        y=da_gauges_t.isel(id=0).y,
    ).data
    cml_r = da_gauges_t.isel(id=0).data
    assert cml_r == merge_r

    # Test that second obs is ignored
    merge_r = adjusted.sel(
        x=da_gauges_t.isel(id=1).x,
        y=da_gauges_t.isel(id=1).y,
    ).data
    cml_r = da_gauges_t.isel(id=1).data
    assert cml_r != merge_r


def test_MergeDifferenceIDW():
    # CML and rain gauge overlapping sets
    da_cml_t1 = ds_cmls.isel(cml_id=[2, 1], time=0).R
    da_cml_t2 = ds_cmls.isel(cml_id=[1, 0], time=0).R
    da_gauges_t1 = ds_gauges.isel(id=[2, 1], time=0).R
    da_gauges_t2 = ds_gauges.isel(id=[1, 0], time=0).R

    # Select radar timestep
    da_rad_t = ds_rad.isel(time=0).R

    # Initialize highlevel-class
    merger = merge.MergeDifferenceIDW(
        ds_rad=ds_rad,
        ds_cmls=ds_cmls,
        ds_gauges=ds_gauges,
        nnear=8,
        min_observations=1,
        method="additive",
    )

    # Adjust field
    adjusted = merger(
        da_rad=da_rad_t,
        da_cmls=da_cml_t1,
        da_gauges=da_gauges_t1,
    )

    # Check CMLs
    for cml_id in da_cml_t1.cml_id:
        merge_r = adjusted.sel(
            x=da_cml_t1.sel(cml_id=cml_id).x,
            y=da_cml_t1.sel(cml_id=cml_id).y,
        ).data
        cml_r = da_cml_t1.sel(cml_id=cml_id).data
        assert cml_r == merge_r

    # Check gauges
    for id in da_gauges_t1.id:
        merge_r = adjusted.sel(
            x=da_gauges_t1.sel(id=id).x,
            y=da_gauges_t1.sel(id=id).y,
        ).data
        gauge_r = da_gauges_t1.sel(id=id).data
        assert gauge_r == merge_r

    # Update the weights using some new links
    adjusted = merger(
        da_rad=da_rad_t,
        da_cmls=da_cml_t2,
        da_gauges=da_gauges_t2,
    )

    # Test that CML names is correctly updated and sorted in the class
    assert (merger.intersect_weights.cml_id == da_cml_t2.cml_id).all()

    # Test that coordinates are updated
    assert (merger.x_0_cml == da_cml_t2.site_0_x).all()
    assert (merger.x_1_cml == da_cml_t2.site_1_x).all()
    assert (merger.y_0_cml == da_cml_t2.site_0_y).all()
    assert (merger.y_1_cml == da_cml_t2.site_1_y).all()
    assert (merger.x_gauge == da_gauges_t2.x).all()
    assert (merger.y_gauge == da_gauges_t2.y).all()

    # Check that field is fit to CMLs using updated weights and
    # multiplicative adjustment
    for cml_id in da_cml_t2.cml_id:
        merge_r = adjusted.sel(
            x=da_cml_t2.sel(cml_id=cml_id).x,
            y=da_cml_t2.sel(cml_id=cml_id).y,
        ).data
        cml_r = da_cml_t2.sel(cml_id=cml_id).data
        assert cml_r == merge_r

    # Check that field is fit to gauges
    for id in da_gauges_t2.id:
        merge_r = adjusted.sel(
            x=da_gauges_t2.sel(id=id).x,
            y=da_gauges_t2.sel(id=id).y,
        ).data
        gauge_r = da_gauges_t2.sel(id=id).data
        assert gauge_r == merge_r

    # Update with gauges only
    adjusted = merger(
        da_rad=da_rad_t,
        da_gauges=da_gauges_t2,
    )

    # Check that field is fit to gauges
    for id in da_gauges_t2.id:
        merge_r = adjusted.sel(
            x=da_gauges_t2.sel(id=id).x,
            y=da_gauges_t2.sel(id=id).y,
        ).data
        gauge_r = da_gauges_t2.sel(id=id).data
        assert gauge_r == merge_r


def test_MergeDifferenceOrdinaryKriging_c0():
    # CML and rain gauge overlapping sets
    ds_cmls_t = ds_cmls.isel(cml_id=[0], time=0)
    ds_gauges_t = ds_gauges.isel(id=[1], time=0)

    # Select radar timestep
    ds_rad_t = ds_rad.isel(time=0)

    # Initialize highlevel-class
    merger = merge.MergeDifferenceOrdinaryKriging(
        ds_rad=ds_rad_t,
        ds_cmls=ds_cmls_t,
        ds_gauges=ds_gauges_t,
        nnear=8,
        min_observations=1,
        method="additive",
        discretization=20,
        variogram_model="spherical",
        variogram_parameters={"sill": 1, "range": 1, "nugget": 0},
        c0_within=False,
    )

    # Adjust field
    adjusted = merger(
        da_rad=ds_rad_t.R,
        da_cmls=ds_cmls_t.R,
        da_gauges=ds_gauges_t.R,
    )

    # test that the adjusted field is the same as first run
    data_check = np.array(
        [
            [1.1788793, 1.5957385, 1.6926061, 1.6926061],
            [1.5957385, 0.8272601, 1.5957385, 1.6926061],
            [1.6926061, 5.0, 1.1788793, 1.6926061],
            [1.6926061, 1.6926061, 1.6926061, 1.6926061],
        ]
    )

    np.testing.assert_almost_equal(adjusted, data_check)


def test_MergeDifferenceOrdinaryKriging():
    # CML and rain gauge overlapping sets
    ds_cmls_t = ds_cmls.isel(cml_id=[0], time=0)
    ds_gauges_t = ds_gauges.isel(id=[1], time=0)
    ds_cmls_t2 = ds_cmls.isel(cml_id=[2, 1], time=0)
    ds_gauges_t2 = ds_gauges.isel(id=[2], time=0)

    # Select radar timestep
    ds_rad_t = ds_rad.isel(time=0)

    # Initialize highlevel-class
    merger = merge.MergeDifferenceOrdinaryKriging(
        ds_rad=ds_rad_t,
        ds_cmls=ds_cmls_t,
        ds_gauges=ds_gauges_t,
        nnear=8,
        min_observations=1,
        method="additive",
        discretization=40,
        variogram_model="spherical",
        variogram_parameters={"sill": 1, "range": 1, "nugget": 0},
        c0_within=True,
    )

    # Adjust field
    adjusted = merger(
        da_rad=ds_rad_t.R,
        da_cmls=ds_cmls_t.R,
        da_gauges=ds_gauges_t.R,
    )

    # Test that CML names is correctly updated and sorted in the class
    assert (merger.intersect_weights.cml_id == ds_cmls_t.cml_id).all()

    # Test that cml midpoint and end coordinates corresponds to x0_cml
    assert (merger.x_0_cml == ds_cmls_t.site_0_x).all()
    assert (merger.x_1_cml == ds_cmls_t.site_1_x).all()
    assert (merger.y_0_cml == ds_cmls_t.site_0_y).all()
    assert (merger.y_1_cml == ds_cmls_t.site_1_y).all()

    # Test that gauge midpoint coordinates corresponds to x0_gauge
    assert (merger.x_gauge == ds_gauges_t.x).all()
    assert (merger.y_gauge == ds_gauges_t.y).all()

    # test that the adjusted field is the same as first run
    data_check = np.array(
        [
            [2.7079736, 2.9399488, 3.0, 3.0],
            [2.9399488, 2.4661924, 2.9399488, 3.0],
            [3.0, 5.0, 2.7079736, 3.0],
            [3.0, 3.0, 3.0, 3.0],
        ]
    )

    np.testing.assert_almost_equal(adjusted, data_check)

    # Long CMLs average large areas; under stationarity, they
    # observe the mean with uncertainty. The field need not
    # match the CML exactly, so we skip path-wise agreement tests.

    # Test adjusted field at rain gauges
    for gauge_id in ds_gauges_t.id:
        merge_r = adjusted.sel(
            x=ds_gauges_t.sel(id=gauge_id).x.data,
            y=ds_gauges_t.sel(id=gauge_id).y.data,
        ).data
        gauge_r = ds_gauges_t.sel(id=gauge_id).R.data
        np.testing.assert_almost_equal(merge_r, gauge_r, decimal=8)

    # Adjust field
    adjusted = merger(
        da_rad=ds_rad_t.R,
        da_cmls=ds_cmls_t2.R,
        da_gauges=ds_gauges_t2.R,
    )

    # Test that CML names is correctly updated and sorted in the class
    assert (merger.intersect_weights.cml_id == ds_cmls_t2.cml_id).all()

    # Test that cml midpoint and end coordinates corresponds to x0_cml
    assert (merger.x_0_cml == ds_cmls_t2.site_0_x).all()
    assert (merger.x_1_cml == ds_cmls_t2.site_1_x).all()
    assert (merger.y_0_cml == ds_cmls_t2.site_0_y).all()
    assert (merger.y_1_cml == ds_cmls_t2.site_1_y).all()

    # Test that gauge midpoint coordinates corresponds to x0_gauge
    assert (merger.x_gauge == ds_gauges_t2.x).all()
    assert (merger.y_gauge == ds_gauges_t2.y).all()

    # test that the adjusted field is the same as first run
    data_check = np.array(
        [
            [7.6986128, 7.3003201, 7.6167094, 7.6986128],
            [7.6986128, 7.6167094, 7.0145239, 7.8305215],
            [7.6986128, 7.7425803, 8.0075458, 7.3442876],
            [7.6986128, 7.9124249, 7.7425803, 9.0],
        ]
    )

    np.testing.assert_almost_equal(adjusted, data_check)

    # Test adjusted field at rain gauges
    for gauge_id in ds_gauges_t2.id:
        merge_r = adjusted.sel(
            x=ds_gauges_t2.sel(id=gauge_id).x.data,
            y=ds_gauges_t2.sel(id=gauge_id).y.data,
        ).data
        gauge_r = ds_gauges_t2.sel(id=gauge_id).R.data
        np.testing.assert_almost_equal(
            merge_r,
            gauge_r,
            decimal=8,  # not very precise, but decent
        )


def test_MergeBlockKrigingExternalDrift_c0():
    # CML and rain gauge overlapping sets
    ds_cmls_t = ds_cmls.isel(cml_id=[0], time=0)
    ds_gauges_t = ds_gauges.isel(id=[1, 2], time=0)

    # Select radar timestep
    ds_rad_t = ds_rad.isel(time=0).copy()

    # Set some drift so that matrix is not singular
    ds_rad_t["R"].data = np.array(
        [
            [2.7079736, 2.9399488, 3.0, 3.0],
            [2.9399488, 2.4661924, 2.9399488, 3.0],
            [3.0, 5.0, 2.7079736, 3.0],
            [3.0, 3.0, 3.0, 3.0],
        ]
    )

    # Initialize highlevel-class
    merger = merge.MergeKrigingExternalDrift(
        ds_rad=ds_rad_t,
        ds_cmls=ds_cmls_t,
        ds_gauges=ds_gauges_t,
        nnear=8,
        min_observations=1,
        discretization=8,
        variogram_parameters={"sill": 1, "range": 2, "nugget": 0},
        c0_within=True,
    )

    # Adjust field
    adjusted = merger(
        da_rad=ds_rad_t.R,
        da_cmls=ds_cmls_t.R,
        da_gauges=ds_gauges_t.R,
    )

    # test that the adjusted field is the same as first run
    data_check = np.array(
        [
            [3.6450039, 3.9961395, 4.7164175, 4.8992262],
            [3.9136239, 2.6078412, 3.9136239, 4.7856179],
            [4.4943462, 5.0, 3.9051202, 5.9863823],
            [4.8167106, 4.5635466, 5.9038667, 9.0],
        ]
    )

    np.testing.assert_almost_equal(adjusted, data_check)


def test_MergeBlockKrigingExternalDrift():
    # CML and rain gauge overlapping sets
    ds_cmls_t = ds_cmls.isel(cml_id=[0], time=0)
    ds_gauges_t = ds_gauges.isel(id=[1], time=0)
    ds_cmls_t2 = ds_cmls.isel(cml_id=[2, 1], time=0)
    ds_gauges_t2 = ds_gauges.isel(id=[2], time=0)

    # Select radar timestep
    ds_rad_t = ds_rad.isel(time=0).copy()

    # Set some drift so that matrix is not singular
    ds_rad_t["R"].data = np.array(
        [
            [2.7079736, 2.9399488, 3.0, 3.0],
            [2.9399488, 2.4661924, 2.9399488, 3.0],
            [3.0, 5.0, 2.7079736, 3.0],
            [3.0, 3.0, 3.0, 3.0],
        ]
    )

    # Initialize highlevel-class
    merger = merge.MergeKrigingExternalDrift(
        ds_rad=ds_rad_t,
        ds_cmls=ds_cmls_t,
        ds_gauges=ds_gauges_t,
        nnear=8,
        min_observations=1,
        discretization=8,
        variogram_parameters={"sill": 1, "range": 2, "nugget": 0},
        c0_within=True,
    )

    # Adjust field
    adjusted = merger(
        da_rad=ds_rad_t.R,
        da_cmls=ds_cmls_t.R,
        da_gauges=ds_gauges_t.R,
    )

    # Test that CML names is correctly updated and sorted in the class
    assert (merger.intersect_weights.cml_id == ds_cmls_t.cml_id).all()

    # Test that cml midpoint and end coordinates corresponds to x0_cml
    assert (merger.x_0_cml == ds_cmls_t.site_0_x).all()
    assert (merger.x_1_cml == ds_cmls_t.site_1_x).all()
    assert (merger.y_0_cml == ds_cmls_t.site_0_y).all()
    assert (merger.y_1_cml == ds_cmls_t.site_1_y).all()

    # Test that gauge midpoint coordinates corresponds to x0_gauge
    assert (merger.x_gauge == ds_gauges_t.x).all()
    assert (merger.y_gauge == ds_gauges_t.y).all()

    # test that the adjusted field is the same as first run
    data_check = np.array(
        [
            [1.2004057, 1.5849614, 1.6845109, 1.6845109],
            [1.5849614, 0.7995943, 1.5849614, 1.6845109],
            [1.6845109, 5.0, 1.2004057, 1.6845109],
            [1.6845109, 1.6845109, 1.6845109, 1.6845109],
        ]
    )

    np.testing.assert_almost_equal(adjusted, data_check)

    # Long CMLs average large areas; under stationarity, they
    # observe the mean with uncertainty. The field need not
    # match the CML exactly, so we skip path-wise agreement tests.

    # Test adjusted field at rain gauges
    for gauge_id in ds_gauges_t.id:
        merge_r = adjusted.sel(
            x=ds_gauges_t.sel(id=gauge_id).x.data,
            y=ds_gauges_t.sel(id=gauge_id).y.data,
        ).data
        gauge_r = ds_gauges_t.R.sel(id=gauge_id).data
        np.testing.assert_almost_equal(merge_r, gauge_r, decimal=8)

    # Create a new drift field for testing
    ds_rad_t["R"].data = np.array(
        [
            [7.6986128, 7.3003201, 7.6167094, 7.6986128],
            [7.6986128, 7.6167094, 7.0145239, 7.8305215],
            [7.6986128, 7.7425803, 8.0075458, 7.3442876],
            [7.6986128, 7.9124249, 7.7425803, 9.0],
        ]
    )

    # Update the weights using some new links
    adjusted = merger(
        da_rad=ds_rad_t.R,
        da_cmls=ds_cmls_t2.R,
        da_gauges=ds_gauges_t2.R,
    )

    # Test that CML names is correctly updated and sorted in the class
    assert (merger.intersect_weights.cml_id == ds_cmls_t2.cml_id).all()

    # Test that cml midpoint and end coordinates corresponds to x0_cml
    assert (merger.x_0_cml == ds_cmls_t2.site_0_x).all()
    assert (merger.x_1_cml == ds_cmls_t2.site_1_x).all()
    assert (merger.y_0_cml == ds_cmls_t2.site_0_y).all()
    assert (merger.y_1_cml == ds_cmls_t2.site_1_y).all()

    # Test that gauge midpoint coordinates corresponds to x0_gauge
    assert (merger.x_gauge == ds_gauges_t2.x).all()
    assert (merger.y_gauge == ds_gauges_t2.y).all()

    # test that the adjusted field is the same as first run
    data_check = np.array(
        [
            [6.8297455, 5.711312, 6.512705, 6.950672],
            [6.8596963, 6.5496356, 5.3518979, 7.4593091],
            [6.9554314, 7.3897511, 8.084257, 6.021082],
            [7.0071598, 7.9020355, 7.1694663, 9.0],
        ]
    )

    np.testing.assert_almost_equal(adjusted, data_check)

    # Long CMLs average large areas; under stationarity, they
    # observe the mean with uncertainty. The field need not
    # match the CML exactly, so we skip path-wise agreement tests.

    # Test adjusted field at rain gauges
    for gauge_id in ds_gauges_t2.id:
        merge_r = adjusted.sel(
            x=ds_gauges_t2.sel(id=gauge_id).x.data,
            y=ds_gauges_t2.sel(id=gauge_id).y.data,
        ).data
        gauge_r = ds_gauges_t2.sel(id=gauge_id).R.data
        np.testing.assert_almost_equal(merge_r, gauge_r, decimal=8)
