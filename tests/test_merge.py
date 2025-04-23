from __future__ import annotations

import numpy as np
import poligrain as plg
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
        min_observations=2,
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


def test_MergeDifferenceOrdinaryKriging():
    # CML and rain gauge overlapping sets
    da_cml_t1 = ds_cmls.isel(cml_id=[2, 1], time=0).R
    da_cml_t2 = ds_cmls.isel(cml_id=[1, 0], time=0).R
    da_gauges_t1 = ds_gauges.isel(id=[2, 1], time=0).R
    da_gauges_t2 = ds_gauges.isel(id=[1, 0], time=0).R

    # Select radar timestep
    da_rad_t = ds_rad.isel(time=0).R

    # Initialize highlevel-class
    merger = merge.MergeDifferenceOrdinaryKriging(
        ds_rad=ds_rad,
        ds_cmls=ds_cmls,
        ds_gauges=ds_gauges,
        nnear=8,
        min_observations=2,
        method="additive",
        discretization=8,
        variogram_parameters={"sill": 1, "range": 1, "nugget": 0.1},
    )

    # Adjust field
    adjusted = merger(
        da_rad=da_rad_t,
        da_cmls=da_cml_t1,
        da_gauges=da_gauges_t1,
    )

    # Test that CML names is correctly updated and sorted in the class
    assert (merger.intersect_weights.cml_id == da_cml_t1.cml_id).all()

    # Test that cml midpoint and end coordinates corresponds to x0_cml
    assert (merger.x_0_cml == da_cml_t1.site_0_x).all()
    assert (merger.x_1_cml == da_cml_t1.site_1_x).all()
    assert (merger.y_0_cml == da_cml_t1.site_0_y).all()
    assert (merger.y_1_cml == da_cml_t1.site_1_y).all()

    # Test that gauge midpoint coordinates corresponds to x0_gauge
    assert (merger.x_gauge == da_gauges_t1.x).all()
    assert (merger.y_gauge == da_gauges_t1.y).all()

    # test that the adjusted field is the same as first run
    data_check = np.array(
        [
            [7.0564348, 5.7576878, 6.8637781, 7.0564348],
            [7.0564348, 6.8637781, 5.4636964, 8.1919296],
            [7.0564348, 5.2825001, 8.6940564, 5.9547064],
            [7.0564348, 8.3845863, 7.2534534, 8.8627983],
        ]
    )

    np.testing.assert_almost_equal(adjusted, data_check)

    # calculate the adjusted field along CMLs
    intersect_weights = plg.spatial.calc_sparse_intersect_weights_for_several_cmls(
        x1_line=da_cml_t1.site_0_x.data,
        y1_line=da_cml_t1.site_0_y.data,
        x2_line=da_cml_t1.site_1_x.data,
        y2_line=da_cml_t1.site_1_y.data,
        cml_id=da_cml_t1.cml_id.data,
        x_grid=adjusted.x_grid.data,
        y_grid=adjusted.y_grid.data,
        grid_point_location="center",
    )
    adjusted_at_cmls = plg.spatial.get_grid_time_series_at_intersections(
        grid_data=adjusted.expand_dims("time"),
        intersect_weights=intersect_weights,
    )

    # As the block kriging adjustment uses the variogram to fit the rainfall field,
    # the get_grid_time_series_at_intersections uses the grid intersections and
    # the grid is discretized, the adjusted radar along will not perfectly fit the
    # CML observation. Thus we only test up to a certain decimal place.

    np.testing.assert_almost_equal(
        adjusted_at_cmls.data.ravel(),
        da_cml_t1.data.ravel(),
        decimal=0,  # not very precise, but decent
    )

    # Test adjusted field at rain gauges
    for gauge_id in da_gauges_t1.id:
        merge_r = adjusted.sel(
            x=da_gauges_t1.sel(id=gauge_id).x.data,
            y=da_gauges_t1.sel(id=gauge_id).y.data,
        ).data
        gauge_r = da_gauges_t1.sel(id=gauge_id).data
        np.testing.assert_almost_equal(merge_r, gauge_r, decimal=0)

    # Adjust field
    adjusted = merger(
        da_rad=da_rad_t,
        da_cmls=da_cml_t2,
        da_gauges=da_gauges_t2,
    )

    # Test that CML names is correctly updated and sorted in the class
    assert (merger.intersect_weights.cml_id == da_cml_t2.cml_id).all()

    # Test that cml midpoint and end coordinates corresponds to x0_cml
    assert (merger.x_0_cml == da_cml_t2.site_0_x).all()
    assert (merger.x_1_cml == da_cml_t2.site_1_x).all()
    assert (merger.y_0_cml == da_cml_t2.site_0_y).all()
    assert (merger.y_1_cml == da_cml_t2.site_1_y).all()

    # Test that gauge midpoint coordinates corresponds to x0_gauge
    assert (merger.x_gauge == da_gauges_t2.x).all()
    assert (merger.y_gauge == da_gauges_t2.y).all()

    # test that the adjusted field is the same as first run
    data_check = np.array(
        [
            [1.8034557, 3.807717, 3.1030118, 2.9506274],
            [2.7804554, 1.5221354, 4.1960881, 3.1030118],
            [2.9506274, 4.5449719, 0.8625104, 3.977889],
            [2.9506274, 2.9506274, 2.9506274, 2.9506274],
        ]
    )

    np.testing.assert_almost_equal(adjusted, data_check)

    # calculate the adjusted field along CMLs
    intersect_weights = plg.spatial.calc_sparse_intersect_weights_for_several_cmls(
        x1_line=da_cml_t2.site_0_x.data,
        y1_line=da_cml_t2.site_0_y.data,
        x2_line=da_cml_t2.site_1_x.data,
        y2_line=da_cml_t2.site_1_y.data,
        cml_id=da_cml_t2.cml_id.data,
        x_grid=adjusted.x_grid.data,
        y_grid=adjusted.y_grid.data,
        grid_point_location="center",
    )
    adjusted_at_cmls = plg.spatial.get_grid_time_series_at_intersections(
        grid_data=adjusted.expand_dims("time"),
        intersect_weights=intersect_weights,
    )

    # As the block kriging adjustment uses the variogram to fit the rainfall field,
    # the get_grid_time_series_at_intersections uses the grid intersections and
    # the grid is discretized, the adjusted radar along will not perfectly fit the
    # CML observation. Thus we only test up to a certain decimal place.
    np.testing.assert_almost_equal(
        adjusted_at_cmls.data.ravel(),
        da_cml_t2.data.ravel(),
        decimal=0,  # not very precise, but decent
    )

    # Test adjusted field at rain gauges
    for gauge_id in da_gauges_t2.id:
        merge_r = adjusted.sel(
            x=da_gauges_t2.sel(id=gauge_id).x.data,
            y=da_gauges_t2.sel(id=gauge_id).y.data,
        ).data
        gauge_r = da_gauges_t2.sel(id=gauge_id).data
        np.testing.assert_almost_equal(
            merge_r,
            gauge_r,
            decimal=0,  # not very precise, but decent
        )


def test_MergeBlockKrigingExternalDrift():
    # CML and rain gauge overlapping sets
    da_cml_t1 = ds_cmls.isel(cml_id=[2, 1], time=0).R
    da_cml_t2 = ds_cmls.isel(cml_id=[1, 0], time=0).R
    da_gauges_t1 = ds_gauges.isel(id=[2, 1], time=0).R
    da_gauges_t2 = ds_gauges.isel(id=[1, 0], time=0).R

    # Select radar timestep
    da_rad_t = ds_rad.isel(time=0).R.copy()

    # Set some drift so that matrix is not singular
    da_rad_t.data = np.array(
        [
            [7.0564348, 5.7576878, 6.8637781, 7.0564348],
            [7.0564348, 6.8637781, 5.4636964, 8.1919296],
            [7.0564348, 5.2825001, 8.6940564, 5.9547064],
            [7.0564348, 8.3845863, 7.2534534, 8.8627983],
        ]
    )

    # Initialize highlevel-class
    merger = merge.MergeKrigingExternalDrift(
        ds_rad=ds_rad,
        ds_cmls=ds_cmls,
        ds_gauges=ds_gauges,
        nnear=8,
        min_observations=1,
        discretization=8,
        variogram_parameters={"sill": 1, "range": 1, "nugget": 0.1},
    )

    # Adjust field
    adjusted = merger(
        da_rad=da_rad_t,
        da_cmls=da_cml_t1,
        da_gauges=da_gauges_t1,
    )

    # Test that CML names is correctly updated and sorted in the class
    assert (merger.intersect_weights.cml_id == da_cml_t1.cml_id).all()

    # Test that cml midpoint and end coordinates corresponds to x0_cml
    assert (merger.x_0_cml == da_cml_t1.site_0_x).all()
    assert (merger.x_1_cml == da_cml_t1.site_1_x).all()
    assert (merger.y_0_cml == da_cml_t1.site_0_y).all()
    assert (merger.y_1_cml == da_cml_t1.site_1_y).all()

    # Test that gauge midpoint coordinates corresponds to x0_gauge
    assert (merger.x_gauge == da_gauges_t1.x).all()
    assert (merger.y_gauge == da_gauges_t1.y).all()

    # test that the adjusted field is the same as first run
    data_check = np.array(
        [
            [6.9695107, 5.169682, 6.7025233, 6.9695107],
            [6.9695107, 6.7025233, 4.7619131, 8.5407423],
            [6.9695107, 4.9613212, 9.2357075, 5.4423643],
            [6.9695107, 8.8077297, 7.2421929, 9.0354948],
        ]
    )

    np.testing.assert_almost_equal(adjusted, data_check)

    # calculate the adjusted field along CMLs
    intersect_weights = plg.spatial.calc_sparse_intersect_weights_for_several_cmls(
        x1_line=da_cml_t1.site_0_x.data,
        y1_line=da_cml_t1.site_0_y.data,
        x2_line=da_cml_t1.site_1_x.data,
        y2_line=da_cml_t1.site_1_y.data,
        cml_id=da_cml_t1.cml_id.data,
        x_grid=adjusted.x_grid.data,
        y_grid=adjusted.y_grid.data,
        grid_point_location="center",
    )
    adjusted_at_cmls = plg.spatial.get_grid_time_series_at_intersections(
        grid_data=adjusted.expand_dims("time"),
        intersect_weights=intersect_weights,
    )

    # As the block kriging adjustment uses the variogram to fit the rainfall field,
    # the get_grid_time_series_at_intersections uses the grid intersections and
    # the grid is discretized, the adjusted radar along will not perfectly fit the
    # CML observation. Thus we only test up to a certain decimal place.
    np.testing.assert_almost_equal(
        adjusted_at_cmls.data.ravel(),
        da_cml_t1.data.ravel(),
        decimal=1,  # not very precise, but decent
    )

    # Test adjusted field at rain gauges
    for gauge_id in da_gauges_t1.id:
        merge_r = adjusted.sel(
            x=da_gauges_t1.sel(id=gauge_id).x.data,
            y=da_gauges_t1.sel(id=gauge_id).y.data,
        ).data
        gauge_r = da_gauges_t1.sel(id=gauge_id).data
        np.testing.assert_almost_equal(merge_r, gauge_r, decimal=1)

    # New drift field based on other observations
    da_rad_t.data = np.array(
        [
            [1.8034557, 3.807717, 3.1030118, 2.9506274],
            [2.7804554, 1.5221354, 4.1960881, 3.1030118],
            [2.9506274, 4.5449719, 0.8625104, 3.977889],
            [2.9506274, 2.9506274, 2.9506274, 2.9506274],
        ]
    )

    # Update the weights using some new links
    adjusted = merger(
        da_rad=da_rad_t,
        da_cmls=da_cml_t2,
        da_gauges=da_gauges_t2,
    )

    # Test that CML names is correctly updated and sorted in the class
    assert (merger.intersect_weights.cml_id == da_cml_t2.cml_id).all()

    # Test that cml midpoint and end coordinates corresponds to x0_cml
    assert (merger.x_0_cml == da_cml_t2.site_0_x).all()
    assert (merger.x_1_cml == da_cml_t2.site_1_x).all()
    assert (merger.y_0_cml == da_cml_t2.site_0_y).all()
    assert (merger.y_1_cml == da_cml_t2.site_1_y).all()

    # Test that gauge midpoint coordinates corresponds to x0_gauge
    assert (merger.x_gauge == da_gauges_t2.x).all()
    assert (merger.y_gauge == da_gauges_t2.y).all()

    # test that the adjusted field is the same as first run
    data_check = np.array(
        [
            [1.4398873, 4.4498312, 3.4063288, 3.1796229],
            [2.9215497, 1.0088612, 5.0276201, 3.4063288],
            [3.1796229, 4.9455518, 0.8261877, 4.7079044],
            [3.1796229, 3.1796229, 3.1796229, 3.1796229],
        ]
    )

    np.testing.assert_almost_equal(adjusted, data_check)

    # calculate the adjusted field along CMLs
    intersect_weights = plg.spatial.calc_sparse_intersect_weights_for_several_cmls(
        x1_line=da_cml_t2.site_0_x.data,
        y1_line=da_cml_t2.site_0_y.data,
        x2_line=da_cml_t2.site_1_x.data,
        y2_line=da_cml_t2.site_1_y.data,
        cml_id=da_cml_t2.cml_id.data,
        x_grid=adjusted.x_grid.data,
        y_grid=adjusted.y_grid.data,
        grid_point_location="center",
    )
    adjusted_at_cmls = plg.spatial.get_grid_time_series_at_intersections(
        grid_data=adjusted.expand_dims("time"),
        intersect_weights=intersect_weights,
    )

    # As the block kriging adjustment uses the variogram to fit the rainfall field,
    # the get_grid_time_series_at_intersections uses the grid intersections and
    # the grid is discretized, the adjusted radar along will not perfectly fit the
    # CML observation. Thus we only test up to a certain decimal place.
    np.testing.assert_almost_equal(
        adjusted_at_cmls.data.ravel(),
        da_cml_t2.data.ravel(),
        decimal=0,  # not very precise, but decent
    )

    # Test adjusted field at rain gauges
    for gauge_id in da_gauges_t2.id:
        merge_r = adjusted.sel(
            x=da_gauges_t2.sel(id=gauge_id).x.data,
            y=da_gauges_t2.sel(id=gauge_id).y.data,
        ).data
        gauge_r = da_gauges_t2.sel(id=gauge_id).data
        np.testing.assert_almost_equal(merge_r, gauge_r, decimal=0)
