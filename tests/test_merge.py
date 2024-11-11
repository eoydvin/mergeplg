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
        "xs": (("y", "x"), np.meshgrid([-1, 0, 1, 2], [-1, 0, 1, 2])[0]),
        "ys": (("y", "x"), np.meshgrid([-1, 0, 1, 2], [-1, 0, 1, 2])[1]),
    },
)


def test_MergeAdditiveIDW():
    # CML and rain gauge overlapping sets
    da_cml_t1 = ds_cmls.isel(cml_id=[2, 1], time=[0]).R
    da_cml_t2 = ds_cmls.isel(cml_id=[1, 0], time=[0]).R
    da_gauges_t1 = ds_gauges.isel(id=[2, 1], time=[0]).R
    da_gauges_t2 = ds_gauges.isel(id=[1, 0], time=[0]).R

    # Select radar timestep
    da_rad_t = ds_rad.isel(time=[0]).R

    # Initialize highlevel-class
    merge_IDW = merge.MergeAdditiveIDW(min_obs=1)

    # Update geometry to set1
    merge_IDW.update(
        da_rad_t,
        da_cml=da_cml_t1,
        da_gauge=da_gauges_t1,
    )

    # Adjust field
    adjusted = merge_IDW.adjust(
        da_rad_t,
        da_cml=da_cml_t1,
        da_gauge=da_gauges_t1,
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
    merge_IDW.update(
        da_rad_t,
        da_cml=da_cml_t2,
        da_gauge=da_gauges_t2,
    )

    # Test that CML names is correctly updated and sorted in the class
    assert (merge_IDW.intersect_weights.cml_id == da_cml_t2.cml_id).all()

    # Test that midpoint coordinates corresponds to x0_cml and x0_gauge
    assert (merge_IDW.x0_cml.isel(yx=1) == da_cml_t2.x).all()
    assert (merge_IDW.x0_cml.isel(yx=0) == da_cml_t2.y).all()
    assert (merge_IDW.x0_gauge.isel(yx=1) == da_gauges_t2.x).all()
    assert (merge_IDW.x0_gauge.isel(yx=0) == da_gauges_t2.y).all()

    # Adjust field using updated geometry
    adjusted = merge_IDW.adjust(
        da_rad_t,
        da_cml=da_cml_t2,
        da_gauge=da_gauges_t2,
    )

    # Check that field is fit to CMLs
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

    # Test that da_rad is return if too few obs is provided
    merge_IDW.min_obs_ = 10
    adjusted = merge_IDW.adjust(
        da_rad_t,
        da_cml=da_cml_t2,
        da_gauge=None,
    )
    np.testing.assert_almost_equal(
        adjusted.data,
        da_rad_t.data,
    )
    merge_IDW.min_obs_ = 1  # reset


def test_MergeAdditiveBlockKriging():
    # CML and rain gauge overlapping sets
    da_cml_t1 = ds_cmls.isel(cml_id=[2, 1], time=[0]).R
    da_cml_t2 = ds_cmls.isel(cml_id=[1, 0], time=[0]).R
    da_gauges_t1 = ds_gauges.isel(id=[2, 1], time=[0]).R
    da_gauges_t2 = ds_gauges.isel(id=[1, 0], time=[0]).R

    # Select radar timestep
    da_rad_t = ds_rad.isel(time=[0]).R

    # Initialize highlevel-class
    merge_BK = merge.MergeAdditiveBlockKriging(min_obs=1, discretization=8)

    # Update geometry to set1
    merge_BK.update(
        da_rad_t,
        da_cml=da_cml_t1,
        da_gauge=da_gauges_t1,
    )

    # Test that CML names is correctly updated and sorted in the class
    assert (merge_BK.intersect_weights.cml_id == da_cml_t1.cml_id).all()

    # Test that cml midpoint and end coordinates corresponds to x0_cml
    assert (merge_BK.x0_cml.isel(yx=1, discretization=0) == da_cml_t1.site_0_x).all()
    assert (merge_BK.x0_cml.isel(yx=1, discretization=4) == da_cml_t1.x).all()
    assert (merge_BK.x0_cml.isel(yx=1, discretization=8) == da_cml_t1.site_1_x).all()
    assert (merge_BK.x0_cml.isel(yx=0, discretization=0) == da_cml_t1.site_0_y).all()
    assert (merge_BK.x0_cml.isel(yx=0, discretization=4) == da_cml_t1.y).all()
    assert (merge_BK.x0_cml.isel(yx=0, discretization=8) == da_cml_t1.site_1_y).all()

    # Test that gauge midpoint coordinates corresponds to x0_gauge
    assert (merge_BK.x0_gauge.isel(yx=1) == da_gauges_t1.x).all()
    assert (merge_BK.x0_gauge.isel(yx=0) == da_gauges_t1.y).all()

    # Test that automatic variogram estimation works
    adjusted = merge_BK.adjust(
        da_rad_t, da_cml=da_cml_t1, da_gauge=da_gauges_t1, variogram="exponential"
    )

    # test that the adjusted field is the same as first run
    data_check = np.array(
        [
            [6.8837209, 4.7383721, 6.8837209, 6.8837209],
            [6.8837209, 6.8837209, 4.7383721, 9.2616279],
            [6.8837209, 5.0, 9.2383721, 4.7616279],
            [6.8837209, 9.0, 7.1162791, 9.0],
        ]
    )

    np.testing.assert_almost_equal(adjusted, data_check)

    # Test the rest of the code with a simple variogram
    def variogram(h):
        return h

    # Adjust field
    adjusted = merge_BK.adjust(
        da_rad_t, da_cml=da_cml_t1, da_gauge=da_gauges_t1, variogram=variogram
    )

    # test that the adjusted field is the same as first run
    data_check = np.array(
        [
            [1.2673279, 0.9159766, 2.8557292, 5.2944255],
            [2.5462022, 2.7732177, 4.7644577, 6.2127718],
            [4.7542038, 5.0, 9.1808035, 7.4418168],
            [7.4271479, 7.2153635, 10.724253, 9.0],
        ]
    )

    np.testing.assert_almost_equal(adjusted, data_check)

    # calculate the adjusted field along CMLs
    intersect_weights = plg.spatial.calc_sparse_intersect_weights_for_several_cmls(
        x1_line=da_cml_t1.site_0_lon.data,
        y1_line=da_cml_t1.site_0_lat.data,
        x2_line=da_cml_t1.site_1_lon.data,
        y2_line=da_cml_t1.site_1_lat.data,
        cml_id=da_cml_t1.cml_id.data,
        x_grid=adjusted.lon.data,
        y_grid=adjusted.lat.data,
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
        np.testing.assert_almost_equal(
            merge_r,
            gauge_r,
        )

    # Update the weights using some new links
    merge_BK.update(
        da_rad_t,
        da_cml=da_cml_t2,
        da_gauge=da_gauges_t2,
    )

    # Test that CML names is correctly updated and sorted in the class
    assert (merge_BK.intersect_weights.cml_id == da_cml_t2.cml_id).all()

    # Test that cml midpoint and end coordinates corresponds to x0_cml
    assert (merge_BK.x0_cml.isel(yx=1, discretization=0) == da_cml_t2.site_0_x).all()
    assert (merge_BK.x0_cml.isel(yx=1, discretization=4) == da_cml_t2.x).all()
    assert (merge_BK.x0_cml.isel(yx=1, discretization=8) == da_cml_t2.site_1_x).all()
    assert (merge_BK.x0_cml.isel(yx=0, discretization=0) == da_cml_t2.site_0_y).all()
    assert (merge_BK.x0_cml.isel(yx=0, discretization=4) == da_cml_t2.y).all()
    assert (merge_BK.x0_cml.isel(yx=0, discretization=8) == da_cml_t2.site_1_y).all()

    # Test that gauge midpoint coordinates corresponds to x0_gauge
    assert (merge_BK.x0_gauge.isel(yx=1) == da_gauges_t2.x).all()
    assert (merge_BK.x0_gauge.isel(yx=0) == da_gauges_t2.y).all()

    # Adjust field
    adjusted = merge_BK.adjust(
        da_rad_t, da_cml=da_cml_t2, da_gauge=da_gauges_t2, variogram=variogram
    )

    # test that the adjusted field is the same as first run
    data_check = np.array(
        [
            [0.0, 0.7627281, 6.0610102, 6.9895002],
            [0.0, 0.754008, 5.2116101, 6.7336794],
            [3.7761221, 5.0, 1.0, 5.5185573],
            [4.2335087, 4.0556517, 1.2108774, 3.4961416],
        ]
    )

    np.testing.assert_almost_equal(adjusted, data_check)

    # calculate the adjusted field along CMLs
    intersect_weights = plg.spatial.calc_sparse_intersect_weights_for_several_cmls(
        x1_line=da_cml_t2.site_0_lon.data,
        y1_line=da_cml_t2.site_0_lat.data,
        x2_line=da_cml_t2.site_1_lon.data,
        y2_line=da_cml_t2.site_1_lat.data,
        cml_id=da_cml_t2.cml_id.data,
        x_grid=adjusted.lon.data,
        y_grid=adjusted.lat.data,
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
        )

    # Test that da_rad is return if too few obs is provided
    merge_BK.min_obs_ = 10
    adjusted = merge_BK.adjust(
        da_rad_t,
        da_cml=da_cml_t2,
        da_gauge=None,
        variogram=variogram,
    )
    np.testing.assert_almost_equal(
        adjusted.data,
        da_rad_t.data,
    )
    merge_BK.min_obs_ = 1  # reset


def test_MergeBlockKrigingExternalDrift():
    # CML and rain gauge overlapping sets
    da_cml_t1 = ds_cmls.isel(cml_id=[2, 1], time=[0]).R
    da_cml_t2 = ds_cmls.isel(cml_id=[1, 0], time=[0]).R
    da_gauges_t1 = ds_gauges.isel(id=[2, 1], time=[0]).R
    da_gauges_t2 = ds_gauges.isel(id=[1, 0], time=[0]).R

    # Select radar timestep
    da_rad_t = ds_rad.isel(time=[0]).R

    # Initialize highlevel-class
    merge_KED = merge.MergeBlockKrigingExternalDrift(
        min_obs=1,  # We use few obs in this test
        discretization=8,
    )

    # Update geometry to set1
    merge_KED.update(
        da_rad_t,
        da_cml=da_cml_t1,
        da_gauge=da_gauges_t1,
    )

    # Test that CML names is correctly updated and sorted in the class
    assert (merge_KED.intersect_weights.cml_id == da_cml_t1.cml_id).all()

    # Test that cml midpoint and end coordinates corresponds to x0_cml
    assert (merge_KED.x0_cml.isel(yx=1, discretization=0) == da_cml_t1.site_0_x).all()
    assert (merge_KED.x0_cml.isel(yx=1, discretization=4) == da_cml_t1.x).all()
    assert (merge_KED.x0_cml.isel(yx=1, discretization=8) == da_cml_t1.site_1_x).all()
    assert (merge_KED.x0_cml.isel(yx=0, discretization=0) == da_cml_t1.site_0_y).all()
    assert (merge_KED.x0_cml.isel(yx=0, discretization=4) == da_cml_t1.y).all()
    assert (merge_KED.x0_cml.isel(yx=0, discretization=8) == da_cml_t1.site_1_y).all()

    # Test that gauge midpoint coordinates corresponds to x0_gauge
    assert (merge_KED.x0_gauge.isel(yx=1) == da_gauges_t1.x).all()
    assert (merge_KED.x0_gauge.isel(yx=0) == da_gauges_t1.y).all()

    # Test that automatic variogram estimation and transformation function works
    adjusted = merge_KED.adjust(
        da_rad_t,
        da_cml=da_cml_t1,
        da_gauge=da_gauges_t1,
        variogram="exponential",
    )

    # test that the adjusted field is the same as first run
    data_check = np.array(
        [
            [6.8837209, 4.7383721, 6.8837209, 6.8837209],
            [6.8837209, 6.8837209, 4.7383721, 9.2616279],
            [6.8837209, 5.0, 9.2383721, 4.7616279],
            [6.8837209, 9.0, 7.1162791, 9.0],
        ]
    )

    np.testing.assert_almost_equal(adjusted, data_check)

    # Simple linear variogram and transformation functions for testing
    def variogram(h):
        return h

    # Adjust field
    adjusted = merge_KED.adjust(
        da_rad_t,
        da_cml=da_cml_t1,
        da_gauge=da_gauges_t1,
        variogram=variogram,
    )

    # test that the adjusted field is the same as first run
    data_check = np.array(
        [
            [1.2673279, 0.9159766, 2.8557292, 5.2944255],
            [2.5462022, 2.7732177, 4.7644577, 6.2127718],
            [4.7542038, 5.0, 9.1808035, 7.4418168],
            [7.4271479, 7.2153635, 10.724253, 9.0],
        ]
    )

    np.testing.assert_almost_equal(adjusted, data_check)

    # calculate the adjusted field along CMLs
    intersect_weights = plg.spatial.calc_sparse_intersect_weights_for_several_cmls(
        x1_line=da_cml_t1.site_0_lon.data,
        y1_line=da_cml_t1.site_0_lat.data,
        x2_line=da_cml_t1.site_1_lon.data,
        y2_line=da_cml_t1.site_1_lat.data,
        cml_id=da_cml_t1.cml_id.data,
        x_grid=adjusted.lon.data,
        y_grid=adjusted.lat.data,
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
        np.testing.assert_almost_equal(
            merge_r,
            gauge_r,
        )

    # Update the weights using some new links
    merge_KED.update(
        da_rad_t,
        da_cml=da_cml_t2,
        da_gauge=da_gauges_t2,
    )

    # Test that CML names is correctly updated and sorted in the class
    assert (merge_KED.intersect_weights.cml_id == da_cml_t2.cml_id).all()

    # Test that cml midpoint and end coordinates corresponds to x0_cml
    assert (merge_KED.x0_cml.isel(yx=1, discretization=0) == da_cml_t2.site_0_x).all()
    assert (merge_KED.x0_cml.isel(yx=1, discretization=4) == da_cml_t2.x).all()
    assert (merge_KED.x0_cml.isel(yx=1, discretization=8) == da_cml_t2.site_1_x).all()
    assert (merge_KED.x0_cml.isel(yx=0, discretization=0) == da_cml_t2.site_0_y).all()
    assert (merge_KED.x0_cml.isel(yx=0, discretization=4) == da_cml_t2.y).all()
    assert (merge_KED.x0_cml.isel(yx=0, discretization=8) == da_cml_t2.site_1_y).all()

    # Test that gauge midpoint coordinates corresponds to x0_gauge
    assert (merge_KED.x0_gauge.isel(yx=1) == da_gauges_t2.x).all()
    assert (merge_KED.x0_gauge.isel(yx=0) == da_gauges_t2.y).all()

    # Adjust field
    adjusted = merge_KED.adjust(
        da_rad_t,
        da_cml=da_cml_t2,
        da_gauge=da_gauges_t2,
        variogram=variogram,
    )

    # test that the adjusted field is the same as first run
    data_check = np.array(
        [
            [-2.6950506, 0.7627281, 6.0610102, 6.9895002],
            [-0.9672861, 0.754008, 5.2116101, 6.7336794],
            [3.7761221, 5.0, 1.0, 5.5185573],
            [4.2335087, 4.0556517, 1.2108774, 3.4961416],
        ]
    )

    np.testing.assert_almost_equal(adjusted, data_check)

    # calculate the adjusted field along CMLs
    intersect_weights = plg.spatial.calc_sparse_intersect_weights_for_several_cmls(
        x1_line=da_cml_t2.site_0_lon.data,
        y1_line=da_cml_t2.site_0_lat.data,
        x2_line=da_cml_t2.site_1_lon.data,
        y2_line=da_cml_t2.site_1_lat.data,
        cml_id=da_cml_t2.cml_id.data,
        x_grid=adjusted.lon.data,
        y_grid=adjusted.lat.data,
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
        )

    # Test that providing only rain gauge adjusts only at rain gauge
    adjusted = merge_KED.adjust(
        da_rad_t,
        da_cml=None,
        da_gauge=da_gauges_t2,
        variogram=variogram,
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
        )

    # Test that providing only cml adjusts at cml
    adjusted = merge_KED.adjust(
        da_rad_t,
        da_cml=da_cml_t2,
        da_gauge=None,
        variogram=variogram,
    )

    # calculate the adjusted field along CMLs
    intersect_weights = plg.spatial.calc_sparse_intersect_weights_for_several_cmls(
        x1_line=da_cml_t2.site_0_lon.data,
        y1_line=da_cml_t2.site_0_lat.data,
        x2_line=da_cml_t2.site_1_lon.data,
        y2_line=da_cml_t2.site_1_lat.data,
        cml_id=da_cml_t2.cml_id.data,
        x_grid=adjusted.lon.data,
        y_grid=adjusted.lat.data,
        grid_point_location="center",
    )
    adjusted_at_cmls = plg.spatial.get_grid_time_series_at_intersections(
        grid_data=adjusted.expand_dims("time"),
        intersect_weights=intersect_weights,
    )

    # Test cml almost equal
    np.testing.assert_almost_equal(
        adjusted_at_cmls.data.ravel(),
        da_cml_t2.data.ravel(),
        decimal=0,  # not very precise, but decent
    )

    # Test that da_rad is return if too few obs is provided
    merge_KED.min_obs_ = 10
    adjusted = merge_KED.adjust(
        da_rad_t,
        da_cml=da_cml_t2,
        da_gauge=None,
        variogram=variogram,
    )
    np.testing.assert_almost_equal(
        adjusted.data,
        da_rad_t.data,
    )
    merge_KED.min_obs_ = 1  # reset
