from __future__ import annotations

import numpy as np
import xarray as xr

from mergeplg import merge_functions

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


def test_calculate_cml_line():
    # Test that the CML geometry is correctly esimtated
    y, x = merge_functions.calculate_cml_line(
        ds_cmls.isel(cml_id=[1]),
        discretization=2,  # divides the line into two intervals, 3 points
    ).data[0]
    assert (y == np.array([-1, 0, 1])).all()
    assert (x == np.array([0, 1, 2])).all()


def test_block_points_to_lengths():
    # Check that the length matrix is correctly estimated
    line = merge_functions.block_points_to_lengths(
        merge_functions.calculate_cml_line(
            ds_cmls.isel(cml_id=[0, 1]), discretization=2
        ).data
    )
    l0l0 = line[0, 0]  # Lengths from link0 to link0
    l0l1 = line[0, 1]  # Lengths from link0 to link0
    l1l0 = line[1, 0]  # Lengths from link0 to link0
    l1l1 = line[1, 1]  # Lengths from link0 to link0

    # Length matrix from line1 to lin1
    assert (
        l0l0
        == np.array(
            [
                [0, np.sqrt(2), 2 * np.sqrt(2)],
                [np.sqrt(2), 0, np.sqrt(2)],
                [2 * np.sqrt(2), np.sqrt(2), 0],
            ]
        )
    ).all()

    # Length matrix from line1 to lin2
    assert (
        l0l1
        == np.array(
            [
                [1, 1, np.sqrt(2**2 + 1)],
                [np.sqrt(2**2 + 1), 1, 1],
                [np.sqrt(3**2 + 2**2), np.sqrt(2**2 + 1), 1],
            ]
        )
    ).all()

    # Length matrix from line2 to lin1
    assert (
        l1l0
        == np.array(
            [
                [1, np.sqrt(2**2 + 1), np.sqrt(3**2 + 2**2)],
                [1, 1, np.sqrt(2**2 + 1)],
                [np.sqrt(1 + 2**2), 1, 1],
            ]
        )
    ).all()

    # Length matrix from line2 to lin2
    assert (
        l1l1
        == np.array(
            [
                [0, np.sqrt(2), 2 * np.sqrt(2)],
                [np.sqrt(2), 0, np.sqrt(2)],
                [2 * np.sqrt(2), np.sqrt(2), 0],
            ]
        )
    ).all()


def test_merge_additive_idw():
    # Calculate CML-radar difference
    ds_cmls["R_diff"] = xr.full_like(ds_cmls.R, 0)
    for cml_id in ds_cmls.cml_id:
        rad_r = ds_rad.sel(
            x=ds_cmls.sel(cml_id=cml_id).x,
            y=ds_cmls.sel(cml_id=cml_id).y,
        ).R
        cml_r = ds_cmls.sel(cml_id=cml_id).R
        ds_cmls["R_diff"].loc[{"cml_id": cml_id}] = cml_r - rad_r

    x0 = np.hstack(
        [
            ds_cmls.y.data.reshape(-1, 1),
            ds_cmls.x.data.reshape(-1, 1),
        ]
    )
    additive_idw = merge_functions.merge_additive_idw(
        ds_rad.R.isel(time=0), ds_cmls.R_diff.isel(time=0).data.ravel(), x0
    )

    # Check that radar is conditioned to point observations
    for cml_id in ds_cmls.cml_id:
        merge_r = additive_idw.sel(
            x=ds_cmls.sel(cml_id=cml_id).x,
            y=ds_cmls.sel(cml_id=cml_id).y,
        ).data
        cml_r = ds_cmls.sel(cml_id=cml_id).isel(time=0).R.data
        assert cml_r == merge_r


def test_merge_additive_blockkriging():
    # Calculate CML-radar difference
    ds_cmls["R_diff"] = xr.full_like(ds_cmls.R, 0)
    for cml_id in ds_cmls.cml_id:
        rad_r = ds_rad.sel(
            x=ds_cmls.sel(cml_id=cml_id).x,
            y=ds_cmls.sel(cml_id=cml_id).y,
        ).R
        cml_r = ds_cmls.sel(cml_id=cml_id).R
        ds_cmls["R_diff"].loc[{"cml_id": cml_id}] = cml_r - rad_r

    # Calculate CML geometry
    x0 = merge_functions.calculate_cml_line(ds_cmls).data

    # Define variogram (exponential)
    def variogram(h):  # Exponential variogram
        return 0 + (1 - 0) * (1 - np.exp(-h * 3 / 1))

    n_obs = ds_cmls.R_diff.isel(time=0).data.size

    # do additive blockkriging
    additive_blockkriging = merge_functions.merge_additive_blockkriging(
        ds_rad.R.isel(time=0),
        ds_cmls.R_diff.isel(time=0).data,
        x0,
        variogram,
        n_obs - 1 if n_obs <= 8 else 8,
    )

    # Is not strightforward to check the line integrals as the discretization
    # only approximates the real field, thus we just check the full matrix
    # against a copy for now.
    data_check = np.array(
        [
            [1.1007268, 4.4315058, 3.4856829, 7.0209609],
            [2.4789288, 0.8917858, 5.1082142, 8.6748483],
            [4.9256465, 5.0, 6.421814, 5.3251517],
            [5.2115603, 9.1025073, 5.9828137, 3.0836016],
        ]
    )

    np.testing.assert_almost_equal(additive_blockkriging, data_check)


def test_merge_ked_blockkriging():
    # Calculate CML-radar difference
    ds_cmls["cml_rad"] = xr.full_like(ds_cmls.R, 0)
    for cml_id in ds_cmls.cml_id:
        rad_r = ds_rad.sel(
            x=ds_cmls.sel(cml_id=cml_id).x,
            y=ds_cmls.sel(cml_id=cml_id).y,
        ).R

        ds_cmls["cml_rad"].loc[{"cml_id": cml_id}] = rad_r

    # Calculate CML geometry
    x0 = merge_functions.calculate_cml_line(ds_cmls).data

    # Define variogram (exponential)
    def variogram(h):  # Exponential variogram
        return 0 + (1 - 0) * (1 - np.exp(-h * 3 / 1))

    n_obs = ds_cmls.R_diff.isel(time=0).data.size

    # do additive blockkriging
    ked = merge_functions.merge_ked_blockkriging(
        ds_rad.R.isel(time=0),
        ds_cmls.cml_rad.isel(time=0).data,
        ds_cmls.R.isel(time=0).data,
        x0,
        variogram,
        n_obs - 1 if n_obs <= 8 else 8,
    )

    # Is not strightforward to check the line integrals as the discretization
    # only approximates the real field, thus we just check the full matrix
    # against a copy for now.
    data_check = np.array(
        [
            [1.1007268, 4.4315058, 3.4856829, 7.0209609],
            [2.4789288, 0.8917858, 5.1082142, 8.6748483],
            [4.9256465, 5.0, 6.421814, 5.3251517],
            [5.2115603, 9.1025073, 5.9828137, 3.0836016],
        ]
    )

    np.testing.assert_almost_equal(ked, data_check)
