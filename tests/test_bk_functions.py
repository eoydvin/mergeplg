from __future__ import annotations

import numpy as np
import xarray as xr

from mergeplg import bk_functions

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
    y, x = bk_functions.calculate_cml_line(
        ds_cmls.isel(cml_id=[1]),
        discretization=2,  # divides the line into two intervals, 3 points
    ).data[0]
    assert (y == np.array([-1, 0, 1])).all()
    assert (x == np.array([0, 1, 2])).all()


def test_block_points_to_lengths():
    # Check that the length matrix is correctly estimated
    line = bk_functions.block_points_to_lengths(
        bk_functions.calculate_cml_line(
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
