import numpy as np
import pandas as pd
import pytest
import xarray as xr

import mergeplg as mrg

from .test_radolan_adjust import get_test_data


def test_rounding_down():
    a = np.array([0.0999999, 0.11111, 0.19999, 1.99999])

    np.testing.assert_almost_equal(
        mrg.radolan.processing.round_down(a, decimal=0), np.array([0.0, 0.0, 0.0, 1])
    )

    np.testing.assert_almost_equal(
        mrg.radolan.processing.round_down(a, decimal=1), np.array([0.0, 0.1, 0.1, 1.9])
    )

    np.testing.assert_almost_equal(
        mrg.radolan.processing.round_down(a, decimal=2),
        np.array([0.09, 0.11, 0.19, 1.99]),
    )

    # also test passing xr.DataArray
    np.testing.assert_almost_equal(
        mrg.radolan.processing.round_down(xr.DataArray(a), decimal=2),
        np.array([0.09, 0.11, 0.19, 1.99]),
    )


def test_rh_to_rw():
    ds_radolan, df_stations = get_test_data()
    RY_sum = ds_radolan.RY.sum(dim="time", min_count=12).to_dataset(name="RH")
    RY_sum["time"] = ds_radolan.time.values[-1] + pd.Timedelta("5min")

    # add a dummy station outside of the radar coverage to test that interpolated
    # station data is added in these regions (see test further down below)
    dummy_station = pd.DataFrame(
        index=[
            1,
        ],
        data={
            "time": "2021-08-23 09:50:00",
            "station_id": "foo",
            "rainfall_amount": 5.5,
            "station_name": "le_dummy",
            "longitude": 1.1,
            "latitude": 1.1,
            "x": -420.0,
            "y": -4500.0,
        },
    )
    df_stations = pd.concat([df_stations, dummy_station], ignore_index=True)

    ds_radolan_result, df_stations_result = mrg.radolan.processing.rh_to_rw(
        ds_radolan_t=RY_sum,
        df_stations_t=df_stations,
        idw_method="radolan",
        start_index_in_relevant_stations=0,
        nnear=20,
        max_distance=60,
    )

    # I visually checked that these results make sense, but
    # there is no 100 percent guarantee that all parts of the
    # RADOLAN-RW production process work correctly in the processing.
    np.testing.assert_array_almost_equal(
        ds_radolan_result.RW.values[420:423, 710:713],
        np.array(
            [
                [4.7, 3.8, 4.0],
                [4.6, 4.5, 4.1],
                [5.3, 4.9, 5.1],
            ]
        ),
    )

    np.testing.assert_array_almost_equal(
        (ds_radolan_result.RW - ds_radolan_result.RH).values[420:423, 710:713],
        np.array(
            [
                [0.33, 0.41, 0.54],
                [0.33, 0.46, 0.61],
                [0.42, 0.6, 0.82],
            ]
        ),
    )

    # test that interpolated station data is in the regions where RW would be NaN
    nan = np.nan
    np.testing.assert_array_almost_equal(
        ds_radolan_result.RW.values[118:121, 102:105],
        np.array(
            [
                [nan, nan, nan],
                [5.5, 5.5, 5.5],
                [5.5, 5.5, 5.5],
            ]
        ),
    )

    # test the raise if there is a `time` dimension
    RY_sum = ds_radolan.RY.sum(dim="time", min_count=12).to_dataset(name="RH")
    RY_sum["time"] = [
        ds_radolan.time.values[-1] + pd.Timedelta("5min"),
    ]
    with pytest.raises(ValueError, match=r"`ds_radolan_t` must have *"):
        _, _ = mrg.radolan.processing.rh_to_rw(
            ds_radolan_t=RY_sum,
            df_stations_t=df_stations,
        )
