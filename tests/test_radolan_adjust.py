import numpy as np
import pandas as pd
import pytest
import xarray as xr

import mergeplg as mrg


def get_test_data():
    df_stations = pd.read_csv(
        "tests/test_data/radolan_rain_gauge_data.csv", index_col=0
    )
    ds_radolan = xr.open_dataset("tests/test_data/radolan_ry_data_compressed.nc")

    assert len(ds_radolan.time) == 12

    mrg.radolan.check_data_struct.check_radar_dataset_or_dataarray(
        ds_radolan, only_single_time_step=False
    )
    mrg.radolan.check_data_struct.check_station_dataframe(df_stations)

    return ds_radolan, df_stations


def test_get_test_data():
    # this is just to run the `get_test_data` function from above when running pytest
    _, _ = get_test_data()


def test_check_for_radar_coverage():
    ds_radolan, df_stations = get_test_data()
    RY_sum = ds_radolan.RY.sum(dim="time", min_count=12)

    no_radar_coverage = mrg.radolan.adjust.check_for_radar_coverage(
        x_gage=[-1500, -350, -350, 100],
        y_gage=[-4500, -4500, -4300, -4300],
        x_radar=RY_sum.x.data.flatten(),
        y_radar=RY_sum.y.data.flatten(),
        no_radar_coverage_grid=RY_sum.isnull().values,  # noqa: PD003
    )

    np.testing.assert_equal(
        no_radar_coverage,
        np.array([True, True, False, False]),
    )


def test_label_relevant_audit_interim_in_gageset_fixed_start_index():
    ds_radolan, df_stations = get_test_data()
    RY_sum = ds_radolan.RY.sum(dim="time", min_count=12)
    df_stations_with_audit_interim = (
        mrg.radolan.adjust.label_relevant_audit_interim_in_gageset(
            df_gageset_t=df_stations,
            da_radolan=RY_sum,
            start_index_in_relevant=2,
        )
    )

    assert df_stations_with_audit_interim.audit.sum() == 228
    assert df_stations_with_audit_interim.interim.sum() == 914
    audit_stations = df_stations_with_audit_interim[
        df_stations_with_audit_interim.audit
    ]
    assert audit_stations.station_id.iloc[101] == "L521"


def test_label_relevant_audit_interim_in_gageset_random_start_index():
    ds_radolan, df_stations = get_test_data()
    RY_sum = ds_radolan.RY.sum(dim="time", min_count=12)

    audit_station_id_previous = None
    N_random_runs = 5
    for i in range(N_random_runs):  # noqa: B007
        # Note that the default for start_index_in_relevant='random',
        # hence we do not set it here
        df_stations_with_audit_interim = (
            mrg.radolan.adjust.label_relevant_audit_interim_in_gageset(
                df_gageset_t=df_stations,
                da_radolan=RY_sum,
            )
        )
        audit_station_id = df_stations_with_audit_interim[
            df_stations_with_audit_interim.audit
        ].station_id.iloc[99]

        if audit_station_id_previous is None:
            audit_station_id_previous = audit_station_id
            continue
        if audit_station_id != audit_station_id_previous:
            break
        audit_station_id_previous = audit_station_id

    # This fails if all runs with random start index produced the same station_id
    assert i < N_random_runs - 1


def test_label_relevant_audit_interim_in_gageset_raise():
    ds_radolan, df_stations = get_test_data()
    RY_sum = ds_radolan.RY.sum(dim="time", min_count=12)
    with pytest.raises(TypeError):
        mrg.radolan.adjust.label_relevant_audit_interim_in_gageset(
            df_gageset_t=df_stations,
            da_radolan=RY_sum,
            start_index_in_relevant=1.42,
        )


def test_get_grid_rainfall_at_points():
    ds_radolan, df_stations = get_test_data()
    RY_sum = ds_radolan.RY.sum(dim="time", min_count=12)

    df_stations["radar_at_gauge"] = mrg.radolan.adjust.get_grid_rainfall_at_points(
        RY_sum,
        df_stations,
    )

    df_stations_sorted = df_stations.sort_values("radar_at_gauge", kind="stable")
    np.testing.assert_array_almost_equal(
        df_stations_sorted.radar_at_gauge.to_numpy()[-10:],
        np.array([4.69, 4.75, 4.76, 4.89, 5.73, 5.99, 6.39, 7.3, 7.57, 7.57]),
    )
    np.testing.assert_array_equal(
        df_stations_sorted.station_id.to_numpy()[-5:],
        np.array(["O980", "O811", "M500", "F598", "O708"], dtype=object),
    )


def test_interpolate_station_values():
    ds_radolan, df_stations = get_test_data()
    RY_sum = ds_radolan.RY.sum(dim="time", min_count=12)

    interpolated_grid = mrg.radolan.adjust.interpolate_station_values(
        df_stations=df_stations,
        col_name="rainfall_amount",
        ds_grid=RY_sum,
        nnear=8,
        p=2,
        max_distance=60,
        idw_method="standard",
    )
    np.testing.assert_array_almost_equal(
        interpolated_grid.data[410:414, 710:714],
        np.array(
            [
                [2.72487096, 2.800001, 2.83147212, 2.82727883],
                [2.80509236, 2.88496315, 2.92157422, 2.92241554],
                [2.94312663, 3.00469669, 3.03208744, 3.02933747],
                [3.13244699, 3.15721331, 3.16299551, 3.14901499],
            ]
        ),
    )

    interpolated_grid = mrg.radolan.adjust.interpolate_station_values(
        df_stations=df_stations,
        col_name="rainfall_amount",
        ds_grid=RY_sum,
        nnear=8,
        p=2,
        max_distance=60,
        idw_method="radolan",
    )
    np.testing.assert_array_almost_equal(
        interpolated_grid.data[410:414, 710:714],
        np.array(
            [
                [2.68921691, 2.6993467, 2.67471823, 2.61990556],
                [2.78720804, 2.8033063, 2.78501507, 2.73558897],
                [2.93226004, 2.93803843, 2.91615941, 2.86679836],
                [3.12492011, 3.10614845, 3.07118028, 3.01634675],
            ]
        ),
    )

    # test filling of NaNs (first check that NaNs are there at this specific
    # location, which is at the edge of max_distance for interpolation, and then do
    # interpolation again with filling the NaNs)
    nan = np.nan
    np.testing.assert_array_almost_equal(
        interpolated_grid.data[355:359, 800:804],
        np.array(
            [
                [nan, nan, nan, nan],
                [1.02, 1.02, nan, nan],
                [1.23399745, 1.02, 1.02, 1.02],
                [1.23115475, 1.22518546, 1.21935005, 1.02],
            ]
        ),
    )

    interpolated_grid = mrg.radolan.adjust.interpolate_station_values(
        df_stations=df_stations,
        col_name="rainfall_amount",
        ds_grid=RY_sum,
        nnear=8,
        p=2,
        max_distance=60,
        idw_method="radolan",
        fill_value=0,
    )
    np.testing.assert_array_almost_equal(
        interpolated_grid.data[355:359, 800:804],
        np.array(
            [
                [0, 0, 0, 0.0],
                [1.02, 1.02, 0, 0],
                [1.23399745, 1.02, 1.02, 1.02],
                [1.23115475, 1.22518546, 1.21935005, 1.02],
            ]
        ),
    )

    # test case where all stations are NaN
    df_stations_all_nan = df_stations.copy()
    df_stations_all_nan["rainfall_amount"] = np.nan

    interpolated_grid = mrg.radolan.adjust.interpolate_station_values(
        df_stations=df_stations_all_nan,
        col_name="rainfall_amount",
        ds_grid=RY_sum,
        nnear=8,
        p=2,
        max_distance=60,
        idw_method="radolan",
    )
    nan = np.nan
    np.testing.assert_array_almost_equal(
        interpolated_grid.data[355:359, 800:804],
        np.array(
            [
                [nan, nan, nan, nan],
                [nan, nan, nan, nan],
                [nan, nan, nan, nan],
                [nan, nan, nan, nan],
            ]
        ),
    )


def test_bogra_like_smoothing():
    ds_radolan, df_stations = get_test_data()
    RY_sum = ds_radolan.RY.sum(dim="time", min_count=12)

    # Test with xarray.DataArray
    smoothed_data = mrg.radolan.adjust.bogra_like_smoothing(RY_sum)
    np.testing.assert_array_almost_equal(
        (smoothed_data - RY_sum).data[138:142, 679:682],
        np.array(
            [
                [0.0, 0.0, -0.03349609],
                [0.0, -1.08796875, 0.0],
                [-0.2034375, -0.99517578, 0.0],
                [0.0, 0.0, -0.32083984],
            ],
        ),
    )

    # Test again, but with different threshold
    smoothed_data = mrg.radolan.adjust.bogra_like_smoothing(
        RY_sum,
        max_allowed_relative_diff=5,
    )
    np.testing.assert_array_almost_equal(
        (smoothed_data - RY_sum).data[138:142, 679:682],
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, -0.98375, 0.0],
                [0.0, -0.83375, 0.0],
                [0.0, 0.0, 0.0],
            ],
        ),
    )

    # Test again, but with high threshold and less iterations, just to check
    # that the `break` in the bogra loop is called.
    smoothed_data = mrg.radolan.adjust.bogra_like_smoothing(
        RY_sum, max_allowed_relative_diff=10, max_iterations=10
    )
    np.testing.assert_array_almost_equal(
        (smoothed_data - RY_sum).data[138:142, 679:682],
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0, 0.0],
                [0.0, 0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        ),
    )

    # Test that results are the same for numpy array xarray.DataArray as input
    np.testing.assert_array_almost_equal(
        mrg.radolan.adjust.bogra_like_smoothing(RY_sum).values,
        mrg.radolan.adjust.bogra_like_smoothing(RY_sum.values),
    )
