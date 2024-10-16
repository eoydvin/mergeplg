"""Functions to do one full processing run from RADOLAN-RH to RW."""

import numpy as np
import poligrain as plg

from . import adjust

# This is just here because I am too lazy to fix pylint errors and
# and cannot use the per-line ignoring for pylint...
#
# pylint: skip-file


def round_down(a, decimal):
    """Round down floating points number with selectable decimal precision.

    Note that this does not work consistently for floats very close to the
    next value of the given decimal precision. This is due to inexact
    floating point representation. E.g.
    >>> round_down(0.099999999999, 1)
    >>> 0.0
    but
    >>> round_down(0.099999999999999999, 1)
    >>> 0.1

    """
    scale = 10**decimal
    return np.trunc(a * scale) / scale


def rh_to_rw(
    ds_radolan_t,
    df_stations_t,
    start_index_in_relevant_stations="random",
    idw_method="radolan",
    nnear=20,
    p=0.7,
    max_distance=60,
    bogra_kwargs={"max_iterations": 100, "max_allowed_relative_diff": 3},  # noqa: B006
    allow_gauge_and_cml=False,
    intersect_weights=None,
):
    """Produce RW from RH.

    ds_radolan_t : xr.Dataset
        Radar data which has to contain a data_var `RH`. See the function
        `check_data_struct.check_radar_dataset_or_dataarray` for details
        of what structure is expected
    df_stations_t : pd.DataFrame
        Data from rain gauges or CMLs as pd.DataFrame. See the function
        `check_data_struct.check_station_dataframe` for the expected
        structure.
    start_index_in_relevant_stations : str or int
        If set to "random" (which is the default) a random station will be picked as
        starting point for selecting the audit stations.
    idw_method : str
        With the default ("radolan") an exponential decay with range, as in the
        original RADOLAN implementation, is used for IDW. When set to "standard"
        a normal IDW decay with 1/d**p is used.
    nnear : int
        Number of nearest neighbors to use for interpolation.
    p : float
        The exponent used for "standard" IDW with 1/d**p.
    max_distance : int or float
        The maximum distance around a station (or mid-point of a CML) to use
        for interpolation.
    bogra_kwargs : dict
        Parameters for the BOGRA smoothing function, see `adjust.bogra_like_smoothing`
        for details.
    allow_gauge_and_cml : bool, optional
        If this is set to True, the column `sensor_type` has to be present in
        `df_stations_t` with either 'gauge_dwd' or 'cml_ericsson' as entry. Based
        on that an index will be created that is then used to select gauge and CML
        at the relevant parts of the code, currently important when getting the
        radar values at gauge or along CML.
    intersect_weights : xr.Dataset
        The CML intersection weights with the radar grid as returned by the function
        `poligrain.spatial.calc_sparse_intersect_weights_for_several_cmls`.

    Returns
    -------
    tuple (ds_radolan_t, df_stations_t)
        ds_radolan_t: All radar fields, also the intermediate ones used for adjustment
        df_stations_t: All station (and CML) data also with intermediate data

    """
    # It is important that we do not have a time dimension, not even one with
    # length=1, because we expect the data in `ds_radolan_t` to be 2D matrices
    # and not NxMx1. But we want to have the time stamp so that we can use it in
    # this function (even though, I thing it is not yet used to check if the radar
    # and gauge data are from the same time step).
    if not isinstance(ds_radolan_t.time.values, np.datetime64):
        msg = (
            "`ds_radolan_t` must have a `time` variable but no time dimension, "
            "i.e. there must only be one timestamp in `time`"
        )
        raise ValueError(msg)

    if allow_gauge_and_cml:
        sensor_is_dwd_gauge = df_stations_t.sensor_type == "gauge_dwd"
        sensor_is_cml = df_stations_t.sensor_type == "cml_ericsson"
        if intersect_weights is None:
            msg = "You must pass `intersect_weights` if you allow CML data"
            raise ValueError(msg)
    else:
        sensor_is_dwd_gauge = np.ones(shape=len(df_stations_t), dtype=bool)
        sensor_is_cml = np.zeros(shape=len(df_stations_t), dtype=bool)

    df_stations_t = adjust.label_relevant_audit_interim_in_gageset(
        df_gageset_t=df_stations_t,
        da_radolan=ds_radolan_t.RH,
        start_index_in_relevant=start_index_in_relevant_stations,
    )

    # BOGRA
    ds_radolan_t["RG"] = adjust.bogra_like_smoothing(
        radar_img=ds_radolan_t.RH,
        **bogra_kwargs,
    )

    # BORAMA (we do not do anything here for now, because it does not make much sense)
    ds_radolan_t["RB"] = ds_radolan_t.RG

    # Radar at gauge
    df_stations_t.loc[sensor_is_dwd_gauge, "radar_RB_rainfall"] = (
        adjust.get_grid_rainfall_at_points(
            da_grid=ds_radolan_t.RB,
            df_stations=df_stations_t.loc[sensor_is_dwd_gauge, :],
        )
    )

    # Radar at CML
    if allow_gauge_and_cml:
        cml_ids_df = df_stations_t.loc[sensor_is_cml, "station_id"]
        radar_along_cmls = plg.spatial.get_grid_time_series_at_intersections(
            grid_data=ds_radolan_t.RB.expand_dims(dim="time"),
            intersect_weights=intersect_weights.sel(cml_id=cml_ids_df.values),
        )
        df_stations_t.loc[sensor_is_cml, "radar_RB_rainfall"] = (
            radar_along_cmls.isel(time=0).sel(cml_id=cml_ids_df.values).values
        )

    # Diff and fact
    df_stations_t.loc[:, "radar_RB_rainfall_diff"] = (
        df_stations_t.rainfall_amount - df_stations_t.radar_RB_rainfall
    )
    df_stations_t.loc[:, "radar_RB_rainfall_fact"] = (
        df_stations_t.rainfall_amount / df_stations_t.radar_RB_rainfall
    )

    # Bodcorr
    diff = df_stations_t.radar_RB_rainfall_diff
    fact = df_stations_t.radar_RB_rainfall_fact

    df_stations_t.loc[:, "bodcorr_diff_out_of_range"] = (diff > 10) | (diff < -10)
    df_stations_t.loc[:, "bodcorr_fact_out_of_range"] = (fact > 15) | (fact < 0.1)

    # set to neutral for small rain rates
    threshold = 0.1
    df_stations_t.loc[
        df_stations_t.rainfall_amount < threshold, "radar_RB_rainfall_diff"
    ] = 0
    df_stations_t.loc[
        df_stations_t.rainfall_amount < threshold, "radar_RB_rainfall_fact"
    ] = 1

    # set to neutral if radar is zero
    df_stations_t.loc[
        df_stations_t.radar_RB_rainfall <= 0, "radar_RB_rainfall_diff"
    ] = 0
    df_stations_t.loc[
        df_stations_t.radar_RB_rainfall <= 0, "radar_RB_rainfall_fact"
    ] = 1

    # Interpolate adjustment diff and fact from interim stations
    ds_radolan_t["dbr_interim"] = adjust.interpolate_station_values(
        df_stations=df_stations_t[
            df_stations_t.interim & ~df_stations_t.bodcorr_diff_out_of_range
        ],
        col_name="radar_RB_rainfall_diff",
        station_ids_to_exclude=[],
        ds_grid=ds_radolan_t,
        idw_method=idw_method,
        nnear=nnear,
        p=p,
        max_distance=max_distance,
        fill_value=0,
    )

    ds_radolan_t["fbr_interim"] = adjust.interpolate_station_values(
        df_stations=df_stations_t[
            df_stations_t.interim & ~df_stations_t.bodcorr_fact_out_of_range
        ],
        col_name="radar_RB_rainfall_fact",
        station_ids_to_exclude=[],
        ds_grid=ds_radolan_t,
        idw_method=idw_method,
        nnear=nnear,
        p=p,
        max_distance=max_distance,
        fill_value=1,
    )

    # Do adjustment based on interim stations
    ds_radolan_t["addiff_interim"] = ds_radolan_t.RB + ds_radolan_t.dbr_interim
    ds_radolan_t["mulfak_interim"] = ds_radolan_t.RB * ds_radolan_t.fbr_interim

    # Set values smaller 0 to 0
    ds_radolan_t["addiff_interim"] = ds_radolan_t.addiff_interim.where(
        ~(ds_radolan_t.addiff_interim < 0), 0
    )
    ds_radolan_t["mulfak_interim"] = ds_radolan_t.mulfak_interim.where(
        ~(ds_radolan_t.mulfak_interim < 0), 0
    )

    # Get adjusted radar values at all stations (even though we will only use the audit
    # station in the next step)
    df_stations_t.loc[:, "radar_addiff_interim"] = adjust.get_grid_rainfall_at_points(
        da_grid=ds_radolan_t.addiff_interim,
        df_stations=df_stations_t,
        nnear=1,
        stat="best",
    )

    df_stations_t.loc[:, "radar_mulfak_interim"] = adjust.get_grid_rainfall_at_points(
        da_grid=ds_radolan_t.mulfak_interim,
        df_stations=df_stations_t,
        nnear=1,
        stat="best",
    )

    # Calc differences
    df_stations_t.loc[:, "diff_addiff_interim"] = (
        df_stations_t.rainfall_amount - df_stations_t.radar_addiff_interim
    )
    df_stations_t.loc[:, "diff_mulfak_interim"] = (
        df_stations_t.rainfall_amount - df_stations_t.radar_mulfak_interim
    )

    # Determine the winner of the two methods and set a weight of 1 for it
    df_stations_t.loc[:, "weight_addiff_interim"] = (
        df_stations_t.diff_addiff_interim.fillna(1e10).abs()
        <= df_stations_t.diff_mulfak_interim.fillna(1e10).abs()
    ).astype("float")
    df_stations_t.loc[:, "weight_mulfak_interim"] = (
        df_stations_t.diff_addiff_interim.fillna(1e10).abs()
        >= df_stations_t.diff_mulfak_interim.fillna(1e10).abs()
    ).astype("float")

    # Set 0.5 in case both are equally good, or bad...
    # TODO: Check if we should fill NaNs here
    # equal_index = (
    #     df_stations_t.diff_addiff_interim.fillna(1e10).abs()
    #     == df_stations_t.diff_mulfakt_interim.fillna(1e10).abs()
    # )
    equal_index = (
        df_stations_t.diff_addiff_interim.abs()
        == df_stations_t.diff_mulfak_interim.abs()
    )
    df_stations_t.loc[equal_index, "weight_addiff_interim"] = 0.5
    df_stations_t.loc[equal_index, "weight_mulfak_interim"] = 0.5

    # Interpolate the weights onto radar grid
    ds_radolan_t["weight_addiff_interim_audit"] = adjust.interpolate_station_values(
        df_stations=df_stations_t[
            df_stations_t.audit & ~df_stations_t.bodcorr_diff_out_of_range
        ],
        col_name="weight_addiff_interim",
        station_ids_to_exclude=[],
        ds_grid=ds_radolan_t,
        idw_method=idw_method,
        nnear=nnear,
        p=p,
        max_distance=max_distance,
        fill_value=0.5,
    )

    ds_radolan_t["weight_mulfak_interim_audit"] = adjust.interpolate_station_values(
        df_stations=df_stations_t[
            df_stations_t.audit & ~df_stations_t.bodcorr_diff_out_of_range
        ],
        col_name="weight_mulfak_interim",
        station_ids_to_exclude=[],
        ds_grid=ds_radolan_t,
        idw_method=idw_method,
        nnear=nnear,
        p=p,
        max_distance=max_distance,
        fill_value=0.5,
    )

    # Interpolate adjustment diff and fact from all relevant stations
    ds_radolan_t["dbr_relevant"] = adjust.interpolate_station_values(
        df_stations=df_stations_t[
            df_stations_t.relevant & ~df_stations_t.bodcorr_diff_out_of_range
        ],
        col_name="radar_RB_rainfall_diff",
        station_ids_to_exclude=[],
        ds_grid=ds_radolan_t,
        idw_method=idw_method,
        nnear=nnear,
        p=p,
        max_distance=max_distance,
        fill_value=0,
    )

    ds_radolan_t["fbr_relevant"] = adjust.interpolate_station_values(
        df_stations=df_stations_t[
            df_stations_t.relevant & ~df_stations_t.bodcorr_fact_out_of_range
        ],
        col_name="radar_RB_rainfall_fact",
        station_ids_to_exclude=[],
        ds_grid=ds_radolan_t,
        idw_method=idw_method,
        nnear=nnear,
        p=p,
        max_distance=max_distance,
        fill_value=1,
    )

    # Do adjustment based on relevant stations
    ds_radolan_t["addiff_relevant"] = ds_radolan_t.RB + ds_radolan_t.dbr_relevant
    ds_radolan_t["mulfak_relevant"] = ds_radolan_t.RB * ds_radolan_t.fbr_relevant
    # Set original value in locations where adjustments have NaNs
    ds_radolan_t["addiff_relevant"] = ds_radolan_t.addiff_relevant.where(
        ~ds_radolan_t.dbr_relevant.isnull(),  # noqa: PD003
        ds_radolan_t.RB,
    )
    ds_radolan_t["mulfak_relevant"] = ds_radolan_t.mulfak_relevant.where(
        ~ds_radolan_t.fbr_relevant.isnull(),  # noqa: PD003
        ds_radolan_t.RB,
    )
    # Set original value in locations where radar had zero rainfall
    ds_radolan_t["addiff_relevant"] = ds_radolan_t.addiff_relevant.where(
        ~(ds_radolan_t.RB == 0), ds_radolan_t.RB
    )
    ds_radolan_t["mulfak_relevant"] = ds_radolan_t.mulfak_relevant.where(
        ~(ds_radolan_t.RB == 0), ds_radolan_t.RB
    )

    # Set values smaller 0 to 0
    ds_radolan_t["addiff_relevant"] = ds_radolan_t.addiff_relevant.where(
        ~(ds_radolan_t.addiff_relevant < 0), 0
    )
    ds_radolan_t["mulfak_relevant"] = ds_radolan_t.mulfak_relevant.where(
        ~(ds_radolan_t.mulfak_relevant < 0), 0
    )

    # Get adjusted radar values at all stations
    df_stations_t.loc[:, "radar_addiff_relevant"] = adjust.get_grid_rainfall_at_points(
        da_grid=ds_radolan_t.addiff_relevant,
        df_stations=df_stations_t,
        nnear=1,
        stat="best",
    )

    df_stations_t.loc[:, "radar_mulfak_relevant"] = adjust.get_grid_rainfall_at_points(
        da_grid=ds_radolan_t.mulfak_relevant,
        df_stations=df_stations_t,
        nnear=1,
        stat="best",
    )

    # Calc differences
    df_stations_t.loc[:, "diff_addiff_relevant"] = (
        df_stations_t.rainfall_amount - df_stations_t.radar_addiff_relevant
    )
    df_stations_t.loc[:, "diff_mulfak_relevant"] = (
        df_stations_t.rainfall_amount - df_stations_t.radar_mulfak_relevant
    )

    # Generate RW und the RW from the adjustments that were only made based
    # on the interim stations
    ds_radolan_t["RW_not_rounded"] = (
        ds_radolan_t.addiff_relevant * ds_radolan_t.weight_addiff_interim_audit
        + ds_radolan_t.mulfak_relevant * ds_radolan_t.weight_mulfak_interim_audit
    ) / (
        ds_radolan_t.weight_addiff_interim_audit
        + ds_radolan_t.weight_mulfak_interim_audit
    )
    ds_radolan_t["RW"] = round_down(ds_radolan_t.RW_not_rounded, decimal=1)

    ds_radolan_t["RW_interim"] = (
        ds_radolan_t.addiff_interim * ds_radolan_t.weight_addiff_interim_audit
        + ds_radolan_t.mulfak_interim * ds_radolan_t.weight_mulfak_interim_audit
    ) / (
        ds_radolan_t.weight_addiff_interim_audit
        + ds_radolan_t.weight_mulfak_interim_audit
    )

    ds_radolan_t["RR"] = adjust.interpolate_station_values(
        df_stations=df_stations_t,
        col_name="rainfall_amount",
        ds_grid=ds_radolan_t,
        idw_method=idw_method,
        nnear=nnear,
        p=p,
        max_distance=40,
    )

    # Set interpolated station data at locations where RW is NaN
    ds_radolan_t["RW_no_station_fill"] = ds_radolan_t.RW.copy(deep=True)
    ds_radolan_t["RW"] = ds_radolan_t.RW.where(
        ~ds_radolan_t.RW.isnull(),  # noqa: PD003
        ds_radolan_t.RR,
    )
    ds_radolan_t["RW_interim"] = ds_radolan_t.RW_interim.where(
        ~ds_radolan_t.RW_interim.isnull(),  # noqa: PD003
        ds_radolan_t.RR,
    )

    # Set values smaller 0 to 0
    ds_radolan_t["RW"] = ds_radolan_t.RW.where(~(ds_radolan_t.RW < 0), 0)
    ds_radolan_t["RW_interim"] = ds_radolan_t.RW_interim.where(
        ~(ds_radolan_t.RW_interim < 0), 0
    )

    # Get RW at station (just for evaluation)
    df_stations_t.loc[:, "radar_RW_rainfall_1"] = adjust.get_grid_rainfall_at_points(
        da_grid=ds_radolan_t.RW,
        df_stations=df_stations_t,
        nnear=1,
        stat="best",
    )
    df_stations_t.loc[:, "radar_RW_rainfall_9"] = adjust.get_grid_rainfall_at_points(
        da_grid=ds_radolan_t.RW,
        df_stations=df_stations_t,
        nnear=9,
        stat="best",
    )

    return ds_radolan_t, df_stations_t
