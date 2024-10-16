"""Functions to check data structures."""


def check_station_dataframe(df, require_lat_lon=True, only_single_time_step=True):
    """Check that Dataframe with station data has the correct structure.

    Requirements are:
        - columns contain ["station_id", "rainfall_amount", "station_name", "x", "y"]
        - if `require_lat_lon=True` then ["longitude", "latitude"] must be in columns
        - if `only_single_time_step=True` then no duplicated `station_id` and no
          duplicate time stamp in the index is allowed

    Parameters
    ----------
    df : pandas.Dataframe
       Dataframe to check
    require_lat_lon : bool (default True)
       Check if column for latitude and longitude is there
    only_single_time_step : bool (default True)
       Check that no duplicated time stamps are in the index

    Raises
    ------
    ValueError in case one of the requirements is not met

    """
    required_coord_columns = ["x", "y"]
    if require_lat_lon:
        required_coord_columns += ["longitude", "latitude"]

    required_columns = [  # noqa: RUF005
        "station_id",
        "rainfall_amount",
        "station_name",
    ] + required_coord_columns
    df_columns = df.columns
    for col_name in required_columns:
        if col_name not in df_columns:
            msg = f"Column `{col_name}` is not present in Dataframe"
            raise ValueError(msg)
    if only_single_time_step:
        unique_ts = df.index.unique()
        duplicated_station_ids = df.station_id[df.station_id.duplicated()]
        if len(unique_ts) != 1:
            msg = (
                "There must be only one unique time stamp in the Dataframe, "
                f"but unique time stamps are: {unique_ts}"
            )
            raise ValueError(msg)
        if len(duplicated_station_ids) > 0:
            msg = (
                "There must not be duplicated station_id entries in Dataframe, "
                "if only one time step is allowed. The duplicated station_id "
                f"entries are: {duplicated_station_ids}"
            )
            raise ValueError(msg)
    if df[required_coord_columns].isna().any().any():
        msg = "There must not be NaNs in the coordinate columns"
        raise ValueError(msg)


def check_radar_dataset_or_dataarray(
    ds, require_lat_lon=True, only_single_time_step=True
):
    """Check that Dataset or DataArray have correct structure.

    Beware that this does not check the dimensions of your data variables. It is
    expected that the data variables have (time, y, x) as dimensions.

    The requirements for the structure are:
        - coordinates contain ["x", "y"]
        - if `require_lat_lon=True` then ["longitudes", "latitudes"] have to be in
          the coordinates
        - if `only_single_time_step=False`, there must be a coordinate "time"
        - if `only_single_time_step=True`, the "time" dimension has to have a length=1

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Dataset or DataArray to check
    require_lat_lon : bool (default True)
       Check if column for latitude and longitude is there
    only_single_time_step : bool (default True)
       Check that time dimension has a length == 1

    Raises
    ------
    ValueError in case one of the requirements is not met

    """
    required_coords = ["x", "y"]
    if require_lat_lon:
        required_coords += ["longitude", "latitude"]
    if not only_single_time_step:
        required_coords += ["time"]

    for coord in required_coords:
        if coord not in ds.coords:
            msg = f"Coordinate `{coord}` missing in Dataset or Dataarray"
            raise ValueError(msg)

    if only_single_time_step and "time" in ds.coords and ds.time.size != 1:
        msg = f"Length of time dimension is {len(ds.time)} but should be 1"
        raise ValueError(msg)
