"""Function to get data in the specific format needed for RADOLAN code"""

import pandas as pd

# TODO: This module could be removed in case the RADOLAN code is updated
# to use xr.Dataset as input for gauge and CML data.


def transform_openmrg_data_for_old_radolan_code(ds_cmls=None, ds_gauges=None):
    """Transform OpenMRG CML Dataset to DataFrame as needed by RADOLAN code.

    The old RADOLAN code requires input from gauges and CML in one pandas.DataFrame.
    This function creates such a DataFrame for the OpenMRG CML data based on the
    xr.Dataset that we normally use for CML data.

    The "sensor_type" column contains the values "cml" for CML data and "gauge"
    for gauge data.

    Parameters
    ----------
    ds_cmls : xr.Dataset
        The CML data in an xr.Dataset as returned
        by `io.load_and_transform_openmrg_data`
    ds_gauges : xr.Dataset
        The gauge data in an xr.Dataset

    Returns
    -------
    df_stations : pd.DataFrame
        A DataFrame with columns "time", "station_id", "sensor_type", and
        "rainfall_amount" containing the CML and gauge data in the format needed for
        the old RADOLAN rh_to_rw function.
    """
    if ds_cmls is not None:
        df_cmls = ds_cmls.to_dataframe().sort_index(level=["time", "cml_id"])
        df_cmls["station_id"] = df_cmls.index.get_level_values("cml_id")
        df_cmls.index = df_cmls.index.droplevel("cml_id")
        df_cmls["sensor_type"] = "cml"
        df_cmls["rainfall_amount"] = df_cmls.R
    if ds_gauges is not None:
        df_gauges = ds_gauges.to_dataframe().sort_index(level=["time", "id"])
        df_gauges["station_id"] = df_gauges.index.get_level_values("id")
        df_gauges.index = df_gauges.index.droplevel("id")
        df_gauges["sensor_type"] = "gauge"
        df_gauges["rainfall_amount"] = df_gauges.R

    if ds_cmls is not None and ds_gauges is not None:
        df_stations = pd.concat(
            [
                df_cmls[["station_id", "sensor_type", "rainfall_amount", "x", "y"]],
                df_gauges[["station_id", "sensor_type", "rainfall_amount", "x", "y"]],
            ],
            ignore_index=True,
        )
    elif ds_cmls is not None:
        df_stations = df_cmls[
            ["station_id", "sensor_type", "rainfall_amount", "x", "y"]
        ]
    elif ds_gauges is not None:
        df_stations = df_gauges[
            ["station_id", "sensor_type", "rainfall_amount", "x", "y"]
        ]
    else:
        msg = "At least one of ds_cmls and ds_gauges must be provided."
        raise ValueError(msg)

    return df_stations
