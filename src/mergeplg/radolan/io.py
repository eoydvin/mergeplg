"""Function to get data in the specific format needed for RADOLAN code"""

# TODO: This module could be removed in case the RADOLAN code is updated
# to use xr.Dataset as input for gauge and CML data.


def transform_openmrg_data_for_old_radolan_code(ds_cmls):
    """Transform OpenMRG CML Dataset to DataFrame as needed by RADOLAN code.

    The old RADOLAN code requires input from gauges and CML in one pandas.DataFrame.
    This function creates such a DataFrame for the OpenMRG CML data based on the
    xr.Dataset that we normally use for CML data.

    Parameters
    ----------
    ds_cmls : xr.Dataset
        The CML data in an xr.Dataset as returned
        by `io.load_and_transform_openmrg_data`

    Returns
    -------
    df_cmls : pd.DataFrame
        The CML data as DataFrame

    """
    df_cmls = ds_cmls.to_dataframe().sort_index(level=["time", "cml_id"])
    df_cmls["station_id"] = df_cmls.index.get_level_values("cml_id")
    df_cmls.index = df_cmls.index.droplevel("cml_id")
    df_cmls["sensor_type"] = "cml_ericsson"
    df_cmls["rainfall_amount"] = df_cmls.R
    return df_cmls
