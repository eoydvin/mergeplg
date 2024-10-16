"""Functions to load data"""

from importlib.resources import files as importlib_files
from pathlib import Path

import poligrain as plg
import xarray as xr


def load_and_transform_openmrg_data():
    """Load OpenMRG example data and adjust variables names.

    Returns
    -------
    tuple (ds_rad, ds_cmls, ds_gauges, ds_gauges_smhi)
        ds_rad: Radar data
        ds_cmls: CML data
        ds_gauges: Municipal rain gauges
        ds_gauges_smhi: SMHI gauge

    """
    data_path = importlib_files("mergeplg") / "example_data"
    ds_gauges = xr.open_dataset(Path(data_path) / "openmrg_municp_gauge.nc")
    ds_cmls = xr.open_dataset(Path(data_path) / "openmrg_cml.nc")
    ds_rad = xr.open_dataset(Path(data_path) / "openmrg_rad.nc")
    ds_gauges_smhi = xr.open_dataset(Path(data_path) / "openmrg_smhi_gauge.nc")

    # UTM32N: https://epsg.io/32632
    ref_str = "EPSG:32632"

    ds_gauges.coords["x"], ds_gauges.coords["y"] = (
        plg.spatial.project_point_coordinates(ds_gauges.lon, ds_gauges.lat, ref_str)
    )

    ds_gauges_smhi.coords["x"], ds_gauges_smhi.coords["y"] = (
        plg.spatial.project_point_coordinates(
            ds_gauges_smhi.lon, ds_gauges_smhi.lat, ref_str
        )
    )

    # For CML
    (
        ds_cmls.coords["site_0_x"],
        ds_cmls.coords["site_0_y"],
    ) = plg.spatial.project_point_coordinates(
        ds_cmls.site_0_lon, ds_cmls.site_0_lat, ref_str
    )
    (
        ds_cmls.coords["site_1_x"],
        ds_cmls.coords["site_1_y"],
    ) = plg.spatial.project_point_coordinates(
        ds_cmls.site_1_lon, ds_cmls.site_1_lat, ref_str
    )

    # Midpoint
    ds_cmls["x"] = (ds_cmls.site_0_x + ds_cmls.site_1_x) / 2
    ds_cmls["y"] = (ds_cmls.site_0_y + ds_cmls.site_1_y) / 2

    # Projected radar coords
    ds_rad.coords["xs"], ds_rad.coords["ys"] = plg.spatial.project_point_coordinates(
        ds_rad.longitudes, ds_rad.latitudes, ref_str
    )

    # This is not correct, but I have to quickly create a 1D array
    # for x and y with the projection that has been used above
    ds_rad["x"] = ds_rad.xs.isel(y=20)
    ds_rad["y"] = ds_rad.ys.isel(x=20)

    return ds_rad, ds_cmls, ds_gauges, ds_gauges_smhi
