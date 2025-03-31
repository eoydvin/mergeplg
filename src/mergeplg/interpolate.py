"""Module for merging CML and radar data."""

from __future__ import annotations

import numpy as np
import xarray as xr

from mergeplg import bk_functions
from mergeplg.base import Base

from .radolan import idw


class InterpolateIDW(Base):
    """Interpolate CML and rain gauge using IDW (CML midpoint)."""

    def __init__(
        self,
        grid_location_radar="center",
        min_observations=5,
        p=2,
        idw_method="radolan",
        nnear=8,
        max_distance=60000,
    ):
        """
        p: float
            IDW interpolation parameter
        idw_method: str
            by default "radolan"
        nnear: int
            number of neighbours to use for interpolation
        max_distance: float
            max distance allowed interpolation distance
        """
        Base.__init__(self, grid_location_radar)
        self.min_observations = min_observations
        self.p = p
        self.idw_method=idw_method
        self.nnear=nnear
        self.max_distance=max_distance

    def update(self, da_cml=None, da_gauge=None):
        """Initilize interpolator if observations have changed.

        Checks cml and gauge names from previous run. Return observations
        in correct order. 
        """

        return self.update_interpolator_idw_(da_cml, da_gauge)
        
    def interpolate(
        self,
        da_grid,
        da_cml=None,
        da_gauge=None,
    ):
        """Interpolate observations for one time step using IDW

        Interpolate observations for one time step. 

        Input data can have a time dimension of length 1 or no time dimension.

        Parameters
        ----------
        da_grid: xarray.DataArray
            Dataframe providing the grid for interpolation. Must contain
            projected x_grid and y_grid coordinates.
        da_cml: xarray.DataArray
            CML observations. Must contain the projected midpoint
            coordinates (x, y).
        da_gauge: xarray.DataArray
            Gauge observations. Must contain the coordinates projected
            coordinates (x, y).

        Returns
        -------
        da_field_out: xarray.DataArray
            DataArray with the same structure as the ds_rad but with the
            interpolated field.
        """
        time_dim_was_expanded = False
        if da_cml is not None and "time" not in da_cml.dims:
            da_cml = da_cml.copy().expand_dims("time")
            time_dim_was_expanded = True
        if da_gauge is not None and "time" not in da_gauge.dims:
            da_gauge = da_gauge.copy().expand_dims("time")
            time_dim_was_expanded = True
        if "time" not in da_grid.dims:
            da_grid = da_grid.copy().expand_dims("time")
            time_dim_was_expanded = True

        # Update interpolator
        obs = self.update(da_cml=da_cml, da_gauge=da_gauge)

        # If few observations return zero grid
        if (~np.isnan(obs)).sum() <= self.min_observations:
            return xr.DataArray(
                data=[np.zeros(da_grid.x_grid.shape)],
                coords=da_grid.coords,
                dims=da_grid.dims,
            )

        # Coordinates to predict
        coord_pred = np.hstack(
            [da_grid.y_grid.data.reshape(-1, 1), da_grid.x_grid.data.reshape(-1, 1)]
        )
        
        # IDW interpolator invdisttree
        interpolated = self.interpolator(
            q=coord_pred,
            z=obs,
            nnear=obs.size if obs.size <= self.nnear else self.nnear,
            p=self.p,
            idw_method=self.idw_method,
            max_distance=self.max_distance,
        ).reshape(da_grid.x_grid.shape)

        da_interpolated = xr.DataArray(
            data=[interpolated], coords=da_grid.coords, dims=da_grid.dims
        )
        if time_dim_was_expanded:
            da_interpolated = da_interpolated.isel(time=0)
            da_interpolated = da_interpolated.drop_vars("time")
        return da_interpolated


class InterpolateOrdinaryKriging(Base):
    """Interpolate CML and radar using neighbourhood ordinary kriging

    Interpolates the provided CML and rain gauge observations using
    ordinary kriging. The class defaults to interpolation using neighbouring
    observations. It also by default uses the full line geometry for
    interpolation, but can treat the lines as points by setting full_line
    to False.
    """

    def __init__(
        self,
        variogram_model="spherical",
        variogram_parameters=None,
        grid_location_radar="center",
        discretization=8,
        min_observations=1,
        nnear=8,
        max_distance=60000,
        full_line=True,
    ):
        """
        Parameters
        ----------
        variogram_model: str
            Must be a valid variogram type in pykrige.
        variogram_parameters: str
            Must be a valid parameters corresponding to variogram_model.
        nnear: int
            Number of closest links to use for interpolation
        max_distance: float
            Largest distance allowed for including an observation.
        full_line: bool
            Whether to use the full line for block kriging. If set to false, the
            x0 geometry is reformatted to simply reflect the midpoint of the CML.
        """
        Base.__init__(self, grid_location_radar)

        self.discretization = discretization
        self.min_observations = min_observations
        self.nnear = nnear
        self.max_distance = max_distance
        self.full_line=True

        # Construct variogram using parameters provided by user
        if variogram_parameters is None:
            variogram_parameters = {"sill": 0.9, "range": 5000, "nugget": 0.1}
        obs = np.array([0, 1]) # dummy variables
        coord = np.array([[0, 1], [1, 0]])

        self.variogram = bk_functions.construct_variogram(
            obs, coord, variogram_parameters, variogram_model
        )

    def update(self, da_cml=None, da_gauge=None):
        """Initilize interpolator if observations have changed.

        Checks cml and gauge names from previous run. Initialize
        if needed.
        """

        return self.update_interpolator_obk_(da_cml, da_gauge)

    def interpolate(
        self,
        da_grid,
        da_cml=None,
        da_gauge=None,
    ):
        """Interpolate observations for one time step.

        Interpolates ground observations for one time step. 

        Input data can have a time dimension of length 1 or no time dimension.

        Parameters
        ----------
        da_grid: xarray.DataArray
            Dataframe providing the grid for interpolation. Must contain
            projected x_grid and y_grid coordinates.
        da_cml: xarray.DataArray
            CML observations. Must contain the projected midpoint
            coordinates (x, y).
        da_gauge: xarray.DataArray
            Gauge observations. Must contain the projected
            coordinates (x, y).

        Returns
        -------
        da_field_out: xarray.DataArray
            DataArray with the same structure as the ds_rad but with the
            interpolated field.
        """
        time_dim_was_expanded = False
        if da_cml is not None and "time" not in da_cml.dims:
            da_cml = da_cml.copy().expand_dims("time")
            time_dim_was_expanded = True
        if da_gauge is not None and "time" not in da_gauge.dims:
            da_gauge = da_gauge.copy().expand_dims("time")
            time_dim_was_expanded = True
        if "time" not in da_grid.dims:
            da_grid = da_grid.copy().expand_dims("time")
            time_dim_was_expanded = True

        # Check if the cml or rain gauges are updated
        obs = self.update(da_cml=da_cml, da_gauge=da_gauge)
        
        # If few observations return zero grid
        if (~np.isnan(obs)).sum() <= self.min_observations:
            return xr.DataArray(
                data=[np.zeros(da_grid.x_grid.shape)],
                coords=da_grid.coords,
                dims=da_grid.dims,
            )

        # Points to interpolate
        points = np.hstack([
            da_grid.y_grid.data.reshape(-1, 1),
            da_grid.x_grid.data.reshape(-1, 1),
        ])
        
        # Else do neighbourhood kriging
        interpolated = self.interpolator(
            points,
            da_cmls=da_cml,
            da_gauges= da_gauge,
        ).reshape(da_grid.x_grid.shape)

        da_interpolated = xr.DataArray(
            data=[interpolated], coords=da_grid.coords, dims=da_grid.dims
        )
        if time_dim_was_expanded:
            da_interpolated = da_interpolated.isel(time=0)
            da_interpolated = da_interpolated.drop_vars("time")
        return da_interpolated
