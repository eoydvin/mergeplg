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
    ):
        Base.__init__(self, grid_location_radar)

        # Minimum number of observations needed to perform interpolation
        self.min_observations = min_observations

    def update(self, da_cml=None, da_gauge=None):
        """Update x0 geometry for CML and gauge

        This function uses the midpoint of the CML as CML reference.
        """
        # Update x0 and radar weights
        self.update_x0_(da_cml=da_cml, da_gauge=da_gauge)

    def interpolate(
        self,
        da_grid,
        da_cml=None,
        da_gauge=None,
        p=2,
        idw_method="radolan",
        nnear=8,
        max_distance=60000,
    ):
        """Interpolate observations for one time step using IDW

        Interpolate observations for one time step. The function assumes that
        the x0 are updated using the update class method.

        Parameters
        ----------
        da_grid: xarray.DataArray
            Dataframe providing the grid for interpolation. Must contain
            projected xs and ys coordinates.
        da_cml: xarray.DataArray
            CML observations. Must contain the projected midpoint
            coordinates (x, y).
        da_gauge: xarray.DataArray
            Gauge observations. Must contain the coordinates projected
            coordinates (x, y).
        p: float
            IDW interpolation parameter
        idw_method: str
            by default "radolan"
        nnear: int
            number of neighbours to use for interpolation
        max_distance: float
            max distance allowed interpolation distance

        Returns
        -------
        da_field_out: xarray.DataArray
            DataArray with the same structure as the ds_rad but with the
            interpolated field.
        """
        # Update x0 geometry for CML and gauge
        self.update(da_cml=da_cml, da_gauge=da_gauge)

        # Get ground observations and x0 geometry
        obs, x0 = self.get_obs_x0_(da_cml=da_cml, da_gauge=da_gauge)

        # Get index of not-nan obs
        keep = np.where(~np.isnan(obs))[0]

        # Return gridded data with zeros if too few observations
        if obs[keep].size <= self.min_observations:
            return xr.DataArray(
                data=[np.zeros(da_grid.x_grid.shape)],
                coords=da_grid.coords,
                dims=da_grid.dims,
            )

        # Coordinates to predict
        coord_pred = np.hstack(
            [da_grid.y_grid.data.reshape(-1, 1), da_grid.x_grid.data.reshape(-1, 1)]
        )

        # Ensure same functionality as in kriging
        if not nnear:
            nnear = obs[keep].size

        # IDW interpolator invdisttree
        idw_interpolator = idw.Invdisttree(x0[keep])
        interpolated = idw_interpolator(
            q=coord_pred,
            z=obs[keep],
            nnear=obs[keep].size if obs[keep].size <= nnear else nnear,
            p=p,
            idw_method=idw_method,
            max_distance=max_distance,
        ).reshape(da_grid.x_grid.shape)

        return xr.DataArray(
            data=[interpolated], coords=da_grid.coords, dims=da_grid.dims
        )


class InterpolateOrdinaryKriging(Base):
    """Interpolate CML and radar using neighbourhood ordinary kriging

    Interpolates the provided CML and rain gauge observations using
    ordinary kriging. The class defaults to interpolation using neighbouring
    observations, but it can also consider all observations by setting
    n_closest to False. It also by default uses the full line geometry for
    interpolation, but can treat the lines as points by setting full_line
    to False.
    """

    def __init__(
        self,
        grid_location_radar="center",
        discretization=8,
        min_observations=5,
    ):
        Base.__init__(self, grid_location_radar)

        # Number of discretization points along CML
        self.discretization = discretization

        # Minimum number of observations needed to perform interpolation
        self.min_observations = min_observations

    def update(self, da_cml=None, da_gauge=None):
        """Update weights and x0 geometry for CML and gauge assuming block data

        This function uses the full CML geometry and makes the gauge geometry
        appear as a rain gauge.
        """
        self.update_x0_block_(self.discretization, da_cml=da_cml, da_gauge=da_gauge)

    def interpolate(
        self,
        da_grid,
        da_cml=None,
        da_gauge=None,
        variogram_model="spherical",
        variogram_parameters=None,
        nnear=8,
        full_line=True,
    ):
        """Interpolate observations for one time step.

        Interpolates ground observations for one time step. The function assumes
        that the x0 are updated using the update class method.

        Parameters
        ----------
        da_grid: xarray.DataArray
            Dataframe providing the grid for interpolation. Must contain
            projected xs and ys coordinates.
        da_cml: xarray.DataArray
            CML observations. Must contain the projected midpoint
            coordinates (x, y).
        da_gauge: xarray.DataArray
            Gauge observations. Must contain the projected
            coordinates (x, y).
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

        Returns
        -------
        da_field_out: xarray.DataArray
            DataArray with the same structure as the ds_rad but with the
            interpolated field.
        """
        # Initialize variogram parameters
        if variogram_parameters is None:
            variogram_parameters = {"sill": 0.9, "range": 5000, "nugget": 0.1}

        # Update x0 geometry for CML and gauge
        self.update(da_cml=da_cml, da_gauge=da_gauge)

        # Get ground observations and x0 geometry
        obs, x0 = self.get_obs_x0_(da_cml=da_cml, da_gauge=da_gauge)

        # Get index of not-nan obs
        keep = np.where(~np.isnan(obs))[0]

        # Return gridded data with zeros if too few observations
        if obs[keep].size <= self.min_observations:
            return xr.DataArray(
                data=[np.zeros(da_grid.x_grid.shape)],
                coords=da_grid.coords,
                dims=da_grid.dims,
            )

        # Force interpolator to use only midpoint
        if full_line is False:
            x0 = x0[:, :, [int(x0.shape[2] / 2)]]

        # Construct variogram using parameters provided by user
        variogram = bk_functions.construct_variogram(
            obs[keep], x0[keep], variogram_parameters, variogram_model
        )

        # If nnear is set to False, use all observations in kriging
        if not nnear:
            interpolated = bk_functions.interpolate_block_kriging(
                da_grid.x_grid.data,
                da_grid.y_grid.data,
                obs[keep],
                x0[keep],
                variogram,
            )

        # Else do neighbourhood kriging
        else:
            interpolated = bk_functions.interpolate_neighbourhood_block_kriging(
                da_grid.x_grid.data,
                da_grid.y_grid.data,
                obs[keep],
                x0[keep],
                variogram,
                obs[keep].size if obs[keep].size <= nnear else nnear,
            )

        return xr.DataArray(
            data=[interpolated], coords=da_grid.coords, dims=da_grid.dims
        )
